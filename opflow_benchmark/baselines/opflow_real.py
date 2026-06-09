"""P-only latent SIT OpFlow model for real-world OD allocation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from opflow_benchmark.baselines.common import (
    EPS,
    FeatureStats,
    timed_evaluate_area_allocation,
    mean_scale_prediction,
    oracle_scale_prediction,
    profile_record,
    resolve_device,
    tensor_area,
    timed_evaluate_area_predictions,
)
from opflow_benchmark.exposure import local_destination_exposure, row_center


def _split_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(int(batch_size), 1)
    return [items[start : start + size] for start in range(0, len(items), size)]


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _smoothed_share(y: torch.Tensor, p: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return p
    n = int(y.shape[1])
    prior = torch.full_like(y, 1.0 / max(n, 1))
    denom = y.sum(dim=1, keepdim=True) + float(alpha)
    return (y + float(alpha) * prior) / torch.clamp(denom, min=EPS)


def _row_kl_values(p_true: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(torch.clamp(p_true, min=0.0) * logp).sum(dim=1)


def _risk_var(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_tensor(0.0)
    return values.var(unbiased=False)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpFlowRealNet(nn.Module):
    """Latent SIT operators plus additive operator potential."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        omega: float,
        tau: float = 1.0,
        rank_prefix_row_chunk_size: int | None = None,
        local_exposure_mode: str = "kernel_sum",
        scale_residual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.omega = float(omega)
        self.tau = float(tau)
        self.rank_prefix_row_chunk_size = rank_prefix_row_chunk_size
        self.local_exposure_mode = local_exposure_mode
        self.scale_residual_weight = float(scale_residual_weight)
        self.mean_outflow = 1.0
        self.log1p_outflow_mean = 0.0
        self.log1p_outflow_std = 1.0

        self.opportunity = _MLP(node_dim, hidden_dim, 1)
        self.impedance = _MLP(node_dim * 2 + 3, hidden_dim, 1)
        self.deterrence = _MLP(1, hidden_dim, 1)
        self.benefit = _MLP(1, hidden_dim, 1)
        self.exposure = _MLP(1, hidden_dim, 1)
        self.competition = _MLP(1, hidden_dim, 1)
        self.modifier = _MLP(node_dim + 3, hidden_dim, 1)
        self.generic_potential = _MLP(node_dim * 2 + 7, hidden_dim, 1)
        self.scale_encoder = _MLP(4, hidden_dim, 3)
        self.scale = _MLP(node_dim + 3, hidden_dim, 1)
        nn.init.zeros_(self.scale.net[-1].weight)
        nn.init.zeros_(self.scale.net[-1].bias)

    def opportunity_field(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.opportunity(x).squeeze(-1)) + EPS

    def _pair_geo(
        self,
        distance_raw: torch.Tensor,
        distance_scaled: torch.Tensor,
        adjacency: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = distance_scaled[start:end]
        adj = adjacency[start:end]
        kernel = torch.exp(-torch.clamp(distance_raw[start:end], min=0.0) / max(self.omega, EPS))
        geo = torch.stack([dist, adj, kernel], dim=-1)
        return geo, kernel

    def _pair_features(
        self,
        x: torch.Tensor,
        distance_raw: torch.Tensor,
        distance_scaled: torch.Tensor,
        adjacency: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_dest = int(x.shape[0])
        n_rows = int(end - start)
        xi = x[start:end, None, :].expand(n_rows, n_dest, x.shape[1])
        xj = x[None, :, :].expand(n_rows, n_dest, x.shape[1])
        geo, _ = self._pair_geo(distance_raw, distance_scaled, adjacency, start, end)
        return torch.cat([xi, xj, geo], dim=-1), geo

    def pair_impedance(
        self,
        x: torch.Tensor,
        distance_raw: torch.Tensor,
        distance_scaled: torch.Tensor,
        adjacency: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        pair_features, _ = self._pair_features(x, distance_raw, distance_scaled, adjacency, start, end)
        return F.softplus(self.impedance(pair_features).squeeze(-1)) + EPS

    def impedance_matrix(
        self,
        x: torch.Tensor,
        distance_raw: torch.Tensor,
        distance_scaled: torch.Tensor,
        adjacency: torch.Tensor,
        row_chunk_size: int,
    ) -> torch.Tensor:
        chunks = []
        n = int(x.shape[0])
        for start in range(0, n, max(int(row_chunk_size), 1)):
            end = min(start + int(row_chunk_size), n)
            chunks.append(self.pair_impedance(x, distance_raw, distance_scaled, adjacency, start, end))
        return torch.cat(chunks, dim=0) if chunks else x.new_zeros((0, 0))

    def rank_prefix_from_impedance(
        self,
        opportunity: torch.Tensor,
        impedance: torch.Tensor,
        row_chunk_size: int | None = None,
    ) -> torch.Tensor:
        n = int(opportunity.shape[0])
        result = torch.zeros((n, n), dtype=opportunity.dtype, device=opportunity.device)
        if n == 0:
            return result
        clean = torch.nan_to_num(impedance, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        chunk = int(row_chunk_size or self.rank_prefix_row_chunk_size or min(max(64, 262144 // max(n, 1)), n))
        chunk = max(min(chunk, n), 1)
        positions = torch.arange(n, device=opportunity.device, dtype=torch.long)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            imp_chunk = clean[start:end]
            order = torch.argsort(imp_chunk, dim=1, stable=True)
            sorted_imp = torch.gather(imp_chunk, 1, order)
            row_ids = torch.arange(start, end, device=opportunity.device, dtype=torch.long)[:, None]
            sorted_opp = opportunity[order].masked_fill(order == row_ids, 0.0)
            exclusive_prefix = sorted_opp.cumsum(dim=1) - sorted_opp
            group_start = torch.ones_like(sorted_imp, dtype=torch.bool)
            group_start[:, 1:] = sorted_imp[:, 1:] != sorted_imp[:, :-1]
            start_positions = torch.where(group_start, positions.expand(end - start, n), torch.zeros_like(order))
            group_start_positions = torch.cummax(start_positions, dim=1).values
            sorted_prefix = torch.gather(exclusive_prefix, 1, group_start_positions)
            result[start:end].scatter_(1, order, sorted_prefix)
        return result

    def scale_prediction(
        self,
        x: torch.Tensor,
        opportunity: torch.Tensor,
        impedance: torch.Tensor,
        prefix: torch.Tensor,
        local: torch.Tensor,
    ) -> torch.Tensor:
        n = int(x.shape[0])
        summary = torch.stack(
            [
                torch.log1p(opportunity).mean().expand(n),
                torch.log1p(torch.clamp(impedance, min=0.0)).mean(dim=1),
                torch.log1p(torch.clamp(prefix, min=0.0)).mean(dim=1),
                torch.log1p(torch.clamp(local, min=0.0)).mean().expand(n),
            ],
            dim=1,
        ).detach()
        encoded = self.scale_encoder(summary)
        raw_residual = self.scale(torch.cat([x, encoded], dim=1)).squeeze(-1)
        log_base = torch.as_tensor(self.log1p_outflow_mean, dtype=x.dtype, device=x.device)
        log_scale = torch.clamp(log_base + self.scale_residual_weight * raw_residual, 0.0, 16.0)
        learned_scale = torch.expm1(log_scale) + EPS
        mean_scale = torch.full_like(learned_scale, float(self.mean_outflow))
        weight = min(max(float(self.scale_residual_weight), 0.0), 1.0)
        return (1.0 - weight) * mean_scale + weight * learned_scale

    def chunk_outputs(
        self,
        x: torch.Tensor,
        distance_raw: torch.Tensor,
        distance_scaled: torch.Tensor,
        adjacency: torch.Tensor,
        opportunity: torch.Tensor,
        impedance: torch.Tensor,
        prefix: torch.Tensor,
        local: torch.Tensor,
        start: int,
        end: int,
        *,
        use_sit_operator_potential: bool,
        use_row_centering: bool,
    ) -> dict[str, torch.Tensor]:
        n_dest = int(x.shape[0])
        n_rows = int(end - start)
        o = opportunity[None, :].expand(n_rows, n_dest)
        c = impedance[start:end]
        s = prefix[start:end]
        ell = local[None, :].expand(n_rows, n_dest)
        pair_features, geo = self._pair_features(x, distance_raw, distance_scaled, adjacency, start, end)
        xi = x[start:end, None, :].expand(n_rows, n_dest, x.shape[1])

        if use_sit_operator_potential:
            psi = (
                self.deterrence(torch.log1p(c).unsqueeze(-1)).squeeze(-1)
                + self.benefit(torch.log1p(o).unsqueeze(-1)).squeeze(-1)
                + self.exposure(torch.log1p(s).unsqueeze(-1)).squeeze(-1)
                + self.competition(torch.log1p(ell).unsqueeze(-1)).squeeze(-1)
                + self.modifier(torch.cat([xi, geo], dim=-1)).squeeze(-1)
            )
        else:
            generic = torch.cat(
                [
                    pair_features,
                    torch.log1p(o).unsqueeze(-1),
                    torch.log1p(c).unsqueeze(-1),
                    torch.log1p(s).unsqueeze(-1),
                    torch.log1p(ell).unsqueeze(-1),
                ],
                dim=-1,
            )
            psi = self.generic_potential(generic).squeeze(-1)
        psi_bar = row_center(psi) if use_row_centering else psi
        logits = psi_bar / max(self.tau, EPS)
        return {"psi": psi, "psi_bar": psi_bar, "logits": logits, "o": o, "c": c, "s": s, "ell": ell}

    def monotone_penalty(
        self,
        impedance: torch.Tensor,
        prefix: torch.Tensor,
        opportunity: torch.Tensor,
        local: torch.Tensor,
        *,
        max_points: int = 4096,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def sample(values: torch.Tensor) -> torch.Tensor:
            flat = torch.log1p(torch.clamp(values.reshape(-1), min=0.0)).detach()
            if flat.numel() > max_points:
                flat = flat[:max_points]
            return flat.reshape(-1, 1).requires_grad_(True)

        def grad_violation(net: nn.Module, values: torch.Tensor, sign: str) -> torch.Tensor:
            if values.numel() == 0:
                return impedance.new_tensor(0.0)
            out = net(values).sum()
            grad = torch.autograd.grad(out, values, create_graph=True, retain_graph=True)[0]
            if sign == "decreasing":
                return F.relu(grad).mean()
            if sign == "increasing":
                return F.relu(-grad).mean()
            raise ValueError(f"Unknown monotone sign: {sign}")

        with torch.enable_grad():
            d_violation = grad_violation(self.deterrence, sample(impedance), "decreasing")
            s_violation = grad_violation(self.exposure, sample(prefix), "decreasing")
            b_violation = grad_violation(self.benefit, sample(opportunity), "increasing")
            penalty = d_violation + s_violation + b_violation
        return penalty, {
            "op_monotone_violation_D": d_violation.detach(),
            "op_monotone_violation_S": s_violation.detach(),
            "op_monotone_violation_B": b_violation.detach(),
        }


class OpFlowRealBaseline:
    """Real-world OpFlow baseline with P-only latent SIT operators."""

    REMOVED_VARIANTS = {"leaky_opflow", "opflow_no_routing", "opflow_no_diagnostic"}
    VARIANT_ALIASES = {
        "opflow_real": "opflow_full",
        "opflow_full": "opflow_full",
        "full": "opflow_full",
        "opflow_no_rank_exposure": "opflow_no_rank_exposure",
        "opflow_no_competition": "opflow_no_competition",
        "opflow_no_opportunity_field": "opflow_no_opportunity_field",
        "opflow_no_scale_detach": "opflow_no_scale_detach",
        "opflow_no_scale_branch": "opflow_no_scale_branch",
        "opflow_no_exposure_ops": "opflow_no_exposure_ops",
        "opflow_no_row_centering": "opflow_no_row_centering",
        "opflow_full_softmax_local": "opflow_full_softmax_local",
        "opflow_generic_potential_mlp": "opflow_generic_potential_mlp",
        "opflow_no_groupdro": "opflow_no_groupdro",
        "opflow_no_vrex": "opflow_no_vrex",
        "opflow_no_monotone_penalty": "opflow_no_monotone_penalty",
    }

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 50,
        weight_decay: float = 0.0,
        lambda_count: float = 0.1,
        lambda_scale: float = 1.0,
        lambda_vrex: float = 0.1,
        lambda_op: float = 0.01,
        lambda_chal: float = 0.0,
        hidden_dim: int = 64,
        area_batch_size: int = 4,
        row_chunk_size: int = 16,
        device: str = "auto",
        patience: int = 20,
        grad_clip: float = 5.0,
        alpha: float = 1.0,
        tau: float = 1.0,
        omega_grid: list[float] | tuple[float, ...] | None = None,
        rank_prefix_row_chunk_size: int | None = None,
        progress_path: str | None = None,
        progress_prefix: str | None = None,
        model_variant: str = "opflow_full",
        report_name: str | None = None,
        local_exposure_mode: str = "kernel_sum",
        scale_residual_weight: float = 0.1,
        use_groupdro: bool = True,
        use_rank_exposure: bool = True,
        use_competition: bool = True,
        use_opportunity_field: bool = True,
        use_row_center: bool = True,
        use_row_centering: bool | None = None,
        detach_count_allocation: bool = True,
        use_sit_operator_potential: bool = True,
        use_monotone_penalty: bool = True,
        use_vrex: bool = True,
        **_: Any,
    ) -> None:
        del lambda_chal
        self.device = resolve_device(device)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.lambda_count = float(lambda_count)
        self.lambda_scale = float(lambda_scale)
        self.lambda_vrex = float(lambda_vrex)
        self.lambda_op = float(lambda_op)
        self.lambda_chal = 0.0
        self.hidden_dim = int(hidden_dim)
        self.area_batch_size = max(int(area_batch_size), 1)
        self.row_chunk_size = max(int(row_chunk_size), 1)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.omega_grid = [float(v) for v in (omega_grid or [10000.0, 25000.0, 50000.0])]
        self.omega = self.omega_grid[0]
        self.rank_prefix_row_chunk_size = rank_prefix_row_chunk_size
        self.progress_path = Path(progress_path) if progress_path else None
        self.progress_prefix = progress_prefix or "model=opflow_real"
        canonical = self.VARIANT_ALIASES.get(model_variant, model_variant)
        if canonical in self.REMOVED_VARIANTS or model_variant in self.REMOVED_VARIANTS:
            raise ValueError(f"{model_variant} was removed from P-only OpFlow")
        self.model_variant = canonical
        self.report_name = report_name or self.model_variant
        self.local_exposure_mode = "softmax" if self.model_variant == "opflow_full_softmax_local" else local_exposure_mode
        self.scale_residual_weight = float(scale_residual_weight)

        row_center_flag = use_row_center if use_row_centering is None else use_row_centering
        self.use_rank_exposure = bool(use_rank_exposure) and self.model_variant not in {
            "opflow_no_rank_exposure",
            "opflow_no_exposure_ops",
        }
        self.use_competition = bool(use_competition) and self.model_variant not in {
            "opflow_no_competition",
            "opflow_no_exposure_ops",
        }
        self.use_opportunity_field = bool(use_opportunity_field) and self.model_variant != "opflow_no_opportunity_field"
        self.detach_count_allocation = bool(detach_count_allocation) and self.model_variant != "opflow_no_scale_detach"
        self.use_scale_branch = self.model_variant != "opflow_no_scale_branch"
        self.use_row_centering = bool(row_center_flag) and self.model_variant != "opflow_no_row_centering"
        self.use_sit_operator_potential = bool(use_sit_operator_potential) and self.model_variant != "opflow_generic_potential_mlp"
        self.use_groupdro = bool(use_groupdro) and self.model_variant != "opflow_no_groupdro"
        self.use_vrex = bool(use_vrex) and self.model_variant != "opflow_no_vrex"
        self.use_monotone_penalty = bool(use_monotone_penalty) and self.model_variant != "opflow_no_monotone_penalty"

        self.mean_outflow = 1.0
        self.log1p_outflow_mean = 0.0
        self.log1p_outflow_std = 1.0
        self.stats = FeatureStats()
        self.model: OpFlowRealNet | None = None
        self.profile: dict[str, float] = {}

    def _format_seconds(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        return f"{minutes / 60.0:.2f}h"

    def _emit_progress(self, message: str) -> None:
        print(message, flush=True)
        if self.progress_path is None:
            return
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    def _ensure_model(self, node_dim: int) -> None:
        if self.model is None:
            self.model = OpFlowRealNet(
                node_dim=node_dim,
                hidden_dim=self.hidden_dim,
                omega=self.omega,
                tau=self.tau,
                rank_prefix_row_chunk_size=self.rank_prefix_row_chunk_size,
                local_exposure_mode=self.local_exposure_mode,
                scale_residual_weight=self.scale_residual_weight,
            ).to(self.device)
            self.model.log1p_outflow_mean = self.log1p_outflow_mean
            self.model.log1p_outflow_std = self.log1p_outflow_std
            self.model.mean_outflow = self.mean_outflow

    def _select_omega(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> None:
        ref_areas = val_areas if val_areas else train_areas
        scores = []
        for omega in self.omega_grid:
            area_scores = []
            for area in ref_areas[: min(len(ref_areas), 32)]:
                x = torch.log1p(torch.clamp(area["node_feat"].float(), min=0.0)).sum(dim=1).to(self.device) + EPS
                distance = area["distance"].to(self.device)
                local = local_destination_exposure(x, distance, omega, mode=self.local_exposure_mode).detach().cpu().numpy()
                dest_inflow = area["y_raw"].sum(dim=0).detach().cpu().numpy()
                corr = _safe_corr(local, dest_inflow)
                if np.isfinite(corr):
                    area_scores.append(corr)
            scores.append(float(np.mean(area_scores)) if area_scores else float("-inf"))
        best_idx = int(np.argmax(scores)) if scores else 0
        self.omega = self.omega_grid[best_idx]

    def _batch_from_area(self, area: dict[str, Any]) -> dict[str, Any]:
        return tensor_area(area, self.device, self.stats)

    def _operator_parts(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        x = batch["x"]
        n = int(x.shape[0])
        opportunity = self.model.opportunity_field(x) if self.use_opportunity_field else torch.ones((n,), dtype=x.dtype, device=x.device)
        impedance = self.model.impedance_matrix(
            x,
            batch["distance_raw"],
            batch["distance"],
            batch["adjacency"],
            self.row_chunk_size,
        )
        if self.use_rank_exposure:
            prefix = self.model.rank_prefix_from_impedance(
                opportunity,
                impedance,
                row_chunk_size=self.rank_prefix_row_chunk_size,
            )
        else:
            prefix = torch.zeros_like(impedance)
        if self.use_competition:
            local = local_destination_exposure(opportunity, batch["distance_raw"], self.omega, mode=self.local_exposure_mode)
        else:
            local = torch.zeros_like(opportunity)
        if self.use_scale_branch:
            outflow_hat = self.model.scale_prediction(x, opportunity, impedance, prefix, local)
        else:
            outflow_hat = torch.full((n,), self.mean_outflow, dtype=x.dtype, device=x.device)
        return {
            "opportunity": opportunity,
            "impedance": impedance,
            "prefix": prefix,
            "local": local,
            "outflow_hat": outflow_hat,
        }

    def _loss_area_parts(
        self,
        area: dict[str, Any],
        *,
        collect_outputs: bool = False,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        batch = self._batch_from_area(area)
        x = batch["x"]
        y = batch["y"]
        valid = batch["valid"]
        p_target = _smoothed_share(y, batch["p"], self.alpha)
        n = int(x.shape[0])
        ops = self._operator_parts(batch)
        outflow_hat = ops["outflow_hat"]
        zero = x.new_tensor(0.0)
        alloc_sum = zero.clone()
        count_sum = zero.clone()
        valid_count = zero.clone()
        count_cells = 0
        y_chunks = []
        p_chunks = []
        psi_bar_chunks = []

        for start in range(0, n, self.row_chunk_size):
            end = min(start + self.row_chunk_size, n)
            output = self.model.chunk_outputs(
                x,
                batch["distance_raw"],
                batch["distance"],
                batch["adjacency"],
                ops["opportunity"],
                ops["impedance"],
                ops["prefix"],
                ops["local"],
                start,
                end,
                use_sit_operator_potential=self.use_sit_operator_potential,
                use_row_centering=self.use_row_centering,
            )
            logits = output["logits"]
            row_valid = valid[start:end]
            row_target = p_target[start:end]
            if bool(row_valid.any()):
                row_kl = _row_kl_values(row_target, logits)
                alloc_sum = alloc_sum + row_kl[row_valid].sum()
                valid_count = valid_count + row_valid.to(dtype=x.dtype).sum()
            p_hat = F.softmax(logits, dim=1)
            count_pi = p_hat.detach() if self.detach_count_allocation else p_hat
            y_hat_scale_only = outflow_hat[start:end, None] * count_pi
            if bool(row_valid.any()):
                count_sum = count_sum + F.smooth_l1_loss(
                    torch.log1p(y_hat_scale_only[row_valid]),
                    torch.log1p(y[start:end][row_valid]),
                    reduction="sum",
                )
                count_cells += int(row_valid.to(dtype=torch.int64).sum().item()) * int(y_hat_scale_only.shape[1])
            if collect_outputs:
                p_chunks.append(p_hat.detach())
                y_chunks.append((outflow_hat[start:end, None] * p_hat).detach())
                psi_bar_chunks.append(output["psi_bar"].detach())

        if bool(valid.any()):
            scale_rows = F.smooth_l1_loss(
                torch.log1p(outflow_hat[valid]),
                torch.log1p(batch["outflow"][valid]),
                reduction="none",
            )
            scale_loss = scale_rows.mean()
        else:
            scale_loss = F.smooth_l1_loss(torch.log1p(outflow_hat), torch.log1p(batch["outflow"]))
        safe_valid = torch.clamp(valid_count, min=1.0)
        op_penalty = zero.clone()
        op_diag = {
            "op_monotone_violation_D": zero.detach(),
            "op_monotone_violation_S": zero.detach(),
            "op_monotone_violation_B": zero.detach(),
        }
        if self.use_monotone_penalty and self.use_sit_operator_potential:
            op_penalty, op_diag = self.model.monotone_penalty(
                ops["impedance"],
                ops["prefix"],
                ops["opportunity"],
                ops["local"],
            )
        parts: dict[str, Any] = {
            "alloc": alloc_sum / safe_valid,
            "scale": scale_loss,
            "count": count_sum / max(count_cells, 1),
            "op_penalty": op_penalty,
            "outflow_hat": outflow_hat,
            **ops,
            **op_diag,
        }
        if collect_outputs:
            p_hat = torch.cat(p_chunks, dim=0) if p_chunks else x.new_zeros((n, n))
            y_hat = torch.cat(y_chunks, dim=0) if y_chunks else x.new_zeros((n, n))
            psi_bar = torch.cat(psi_bar_chunks, dim=0) if psi_bar_chunks else x.new_zeros((n, n))
            parts.update(
                {
                    "p_hat": p_hat,
                    "y_hat": y_hat,
                    "psi_bar": psi_bar,
                    "psi_bar_row_mean_abs": float(psi_bar.mean(dim=1).abs().mean().detach().cpu()),
                    "operator_mean_o_tilde": float(ops["opportunity"].mean().detach().cpu()),
                    "operator_mean_c_tilde": float(ops["impedance"].mean().detach().cpu()),
                    "gate_opportunity_corr": _safe_corr(
                        ops["opportunity"].detach().cpu().numpy(),
                        area["y_raw"].sum(dim=0).detach().cpu().numpy(),
                    ),
                    "gate_distance_corr": _safe_corr(
                        ops["impedance"].detach().cpu().numpy().reshape(-1),
                        batch["distance_raw"].detach().cpu().numpy().reshape(-1),
                    ),
                }
            )
        return parts

    def _loss_area(self, area: dict[str, Any]) -> torch.Tensor:
        parts = self._loss_area_parts(area)
        return parts["alloc"] + self.lambda_scale * parts["scale"] + self.lambda_count * parts["count"] + self.lambda_op * parts["op_penalty"]

    def _loss_batch_parts(self, areas: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, float]]:
        parts_list = [self._loss_area_parts(area) for area in areas]
        alloc = torch.stack([parts["alloc"] for parts in parts_list])
        scale = torch.stack([parts["scale"] for parts in parts_list])
        count = torch.stack([parts["count"] for parts in parts_list])
        op_penalty = torch.stack([parts["op_penalty"] for parts in parts_list])
        alloc_risk_mean = alloc.mean()
        alloc_risk_max = alloc.max() if self.use_groupdro else alloc_risk_mean
        vrex_alloc_var = _risk_var(alloc) if self.use_vrex else alloc.new_tensor(0.0)
        scale_risk_mean = scale.mean()
        scale_risk_max = scale.max()
        loss = (
            alloc_risk_max
            + self.lambda_vrex * vrex_alloc_var
            + self.lambda_scale * scale_risk_max
            + self.lambda_count * count.mean()
            + self.lambda_op * op_penalty.mean()
        )
        info = {
            "alloc_risk_max": float(alloc.max().detach().cpu()),
            "alloc_risk_mean": float(alloc_risk_mean.detach().cpu()),
            "vrex_alloc_var": float(vrex_alloc_var.detach().cpu()),
            "scale_risk_max": float(scale.max().detach().cpu()),
            "scale_risk_mean": float(scale_risk_mean.detach().cpu()),
            "op_monotone_violation_D": float(torch.stack([p["op_monotone_violation_D"] for p in parts_list]).mean().detach().cpu()),
            "op_monotone_violation_S": float(torch.stack([p["op_monotone_violation_S"] for p in parts_list]).mean().detach().cpu()),
            "op_monotone_violation_B": float(torch.stack([p["op_monotone_violation_B"] for p in parts_list]).mean().detach().cpu()),
        }
        return loss, info

    def _loss_batch(self, areas: list[dict[str, Any]]) -> torch.Tensor:
        return self._loss_batch_parts(areas)[0]

    def _eval_loss(self, areas: list[dict[str, Any]]) -> float:
        if not areas:
            return float("inf")
        with torch.no_grad():
            alloc = torch.stack([self._loss_area_parts(area)["alloc"].detach() for area in areas])
            objective = (alloc.max() if self.use_groupdro else alloc.mean()) + self.lambda_vrex * (_risk_var(alloc) if self.use_vrex else alloc.new_tensor(0.0))
        return float(objective.cpu())

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        if not train_areas:
            raise ValueError("No training areas available")
        self.stats.fit(train_areas)
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            outflow_values = torch.cat(valid_outflows)
            log_outflow = torch.log1p(torch.clamp(outflow_values, min=0.0))
            self.mean_outflow = float(outflow_values.mean().item())
            self.log1p_outflow_mean = float(log_outflow.mean().item())
            self.log1p_outflow_std = float(log_outflow.std(unbiased=False).item())
            if self.log1p_outflow_std <= EPS:
                self.log1p_outflow_std = 1.0
        self._select_omega(train_areas, val_areas)
        first = tensor_area(train_areas[0], self.device, self.stats)
        self._ensure_model(first["x"].shape[1])
        assert self.model is not None
        self.model.omega = self.omega
        self.model.local_exposure_mode = self.local_exposure_mode
        self.model.scale_residual_weight = self.scale_residual_weight
        self.model.mean_outflow = self.mean_outflow
        self.model.log1p_outflow_mean = self.log1p_outflow_mean
        self.model.log1p_outflow_std = self.log1p_outflow_std
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        history = {
            "train_loss": [],
            "val_loss": [],
            "alloc_risk_max": [],
            "vrex_alloc_var": [],
            "omega": [self.omega],
        }
        best_value = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        no_improve = 0
        val_ref = val_areas if val_areas else train_areas
        start_time = time.perf_counter()
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            last_info = {"alloc_risk_max": float("nan"), "vrex_alloc_var": float("nan")}
            batches = _split_batches(train_areas, self.area_batch_size)
            for batch_areas in batches:
                optimizer.zero_grad()
                loss, info = self._loss_batch_parts(batch_areas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                total += float(loss.detach().cpu())
                last_info = info
            self.model.eval()
            train_loss = total / max(len(batches), 1)
            val_loss = self._eval_loss(val_ref)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["alloc_risk_max"].append(last_info["alloc_risk_max"])
            history["vrex_alloc_var"].append(last_info["vrex_alloc_var"])
            completed = epoch + 1
            elapsed = time.perf_counter() - start_time
            avg = elapsed / completed
            eta = avg * (self.epochs - completed)
            self._emit_progress(
                "[real-world][epoch] "
                f"{self.progress_prefix} epoch={completed}/{self.epochs} "
                f"elapsed={self._format_seconds(elapsed)} eta={self._format_seconds(eta)} "
                f"train_loss={train_loss:.6g} val_alloc_obj={val_loss:.6g} "
                f"alloc_max={last_info['alloc_risk_max']:.6g} vrex={last_info['vrex_alloc_var']:.6g}"
            )
            if val_loss < best_value - 1e-6:
                best_value = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return history

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.eval()
        with torch.no_grad():
            parts = self._loss_area_parts(area, collect_outputs=True)
        extra = {
            "mean_pred_outflow": float(parts["outflow_hat"].mean().detach().cpu()),
            "train_mean_outflow": float(self.mean_outflow),
            "log1p_outflow_mean": float(self.log1p_outflow_mean),
            "log1p_outflow_std": float(self.log1p_outflow_std),
            "local_exposure_mode_kernel_sum": float(self.local_exposure_mode == "kernel_sum"),
            "scale_residual_weight": float(self.scale_residual_weight),
            "uses_scale_branch": float(self.use_scale_branch),
            "uses_row_centering": float(self.use_row_centering),
            "omega": float(self.omega),
            "alloc_risk_max": float(parts["alloc"].detach().cpu()),
            "alloc_risk_mean": float(parts["alloc"].detach().cpu()),
            "vrex_alloc_var": 0.0,
            "scale_risk_max": float(parts["scale"].detach().cpu()),
            "scale_risk_mean": float(parts["scale"].detach().cpu()),
            "psi_bar_row_mean_abs": float(parts["psi_bar_row_mean_abs"]),
            "operator_mean_o_tilde": float(parts["operator_mean_o_tilde"]),
            "operator_mean_c_tilde": float(parts["operator_mean_c_tilde"]),
            "gate_opportunity_corr": float(parts["gate_opportunity_corr"]),
            "gate_distance_corr": float(parts["gate_distance_corr"]),
            "op_monotone_violation_D": float(parts["op_monotone_violation_D"].detach().cpu()),
            "op_monotone_violation_S": float(parts["op_monotone_violation_S"].detach().cpu()),
            "op_monotone_violation_B": float(parts["op_monotone_violation_B"].detach().cpu()),
        }
        return parts["y_hat"], parts["p_hat"], extra

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload: dict[str, Any] = {
            f"{self.report_name}_learned_scale": {},
            f"{self.report_name}_oracle_scale": {},
            f"{self.report_name}_mean_scale": {},
        }
        for split, areas in split_areas.items():
            learned_rows = []
            oracle_rows = []
            mean_rows = []
            for area in areas:
                start = time.perf_counter()
                y_hat, p_hat, extra = self.predict_area(area)
                profile_record(self.profile, "predict_seconds", time.perf_counter() - start)
                self.profile["areas"] += 1.0
                allocation = timed_evaluate_area_allocation(self.profile, area, p_hat, topk=topk)
                learned_rows.append(timed_evaluate_area_predictions(self.profile, area, y_hat, p_hat, topk=topk, extra=extra, allocation=allocation))
                oracle_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        oracle_scale_prediction(area, p_hat),
                        p_hat,
                        topk=topk,
                        extra={**extra, "oracle_scale": 1.0},
                        allocation=allocation,
                    )
                )
                mean_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        mean_scale_prediction(area, p_hat, self.mean_outflow),
                        p_hat,
                        topk=topk,
                        extra={**extra, "mean_scale": 1.0},
                        allocation=allocation,
                    )
                )
            payload[f"{self.report_name}_learned_scale"][split] = learned_rows
            payload[f"{self.report_name}_oracle_scale"][split] = oracle_rows
            payload[f"{self.report_name}_mean_scale"][split] = mean_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return {
            "model_type": "opflow_real",
            "model_family": "p_only_sit_operator_real",
            "model_variant": self.model_variant,
            "report_name": self.report_name,
            "model_state": self.model.state_dict(),
            "stats": self.stats.state_dict(),
            "node_dim": self.model.node_dim,
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "lambda_count": self.lambda_count,
                "lambda_scale": self.lambda_scale,
                "lambda_vrex": self.lambda_vrex,
                "lambda_op": self.lambda_op,
                "hidden_dim": self.hidden_dim,
                "area_batch_size": self.area_batch_size,
                "row_chunk_size": self.row_chunk_size,
                "patience": self.patience,
                "grad_clip": self.grad_clip,
                "alpha": self.alpha,
                "tau": self.tau,
                "omega": self.omega,
                "omega_grid": self.omega_grid,
                "rank_prefix_row_chunk_size": self.rank_prefix_row_chunk_size,
                "model_variant": self.model_variant,
                "report_name": self.report_name,
                "mean_outflow": self.mean_outflow,
                "log1p_outflow_mean": self.log1p_outflow_mean,
                "log1p_outflow_std": self.log1p_outflow_std,
                "local_exposure_mode": self.local_exposure_mode,
                "scale_residual_weight": self.scale_residual_weight,
                "use_groupdro": self.use_groupdro,
                "use_rank_exposure": self.use_rank_exposure,
                "use_competition": self.use_competition,
                "use_opportunity_field": self.use_opportunity_field,
                "use_row_centering": self.use_row_centering,
                "detach_count_allocation": self.detach_count_allocation,
                "use_sit_operator_potential": self.use_sit_operator_potential,
                "use_monotone_penalty": self.use_monotone_penalty,
                "use_vrex": self.use_vrex,
            },
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.stats = FeatureStats.from_state_dict(checkpoint["stats"])
        config = checkpoint.get("config", {})
        self.hidden_dim = int(config.get("hidden_dim", self.hidden_dim))
        self.tau = float(config.get("tau", self.tau))
        self.row_chunk_size = int(config.get("row_chunk_size", self.row_chunk_size))
        self.omega = float(config.get("omega", self.omega))
        model_variant = config.get("model_variant", checkpoint.get("model_variant", self.model_variant))
        canonical = self.VARIANT_ALIASES.get(model_variant, model_variant)
        if canonical in self.REMOVED_VARIANTS:
            raise ValueError(f"{model_variant} was removed from P-only OpFlow")
        self.model_variant = canonical
        self.report_name = config.get("report_name", checkpoint.get("report_name", self.report_name))
        self.local_exposure_mode = config.get("local_exposure_mode", self.local_exposure_mode)
        self.scale_residual_weight = float(config.get("scale_residual_weight", self.scale_residual_weight))
        if self.model_variant == "opflow_full_softmax_local":
            self.local_exposure_mode = "softmax"
        self.use_rank_exposure = bool(config.get("use_rank_exposure", self.use_rank_exposure)) and self.model_variant not in {
            "opflow_no_rank_exposure",
            "opflow_no_exposure_ops",
        }
        self.use_competition = bool(config.get("use_competition", self.use_competition)) and self.model_variant not in {
            "opflow_no_competition",
            "opflow_no_exposure_ops",
        }
        self.use_opportunity_field = bool(config.get("use_opportunity_field", self.use_opportunity_field)) and self.model_variant != "opflow_no_opportunity_field"
        self.detach_count_allocation = bool(config.get("detach_count_allocation", self.detach_count_allocation)) and self.model_variant != "opflow_no_scale_detach"
        self.use_scale_branch = self.model_variant != "opflow_no_scale_branch"
        self.use_row_centering = bool(config.get("use_row_centering", self.use_row_centering)) and self.model_variant != "opflow_no_row_centering"
        self.use_sit_operator_potential = bool(config.get("use_sit_operator_potential", self.use_sit_operator_potential)) and self.model_variant != "opflow_generic_potential_mlp"
        self.use_groupdro = bool(config.get("use_groupdro", self.use_groupdro)) and self.model_variant != "opflow_no_groupdro"
        self.use_vrex = bool(config.get("use_vrex", self.use_vrex)) and self.model_variant != "opflow_no_vrex"
        self.use_monotone_penalty = bool(config.get("use_monotone_penalty", self.use_monotone_penalty)) and self.model_variant != "opflow_no_monotone_penalty"
        self.lambda_vrex = float(config.get("lambda_vrex", self.lambda_vrex))
        self.lambda_op = float(config.get("lambda_op", self.lambda_op))
        self.mean_outflow = float(config.get("mean_outflow", self.mean_outflow))
        self.log1p_outflow_mean = float(config.get("log1p_outflow_mean", np.log1p(max(self.mean_outflow, 0.0))))
        self.log1p_outflow_std = float(config.get("log1p_outflow_std", self.log1p_outflow_std))
        self.rank_prefix_row_chunk_size = config.get("rank_prefix_row_chunk_size", self.rank_prefix_row_chunk_size)
        self._ensure_model(int(checkpoint["node_dim"]))
        assert self.model is not None
        self.model.omega = self.omega
        self.model.local_exposure_mode = self.local_exposure_mode
        self.model.scale_residual_weight = self.scale_residual_weight
        self.model.mean_outflow = self.mean_outflow
        self.model.log1p_outflow_mean = self.log1p_outflow_mean
        self.model.log1p_outflow_std = self.log1p_outflow_std
        self.model.rank_prefix_row_chunk_size = self.rank_prefix_row_chunk_size
        self.model.load_state_dict(checkpoint["model_state"])
