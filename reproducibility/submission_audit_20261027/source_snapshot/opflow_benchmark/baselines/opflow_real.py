"""P-only latent SIT OpFlow model for real-world OD allocation."""

from __future__ import annotations

from collections import OrderedDict
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


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _reset_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


def _raw_opportunity_proxy(area: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Deployable non-label opportunity proxy from raw tract attributes."""
    raw = area.get("node_feat")
    if raw is None:
        n = int(area["y_raw"].shape[0])
        return torch.ones((n,), dtype=torch.float32, device=device)
    values = torch.log1p(torch.clamp(raw.to(device).float(), min=0.0))
    proxy = values.sum(dim=1)
    return torch.clamp(proxy, min=EPS)


def _proxy_entropy(proxy: torch.Tensor) -> torch.Tensor:
    n = int(proxy.numel())
    if n <= 1:
        return proxy.new_tensor(0.0)
    mass = torch.clamp(proxy, min=EPS)
    p = mass / torch.clamp(mass.sum(), min=EPS)
    return -(p * torch.log(torch.clamp(p, min=EPS))).sum() / np.log(float(n))


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, *, zero_last: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if zero_last:
            _reset_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ScalarMLP(nn.Module):
    def __init__(self, hidden_dim: int, *, skip: float = 0.0, zero_last: bool = True) -> None:
        super().__init__()
        self.skip = float(skip)
        self.net = _MLP(1, hidden_dim, 1, zero_last=zero_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x.reshape(-1, 1)).reshape_as(x)
        return self.skip * x + residual


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
        residual_hidden_dim: int = 16,
        pair_residual_hidden_dim: int | None = None,
        benefit_skip: float = 1.15,
        deterrence_skip: float = -0.35,
        exposure_skip: float = -0.65,
        competition_skip: float = 0.45,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.omega = float(omega)
        self.tau = float(tau)
        self.rank_prefix_row_chunk_size = rank_prefix_row_chunk_size
        self.local_exposure_mode = local_exposure_mode
        self.scale_residual_weight = float(scale_residual_weight)
        self.benefit_skip = float(benefit_skip)
        self.deterrence_skip = float(deterrence_skip)
        self.exposure_skip = float(exposure_skip)
        self.competition_skip = float(competition_skip)
        self.mean_outflow = 1.0
        self.log1p_outflow_mean = 0.0
        self.log1p_outflow_std = 1.0

        self.opportunity = _MLP(node_dim, hidden_dim, 1)
        self.opp_reconstruction = _MLP(node_dim + 1, hidden_dim, 1, zero_last=True)
        self.impedance = _MLP(node_dim * 2 + 3, hidden_dim, 1)
        self.deterrence = _ScalarMLP(hidden_dim, skip=abs(self.deterrence_skip))
        self.benefit = _ScalarMLP(hidden_dim, skip=abs(self.benefit_skip))
        self.exposure = _ScalarMLP(hidden_dim, skip=abs(self.exposure_skip))
        self.competition = _ScalarMLP(hidden_dim, skip=abs(self.competition_skip))
        self.coefficient_head = _MLP(node_dim, hidden_dim, 4, zero_last=True)
        self.residual_interaction = _MLP(node_dim + 4 + 3, max(int(residual_hidden_dim), 4), 1, zero_last=True)
        pair_hidden = max(int(pair_residual_hidden_dim or residual_hidden_dim), 4)
        self.pair_residual_interaction = _MLP(node_dim * 4 + 3, pair_hidden, 1, zero_last=True)
        self.modifier = _MLP(node_dim + 3, hidden_dim, 1)
        self.generic_potential = _MLP(node_dim * 2 + 7, hidden_dim, 1)
        self.scale_encoder = _MLP(4, hidden_dim, 3)
        self.scale = _MLP(node_dim + 3, hidden_dim, 1)
        nn.init.zeros_(self.scale.net[-1].weight)
        nn.init.zeros_(self.scale.net[-1].bias)

    def opportunity_field(
        self,
        x: torch.Tensor,
        proxy: torch.Tensor | None = None,
        *,
        use_reconstruction: bool = False,
        opp_recon_delta_scale: float = 0.25,
    ) -> torch.Tensor:
        if use_reconstruction and proxy is not None:
            base = torch.clamp(proxy.to(dtype=x.dtype, device=x.device), min=EPS)
            recon_input = torch.cat([x, torch.log1p(base).unsqueeze(-1)], dim=1)
            delta = float(opp_recon_delta_scale) * torch.tanh(self.opp_reconstruction(recon_input).squeeze(-1))
            return torch.clamp(base * torch.exp(delta), min=EPS)
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

    def soft_rank_prefix_from_impedance(
        self,
        opportunity: torch.Tensor,
        impedance: torch.Tensor,
        *,
        temperature: float = 0.05,
        row_chunk_size: int | None = None,
    ) -> torch.Tensor:
        n = int(opportunity.shape[0])
        result = torch.zeros((n, n), dtype=opportunity.dtype, device=opportunity.device)
        if n == 0:
            return result
        temp = max(float(temperature), EPS)
        clean = torch.nan_to_num(impedance, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        chunk = int(row_chunk_size or self.rank_prefix_row_chunk_size or min(max(16, 131072 // max(n, 1)), n))
        chunk = max(min(chunk, n), 1)
        positions = torch.arange(n, device=opportunity.device, dtype=torch.long)
        same_dest = torch.eye(n, dtype=torch.bool, device=opportunity.device).unsqueeze(0)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            cost = clean[start:end]
            pairwise_order = torch.sigmoid((cost.unsqueeze(2) - cost.unsqueeze(1)) / temp)
            row_ids = torch.arange(start, end, device=opportunity.device, dtype=torch.long)
            origin_self = positions[None, None, :] == row_ids[:, None, None]
            pairwise_order = pairwise_order.masked_fill(same_dest, 0.0).masked_fill(origin_self, 0.0)
            result[start:end] = (pairwise_order * opportunity[None, None, :]).sum(dim=2)
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
        return torch.expm1(log_scale) + EPS

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
        use_origin_gates: bool = True,
        origin_gate_rho: float = 0.5,
        use_residual_interaction: bool = True,
        eta_residual: float = 0.2,
        use_pair_residual_interaction: bool = True,
        pair_residual_weight: float = 0.2,
        use_crowding_shift_gate: bool = False,
        crowding_gate: torch.Tensor | None = None,
        crowding_fallback_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        n_dest = int(x.shape[0])
        n_rows = int(end - start)
        o = opportunity[None, :].expand(n_rows, n_dest)
        c = impedance[start:end]
        s = prefix[start:end]
        ell = local[None, :].expand(n_rows, n_dest)
        pair_features, geo = self._pair_features(x, distance_raw, distance_scaled, adjacency, start, end)
        xi = x[start:end, None, :].expand(n_rows, n_dest, x.shape[1])
        xj = x[None, :, :].expand(n_rows, n_dest, x.shape[1])
        log_o = torch.log1p(torch.clamp(o, min=0.0))
        log_c = torch.log1p(torch.clamp(c, min=0.0))
        log_s = torch.log1p(torch.clamp(s, min=0.0))
        log_ell = torch.log1p(torch.clamp(ell, min=0.0))

        if use_sit_operator_potential:
            if use_origin_gates:
                coeff = 1.0 + float(origin_gate_rho) * torch.tanh(self.coefficient_head(x[start:end]))
            else:
                coeff = torch.ones((n_rows, 4), dtype=x.dtype, device=x.device)
            a_i, b_i, r_i, local_i = [part[:, None] for part in coeff.unbind(dim=1)]
            u_benefit = a_i * self.benefit(log_o)
            u_deterrence = -b_i * self.deterrence(log_c)
            u_exposure = -r_i * self.exposure(log_s)
            u_competition = local_i * self.competition(log_ell)
            u_struct = u_deterrence + u_benefit + u_exposure + u_competition
            if use_residual_interaction and eta_residual > 0.0:
                residual_input = torch.cat([xi, torch.stack([log_o, log_c, log_s, log_ell], dim=-1), geo], dim=-1)
                u_residual = float(eta_residual) * self.residual_interaction(residual_input).squeeze(-1)
                if use_pair_residual_interaction and pair_residual_weight > 0.0:
                    pair_residual_input = torch.cat([xi, xj, xj - xi, torch.abs(xj - xi), geo], dim=-1)
                    u_residual = u_residual + float(pair_residual_weight) * self.pair_residual_interaction(pair_residual_input).squeeze(-1)
            else:
                u_residual = torch.zeros_like(u_struct)
            psi = u_struct + u_residual
            if use_crowding_shift_gate and crowding_fallback_weight > 0.0 and crowding_gate is not None:
                fallback = (
                    abs(self.benefit_skip) * log_o
                    - abs(self.deterrence_skip) * log_c
                    - abs(self.exposure_skip) * log_s
                    + abs(self.competition_skip) * log_ell
                )
                blend = torch.clamp(crowding_gate.to(dtype=x.dtype, device=x.device) * float(crowding_fallback_weight), 0.0, 1.0)
                while blend.ndim < psi.ndim:
                    blend = blend.unsqueeze(-1)
                psi = (1.0 - blend) * psi + blend * fallback
        else:
            generic = torch.cat(
                [
                    pair_features,
                    log_o.unsqueeze(-1),
                    log_c.unsqueeze(-1),
                    log_s.unsqueeze(-1),
                    log_ell.unsqueeze(-1),
                ],
                dim=-1,
            )
            psi = self.generic_potential(generic).squeeze(-1)
            u_struct = torch.zeros_like(psi)
            u_residual = torch.zeros_like(psi)
            coeff = torch.ones((n_rows, 4), dtype=x.dtype, device=x.device)
        psi_bar = row_center(psi) if use_row_centering else psi
        logits = psi_bar / max(self.tau, EPS)
        return {
            "psi": psi,
            "psi_bar": psi_bar,
            "logits": logits,
            "o": o,
            "c": c,
            "s": s,
            "ell": ell,
            "u_struct": u_struct,
            "u_residual": u_residual,
            "origin_gates": coeff,
        }

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

        def grad_violation(values: torch.Tensor, contribution: torch.Tensor, sign: str) -> torch.Tensor:
            if values.numel() == 0:
                return impedance.new_tensor(0.0)
            grad = torch.autograd.grad(contribution.sum(), values, create_graph=True, retain_graph=True)[0]
            if sign == "decreasing":
                return F.relu(grad).mean()
            if sign == "increasing":
                return F.relu(-grad).mean()
            raise ValueError(f"Unknown monotone sign: {sign}")

        with torch.enable_grad():
            d_values = sample(impedance)
            s_values = sample(prefix)
            b_values = sample(opportunity)
            d_violation = grad_violation(d_values, -self.deterrence(d_values), "decreasing")
            s_violation = grad_violation(s_values, -self.exposure(s_values), "decreasing")
            b_violation = grad_violation(b_values, self.benefit(b_values), "increasing")
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
        lambda_anchor: float = 0.005,
        lambda_smooth: float = 0.0005,
        lambda_selection_scale: float = 0.1,
        selection_protocol: str = "validation",
        leo_selection_metric: str = "joint_selection_score",
        leo_max_epochs: int | None = None,
        leo_patience: int | None = None,
        leo_max_folds: int = 0,
        leo_group_key: str | None = "auto",
        leo_fold_rank_prefix_mode: str | None = None,
        leo_min_folds_per_epoch: int | None = None,
        leo_min_fold_fraction: float = 0.75,
        rank_prefix_mode: str = "hard",
        rank_prefix_temperature: float = 0.05,
        use_learned_opportunity_reconstruction: bool = True,
        use_origin_gates: bool = True,
        use_residual_interaction: bool = True,
        use_crowding_shift_gate: bool = False,
        opp_recon_delta_scale: float = 0.25,
        origin_gate_rho: float = 0.5,
        eta_residual: float = 0.2,
        residual_hidden_dim: int = 16,
        use_pair_residual_interaction: bool = True,
        pair_residual_weight: float = 0.2,
        pair_residual_hidden_dim: int | None = None,
        benefit_skip: float = 1.15,
        deterrence_skip: float = -0.35,
        exposure_skip: float = -0.65,
        competition_skip: float = 0.45,
        crowding_gate_threshold: float = 1.5,
        crowding_gate_temperature: float = 0.5,
        crowding_fallback_weight: float = 1.0,
        omega_selection: str = "fixed",
        checkpoint_metric: str | None = None,
        cache_area_tensors: bool = True,
        cache_area_max_mb: float = 64.0,
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
        self.rank_prefix_mode = str(rank_prefix_mode).lower()
        if self.rank_prefix_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown rank_prefix_mode: {rank_prefix_mode}")
        self.rank_prefix_temperature = float(rank_prefix_temperature)
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
        self.use_learned_opportunity_reconstruction = bool(use_learned_opportunity_reconstruction) and self.use_opportunity_field
        self.use_origin_gates = bool(use_origin_gates) and self.use_sit_operator_potential
        self.use_residual_interaction = bool(use_residual_interaction) and self.use_sit_operator_potential
        self.use_pair_residual_interaction = bool(use_pair_residual_interaction) and self.use_residual_interaction
        self.use_crowding_shift_gate = bool(use_crowding_shift_gate) and self.use_sit_operator_potential
        self.opp_recon_delta_scale = float(opp_recon_delta_scale)
        self.origin_gate_rho = float(origin_gate_rho)
        self.eta_residual = float(eta_residual)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.pair_residual_weight = float(pair_residual_weight)
        self.pair_residual_hidden_dim = int(pair_residual_hidden_dim) if pair_residual_hidden_dim is not None else None
        self.benefit_skip = float(benefit_skip)
        self.deterrence_skip = float(deterrence_skip)
        self.exposure_skip = float(exposure_skip)
        self.competition_skip = float(competition_skip)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_selection_scale = float(lambda_selection_scale)
        self.selection_protocol = str(selection_protocol).lower()
        self.leo_selection_metric = str(leo_selection_metric)
        self.checkpoint_metric = str(checkpoint_metric or "alloc_loss")
        self.leo_max_epochs = int(leo_max_epochs) if leo_max_epochs is not None else self.epochs
        if self.leo_max_epochs <= 0:
            self.leo_max_epochs = self.epochs
        self.leo_patience = int(leo_patience) if leo_patience is not None else self.patience
        self.leo_max_folds = int(leo_max_folds)
        self.leo_group_key = str(leo_group_key or "auto")
        self.leo_fold_rank_prefix_mode = (
            None
            if leo_fold_rank_prefix_mode is None or str(leo_fold_rank_prefix_mode).lower() in {"", "none", "same"}
            else str(leo_fold_rank_prefix_mode).lower()
        )
        if self.leo_fold_rank_prefix_mode is not None and self.leo_fold_rank_prefix_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown leo_fold_rank_prefix_mode: {leo_fold_rank_prefix_mode}")
        self.leo_min_folds_per_epoch = int(leo_min_folds_per_epoch or 0)
        self.leo_min_fold_fraction = float(leo_min_fold_fraction)
        self.crowding_gate_threshold = float(crowding_gate_threshold)
        self.crowding_gate_temperature = float(crowding_gate_temperature)
        self.crowding_fallback_weight = float(crowding_fallback_weight)
        self.omega_selection = str(omega_selection).lower()
        self.cache_area_tensors = bool(cache_area_tensors)
        self.cache_area_max_bytes = max(int(float(cache_area_max_mb) * 1024 * 1024), 0)
        self._area_batch_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._area_batch_cache_bytes = 0
        self.crowding_entropy_mean = 0.0
        self.crowding_entropy_std = 1.0
        self.selection_summary_: dict[str, Any] = {
            "selection_protocol": "unfit",
            "selection_metric": self.leo_selection_metric,
            "selected_epoch": 0,
        }

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

    def _clear_area_cache(self) -> None:
        self._area_batch_cache.clear()
        self._area_batch_cache_bytes = 0

    def _cache_key(self, area: dict[str, Any]) -> str:
        return str(area.get("area_id", id(area)))

    def _remember_batch(self, key: str, batch: dict[str, Any]) -> None:
        if not self.cache_area_tensors or self.cache_area_max_bytes <= 0:
            return
        size = _tensor_bytes(batch)
        if size <= 0 or size > self.cache_area_max_bytes:
            return
        while self._area_batch_cache_bytes + size > self.cache_area_max_bytes and self._area_batch_cache:
            _, old = self._area_batch_cache.popitem(last=False)
            self._area_batch_cache_bytes -= _tensor_bytes(old)
        self._area_batch_cache[key] = batch
        self._area_batch_cache_bytes += size

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
                residual_hidden_dim=self.residual_hidden_dim,
                pair_residual_hidden_dim=self.pair_residual_hidden_dim,
                benefit_skip=self.benefit_skip,
                deterrence_skip=self.deterrence_skip,
                exposure_skip=self.exposure_skip,
                competition_skip=self.competition_skip,
            ).to(self.device)
            self.model.log1p_outflow_mean = self.log1p_outflow_mean
            self.model.log1p_outflow_std = self.log1p_outflow_std
            self.model.mean_outflow = self.mean_outflow

    def _fit_crowding_reference(self, train_areas: list[dict[str, Any]]) -> None:
        entropies = []
        for area in train_areas:
            proxy = _raw_opportunity_proxy(area, self.device)
            entropies.append(float(_proxy_entropy(proxy).detach().cpu()))
        if not entropies:
            self.crowding_entropy_mean = 0.0
            self.crowding_entropy_std = 1.0
            return
        arr = np.asarray(entropies, dtype=float)
        self.crowding_entropy_mean = float(arr.mean())
        std = float(arr.std())
        self.crowding_entropy_std = std if std > EPS else 1.0

    def _crowding_shift_gate(self, opportunity_proxy: torch.Tensor) -> torch.Tensor:
        if not self.use_crowding_shift_gate:
            return opportunity_proxy.new_tensor(0.0)
        entropy = _proxy_entropy(opportunity_proxy)
        z = torch.abs((entropy - float(self.crowding_entropy_mean)) / max(float(self.crowding_entropy_std), EPS))
        temp = max(float(self.crowding_gate_temperature), EPS)
        return torch.sigmoid((z - float(self.crowding_gate_threshold)) / temp)

    def _select_omega(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> None:
        if self.omega_selection in {"fixed", "first", "none"}:
            self.omega = self.omega_grid[0]
            return
        ref_areas = val_areas if val_areas else train_areas
        scores = []
        for omega in self.omega_grid:
            area_scores = []
            for area in ref_areas[: min(len(ref_areas), 32)]:
                x = _raw_opportunity_proxy(area, self.device)
                distance = area["distance"].to(self.device)
                local = local_destination_exposure(x, distance, omega, mode=self.local_exposure_mode).detach().cpu().numpy()
                target = (
                    area["y_raw"].sum(dim=0).detach().cpu().numpy()
                    if self.omega_selection in {"label_corr", "legacy_label_corr"}
                    else x.detach().cpu().numpy()
                )
                corr = _safe_corr(local, target)
                if np.isfinite(corr):
                    area_scores.append(corr)
            scores.append(float(np.mean(area_scores)) if area_scores else float("-inf"))
        best_idx = int(np.argmax(scores)) if scores else 0
        self.omega = self.omega_grid[best_idx]

    def _batch_from_area(self, area: dict[str, Any]) -> dict[str, Any]:
        key = self._cache_key(area)
        if self.cache_area_tensors and key in self._area_batch_cache:
            batch = self._area_batch_cache.pop(key)
            self._area_batch_cache[key] = batch
            self.profile["area_cache_hits"] = float(self.profile.get("area_cache_hits", 0.0) + 1.0)
            return batch
        self.profile["area_cache_misses"] = float(self.profile.get("area_cache_misses", 0.0) + 1.0)
        batch = tensor_area(area, self.device, self.stats)
        batch["opportunity_proxy"] = _raw_opportunity_proxy(area, self.device)
        batch["crowding_gate"] = self._crowding_shift_gate(batch["opportunity_proxy"])
        self._remember_batch(key, batch)
        return batch

    def _operator_parts(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        x = batch["x"]
        n = int(x.shape[0])
        proxy = batch.get("opportunity_proxy")
        opportunity = (
            self.model.opportunity_field(
                x,
                proxy,
                use_reconstruction=self.use_learned_opportunity_reconstruction,
                opp_recon_delta_scale=self.opp_recon_delta_scale,
            )
            if self.use_opportunity_field
            else torch.ones((n,), dtype=x.dtype, device=x.device)
        )
        impedance = self.model.impedance_matrix(
            x,
            batch["distance_raw"],
            batch["distance"],
            batch["adjacency"],
            self.row_chunk_size,
        )
        if self.use_rank_exposure:
            if self.rank_prefix_mode == "soft":
                prefix = self.model.soft_rank_prefix_from_impedance(
                    opportunity,
                    impedance,
                    temperature=self.rank_prefix_temperature,
                    row_chunk_size=self.rank_prefix_row_chunk_size,
                )
            else:
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
            "opportunity_proxy": proxy if proxy is not None else torch.ones_like(opportunity),
            "crowding_gate": batch.get("crowding_gate", x.new_tensor(0.0)),
        }

    def _opportunity_anchor_loss(self, ops: dict[str, torch.Tensor]) -> torch.Tensor:
        opportunity = ops["opportunity"]
        if not self.use_learned_opportunity_reconstruction or self.lambda_anchor <= 0.0:
            return opportunity.new_tensor(0.0)
        proxy = torch.clamp(ops["opportunity_proxy"].to(dtype=opportunity.dtype, device=opportunity.device), min=EPS)
        return F.mse_loss(torch.log1p(opportunity), torch.log1p(proxy))

    def _opportunity_smoothness_loss(self, batch: dict[str, Any], ops: dict[str, torch.Tensor]) -> torch.Tensor:
        opportunity = ops["opportunity"]
        if not self.use_learned_opportunity_reconstruction or self.lambda_smooth <= 0.0 or int(opportunity.numel()) <= 1:
            return opportunity.new_tensor(0.0)
        log_o = torch.log1p(torch.clamp(opportunity, min=0.0))
        distance = torch.clamp(batch["distance_raw"].to(dtype=log_o.dtype, device=log_o.device), min=0.0)
        weights = torch.exp(-distance / max(float(self.omega), EPS))
        weights = weights.masked_fill(torch.eye(int(log_o.numel()), dtype=torch.bool, device=log_o.device), 0.0)
        denom = torch.clamp(weights.sum(), min=EPS)
        diff = log_o[:, None] - log_o[None, :]
        return (weights * diff.square()).sum() / denom

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
        residual_chunks = []

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
                use_origin_gates=self.use_origin_gates,
                origin_gate_rho=self.origin_gate_rho,
                use_residual_interaction=self.use_residual_interaction,
                eta_residual=self.eta_residual,
                use_pair_residual_interaction=self.use_pair_residual_interaction,
                pair_residual_weight=self.pair_residual_weight,
                use_crowding_shift_gate=self.use_crowding_shift_gate,
                crowding_gate=ops["crowding_gate"],
                crowding_fallback_weight=self.crowding_fallback_weight,
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
                residual_chunks.append(output["u_residual"].detach())

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
        opportunity_anchor_loss = self._opportunity_anchor_loss(ops)
        opportunity_smooth_loss = self._opportunity_smoothness_loss(batch, ops)
        parts: dict[str, Any] = {
            "alloc": alloc_sum / safe_valid,
            "scale": scale_loss,
            "count": count_sum / max(count_cells, 1),
            "op_penalty": op_penalty,
            "opportunity_anchor_loss": opportunity_anchor_loss,
            "opportunity_smooth_loss": opportunity_smooth_loss,
            "outflow_hat": outflow_hat,
            **ops,
            **op_diag,
        }
        if collect_outputs:
            p_hat = torch.cat(p_chunks, dim=0) if p_chunks else x.new_zeros((n, n))
            y_hat = torch.cat(y_chunks, dim=0) if y_chunks else x.new_zeros((n, n))
            psi_bar = torch.cat(psi_bar_chunks, dim=0) if psi_bar_chunks else x.new_zeros((n, n))
            residual = torch.cat(residual_chunks, dim=0) if residual_chunks else x.new_zeros((n, n))
            parts.update(
                {
                    "p_hat": p_hat,
                    "y_hat": y_hat,
                    "psi_bar": psi_bar,
                    "psi_bar_row_mean_abs": float(psi_bar.mean(dim=1).abs().mean().detach().cpu()),
                    "operator_mean_o_tilde": float(ops["opportunity"].mean().detach().cpu()),
                    "operator_mean_c_tilde": float(ops["impedance"].mean().detach().cpu()),
                    "opportunity_anchor_loss_value": float(opportunity_anchor_loss.detach().cpu()),
                    "opportunity_smooth_loss_value": float(opportunity_smooth_loss.detach().cpu()),
                    "crowding_shift_gate": float(ops["crowding_gate"].detach().cpu()),
                    "residual_interaction_abs_mean": float(residual.abs().mean().detach().cpu()),
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
        return (
            parts["alloc"]
            + self.lambda_scale * parts["scale"]
            + self.lambda_count * parts["count"]
            + self.lambda_op * parts["op_penalty"]
            + self.lambda_anchor * parts.get("opportunity_anchor_loss", parts["alloc"].new_tensor(0.0))
            + self.lambda_smooth * parts.get("opportunity_smooth_loss", parts["alloc"].new_tensor(0.0))
        )

    def _loss_batch_parts(self, areas: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, float]]:
        parts_list = [self._loss_area_parts(area) for area in areas]
        alloc = torch.stack([parts["alloc"] for parts in parts_list])
        scale = torch.stack([parts["scale"] for parts in parts_list])
        count = torch.stack([parts["count"] for parts in parts_list])
        op_penalty = torch.stack([parts["op_penalty"] for parts in parts_list])
        zero_like_alloc = alloc.new_tensor(0.0)
        opportunity_anchor = torch.stack([parts.get("opportunity_anchor_loss", zero_like_alloc) for parts in parts_list])
        opportunity_smooth = torch.stack([parts.get("opportunity_smooth_loss", zero_like_alloc) for parts in parts_list])
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
            + self.lambda_anchor * opportunity_anchor.mean()
            + self.lambda_smooth * opportunity_smooth.mean()
        )
        info = {
            "alloc_risk_max": float(alloc.max().detach().cpu()),
            "alloc_risk_mean": float(alloc_risk_mean.detach().cpu()),
            "vrex_alloc_var": float(vrex_alloc_var.detach().cpu()),
            "scale_risk_max": float(scale.max().detach().cpu()),
            "scale_risk_mean": float(scale_risk_mean.detach().cpu()),
            "opportunity_anchor_loss": float(opportunity_anchor.mean().detach().cpu()),
            "opportunity_smooth_loss": float(opportunity_smooth.mean().detach().cpu()),
            "op_monotone_violation_D": float(torch.stack([p["op_monotone_violation_D"] for p in parts_list]).mean().detach().cpu()),
            "op_monotone_violation_S": float(torch.stack([p["op_monotone_violation_S"] for p in parts_list]).mean().detach().cpu()),
            "op_monotone_violation_B": float(torch.stack([p["op_monotone_violation_B"] for p in parts_list]).mean().detach().cpu()),
        }
        return loss, info

    def _loss_batch(self, areas: list[dict[str, Any]]) -> torch.Tensor:
        return self._loss_batch_parts(areas)[0]

    def _score_from_parts(self, parts: dict[str, Any], metric: str) -> torch.Tensor:
        metric_key = str(metric).lower()
        if metric_key in {"joint_selection_score", "joint", "selection"}:
            return parts["alloc"] + float(self.lambda_selection_scale) * parts["scale"]
        if metric_key in {"scale_loss", "scale"}:
            return parts["scale"]
        if metric_key in {"loss", "train_loss"}:
            return (
                parts["alloc"]
                + self.lambda_scale * parts["scale"]
                + self.lambda_count * parts["count"]
                + self.lambda_op * parts["op_penalty"]
                + self.lambda_anchor * parts.get("opportunity_anchor_loss", parts["alloc"].new_tensor(0.0))
                + self.lambda_smooth * parts.get("opportunity_smooth_loss", parts["alloc"].new_tensor(0.0))
            )
        return parts["alloc"]

    def _eval_score(self, areas: list[dict[str, Any]], metric: str) -> float:
        if not areas:
            return float("inf")
        with torch.no_grad():
            scores = torch.stack([self._score_from_parts(self._loss_area_parts(area), metric).detach() for area in areas])
        return float(scores.mean().cpu())

    def _eval_loss(self, areas: list[dict[str, Any]]) -> float:
        if not areas:
            return float("inf")
        with torch.no_grad():
            alloc = torch.stack([self._loss_area_parts(area)["alloc"].detach() for area in areas])
            objective = (alloc.max() if self.use_groupdro else alloc.mean()) + self.lambda_vrex * (_risk_var(alloc) if self.use_vrex else alloc.new_tensor(0.0))
        return float(objective.cpu())

    def _sync_model_runtime_fields(self) -> None:
        if self.model is None:
            return
        self.model.omega = self.omega
        self.model.local_exposure_mode = self.local_exposure_mode
        self.model.scale_residual_weight = self.scale_residual_weight
        self.model.mean_outflow = self.mean_outflow
        self.model.log1p_outflow_mean = self.log1p_outflow_mean
        self.model.log1p_outflow_std = self.log1p_outflow_std
        self.model.rank_prefix_row_chunk_size = self.rank_prefix_row_chunk_size

    def _initialize_training_context(
        self,
        train_areas: list[dict[str, Any]],
        val_areas: list[dict[str, Any]],
        *,
        reset_model: bool,
    ) -> None:
        if not train_areas:
            raise ValueError("No training areas available")
        self._clear_area_cache()
        self.stats = FeatureStats().fit(train_areas)
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            outflow_values = torch.cat(valid_outflows)
            log_outflow = torch.log1p(torch.clamp(outflow_values, min=0.0))
            self.mean_outflow = float(outflow_values.mean().item())
            self.log1p_outflow_mean = float(log_outflow.mean().item())
            self.log1p_outflow_std = float(log_outflow.std(unbiased=False).item())
            if self.log1p_outflow_std <= EPS:
                self.log1p_outflow_std = 1.0
        else:
            self.mean_outflow = 1.0
            self.log1p_outflow_mean = 0.0
            self.log1p_outflow_std = 1.0
        self._fit_crowding_reference(train_areas)
        self._select_omega(train_areas, val_areas)
        if reset_model:
            self.model = None
        first = self._batch_from_area(train_areas[0])
        self._ensure_model(first["x"].shape[1])
        assert self.model is not None
        self._sync_model_runtime_fields()

    def _fit_once(
        self,
        train_areas: list[dict[str, Any]],
        val_areas: list[dict[str, Any]],
        *,
        max_epochs: int | None = None,
        patience_override: int | None = None,
        selection_metric: str | None = None,
        reset_model: bool = True,
        track_best: bool = True,
        emit_progress: bool = True,
        rank_prefix_mode_override: str | None = None,
    ) -> dict[str, list[float]]:
        original_rank_prefix_mode = self.rank_prefix_mode
        if rank_prefix_mode_override is not None:
            override = str(rank_prefix_mode_override).lower()
            if override not in {"hard", "soft"}:
                raise ValueError(f"Unknown rank_prefix_mode override: {rank_prefix_mode_override}")
            self.rank_prefix_mode = override
        try:
            self._initialize_training_context(train_areas, val_areas, reset_model=reset_model)
            assert self.model is not None
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            total_epochs = int(self.epochs if max_epochs is None else max_epochs)
            total_epochs = max(total_epochs, 0)
            patience_value = self.patience if patience_override is None else int(patience_override)
            metric = selection_metric or "alloc_loss"
            history = {
                "train_loss": [],
                "val_loss": [],
                "selection_score": [],
                "alloc_risk_max": [],
                "vrex_alloc_var": [],
                "opportunity_anchor_loss": [],
                "opportunity_smooth_loss": [],
                "omega": [self.omega],
                "rank_prefix_mode": [self.rank_prefix_mode],
            }
            best_value = float("inf")
            best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            no_improve = 0
            val_ref = val_areas if val_areas else train_areas
            start_time = time.perf_counter()
            for epoch in range(total_epochs):
                self.model.train()
                total = 0.0
                last_info = {
                    "alloc_risk_max": float("nan"),
                    "vrex_alloc_var": float("nan"),
                    "opportunity_anchor_loss": float("nan"),
                    "opportunity_smooth_loss": float("nan"),
                }
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
                selection_score = self._eval_score(val_ref, metric)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["selection_score"].append(selection_score)
                history["alloc_risk_max"].append(last_info["alloc_risk_max"])
                history["vrex_alloc_var"].append(last_info["vrex_alloc_var"])
                history["opportunity_anchor_loss"].append(last_info["opportunity_anchor_loss"])
                history["opportunity_smooth_loss"].append(last_info["opportunity_smooth_loss"])
                completed = epoch + 1
                elapsed = time.perf_counter() - start_time
                avg = elapsed / completed
                eta = avg * (total_epochs - completed)
                if emit_progress:
                    self._emit_progress(
                        "[real-world][epoch] "
                        f"{self.progress_prefix} epoch={completed}/{total_epochs} "
                        f"elapsed={self._format_seconds(elapsed)} eta={self._format_seconds(eta)} "
                        f"train_loss={train_loss:.6g} val_alloc_obj={val_loss:.6g} "
                        f"selection_{metric}={selection_score:.6g} "
                        f"alloc_max={last_info['alloc_risk_max']:.6g} vrex={last_info['vrex_alloc_var']:.6g}"
                    )
                if track_best:
                    if selection_score < best_value - 1e-6:
                        best_value = selection_score
                        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if patience_value > 0 and no_improve >= patience_value:
                            break
            if track_best and self.model is not None:
                self.model.load_state_dict(best_state)
            history["selected_epoch"] = [int(np.argmin(history["selection_score"]) + 1) if history["selection_score"] else 0]
            history["best_score"] = [float(min(history["selection_score"])) if history["selection_score"] else float("inf")]
            return history
        finally:
            self.rank_prefix_mode = original_rank_prefix_mode

    def _use_leo_selection(self, train_areas: list[dict[str, Any]]) -> bool:
        return self.selection_protocol in {"leo", "leave_one_env", "leave_one_environment_out"} and len(train_areas) > 1

    def _leo_area_group(self, area: dict[str, Any]) -> str:
        key = self.leo_group_key.strip().lower()
        if key in {"", "none", "county", "area", "area_id"}:
            return str(area.get("area_id", id(area)))
        if key == "auto":
            for candidate in ("leo_group", "census_division", "STATEFP", "statefp", "opp_group"):
                value = area.get(candidate)
                if value not in (None, ""):
                    return str(value)
            diagnostics = area.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                for candidate in ("leo_group", "census_division", "STATEFP", "statefp", "opp_group"):
                    value = diagnostics.get(candidate)
                    if value not in (None, ""):
                        return str(value)
            return str(area.get("area_id", id(area)))
        value = area.get(self.leo_group_key)
        if value in (None, ""):
            value = area.get(key)
        if value in (None, ""):
            diagnostics = area.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                value = diagnostics.get(self.leo_group_key, diagnostics.get(key))
        return str(value) if value not in (None, "") else str(area.get("area_id", id(area)))

    def _leo_candidate_groups(self, train_areas: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
        by_group: dict[str, list[int]] = {}
        for idx, area in enumerate(train_areas):
            group = self._leo_area_group(area)
            by_group.setdefault(group, []).append(idx)
        groups = [(group, indices) for group, indices in by_group.items() if len(indices) < len(train_areas)]
        if not groups:
            groups = [(str(idx), [idx]) for idx in range(len(train_areas))]
        return sorted(groups, key=lambda item: (item[0], item[1][0]))

    def _leo_holdout_groups(self, train_areas: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
        groups = self._leo_candidate_groups(train_areas)
        if self.leo_max_folds <= 0 or self.leo_max_folds >= len(groups):
            return groups
        positions = np.linspace(0, len(groups) - 1, num=max(int(self.leo_max_folds), 1), dtype=int)
        return [groups[int(pos)] for pos in sorted(set(int(pos) for pos in positions))]

    def _aggregate_leo_scores(self, fold_scores: list[list[float]]) -> tuple[list[float | None], list[int], int, int]:
        max_epochs = max((len(scores) for scores in fold_scores), default=0)
        if max_epochs <= 0:
            return [], [], 0, 0
        score_matrix = np.full((len(fold_scores), max_epochs), np.nan, dtype=float)
        for row, scores in enumerate(fold_scores):
            if scores:
                score_matrix[row, : len(scores)] = np.asarray(scores, dtype=float)
        fold_counts = np.isfinite(score_matrix).sum(axis=0).astype(int)
        if self.leo_min_folds_per_epoch > 0:
            min_folds = min(max(int(self.leo_min_folds_per_epoch), 1), max(len(fold_scores), 1))
        else:
            fraction = min(max(float(self.leo_min_fold_fraction), 0.0), 1.0)
            min_folds = max(1, int(np.ceil(max(len(fold_scores), 1) * fraction)))
        mean_scores: list[float | None] = []
        for epoch_idx in range(max_epochs):
            if fold_counts[epoch_idx] < min_folds:
                mean_scores.append(None)
                continue
            mean_scores.append(float(np.nanmean(score_matrix[:, epoch_idx])))
        valid = [(idx, score) for idx, score in enumerate(mean_scores) if score is not None and np.isfinite(score)]
        selected_epoch = int(min(valid, key=lambda item: item[1])[0] + 1) if valid else max(1, min(max_epochs, self.epochs))
        complete_epochs = int((fold_counts == len(fold_scores)).sum())
        return mean_scores, [int(value) for value in fold_counts.tolist()], selected_epoch, complete_epochs

    def _fit_with_leo_selection(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        fold_epochs = max(1, min(int(self.leo_max_epochs), int(self.epochs)))
        candidate_groups = self._leo_candidate_groups(train_areas)
        holdout_groups = self._leo_holdout_groups(train_areas)
        fold_scores: list[list[float]] = []
        self._emit_progress(
            "[real-world][leo] "
            f"{self.progress_prefix} folds={len(holdout_groups)}/{len(candidate_groups)} train_areas={len(train_areas)} "
            f"group_key={self.leo_group_key} fold_epochs={fold_epochs} metric={self.leo_selection_metric} "
            f"fold_rank_prefix={self.leo_fold_rank_prefix_mode or self.rank_prefix_mode}"
        )
        heldout_group_names: list[str] = []
        heldout_area_counts: list[int] = []
        for fold_num, (group_name, heldout_indices) in enumerate(holdout_groups, start=1):
            heldout = set(heldout_indices)
            fold_train = [area for idx, area in enumerate(train_areas) if idx not in heldout]
            fold_val = [train_areas[idx] for idx in heldout_indices]
            if not fold_train or not fold_val:
                continue
            history = self._fit_once(
                fold_train,
                fold_val,
                max_epochs=fold_epochs,
                patience_override=self.leo_patience,
                selection_metric=self.leo_selection_metric,
                reset_model=True,
                track_best=True,
                emit_progress=False,
                rank_prefix_mode_override=self.leo_fold_rank_prefix_mode,
            )
            fold_scores.append([float(v) for v in history["selection_score"]])
            heldout_group_names.append(str(group_name))
            heldout_area_counts.append(len(fold_val))
            self._emit_progress(
                "[real-world][leo-fold] "
                f"{self.progress_prefix} fold={fold_num}/{len(holdout_groups)} "
                f"heldout_group={group_name} heldout_areas={len(fold_val)} "
                f"epochs={len(history['selection_score'])} best={min(history['selection_score']) if history['selection_score'] else float('inf'):.6g}"
            )
        mean_score_by_epoch, fold_counts_by_epoch, selected_epoch, complete_epochs = self._aggregate_leo_scores(fold_scores)
        self._emit_progress(
            "[real-world][leo-selected] "
            f"{self.progress_prefix} selected_epoch={selected_epoch} "
            f"complete_epochs={complete_epochs} capped={int(self.leo_max_folds > 0 and self.leo_max_folds < len(candidate_groups))}"
        )
        final_history = self._fit_once(
            train_areas,
            val_areas,
            max_epochs=selected_epoch,
            patience_override=0,
            selection_metric=self.leo_selection_metric,
            reset_model=True,
            track_best=False,
            emit_progress=True,
        )
        final_history["leo_mean_score_by_epoch"] = mean_score_by_epoch
        self.selection_summary_ = {
            "selection_protocol": "leo",
            "selection_metric": self.leo_selection_metric,
            "selected_epoch": int(selected_epoch),
            "lambda_selection_scale": float(self.lambda_selection_scale),
            "leo_max_epochs": int(fold_epochs),
            "leo_patience": int(self.leo_patience),
            "leo_complete_epochs": int(complete_epochs),
            "leo_group_key": self.leo_group_key,
            "leo_fold_rank_prefix_mode": self.leo_fold_rank_prefix_mode or self.rank_prefix_mode,
            "leo_min_folds_per_epoch": int(self.leo_min_folds_per_epoch),
            "leo_min_fold_fraction": float(self.leo_min_fold_fraction),
            "num_train_areas": int(len(train_areas)),
            "num_train_environments": int(len(candidate_groups)),
            "num_candidate_environments": int(len(candidate_groups)),
            "num_folds": int(len(heldout_group_names)),
            "leo_max_folds": int(self.leo_max_folds),
            "folds_capped": bool(self.leo_max_folds > 0 and self.leo_max_folds < len(candidate_groups)),
            "heldout_groups": heldout_group_names,
            "heldout_area_counts": heldout_area_counts,
            "mean_score_by_epoch": mean_score_by_epoch,
            "fold_counts_by_epoch": fold_counts_by_epoch,
            "retrained_on_all_train_envs": True,
        }
        return final_history

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        if self._use_leo_selection(train_areas):
            return self._fit_with_leo_selection(train_areas, val_areas)
        history = self._fit_once(
            train_areas,
            val_areas,
            max_epochs=self.epochs,
            patience_override=self.patience,
            selection_metric=self.checkpoint_metric,
            reset_model=True,
            track_best=True,
            emit_progress=True,
        )
        self.selection_summary_ = {
            "selection_protocol": self.selection_protocol if self.selection_protocol else "validation",
            "selection_metric": self.checkpoint_metric,
            "selected_epoch": int(history["selected_epoch"][0]) if history["selected_epoch"] else 0,
            "best_score": float(history["best_score"][0]) if history["best_score"] else float("inf"),
            "retrained_on_all_train_envs": False,
        }
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
            "opportunity_anchor_loss": float(parts["opportunity_anchor_loss_value"]),
            "opportunity_smooth_loss": float(parts["opportunity_smooth_loss_value"]),
            "crowding_shift_gate": float(parts["crowding_shift_gate"]),
            "residual_interaction_abs_mean": float(parts["residual_interaction_abs_mean"]),
            "uses_learned_opportunity_reconstruction": float(self.use_learned_opportunity_reconstruction),
            "uses_origin_gates": float(self.use_origin_gates),
            "uses_residual_interaction": float(self.use_residual_interaction),
            "uses_pair_residual_interaction": float(self.use_pair_residual_interaction),
            "uses_crowding_shift_gate": float(self.use_crowding_shift_gate),
            "rank_prefix_soft": float(self.rank_prefix_mode == "soft"),
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
                "rank_prefix_mode": self.rank_prefix_mode,
                "rank_prefix_temperature": self.rank_prefix_temperature,
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
                "use_learned_opportunity_reconstruction": self.use_learned_opportunity_reconstruction,
                "use_origin_gates": self.use_origin_gates,
                "use_residual_interaction": self.use_residual_interaction,
                "use_pair_residual_interaction": self.use_pair_residual_interaction,
                "use_crowding_shift_gate": self.use_crowding_shift_gate,
                "opp_recon_delta_scale": self.opp_recon_delta_scale,
                "origin_gate_rho": self.origin_gate_rho,
                "eta_residual": self.eta_residual,
                "residual_hidden_dim": self.residual_hidden_dim,
                "pair_residual_weight": self.pair_residual_weight,
                "pair_residual_hidden_dim": self.pair_residual_hidden_dim,
                "benefit_skip": self.benefit_skip,
                "deterrence_skip": self.deterrence_skip,
                "exposure_skip": self.exposure_skip,
                "competition_skip": self.competition_skip,
                "lambda_anchor": self.lambda_anchor,
                "lambda_smooth": self.lambda_smooth,
                "lambda_selection_scale": self.lambda_selection_scale,
                "selection_protocol": self.selection_protocol,
                "checkpoint_metric": self.checkpoint_metric,
                "leo_selection_metric": self.leo_selection_metric,
                "leo_max_epochs": self.leo_max_epochs,
                "leo_patience": self.leo_patience,
                "leo_max_folds": self.leo_max_folds,
                "leo_group_key": self.leo_group_key,
                "leo_fold_rank_prefix_mode": self.leo_fold_rank_prefix_mode,
                "leo_min_folds_per_epoch": self.leo_min_folds_per_epoch,
                "leo_min_fold_fraction": self.leo_min_fold_fraction,
                "crowding_gate_threshold": self.crowding_gate_threshold,
                "crowding_gate_temperature": self.crowding_gate_temperature,
                "crowding_fallback_weight": self.crowding_fallback_weight,
                "crowding_entropy_mean": self.crowding_entropy_mean,
                "crowding_entropy_std": self.crowding_entropy_std,
                "omega_selection": self.omega_selection,
                "cache_area_tensors": self.cache_area_tensors,
                "cache_area_max_mb": self.cache_area_max_bytes / 1024**2,
            },
            "selection_summary": self.selection_summary_,
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
        self.use_learned_opportunity_reconstruction = bool(
            config.get("use_learned_opportunity_reconstruction", self.use_learned_opportunity_reconstruction)
        ) and self.use_opportunity_field
        self.use_origin_gates = bool(config.get("use_origin_gates", self.use_origin_gates)) and self.use_sit_operator_potential
        self.use_residual_interaction = bool(config.get("use_residual_interaction", self.use_residual_interaction)) and self.use_sit_operator_potential
        self.use_pair_residual_interaction = bool(
            config.get("use_pair_residual_interaction", self.use_pair_residual_interaction)
        ) and self.use_residual_interaction
        self.use_crowding_shift_gate = bool(config.get("use_crowding_shift_gate", self.use_crowding_shift_gate)) and self.use_sit_operator_potential
        self.lambda_vrex = float(config.get("lambda_vrex", self.lambda_vrex))
        self.lambda_op = float(config.get("lambda_op", self.lambda_op))
        self.lambda_anchor = float(config.get("lambda_anchor", self.lambda_anchor))
        self.lambda_smooth = float(config.get("lambda_smooth", self.lambda_smooth))
        self.lambda_selection_scale = float(config.get("lambda_selection_scale", self.lambda_selection_scale))
        self.opp_recon_delta_scale = float(config.get("opp_recon_delta_scale", self.opp_recon_delta_scale))
        self.origin_gate_rho = float(config.get("origin_gate_rho", self.origin_gate_rho))
        self.eta_residual = float(config.get("eta_residual", self.eta_residual))
        self.residual_hidden_dim = int(config.get("residual_hidden_dim", self.residual_hidden_dim))
        self.pair_residual_weight = float(config.get("pair_residual_weight", self.pair_residual_weight))
        pair_hidden = config.get("pair_residual_hidden_dim", self.pair_residual_hidden_dim)
        self.pair_residual_hidden_dim = int(pair_hidden) if pair_hidden is not None else None
        self.benefit_skip = float(config.get("benefit_skip", self.benefit_skip))
        self.deterrence_skip = float(config.get("deterrence_skip", self.deterrence_skip))
        self.exposure_skip = float(config.get("exposure_skip", self.exposure_skip))
        self.competition_skip = float(config.get("competition_skip", self.competition_skip))
        self.selection_protocol = str(config.get("selection_protocol", self.selection_protocol)).lower()
        self.checkpoint_metric = str(config.get("checkpoint_metric", self.checkpoint_metric))
        self.leo_selection_metric = str(config.get("leo_selection_metric", self.leo_selection_metric))
        self.leo_max_epochs = int(config.get("leo_max_epochs", self.leo_max_epochs))
        self.leo_patience = int(config.get("leo_patience", self.leo_patience))
        self.leo_max_folds = int(config.get("leo_max_folds", self.leo_max_folds))
        self.leo_group_key = str(config.get("leo_group_key", self.leo_group_key))
        fold_rank = config.get("leo_fold_rank_prefix_mode", self.leo_fold_rank_prefix_mode)
        self.leo_fold_rank_prefix_mode = None if fold_rank in (None, "", "none", "same") else str(fold_rank).lower()
        self.leo_min_folds_per_epoch = int(config.get("leo_min_folds_per_epoch", self.leo_min_folds_per_epoch))
        self.leo_min_fold_fraction = float(config.get("leo_min_fold_fraction", self.leo_min_fold_fraction))
        self.crowding_gate_threshold = float(config.get("crowding_gate_threshold", self.crowding_gate_threshold))
        self.crowding_gate_temperature = float(config.get("crowding_gate_temperature", self.crowding_gate_temperature))
        self.crowding_fallback_weight = float(config.get("crowding_fallback_weight", self.crowding_fallback_weight))
        self.crowding_entropy_mean = float(config.get("crowding_entropy_mean", self.crowding_entropy_mean))
        self.crowding_entropy_std = float(config.get("crowding_entropy_std", self.crowding_entropy_std))
        self.omega_selection = str(config.get("omega_selection", self.omega_selection)).lower()
        self.mean_outflow = float(config.get("mean_outflow", self.mean_outflow))
        self.log1p_outflow_mean = float(config.get("log1p_outflow_mean", np.log1p(max(self.mean_outflow, 0.0))))
        self.log1p_outflow_std = float(config.get("log1p_outflow_std", self.log1p_outflow_std))
        self.rank_prefix_row_chunk_size = config.get("rank_prefix_row_chunk_size", self.rank_prefix_row_chunk_size)
        self.rank_prefix_mode = str(config.get("rank_prefix_mode", self.rank_prefix_mode)).lower()
        self.rank_prefix_temperature = float(config.get("rank_prefix_temperature", self.rank_prefix_temperature))
        self.cache_area_tensors = bool(config.get("cache_area_tensors", self.cache_area_tensors))
        self.cache_area_max_bytes = max(int(float(config.get("cache_area_max_mb", self.cache_area_max_bytes / 1024**2)) * 1024 * 1024), 0)
        self._clear_area_cache()
        self.selection_summary_ = checkpoint.get("selection_summary", self.selection_summary_)
        self.model = None
        self._ensure_model(int(checkpoint["node_dim"]))
        assert self.model is not None
        self._sync_model_runtime_fields()
        state = dict(checkpoint["model_state"])
        for module_name in ("deterrence", "benefit", "exposure", "competition"):
            old_prefix = f"{module_name}.net."
            new_prefix = f"{module_name}.net.net."
            for key in list(state):
                if key.startswith(old_prefix) and not key.startswith(new_prefix):
                    migrated = new_prefix + key[len(old_prefix) :]
                    state.setdefault(migrated, state[key])
        self.model.load_state_dict(state, strict=False)
