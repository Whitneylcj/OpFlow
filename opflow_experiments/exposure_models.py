"""Exposure-aware synthetic models for the final DGP four-question suite."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from opflow_experiments.models.torch_mlp import StandardScaler, resolve_device


def geo_columns(records: pd.DataFrame, include_raw_cost: bool = True) -> list[str]:
    cols = sorted([col for col in records.columns if col.startswith("x_geo_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    if not include_raw_cost:
        cols = [col for col in cols if col != "x_geo_4"]
    return cols


def res_columns(records: pd.DataFrame) -> list[str]:
    return sorted([col for col in records.columns if col.startswith("x_res_")], key=lambda x: int(x.rsplit("_", 1)[1]))


def origin_columns(records: pd.DataFrame) -> list[str]:
    return sorted([col for col in records.columns if col.startswith("x_origin_")], key=lambda x: int(x.rsplit("_", 1)[1]))


def deployed_pair_columns(records: pd.DataFrame, include_raw_cost: bool = True) -> list[str]:
    """Admissible deployed pair features for non-OpFlow baselines."""
    return origin_columns(records) + geo_columns(records, include_raw_cost=include_raw_cost)


@dataclass
class _RowLayout:
    row_id: np.ndarray
    slot: np.ndarray
    flat_index: np.ndarray
    first_index: np.ndarray
    n_rows: int
    k: int
    mask: np.ndarray


def _row_layout(records: pd.DataFrame) -> _RowLayout:
    keys = pd.MultiIndex.from_frame(records[["env_id", "origin_id"]])
    row_id, uniques = pd.factorize(keys, sort=False)
    counts = np.bincount(row_id)
    if len(counts) == 0:
        raise ValueError("Exposure models require at least one origin row.")
    n_rows = len(uniques)
    k = int(counts.max())
    slot = np.zeros(len(records), dtype=np.int64)
    seen = np.zeros(n_rows, dtype=np.int64)
    first_index = np.full(n_rows, -1, dtype=np.int64)
    for idx, rid in enumerate(row_id.astype(np.int64)):
        slot[idx] = seen[rid]
        seen[rid] += 1
        if first_index[rid] < 0:
            first_index[rid] = idx
    flat_index = row_id.astype(np.int64) * k + slot
    mask = np.zeros((n_rows, k), dtype=bool)
    mask.reshape(-1)[flat_index] = True
    return _RowLayout(
        row_id=row_id.astype(np.int64),
        slot=slot,
        flat_index=flat_index,
        first_index=first_index,
        n_rows=n_rows,
        k=k,
        mask=mask,
    )


def _row_scale_true(scale: torch.Tensor, row_first_index: torch.Tensor) -> torch.Tensor:
    return scale.index_select(0, row_first_index)


def _scatter_to_matrix(values: torch.Tensor, flat_index: torch.Tensor, n_rows: int, k: int, fill: float = 0.0) -> torch.Tensor:
    matrix = values.new_full((n_rows * k,), fill)
    matrix.index_copy_(0, flat_index, values.reshape(-1))
    return matrix.reshape(n_rows, k)


def _gather_from_matrix(matrix: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    return matrix.reshape(-1).index_select(0, flat_index)


def _masked_mean(matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(dim=1, keepdim=True).to(matrix.dtype), min=1.0)
    return (matrix * mask.to(matrix.dtype)).sum(dim=1, keepdim=True) / denom


def _masked_row_center(matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    centered = matrix - _masked_mean(matrix, mask)
    return centered.masked_fill(~mask, 0.0)


def _softmax_matrix(logits: torch.Tensor, tau: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        centered = logits - logits.max(dim=1, keepdim=True).values
        return torch.softmax(centered / max(float(tau), 1e-8), dim=1)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted = (masked_logits - row_max) / max(float(tau), 1e-8)
    exp = torch.exp(shifted).masked_fill(~mask, 0.0)
    return exp / torch.clamp(exp.sum(dim=1, keepdim=True), min=1e-8)


def _kl_matrix(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    true = torch.clamp(target, min=1e-8)
    hat = torch.clamp(pred, min=1e-8)
    term = true * (torch.log(true) - torch.log(hat))
    if mask is not None:
        term = term.masked_fill(~mask, 0.0)
    return term.sum(dim=1).mean()


def _row_kl_vector(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    true = torch.clamp(target, min=1e-8)
    hat = torch.clamp(pred, min=1e-8)
    term = true * (torch.log(true) - torch.log(hat))
    if mask is not None:
        term = term.masked_fill(~mask, 0.0)
    return term.sum(dim=1)


def _env_risks_from_rows(per_row: torch.Tensor, row_env: torch.Tensor) -> torch.Tensor:
    risks = []
    for env in torch.unique(row_env, sorted=True):
        env_mask = row_env == env
        if bool(env_mask.any()):
            risks.append(per_row[env_mask].mean())
    return torch.stack(risks) if risks else per_row.new_zeros((1,))


def _positive(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + 1e-8


def _scale_from_log_residual(raw: torch.Tensor, mean_scale: float) -> torch.Tensor:
    """Parameterize scale so a zero residual predicts the train mean scale."""
    base = float(np.log(max(float(mean_scale), 1e-8)))
    return torch.exp(torch.clamp(raw.reshape(-1) + base, min=-10.0, max=12.0))


def _float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).copy()


def _frame_array(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    return _float_array(frame.to_numpy(dtype=float))


def _safe_std(values: np.ndarray) -> float:
    return float(np.asarray(values, dtype=float).std(ddof=0) + 1e-6)


def _fit_component_standardizer(components: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        key: {"mean": float(np.asarray(values, dtype=float).mean()), "std": _safe_std(values)}
        for key, values in components.items()
    }


def _apply_component_standardizer(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(stats["mean"])) / max(float(stats["std"]), 1e-8)


def _proxy_rank_prefix_numpy(records: pd.DataFrame, *, mode: str, temperature: float) -> np.ndarray:
    out = np.zeros(len(records), dtype=float)
    temp = max(float(temperature), 1e-6)
    for _, positions in records.groupby(["env_id", "origin_id"], sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        group = records.iloc[idx]
        opp = group["x_geo_3"].to_numpy(dtype=float)
        cost = group["cost"].to_numpy(dtype=float)
        if mode == "soft":
            weights = 1.0 / (1.0 + np.exp(-(cost[:, None] - cost[None, :]) / temp))
            np.fill_diagonal(weights, 0.0)
            exposure = weights @ opp
        else:
            order = np.argsort(cost, kind="mergesort")
            sorted_opp = opp[order]
            prefix = np.cumsum(sorted_opp) - sorted_opp
            exposure = np.zeros_like(opp)
            exposure[order] = prefix
        out[idx] = exposure
    return out


def _proxy_local_field_numpy(records: pd.DataFrame, omega: float) -> np.ndarray:
    unique = records.groupby(["env_id", "dest_id"], sort=False).agg(
        opp=("x_geo_3", "first"),
        x=("x_geo_0", "first"),
        y=("x_geo_1", "first"),
    )
    local_values: dict[tuple, float] = {}
    for env_id, group in unique.groupby(level=0, sort=False):
        opp = group["opp"].to_numpy(dtype=float)
        xy = group[["x", "y"]].to_numpy(dtype=float)
        dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        weights = np.exp(-dist / max(float(omega), 1e-8))
        np.fill_diagonal(weights, 0.0)
        local = weights @ opp
        for (_, dest_id), value in zip(group.index, local):
            local_values[(env_id, dest_id)] = float(value)
    return np.asarray([local_values[(row.env_id, row.dest_id)] for row in records.itertuples(index=False)], dtype=float)


def _proxy_components_numpy(records: pd.DataFrame, *, rank_prefix_mode: str, rank_prefix_temperature: float, omega: float) -> dict[str, np.ndarray]:
    proxy_opp = np.maximum(records["x_geo_3"].to_numpy(dtype=float), 1e-8)
    proxy_prefix = np.maximum(
        _proxy_rank_prefix_numpy(records, mode=rank_prefix_mode, temperature=rank_prefix_temperature),
        0.0,
    )
    proxy_local = np.maximum(_proxy_local_field_numpy(records, omega), 0.0)
    return {
        "O": np.log1p(proxy_opp),
        "C": records["cost"].to_numpy(dtype=float),
        "S": np.log1p(proxy_prefix),
        "L": np.log1p(proxy_local),
    }


def _float_tensor(values, *, device=None) -> torch.Tensor:
    return torch.as_tensor(_float_array(values), dtype=torch.float32, device=device)


def _smoothed_target(tensors: dict, alpha: float) -> torch.Tensor:
    n_rows = tensors["n_rows"]
    k = tensors["k"]
    mask = tensors["candidate_mask"]
    y_m = _scatter_to_matrix(tensors["y"], tensors["flat_index"], n_rows, k, fill=0.0)
    counts = torch.clamp(mask.sum(dim=1, keepdim=True).to(y_m.dtype), min=1.0)
    target = (y_m + float(alpha) / counts) / torch.clamp(y_m.sum(dim=1, keepdim=True) + float(alpha), min=1e-8)
    return target.masked_fill(~mask, 0.0)


def _prediction_dict(
    lam: torch.Tensor,
    pi: torch.Tensor,
    score: torch.Tensor,
    extra: dict | None = None,
    *,
    score_kind: str = "choice_potential",
) -> dict:
    extra_payload = {"score_hat": score.detach().cpu(), "score_kind": score_kind}
    extra_payload.update(extra or {})
    return {
        "lambda_hat": lam.detach().cpu(),
        "pi_hat": pi.detach().cpu(),
        "z": None,
        "extra": extra_payload,
    }


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coeff * grad_output, None


def _grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
    return _GradientReverseFn.apply(x, float(coeff))


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
            last = self.net[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PairAllocationNet(nn.Module):
    def __init__(self, input_dim: int, origin_dim: int, hidden_dim: int, with_scale: bool, num_domains: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.scale = _MLP(origin_dim, hidden_dim, 1) if with_scale else None
        self.domain_head = nn.Linear(hidden_dim, max(int(num_domains), 1)) if int(num_domains) > 1 else None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.features(x))

    def score_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_head(features)


class ExposureBaselineModel:
    """Conventional MLP baselines over configured feature groups."""

    def __init__(self, method: str, config: dict, seed: int) -> None:
        self.method = method
        cfg = config.get("exposure_model", {})
        self.hidden_dim = int(cfg.get("hidden_dim", 48))
        self.epochs = int(cfg.get("epochs", 100))
        self.lr = float(cfg.get("learning_rate", 0.003))
        self.patience = int(cfg.get("patience", 15))
        self.lambda_count = float(cfg.get("lambda_count", 0.05))
        self.lambda_scale = float(cfg.get("lambda_scale", 0.1))
        self.lambda_vrex = float(cfg.get("lambda_vrex", 0.1))
        self.lambda_irm = float(cfg.get("lambda_irm", 0.1))
        self.lambda_dann = float(cfg.get("lambda_dann", 0.1))
        self.lambda_selection_scale = float(cfg.get("lambda_selection_scale", 0.1))
        self.baseline_groupdro_eta = float(cfg.get("baseline_groupdro_eta", 0.05))
        self.dann_warmup_epochs = int(cfg.get("dann_warmup_epochs", 0))
        self.dann_grl_max = float(cfg.get("dann_grl_max", 1.0))
        self.raw_count_loss = str(cfg.get("raw_count_loss", "mse")).lower()
        self.alpha = float(cfg.get("alpha_smooth", 0.5))
        self.tau = float(config.get("dgp", {}).get("tau", config.get("tau", 0.7)))
        self.device = resolve_device(cfg.get("device", "cpu"))
        self._raw_count_methods = {"RawCountMLP", "RawCountGNN"}
        self._scale_aware_methods = {"ScaleAllocationMLP", "ScaleAwareMLP", "ScaleAwareMLPAllocSelected"}
        self.feature_scaler = StandardScaler()
        self.origin_scaler = StandardScaler()
        self.columns: list[str] = []
        self.origin_cols: list[str] = []
        self.net: _PairAllocationNet | None = None
        self.train_mean_scale = 1.0
        self.scale_target_mean = 1.0
        self.share_lookup: dict[tuple[int, int], float] = {}
        self.dest_share_lookup: dict[int, float] = {}
        self.global_share = 1.0
        self.num_train_envs = 1
        self.groupdro_weights: torch.Tensor | None = None
        self.selected_epoch_: int | None = None
        self.selection_summary_: dict = {}
        torch.manual_seed(seed)

    def _columns_for_method(self, records: pd.DataFrame) -> list[str]:
        if self.method == "ShortcutOnly":
            cols = res_columns(records)
            if not cols:
                raise ValueError("ShortcutOnly requires explicit x_res_* columns, which are not part of the main DGP.")
            return cols
        if self.method in {"RawCountMLP", "RawCountGNN"}:
            return deployed_pair_columns(records)
        if self.method in self._scale_aware_methods:
            return deployed_pair_columns(records)
        if self.method == "ExposureMLPNoS":
            return origin_columns(records) + [col for col in geo_columns(records) if col not in {"x_geo_2"}]
        return deployed_pair_columns(records)

    def fit(self, train_records: pd.DataFrame, val_records: pd.DataFrame) -> dict:
        row_scale = train_records.groupby(["env_id", "origin_id"], sort=False)["y_obs"].sum()
        self.train_mean_scale = float(row_scale.mean())
        if "R_oracle" in train_records:
            true_scale = train_records.groupby(["env_id", "origin_id"], sort=False)["R_oracle"].first()
            self.scale_target_mean = float(true_scale.mean())
        else:
            self.scale_target_mean = self.train_mean_scale
        if self.method in {"GravityDirect", "GravityOnly", "OraclePotential", "InterveningOpportunity", "Radiation", "CompetingDestination"}:
            self.selection_summary_ = {"selection_protocol": "closed_form", "selection_metric": "none", "selected_epoch": 0}
            return {"train_loss": [], "val_loss": []}
        if self.method in {"ShareStable", "LastShare"}:
            working = train_records.copy()
            grouped = working.groupby(["env_id", "origin_id"], sort=False)
            row_sum = grouped["y_obs"].transform("sum").to_numpy(dtype=float)
            row_count = np.maximum(grouped["dest_id"].transform("count").to_numpy(dtype=float), 1.0)
            working["_share"] = (working["y_obs"].to_numpy(dtype=float) + self.alpha / row_count) / np.maximum(row_sum + self.alpha, 1e-8)
            by_pair = working.groupby(["origin_id", "dest_id"], sort=False)["_share"].mean()
            by_dest = working.groupby("dest_id", sort=False)["_share"].mean()
            self.share_lookup = {(int(i), int(j)): float(v) for (i, j), v in by_pair.items()}
            self.dest_share_lookup = {int(j): float(v) for j, v in by_dest.items()}
            self.global_share = float(working["_share"].mean())
            self.selection_summary_ = {"selection_protocol": "empirical_share", "selection_metric": "none", "selected_epoch": 0}
            return {"train_loss": [], "val_loss": []}
        self.columns = self._columns_for_method(train_records)
        self.origin_cols = origin_columns(train_records)
        x_train = self.feature_scaler.fit_transform(_frame_array(train_records[self.columns]))
        x_val = self.feature_scaler.transform(_frame_array(val_records[self.columns]))
        o_train = self.origin_scaler.fit_transform(_frame_array(train_records[self.origin_cols]))
        o_val = self.origin_scaler.transform(_frame_array(val_records[self.origin_cols]))
        with_scale = self.method in self._scale_aware_methods
        self.num_train_envs = int(train_records["env_id"].nunique())
        self.net = _PairAllocationNet(
            len(self.columns),
            len(self.origin_cols),
            self.hidden_dim,
            with_scale=with_scale,
            num_domains=self.num_train_envs if self.method == "DANN-AllocMLP" else 1,
        ).to(self.device)
        train_tensors = self._make_tensors(train_records, x_train, o_train)
        val_tensors = self._make_tensors(val_records, x_val, o_val)
        self.groupdro_weights = torch.full(
            (max(int(torch.unique(train_tensors["row_env"]).numel()), 1),),
            1.0 / max(int(torch.unique(train_tensors["row_env"]).numel()), 1),
            dtype=torch.float32,
            device=self.device,
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        best_state = copy.deepcopy(self.net.state_dict())
        best_val = float("inf")
        best_epoch = 0
        stale = 0
        selects_raw_count = self.method in self._raw_count_methods
        selects_joint_scale = self.method in self._scale_aware_methods and self.method != "ScaleAwareMLPAllocSelected"
        if selects_raw_count:
            selection_metric = "raw_count_loss"
        elif selects_joint_scale:
            selection_metric = "joint_selection_score"
        else:
            selection_metric = "alloc_loss"
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_alloc_loss": [],
            "val_alloc_loss": [],
            "train_count_loss": [],
            "val_count_loss": [],
            "selection_score": [],
        }
        for epoch in range(self.epochs):
            self.net.train()
            loss, _ = self._loss(
                train_tensors,
                epoch=epoch,
                update_groupdro=self.method == "GroupDRO-AllocMLP",
                include_domain=self.method == "DANN-AllocMLP",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.net.eval()
            with torch.no_grad():
                train_loss_t, train_pred = self._loss(train_tensors)
                val_loss_t, val_pred = self._loss(val_tensors)
                train_loss = float(train_loss_t.item())
                val_loss = float(val_loss_t.item())
                train_alloc = float(train_pred["alloc_loss"].item())
                val_alloc = float(val_pred["alloc_loss"].item())
                train_count = float(train_pred["count_loss"].item())
                val_count = float(val_pred["count_loss"].item())
                if selects_raw_count:
                    selection_score = val_count
                elif selects_joint_scale:
                    selection_score = float(val_pred["joint_selection_score"].item())
                else:
                    selection_score = val_alloc
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_alloc_loss"].append(train_alloc)
            history["val_alloc_loss"].append(val_alloc)
            history["train_count_loss"].append(train_count)
            history["val_count_loss"].append(val_count)
            history["selection_score"].append(selection_score)
            if selection_score < best_val - 1e-6:
                best_val = selection_score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.selected_epoch_ = max(best_epoch, 1)
        self.selection_summary_ = {
            "selection_protocol": "validation",
            "selection_metric": selection_metric,
            "lambda_selection_scale": float(self.lambda_selection_scale),
            "selected_epoch": self.selected_epoch_,
            "best_score": float(best_val),
            "retrained_on_all_train_envs": False,
        }
        return history

    def _make_tensors(self, records: pd.DataFrame, x: np.ndarray, origin: np.ndarray) -> dict:
        layout = _row_layout(records)
        env_codes, env_uniques = pd.factorize(records["env_id"], sort=False)
        row_env = np.zeros(layout.n_rows, dtype=np.int64)
        for idx, rid in enumerate(layout.row_id):
            row_env[rid] = env_codes[idx]
        return {
            "x": _float_tensor(x, device=self.device),
            "origin": _float_tensor(origin, device=self.device),
            "y": _float_tensor(records["y_obs"].to_numpy(dtype=float), device=self.device),
            "scale": _float_tensor(records["R_oracle"].to_numpy(dtype=float), device=self.device),
            "env": torch.as_tensor(env_codes, dtype=torch.long, device=self.device),
            "row_env": torch.as_tensor(row_env, dtype=torch.long, device=self.device),
            "num_envs": len(env_uniques),
            "flat_index": torch.as_tensor(layout.flat_index, dtype=torch.long, device=self.device),
            "row_first_index": torch.as_tensor(layout.first_index, dtype=torch.long, device=self.device),
            "candidate_mask": torch.as_tensor(layout.mask, dtype=torch.bool, device=self.device),
            "n_rows": layout.n_rows,
            "k": layout.k,
        }

    def _dann_coeff(self, epoch: int | None) -> float:
        if epoch is None:
            return 0.0
        if epoch < self.dann_warmup_epochs:
            return 0.0
        remaining = max(self.epochs - self.dann_warmup_epochs, 1)
        progress = min(max((epoch - self.dann_warmup_epochs + 1) / remaining, 0.0), 1.0)
        return float(self.dann_grl_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))

    def _irm_penalty(self, score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, row_env: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return score.new_tensor(0.0)
        scale = torch.tensor(1.0, dtype=score.dtype, device=score.device, requires_grad=True)
        pi = _softmax_matrix(score * scale, self.tau, mask)
        per_row = _row_kl_vector(target, pi, mask)
        penalties = []
        for env_risk in _env_risks_from_rows(per_row, row_env):
            grad = torch.autograd.grad(env_risk, [scale], create_graph=True, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                penalties.append(grad.square())
        return torch.stack(penalties).mean() if penalties else score.new_tensor(0.0)

    def _loss(
        self,
        tensors: dict,
        *,
        epoch: int | None = None,
        update_groupdro: bool = False,
        include_domain: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        assert self.net is not None
        n_rows = tensors["n_rows"]
        k = tensors["k"]
        mask = tensors["candidate_mask"]
        flat_index = tensors["flat_index"]
        features = self.net.features(tensors["x"])
        score_flat = self.net.score_from_features(features).reshape(-1)
        score = _scatter_to_matrix(score_flat, flat_index, n_rows, k, fill=0.0)
        pi = _softmax_matrix(score, self.tau, mask)
        target = _smoothed_target(tensors, self.alpha)
        per_row_alloc = _row_kl_vector(target, pi, mask)
        alloc_loss = per_row_alloc.mean()
        env_alloc = _env_risks_from_rows(per_row_alloc, tensors["row_env"])
        env_alloc_var = env_alloc.var(unbiased=False) if env_alloc.numel() > 1 else env_alloc.new_tensor(0.0)
        irm_penalty = score.new_tensor(0.0)
        domain_loss = score.new_tensor(0.0)
        if self.method in self._scale_aware_methods:
            row_origin = tensors["origin"].index_select(0, tensors["row_first_index"])
            row_scale = _scale_from_log_residual(self.net.scale(row_origin).reshape(-1), self.scale_target_mean)
            row_true = _row_scale_true(tensors["scale"], tensors["row_first_index"])
            scale_loss = F.mse_loss(torch.log(row_scale), torch.log(torch.clamp(row_true, min=1e-8)))
            lam = row_scale[:, None] * pi.detach()
            count_loss = F.mse_loss(
                torch.log1p(_gather_from_matrix(lam, flat_index)),
                torch.log1p(tensors["y"]),
            )
            loss = alloc_loss + self.lambda_scale * scale_loss + self.lambda_count * count_loss
        else:
            scale_loss = score.new_tensor(0.0)
            row_scale = torch.full((n_rows,), self.train_mean_scale, dtype=pi.dtype, device=pi.device)
            lam = row_scale[:, None] * pi
            if self.method in self._raw_count_methods:
                raw_lam_flat = _positive(score_flat)
                raw_lam = _scatter_to_matrix(raw_lam_flat, flat_index, n_rows, k, fill=0.0)
                raw_lam = raw_lam.masked_fill(~mask, 0.0)
                pi = raw_lam / torch.clamp(raw_lam.sum(dim=1, keepdim=True), min=1e-8)
                lam = raw_lam
                per_row_alloc = _row_kl_vector(target, pi, mask)
                alloc_loss = per_row_alloc.mean()
                # RawCountMLP is intentionally supervised on raw OD counts, not
                # row-normalized allocation targets.
                if self.raw_count_loss == "poisson_nll":
                    loss = F.poisson_nll_loss(torch.clamp(raw_lam_flat, min=1e-8), tensors["y"], log_input=False, full=False)
                elif self.raw_count_loss == "log_mse":
                    loss = F.mse_loss(torch.log1p(raw_lam_flat), torch.log1p(tensors["y"]))
                else:
                    loss = F.mse_loss(raw_lam_flat, tensors["y"])
                count_loss = loss
            else:
                count_loss = F.mse_loss(
                    torch.log1p(_gather_from_matrix(lam, flat_index)),
                    torch.log1p(tensors["y"]),
                )
                alloc_objective = alloc_loss
                if self.method == "GroupDRO-AllocMLP":
                    if update_groupdro and self.groupdro_weights is not None and self.groupdro_weights.numel() == env_alloc.numel():
                        with torch.no_grad():
                            self.groupdro_weights *= torch.exp(float(self.baseline_groupdro_eta) * env_alloc.detach())
                            self.groupdro_weights /= torch.clamp(self.groupdro_weights.sum(), min=1e-8)
                    if self.groupdro_weights is not None and self.groupdro_weights.numel() == env_alloc.numel():
                        alloc_objective = (self.groupdro_weights.detach() * env_alloc).sum()
                    elif env_alloc.numel() > 0:
                        alloc_objective = env_alloc.max()
                elif self.method == "VREx-AllocMLP":
                    alloc_objective = env_alloc.mean() + self.lambda_vrex * env_alloc_var
                elif self.method == "IRM-AllocMLP":
                    irm_penalty = self._irm_penalty(score, target, mask, tensors["row_env"])
                    alloc_objective = alloc_loss + self.lambda_irm * irm_penalty
                elif self.method == "DANN-AllocMLP" and include_domain and self.net.domain_head is not None:
                    coeff = self._dann_coeff(epoch)
                    if coeff > 0.0 and int(tensors["env"].max().item()) < self.net.domain_head.out_features:
                        domain_logits = self.net.domain_head(_grad_reverse(features, coeff))
                        domain_loss = F.cross_entropy(domain_logits, tensors["env"])
                loss = alloc_objective + self.lambda_count * count_loss + self.lambda_dann * domain_loss
        return loss, {
            "lambda_hat": _gather_from_matrix(lam, flat_index),
            "pi_hat": _gather_from_matrix(pi, flat_index),
            "score_hat": score_flat,
            "row_scale_hat": row_scale,
            "alloc_loss": alloc_loss,
            "scale_loss": scale_loss,
            "count_loss": count_loss,
            "joint_selection_score": alloc_loss + float(self.lambda_selection_scale) * scale_loss,
            "env_alloc_var": env_alloc_var,
            "irm_penalty": irm_penalty,
            "domain_loss": domain_loss,
        }

    def _gravity_prediction(self, records: pd.DataFrame) -> dict:
        layout = _row_layout(records)
        flat_index = torch.as_tensor(layout.flat_index, dtype=torch.long)
        mask = torch.as_tensor(layout.mask, dtype=torch.bool)
        log_opp = _float_tensor(records["x_geo_3"].to_numpy(dtype=float))
        cost = _float_tensor(records["x_geo_4"].to_numpy(dtype=float))
        score_flat = log_opp - 0.35 * cost
        score = _scatter_to_matrix(score_flat, flat_index, layout.n_rows, layout.k, fill=0.0)
        pi = _softmax_matrix(score, self.tau, mask)
        row_scale = torch.full((layout.n_rows, 1), self.train_mean_scale, dtype=pi.dtype)
        lam = row_scale * pi
        return _prediction_dict(
            _gather_from_matrix(lam, flat_index),
            _gather_from_matrix(pi, flat_index),
            score_flat,
            {"tau": self.tau},
        )

    def _intervening_opportunity_prediction(self, records: pd.DataFrame) -> dict:
        layout = _row_layout(records)
        flat_index = torch.as_tensor(layout.flat_index, dtype=torch.long)
        mask = torch.as_tensor(layout.mask, dtype=torch.bool)
        opp_flat = _float_tensor(records["x_geo_3"].to_numpy(dtype=float))
        cost_flat = _float_tensor(records["x_geo_4"].to_numpy(dtype=float))
        opp = _scatter_to_matrix(opp_flat, flat_index, layout.n_rows, layout.k, fill=0.0)
        cost = _scatter_to_matrix(cost_flat, flat_index, layout.n_rows, layout.k, fill=0.0)
        exposure = torch.zeros_like(opp)
        order = torch.argsort(cost.masked_fill(~mask, torch.inf), dim=1, stable=True)
        sorted_opp = torch.gather(opp, 1, order)
        prefix = torch.cumsum(sorted_opp, dim=1) - sorted_opp
        exposure.scatter_(1, order, prefix)
        score = opp - torch.log1p(torch.clamp(exposure, min=0.0)) - 0.2 * cost
        pi = _softmax_matrix(score, self.tau, mask)
        row_scale = torch.full((layout.n_rows, 1), self.train_mean_scale, dtype=pi.dtype)
        lam = row_scale * pi
        return _prediction_dict(
            _gather_from_matrix(lam, flat_index),
            _gather_from_matrix(pi, flat_index),
            _gather_from_matrix(score, flat_index),
            {"tau": self.tau},
        )

    def _competing_destination_prediction(self, records: pd.DataFrame) -> dict:
        layout = _row_layout(records)
        flat_index = torch.as_tensor(layout.flat_index, dtype=torch.long)
        mask = torch.as_tensor(layout.mask, dtype=torch.bool)
        opp = _scatter_to_matrix(_float_tensor(records["x_geo_3"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        cost = _scatter_to_matrix(_float_tensor(records["x_geo_4"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        dest_x = _scatter_to_matrix(_float_tensor(records["x_geo_0"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        dest_y = _scatter_to_matrix(_float_tensor(records["x_geo_1"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        xy = torch.stack([dest_x, dest_y], dim=2)
        pair_dist = torch.cdist(xy, xy)
        weights = torch.exp(-pair_dist / 0.15)
        eye = torch.eye(layout.k, dtype=torch.bool, device=weights.device).unsqueeze(0)
        weights = weights.masked_fill(eye, 0.0)
        weights = weights * mask.unsqueeze(1).to(weights.dtype) * mask.unsqueeze(2).to(weights.dtype)
        competition = (weights * opp.unsqueeze(1)).sum(dim=2).masked_fill(~mask, 0.0)
        score = torch.log1p(torch.clamp(opp, min=0.0)) + 0.35 * torch.log1p(torch.clamp(competition, min=0.0)) - 0.35 * cost
        pi = _softmax_matrix(score, self.tau, mask)
        row_scale = torch.full((layout.n_rows, 1), self.train_mean_scale, dtype=pi.dtype)
        lam = row_scale * pi
        return _prediction_dict(
            _gather_from_matrix(lam, flat_index),
            _gather_from_matrix(pi, flat_index),
            _gather_from_matrix(score, flat_index),
            {"tau": self.tau},
        )

    def _radiation_prediction(self, records: pd.DataFrame) -> dict:
        layout = _row_layout(records)
        flat_index = torch.as_tensor(layout.flat_index, dtype=torch.long)
        mask = torch.as_tensor(layout.mask, dtype=torch.bool)
        opp = _scatter_to_matrix(_float_tensor(records["x_geo_3"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        cost = _scatter_to_matrix(_float_tensor(records["x_geo_4"].to_numpy(dtype=float)), flat_index, layout.n_rows, layout.k, fill=0.0)
        exposure = torch.zeros_like(opp)
        order = torch.argsort(cost.masked_fill(~mask, torch.inf), dim=1, stable=True)
        sorted_opp = torch.gather(opp, 1, order)
        prefix = torch.cumsum(sorted_opp, dim=1) - sorted_opp
        exposure.scatter_(1, order, prefix)
        scale = torch.full((layout.n_rows, 1), self.train_mean_scale, dtype=opp.dtype)
        score = torch.log(torch.clamp(opp, min=1e-8)) - torch.log1p(torch.clamp(scale + exposure, min=0.0))
        pi = _softmax_matrix(score, self.tau, mask)
        lam = scale * pi
        return _prediction_dict(
            _gather_from_matrix(lam, flat_index),
            _gather_from_matrix(pi, flat_index),
            _gather_from_matrix(score, flat_index),
            {"tau": self.tau},
        )

    def _share_stable_prediction(self, records: pd.DataFrame) -> dict:
        layout = _row_layout(records)
        flat_index = torch.as_tensor(layout.flat_index, dtype=torch.long)
        values = []
        for row in records.itertuples(index=False):
            key = (int(row.origin_id), int(row.dest_id))
            values.append(self.share_lookup.get(key, self.dest_share_lookup.get(int(row.dest_id), self.global_share)))
        raw = _scatter_to_matrix(torch.as_tensor(values, dtype=torch.float32), flat_index, layout.n_rows, layout.k, fill=0.0)
        mask = torch.as_tensor(layout.mask, dtype=torch.bool)
        raw = raw.masked_fill(~mask, 0.0)
        pi = raw / torch.clamp(raw.sum(dim=1, keepdim=True), min=1e-8)
        row_scale = torch.full((layout.n_rows, 1), self.train_mean_scale, dtype=pi.dtype)
        score = torch.log(torch.clamp(pi, min=1e-8))
        lam = row_scale * pi
        return _prediction_dict(
            _gather_from_matrix(lam, flat_index),
            _gather_from_matrix(pi, flat_index),
            _gather_from_matrix(score, flat_index),
            {"tau": self.tau},
            score_kind="allocation_log_share",
        )

    def predict(self, records: pd.DataFrame) -> dict:
        if self.method == "OraclePotential":
            return _prediction_dict(
                _float_tensor(records["lambda_oracle"].to_numpy(dtype=float)),
                _float_tensor(records["pi_oracle"].to_numpy(dtype=float)),
                _float_tensor(records["psi_oracle"].to_numpy(dtype=float)),
                {"tau": self.tau},
            )
        if self.method in {"GravityDirect", "GravityOnly"}:
            return self._gravity_prediction(records)
        if self.method == "InterveningOpportunity":
            return self._intervening_opportunity_prediction(records)
        if self.method == "CompetingDestination":
            return self._competing_destination_prediction(records)
        if self.method == "Radiation":
            return self._radiation_prediction(records)
        if self.method in {"ShareStable", "LastShare"}:
            return self._share_stable_prediction(records)
        x = self.feature_scaler.transform(_frame_array(records[self.columns]))
        origin = self.origin_scaler.transform(_frame_array(records[self.origin_cols]))
        tensors = self._make_tensors(records, x, origin)
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            _, pred = self._loss(tensors)
        score_kind = "raw_count_rate" if self.method in self._raw_count_methods else "choice_potential"
        return _prediction_dict(
            pred["lambda_hat"],
            pred["pi_hat"],
            pred["score_hat"],
            {"row_scale_hat": pred["row_scale_hat"].detach().cpu(), "tau": self.tau},
            score_kind=score_kind,
        )


class _ScalarMLP(nn.Module):
    def __init__(self, hidden_dim: int, *, skip: float = 0.0, zero_last: bool = True) -> None:
        super().__init__()
        self.skip = float(skip)
        self.net = _MLP(1, hidden_dim, 1, zero_last=zero_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x.reshape(-1, 1)).reshape_as(x)
        return self.skip * x + residual


class _ExposureOpFlowNet(nn.Module):
    """P-only latent SIT operator network for synthetic OpFlow."""

    def __init__(
        self,
        p_dest_dim: int,
        p_pair_dim: int,
        p_geo_dim: int,
        origin_dim: int,
        hidden_dim: int,
        *,
        benefit_skip: float,
        deterrence_skip: float,
        exposure_skip: float,
        competition_skip: float,
        residual_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.opp_net = _MLP(p_dest_dim, hidden_dim, 1)
        self.impedance_net = _MLP(origin_dim + p_dest_dim + p_pair_dim + p_geo_dim, hidden_dim, 1)
        self.deterrence = _ScalarMLP(hidden_dim, skip=abs(float(deterrence_skip)))
        self.benefit = _ScalarMLP(hidden_dim, skip=abs(float(benefit_skip)))
        self.exposure = _ScalarMLP(hidden_dim, skip=abs(float(exposure_skip)))
        self.local_field = _ScalarMLP(hidden_dim, skip=abs(float(competition_skip)))
        self.coefficient_head = _MLP(origin_dim, hidden_dim, 4, zero_last=True)
        residual_input_dim = origin_dim + 4 + p_geo_dim
        self.residual_interaction = _MLP(residual_input_dim, max(int(residual_hidden_dim), 4), 1, zero_last=True)
        self.generic_potential = _MLP(origin_dim + p_dest_dim + p_pair_dim + p_geo_dim, hidden_dim, 1)
        self.scale_encoder = _MLP(4, hidden_dim, 3)
        self.scale_net = _MLP(origin_dim + 3, hidden_dim, 1)

    def allocation_parameters(self):
        yield from self.opp_net.parameters()
        yield from self.impedance_net.parameters()
        yield from self.deterrence.parameters()
        yield from self.benefit.parameters()
        yield from self.exposure.parameters()
        yield from self.local_field.parameters()
        yield from self.coefficient_head.parameters()
        yield from self.residual_interaction.parameters()
        yield from self.generic_potential.parameters()


@dataclass
class ExposureAudit:
    count_scale_potential_grad_max: float
    row_center_mean_abs: float
    op_monotone_violation_D: float
    op_monotone_violation_S: float
    op_monotone_violation_B: float


class ExposureOpFlowModel:
    """Synthetic P-only latent SIT operator OpFlow."""

    def __init__(self, method: str, config: dict, seed: int) -> None:
        self.method = method
        cfg = config.get("exposure_model", {})
        self.hidden_dim = int(cfg.get("hidden_dim", 48))
        self.epochs = int(cfg.get("epochs", 100))
        self.lr = float(cfg.get("learning_rate", 0.003))
        self.patience = int(cfg.get("patience", 15))
        self.alpha = float(cfg.get("alpha_smooth", 0.5))
        self.lambda_count = float(cfg.get("lambda_count", 0.05))
        self.lambda_scale = float(cfg.get("lambda_scale", 0.1))
        self.lambda_vrex = float(cfg.get("lambda_vrex", 0.1))
        self.lambda_op = float(cfg.get("lambda_op", 0.01))
        self.groupdro_weight = float(cfg.get("groupdro_weight", 0.005))
        self.lambda_operator_anchor = float(cfg.get("lambda_operator_anchor", 0.0))
        self.lambda_o_anchor = float(cfg.get("lambda_o_anchor", 1.0))
        self.lambda_c_anchor = float(cfg.get("lambda_c_anchor", 1.0))
        self.lambda_selection_scale = float(cfg.get("lambda_selection_scale", 0.1))
        self.eta_residual = float(cfg.get("eta_residual", 0.2))
        self.residual_hidden_dim = int(cfg.get("residual_hidden_dim", max(int(self.hidden_dim // 4), 4)))
        self.checkpoint_metric = str(cfg.get("checkpoint_metric", "joint_selection_score"))
        self.selection_protocol = str(cfg.get("selection_protocol", "leo")).lower()
        self.leo_selection_metric = str(cfg.get("leo_selection_metric", "joint_selection_score"))
        self.rank_prefix_mode = str(cfg.get("rank_prefix_mode", "hard")).lower()
        if self.rank_prefix_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown rank_prefix_mode: {self.rank_prefix_mode}")
        self.rank_prefix_temperature = float(cfg.get("rank_prefix_temperature", 0.05))
        mech_cfg = config.get("dgp", config.get("mechanism", {}))
        self.benefit_skip = float(cfg.get("benefit_skip", mech_cfg.get("beta_o", 1.15)))
        self.deterrence_skip = float(cfg.get("deterrence_skip", -float(mech_cfg.get("beta_c", 0.35))))
        self.exposure_skip = float(cfg.get("exposure_skip", -float(mech_cfg.get("beta_s", 0.65))))
        self.competition_skip = float(cfg.get("competition_skip", 0.0))
        self.use_crowding_shift_gate = (
            bool(cfg.get("use_crowding_shift_gate", False))
            and method not in {"NoCompetition", "NoExposureOps"}
        )
        self.crowding_ref_corr_threshold = float(cfg.get("crowding_ref_corr_threshold", 0.8))
        self.crowding_ref_corr_temperature = float(cfg.get("crowding_ref_corr_temperature", 0.05))
        self.crowding_fallback_weight = float(cfg.get("crowding_fallback_weight", 1.0))
        self.tau = float(config.get("dgp", {}).get("tau", config.get("tau", 0.7)))
        self.device = resolve_device(cfg.get("device", "cpu"))
        self.use_rank_exposure = bool(cfg.get("use_rank_exposure", True)) and method not in {"NoRankExposure", "NoExposureOps"}
        self.use_competition = bool(cfg.get("use_competition", True)) and method not in {"NoCompetition", "NoExposureOps"}
        self.use_opportunity_field = bool(cfg.get("use_opportunity_field", True)) and method != "NoOpportunityField"
        self.use_row_center = bool(cfg.get("use_row_center", True)) and method != "NoRowCenter"
        self.detach_count_allocation = bool(cfg.get("detach_count_allocation", True)) and method != "NoScaleDetach"
        self.use_groupdro = bool(cfg.get("use_groupdro", True)) and method not in {"NoGroupDRO", "OpFlowERM"}
        if not self.use_groupdro:
            self.groupdro_weight = 0.0
        self.use_vrex = bool(cfg.get("use_vrex", True)) and method not in {"NoVREX", "OpFlowERM"}
        self.use_monotone_penalty = bool(cfg.get("use_monotone_penalty", True)) and method != "NoMonotonePenalty"
        self.use_sit_operator_potential = bool(cfg.get("use_sit_operator_potential", True)) and method != "GenericPotentialMLP"
        self.use_residual_interaction = bool(cfg.get("use_residual_interaction", True)) and method != "NoResidualInteraction"
        if method in {"OpFlowAllocSelected", "OpFlow-AllocSelected"}:
            self.checkpoint_metric = "alloc_loss"
            self.leo_selection_metric = "alloc_loss"
        self.competition_omega = float(config.get("dgp", {}).get("omega", 0.15))
        self.component_standardizer: dict[str, dict[str, float]] = {}
        self.geo_scaler = StandardScaler()
        self.origin_scaler = StandardScaler()
        self.geo_cols: list[str] = []
        self.origin_cols: list[str] = []
        self.dest_proxy_reference: pd.Series | None = None
        self.net: _ExposureOpFlowNet | None = None
        self.train_mean_scale = 1.0
        self.seed = int(seed)
        self.selected_epoch_: int | None = None
        self.selection_summary_: dict = {}
        torch.manual_seed(seed)

    def _new_net(self) -> _ExposureOpFlowNet:
        return _ExposureOpFlowNet(
            p_dest_dim=3,
            p_pair_dim=3,
            p_geo_dim=3,
            origin_dim=len(self.origin_cols),
            hidden_dim=self.hidden_dim,
            benefit_skip=self.benefit_skip,
            deterrence_skip=self.deterrence_skip,
            exposure_skip=self.exposure_skip,
            competition_skip=self.competition_skip,
            residual_hidden_dim=self.residual_hidden_dim,
        ).to(self.device)

    def _empty_history(self) -> dict[str, list[float]]:
        return {
            "train_loss": [],
            "val_loss": [],
            "train_alloc_risk_max": [],
            "val_alloc_risk_max": [],
            "train_vrex_alloc_var": [],
            "val_vrex_alloc_var": [],
            "train_operator_anchor_loss": [],
            "val_operator_anchor_loss": [],
            "train_scale_loss": [],
            "val_scale_loss": [],
            "train_joint_selection_score": [],
            "val_joint_selection_score": [],
            "selection_score": [],
        }

    def _prepare_fit(self, train_records: pd.DataFrame, val_records: pd.DataFrame) -> tuple[torch.optim.Optimizer, dict, dict]:
        self.geo_cols = geo_columns(train_records, include_raw_cost=True)
        self.origin_cols = origin_columns(train_records)
        self.dest_proxy_reference = train_records.groupby("dest_id", sort=False)["x_geo_3"].mean()
        if "R_oracle" in train_records:
            self.train_mean_scale = float(train_records.groupby(["env_id", "origin_id"], sort=False)["R_oracle"].first().mean())
        else:
            self.train_mean_scale = float(train_records.groupby(["env_id", "origin_id"], sort=False)["y_obs"].sum().mean())
        self.geo_scaler = StandardScaler()
        self.origin_scaler = StandardScaler()
        x_geo_train = self.geo_scaler.fit_transform(_frame_array(train_records[self.geo_cols]))
        x_geo_val = self.geo_scaler.transform(_frame_array(val_records[self.geo_cols]))
        origin_train = self.origin_scaler.fit_transform(_frame_array(train_records[self.origin_cols]))
        origin_val = self.origin_scaler.transform(_frame_array(val_records[self.origin_cols]))
        train_components = _proxy_components_numpy(
            train_records,
            rank_prefix_mode=self.rank_prefix_mode,
            rank_prefix_temperature=self.rank_prefix_temperature,
            omega=self.competition_omega,
        )
        self.component_standardizer = _fit_component_standardizer(train_components)
        torch.manual_seed(self.seed)
        self.net = self._new_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        train_tensors = self._make_tensors(train_records, x_geo_train, origin_train)
        val_tensors = self._make_tensors(val_records, x_geo_val, origin_val)
        return optimizer, train_tensors, val_tensors

    def _selection_score(self, train_loss: float, val_pred: dict, score_mode: str) -> float:
        if score_mode == "train_loss":
            return train_loss
        metric = self.leo_selection_metric if score_mode == "leo" else self.checkpoint_metric
        if metric not in val_pred:
            raise KeyError(f"Unknown checkpoint metric: {metric}")
        return float(val_pred[metric].item())

    def _fit_once(
        self,
        train_records: pd.DataFrame,
        val_records: pd.DataFrame,
        *,
        score_mode: str,
        max_epochs: int | None = None,
        use_patience: bool = True,
        keep_final_state: bool = False,
    ) -> tuple[dict, int, float]:
        optimizer, train_tensors, val_tensors = self._prepare_fit(train_records, val_records)
        best_state = copy.deepcopy(self.net.state_dict())
        best_val = float("inf")
        best_epoch = 0
        stale = 0
        history = self._empty_history()
        total_epochs = int(max_epochs or self.epochs)
        for epoch in range(total_epochs):
            self.net.train()
            loss, _ = self._loss(train_tensors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.net.eval()
            with torch.no_grad():
                train_loss, train_pred = self._loss(train_tensors)
                val_loss, val_pred = self._loss(val_tensors)
                train_loss = float(train_loss.item())
                val_loss = float(val_loss.item())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_alloc_risk_max"].append(float(train_pred["alloc_risk_max"].item()))
            history["val_alloc_risk_max"].append(float(val_pred["alloc_risk_max"].item()))
            history["train_vrex_alloc_var"].append(float(train_pred["vrex_alloc_var"].item()))
            history["val_vrex_alloc_var"].append(float(val_pred["vrex_alloc_var"].item()))
            history["train_operator_anchor_loss"].append(float(train_pred["operator_anchor_loss"].item()))
            history["val_operator_anchor_loss"].append(float(val_pred["operator_anchor_loss"].item()))
            history["train_scale_loss"].append(float(train_pred["scale_loss"].item()))
            history["val_scale_loss"].append(float(val_pred["scale_loss"].item()))
            history["train_joint_selection_score"].append(float(train_pred["joint_selection_score"].item()))
            history["val_joint_selection_score"].append(float(val_pred["joint_selection_score"].item()))
            val_score = self._selection_score(train_loss, val_pred, score_mode)
            history["selection_score"].append(val_score)
            if keep_final_state:
                best_val = val_score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())
                continue
            if val_score < best_val - 1e-6:
                best_val = val_score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())
                stale = 0
            else:
                stale += 1
                if use_patience and stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return history, max(best_epoch, 1), best_val

    def _use_leo_selection(self, train_records: pd.DataFrame) -> bool:
        if self.method == "NoEnvSelection":
            return False
        if self.selection_protocol not in {"leo", "leave_one_env", "leave_one_environment_out"}:
            return False
        return int(train_records["env_id"].nunique()) >= 2

    def _fit_with_leo_selection(self, train_records: pd.DataFrame, val_records: pd.DataFrame) -> dict:
        env_ids = list(pd.unique(train_records["env_id"]))
        fold_scores: list[list[float]] = []
        for heldout_env in env_ids:
            fold_train = train_records.loc[train_records["env_id"] != heldout_env].copy()
            fold_val = train_records.loc[train_records["env_id"] == heldout_env].copy()
            fold_history, _, _ = self._fit_once(
                fold_train,
                fold_val,
                score_mode="leo",
                max_epochs=self.epochs,
                use_patience=False,
            )
            fold_scores.append(list(fold_history["selection_score"]))
        score_matrix = np.asarray(fold_scores, dtype=float)
        mean_scores = np.nanmean(score_matrix, axis=0)
        selected_epoch = int(np.nanargmin(mean_scores)) + 1
        history, _, final_score = self._fit_once(
            train_records,
            val_records,
            score_mode="leo",
            max_epochs=selected_epoch,
            use_patience=False,
            keep_final_state=True,
        )
        self.selected_epoch_ = selected_epoch
        self.selection_summary_ = {
            "selection_protocol": "leo",
            "selection_metric": self.leo_selection_metric,
            "lambda_selection_scale": float(self.lambda_selection_scale),
            "selected_epoch": selected_epoch,
            "num_folds": len(env_ids),
            "fold_env_ids": [int(env) if isinstance(env, (int, np.integer)) else env for env in env_ids],
            "mean_score_by_epoch": mean_scores.tolist(),
            "final_val_score": float(final_score),
            "retrained_on_all_train_envs": True,
        }
        return history

    def fit(self, train_records: pd.DataFrame, val_records: pd.DataFrame) -> dict:
        if self._use_leo_selection(train_records):
            return self._fit_with_leo_selection(train_records, val_records)
        score_mode = "train_loss" if self.method == "NoEnvSelection" else "validation"
        history, selected_epoch, best_score = self._fit_once(train_records, val_records, score_mode=score_mode)
        self.selected_epoch_ = selected_epoch
        self.selection_summary_ = {
            "selection_protocol": "train_loss" if self.method == "NoEnvSelection" else "validation",
            "selection_metric": "train_loss" if self.method == "NoEnvSelection" else self.checkpoint_metric,
            "lambda_selection_scale": float(self.lambda_selection_scale),
            "selected_epoch": selected_epoch,
            "best_score": float(best_score),
            "retrained_on_all_train_envs": False,
        }
        return history

    def _crowding_shift_gate(self, records: pd.DataFrame) -> np.ndarray:
        gates = np.zeros(len(records), dtype=np.float32)
        if not self.use_crowding_shift_gate or self.dest_proxy_reference is None or self.dest_proxy_reference.empty:
            return gates
        temp = max(float(self.crowding_ref_corr_temperature), 1e-6)
        for _, positions in records.groupby("env_id", sort=False).indices.items():
            group = records.iloc[positions]
            current = group.groupby("dest_id", sort=False)["x_geo_3"].mean()
            joined = pd.concat([self.dest_proxy_reference.rename("ref"), current.rename("cur")], axis=1).dropna()
            if len(joined) < 2:
                gate = 0.0
            else:
                ref = joined["ref"].to_numpy(dtype=float)
                cur = joined["cur"].to_numpy(dtype=float)
                if np.isclose(ref.std(), 0.0) or np.isclose(cur.std(), 0.0):
                    gate = 0.0
                else:
                    corr = float(np.corrcoef(ref, cur)[0, 1])
                    gate = float(1.0 / (1.0 + np.exp((corr - self.crowding_ref_corr_threshold) / temp)))
            gates[np.asarray(positions, dtype=int)] = gate
        return gates

    def _make_tensors(self, records: pd.DataFrame, x_geo: np.ndarray, origin: np.ndarray) -> dict:
        layout = _row_layout(records)
        row_id = layout.row_id
        n_rows = layout.n_rows
        k = layout.k
        env_codes, _ = pd.factorize(records["env_id"], sort=False)
        row_env = np.zeros(n_rows, dtype=np.int64)
        for idx, rid in enumerate(row_id):
            row_env[rid] = env_codes[idx]
        env_dest_keys = pd.MultiIndex.from_frame(records[["env_id", "dest_id"]])
        env_dest_idx, env_dest_uniques = pd.factorize(env_dest_keys, sort=False)
        first_indices = np.zeros(len(env_dest_uniques), dtype=np.int64)
        seen = np.zeros(len(env_dest_uniques), dtype=bool)
        for record_idx, group_idx in enumerate(env_dest_idx):
            if not seen[group_idx]:
                first_indices[group_idx] = record_idx
                seen[group_idx] = True
        env_by_unique = env_codes[first_indices]
        env_unique_dest_indices = [
            torch.as_tensor(np.flatnonzero(env_by_unique == env_id), dtype=torch.long, device=self.device)
            for env_id in np.unique(env_by_unique)
        ]
        origin_raw = _frame_array(records[["x_origin_0", "x_origin_1"]])
        dest_raw = _frame_array(records[["x_geo_0", "x_geo_1"]])
        p_geo = np.column_stack(
            [
                dest_raw[:, 0] - origin_raw[:, 0],
                dest_raw[:, 1] - origin_raw[:, 1],
                _frame_array(records["distance"]),
            ]
        )
        p_pair = np.column_stack(
            [
                x_geo[:, 2],
                x_geo[:, 4],
                _frame_array(records["cost"]),
            ]
        )
        raw_components = _proxy_components_numpy(
            records,
            rank_prefix_mode=self.rank_prefix_mode,
            rank_prefix_temperature=self.rank_prefix_temperature,
            omega=self.competition_omega,
        )
        if not self.component_standardizer:
            self.component_standardizer = _fit_component_standardizer(raw_components)
        proxy_components = {
            key: _apply_component_standardizer(values, self.component_standardizer[key])
            for key, values in raw_components.items()
        }
        return {
            "x_geo": _float_tensor(x_geo, device=self.device),
            "origin": _float_tensor(origin, device=self.device),
            "p_pair": _float_tensor(p_pair, device=self.device),
            "p_geo": _float_tensor(p_geo, device=self.device),
            "cost_z": _float_tensor(records["x_geo_4"].to_numpy(dtype=float), device=self.device),
            "proxy_o": _float_tensor(proxy_components["O"], device=self.device),
            "proxy_c": _float_tensor(proxy_components["C"], device=self.device),
            "proxy_s": _float_tensor(proxy_components["S"], device=self.device),
            "proxy_l": _float_tensor(proxy_components["L"], device=self.device),
            "crowding_gate": _float_tensor(self._crowding_shift_gate(records), device=self.device),
            "dest_input": _float_tensor(x_geo[first_indices][:, [0, 1, 3]], device=self.device),
            "dest_xy": _float_tensor(_frame_array(records[["x_geo_0", "x_geo_1"]])[first_indices], device=self.device),
            "env_dest_idx": torch.as_tensor(env_dest_idx, dtype=torch.long, device=self.device),
            "env_unique_dest_indices": env_unique_dest_indices,
            "row_env": torch.as_tensor(row_env, dtype=torch.long, device=self.device),
            "y": _float_tensor(records["y_obs"].to_numpy(dtype=float), device=self.device),
            "scale": _float_tensor(records["R_oracle"].to_numpy(dtype=float), device=self.device),
            "flat_index": torch.as_tensor(layout.flat_index, dtype=torch.long, device=self.device),
            "row_first_index": torch.as_tensor(layout.first_index, dtype=torch.long, device=self.device),
            "candidate_mask": torch.as_tensor(layout.mask, dtype=torch.bool, device=self.device),
            "n_rows": n_rows,
            "k": k,
        }

    def _rank_prefix(
        self,
        opportunity: torch.Tensor,
        impedance: torch.Tensor,
        n_rows: int,
        k: int,
        flat_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_rank_exposure:
            return torch.zeros_like(opportunity)
        opp_m = _scatter_to_matrix(opportunity, flat_index, n_rows, k, fill=0.0).masked_fill(~mask, 0.0)
        cost_m = _scatter_to_matrix(impedance, flat_index, n_rows, k, fill=0.0)
        if self.rank_prefix_mode == "soft":
            temp = max(float(self.rank_prefix_temperature), 1e-6)
            pairwise_order = torch.sigmoid((cost_m.unsqueeze(2) - cost_m.unsqueeze(1)) / temp)
            self_mask = torch.eye(k, dtype=torch.bool, device=opportunity.device).unsqueeze(0)
            pairwise_order = pairwise_order.masked_fill(self_mask, 0.0)
            pairwise_order = pairwise_order * mask.unsqueeze(1).to(pairwise_order.dtype) * mask.unsqueeze(2).to(pairwise_order.dtype)
            exposure = (pairwise_order * opp_m.unsqueeze(1)).sum(dim=2).masked_fill(~mask, 0.0)
            return _gather_from_matrix(exposure, flat_index)
        order = torch.argsort(cost_m.masked_fill(~mask, torch.inf), dim=1, stable=True)
        sorted_opp = torch.gather(opp_m, 1, order)
        prefix = torch.cumsum(sorted_opp, dim=1) - sorted_opp
        exposure = torch.zeros_like(prefix).scatter(1, order, prefix)
        exposure = exposure.masked_fill(~mask, 0.0)
        return _gather_from_matrix(exposure, flat_index)

    def _local_competition(self, opportunity_unique: torch.Tensor, dest_xy: torch.Tensor, env_indices: list[torch.Tensor]) -> torch.Tensor:
        ell = torch.zeros_like(opportunity_unique)
        for indices in env_indices:
            if int(indices.numel()) == 0:
                continue
            xy = dest_xy.index_select(0, indices)
            opp = opportunity_unique.index_select(0, indices)
            dist = torch.cdist(xy, xy)
            weights = torch.exp(-dist / max(self.competition_omega, 1e-8))
            weights.fill_diagonal_(0.0)
            ell.index_copy_(0, indices, weights @ opp)
        return ell

    def _parts(self, tensors: dict) -> dict:
        assert self.net is not None
        n_rows = tensors["n_rows"]
        k = tensors["k"]
        flat_index = tensors["flat_index"]
        mask = tensors["candidate_mask"]
        p_dest_unique = tensors["dest_input"]
        p_dest_pair = p_dest_unique.index_select(0, tensors["env_dest_idx"])
        origin = tensors["origin"]
        p_pair = tensors["p_pair"]
        p_geo = tensors["p_geo"]
        require_component_grad = self.use_monotone_penalty and self.use_sit_operator_potential
        log_o = tensors["proxy_o"].detach().clone().requires_grad_(require_component_grad)
        log_c = tensors["proxy_c"].detach().clone().requires_grad_(require_component_grad)
        log_s = tensors["proxy_s"].detach().clone().requires_grad_(require_component_grad)
        log_ell = tensors["proxy_l"].detach().clone().requires_grad_(require_component_grad)
        if not self.use_opportunity_field:
            log_o = torch.zeros_like(log_o)
        if not self.use_rank_exposure:
            log_s = torch.zeros_like(log_s)
        if not self.use_competition:
            log_ell = torch.zeros_like(log_ell)
        o_hat = log_o
        c_hat = log_c
        s_hat = log_s
        ell_hat = log_ell
        if self.use_sit_operator_potential:
            coeff_raw = self.net.coefficient_head(origin)
            coeff = F.softplus(coeff_raw) / np.log(2.0)
            a_i, b_i, r_i, local_i = coeff.unbind(dim=1)
            u_benefit = a_i * self.net.benefit(log_o)
            u_deterrence = -b_i * self.net.deterrence(log_c)
            u_exposure = -r_i * self.net.exposure(log_s)
            u_competition = local_i * self.net.local_field(log_ell)
            if not self.use_opportunity_field:
                u_benefit = torch.zeros_like(log_o)
            if not self.use_rank_exposure:
                u_exposure = torch.zeros_like(log_o)
            if not self.use_competition:
                u_competition = torch.zeros_like(log_o)
            if self.use_residual_interaction and self.eta_residual > 0.0:
                residual_input = torch.cat([origin, torch.stack([log_o, log_c, log_s, log_ell], dim=1), p_geo], dim=1)
                u_residual = float(self.eta_residual) * self.net.residual_interaction(residual_input).reshape(-1)
            else:
                u_residual = torch.zeros_like(log_o)
            u = (
                u_deterrence
                + u_benefit
                + u_exposure
                + u_competition
                + u_residual
            )
            if self.use_crowding_shift_gate and self.crowding_fallback_weight > 0.0:
                fallback = (
                    abs(float(self.benefit_skip)) * log_o
                    - abs(float(self.deterrence_skip)) * log_c
                    - abs(float(self.exposure_skip)) * log_s
                    + abs(float(self.competition_skip)) * log_ell
                )
                blend = torch.clamp(tensors["crowding_gate"] * float(self.crowding_fallback_weight), min=0.0, max=1.0)
                u = (1.0 - blend) * u + blend * fallback
        else:
            u_deterrence = torch.zeros_like(o_hat)
            u_benefit = torch.zeros_like(o_hat)
            u_exposure = torch.zeros_like(o_hat)
            u_competition = torch.zeros_like(o_hat)
            u_residual = torch.zeros_like(o_hat)
            u = self.net.generic_potential(torch.cat([origin, p_dest_pair, p_pair, p_geo], dim=1)).reshape(-1)
        logits = _scatter_to_matrix(u, flat_index, n_rows, k, fill=0.0)
        if self.use_row_center:
            logits = _masked_row_center(logits, mask)
        else:
            logits = logits.masked_fill(~mask, 0.0)
        pi = _softmax_matrix(logits, self.tau, mask)
        row_origin = origin.index_select(0, tensors["row_first_index"])
        operator_state_flat = torch.stack([log_o, log_c, log_s, log_ell], dim=1)
        encoded_flat = self.net.scale_encoder(operator_state_flat.detach())
        encoded = _scatter_to_matrix(encoded_flat.reshape(-1), flat_index.repeat_interleave(3) * 3 + torch.arange(3, device=flat_index.device).repeat(flat_index.numel()), n_rows, k * 3, fill=0.0).reshape(n_rows, k, 3)
        denom = torch.clamp(mask.sum(dim=1, keepdim=True).to(encoded.dtype), min=1.0)
        operator_summary = (encoded * mask.unsqueeze(2).to(encoded.dtype)).sum(dim=1) / denom
        row_scale = _scale_from_log_residual(
            self.net.scale_net(torch.cat([row_origin, operator_summary], dim=1)).reshape(-1),
            self.train_mean_scale,
        )
        return {
            "o_hat": o_hat,
            "c_hat": c_hat,
            "s_hat": s_hat,
            "logits": logits,
            "pi": pi,
            "u_flat": u,
            "row_scale": row_scale,
            "ell_hat": ell_hat,
            "operator_summary": operator_summary,
            "log_o_hat": log_o,
            "log_c_hat": log_c,
            "log_s_hat": log_s,
            "log_ell_hat": log_ell,
            "u_deterrence": u_deterrence,
            "u_benefit": u_benefit,
            "u_exposure": u_exposure,
            "u_competition": u_competition,
            "u_residual": u_residual,
            "eta_residual": torch.as_tensor(float(self.eta_residual if self.use_residual_interaction else 0.0), dtype=u.dtype, device=u.device),
        }

    def _env_risks(self, per_row: torch.Tensor, row_env: torch.Tensor) -> torch.Tensor:
        risks = []
        for env in torch.unique(row_env, sorted=True):
            mask = row_env == env
            if bool(mask.any()):
                risks.append(per_row[mask].mean())
        return torch.stack(risks) if risks else per_row.new_zeros((1,))

    def _monotone_penalty(self, parts: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.use_monotone_penalty or not self.use_sit_operator_potential:
            zero = parts["logits"].new_tensor(0.0)
            return zero, {"op_monotone_violation_D": zero, "op_monotone_violation_S": zero, "op_monotone_violation_B": zero}

        def grad_of(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
            if not input_tensor.requires_grad or not output.requires_grad:
                return torch.zeros_like(input_tensor)
            grad = torch.autograd.grad(output.sum(), input_tensor, create_graph=True, retain_graph=True, allow_unused=True)[0]
            return torch.zeros_like(input_tensor) if grad is None else grad

        d_grad = grad_of(parts["u_deterrence"], parts["c_hat"])
        s_grad = grad_of(parts["u_exposure"], parts["s_hat"])
        b_grad = grad_of(parts["u_benefit"], parts["o_hat"])
        v_d = torch.relu(d_grad).square().mean()
        v_s = torch.relu(s_grad).square().mean()
        v_b = torch.relu(-b_grad).square().mean()
        return v_d + v_s + v_b, {
            "op_monotone_violation_D": v_d.detach(),
            "op_monotone_violation_S": v_s.detach(),
            "op_monotone_violation_B": v_b.detach(),
        }

    def _row_center_tensor(self, values: torch.Tensor, tensors: dict) -> torch.Tensor:
        matrix = _scatter_to_matrix(values, tensors["flat_index"], tensors["n_rows"], tensors["k"], fill=0.0)
        centered = _masked_row_center(matrix, tensors["candidate_mask"])
        return _gather_from_matrix(centered, tensors["flat_index"])

    def _operator_anchor_loss(self, tensors: dict, parts: dict) -> torch.Tensor:
        if (
            not self.use_sit_operator_potential
            or self.lambda_operator_anchor <= 0.0
            or (self.lambda_o_anchor <= 0.0 and self.lambda_c_anchor <= 0.0)
        ):
            return parts["logits"].new_tensor(0.0)
        loss = parts["logits"].new_tensor(0.0)
        if self.use_opportunity_field and self.lambda_o_anchor > 0.0:
            o_target = tensors["proxy_o"]
            loss = loss + self.lambda_o_anchor * F.mse_loss(
                self._row_center_tensor(parts["log_o_hat"], tensors),
                self._row_center_tensor(o_target, tensors),
            )
        if self.lambda_c_anchor > 0.0:
            rank_target = tensors["proxy_c"]
            loss = loss + self.lambda_c_anchor * F.mse_loss(
                self._row_center_tensor(parts["log_c_hat"], tensors),
                self._row_center_tensor(rank_target, tensors),
            )
        return loss

    def _loss_components(self, tensors: dict) -> dict:
        parts = self._parts(tensors)
        n_rows = tensors["n_rows"]
        k = tensors["k"]
        mask = tensors["candidate_mask"]
        flat_index = tensors["flat_index"]
        target = _smoothed_target(tensors, self.alpha)
        pi_clamped = torch.clamp(parts["pi"], min=1e-8)
        target_clamped = torch.clamp(target, min=1e-8)
        per_row_alloc = (
            target_clamped * (torch.log(target_clamped) - torch.log(pi_clamped))
        ).masked_fill(~mask, 0.0).sum(dim=1)
        env_alloc = self._env_risks(per_row_alloc, tensors["row_env"])
        alloc_risk_mean = env_alloc.mean()
        alloc_risk_max = env_alloc.max()
        alloc_risk_train = alloc_risk_mean + float(self.groupdro_weight) * (alloc_risk_max - alloc_risk_mean)
        vrex_alloc_var = env_alloc.var(unbiased=False) if env_alloc.numel() > 1 else env_alloc.new_tensor(0.0)
        alloc_objective = alloc_risk_train + (self.lambda_vrex * vrex_alloc_var if self.use_vrex else 0.0)
        row_true = _row_scale_true(tensors["scale"], tensors["row_first_index"])
        per_row_scale = (torch.log1p(parts["row_scale"]) - torch.log1p(torch.clamp(row_true, min=0.0))).square()
        env_scale = self._env_risks(per_row_scale, tensors["row_env"])
        scale_risk_max = env_scale.max()
        scale_risk_mean = env_scale.mean()
        scale_loss = per_row_scale.mean()
        joint_selection_score = per_row_alloc.mean() + float(self.lambda_selection_scale) * scale_loss
        count_pi = parts["pi"].detach() if self.detach_count_allocation else parts["pi"]
        count_lam = parts["row_scale"][:, None] * count_pi
        count_loss = F.mse_loss(torch.log1p(_gather_from_matrix(count_lam, flat_index)), torch.log1p(tensors["y"]))
        op_penalty, op_diag = self._monotone_penalty(parts)
        operator_anchor_loss = self._operator_anchor_loss(tensors, parts)
        loss = (
            alloc_objective
            + self.lambda_scale * scale_risk_max
            + self.lambda_count * count_loss
            + self.lambda_op * op_penalty
            + self.lambda_operator_anchor * operator_anchor_loss
        )
        return {
            **parts,
            "alloc_loss": per_row_alloc.mean(),
            "scale_loss": scale_loss,
            "count_loss": count_loss,
            "operator_anchor_loss": operator_anchor_loss,
            "loss": loss,
            "joint_selection_score": joint_selection_score,
            "alloc_objective": alloc_objective,
            "alloc_risk_max": alloc_risk_max,
            "alloc_risk_mean": alloc_risk_mean,
            "alloc_risk_train": alloc_risk_train,
            "vrex_alloc_var": vrex_alloc_var,
            "scale_risk_max": scale_risk_max,
            "scale_risk_mean": scale_risk_mean,
            "op_penalty": op_penalty,
            **op_diag,
        }

    def _loss(self, tensors: dict) -> tuple[torch.Tensor, dict]:
        comp = self._loss_components(tensors)
        lam = comp["row_scale"][:, None] * comp["pi"]
        return comp["loss"], {
            "lambda_hat": _gather_from_matrix(lam, tensors["flat_index"]),
            "pi_hat": _gather_from_matrix(comp["pi"], tensors["flat_index"]),
            "score_hat": _gather_from_matrix(comp["logits"], tensors["flat_index"]),
            "row_scale_hat": comp["row_scale"],
            "o_hat": comp["o_hat"],
            "c_hat": comp["c_hat"],
            "s_hat": comp["s_hat"],
            "ell_hat": comp["ell_hat"],
            "operator_summary": comp["operator_summary"],
            **{key: comp[key] for key in (
                "alloc_loss",
                "scale_loss",
                "count_loss",
                "operator_anchor_loss",
                "joint_selection_score",
                "alloc_objective",
                "alloc_risk_max",
                "alloc_risk_mean",
                "alloc_risk_train",
                "vrex_alloc_var",
                "scale_risk_max",
                "scale_risk_mean",
                "op_penalty",
                "op_monotone_violation_D",
                "op_monotone_violation_S",
                "op_monotone_violation_B",
            )},
            "eta_residual": comp["eta_residual"],
        }

    def audit_gradients(self, records: pd.DataFrame) -> ExposureAudit:
        assert self.net is not None
        x_geo = self.geo_scaler.transform(_frame_array(records[self.geo_cols]))
        origin = self.origin_scaler.transform(_frame_array(records[self.origin_cols]))
        tensors = self._make_tensors(records, x_geo, origin)
        alloc_params = list(self.net.allocation_parameters())

        def max_potential_grad(loss: torch.Tensor) -> float:
            if not loss.requires_grad:
                return 0.0
            self.net.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            vals = [float(param.grad.detach().abs().max().item()) for param in alloc_params if param.grad is not None]
            self.net.zero_grad(set_to_none=True)
            return max(vals) if vals else 0.0

        comp = self._loss_components(tensors)
        count_scale_grad = max_potential_grad(self.lambda_scale * comp["scale_loss"] + self.lambda_count * comp["count_loss"])
        row_center_mean_abs = float(_masked_mean(comp["logits"], tensors["candidate_mask"]).abs().mean().detach().cpu())
        return ExposureAudit(
            count_scale_potential_grad_max=count_scale_grad,
            row_center_mean_abs=row_center_mean_abs,
            op_monotone_violation_D=float(comp["op_monotone_violation_D"].detach().cpu()),
            op_monotone_violation_S=float(comp["op_monotone_violation_S"].detach().cpu()),
            op_monotone_violation_B=float(comp["op_monotone_violation_B"].detach().cpu()),
        )

    def predict(self, records: pd.DataFrame) -> dict:
        x_geo = self.geo_scaler.transform(_frame_array(records[self.geo_cols]))
        origin = self.origin_scaler.transform(_frame_array(records[self.origin_cols]))
        tensors = self._make_tensors(records, x_geo, origin)
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            _, pred = self._loss(tensors)
        return _prediction_dict(
            pred["lambda_hat"],
            pred["pi_hat"],
            pred["score_hat"],
            {
                "row_scale_hat": pred["row_scale_hat"].detach().cpu(),
                "tau": self.tau,
                "o_hat": pred["o_hat"].detach().cpu(),
                "c_hat": pred["c_hat"].detach().cpu(),
                "s_hat": pred["s_hat"].detach().cpu(),
                "ell_hat": pred["ell_hat"].detach().cpu(),
                "operator_summary": pred["operator_summary"].detach().cpu(),
                "alloc_risk_max": float(pred["alloc_risk_max"].detach().cpu()),
                "alloc_risk_mean": float(pred["alloc_risk_mean"].detach().cpu()),
                "vrex_alloc_var": float(pred["vrex_alloc_var"].detach().cpu()),
                "scale_risk_max": float(pred["scale_risk_max"].detach().cpu()),
                "scale_risk_mean": float(pred["scale_risk_mean"].detach().cpu()),
                "eta_residual": float(pred["eta_residual"].detach().cpu()),
            },
        )


def build_exposure_model(method: str, config: dict, seed: int):
    aliases = {
        "FullOpFlow": "OpFlow",
        "OpFlow-JointSelected": "OpFlow",
        "OpFlowJointSelected": "OpFlow",
        "OpFlow-AllocSelected": "OpFlowAllocSelected",
        "OpFlowAllocSelected": "OpFlowAllocSelected",
        "OpFlow-ERM": "OpFlowERM",
        "RowCountMLP": "RawCountMLP",
        "PairAllocMLP": "PairMLP",
        "FixedShare": "ShareStable",
        "Gravity": "GravityOnly",
        "NoRankPrefix": "NoRankExposure",
        "NoLocalField": "NoCompetition",
        "NoScaleIsolation": "NoScaleDetach",
        "NoMonotonicity": "NoMonotonePenalty",
        "ScaleAwareMLP": "ScaleAwareMLP",
        "NoRankExposure": "NoRankExposure",
        "NoCompetition": "NoCompetition",
        "NoOpportunityField": "NoOpportunityField",
        "NoResidualInteraction": "NoResidualInteraction",
        "NoScaleDetach": "NoScaleDetach",
        "NoRowCenter": "NoRowCenter",
        "LastShare": "LastShare",
        "GravityOnly": "GravityOnly",
        "ScaleAwareMLP-JointSelected": "ScaleAwareMLP",
        "ScaleAwareMLPJointSelected": "ScaleAwareMLP",
        "ScaleAwareMLP-AllocSelected": "ScaleAwareMLPAllocSelected",
        "ScaleAwareMLPAllocSelected": "ScaleAwareMLPAllocSelected",
    }
    method = aliases.get(method, method)
    removed = {
        "LeakyOpFlow",
        "NoRouting",
        "NoDiagnostic",
        "OpFlowNoResidual",
        "AblationNoInputRouting",
        "AblationNoResidualChallenger",
        "NoPrefixNoCompetition",
    }
    if method in removed:
        raise ValueError(f"{method} was removed by the P-only SIT refactor.")
    if method in {
        "OpFlow",
        "OpFlowAllocSelected",
        "OpFlowRank",
        "NoRankExposure",
        "NoCompetition",
        "NoOpportunityField",
        "NoResidualInteraction",
        "NoScaleDetach",
        "NoRowCenter",
        "NoExposureOps",
        "GenericPotentialMLP",
        "OpFlowERM",
        "NoGroupDRO",
        "NoVREX",
        "NoMonotonePenalty",
        "NoEnvSelection",
    }:
        return ExposureOpFlowModel(method, config, seed)
    if method in {
        "RawCountMLP",
        "RawCountGNN",
        "PairMLP",
        "ScaleAllocationMLP",
        "ScaleAwareMLP",
        "ScaleAwareMLPAllocSelected",
        "ShortcutOnly",
        "ShareStable",
        "LastShare",
        "GravityDirect",
        "GravityOnly",
        "CompetingDestination",
        "InterveningOpportunity",
        "Radiation",
        "GroupDRO-AllocMLP",
        "VREx-AllocMLP",
        "IRM-AllocMLP",
        "DANN-AllocMLP",
        "ExposureMLPNoS",
        "OraclePotential",
    }:
        return ExposureBaselineModel(method, config, seed)
    raise KeyError(f"Unknown exposure model: {method}")
