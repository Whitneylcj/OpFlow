"""DGL-free rewrites of the official CommutingODGen benchmark baselines.

These classes preserve the official model names and high-level model families
while adapting them to the OpFlow benchmark interface and Mac-friendly Torch.
They are intended for unified evaluation, not byte-for-byte official CUDA/DGL
reproduction.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from opflow_benchmark.baselines.common import (
    timed_evaluate_area_allocation,
    oracle_scale_prediction,
    profile_record,
    timed_evaluate_area_predictions,
    resolve_device,
)

try:  # pragma: no cover - exercised only when sklearn is installed.
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except ImportError:  # local smoke checks should still cover every official name.
    GradientBoostingRegressor = None
    RandomForestRegressor = None
    SVR = None

    class StandardScaler:
        def __init__(self) -> None:
            self.mean_: np.ndarray | None = None
            self.scale_: np.ndarray | None = None
            self._n_seen = 0
            self._sum: np.ndarray | None = None
            self._sum_sq: np.ndarray | None = None

        def partial_fit(self, x: np.ndarray) -> "StandardScaler":
            x = np.asarray(x, dtype=np.float64)
            if self._sum is None:
                self._sum = x.sum(axis=0)
                self._sum_sq = (x * x).sum(axis=0)
            else:
                self._sum += x.sum(axis=0)
                self._sum_sq += (x * x).sum(axis=0)
            self._n_seen += int(x.shape[0])
            mean = self._sum / max(self._n_seen, 1)
            var = self._sum_sq / max(self._n_seen, 1) - mean * mean
            self.mean_ = mean.astype(np.float32)
            self.scale_ = np.sqrt(np.maximum(var, EPS)).astype(np.float32)
            return self

        def fit(self, x: np.ndarray) -> "StandardScaler":
            self._n_seen = 0
            self._sum = None
            self._sum_sq = None
            return self.partial_fit(x)

        def transform(self, x: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                return np.asarray(x, dtype=np.float32)
            return ((np.asarray(x, dtype=np.float32) - self.mean_) / np.maximum(self.scale_, EPS)).astype(np.float32)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for progress display.
    tqdm = None


EPS = 1e-8


class NumpyFallbackRegressor:
    """Small deterministic fallback for official sklearn-style baselines."""

    def __init__(self, kind: str, ridge: float = 1e-3, max_prototypes: int = 128) -> None:
        self.kind = kind
        self.ridge = float(ridge)
        self.max_prototypes = int(max_prototypes)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.prototypes: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.x_mean is None or self.x_std is None:
            return x
        return ((x - self.x_mean) / np.maximum(self.x_std, EPS)).astype(np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        z = self._scale(x)
        parts = [np.ones((z.shape[0], 1), dtype=np.float32), z]
        if self.kind in {"gbrt", "rf"}:
            parts.append(np.square(z).astype(np.float32))
        if self.kind == "rf":
            rng = np.random.default_rng(17)
            proj = rng.normal(size=(z.shape[1], min(16, max(4, z.shape[1] * 2)))).astype(np.float32)
            parts.append(np.maximum(z @ proj, 0.0).astype(np.float32))
        if self.kind == "svr" and self.prototypes is not None and len(self.prototypes) > 0:
            diff = z[:, None, :] - self.prototypes[None, :, :]
            dist2 = np.square(diff).mean(axis=2)
            parts.append(np.exp(-0.5 * dist2).astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyFallbackRegressor":
        x = np.asarray(x, dtype=np.float32)
        y_log = np.log1p(np.maximum(np.asarray(y, dtype=np.float32), 0.0))
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + EPS
        z = self._scale(x)
        if self.kind == "svr":
            step = max(len(z) // max(self.max_prototypes, 1), 1)
            self.prototypes = z[::step][: self.max_prototypes].copy()
        phi = self._features(x)
        reg = self.ridge * np.eye(phi.shape[1], dtype=np.float32)
        lhs = phi.T @ phi + reg
        rhs = phi.T @ y_log
        self.weights = np.linalg.solve(lhs.astype(np.float64), rhs.astype(np.float64)).astype(np.float32)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(f"official_{self.kind} fallback regressor is not fitted")
        pred_log = self._features(x) @ self.weights
        return np.maximum(np.expm1(np.clip(pred_log, -5.0, 16.0)), 0.0).astype(np.float32)


def _as_np(value: torch.Tensor | np.ndarray | None) -> np.ndarray:
    if value is None:
        raise ValueError("Official baselines require complete area tensors")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _population_feature(area: dict[str, Any]) -> np.ndarray:
    node = _as_np(area["node_feat"]).astype(np.float32)
    return np.maximum(node[:, :1], 0.0)


def _node_features(area: dict[str, Any]) -> np.ndarray:
    return np.maximum(_as_np(area["node_feat"]).astype(np.float32), 0.0)


def _distance(area: dict[str, Any]) -> np.ndarray:
    return np.maximum(_as_np(area["distance"]).astype(np.float32), 0.0)


def _adjacency(area: dict[str, Any]) -> np.ndarray:
    return np.maximum(_as_np(area["adjacency"]).astype(np.float32), 0.0)


def _target(area: dict[str, Any]) -> np.ndarray:
    return np.maximum(_as_np(area["y_raw"]).astype(np.float32), 0.0)


def _official_pair_features(area: dict[str, Any], *, population_only: bool = False) -> np.ndarray:
    feat = _population_feature(area) if population_only else _node_features(area)
    n = feat.shape[0]
    origin = np.broadcast_to(feat[:, None, :], (n, n, feat.shape[1]))
    dest = np.broadcast_to(feat[None, :, :], (n, n, feat.shape[1]))
    dis = _distance(area)[:, :, None]
    return np.concatenate([origin, dest, dis], axis=2).reshape(n * n, -1).astype(np.float32)


def _pair_features_at(area: dict[str, Any], rows: np.ndarray, cols: np.ndarray, *, population_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    feat = _population_feature(area) if population_only else _node_features(area)
    dist = _distance(area)
    y = _target(area)
    x = np.concatenate([feat[rows], feat[cols], dist[rows, cols, None]], axis=1)
    return x.astype(np.float32), y[rows, cols].astype(np.float32)


def _resolve_pair_row_chunk_size(
    n: int,
    feature_dim: int,
    row_chunk_size: int | str | None,
    *,
    target_mb: float = 64.0,
    max_rows: int = 128,
) -> int:
    if row_chunk_size is None:
        return min(max(int(max_rows), 1), max(n, 1))
    if isinstance(row_chunk_size, str):
        if row_chunk_size.lower() != "auto":
            return max(int(row_chunk_size), 1)
        bytes_per_row = max(n, 1) * max(2 * int(feature_dim) + 1, 1) * np.dtype(np.float32).itemsize
        rows = int((float(target_mb) * 1024 * 1024) // max(bytes_per_row, 1))
        return max(1, min(n, max(int(max_rows), 1), rows))
    return max(int(row_chunk_size), 1)


def _iter_pair_feature_chunks(
    area: dict[str, Any],
    *,
    population_only: bool = False,
    row_chunk_size: int | str | None = 16,
    pair_chunk_target_mb: float = 64.0,
    pair_chunk_max_rows: int = 128,
):
    feat = _population_feature(area) if population_only else _node_features(area)
    dist = _distance(area)
    y = _target(area)
    n = feat.shape[0]
    chunk = _resolve_pair_row_chunk_size(
        n,
        int(feat.shape[1]),
        row_chunk_size,
        target_mb=pair_chunk_target_mb,
        max_rows=pair_chunk_max_rows,
    )
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = end - start
        origin = np.broadcast_to(feat[start:end, None, :], (rows, n, feat.shape[1]))
        dest = np.broadcast_to(feat[None, :, :], (rows, n, feat.shape[1]))
        dis = dist[start:end, :, None]
        x = np.concatenate([origin, dest, dis], axis=2).reshape(rows * n, -1).astype(np.float32)
        target = y[start:end, :].reshape(-1).astype(np.float32)
        yield start, end, x, target


def _progress(iterable, *, total: int | None, desc: str, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, file=sys.stdout, dynamic_ncols=True, mininterval=1.0, leave=False)


def _row_share_from_count(y_hat: torch.Tensor) -> torch.Tensor:
    y_pos = torch.clamp(y_hat, min=0.0)
    denom = y_pos.sum(dim=1, keepdim=True)
    return torch.where(denom > EPS, y_pos / torch.clamp(denom, min=EPS), torch.zeros_like(y_pos))


def _evaluate_count_model(
    model_name: str,
    predict_area,
    split_areas: dict[str, list[dict[str, Any]]],
    topk: int,
    profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile = profile if profile is not None else {}
    payload: dict[str, Any] = {f"{model_name}_learned_scale": {}, f"{model_name}_oracle_scale": {}}
    for split, areas in split_areas.items():
        learned_rows = []
        oracle_rows = []
        iterator = _progress(areas, total=len(areas), desc=f"{model_name} eval {split}", unit="area")
        for area in iterator:
            start = time.perf_counter()
            y_hat, extra = predict_area(area)
            p_hat = _row_share_from_count(y_hat)
            profile_record(profile, "predict_seconds", time.perf_counter() - start)
            profile["areas"] = float(profile.get("areas", 0.0) + 1.0)
            allocation = timed_evaluate_area_allocation(profile, area, p_hat, topk=topk)
            learned_rows.append(timed_evaluate_area_predictions(profile, area, y_hat, p_hat, topk=topk, extra=extra, allocation=allocation))
            y_oracle = oracle_scale_prediction(area, p_hat)
            oracle_rows.append(
                timed_evaluate_area_predictions(
                    profile,
                    area,
                    y_oracle,
                    p_hat,
                    topk=topk,
                    extra={**extra, "oracle_scale": 1.0},
                    allocation=allocation,
                )
            )
        payload[f"{model_name}_learned_scale"][split] = learned_rows
        payload[f"{model_name}_oracle_scale"][split] = oracle_rows
    return payload


class OfficialSklearnBaseline:
    """Official SVR/RF/GBRT pair-feature baselines."""

    def __init__(
        self,
        model_name: str,
        estimator: Any,
        max_train_pairs: int | None = 300_000,
        pair_row_chunk_size: int | str | None = 16,
        pair_chunk_target_mb: float = 64.0,
        pair_chunk_max_rows: int = 128,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.estimator = estimator
        self.max_train_pairs = None if max_train_pairs is None else int(max_train_pairs)
        self.pair_row_chunk_size = pair_row_chunk_size
        self.pair_chunk_target_mb = float(pair_chunk_target_mb)
        self.pair_chunk_max_rows = max(int(pair_chunk_max_rows), 1)
        self.actual_train_pairs = 0
        self.fitted = False
        self.profile: dict[str, float] = {}

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        total_pairs = sum(int(area["y_raw"].shape[0]) ** 2 for area in train_areas)
        cap = total_pairs if self.max_train_pairs is None else min(self.max_train_pairs, total_pairs)
        xs = []
        ys = []
        taken = 0
        iterator = _progress(train_areas, total=len(train_areas), desc=f"{self.model_name} collect", unit="area")
        for area in iterator:
            n = int(area["y_raw"].shape[0])
            area_pairs = n * n
            remaining = cap - taken
            if remaining <= 0:
                break
            take = area_pairs if cap == total_pairs else max(1, min(area_pairs, int(round(cap * area_pairs / max(total_pairs, 1)))))
            take = min(take, remaining)
            if take >= area_pairs:
                x_area = _official_pair_features(area, population_only=False)
                y_area = _target(area).reshape(-1).astype(np.float32)
            else:
                flat = np.linspace(0, area_pairs - 1, num=take, dtype=np.int64)
                rows = flat // n
                cols = flat % n
                x_area, y_area = _pair_features_at(area, rows, cols, population_only=False)
            xs.append(x_area)
            ys.append(y_area)
            taken += len(y_area)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        self.actual_train_pairs = int(len(y))
        self.estimator.fit(x, y)
        self.fitted = True
        return {"train_loss": [], "val_loss": []}

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.fitted:
            raise RuntimeError(f"{self.model_name} is not fitted")
        n = int(area["y_raw"].shape[0])
        pred = np.zeros((n, n), dtype=np.float32)
        iterator = _iter_pair_feature_chunks(
            area,
            population_only=False,
            row_chunk_size=self.pair_row_chunk_size,
            pair_chunk_target_mb=self.pair_chunk_target_mb,
            pair_chunk_max_rows=self.pair_chunk_max_rows,
        )
        for start, end, x, _ in iterator:
            pred[start:end, :] = np.asarray(self.estimator.predict(x), dtype=np.float32).reshape(end - start, n)
        pred = np.maximum(pred, 0.0)
        y_hat = torch.as_tensor(pred, dtype=torch.float32)
        return y_hat, {"mean_pred_count": float(pred.mean())}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        return _evaluate_count_model(self.model_name, self.predict_area, split_areas, topk, profile=self.profile)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": self.model_name,
            "fitted": self.fitted,
            "estimator": self.estimator,
            "config": {
                "max_train_pairs": self.max_train_pairs,
                "actual_train_pairs": self.actual_train_pairs,
                "pair_row_chunk_size": self.pair_row_chunk_size,
                "pair_chunk_target_mb": self.pair_chunk_target_mb,
                "pair_chunk_max_rows": self.pair_chunk_max_rows,
            },
        }


class OfficialGravityNet(nn.Module):
    def __init__(self, linear: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1) if linear else None
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_g = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        safe = torch.clamp(x, min=EPS)
        if self.linear is not None:
            return torch.exp(torch.clamp(self.linear(torch.log(safe)).squeeze(-1) + self.log_g, -5.0, 16.0))
        gamma = F.softplus(self.gamma_raw) + EPS
        logy = self.alpha * torch.log(safe[:, 0]) + self.beta * torch.log(safe[:, 1]) + safe[:, 2] * torch.log(gamma.square())
        return torch.exp(torch.clamp(logy + self.log_g, -5.0, 16.0))


class OfficialTorchPairBaseline:
    """Shared Torch trainer for official pair-count neural baselines."""

    def __init__(
        self,
        model_name: str,
        net_kind: str,
        lr: float = 1e-3,
        epochs: int = 50,
        weight_decay: float = 0.0,
        device: str = "auto",
        patience: int = 20,
        grad_clip: float = 5.0,
        pair_row_chunk_size: int | str | None = 16,
        pair_chunk_target_mb: float = 64.0,
        pair_chunk_max_rows: int = 128,
        max_train_pairs: int | None = 300_000,
        dgm_hidden_dim: int = 256,
        dgm_layers: int = 15,
        residual_hidden_dim: int | None = None,
        residual_layers: int | None = None,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.net_kind = net_kind
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device = resolve_device(device)
        self.patience = patience
        self.grad_clip = grad_clip
        self.pair_row_chunk_size = pair_row_chunk_size
        self.pair_chunk_target_mb = float(pair_chunk_target_mb)
        self.pair_chunk_max_rows = max(int(pair_chunk_max_rows), 1)
        self.max_train_pairs = None if max_train_pairs is None else int(max_train_pairs)
        self.dgm_hidden_dim = int(dgm_hidden_dim)
        self.dgm_layers = int(dgm_layers)
        self.residual_hidden_dim = None if residual_hidden_dim is None else int(residual_hidden_dim)
        self.residual_layers = None if residual_layers is None else int(residual_layers)
        self.model: nn.Module | None = None
        self.x_scaler: StandardScaler | None = None
        self.x_min: np.ndarray | None = None
        self.x_max: np.ndarray | None = None
        self.y_min = 0.0
        self.y_max = 1.0
        self.profile: dict[str, float] = {}

    def _make_model(self, input_dim: int) -> nn.Module:
        if self.net_kind == "gm_p":
            return OfficialGravityNet(linear=False)
        if self.net_kind == "gm_e":
            return OfficialGravityNet(linear=True)
        if self.net_kind == "dgm":
            return OfficialDGMNet(input_dim=input_dim, hidden_dim=self.dgm_hidden_dim, layers=self.dgm_layers)
        if self.net_kind == "netgan":
            return OfficialResidualPairNet(
                input_dim=input_dim,
                hidden_dim=self.residual_hidden_dim or 96,
                layers=self.residual_layers or 3,
            )
        if self.net_kind == "diffodgen":
            return OfficialResidualPairNet(
                input_dim=input_dim,
                hidden_dim=self.residual_hidden_dim or 128,
                layers=self.residual_layers or 4,
            )
        if self.net_kind == "wedan":
            return OfficialResidualPairNet(
                input_dim=input_dim,
                hidden_dim=self.residual_hidden_dim or 160,
                layers=self.residual_layers or 4,
            )
        raise ValueError(f"Unknown official Torch net kind: {self.net_kind}")

    def _features(self, area: dict[str, Any]) -> np.ndarray:
        return _official_pair_features(area, population_only=self.net_kind in {"gm_p", "gm_e"})

    def _fit_streaming_stats(self, train_areas: list[dict[str, Any]]) -> int:
        input_dim = 0
        cap = float("inf") if self.max_train_pairs is None else max(int(self.max_train_pairs), 1)
        if self.net_kind == "dgm":
            x_min = None
            x_max = None
            y_min = float("inf")
            y_max = float("-inf")
            seen = 0
            for area in _progress(train_areas, total=len(train_areas), desc=f"{self.model_name} stats", unit="area"):
                for _, _, x, y in _iter_pair_feature_chunks(
                    area,
                    population_only=False,
                    row_chunk_size=self.pair_row_chunk_size,
                    pair_chunk_target_mb=self.pair_chunk_target_mb,
                    pair_chunk_max_rows=self.pair_chunk_max_rows,
                ):
                    if seen >= cap:
                        break
                    if seen + len(y) > cap:
                        keep = int(cap - seen)
                        x = x[:keep]
                        y = y[:keep]
                    input_dim = x.shape[1]
                    cur_min = x.min(axis=0)
                    cur_max = x.max(axis=0)
                    x_min = cur_min if x_min is None else np.minimum(x_min, cur_min)
                    x_max = cur_max if x_max is None else np.maximum(x_max, cur_max)
                    y_min = min(y_min, float(y.min()))
                    y_max = max(y_max, float(y.max()))
                    seen += len(y)
                if seen >= cap:
                    break
            self.x_min = x_min.astype(np.float32) if x_min is not None else None
            self.x_max = x_max.astype(np.float32) if x_max is not None else None
            self.y_min = y_min if np.isfinite(y_min) else 0.0
            self.y_max = y_max if np.isfinite(y_max) else 1.0
            return input_dim
        if self.net_kind in {"netgan", "diffodgen", "wedan"}:
            self.x_scaler = StandardScaler()
            seen = 0
            for area in _progress(train_areas, total=len(train_areas), desc=f"{self.model_name} stats", unit="area"):
                for _, _, x, _ in _iter_pair_feature_chunks(
                    area,
                    population_only=False,
                    row_chunk_size=self.pair_row_chunk_size,
                    pair_chunk_target_mb=self.pair_chunk_target_mb,
                    pair_chunk_max_rows=self.pair_chunk_max_rows,
                ):
                    if seen >= cap:
                        break
                    if seen + x.shape[0] > cap:
                        x = x[: int(cap - seen)]
                    input_dim = x.shape[1]
                    self.x_scaler.partial_fit(x)
                    seen += x.shape[0]
                if seen >= cap:
                    break
            return input_dim
        first = train_areas[0]
        return next(
            _iter_pair_feature_chunks(
                first,
                population_only=self.net_kind in {"gm_p", "gm_e"},
                row_chunk_size=self.pair_row_chunk_size,
                pair_chunk_target_mb=self.pair_chunk_target_mb,
                pair_chunk_max_rows=self.pair_chunk_max_rows,
            )
        )[2].shape[1]

    def _transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.net_kind == "dgm" and self.x_min is not None and self.x_max is not None:
            denom = np.maximum(self.x_max - self.x_min, EPS)
            return (2.0 * ((x - self.x_min) / denom) - 1.0).astype(np.float32)
        if self.x_scaler is None:
            return x.astype(np.float32)
        return self.x_scaler.transform(x).astype(np.float32)

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.net_kind == "dgm":
            scale = max(self.y_max - self.y_min, EPS)
            return (2.0 * ((y - self.y_min) / scale) - 1.0).astype(np.float32)
        if self.net_kind in {"netgan", "diffodgen", "wedan"}:
            return np.log1p(y).astype(np.float32)
        return y.astype(np.float32)

    def _invert_y(self, pred: np.ndarray) -> np.ndarray:
        if self.net_kind == "dgm":
            return np.maximum(((np.clip(pred, -1.0, 1.0) + 1.0) / 2.0) * max(self.y_max - self.y_min, EPS) + self.y_min, 0.0)
        if self.net_kind in {"netgan", "diffodgen", "wedan"}:
            return np.maximum(np.expm1(np.clip(pred, -5.0, 16.0)), 0.0)
        return np.maximum(pred, 0.0)

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        input_dim = self._fit_streaming_stats(train_areas)
        self.model = self._make_model(input_dim).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        no_improve = 0
        population_only = self.net_kind in {"gm_p", "gm_e"}
        cap = float("inf") if self.max_train_pairs is None else max(int(self.max_train_pairs), 1)
        for epoch in range(self.epochs):
            self.model.train()
            loss_sum = 0.0
            n_obs = 0
            area_iter = _progress(train_areas, total=len(train_areas), desc=f"{self.model_name} epoch {epoch + 1}/{self.epochs}", unit="area")
            for area in area_iter:
                for _, _, x_np, y_np in _iter_pair_feature_chunks(
                    area,
                    population_only=population_only,
                    row_chunk_size=self.pair_row_chunk_size,
                    pair_chunk_target_mb=self.pair_chunk_target_mb,
                    pair_chunk_max_rows=self.pair_chunk_max_rows,
                ):
                    if n_obs >= cap:
                        break
                    if n_obs + len(y_np) > cap:
                        keep = int(cap - n_obs)
                        x_np = x_np[:keep]
                        y_np = y_np[:keep]
                    x = torch.as_tensor(self._transform_x(x_np), dtype=torch.float32, device=self.device)
                    y = torch.as_tensor(self._transform_y(y_np), dtype=torch.float32, device=self.device)
                    opt.zero_grad()
                    pred = self.model(x).reshape(-1)
                    loss = F.mse_loss(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    opt.step()
                    rows = int(y.numel())
                    loss_sum += float(loss.detach().cpu()) * rows
                    n_obs += rows
                if n_obs >= cap:
                    break
            train_loss = loss_sum / max(n_obs, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(train_loss)
            if train_loss < best_loss - 1e-7:
                best_loss = train_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return history

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        n = int(area["y_raw"].shape[0])
        y_np = np.zeros((n, n), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for start, end, x_chunk, _ in _iter_pair_feature_chunks(
                area,
                population_only=self.net_kind in {"gm_p", "gm_e"},
                row_chunk_size=self.pair_row_chunk_size,
                pair_chunk_target_mb=self.pair_chunk_target_mb,
                pair_chunk_max_rows=self.pair_chunk_max_rows,
            ):
                x = torch.as_tensor(self._transform_x(x_chunk), dtype=torch.float32, device=self.device)
                pred = self.model(x).reshape(-1).detach().cpu().numpy()
                y_np[start:end, :] = self._invert_y(pred).reshape(end - start, n).astype(np.float32)
        y_hat = torch.as_tensor(y_np, dtype=torch.float32)
        return y_hat, {"mean_pred_count": float(y_np.mean())}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        return _evaluate_count_model(self.model_name, self.predict_area, split_areas, topk, profile=self.profile)

    def checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        return {
            "model_type": self.model_name,
            "net_kind": self.net_kind,
            "model_state": self.model.state_dict(),
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "grad_clip": self.grad_clip,
                "y_min": self.y_min,
                "y_max": self.y_max,
                "pair_row_chunk_size": self.pair_row_chunk_size,
                "pair_chunk_target_mb": self.pair_chunk_target_mb,
                "pair_chunk_max_rows": self.pair_chunk_max_rows,
                "max_train_pairs": self.max_train_pairs,
                "dgm_hidden_dim": self.dgm_hidden_dim,
                "dgm_layers": self.dgm_layers,
                "residual_hidden_dim": self.residual_hidden_dim,
                "residual_layers": self.residual_layers,
            },
        }


class OfficialDGMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 15) -> None:
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.linears:
            x = torch.relu(layer(x)) + x
        return torch.tanh(self.linear_out(x)).squeeze(-1)


class OfficialResidualPairNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear_in(x))
        for block in self.blocks:
            x = torch.relu(block(x)) + x
        return self.linear_out(x).squeeze(-1)


class OfficialDenseGraphBaseline:
    """Dense PyTorch replacement for official DGL graph baselines."""

    def __init__(
        self,
        model_name: str,
        lr: float = 1e-3,
        epochs: int = 50,
        weight_decay: float = 0.0,
        hidden_dim: int = 64,
        device: str = "auto",
        patience: int = 20,
        grad_clip: float = 5.0,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.device = resolve_device(device)
        self.patience = patience
        self.grad_clip = grad_clip
        self.node_scaler = StandardScaler()
        self.dist_scaler = StandardScaler()
        self.y_min = 0.0
        self.y_max = 1.0
        self.model: OfficialDenseGraphNet | None = None
        self.profile: dict[str, float] = {}

    def _fit_scalers(self, areas: list[dict[str, Any]]) -> None:
        nodes = np.concatenate([_node_features(area) for area in areas], axis=0)
        dists = np.concatenate([_distance(area).reshape(-1, 1) for area in areas], axis=0)
        ys = np.concatenate([_target(area).reshape(-1) for area in areas], axis=0)
        self.node_scaler.fit(nodes)
        self.dist_scaler.fit(dists)
        self.y_min = float(ys.min())
        self.y_max = float(ys.max())

    def _area_tensors(self, area: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.node_scaler.transform(_node_features(area)).astype(np.float32)
        d = self.dist_scaler.transform(_distance(area).reshape(-1, 1)).reshape(area["y_raw"].shape).astype(np.float32)
        adj = _adjacency(area).astype(np.float32)
        y = _target(area)
        scale = max(self.y_max - self.y_min, EPS)
        y_norm = (2.0 * ((y - self.y_min) / scale) - 1.0).astype(np.float32)
        return (
            torch.as_tensor(x, dtype=torch.float32, device=self.device),
            torch.as_tensor(d, dtype=torch.float32, device=self.device),
            torch.as_tensor(adj, dtype=torch.float32, device=self.device),
            torch.as_tensor(y_norm, dtype=torch.float32, device=self.device),
        )

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        self._fit_scalers(train_areas)
        node_dim = int(train_areas[0]["node_feat"].shape[1])
        self.model = OfficialDenseGraphNet(node_dim=node_dim, hidden_dim=self.hidden_dim).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        no_improve = 0
        for _ in range(self.epochs):
            self.model.train()
            total = 0.0
            for area in train_areas:
                x, dist, adj, y = self._area_tensors(area)
                opt.zero_grad()
                pred = self.model(x, dist, adj)
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                total += float(loss.detach().cpu())
            train_loss = total / max(len(train_areas), 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(train_loss)
            if train_loss < best_loss - 1e-7:
                best_loss = train_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return history

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        x, dist, adj, _ = self._area_tensors(area)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, dist, adj).detach().cpu().numpy()
        y_np = np.maximum(((np.clip(pred, -1.0, 1.0) + 1.0) / 2.0) * max(self.y_max - self.y_min, EPS) + self.y_min, 0.0).astype(np.float32)
        y_hat = torch.as_tensor(y_np, dtype=torch.float32)
        return y_hat, {"mean_pred_count": float(y_np.mean())}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        return _evaluate_count_model(self.model_name, self.predict_area, split_areas, topk, profile=self.profile)

    def checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        return {
            "model_type": self.model_name,
            "model_state": self.model.state_dict(),
            "config": {"lr": self.lr, "epochs": self.epochs, "hidden_dim": self.hidden_dim, "y_min": self.y_min, "y_max": self.y_max},
        }


class OfficialDenseGraphNet(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, x: torch.Tensor, distance: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_in(x))
        weights = torch.softmax(adjacency + torch.eye(adjacency.shape[0], device=adjacency.device), dim=1)
        h = torch.relu(self.message(weights @ h)) + h
        n = h.shape[0]
        hi = h[:, None, :].expand(n, n, h.shape[1])
        hj = h[None, :, :].expand(n, n, h.shape[1])
        pair = torch.cat([hi, hj, distance[..., None]], dim=-1)
        return self.out(pair).squeeze(-1)


def build_official_model(model_name: str, **kwargs: Any):
    if model_name == "official_svr":
        estimator = SVR(gamma=0.01, C=100) if SVR is not None else NumpyFallbackRegressor("svr", ridge=1e-2)
        return OfficialSklearnBaseline(model_name, estimator, **kwargs)
    if model_name == "official_rf":
        estimator = (
            RandomForestRegressor(n_estimators=20, oob_score=True, max_depth=None, min_samples_split=2, min_samples_leaf=2, n_jobs=1)
            if RandomForestRegressor is not None
            else NumpyFallbackRegressor("rf", ridge=1e-2)
        )
        return OfficialSklearnBaseline(model_name, estimator, **kwargs)
    if model_name == "official_gbrt":
        estimator = (
            GradientBoostingRegressor(n_estimators=20, min_samples_split=2, min_samples_leaf=2, max_depth=None)
            if GradientBoostingRegressor is not None
            else NumpyFallbackRegressor("gbrt", ridge=1e-2)
        )
        return OfficialSklearnBaseline(model_name, estimator, **kwargs)
    if model_name == "official_gm_p":
        return OfficialTorchPairBaseline(model_name, "gm_p", **kwargs)
    if model_name == "official_gm_e":
        return OfficialTorchPairBaseline(model_name, "gm_e", **kwargs)
    if model_name == "official_dgm":
        return OfficialTorchPairBaseline(model_name, "dgm", **kwargs)
    if model_name == "official_netgan":
        return OfficialTorchPairBaseline(model_name, "netgan", **kwargs)
    if model_name == "official_diffodgen":
        return OfficialTorchPairBaseline(model_name, "diffodgen", **kwargs)
    if model_name == "official_wedan":
        return OfficialTorchPairBaseline(model_name, "wedan", **kwargs)
    if model_name == "official_gmel":
        return OfficialDenseGraphBaseline(model_name, **kwargs)
    raise ValueError(f"Unknown official model: {model_name}")
