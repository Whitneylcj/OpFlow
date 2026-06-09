"""Shared helpers for LargeCommutingOD baselines."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from opflow_benchmark.metrics import allocation_metrics, count_metrics, evaluate_area, summarize


EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(device)


def finite_mean_std(values: list[np.ndarray]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    mean = float(arr.mean())
    std = float(arr.std())
    return mean, std if std > EPS else 1.0


class FeatureStats:
    """Log-and-standardize node features and distances from train areas."""

    def __init__(self) -> None:
        self.node_mean: np.ndarray | None = None
        self.node_std: np.ndarray | None = None
        self.dist_mean: float = 0.0
        self.dist_std: float = 1.0

    def fit(self, areas: list[dict[str, Any]]) -> "FeatureStats":
        node_chunks = []
        dist_chunks = []
        for area in areas:
            if area["node_feat"] is not None:
                node_chunks.append(np.log1p(area["node_feat"].detach().cpu().numpy()))
            if area["distance"] is not None:
                dist = np.log1p(np.maximum(area["distance"].detach().cpu().numpy(), 0.0))
                dist_chunks.append(dist[np.isfinite(dist)])
        if node_chunks:
            nodes = np.concatenate(node_chunks, axis=0)
            self.node_mean = nodes.mean(axis=0)
            self.node_std = nodes.std(axis=0)
            self.node_std[self.node_std <= EPS] = 1.0
        self.dist_mean, self.dist_std = finite_mean_std(dist_chunks)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "node_mean": None if self.node_mean is None else self.node_mean.tolist(),
            "node_std": None if self.node_std is None else self.node_std.tolist(),
            "dist_mean": self.dist_mean,
            "dist_std": self.dist_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureStats":
        obj = cls()
        if state.get("node_mean") is not None:
            obj.node_mean = np.asarray(state["node_mean"], dtype=float)
            obj.node_std = np.asarray(state["node_std"], dtype=float)
        obj.dist_mean = float(state.get("dist_mean", 0.0))
        obj.dist_std = float(state.get("dist_std", 1.0)) or 1.0
        return obj

    def transform_node(self, node_feat: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if node_feat is None:
            raise ValueError("node_feat is required for this baseline")
        x = torch.log1p(torch.clamp(node_feat.to(device), min=0.0))
        if self.node_mean is None or self.node_std is None:
            return x
        mean = torch.as_tensor(self.node_mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.node_std, dtype=torch.float32, device=device)
        return (x - mean) / std

    def transform_distance(self, distance: torch.Tensor | None, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if distance is None:
            raw = torch.zeros((n, n), dtype=torch.float32, device=device)
        else:
            raw = torch.clamp(distance.to(device), min=0.0)
        logd = torch.log1p(raw)
        scaled = (logd - self.dist_mean) / max(self.dist_std, EPS)
        return raw, scaled


def tensor_area(area: dict[str, Any], device: torch.device, stats: FeatureStats | None = None) -> dict[str, Any]:
    y = area["y_raw"].to(device)
    p = area["p_share"].to(device)
    outflow = area["outflow"].to(device)
    valid = area["valid_origin"].to(device)
    n = int(y.shape[0])
    if stats is None:
        x = area["node_feat"].to(device) if area["node_feat"] is not None else None
        dist_raw = area["distance"].to(device) if area["distance"] is not None else torch.zeros((n, n), dtype=torch.float32, device=device)
        dist_scaled = torch.log1p(torch.clamp(dist_raw, min=0.0))
    else:
        x = stats.transform_node(area["node_feat"], device)
        dist_raw, dist_scaled = stats.transform_distance(area["distance"], n, device)
    adj = area["adjacency"].to(device) if area["adjacency"] is not None else torch.zeros((n, n), dtype=torch.float32, device=device)
    return {
        "area_id": area["area_id"],
        "y": y,
        "p": p,
        "outflow": outflow,
        "valid": valid,
        "x": x,
        "distance_raw": dist_raw,
        "distance": dist_scaled,
        "adjacency": adj,
    }


def pair_node_features(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    xi = x[:, None, :].expand(n, n, x.shape[1])
    xj = x[None, :, :].expand(n, n, x.shape[1])
    return torch.cat([xi, xj, xj - xi, torch.abs(xj - xi)], dim=-1)


def masked_share_loss(p_true: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    logp = F.log_softmax(logits[valid], dim=1)
    return -(torch.clamp(p_true[valid], min=0.0) * logp).sum(dim=1).mean()


def compose_prediction(logits: torch.Tensor, outflow_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p_hat = F.softmax(logits, dim=1)
    y_hat = outflow_hat[:, None] * p_hat
    return y_hat, p_hat


def positive_scale_from_log(raw_log_scale: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained log-scale prediction to positive origin flow."""
    return torch.exp(torch.clamp(raw_log_scale, -5.0, 16.0)) + EPS


def evaluate_area_predictions(
    area: dict[str, Any],
    y_hat: torch.Tensor,
    p_hat: torch.Tensor,
    topk: int = 10,
    extra: dict[str, float] | None = None,
    allocation: dict[str, float] | None = None,
) -> dict[str, Any]:
    y_true = area["y_raw"].detach().cpu().numpy()
    y_pred = y_hat.detach().cpu().numpy()
    if allocation is None:
        metrics = evaluate_area(
            y_true,
            y_pred,
            p_true=area["p_share"].detach().cpu().numpy(),
            p_pred=p_hat.detach().cpu().numpy(),
            valid_origin=area["valid_origin"].detach().cpu().numpy(),
            topk=topk,
        )
    else:
        metrics = count_metrics(y_true, y_pred)
        metrics.update(allocation)
    metrics["area_id"] = area["area_id"]
    metrics["n_regions"] = int(area["y_raw"].shape[0])
    if extra:
        metrics.update(extra)
    return metrics


def evaluate_area_allocation(area: dict[str, Any], p_hat: torch.Tensor, topk: int = 10) -> dict[str, float]:
    return allocation_metrics(
        area["p_share"].detach().cpu().numpy(),
        p_hat.detach().cpu().numpy(),
        valid_origin=area["valid_origin"].detach().cpu().numpy(),
        topk=topk,
    )


def timed_evaluate_area_allocation(profile: dict[str, float], area: dict[str, Any], p_hat: torch.Tensor, topk: int = 10) -> dict[str, float]:
    start = time.perf_counter()
    metrics = evaluate_area_allocation(area, p_hat, topk=topk)
    profile_record(profile, "metric_seconds", time.perf_counter() - start)
    return metrics


def profile_record(profile: dict[str, float], key: str, seconds: float) -> None:
    profile[key] = float(profile.get(key, 0.0) + seconds)


def profile_count(profile: dict[str, float], key: str, count: int = 1) -> None:
    profile[key] = float(profile.get(key, 0.0) + int(count))


def timed_evaluate_area_predictions(
    profile: dict[str, float],
    area: dict[str, Any],
    y_hat: torch.Tensor,
    p_hat: torch.Tensor,
    topk: int = 10,
    extra: dict[str, float] | None = None,
    allocation: dict[str, float] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    row = evaluate_area_predictions(area, y_hat, p_hat, topk=topk, extra=extra, allocation=allocation)
    profile_record(profile, "metric_seconds", time.perf_counter() - start)
    profile_count(profile, "metric_rows", 1)
    return row


def oracle_scale_prediction(area: dict[str, Any], p_hat: torch.Tensor) -> torch.Tensor:
    """Compose true origin outflow with predicted allocation shares."""
    return area["outflow"].to(p_hat.device)[:, None] * p_hat


def mean_scale_prediction(area: dict[str, Any], p_hat: torch.Tensor, mean_outflow: float) -> torch.Tensor:
    """Compose a train-mean origin outflow with predicted allocation shares."""
    outflow = torch.full((p_hat.shape[0],), float(mean_outflow), dtype=p_hat.dtype, device=p_hat.device)
    return outflow[:, None] * p_hat


def aggregate_area_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted({key for row in rows for key in row if key not in {"area_id", "n_regions", "split", "variant"}})
    return {key: summarize(float(row[key]) for row in rows if key in row) for key in metric_keys}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
