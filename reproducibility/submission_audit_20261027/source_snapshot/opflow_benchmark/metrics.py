"""Metrics for raw OD counts and row-normalized destination allocation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-8


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _valid_rows(p_true: np.ndarray, valid_origin=None, eps: float = EPS) -> np.ndarray:
    if valid_origin is not None:
        valid = np.asarray(valid_origin, dtype=bool).copy()
    else:
        valid = np.sum(np.asarray(p_true, dtype=float), axis=1) > eps
    return valid & np.isfinite(p_true).all(axis=1)


def _normalize_rows(arr: np.ndarray, valid: np.ndarray, eps: float = EPS) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    safe = np.maximum(arr[valid], 0.0)
    denom = np.maximum(safe.sum(axis=1, keepdims=True), eps)
    out[valid] = safe / denom
    return out


def rmse(y_true, y_pred) -> float:
    """Root mean squared error over all OD cells."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error over all OD cells."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def log_rmse(y_true, y_pred) -> float:
    """RMSE over log1p-transformed non-negative OD counts."""
    y_true = np.maximum(_as_float_array(y_true), 0.0)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    return rmse(np.log1p(y_true), np.log1p(y_pred))


def nrmse_mean(y_true, y_pred, eps: float = EPS) -> float:
    """RMSE divided by mean true flow. This is the benchmark default."""
    y_true = _as_float_array(y_true)
    denom = float(np.mean(y_true))
    if abs(denom) <= eps:
        return float("nan")
    return rmse(y_true, y_pred) / denom


def nrmse_official(y_true, y_pred, eps: float = EPS) -> float:
    """RMSE divided by true-flow standard deviation, matching the official repo style."""
    y_true = _as_float_array(y_true)
    denom = float(np.std(y_true))
    if denom <= eps:
        return float("nan")
    return rmse(y_true, y_pred) / denom


def nrmse(y_true, y_pred, eps: float = EPS) -> float:
    """Backward-compatible alias for official-style NRMSE."""
    return nrmse_official(y_true, y_pred, eps=eps)


def global_cpc(y_true, y_pred, eps: float = EPS) -> float:
    """Common part of commuters for non-negative OD matrices."""
    y_true = np.maximum(_as_float_array(y_true), 0.0)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    denom = float(y_true.sum() + y_pred.sum())
    if denom <= eps:
        return float("nan")
    return float(2.0 * np.minimum(y_true, y_pred).sum() / denom)


def row_kl(p_true, p_pred, valid_origin=None, eps: float = EPS) -> float:
    """Average KL(P_true || P_pred) over valid origin rows."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if not np.any(valid):
        return float("nan")
    p_true = _normalize_rows(p_true, valid, eps=eps)
    p_pred = _normalize_rows(np.maximum(p_pred, eps), valid, eps=eps)
    terms = np.where(p_true[valid] > 0, p_true[valid] * (np.log(p_true[valid] + eps) - np.log(p_pred[valid] + eps)), 0.0)
    return float(np.mean(np.sum(terms, axis=1)))


def row_cpc(p_true, p_pred, valid_origin=None, eps: float = EPS) -> float:
    """Average row-wise CPC for destination-share distributions."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if not np.any(valid):
        return float("nan")
    p_true = _normalize_rows(p_true, valid, eps=eps)
    p_pred = _normalize_rows(p_pred, valid, eps=eps)
    denom = p_true[valid].sum(axis=1) + p_pred[valid].sum(axis=1)
    scores = np.divide(2.0 * np.minimum(p_true[valid], p_pred[valid]).sum(axis=1), denom, out=np.full_like(denom, np.nan), where=denom > eps)
    return float(np.nanmean(scores))


def row_rmse(p_true, p_pred, valid_origin=None, eps: float = EPS) -> float:
    """Mean per-origin RMSE between row-share vectors."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if not np.any(valid):
        return float("nan")
    p_true = _normalize_rows(p_true, valid, eps=eps)
    p_pred = _normalize_rows(p_pred, valid, eps=eps)
    return float(np.mean(np.sqrt(np.mean((p_true[valid] - p_pred[valid]) ** 2, axis=1))))


def row_mae(p_true, p_pred, valid_origin=None, eps: float = EPS) -> float:
    """Mean per-origin MAE between row-share vectors."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if not np.any(valid):
        return float("nan")
    p_true = _normalize_rows(p_true, valid, eps=eps)
    p_pred = _normalize_rows(p_pred, valid, eps=eps)
    return float(np.mean(np.mean(np.abs(p_true[valid] - p_pred[valid]), axis=1)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(left * right) / denom)


def rank_corr(p_true, p_pred, valid_origin=None, eps: float = EPS) -> float:
    """Average row-wise Spearman correlation, skipping undefined rows."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    values = []
    for i in np.flatnonzero(valid):
        if np.unique(p_true[i]).size <= 1 or np.unique(p_pred[i]).size <= 1:
            continue
        corr = _pearson(_average_ranks(p_true[i]), _average_ranks(p_pred[i]))
        if np.isfinite(corr):
            values.append(corr)
    return float(np.mean(values)) if values else float("nan")


def topk_overlap(p_true, p_pred, valid_origin=None, k: int = 10, eps: float = EPS) -> float:
    """Average intersection-over-k for eligible top-k rows."""
    return topk_overlap_with_coverage(p_true, p_pred, valid_origin=valid_origin, k=k, eps=eps)["value"]


def topk_overlap_with_coverage(p_true, p_pred, valid_origin=None, k: int = 10, eps: float = EPS) -> dict[str, float]:
    """Top-k overlap with explicit row coverage.

    Rows with N <= k or <= k positive true destinations are excluded so the
    metric cannot become 1.0 only because top-k covers the whole row.
    """
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if p_true.ndim != 2:
        raise ValueError("p_true must be a 2-D matrix")
    k_eff = int(k)
    n_total = int(np.sum(valid))
    if k_eff <= 0 or not np.any(valid) or p_true.shape[1] <= k_eff:
        return {"value": float("nan"), "coverage": 0.0, "n_rows_used": 0, "n_rows_total": n_total}
    scores = []
    for i in np.flatnonzero(valid):
        if int(np.sum(p_true[i] > eps)) <= k_eff:
            continue
        true_idx = set(np.argpartition(-p_true[i], k_eff - 1)[:k_eff].tolist())
        pred_idx = set(np.argpartition(-p_pred[i], k_eff - 1)[:k_eff].tolist())
        scores.append(len(true_idx & pred_idx) / float(k_eff))
    return {
        "value": float(np.mean(scores)) if scores else float("nan"),
        "coverage": float(len(scores) / n_total) if n_total else 0.0,
        "n_rows_used": int(len(scores)),
        "n_rows_total": n_total,
    }


def ndcg_at_k(p_true, p_pred, valid_origin=None, k: int = 10, eps: float = EPS) -> float:
    """Average row-wise NDCG@k using true shares as graded relevance."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    valid = _valid_rows(p_true, valid_origin, eps=eps)
    if p_true.ndim != 2:
        raise ValueError("p_true must be a 2-D matrix")
    k_eff = min(int(k), p_true.shape[1])
    if k_eff <= 0 or not np.any(valid):
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=float))
    scores = []
    for i in np.flatnonzero(valid):
        pred_idx = np.argsort(-p_pred[i], kind="mergesort")[:k_eff]
        ideal_idx = np.argsort(-p_true[i], kind="mergesort")[:k_eff]
        dcg = float(np.sum(p_true[i, pred_idx] * discounts))
        idcg = float(np.sum(p_true[i, ideal_idx] * discounts))
        if idcg > eps:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else float("nan")


def total_flow_ratio(y_true, y_pred, eps: float = EPS) -> float:
    y_true = np.maximum(_as_float_array(y_true), 0.0)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    return float(y_pred.sum() / max(float(y_true.sum()), eps))


def outflow_rmse(outflow_true, outflow_pred) -> float:
    return rmse(outflow_true, outflow_pred)


def outflow_log_rmse(outflow_true, outflow_pred) -> float:
    outflow_true = np.maximum(_as_float_array(outflow_true), 0.0)
    outflow_pred = np.maximum(_as_float_array(outflow_pred), 0.0)
    return rmse(np.log1p(outflow_true), np.log1p(outflow_pred))


def outflow_corr(outflow_true, outflow_pred) -> float:
    return _pearson(_as_float_array(outflow_true), _as_float_array(outflow_pred))


def scale_diagnostics(y_true, y_pred, eps: float = EPS) -> dict[str, float]:
    y_true = np.maximum(_as_float_array(y_true), 0.0)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    out_true = y_true.sum(axis=1)
    out_pred = y_pred.sum(axis=1)
    return {
        "total_flow_ratio": total_flow_ratio(y_true, y_pred, eps=eps),
        "outflow_rmse": outflow_rmse(out_true, out_pred),
        "outflow_log_rmse": outflow_log_rmse(out_true, out_pred),
        "outflow_corr": outflow_corr(out_true, out_pred),
        "true_outflow_mean": float(np.mean(out_true)) if out_true.size else float("nan"),
        "true_outflow_std": float(np.std(out_true)) if out_true.size else float("nan"),
        "pred_outflow_mean": float(np.mean(out_pred)) if out_pred.size else float("nan"),
        "pred_outflow_std": float(np.std(out_pred)) if out_pred.size else float("nan"),
        "true_total_flow": float(y_true.sum()),
        "pred_total_flow": float(y_pred.sum()),
        "median_true_outflow": float(np.median(out_true)) if out_true.size else float("nan"),
        "median_pred_outflow": float(np.median(out_pred)) if out_pred.size else float("nan"),
    }


def allocation_metrics(p_true, p_pred, valid_origin=None, topk: int = 10, eps: float = EPS) -> dict[str, float]:
    """Compute metrics that depend only on row-normalized allocation."""
    p_true = _as_float_array(p_true)
    p_pred = _as_float_array(p_pred)
    metrics = {
        "row_kl": row_kl(p_true, p_pred, valid_origin=valid_origin, eps=eps),
        "row_cpc": row_cpc(p_true, p_pred, valid_origin=valid_origin, eps=eps),
        "row_rmse": row_rmse(p_true, p_pred, valid_origin=valid_origin, eps=eps),
        "row_mae": row_mae(p_true, p_pred, valid_origin=valid_origin, eps=eps),
        "rank_corr": rank_corr(p_true, p_pred, valid_origin=valid_origin, eps=eps),
    }
    for k in (1, 3, 5, 10):
        tk = topk_overlap_with_coverage(p_true, p_pred, valid_origin=valid_origin, k=k, eps=eps)
        metrics[f"topk@{k}"] = tk["value"]
        metrics[f"topk@{k}_coverage"] = tk["coverage"]
        metrics[f"topk@{k}_n_rows_used"] = tk["n_rows_used"]
        metrics[f"topk@{k}_n_rows_total"] = tk["n_rows_total"]
        metrics[f"topk_overlap@{k}"] = tk["value"]
        metrics[f"ndcg@{k}"] = ndcg_at_k(p_true, p_pred, valid_origin=valid_origin, k=k, eps=eps)
    metrics[f"topk_overlap@{topk}"] = metrics.get(f"topk@{topk}", topk_overlap(p_true, p_pred, valid_origin=valid_origin, k=topk, eps=eps))
    return metrics


def count_metrics(y_true, y_pred, eps: float = EPS) -> dict[str, float]:
    """Compute metrics that depend on count scale."""
    y_true = _as_float_array(y_true)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "log_rmse": log_rmse(y_true, y_pred),
        "nrmse_mean": nrmse_mean(y_true, y_pred, eps=eps),
        "nrmse_official": nrmse_official(y_true, y_pred, eps=eps),
        "nrmse": nrmse_official(y_true, y_pred, eps=eps),
        "global_cpc": global_cpc(y_true, y_pred, eps=eps),
    }
    metrics.update(scale_diagnostics(y_true, y_pred, eps=eps))
    return metrics


def ood_gap(metric_id: float, metric_ood: float, higher_is_better: bool) -> float:
    """OOD degradation: score drops are positive, error increases are positive."""
    if metric_id is None or metric_ood is None:
        return float("nan")
    if not np.isfinite(metric_id) or not np.isfinite(metric_ood):
        return float("nan")
    return float(metric_id - metric_ood if higher_is_better else metric_ood - metric_id)


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Return mean and standard error for finite scalar values."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "se": se, "n": int(arr.size)}


def matrix_to_share(y_raw, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute row-normalized shares, outflow, and valid-origin mask."""
    y_raw = np.maximum(_as_float_array(y_raw), 0.0)
    outflow = y_raw.sum(axis=1)
    valid = outflow > eps
    share = np.zeros_like(y_raw, dtype=float)
    share[valid] = y_raw[valid] / outflow[valid, None]
    return share, outflow, valid


def evaluate_area(y_true, y_pred, p_true=None, p_pred=None, valid_origin=None, topk: int = 10) -> dict[str, float]:
    """Compute all benchmark metrics for one area."""
    y_true = _as_float_array(y_true)
    y_pred = np.maximum(_as_float_array(y_pred), 0.0)
    if p_true is None:
        p_true, _, valid_origin = matrix_to_share(y_true)
    if p_pred is None:
        p_pred, _, _ = matrix_to_share(y_pred)
    metrics = count_metrics(y_true, y_pred)
    metrics.update(allocation_metrics(p_true, p_pred, valid_origin=valid_origin, topk=topk))
    return metrics
