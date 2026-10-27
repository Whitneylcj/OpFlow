"""Row-wise ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ("env_id", "origin_id")


def _safe_spearman(group: pd.DataFrame, true_col: str, pred_col: str) -> float:
    true_vals = group[true_col].astype(float)
    pred_vals = group[pred_col].astype(float)
    if true_vals.nunique() <= 1 or pred_vals.nunique() <= 1:
        return float("nan")
    true_rank = true_vals.rank(method="average")
    pred_rank = pred_vals.rank(method="average")
    return float(true_rank.corr(pred_rank))


def mean_spearman_by_row(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    group_cols=DEFAULT_GROUP_COLS,
) -> float:
    """Compute mean row-wise Spearman rank correlation."""
    values = []
    for _, group in records.groupby(list(group_cols)):
        metric = _safe_spearman(group, true_col, pred_col)
        if not np.isnan(metric):
            values.append(metric)
    return float(np.mean(values)) if values else float("nan")


def mean_topk_overlap_by_row(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    k: int = 5,
    group_cols=DEFAULT_GROUP_COLS,
) -> float:
    """Compute mean row-wise Top-k overlap."""
    overlaps = []
    for _, group in records.groupby(list(group_cols)):
        true_idx = set(group.nlargest(k, true_col)["dest_id"].tolist())
        pred_idx = set(group.nlargest(k, pred_col)["dest_id"].tolist())
        denom = max(len(true_idx | pred_idx), 1)
        overlaps.append(len(true_idx & pred_idx) / denom)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def mean_ndcg_by_row(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    k: int = 5,
    group_cols=DEFAULT_GROUP_COLS,
) -> float:
    """Compute mean row-wise NDCG@k using true scores as graded relevance."""
    ndcgs = []
    for _, group in records.groupby(list(group_cols)):
        ranked = group.sort_values(pred_col, ascending=False).head(k)
        ideal = group.sort_values(true_col, ascending=False).head(k)
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum((2 ** ranked[true_col].to_numpy(dtype=float) - 1) * discounts))
        idcg = float(np.sum((2 ** ideal[true_col].to_numpy(dtype=float) - 1) * discounts))
        ndcgs.append(dcg / idcg if idcg > 0 else float("nan"))
    valid = [value for value in ndcgs if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")
