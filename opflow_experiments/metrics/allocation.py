"""Row-normalized allocation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ("env_id", "origin_id")


def compute_row_shares(
    records: pd.DataFrame,
    value_col: str,
    group_cols=DEFAULT_GROUP_COLS,
    eps: float = 1e-8,
) -> pd.Series:
    """Compute row-normalized shares for a value column."""
    values = records[value_col].astype(float)
    row_sum = records.groupby(list(group_cols))[value_col].transform("sum").astype(float)
    return values / np.maximum(row_sum, eps)


def row_kl(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    group_cols=DEFAULT_GROUP_COLS,
    eps: float = 1e-8,
) -> float:
    """Compute average row-wise KL divergence."""
    working = records.copy()
    working["_true"] = np.maximum(working[true_col].astype(float), eps)
    working["_pred"] = np.maximum(working[pred_col].astype(float), eps)
    row_sum = working.groupby(list(group_cols))[true_col].transform("sum").astype(float)
    valid = row_sum > eps
    kl = working.loc[valid, "_true"] * np.log(working.loc[valid, "_true"] / working.loc[valid, "_pred"])
    per_row = kl.groupby([working.loc[valid, col] for col in group_cols]).sum()
    return float(per_row.mean()) if not per_row.empty else float("nan")


def row_js(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    group_cols=DEFAULT_GROUP_COLS,
    eps: float = 1e-8,
) -> float:
    """Compute average row-wise Jensen-Shannon divergence."""
    working = records.copy()
    working["_m"] = 0.5 * (working[true_col].astype(float) + working[pred_col].astype(float))
    kl_true = row_kl(working, true_col=true_col, pred_col="_m", group_cols=group_cols, eps=eps)
    kl_pred = row_kl(working, true_col=pred_col, pred_col="_m", group_cols=group_cols, eps=eps)
    return 0.5 * (kl_true + kl_pred)


def row_l1(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    group_cols=DEFAULT_GROUP_COLS,
) -> float:
    """Compute average row-wise L1 distance."""
    working = records.copy()
    working["_err"] = np.abs(working[true_col].astype(float) - working[pred_col].astype(float))
    per_row = working.groupby(list(group_cols))["_err"].sum()
    return float(per_row.mean()) if not per_row.empty else float("nan")


def row_cpc(
    records: pd.DataFrame,
    true_col: str = "pi_true",
    pred_col: str = "pi_hat",
    group_cols=DEFAULT_GROUP_COLS,
    eps: float = 1e-8,
) -> float:
    """Compute average row-wise common-part-of-commuters score."""
    working = records.copy()
    working["_true"] = np.maximum(working[true_col].astype(float), 0.0)
    working["_pred"] = np.maximum(working[pred_col].astype(float), 0.0)
    grouped = working.groupby(list(group_cols), sort=False)
    numer = grouped[["_true", "_pred"]].apply(lambda g: 2.0 * np.minimum(g["_true"], g["_pred"]).sum())
    denom = grouped["_true"].sum() + grouped["_pred"].sum()
    cpc = numer / np.maximum(denom, eps)
    return float(cpc.mean()) if not cpc.empty else float("nan")
