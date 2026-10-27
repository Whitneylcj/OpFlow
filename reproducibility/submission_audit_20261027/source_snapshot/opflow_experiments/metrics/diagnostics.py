"""DGP diagnostic metrics and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opflow_experiments.metrics.allocation import compute_row_shares, row_kl, row_l1
from opflow_experiments.metrics.prediction import mae, poisson_deviance, rmse
from opflow_experiments.metrics.ranking import mean_spearman_by_row, mean_topk_overlap_by_row


def _safe_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    subset = frame[[left, right]].dropna()
    if subset.empty or subset[left].nunique() <= 1 or subset[right].nunique() <= 1:
        return float("nan")
    return float(subset[left].corr(subset[right]))


def dgp_correlation_report(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return correlation diagnostics by split."""
    report: dict[str, dict[str, float]] = {}
    for split, frame in records.groupby("split"):
        row_entropy = (
            frame.assign(_entropy_term=-(frame["pi_true"] * np.log(np.maximum(frame["pi_true"], 1e-8))))
            .groupby(["env_id", "origin_id"])["_entropy_term"]
            .sum()
        )
        report[split] = {
            "corr_neg_distance_true_score": _safe_corr(frame.assign(neg_distance=-frame["distance"]), "neg_distance", "opportunity_score_true"),
            "corr_history_true_score": _safe_corr(frame, "shortcut_history", "opportunity_score_true"),
            "corr_opportunity_true_score": _safe_corr(frame, "opportunity_dest", "opportunity_score_true"),
            "origin_scale_mean": float(frame["origin_scale_true"].mean()),
            "origin_scale_std": float(frame["origin_scale_true"].std(ddof=0)),
            "row_entropy_mean": float(row_entropy.mean()),
            "row_entropy_std": float(row_entropy.std(ddof=0)),
        }
    return report


def corr_distance_score(records: pd.DataFrame) -> dict[str, float]:
    """Correlation between negative distance and true score by split."""
    return {
        split: _safe_corr(frame.assign(neg_distance=-frame["distance"]), "neg_distance", "opportunity_score_true")
        for split, frame in records.groupby("split")
    }


def corr_history_score(records: pd.DataFrame) -> dict[str, float]:
    """Correlation between history shortcut and true score by split."""
    return {split: _safe_corr(frame, "shortcut_history", "opportunity_score_true") for split, frame in records.groupby("split")}


def corr_opportunity_score(records: pd.DataFrame) -> dict[str, float]:
    """Correlation between destination opportunity and true score by split."""
    return {split: _safe_corr(frame, "opportunity_dest", "opportunity_score_true") for split, frame in records.groupby("split")}


def origin_scale_drift(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean and std of log origin scale by split."""
    report = {}
    for split, frame in records.groupby("split"):
        log_scale = np.log(np.maximum(frame["origin_scale_true"].to_numpy(dtype=float), 1e-8))
        report[split] = {
            "mean_log_scale": float(log_scale.mean()),
            "std_log_scale": float(log_scale.std()),
        }
    return report


def id_ood_gap(metric_id: float, metric_ood: float) -> float:
    """Simple ID/OOD metric gap."""
    return float(metric_ood - metric_id)


def oracle_comparison_report(records: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Compare oracle-style predictors by split."""
    report: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in records.groupby("split"):
        split_report = {}
        for label, lambda_col in (
            ("opportunity_oracle", "lambda_hat_opp"),
            ("shortcut_oracle", "lambda_hat_shortcut"),
        ):
            working = frame.copy()
            working["pi_hat"] = compute_row_shares(working, lambda_col)
            split_report[label] = {
                "rmse": rmse(working["y_obs"], working[lambda_col]),
                "mae": mae(working["y_obs"], working[lambda_col]),
                "poisson_deviance": poisson_deviance(working["y_obs"], working[lambda_col]),
                "row_kl": row_kl(working),
                "row_l1": row_l1(working),
                "rank_corr": mean_spearman_by_row(working),
                "top5_overlap": mean_topk_overlap_by_row(working, k=5),
            }
        report[split] = split_report
    return report
