"""Unified evaluator for pair prediction and row allocation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from opflow_experiments.metrics.allocation import row_cpc, row_kl
from opflow_experiments.metrics.prediction import mae, rmse
from opflow_experiments.metrics.ranking import mean_spearman_by_row, mean_topk_overlap_by_row


def _safe_corr(left, right) -> float:
    """Correlation with stable NaN behavior for constant or tiny inputs."""
    left_s = pd.Series(left, dtype=float)
    right_s = pd.Series(right, dtype=float)
    valid = left_s.notna() & right_s.notna()
    left_s = left_s.loc[valid]
    right_s = right_s.loc[valid]
    if len(left_s) < 2:
        return float("nan")
    if np.isclose(left_s.std(ddof=0), 0.0) or np.isclose(right_s.std(ddof=0), 0.0):
        return float("nan")
    return float(left_s.corr(right_s))


def _origin_scale_error(frame: pd.DataFrame) -> float:
    """Mean log-scale error at the origin row level."""
    scale_col = "R_oracle" if "R_oracle" in frame else "a_true"
    rows = (
        frame.groupby(["env_id", "origin_id"], sort=False)
        .agg(lambda_hat_sum=("lambda_hat", "sum"), scale_true=(scale_col, "first"))
        .reset_index()
    )
    err = np.abs(np.log(np.maximum(rows["lambda_hat_sum"].to_numpy(dtype=float), 1e-8)) - np.log(np.maximum(rows["scale_true"].to_numpy(dtype=float), 1e-8)))
    return float(np.mean(err))


def _row_share_sum_error(frame: pd.DataFrame) -> float:
    """Maximum row-normalization error for predicted shares."""
    row_sums = frame.groupby(["env_id", "origin_id"], sort=False)["pi_hat"].sum().to_numpy(dtype=float)
    return float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else float("nan")


def _log_rmse(true, pred) -> float:
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(np.maximum(pred_arr, 0.0)) - np.log1p(np.maximum(true_arr, 0.0))) ** 2)))


def _global_cpc(true, pred, eps: float = 1e-8) -> float:
    true_arr = np.maximum(np.asarray(true, dtype=float), 0.0)
    pred_arr = np.maximum(np.asarray(pred, dtype=float), 0.0)
    return float(2.0 * np.minimum(true_arr, pred_arr).sum() / max(true_arr.sum() + pred_arr.sum(), eps))


def _to_numpy(value):
    """Convert tensor-like values to NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _center_by_row(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    out = pd.Series(values, index=frame.index, dtype=float)
    return (out - out.groupby([frame["env_id"], frame["origin_id"]]).transform("mean")).to_numpy(dtype=float)


def _score_mse_calibrated(true_score: np.ndarray, pred_score: np.ndarray) -> float:
    valid = np.isfinite(true_score) & np.isfinite(pred_score)
    if valid.sum() < 2:
        return float("nan")
    x = pred_score[valid]
    y = true_score[valid]
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    if not np.isfinite(coef).all():
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fitted = design @ coef
    if not np.isfinite(fitted).all():
        return float("nan")
    return float(np.mean((y - fitted) ** 2))


def _potential_mse(true_score: np.ndarray, pred_score: np.ndarray) -> float:
    valid = np.isfinite(true_score) & np.isfinite(pred_score)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((true_score[valid] - pred_score[valid]) ** 2))


def _rowwise_potential_mse(frame: pd.DataFrame, true_score: np.ndarray, pred_score: np.ndarray) -> float:
    valid = np.isfinite(true_score) & np.isfinite(pred_score)
    if valid.sum() == 0:
        return float("nan")
    working = frame.loc[valid, ["env_id", "origin_id"]].copy()
    working["_err"] = (true_score[valid] - pred_score[valid]) ** 2
    per_row = working.groupby(["env_id", "origin_id"], sort=False)["_err"].mean()
    return float(per_row.mean()) if not per_row.empty else float("nan")


def evaluate_predictions(records: pd.DataFrame, prediction: dict, topk: int = 5) -> dict[str, dict[str, float]]:
    """Evaluate a prediction bundle by split with row-wise aggregation."""
    enriched = records.copy()
    enriched["lambda_hat"] = _to_numpy(prediction["lambda_hat"])
    enriched["pi_hat"] = _to_numpy(prediction["pi_hat"])
    score_hat = prediction.get("extra", {}).get("score_hat")
    if score_hat is not None:
        enriched["score_hat"] = _to_numpy(score_hat)
    report: dict[str, dict[str, float]] = {}
    for split, frame in enriched.groupby("split"):
        if {"distance", "shortcut_history"}.issubset(frame.columns):
            shortcut_sensitivity = _safe_corr(
                frame["pi_hat"],
                ((-frame["distance"]) + frame["shortcut_history"]) / 2.0,
            )
            distance_sensitivity = _safe_corr(frame["pi_hat"], -frame["distance"])
            history_sensitivity = _safe_corr(frame["pi_hat"], frame["shortcut_history"])
        else:
            shortcut_sensitivity = float("nan")
            distance_sensitivity = float("nan")
            history_sensitivity = float("nan")
        split_report = {
            "rmse": rmse(frame["y_obs"], frame["lambda_hat"]),
            "mae": mae(frame["y_obs"], frame["lambda_hat"]),
            "log_rmse": _log_rmse(frame["y_obs"], frame["lambda_hat"]),
            "global_cpc": _global_cpc(frame["y_obs"], frame["lambda_hat"]),
            "row_kl": row_kl(frame, true_col="pi_oracle", pred_col="pi_hat"),
            "row_cpc": row_cpc(frame, true_col="pi_oracle", pred_col="pi_hat"),
            "rank_corr": mean_spearman_by_row(frame, true_col="pi_oracle", pred_col="pi_hat"),
            "topk_overlap": mean_topk_overlap_by_row(frame, true_col="pi_oracle", pred_col="pi_hat", k=topk),
            "shortcut_sensitivity": shortcut_sensitivity,
            "distance_sensitivity": distance_sensitivity,
            "history_sensitivity": history_sensitivity,
            "origin_scale_error": _origin_scale_error(frame),
            "row_share_sum_error": _row_share_sum_error(frame),
        }
        if "psi_oracle" in frame and "score_hat" in frame:
            true_centered = frame["psi_oracle"].to_numpy(dtype=float)
            pred_centered = _center_by_row(frame, frame["score_hat"].to_numpy(dtype=float))
            score_frame = frame.copy()
            score_frame["_qbar_hat"] = pred_centered
            split_report["score_mse"] = _score_mse_calibrated(true_centered, pred_centered)
            split_report["pmse"] = _rowwise_potential_mse(frame, true_centered, pred_centered)
            split_report["score_corr"] = mean_spearman_by_row(score_frame, true_col="psi_oracle", pred_col="_qbar_hat")
        else:
            split_report["score_mse"] = float("nan")
            split_report["pmse"] = float("nan")
            split_report["score_corr"] = float("nan")
        report[split] = split_report
    report["ood_id_gap"] = {
        "rmse_gap": report["test"]["rmse"] - report["train"]["rmse"],
        "mae_gap": report["test"]["mae"] - report["train"]["mae"],
        "row_kl_gap": report["test"]["row_kl"] - report["train"]["row_kl"],
        "rank_corr_gap": report["test"]["rank_corr"] - report["train"]["rank_corr"],
        "topk_gap": report["test"]["topk_overlap"] - report["train"]["topk_overlap"],
        "rmse_ood_gap": report["test"]["rmse"] - report["train"]["rmse"],
        "mae_ood_gap": report["test"]["mae"] - report["train"]["mae"],
        "row_kl_ood_gap": report["test"]["row_kl"] - report["train"]["row_kl"],
        "rankcorr_ood_drop": report["train"]["rank_corr"] - report["test"]["rank_corr"],
        "topk_ood_drop": report["train"]["topk_overlap"] - report["test"]["topk_overlap"],
    }
    return report


def build_results_table(regime: str, model_reports: dict[str, dict]) -> pd.DataFrame:
    """Build a compact sanity table for the test split."""
    rows = []
    for model_name, metrics in model_reports.items():
        test_metrics = metrics["metrics"]["test"]
        gaps = metrics["metrics"]["ood_id_gap"]
        rows.append(
            {
                "regime": regime,
                "model": model_name,
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_row_kl": test_metrics["row_kl"],
                "test_rank_corr": test_metrics["rank_corr"],
                "test_topk_overlap": test_metrics["topk_overlap"],
                "row_kl_gap": gaps["row_kl_gap"],
                "rank_corr_gap": gaps["rank_corr_gap"],
            }
        )
    return pd.DataFrame(rows)
