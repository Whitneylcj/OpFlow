"""Canonical runner for the final DGP four-question synthetic suite."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from opflow_experiments.data.dgp import DGP_REGIME, apply_residual_behavioral_drift, generate_dgp_data
from opflow_experiments.evaluation.evaluator import evaluate_predictions
from opflow_experiments.evaluation.standard_outputs import metrics_frame_to_standard, write_standard_outputs
from opflow_experiments.exposure_models import ExposureOpFlowModel, build_exposure_model
from opflow_experiments.utils.config import deep_merge, load_resolved_config, save_resolved_config
from opflow_experiments.utils.io import ensure_dir
from opflow_experiments.utils.paths import create_run_dir


REGIME_LABELS = {DGP_REGIME: DGP_REGIME}
MAIN_QUESTION_CHOICES = ("q1_scale", "q2_opportunity", "q3_component_corr", "q4_ablation")
STRESS_RESIDUAL_DRIFT = "stress_residual_drift"
QUESTION_CHOICES = (*MAIN_QUESTION_CHOICES, STRESS_RESIDUAL_DRIFT)
DG_BASELINE_METHODS = ["GroupDRO-AllocMLP", "VREx-AllocMLP", "IRM-AllocMLP", "DANN-AllocMLP"]
CLASSICAL_BASELINE_METHODS = ["Gravity", "CompetingDestination", "InterveningOpportunity", "Radiation"]
BEST_DG_METHOD = "BestDG-AllocMLP"
BEST_CLASSICAL_METHOD = "BestClassical"
BEST_SELECTOR_POOLS = {
    BEST_DG_METHOD: DG_BASELINE_METHODS,
    BEST_CLASSICAL_METHOD: CLASSICAL_BASELINE_METHODS,
}
Q1_METHODS = ["RawCountMLP", "PairAllocMLP", "Gravity", "OpFlow"]
MAIN_METHODS = [
    "RawCountMLP",
    "PairAllocMLP",
    "ScaleAwareMLP",
    "ShareStable",
    *DG_BASELINE_METHODS,
    *CLASSICAL_BASELINE_METHODS,
    BEST_DG_METHOD,
    BEST_CLASSICAL_METHOD,
    "OpFlow",
]
Q1_PLOT_METHODS = list(Q1_METHODS)
MAIN_PLOT_METHODS = ["RawCountMLP", "PairAllocMLP", "ScaleAwareMLP", BEST_DG_METHOD, BEST_CLASSICAL_METHOD, "OpFlow"]
STRESS_METHODS = list(MAIN_METHODS)
Q4_METHODS = ["FullOpFlow", "OpFlow-ERM", "NoRankPrefix", "NoLocalField", "NoScaleIsolation", "NoEnvSelection", "NoMonotonicity"]
DEFAULT_METHODS = {DGP_REGIME: MAIN_METHODS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--regime", choices=sorted(REGIME_LABELS), default=None)
    parser.add_argument("--regimes", nargs="*", choices=sorted(REGIME_LABELS), default=None)
    parser.add_argument("--questions", nargs="*", choices=QUESTION_CHOICES, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=["debug", "full"], default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--tables-dir", default="tables")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--skip-paper-outputs", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _ensure_defaults(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out.setdefault("suite", DGP_REGIME)
    out.setdefault("N", 80)
    out.setdefault("N_o", int(out.get("N", 80)))
    out.setdefault("N_d", int(out.get("N", 80)))
    out.setdefault("tau", 0.7)
    out.setdefault("experiment", {}).setdefault("regimes", [DGP_REGIME])
    out["experiment"].setdefault("questions", list(MAIN_QUESTION_CHOICES))
    out["experiment"].setdefault("methods", list(MAIN_METHODS))
    out["experiment"].setdefault("q4_methods", list(Q4_METHODS))
    dgp = dict(out.get("dgp", {}))
    dgp.setdefault("N", int(out.get("N", 80)))
    dgp.setdefault("N_o", int(out.get("N_o", dgp["N"])))
    dgp.setdefault("N_d", int(out.get("N_d", dgp["N"])))
    dgp.setdefault("K_mean", 25)
    dgp.setdefault("K_std", 5)
    dgp.setdefault("K_min", 15)
    dgp.setdefault("K_max", 40)
    dgp.setdefault("num_train_env", 8)
    dgp.setdefault("num_val_env", 2)
    dgp.setdefault("num_test_env", 4)
    dgp.setdefault("tau", out.get("tau", 0.7))
    dgp.setdefault("omega", 0.15)
    dgp.setdefault("beta_o", 1.15)
    dgp.setdefault("beta_c", 0.35)
    dgp.setdefault("beta_s", 0.65)
    dgp.setdefault("beta_l", 0.45)
    dgp.setdefault("gamma_o", 0.25)
    dgp.setdefault("gamma_c", 0.25)
    dgp.setdefault("sigma_delta", 0.03)
    dgp.setdefault("sigma_proxy", 0.35)
    dgp.setdefault("q2_stable_proxy_snr", True)
    dgp.setdefault("proxy_target_corr", 0.95)
    out["dgp"] = dgp
    out.setdefault("synthetic_questions", {})
    out["synthetic_questions"].setdefault("q1_scale", {"scale_severity_list": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]})
    out["synthetic_questions"]["q1_scale"].setdefault("methods", list(Q1_METHODS))
    out["synthetic_questions"].setdefault("q2_opportunity", {"opp_severity_list": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]})
    out["synthetic_questions"]["q2_opportunity"].setdefault("reuse_train_fit", True)
    out["synthetic_questions"].setdefault(
        "q3_component_corr",
        {"alpha_train": 0.7, "alpha_val": 0.5, "alpha_test_list": [0.7, 0.3, 0.0, -0.3, -0.6]},
    )
    out["synthetic_questions"]["q3_component_corr"].setdefault("reuse_train_fit", True)
    out["synthetic_questions"].setdefault("q4_ablation", {"stress_opp_severity": 0.9, "stress_alpha_test": -0.6})
    out["synthetic_questions"].setdefault(STRESS_RESIDUAL_DRIFT, {"kappa_list": [0.0, 0.1, 0.2, 0.3, 0.5], "methods": list(STRESS_METHODS)})
    model = dict(out.get("exposure_model", {}))
    model.setdefault("device", "cpu")
    model.setdefault("hidden_dim", 64)
    model.setdefault("epochs", 180)
    model.setdefault("learning_rate", 0.002)
    model.setdefault("patience", 25)
    model.setdefault("alpha_smooth", 0.5)
    model.setdefault("lambda_count", 0.05)
    model.setdefault("lambda_scale", 0.1)
    model.setdefault("raw_count_loss", "mse")
    model.setdefault("lambda_vrex", 0.1)
    model.setdefault("lambda_irm", 0.1)
    model.setdefault("lambda_dann", 0.1)
    model.setdefault("baseline_groupdro_eta", 0.05)
    model.setdefault("dann_warmup_epochs", 0)
    model.setdefault("dann_grl_max", 1.0)
    model.setdefault("lambda_op", 0.01)
    model.setdefault("groupdro_weight", 0.005)
    model.setdefault("lambda_operator_anchor", 0.0)
    model.setdefault("lambda_o_anchor", 1.0)
    model.setdefault("lambda_c_anchor", 1.0)
    model.setdefault("lambda_anchor", 0.005)
    model.setdefault("lambda_smooth", 0.0005)
    model.setdefault("checkpoint_metric", "joint_selection_score")
    model.setdefault("selection_protocol", "leo")
    model.setdefault("leo_selection_metric", "joint_selection_score")
    model.setdefault("leo_max_epochs", 80)
    model.setdefault("leo_patience", 10)
    model.setdefault("leo_min_fold_fraction", 0.75)
    model.setdefault("lambda_selection_scale", 0.1)
    model.setdefault("scale_residual_weight", 0.1)
    model.setdefault("eta_residual", 0.2)
    model.setdefault("residual_hidden_dim", max(int(model.get("hidden_dim", 64)) // 4, 4))
    model.setdefault("origin_gate_rho", 0.5)
    model.setdefault("opp_recon_delta_scale", 0.25)
    model.setdefault("rank_prefix_mode", "soft")
    model.setdefault("rank_prefix_temperature", 0.05)
    model.setdefault("benefit_skip", float(dgp.get("beta_o", 1.15)))
    model.setdefault("deterrence_skip", -float(dgp.get("beta_c", 0.35)))
    model.setdefault("exposure_skip", -float(dgp.get("beta_s", 0.65)))
    model.setdefault("competition_skip", float(dgp.get("beta_l", 0.45)))
    model.setdefault("use_crowding_shift_gate", True)
    model.setdefault("use_groupdro", True)
    model.setdefault("use_rank_exposure", True)
    model.setdefault("use_competition", True)
    model.setdefault("use_opportunity_field", True)
    model.setdefault("use_learned_opportunity_reconstruction", True)
    model.setdefault("use_origin_gates", True)
    model.setdefault("use_row_center", True)
    model.setdefault("detach_count_allocation", True)
    model.setdefault("use_sit_operator_potential", True)
    model.setdefault("use_residual_interaction", True)
    model.setdefault("use_monotone_penalty", True)
    model.setdefault("use_vrex", True)
    out["exposure_model"] = model
    out.setdefault("baseline", {}).setdefault("topk", 5)
    return out


def _apply_debug(config: dict[str, Any]) -> dict[str, Any]:
    out = _ensure_defaults(config)
    dgp = out["dgp"]
    n = min(int(dgp.get("N", dgp.get("N_o", 80))), 18)
    out["N"] = out["N_o"] = out["N_d"] = n
    dgp["N"] = dgp["N_o"] = dgp["N_d"] = n
    dgp["K_mean"] = min(float(dgp.get("K_mean", 25)), 7.0)
    dgp["K_std"] = min(float(dgp.get("K_std", 5)), 1.5)
    dgp["K_min"] = min(int(dgp.get("K_min", 15)), 4, n - 1)
    dgp["K_max"] = min(int(dgp.get("K_max", 40)), 10, n - 1)
    dgp["num_train_env"] = 2
    dgp["num_val_env"] = 1
    dgp["num_test_env"] = 1
    model = out["exposure_model"]
    model["hidden_dim"] = min(int(model.get("hidden_dim", 64)), 12)
    model["epochs"] = min(int(model.get("epochs", 180)), 4)
    model["patience"] = min(int(model.get("patience", 25)), 2)
    return out


def load_exposure_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    wrapper = load_resolved_config(config_path, overrides)
    base_path = wrapper.get("base_config")
    if base_path:
        base = load_exposure_config(base_path)
        wrapper = deep_merge(base, {key: value for key, value in wrapper.items() if key != "base_config"})
    return _ensure_defaults(wrapper)


def _sort_records(records: pd.DataFrame) -> pd.DataFrame:
    return records.sort_values(["env_index", "origin_id", "candidate_slot", "dest_id"], kind="mergesort").reset_index(drop=True)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 2 or np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _row_center(records: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.reshape(-1), index=records.index, dtype=float)
    return (series - series.groupby([records["env_id"], records["origin_id"]]).transform("mean")).to_numpy(dtype=float)


def _metric_rows(
    *,
    question: str,
    scenario_name: str,
    scenario_value_name: str,
    scenario_value: float | str,
    stress_type: str,
    regime: str,
    method: str,
    seed: int,
    metrics: dict,
    extra: dict[str, float],
) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        split_metrics = metrics[split]
        rows.append(
            {
                "question": question,
                "scenario": scenario_name,
                "scenario_value_name": scenario_value_name,
                "scenario_value": scenario_value,
                "stress_type": stress_type,
                "regime": regime,
                "method": method,
                "seed": seed,
                "split": split,
                "rmse": split_metrics["rmse"],
                "mae": split_metrics["mae"],
                "log_rmse": split_metrics.get("log_rmse"),
                "global_cpc": split_metrics.get("global_cpc"),
                "row_kl": split_metrics["row_kl"],
                "row_cpc": split_metrics.get("row_cpc"),
                "rank_corr": split_metrics["rank_corr"],
                "topk_overlap": split_metrics["topk_overlap"],
                "score_mse": split_metrics.get("score_mse"),
                "score_pmse": split_metrics.get("score_pmse"),
                "pmse": split_metrics.get("pmse"),
                "potential_corr": split_metrics.get("potential_corr"),
                "score_corr": split_metrics.get("score_corr"),
                "scale_rmse": split_metrics.get("scale_rmse"),
                "scale_mae": split_metrics.get("scale_mae"),
                "scale_corr": split_metrics.get("scale_corr"),
                "scale_hat_mean": split_metrics.get("scale_hat_mean"),
                "scale_true_mean": split_metrics.get("scale_true_mean"),
                "exposure_alignment": extra.get("exposure_alignment", np.nan) if split == "test" else np.nan,
                "o_hat_within_dest_std_max": extra.get("o_hat_within_dest_std_max", np.nan) if split == "test" else np.nan,
                "o_hat_within_dest_std_mean": extra.get("o_hat_within_dest_std_mean", np.nan) if split == "test" else np.nan,
                "ell_hat_vs_true_corr": extra.get("ell_hat_vs_true_corr", np.nan) if split == "test" else np.nan,
                "alloc_risk_max": extra.get("alloc_risk_max", np.nan) if split == "test" else np.nan,
                "alloc_risk_mean": extra.get("alloc_risk_mean", np.nan) if split == "test" else np.nan,
                "vrex_alloc_var": extra.get("vrex_alloc_var", np.nan) if split == "test" else np.nan,
                "scale_risk_max": extra.get("scale_risk_max", np.nan) if split == "test" else np.nan,
                "scale_risk_mean": extra.get("scale_risk_mean", np.nan) if split == "test" else np.nan,
                "eta_residual": extra.get("eta_residual", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_D": extra.get("op_monotone_violation_D", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_S": extra.get("op_monotone_violation_S", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_B": extra.get("op_monotone_violation_B", np.nan) if split == "test" else np.nan,
                "operator_mean_o_tilde": extra.get("operator_mean_o_tilde", np.nan) if split == "test" else np.nan,
                "operator_mean_c_tilde": extra.get("operator_mean_c_tilde", np.nan) if split == "test" else np.nan,
                "psi_bar_row_mean_abs": extra.get("psi_bar_row_mean_abs", np.nan) if split == "test" else np.nan,
                "raw_proxy_log_opp_corr": extra.get("raw_proxy_log_opp_corr", np.nan) if split == "test" else np.nan,
                "learned_opp_log_corr": extra.get("learned_opp_log_corr", np.nan) if split == "test" else np.nan,
                "opportunity_anchor_loss": extra.get("opportunity_anchor_loss", np.nan) if split == "test" else np.nan,
                "opportunity_smooth_loss": extra.get("opportunity_smooth_loss", np.nan) if split == "test" else np.nan,
                "residual_contribution_ratio": extra.get("residual_contribution_ratio", np.nan) if split == "test" else np.nan,
                "gate_a_mean": extra.get("gate_a_mean", np.nan) if split == "test" else np.nan,
                "gate_a_std": extra.get("gate_a_std", np.nan) if split == "test" else np.nan,
                "gate_a_min": extra.get("gate_a_min", np.nan) if split == "test" else np.nan,
                "gate_a_max": extra.get("gate_a_max", np.nan) if split == "test" else np.nan,
                "gate_b_mean": extra.get("gate_b_mean", np.nan) if split == "test" else np.nan,
                "gate_b_std": extra.get("gate_b_std", np.nan) if split == "test" else np.nan,
                "gate_b_min": extra.get("gate_b_min", np.nan) if split == "test" else np.nan,
                "gate_b_max": extra.get("gate_b_max", np.nan) if split == "test" else np.nan,
                "gate_r_mean": extra.get("gate_r_mean", np.nan) if split == "test" else np.nan,
                "gate_r_std": extra.get("gate_r_std", np.nan) if split == "test" else np.nan,
                "gate_r_min": extra.get("gate_r_min", np.nan) if split == "test" else np.nan,
                "gate_r_max": extra.get("gate_r_max", np.nan) if split == "test" else np.nan,
                "gate_l_mean": extra.get("gate_l_mean", np.nan) if split == "test" else np.nan,
                "gate_l_std": extra.get("gate_l_std", np.nan) if split == "test" else np.nan,
                "gate_l_min": extra.get("gate_l_min", np.nan) if split == "test" else np.nan,
                "gate_l_max": extra.get("gate_l_max", np.nan) if split == "test" else np.nan,
                "shortcut_sensitivity": split_metrics.get("shortcut_sensitivity"),
                "distance_sensitivity": split_metrics.get("distance_sensitivity"),
                "origin_scale_error": split_metrics.get("origin_scale_error"),
                "row_share_sum_error": split_metrics.get("row_share_sum_error"),
                "row_kl_ood_gap": metrics["ood_id_gap"]["row_kl_ood_gap"],
                "topk_ood_drop": metrics["ood_id_gap"]["topk_ood_drop"],
                "potential_rank_corr": split_metrics.get("score_corr"),
                "rankcorr_ood_drop": metrics["ood_id_gap"]["rankcorr_ood_drop"],
                "notes": question,
            }
        )
    return rows


def _within_dest_std(records: pd.DataFrame, values: np.ndarray) -> tuple[float, float]:
    frame = records[["env_id", "dest_id"]].copy()
    frame["_value"] = np.asarray(values, dtype=float).reshape(-1)
    std = frame.groupby(["env_id", "dest_id"], sort=False)["_value"].std(ddof=0).dropna()
    if std.empty:
        return float("nan"), float("nan")
    return float(std.max()), float(std.mean())


def _extra_metrics(model, records: pd.DataFrame, prediction: dict) -> dict[str, float]:
    extra = {
        "exposure_alignment": float("nan"),
        "o_hat_within_dest_std_max": float("nan"),
        "o_hat_within_dest_std_mean": float("nan"),
        "ell_hat_vs_true_corr": float("nan"),
        "operator_mean_o_tilde": float("nan"),
        "operator_mean_c_tilde": float("nan"),
        "psi_bar_row_mean_abs": float("nan"),
        "eta_residual": float("nan"),
        "raw_proxy_log_opp_corr": float("nan"),
        "learned_opp_log_corr": float("nan"),
        "opportunity_anchor_loss": float("nan"),
        "opportunity_smooth_loss": float("nan"),
        "residual_contribution_ratio": float("nan"),
    }
    for gate in ("a", "b", "r", "l"):
        for stat in ("mean", "std", "min", "max"):
            extra[f"gate_{gate}_{stat}"] = float("nan")
    pred_extra = prediction.get("extra", {})
    score = pred_extra.get("score_hat")
    test_mask = records["split"].to_numpy() == "test"
    if test_mask.any() and {"opp_proxy", "o_true"}.issubset(records.columns):
        extra["raw_proxy_log_opp_corr"] = _safe_corr(
            np.log1p(records.loc[test_mask, "opp_proxy"].to_numpy(dtype=float)),
            np.log1p(records.loc[test_mask, "o_true"].to_numpy(dtype=float)),
        )
    if score is not None and str(pred_extra.get("score_kind", "choice_potential")) == "choice_potential":
        extra["exposure_alignment"] = _safe_corr(
            _row_center(records.loc[test_mask], _as_numpy(score)[test_mask]),
            records.loc[test_mask, "psi_oracle"].to_numpy(dtype=float),
        )
        row_mean = pd.Series(_as_numpy(score)[test_mask].reshape(-1), index=records.loc[test_mask].index)
        centered = row_mean.groupby([records.loc[test_mask, "env_id"], records.loc[test_mask, "origin_id"]]).transform("mean")
        extra["psi_bar_row_mean_abs"] = float(np.nanmean(np.abs(centered.to_numpy(dtype=float))))
    o_hat = pred_extra.get("o_hat")
    if o_hat is not None:
        max_std, mean_std = _within_dest_std(records.loc[test_mask], _as_numpy(o_hat)[test_mask])
        extra["o_hat_within_dest_std_max"] = max_std
        extra["o_hat_within_dest_std_mean"] = mean_std
        extra["operator_mean_o_tilde"] = float(np.nanmean(_as_numpy(o_hat)[test_mask]))
    c_hat = pred_extra.get("c_hat")
    if c_hat is not None:
        extra["operator_mean_c_tilde"] = float(np.nanmean(_as_numpy(c_hat)[test_mask]))
    ell_hat = pred_extra.get("ell_hat")
    if ell_hat is not None:
        extra["ell_hat_vs_true_corr"] = _safe_corr(_as_numpy(ell_hat)[test_mask], records.loc[test_mask, "ell_true"].to_numpy(dtype=float))
    o_tilde = pred_extra.get("o_tilde")
    if o_tilde is not None and test_mask.any() and "o_true" in records:
        extra["learned_opp_log_corr"] = _safe_corr(
            np.log1p(np.maximum(_as_numpy(o_tilde)[test_mask], 0.0)),
            np.log1p(records.loc[test_mask, "o_true"].to_numpy(dtype=float)),
        )
    for gate in ("a", "b", "r", "l"):
        values = pred_extra.get(f"gate_{gate}")
        if values is None:
            continue
        arr = _as_numpy(values)[test_mask]
        if arr.size:
            extra[f"gate_{gate}_mean"] = float(np.nanmean(arr))
            extra[f"gate_{gate}_std"] = float(np.nanstd(arr))
            extra[f"gate_{gate}_min"] = float(np.nanmin(arr))
            extra[f"gate_{gate}_max"] = float(np.nanmax(arr))
    for key in (
        "alloc_risk_max",
        "alloc_risk_mean",
        "vrex_alloc_var",
        "scale_risk_max",
        "scale_risk_mean",
        "eta_residual",
        "opportunity_anchor_loss",
        "opportunity_smooth_loss",
        "residual_contribution_ratio",
    ):
        if key in pred_extra:
            extra[key] = float(pred_extra[key])
    return extra


def _env_risk_rows(records: pd.DataFrame, prediction: dict, *, question: str, scenario: str, regime: str, method: str, seed: int) -> list[dict[str, Any]]:
    working = records[["env_id", "split", "origin_id", "pi_oracle"]].copy()
    working["pi_hat"] = _as_numpy(prediction["pi_hat"]).reshape(-1)
    rows = []
    for (env_id, split, origin_id), group in working.groupby(["env_id", "split", "origin_id"], sort=False):
        p_true = np.clip(group["pi_oracle"].to_numpy(dtype=float), 1e-8, None)
        p_hat = np.clip(group["pi_hat"].to_numpy(dtype=float), 1e-8, None)
        rows.append({"env_id": env_id, "split": split, "origin_id": origin_id, "row_kl": float(np.sum(p_true * (np.log(p_true) - np.log(p_hat))))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out = []
    for (env_id, split), group in frame.groupby(["env_id", "split"], sort=False):
        out.append(
            {
                "question": question,
                "scenario": scenario,
                "regime": regime,
                "method": method,
                "seed": seed,
                "env_id": env_id,
                "split": split,
                "n_rows": int(len(group)),
                "alloc_risk": float(group["row_kl"].mean()),
            }
        )
    return out


def _operator_diagnostic_row(question: str, scenario: str, regime: str, method: str, seed: int, extra: dict[str, float]) -> dict[str, Any]:
    keys = [
        "alloc_risk_max",
        "alloc_risk_mean",
        "vrex_alloc_var",
        "scale_risk_max",
        "scale_risk_mean",
        "eta_residual",
        "op_monotone_violation_D",
        "op_monotone_violation_S",
        "op_monotone_violation_B",
        "operator_mean_o_tilde",
        "operator_mean_c_tilde",
        "psi_bar_row_mean_abs",
        "o_hat_within_dest_std_max",
        "o_hat_within_dest_std_mean",
        "ell_hat_vs_true_corr",
        "raw_proxy_log_opp_corr",
        "learned_opp_log_corr",
        "opportunity_anchor_loss",
        "opportunity_smooth_loss",
        "residual_contribution_ratio",
    ]
    for gate in ("a", "b", "r", "l"):
        for stat in ("mean", "std", "min", "max"):
            keys.append(f"gate_{gate}_{stat}")
    return {"question": question, "scenario": scenario, "regime": regime, "method": method, "seed": seed, **{key: extra.get(key, np.nan) for key in keys}}


def _check_dgp_acceptance(data) -> list[str]:
    warnings: list[str] = []
    meta = data.env_meta
    if not bool(data.metadata.get("shared_support")):
        raise AssertionError("Final DGP must use one shared origin/destination support.")
    if np.any(data.candidate_indices[data.candidate_mask] < 0):
        raise AssertionError("Candidate mask marks invalid padded destinations.")
    if np.any(data.candidate_indices[data.candidate_mask] == np.repeat(np.arange(data.candidate_indices.shape[0]), data.candidate_mask.sum(axis=1))):
        raise AssertionError("Self-loops must be excluded from candidate sets.")
    if float(data.metadata["prefix_self_exclusion_max_abs_error"]) > 1e-8:
        raise AssertionError("Rank exposure prefix failed self-exclusion check.")
    if float(meta["pi_oracle_row_sum_error"].max()) > 1e-6:
        raise AssertionError("Oracle allocation rows do not sum to one.")
    if float(meta["psi_oracle_row_mean_abs"].max()) > 1e-6:
        raise AssertionError("Oracle potential is not row-centered.")
    if float(meta["lambda_identity_max_abs_error"].max()) > 1e-8:
        raise AssertionError("lambda_oracle != R_oracle * pi_oracle.")
    if not np.allclose(meta["opportunity_mean"].to_numpy(dtype=float), 1.0, atol=1e-6):
        raise AssertionError("Environment opportunity fields are not mean-normalized.")
    if float(data.metadata.get("opportunity_env_corr_max", 1.0)) > 0.995:
        warnings.append("Some environment opportunity fields are nearly identical.")
    if str(data.metadata.get("standardizer_fit_split")) != "train":
        raise AssertionError("Oracle standardizer must be fitted on training environments only.")
    return warnings


def _check_opflow_acceptance(model, train: pd.DataFrame, method: str) -> dict[str, float | bool]:
    if not isinstance(model, ExposureOpFlowModel):
        return {}
    audit = model.audit_gradients(train)
    model_method = getattr(model, "method", method)
    tol = 1e-7
    if model_method != "NoScaleDetach" and audit.count_scale_potential_grad_max > tol:
        raise AssertionError(f"{method} scale/count loss updates potential branch: {audit.count_scale_potential_grad_max:.3e}")
    if model_method != "NoRowCenter" and audit.row_center_mean_abs > 1e-6:
        raise AssertionError(f"{method} row-centered potential has nonzero row mean: {audit.row_center_mean_abs:.3e}")
    return {
        "count_scale_potential_grad_max": audit.count_scale_potential_grad_max,
        "row_center_mean_abs": audit.row_center_mean_abs,
        "op_monotone_violation_D": audit.op_monotone_violation_D,
        "op_monotone_violation_S": audit.op_monotone_violation_S,
        "op_monotone_violation_B": audit.op_monotone_violation_B,
    }


def _scenario_config(config: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out["scenario"] = dict(scenario)
    return out


def _unique_preserve_order(methods: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for method in methods:
        if method not in seen:
            out.append(method)
            seen.add(method)
    return out


def _fit_methods_for_selectors(methods: list[str]) -> tuple[list[str], set[str], list[str]]:
    reported = {method for method in methods if method not in BEST_SELECTOR_POOLS}
    aliases = [method for method in methods if method in BEST_SELECTOR_POOLS]
    fit_methods = [method for method in methods if method not in BEST_SELECTOR_POOLS]
    for alias in aliases:
        fit_methods.extend(BEST_SELECTOR_POOLS[alias])
    return _unique_preserve_order(fit_methods), reported, _unique_preserve_order(aliases)


def _select_best_by_val_row_kl(method_cache: dict[str, dict], candidates: list[str]) -> tuple[str | None, dict[str, float]]:
    scores: dict[str, float] = {}
    for method in candidates:
        metrics = method_cache.get(method, {}).get("metrics")
        if not metrics or "val" not in metrics:
            continue
        score = float(metrics["val"].get("row_kl", float("nan")))
        scores[method] = score
    finite = {method: score for method, score in scores.items() if np.isfinite(score)}
    if not finite:
        return None, scores
    return min(finite, key=finite.get), scores


def _run_one(
    config: dict,
    regime: str,
    seed: int,
    methods: list[str],
    run_dir: Path,
    *,
    question: str,
    scenario_name: str,
    scenario_value_name: str,
    scenario_value: float | str,
    stress_type: str = "",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regime != DGP_REGIME:
        raise ValueError(f"Only DGP regime is supported, got {regime!r}.")
    data = generate_dgp_data(config, seed=seed)
    warnings = _check_dgp_acceptance(data)
    records = _sort_records(data.records)
    train = records.loc[records["split"] == "train"].copy()
    val = records.loc[records["split"] == "val"].copy()
    artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / question / scenario_name / f"seed_{seed}")
    data.env_meta.to_csv(artifacts / "env_meta.csv", index=False)
    with (artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(data.metadata, handle, indent=2, sort_keys=True)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    env_risk_rows: list[dict] = []
    operator_rows: list[dict] = []
    selection_rows: list[dict] = []
    fit_methods, reported_methods, selector_aliases = _fit_methods_for_selectors(methods)
    method_cache: dict[str, dict] = {}
    method_iter = tqdm(fit_methods, desc=f"{question} {scenario_name} seed {seed}", leave=False) if show_progress else fit_methods
    for method in method_iter:
        model = build_exposure_model(method, config, seed)
        model.fit(train, val)
        selection = getattr(model, "selection_summary_", {})
        if selection:
            selection_rows.append(
                {
                    "question": question,
                    "scenario": scenario_name,
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    **{
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in selection.items()
                    },
                }
            )
        pred = model.predict(records)
        metrics = evaluate_predictions(records, pred, topk=int(config.get("baseline", {}).get("topk", 5)))
        extra = _extra_metrics(model, records, pred)
        audit = _check_opflow_acceptance(model, train, method)
        if audit:
            audit_rows.append({"question": question, "scenario": scenario_name, "regime": regime, "seed": seed, "method": method, **audit})
            for key in ("op_monotone_violation_D", "op_monotone_violation_S", "op_monotone_violation_B"):
                extra[key] = float(audit[key])
        method_cache[method] = {"model": model, "prediction": pred, "metrics": metrics, "extra": extra}
        if method in reported_methods:
            env_risk_rows.extend(_env_risk_rows(records, pred, question=question, scenario=scenario_name, regime=regime, method=method, seed=seed))
            operator_rows.append(_operator_diagnostic_row(question, scenario_name, regime, method, seed, extra))
            rows.extend(
                _metric_rows(
                    question=question,
                    scenario_name=scenario_name,
                    scenario_value_name=scenario_value_name,
                    scenario_value=scenario_value,
                    stress_type=stress_type,
                    regime=regime,
                    method=method,
                    seed=seed,
                    metrics=metrics,
                    extra=extra,
                )
            )
    for alias in selector_aliases:
        selected, scores = _select_best_by_val_row_kl(method_cache, BEST_SELECTOR_POOLS[alias])
        selection_rows.append(
            {
                "question": question,
                "scenario": scenario_name,
                "regime": regime,
                "seed": seed,
                "method": alias,
                "selection_protocol": "validation_alias",
                "selection_metric": "row_kl",
                "selected_method": selected or "",
                "candidate_val_row_kl": json.dumps(scores, sort_keys=True),
            }
        )
        if selected is None:
            continue
        cached = method_cache[selected]
        pred = cached["prediction"]
        metrics = cached["metrics"]
        extra = cached["extra"]
        env_risk_rows.extend(_env_risk_rows(records, pred, question=question, scenario=scenario_name, regime=regime, method=alias, seed=seed))
        operator_rows.append(_operator_diagnostic_row(question, scenario_name, regime, alias, seed, extra))
        rows.extend(
            _metric_rows(
                question=question,
                scenario_name=scenario_name,
                scenario_value_name=scenario_value_name,
                scenario_value=scenario_value,
                stress_type=stress_type,
                regime=regime,
                method=alias,
                seed=seed,
                metrics=metrics,
                extra=extra,
            )
        )
    diag = data.env_meta.copy()
    diag["regime"] = regime
    diag["seed"] = seed
    diag["scenario"] = scenario_name
    diag["scenario_value_name"] = scenario_value_name
    diag["scenario_value"] = scenario_value
    diag["stress_type"] = stress_type
    diag["diagnostic_note"] = "; ".join(warnings)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(artifacts / "gradient_audit.csv", index=False)
    if env_risk_rows:
        pd.DataFrame(env_risk_rows).to_csv(artifacts / "env_risks.csv", index=False)
    if operator_rows:
        pd.DataFrame(operator_rows).to_csv(artifacts / "operator_diagnostics.csv", index=False)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(artifacts / "selection_diagnostics.csv", index=False)
    return pd.DataFrame(rows), diag


def _assert_q2_train_val_fixed(base_records: pd.DataFrame, records: pd.DataFrame, scenario_name: str) -> None:
    base = base_records.loc[base_records["split"].isin(["train", "val"])].reset_index(drop=True)
    other = records.loc[records["split"].isin(["train", "val"])].reset_index(drop=True)
    if len(base) != len(other):
        raise AssertionError(f"Q2 train/val row count changed under {scenario_name}.")
    equality_cols = ["env_id", "env_index", "split", "origin_id", "dest_id", "candidate_slot", "k_i"]
    numeric_cols = [
        "origin_x",
        "origin_y",
        "dest_x",
        "dest_y",
        "cost",
        "distance",
        "dx",
        "dy",
        "cost_rank_frac",
        "row_standardized_cost",
        "opp_proxy",
        "o_true",
        "s_true",
        "ell_true",
        "g_oracle",
        "psi_oracle",
        "pi_oracle",
        "R_oracle",
        "lambda_oracle",
        "y_obs",
    ]
    for col in equality_cols:
        if not base[col].equals(other[col]):
            raise AssertionError(f"Q2 train/val column {col!r} changed under {scenario_name}; cannot reuse fit.")
    for col in numeric_cols:
        if col in base and col in other and not np.allclose(base[col].to_numpy(dtype=float), other[col].to_numpy(dtype=float), atol=1e-10):
            raise AssertionError(f"Q2 train/val column {col!r} changed under {scenario_name}; cannot reuse fit.")


def _assert_q3_train_val_fixed(base_records: pd.DataFrame, records: pd.DataFrame, scenario_name: str) -> None:
    base = base_records.loc[base_records["split"].isin(["train", "val"])].reset_index(drop=True)
    other = records.loc[records["split"].isin(["train", "val"])].reset_index(drop=True)
    if len(base) != len(other):
        raise AssertionError(f"Q3 train/val row count changed under {scenario_name}.")
    equality_cols = ["env_id", "env_index", "split", "origin_id", "dest_id", "candidate_slot", "k_i"]
    numeric_cols = [
        "origin_x",
        "origin_y",
        "dest_x",
        "dest_y",
        "cost",
        "distance",
        "dx",
        "dy",
        "cost_rank_frac",
        "row_standardized_cost",
        "opp_proxy",
        "o_true",
        "s_true",
        "ell_true",
        "g_oracle",
        "psi_oracle",
        "pi_oracle",
        "R_oracle",
        "lambda_oracle",
        "y_obs",
    ]
    for col in equality_cols:
        if not base[col].equals(other[col]):
            raise AssertionError(f"Q3 train/val column {col!r} changed under {scenario_name}; cannot reuse fit.")
    for col in numeric_cols:
        if col in base and col in other and not np.allclose(base[col].to_numpy(dtype=float), other[col].to_numpy(dtype=float), atol=1e-10):
            raise AssertionError(f"Q3 train/val column {col!r} changed under {scenario_name}; cannot reuse fit.")


def _with_scenario(rows: list[dict[str, Any]], scenario: str, *, reused_fit_from: str = "") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["scenario"] = scenario
        if reused_fit_from:
            copied["reused_fit_from"] = reused_fit_from
        out.append(copied)
    return out


def _run_q2_opportunity_sweep(
    config: dict,
    regime: str,
    seed: int,
    tasks: list[dict[str, Any]],
    run_dir: Path,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()
    if regime != DGP_REGIME:
        raise ValueError(f"Only DGP regime is supported, got {regime!r}.")
    methods = list(tasks[0]["methods"])
    if any(list(task["methods"]) != methods for task in tasks):
        raise ValueError("Q2 fit reuse requires all Q2 severity tasks to use the same method list.")

    base_scenario = {"question": "q2_opportunity", "opp_severity": 0.0}
    base_config = _scenario_config(config, base_scenario)
    base_data = generate_dgp_data(base_config, seed=seed)
    warnings = _check_dgp_acceptance(base_data)
    base_records = _sort_records(base_data.records)
    train = base_records.loc[base_records["split"] == "train"].copy()
    val = base_records.loc[base_records["split"] == "val"].copy()
    shared_name = "shared_fit"
    shared_artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / "q2_opportunity" / shared_name / f"seed_{seed}")
    base_data.env_meta.to_csv(shared_artifacts / "base_env_meta.csv", index=False)
    with (shared_artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(base_data.metadata, handle, indent=2, sort_keys=True)

    rows: list[dict] = []
    diag_frames: list[pd.DataFrame] = []
    shared_audit_rows: list[dict] = []
    shared_selection_rows: list[dict] = []
    fit_methods, reported_methods, selector_aliases = _fit_methods_for_selectors(methods)
    method_cache: dict[str, dict] = {}
    method_iter = tqdm(fit_methods, desc=f"q2_opportunity shared_fit seed {seed}", leave=False) if show_progress else fit_methods
    for method in method_iter:
        model = build_exposure_model(method, base_config, seed)
        model.fit(train, val)
        selection = getattr(model, "selection_summary_", {})
        if selection:
            shared_selection_rows.append(
                {
                    "question": "q2_opportunity",
                    "scenario": shared_name,
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    **{
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in selection.items()
                    },
                }
            )
        audit = _check_opflow_acceptance(model, train, method)
        if audit:
            shared_audit_rows.append({"question": "q2_opportunity", "scenario": shared_name, "regime": regime, "seed": seed, "method": method, **audit})
        base_pred = model.predict(base_records)
        base_metrics = evaluate_predictions(base_records, base_pred, topk=int(base_config.get("baseline", {}).get("topk", 5)))
        method_cache[method] = {"model": model, "base_metrics": base_metrics, "audit": audit}

    if shared_audit_rows:
        pd.DataFrame(shared_audit_rows).to_csv(shared_artifacts / "gradient_audit.csv", index=False)
    if shared_selection_rows:
        pd.DataFrame(shared_selection_rows).to_csv(shared_artifacts / "selection_diagnostics.csv", index=False)

    for task in tasks:
        task_config = _scenario_config(config, task["scenario"])
        data = generate_dgp_data(task_config, seed=seed)
        task_warnings = _check_dgp_acceptance(data)
        records = _sort_records(data.records)
        _assert_q2_train_val_fixed(base_records, records, task["name"])
        artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / "q2_opportunity" / task["name"] / f"seed_{seed}")
        data.env_meta.to_csv(artifacts / "env_meta.csv", index=False)
        with (artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(data.metadata, handle, indent=2, sort_keys=True)

        env_risk_rows: list[dict] = []
        operator_rows: list[dict] = []
        scenario_cache: dict[str, dict] = {}
        for method in fit_methods:
            cached = method_cache[method]
            model = cached["model"]
            pred = model.predict(records)
            metrics = evaluate_predictions(records, pred, topk=int(task_config.get("baseline", {}).get("topk", 5)))
            extra = _extra_metrics(model, records, pred)
            audit = cached.get("audit", {})
            for key in ("op_monotone_violation_D", "op_monotone_violation_S", "op_monotone_violation_B"):
                if key in audit:
                    extra[key] = float(audit[key])
            scenario_cache[method] = {"prediction": pred, "metrics": metrics, "extra": extra}
            if method in reported_methods:
                env_risk_rows.extend(_env_risk_rows(records, pred, question="q2_opportunity", scenario=task["name"], regime=regime, method=method, seed=seed))
                operator_rows.append(_operator_diagnostic_row("q2_opportunity", task["name"], regime, method, seed, extra))
                rows.extend(
                    _metric_rows(
                        question="q2_opportunity",
                        scenario_name=task["name"],
                        scenario_value_name=task["value_name"],
                        scenario_value=task["value"],
                        stress_type=task["stress_type"],
                        regime=regime,
                        method=method,
                        seed=seed,
                        metrics=metrics,
                        extra=extra,
                    )
                )

        selection_rows = _with_scenario(shared_selection_rows, task["name"], reused_fit_from=shared_name)
        for alias in selector_aliases:
            selected, scores = _select_best_by_val_row_kl(
                {method: {"metrics": cached["base_metrics"]} for method, cached in method_cache.items()},
                BEST_SELECTOR_POOLS[alias],
            )
            selection_rows.append(
                {
                    "question": "q2_opportunity",
                    "scenario": task["name"],
                    "regime": regime,
                    "seed": seed,
                    "method": alias,
                    "selection_protocol": "validation_alias",
                    "selection_metric": "row_kl",
                    "selected_method": selected or "",
                    "candidate_val_row_kl": json.dumps(scores, sort_keys=True),
                    "reused_fit_from": shared_name,
                }
            )
            if selected is None:
                continue
            cached = scenario_cache[selected]
            pred = cached["prediction"]
            metrics = cached["metrics"]
            extra = cached["extra"]
            env_risk_rows.extend(_env_risk_rows(records, pred, question="q2_opportunity", scenario=task["name"], regime=regime, method=alias, seed=seed))
            operator_rows.append(_operator_diagnostic_row("q2_opportunity", task["name"], regime, alias, seed, extra))
            rows.extend(
                _metric_rows(
                    question="q2_opportunity",
                    scenario_name=task["name"],
                    scenario_value_name=task["value_name"],
                    scenario_value=task["value"],
                    stress_type=task["stress_type"],
                    regime=regime,
                    method=alias,
                    seed=seed,
                    metrics=metrics,
                    extra=extra,
                )
            )

        diag = data.env_meta.copy()
        diag["regime"] = regime
        diag["seed"] = seed
        diag["scenario"] = task["name"]
        diag["scenario_value_name"] = task["value_name"]
        diag["scenario_value"] = task["value"]
        diag["stress_type"] = task["stress_type"]
        diag["fit_reuse_group"] = shared_name
        diag["diagnostic_note"] = "; ".join([*warnings, *task_warnings])
        diag_frames.append(diag)
        if shared_audit_rows:
            pd.DataFrame(_with_scenario(shared_audit_rows, task["name"], reused_fit_from=shared_name)).to_csv(artifacts / "gradient_audit.csv", index=False)
        if env_risk_rows:
            pd.DataFrame(env_risk_rows).to_csv(artifacts / "env_risks.csv", index=False)
        if operator_rows:
            pd.DataFrame(operator_rows).to_csv(artifacts / "operator_diagnostics.csv", index=False)
        if selection_rows:
            pd.DataFrame(selection_rows).to_csv(artifacts / "selection_diagnostics.csv", index=False)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    return pd.DataFrame(rows), diagnostics


def _run_q3_component_corr_sweep(
    config: dict,
    regime: str,
    seed: int,
    tasks: list[dict[str, Any]],
    run_dir: Path,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()
    if regime != DGP_REGIME:
        raise ValueError(f"Only DGP regime is supported, got {regime!r}.")
    methods = list(tasks[0]["methods"])
    if any(list(task["methods"]) != methods for task in tasks):
        raise ValueError("Q3 fit reuse requires all alpha-test tasks to use the same method list.")

    base_scenario = dict(tasks[0]["scenario"])
    base_config = _scenario_config(config, base_scenario)
    base_data = generate_dgp_data(base_config, seed=seed)
    warnings = _check_dgp_acceptance(base_data)
    base_records = _sort_records(base_data.records)
    train = base_records.loc[base_records["split"] == "train"].copy()
    val = base_records.loc[base_records["split"] == "val"].copy()
    shared_name = "shared_fit"
    shared_artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / "q3_component_corr" / shared_name / f"seed_{seed}")
    base_data.env_meta.to_csv(shared_artifacts / "base_env_meta.csv", index=False)
    with (shared_artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(base_data.metadata, handle, indent=2, sort_keys=True)

    rows: list[dict] = []
    diag_frames: list[pd.DataFrame] = []
    shared_audit_rows: list[dict] = []
    shared_selection_rows: list[dict] = []
    fit_methods, reported_methods, selector_aliases = _fit_methods_for_selectors(methods)
    method_cache: dict[str, dict] = {}
    method_iter = tqdm(fit_methods, desc=f"q3_component_corr shared_fit seed {seed}", leave=False) if show_progress else fit_methods
    for method in method_iter:
        model = build_exposure_model(method, base_config, seed)
        model.fit(train, val)
        selection = getattr(model, "selection_summary_", {})
        if selection:
            shared_selection_rows.append(
                {
                    "question": "q3_component_corr",
                    "scenario": shared_name,
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    **{
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in selection.items()
                    },
                }
            )
        audit = _check_opflow_acceptance(model, train, method)
        if audit:
            shared_audit_rows.append({"question": "q3_component_corr", "scenario": shared_name, "regime": regime, "seed": seed, "method": method, **audit})
        base_pred = model.predict(base_records)
        base_metrics = evaluate_predictions(base_records, base_pred, topk=int(base_config.get("baseline", {}).get("topk", 5)))
        method_cache[method] = {"model": model, "base_metrics": base_metrics, "audit": audit}

    if shared_audit_rows:
        pd.DataFrame(shared_audit_rows).to_csv(shared_artifacts / "gradient_audit.csv", index=False)
    if shared_selection_rows:
        pd.DataFrame(shared_selection_rows).to_csv(shared_artifacts / "selection_diagnostics.csv", index=False)

    for task in tasks:
        task_config = _scenario_config(config, task["scenario"])
        data = generate_dgp_data(task_config, seed=seed)
        task_warnings = _check_dgp_acceptance(data)
        records = _sort_records(data.records)
        _assert_q3_train_val_fixed(base_records, records, task["name"])
        artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / "q3_component_corr" / task["name"] / f"seed_{seed}")
        data.env_meta.to_csv(artifacts / "env_meta.csv", index=False)
        with (artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(data.metadata, handle, indent=2, sort_keys=True)

        env_risk_rows: list[dict] = []
        operator_rows: list[dict] = []
        scenario_cache: dict[str, dict] = {}
        for method in fit_methods:
            cached = method_cache[method]
            model = cached["model"]
            pred = model.predict(records)
            metrics = evaluate_predictions(records, pred, topk=int(task_config.get("baseline", {}).get("topk", 5)))
            extra = _extra_metrics(model, records, pred)
            audit = cached.get("audit", {})
            for key in ("op_monotone_violation_D", "op_monotone_violation_S", "op_monotone_violation_B"):
                if key in audit:
                    extra[key] = float(audit[key])
            scenario_cache[method] = {"prediction": pred, "metrics": metrics, "extra": extra}
            if method in reported_methods:
                env_risk_rows.extend(_env_risk_rows(records, pred, question="q3_component_corr", scenario=task["name"], regime=regime, method=method, seed=seed))
                operator_rows.append(_operator_diagnostic_row("q3_component_corr", task["name"], regime, method, seed, extra))
                rows.extend(
                    _metric_rows(
                        question="q3_component_corr",
                        scenario_name=task["name"],
                        scenario_value_name=task["value_name"],
                        scenario_value=task["value"],
                        stress_type=task["stress_type"],
                        regime=regime,
                        method=method,
                        seed=seed,
                        metrics=metrics,
                        extra=extra,
                    )
                )

        selection_rows = _with_scenario(shared_selection_rows, task["name"], reused_fit_from=shared_name)
        for alias in selector_aliases:
            selected, scores = _select_best_by_val_row_kl(
                {method: {"metrics": cached["base_metrics"]} for method, cached in method_cache.items()},
                BEST_SELECTOR_POOLS[alias],
            )
            selection_rows.append(
                {
                    "question": "q3_component_corr",
                    "scenario": task["name"],
                    "regime": regime,
                    "seed": seed,
                    "method": alias,
                    "selection_protocol": "validation_alias",
                    "selection_metric": "row_kl",
                    "selected_method": selected or "",
                    "candidate_val_row_kl": json.dumps(scores, sort_keys=True),
                    "reused_fit_from": shared_name,
                }
            )
            if selected is None:
                continue
            cached = scenario_cache[selected]
            pred = cached["prediction"]
            metrics = cached["metrics"]
            extra = cached["extra"]
            env_risk_rows.extend(_env_risk_rows(records, pred, question="q3_component_corr", scenario=task["name"], regime=regime, method=alias, seed=seed))
            operator_rows.append(_operator_diagnostic_row("q3_component_corr", task["name"], regime, alias, seed, extra))
            rows.extend(
                _metric_rows(
                    question="q3_component_corr",
                    scenario_name=task["name"],
                    scenario_value_name=task["value_name"],
                    scenario_value=task["value"],
                    stress_type=task["stress_type"],
                    regime=regime,
                    method=alias,
                    seed=seed,
                    metrics=metrics,
                    extra=extra,
                )
            )

        diag = data.env_meta.copy()
        diag["regime"] = regime
        diag["seed"] = seed
        diag["scenario"] = task["name"]
        diag["scenario_value_name"] = task["value_name"]
        diag["scenario_value"] = task["value"]
        diag["stress_type"] = task["stress_type"]
        diag["fit_reuse_group"] = shared_name
        diag["diagnostic_note"] = "; ".join([*warnings, *task_warnings])
        diag_frames.append(diag)
        if shared_audit_rows:
            pd.DataFrame(_with_scenario(shared_audit_rows, task["name"], reused_fit_from=shared_name)).to_csv(artifacts / "gradient_audit.csv", index=False)
        if env_risk_rows:
            pd.DataFrame(env_risk_rows).to_csv(artifacts / "env_risks.csv", index=False)
        if operator_rows:
            pd.DataFrame(operator_rows).to_csv(artifacts / "operator_diagnostics.csv", index=False)
        if selection_rows:
            pd.DataFrame(selection_rows).to_csv(artifacts / "selection_diagnostics.csv", index=False)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    return pd.DataFrame(rows), diagnostics


def _standard_dgp_config(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out.pop("scenario", None)
    if isinstance(out.get("dgp"), dict):
        out["dgp"] = dict(out["dgp"])
        out["dgp"].pop("scenario", None)
    return out


def _run_residual_drift(
    config: dict,
    regime: str,
    seed: int,
    methods: list[str],
    kappa_values: list[float],
    run_dir: Path,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regime != DGP_REGIME:
        raise ValueError(f"Only DGP regime is supported, got {regime!r}.")
    base_config = _standard_dgp_config(config)
    data = generate_dgp_data(base_config, seed=seed)
    warnings = _check_dgp_acceptance(data)
    base_records = _sort_records(data.records)
    train = base_records.loc[base_records["split"] == "train"].copy()
    val = base_records.loc[base_records["split"] == "val"].copy()
    artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / STRESS_RESIDUAL_DRIFT / f"seed_{seed}")
    data.env_meta.to_csv(artifacts / "base_env_meta.csv", index=False)
    with (artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(data.metadata, handle, indent=2, sort_keys=True)
    rows: list[dict] = []
    diag_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    env_risk_rows: list[dict] = []
    operator_rows: list[dict] = []
    selection_rows: list[dict] = []
    drifted_records: dict[float, pd.DataFrame] = {}
    for kappa in kappa_values:
        drifted = apply_residual_behavioral_drift(data, base_config, seed=seed, kappa=float(kappa))
        records = _sort_records(drifted.records)
        drifted_records[float(kappa)] = records
        diag = drifted.env_meta.copy()
        diag["regime"] = regime
        diag["seed"] = seed
        diag["scenario"] = f"kappa_{float(kappa):g}"
        diag["scenario_value_name"] = "kappa"
        diag["scenario_value"] = float(kappa)
        diag["stress_type"] = "residual_behavioral_drift"
        diag["diagnostic_note"] = "; ".join(warnings)
        diag_frames.append(diag)
    fit_methods, reported_methods, selector_aliases = _fit_methods_for_selectors(methods)
    method_cache: dict[str, dict] = {}
    method_kappa_cache: dict[tuple[str, float], dict] = {}
    method_iter = tqdm(fit_methods, desc=f"{STRESS_RESIDUAL_DRIFT} seed {seed}", leave=False) if show_progress else fit_methods
    for method in method_iter:
        model = build_exposure_model(method, base_config, seed)
        model.fit(train, val)
        selection = getattr(model, "selection_summary_", {})
        if selection:
            selection_rows.append(
                {
                    "question": STRESS_RESIDUAL_DRIFT,
                    "scenario": "kappa_sweep",
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    **{
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in selection.items()
                    },
                }
            )
        audit = _check_opflow_acceptance(model, train, method)
        if audit:
            audit_rows.append({"question": STRESS_RESIDUAL_DRIFT, "scenario": "kappa_sweep", "regime": regime, "seed": seed, "method": method, **audit})
        base_pred = model.predict(base_records)
        base_metrics = evaluate_predictions(base_records, base_pred, topk=int(base_config.get("baseline", {}).get("topk", 5)))
        method_cache[method] = {"model": model, "metrics": base_metrics}
        for kappa, records in drifted_records.items():
            scenario_name = f"kappa_{float(kappa):g}"
            pred = model.predict(records)
            metrics = evaluate_predictions(records, pred, topk=int(base_config.get("baseline", {}).get("topk", 5)))
            extra = _extra_metrics(model, records, pred)
            for key in ("op_monotone_violation_D", "op_monotone_violation_S", "op_monotone_violation_B"):
                if key in audit:
                    extra[key] = float(audit[key])
            method_kappa_cache[(method, float(kappa))] = {"records": records, "prediction": pred, "metrics": metrics, "extra": extra}
            if method in reported_methods:
                env_risk_rows.extend(_env_risk_rows(records, pred, question=STRESS_RESIDUAL_DRIFT, scenario=scenario_name, regime=regime, method=method, seed=seed))
                operator_rows.append(_operator_diagnostic_row(STRESS_RESIDUAL_DRIFT, scenario_name, regime, method, seed, extra))
                for row in _metric_rows(
                    question=STRESS_RESIDUAL_DRIFT,
                    scenario_name=scenario_name,
                    scenario_value_name="kappa",
                    scenario_value=float(kappa),
                    stress_type="residual_behavioral_drift",
                    regime=regime,
                    method=method,
                    seed=seed,
                    metrics=metrics,
                    extra=extra,
                ):
                    if row["split"] == "test":
                        rows.append(row)
    for alias in selector_aliases:
        selected, scores = _select_best_by_val_row_kl(method_cache, BEST_SELECTOR_POOLS[alias])
        selection_rows.append(
            {
                "question": STRESS_RESIDUAL_DRIFT,
                "scenario": "kappa_sweep",
                "regime": regime,
                "seed": seed,
                "method": alias,
                "selection_protocol": "validation_alias",
                "selection_metric": "row_kl",
                "selected_method": selected or "",
                "candidate_val_row_kl": json.dumps(scores, sort_keys=True),
            }
        )
        if selected is None:
            continue
        for kappa in kappa_values:
            scenario_name = f"kappa_{float(kappa):g}"
            cached = method_kappa_cache[(selected, float(kappa))]
            records = cached["records"]
            pred = cached["prediction"]
            metrics = cached["metrics"]
            extra = cached["extra"]
            env_risk_rows.extend(_env_risk_rows(records, pred, question=STRESS_RESIDUAL_DRIFT, scenario=scenario_name, regime=regime, method=alias, seed=seed))
            operator_rows.append(_operator_diagnostic_row(STRESS_RESIDUAL_DRIFT, scenario_name, regime, alias, seed, extra))
            for row in _metric_rows(
                question=STRESS_RESIDUAL_DRIFT,
                scenario_name=scenario_name,
                scenario_value_name="kappa",
                scenario_value=float(kappa),
                stress_type="residual_behavioral_drift",
                regime=regime,
                method=alias,
                seed=seed,
                metrics=metrics,
                extra=extra,
            ):
                if row["split"] == "test":
                    rows.append(row)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(artifacts / "gradient_audit.csv", index=False)
    if env_risk_rows:
        pd.DataFrame(env_risk_rows).to_csv(artifacts / "env_risks.csv", index=False)
    if operator_rows:
        pd.DataFrame(operator_rows).to_csv(artifacts / "operator_diagnostics.csv", index=False)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(artifacts / "selection_diagnostics.csv", index=False)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    if not diagnostics.empty:
        diagnostics.to_csv(artifacts / "drift_env_meta.csv", index=False)
    return pd.DataFrame(rows), diagnostics


def _ci95(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return float("nan")
    return float(1.96 * numeric.std(ddof=1) / math.sqrt(len(numeric)))


def _debug_values(values: list[float]) -> list[float]:
    if len(values) <= 2:
        return values
    return [values[0], values[-1]]


def _question_tasks(config: dict[str, Any], questions: list[str], debug: bool, requested_methods: list[str] | None) -> list[dict[str, Any]]:
    qcfg = config.get("synthetic_questions", {})
    main_methods = requested_methods or list(config.get("experiment", {}).get("methods", MAIN_METHODS))
    q1_methods = requested_methods or list(qcfg.get("q1_scale", {}).get("methods", Q1_METHODS))
    q4_methods = requested_methods or list(config.get("experiment", {}).get("q4_methods", Q4_METHODS))
    tasks: list[dict[str, Any]] = []
    if "q1_scale" in questions:
        values = [float(v) for v in qcfg.get("q1_scale", {}).get("scale_severity_list", [0.0, 0.5, 1.0, 1.5])]
        for value in (_debug_values(values) if debug else values):
            scenario = {"question": "q1_scale", "scale_severity": value}
            tasks.append({"question": "q1_scale", "scenario": scenario, "name": f"scale_{value:g}", "value_name": "severity", "value": value, "stress_type": "", "methods": q1_methods})
    if "q2_opportunity" in questions:
        values = [float(v) for v in qcfg.get("q2_opportunity", {}).get("opp_severity_list", [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])]
        for value in (_debug_values(values) if debug else values):
            scenario = {"question": "q2_opportunity", "opp_severity": value}
            tasks.append({"question": "q2_opportunity", "scenario": scenario, "name": f"opp_{value:g}", "value_name": "severity", "value": value, "stress_type": "", "methods": main_methods})
    if "q3_component_corr" in questions:
        q3 = qcfg.get("q3_component_corr", {})
        alpha_train = float(q3.get("alpha_train", 0.7))
        alpha_val = float(q3.get("alpha_val", 0.5))
        values = [float(v) for v in q3.get("alpha_test_list", [0.7, 0.3, 0.0, -0.3, -0.6])]
        for value in (_debug_values(values) if debug else values):
            scenario = {"question": "q3_component_corr", "alpha_train": alpha_train, "alpha_val": alpha_val, "alpha_test": value}
            tasks.append({"question": "q3_component_corr", "scenario": scenario, "name": f"alpha_{value:g}", "value_name": "alpha_test", "value": value, "stress_type": "", "methods": main_methods})
    if "q4_ablation" in questions:
        q4 = qcfg.get("q4_ablation", {})
        q3 = qcfg.get("q3_component_corr", {})
        opp_value = float(q4.get("stress_opp_severity", 0.9))
        alpha_value = float(q4.get("stress_alpha_test", -0.6))
        tasks.append(
            {
                "question": "q4_ablation",
                "scenario": {"question": "q2_opportunity", "opp_severity": opp_value},
                "name": f"stress_q2_opp_{opp_value:g}",
                "value_name": "stress",
                "value": opp_value,
                "stress_type": "q2_opportunity",
                "methods": q4_methods,
            }
        )
        tasks.append(
            {
                "question": "q4_ablation",
                "scenario": {
                    "question": "q3_component_corr",
                    "alpha_train": float(q3.get("alpha_train", 0.7)),
                    "alpha_val": float(q3.get("alpha_val", 0.5)),
                    "alpha_test": alpha_value,
                },
                "name": f"stress_q3_alpha_{alpha_value:g}",
                "value_name": "stress",
                "value": alpha_value,
                "stress_type": "q3_component_corr",
                "methods": q4_methods,
            }
        )
    return tasks


def _test_rows(metrics: pd.DataFrame, question: str) -> pd.DataFrame:
    return metrics.loc[(metrics["question"] == question) & (metrics["split"] == "test")].copy()


def _write_question_metrics(metrics: pd.DataFrame, diagnostics: pd.DataFrame, results_root: str | Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    root = Path(results_root)
    common_metrics = ["row_kl", "row_cpc", "rmse", "mae", "pmse"]
    q1 = _test_rows(metrics, "q1_scale")
    if not q1.empty:
        out = ensure_dir(root / "q1_scale")
        frame = q1.rename(columns={"method": "model", "scenario_value": "severity"})[["seed", "severity", "model", *common_metrics]]
        frame.to_csv(out / "metrics.csv", index=False)
        written["q1_scale"] = out / "metrics.csv"
    q2 = _test_rows(metrics, "q2_opportunity")
    if not q2.empty:
        out = ensure_dir(root / "q2_opportunity")
        frame = q2.rename(columns={"method": "model", "scenario_value": "severity"})[["seed", "severity", "model", *common_metrics]]
        frame.to_csv(out / "metrics.csv", index=False)
        written["q2_opportunity"] = out / "metrics.csv"
    q3 = _test_rows(metrics, "q3_component_corr")
    if not q3.empty:
        out = ensure_dir(root / "q3_component_corr")
        frame = q3.rename(columns={"method": "model", "scenario_value": "alpha_test"})[["seed", "alpha_test", "model", *common_metrics]]
        frame.to_csv(out / "metrics.csv", index=False)
        corr_cols = ["seed", "env_id", "split", "alpha_target", "empirical_corr_log_o_accessibility"]
        corr = diagnostics.loc[diagnostics["question"].isin(["q3_component_corr"]) & diagnostics["alpha_target"].notna(), corr_cols].copy()
        corr.to_csv(out / "correlation_checks.csv", index=False)
        written["q3_component_corr"] = out / "metrics.csv"
    q4 = _test_rows(metrics, "q4_ablation")
    if not q4.empty:
        out = ensure_dir(root / "q4_ablation")
        frame = q4.rename(columns={"method": "ablation"})[["seed", "stress_type", "ablation", *common_metrics]]
        frame.to_csv(out / "metrics.csv", index=False)
        written["q4_ablation"] = out / "metrics.csv"
    stress = _test_rows(metrics, STRESS_RESIDUAL_DRIFT)
    if not stress.empty:
        out = ensure_dir(root / STRESS_RESIDUAL_DRIFT)
        frame = stress.rename(columns={"method": "model", "scenario_value": "kappa"})[
            ["seed", "kappa", "model", "row_kl", "row_cpc", "rmse", "mae", "pmse"]
        ]
        frame.to_csv(out / "metrics.csv", index=False)
        written[STRESS_RESIDUAL_DRIFT] = out / "metrics.csv"
    return written


def _mean_ci(frame: pd.DataFrame, group_cols: list[str], value_col: str = "row_kl") -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        values = pd.to_numeric(group[value_col], errors="coerce")
        row[f"{value_col}_mean"] = float(values.mean())
        row[f"{value_col}_ci95"] = _ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_line(ax, frame: pd.DataFrame, x_col: str, title: str, ylabel: str = "Row-KL", methods: list[str] | None = None) -> None:
    if frame.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    if methods is not None:
        frame = frame.loc[frame["model"].isin(methods)].copy()
        if frame.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No plotted models", ha="center", va="center")
            return
    summary = _mean_ci(frame, [x_col, "model"], "row_kl")
    for model, group in summary.groupby("model", sort=False):
        group = group.sort_values(x_col)
        ax.errorbar(group[x_col], group["row_kl_mean"], yerr=group["row_kl_ci95"].fillna(0.0), marker="o", linewidth=1.6, capsize=2, label=model)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _write_main_figure(results_root: str | Path, figures_dir: str | Path = "figures") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_root)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if (root / "q1_scale" / "metrics.csv").exists():
        _plot_line(axes[0, 0], pd.read_csv(root / "q1_scale" / "metrics.csv"), "severity", "A. Training scale heterogeneity", methods=Q1_PLOT_METHODS)
    else:
        _plot_line(axes[0, 0], pd.DataFrame(), "severity", "A. Training scale heterogeneity")
    if (root / "q2_opportunity" / "metrics.csv").exists():
        _plot_line(axes[0, 1], pd.read_csv(root / "q2_opportunity" / "metrics.csv"), "severity", "B. Opportunity redistribution", methods=MAIN_PLOT_METHODS)
    else:
        _plot_line(axes[0, 1], pd.DataFrame(), "severity", "B. Opportunity redistribution")
    if (root / "q3_component_corr" / "metrics.csv").exists():
        _plot_line(axes[1, 0], pd.read_csv(root / "q3_component_corr" / "metrics.csv"), "alpha_test", "C. Component-correlation shift", methods=MAIN_PLOT_METHODS)
    else:
        _plot_line(axes[1, 0], pd.DataFrame(), "alpha_test", "C. Component-correlation shift")
    ax = axes[1, 1]
    q4_path = root / "q4_ablation" / "metrics.csv"
    if q4_path.exists():
        q4 = pd.read_csv(q4_path)
        per_seed = q4.groupby(["seed", "ablation"], as_index=False)["row_kl"].mean()
        summary = _mean_ci(per_seed, ["ablation"], "row_kl")
        ax.bar(summary["ablation"], summary["row_kl_mean"], yerr=summary["row_kl_ci95"].fillna(0.0), capsize=3)
        ax.tick_params(axis="x", rotation=35)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("D. Load-bearing ablation")
    ax.set_ylabel("Avg. OOD Row-KL")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
        fig.subplots_adjust(top=0.88)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir = ensure_dir(figures_dir)
    fig.savefig(out_dir / "synthetic_main.png", dpi=200)
    fig.savefig(out_dir / "synthetic_main.pdf")
    plt.close(fig)


def _write_stress_figure(results_root: str | Path, figures_dir: str | Path = "figures") -> tuple[Path, Path] | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(results_root) / STRESS_RESIDUAL_DRIFT / "metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_line(ax, frame, "kappa", "Residual Behavioral Drift", methods=MAIN_PLOT_METHODS)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False)
    fig.tight_layout()
    out_dir = ensure_dir(figures_dir)
    png = out_dir / "stress_residual_drift.png"
    pdf = out_dir / "stress_residual_drift.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _value_at_endpoint(frame: pd.DataFrame, value_col: str, endpoint) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    view = frame.loc[np.isclose(pd.to_numeric(frame[value_col], errors="coerce"), float(endpoint))].copy()
    if view.empty:
        return pd.DataFrame()
    return view.groupby("model", as_index=False).agg(
        row_kl=("row_kl", "mean"),
        row_cpc=("row_cpc", "mean"),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        pmse=("pmse", "mean"),
    )


def _write_endpoint_table(results_root: str | Path, config: dict[str, Any], tables_dir: str | Path = "tables") -> Path:
    root = Path(results_root)
    qcfg = config.get("synthetic_questions", {})
    q1_path = root / "q1_scale" / "metrics.csv"
    q2_path = root / "q2_opportunity" / "metrics.csv"
    q3_path = root / "q3_component_corr" / "metrics.csv"
    q4_path = root / "q4_ablation" / "metrics.csv"
    q1 = pd.read_csv(q1_path) if q1_path.exists() else pd.DataFrame()
    q2 = pd.read_csv(q2_path) if q2_path.exists() else pd.DataFrame()
    q3 = pd.read_csv(q3_path) if q3_path.exists() else pd.DataFrame()
    q4 = pd.read_csv(q4_path) if q4_path.exists() else pd.DataFrame()
    q1_endpoint = max(qcfg.get("q1_scale", {}).get("scale_severity_list", [1.5]))
    q2_endpoint = max(qcfg.get("q2_opportunity", {}).get("opp_severity_list", [0.9]))
    q3_endpoint = min(qcfg.get("q3_component_corr", {}).get("alpha_test_list", [-0.6]))
    rows: dict[str, dict[str, Any]] = {}
    for label, frame in (
        ("q1_strong", _value_at_endpoint(q1, "severity", q1_endpoint)),
        ("q2_strong", _value_at_endpoint(q2, "severity", q2_endpoint)),
        ("q3_strong", _value_at_endpoint(q3, "alpha_test", q3_endpoint)),
    ):
        for row in frame.itertuples(index=False):
            item = rows.setdefault(str(row.model), {"model_or_ablation": str(row.model)})
            for metric in ("row_kl", "row_cpc", "rmse", "mae", "pmse"):
                item[f"{label}_{metric}"] = float(getattr(row, metric))
    if not q4.empty:
        per_seed = q4.groupby(["seed", "ablation"], as_index=False)[["row_kl", "row_cpc", "rmse", "mae", "pmse"]].mean()
        for row in per_seed.groupby("ablation", as_index=False).agg(
            row_kl=("row_kl", "mean"),
            row_cpc=("row_cpc", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            pmse=("pmse", "mean"),
        ).itertuples(index=False):
            item = rows.setdefault(str(row.ablation), {"model_or_ablation": str(row.ablation)})
            for metric in ("row_kl", "row_cpc", "rmse", "mae", "pmse"):
                item[f"avg_ood_{metric}"] = float(getattr(row, metric))
    out = pd.DataFrame(rows.values())
    path = ensure_dir(tables_dir) / "synthetic_endpoint.csv"
    out.to_csv(path, index=False)
    return path


def _best_line(path: Path, model_col: str = "model") -> str:
    if not path.exists():
        return "not run"
    frame = pd.read_csv(path)
    if frame.empty:
        return "not run"
    summary = frame.groupby(model_col, as_index=False)["row_kl"].mean().sort_values("row_kl")
    row = summary.iloc[0]
    return f"{row[model_col]} Row-KL={float(row['row_kl']):.4f}"


def _write_report(
    results_root: str | Path,
    config: dict[str, Any],
    run_dir: Path,
    written: dict[str, Path],
    *,
    figures_dir: str | Path = "figures",
    tables_dir: str | Path = "tables",
    reports_dir: str | Path = "reports",
) -> Path:
    root = Path(results_root)
    lines = [
        "# Synthetic Four-Question Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## DGP Settings",
        "",
        f"- suite/regime: `{DGP_REGIME}`",
        f"- N: `{config.get('dgp', {}).get('N', config.get('N', 80))}` shared origin/destination units; self-loops excluded.",
        f"- train/val/test envs: `{config.get('dgp', {}).get('num_train_env')}/{config.get('dgp', {}).get('num_val_env')}/{config.get('dgp', {}).get('num_test_env')}`",
        "- oracle fields are diagnostics only; deployed models use coordinates, opportunity proxy, cost/rank, and row-standardized cost.",
        f"- RawCountMLP objective: `{config.get('exposure_model', {}).get('raw_count_loss', 'mse')}` raw-count reconstruction.",
        "",
        "## Metrics",
        "",
        "- Reported metrics: Row-KL, Row-CPC, raw-domain RMSE, raw-domain MAE, and PMSE.",
        "- Row-KL/Row-CPC evaluate row-normalized allocation; RMSE/MAE evaluate raw OD counts; PMSE evaluates row-centered oracle potential.",
        "",
        "## Main Findings",
        "",
        f"- Q1 training scale heterogeneity: {_best_line(root / 'q1_scale' / 'metrics.csv')}.",
        f"- Q2 opportunity redistribution: {_best_line(root / 'q2_opportunity' / 'metrics.csv')}.",
        f"- Q3 component-correlation shift: {_best_line(root / 'q3_component_corr' / 'metrics.csv')}.",
        f"- Q4 load-bearing ablation: {_best_line(root / 'q4_ablation' / 'metrics.csv', model_col='ablation')}.",
        "",
        "## Interpretation Guardrails",
        "",
        "- Q1 varies train/validation origin-scale dispersion while holding oracle allocation fixed; it tests raw-count supervision rather than a pure test-time scale shock.",
        "- Q2 tests equivariant adaptation to opportunity redistribution.",
        "- Q3 uses admissible opportunity/accessibility variables and no explicit shortcut residual.",
        "- Q4 identifies load-bearing components under controlled OOD stress.",
        "",
        "## Deliverables",
        "",
    ]
    for key, path in sorted(written.items()):
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            f"- `{Path(figures_dir) / 'synthetic_main.png'}`",
            f"- `{Path(figures_dir) / 'synthetic_main.pdf'}`",
            f"- `{Path(tables_dir) / 'synthetic_endpoint.csv'}`",
        ]
    )
    path = ensure_dir(reports_dir) / "synthetic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_four_question_outputs(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: dict,
    run_dir: Path,
    results_root: str | Path,
    *,
    figures_dir: str | Path = "figures",
    tables_dir: str | Path = "tables",
    reports_dir: str | Path = "reports",
    skip_paper_outputs: bool = False,
) -> dict[str, Path]:
    if skip_paper_outputs:
        return {}
    written = _write_question_metrics(metrics, diagnostics, results_root)
    questions = set(metrics["question"].dropna()) if "question" in metrics else set()
    if questions.intersection(MAIN_QUESTION_CHOICES):
        _write_main_figure(results_root, figures_dir=figures_dir)
        endpoint = _write_endpoint_table(results_root, config, tables_dir=tables_dir)
        report = _write_report(
            results_root,
            config,
            run_dir,
            written,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            reports_dir=reports_dir,
        )
        written["synthetic_endpoint"] = endpoint
        written["synthetic_report"] = report
    if STRESS_RESIDUAL_DRIFT in questions:
        stress_figures = _write_stress_figure(results_root, figures_dir=figures_dir)
        if stress_figures is not None:
            written["stress_residual_drift_png"] = stress_figures[0]
            written["stress_residual_drift_pdf"] = stress_figures[1]
    return written


def main() -> None:
    args = parse_args()
    config = load_exposure_config(args.config, args.override)
    if args.mode == "debug":
        args.debug = True
    if args.debug:
        config = _apply_debug(config)
    if args.mode == "debug" and args.seeds is None:
        seeds = [0]
    elif args.mode == "full" and args.seeds is None:
        seeds = list(range(30))
    else:
        seeds = args.seeds or [args.seed]
    regimes = args.regimes or ([args.regime] if args.regime else config.get("experiment", {}).get("regimes", [DGP_REGIME]))
    questions = args.questions or config.get("experiment", {}).get("questions", list(MAIN_QUESTION_CHOICES))
    requested_methods = args.methods or None
    run_dir = Path(create_run_dir("DGP", run_id=args.run_id, run_name=args.run_name, output_root=args.output_dir))
    save_resolved_config(config, run_dir)
    questions = list(questions)
    stress_requested = STRESS_RESIDUAL_DRIFT in questions
    tasks = _question_tasks(config, [question for question in questions if question != STRESS_RESIDUAL_DRIFT], args.debug, requested_methods)
    stress_cfg = config.get("synthetic_questions", {}).get(STRESS_RESIDUAL_DRIFT, {})
    stress_kappas = [float(v) for v in stress_cfg.get("kappa_list", [0.0, 0.1, 0.2, 0.3, 0.5])]
    if args.debug:
        stress_kappas = _debug_values(stress_kappas)
    stress_methods = requested_methods or list(stress_cfg.get("methods", STRESS_METHODS))
    metric_frames = []
    diag_frames = []
    q2_cfg = config.get("synthetic_questions", {}).get("q2_opportunity", {})
    reuse_q2_fit = bool(q2_cfg.get("reuse_train_fit", True))
    q2_tasks = [task for task in tasks if task["question"] == "q2_opportunity"]
    q3_cfg = config.get("synthetic_questions", {}).get("q3_component_corr", {})
    reuse_q3_fit = bool(q3_cfg.get("reuse_train_fit", True))
    q3_tasks = [task for task in tasks if task["question"] == "q3_component_corr"]
    for seed in seeds:
        for regime in regimes:
            q2_sweep_done = False
            q3_sweep_done = False
            for task in tasks:
                if reuse_q2_fit and task["question"] == "q2_opportunity":
                    if not q2_sweep_done:
                        metrics, diagnostics = _run_q2_opportunity_sweep(
                            config,
                            regime,
                            seed,
                            q2_tasks,
                            run_dir,
                            show_progress=True,
                        )
                        metrics["experiment"] = "q2_opportunity"
                        metric_frames.append(metrics)
                        diag_frames.append(diagnostics)
                        raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
                        pd.concat(metric_frames, ignore_index=True).to_csv(raw_dir / "raw_metrics.partial.csv", index=False)
                        q2_sweep_done = True
                    continue
                if reuse_q3_fit and task["question"] == "q3_component_corr":
                    if not q3_sweep_done:
                        metrics, diagnostics = _run_q3_component_corr_sweep(
                            config,
                            regime,
                            seed,
                            q3_tasks,
                            run_dir,
                            show_progress=True,
                        )
                        metrics["experiment"] = "q3_component_corr"
                        metric_frames.append(metrics)
                        diag_frames.append(diagnostics)
                        raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
                        pd.concat(metric_frames, ignore_index=True).to_csv(raw_dir / "raw_metrics.partial.csv", index=False)
                        q3_sweep_done = True
                    continue
                task_config = _scenario_config(config, task["scenario"])
                metrics, diagnostics = _run_one(
                    task_config,
                    regime,
                    seed,
                    list(task["methods"]),
                    run_dir,
                    question=task["question"],
                    scenario_name=task["name"],
                    scenario_value_name=task["value_name"],
                    scenario_value=task["value"],
                    stress_type=task["stress_type"],
                    show_progress=True,
                )
                metrics["experiment"] = task["question"]
                metric_frames.append(metrics)
                diag_frames.append(diagnostics)
                raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
                pd.concat(metric_frames, ignore_index=True).to_csv(raw_dir / "raw_metrics.partial.csv", index=False)
            if stress_requested:
                metrics, diagnostics = _run_residual_drift(
                    config,
                    regime,
                    seed,
                    list(stress_methods),
                    list(stress_kappas),
                    run_dir,
                    show_progress=True,
                )
                metrics["experiment"] = STRESS_RESIDUAL_DRIFT
                metric_frames.append(metrics)
                diag_frames.append(diagnostics)
                raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
                pd.concat(metric_frames, ignore_index=True).to_csv(raw_dir / "raw_metrics.partial.csv", index=False)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
    metrics.to_csv(raw_dir / "raw_metrics.csv", index=False)
    diagnostics.to_csv(raw_dir / "diagnostics.csv", index=False)
    written = _write_four_question_outputs(
        metrics,
        diagnostics,
        config,
        run_dir,
        args.results_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        reports_dir=args.reports_dir,
        skip_paper_outputs=bool(args.skip_paper_outputs),
    )
    standard = metrics_frame_to_standard(
        metrics,
        experiment="DGP",
        regime="DGP",
        config=config,
        gate_diagnostics=pd.DataFrame(),
        notes="debug" if args.debug else "DGP four-question suite",
    )
    write_standard_outputs(
        run_dir,
        experiment="DGP",
        config_path=args.config,
        standard_metrics=standard,
        diagnostics={
            "regimes": regimes,
            "questions": questions,
            "tasks": len(tasks) + int(stress_requested),
            "diagnostic_rows": int(len(diagnostics)),
            "outputs": {key: str(path) for key, path in written.items()},
        },
        warnings=[],
    )
    print(f"Run directory: {run_dir}")
    if args.skip_paper_outputs:
        print("Paper-facing outputs skipped by --skip-paper-outputs.")
    else:
        for key, path in sorted(written.items()):
            print(f"{key}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
