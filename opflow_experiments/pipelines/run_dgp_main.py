"""Canonical runner for the final DGP."""

from __future__ import annotations

import argparse
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

from opflow_experiments.data.dgp import DGP_REGIME, generate_dgp_data
from opflow_experiments.evaluation.evaluator import evaluate_predictions
from opflow_experiments.evaluation.standard_outputs import metrics_frame_to_standard, write_standard_outputs
from opflow_experiments.exposure_models import ExposureOpFlowModel, build_exposure_model
from opflow_experiments.utils.config import deep_merge, load_resolved_config, save_resolved_config
from opflow_experiments.utils.io import ensure_dir
from opflow_experiments.utils.paths import create_run_dir


REGIME_LABELS = {DGP_REGIME: DGP_REGIME}
DEFAULT_METHODS = {
    DGP_REGIME: [
        "RawCountMLP",
        "RawCountGNN",
        "PairMLP",
        "ShortcutOnly",
        "GenericPotentialMLP",
        "ScaleAwareMLP",
        "OpFlow",
        "OraclePotential",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--regime", choices=sorted(REGIME_LABELS), default=None)
    parser.add_argument("--regimes", nargs="*", choices=sorted(REGIME_LABELS), default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=["debug", "full"], default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _ensure_defaults(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    out.setdefault("suite", DGP_REGIME)
    out.setdefault("N_o", 80)
    out.setdefault("N_d", 80)
    out.setdefault("tau", 0.7)
    out.setdefault("experiment", {}).setdefault("regimes", [DGP_REGIME])
    dgp = dict(out.get("dgp", {}))
    dgp.setdefault("N_o", int(out.get("N_o", 80)))
    dgp.setdefault("N_d", int(out.get("N_d", 80)))
    dgp.setdefault("K_mean", 25)
    dgp.setdefault("K_std", 5)
    dgp.setdefault("K_min", 15)
    dgp.setdefault("K_max", 40)
    dgp.setdefault("num_train_env", 8)
    dgp.setdefault("num_val_env", 2)
    dgp.setdefault("num_test_env", 4)
    dgp.setdefault("tau", out.get("tau", 0.7))
    dgp.setdefault("omega", 0.15)
    out["dgp"] = dgp
    model = dict(out.get("exposure_model", {}))
    model.setdefault("device", "cpu")
    model.setdefault("hidden_dim", 64)
    model.setdefault("epochs", 180)
    model.setdefault("learning_rate", 0.002)
    model.setdefault("patience", 25)
    model.setdefault("alpha_smooth", 0.5)
    model.setdefault("lambda_count", 0.05)
    model.setdefault("lambda_scale", 0.1)
    model.setdefault("lambda_vrex", 0.1)
    model.setdefault("lambda_op", 0.01)
    model.setdefault("groupdro_weight", 0.005)
    model.setdefault("lambda_operator_anchor", 0.03)
    model.setdefault("lambda_o_anchor", 1.0)
    model.setdefault("lambda_c_anchor", 1.0)
    model.setdefault("checkpoint_metric", "alloc_loss")
    model.setdefault("rank_prefix_mode", "soft")
    model.setdefault("rank_prefix_temperature", 0.05)
    model.setdefault("benefit_skip", float(dgp.get("beta_o", 1.15)))
    model.setdefault("deterrence_skip", -float(dgp.get("beta_c", 0.35)))
    model.setdefault("exposure_skip", -float(dgp.get("beta_s", 0.65)))
    model.setdefault("competition_skip", float(dgp.get("beta_l", 0.45)))
    model.setdefault("use_groupdro", True)
    model.setdefault("use_rank_exposure", True)
    model.setdefault("use_competition", True)
    model.setdefault("use_opportunity_field", True)
    model.setdefault("use_row_center", True)
    model.setdefault("detach_count_allocation", True)
    model.setdefault("use_sit_operator_potential", True)
    model.setdefault("use_monotone_penalty", True)
    model.setdefault("use_vrex", True)
    out["exposure_model"] = model
    out.setdefault("baseline", {}).setdefault("topk", 5)
    return out


def _apply_debug(config: dict[str, Any]) -> dict[str, Any]:
    out = _ensure_defaults(config)
    dgp = out["dgp"]
    dgp["N_o"] = min(int(dgp.get("N_o", 80)), 16)
    dgp["N_d"] = min(int(dgp.get("N_d", 80)), 24)
    dgp["K_mean"] = min(float(dgp.get("K_mean", 25)), 8.0)
    dgp["K_std"] = min(float(dgp.get("K_std", 5)), 2.0)
    dgp["K_min"] = min(int(dgp.get("K_min", 15)), 4, int(dgp["N_d"]))
    dgp["K_max"] = min(int(dgp.get("K_max", 40)), 12, int(dgp["N_d"]))
    dgp["num_train_env"] = 2
    dgp["num_val_env"] = 1
    dgp["num_test_env"] = 1
    model = out["exposure_model"]
    model["hidden_dim"] = min(int(model.get("hidden_dim", 64)), 16)
    model["epochs"] = min(int(model.get("epochs", 180)), 10)
    model["patience"] = min(int(model.get("patience", 25)), 4)
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


def _metric_rows(regime: str, method: str, seed: int, metrics: dict, extra: dict[str, float]) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        split_metrics = metrics[split]
        rows.append(
            {
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
                "pmse": split_metrics.get("pmse"),
                "score_corr": split_metrics.get("score_corr"),
                "exposure_alignment": extra.get("exposure_alignment", np.nan) if split == "test" else np.nan,
                "o_hat_within_dest_std_max": extra.get("o_hat_within_dest_std_max", np.nan) if split == "test" else np.nan,
                "o_hat_within_dest_std_mean": extra.get("o_hat_within_dest_std_mean", np.nan) if split == "test" else np.nan,
                "ell_hat_vs_true_corr": extra.get("ell_hat_vs_true_corr", np.nan) if split == "test" else np.nan,
                "alloc_risk_max": extra.get("alloc_risk_max", np.nan) if split == "test" else np.nan,
                "alloc_risk_mean": extra.get("alloc_risk_mean", np.nan) if split == "test" else np.nan,
                "vrex_alloc_var": extra.get("vrex_alloc_var", np.nan) if split == "test" else np.nan,
                "scale_risk_max": extra.get("scale_risk_max", np.nan) if split == "test" else np.nan,
                "scale_risk_mean": extra.get("scale_risk_mean", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_D": extra.get("op_monotone_violation_D", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_S": extra.get("op_monotone_violation_S", np.nan) if split == "test" else np.nan,
                "op_monotone_violation_B": extra.get("op_monotone_violation_B", np.nan) if split == "test" else np.nan,
                "operator_mean_o_tilde": extra.get("operator_mean_o_tilde", np.nan) if split == "test" else np.nan,
                "operator_mean_c_tilde": extra.get("operator_mean_c_tilde", np.nan) if split == "test" else np.nan,
                "psi_bar_row_mean_abs": extra.get("psi_bar_row_mean_abs", np.nan) if split == "test" else np.nan,
                "shortcut_sensitivity": split_metrics.get("shortcut_sensitivity"),
                "distance_sensitivity": split_metrics.get("distance_sensitivity"),
                "origin_scale_error": split_metrics.get("origin_scale_error"),
                "row_share_sum_error": split_metrics.get("row_share_sum_error"),
                "row_kl_ood_gap": metrics["ood_id_gap"]["row_kl_ood_gap"],
                "topk_ood_drop": metrics["ood_id_gap"]["topk_ood_drop"],
                "potential_rank_corr": split_metrics.get("score_corr"),
                "rankcorr_ood_drop": metrics["ood_id_gap"]["rankcorr_ood_drop"],
                "notes": "DGP",
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
    }
    pred_extra = prediction.get("extra", {})
    score = pred_extra.get("score_hat")
    test_mask = records["split"].to_numpy() == "test"
    if score is not None:
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
    for key in (
        "alloc_risk_max",
        "alloc_risk_mean",
        "vrex_alloc_var",
        "scale_risk_max",
        "scale_risk_mean",
    ):
        if key in pred_extra:
            extra[key] = float(pred_extra[key])
    return extra


def _env_risk_rows(records: pd.DataFrame, prediction: dict, *, regime: str, method: str, seed: int) -> list[dict[str, Any]]:
    working = records[["env_id", "split", "origin_id", "pi_oracle"]].copy()
    working["pi_hat"] = _as_numpy(prediction["pi_hat"]).reshape(-1)
    rows = []
    for (env_id, split, origin_id), group in working.groupby(["env_id", "split", "origin_id"], sort=False):
        p_true = np.clip(group["pi_oracle"].to_numpy(dtype=float), 1e-8, None)
        p_hat = np.clip(group["pi_hat"].to_numpy(dtype=float), 1e-8, None)
        rows.append(
            {
                "env_id": env_id,
                "split": split,
                "origin_id": origin_id,
                "row_kl": float(np.sum(p_true * (np.log(p_true) - np.log(p_hat)))),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    out = []
    for (env_id, split), group in frame.groupby(["env_id", "split"], sort=False):
        out.append(
            {
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


def _operator_diagnostic_row(regime: str, method: str, seed: int, extra: dict[str, float]) -> dict[str, Any]:
    keys = [
        "alloc_risk_max",
        "alloc_risk_mean",
        "vrex_alloc_var",
        "scale_risk_max",
        "scale_risk_mean",
        "op_monotone_violation_D",
        "op_monotone_violation_S",
        "op_monotone_violation_B",
        "operator_mean_o_tilde",
        "operator_mean_c_tilde",
        "psi_bar_row_mean_abs",
        "o_hat_within_dest_std_max",
        "o_hat_within_dest_std_mean",
        "ell_hat_vs_true_corr",
    ]
    return {"regime": regime, "method": method, "seed": seed, **{key: extra.get(key, np.nan) for key in keys}}


def _check_dgp_acceptance(data) -> list[str]:
    warnings: list[str] = []
    meta = data.env_meta
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
    tol = 1e-7
    if method != "NoScaleDetach" and audit.count_scale_potential_grad_max > tol:
        raise AssertionError(f"{method} scale/count loss updates potential branch: {audit.count_scale_potential_grad_max:.3e}")
    if method != "NoRowCenter" and audit.row_center_mean_abs > 1e-6:
        raise AssertionError(f"{method} row-centered potential has nonzero row mean: {audit.row_center_mean_abs:.3e}")
    return {
        "count_scale_potential_grad_max": audit.count_scale_potential_grad_max,
        "row_center_mean_abs": audit.row_center_mean_abs,
        "op_monotone_violation_D": audit.op_monotone_violation_D,
        "op_monotone_violation_S": audit.op_monotone_violation_S,
        "op_monotone_violation_B": audit.op_monotone_violation_B,
    }


def _run_one(
    config: dict,
    regime: str,
    seed: int,
    methods: list[str],
    run_dir: Path,
    show_progress: bool = True,
    grounding_mode: str = "true_opportunity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regime != DGP_REGIME:
        raise ValueError(f"Only DGP regime is supported, got {regime!r}.")
    data = generate_dgp_data(config, seed=seed)
    warnings = _check_dgp_acceptance(data)
    records = _sort_records(data.records)
    train = records.loc[records["split"] == "train"].copy()
    val = records.loc[records["split"] == "val"].copy()
    artifacts = ensure_dir(run_dir / "artifacts" / "DGP" / f"seed_{seed}")
    data.env_meta.to_csv(artifacts / "env_meta.csv", index=False)
    with (artifacts / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(data.metadata, handle, indent=2, sort_keys=True)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    env_risk_rows: list[dict] = []
    operator_rows: list[dict] = []
    method_iter = tqdm(methods, desc=f"DGP seed {seed}", leave=False) if show_progress else methods
    for method in method_iter:
        model = build_exposure_model(method, config, seed)
        model.fit(train, val)
        pred = model.predict(records)
        metrics = evaluate_predictions(records, pred, topk=int(config.get("baseline", {}).get("topk", 5)))
        extra = _extra_metrics(model, records, pred)
        audit = _check_opflow_acceptance(model, train, method)
        if audit:
            audit_rows.append({"regime": regime, "seed": seed, "method": method, **audit})
            for key in ("op_monotone_violation_D", "op_monotone_violation_S", "op_monotone_violation_B"):
                extra[key] = float(audit[key])
        env_risk_rows.extend(_env_risk_rows(records, pred, regime=regime, method=method, seed=seed))
        operator_rows.append(_operator_diagnostic_row(regime, method, seed, extra))
        rows.extend(_metric_rows(regime, method, seed, metrics, extra))
    diag = data.env_meta.copy()
    diag["regime"] = regime
    diag["seed"] = seed
    diag["warning"] = "; ".join(warnings)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(artifacts / "gradient_audit.csv", index=False)
    if env_risk_rows:
        pd.DataFrame(env_risk_rows).to_csv(artifacts / "env_risks.csv", index=False)
    if operator_rows:
        pd.DataFrame(operator_rows).to_csv(artifacts / "operator_diagnostics.csv", index=False)
    return pd.DataFrame(rows), diag


def _ci95(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return float("nan")
    return float(1.96 * numeric.std(ddof=1) / math.sqrt(len(numeric)))


def _mean_ci_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"] == "test"].copy()
    metric_cols = [
        "row_kl",
        "row_cpc",
        "rank_corr",
        "topk_overlap",
        "log_rmse",
        "global_cpc",
        "pmse",
        "potential_rank_corr",
        "origin_scale_error",
        "row_kl_ood_gap",
        "topk_ood_drop",
        "alloc_risk_max",
        "vrex_alloc_var",
        "op_monotone_violation_D",
        "op_monotone_violation_S",
        "op_monotone_violation_B",
    ]
    rows = []
    for (regime, method), group in test.groupby(["regime", "method"], sort=False):
        row = {"regime": regime, "method": method}
        for col in metric_cols:
            if col in group:
                values = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_ci95"] = _ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in view.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_dgp_outputs(metrics: pd.DataFrame, diagnostics: pd.DataFrame, config: dict, run_dir: Path, results_root: str | Path) -> Path:
    out = ensure_dir(Path(results_root) / "DGP")
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    summary = _mean_ci_frame(metrics)
    summary.to_csv(out / "summary_mean_ci.csv", index=False)
    with (out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    report_lines = [
        f"# DGP Run: {run_dir.name}",
        "",
        "Final synthetic DGP with environment-specific opportunity fields and train-only oracle standardization.",
        "",
        "## Test Summary",
        "",
        _markdown_table(summary),
        "",
        "## Diagnostics",
    ]
    for _, row in diagnostics.groupby("env_id", sort=False).first().reset_index().iterrows():
        report_lines.append(
            f"- env {row['env_id']} ({row['split']}): opportunity_mean={float(row['opportunity_mean']):.4f}, "
            f"row_total_mean={float(row['row_total_mean']):.2f}, warning={row.get('warning', '')}"
        )
    report = "\n".join(report_lines) + "\n"
    (out / "summary.md").write_text(report, encoding="utf-8")
    ensure_dir("outputs/reports").joinpath("DGP_report.md").write_text(report, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    config = load_exposure_config(args.config, args.override)
    if args.mode == "debug":
        args.debug = True
    if args.debug:
        config = _apply_debug(config)
    if args.mode == "debug" and args.seeds is None:
        seeds = [0, 1]
    elif args.mode == "full" and args.seeds is None:
        seeds = list(range(30))
    else:
        seeds = args.seeds or [args.seed]
    regimes = args.regimes or ([args.regime] if args.regime else config.get("experiment", {}).get("regimes", [DGP_REGIME]))
    requested_methods = args.methods or config.get("experiment", {}).get("methods")
    run_dir = Path(create_run_dir("DGP", run_id=args.run_id, run_name=args.run_name, output_root=args.output_dir))
    save_resolved_config(config, run_dir)
    metric_frames = []
    diag_frames = []
    for seed in seeds:
        for regime in regimes:
            methods = requested_methods or DEFAULT_METHODS[regime]
            metrics, diagnostics = _run_one(config, regime, seed, methods, run_dir, show_progress=True)
            metrics["experiment"] = REGIME_LABELS[regime]
            metric_frames.append(metrics)
            diag_frames.append(diagnostics)
            raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
            pd.concat(metric_frames, ignore_index=True).to_csv(raw_dir / "raw_metrics.partial.csv", index=False)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    raw_dir = ensure_dir(run_dir / "artifacts" / "DGP")
    metrics.to_csv(raw_dir / "raw_metrics.csv", index=False)
    diagnostics.to_csv(raw_dir / "diagnostics.csv", index=False)
    paper_dir = _write_dgp_outputs(metrics, diagnostics, config, run_dir, args.results_dir)
    standard = metrics_frame_to_standard(
        metrics,
        experiment="DGP",
        regime="DGP",
        config=config,
        gate_diagnostics=pd.DataFrame(),
        notes="debug" if args.debug else "DGP",
    )
    write_standard_outputs(
        run_dir,
        experiment="DGP",
        config_path=args.config,
        standard_metrics=standard,
        diagnostics={
            "regimes": regimes,
            "diagnostic_rows": int(len(diagnostics)),
            "paper_results_dir": str(Path(paper_dir).resolve()),
        },
        warnings=["Allocation metrics are primary; RMSE is secondary under origin-scale shift."],
    )
    print(f"Run directory: {run_dir}")
    print(f"Paper results: {Path(paper_dir).resolve()}")


if __name__ == "__main__":
    main()
