"""Standard output schema for synthetic experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from opflow_experiments.utils.io import ensure_dir


STANDARD_METRIC_COLUMNS = [
    "experiment",
    "regime",
    "method",
    "seed",
    "split",
    "n_units",
    "noise_level",
    "shift_strength",
    "shortcut_strength",
    "graph_noise",
    "ood_rmse",
    "id_rmse",
    "mae",
    "log_rmse",
    "global_cpc",
    "row_kl",
    "row_cpc",
    "rank_corr",
    "topk_overlap",
    "score_mse",
    "pmse",
    "score_corr",
    "scale_rmse",
    "scale_mae",
    "scale_corr",
    "scale_hat_mean",
    "scale_true_mean",
    "oie",
    "share_drift",
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
    "shortcut_sensitivity",
    "distance_sensitivity",
    "gate_opp_alignment",
    "gate_distance_corr",
    "gate_source_corr",
    "origin_scale_error",
    "row_share_sum_error",
    "graph_gain",
    "notes",
]


def standard_context(config: dict[str, Any], regime: str) -> dict[str, Any]:
    """Build static context columns for the standard metrics schema."""
    shift_strength: float | str = "NA"
    if regime == "DGP":
        shift_strength = "environment_opportunity"
    return {
        "n_units": int(config.get("dgp", {}).get("N_o", config.get("latent", {}).get("N_o", config.get("N_o", 0)))),
        "noise_level": config.get("dgp", {}).get("sigma_proxy", config.get("latent", {}).get("feature_noise", config.get("sigma_q", "NA"))),
        "shift_strength": shift_strength,
        "shortcut_strength": "shortcut_proxy" if regime == "DGP" else "NA",
        "graph_noise": config.get("latent", {}).get("graph_noise", config.get("graph_noise", "NA")),
    }


def metrics_frame_to_standard(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    regime: str,
    config: dict[str, Any],
    gate_diagnostics: pd.DataFrame | None = None,
    notes: str = "",
) -> pd.DataFrame:
    """Map existing per-seed metric rows into the standard synthetic schema."""
    if metrics.empty:
        return pd.DataFrame(columns=STANDARD_METRIC_COLUMNS)
    frame = metrics.copy()
    if "method" not in frame and "model" in frame:
        frame["method"] = frame["model"]
    if "rank_corr" not in frame and "rankcorr" in frame:
        frame["rank_corr"] = frame["rankcorr"]
    if "graph_gain" not in frame:
        graph_gain_lookup = {}
        baseline_rows = frame.loc[frame["method"].isin(["raw_count_mlp", "all_feature_mlp"])].copy()
        if not baseline_rows.empty and "row_kl" in baseline_rows:
            for base_row in baseline_rows.itertuples(index=False):
                graph_gain_lookup[(getattr(base_row, "regime", regime), base_row.seed, base_row.split)] = base_row.row_kl
        if graph_gain_lookup:
            frame["graph_gain"] = [
                graph_gain_lookup.get((getattr(row, "regime", regime), row.seed, row.split), float("nan")) - row.row_kl
                if row.method not in {"raw_count_mlp", "all_feature_mlp"}
                and pd.notna(graph_gain_lookup.get((getattr(row, "regime", regime), row.seed, row.split), float("nan")))
                else 0.0
                for row in frame.itertuples(index=False)
            ]

    id_lookup = {}
    ood_lookup = {}
    for row in frame.itertuples(index=False):
        row_regime = getattr(row, "regime", regime)
        method = getattr(row, "method", "NA")
        seed = getattr(row, "seed", "NA")
        split = getattr(row, "split", "NA")
        rmse = getattr(row, "rmse", "NA")
        if split == "train":
            id_lookup[(row_regime, method, seed)] = rmse
        if split == "test":
            ood_lookup[(row_regime, method, seed)] = rmse

    gate_lookup: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    if gate_diagnostics is not None and not gate_diagnostics.empty:
        gate_frame = gate_diagnostics.copy()
        if "method" not in gate_frame and "model" in gate_frame:
            gate_frame["method"] = gate_frame["model"]
        for gate_row in gate_frame.itertuples(index=False):
            gate_lookup[(getattr(gate_row, "regime", regime), gate_row.method, gate_row.seed, gate_row.split)] = {
                "gate_opp_alignment": getattr(gate_row, "corr_gate_true_score", "NA"),
                "gate_distance_corr": getattr(gate_row, "corr_gate_shortcut_score", "NA"),
                "gate_source_corr": getattr(gate_row, "corr_gate_source_scale", getattr(gate_row, "corr_gate_history", "NA")),
            }

    rows = []
    for row in frame.itertuples(index=False):
        method = getattr(row, "method", "NA")
        seed = getattr(row, "seed", "NA")
        split = getattr(row, "split", "NA")
        row_regime = getattr(row, "regime", regime)
        context = standard_context(config, row_regime)
        gate_values = gate_lookup.get((row_regime, method, seed, split), {})
        rows.append(
            {
                "experiment": getattr(row, "experiment", experiment),
                "regime": row_regime,
                "method": method,
                "seed": seed,
                "split": split,
                **context,
                "ood_rmse": ood_lookup.get((row_regime, method, seed), "NA"),
                "id_rmse": id_lookup.get((row_regime, method, seed), "NA"),
                "mae": getattr(row, "mae", "NA"),
                "log_rmse": getattr(row, "log_rmse", "NA"),
                "global_cpc": getattr(row, "global_cpc", "NA"),
                "row_kl": getattr(row, "row_kl", "NA"),
                "row_cpc": getattr(row, "row_cpc", "NA"),
                "rank_corr": getattr(row, "rank_corr", "NA"),
                "topk_overlap": getattr(row, "topk_overlap", "NA"),
                "score_mse": getattr(row, "score_mse", "NA"),
                "pmse": getattr(row, "pmse", "NA"),
                "score_corr": getattr(row, "score_corr", "NA"),
                "scale_rmse": getattr(row, "scale_rmse", "NA"),
                "scale_mae": getattr(row, "scale_mae", "NA"),
                "scale_corr": getattr(row, "scale_corr", "NA"),
                "scale_hat_mean": getattr(row, "scale_hat_mean", "NA"),
                "scale_true_mean": getattr(row, "scale_true_mean", "NA"),
                "oie": getattr(row, "oie", "NA"),
                "share_drift": getattr(row, "share_drift", "NA"),
                "alloc_risk_max": getattr(row, "alloc_risk_max", "NA"),
                "alloc_risk_mean": getattr(row, "alloc_risk_mean", "NA"),
                "vrex_alloc_var": getattr(row, "vrex_alloc_var", "NA"),
                "scale_risk_max": getattr(row, "scale_risk_max", "NA"),
                "scale_risk_mean": getattr(row, "scale_risk_mean", "NA"),
                "eta_residual": getattr(row, "eta_residual", "NA"),
                "op_monotone_violation_D": getattr(row, "op_monotone_violation_D", "NA"),
                "op_monotone_violation_S": getattr(row, "op_monotone_violation_S", "NA"),
                "op_monotone_violation_B": getattr(row, "op_monotone_violation_B", "NA"),
                "operator_mean_o_tilde": getattr(row, "operator_mean_o_tilde", "NA"),
                "operator_mean_c_tilde": getattr(row, "operator_mean_c_tilde", "NA"),
                "psi_bar_row_mean_abs": getattr(row, "psi_bar_row_mean_abs", "NA"),
                "shortcut_sensitivity": getattr(row, "shortcut_sensitivity", getattr(row, "rankcorr_ood_drop", "NA")),
                "distance_sensitivity": getattr(row, "distance_sensitivity", "NA"),
                "gate_opp_alignment": gate_values.get("gate_opp_alignment", "NA"),
                "gate_distance_corr": gate_values.get("gate_distance_corr", "NA"),
                "gate_source_corr": gate_values.get("gate_source_corr", "NA"),
                "origin_scale_error": getattr(row, "origin_scale_error", "NA"),
                "row_share_sum_error": getattr(row, "row_share_sum_error", "NA"),
                "graph_gain": getattr(row, "graph_gain", "NA"),
                "notes": getattr(row, "notes", notes) or notes,
            }
        )
    return pd.DataFrame(rows, columns=STANDARD_METRIC_COLUMNS)


def aggregate_summary(standard_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate test split metrics by method for summary display."""
    if standard_metrics.empty:
        return pd.DataFrame()
    test = standard_metrics.loc[standard_metrics["split"] == "test"].copy()
    numeric_cols = [
        "ood_rmse",
        "mae",
        "row_kl",
        "row_cpc",
        "log_rmse",
        "global_cpc",
        "rank_corr",
        "topk_overlap",
        "score_mse",
        "pmse",
        "score_corr",
        "scale_rmse",
        "scale_mae",
        "scale_corr",
        "scale_hat_mean",
        "scale_true_mean",
        "oie",
        "share_drift",
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
        "shortcut_sensitivity",
        "distance_sensitivity",
        "gate_opp_alignment",
        "gate_distance_corr",
        "gate_source_corr",
        "origin_scale_error",
        "row_share_sum_error",
        "graph_gain",
    ]
    for col in numeric_cols:
        test[col] = pd.to_numeric(test[col], errors="coerce")
    return test.groupby("method", as_index=False)[numeric_cols].mean(numeric_only=True)


def build_summary_payload(
    standard_metrics: pd.DataFrame,
    *,
    experiment: str,
    config_path: str | None,
    run_id: str,
    diagnostics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a stable summary.json payload."""
    summary = aggregate_summary(standard_metrics)
    best_methods: dict[str, str] = {}
    if not summary.empty:
        for metric, direction in (("row_kl", "min"), ("row_cpc", "max"), ("ood_rmse", "min"), ("mae", "min"), ("pmse", "min")):
            valid = summary.dropna(subset=[metric])
            if valid.empty:
                continue
            idx = valid[metric].idxmin() if direction == "min" else valid[metric].idxmax()
            best_methods[metric] = str(valid.loc[idx, "method"])
    main_metrics = summary.fillna("NA").to_dict(orient="records") if not summary.empty else []
    payload = {
        "experiment": experiment,
        "config_path": config_path,
        "run_id": run_id,
        "status": "success",
        "main_metrics": main_metrics,
        "best_methods": best_methods,
        "diagnostics": diagnostics or {},
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def write_standard_outputs(
    run_dir: str | Path,
    *,
    experiment: str,
    config_path: str | None,
    standard_metrics: pd.DataFrame,
    diagnostics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Write metrics.csv, summary.json, and summary.md to the run root."""
    run_path = ensure_dir(run_dir)
    run_id = run_path.name
    metrics_out = standard_metrics.copy()
    for column in STANDARD_METRIC_COLUMNS:
        if column not in metrics_out:
            metrics_out[column] = "NA"
    metrics_out = metrics_out.loc[:, STANDARD_METRIC_COLUMNS].fillna("NA")
    metrics_out.to_csv(run_path / "metrics.csv", index=False)

    payload = build_summary_payload(
        metrics_out,
        experiment=experiment,
        config_path=config_path,
        run_id=run_id,
        diagnostics=diagnostics,
        warnings=warnings,
    )
    with (run_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    _write_summary_markdown(payload, run_path / "summary.md")
    return payload


def _write_summary_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {payload['experiment']}",
        "",
        f"Run ID: `{payload['run_id']}`",
        f"Status: `{payload['status']}`",
        "",
        "## Best Methods",
        "",
    ]
    if payload["best_methods"]:
        for metric, method in payload["best_methods"].items():
            lines.append(f"- `{metric}`: `{method}`")
    else:
        lines.append("- NA")
    lines.extend(["", "## Main Metrics", ""])
    main_metrics = payload["main_metrics"]
    if main_metrics:
        keys = ["method", "row_kl", "row_cpc", "ood_rmse", "mae", "pmse"]
        lines.append("| method | row_kl | row_cpc | rmse | mae | pmse |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in main_metrics:
            values = [row.get(key, "NA") for key in keys]
            lines.append(
                f"| {values[0]} | {_fmt(values[1])} | {_fmt(values[2])} | {_fmt(values[3])} | {_fmt(values[4])} | {_fmt(values[5])} |"
            )
    else:
        lines.append("NA")
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
