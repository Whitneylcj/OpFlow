"""Diagnostic runner for Q2 opportunity redistribution severity behavior."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opflow_experiments.data.dgp import generate_dgp_data
from opflow_experiments.evaluation.evaluator import evaluate_predictions
from opflow_experiments.exposure_models import (
    _apply_component_standardizer,
    _fit_component_standardizer,
    _proxy_components_numpy,
    build_exposure_model,
)
from opflow_experiments.pipelines.run_dgp_main import (
    _apply_debug,
    _extra_metrics,
    _sort_records,
    load_exposure_config,
)
from opflow_experiments.utils.config import save_resolved_config
from opflow_experiments.utils.io import ensure_dir
from opflow_experiments.utils.paths import create_run_dir


DEFAULT_Q2_SEVERITIES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
DEFAULT_Q2_SEEDS = [0, 1, 2]
METRIC_COLUMNS = [
    "row_kl",
    "row_cpc",
    "pmse",
    "rank_corr",
    "topk_overlap",
    "log_rmse",
    "global_cpc",
    "origin_scale_error",
    "row_share_sum_error",
]


@dataclass(frozen=True)
class ProbeSpec:
    label: str
    method: str = "OpFlow"
    exposure_updates: dict[str, Any] | None = None
    dgp_updates: dict[str, Any] | None = None
    shifted_validation: bool = False


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def _with_scenario(config: dict[str, Any], severity: float) -> dict[str, Any]:
    out = _copy_config(config)
    out["scenario"] = {"question": "q2_opportunity", "opp_severity": float(severity)}
    return out


def _updated_config(
    config: dict[str, Any],
    *,
    exposure_updates: dict[str, Any] | None = None,
    dgp_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = _copy_config(config)
    if exposure_updates:
        out.setdefault("exposure_model", {}).update(exposure_updates)
    if dgp_updates:
        out.setdefault("dgp", {}).update(dgp_updates)
    return out


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _center_by_row(records: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float), index=records.index)
    grouped = series.groupby([records["env_id"], records["origin_id"]], sort=False)
    return (series - grouped.transform("mean")).to_numpy(dtype=float)


def _softmax_by_row(records: pd.DataFrame, score: np.ndarray, tau: float) -> np.ndarray:
    score = np.asarray(score, dtype=float).reshape(-1)
    out = np.zeros(len(records), dtype=float)
    for _, positions in records.groupby(["env_id", "origin_id"], sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        logits = score[idx] / max(float(tau), 1e-8)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        out[idx] = weights / max(float(weights.sum()), 1e-12)
    return out


def _standardize(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(stats["mean"])) / max(float(stats["std"]), 1e-8)


def _prediction_from_pi(records: pd.DataFrame, pi_hat: np.ndarray, score_hat: np.ndarray, tau: float) -> dict:
    pi_hat = np.asarray(pi_hat, dtype=float).reshape(-1)
    row_scale = records["R_oracle"].to_numpy(dtype=float)
    return {
        "lambda_hat": row_scale * pi_hat,
        "pi_hat": pi_hat,
        "z": None,
        "extra": {
            "score_hat": np.asarray(score_hat, dtype=float).reshape(-1),
            "score_kind": "choice_potential",
            "tau": float(tau),
            "row_scale_hat": row_scale,
        },
    }


def metric_oracle_prediction(records: pd.DataFrame, config: dict[str, Any]) -> dict:
    """Prediction that reads oracle allocation directly; should make Row-KL zero."""
    tau = float(config.get("dgp", {}).get("tau", config.get("tau", 0.7)))
    return _prediction_from_pi(
        records,
        records["pi_oracle"].to_numpy(dtype=float),
        records["psi_oracle"].to_numpy(dtype=float),
        tau,
    )


def true_exposure_oracle_prediction(records: pd.DataFrame, metadata: dict[str, Any], config: dict[str, Any]) -> dict:
    """Closed-form exposure law using true admissible DGP components, without reading oracle psi/pi."""
    standardizer = metadata["standardizer"]
    params = metadata.get("mechanism_params", config.get("dgp", {}))
    z_log_o = _standardize(np.log1p(records["o_true"].to_numpy(dtype=float)), standardizer["log_o"])
    z_log_s = _standardize(np.log1p(records["s_true"].to_numpy(dtype=float)), standardizer["log_s"])
    z_log_l = _standardize(np.log1p(records["ell_true"].to_numpy(dtype=float)), standardizer["log_l"])
    z_cost = _standardize(records["cost"].to_numpy(dtype=float), standardizer["cost"])
    g = (
        records["origin_beta_o"].to_numpy(dtype=float) * z_log_o
        - records["origin_beta_c"].to_numpy(dtype=float) * z_cost
        - float(params["beta_s"]) * z_log_s
        + float(params["beta_l"]) * z_log_l
    )
    psi = _center_by_row(records, g)
    tau = float(config.get("dgp", {}).get("tau", config.get("tau", 0.7)))
    pi = _softmax_by_row(records, psi, tau)
    return _prediction_from_pi(records, pi, psi, tau)


def proxy_oracle_prediction(
    records: pd.DataFrame,
    component_standardizer: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict:
    """Closed-form exposure law using deployed proxy-derived O/C/S/L components."""
    model_cfg = config.get("exposure_model", {})
    dgp_cfg = config.get("dgp", {})
    raw_components = _proxy_components_numpy(
        records,
        rank_prefix_mode=str(model_cfg.get("rank_prefix_mode", "soft")),
        rank_prefix_temperature=float(model_cfg.get("rank_prefix_temperature", 0.05)),
        omega=float(dgp_cfg.get("omega", 0.15)),
    )
    components = {
        key: _apply_component_standardizer(values, component_standardizer[key])
        for key, values in raw_components.items()
    }
    params = config.get("dgp", {})
    g = (
        records["origin_beta_o"].to_numpy(dtype=float) * components["O"]
        - records["origin_beta_c"].to_numpy(dtype=float) * components["C"]
        - float(params.get("beta_s", 0.65)) * components["S"]
        + float(params.get("beta_l", 0.45)) * components["L"]
    )
    psi = _center_by_row(records, g)
    tau = float(config.get("dgp", {}).get("tau", config.get("tau", 0.7)))
    pi = _softmax_by_row(records, psi, tau)
    return _prediction_from_pi(records, pi, psi, tau)


def _fit_proxy_component_standardizer(train_records: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    model_cfg = config.get("exposure_model", {})
    dgp_cfg = config.get("dgp", {})
    components = _proxy_components_numpy(
        train_records,
        rank_prefix_mode=str(model_cfg.get("rank_prefix_mode", "soft")),
        rank_prefix_temperature=float(model_cfg.get("rank_prefix_temperature", 0.05)),
        omega=float(dgp_cfg.get("omega", 0.15)),
    )
    return _fit_component_standardizer(components)


def _flatten_metrics(
    metrics: dict[str, dict[str, float]],
    *,
    seed: int,
    severity: float,
    model: str,
    family: str,
    extra: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_metrics in metrics.items():
        if split == "ood_id_gap":
            continue
        row = {
            "family": family,
            "model": model,
            "seed": int(seed),
            "severity": float(severity),
            "split": split,
        }
        for key in METRIC_COLUMNS:
            row[key] = split_metrics.get(key, float("nan"))
        if extra and split == "test":
            row.update(extra)
        rows.append(row)
    return rows


def _test_env_meta_summary(env_meta: pd.DataFrame, *, seed: int, severity: float) -> dict[str, Any]:
    test = env_meta.loc[env_meta["split"] == "test"].copy()
    fields = [
        "opportunity_std",
        "proxy_true_opp_corr",
        "proxy_log_opp_corr",
        "allocation_entropy_mean",
        "effective_destinations_mean",
        "zero_count_ratio",
        "empirical_corr_log_o_accessibility",
    ]
    row = {"seed": int(seed), "severity": float(severity)}
    for field in fields:
        row[field] = float(test[field].mean()) if field in test else float("nan")
    return row


def _component_values(records: pd.DataFrame, config: dict[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    model_cfg = config.get("exposure_model", {})
    dgp_cfg = config.get("dgp", {})
    proxy_components = _proxy_components_numpy(
        records,
        rank_prefix_mode=str(model_cfg.get("rank_prefix_mode", "soft")),
        rank_prefix_temperature=float(model_cfg.get("rank_prefix_temperature", 0.05)),
        omega=float(dgp_cfg.get("omega", 0.15)),
    )
    return {
        ("true", "O"): np.log1p(records["o_true"].to_numpy(dtype=float)),
        ("true", "S"): np.log1p(records["s_true"].to_numpy(dtype=float)),
        ("true", "L"): np.log1p(records["ell_true"].to_numpy(dtype=float)),
        ("proxy", "O"): proxy_components["O"],
        ("proxy", "S"): proxy_components["S"],
        ("proxy", "L"): proxy_components["L"],
    }


def coverage_diagnostics(
    train_records: pd.DataFrame,
    test_records: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    severity: float,
) -> list[dict[str, Any]]:
    train_values = _component_values(train_records, config)
    test_values = _component_values(test_records, config)
    rows = []
    for key, train_arr in train_values.items():
        source, component = key
        test_arr = np.asarray(test_values[key], dtype=float)
        train_arr = np.asarray(train_arr, dtype=float)
        q01, q99 = np.quantile(train_arr, [0.01, 0.99])
        train_std = float(train_arr.std(ddof=0))
        rows.append(
            {
                "seed": int(seed),
                "severity": float(severity),
                "source": source,
                "component": component,
                "train_q01": float(q01),
                "train_q99": float(q99),
                "test_mean": float(test_arr.mean()),
                "train_mean": float(train_arr.mean()),
                "outside_train_1_99": float(((test_arr < q01) | (test_arr > q99)).mean()),
                "zmean_shift": float((test_arr.mean() - train_arr.mean()) / max(train_std, 1e-12)),
            }
        )
    return rows


def oracle_ladder_for_seed(
    config: dict[str, Any],
    *,
    seed: int,
    severities: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_data = generate_dgp_data(_with_scenario(config, 0.0), seed=seed)
    base_records = _sort_records(base_data.records)
    train_records = base_records.loc[base_records["split"] == "train"].copy()
    proxy_standardizer = _fit_proxy_component_standardizer(train_records, config)
    metric_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    topk = int(config.get("baseline", {}).get("topk", 5))
    for severity in severities:
        data = generate_dgp_data(_with_scenario(config, severity), seed=seed)
        records = _sort_records(data.records)
        test_records = records.loc[records["split"] == "test"].copy()
        shape_rows.append(_test_env_meta_summary(data.env_meta, seed=seed, severity=severity))
        coverage_rows.extend(coverage_diagnostics(train_records, test_records, config, seed=seed, severity=severity))
        predictions = {
            "MetricOracle": metric_oracle_prediction(records, config),
            "TrueExposureOracle": true_exposure_oracle_prediction(records, data.metadata, config),
            "ProxyOracle": proxy_oracle_prediction(records, proxy_standardizer, config),
        }
        for model, prediction in predictions.items():
            metrics = evaluate_predictions(records, prediction, topk=topk)
            metric_rows.extend(
                _flatten_metrics(metrics, seed=seed, severity=severity, model=model, family="oracle_ladder")
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(shape_rows), pd.DataFrame(coverage_rows)


def _probe_specs(profile: str) -> list[ProbeSpec]:
    specs = [
        ProbeSpec("CurrentOpFlow"),
        ProbeSpec("AllocSelectedOpFlow", method="OpFlowAllocSelected"),
        ProbeSpec(
            "ShiftedValOpFlow",
            exposure_updates={"selection_protocol": "validation", "checkpoint_metric": "joint_selection_score"},
            shifted_validation=True,
        ),
    ]
    if profile in {"standard", "full"}:
        specs.extend(
            [
                ProbeSpec("NoOppRecon", exposure_updates={"use_learned_opportunity_reconstruction": False}),
                ProbeSpec("ReconDelta0.0", exposure_updates={"opp_recon_delta_scale": 0.0}),
                ProbeSpec("ReconDelta0.5", exposure_updates={"opp_recon_delta_scale": 0.5}),
                ProbeSpec("ProxyCorr0.99", dgp_updates={"proxy_target_corr": 0.99}),
            ]
        )
    if profile == "full":
        specs.extend(
            [
                ProbeSpec("ReconDelta0.1", exposure_updates={"opp_recon_delta_scale": 0.1}),
                ProbeSpec("NoReconReg", exposure_updates={"lambda_anchor": 0.0, "lambda_smooth": 0.0}),
                ProbeSpec("StrongerReconReg", exposure_updates={"lambda_anchor": 0.05, "lambda_smooth": 0.005}),
                ProbeSpec("ProxyCorr1.0", dgp_updates={"proxy_target_corr": 1.0}),
            ]
        )
    return specs


def _fit_probe_model(
    config: dict[str, Any],
    spec: ProbeSpec,
    *,
    seed: int,
    shifted_val_severity: float,
):
    probe_config = _updated_config(config, exposure_updates=spec.exposure_updates, dgp_updates=spec.dgp_updates)
    base_data = generate_dgp_data(_with_scenario(probe_config, 0.0), seed=seed)
    base_records = _sort_records(base_data.records)
    train = base_records.loc[base_records["split"] == "train"].copy()
    if spec.shifted_validation:
        shifted_data = generate_dgp_data(_with_scenario(probe_config, shifted_val_severity), seed=seed)
        shifted_records = _sort_records(shifted_data.records)
        val = shifted_records.loc[shifted_records["split"] == "test"].copy()
        val["split"] = "val"
    else:
        val = base_records.loc[base_records["split"] == "val"].copy()
    model = build_exposure_model(spec.method, probe_config, seed)
    model.fit(train, val)
    return model, probe_config


def model_probes_for_seed(
    config: dict[str, Any],
    *,
    seed: int,
    severities: list[float],
    specs: list[ProbeSpec],
    shifted_val_severity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    topk = int(config.get("baseline", {}).get("topk", 5))
    for spec in specs:
        model, probe_config = _fit_probe_model(
            config,
            spec,
            seed=seed,
            shifted_val_severity=shifted_val_severity,
        )
        summary = dict(getattr(model, "selection_summary_", {}) or {})
        selection_rows.append(
            {
                "model": spec.label,
                "method": spec.method,
                "seed": int(seed),
                "shifted_validation": bool(spec.shifted_validation),
                "shifted_val_severity": float(shifted_val_severity) if spec.shifted_validation else float("nan"),
                **{
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in summary.items()
                },
            }
        )
        for severity in severities:
            data = generate_dgp_data(_with_scenario(probe_config, severity), seed=seed)
            records = _sort_records(data.records)
            pred = model.predict(records)
            metrics = evaluate_predictions(records, pred, topk=topk)
            extra = _extra_metrics(model, records, pred)
            metric_rows.extend(
                _flatten_metrics(
                    metrics,
                    seed=seed,
                    severity=severity,
                    model=spec.label,
                    family="model_probe",
                    extra=extra,
                )
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows)


def _plot_line(
    frame: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    metric: str = "row_kl",
    split: str = "test",
    max_models: int | None = None,
) -> None:
    plot = frame.loc[frame["split"] == split].copy()
    if plot.empty:
        return
    models = plot["model"].drop_duplicates().tolist()
    if max_models is not None:
        models = models[:max_models]
        plot = plot.loc[plot["model"].isin(models)]
    summary = (
        plot.groupby(["severity", "model"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model, group in summary.groupby("model", sort=False):
        err = 1.96 * group["std"].fillna(0.0) / np.sqrt(group["count"].clip(lower=1))
        ax.errorbar(group["severity"], group["mean"], yerr=err, marker="o", linewidth=1.8, capsize=3, label=model)
    ax.set_title(title)
    ax.set_xlabel("Opportunity severity")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_severity_shape(frame: pd.DataFrame, out_path: Path) -> None:
    if frame.empty:
        return
    metrics = [
        ("opportunity_std", "Opportunity std"),
        ("allocation_entropy_mean", "Allocation entropy"),
        ("effective_destinations_mean", "Effective destinations"),
        ("zero_count_ratio", "Zero-count ratio"),
    ]
    summary = frame.groupby("severity", as_index=False).agg({metric: "mean" for metric, _ in metrics})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        ax.plot(summary["severity"], summary[metric], marker="o", linewidth=1.8)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Opportunity severity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_coverage(frame: pd.DataFrame, out_path: Path) -> None:
    if frame.empty:
        return
    plot = frame.copy()
    plot["series"] = plot["source"] + " " + plot["component"]
    summary = plot.groupby(["severity", "series"], as_index=False)["outside_train_1_99"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for series, group in summary.groupby("series", sort=False):
        ax.plot(group["severity"], group["outside_train_1_99"], marker="o", linewidth=1.6, label=series)
    ax.set_title("Q2 exposure coverage outside train 1-99% range")
    ax.set_xlabel("Opportunity severity")
    ax.set_ylabel("Outside fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _delta(frame: pd.DataFrame, model: str, low: float, high: float) -> float:
    test = frame.loc[(frame["split"] == "test") & (frame["model"] == model)]
    means = test.groupby("severity")["row_kl"].mean()
    if low not in means.index or high not in means.index:
        return float("nan")
    return float(means.loc[high] - means.loc[low])


def _write_report(
    out_path: Path,
    *,
    severities: list[float],
    oracle_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    shape: pd.DataFrame,
) -> None:
    low = min(severities)
    high = max(severities)
    metric_max = float(
        oracle_metrics.loc[
            (oracle_metrics["split"] == "test") & (oracle_metrics["model"] == "MetricOracle"),
            "row_kl",
        ].max()
    )
    true_delta = _delta(oracle_metrics, "TrueExposureOracle", low, high)
    proxy_delta = _delta(oracle_metrics, "ProxyOracle", low, high)
    current_delta = _delta(model_metrics, "CurrentOpFlow", low, high) if not model_metrics.empty else float("nan")
    lines = [
        "# Q2 Row-KL Severity Diagnostics",
        "",
        "This report is diagnostic-only. Oracle probes are not deployed baselines.",
        "",
        "## Decision Checks",
        f"- MetricOracle max test Row-KL: {metric_max:.6g}.",
        f"- TrueExposureOracle test Row-KL delta ({low:g}->{high:g}): {true_delta:.6g}.",
        f"- ProxyOracle test Row-KL delta ({low:g}->{high:g}): {proxy_delta:.6g}.",
        f"- CurrentOpFlow test Row-KL delta ({low:g}->{high:g}): {current_delta:.6g}.",
        "",
        "## Severity Shape",
    ]
    if not shape.empty:
        shape_summary = shape.groupby("severity", as_index=False).mean(numeric_only=True)
        cols = [
            "severity",
            "opportunity_std",
            "allocation_entropy_mean",
            "effective_destinations_mean",
            "zero_count_ratio",
        ]
        lines.append("```text")
        lines.append(shape_summary[cols].round(4).to_string(index=False))
        lines.append("```")
    if not model_metrics.empty:
        lines.extend(["", "## Model Probe Summary"])
        test = model_metrics.loc[model_metrics["split"] == "test"].copy()
        high = max(severities)
        low = min(severities)
        high_summary = (
            test.loc[np.isclose(test["severity"], high)]
            .groupby("model", as_index=False)
            .agg(
                row_kl=("row_kl", "mean"),
                row_kl_std=("row_kl", "std"),
                pmse=("pmse", "mean"),
                exposure_alignment=("exposure_alignment", "mean"),
                learned_opp_log_corr=("learned_opp_log_corr", "mean"),
            )
        )
        deltas = []
        for model in high_summary["model"]:
            deltas.append(_delta(model_metrics, str(model), low, high))
        high_summary["row_kl_delta"] = deltas
        high_summary = high_summary.sort_values("row_kl")
        lines.append(f"High-severity test summary at severity={high:g}:")
        lines.append("```text")
        lines.append(high_summary.round(4).to_string(index=False))
        lines.append("```")
    if not selection.empty:
        keep = [
            "model",
            "seed",
            "selection_protocol",
            "selection_metric",
            "selected_epoch",
            "shifted_validation",
            "shifted_val_severity",
        ]
        keep = [col for col in keep if col in selection.columns]
        lines.extend(["", "## Selection Summary", "```text"])
        lines.append(selection[keep].to_string(index=False))
        lines.append("```")
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "- If MetricOracle is not near zero, inspect metric masking and evaluation.",
            "- If TrueExposureOracle is flat but ProxyOracle is not, proxy/exposure observability is the likely bottleneck.",
            "- If TrueExposureOracle is not flat, the DGP or the expected flatness claim needs review.",
            "- If shifted-val probes improve CurrentOpFlow, Q2 selection coverage is likely mismatched.",
            "- If reconstruction probes improve CurrentOpFlow, opportunity reconstruction is the likely implementation bottleneck.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_q2_diagnostics(
    config: dict[str, Any],
    *,
    seeds: list[int],
    severities: list[float],
    run_dir: Path,
    figures_dir: Path,
    profile: str,
) -> dict[str, Path]:
    diagnostic_dir = ensure_dir(run_dir / "diagnostics" / "q2_debug")
    figures_dir = ensure_dir(figures_dir)
    oracle_frames: list[pd.DataFrame] = []
    shape_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for seed in seeds:
        oracle, shape, coverage = oracle_ladder_for_seed(config, seed=seed, severities=severities)
        oracle_frames.append(oracle)
        shape_frames.append(shape)
        coverage_frames.append(coverage)
    oracle_metrics = pd.concat(oracle_frames, ignore_index=True) if oracle_frames else pd.DataFrame()
    severity_shape = pd.concat(shape_frames, ignore_index=True) if shape_frames else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()

    model_metrics = pd.DataFrame()
    selection = pd.DataFrame()
    if profile != "oracle":
        specs = _probe_specs(profile)
        metric_frames: list[pd.DataFrame] = []
        selection_frames: list[pd.DataFrame] = []
        shifted_val_severity = max(severities)
        for seed in seeds:
            metrics, selection_rows = model_probes_for_seed(
                config,
                seed=seed,
                severities=severities,
                specs=specs,
                shifted_val_severity=shifted_val_severity,
            )
            metric_frames.append(metrics)
            selection_frames.append(selection_rows)
        model_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
        selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()

    paths = {
        "oracle_metrics": diagnostic_dir / "oracle_ladder_metrics.csv",
        "severity_shape": diagnostic_dir / "severity_shape.csv",
        "coverage": diagnostic_dir / "coverage_diagnostics.csv",
        "model_metrics": diagnostic_dir / "model_probe_metrics.csv",
        "selection": diagnostic_dir / "selection_diagnostics.csv",
        "report": diagnostic_dir / "q2_debug_report.md",
        "oracle_figure": figures_dir / "q2_oracle_ladder_row_kl.png",
        "shape_figure": figures_dir / "q2_severity_shape.png",
        "coverage_figure": figures_dir / "q2_coverage.png",
        "model_figure": figures_dir / "q2_model_probes_row_kl.png",
    }
    oracle_metrics.to_csv(paths["oracle_metrics"], index=False)
    severity_shape.to_csv(paths["severity_shape"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    selection.to_csv(paths["selection"], index=False)
    _plot_line(oracle_metrics, out_path=paths["oracle_figure"], title="Q2 oracle ladder Row-KL")
    _plot_severity_shape(severity_shape, paths["shape_figure"])
    _plot_coverage(coverage, paths["coverage_figure"])
    if not model_metrics.empty:
        _plot_line(model_metrics, out_path=paths["model_figure"], title="Q2 model probe Row-KL", max_models=12)
    else:
        paths.pop("model_figure")
    _write_report(
        paths["report"],
        severities=severities,
        oracle_metrics=oracle_metrics,
        model_metrics=model_metrics,
        selection=selection,
        shape=severity_shape,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q2 diagnostic oracle and model probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_Q2_SEEDS))
    parser.add_argument("--severities", type=float, nargs="*", default=list(DEFAULT_Q2_SEVERITIES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["debug", "full"], default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--profile", choices=["oracle", "selection", "standard", "full"], default="standard")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_exposure_config(args.config, args.override)
    if args.mode == "debug":
        config = _apply_debug(config)
    run_id = args.run_id or f"q2_debug_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = create_run_dir(args.output_dir, run_id)
    figures_dir = ensure_dir(Path(args.figures_dir) / run_id)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    paths = run_q2_diagnostics(
        config,
        seeds=[int(seed) for seed in args.seeds],
        severities=[float(value) for value in args.severities],
        run_dir=run_dir,
        figures_dir=figures_dir,
        profile=str(args.profile),
    )
    print("Q2 diagnostics written:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
