"""Constrained spatial-interaction real-world V2 outputs."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opflow_experiments.pipelines.realworld_upgrade import _operator_diagnostics, _safe_json, _scale_diagnostics, _summary_mean_ci


CONSTRAINED_SI_SPLITS = ["RandSP", "GeoSP", "OppSP"]
CONSTRAINED_SI_MAIN_MODELS = [
    "gravity",
    "radiation",
    "scale_aware_mlp",
    "pair_mlp_sampled",
    "opflow_full",
    "opflow_full_softmax_local",
    "opflow_no_scale_branch",
    "opflow_no_rank_exposure",
    "opflow_no_competition",
    "opflow_generic_potential_mlp",
    "opflow_no_groupdro",
    "opflow_no_vrex",
    "opflow_no_monotone_penalty",
]
CONSTRAINED_SI_APPENDIX_MODELS = [
    "intervening_opportunity",
    "raw_count_mlp",
    "opflow_no_exposure_ops",
    "opflow_no_row_centering",
    "opflow_lite",
]


def is_constrained_si(config: dict[str, Any]) -> bool:
    return str(config.get("suite", "")).lower() == "constrained_si"


def apply_constrained_si_debug(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    requested = list(resolved.get("splits", CONSTRAINED_SI_SPLITS))
    resolved["splits"] = ["RandSP_seed0"] if "RandSP" in requested else requested[:1]
    debug_models = list(resolved.get("models", CONSTRAINED_SI_MAIN_MODELS))
    for model in ("intervening_opportunity", "raw_count_mlp"):
        if model not in debug_models:
            debug_models.append(model)
    resolved["models"] = debug_models
    resolved["seeds"] = list(resolved.get("seeds", [resolved.get("seed", 2026)]))[:1]
    training = dict(resolved.get("training", {}))
    training["epochs"] = min(int(training.get("epochs", 2)), 2)
    training["hidden_dim"] = min(int(training.get("hidden_dim", 8)), 8)
    training["patience"] = min(int(training.get("patience", 2)), 2)
    training["leo_max_epochs"] = min(int(training.get("leo_max_epochs", training["epochs"])), training["epochs"])
    training["leo_patience"] = min(int(training.get("leo_patience", 1)), 1)
    training["leo_max_folds"] = min(int(training.get("leo_max_folds", 2)), 2)
    training["row_chunk_size"] = min(int(training.get("row_chunk_size", 8)), 8)
    training["rank_prefix_row_chunk_size"] = min(int(training.get("rank_prefix_row_chunk_size", 32)), 32)
    training["max_pair_mlp_train_areas"] = min(int(training.get("max_pair_mlp_train_areas") or 3), 3)
    training["max_raw_count_train_areas"] = min(int(training.get("max_raw_count_train_areas") or 3), 3)
    training["process_workers"] = 1
    training["torch_num_threads_per_worker"] = 1
    resolved["training"] = training
    reporting = dict(resolved.get("reporting", {}))
    reporting["exposure_signature_max_areas"] = min(int(reporting.get("exposure_signature_max_areas") or 8), 8)
    reporting["exposure_full_arrays_max_areas"] = min(int(reporting.get("exposure_full_arrays_max_areas") or 2), 2)
    resolved["reporting"] = reporting
    limits = dict(resolved.get("area_limits", {}))
    limits["max_regions"] = min(int(limits.get("max_regions") or 28), 28)
    limits["max_train_areas"] = min(int(limits.get("max_train_areas") or 3), 3)
    limits["max_val_areas"] = min(int(limits.get("max_val_areas") or 1), 1)
    limits["max_test_areas"] = min(int(limits.get("max_test_areas") or 2), 2)
    resolved["area_limits"] = limits
    if resolved.get("stage", "train") == "all":
        resolved["stage"] = "train"
    return resolved


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "NA"
    display = frame.head(max_rows).copy().fillna("NA")
    headers = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.itertuples(index=False):
        values = []
        for value in row:
            values.append(f"{value:.5g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_latex(index=False, float_format="%.5g"), encoding="utf-8")


def _metric_summary(results: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    primary = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    if models is not None:
        primary = primary.loc[primary["model"].isin(models)].copy()
    metrics = ["row_kl", "rank_corr", "topk@5", "ndcg@5", "rmse", "log_rmse", "global_cpc", "outflow_log_rmse"]
    return _summary_mean_ci(primary, ["split_name", "model"], [col for col in metrics if col in primary])


def _scale_diagnostics_v2(results: pd.DataFrame) -> pd.DataFrame:
    test = results.loc[results["part"] == "test"].copy()
    learned = test.loc[test["variant"].str.endswith("learned_scale", na=False)].copy()
    oracle = test.loc[test["variant"].str.endswith("oracle_scale", na=False)].copy()
    mean = test.loc[test["variant"].str.endswith("mean_scale", na=False)].copy()
    oracle = oracle.rename(columns={"global_cpc": "oracle_global_cpc", "log_rmse": "oracle_log_rmse", "row_cpc": "oracle_row_cpc", "row_kl": "oracle_row_kl"})
    mean = mean.rename(columns={"global_cpc": "mean_scale_global_cpc", "log_rmse": "mean_scale_log_rmse", "outflow_log_rmse": "mean_scale_outflow_log_rmse"})
    merged = learned.merge(
        oracle[["split_name", "model", "seed", "oracle_global_cpc", "oracle_log_rmse", "oracle_row_cpc", "oracle_row_kl"]],
        on=["split_name", "model", "seed"],
        how="left",
    )
    merged = merged.merge(
        mean[["split_name", "model", "seed", "mean_scale_global_cpc", "mean_scale_log_rmse", "mean_scale_outflow_log_rmse"]],
        on=["split_name", "model", "seed"],
        how="left",
    )
    keep = [
        "split_name",
        "model",
        "seed",
        "true_outflow_mean",
        "true_outflow_std",
        "pred_outflow_mean",
        "pred_outflow_std",
        "total_flow_ratio",
        "outflow_log_rmse",
        "outflow_corr",
        "global_cpc",
        "oracle_global_cpc",
        "mean_scale_global_cpc",
        "log_rmse",
        "oracle_log_rmse",
        "mean_scale_log_rmse",
        "mean_scale_outflow_log_rmse",
        "oracle_row_cpc",
        "oracle_row_kl",
    ]
    merged = merged.rename(columns={"global_cpc": "learned_scale_global_cpc", "log_rmse": "learned_scale_log_rmse"})
    keep = [col.replace("global_cpc", "learned_scale_global_cpc") if col == "global_cpc" else col for col in keep]
    keep = [col.replace("log_rmse", "learned_scale_log_rmse") if col == "log_rmse" else col for col in keep]
    return merged[[col for col in keep if col in merged]]


def _write_section(
    out_dir: Path,
    name: str,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    note: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "raw_per_seed.csv", index=False)
    summary.to_csv(out_dir / "aggregated.csv", index=False)
    _write_latex(summary, out_dir / "table.tex")
    report = ["# " + title, "", note, "", "## Aggregated Results", "", _markdown_table(summary)]
    (out_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _temporal_infeasible(out_dir: Path, config: dict[str, Any]) -> None:
    lines = [
        "# Temporal OOD Split Infeasible",
        "",
        "The current LargeCommutingOD checkout is a static cross-area OD benchmark.",
        "The metadata contains county/area fields but no date, period, environment, or chronological identifier.",
        "Therefore this run does not construct a pseudo-temporal split and does not claim temporal OOD evidence.",
        "",
        "RandSP is a random county-level reference; GeoSP holds out complete Census divisions; OppSP holds out opportunity-concentration tails.",
        "Model inputs must not use labels, split labels, fold IDs, area IDs, or environment IDs.",
        "",
        f"Dataset root: `{config.get('dataset', {}).get('data_root', 'data/opflow_large_commuting_od')}`",
    ]
    (out_dir / "temporal_ood_infeasible.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gini(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if values.size == 0 or total <= 1e-8:
        return float("nan")
    sorted_values = np.sort(values)
    n = sorted_values.size
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(weights * sorted_values) / (n * total)) - ((n + 1.0) / n))


def _entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 1e-8:
        return float("nan")
    p = values / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-8)))


def _write_exposure_signatures(out_dir: Path, config: dict[str, Any]) -> None:
    import torch

    from opflow_benchmark.dataset import AreaODDataset
    from opflow_benchmark.exposure import exogenous_opportunity_proxy, local_destination_exposure, rank_prefix_exposure

    data_root = Path(config.get("dataset", {}).get("data_root", "data/opflow_large_commuting_od"))
    dataset = AreaODDataset(data_root)
    max_saved = int(config.get("reporting", {}).get("exposure_full_arrays_max_areas", 3))
    max_summary = config.get("reporting", {}).get("exposure_signature_max_areas")
    max_summary = None if max_summary in (None, "") else int(max_summary)
    omega = float(config.get("training", {}).get("omega_grid", [25000.0])[0])
    rows = []
    saved = 0
    arrays_dir = out_dir / "exposure_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    n_areas = len(dataset) if max_summary is None else min(len(dataset), max_summary)
    for idx in range(n_areas):
        area = dataset[idx]
        x = torch.log1p(torch.clamp(area["node_feat"].float(), min=0.0))
        opp = exogenous_opportunity_proxy(x)
        prefix = rank_prefix_exposure(opp, area["distance"].float(), row_chunk_size=config.get("training", {}).get("rank_prefix_row_chunk_size", 128))
        local = local_destination_exposure(
            opp,
            area["distance"].float(),
            omega,
            mode=str(config.get("training", {}).get("local_exposure_mode", "kernel_sum")),
        )
        opp_np = opp.detach().cpu().numpy()
        prefix_np = prefix.detach().cpu().numpy()
        local_np = local.detach().cpu().numpy()
        rows.append(
            {
                "area_id": area["area_id"],
                "n_regions": int(area["y_raw"].shape[0]),
                "opportunity_mean": float(np.mean(opp_np)),
                "opportunity_entropy": _entropy(opp_np),
                "opportunity_gini": _gini(opp_np),
                "prefix_mean": float(np.mean(prefix_np)),
                "prefix_std": float(np.std(prefix_np)),
                "competition_mean": float(np.mean(local_np)),
                "competition_std": float(np.std(local_np)),
            }
        )
        if saved < max_saved:
            np.savez_compressed(
                arrays_dir / f"{area['area_id']}_exposure_signature.npz",
                opportunity=opp_np,
                rank_prefix=prefix_np,
                local_competition=local_np,
            )
            saved += 1
    pd.DataFrame(rows).to_csv(out_dir / "exposure_signature_summary.csv", index=False)


def _write_plots(out_root: Path, summary: pd.DataFrame, scale: pd.DataFrame, diag: pd.DataFrame) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
    import matplotlib.pyplot as plt

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        frame = summary.copy()
        frame["row_kl_mean"] = pd.to_numeric(frame.get("row_kl_mean"), errors="coerce")
        pivot = frame.pivot_table(index="model", columns="split_name", values="row_kl_mean", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Row-KL")
        ax.set_title("Main OOD allocation error")
        fig.tight_layout()
        fig.savefig(out_root / "main_ood" / "main_ood_row_kl.png", dpi=170)
        plt.close(fig)
    if not scale.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        frame = scale.copy()
        frame["oracle_gain_cpc"] = pd.to_numeric(frame["oracle_global_cpc"], errors="coerce") - pd.to_numeric(
            frame["learned_scale_global_cpc"], errors="coerce"
        )
        frame["mean_gain_cpc"] = pd.to_numeric(frame["mean_scale_global_cpc"], errors="coerce") - pd.to_numeric(
            frame["learned_scale_global_cpc"], errors="coerce"
        )
        frame.groupby(["split_name", "model"])["mean_gain_cpc"].mean().unstack("model").plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean-scale Global-CPC gain")
        ax.set_title("Scale-allocation decomposition")
        fig.tight_layout()
        fig.savefig(out_root / "scale_allocation" / "scale_decomposition.png", dpi=170)
        plt.close(fig)
    if not diag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        frame = diag.loc[diag["model"].isin(["opflow_full", "opflow_no_groupdro", "opflow_no_vrex", "opflow_no_monotone_penalty"])].copy()
        frame["alloc_risk_max"] = pd.to_numeric(frame.get("alloc_risk_max"), errors="coerce")
        frame.groupby(["split_name", "model"])["alloc_risk_max"].mean().unstack("model").plot(kind="bar", ax=ax)
        ax.set_ylabel("Allocation max risk")
        ax.set_title("Operator diagnostics")
        fig.tight_layout()
        fig.savefig(out_root / "operator_diagnostics" / "alloc_risk_max.png", dpi=170)
        plt.close(fig)


def _write_case_study(out_dir: Path, results: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    keep = candidates.loc[candidates["model"].isin(["opflow_full", "pair_mlp_sampled", "opflow_generic_potential_mlp"])]
    keep.to_csv(out_dir / "case_study_candidate_metrics.csv", index=False)
    if not keep.empty:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
        import matplotlib.pyplot as plt

        frame = keep.copy()
        frame["row_kl"] = pd.to_numeric(frame["row_kl"], errors="coerce")
        plot_frame = frame.groupby(["split_name", "model"])["row_kl"].mean().unstack("model")
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_frame.plot(kind="bar", ax=ax)
        ax.set_ylabel("Row-KL")
        ax.set_title("Case-study candidate allocation error")
        fig.tight_layout()
        fig.savefig(out_dir / "case_study_candidate_row_kl.png", dpi=170)
        plt.close(fig)
    (out_dir / "summary.md").write_text(
        "# Case Study Candidates\n\n"
        "This directory records OOD test rows suitable for qualitative visualization. "
        "Full top-destination plots should be generated from selected checkpoints after the main V2 run completes.\n",
        encoding="utf-8",
    )


def _mirror_section_surfaces(out_root: Path, base_root: Path, run_name: str) -> None:
    for section in ("main_ood", "scale_allocation", "exposure_ablation", "operator_diagnostics", "case_study"):
        src = out_root / section
        if not src.exists():
            continue
        dst = base_root / section / run_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (base_root / section / "latest_run.txt").write_text(run_name + "\n", encoding="utf-8")


def write_constrained_si_outputs(config: dict[str, Any], run_dir: Path) -> Path:
    from scripts.make_tables import collect_results

    run_name = run_dir.name
    base_root = Path(config.get("reporting", {}).get("constrained_si_output_root", "results/real_world"))
    out_root = base_root / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    results = collect_results(run_dir / "artifacts" / "real_world_models")
    results.to_csv(out_root / "all_metrics_by_seed.csv", index=False)
    (out_root / "config.json").write_text(json.dumps(_safe_json(config), indent=2, sort_keys=True), encoding="utf-8")
    _temporal_infeasible(out_root, config)
    _write_exposure_signatures(out_root, config)

    main_summary = _metric_summary(results, CONSTRAINED_SI_MAIN_MODELS)
    appendix_summary = _metric_summary(results, CONSTRAINED_SI_APPENDIX_MODELS)
    primary = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    _write_section(
        out_root / "main_ood",
        "main_ood",
        primary.loc[primary["model"].isin(CONSTRAINED_SI_MAIN_MODELS)].copy(),
        main_summary,
        "Main OOD Performance",
        "Allocation metrics are primary; count RMSE is auxiliary because counts mix production scale and allocation.",
    )
    scale = _scale_diagnostics_v2(results) if not results.empty else pd.DataFrame()
    _write_section(
        out_root / "scale_allocation",
        "scale_allocation",
        scale,
        _summary_mean_ci(
            scale,
            ["split_name", "model"],
            [
                col
                for col in [
                    "true_outflow_mean",
                    "pred_outflow_mean",
                    "total_flow_ratio",
                    "outflow_log_rmse",
                    "learned_scale_global_cpc",
                    "oracle_global_cpc",
                    "mean_scale_global_cpc",
                ]
                if col in scale
            ],
        )
        if not scale.empty
        else pd.DataFrame(),
        "Scale-Allocation Decomposition",
        "This section separates origin-production prediction from destination-allocation quality.",
    )
    exposure_models = ["opflow_full", "opflow_full_softmax_local", "opflow_no_rank_exposure", "opflow_no_competition", "opflow_no_exposure_ops"]
    exposure_raw = primary.loc[primary["model"].isin(exposure_models)].copy()
    _write_section(
        out_root / "exposure_ablation",
        "exposure_ablation",
        exposure_raw,
        _metric_summary(results, exposure_models),
        "Exposure Operator Ablation",
        "Compares full exposure signature against removals of rank-prefix exposure and local destination competition.",
    )
    diag = _operator_diagnostics(results) if not results.empty else pd.DataFrame()
    _write_section(
        out_root / "operator_diagnostics",
        "operator_diagnostics",
        diag,
        _summary_mean_ci(
            diag,
            ["split_name", "model"],
            [
                col
                for col in [
                    "alloc_risk_max",
                    "alloc_risk_mean",
                    "vrex_alloc_var",
                    "op_monotone_violation_D",
                    "op_monotone_violation_S",
                    "op_monotone_violation_B",
                    "psi_bar_row_mean_abs",
                ]
                if col in diag
            ],
        )
        if not diag.empty
        else pd.DataFrame(),
        "Operator Diagnostics",
        "Summarizes GroupDRO/V-REx allocation risks and additive-operator monotonicity diagnostics.",
    )
    _write_case_study(out_root / "case_study", results)
    _write_plots(out_root, main_summary, scale, diag)

    report_lines = [
        "# Real-World Constrained Spatial-Interaction Report",
        "",
        "OpFlow is reported here as a neural generalization of constrained spatial interaction models under an entropy-regularized choice-potential principle.",
        "",
        "The current LargeCommutingOD data are static cross-area OD snapshots. RandSP/GeoSP/OppSP are spatial or opportunity-structure environment splits, not temporal OOD evidence.",
        "",
        "Opportunity features are deployment-observable attraction proxies, not identified semantic opportunity covariates. Operator diagnostics are descriptive, not causal identification.",
        "",
        "## Main OOD Table",
        "",
        _markdown_table(main_summary),
        "",
        "## Appendix / Extra Variants",
        "",
        _markdown_table(appendix_summary),
        "",
        "## Required Artifacts",
        "",
        "- `temporal_ood_infeasible.md` documents why true temporal OOD is unavailable.",
        "- `exposure_signature_summary.csv` documents deployment-observable attraction proxies and derived exposure signatures.",
        "- Section directories contain raw per-seed CSVs, aggregated CSVs, LaTeX tables, Markdown summaries, and plots where applicable.",
    ]
    (out_root / "realworld_constrained_si_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _mirror_section_surfaces(out_root, base_root, run_name)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_root / "realworld_constrained_si_report.md", reports_dir / "realworld_constrained_si_report.md")
    return out_root
