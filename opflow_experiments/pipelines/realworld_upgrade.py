"""Paper-aligned real-world upgrade outputs for LargeCommutingOD."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UPGRADE_SPLITS = ["random", "opportunity_shift", "shortcut_shift", "scale_shift"]
UPGRADE_MODELS = [
    "distance_only",
    "gravity",
    "intervening_opportunity",
    "pair_mlp",
    "scale_aware_mlp",
    "opflow_full",
    "opflow_no_rank_exposure",
    "opflow_no_competition",
    "opflow_no_opportunity_field",
    "opflow_no_scale_detach",
    "opflow_generic_potential_mlp",
    "opflow_no_groupdro",
    "opflow_no_vrex",
    "opflow_no_monotone_penalty",
]

FEATURE_PARTITION = {
    "deployed_geographic_opportunity_variables": [
        "node_feat destination opportunity proxy from deployment-available node attributes",
        "node_feat origin context attributes",
        "pairwise distance / impedance",
        "rank-prefix exposure computed from learned opportunity and learned impedance",
        "destination local competition exposure computed from learned opportunity and spatial kernels",
        "adjacency only as deployment-available geography when explicitly used by non-full variants",
    ],
    "shortcut_prone_variables": [
        "OD labels y_raw",
        "row-normalized labels p_share",
        "origin outflow labels except as supervised scale target",
        "destination inflow and label-derived topology summaries",
        "distance-flow correlation split scores",
        "destination allocation entropy split descriptors",
        "environment IDs, split labels, and area IDs",
        "historical co-flow or target-derived graph artifacts",
    ],
    "allowed_deployed_models": ["opflow_full", "opflow_no_scale_detach", "opflow_generic_potential_mlp"],
    "operator_ablation_models": ["opflow_no_groupdro", "opflow_no_vrex", "opflow_no_monotone_penalty"],
    "notes": [
        "Full OpFlow receives only node features, learned impedance, rank-prefix exposure, local competition, and geographic modifiers.",
        "Label-derived split diagnostics are saved for evaluation but are not passed as model features.",
        "The current LargeCommutingOD checkout uses node-feature opportunity proxies, not external semantic opportunity covariates.",
    ],
}


def apply_realworld_upgrade_debug(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved["splits"] = list(resolved.get("splits", UPGRADE_SPLITS))
    resolved["models"] = list(resolved.get("models", UPGRADE_MODELS))
    resolved["seeds"] = list(resolved.get("seeds", [resolved.get("seed", 2026)]))[:1]
    training = dict(resolved.get("training", {}))
    training["epochs"] = min(int(training.get("epochs", 2)), 2)
    training["hidden_dim"] = min(int(training.get("hidden_dim", 8)), 8)
    training["patience"] = min(int(training.get("patience", 2)), 2)
    training["row_chunk_size"] = min(int(training.get("row_chunk_size", 8)), 8)
    training["rank_prefix_row_chunk_size"] = min(int(training.get("rank_prefix_row_chunk_size", 32)), 32)
    training["process_workers"] = 1
    training["torch_num_threads_per_worker"] = 1
    resolved["training"] = training
    limits = dict(resolved.get("area_limits", {}))
    limits["max_regions"] = min(int(limits.get("max_regions") or 28), 28)
    limits["max_train_areas"] = min(int(limits.get("max_train_areas") or 3), 3)
    limits["max_val_areas"] = min(int(limits.get("max_val_areas") or 1), 1)
    limits["max_test_areas"] = min(int(limits.get("max_test_areas") or 2), 2)
    resolved["area_limits"] = limits
    if resolved.get("stage", "train") == "all":
        resolved["stage"] = "train"
    return resolved


def is_realworld_upgrade(config: dict[str, Any]) -> bool:
    return str(config.get("suite", "")).lower() == "realworld_upgrade"


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(val) for val in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary_mean_ci(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_ci95"] = float("nan")
            else:
                mean = float(values.mean())
                ci = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def _split_diagnostics(config: dict[str, Any]) -> pd.DataFrame:
    data_root = Path(config.get("dataset", {}).get("data_root", "data/opflow_large_commuting_od"))
    balance_path = data_root / "split_balance.json"
    if not balance_path.exists():
        return pd.DataFrame()
    frame = pd.read_json(balance_path)
    reverse = {
        "matched_opportunity_low_to_high": "opportunity_shift",
        "matched_shortcut_high_to_low": "shortcut_shift",
        "matched_scale_low_to_high": "scale_shift",
    }
    frame["public_split_name"] = frame["split_name"].map(reverse).fillna(frame["split_name"])
    keep = set(config.get("splits", UPGRADE_SPLITS))
    frame = frame.loc[frame["public_split_name"].isin(keep)].copy()
    frame["opportunity_descriptor"] = "node_feature_proxy_and_empirical_allocation_entropy"
    frame["claim_boundary"] = np.where(
        frame["public_split_name"].eq("opportunity_shift"),
        "empirical opportunity-structure proxy shift, not semantic opportunity recovery",
        "benchmark split diagnostic",
    )
    return frame


def _scale_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    test = results.loc[results["part"] == "test"].copy()
    learned = test.loc[test["variant"].str.endswith("learned_scale", na=False)].copy()
    oracle = test.loc[test["variant"].str.endswith("oracle_scale", na=False)].copy()
    oracle = oracle.rename(
        columns={
            "global_cpc": "oracle_global_cpc",
            "log_rmse": "oracle_log_rmse",
            "row_cpc": "oracle_row_cpc",
            "row_kl": "oracle_row_kl",
        }
    )
    merged = learned.merge(
        oracle[
            [
                "split_name",
                "model",
                "seed",
                "oracle_global_cpc",
                "oracle_log_rmse",
                "oracle_row_cpc",
                "oracle_row_kl",
            ]
        ],
        on=["split_name", "model", "seed"],
        how="left",
    )
    return merged.rename(columns={"global_cpc": "learned_global_cpc", "log_rmse": "learned_log_rmse"})[
        [
            "split_name",
            "model",
            "seed",
            "total_flow_ratio",
            "outflow_log_rmse",
            "outflow_corr",
            "learned_global_cpc",
            "oracle_global_cpc",
            "learned_log_rmse",
            "oracle_log_rmse",
            "oracle_row_cpc",
            "oracle_row_kl",
        ]
    ]


def _operator_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    test = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    rows = []
    for row in test.itertuples(index=False):
        rows.append(
            {
                "split_name": row.split_name,
                "model": row.model,
                "seed": row.seed,
                "alloc_risk_max": getattr(row, "alloc_risk_max", np.nan),
                "alloc_risk_mean": getattr(row, "alloc_risk_mean", np.nan),
                "vrex_alloc_var": getattr(row, "vrex_alloc_var", np.nan),
                "scale_risk_max": getattr(row, "scale_risk_max", np.nan),
                "scale_risk_mean": getattr(row, "scale_risk_mean", np.nan),
                "op_monotone_violation_D": getattr(row, "op_monotone_violation_D", np.nan),
                "op_monotone_violation_S": getattr(row, "op_monotone_violation_S", np.nan),
                "op_monotone_violation_B": getattr(row, "op_monotone_violation_B", np.nan),
                "operator_mean_o_tilde": getattr(row, "operator_mean_o_tilde", np.nan),
                "operator_mean_c_tilde": getattr(row, "operator_mean_c_tilde", np.nan),
                "psi_bar_row_mean_abs": getattr(row, "psi_bar_row_mean_abs", np.nan),
                "row_kl": getattr(row, "row_kl", np.nan),
                "row_cpc": getattr(row, "row_cpc", np.nan),
                "rank_corr": getattr(row, "rank_corr", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _write_figures(out_dir: Path, results: pd.DataFrame, split_diag: pd.DataFrame, scale_diag: pd.DataFrame, operator_diag: pd.DataFrame) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not split_diag.empty:
        cols = [
            "mean_feature_opportunity_entropy_norm",
            "mean_dest_entropy_norm",
            "mean_distance_flow_corr",
            "mean_log_total_flow",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col in zip(axes.reshape(-1), cols):
            if col not in split_diag:
                ax.text(0.5, 0.5, "not available", ha="center", va="center")
                ax.set_title(col)
                continue
            for part, color in (("train", "#4C78A8"), ("test", "#F58518")):
                sub = split_diag.loc[split_diag["part"].eq(part)]
                ax.bar(sub["public_split_name"] + ":" + part, sub[col], color=color, alpha=0.7)
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures / "fig1_split_diagnostics.png", dpi=170)
        plt.close(fig)

    primary = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    if not primary.empty:
        primary["row_cpc"] = pd.to_numeric(primary["row_cpc"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot = primary.pivot_table(index="model", columns="split_name", values="row_cpc", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Row-CPC")
        ax.set_title("Allocation performance by split")
        fig.tight_layout()
        fig.savefig(figures / "fig2_opportunity_response_curve.png", dpi=170)
        plt.close(fig)

    if not operator_diag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        frame = operator_diag.loc[operator_diag["model"].isin(["opflow_full", "opflow_no_groupdro", "opflow_no_vrex", "opflow_no_monotone_penalty"])].copy()
        if frame.empty:
            frame = operator_diag.copy()
        frame["alloc_risk_max"] = pd.to_numeric(frame.get("alloc_risk_max"), errors="coerce")
        frame.groupby(["split_name", "model"])["alloc_risk_max"].mean().unstack("model").plot(kind="bar", ax=ax)
        ax.set_ylabel("allocation max risk")
        ax.set_title("Operator allocation-risk diagnostics")
        fig.tight_layout()
        fig.savefig(figures / "fig3_operator_diagnostics.png", dpi=170)
        plt.close(fig)

    if not scale_diag.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        frame = scale_diag.copy()
        frame["oracle_gain_cpc"] = pd.to_numeric(frame["oracle_global_cpc"], errors="coerce") - pd.to_numeric(
            frame["learned_global_cpc"], errors="coerce"
        )
        frame.groupby(["split_name", "model"])["oracle_gain_cpc"].mean().unstack("model").plot(kind="bar", ax=ax)
        ax.set_ylabel("Oracle-scale Global-CPC gain")
        ax.set_title("Scale decomposition")
        fig.tight_layout()
        fig.savefig(figures / "fig4_scale_decomposition.png", dpi=170)
        plt.close(fig)


def _write_opportunity_response_curve(config: dict[str, Any], run_dir: Path, out_dir: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/opflow_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow_cache")
    import matplotlib.pyplot as plt
    import torch

    from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline
    from opflow_experiments.pipelines import real_world_large_commuting_od as rw

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    splits = list(config.get("splits", UPGRADE_SPLITS))
    seeds = list(config.get("seeds", [config.get("seed", 2026)]))
    seed = int(seeds[0]) if seeds else int(config.get("seed", 2026))
    model_name = "opflow_full" if "opflow_full" in config.get("models", []) else "opflow_real"
    rows = []
    max_areas = int(config.get("reporting", {}).get("response_max_areas_per_part", 4))
    max_points = int(config.get("reporting", {}).get("response_max_points_per_split", 20000))
    for split_name in splits:
        ckpt_path = run_dir / "artifacts" / "real_world_models" / split_name / model_name / f"seed_{seed}" / "checkpoint.pt"
        if not ckpt_path.exists():
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = OpFlowRealBaseline(device="cpu", model_variant=checkpoint.get("model_variant", "opflow_full"))
        model.load_checkpoint(checkpoint)
        split_areas = rw._load_split_areas(config, split_name)
        for part in ("train", "test"):
            points = []
            for area in split_areas.get(part, [])[:max_areas]:
                with torch.no_grad():
                    parts = model._loss_area_parts(area, collect_outputs=True)
                psi = parts["psi_bar"].detach().cpu().numpy()
                p_true = area["p_share"].detach().cpu().numpy()
                valid = area["valid_origin"].detach().cpu().numpy().astype(bool)
                empirical = np.log(np.clip(p_true, 1e-8, None))
                empirical = empirical - empirical.mean(axis=1, keepdims=True)
                mask = np.broadcast_to(valid[:, None], p_true.shape)
                pair = pd.DataFrame({"predicted_potential": psi[mask], "empirical_potential": empirical[mask]})
                points.append(pair)
            if not points:
                continue
            frame = pd.concat(points, ignore_index=True)
            if len(frame) > max_points:
                frame = frame.sample(max_points, random_state=2026)
            frame["bin"] = pd.qcut(frame["predicted_potential"].rank(method="first"), q=min(12, len(frame)), labels=False, duplicates="drop")
            curve = frame.groupby("bin", as_index=False).agg(
                predicted_potential=("predicted_potential", "mean"),
                empirical_potential=("empirical_potential", "mean"),
            )
            curve["split_name"] = split_name
            curve["part"] = part
            rows.append(curve)
    if not rows:
        return
    curves = pd.concat(rows, ignore_index=True)
    curves.to_csv(out_dir / "opportunity_response_curve.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, split_name in zip(axes.reshape(-1), splits):
        sub = curves.loc[curves["split_name"].eq(split_name)]
        if sub.empty:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.set_title(split_name)
            continue
        for part, color in (("train", "#4C78A8"), ("test", "#F58518")):
            part_frame = sub.loc[sub["part"].eq(part)].sort_values("predicted_potential")
            ax.plot(part_frame["predicted_potential"], part_frame["empirical_potential"], marker="o", label=part, color=color)
        ax.set_title(split_name)
        ax.set_xlabel("binned predicted exposure potential")
        ax.set_ylabel("empirical row-centered log-share")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig2_opportunity_response_curve.png", dpi=170)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "NA"
    display = frame.head(max_rows).copy().fillna("NA")
    headers = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.itertuples(index=False):
        vals = []
        for value in row:
            vals.append(f"{value:.5g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    out_dir: Path,
    config: dict[str, Any],
    summary: pd.DataFrame,
    split_diag: pd.DataFrame,
    scale_diag: pd.DataFrame,
    operator_diag: pd.DataFrame,
) -> None:
    metric_cols = [col for col in summary.columns if col in {"split_name", "model", "row_kl_mean", "row_cpc_mean", "rank_corr_mean", "ndcg@5_mean", "global_cpc_mean"}]
    scale_cols = [col for col in scale_diag.columns if col in {"split_name", "model", "seed", "total_flow_ratio", "outflow_log_rmse", "learned_global_cpc", "oracle_global_cpc"}]
    lines = [
        "# Real-World OpFlow Upgrade Report",
        "",
        "## Dataset and Task Summary",
        "",
        f"- Dataset: `{config.get('dataset', {}).get('name', 'large_commuting_od')}` at `{config.get('dataset', {}).get('data_root', 'data/opflow_large_commuting_od')}`.",
        "- Task: cross-area OD flow prediction with allocation-first evaluation under random, opportunity-structure, shortcut, and scale shifts.",
        "",
        "## Feature Partition",
        "",
        "- Full OpFlow uses node-feature opportunity proxies, learned impedance, rank-prefix exposure, local competition, and geographic modifiers.",
        "- Shortcut-prone label-derived descriptors are used only for split diagnostics.",
        "- This run does not claim semantic opportunity recovery because no external semantic opportunity covariates are assumed.",
        "",
        "## Split Construction and Diagnostics",
        "",
        _markdown_table(split_diag[[col for col in ["public_split_name", "part", "n_areas", "mean_n_regions", "mean_flow_density", "mean_distance_flow_corr", "mean_dest_entropy_norm", "mean_log_total_flow"] if col in split_diag]]),
        "",
        "## Main Results by Split",
        "",
        _markdown_table(summary[metric_cols] if metric_cols else summary),
        "",
        "## Ablations",
        "",
        "Ablation rows are included in `metrics_by_seed.csv` and `summary_mean_ci.csv` using `opflow_no_*` and `opflow_generic_potential_mlp` model names.",
        "",
        "## Opportunity-Response Diagnostics",
        "",
        "Figure `figures/fig2_opportunity_response_curve.png` summarizes allocation behavior by shift split. Interpret it as opportunity-structure proxy evidence, not proof of semantic opportunity identification.",
        "",
        "## Scale Decomposition",
        "",
        _markdown_table(scale_diag[scale_cols] if scale_cols else scale_diag),
        "",
        "## Operator Diagnostics",
        "",
        _markdown_table(operator_diag),
        "",
        "## Paper-Ready Conclusions",
        "",
        "- Report allocation metrics first; raw-count metrics mix scale and allocation.",
        "- Treat OpportunityShift as an empirical opportunity-structure proxy shift unless external covariates are added.",
        "- Use ShortcutShift with operator ablations to test shortcut-failure claims.",
        "",
        "## Known Limitations",
        "",
        "- Node-feature opportunity proxies are not semantic opportunity labels.",
        "- Real-world splits are observational diagnostics and do not provide causal identification.",
        "- Full paper-facing evidence requires the uncapped three-seed command.",
    ]
    (out_dir / "realworld_upgrade_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_realworld_upgrade_outputs(config: dict[str, Any], run_dir: Path) -> Path:
    from scripts.make_tables import collect_results

    run_name = run_dir.name
    out_dir = Path(config.get("reporting", {}).get("upgrade_output_root", "results/realworld_upgrade")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_root = run_dir / "artifacts" / "real_world_models"
    results = collect_results(results_root)
    results.to_csv(out_dir / "metrics_by_seed.csv", index=False)

    primary = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    metrics = [
        "row_kl",
        "row_cpc",
        "rank_corr",
        "topk@5",
        "ndcg@5",
        "log_rmse",
        "global_cpc",
        "total_flow_ratio",
        "outflow_log_rmse",
    ]
    summary_split = _summary_mean_ci(primary, ["split_name", "model"], [col for col in metrics if col in primary])
    summary_split.to_csv(out_dir / "summary_by_split.csv", index=False)
    summary_mean = _summary_mean_ci(primary, ["model"], [col for col in metrics if col in primary])
    summary_mean.to_csv(out_dir / "summary_mean_ci.csv", index=False)

    split_diag = _split_diagnostics(config)
    split_diag.to_csv(out_dir / "split_diagnostics.csv", index=False)
    scale_diag = _scale_diagnostics(results) if not results.empty else pd.DataFrame()
    scale_diag.to_csv(out_dir / "scale_diagnostics.csv", index=False)
    operator_diag = _operator_diagnostics(results) if not results.empty else pd.DataFrame()
    operator_diag.to_csv(out_dir / "operator_diagnostics.csv", index=False)

    (out_dir / "feature_partition.json").write_text(json.dumps(FEATURE_PARTITION, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(_safe_json(config), indent=2, sort_keys=True), encoding="utf-8")
    _write_figures(out_dir, results, split_diag, scale_diag, operator_diag)
    error_path = out_dir / "opportunity_response_curve.error.txt"
    if error_path.exists():
        error_path.unlink()
    try:
        _write_opportunity_response_curve(config, run_dir, out_dir)
    except Exception as exc:  # pragma: no cover - best-effort diagnostic figure.
        (out_dir / "opportunity_response_curve.error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    _write_report(out_dir, config, summary_split, split_diag, scale_diag, operator_diag)

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "realworld_upgrade_report.md", reports_dir / "realworld_upgrade_report.md")
    return out_dir
