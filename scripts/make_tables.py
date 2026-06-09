"""Create paper-facing CSV and LaTeX tables from benchmark results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-opflow")

import pandas as pd


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return value.get("mean") if isinstance(value, dict) else None


def collect_results(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.rglob("metrics.json")):
        payload = _read(path)
        config = payload.get("config", {})
        split_name = config.get("split_name", path.parents[2].name)
        model = config.get("model", path.parents[1].name)
        seed = config.get("seed", path.parent.name.replace("seed_", ""))
        primary = payload.get("primary_variant", f"{model}_learned_scale")
        gaps = payload.get("ood_gap", {})
        for variant, split_metrics in payload.get("metrics", {}).items():
            for part, summary in split_metrics.items():
                row = {
                    "split_name": split_name,
                    "model": model,
                    "variant": variant,
                    "is_primary": variant == primary,
                    "seed": seed,
                    "part": part,
                    "row_kl": _mean(summary, "row_kl"),
                    "row_cpc": _mean(summary, "row_cpc"),
                    "rank_corr": _mean(summary, "rank_corr"),
                    "topk@1": _mean(summary, "topk@1"),
                    "topk@3": _mean(summary, "topk@3"),
                    "topk@5": _mean(summary, "topk@5"),
                    "topk@5_coverage": _mean(summary, "topk@5_coverage"),
                    "topk@10": _mean(summary, "topk@10"),
                    "topk@10_coverage": _mean(summary, "topk@10_coverage"),
                    "ndcg@5": _mean(summary, "ndcg@5"),
                    "ndcg@10": _mean(summary, "ndcg@10"),
                    "rmse": _mean(summary, "rmse"),
                    "log_rmse": _mean(summary, "log_rmse"),
                    "nrmse_mean": _mean(summary, "nrmse_mean"),
                    "global_cpc": _mean(summary, "global_cpc"),
                    "total_flow_ratio": _mean(summary, "total_flow_ratio"),
                    "outflow_log_rmse": _mean(summary, "outflow_log_rmse"),
                    "outflow_corr": _mean(summary, "outflow_corr"),
                    "true_outflow_mean": _mean(summary, "true_outflow_mean"),
                    "true_outflow_std": _mean(summary, "true_outflow_std"),
                    "pred_outflow_mean": _mean(summary, "pred_outflow_mean"),
                    "pred_outflow_std": _mean(summary, "pred_outflow_std"),
                    "mean_true_total_flow": _mean(summary, "true_total_flow"),
                    "mean_pred_total_flow": _mean(summary, "pred_total_flow"),
                    "median_true_total_flow": _mean(summary, "median_true_outflow"),
                    "median_pred_total_flow": _mean(summary, "median_pred_outflow"),
                    "gate_distance_corr": _mean(summary, "gate_distance_corr"),
                    "gate_opportunity_corr": _mean(summary, "gate_opportunity_corr"),
                    "gate_mean": _mean(summary, "gate_mean"),
                    "distance_beta": _mean(summary, "distance_beta"),
                    "diagnostic_gap": _mean(summary, "diagnostic_gap"),
                    "diagnostic_alloc_kl": _mean(summary, "diagnostic_alloc_kl"),
                    "diagnostic_challenger_kl": _mean(summary, "diagnostic_challenger_kl"),
                    "train_mean_outflow": _mean(summary, "train_mean_outflow"),
                    "log1p_outflow_mean": _mean(summary, "log1p_outflow_mean"),
                    "log1p_outflow_std": _mean(summary, "log1p_outflow_std"),
                    "local_exposure_mode_kernel_sum": _mean(summary, "local_exposure_mode_kernel_sum"),
                }
                row["distance_sensitivity"] = row["distance_beta"] if row["distance_beta"] is not None else row["gate_distance_corr"]
                row["ood_gap_row_kl"] = gaps.get(variant, {}).get("row_kl")
                row["ood_gap_row_cpc"] = gaps.get(variant, {}).get("row_cpc")
                rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(data_root: Path) -> pd.DataFrame:
    meta = _read(data_root / "meta.json")
    areas = pd.DataFrame(meta["areas"])
    return pd.DataFrame(
        [
            {
                "subset": "county",
                "n_areas": len(areas),
                "min_regions": areas["n_regions"].min(),
                "median_regions": areas["n_regions"].median(),
                "max_regions": areas["n_regions"].max(),
                "median_total_flow": areas["total_flow"].median(),
                "max_total_flow": areas["total_flow"].max(),
                "has_distance": bool(areas["has_distance"].all()),
                "has_adjacency": bool(areas["has_adjacency"].all()),
                "has_node_features": bool(areas["has_node_feat"].all()),
            }
        ]
    )


def _write(frame: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, float_format="%.4g"), encoding="utf-8")


def _main_results(results: pd.DataFrame, split_name: str) -> pd.DataFrame:
    frame = results.loc[(results["split_name"] == split_name) & (results["part"] == "test") & (results["is_primary"])].copy()
    cols = ["model", "row_kl", "row_cpc", "rank_corr", "topk@5", "topk@5_coverage", "log_rmse", "global_cpc"]
    return frame.loc[:, cols].sort_values("model")


def _shift_results(results: pd.DataFrame, split_name: str) -> pd.DataFrame:
    frame = results.loc[(results["split_name"] == split_name) & (results["part"] == "test") & (results["is_primary"])].copy()
    cols = ["model", "row_kl", "row_cpc", "rank_corr", "topk@5", "ood_gap_row_kl", "ood_gap_row_cpc", "distance_sensitivity"]
    return frame.loc[:, [col for col in cols if col in frame]].sort_values("model")


def _scale(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.loc[results["part"] == "test"].copy()
    learned = frame.loc[frame["variant"].str.endswith("learned_scale", na=False)].copy()
    oracle = frame.loc[frame["variant"].str.endswith("oracle_scale", na=False), ["split_name", "model", "seed", "global_cpc"]].rename(
        columns={"global_cpc": "oracle_scale_global_cpc"}
    )
    merged = learned.merge(oracle, on=["split_name", "model", "seed"], how="left")
    merged = merged.rename(columns={"global_cpc": "learned_scale_global_cpc"})
    cols = [
        "model",
        "split_name",
        "variant",
        "total_flow_ratio",
        "outflow_log_rmse",
        "outflow_corr",
        "learned_scale_global_cpc",
        "oracle_scale_global_cpc",
    ]
    return merged.loc[:, cols].sort_values(["split_name", "model"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/opflow_large_commuting_od")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--tables_dir", default="tables")
    args = parser.parse_args()
    tables = Path(args.tables_dir)
    results = collect_results(Path(args.results_root))
    _write(dataset_summary(Path(args.data_root)), tables / "dataset_summary.csv", tables / "dataset_summary_latex.tex")
    balance = pd.read_json(Path(args.data_root) / "split_balance.json")
    _write(balance, tables / "split_balance_table.csv", tables / "split_balance_table_latex.tex")
    _write(_main_results(results, "random"), tables / "random_main_results.csv", tables / "random_main_results_latex.tex")
    _write(_shift_results(results, "matched_shortcut_high_to_low"), tables / "matched_shortcut_results.csv", tables / "matched_shortcut_results_latex.tex")
    _write(_shift_results(results, "matched_opportunity_low_to_high"), tables / "matched_opportunity_results.csv", tables / "matched_opportunity_results_latex.tex")
    _write(_scale(results), tables / "scale_diagnostics.csv", tables / "scale_diagnostics_latex.tex")
    results.to_csv(tables / "all_result_records.csv", index=False)


if __name__ == "__main__":
    main()
