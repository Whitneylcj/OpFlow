"""Create benchmark diagnostic figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-opflow")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_tables import collect_results


def _split_distribution(records: pd.DataFrame, splits: dict, split_name: str, cols: list[str], out: Path) -> None:
    rows = []
    for part in ("train", "test"):
        ids = set(splits[split_name][part])
        sub = records.loc[records["area_id"].isin(ids)]
        for col in cols:
            for value in sub[col].dropna():
                rows.append({"part": part, "metric": col, "value": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No split distribution rows found for {split_name}; check area_id types")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.reshape(-1)
    for ax, col in zip(axes, cols):
        for part, color in (("train", "#4C78A8"), ("test", "#F58518")):
            values = frame.loc[(frame["part"] == part) & (frame["metric"] == col), "value"]
            ax.hist(values, bins=20, alpha=0.55, label=part, color=color)
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _bar_results(results: pd.DataFrame, split_name: str, metric: str, out: Path) -> None:
    frame = results.loc[(results["split_name"] == split_name) & (results["part"] == "test") & (results["is_primary"])].copy()
    frame = frame.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(frame["model"], frame[metric], color="#4C78A8")
    ax.set_ylabel(metric)
    ax.set_title(split_name)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _scale(results: pd.DataFrame, out: Path) -> None:
    frame = results.loc[(results["part"] == "test") & (results["variant"].str.endswith("learned_scale", na=False))].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = frame["split_name"] + "\n" + frame["model"]
    ax.bar(range(len(frame)), frame["total_flow_ratio"], color="#54A24B")
    ax.set_ylabel("pred / true total flow")
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _gate(results: pd.DataFrame, out: Path) -> None:
    frame = results.loc[(results["model"] == "opflow_lite") & (results["variant"].str.endswith("learned_scale", na=False)) & (results["part"] == "test")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, col in zip(axes, ["gate_mean", "gate_distance_corr", "gate_opportunity_corr"]):
        if col in frame and frame[col].notna().any():
            ax.bar(frame["split_name"], frame[col], color="#B279A2")
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(col if col in frame and frame[col].notna().any() else f"{col}: not_available")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/opflow_large_commuting_od")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()
    with (Path(args.data_root) / "split_records.json").open("r", encoding="utf-8") as handle:
        records = pd.DataFrame(json.load(handle))
    records["area_id"] = records["area_id"].astype(str)
    with (Path(args.data_root) / "splits.json").open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    results = collect_results(Path(args.results_root))
    figures = Path(args.figures_dir)
    _split_distribution(records, splits, "matched_shortcut_high_to_low", ["distance_flow_corr", "dest_entropy_norm", "log_n_regions", "log_total_flow"], figures / "split_balance_shortcut.png")
    _split_distribution(records, splits, "matched_opportunity_low_to_high", ["distance_flow_corr", "dest_entropy_norm", "log_n_regions", "log_total_flow"], figures / "split_balance_opportunity.png")
    _bar_results(results, "matched_shortcut_high_to_low", "row_cpc", figures / "shortcut_shift_allocation_results.png")
    _bar_results(results, "matched_opportunity_low_to_high", "row_cpc", figures / "opportunity_shift_allocation_results.png")
    _scale(results, figures / "scale_diagnostics.png")
    _gate(results, figures / "opflow_lite_gate_diagnostics.png")


if __name__ == "__main__":
    main()
