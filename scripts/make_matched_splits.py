"""Construct scale-matched shortcut and opportunity OOD splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_benchmark.diagnostics import save_area_diagnostics
from opflow_benchmark.splits import balance_acceptance, matched_extreme_split, random_split, split_balance, write_splits


def _summary_lines(balance: pd.DataFrame, acceptance: dict[str, dict]) -> list[str]:
    lines = []
    for split_name in balance["split_name"].drop_duplicates():
        lines.append(f"[{split_name}]")
        for row in balance.loc[balance["split_name"] == split_name].itertuples(index=False):
            lines.append(
                " ".join(
                    [
                        f"{row.part}: n={row.n_areas};",
                        f"mean_regions={row.mean_n_regions:.3f};",
                        f"mean_total_flow={row.mean_total_flow:.3f};",
                        f"mean_distance_flow_corr={row.mean_distance_flow_corr:.6f};",
                        f"mean_dest_entropy_norm={row.mean_dest_entropy_norm:.6f};",
                        f"mean_dest_hhi={row.mean_dest_hhi:.6f};",
                        f"mean_flow_density={row.mean_flow_density:.6f}",
                    ]
                )
            )
        if split_name in acceptance:
            lines.append(f"acceptance: {acceptance[split_name]}")
        lines.append("")
    return lines


def make_matched_splits(data_root: Path, seed: int, out_tables: Path, log_dir: Path, reports_dir: Path) -> dict:
    records = save_area_diagnostics(data_root, out_tables / "area_diagnostics.csv")
    splits = {"random": random_split(records["area_id"].tolist(), seed=seed)}
    reports = {}

    shortcut_hl, shortcut_hl_bins = matched_extreme_split(records, target_col="distance_flow_corr", n_bins=5, high_to_low=True)
    shortcut_lh, shortcut_lh_bins = matched_extreme_split(records, target_col="distance_flow_corr", n_bins=5, high_to_low=False)
    if len(shortcut_hl["train"]) < 20 or len(shortcut_hl["test"]) < 20:
        shortcut_hl, shortcut_hl_bins = matched_extreme_split(records, target_col="distance_flow_corr", n_bins=4, high_to_low=True)
        shortcut_lh, shortcut_lh_bins = matched_extreme_split(records, target_col="distance_flow_corr", n_bins=4, high_to_low=False)
        reports["shortcut_bins"] = "loosened_to_4x4"
    else:
        reports["shortcut_bins"] = "5x5"
    splits["matched_shortcut_high_to_low"] = shortcut_hl
    splits["matched_shortcut_low_to_high"] = shortcut_lh

    opp_lh, opp_lh_bins = matched_extreme_split(records, target_col="dest_entropy_norm", n_bins=5, high_to_low=False)
    opp_hl, opp_hl_bins = matched_extreme_split(records, target_col="dest_entropy_norm", n_bins=5, high_to_low=True)
    if len(opp_lh["train"]) < 20 or len(opp_lh["test"]) < 20:
        opp_lh, opp_lh_bins = matched_extreme_split(records, target_col="dest_entropy_norm", n_bins=4, high_to_low=False)
        opp_hl, opp_hl_bins = matched_extreme_split(records, target_col="dest_entropy_norm", n_bins=4, high_to_low=True)
        reports["opportunity_bins"] = "loosened_to_4x4"
    else:
        reports["opportunity_bins"] = "5x5"
    splits["matched_opportunity_low_to_high"] = opp_lh
    splits["matched_opportunity_high_to_low"] = opp_hl

    scale_lh, scale_lh_bins = matched_extreme_split(
        records,
        target_col="log_total_flow",
        control_cols=("log_n_regions", "flow_density"),
        n_bins=5,
        high_to_low=False,
    )
    scale_hl, scale_hl_bins = matched_extreme_split(
        records,
        target_col="log_total_flow",
        control_cols=("log_n_regions", "flow_density"),
        n_bins=5,
        high_to_low=True,
    )
    if len(scale_lh["train"]) < 20 or len(scale_lh["test"]) < 20:
        scale_lh, scale_lh_bins = matched_extreme_split(
            records,
            target_col="log_total_flow",
            control_cols=("log_n_regions", "flow_density"),
            n_bins=4,
            high_to_low=False,
        )
        scale_hl, scale_hl_bins = matched_extreme_split(
            records,
            target_col="log_total_flow",
            control_cols=("log_n_regions", "flow_density"),
            n_bins=4,
            high_to_low=True,
        )
        reports["scale_bins"] = "loosened_to_4x4"
    else:
        reports["scale_bins"] = "5x5"
    splits["matched_scale_low_to_high"] = scale_lh
    splits["matched_scale_high_to_low"] = scale_hl

    balance = split_balance(records, splits)
    acceptance = {
        split_name: balance_acceptance(balance, split_name)
        for split_name in splits
        if split_name != "random"
    }
    write_splits(data_root, splits, balance)
    out_tables.mkdir(parents=True, exist_ok=True)
    balance.to_csv(out_tables / "split_balance_table.csv", index=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "split_summary.txt").write_text("\n".join(_summary_lines(balance, acceptance)), encoding="utf-8")

    reports_dir.mkdir(parents=True, exist_ok=True)
    matching_report = {
        "bin_reports": reports,
        "acceptance": acceptance,
        "balance": balance.to_dict(orient="records"),
        "shortcut_high_to_low_bins": shortcut_hl_bins.to_dict(orient="records"),
        "opportunity_low_to_high_bins": opp_lh_bins.to_dict(orient="records"),
        "scale_low_to_high_bins": scale_lh_bins.to_dict(orient="records"),
    }
    (reports_dir / "split_matching_report.md").write_text(_markdown_report(matching_report), encoding="utf-8")
    failures = {name: row for name, row in acceptance.items() if not row.get("accepted")}
    if failures:
        (reports_dir / "split_matching_failure.md").write_text(_failure_report(failures), encoding="utf-8")
    return {"splits": splits, "balance": balance.to_dict(orient="records"), "acceptance": acceptance}


def _markdown_report(payload: dict) -> str:
    lines = ["# Split Matching Report", "", f"Bin settings: {payload['bin_reports']}", "", "## Acceptance", ""]
    for split, values in payload["acceptance"].items():
        lines.append(f"- `{split}`: {values}")
    balance = pd.DataFrame(payload.get("balance", []))
    if not balance.empty:
        lines.extend(["", "## Target Shifts", ""])
        for split in payload["acceptance"]:
            parts = balance.loc[balance["split_name"] == split].set_index("part")
            if not {"train", "test"} <= set(parts.index):
                continue
            train = parts.loc["train"]
            test = parts.loc["test"]
            lines.append(
                "- "
                + f"`{split}`: "
                + f"distance_flow_corr {train.mean_distance_flow_corr:.4f} -> {test.mean_distance_flow_corr:.4f}; "
                + f"dest_entropy_norm {train.mean_dest_entropy_norm:.4f} -> {test.mean_dest_entropy_norm:.4f}; "
                + f"log_total_flow {train.mean_log_total_flow:.4f} -> {test.mean_log_total_flow:.4f}; "
                + f"mean_n_regions ratio {max(train.mean_n_regions, test.mean_n_regions) / max(min(train.mean_n_regions, test.mean_n_regions), 1e-8):.3f}; "
                + f"mean_total_flow ratio {max(train.mean_total_flow, test.mean_total_flow) / max(min(train.mean_total_flow, test.mean_total_flow), 1e-8):.3f}"
            )
    lines.append("")
    return "\n".join(lines)


def _failure_report(failures: dict) -> str:
    lines = ["# Split Matching Failure Diagnostics", "", "Some matched splits did not meet all strict acceptance thresholds.", ""]
    for split, values in failures.items():
        lines.append(f"- `{split}`: {values}")
    lines.extend(
        [
            "",
            "Use these splits with the reported balance table. If scale ratios remain high, label the result as partly structural-scale OOD rather than pure shortcut/opportunity OOD.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/opflow_large_commuting_od")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tables_dir", default="tables")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()
    result = make_matched_splits(Path(args.data_root), args.seed, Path(args.tables_dir), Path(args.log_dir), Path(args.reports_dir))
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
