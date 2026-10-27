#!/usr/bin/env python3
"""Recompute CIs from the available-local county evidence bundle.

Requirements: Python 3.10+, NumPy and pandas. No repository imports, training,
checkpoint loading, or network access. Input files are read only. These outputs
are not verified submission results: they condition on training seed 2026,
compare the identified saved runs, and use SouthAtlantic as the GeoSP fold.

Example:
    python recompute_county_ci.py --bundle-dir /path/to/extracted_bundle

The bootstrap exactly follows audit_available_county_outputs.py: fixed split,
metric and model order, NumPy SeedSequence([20261027, split_index, metric_index]),
20,000 replicates by default, batches of 256, and percentile 95% intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 20261027
TRAINING_SEED = 2026
SOURCES = {
    "OpFlow": ("realworld_opflow_strong_seed2026_3splits_20260619", "opflow_full"),
    "PairMLP": ("realworld_seed2026_core3splits_20260613", "pair_mlp"),
    "ScaleMLP": ("realworld_seed2026_core3splits_20260613", "scale_aware_mlp"),
}
SPLITS = ["RandSP_seed0", "GeoSP_SouthAtlantic", "OppSP"]
METRICS = ["row_kl", "row_cpc", "log_rmse", "global_cpc", "rmse", "nrmse_mean", "nrmse_official"]
OUTPUT_NAMES = ["available_local_model_summary_ci.csv", "available_local_paired_county_ci.csv"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(bundle_dir: Path) -> pd.DataFrame:
    """Validate fixed source identities and the exact test county alignment."""
    source = bundle_dir / "available_local_county_metrics.csv"
    split_file = bundle_dir / "evidence/data/opflow_large_commuting_od/splits.json"
    frame = pd.read_csv(source, dtype={"area_id": str}, float_precision="round_trip")
    definitions = json.loads(split_file.read_text(encoding="utf-8"))
    required = {
        "source_run", "split", "model", "internal_model", "seed", "variant",
        "area_id", "n_regions", "true_total_flow", "source_eval", *METRICS,
    }
    require(required.issubset(frame.columns), f"Missing columns: {sorted(required - set(frame.columns))}")
    require(not frame.empty, "County input is empty")
    require(set(frame["split"]) == set(SPLITS), "Unexpected or missing split")
    require(set(frame["model"]) == set(SOURCES), "Unexpected or missing model")
    require(frame["area_id"].notna().all(), "Missing county ID")
    require(frame["area_id"].str.fullmatch(r"[0-9]{5}").all(), "County IDs must be five-digit strings")
    require(not frame.duplicated(["split", "model", "seed", "area_id"]).any(), "Duplicate split/model/seed/county key")
    for column in ["seed", "n_regions", "true_total_flow", *METRICS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    require(frame["seed"].eq(TRAINING_SEED).all(), "All records must use training seed 2026")
    nodes = frame["n_regions"].to_numpy(dtype=float)
    totals = frame["true_total_flow"].to_numpy(dtype=float)
    require(np.isfinite(nodes).all() and (nodes > 0).all() and (nodes == np.floor(nodes)).all(), "Node counts must be positive finite integers")
    require(np.isfinite(totals).all() and (totals >= 0).all(), "True total flows must be finite and nonnegative")
    for name, (run_id, internal) in SOURCES.items():
        for split in SPLITS:
            selected = frame.loc[frame["model"].eq(name) & frame["split"].eq(split)]
            require(not selected.empty, f"Missing records for {split}/{name}")
            expected_list = [str(v) for v in definitions[split]["test"]]
            require(len(expected_list) == len(set(expected_list)), f"Duplicate IDs in split definition {split}")
            require(set(selected["area_id"]) == set(expected_list), f"Test county mismatch for {split}/{name}")
            require(selected["source_run"].eq(run_id).all(), f"Source run mismatch for {split}/{name}")
            require(selected["internal_model"].eq(internal).all(), f"Internal model mismatch for {split}/{name}")
            require(selected["variant"].eq(f"{internal}_learned_scale").all(), f"Prediction variant mismatch for {split}/{name}")
            expected_source = f"outputs/runs/{run_id}/artifacts/real_world_models/{split}/{internal}/seed_2026/eval.json"
            require(selected["source_eval"].eq(expected_source).all(), f"Source evaluation identity mismatch for {split}/{name}")
    return frame


def recompute(frame: pd.DataFrame, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, comparisons = [], []
    names = list(SOURCES)
    for split_index, split in enumerate(SPLITS):
        selected = frame.loc[frame["split"].eq(split)]
        county_ids = sorted(selected["area_id"].unique())
        aligned = [selected.loc[selected["model"].eq(name)].set_index("area_id").loc[county_ids] for name in names]
        for other in aligned[1:]:
            require(np.array_equal(aligned[0]["n_regions"], other["n_regions"]), f"Node counts differ across models in {split}")
            require(np.allclose(aligned[0]["true_total_flow"], other["true_total_flow"], rtol=0, atol=1e-8), f"True flow totals differ across models in {split}")
        for metric_index, metric in enumerate(METRICS):
            values = np.column_stack([a[metric].to_numpy(dtype=float) for a in aligned])
            valid = np.isfinite(values).all(axis=1)
            values = values[valid]
            n = len(values)
            require(n > 1, f"Fewer than two paired finite counties: {split}/{metric}")
            rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, split_index, metric_index]))
            boot = np.empty((replicates, len(names)))
            for start in range(0, replicates, 256):
                stop = min(start + 256, replicates)
                index = rng.integers(0, n, size=(stop - start, n))
                boot[start:stop] = values[index].mean(axis=1)
            means = values.mean(axis=0)
            limits = np.quantile(boot, [0.025, 0.975], axis=0)
            for i, name in enumerate(names):
                summaries.append({"split": split, "model": name, "source_run": SOURCES[name][0], "seed": TRAINING_SEED,
                                  "metric": metric, "county_mean": means[i], "ci95_low": limits[0, i], "ci95_high": limits[1, i],
                                  "n_counties": n, "excluded_nonfinite_counties": int((~valid).sum())})
            for i in range(1, len(names)):
                delta = boot[:, 0] - boot[:, i]
                low, high = np.quantile(delta, [0.025, 0.975])
                comparisons.append({"split": split, "comparison": f"OpFlow - {names[i]}", "metric": metric,
                                    "mean_difference": means[0] - means[i], "ci95_low": low, "ci95_high": high,
                                    "n_paired_counties": n, "excluded_nonfinite_counties": int((~valid).sum()),
                                    "training_seed": TRAINING_SEED, "opflow_run": SOURCES["OpFlow"][0], "baseline_run": SOURCES[names[i]][0]})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing available_local_county_metrics.csv and evidence/ (default: script directory)")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "outputs/recomputed_county_ci",
                        help="New output directory (default: ./outputs/recomputed_county_ci)")
    parser.add_argument("--replicates", type=int, default=20000, help="Bootstrap replicates; at least 1000 (default: 20000)")
    args = parser.parse_args()
    require(args.replicates >= 1000, "Use at least 1000 bootstrap replicates")
    bundle = args.bundle_dir.resolve()
    out = args.output_dir.resolve()
    evidence = bundle / "evidence"
    require(out != bundle and out != evidence and evidence not in out.parents, "Output must not overwrite the bundle root or evidence directory")
    for filename in OUTPUT_NAMES:
        require(not (out / filename).exists(), f"Refusing to overwrite existing output: {out / filename}; choose a new --output-dir")
    frame = load_and_validate(bundle)
    summaries, comparisons = recompute(frame, args.replicates)
    out.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(out / OUTPUT_NAMES[0], index=False)
    comparisons.to_csv(out / OUTPUT_NAMES[1], index=False)
    print(json.dumps({"status": "AVAILABLE_LOCAL_RUNS_ONLY_NOT_VERIFIED_SUBMISSION", "output_dir": str(out),
                      "county_rows": len(frame), "model_summaries": len(summaries), "paired_intervals": len(comparisons),
                      "bootstrap_seed": BOOTSTRAP_SEED, "training_seed": TRAINING_SEED, "replicates": args.replicates,
                      "checks": ["unique keys", "exact test county IDs", "run/model/variant/seed/source identity",
                                 "positive integer node counts", "finite nonnegative true flows", "paired node-count and true-flow equality"]}))


if __name__ == "__main__":
    main()
