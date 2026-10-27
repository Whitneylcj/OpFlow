"""Audit saved local evaluations; these are NOT verified submission outputs.

No training or checkpoint loading. Bootstrap counties within each split, pairing
models by county ID. Each interval conditions on the saved training seed 2026.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "OpFlow": ("realworld_opflow_strong_seed2026_3splits_20260619", "opflow_full"),
    "PairMLP": ("realworld_seed2026_core3splits_20260613", "pair_mlp"),
    "ScaleMLP": ("realworld_seed2026_core3splits_20260613", "scale_aware_mlp"),
}
SPLITS = ["RandSP_seed0", "GeoSP_SouthAtlantic", "OppSP"]
METRICS = ["row_kl", "row_cpc", "log_rmse", "global_cpc", "rmse", "nrmse_mean", "nrmse_official"]
RUNS = [
    "realworld_seed2026_core3splits_20260613",
    "realworld_main_strict_seed2026_all_geosp",
    "realworld_opflow_strong_seed2026_3splits_20260619",
    "realworld_classical_si_3splits_geosp9_20260619",
    "realworld_rf_gbrt_official_gm_seed2026_geosp9",
    "realworld_official_target_1seed_3splits_20260618",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/audits/submission_20261027")
    parser.add_argument("--bootstrap-seed", type=int, default=20261027)
    parser.add_argument("--replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    splits_path = ROOT / "data/opflow_large_commuting_od/splits.json"
    split_definitions = json.loads(splits_path.read_text())
    rows, provenance, model_configs = [], [], []
    configs_dir = out / "saved_configs"
    configs_dir.mkdir(exist_ok=True)
    for display_name, (run_id, model) in SOURCES.items():
        for split in SPLITS:
            directory = ROOT / "outputs/runs" / run_id / "artifacts/real_world_models" / split / model / "seed_2026"
            path = directory / "eval.json"
            payload = json.loads(path.read_text())
            variant = f"{model}_learned_scale"
            records = payload["area_metrics"][variant]["test"]
            expected = set(str(v) for v in split_definitions[split]["test"])
            ids = [str(r["area_id"]) for r in records]
            assert len(ids) == len(set(ids)), (split, model, "duplicate county")
            assert set(ids) == expected, (split, model, "test county mismatch")
            for metric in METRICS:
                values = np.array([r.get(metric, np.nan) for r in records], dtype=float)
                reported = payload["metrics"][variant]["test"][metric]["mean"]
                assert np.isclose(np.nanmean(values), reported, rtol=1e-10, atol=1e-10), (path, metric)
            for record in records:
                rows.append({
                    "source_run": run_id, "split": split, "model": display_name,
                    "internal_model": model, "seed": 2026, "variant": variant,
                    "area_id": str(record["area_id"]), "n_regions": record["n_regions"],
                    "true_total_flow": record["true_total_flow"],
                    "source_eval": str(path.relative_to(ROOT)),
                    **{m: record.get(m, np.nan) for m in METRICS},
                })
            provenance.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "test_counties": len(ids)})
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text())
            model_configs.append({"run_id": run_id, "split": split, "model": display_name, "seed": 2026,
                                  "training": config.get("training", {}), "selection_summary": config.get("selection_summary", {}),
                                  "source": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "available_local_county_metrics.csv", index=False)
    summaries, comparisons = [], []
    names = list(SOURCES)
    for split_index, split in enumerate(SPLITS):
        selected = frame.loc[frame["split"].eq(split)]
        county_ids = sorted(selected["area_id"].unique())
        aligned = [selected.loc[selected["model"].eq(name)].set_index("area_id").loc[county_ids] for name in names]
        for other in aligned[1:]:
            assert np.array_equal(aligned[0]["n_regions"], other["n_regions"])
            assert np.allclose(aligned[0]["true_total_flow"], other["true_total_flow"], rtol=0, atol=1e-8)
        for metric_index, metric in enumerate(METRICS):
            values = np.column_stack([a[metric].to_numpy(dtype=float) for a in aligned])
            valid = np.isfinite(values).all(axis=1)
            values = values[valid]
            n = len(values)
            assert n > 1, (split, metric)
            rng = np.random.default_rng(np.random.SeedSequence([args.bootstrap_seed, split_index, metric_index]))
            boot = np.empty((args.replicates, len(names)))
            for start in range(0, args.replicates, 256):
                stop = min(start + 256, args.replicates)
                index = rng.integers(0, n, size=(stop - start, n))
                boot[start:stop] = values[index].mean(axis=1)
            means = values.mean(axis=0)
            limits = np.quantile(boot, [0.025, 0.975], axis=0)
            for i, name in enumerate(names):
                summaries.append({"split": split, "model": name, "source_run": SOURCES[name][0], "seed": 2026,
                                  "metric": metric, "county_mean": means[i], "ci95_low": limits[0, i], "ci95_high": limits[1, i],
                                  "n_counties": n, "excluded_nonfinite_counties": int((~valid).sum())})
            for i in range(1, len(names)):
                delta = boot[:, 0] - boot[:, i]
                low, high = np.quantile(delta, [0.025, 0.975])
                comparisons.append({"split": split, "comparison": f"OpFlow - {names[i]}", "metric": metric,
                                    "mean_difference": means[0] - means[i], "ci95_low": low, "ci95_high": high,
                                    "n_paired_counties": n, "excluded_nonfinite_counties": int((~valid).sum()),
                                    "training_seed": 2026, "opflow_run": SOURCES["OpFlow"][0], "baseline_run": SOURCES[names[i]][0]})
    pd.DataFrame(summaries).to_csv(out / "available_local_model_summary_ci.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "available_local_paired_county_ci.csv", index=False)
    (out / "available_local_effective_configs.json").write_text(json.dumps(model_configs, indent=2) + "\n")
    command_lines = ["# Reconstructed commands for saved runs; NOT original captured argv or a submission reproduction guarantee.",
                     "# These commands would TRAIN models. The audit did not execute them.", "conda activate opflow", f"cd {ROOT}"]
    for run_id in RUNS:
        source = ROOT / "outputs/runs" / run_id / "resolved_config.yaml"
        target = configs_dir / f"{run_id}.yaml"
        shutil.copyfile(source, target)
        provenance.append({"path": str(source.relative_to(ROOT)), "sha256": sha256(source)})
        command_lines.append(f"python experiments/real_world/run_large_commuting_od.py --config {target} --run-id audit_replay_{run_id}")
    (out / "reconstructed_commands.txt").write_text("\n\n".join(command_lines) + "\n")
    manifest = {
        "status": "AVAILABLE_LOCAL_RUNS_ONLY_NOT_VERIFIED_SUBMISSION",
        "paper": "https://arxiv.org/html/2607.03200v1#S5.T2",
        "bootstrap": {"unit": "county", "within_split": True, "paired": True, "replicates": args.replicates,
                      "seed": args.bootstrap_seed, "interval": "percentile 95%", "aggregation": "equal weight counties"},
        "limitations": [
            "PairMLP and ScaleMLP source metrics do not match published Table 2.",
            "OpFlow source is a later run than the two baselines; all are explicitly identified, not a frozen submission bundle.",
            "Only one training seed (2026); intervals do not estimate training randomness.",
            "GeoSP here is SouthAtlantic only, not the nine-fold GeoSP protocol average.",
            "Count metrics reuse saved all-cell county evaluations, not newly masked OD recomputation.",
            "global_cpc is count-CPC within each county, summarized as a county mean, not a pooled OD-level CPC.",
            "nrmse_mean divides by mean true flow; nrmse_official divides by true-flow population standard deviation.",
            "No overall bootstrap pooling across splits because counties overlap between protocols.",
            "County bootstrap assumes exchangeable counties; no spatial-block uncertainty correction.",
        ],
        "split_file": {"path": str(splits_path.relative_to(ROOT)), "sha256": sha256(splits_path)},
        "source_files": provenance,
        "checks": ["unique IDs", "exact saved/current test ID equality", "same county node counts and true totals across models",
                   "recomputed county means equal saved summaries for all seven metrics"],
        "county_metric_rows": len(frame),
    }
    (out / "available_local_audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "county_rows": len(frame), "model_summaries": len(summaries), "paired_intervals": len(comparisons)}))


if __name__ == "__main__":
    main()
