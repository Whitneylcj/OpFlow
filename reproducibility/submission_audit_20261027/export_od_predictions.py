#!/usr/bin/env python3
"""Export saved local-run OD predictions, without training.

The bundle must contain source_snapshot/opflow_benchmark, checkpoints/, and
available_local_county_metrics.csv. The data root must contain county folders
with the converted NumPy arrays and region_meta.csv. A parent data/ directory
containing opflow_large_commuting_od/ is also accepted.

One CSV.gz is written per county. Model inference may materialize a complete
county matrix, but counties and CSV rows are never accumulated across counties.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


MODEL_NAMES = {
    "OpFlow": "opflow_full",
    "PairMLP": "pair_mlp",
    "ScaleMLP": "scale_aware_mlp",
}
METRICS = ("row_kl", "row_cpc", "global_cpc", "rmse", "nrmse_mean", "nrmse_official")
FIELDS = (
    "source_run", "split", "model", "seed", "county_id", "origin_id",
    "destination_id", "y_true", "y_pred", "valid_origin", "count_eval_mask",
    "allocation_eval_mask",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES),
                        help="Paper names or internal names; default: all three models.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Exact CSV split names; default: every available split.")
    parser.add_argument("--max-counties", type=int, default=None,
                        help="Maximum counties per model/split/seed, preserving CSV order.")
    parser.add_argument("--county-ids", nargs="+", default=None,
                        help="Optional county FIPS allowlist, preserving leading zeros.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing county exports.")
    args = parser.parse_args()
    if args.max_counties is not None and args.max_counties < 1:
        parser.error("--max-counties must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    canonical = set(MODEL_NAMES) | set(MODEL_NAMES.values())
    if set(args.models) - canonical:
        parser.error(f"Unknown models: {sorted(set(args.models) - canonical)}")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def county_node_ids(path, n):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"node_index", "GEOID"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing node_index/GEOID in {path}")
        records = list(reader)
    if len(records) != n or sorted(int(r["node_index"]) for r in records) != list(range(n)):
        raise ValueError(f"region_meta.csv does not map each matrix index exactly once: {path}")
    ids = [""] * n
    for row in records:
        ids[int(row["node_index"])] = row["GEOID"]
    if any(not value for value in ids) or len(set(ids)) != n:
        raise ValueError(f"Missing or duplicate GEOID in {path}")
    return ids


def select_groups(args, metrics_path):
    wanted_models = {MODEL_NAMES.get(name, name) for name in args.models}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_run", "split", "model", "internal_model", "seed", "area_id", "n_regions"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {metrics_path}: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    available_splits = {row["split"] for row in rows}
    if args.splits and set(args.splits) - available_splits:
        raise ValueError(f"Unknown splits: {sorted(set(args.splits) - available_splits)}")
    groups = {}
    seen = set()
    selected_counties = set()
    for row in rows:
        if row["internal_model"] not in wanted_models:
            continue
        if args.splits and row["split"] not in args.splits:
            continue
        if args.county_ids and row["area_id"] not in args.county_ids:
            continue
        if row.get("variant") and row["variant"] != row["internal_model"] + "_learned_scale":
            raise ValueError("Only learned-scale predictions may be exported")
        if row.get("part", "test") != "test":
            raise ValueError("The metrics inventory must contain test counties only")
        key = (row["source_run"], row["split"], row["internal_model"], row["seed"])
        identity = (*key, row["area_id"])
        if identity in seen:
            raise ValueError(f"Duplicate county metric row: {identity}")
        seen.add(identity)
        group = groups.setdefault(key, [])
        if args.max_counties is None or len(group) < args.max_counties:
            group.append(row)
            selected_counties.add(row["area_id"])
    groups = {key: rows for key, rows in groups.items() if rows}
    if not groups:
        raise ValueError("No test counties match the requested filters")
    if args.county_ids and set(args.county_ids) - selected_counties:
        print("Note: county IDs not selected (filters or --max-counties): "
              + ", ".join(sorted(set(args.county_ids) - selected_counties)), file=sys.stderr)
    return groups


def main():
    args = parse_args()
    bundle = args.bundle_dir.resolve()
    source = bundle / "source_snapshot"
    if not (source / "opflow_benchmark" / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing bundled source snapshot: {source}")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from opflow_benchmark.baselines import build_model
    from opflow_benchmark.baselines.common import evaluate_area_predictions
    from opflow_benchmark.dataset import AreaODDataset

    torch.set_num_threads(args.threads)
    data = args.data_root.resolve()
    if (data / "opflow_large_commuting_od").is_dir():
        data = data / "opflow_large_commuting_od"
    metrics_path = bundle / "available_local_county_metrics.csv"
    groups = select_groups(args, metrics_path)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle), "data_root": str(data),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "numpy_version": np.__version__, "device": "cpu", "threads": args.threads,
        "metrics_csv_sha256": sha256(metrics_path),
        "load_policy": "weights_only=True, map_location=cpu; no unrestricted pickle fallback",
        "selection": {"models": args.models, "splits": args.splits,
                      "max_counties_per_model_split_seed": args.max_counties,
                      "county_ids": args.county_ids},
        "count_mask": "All n*n cells, including diagonal and zero-outflow origins.",
        "allocation_mask": "valid_origin broadcast over destinations; includes diagonal.",
        "metric_check_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "provenance_note": "Saved local runs; not a verified reproduction of submitted Table 2 baseline numbers.",
        "counties": [],
    }
    for (run, split, internal_model, seed), rows in groups.items():
        model_dir = bundle / "checkpoints" / split / internal_model / f"seed_{seed}"
        checkpoint_path = model_dir / "checkpoint.pt"
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Safe checkpoint loading failed for {checkpoint_path}; "
                               "no unrestricted pickle fallback was attempted") from exc
        config = dict(checkpoint.get("config", {}))
        config.pop("model_variant", None)
        if internal_model == "opflow_full":
            config.pop("report_name", None)
        config["device"] = "cpu"
        model = build_model(internal_model, **config)
        model.load_checkpoint(checkpoint)
        # Fail on missing parameters: OpFlow's compatibility loader uses strict=False.
        state = dict(checkpoint["model_state"])
        if internal_model == "opflow_full":
            for module in ("deterrence", "benefit", "exposure", "competition"):
                old, new = module + ".net.", module + ".net.net."
                for key in list(state):
                    if key.startswith(old) and not key.startswith(new):
                        state.setdefault(new + key[len(old):], state[key])
        current = model.model.state_dict()
        missing = set(current) - set(state)
        mismatched = [key for key in current if key in state and current[key].shape != state[key].shape]
        if missing or mismatched:
            raise RuntimeError(f"Incompatible checkpoint: missing={sorted(missing)}, shapes={mismatched}")
        checkpoint_hash = sha256(checkpoint_path)
        dataset = AreaODDataset(data, area_ids=[row["area_id"] for row in rows])
        for index, row in enumerate(rows):
            county = row["area_id"]
            target = output_root / split / internal_model / f"seed_{seed}" / f"{county}.csv.gz"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {target}; use --overwrite to replace it")
            area = dataset[index]
            n = int(area["y_raw"].shape[0])
            if n != int(row["n_regions"]):
                raise ValueError(f"County {county}: n_regions differs from saved metric inventory")
            ids = county_node_ids(data / county / "region_meta.csv", n)
            with torch.inference_mode():
                y_hat, p_hat, _ = model.predict_area(area)
            actual_metrics = evaluate_area_predictions(area, y_hat, p_hat, topk=10)
            errors = {metric: abs(float(actual_metrics[metric]) - float(row[metric]))
                      for metric in METRICS if row.get(metric) not in (None, "")}
            mismatched_metrics = [metric for metric in errors
                                  if not np.isclose(actual_metrics[metric], float(row[metric]),
                                                    rtol=1e-5, atol=1e-5)]
            if mismatched_metrics:
                raise RuntimeError(f"County {county}, {split}, {internal_model}: metrics differ "
                                   f"from saved results: {mismatched_metrics}; abs_errors={errors}")
            y_true = area["y_raw"].detach().cpu().numpy()
            y_pred = y_hat.detach().cpu().numpy()
            valid_origin = area["valid_origin"].detach().cpu().numpy()
            if y_true.shape != (n, n) or y_pred.shape != (n, n) or valid_origin.shape != (n,):
                raise ValueError(f"Unexpected prediction or mask shape for {county}")
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                raise ValueError(f"Non-finite OD values in {county}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            try:
                with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(FIELDS)
                    for i in range(n):
                        valid = int(valid_origin[i])
                        for j in range(n):
                            writer.writerow((run, split, row["model"], seed, county,
                                             ids[i], ids[j], float(y_true[i, j]),
                                             float(y_pred[i, j]), valid, 1, valid))
                os.replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            entry = {"source_run": run, "split": split, "model": row["model"],
                     "internal_model": internal_model, "seed": int(seed), "county_id": county,
                     "n_regions": n, "od_rows": n * n, "valid_origins": int(valid_origin.sum()),
                     "csv_gz": str(target.relative_to(output_root)),
                     "checkpoint_sha256": checkpoint_hash, "metric_abs_errors": errors}
            manifest["counties"].append(entry)
            # Update the manifest after each finished county, so a later failure is inspectable.
            manifest_path = output_root / "export_manifest.json"
            temp_manifest = manifest_path.with_suffix(".json.tmp")
            temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temp_manifest, manifest_path)
            print(f"Exported {split} {internal_model} seed={seed} county={county}: {n*n} OD rows", flush=True)
            # OpFlow can cache area tensors; release them before the next county.
            if hasattr(model, "_clear_area_cache"):
                model._clear_area_cache()
            del area, y_hat, p_hat, y_true, y_pred, valid_origin
    print(f"Finished: {len(manifest['counties'])} county/model exports in {output_root}")


if __name__ == "__main__":
    main()
