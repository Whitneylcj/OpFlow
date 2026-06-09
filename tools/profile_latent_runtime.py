"""Profile latent synthetic runtime by stage without changing experiment logic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.data.latent_synthetic_dgp import generate_latent_synthetic_od_data, perturb_nuisance_records
from opflow_experiments.evaluation.evaluator import evaluate_predictions
from opflow_experiments.latent_models import build_latent_model
from opflow_experiments.pipelines.run_latent_synthetic_main import (
    _latent_diagnostic_row,
    _oracle_score_prediction,
    default_methods_for_regime,
    load_latent_config,
    normalize_latent_regime,
)


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/latent_opflow_e1_e4.yaml")
    parser.add_argument("--regime", default="stable_residual_ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--output", default="outputs/reports/latent_runtime_profile.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_latent_config(args.config)
    regime = normalize_latent_regime(args.regime)
    methods = args.methods or default_methods_for_regime(regime)
    rows = []

    start = time.perf_counter()
    data = generate_latent_synthetic_od_data(config, regime=regime, seed=args.seed)
    rows.append({"stage": "dgp", "method": "all", "seconds": _elapsed(start), "records": len(data.records)})

    records = data.records.copy()
    train = records.loc[records["split"] == "train"].copy()
    val = records.loc[records["split"] == "val"].copy()

    predictions = {}
    for method in methods:
        if method == "OracleScore":
            start = time.perf_counter()
            pred = _oracle_score_prediction(records)
            rows.append({"stage": "oracle_predict", "method": method, "seconds": _elapsed(start), "records": len(records)})
        else:
            model = build_latent_model(method, config, args.seed)
            start = time.perf_counter()
            model.fit(train, val)
            rows.append({"stage": "fit", "method": method, "seconds": _elapsed(start), "records": len(train)})

            start = time.perf_counter()
            pred = model.predict(records)
            rows.append({"stage": "predict", "method": method, "seconds": _elapsed(start), "records": len(records)})

        start = time.perf_counter()
        evaluate_predictions(records, pred, topk=int(config.get("baseline", {}).get("topk", 5)))
        rows.append({"stage": "evaluate", "method": method, "seconds": _elapsed(start), "records": len(records)})
        predictions[method] = (model if method != "OracleScore" else None, pred)

    for method, (model, pred) in predictions.items():
        if model is None or "stable_mask" not in pred.get("extra", {}):
            continue
        if "perturbed_records" not in locals():
            sigma = float(config.get("latent", {}).get("nuisance_perturb_sigma", 1.0))
            start = time.perf_counter()
            perturbed_records = perturb_nuisance_records(data, sigma=sigma, seed=args.seed + 100_003)
            rows.append({"stage": "diagnostic_perturb_records", "method": "all", "seconds": _elapsed(start), "records": len(records)})
        start = time.perf_counter()
        _latent_diagnostic_row(config, data, records, model, method, regime, args.seed, pred, perturbed_records=perturbed_records)
        rows.append({"stage": "diagnostics", "method": method, "seconds": _elapsed(start), "records": len(records)})

    profile = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    profile.to_csv(output, index=False)
    write_seconds = _elapsed(start)
    profile.loc[len(profile)] = {"stage": "profile_csv_write", "method": "all", "seconds": write_seconds, "records": len(profile)}
    profile.to_csv(output, index=False)

    summary = profile.groupby("stage", as_index=False)["seconds"].sum().sort_values("seconds", ascending=False)
    print(summary.to_string(index=False))
    print(f"Profile CSV: {output}")


if __name__ == "__main__":
    main()
