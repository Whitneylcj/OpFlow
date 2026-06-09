# OpFlow

OpFlow is organized as a reproducible experiment project for opportunity-invariant spatial flow prediction.

```text
configs/              YAML configs for synthetic and real-world experiments
experiments/          Canonical CLI entrypoints; run commands should start here
opflow_experiments/   Experiment orchestration package: pipelines, configs, outputs, final synthetic DGP, synthetic models
opflow_benchmark/     Real-world OD benchmark package: LargeCommutingOD dataset loader, real-world baselines, OD metrics, splits
scripts/              Data preparation, split construction, plotting, and result export utilities
tools/                One-off diagnostics and sanity checks
data/                 Converted local datasets; current real-world data is data/opflow_large_commuting_od
results/              Generated result mirrors, recreated by experiment runs
outputs/runs/         Generated run-level experiment outputs
outputs/reports/      Generated analysis reports
```

`opflow_experiments/` and `opflow_benchmark/` are not duplicate packages. The
first one is the experiment runner layer shared by synthetic and real-world
experiments. The second one is the real-world OD benchmark library used by the
real-world pipeline.

All commands below use the `opflow` conda environment.

Every canonical run writes:

```text
outputs/runs/{run_id}/
├── resolved_config.yaml
├── metrics.csv
├── summary.json
├── summary.md
├── artifacts/
├── figures/
└── logs/
```

## Call Graphs

### Synthetic Suite

The synthetic experiments are fully contained in the `opflow_experiments/`
package. The canonical call chain is:

```text
experiments/synthetic/run_synthetic_main.py
  -> opflow_experiments/pipelines/run_synthetic_main.py
     -> opflow_experiments/pipelines/run_dgp_main.py
        -> opflow_experiments/data/dgp.py
        -> opflow_experiments/exposure_models.py
        -> opflow_experiments/evaluation/evaluator.py
        -> opflow_experiments/metrics/*
        -> opflow_experiments/evaluation/standard_outputs.py
```

Responsibilities:

- `experiments/synthetic/run_synthetic_main.py`: thin CLI entrypoint.
- `configs/synthetic/DGP.yaml`: final DGP, model settings, and output options.
- `opflow_experiments/pipelines/run_dgp_main.py`: loops over seeds, trains models, evaluates metrics, and writes standard outputs.
- `opflow_experiments/data/dgp.py`: the single final synthetic data-generating process.
- `opflow_experiments/exposure_models.py`: synthetic OpFlow and baseline model implementations.
- `opflow_experiments/metrics/`: DataFrame-style allocation, ranking, prediction, and diagnostic metrics for synthetic records.
- `opflow_experiments/evaluation/standard_outputs.py`: common `metrics.csv`, `summary.json`, and `summary.md` writer.

### Real-World LargeCommutingOD

The real-world experiment uses `opflow_experiments/` for orchestration and
`opflow_benchmark/` for the actual OD benchmark implementation:

```text
experiments/real_world/run_large_commuting_od.py
  -> opflow_experiments/pipelines/real_world_large_commuting_od.py
     -> opflow_experiments/data/real_world_registry.py
        -> opflow_benchmark/dataset.py
     -> opflow_benchmark/baselines/__init__.py
        -> opflow_benchmark/baselines/*
     -> opflow_benchmark/metrics.py
     -> opflow_benchmark/exposure.py
     -> opflow_experiments/evaluation/standard_outputs.py
     -> scripts/make_tables.py and scripts/make_figures.py for paper-facing exports
```

Responsibilities:

- `experiments/real_world/run_large_commuting_od.py`: thin CLI entrypoint.
- `configs/real_world/large_commuting_od.yaml`: active LargeCommutingOD config, including splits, model list, seeds, chunk sizes, and reporting mirror.
- `opflow_experiments/pipelines/real_world_large_commuting_od.py`: resolves config, loads split areas, runs each split/model/seed task, emits progress, and writes run outputs.
- `opflow_experiments/data/real_world_registry.py`: maps the config dataset name to the converted dataset root and dataset class.
- `opflow_benchmark/dataset.py`: loads one variable-size OD area from `data/opflow_large_commuting_od`.
- `opflow_benchmark/baselines/`: real-world models and baselines, including `distance_only`, `gravity`, `intervening_opportunity`, `pair_mlp`, `scale_aware_mlp`, and `opflow_real`.
- `opflow_benchmark/exposure.py`: opportunity/exposure feature construction used by real-world baselines and OpFlow-real.
- `opflow_benchmark/metrics.py`: matrix-style OD metrics such as RMSE, log-RMSE, global CPC, Row-KL, Row-CPC, rank correlation, and top-k overlap.
- `scripts/build_opflow_dataset.py`, `scripts/inspect_large_commuting_od.py`, and `scripts/make_matched_splits.py`: preparation utilities only needed when rebuilding the converted dataset or splits.

Current real-world evidence is:

```text
outputs/runs/real_opflow_exposure_potential_full_no_chal_cache/
results/real_opflow_exposure_potential/
outputs/reports/real_world_exposure_potential_full_no_chal_cache_report.md
```

## Environment

Create the Mac-compatible OpFlow environment with:

```bash
conda env create -f environment-opflow.yml
```

This environment intentionally does not install DGL. The official
CommutingODGen baselines are integrated as DGL-free, CUDA-free rewrites under
the `official_*` model names.

## Synthetic Experiment

The active synthetic experiment is the single final `DGP`. It generates
environment-specific opportunity fields, fits oracle standardizers on training
environments only, and exposes only raw/proxy deployed features to OpFlow.

Full DGP run, 30 seeds:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode full \
  --run-id DGP_30seed
```

Fast DGP smoke check:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --run-id DGP_debug
```

Primary metrics are Row-KL, Row-CPC, rank correlation, top-k overlap, PMSE,
log-RMSE, and global CPC. Raw-count RMSE is secondary because it also depends
on origin-scale recovery.

DGP outputs are written to:

```text
results/DGP/metrics_by_seed.csv
results/DGP/summary_mean_ci.csv
results/DGP/diagnostics.csv
results/DGP/summary.md
outputs/reports/DGP_report.md
```

## Real-World Experiment

LargeCommutingOD full benchmark:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od.yaml \
  --run-id real_large_commuting_od_full
```

The real-world runner prints one flushed progress line per task and also writes
`outputs/runs/{run_id}/logs/real_world_progress.log`, which can be watched with
`tail -f`. The default real-world config runs all configured splits, seeds,
area sizes, the local classical/neural baselines, OpFlow, and all 10
official-style CommutingODGen baselines:

```text
official_gm_p, official_gm_e, official_svr, official_rf, official_gbrt,
official_dgm, official_gmel, official_netgan, official_diffodgen, official_wedan
```

The official-style rewrites are DGL-free. `official_svr`, `official_rf`, and
`official_gbrt` use sklearn when it is installed and deterministic numpy
fallback regressors otherwise, so smoke checks can run in the current local
environment without installing extra packages.

To run only one stage, edit `stage` in
`configs/real_world/large_commuting_od.yaml` or pass an override:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od.yaml \
  --override stage=train \
  --run-id real_large_commuting_od_train_full
```

Official CommutingODGen baseline smoke:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od_official_smoke.yaml \
  --run-id official_baselines_smoke
```

The official smoke config runs the 10 `official_*` models on a tiny random
split. It is a compatibility check, not paper evidence. With
`training.official_runtime: auto`, the runner uses a CUDA profile when
`torch.cuda.is_available()` is true; otherwise it falls back to the CPU/Mac
profile used for local checks. The resolved choice is saved in
`resolved_config.yaml` as `training.official_runtime_resolved`.

Official full-area baseline checks, one baseline per command:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_gm_p]' --run-id official_full_check_official_gm_p
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_gm_e]' --run-id official_full_check_official_gm_e
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_svr]' --override training.official_max_train_pairs=200 --run-id official_full_check_official_svr
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_rf]' --run-id official_full_check_official_rf
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_gbrt]' --run-id official_full_check_official_gbrt
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_dgm]' --run-id official_full_check_official_dgm
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_gmel]' --run-id official_full_check_official_gmel
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_netgan]' --run-id official_full_check_official_netgan
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_diffodgen]' --run-id official_full_check_official_diffodgen
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py --config configs/real_world/large_commuting_od_official_full_check.yaml --override 'models=[official_wedan]' --run-id official_full_check_official_wedan
```

This full-area check uses all areas in the random split with one seed and no
`area_limits`. Runtime defaults are automatic: CUDA machines use `device=cuda`
and the closer-to-official Torch architecture defaults, while non-CUDA machines
use the Mac-friendly sklearn pair caps and lightweight neural dimensions. Full
train, validation, and test evaluation still scans all areas and prints progress
bars.

## Common Overrides

Override any nested config key from the command line:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/exposure_opflow_e1_e3.yaml \
  --override exposure_model.epochs=150 \
  --run-id exposure_opflow_epochs_150
```

For smoke tests only, add `--debug`; do not use debug outputs as paper evidence.
