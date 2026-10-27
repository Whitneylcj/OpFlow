# OpFlow


## Shared reproducibility audit (2026-10-27)

[Collaborator materials and runnable commands](reproducibility/submission_audit_20261027/README.md) · [中文审计报告](reports/submission_reproducibility_audit_20261027.md) · [Download the evidence bundle](dist/OpFlow_collaborator_materials_20261027.zip)

The bundle includes county split IDs, saved configs and logs, raw metrics, county-level confidence intervals, nine checkpoints, and portable CI/OD-export scripts. It records unresolved Table 2 provenance and GeoSP-protocol differences; it is **not a verified submission-code release**. Its source snapshot is used by the bundled scripts; the main code below has not been replaced by that snapshot.


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
- `opflow_benchmark/metrics.py`: matrix-style OD metrics; report-facing synthetic metrics are Row-KL, Row-CPC, raw-domain RMSE, raw-domain MAE, and PMSE, with log-RMSE/global CPC/ranking metrics available as auxiliary diagnostics.
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

The active synthetic suite uses the single final `DGP` regime and organizes the
paper-facing simulations as four questions: `q1_scale`, `q2_opportunity`,
`q3_component_corr`, and `q4_ablation`. It uses one shared origin/destination
support, excludes self-loops, fits oracle standardizers on training
environments only, and exposes only admissible deployed proxy features.
`q1_scale` is a training/validation scale-heterogeneity sweep: it holds the
oracle allocation law fixed and changes only origin production scale, generated
counts, and implied count intensity in the training objective. `RawCountMLP`
uses raw-count MSE for this suite, while allocation methods optimize
row-level allocation risk. Q1-Q3 and the appendix stress test use the same
baseline roster: mechanism probes, DG allocation MLPs (`GroupDRO`, `VREx`,
`IRM`, `DANN`), classical opportunity laws, validation-selected `BestDG` /
`BestClassical`, and `OpFlow`; Q4 remains OpFlow ablation-only. The default
synthetic OpFlow enables the opportunity-shift gate and a stronger operator
anchor so Q2 redistribution uses opportunity-supported fallback structure
instead of over-amplifying the learned local field.

Full four-question DGP run, 30 seeds:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode full \
  --run-id synthetic_four_30seed
```

Fast four-question smoke check:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --questions q1_scale q2_opportunity q3_component_corr q4_ablation \
  --run-id synthetic_four_debug
```

Appendix residual behavioral drift stress test:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --questions stress_residual_drift \
  --run-id stress_residual_drift_debug
```

Reported synthetic metrics are Row-KL, Row-CPC, raw-domain RMSE,
raw-domain MAE, and PMSE.

DGP outputs are written to:

```text
results/q1_scale/metrics.csv
results/q2_opportunity/metrics.csv
results/q3_component_corr/metrics.csv
results/q3_component_corr/correlation_checks.csv
results/q4_ablation/metrics.csv
results/stress_residual_drift/metrics.csv
figures/synthetic_main.png
figures/synthetic_main.pdf
figures/stress_residual_drift.png
figures/stress_residual_drift.pdf
tables/synthetic_endpoint.csv
reports/synthetic_report.md
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
  --config configs/synthetic/DGP.yaml \
  --questions q3_component_corr \
  --override exposure_model.epochs=150 \
  --run-id q3_epochs_150
```

For smoke tests only, add `--debug`; do not use debug outputs as paper evidence.
