# Experiment Registry

The current canonical experiments are:

- Synthetic final DGP: `experiments/synthetic/run_synthetic_main.py`
- Real-world LargeCommutingOD benchmark: `experiments/real_world/run_large_commuting_od.py`

Every run writes:

```text
outputs/runs/{run_id}/
├── resolved_config.yaml
├── metrics.csv
├── summary.json
├── summary.md
└── artifacts/
```

## Synthetic Experiments

`q1_scale` is a training/validation origin-scale heterogeneity experiment. It
keeps the oracle allocation mechanism fixed across severity values, so Row-KL
movement comes from raw-count supervision being dominated by high-scale rows,
not from a pure test-time production shock. `RawCountMLP` uses raw-count MSE in
the synthetic suite; allocation baselines use row-level allocation risk.

Run the final four-question DGP suite for paper-scale 30 seeds:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode full \
  --run-id synthetic_four_30seed
```

Run a quicker four-question DGP smoke check:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --questions q1_scale q2_opportunity q3_component_corr q4_ablation \
  --run-id synthetic_four_debug
```

Run the opt-in appendix residual behavioral drift stress test:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --questions stress_residual_drift \
  --run-id stress_residual_drift_debug
```

DGP outputs include per-run artifacts under `outputs/runs/{run_id}/`,
paper-facing CSVs under `results/q1_scale/`, `results/q2_opportunity/`,
`results/q3_component_corr/`, and `results/q4_ablation/`, plus
`figures/synthetic_main.*`, `tables/synthetic_endpoint.csv`, and
`reports/synthetic_report.md`. The appendix stress writes
`results/stress_residual_drift/metrics.csv` and
`figures/stress_residual_drift.*` when explicitly requested.

## Real-World Experiments

LargeCommutingOD remains under the real-world runner:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od.yaml \
  --run-id real_large_commuting_od_full
```

This runner writes flushed progress lines to the terminal and to
`outputs/runs/{run_id}/logs/real_world_progress.log` for `tail -f` monitoring.
The default real-world config includes the local OD baselines, OpFlow, and all
10 official-style CommutingODGen baselines. The official sklearn-family names
fall back to deterministic numpy regressors when sklearn is unavailable.

Run the Mac-compatible rewrites of the 10 official CommutingODGen baselines as
a smoke check:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od_official_smoke.yaml \
  --run-id official_baselines_smoke
```

For full-area compatibility checks, run one official baseline per command using
`configs/real_world/large_commuting_od_official_full_check.yaml` and override
`models=[...]`. Example:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od_official_full_check.yaml \
  --override 'models=[official_wedan]' \
  --run-id official_full_check_official_wedan
```

The root `README.md` lists all 10 one-baseline commands. The full-check config
uses all random-split areas with one seed and resolves
`training.official_runtime: auto` at runtime: CUDA hosts use `device=cuda`,
while non-CUDA hosts use the Mac-friendly sklearn pair caps and lightweight
neural dimensions.
