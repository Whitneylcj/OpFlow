# Synthetic Four-Question Report

Run directory: `outputs/runs/q1_3seed_current`

## DGP Settings

- suite/regime: `DGP`
- N: `80` shared origin/destination units; self-loops excluded.
- train/val/test envs: `8/2/4`
- oracle fields are diagnostics only; deployed models use coordinates, opportunity proxy, cost/rank, and row-standardized cost.
- RawCountMLP objective: `mse` raw-count reconstruction.

## Metrics

- Reported metrics: Row-KL, Row-CPC, raw-domain RMSE, raw-domain MAE, and PMSE.
- Row-KL/Row-CPC evaluate row-normalized allocation; RMSE/MAE evaluate raw OD counts; PMSE evaluates row-centered oracle potential.

## Main Findings

- Q1 training scale heterogeneity: ScaleAwareMLP Row-KL=0.3739.
- Q2 opportunity redistribution: OpFlow Row-KL=0.2492.
- Q3 component-correlation shift: OpFlow Row-KL=0.2539.
- Q4 load-bearing ablation: NoEnvSelection Row-KL=0.4010.

## Interpretation Guardrails

- Q1 varies train/validation origin-scale dispersion while holding oracle allocation fixed; it tests raw-count supervision rather than a pure test-time scale shock.
- Q2 tests equivariant adaptation to opportunity redistribution.
- Q3 uses admissible opportunity/accessibility variables and no explicit shortcut residual.
- Q4 identifies load-bearing components under controlled OOD stress.

## Deliverables

- `q1_scale`: `results/q1_scale/metrics.csv`
- `figures/synthetic_main.png`
- `figures/synthetic_main.pdf`
- `tables/synthetic_endpoint.csv`
