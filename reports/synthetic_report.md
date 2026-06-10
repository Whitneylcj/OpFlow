# Synthetic Four-Question Report

Run directory: `outputs/runs/synthetic_four_debug`

## DGP Settings

- suite/regime: `DGP`
- N: `18` shared origin/destination units; self-loops excluded.
- train/val/test envs: `2/1/1`
- oracle fields are diagnostics only; deployed models use coordinates, opportunity proxy, cost/rank, and row-standardized cost.
- RawCountMLP objective: `mse` raw-count reconstruction.

## Metrics

- Main: Row-KL against oracle allocation.
- Secondary: PMSE after row-centering predicted and oracle potentials by origin row.
- Appendix: log-RMSE, global CPC, Row-CPC, rank correlation, top-k overlap.

## Main Findings

- Q1 training scale heterogeneity: OpFlow Row-KL=0.6938.
- Q2 opportunity redistribution: OpFlow Row-KL=0.7237.
- Q3 component-correlation shift: OpFlow Row-KL=0.6703.
- Q4 load-bearing ablation: NoLocalField Row-KL=0.7466.

## Interpretation Guardrails

- Q1 varies train/validation origin-scale dispersion while holding oracle allocation fixed; it tests raw-count supervision rather than a pure test-time scale shock.
- Q2 tests equivariant adaptation to opportunity redistribution.
- Q3 uses admissible opportunity/accessibility variables and no explicit shortcut residual.
- Q4 identifies load-bearing components under controlled OOD stress.

## Deliverables

- `q1_scale`: `results/q1_scale/metrics.csv`
- `q2_opportunity`: `results/q2_opportunity/metrics.csv`
- `q3_component_corr`: `results/q3_component_corr/metrics.csv`
- `q4_ablation`: `results/q4_ablation/metrics.csv`
- `figures/synthetic_main.png`
- `figures/synthetic_main.pdf`
- `tables/synthetic_endpoint.csv`
