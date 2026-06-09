# OpFlow P-only Latent SIT Operator Report

## Model

默认 OpFlow 实现为：

```text
P-only inputs
  -> latent SIT operators: o_tilde, c_tilde, s_tilde, ell_tilde
  -> additive operator potential
  -> row-centered softmax allocation pi_hat
  -> detached-allocation count reconstruction
```

Synthetic path: `opflow_experiments/exposure_models.py`

Real-world path: `opflow_benchmark/baselines/opflow_real.py`

Core operators:

```python
o_tilde_j = softplus(H_o(P_dest_j)) + eps
c_tilde_ij = softplus(H_c(P_ori_i, P_dest_j, P_pair_ij, P_geo_ij)) + eps
s_tilde_ij = prefix_sum(o_tilde_k ordered by c_tilde_ik, excluding j)
ell_tilde_j = destination_side_kernel(o_tilde, geography)
```

Potential:

```python
psi_ij = (
    D_theta(c_tilde_ij)
    + B_theta(o_tilde_j)
    + S_theta(s_tilde_ij)
    + C_theta(ell_tilde_j)
    + M_theta(P_ori_i, P_geo_ij)
)
psi_bar_ij = psi_ij - mean_j(psi_ij)
pi_hat_ij = softmax(psi_bar_i / tau)
lambda_hat_ij = scale_hat_i * pi_hat_ij
```

Removed from default OpFlow:

```text
x_res_* inputs
unrestricted raw pair concat potential
training shortcut heads
LeakyOpFlow / NoRouting / NoDiagnostic
```

## Synthetic DGP

The active synthetic data generator is the single final `DGP`.

P mapping:

```text
P_ori  = x_origin_*
P_dest = x_geo_0, x_geo_1, x_geo_3
P_pair = x_geo_2, x_geo_4, cost
P_geo  = dest_x - origin_x, dest_y - origin_y, distance
```

`x_res_*` is generated only for unrestricted/raw shortcut baselines such as
`ShortcutOnly`, `RawCountMLP`, and `RawCountGNN`. Default `OpFlow` does not
scale, tensorize, or read it. Oracle latent columns such as `o_true`,
`s_true`, `ell_true`, `g_oracle`, `psi_oracle`, `pi_oracle`, and
`lambda_oracle` are diagnostics/evaluation columns only.

## Training

Synthetic and real-world OpFlow use allocation-first training:

```python
L_alloc = max_e R_alloc[e] + lambda_vrex * var_e(R_alloc[e])
L_scale = max_e R_scale[e]
L_count = d_count(log1p(scale_hat * pi_hat.detach()), log1p(y))
L = L_alloc + lambda_scale * L_scale + lambda_count * L_count + lambda_op * L_op
```

Defaults:

```yaml
lambda_vrex: 0.1
lambda_op: 0.01
use_groupdro: true
use_rank_exposure: true
use_competition: true
use_opportunity_field: true
use_row_center: true
detach_count_allocation: true
use_sit_operator_potential: true
use_monotone_penalty: true
```

Monotonicity penalty:

```text
D_theta: penalize positive derivative
S_theta: penalize positive derivative
B_theta: penalize negative derivative
C_theta: sign-flexible
```

Real-world environments are areas. Early stopping uses the validation allocation objective.

## Variants

Kept OpFlow variants:

```text
OpFlow / opflow_full
NoRankExposure / opflow_no_rank_exposure
NoCompetition / opflow_no_competition
NoOpportunityField / opflow_no_opportunity_field
NoScaleDetach / opflow_no_scale_detach
NoScaleBranch / opflow_no_scale_branch
NoExposureOps / opflow_no_exposure_ops
NoRowCenter / opflow_no_row_centering
GenericPotentialMLP / opflow_generic_potential_mlp
NoGroupDRO / opflow_no_groupdro
NoVREX / opflow_no_vrex
NoMonotonePenalty / opflow_no_monotone_penalty
```

Removed variants raise `ValueError`.

## Metrics And Artifacts

Primary metrics:

```text
row_kl
row_cpc
rank_corr
topk_overlap / ndcg@k
log_rmse
global_cpc
```

Operator/training diagnostics:

```text
alloc_risk_max
alloc_risk_mean
vrex_alloc_var
scale_risk_max
scale_risk_mean
op_monotone_violation_D
op_monotone_violation_S
op_monotone_violation_B
operator_mean_o_tilde
operator_mean_c_tilde
psi_bar_row_mean_abs
gate_opportunity_corr
gate_distance_corr
```

Run outputs:

```text
resolved_config.yaml
metrics.csv
summary.json
summary.md
artifacts/DGP/*/env_meta.csv
artifacts/DGP/*/gradient_audit.csv
operator_diagnostics.csv for real-world upgrade reports
```

## Commands

Unit tests:

```bash
python -m unittest discover tests
```

Synthetic debug:

```bash
conda run --no-capture-output -n opflow python experiments/synthetic/run_synthetic_main.py \
  --config configs/synthetic/DGP.yaml \
  --mode debug \
  --run-id DGP_debug
```

Real-world debug:

```bash
conda run --no-capture-output -n opflow python -u experiments/real_world/run_large_commuting_od.py \
  --config configs/real_world/large_commuting_od.yaml \
  --debug \
  --run-id real_sit_refactor_debug
```
