# AGENT.md

## Project

This repository supports the paper:

**OpFlow: Learning Opportunity-Conditioned Choice Potentials for Robust OD Flow Prediction**

The goal is robust origin-destination (OD) flow prediction under environment shift.

## Core Idea

OpFlow treats OD prediction as a scale-allocation problem. A raw OD row mixes two objects:

\[
\lambda_{ij}^e = R_i^e \pi_{ij}^e ,
\]

where (R_i^e) is the origin production scale and (\pi_i^e) is the destination allocation. Raw-count supervision can fit origin scale, destination popularity, and unstable cross-environment correlations without recovering the transferable choice mechanism.

OpFlow instead learns an **opportunity-conditioned exposure-to-choice law**. Spatial exposures may shift and allocations may change, but the conditional response from exposure states to row-centered choice potentials is the deployable target within a specified environment family.

The paper does **not** claim full causal identification or universal human mobility laws. It uses spatial interaction theory and opportunity structure as domain-informed inductive bias. Robustness claims should always be stated under exposure coverage and bounded residual drift.

## Main Theoretical Logic

The paper follows this chain:

1. The OD conditional mean decomposes into origin production scale and destination allocation:
   [
   \lambda_{ij}^e = R_i^e \pi_{ij}^e.
   ]

2. The identifiable allocation mechanism is not an uncentered utility but a row-centered choice potential:
   [
   \psi_i^e = H_i g_i^e,
   \qquad
   \pi_i^e = \mathrm{softmax}(\psi_i^e/\tau).
   ]

3. The transferable object is the exposure-to-choice response:
   [
   \psi_i^{e,*}=H_iF^\star(Z_i^e)+\Delta_i^e,
   ]
   where (Z_i^e) contains opportunity, impedance, intervening exposure, local field, origin context, and geographic covariates. The residual (\Delta_i^e) captures bounded environment-specific structural drift.

4. Classical spatial interaction laws are restricted log-potential cases. OpFlow neuralizes this potential while preserving the scale-allocation and row-centering structure.

5. OOD robustness is not claimed for arbitrary shifts. It depends on covered exposure shifts and limited unexplained residual drift.

## Paper-Facing Terminology

Use the following terms consistently:

* **origin production scale** (R_i^e)
* **destination allocation** (\pi_i^e)
* **choice intensity** (A_{ij}^e)
* **log-intensity / uncentered potential** (g_{ij}^e) or (u_{\theta,ij}^e)
* **row-centered potential** (\psi_i^e=H_i g_i^e)
* **exposure state** (Z_{ij}^e=(m_i^e,o_j^e,c_{ij}^e,s_{ij}^e,\ell_j^e,\ldots))
* **exposure-to-choice law** (F^\star)
* **scale-isolated reconstruction**
* **cross-environment validation / LEO selection**

Avoid these outdated or misleading terms unless explicitly needed for legacy code:

* opportunity-invariant representation
* invariant allocation
* generic OD-GNN
* shortcut routing as a core method
* gate regularization as a core method
* opportunity-guided gated propagation

## Method Focus

OpFlow should be treated as a **mechanism-constrained OD prediction framework**, not as a generic graph neural network with more capacity.

Core components:

1. **Exposure construction.** Build opportunity-conditioned exposure states from admissible spatial inputs, including opportunity proxies, travel impedance, rank-prefix exposure, local opportunity field, origin context, and geographic pair features.

2. **Operator-structured choice potential.** Map exposure states to uncentered choice scores through neural operators aligned with spatial interaction theory.

3. **Row-centering and allocation.** Apply (H_i) to remove row-wise offsets and obtain destination allocation by softmax.

4. **Scale-isolated reconstruction.** Predict origin production scale using a separate branch. Use stop-gradient where needed so count/scale losses do not corrupt the allocation potential.

5. **Environment-robust training.** Optimize stable allocation risk across environments and use robust penalties when configured.

6. **Cross-environment model selection.** Select candidates by held-out environment allocation risk, not random row-wise validation. Leave-one-environment-out (LEO) is the special case where each validation fold contains one complete environment.

The graph or spatial operator module should help only when relational opportunity structure exists. Do not describe improvements as generic GNN capacity gains.

## Synthetic Experiments

Synthetic experiments should validate theory-driven mechanisms, not only rank models.

Use one unified DGP and organize experiments with the following question identifiers:

### Q1: Scale-target disentanglement

Question: Does row-normalized potential learning remove origin-scale nuisance that misleads raw-count supervision?

Design: Vary only the origin production scale (R_i^e) in test environments while keeping the oracle allocation fixed.

Interpretation: This is a controlled counterfactual diagnostic, not a claim that real-world scale shifts occur alone. OpFlow need not uniquely dominate all allocation-supervised methods here. The expected result is that raw-count models degrade, while allocation-targeted methods remain stable.

### Q2: Opportunity redistribution

Question: Can a shared exposure-to-choice law adapt equivariantly when destination opportunities are redistributed?

Design: Shift the spatial distribution of the opportunity field (o_j^e) while keeping the oracle response function (F^\star) fixed.

Interpretation: Allocation is allowed to change. The expected result is that models relying on stable shares degrade, while exposure-conditioned models adapt better.

### Q3: Component-correlation shift

Question: Can structured exposure modeling separate opportunity attraction from travel impedance when their training correlation breaks?

Design: In training environments, make opportunity positively correlated with accessibility or low cost. In test environments, weaken or reverse this correlation while keeping all variables admissible and the oracle signs fixed.

Interpretation: This replaces the old explicit shortcut-residual experiment. The goal is to test robustness to unstable correlations among admissible components, not to rely on manually excluded shortcut features.

### Q4: Load-bearing ablation

Question: Which OpFlow components are necessary under controlled OOD stress?

Design: Evaluate ablations under strong Q1/Q2/Q3 stress or a clearly defined average OOD stress setting.

Recommended ablations:

* `NoRankPrefix`: remove rank-prefix exposure (s_{ij})
* `NoLocalField`: remove local opportunity field (\ell_j)
* `NoScaleIsolation`: remove scale/allocation gradient isolation
* `NoEnvSelection`: replace cross-environment selection with training loss or random row-wise validation
* `NoMonotonicity`: remove monotonicity penalty, optional if runtime is limited

Do not use explicit `ShortcutOnly`, `LeakyOpFlow`, or `NoRouting` as main synthetic baselines unless they are clearly marked as appendix diagnostics.

## Synthetic DGP Requirements

Use one updated DGP generator for all four questions.

Basic structure:

* sample one shared set of (N=80) spatial units per seed;
* each unit can act as both origin and destination;
* exclude self-loops from candidate sets;
* each origin has a candidate destination set with (K_i) sampled around 25 and clipped to a fixed range;
* all row-wise operations must use the candidate mask.

The oracle count generation must follow:

[
\lambda_{ij}^e=R_i^e\pi_{ij}^{e,*},
\qquad
Y_{ij}^e\sim\mathrm{Poisson}(\lambda_{ij}^e).
]

The oracle potential should use the simplified four-component structure:

[
g_{ij}^{e,*}
============

## a_i O_j^e

## b_i C_{ij}

\beta_s S_{ij}^e
+
\beta_\ell L_j^e
+
\Delta_{ij}^e.
]

Here (O), (C), (S), and (L) denote standardized opportunity, cost, rank-prefix exposure, and local field components. Standardization statistics must be fitted on training environments only and then reused for validation and test environments.

Deployed models must not read oracle latent variables:

* true (o_j^e)
* true (s_{ij}^e)
* true (\ell_j^e)
* oracle (g_{ij}^{e,*})
* oracle (\psi_{ij}^{e,*})
* oracle (\pi_{ij}^{e,*})
* oracle (\lambda_{ij}^{e,*})

They may read admissible proxies:

* origin coordinates
* destination coordinates
* travel cost / distance
* displacement ((dx,dy))
* cost rank fraction
* row-standardized cost
* noisy opportunity proxy

## Metrics

Do not report only RMSE. Always include allocation-oriented metrics.

Main paper metric:

* Row-KL

Recommended synthetic metrics:

* Row-KL
* Row-CPC
* PMSE for row-centered potential recovery
* Log-RMSE for count reconstruction
* Global-CPC
* RankCorr or NDCG@K when ranking quality is discussed

PMSE must be computed only after row-centering both predicted and oracle potentials and averaging over valid candidates.

Outdated metrics such as gate-opportunity alignment, gate-distance correlation, and gate-source-scale correlation should not be used as main evidence unless a gate module is explicitly part of the current method.

## Main Baselines

Use diagnostic baselines that correspond to theory questions.

Recommended main baselines:

* `OpFlow`
* `RawCountMLP`
* `PairAllocMLP`
* `ScaleAwareMLP`
* `ShareStable` or `FixedShare` for Q2
* OpFlow ablations for Q4

Classical Gravity/Radiation baselines may be reported in appendix or as secondary references if implemented cleanly.

Paper-facing names must be consistent:

* `PairAllocMLP` may be displayed as `PairMLP` only if defined once.
* `FullOpFlow` should be displayed as `OpFlow`.
* `NoCompetition` should be renamed to `NoLocalField` or `NoLocalContext`.
* `NoScaleDetach` should be displayed as `NoScaleIsolation`.
* `NoMonotonePenalty` should be displayed as `NoMonotonicity`.

## Required Synthetic Outputs

The canonical command should be:

```bash
python experiments/synthetic/run_synthetic_main.py --config configs/synthetic/DGP.yaml
```

The runner should support:

```bash
--questions q1_scale q2_opportunity q3_component_corr q4_ablation
--mode debug
--run-id <id>
```

Required paper-facing artifacts:

```text
results/q1_scale/metrics.csv
results/q2_opportunity/metrics.csv
results/q3_component_corr/metrics.csv
results/q3_component_corr/correlation_checks.csv
results/q4_ablation/metrics.csv

figures/synthetic_main.pdf
figures/synthetic_main.png

tables/synthetic_endpoint.csv
reports/synthetic_report.md
```

Also save run-level reproducibility artifacts under:

```text
outputs/runs/{run_id}/
```

Each run should save:

* resolved config
* random seeds
* raw per-split metrics
* model list and hyperparameters
* selected checkpoint/configuration
* diagnostics
* logs

## Coding Rules

Use:

* `experiments/` for runnable experiment scripts;
* `configs/` for experiment configs;
* `opflow_experiments/` for reusable source code;
* `outputs/` for run-level outputs;
* `results/`, `figures/`, `tables/`, and `reports/` for paper-facing artifacts.

Avoid duplicated experiment logic. Root-level `experiments/` should be the canonical entry point.

All experiment code must enforce:

* no oracle feature leakage;
* no shortcut residuals in main deployed features;
* train-only standardization for oracle components;
* masked row operations for variable candidate sets;
* reproducible seed handling;
* consistent model naming between CSV, reports, and figures.

## Validation Tests

Required unit or integration checks:

* shared support shape is (N=80);
* self-loops are excluded;
* candidate masks match valid candidates;
* all row-wise metrics respect candidate masks;
* train-only standardization is used for (O,C,S,L);
* Q1 changes test scale and counts but not oracle allocation;
* Q2 changes opportunity distribution and allocation;
* Q2 logs opportunity-accessibility correlation diagnostics;
* Q3 empirical opportunity-accessibility correlation follows target (\alpha);
* OpFlow and main baselines do not read oracle columns;
* OpFlow and main baselines do not read `x_res_*`;
* PMSE is computed after row-centering;
* `NoEnvSelection` differs from full OpFlow only in selection protocol;
* all required CSV, figure, table, and report files are generated.

Default checks:

```bash
python -m unittest discover tests
python experiments/synthetic/run_synthetic_main.py --config configs/synthetic/DGP.yaml --mode debug --questions q1_scale q2_opportunity q3_component_corr q4_ablation --run-id synthetic_four_debug
```

After running, scan generated artifacts and docs to ensure no old synthetic terminology remains, including:

* old E1/E2/E3/E4 labels
* `latent_opflow`
* `synthetic_upgrade`
* `exposure_synthetic`
* old shortcut-break main experiment wording
* gate-regularization main experiment wording

## Writing Rules

When generating paper text, keep the following interpretation constraints:

1. Q1 is a target-level diagnostic, not a realistic claim that scale shifts occur alone.
2. Q2 tests equivariant adaptation, not invariant allocation.
3. Q3 tests admissible component-correlation shift, not explicit shortcut exclusion.
4. Q4 tests load-bearing components and should not overclaim weak ablations.
5. OpFlow is not claimed to solve arbitrary behavioral response drift. If residual drift grows large, all methods may degrade.
6. The main claim is that OpFlow combines the correct allocation target, structured exposure modeling, scale-isolated reconstruction, and environment-level selection to improve robust allocation under controlled mechanism shifts.
