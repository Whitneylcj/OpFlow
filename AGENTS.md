# AGENT.md

## Project

This repository supports the paper:

**OpFlow: Learning Opportunity-Invariant Representations for Spatial Flow Prediction**

The goal is robust spatial flow prediction under environment shift.

## Core Idea

Standard OD/GNN predictors may fit training data by exploiting shortcuts such as distance, historical flow, static topology, source scale, or destination popularity. These signals can be predictive in-distribution but brittle when opportunity landscapes shift.

OpFlow instead aims to learn **opportunity-supported destination allocation**.

The paper does **not** claim full causal identification. It uses opportunity structure as a domain-informed, testable inductive bias.

## Main Theoretical Logic

The paper follows this chain:

1. Observational OD logs cannot identify an unrestricted stable flow mechanism when opportunity features and shortcut features are entangled.
2. Distance/topology can mimic opportunity structure in training environments, giving good ID performance but poor OOD robustness.
3. A more deployable target is row-normalized opportunity allocation:
   \[
   \lambda_{ij}^e =
   a_i^e
   \frac{\rho(U_{ij}^e)}
   {\sum_k \rho(U_{ik}^e)}.
   \]
   Here \(a_i^e\) is origin demand scale, while the normalized term is destination allocation.
4. OpFlow should approximate this opportunity-driven allocation rule, not just raw OD counts.

## Method Focus

OpFlow should be treated as an **opportunity-constrained graph learning framework**, not just another OD-GNN.

Core components:

- opportunity-aware representation;
- opportunity-guided gated propagation;
- scale--allocation decoder;
- shortcut-control diagnostics or regularization.

The graph module should help only when relational opportunity structure exists. Avoid making the method look like a generic GNN capacity increase.

## Experiments

Experiments should validate the theory, not only rank models.

Main synthetic experiments:

1. **Shortcut break**: distance/topology works ID but fails OOD when opportunity-distance alignment changes.
2. **Scale drift**: raw counts drift, but row-normalized allocation and ranking are more stable.
3. **Relational opportunity**: graph propagation helps when opportunity is relational, and helps less when it is not.
4. **Gate regularization**: opportunity gates should align with opportunity signals and avoid distance/source shortcuts.

Appendix experiments:

- negative control when opportunity invariance fails;
- sample size / noise / shift-strength sweeps;
- component ablations.

Expected conclusion:

OpFlow should improve in regimes predicted by the opportunity-invariance theory, and weaken when the assumption is violated.

## Metrics

Do not report only RMSE. Always include allocation-oriented metrics when possible:

- ID / OOD RMSE;
- Row-KL;
- ranking correlation;
- top-k destination overlap;
- gate-opportunity alignment;
- gate-distance correlation;
- gate-source-scale or degree correlation.

## Coding Rules

Use:

- `experiments/` for runnable experiment scripts;
- `configs/` for experiment configs;
- `opflow_experiments/` for reusable source code;
- `outputs/` for results.

Avoid duplicated experiment logic. Root-level `experiments/` should be the canonical entry point.

Every run should save:

- resolved config;
- metrics CSV;
- summary JSON or markdown;
- figures/tables if generated.