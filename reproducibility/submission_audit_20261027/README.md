# OpFlow reproducibility audit and collaborator materials

This bundle shares the **available local experiment evidence audited on 2026-10-27**. It is **not a verified submission-code release or a corrected Table 2**. The audit compared local files with [arXiv:2607.03200v1](https://arxiv.org/html/2607.03200v1#S5.T2) and GitHub commit `d32a49dc048477711288526ea096dcc60b8872e2`.

合作者请先读[中文审计报告](AUDIT_REPORT.md)。本目录包含已核验的县清单、原始指标及日志、保存配置、4098条逐县记录、配对置信区间、9个checkpoint和可直接运行的复算／推理脚本。它不能作为“已经确认与投稿版本一致”的声明。

## Evidence and limitations

- PairMLP and ScaleMLP's saved results do **not** match the published Table 2 entries. The PairMLP displayed Row-KL values average to 0.4196667, not 0.413 or 0.410. Other table averages also need reconciliation.
- The closest available OpFlow run contains **GeoSP South Atlantic only**, while the traceable classical, tree/GM and deep-baseline Table 2 results use nine GeoSP folds. Do not compare these as if they were the same protocol.
- The remote deep-baseline run has a local metrics CSV, but its corresponding configuration, training logs and checkpoints were not recovered. Local CPU baseline configs must not substitute for those missing artifacts.
- The saved three-model comparison uses one training seed, **2026**. The 95% intervals resample paired counties, not model-training seeds; no spatial-block uncertainty correction is applied.
- The split universe contains **2265 counties**; the paper mentions 3333. A complete record of the eligibility/filtering difference remains missing.
- The source snapshot is the local code available at audit time. It is included to operate the saved checkpoints and inspect the documented behavior; it is **not** an authenticated historical submission snapshot.

## Contents

| Material | Location |
| --- | --- |
| Chinese audit report | [AUDIT_REPORT.md](AUDIT_REPORT.md) |
| Table 2 cell-level provenance and discrepancies | [table2_audit.json](table2_audit.json) |
| Full train/validation/test county IDs | [splits.json](evidence/data/opflow_large_commuting_od/splits.json), [per-county CSVs](evidence/data/opflow_large_commuting_od/splits/) |
| Split seeds, counts and run checks | [split_inventory.csv](split_inventory.csv), [split_audit.json](split_audit.json) |
| 4098 model/split/seed/county observations | [available_local_county_metrics.csv](available_local_county_metrics.csv) |
| County means and 95% intervals | [available_local_model_summary_ci.csv](available_local_model_summary_ci.csv) |
| Paired differences and 95% intervals | [available_local_paired_county_ci.csv](available_local_paired_county_ci.csv) |
| Saved configurations and selection records | [saved_configs/](saved_configs/), [local_run_config_catalog.json](local_run_config_catalog.json), [available_local_effective_configs.json](available_local_effective_configs.json) |
| Source metrics and logs | [evidence/results/](evidence/results/), [evidence/outputs/runs/](evidence/outputs/runs/), [evidence/tables/](evidence/tables/) |
| Three models × three splits, seed 2026 | [checkpoints/](checkpoints/), [checkpoint_manifest.json](checkpoint_manifest.json) |
| Model/runner/source snapshot and split generator | [source_snapshot/](source_snapshot/), [evidence/scripts/make_real_spatial_splits.py](evidence/scripts/make_real_spatial_splits.py) |
| Checkpoint validation | [inference_probe.json](inference_probe.json) |
| Portable CI recomputation | [recompute_county_ci.py](recompute_county_ci.py) |
| Portable OD prediction export | [export_od_predictions.py](export_od_predictions.py) |
| Reconstructed training commands | [reconstructed_commands.txt](reconstructed_commands.txt) |
| Tested runtime versions | [tested_environment.json](tested_environment.json) |
| Published file integrity | [SHA256SUMS](SHA256SUMS) |

The OpFlow checkpoints come from `realworld_opflow_strong_seed2026_3splits_20260619`; PairMLP and ScaleMLP come from `realworld_seed2026_core3splits_20260613`. All use exactly matching test counties within each of `RandSP_seed0`, `GeoSP_SouthAtlantic`, and `OppSP` (453, 459 and 454 counties respectively).

## Recompute the confidence intervals

From the repository root, with Python 3.10+ and NumPy/pandas installed:

```bash
conda activate opflow
OPFLOW_AUDIT_BUNDLE=reproducibility/submission_audit_20261027
python "$OPFLOW_AUDIT_BUNDLE/recompute_county_ci.py" \
  --output-dir outputs/recomputed_county_ci
```

This needs only the published county CSV and split IDs: no dataset, training or checkpoint loading. It uses 20,000 paired county-bootstrap draws, seed 20261027, and percentile 95% intervals. The published 63 summary rows and 42 paired intervals were reproduced byte for byte in the recorded environment. Choose a new output directory for each run; the script does not overwrite existing outputs.

For an extracted archive, run `python /path/to/bundle/recompute_county_ci.py --output-dir /path/to/new_output` or pass `--bundle-dir` explicitly.

## Export OD predictions without training

The converted dataset is in the repository's `data/opflow_large_commuting_od/`. Use the bundled model source and the nine saved checkpoints. The script defaults to CPU and `torch.load(weights_only=True)`; there is no unrestricted pickle fallback. The recorded environment used PyTorch 2.1.0. See [environment-opflow.yml](source_snapshot/environment-opflow.yml) for the project's environment specification and [tested_environment.json](tested_environment.json) for actual versions used in validation.

Small verification export (one county per model/split):

```bash
OPFLOW_AUDIT_BUNDLE=reproducibility/submission_audit_20261027
python "$OPFLOW_AUDIT_BUNDLE/export_od_predictions.py" \
  --data-root data/opflow_large_commuting_od \
  --output-dir outputs/shared_od_probe \
  --county-ids 01093 13137 21121 \
  --max-counties 1
```

Export all available test counties:

```bash
python "$OPFLOW_AUDIT_BUNDLE/export_od_predictions.py" \
  --data-root data/opflow_large_commuting_od \
  --output-dir outputs/shared_od_predictions
```

Optional filters: `--models OpFlow PairMLP`, `--splits OppSP`, or `--max-counties 10`. The script writes one compressed CSV per county and a provenance/metric-check manifest. It checks the saved county metrics before exporting. Full export was not run as part of publication; the 9-checkpoint, 576-OD-cell smoke export matched the historical county metrics.

Fields are `source_run`, `split`, `model`, `seed`, `county_id`, `origin_id`, `destination_id`, `y_true`, `y_pred`, `valid_origin`, `count_eval_mask`, and `allocation_eval_mask`. IDs are strings: preserve their leading zeros when reading CSVs.

Metric/mask semantics match the saved evaluations:

- Count metrics include the complete county OD matrix, including diagonal cells and zero-outflow origins; `count_eval_mask=1` for every exported cell.
- Allocation metrics use `valid_origin` broadcast across destinations; diagonal destinations are retained.
- `global_cpc` means count-CPC within each county, followed by equal-county averaging. It is not a pooled all-counties OD CPC.
- `nrmse_mean=RMSE/mean(y)`; `nrmse_official=RMSE/std(y)`. Both definitions are included explicitly.

Do not silently change masks or replace learned-scale predictions with oracle-scale outputs when comparing to the saved evidence.

## Provenance and training replay

Historical JSON/config/log contents are preserved, including original local paths. Those paths are provenance, not runnable paths on another computer. `reconstructed_commands.txt` supplies bundle-relative configs and overrides the dataset/split locations. Its commands would **train** models and are reconstructed from saved configurations, not captured original shell commands; no training replay was performed during publication.

`provenance/original_local_audit.py` is the original audit-script snapshot. It depends on large original `eval.json` files that are not bundled. Use `recompute_county_ci.py` for portable recomputation. The extracted county CSV, source metrics, source hashes, saved configs and checkpoints are included instead of duplicating all large train/validation evaluation JSONs.

`evidence_manifest.json` and other historical manifests retain original source-file hashes and may refer to files that are represented by extracted evidence here. `SHA256SUMS` checks the files actually published in this bundle:

```bash
cd reproducibility/submission_audit_20261027
shasum -a 256 -c SHA256SUMS
```

The historical 235 MB `OpFlow_minimal_runnable_20260615.zip` is documented in the audit but is not attached here: it is not a verified submission release. This smaller collaborator bundle contains the audited material, the relevant code snapshot and the verified saved checkpoints.
