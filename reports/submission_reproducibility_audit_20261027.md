# OpFlow 投稿复现审计（2026-10-27）

审计对象：本地当前工作区、GitHub main、已有运行产物，以及 [arXiv:2607.03200v1（2026-07-03）](https://arxiv.org/html/2607.03200v1#S5.T2)。该公开稿含用户指出的表2数字；公开稿与实际投稿文件是否完全相同仍无独立提交记录证明。

**结论：GitHub 不是可验证的投稿代码快照；本地材料可以恢复相当部分实验，但尚不足以完整重建表2。不能把问题仅归为一个平均数笔误。**

## 1. 代码版本、配置、命令与模型选择

实时 Git 远端 main/HEAD 与本地 HEAD 均为 `d32a49dc048477711288526ea096dcc60b8872e2`（2026-06-12，`real-wolrd data add`），未发现远端 tag。这个提交的真实实验配置仍为 `random / matched_shortcut_high_to_low / matched_opportunity_low_to_high`；当前 RandSP/GeoSP/OppSP 生成代码与清单尚未进入该提交。当前工作区亦有大量未提交修改。因此该 GitHub commit 不能作为公开稿三协议实验已完整发布的证据。

已有代码压缩包：`OpFlow_minimal_runnable_20260615.zip`（历史本地包未上传；见本目录README），大小235,859,132字节，SHA-256：`ebc9b1eb227593b901b16b78ff1dbbe894030b5edfe7ed0a7a32cd3112b1c998`。包内 `PACKAGE_MANIFEST.json` 标记用途为最小可运行包，排除了 outputs/results/figures/tables/reports，也没有投稿 commit 绑定。包内真实实验配置与当前工作区不一致；不能将其改称已核实的投稿版本。

下面是数值追溯结果。吻合只代表已存指标按显示精度吻合，不能单独证明代码快照一致：

| 表2模型 | 本地来源 run | 三个协议的9个指标全部吻合 | GeoSP口径 |
| --- | --- | --- | --- |
| Gravity | realworld_classical_si_3splits_geosp9_20260619 | 是 | 9折均值 |
| Int. Opp. | realworld_classical_si_3splits_geosp9_20260619 | 是 | 9折均值 |
| Radiation | realworld_classical_si_3splits_geosp9_20260619 | 是 | 9折均值 |
| RF | realworld_rf_gbrt_official_gm_seed2026_geosp9 | 是 | 9折均值 |
| GBRT | realworld_rf_gbrt_official_gm_seed2026_geosp9 | 是 | 9折均值 |
| GM | realworld_rf_gbrt_official_gm_seed2026_geosp9 | 是 | 9折均值 |
| PairMLP | realworld_main_strict_seed2026_all_geosp | 否，见审计JSON | 原始记录不匹配 |
| ScaleMLP | realworld_seed2026_core3splits_20260613 | 否，见审计JSON | 原始记录不匹配 |
| DGM | realworld_remote_deepgraph_cuda_seed2026_geosp9 | 是 | 9折均值 |
| NetGAN | realworld_remote_deepgraph_cuda_seed2026_geosp9 | 是 | 9折均值 |
| DiffODGen | realworld_remote_deepgraph_cuda_seed2026_geosp9 | 是 | 9折均值 |
| WeDAN | realworld_remote_deepgraph_cuda_seed2026_geosp9 | 是 | 9折均值 |
| GMEL | realworld_remote_deepgraph_cuda_seed2026_geosp9 | 是 | 9折均值 |
| OpFlow | realworld_opflow_strong_seed2026_3splits_20260619 | 否，见审计JSON | SouthAtlantic单折（最接近记录） |

- 可核对配置：每个本地完整 run 的 `resolved_config.yaml`，及 `artifacts/real_world_models/{split}/{model}/seed_2026/config.json`。
- 已整理 [本地模型任务配置目录](../reproducibility/submission_audit_20261027/local_run_config_catalog.json)（190条，去掉大体积normalizer数组，原文件路径与哈希保留），及 [本次三模型补算配置](../reproducibility/submission_audit_20261027/available_local_effective_configs.json)。
- 保存配置副本在 [saved_configs](../reproducibility/submission_audit_20261027/saved_configs)；基于保存配置重建的命令在 [reconstructed_commands.txt](../reproducibility/submission_audit_20261027/reconstructed_commands.txt)。这些不是捕获到的原始 shell argv，也不是投稿复现保证；本次没有执行训练命令。
- **表2深度5模型**的可吻合来源为 `realworld_remote_deepgraph_cuda_seed2026_geosp9`；其本地目录只有 `metrics.csv`，缺相应配置、日志和checkpoint。不能拿 `realworld_main_strict_seed2026_all_geosp` 的配置替代远程版本。

目前可核实的模型选择行为（以源码和已存配置为依据）：

| 模型 | 选择行为 |
| --- | --- |
| OpFlow strong | 训练县按 Census division 分组，在最多4个内部留出组上用 `alloc_loss + 0.1 * scale_loss` 选择epoch，再对全训练县训练所选epoch；Rand/Geo SouthAtlantic/Opp分别选3/4/3 epoch。最多8 epoch，LEO patience=3，要求所有选定折覆盖该epoch。不是完整9折外层GeoSP实验。 |
| PairMLP、ScaleMLP | 验证县上复合损失 `share + 1.0*scale log-MSE + 0.1*count log-MSE` 最小的epoch；最多20epoch、patience=15、hidden_dim=32、lr=.001、weight_decay=1e-5。它们不因共享YAML写了 `selection_protocol: leo` 就实际执行LEO。 |
| Gravity | 训练数据拟合固定岭回归，无验证集模型选择。 |
| Intervening Opportunity | 验证损失选择参数并early stop。 |
| Radiation | 固定allocation形式，训练只估计mean outflow。 |
| RF、GBRT | 当前实现以固定estimator参数拟合训练对，fit未使用验证县。 |
| GM（official_gm_e） | 当前Torch实现按训练损失保留参数；history的val_loss实际写入train_loss，不能解读成验证选择。 |
| 表2远程深度5模型 | 本地缺对应配置/日志，无法据现有源码确认其当时的最终配置与选择规则。 |

详见 [opflow_real.py](../reproducibility/submission_audit_20261027/source_snapshot/opflow_benchmark/baselines/opflow_real.py)、[pair_mlp.py](../reproducibility/submission_audit_20261027/source_snapshot/opflow_benchmark/baselines/pair_mlp.py)、[scale_aware_mlp.py](../reproducibility/submission_audit_20261027/source_snapshot/opflow_benchmark/baselines/scale_aware_mlp.py)、[official.py](../reproducibility/submission_audit_20261027/source_snapshot/opflow_benchmark/baselines/official.py)；源码检查不替代历史代码快照。

## 2. RandSP / GeoSP / OppSP 实际县 ID

完整清单为 [splits.json](../reproducibility/submission_audit_20261027/evidence/data/opflow_large_commuting_od/splits.json) 和 [逐县CSV目录](../reproducibility/submission_audit_20261027/evidence/data/opflow_large_commuting_od/splits)。`area_id`是县FIPS字符串，`role`是train/val/test；读取时要保留前导零。

| 协议 | split种子/折 | 训练县 | 验证县 | 测试县 |
| --- | --- | --- | --- | --- |
| RandSP_seed0～4 | 0,1,2,3,4 | 1585 | 227 | 453 |
| GeoSP_SouthAtlantic | 验证seed=0 | 1625 | 181 | 459 |
| GeoSP其余8折 | 验证seed=0 | 见split_inventory.csv | 见清单 | 与SouthAtlantic合计覆盖2265县 |
| OppSP | seed=0 | 1585 | 226 | 454 |

生成脚本：[make_real_spatial_splits.py](../reproducibility/submission_audit_20261027/evidence/scripts/make_real_spatial_splits.py)；算法：[real_spatial_splits.py](../reproducibility/submission_audit_20261027/evidence/opflow_benchmark/real_spatial_splits.py)。生成入口命令（本次未运行；它会重写data_root内的划分文件，应只对数据副本执行）：

```bash
conda activate opflow
python scripts/make_real_spatial_splits.py --data_root /path/to/copied_opflow_large_commuting_od
```

注意：当前生成器即使指定 `--output_dir`，仍会写回data_root下的 `splits.json`、`split_summary.csv`、`split_balance.json`，不能靠这个参数隔离历史清单。

RandSP先分20% test，val再使用 seed+7919。GeoSP按完整Census division作test，剩余县中10%作val。OppSP按POI浓度HHI低/高10%尾部作test，两端各227县，POI列与阈值在 `OppSP_summary.json`。

已验证15个split的JSON和CSV完全一致、各split内角色互斥、9Geo测试折覆盖2265县且每县一次。6个完整run的190个逐县风险文件与当前清单精确相等；main_strict实际运行RandSP_seed0+9Geo+Opp共11split，模型训练seed=2026。定义5个RandSP种子不代表5个都运行过。证据：[split_audit.json](../reproducibility/submission_audit_20261027/split_audit.json)、[split_inventory.csv](../reproducibility/submission_audit_20261027/split_inventory.csv)。

公开稿提及3333县，本地实验支持集是2265县；尚未建立差额1068县的完整筛选溯源。需说明原始数据规模与最终有效样本量，不能默认两者相同。

## 3. 表2原始指标、日志和平均数冲突

详细逐格追溯（含原始未舍入值、CSV行号、source run）在 [table2_audit.json](../reproducibility/submission_audit_20261027/table2_audit.json)。原始逐任务记录位于各run的 `metrics.csv`、`artifacts/real_world_models/.../metrics.json`、`eval.json`；日志位于 `logs/real_world_progress.log` 及模型子目录 `train.log`。后者主要是轮次和耗时摘要，不应宣称拥有完整逐步训练日志。远程深度run只有metrics CSV。

提取与汇总代码是 [make_tables.py](../reproducibility/submission_audit_20261027/evidence/scripts/make_tables.py) 和 [realworld_upgrade.py](../reproducibility/submission_audit_20261027/evidence/opflow_experiments/pipelines/realworld_upgrade.py)。未找到能从固定run清单一键生成整张表2的唯一脚本或manifest。现有汇总按split/seed记录做无权重均值；11split直接平均会给GeoSP九倍权重，不能当三协议等权平均。现有summary CI是记录间1.96SE，不是配对县级CI。

PairMLP公开稿列出的三个Row-KL是.399/.392/.468：

`(.399 + .392 + .468) / 3 = .4196666667`，应约为 **.420**。若三分项均按三位小数四舍五入，其原数等权均值只能在 [.4191667,.4201667) 内，不能是表中的.413或正文的.410。

本地现存PairMLP与ScaleMLP记录本身也不吻合公开稿：

| 模型/来源 | RandSP | GeoSP SouthAtlantic | OppSP |
| --- | --- | --- | --- |
| PairMLP，core3或main_strict对应折 | .336770 | .378846 | .454752 |
| ScaleMLP，core3 | .338161 | .400304 | .477464 |
| OpFlow，strong | .322404 | .370750 | .438233 |

PairMLP九Geo折均值=.3892475111；Rand+该Geo均值+Opp的三协议均值=.3935898680；11split直接均值=.3904317903，均无法追溯.413/.410。OpFlow原数也未全部按统一常规四舍五入呈现，例如Geo=.3707497应为.371，而稿件写.370；仅凭接近不能证明来源。

其它平均也应复核：Pair Log-RMSE三分项均值约1.226667（表1.220），Scale Row-CPC约.680333（表.685），Scale Log-RMSE约1.198333（表1.201）。结论是**已证实表内算术/版本不一致，但尚无证据区分笔误、旧值残留或不同结果版本**；不宜只改Pair一个单元格。

## 4. 现成输出、补算和是否需要训练

已有三模型逐县 `eval.json` 包含Row-KL、Row-CPC、count-CPC、RMSE和两种NRMSE。本次使用 strong OpFlow + core3 PairMLP/ScaleMLP，严格按同一split/seed/县ID配对，并核验县节点数和真实总流量一致。它们是可检查的本地run，**不是已核实的投稿表2复现**。

| split | 模型 | count-CPC县均值 | RMSE县均值 | NRMSE/mean(y) | NRMSE/std(y) |
| --- | --- | --- | --- | --- | --- |
| RandSP_seed0 | OpFlow | 0.669256 | 71.401803 | 1.204468 | 0.801204 |
| RandSP_seed0 | PairMLP | 0.650149 | 91.893530 | 1.530106 | 1.057816 |
| RandSP_seed0 | ScaleMLP | 0.636530 | 106.896530 | 1.803126 | 1.259419 |
| GeoSP_SouthAtlantic | OpFlow | 0.646949 | 64.274970 | 1.405680 | 0.987799 |
| GeoSP_SouthAtlantic | PairMLP | 0.585031 | 107.818175 | 2.139758 | 1.580873 |
| GeoSP_SouthAtlantic | ScaleMLP | 0.545040 | 141.372997 | 2.781310 | 2.069927 |
| OppSP | OpFlow | 0.628434 | 68.269282 | 1.442354 | 0.785671 |
| OppSP | PairMLP | 0.613708 | 81.803280 | 1.651022 | 0.931080 |
| OppSP | ScaleMLP | 0.616042 | 84.778758 | 1.726947 | 0.987349 |

配对县级bootstrap采用20,000次、随机种子20261027、percentile 95%区间，县为重采样单位。下面为Row-KL差值（OpFlow减baseline）：

| split | 比较 | 差值 | 95%CI | 配对县数 |
| --- | --- | --- | --- | --- |
| RandSP_seed0 | OpFlow - PairMLP | -0.014366 | [-0.020614, -0.008826] | 453 |
| RandSP_seed0 | OpFlow - ScaleMLP | -0.015757 | [-0.021976, -0.009992] | 453 |
| GeoSP_SouthAtlantic | OpFlow - PairMLP | -0.008096 | [-0.013520, -0.002663] | 459 |
| GeoSP_SouthAtlantic | OpFlow - ScaleMLP | -0.029554 | [-0.036801, -0.022311] | 459 |
| OppSP | OpFlow - PairMLP | -0.016519 | [-0.021976, -0.011073] | 454 |
| OppSP | OpFlow - ScaleMLP | -0.039231 | [-0.048023, -0.030736] | 454 |

全指标及区间：[available_local_model_summary_ci.csv](../reproducibility/submission_audit_20261027/available_local_model_summary_ci.csv)；配对差值：[available_local_paired_county_ci.csv](../reproducibility/submission_audit_20261027/available_local_paired_county_ci.csv)；4098条逐县记录：[available_local_county_metrics.csv](../reproducibility/submission_audit_20261027/available_local_county_metrics.csv)。复算脚本：[audit_available_county_outputs.py](../reproducibility/submission_audit_20261027/provenance/original_local_audit.py)。

这些区间固定训练seed=2026，反映县抽样不确定性，不反映重训随机性；GeoSP仅SouthAtlantic；不跨重叠协议合并重采样，也未做空间块bootstrap。`global_cpc`是县内count-CPC，表中是县等权平均，不是将所有OD合并后的CPC。NRMSE两种分母明确分列。沿用历史评估：count指标在完整n×n矩阵上计算（含对角和零行）；allocation使用valid_origin。若审稿人要求不同mask，需要重新推理评估，不能重命名旧指标冒充。

未找到真实实验逐OD预测文件；`predictions_summary.json`仅是汇总。**但9/9相关checkpoint已用weights_only=True安全加载并完成抽样推理**，每split抽一个8节点县，六项指标与历史eval.json完全一致、最大绝对误差0。证据：[inference_probe.json](../reproducibility/submission_audit_20261027/inference_probe.json)。全量逐OD推理原始审计未运行；发布包提供 `export_od_predictions.py` 供合作者直接从9个checkpoint导出。

如需逐OD表，可由现有checkpoint和数据生成 `split / model / seed / county_id / origin_GEOID / destination_GEOID / y_true / y_pred / valid_origin / count_eval_mask / allocation_eval_mask`。`region_meta.csv`提供node_index到GEOID映射；mask需按明确评估口径生成。**应优先推理导出，无需先重训**。只有找不到对应投稿模型、缺少Geo折checkpoint或无法恢复协议时，才考虑补跑相应实验。

## 交付与未解决项

原始审计未修改模型、论文或历史结果，也未训练。后续按用户要求发布协作材料，保留上述审计结论。已保存来源哈希、配置副本、原始指标CSV、划分ID和日志副本。材料包是“审计证据包”，不是“投稿版代码包”。

仍需补齐：PairMLP/ScaleMLP投稿数值来源；OpFlow其余8个Geo外层折或将全表统一成同一折；远程深度模型对应代码/config/log；真实投稿代码快照和实际投稿文档绑定；3333→2265县的筛选说明。获得这些证据后再冻结统一表2与正式复现包。


## GitHub协作包说明

请从 [README](../reproducibility/submission_audit_20261027/README.md) 开始。使用 `recompute_county_ci.py` 从已发布逐县CSV复算；`provenance/original_local_audit.py` 是历史本地脚本快照，依赖未发布的大体积原始eval.json，不能作为独立入口。证据文件中的历史路径和哈希指向原始工作区，发布文件的校验以 `SHA256SUMS` 为准。当前源码快照仅用于复核现存checkpoint，并非已证实的投稿代码。
