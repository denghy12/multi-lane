# 冻结旧完整功能：seed1/2 八任务 Validation 复核

日期2026-10-13。用户明确要求执行既定下一步。本轮只执行“旧功能单独可迁移性复核”；新残差、Test与其他扩展均为后续条件步骤，不自动启动。

## 目标与对照

复用现有seed1/2各自的八任务真实基线及TRAIN缓存；不重训CLIP、Selector、Prompt、Task Forward或分类头。与既有seed0构成三seed证据，不改变专家形式、数据划分、容量、损失或预算。

- 主比较：固定真实基线，对比只组合冻结旧完整功能（old_only）。
- 供体控制（fresh_only）：每任务在fit分区学习两个新完整专家，new5+new5；任务结束封存，从下一任务才允许读取。供体组不是新的候选方法，仍记录Validation/holdout作为来源诊断；不得依据其Validation选择供体。
- old_only：task0无旧专家，严格等于基线；task1–7冻结旧函数全部依赖，只学2t个任务固定输出系数，5epochs。无图像Router、无新专家适配阶段。供体训练成本单列，不声称各组等容量或供体免费。
- 位置：Task Forward最终归一化CLS特征后、固定Full/Person/Face特征融合前。Frozen Forward无新模块。Image-token Adapter与旧patch Projector关闭。三视图共享完整专家池和任务固定系数。

## 完整配置

- EMOTIC TRAIN/Validation，仅seed1、seed2，每seed任务0–7；同源基线训练预算30epochs/task但本轮不重训。
- 每任务两个rank32完整供体专家，16slots；保留出生任务A/B/P/norm/scale，输出组合而非参数矩阵组合。新P零输出起点，旧系数tanh(0)起点，旧函数/旧系数任务结束封存，旧任务不读取未来供体。
- TRAIN原图分组沿用seed0固定SHA256规则：`module-holdout-seed0-v1`，模5为0留出；模型seed变化不改变分区，同原图不同人物/level/任务不跨组。原基线曾看过完整TRAIN，所以称“新增模块留出”。holdout只诊断，不选择轮次、开关、供体或配置。
- 拟合：FP32 Adam，lr0.0004，weight_decay0，batch64，clip_norm1；每阶段5epochs、cosine重置。供体10epochs/task，旧系数5epochs/task（task0无更新）。排列seed=model_seed+1009*task+100000*phase_index。
- 损失：固定融合BCE + 0.1可靠单路辅助BCE，乘当前类别数/26；consistency0；总残差2%平滑范数界、fixed scale1。
- 固定融合：可靠Face时Full/Person/Face=0.64/0.16/0.20，否则0.80/0.20/0；Face条件有效、无歧义、short_side>=24、score>=0.6。
- Validation保持来源AMP/TF32=True、batch64、workers2，按原来源任务人群及已见类评价；最终任务26类作为主要指标，逐阶段均值辅助报告。
- 基线：`../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed{1,2}/baseline`。
- TRAIN缓存：`../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed{1,2}`。逐任务核验cache来源SHA、ID、标签、source config和checkpoint hash。
- 数据：`../multi-lane-main/datasets/EMOTIC`；CLIP：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 入口：`python -u -m multi_lane.track_a.fixed_baseline_staged_functions --validation-only --old-only-recheck ...`；Shell `scripts/emotic/run_fixed_baseline_staged_functions.sh`。
- 结果：`./output/emotic_staged_functional_experts/old_function_recheck_seed{1,2}_validation_20261013_01/`；日志对应`./logs/emotic_staged_functional_experts/<batch>/`。
- 预检：每seed八任务，fit/holdout/val各最多64人物，阶段数和更新规则不变；batch `old_function_recheck_seed{1,2}_smoke_20261013_01`。无Test smoke。
- GPU：优先两张空闲卡3/4，每seed一张；启动前再次资源检查，idle要求used<=512MiB/util<=5/free>=8192MiB，不停止其他任务。若资源变化且不能安全启动则保留未启动状态，不擅自抢占。
- 产物：16份模块checkpoint/seed（供体+old_only），旧函数/旧参数/访问范围/FP32旧anchor/identity审计、原图分区、逐任务fit/holdout/Validation/单路scores、source/cache hashes、config/summary/log/exit_code。总结明确candidate_arms仅baseline、old_only。

## 验收与停止规则

1. Track-A全部单测、compile、bash/diff检查通过。新增测试覆盖seed限制、recheck必须禁Test、完整八任务才可完成、source hash变更拒绝。真实seed1/2八任务smoke检查16checkpoint恢复与旧路径审计、task0 old_only与baseline预测一致，无Test目录/输出，summary.test_accessed=False；供体梯度有效，拟合与留出隔离。
2. 正式前在新Git-only独立服务器worktree运行；原primary/test-only、旧结果和用户dirty文件均不修改。运行HEAD锁定干净，启动后不再拉代码。
3. 正式完成后与seed0同配置结果一起报告：每seed最终old_only-baseline、阶段平均、七个可复用任务的holdout方向、单路表现和旧参数/预测审计。不能只挑正任务或正seed。
4. 只有两个新增seed的最终Validation均正向、三seed平均正向且holdout整体不负，才讨论新残差；小幅差异仍需给出不确定性，不自动宣称突破。否则按既定规则停止当前后置完整专家扩展。
5. 本轮不创建Test数据集、不读取Test分数、不执行Test推理、不按既有Test改配置。即使Validation正向，也不自动运行Test或极小新残差，先统一分析与锁定方案。

```bash
GPU=3 GPU_POLICY=idle SEED=1 VALIDATION_ONLY=1 OLD_ONLY_RECHECK=1 \
 BATCH=old_function_recheck_seed1_validation_20261013_01 \
 bash scripts/emotic/run_fixed_baseline_staged_functions.sh
GPU=4 GPU_POLICY=idle SEED=2 VALIDATION_ONLY=1 OLD_ONLY_RECHECK=1 \
 BATCH=old_function_recheck_seed2_validation_20261013_01 \
 bash scripts/emotic/run_fixed_baseline_staged_functions.sh
```

分支`codex/old-function-seed12-validation-20261013`。本地独立worktree在`~/.codex/worktrees/old-function-recheck/multi-lane-main`；远程独立worktree `../multi-lane-main-old-function-recheck`。只确认启动，用户召回后统一分析，不持续监控。
