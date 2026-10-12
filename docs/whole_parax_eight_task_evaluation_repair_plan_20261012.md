# 完整冻结 ParaX 八任务：修复重放校验并补 validation/test

用户授权：2026-10-12“帮我修改并启动test”。只评价已经完成的 seed0 八任务 checkpoint，不重训、不改变模型或融合。

## 问题与修复

原训练成功，13950 次有效更新、零 AMP 跳步，训练末 validation 43.3248。后续独立评估在 baseline task0 保存分数校验处停止。完整2397人前向后再筛选得到2285人，与原来仅2285人前向相比，人物与标签一致，但 batch 组成不同。已有产物 logits 差最大0.015625、均值0.00005336，中位数0，首64人差0；需用原 batch 重放确认数值来源。

修改 `multi_lane/track_a/evaluate_whole_parax_eight_task.py`：新增原筛选人群完整重放，明确核对人物、标签、类别、batch boundaries、finite logits；原阈值0.01不变。完整人群预测独立保留，跨batch差异作为诊断记录，不用历史预测覆盖新预测。preflight对三方法task0/task7完整重放原人群，不能仅检查第一批。新增回归测试：batch依赖的输出可通过正确重放，而真实预测/人物/标签/类别/batch变化必须失败。

新增 `scripts/emotic/launch_whole_parax_eight_task_evaluation.sh`：没有训练调用，完整validation通过后自动test。使用新批次，失败产物与源checkpoint均保留。

## 锁定配置

- 数据：EMOTIC；validation2397人物、test5368人物；每个checkpoint评价对应已见类，最终26类；seed0。
- 模型：无ParaX三路、共享ParaX持续更新三路、task0后整个ParaX冻结三路。冻结CLIP ViT-B/16，后两者在Frozen Forward索引10后插入rank32、3对矩阵的ParaX；原Image-token Adapter关闭；原固定可靠性特征融合。
- 三方法全部复用已有8个checkpoint，不选择epoch/checkpoint，不搜索test权重。无训练epoch、学习率、优化器或scheduler更新；optimizer_updates=0。
- 推理batch64、workers2、AMP/TF32沿用来源config；预训练CLIP与Face manifest均核对原hash。
- 来源：`../multi-lane-main-whole-parax-eight-task/output/emotic_whole_parax_eight_task/three_view_whole_parax_eight_task_seed0_val_test_20261012_01/frozen`；两个原参照在`../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seed0`。
- 数据`../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- GPU0，启动空闲条件>=14000MiB、util<=15%，最多等待12小时，不干预其他进程。
- 新batch `three_view_whole_parax_eight_task_eval_replayfix_seed0_20261012_01`。
- 输出`./output/emotic_whole_parax_eight_task/<batch>/{control,full_val,full_test}`；日志`./logs/emotic_whole_parax_eight_task/<batch>`。产物包括三方法逐任务NPZ、mAP/新旧类/端点与冻结审计、完整comparison.json/md及各阶段退出码。

入口：

```bash
BATCH_ID=three_view_whole_parax_eight_task_eval_replayfix_seed0_20261012_01 GPU=0 PYTHON=/opt/conda/envs/ddp/bin/python bash scripts/emotic/launch_whole_parax_eight_task_evaluation.sh
```

## Git与验证

本地复用干净three-view-whole-parax-freeze工作树，先创建`fix/whole-parax-evaluation-batch-replay-20261012`。服务器新建独立`../multi-lane-main-whole-parax-eval-repair`，Git-only同步；不更新原训练和其他工作树。先compile/bash/diff、完整单测，再真实六次checkpoint首批及完整原人群重放preflight，不加载test；通过后启动正式评价链。启动确认后不持续监控，结束后同步分析。正式test必须等待本修复HEAD下完整validation成功。
