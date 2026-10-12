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


## 2026-10-12 重放修复验证通过，完整val→test评价队列已启动

新分支fix/whole-parax-evaluation-batch-replay-20261012，代码提交/干净运行HEAD b8660514679338eb32a7efb39336091ddd8bd663。服务器独立worktree /mnt/haoyuan/workspace/multi-lane-main-whole-parax-eval-repair，原训练及主/test-only/其他工作树保留。修复完整分数校验使用原筛选人群和原batch边界，人物/标签/类别/batch/有限值均检查，阈值0.01不变；完整人群输出仍独立计算。真实三方法task0/task7六次首批及完整原人群重放差全部0，确认原baseline task0最大差0.015625源于换batch后的AMP舍入；301/301单测与compile/bash/diff通过，无smoke训练或test加载。

正式batch three_view_whole_parax_eight_task_eval_replayfix_seed0_20261012_01，GPU0，tmux ml_whole_parax_evalfix_20261012，确认state=evaluating_val、主PID3120242、baseline task0完整校验已通过；尚未进入test。仅评价现有seed0三方法各8checkpoint，完整2397人val通过后自动完整5368人test；batch64/workers2/AMP/TF32按原config，固定融合，不训练或选择模型。输出/日志在emotic_whole_parax_eight_task/<batch>的新目录，不覆盖原failed批次。完整配置docs/whole_parax_eight_task_evaluation_repair_plan_20261012.md。启动后不持续监控，结束后同步结果并补导师总览。

预检NPZ/JSON/单测与真实重放日志、正式manifest已同步主本地；包logs/emotic_whole_parax_eight_task/verification/whole_parax_eval_repair_preflight_20261012.tar.gz两端SHA2561472c5a684c60ba4d39dad54815007b3d480f161f4966aa12076d8197a1c0de9。checkpoint留远端。导师报告已注明修复和队列实际状态，43.3248仍是训练末val记录，test等待自动阶段。提交推送获本次用户执行授权，仅实验修复分支；后续文档提交不拉取到正在评价的远程工作树。
