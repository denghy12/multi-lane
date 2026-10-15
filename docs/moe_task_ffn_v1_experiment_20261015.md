# Task-Forward MoE 三视图首版：锁定配置与执行记录

## 方法边界

本版独立于 ParaX。Frozen Forward 保持原图像 token 流不变；在 MULTI-LANE
Task Forward 第 2 个视觉块（零起始索引 1）中，MSA 的 Drop & Replace 完成并与
task-lane residual 相加后，令该状态为 `u`，输出
`u + frozen_FFN(LN2(u)) + 0.1 * sparse_MoE(u)`。MoE 的每个专家为完整
`768 -> 64 -> 768` ReLU Adapter，up projection 零初始化，dropout 0.1。
每个任务独立一个 22 专家池，Full、Person、Face 三视图共享本任务专家；
每个任务/视图有独立的 CLS 内容 Router 和 noise 矩阵，训练期 noisy Top-2，
评估期确定性 Top-2。旧任务的专家、Router 和 noise 整池冻结，禁止新任务访问。
本版不做跨任务专家复用、不使用文本塔或 DDAS，仍做全部已见 task lanes
的多标签推理。新增视觉 MoE 共 18,258,944 参数，每任务 2,282,368，
与前述官方全部 12 层全局池约 35.9M 的假设不是同一配置。

## Smoke 与完整运行

脚本：`scripts/emotic/run_moe_task_ffn_v1.sh`；入口
`python -m multi_lane.track_a.runner`，然后两进程并行运行
`python -m multi_lane.track_a.evaluate_moe_task_ffn --split val/test`。
Smoke 在 GPU0 上仅 task0、2 个成功 optimizer updates，用于形状、梯度、
checkpoint、显存及 Face 数据检查；不将 smoke 指标用作效果结论。
正式运行在 GPU0 上顺序训练 8 tasks，每任务 30 epochs，随后在同一 GPU0
上以两个进程同时评估所有 8 个固定 checkpoint 的 validation 与 test。

- 数据：EMOTIC train / val（2397 人）/ test（5368 人）；类别划分
  `5/3/3/3/3/3/3/3`，seed 0。
- 骨干：冻结 OpenAI CLIP ViT-B/16；Full / Person / Face，可靠 Face manifest，
  10 Selectors，5 层 Prompt，shared Selector/Prompt/head，固定视图融合。
- 训练：batch 64、workers 2，Adam，主参数源 LR 0.05 按 64/256 缩放为
  0.0125，MoE LR 0.0004，weight decay 0；余弦调度，无 warmup / 最小 LR；
  AMP 初始 scale 1024、growth interval 1e9，TF32；输入 CLIP 归一化，
  crop scale 0.05–1.0，Person letterbox、margin 0.15。
- 损失：`legacy_full_zero` BCE，固定三视图辅助 BCE 权重 0.1；
  `joint_bce` 路由，无 ParaX/蒸馏/正则、无新任务选择。
- 训练输入：远程独立 worktree 的兄弟目录
  `../multi-lane-main/datasets/EMOTIC`；CLIP 权重
  `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face manifest
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 输出：`./output/moe_task_ffn_v1/<run_id>/train/compact_checkpoints/task0..7.pth`
  与 `val/`、`test/` 的 per-task 分数、三视图诊断和专家路由计数；
  日志在 `./logs/moe_task_ffn_v1/<run_id>/train.log|val.log|test.log`。

首次完整运行固定这些超参数；不根据 test 调参。比较已有静态 MoE/B0
结果时先核对训练样本、初始化 hash、sampler 和代码版本；如不一致，
应补同版本 B0，不能将跨版本小差值解释为收益。成功判断优先看
validation/test final mAP、平均 mAP、旧类遗忘、Full/Person/可靠 Face
分支 AP 及路由使用是否体现内容与视图差异；路由不同本身不是性能收益。

## 状态

首版代码 `25772a3` 与旧任务完整冻结修复 `af4b07a` 已推送，服务器
287 项 Track-A 单测通过。GPU0 `moe_task_ffn_v1_smoke_20261015_01`
在首批前向暴露 AMP `index_add` 半精度/单精度不一致，零训练更新；
已补 dtype 修复和 autocast 测试，待新 commit/`smoke_02` 检查。
`smoke_02` 在 GPU0 完成 task0 两次更新、零跳步后，validation 的
Router `scatter` 再次暴露 AMP dtype 不一致；已把 Router 显式固定 FP32，
待新 commit/`smoke_03` 验证。正式 8 任务尚未启动；结果和最终
commit/run_id 在执行后补记。
