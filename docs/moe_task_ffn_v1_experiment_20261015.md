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

`smoke_03` 在 `396c495` clean HEAD 上完整通过：2 成功更新、0 跳步，
CLIP SHA 不变，峰值保留显存 2902 MiB，MoE 梯度有限非零；
其 task0 val mAP26.760875 只作检查点重放基准，不作为效果结论。
独立加载 compact checkpoint 后，以训练相同的 eval batch64 重放得到
完全相同的 mAP26.760875；batch32 则为26.746689（AMP 数值/排序
敏感性）。正式两路并行评估固定 batch64，避免引入这个评估批量差异。
smoke 仅 2 updates 时，三个视图分别固定使用两名专家，尚无内容相关
路由多样性；正式运行后必须检查跨样本路由分布，不能以不同视图选择
不同专家就宣称方法有效。

正式 batch `moe_task_ffn_v1_seed0_20261015_01` 已在 GPU0/tmux
`ml_moe_ffn_v1_20261015` 从 clean commit `17a9b84` 启动，目前 task0。
训练代码固定在该 HEAD，训练中不更新远程 worktree。正式输出在
`./output/moe_task_ffn_v1/moe_task_ffn_v1_seed0_20261015_01/`，日志在
`./logs/moe_task_ffn_v1/moe_task_ffn_v1_seed0_20261015_01/`；
8任务训练已完成（240 epochs、13950 updates、0 skipped，峰值
reserved 3022 MiB，冻结 CLIP 权重未变化）。两进程评估完成。
原评估 task7 完整 26 类 fused mAP：val 42.0342，test 31.7789；
同协议旧 B0 为42.1912/32.0610，首版尚无增益。全部任务/视图的
测试路由都只使用固定两名专家，尽管 gate/noise 参数训练后非零；
跨图片内容相关的专家选择未出现。原评估 task0..6 使用已见类别
正例筛选人群，而旧 B0 使用固定全人群；原阶段平均 mAP、遗忘数字
不得与 B0 直接比较。现增加 `--full-cohort` 只读重评估开关，
分别另存 `full_val/` 与 `full_test/`，不覆盖训练或原始评估。
