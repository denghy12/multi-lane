# 全量学习下的 ParaX 与三路视图对照

## 要回答的问题

导师希望区分：ParaX 的退化是否主要发生在后续增量任务，还是即使所有类别同时训练，它也不能有效适配当前模型；另外，Person 和 Face 两路是否带来整体收益。

| 实验修改 | 输入 | 主要比较 |
| --- | --- | --- |
| 不加入 ParaX | Full | 单路全量学习基线 |
| 不加入 ParaX | Full、Person、Face | 对比单路，测额外两路的整体作用 |
| 冻结 ViT 第 11 个 block 后加入 ParaX | Full | 对比单路基线，测 ParaX 自身的作用 |
| 同一位置加入三路共享的 ParaX | Full、Person、Face | 对比三路基线，测 ParaX 与多视图的作用 |

这里的“全量学习”是 EMOTIC **原始 train split 中全部人物样本、全部 26 类标签，从第一步同时参与训练**。不是把原增量程序设置为 `max_tasks=1` 后只学习最初 5 类。新协议明确使用 `task_sizes=[26]`，只有一个 Task Forward pathway。

## 模型与参数

四组共享同一个设计：OpenAI CLIP ViT-B/16 骨干冻结，Selector、Prompt 和 26 类分类头可训练。三路输入复用同一个视觉编码器实例及同一组任务参数，并非三个独立 ViT。为对齐早期 CLIP image-stream ParaX 对照，**四组都关闭 Image-token Adapter**；这轮不是 Image-token Adapter 与 ParaX 同时工作的实验。

ParaX 插在 **Frozen Forward 的第 11 个 transformer block 完整输出之后、第 12 个 block 之前（代码索引 10）**，不插在 Task Forward。它仅变换 patch tokens，CLS 旁路。后续冻结层接收变换后的 tokens；它们的参数不更新，但需要把梯度传回 ParaX。使用论文风格共享参数矩阵专家池 `E_A/E_B`，3 个专家、rank 32、router hidden 16；Full、Person、Face 共用专家池和 router，输入内容产生不同 gate。没有显式 level embedding。

采用此前内部 ParaX 的官方随机残差初始化、可训练 output scale（初值 0.1），所有专家与 router 持续学习。没有 projector、蒸馏、残差约束、中心冻结、额外任务分支或动态融合。这样只移除增量任务切换，避免同时更换 ParaX 修复方案。

三路使用现有固定可靠性权重进行 **Task Forward 输出特征融合**，随后共享分类头预测。可靠 Face：Full/Person/Face 权重 0.64/0.16/0.20；不可靠 Face：0.80/0.20/0。可靠性遵循原 manifest 的 valid、非 ambiguous、短边至少 24 像素、检测分数至少 0.6。

## 完整训练配置

- 数据集：EMOTIC 原始 train/val；不留 calibration 子集，不读取 test split，不用 test 调参。
- 4 组 seed=0，每组 30 个完整 train epochs，batch=64，eval batch=64，workers=2。
- Adam；Selector/Prompt/classifier 学习率 0.0125（源学习率 0.05 × 64/256）；ParaX 学习率 0.0004；weight decay=0。
- cosine scheduler，最小学习率 0，无 warmup；AMP、TF32 开启；temperature=1；F1 threshold=0.5。
- 全部 26 类均计算 BCE，没有旧类/未见类遮蔽；三路损失为融合 BCE + 0.1 × 可用单路 BCE 均值。Full 单路用 1.1 × BCE，以匹配总监督系数；这不意味着三路与单路优化目标完全相同。
- Full 采用原 legacy random resized crop，scale=0.05–1；Person margin=0.15、letterbox；Person/Face 无额外 color jitter；CLIP normalization；10 个 Selectors；其余默认配置落盘到各组 `config.json`。
- GPU0 四组并行，正式启动前必须通过同卡四进程真实 CLIP/EMOTIC smoke；不足显存时先调整执行方式，不擅自改变组间 batch 或训练配置。
- 每轮末记录 current validation mAP；正式比较使用第 30 轮最终模型，不按最佳轮次选择。全量协议不报告增量 average mAP 或 forgetting。

服务器独立 worktree：`/mnt/haoyuan/workspace/multi-lane-main-joint26-parax-view-ablation`。运行环境：`/opt/conda/envs/ddp/bin/python`。

输入路径（相对于该 worktree）：

```text
../multi-lane-main/datasets/EMOTIC
../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt
../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908
```

启动入口：`scripts/emotic/launch_joint26_parax_view_ablation.sh`；单组入口：`scripts/emotic/run_joint26_parax_view_ablation.sh`；组间审计与比较：`python -m multi_lane.track_a.compare_joint26 --batch-root ...`。

输出位于 `./output/emotic_joint26_parax_view_ablation/<batch>/<method>/`；日志位于 `./logs/emotic_joint26_parax_view_ablation/<batch>/`；控制状态位于 `./output/emotic_joint26_control/<batch>/`。产物包括每组 config、最终 compact checkpoint、逐轮 loss/mAP/ParaX gate/梯度/残差统计、26 类 validation scores、三路单路诊断、paired audit，以及四组比较的 Markdown/JSON。

## 对齐检查与结果解释

检查初始 Selector/Prompt/classifier hash、首批样本顺序、完整 train/val sample ID hash；确认四组一致。检查 26 个分类行均有梯度、ParaX 专家与 router 梯度有限非零、原 CLIP 权重训练前后 hash 完全不变。最终四组必须具有相同训练更新数，未跳过更新，validation labels/IDs 完全一致。

优先比较：三路减 Full 的收益；三路 ParaX 减三路基线；Full ParaX 减 Full 基线；再比较两种 ParaX 收益的差值。如果单路 ParaX 提升而三路下降，才支持检查多视图适配或融合；如果两者都下降，说明当前 ParaX 配置在全量学习下也有适配问题，不能归因于增量遗忘 alone。

全量三路 ParaX 提升、此前增量 ParaX 下降，会支持增量稳定性是重要因素，仍不能证明它是唯一原因。因为全量使用一个 pathway，而原增量使用八个 pathway；30 个联合 epochs 与 8×30 个增量 epochs 的数据曝光及优化预算不同。后续若要严格隔离顺序效应，应另设预算与任务路径控制。

三路比 Full 高，只说明额外两路及其联合监督整体有效，不能区分 Person、Face 各自贡献；需要时再增加 Full+Person、Full+Face 两组，不在本轮擅自扩展。

## 执行状态

本批次已经结束，后续实测状态覆盖下面保留的启动记录：四组均完成30epochs，但各有2次AMP跳步，实际有效更新7528而非原计划7530。原严格零跳步汇总拒绝生成报告；显式允许跳步后仅作带警告的探索性分析。完整结果、异常、离线诊断和下一步见 `docs/joint26_parax_results_and_next_plan_20261009.md`。不得把短程smoke的zero skipped转述为正式训练也zero skipped。

已完成修改，并在 GPU0 同时启动四组正式训练。

- 分支：`exp/joint26-parax-view-ablation`；训练代码提交：`1f482dc`。
- batch：`joint26_parax_views_seed0_val_gpu0_20261009_01`。
- tmux：`ml_joint26_gpu0_20261009`。
- 远程全部 266 项单测通过，两个 Shell 入口语法检查通过。
- 初次 smoke 发现 Full 单路脚本误传 Face manifest 参数；已修正，首次失败产物保留。第二次四组并行 smoke 全部完成，各 2 次有效更新，zero skipped；配对审计通过，所有 26 类参与监督，CLIP hash 不变，ParaX 专家与 router 梯度有限非零。
- train 共 16,001 个人物实例，validation 共 2,397 个；每个正式 epoch 为 251 次更新，30 epochs 共 7,530 次更新。四组初始任务参数 hash 与样本 ID hash 一致。
- smoke 四进程同时执行占用约 11 GB GPU 显存。短程 mAP 不用于方法收益分析。
- smoke 输出及验证日志已同步到本地 `./output/emotic_joint26_parax_view_ablation/joint26_fourway_smoke_20261009_02/` 与 `./logs/emotic_joint26_parax_view_ablation/verification/`。

正式结果尚未产生；只做启动检查，不持续监督训练。结束后脚本自动核对四组配置、预算、样本/标签及冻结权重，再生成 `comparison.md` 与 `comparison.json`。届时同步正式结果回本地统一分析。
