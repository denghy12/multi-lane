# 增量任务中的辅助视图贡献：运行前配置

## 目标

确认联合训练中各自有效的Person、Face能否在实际增量协议下产生稳定收益，并为后续路由研究建立共享、稳定、可比的参照。本轮是增量迁移验证，不直接修复ParaX。

## 锁定配置

- 四组：只输入Full、Full+Person、Full+Face、Full+Person+Face；全部关闭ParaX和Image-token Adapter。
- 冻结共享OpenAI CLIP ViT-B/16；每个task有MULTI-LANE任务参数，在启用视图之间共享Selector（10个）、任务提示（10个，前5层）和分类头；按原MULTI-LANE方式保存/冻结旧任务参数。没有视图私有参数或新Router。
- EMOTIC原train split；类别顺序保持项目CLASS_ORDER，任务大小[5,3,3,3,3,3,3,3]。只训练当前task的类别，按原协议选取包含当前类别的人物；没有回放或额外标注。
- seed0/1/2，每组8tasks×30epochs，共240epochs；batch64、eval64、workers2，每模型预期13,950更新，按实际task样本数核算，严格零跳步。
- Adam每task重置；主LR0.0125、WD0、cosine、无warmup、末LR0；AMP/TF32开启，init_scale1024、growth_interval1e9；temperature1、threshold0.5。
- 损失沿用legacy_full_zero和joint BCE：Full-only为1.1×BCE；多视图为BCE(fused)+0.1×可用单路BCE均值，Face辅助损失只用可靠行。旧类别梯度遵循原MULTI-LANE实现，不引入新的旧类损失。
- 固定特征融合：Full+Person=[0.8,0.2]；Full+Face可靠时=[0.64/0.84,0.20/0.84]否则[1,0]；三路可靠时=[0.64,0.16,0.20]否则[0.8,0.2,0]。Face可靠规则=valid、非ambiguous、短边≥24、检测分数≥0.6。
- Full legacy随机裁剪scale0.05–1.0；Person margin0.15、letterbox、无颜色抖动；Face复用原清单，翻转与Full配对。未启用视图不进入encoder和损失。
- 新增显式paired audit采样协议：每task DataLoader generator seed=seed+1009×task_id，在四组之间保持一致。此选项默认关闭，不改变历史实验；新结果不冒充历史sampler配置的完全复现。
- 正式运行每task只报告held-out test，不运行validation前向，不按test选结构/轮次/权重。保存final/average mAP、forgetting、每类AP、单路指标、可靠Face、分数、参数和样本hash、峰值显存、运行时间。
- 固定task0 test人物和旧五类跨task跟踪logit漂移与AP，区分样本扩展效应和真实旧输出漂移。该诊断是评估，不进入训练损失；不需要GPU teacher。
- 输入相对路径：`../multi-lane-main/datasets/EMOTIC`；权重`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face root=`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。启动前验证其train/val与原训练清单逐字节一致，检测器配置仅splits扩展。
- 分支`exp/incremental-auxiliary-view-contribution`，独立服务器worktree，Git-only同步；运行Python=`/opt/conda/envs/ddp/bin/python`。
- 入口`scripts/emotic/launch_incremental_view_contribution.sh`，调用`run_incremental_view_contribution.sh`及`compare_incremental_view_contribution.py`。
- 输出`./output/emotic_incremental_view_contribution/<batch>/seed<seed>/<method>/`；日志`./logs/emotic_incremental_view_contribution/<batch>/`；每task保存compact checkpoint，不保存完整CLIP checkpoint。
- GPU0四组并行，seed0→1→2；每seed前要求至少20GB空闲，临时不足时在服务器有限等待（最多12h），不因临时占用自动重训。
- 先全套unittest、shell语法检查，再真实ViT四组×task0–2×4updates/task smoke（使用validation，不加载test）。smoke通过后启动正式队列；不持续监督训练。

## 判断与边界

主要比较三seed配对的final/average mAP、遗忘和固定旧样本漂移，报告每seed方向，不只看单个最高分。两路与三路参数量相同，计算量不同；联合与增量的路径数、训练预算不同，不直接用绝对mAP相减归因。

如果三路收益保留且稳定性没有明显损失，保留三路；若Person两路已达到相近水平而Face带来不稳定，则后续优先简化并验证Face质量，而非继续改写CLIP内部特征。未预设根据test自动启动后续结构，不做test融合权重搜索。

## 执行状态：2026-10-10

- 运行代码HEAD：`d85e2fa`；服务器独立worktree：`/mnt/haoyuan/workspace/multi-lane-main-incremental-auxiliary-view-contribution`。主工作树和test-only工作树的既有改动未修改。
- 服务器完整单测275/275、两脚本bash语法检查通过；真实ViT四组各task0–2×4updates，共12updates/组，全部zero skipped、scale1024、CLIP不变、初始参数/首批和全量样本hash配对通过，test_loaded全部为false。smoke batch=`incremental_auxiliary_view_smoke_20261010_01`。
- smoke峰值allocated显存Full约1283MiB、两路约1968MiB、三路约2575MiB。固定旧样本最大logit差约0.00049–0.00146；同样本在扩展评估集中的batch组成可能变化，加上AMP精度，存在数值底噪，不能把任何非零差异都认定为表征漂移。后续结合漂移量级、旧类AP和全精度复核判断。
- 正式队列已于09:29 CST启动：batch=`incremental_auxiliary_view_multiseed_test_20261010_01`，tmux=`ml_incremental_aux_views_20261010`，GPU0四组并行，seed0→1→2。最终实验尚未完成，不宣称精度改善。
- 正式输出：`./output/emotic_incremental_view_contribution/incremental_auxiliary_view_multiseed_test_20261010_01/`；日志：对应`./logs/emotic_incremental_view_contribution/`目录。仅做一次启动检查，随后停止监督，结束后统一同步分析。
