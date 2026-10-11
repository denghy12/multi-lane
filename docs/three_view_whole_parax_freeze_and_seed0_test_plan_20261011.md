# seed0历史ParaX保护test与整个ParaX冻结控制

用户授权日期2026-10-11。仅研究原三视图全量配置进入增量任务的Frozen Forward内部ParaX，不混入其他会话Task Forward后受保护专家。

## 两个独立问题

1. 完整validation历史保护41.4777低于基线42.6979；补seed0完整test，不训练、不调权重、不选checkpoint。原始最终模型test已有，本次重新前向是补齐历史状态保护和同人群数值对照。
2. task0后固定整个共享ParaX，检查一个不再变化的图像函数能否同时保护旧任务并支持新任务。这与按任务保留多个历史状态不同。

## 统一输入和架构

- EMOTIC：`../multi-lane-main/datasets/EMOTIC`。
- CLIP ViT-B/16：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，hash与原来源核对，视觉权重冻结。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`，train/val/test及detector hash按用途核对。
- Full/Person/Face共享CLIP；Image-token Adapter关闭；Selector/Prompt/分类头在三视图间共享、按任务拥有参数。
- ParaX-style模块只在Frozen Forward第11个Transformer block完整输出之后、第12个block之前（索引10），处理patch tokens、CLS旁路；rank32、三对低秩专家矩阵、router hidden16、无level条件、official初始化、输出scale初始0.1且task0可学习。无新蒸馏、正则、ratio cap或专家扩展。
- 归一化任务CLS特征固定融合：可靠Face0.64/0.16/0.20，否则0.80/0.20/0；随后共享分类头。只做原joint BCE及0.1可用视图辅助BCE。

## seed0完整test历史保护

- 来源：`../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seed0/{THREE_VIEW,THREE_VIEW_PARAX}`。原训练5f2fefa，八任务各30epoch最终compact checkpoints，不按test选择状态。
- 两个模型各恢复task0..7，评估全部5368个test人物、26类，固定类别所属任务组合历史预测；最终模型和无ParaX历史组合作对照。每checkpoint先重放原validation首批，再首次加载test图像。
- batch64、workers2、OMP1、沿用来源AMP/TF32；无optimizer、无训练或权重搜索；seed0不计算标准差。
- 入口：`run_snapshot_validation.sh`的`EVAL_SPLIT=test`；汇总`compare_snapshot_validation --split test --seeds 0`。
- batch：`three_view_snapshot_test_seed0_20261011_01`。
- 输出`./output/emotic_snapshot_test/<batch>/`；日志`./logs/emotic_snapshot_test/<batch>/`。产物为每任务owned-class NPZ、assembled_test/final_model_test、每类/单路/可靠Face指标及来源不变审计。

## seed0 task0–2完整冻结控制

- 从头训练，仍分配原8任务参数（5,3,3,3,3,3,3,3），只限制max_tasks=3，避免改任务参数数量改变初始化。不能随意从旧task0 compact续训声称严格配对（不含完整RNG/optimizer）。
- 原参数配置全部保持；task1起冻结整个bank参数与buffer，包括专家、Router、Norm、投影和output scale。关闭梯度、清理task0残留grad、退出优化器；activate/restore/model.train均不能重新打开。Router函数固定后仍能对不同输入产生不同gate。
- 30epoch/task、train/eval batch64、workers2、Adam每task重置、任务参数lr0.0125、ParaX初始lr0.0004、weight decay0、每task cosine/minlr0/warmup0；AMP1024/growth interval1e9；TF32沿用原配置；seed0、OMP1。
- 预计90epochs、5010有效updates（2520/2070/420）；禁止AMP跳步。验证task0有非零有限ParaX梯度，task1/2所有ParaX梯度为None且范数0，完整state hash与task0一致。
- 只validation，不加载test。原训练任务末评价人群协议不变；结束后独立进程统一评估完整2397人，task0/1/2分别5/8/11个已见类，同时重放已有无ParaX、shared-live checkpoint作为同人群对照，不新增它们的训练。
- 保留fixed16 val-anchor的FP32/TF32-off旧特征余弦和logit drift；logit max<=1e-4、mean cosine distance绝对值<=1e-5，否则明确失败。它不是训练监督。
- 核对基础初始化、训练人物、首批IDs、sampler seed；task0完整预测相对原ParaX源max差<=0.01后才认可任务后续比较。
- 入口`run_whole_parax_freeze.sh`、`compare_whole_parax_freeze`；新flag`--parax-freeze-all-after-task0`。
- batch：`three_view_whole_parax_freeze_seed0_val_20261011_01`。
- 输出`./output/emotic_whole_parax_control/<batch>/frozen`和`full_validation`；日志`./logs/emotic_whole_parax_control/<batch>/`。预期compact/val scores/training history/完整freeze audit/三组full-cohort对照，test结果不用于决定训练超参数。

## 验证、资源与版本

新分支`exp/three-view-whole-parax-freeze`，独立本地worktree`/Users/denghaoyuan/.codex/worktrees/three-view-whole-parax-freeze/multi-lane-main`；拟创建服务器独立worktree`/mnt/haoyuan/workspace/multi-lane-main-whole-parax-freeze`。只用Git同步受控代码，保留主工作树及test-only和所有其他会话改动。

先compile/bash/全量unittest，再真实三任务各4updates验证；smoke只验接线和冻结机制，不作性能结论。之后短checkpoint恢复预检确认split扩展未破坏原validation重放。

正式GPU0上并行一个test评估进程和一个冻结训练进程，启动前要求至少14000MiB空闲且util<=15%；如忙则有限等待12小时，不停止或驱逐其他任务。统一launcher `launch_whole_parax_control_and_test.sh`，tmux `ml_whole_parax_freeze_20261011`。不持续监控或分析中间test；结束后同步、统一分析。

## 判读

- 旧函数仍漂移：先修冻结/前向实现。
- 冻结稳定但新任务仍差：固定ParaX跨任务适配不足，暂停内部路线。
- 稳定且接近基线：再考虑seed1/2，之后才八任务；本轮不自动扩展。
- 三任务不能直接回答task6（627人）的适配问题。

当前实现完毕，远程单测/smoke/正式启动记录待补；本轮新增实际ParaX优化器初始lr字段，避免Image-token Adapter关闭时adapter_learning_rate=None产生误读。
