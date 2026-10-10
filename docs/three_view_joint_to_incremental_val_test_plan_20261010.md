# 三视图全量学习结果与迁移至八任务增量学习的实验

## 1. 本会话的研究对象

本轮只研究三视图全量学习中使用的**原始视觉主干内 ParaX**能否迁移到八任务增量学习。不把另一条“Task Forward 后受保护专家”路线混入本轮结论。本文的“全量学习”指全部26个类别同时监督、只有一条任务路径，不是只训练第一个增量任务。

## 2. 已完成的三 seed 全量学习结果

来源：`docs/joint26_stableamp_validation_test_analysis_20261009.md`。mAP为百分数，±为三个seed的样本标准差；validation和test来自同一组训练checkpoint。

| 输入与方法 | Validation mAP | Test mAP | 说明 |
| --- | ---: | ---: | --- |
| 仅Full，不加ParaX（参考） | 49.1347 ± 0.5290 | 38.2098 ± 0.2808 | 原始共享CLIP及任务路径 |
| Full、Person、Face，不加ParaX | 50.7980 ± 0.3175 | 39.2278 ± 0.0916 | 固定可靠性特征融合 |
| 仅Full，视觉主干内加入ParaX（参考） | 49.4778 ± 0.2453 | 38.6025 ± 0.1871 | 用于判断是否仅为一般适配收益 |
| Full、Person、Face，视觉主干内加入ParaX | 50.9451 ± 0.1396 | 39.6299 ± 0.1402 | 原始ParaX参数路由，固定视图融合 |

三视图相对Full的收益为validation +1.6633、test +1.0180，说明辅助视图总体有价值。三视图加入ParaX后的收益为validation +0.1471、test +0.4021；三个seed的test均提升，但validation均值未达到原开发门槛+0.3。Full单路加ParaX的test也提升+0.3927，与三路收益接近。因此当前证据支持“小幅的一般适配收益”，不能证明ParaX已经实现有效的视图专门化。三路模型中Face单路test下降0.4553，可靠Face下降0.1838，也不支持三路均得到改善。

这批稳定AMP训练没有跳过更新。其正收益说明不能笼统说“ParaX在本任务完全无效”；但也不能因为全量学习有收益，就断言它适合增量学习。共享变换在后续任务中更新，仍可能改变旧任务接收到的特征。

## 3. 下一步：同一方法进入八任务，单独检验持续更新的影响

两组模型，各seed0/1/2，共六次训练。每次训练8个任务，每个任务结束后用**同一份权重**评估validation和test，不额外训练一套test模型，不用test选超参数、早停或checkpoint。

| 对照 | 具体结构 | 关键问题 |
| --- | --- | --- |
| 三路，不加ParaX | 冻结共享CLIP，三路共享Selector、Prompt和分类头，固定特征融合 | 同一新协议下的增量基线 |
| 三路，原始视觉主干内ParaX | 在Frozen Forward的第11个Transformer block之后（代码索引10）、第12个block之前加ParaX；Task Forward内部不插入ParaX | 全量学习的小幅收益是否在后续任务中消失，是否伴随旧表征漂移 |

ParaX仅改变patch tokens，CLS直接旁路；之后的冻结block与Summarize把变化后的图像信息传给Task Forward。三视图使用同一套CLIP权重和同一个ParaX参数池。参数池包含三对低秩参数矩阵，由输入相关Router混合成有效参数，不是Full/Person/Face各绑定一个专家。本轮不新增level条件、不改动态融合。

共享ParaX的矩阵、Router、归一化及可学习输出scale在全部任务中持续训练；每个旧任务的Selector、Prompt和分类头旧行按原协议冻结。通过同时检查这两类参数，判断旧任务变化是否确实发生在其输入变换上。

已完成的无ParaX八任务视图贡献实验中，三路最终test为32.3065，Full为32.0477，Full+Face为32.5061（均为三seed均值）。三路仍有小幅平均收益，但并非最优组合，说明全量学习中的视图收益不一定完整迁移到增量学习。这组结果没有ParaX，不能据此判断ParaX崩塌。已有八任务辅助视图贡献实验保留，不重复Full、Full+Person、Full+Face四组。本轮重新训练三路无ParaX，是为了与新ParaX组使用完全一致的运行代码、采样和双划分评估，并新增固定anchor诊断；不表示已有结果无效。

## 4. 完整实验配置

- 数据：EMOTIC官方train/val/test；train用于优化；validation和test只在每个任务结束后推理。
- 任务：原类别顺序，`[5,3,3,3,3,3,3,3]`；seed0/1/2。
- 骨干：冻结CLIP ViT-B/16；原Image-token Adapter关闭，避免改变原全量配置；共享10个Patch Selectors、10个Prompt，Prompt在前5层，分类头共享。
- ParaX：image-stream模式，代码层索引10，rank32，3对低秩参数矩阵，Router hidden16，official随机初始化，输出scale初始0.1且可学习；全部组件训练；无center冻结、task-local gate、level embedding、Projector、蒸馏、硬范数裁剪、smooth bound或residual penalty。
- 三视图融合：归一化Task Forward特征加权后进入共享分类头；可靠Face用`[0.64,0.16,0.20]`，否则`[0.80,0.20,0]`。可靠Face条件：有效、匹配不歧义、短边至少24像素、检测分数至少0.6。
- 损失：融合BCE + 0.1 × 可用视图BCE均值；Face辅助监督只用可靠样本；`legacy_full_zero`，监督scale1。
- 优化：30epochs/task；train/eval batch64；workers2；Adam每任务重建；主学习率0.0125（0.05×64/256），ParaX学习率0.0004；WD0；cosine，最低LR0，无warmup。
- 数值：AMP/TF32开启，AMP初始scale1024、growth interval 1e9；梯度日志已去掉AMP scale放大；要求零跳过更新；OMP_NUM_THREADS=1。
- 配对：任务采样generator显式用`seed+1009*task`，比较基础参数hash、第一批训练ID、各划分样本/标签及CLIP权重hash。旧实验的默认采样行为不变。
- 输入相对路径：`../multi-lane-main/datasets/EMOTIC`；`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face manifest为`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`，先核对其train/val与原全量学习manifest一致。
- 入口：`scripts/emotic/launch_three_view_joint_transfer.sh`；单组入口`run_three_view_joint_transfer.sh`。
- 正式输出：`./output/emotic_three_view_joint_transfer/<batch>/seedN/<method>/`；日志：`./logs/emotic_three_view_joint_transfer/<batch>/`；保存逐任务compact checkpoint，不重复保存CLIP。
- GPU：启动时检查空闲；六组各占一张空闲卡，计划GPU0–5。不会终止其他任务。
- 预检：完整unittest；两组真实CLIP/EMOTIC、task0–2各4次更新的validation-only smoke。smoke不加载test，其mAP不作为方法效果证据。

## 5. 诊断与结果判读

每任务保存validation/test的融合和单路预测、mAP与forgetting、ParaX梯度/gate/residual统计、显存和时间。固定task0评价样本追踪旧五类AP与logit漂移，避免评价集合随已见类别扩大造成混淆。另外保留16个固定validation样本，以FP32、TF32关闭的推理记录所有旧任务各视图的Task Forward特征余弦变化和logit变化；这些样本从不进入优化。

1. 若task0有益、后续收益下降，且共享ParaX更新伴随旧特征/logit漂移和遗忘增加，支持“持续更新破坏旧任务输入坐标”的解释。
2. 若task0已不如基线，不能只怪遗忘；应进一步分析较少类别/样本监督下的适配、残差强度和优化。
3. 若增量指标稳定提升且漂移受控，原始内部ParaX才值得继续做机制对照。性能提升仍不能单独证明Router必要，需要后续静态参数量对照。
4. 若融合提升但Face持续下降，只能说整体模型改善，不能称三level专门化成功。
5. 三seed同时报告validation/test，不根据test另选rank、scale或seed。全量与增量的绝对mAP差同时受类别监督、训练数据分配、总更新预算和任务路径数影响，不能全部算作遗忘。

Forgetting还会混入评价人群扩大效应：后续任务纳入含新类别的人物，旧类AP可能因新增负例变化。必须把原协议Forgetting与固定旧样本、固定旧类AP/logit漂移分别报告。

## 6. 代码与运行记录

本轮新增的是独立实验入口、配对校验与漂移诊断，不改变原始ParaX前向结构。正式运行前完成单测、真实短程验证；通过后启动六组，启动核查后停止轮询，结束时统一分析。


### 2026-10-10 三视图原始内部ParaX八任务预检通过

独立worktree /Users/denghaoyuan/.codex/worktrees/three-view-joint-transfer/multi-lane-main；服务器 /mnt/haoyuan/workspace/multi-lane-main-three-view-joint-to-incremental。代码879dc47，远程289/289 unittest通过，local compile/bash/diff检查通过。真实两组三任务各4updates smoke batch three_view_joint_transfer_smoke_20261010_01 完成退出0：每组12有效updates、零skips、AMP1024，初始Selector/Prompt/head hash与样本配对、CLIP hash和旧任务参数hash通过，test未加载。峰值allocated基线2575MiB、内部ParaX3270MiB。固定16样本FP32旧task0最大logit变化基线2.68e-7、ParaX1.12e-4，说明诊断可用，不作为短程性能/遗忘结论。正式batch three_view_joint_to_incremental_val_test_20261010_01，tmux ml_three_view_joint_transfer_20261010，计划GPU0/1 seed0无/有ParaX，GPU2/3 seed1，GPU4/5 seed2，8×30epochs，任务末同权重val+test；启动前再次检查空闲，不持续监控。与protected post-task实验分开。
