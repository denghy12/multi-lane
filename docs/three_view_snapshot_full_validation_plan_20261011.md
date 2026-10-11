# 完整validation的历史ParaX变换保护对照：执行计划

## 本轮要回答的问题

原始三路内部ParaX八任务实验已完成。三seed平均最终validation：无ParaX42.6979，持续更新共享ParaX35.6899；test32.0921/26.1408。固定旧人物和旧类证明旧预测显著漂移。共同人群的历史预测拼接可恢复大部分损失，但不覆盖完整validation。本轮补齐这一覆盖缺口。

不重新训练。每个类别固定使用其出生任务的第30轮checkpoint，没有挑选最好任务、最好epoch或最好seed。三个视图依然共享该任务的CLIP和ParaX状态，使用原固定可靠性特征融合。

## 实验配置

- 研究对象：冻结CLIP ViT-B/16，Full/Person/Face，Image-token Adapter关闭，原始ParaX在Frozen Forward第11个block后／第12个block前；不是Task Forward后受保护专家路线。
- 来源：`../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seedN/{THREE_VIEW,THREE_VIEW_PARAX}/`。
- checkpoint：每来源`compact_checkpoints/task0.pth`至`task7.pth`；源训练HEAD5f2fefa。两组×seed0/1/2，共48个现有checkpoint。
- 数据：EMOTIC全部2397个validation人物，每个checkpoint都评估全部人物，包括对当前任务全为负例的人物。使用全部26类标签保存与核对，不按当前类别过滤评价人群。
- 数据路径：`../multi-lane-main/datasets/EMOTIC`。
- CLIP权重：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，文件SHA与来源一致，恢复后冻结视觉权重hash必须一致。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`；只核对val manifest和detector配置，不读取test图像或test NPZ分数。
- 推理：batch64、workers2、OMP_NUM_THREADS=1，AMP/TF32沿用来源；每次恢复后`eval()`、全部参数`requires_grad=False`、`no_grad()`。没有optimizer、学习率、epoch或任何参数更新。
- 任务与类别：原顺序`[5,3,3,3,3,3,3,3]`；三路固定特征融合，可靠Face用0.64/0.16/0.20，否则0.80/0.20/0。
- 每个checkpoint保持原已见任务状态，计算原seen lanes，再只保留该checkpoint当前任务拥有的类别列；按固定类别顺序组成26类预测。最终checkpoint同时保存完整26类预测作同人群比较。
- 入口：`python -m multi_lane.track_a.snapshot_validation`；汇总：`python -m multi_lane.track_a.compare_snapshot_validation`。
- Shell入口：`scripts/emotic/run_snapshot_validation.sh`及`launch_snapshot_validation.sh`。
- 正式batch：`three_view_snapshot_full_validation_seed012_20261011_01`。
- 输出：`./output/emotic_snapshot_validation/<batch>/seedN/<method>/`；日志：`./logs/emotic_snapshot_validation/<batch>/`。每任务owned-class NPZ、完整最终模型NPZ、历史拼接NPZ、来源/权重/样本审计JSON、三seed配对汇总。
- 资源：当前GPU0–7均有其他训练。正式评估有限排队12小时，等待至少20000MiB空闲且利用率<=15%的GPU，再单GPU依次执行六个来源，不停止已有任务。
- 预检：完整unittest、shell/compile检查；真实seed0两种方法各恢复task0和task7，并进行原validation首批replay及完整validation首批前向。仅短前向，使用至少4500MiB可用显存的卡，串行执行，完成后释放。预检不计算完整validation方法结论。

## 审计重点

1. 来源训练必须完整240epochs、更新数正确、零AMP跳步、CLIP未变、旧任务参数未变，source config/summary一致。
2. 八个checkpoint核对task_id、source_git、方法状态完整性；每个任务自己的Selector/Prompt/head与最终checkpoint对应参数逐值相同。这有助于把历史与最终输出差别定位到共享图像变换。
3. 恢复每个checkpoint后，先在来源原评价人群首批重放，logits最大差异<=0.01；最终完整validation重放也要求<=0.01、mAP差<=0.02。
4. 完整validation不通过`dataset_view`过滤人物；仅来源首批重放使用原过滤逻辑。
5. 预测拼接按sample ID对齐，严格核对完整26类标签、Face可靠mask、每列唯一task归属、无重复/缺失类别；不是直接假设两个数组顺序相同。
6. 各checkpoint前后方法权重hash相同，没有梯度；结束后重新检查全部来源checkpoint/配置/日志摘要/val分数与Face/CLIP状态，防止污染来源。
7. 无ParaX历史拼接与无ParaX最终模型应只有数值级差异（完整mAP差<=0.03），作为拼接机制对照；三个seed的两方法基础初始化hash一致，人物/标签/Face mask一致。
8. 当前评估不读取test图像或test NPZ；来源summary可能含此前test摘要，不用于选择或优化。

## 结果如何决定下一步

先报告完整validation三seed“最终共享变换／历史任务状态保护／无ParaX”的差值及各视图mAP。历史组合保护了类别学完后的输出，但不会改变后面任务当时在漂移状态上学习的过程，因此它不是所有稳定训练方案的理论上界。

若恢复后仍明确低于无ParaX，不自动排队新增训练，先暂停内部ParaX扩展。如果能接近或超过基线，再考虑task0后冻结**整个**ParaX的task0–2配对控制，包括Router、Norm、投影和scale，而非只冻结expert center。

原compact checkpoint不保存完整训练RNG／optimizer状态，不能随意续训后声称逐批严格配对。未来训练控制需要从头配对或实现完整边界恢复。

本轮不修改训练模型或增加冻结开关；第一阶段结论出来后，再决定第二阶段是否必要，不预先消耗训练预算。不能用joint26训练得到的ParaX初始化增量模型，避免未来标签监督泄漏。

## 版本与运行记录

本地独立工作树：`/Users/denghaoyuan/.codex/worktrees/three-view-snapshot-validation/multi-lane-main`，分支`exp/three-view-snapshot-validation`，起点7563364。保留主本地工作树及其他会话所有改动。服务器拟创建`/mnt/haoyuan/workspace/multi-lane-main-snapshot-validation`，通过Git-only同步；不修改已完成来源工作树或test-only工作树。


### 2026-10-11 验证通过，完整评估准备排队

实现提交bd8410d。服务器独立worktree通过291/291 unittest；本地Python编译、两份Shell语法和git diff检查通过。seed0无ParaX与内部ParaX分别恢复task0/task7，四次原validation首批重放最大logit差均为0；完整validation首批各64人前向通过，源文件/CLIP/方法权重hash未变、无参数梯度、无优化更新、未读取test图像或test NPZ。预检批次snapshot_validation_preflight_20261011_01，GPU2短串行推理已结束。8张卡仍被其他训练使用，正式批次three_view_snapshot_full_validation_seed012_20261011_01将通过tmux有限等待空闲GPU（>=20000MiB且util<=15%，最多12小时），随后依次评估六个来源；这不是重新训练。下一阶段完全冻结ParaX的三任务控制尚未实现/启动，先等待完整validation结论。禁止将预检64人当作完整评估结果。


### 2026-10-11 完整validation历史变换保护评估已排队

正式评估已在tmux ml_snapshot_validation_20261011启动队列，当前state=waiting_for_idle_gpu，尚未开始完整推理；batch=three_view_snapshot_full_validation_seed012_20261011_01，运行HEAD=70aa575dc2bdb85690d7c68c34bd9d025d7001e2。服务器worktree /mnt/haoyuan/workspace/multi-lane-main-snapshot-validation。等待空闲>=20000MiB/util<=15%的GPU，最多12小时，自动串行六组；源训练、其他GPU任务、test-only均未修改。评估代码在exp/three-view-snapshot-validation独立本地worktree /Users/denghaoyuan/.codex/worktrees/three-view-snapshot-validation/multi-lane-main：snapshot_validation.py、compare_snapshot_validation.py、两份emotic shell入口和测试；joint26 builder仅增加task_sizes可选参数，旧默认不变。291项测试通过，seed0两方法task0/task7四次真实重放最大logit差全部0，64人预检通过，非完整性能结果。预检日志/分数/摘要已同步主本地项目，未同步checkpoint；同步包SHA256 edaa6374702fafd7cd555cd32fd577a97657f99d67bfaebece547a6b6014d69b，两端一致。正式产物保存output/emotic_snapshot_validation/<batch>，日志logs/emotic_snapshot_validation/<batch>。本次只落实计划第一步，未重训、未新增test评估、未启动完全冻结ParaX训练；完整validation结果出来后再决定是否需要阶段二。详细计划docs/three_view_snapshot_full_validation_plan_20261011.md。运行工作树保持70aa575干净，不拉取启动后的文档提交。主本地工作树保留所有其他会话改动。
