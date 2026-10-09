# 全量学习实验：三路视图有收益，ParaX 尚无明确收益

## 四组结果

EMOTIC 全部 26 类同时训练，seed0、30 epochs、batch64、一个 Task Forward pathway。CLIP ViT-B/16 冻结，Image-token Adapter 关闭，Selector、Prompt、classifier 三视图共享。ParaX 仅插在 Frozen Forward 第 11 个 block 后、第 12 个 block 前（代码索引 10），共享 3 个 parameter-matrix experts，rank32，三路固定可靠性特征融合。以下是最终轮次 validation，未评估 test。

| 改了什么 | mAP | cF1 | oF1 | 比同视图无 ParaX 的 mAP 变化 |
| --- | ---: | ---: | ---: | ---: |
| 仅输入 Full，不加入 ParaX | 49.5663 | 11.7202 | 27.0599 | — |
| 输入 Full、Person、Face，不加入 ParaX | 50.6537 | 13.1458 | 29.4600 | — |
| 仅输入 Full，冻结 ViT 内加入 ParaX | 49.3727 | 12.8542 | 28.3666 | -0.1936 |
| 输入三路，冻结 ViT 内加入三路共享 ParaX | 50.7287 | 13.6492 | 30.0169 | +0.0750 |

不加 ParaX 时，三路比只输入 Full 提高 1.0874 个 mAP 点；加 ParaX 时，三路比只输入 Full 提高 1.3560 点。ParaX 与三路联合的交互差值是 +0.2686，但这只是一个 seed 的结果，不能据此宣称形成了有效的视图专门化。

## 完成状态与数值异常

四组训练 exit code 都是 0，均完成 30 epochs。每组预定 7,530 个训练 batch，AMP 各跳过两次溢出更新，实际有效更新均为 7,528 次，跳步率约 0.0266%。不同组的跳步时点不一致：Full 第28/30轮，三路第27/30轮，Full+ParaX第17/23轮，三路+ParaX第17/30轮。

因此，原来要求 zero skipped 的自动比较脚本拒绝汇总，**不能写成零跳步审计通过**。本批次保留为探索性结果：样本、初始参数、标签和原 CLIP 权重 hash 对齐，模型训练完成，但数值轨迹未满足原注册的零跳步要求。新的汇总选项只有显式开启才允许分析此批次，且保留警告。

原 ParaX 梯度诊断在 GradScaler 反缩放之前读取梯度，记录的是经过 loss scale 放大的梯度；溢出 batch 会使当轮平均梯度出现 NaN。不能把这些数值当作真实未缩放梯度大小，也不能据此说模型最终权重已经崩塌。AMP 保护跳过了对应更新，最终预测文件有限，CLIP hash 未改变。下一轮记录除去 loss scale 后的梯度和实际 scale。

## 融合前的单路变化

以下均为各自**三路联合训练模型**中抽出的单路结果，Full 单路端点不同于上表独立训练的 Full 模型。

| 输出方式 | 不加入 ParaX | 加入 ParaX | 变化 |
| --- | ---: | ---: | ---: |
| Full 端点 | 48.7751 | 48.2694 | -0.5057 |
| Person 端点 | 47.3990 | 47.1352 | -0.2637 |
| Face 端点，全部 validation 人物 | 39.4139 | 39.7432 | +0.3294 |
| Face 端点，仅可靠 Face 的 1,460 人 | 44.4794 | 45.0843 | +0.6050 |
| 固定三路融合 | 50.6537 | 50.7287 | +0.0750 |

可靠 Face 与全部样本的 AP 不能直接比较高低来证明检测筛选的因果收益，因为评估子集不同。上表的可靠 Face 行只在相同可靠子集上比较两个模型。

ParaX 更像重新分配了三路表现：Face 略好，Full/Person 略差，最后相互抵消。三路固定融合分别比各自 Full 端点高 1.8786/2.4593 点，说明“特征完全无法融合”不符合当前结果。ParaX 模型融合相对 Full 修复了 728,935 个正负排序对，同时破坏 425,505 个；无 ParaX 为修复 629,733、破坏 366,211。净修复量增加，但排序对统计不是 mAP 的直接等价指标。

## 路由和特征扰动

最后一轮训练统计：

| 输入视图 | 三个参数矩阵专家的平均 gate 权重 | gate 熵 | 残差 / 输入 token 范数 |
| --- | --- | ---: | ---: |
| Full | 0.4505 / 0.2773 / 0.2721 | 0.9308 | 0.1460 |
| Person | 0.3547 / 0.3328 / 0.3125 | 0.9836 | 0.1409 |
| Face | 0.3258 / 0.3624 / 0.3118 | 1.0213 | 0.1062 |

三视图 gate L1 距离约 0.1724；三个专家均有使用，未完全塌缩为一个专家。输出比例从初始 0.1 学到约 0.1876。Full-only ParaX 最后残差比例约 0.1488。

这说明 ParaX 已经产生非零变换和不同路由，不能归因于模块没训练、没接线或完全 identity。但这些 gate 差异尚未带来明确净收益。该比例是训练 batch 上的相对残差幅度，**不是跨任务 feature drift 的测量**；联合训练本身没有旧任务 forgetting 指标。

## 与之前增量结果串起来

同类 Frozen Forward 内部 ParaX 的八任务配对 validation：无 ParaX 42.9847，ParaX 36.8990，差 -6.0858；本次全量三路差 +0.0750。此前同一批固定 task0 样本的旧五类 validation mAP：无 ParaX 从 task0 到 task7 近乎不变，ParaX 下降约 11.6 点；旧类 logit 平均绝对变化约 1.898，基线约 0.004。

当前最合理的解释是两层问题并存：

1. 后续任务持续更新共享参数矩阵专家，会改变旧任务接收到的图像 token，增量学习明显放大了退化。原始 CLIP 权重仍冻结，漂移发生在增加的可训练特征变换及其后续激活。
2. 去掉后续增量任务后，ParaX 没有大幅崩塌，但它的净收益仍非常小；不能把问题全归结为增量，也不能说它已证明适合这项工作。

全量只有一个 pathway，原增量有八个；30个联合 epochs 与8×30个增量 epochs的数据曝光、loss语义和优化预算不同。这是支持性证据，而非只改变“任务顺序”的严格因果对照。

三路整体有效仍不能说明 Person 与 Face 各自必不可少。本轮已对现有分数完成移除单一路的推理诊断，固定先验权重重新归一，不拟合新权重；它不是重新训练 Full+Person/Full+Face 的消融。

## 额外诊断：Person、Face 是否在当前融合中贡献信息

四组配置/样本/初始参数/冻结权重与保存分数已核对，原始结果及日志共40个文件的远程/本地SHA-256逐项匹配。以下重建利用共享线性分类头：同一三路模型的固定特征加权与相应logit加权在实数计算中等价。使用保存的float32 logits重新计算，与原AMP融合的mAP误差约0.0014/0.0019点，最大logit差约0.0094/0.0100；因此表格注明重建，不冒充重新执行模型推理。

| 在同一个已训练三路模型中移除哪一路 | 不加 ParaX | 加 ParaX |
| --- | ---: | ---: |
| 仅保留 Full 端点 | 48.7751 | 48.2694 |
| 移除 Face，Full/Person 先验重归一 | 49.9533 | 49.8708 |
| 移除 Person，Full/可靠 Face 先验重归一 | 49.8285 | 49.7342 |
| 保留全部三路，固定融合 float32 重建 | 50.6551 | 50.7306 |

没有ParaX时，在Full+Person基础上保留Face使mAP增加约0.7018；在Full+Face基础上保留Person增加约0.8266。加ParaX对应约0.8598/0.9964。它支持两个辅助视图在当前联合训练模型的预测中都有贡献，不能推出分别训练Full+Person/Full+Face一定得到同样收益，也不能忽略重新归一权重本身的变化。

26类中，不加ParaX的三路相对独立Full模型改善18类：Sadness +7.1550、Fatigue +4.4840、Embarrassment +2.7961、Pleasure +2.7835；也有Disapproval -2.4602、Aversion -2.2375等退步。三路ParaX相对三路基线改善14类，但收益与损失相抵：Aversion +4.2495、Surprise +2.3312、Fear +2.2979；Suffering -3.8221、Sadness -2.7108、Pain -2.5636。这符合“路由学到不同偏好，但未形成全面改善”的观察，而不是路由完全失效。

## 额外诊断：差值有多不确定

按原图sample ID分组，将同一图像里的多个人物一起重采样；1,705个原图组，500次配对bootstrap。各次对四组使用完全相同的采样索引。

| 比较 | 实测mAP差值 | 图像组bootstrap 95%区间 |
| --- | ---: | --- |
| 三路无ParaX减Full无ParaX | +1.0874 | +0.3105 到 +1.8893 |
| 三路ParaX减三路无ParaX | +0.0750 | -0.6150 到 +0.8484 |
| Full ParaX减Full无ParaX | -0.1936 | -0.9226 到 +0.5998 |

三路收益在这个条件下比较明确，ParaX收益的区间跨零。该区间只反映**当前四个seed0训练模型下validation图像抽样的不确定性**，不包含训练seed变化，也不是test泛化保证。这正是下一轮需要三seed复核，而不是继续据0.075点选择更复杂模型的原因。

## 下一步：先完成稳定性和多 seed 复核

不增加新架构，也不继续试无约束 Router。此前后置路由、预测动态 gate 和校准背景覆盖均未取得有效收益，本轮不重复它们。

修改 `runner.py`：开放 AMP 初始 loss scale 与增长间隔，默认保持历史值；ParaX 梯度诊断去除 AMP loss scale。修改 `compare_joint26.py`：保留默认严格零跳步审计，显式允许旧批次带警告的探索性分析。脚本支持 seed 参数，新增三 seed 队列及配对汇总。

**下一轮完整配置：**

- 四组结构与本次相同，Full/三路 × 有无 Frozen Forward 内部 ParaX；全部26类联合训练，一个 Task Forward pathway，Image-token Adapter 关闭。
- EMOTIC 同一 train=16001、val=2397；同一 CLIP ViT-B/16 权重和 Face manifest；不读取 test 进行推理/监督，不留 calibration 子集。
- 每组30 epochs、batch64、eval batch64、workers2；seed0、seed1、seed2；seed0同样重跑以保证三 seed 使用完全相同的数值配置，旧seed0不混入新均值。
- Adam，主参数LR0.0125、ParaX LR0.0004、WD0；cosine、无warmup、temperature1、threshold0.5；AMP/TF32保持开启。
- 唯一统一数值变更：AMP初始loss scale=1024，growth interval=1,000,000,000，30轮内不主动增长；仍保留溢出检测和跳步保护，不保证事前一定零跳步。以完成后的严格审计验证。
- ParaX配置不变：代码索引10，rank32，3个参数矩阵专家，router hidden16，official初始化，learnable output scale初值0.1；共享expert/router全程学习；无projector/level embedding/蒸馏/残差约束。
- 三路固定可靠性先验0.64/0.16/0.20或0.80/0.20/0；三路BCE(fused)+0.1×单路BCE均值；Full-only 1.1×BCE。其余图像变换与本次一致。
- 同一seed的四组在GPU0并行，完成后自动运行下一seed；每组预期7530有效更新，零跳步方可进入严格汇总。
- 分支 `exp/joint26-parax-seed-replication`；远程独立 worktree `/mnt/haoyuan/workspace/multi-lane-main-joint26-parax-seed-replication`；入口 `scripts/emotic/launch_joint26_seed_replication.sh`。
- 运行环境 `/opt/conda/envs/ddp/bin/python`；输入仍用相对路径 `../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`、`../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`。
- 输出 `./output/emotic_joint26_parax_view_ablation/<prefix>_seed{0,1,2}/`，日志 `./logs/emotic_joint26_parax_view_ablation/`，控制状态 `./output/emotic_joint26_control/`；各组最终compact checkpoint、scores、审计文件保留，最后自动产生三seed配对均值/标准差Markdown和JSON。
- 只比较最终第30轮validation；不看最佳轮次挑结果，不在test上选结构。

预先设定开发晋级标准：三 seed 中三路 ParaX 的平均 mAP 收益至少 +0.3 点，至少两个 seed 为正，且没有一个低于 -0.3。该标准是控制继续研究成本的开发门槛，不是统计显著性检验。如果仍接近零或方向不稳定，停止把共享 ParaX 放在 Frozen Forward 内部；继续保留有证据的三路共享冻结编码器，同时优先研究各辅助视图的独立贡献，而非继续放大专家池。

## 执行状态

原始四组结果和日志已同步本地，40个原始文件SHA-256核对通过。探索性汇总、视图移除诊断和图像组bootstrap已完成，原AMP跳步警告保留。

新代码提交 `f068414` 已通过远程269/269单测、三个Shell入口语法检查，以及GPU0真实CLIP/EMOTIC四组seed1 smoke（各4次有效更新、zero skipped、strict paired audit通过）。新的梯度诊断去除了AMP scale，原CLIP hash仍完全不变。smoke与验证日志同步本地。

**三seed正式队列已启动**：prefix=`joint26_stableamp_multiseed_val_gpu0_20261009_01`，tmux=`ml_joint26_stableamp_seeds_20261009`，训练HEAD=`f068414`。先seed0四组在GPU0并行，全部完成且严格审计通过后自动seed1、seed2。新正式结果尚未完成，不持续监督，结束后统一同步并分析。文档后续提交与运行代码分开，不更新运行中的worktree。
