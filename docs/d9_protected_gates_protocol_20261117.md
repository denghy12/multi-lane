# 同一条受保护的训练轨迹上比较四种 logits 读出

目标是在当前人物的中间证据上学习对三个任务分类头的信任程度。历史固定参照是 seed1 的保护组，Validation 最终／阶段平均 43.6272／51.1094，Test 33.2019／40.2847。

| 读出 | 输入与训练 |
|---|---|
| 固定 | Face 可靠时 64/16/20，否则 80/20/0，不训练 |
| 任务级 | 每个任务学习两个相对于 Full 的分数，同一可靠性组共享 |
| 第9层关联 | 从 Person–Full、Face–Full 分别读取 Full 注意力总量、匹配集中度、对齐的 Q/K 及 V 空间相似性 |
| 第10层摘要 | 读取三路实际用于该层任务 MSA 的 Summarized Patches，分别在摘要 token 维度平均并小投影，再联合可靠性估计本任务三路份额 |

三种可学习读出从历史先验出发。屏蔽不可用 Face 后沿视图做 softmax，权重乘分支原始 logits，最后逐类 sigmoid。摘要门控屏蔽无效 Face 的中间表示。三种门控的融合 BCE 分别只更新该门控；中间特征和参与加权的分支分数作为常量。从 Full/Person/Face 原任务损失继续训练原三路。每个任务封存此前的门控，当次模型一次分支前向并行给出四种预测。摘要门控有三路可学习投影，参数量与关联门控不同；此比较衡量完整读法，不能仅凭胜负归因于层位。

EMOTIC 旧划分，完整当前任务 TRAIN，不留出及回填。Face 分支单路监督使用原 valid 掩码，融合使用原 reliable 掩码。单次 seed1、8 任务各 30 轮、batch/microbatch 64、2 workers，分支及其 Adam、cosine、零权重衰减、BF16 确定性沿用 54e167d 保护流程；主 LR 0.0125、Adapter／交互 LR 0.0004。门控各自用独立 Adam、LR 0.0004 和相同的余弦日程、当前类别 BCE。Validation／Test 不参与梯度更新。

每阶段用历史固定保护组的训练人群、有效更新次数和完整分支状态验证重放，并比较真实前向下每人的三路 logits 和固定读出。每个可学习读出检查本人、同任务同可靠性组的置换和常量均值。常量用该阶段 Validation 输出的无标签组内权重均值，锁定用于 Test，只作诊断。完整报告全体及两种 Face 人群的八阶段 mAP，结论需要区分任务级平均信任与人物对应收益。

Linux GPU0 入口为 `bash scripts/emotic/run_d9_protected_gates.sh smoke 0` 和 `bash scripts/emotic/run_d9_protected_gates.sh formal 0`。CLIP ViT-B/16 从 `../CODE_DDP/pretrained/clip/ViT-B-16.pt` 初始化，EMOTIC 数据和 manifest 沿用 `run_r1_frozen_cross_view.input_paths()`。使用相邻 `multi-lane-main-d9-protected-fixed-20261117` 工作树的归档作重放核对，新训练从 CLIP 起步。保存逐任务 checkpoint、门控参数、训练历史、逐人物权重与 logits，结果位于 `output/d9_protected_gates/seed1_20261117/`，日志位于 `logs/d9_protected_gates/seed1_20261117/`。
