# 保护三视图训练，并从中间证据学习分类头的样本级比例

## 本轮执行固定比例对照

复用 20261116 批次的联合训练固定组，Validation 最终／阶段平均为 41.2280／48.5886，Test 为 30.8054／38.2658。独立分支训练的历史 D 为外部参照，分别为 44.0329／51.1335 和 33.1770／40.2083。

新增 fixed_protected 从 CLIP ViT-B/16 起步，单轨迹训练，保留 Frozen9 patch 双接收、三个 Task Forward 和三个分类头。每路主参数接收该路原 BCE，Adapter 接收该路原 ASL。融合 BCE 继续记录，本对照断开其对两组参数的梯度。可靠／不可靠 Face 固定融合分别为 64:16:20 和 80:20:0。旧 fixed 的分支 BCE 对 26 列取平均，融合 BCE 对当前任务的类别取平均。两项系数同为 1 时，各自对当前类的梯度尺度仍可能不同。

EMOTIC，完整当前任务 TRAIN，seed1，8 任务各 30 轮，Adam、cosine、零衰减、主 LR 0.0125、Adapter／交互 LR 0.0004，有效和物理 batch 均为 64、2 个数据 worker、BF16 确定性。Face 原有效性掩码用于单路训练，可靠性掩码用于融合。与上一轮固定组逐任务核对人物 ID 哈希、训练行数和实际更新；比较 Person、Full、可靠 Face 及融合最终和阶段平均，并计算融合减自身 Full。历史 D 的独立分支数据加载及 Face 优化节奏与两个共同组批模型有差别。

GPU0，Linux 入口 `bash scripts/emotic/run_d9_relation_batch.sh smoke 0 fixed_protected`，通过后执行 `formal 0 fixed_protected`。沿用 `run_r1_frozen_cross_view.input_paths()`。输出 `output/d9_protected_fixed/seed1_20261117/fixed`，日志 `logs/d9_protected_fixed/seed1_20261117/fixed.log`。每组评估八阶段 Validation／Test，完整权重、logits 和逐类 AP 依原协议保存。

## 下一版的前向和梯度

每个任务从自己的训练轨迹抽取同一目标人物三路中间特征。本人三个分类头给出原始 logits。一个小门控根据当前人物的特征产生视图分数，先屏蔽缺失 Face，再沿视图做 softmax，按所得份额加权三个分类分数，最后逐类 sigmoid。分支保持各自的学习目标。只在通向融合监督的两条边截断梯度：进入门控的中间特征、进入加权和的分支 logits。允许融合 BCE 反传到门控。整个流程在 TRAIN 同期运行，Validation／Test 均不更新参数。保护共同组批轨迹是否恢复历史独立训练水平，需由本轮及后续数据核对判断。

优先保留每任务一个静态门控、上一轮四个关联量的样本门控，以及直接读取三路内容的文献对照。内容候选取同一任务第 10 层 Summarize 的三路摘要，这是第一次有机会读取第 9 层写回的 patch。每路摘要池化并做小投影，三个向量连同可靠性进入门控。无效 Face 在门控输入和 softmax 两处屏蔽。各组匹配比例初始化、梯度保护、读出位置和训练预算。第一轮用每任务每人物一组比例，跨当前任务类别复用。若比较内容和关联，应尽量在同层同输入下再加成对相似度，避免混淆层位和门控形式。最后才检验用 Task Selectors 的注意力汇总局部匹配。比例初始化使用历史先验需明确报告；等权初始化作为后续独立消融。

在相同任务及 Face 可靠性组比较本人权重、打乱后的权重、同组常量和同模型固定比例。若只改善组平均信任，结论应限定为任务／人群级重配。监督可用同一批 TRAIN 多标签；这种合法的数据复用仍可能高估容易拟合 TRAIN 人物的辅助分支。仅在固定底座恢复且动态权重确有条件收益后，才进入扩大种子复验。

## 文献依据

- Khan 等的 WACV 2021 群体情绪识别用场景、人脸、物体三路特征拼接，经全连接层得到按类别、按样本的视图分数，沿视图做 softmax，加权各分类器的 score。该文对群体情绪分类，使用联合监督。我们改为针对单个人物、带缺失视图的增量多标签分类，首版控制分支梯度。[原文 §3.3–3.4](https://openaccess.thecvf.com/content/WACV2021/papers/Khan_Regional_Attention_Networks_With_Context-Aware_Fusion_for_Group_Emotion_Recognition_WACV_2021_paper.pdf)
- Zhu 等从 RGB 与光流中间卷积特征预测视频样本的分支权重，用于融合分类分数。论文把训练好的两条流固定后学习门控，并比较了不同门控输出激活。我们的候选把保护扩展到两个同时进行的学习过程。[原文](https://arxiv.org/pdf/1709.03655)
- Lee 等的 CAER-Net 从人脸和上下文特征动态估计两路权重，在特征层融合后分类。该文支持用输入内容估权；本项目三个分类头分别保留到决策层。[原文 §3.2.2](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Context-Aware_Emotion_Recognition_Networks_ICCV_2019_paper.pdf)
