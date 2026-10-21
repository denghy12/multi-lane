# 三视图融合的模块借鉴与候选设计（2026-10-21）

本轮核对原论文方法章节与本项目实际前向；仅文献和方案整理，没有业务代码改动、新实验或test。
所有迁移设计都是本项目假设，不是原论文在本项目增量协议下的已证实结论。现有负结果不构成某个唯一根因的证明。

## 项目边界

Full/Person/Face是同一原图的嵌套RGB视图，有大量重叠；原始patch位置及同编号Selector摘要不能默认语义对齐。
近期retained三任务先导未胜匹配控制：val最终43.8444，三路aux0 44.3003，固定摘要44.6137。
该实现每层重新Summarize三路，Task CLS读取30个摘要，但Drop & Replace只保留CLS更新，放回原10个Selectors。
因此没有30个跨层持久视图状态，也没有多个跨层更新的融合tokens；图像信息仍可从各层Frozen Forward重新读取。
这与MBT完整架构不同，但不能据此断定单CLS是失败原因。
骨干冻结、任务级参数冻结和分类头位置保持，是候选设计的约束；新模块默认task-local，避免旧输入依赖随新任务变化。

## 1. MulT：跨视图定向读取，优先阅读

原文：Multimodal Transformer for Unaligned Multimodal Language Sequences，ACL2019。
https://aclanthology.org/P19-1656/
方法3.1：https://arxiv.org/html/1906.00295v1#S3.SS1

原模块：Directional Pairwise Crossmodal Attention。目标模态提供Q，源模态提供K/V；以跨序列attention学习关联，原任务为未对齐的语言/音频/视觉时间序列。
本项目借鉴：在Summarize之后、任务MSA之前，以Person摘要为Q，Full摘要为K/V，检索与目标人物相关的上下文；Face作为另一路受有效性约束的证据。
不能把这种迁移称为原论文已解决跨裁剪空间对齐；attention只提供学习关联的机制，正确的人物关联仍需验证。
与此前区别：不按编号平均摘要，也不把Person均值变成统一query delta；直接从另一视图读取内容和value。
创新候选：同一任务内共享交互投影，定向的目标人物条件读取；视图状态保留，质量控制单独验证。
风险：新增attention自由度仍会过拟合；人物条件Selector历史已失败，本候选必须与相同预算的普通cross-attention比较。

## 2. MBT：保留各路状态，通过少量token交换信息，优先阅读

原文：Attention Bottlenecks for Multimodal Fusion，NeurIPS2021。
https://arxiv.org/html/2107.00135v2
重点3.2.3、3.3及附录共享权重消融。

原模块：Fusion Bottleneck Tokens。各模态表示保留，跨模态信息只经少量融合token交换；融合token由各路更新，再聚合传给下一层。原实验是音视频。
本项目借鉴：在任务侧增加少量跨层融合tokens，让三视图仍保留各自证据状态，中高层进行有限交互。Frozen Forward不写入这些tokens。
首版可以保留B0三路Task Forward，先隔离交互机制的作用；若坚持单路径，则需另设计视图状态与融合token的跨层持久化，并调整Drop & Replace规则。
区别：不是把三组摘要提前平均，也不是仅让同一个CLS每层读取三路后丢弃临时摘要。
创新候选：目标人物条件的融合token更新；可靠性约束的写入；限制交互范围。
风险：瓶颈仍会丢信息，原文性能/速度优势不能直接转移；MoE Middle最好也不能证明新交互模块应放相同层。

## 3. CCIM：场景原型与背景偏置控制，任务相关性高

原文：Context De-confounded Emotion Recognition，CVPR2023。
https://arxiv.org/html/2303.11921v2
重点3.1–3.3。

原模块：Contextual Causal Intervention Module。原文在TRAIN图中遮去目标人物，提取背景表示，以PCA和K-Means++构造context prototype dictionary；通过字典加权近似对上下文混杂因素的调整。模块接在融合特征之后、分类器之前。
本项目借鉴：先检查场景依赖，再在融合后、原分类头前引入背景原型条件模块。冻结CLIP及旧任务字典，当前任务训练模块学完后冻结。
不能把Full图像特征直接叫背景特征，因为它包含目标人物；原型只由合法TRAIN fit建立，不能使用val/test/holdout拟合字典。
创新候选：针对三视图质量和任务生命周期的原型访问/保护机制。
风险：当前尚未证实背景偏置是主因；普通注意力加权场景字典不能自动被称为因果干预。字典存储和旧统计使用须报告。

## 4. EmotiCon：显式场景与人物交互证据，适合开辟新信息来源

原文：EmotiCon: Context-Aware Multimodal Emotion Recognition using Frege’s Principle，CVPR2020。
https://arxiv.org/html/2003.06692v1
重点3.3、3.4。

原模块：目标人物遮挡后的场景attention；深度图与基于位置/邻近关系的GCN，表征社会交互。
本项目借鉴：目标人物与他人的相对几何、距离和交互作为relation tokens，辅助任务Selectors读取场景；可先使用已有框几何，若增加姿态/深度估计器须分别报告。
原论文不是我们提出的骨架关键点模块，也不能把二维框距离当作真实三维距离。
价值：引入现有RGB摘要没有显式组织的信息；换融合网络只能重组已有输入。
风险：检测漏人、遮挡、近邻不等于关系；需置信度与人物绑定检查。新分支单路AP更高不代表融合净贡献。

## 5. MISA：共同信息与视图特有信息分开，备选

原文：MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis，ACM MM2020。
https://arxiv.org/html/2005.03545v3
重点3.3–3.5。

原模块：每模态共享/私有表示分解，结合相似性、差异性、重建和任务损失，再进行attention融合。
本项目借鉴：对三份任务摘要分别保留共同语义与视图补充信息，避免只对全部表示做一致性对齐或全部平均。
共享参数不等于共同表示，MoE共享Expert也不自动保证分解；历史私有参数失败不等于所有显式信息分解都失败。
风险：Full包含Person，Person又包含Face，完全私有/正交假设过强；不能靠增加特征距离宣称互补，需任务信息保留与融合收益。故不作为首个新训练候选。

## 建议只选一个主机制

如果继续研究三视图融合模块，优先MulT/MBT，提出可检验问题：目标人物条件的定向交互与有限跨层记忆，能否比无差别交互保留更多互补信息。
一个可讨论的设计是：

Full/Person/Face冻结流 → 各自Summarize → 任务侧定向读取或少量融合tokens交互 → 各路Task Forward/受控读出 → 原分类头。

首版保留原三路状态和head位置，不同时合并任务路径；同一Task内可共享投影参数，任务结束冻结完整依赖。
Face无效softmax前mask属于已有基础处理，不再算创新；软质量权重须先证明检测置信度/清晰度能预测实际分支可靠性。
先实现定向cross-attention或持久融合token其中之一，不把MulT/MBT/CCIM/MISA一次堆在一起。

最小消融建议（尚未授权执行）：

1. 当前B0。
2. 普通跨视图attention，控制新增参数和训练预算。
3. 目标人物条件的定向交互，与2比较人物条件净贡献。
4. 在3上单独加入质量控制；所有组均保留相同的无效Face基础mask。

若选融合token作为主机制，应改为普通跨视图attention/单CLS/持久多token的对应控制，不与定向机制同时变动。
遮挡、打乱视图及模块关闭只做条件依赖/敏感性诊断；最终仍要靠重新训练的匹配组、训练内原图holdout和validation判断净收益。
不以路由多样性、attention热图、特征差异或源论文非增量指标代替当前方法有效性。

把已有模块迁移、拼接或重新命名不自动构成创新。可形成贡献的是明确的结构约束、针对三视图重叠/不对齐/缺失和持续学习依赖保护的设计，以及能排除普通attention/参数量解释的证据。
