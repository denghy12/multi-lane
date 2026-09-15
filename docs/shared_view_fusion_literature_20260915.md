# 共享视图专长与动态融合：文献筛选

本次为文献研究与实验设计建议，未启动新实验。下文“项目建议”是迁移假设，不是原论文在EMOTIC增量协议上的结论。

## 证据边界修正

P2失败只说明当前b32+b4结构及训练协议未胜过P0。尚未证明共享参数造成特征同质化，也未证明弱辅助监督是唯一原因。
独立R1为概率融合，共享P0为特征融合，二者还存在分类头等差异，不能直接把差距归因于参数共享。

[DGL，ICCV 2025](https://arxiv.org/html/2507.10213v1)研究融合造成的编码器梯度减弱，并分别控制单模态与融合损失的反传。
其机制不要求跨视图梯度余弦为负。正余弦不足以排除该机制；固定权重衰减也不足以证明有害欠优化。
现有G2低于G0，支持停止该已测配置，不能据此否定所有DGL式方法。当前BCE/ASL、参数共享和固定加权融合与论文分析设置不同。
P2的Adapter审计混合共享和私有参数，后者具有不相交坐标，不能用整体余弦直接比较两模型共享参数的冲突。

## 优先阅读

### 1. AdapterFusion — EACL 2021

[原文](https://aclanthology.org/2021.eacl-main.39/)；[方法全文](https://aclanthology.org/2021.eacl-main.39.pdf)

先训练任务Adapter，再冻结主干及Adapter，用Query/Key/Value注意力组合它们。原任务是自然语言迁移学习。
项目建议：共享冻结CLIP，保留视图专用Selector/Prompt/Adapter，先学表示再学融合。原文为同一输入经过多个任务Adapter，
本项目是多个裁剪视图，需明确这是方法迁移；若选择逐Selector融合，还要建立跨视图查询的语义对应。

### 2. VLMo — NeurIPS 2022

[会议原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d46662aa53e78a62afd980a29e0c37ed-Abstract-Conference.html)；
[方法全文](https://arxiv.org/html/2111.02358v2)；[官方仓库说明](https://github.com/microsoft/unilm/blob/master/vlmo/README.md)

共享self-attention，使用模态专用FFN；专家选择依据输入模态及层位置，并非直接按样本学习三维融合权重。
项目建议：把“共享哪部分”作为结构因素，尤其比较共享Selector与视图专用Selector，而非只增加b4残差。
原模型为大规模图文预训练，不能据此认定冻结CLIP上的小样本迁移必然有效。

### 3. Attention Bottlenecks for Multimodal Fusion（MBT）— NeurIPS 2021

[原文](https://arxiv.org/html/2107.00135v2)

通过少量瓶颈token在中间层交换跨模态信息。其音视频实验还比较参数共享：早融合时独立编码器更好，晚融合时差异较小。
项目建议：在可训练任务lane侧添加少量共享融合token，从三路Selector读取证据；冻结CLIP视觉流继续保持原样。
这属于MBT启发的变体。应分别验证交互位置和参数共享程度，不能把attention权重本身当作路由有效性的证明。

### 4. MISA — ACM Multimedia 2020

[原文](https://arxiv.org/html/2005.03545v3)；[官方代码](https://github.com/declare-lab/MISA)

每模态分出共享与私有表示，利用相似、差异、重建目标约束，再用attention融合。原任务为语言、视觉和声音的情感/幽默识别。
项目建议：只有证据支持有效信息未被分离时再尝试。三种裁剪存在内容重叠，避免直接强迫全部视图正交；
仅有差异损失也可能产生无用或退化表示，需要任务信息保留检查。

## 补充阅读

- [Cross-Stitch Networks，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf)：
  学习跨分支激活的线性混合，提供软共享参照；原始混合系数并非样本动态Router，裁剪patch也不能假设位置对齐。
- [Context-dependent emotion recognition（CD-Net），2022](https://www.sciencedirect.com/science/article/abs/pii/S1047320322001997)：
  面部、身体、上下文的交互与分层融合，在EMOTIC/CAER-S评估；本次核对出版方摘要与介绍，未核验完整消融。
  与任务接近，可作相关工作，不能直接与本项目增量mAP比较。

## 建议的验证顺序

1. 用同一已学习表示对比概率、logit、特征融合，分离融合运算差异；需要保存compact可训练状态以支持受控复评。
2. 固定训练目标和融合方式，比较共享/专用Selector与Adapter的明确配置，匹配参数量并控制初始化随机数。
3. 表示锁定后比较静态组合和Selector查询驱动的动态组合，加入均值权重、打乱权重检查。分支平均mAP更高不是动态融合成立的必要条件；
   应检验是否存在可预测的样本/情绪条件互补性。若每视图Selector独立训练，不应假定同下标token语义天然对应。
4. 再单独研究MBT式中间交互，或单路监督/梯度解耦。保持Face有效掩码，路由参数按task冻结或保留快照。

共享冻结主干可以避免权重重复存储，但三种裁剪仍需各自前向。若要求大部分可训练参数也共享，应明确报告该预算，
不能仅以共享冻结CLIP宣称解决了可训练参数共享问题。新颖性需要方法与证据支持，不能仅把以上模块组合视为论文贡献。
