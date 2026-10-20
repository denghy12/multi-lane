# 联合残差条件贡献与三组摘要保留

## 已有模型只读验证
目的：检验融合残差对共同训练的任务路径的作用。EMOTIC完整validation2397人，边界task0/2/7。
来源joint_view_summary/formal_seed0_20261020_01的dense/moe，f057e71，CLIP ViT-B/16。
原训练seed0、8task各30epoch、B64、Adam、主LR.0125、融合LR.0004、weight decay0、cosine、aux0。
Dense宽32；MoE每任务每层4名宽16专家softTop2；12层全部融合，scale.1。
本轮无优化器、无训练、无test；B64、workers0、AMP/TF32。将所有已见任务层的scale在内存设0，其余权重不变。
GPU5运行dense，GPU6运行moe，分别on→原预测逐数组精确重放→off。先SMOKE=1仅task0完整val，再正式三边界。
入口GPU=5 ARM=dense bash scripts/emotic/run_joint_residual_attribution.sh，GPU6对应moe。
输入相邻worktree原checkpoint/EMOTIC数据/既有Face manifest，CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt。
产物output/joint_residual_attribution/<run>/<arm>，日志logs同名；含逐实例分数、路由和mAP、来源SHA。
离线报告可靠Face分组、原图分组500次配对bootstrap；不把三边界均值叫8阶段均值。
关闭残差模型仍保留共同训练的Selector/Prompt/head，不等同重训固定融合或B0。

## 后续候选
同一组任务Selectors分别读取Full/Person/Face，各10个摘要保持独立；同一个Task CLS读取30个摘要。
无效Face的Key/Value必须在softmax前屏蔽；Drop & Replace放回原10个Selectors，FFN仍处理11个状态token。
无Expert/Router/可学习视图embedding，无跨任务可变共享模块，Frozen Forward及分类头位置保留。
注意力序列11→31，不能预先宣称提速；取消提前平均，也改变注意力中三视图的隐式先验。
先task0–2的TRAIN原图分组holdout与监督匹配对照；不自动开启8task或test。正式训练配置启动前补齐。
