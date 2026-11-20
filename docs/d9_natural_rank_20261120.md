# 自然图正负人物排序与双模糊融合学习

## 协议

EMOTIC seed1，三路来自原图随机 90% 训练的等步数 Frozen9 分支（13,950 次有效更新）。使用其对应原图留出的 10% TRAIN 预测。八任务各自只访问当前引入的情绪和既有融合训练人群，每人至少有一个当前任务情绪为正。某一情绪的负例是在该人群中该情绪为负的人。旧任务权重封存。同一任务、同一 Face 可用性使用同一套比例。

两组从可用视图等权开始，每任务 300 次全批 Adam 更新，学习率 0.01。共同分类目标在每个人的有效版本平均 BCE，再按人平均；版本是自然、Person 224→112→224、可靠 Face 224→112→224。没有可靠人脸时仅采用前两版。原组逐值复现既有权重。新组在同一个 BCE 上加上系数 1 的逐类成对排序损失，温度 1，直接比较自然输入上的原始融合 logits。对于每个情绪，将所有正例与所有负例配对，取 softplus(负例分数−正例分数) 的均值，再对有两种标签的情绪等权平均。包括跨 Face 可用性组的正负对，两个组的参数都接受梯度。无可比正负对的类只参与原分类目标。上述数值为执行前确定的单一设定。

输入为 ../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal 的 checkpoint 出处、TRAIN 缓存和两套自然评估预测。原对照权重与逐阶段指标核对 ../multi-lane-main-d9-reliable-blur/output/d9_reliable_blur/seed1_20261120/formal 。使用 Linux CPU，入口 bash scripts/emotic/run_d9_natural_rank.sh。产物在 output/d9_natural_rank/seed1_20261120 ，日志在 logs/d9_natural_rank/seed1_20261120 。

执行顺序为单测、真实 task0 的 TRAIN 和 Val 小跑、完整八任务 TRAIN 拟合、八阶段自然 Val 封存、统一评自然 Test。Val 的全体最终与阶段平均均不低于同底座双模糊 BCE，且至少一项提高（容差 1e-10）时才推荐新方案。单独替换一类人物的融合读出仅用于归因。输出正式成绩、两组学习权重、TRAIN 损失及配对数、逐任务和逐类数据。全 TRAIN 历史固定融合不参与选择。
