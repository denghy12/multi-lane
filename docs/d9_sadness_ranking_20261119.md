# 全 TRAIN 与 90/10 的 task6 Sadness 配对排名复盘

主实验复用 seed1、全 TRAIN 的受保护 Frozen9 三路。task6 当前任务用627人、562张原图，441人有可靠Face，含260名 Sadness 正例。分支按历史每任务30轮、batch64学习，Full 与 Person 用过这些人物，Face 遵循原筛样。重新提取该模型第10个 Task Forward block 输出的三路 Task CLS，以既有固定投影与 TRAIN 标准化得到12维。融合用全部627人，从有效视图等权开始，全批 Adam、LR 0.01、300步。A 为组常量，B 为个人 BCE 读出，C 同结构增加系数1的平均真实标签概率监督。分支不更新。全 TRAIN 的同源历史固定比例和同源 A 分别作为对照。仅复盘 task6 引入阶段。另读取90/10已有封存参数完成人群对照，该组拟合用62人。

两侧均在当期全部已见人物上计算单类 Sadness AP。用原 float32 sigmoid 和官方排序复现基准后，以 A 或固定比例为起点重算仅正例换 C、仅负例换 C、所有人换 C 的排名。保留可靠 Face 标记；将人物分数变化按相对 Full 的 Person、Face 两项拆分，并统计翻转的正负对。平均两种替换顺序用于描述 AP 的变化，翻转对数量本身不等于 AP 的贡献。Validation 封存后复核 Test。使用标签的混合方式仅作诊断。

入口为 bash scripts/emotic/run_d9_sadness.sh，产物在 output/d9_sadness/seed1_20261119、日志在 logs/d9_sadness/seed1_20261119。原 checkpoint、历史权重和预测均只读。
