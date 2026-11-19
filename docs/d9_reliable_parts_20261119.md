# 可靠 Face 三路融合的两项比例与分支训练覆盖

## 同一底座上的实验条件

EMOTIC、seed1、8任务、历史 Validation 和 Test。冻结原有 Full、Person、Face 三路自然图预测、Frozen9 patch Cross-view Attention 和分类头。本轮只重算排名。三路使用约90% TRAIN 原图、每任务30轮、batch64、Adam／cosine、主学习率0.0125、Adapter 和交互0.0004。来源 A 在另外约10% TRAIN 原图的合法当前任务人物上学习，每个人使用自然图与辅助裁剪单独降清晰度的配对版本，全批 Adam 0.01、300步、可用视图等权初始化。各任务在引入时学习比例并封存。

以 f 表示 Face 的份额，p 表示非 Face 份额里面 Full 的比例。可靠人物三个分类头的融合权重依序是 ((1−f)p, (1−f)(1−p), f)。固定 f=0.20、p=0.80，得到 0.64/0.16/0.20。可学习项取自同一个已锁定的配对 A。所有格子里的不可靠 Face 人物都用该任务 A 的两路权重，关闭上一轮的组间偏移。

| 可靠 Face 的读出 | Face 份额 f | 剩余 Full/Person 分配 p |
| --- | --- | --- |
| 基点 | 固定 | 固定 |
| 只换 Face 份额 | 学习 | 固定 |
| 只换 Full/Person 分配 | 固定 | 学习 |
| 完整 A | 学习 | 学习 |

先分别以基点和完整 A 重放上轮分人群替换实验的“仅不可用组学习”和“双方学习”。四组在 Face 不可用人物上逐值一致。三项评价人群包括全体、可靠和不可用，分别计算每阶段当时所有已见类别的 mAP，同时保存逐类结果。最终为第八阶段26类 mAP，阶段平均为八阶段算术平均。对每个因素同时考察另一因素固定或已学习时的差值。Validation 封存后评估 Test。

CPU 入口 scripts/emotic/run_d9_reliable_parts.sh（smoke、val、test），输入为 sample_holdout/evaluation、d9_paired_a_eight 的 readout、d9_face_group_swap 的历史端点，输出 output/d9_reliable_parts/seed1_20261119，日志 logs/d9_reliable_parts/seed1_20261119。全 TRAIN 对照另从 sample_full 与 sample_holdout 的历史报告按相同读出复核，可区分分支轨迹的差距及单一轨迹内改读出的差距。
