# 同一 90% 三路模型的八阶段固定融合、自然图 A 与清晰度配对 A

## 实验条件

沿用 EMOTIC 历史划分，seed1，Full、Person、Face 各自的 Task Forward 与分类头，以及 Frozen Forward 第9层的跨视图交互。分支原先以约90% TRAIN 原图训练（每个任务30轮、batch64、Adam/cosine、主学习率0.0125，Adapter/交互0.0004）。融合监督来自另外约10%的 TRAIN 原图，同张原图中的全部人物和裁剪同侧。沿用每个任务的原生引入阶段 checkpoint。8个任务的融合训练人数分别为558、439、80、995、423、251、62、126；同一个人可出现在多个任务。每个任务仅以当前新类标签至少一项为正的合法 TRAIN 人物拟合，包含这些人物在其他当前类别上的阴性标签。

| 读出 | 比例的来源 |
| --- | --- |
| 历史固定 | Face 可用时 Full/Person/Face 为64%/16%/20%，否则为80%/20%/0 |
| 自然图 A | 训练时只用三路看自然图时的输出；重放历史八任务学习结果 |
| 清晰度配对 A | 每名人物加入仅 Person 降清晰度的版本。若 Face 可靠，再加入仅 Face 降清晰度的版本。分别将该视图输入以及向其他分支发送的同源输入从224降至112再还原；每名人物的各版本 BCE 取平均 |

两种 A 的起点和学习预算相同。每个任务按 Face 是否可用从可用视图等权起步，拟合两套组平均 softmax 权重，全批 Adam、学习率0.01、300步。新任务的融合训练只访问该任务的 TRAIN 预测以及标签。每个任务训练完即锁定本任务比例。三组共用每个阶段相同的三路分类分数。按类别所属任务加权，再沿用历史类别拼接与评估规则。

GPU0 的真实 TRAIN 前向仅补 task1、2、4、5、7。task0、3、6 复用上轮经真实前向取得的留出缓存，核对预测重放和来源。自然 Validation/Test 直接复用原模型的八阶段预测，每个阶段评价当时全部已见类别和完整历史评价人群。先封存权重与 Validation，再执行 Test。预先比较配对 A 减自然 A 以及配对 A 减固定的两个差值。最终为 task7 阶段26类 mAP；阶段平均为8个阶段 mAP 的算术平均。判断是否达到同底座固定融合，须分别核对最终及阶段平均；受保护全 TRAIN 固定读出另列为跨底座参照。

入口为 scripts/emotic/run_d9_paired_a_eight.sh 的 smoke、extract、fit-val 和 test。上游产物为 sample_holdout 的 compact_checkpoints、cohorts、evaluation、d9_a_eight 的自然图权重与特征、d9_matched_source 的留出配对。输出目录是 output/d9_paired_a_eight/seed1_20261119，日志目录是 logs/d9_paired_a_eight/seed1_20261119。
