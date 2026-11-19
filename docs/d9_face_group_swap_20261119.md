# 分别替换两路与三路融合比例后的八阶段全人群排序

## 协议

使用 EMOTIC、seed1、Frozen9 三视图模型。每个任务的 Full、Person、Face 分支由约 90% TRAIN 原图训练 30 轮（batch64，Adam/cosine，主学习率0.0125，Adapter及交互0.0004）。本轮读取同一模型过去保存的八阶段自然图预测。学习比例取自上一轮锁定的清晰度配对 A：按原图留出的约10% TRAIN，原图、仅 Person 降清晰度、仅可靠 Face 降清晰度，每人等重；按任务及 Face 可用性等权初始化，全批 Adam、学习率0.01、300步，各任务引入时拟合后封存。本轮评估无参数更新。

| 读出 | Face 不可用人物（Full、Person） | Face 可靠人物（Full、Person、Face） |
| --- | --- | --- |
| 固定 | 80%、20% | 64%、16%、20% |
| 只替换两路 | A 学出的两路比例 | 固定三路 |
| 只替换三路 | 固定两路 | A 学出的三路比例 |
| 完整 A | A 学出的两路比例 | A 学出的三路比例 |

各阶段用同一可靠性标记按人物选择融合后的已见类别分数，在该阶段全体人物内按类别重新排序并计算 mAP。两端的固定和完整 A 重放上一轮结果，两种混合读出的组内指标也核对各自来源。Validation 在 Test 前完成并封存。报告第八阶段26个已见类别的最终 mAP 与八阶段 mAP 的算术平均，同时保留逐阶段、逐类和子群数据。

相对固定的两次单独替换刻画各自的全人群效应。另报告两侧都已学习时的相应条件差值。完整 A 减去两个单独替换再加上固定所得的差额，量化两侧同时改变时 mAP 的非加性；组内排序保持对应来源，全人群 mAP 包含不同可靠性人物之间的排序。四格均为只读归因对照。

使用服务器 CPU 两线程，入口 scripts/emotic/run_d9_face_group_swap.sh（smoke、val、test），输入来自 sample_holdout/evaluation 及 d9_paired_a_eight 的 readout/centers_lock.json 与封存文件。输出 output/d9_face_group_swap/seed1_20261119，日志 logs/d9_face_group_swap/seed1_20261119。
