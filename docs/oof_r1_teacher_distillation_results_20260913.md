# D3 OOF固定R1教师蒸馏：seed0 validation结果

## 完整性

- batch：`oof_r1_teacher_seed0_20260913_192759`。
- 8 tasks x 30 epochs = 240 epochs，13,950 optimizer updates，0 skipped，exit code 0。
- 用时38分47秒；无OOM、NaN或test访问，无checkpoint。
- 19个结果/日志文件已同步到本地，共1.7MB，逐文件SHA-256一致。

## 主结果

| 方案 | Full final mAP | Full average mAP | 固定R1 final mAP | Full cF1/oF1 | Full forgetting |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 fresh Full | 42.3799 | 49.2652 | 43.5812 | 37.8214 / 58.3541 | 0.8788 |
| D3 OOF R1 teacher | 42.2110 | 48.8330 | 43.2541 | 37.3754 / 59.1533 | 0.8902 |
| D3-D0 | **-0.1689** | **-0.4322** | **-0.3271** | -0.4460 / +0.7993 | +0.0114 |

三项预注册门槛全部失败，因此不补seed1/2，不运行test。

Full task mAP相对D0为：
`[-1.5883,-0.6126,-0.3540,-0.2077,-0.2149,-0.1855,-0.1254,-0.1689]`。
与D2不同，D3到task6/7仍没有转为正收益。

task6的Full AP变化为Sadness `+2.4458`、Sensitivity `-1.5274`、Suffering `-0.0566`；
代入固定R1后为`+0.3907/-0.6587/-0.9487`。它不再具有D2对Sadness/Suffering的成组改善。

## 原因分析

1. OOF R1的AP/BCE优势不等于它的原始概率是合适的逐样本软标签。直接BCE使学习目标
   变成`0.8*y + 0.2*q`，本质上稀释了硬标签，而mAP依赖跨样本排序，不保证从点对点
   概率拟合中继承。
2. OOF teacher使用稳定评估视图，学生训练时看随机Full crop。同一个固定Person/Face教师概率
   可能要求学生从已被裁掉的人体/面部证据中复现结果，产生不可约的监督噪声。
3. 推理时固定R1再次使用Person/Face。若Full被拉向同一教师，专家相关性增加、互补性下降，
   这与D3 standalone Full `-0.1689`但固定R1进一步降至`-0.3271`一致。

## 下一步

停止全样本raw-probability BCE蒸馏，不搜索更多mix。若保留OOF教师路线，仅做一轮
“OOF优势限定的排序蒸馏”：

- hard BCE恢复完整权重1.0，Adapter仍只用hard-label ASL。
- 仅启用R1相对Full OOF在3 folds均提升AP的13类：Affection、Annoyance、Confidence、
  Disapproval、Disconnection、Engagement、Esteem、Excitement、Fatigue、Happiness、Pain、
  Pleasure、Sympathy。
- 仅使用“Full OOF排错，R1 OOF纠正”的正负样本对，直接优化student logit排序，
  不拟合teacher绝对概率。合格类每类约1万至114万个可用纠错对，数量充足。
- 排序分支使用与OOF教师一致的确定性Full视图；hard BCE/ASL仍用随机增强视图。
- 先只做seed0 validation；若仍不能通过原三项门槛，正式结束OOF蒸馏路线。
