# CLIP Face + AffectNet补充残差结果（2026-09-13）

## 协议与完整性

- 分支：`codex/face-expression-residual-validation`
- 运行提交：`fe054d5b066f55784a2ca8f8c22f25bb529da23a`，clean tree
- batch：`face_expression_residual_seed0_20260913_143719_retry1`
- seed0、EMOTIC完整8-task validation-only；不读test、不存checkpoint。
- 30 epochs/task，共240 epochs、10,980 optimizer updates、0 skipped；每task的cosine终点LR均为0。
- 运行时间`3323.98s`（约55.4分钟），无OOM、NaN或非有限loss。
- 结果、control与log共17个文件已同步到本地，逐文件SHA-256与服务器一致。

## 核心结果

| 视图 | 旧CLIP Face锚点 | CLIP + AffectNet残差 | 变化 |
| --- | ---: | ---: | ---: |
| Face all final mAP | 33.3521 | 31.7562 | -1.5959 |
| Face all average mAP | 38.6388 | 36.4923 | -2.1465 |
| Reliable Face final mAP | 36.3096 | 35.3436 | -0.9660 |
| Reliable Face average mAP | 41.6852 | 40.7240 | -0.9612 |
| 固定R1 final mAP | 43.5812 | 43.3047 | -0.2765 |
| 固定R1 average mAP | 50.7509 | 50.5523 | -0.1986 |
| 固定R1 forgetting | 0.9039 | 0.9701 | +0.0661（更差） |

固定R1的task0--7 mAP变化依次为：

`-0.1306, -0.1096, -0.1604, -0.1639, -0.1349, -0.2871, -0.3256, -0.2765`。

因此失败从task0就开始，不是另一次只在task6/7出现的后期灾难。后期损失扩大，
且forgetting变差，说明残差不仅没有增加有用排序信息，还使旧类稳定性下降。

固定R1 final下降最多的类别为Pain `-2.8726`、Sadness `-1.6696`、Aversion `-1.6329`；
最大改善仅为Disapproval `+0.5252`和Affection `+0.4022`。task6中Suffering仅`+0.0278`，
Sensitivity `-0.1173`，无法抵消Sadness损失。

R1 final cF1/oF1分别小幅变化`+0.0625/+0.0941`，但mAP下降`0.2765`。这更像0.5阈值附近
的概率校准变化，不是类别排序能力改善，不能作为mAP路线继续优化的依据。

## 原因判断

1. AffectNet的8类表情语义主要描述面部肌肉表情，与EMOTIC的26类情境情绪不对齐；
   Pain、Sadness、Aversion等在EMOTIC中还强依赖姿态、人物与场景。
2. task-specific低秩残差只在约60.9%的reliable-face子集上学习，每个task数据有限；它容易将
   AffectNet的粗粒度偏置写入CLIP特征，而不是补充新信息。
3. 零初始化只保证初始时不破坏CLIP；BCE优化后，加性残差仍会改变所有当前lane的特征和排序。
4. R1中Face仅占0.20，所以将Face自身`-0.9660`的损害稀释为`-0.2765`，但没有将其变成正收益。

## 决策与下一步

预注册的两项`+0.05`门槛均失败，不补seed1/2，不运行test，结束Face专用表征路线。
保留旧CLIP Face和已锁定固定R1作为当前三视图方案。

下一个更有依据的ViT-B/16方向是“OOF跨视图教师蒸馏Full lane”：复用已有无泄漏Person/Face
OOF预测，只对当前task新类给Full主路增加强正则软目标，旧task lane继续冻结。第一轮seed0
validation只比较fresh Full锚点、Person OOF蒸馏、Person+Face OOF蒸馏，并同时检查Full单专家
和固定R1是否提高。这与已失败的特征残差不同：辅助视图只在训练时提供监督，不再直接
扰动Full/Face的推理特征。
