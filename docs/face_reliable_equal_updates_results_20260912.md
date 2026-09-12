# reliable_m15等更新量公平实验结果（2026-09-12）

## 完整性

- batch：`face_reliable_equal_updates_seed0_20260911_094039`
- selection split：validation；`test_accessed=false`，没有checkpoint。
- seed0完整8 tasks成功结束，launcher与summary退出码均为0，无OOM、Traceback、NaN或非有限值。
- 配置和实际成功optimizer updates逐task均为
  `[1920,1620,360,3570,1710,960,240,600]`，总计10,980；各task skipped=0且cosine终点LR=0。
- Full、Person与旧Face锚点复用；可靠Face固定使用`[0.64,0.16,0.20]`，无效Face严格回退
  `[0.80,0.20,0]`，没有搜索权重或阈值。

## 核心结果

| 指标 | 旧m15 Face锚点 | reliable_m15等更新量 | 差值 |
|---|---:|---:|---:|
| Face全量 final mAP | 33.3521 | 33.1041 | -0.2480 |
| 可靠Face final mAP | 36.3096 | 36.1713 | **-0.1384** |
| 固定R1 final mAP | **43.5812** | 43.4675 | **-0.1137** |
| 固定R1 average mAP | **50.7509** | 50.5925 | -0.1583 |
| 固定R1 final cF1 | 37.0978 | **37.2849** | +0.1871 |
| 固定R1 final oF1 | 58.5208 | **58.6577** | +0.1369 |
| 固定R1 forgetting | 0.9039 | **0.8771** | -0.0268 |

可靠Face和固定R1都没有达到各`+0.05`的门槛，反而分别下降`0.1384/0.1137`。因此
`advance_winner_to_seed1_seed2_validation=false`，不补seed1/2、不运行test。

相对上一轮相同过滤、相同margin但仅30 epochs/8,610 updates的`reliable_m15`，等更新量版本的可靠
Face final mAP又下降`1.2229`，R1下降`0.0257`。新增更新量没有修复融合，排除了“过滤版只是少训练
21.6%而欠拟合”的主要解释。

## 逐task与类别现象

固定R1相对旧锚点的task0--7 mAP全部下降：
`-0.4425/-0.2201/-0.1155/-0.0803/-0.0363/-0.0893/-0.1689/-0.1137`。
这不是仅由task6或单次末期波动造成，而是从task0开始的一致性排序损失。

最终26类中，Sympathy、Sensitivity、Fatigue、Disapproval分别提高约
`+1.0865/+0.8126/+0.6887/+0.6326`，但Sadness、Anger、Aversion、Peace分别下降
`-2.7075/-1.2404/-0.9809/-0.9554`，Suffering也下降`-0.1333`。cF1/oF1略升但mAP下降，说明
更多训练主要改变0.5阈值附近的校准，而没有改善跨样本排序；它不能作为mAP提升候选。

## 结论与下一步

本轮已完成训练量公平性闭环。训练过滤确实能在30-epoch版本提高Face可靠子集，但该收益不能稳定传递
到固定R1；把更新量补齐后Face和R1进一步下降。应正式停止继续搜索margin、可靠性过滤阈值和基础
ColorJitter，也不为此候选运行seed1/2或test。

下一阶段应改变信息来源，而非继续重排相同CLIP特征：

1. 复用现有manifest中的五点`face_keypoints`，先审计可靠样本关键点完整率、有限性、左右眼顺序和
   仿射后边界；无需重跑检测器。
2. 引入AffectNet/FERPlus等面部表情预训练encoder并冻结主干。训练恢复为旧锚点的
   `valid && !ambiguous`样本集合，融合仍只在短边≥24且分数≥0.6时启用，避免再次混入过滤变量。
3. 用五点相似变换做人脸对齐，同时保留margin0.15头部上下文；无效关键点严格回退现有letterbox并
   记录mask。
4. 第一轮seed0 validation只比较：旧CLIP Face锚点（复用）、表情encoder+轻量task-specific投影、
   表情encoder+task-specific bottleneck Adapter。Full、Person、R1 beta0.20和训练预算锁定。
5. 新Face expert必须同时提高可靠Face和固定R1 final mAP各至少0.05，才补seed1/2；否则Face表情
   表征也不进入正式test。

## 同步产物

- 本地目录：`output/emotic_track_a_face_equal_updates/face_reliable_equal_updates_seed0_20260911_094039/`
- 压缩包：`output/emotic_track_a_face_equal_updates/face_reliable_equal_updates_seed0_20260911_094039.tar.gz`
- 文件数：19；压缩包约1.2MB；不含checkpoint。
- SHA-256：`e07209990fb4da15a5676679515613cc2a31277345edbbf8fee362c4d800b8a8`
