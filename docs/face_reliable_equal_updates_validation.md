# reliable_m15等更新量公平实验

## 目的

上一轮`reliable_m15`将Face可靠子集final validation mAP提高`1.0846`，但固定R1融合下降
`0.0879`。该候选因过滤训练样本只完成8,610次更新，而旧Face锚点完成10,980次，存在21.6%的训练量
差异。本轮只消除这一混杂，不改变Face输入或融合方法。

## 锁定配置

- 数据：EMOTIC，seed0，完整8-task validation-only；禁止访问test。
- Face训练子集：`valid && !ambiguous && short_side>=24 && score>=0.6`。
- Face输入：原检测框margin0.15，pad/letterbox到正方形后Resize 224；不加ColorJitter。
- 模型：CLIP ViT-B/16，Image-token Adapter zero-based layer1、b32、LR `4e-4`、scale0.1、
  ReLU、independent。
- 优化：主模型BCE、Adapter ASL `9.8/0/0.05`，batch64，AMP/TF32开启，Adam每task重置。
- 成功optimizer updates逐task锁定为
  `[1920,1620,360,3570,1710,960,240,600]`，总计10,980；每task cosine `T_max`使用本task预算，
  min LR=0、无warmup，终点LR必须为0。
- 不保存checkpoint；保存完整validation probabilities、稳定sample ID和targets。
- Full、Person专家完全复用，不重训。可靠Face固定权重为0.20，有效权重
  `[0.64,0.16,0.20]`；无效Face精确回退`[0.80,0.20,0]`。不搜索权重或阈值。

## 决策

相对旧margin0.15 Face锚点，候选必须同时满足：

1. Face可靠子集final validation mAP至少提高0.05；
2. 固定R1 final validation mAP至少提高0.05。

汇总器会同时校验配置预算、各task实际成功updates、skipped steps=0及scheduler终点LR=0。任一门槛
失败即停止margin、过滤和基础增强路线，不补seed1/2、不运行test。随后转向AffectNet/FERPlus等表情
预训练Face encoder，结合五点landmark对齐、margin0.15头部上下文与task-specific轻量投影/Adapter。
