# D3：OOF固定R1三视图教师蒸馏Full lane

## 目标

仅在seed0完整8-task validation上验证：将比Full OOF更强、校准更好的无泄漏
固定R1集成作为软教师，是否能同时提高Full final/average mAP和最终固定R1融合。

## 锁定结构

- 可靠Face：`teacher = 0.64 Full OOF + 0.16 Person OOF + 0.20 Face OOF`。
- 无效/歧义Face：`teacher = 0.80 Full OOF + 0.20 Person OOF`。
- 只蒸馏当前task新类别；旧task lane冻结。
- Full主参数：`0.80 hard-label BCE + 0.20 teacher BCE`。
- Adapter只接收原始hard-label ASL 9.8/0/0.05。

## 训练配置

EMOTIC seed0，30 epochs/task，batch64，Adam每task重置，main LR 0.0125，无warmup的
CosineAnnealingLR。Full legacy crop、CLIP normalization、crop scale 0.05--1.0。Image-token
Adapter zero-based layer1、b32、LR4e-4、scale0.1、ReLU、independent。AMP/TF32开启。
不保存checkpoint，仅保存validation scores，禁止test。

## 选择规则

相对已有同批D0 fresh Full必须同时满足：Full final mAP `>= +0.05`、Full average mAP
`>= 0`、固定R1 final mAP `>= +0.05`。不搜索教师权重或蒸馏mix。
