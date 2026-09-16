# 视图专用 Selector 验证方案

## 目的

在冻结共享 CLIP ViT、Prompt、Image-token Adapter 和分类 head 的条件下，验证 Full、Person、Face 是否需要不同的 Selector 才能从共享视觉表示中提取互补特征。本阶段不解冻 ViT，不引入动态融合 Router，固定使用现有三视图融合规则。

## 实现模式

- `selector_mode=shared`：历史 P0 行为，每个增量 task 一套 Selector。
- `selector_mode=view_specific`：每个 task 有 Full、Person、Face 三套 Selector，形状为 `[task, view, selector, width]`。
- 视图专用模式从一套正交初始化的共享 Selector 复制到三路，不额外消耗全局 RNG；训练前应与 shared 模式逐元素等价。
- 新 task 激活时，三路分别从上一 task 的对应 Selector 复制；训练时只更新当前 task 的 Selector。

## 第一批对照

| 组别 | `selector_mode` | `num_selectors` | 目的 |
| --- | --- | ---: | --- |
| S0 | `shared` | 10 | 历史 P0 基线 |
| S1 | `view_specific` | 10 | Full/Person/Face 视图专用 Selector |
| S2 | `shared` | 30 | 与 S1 Selector 参数量匹配的容量对照 |

三组均使用 seed0、8-task validation、30 epochs/task、batch64、Adam、主学习率 0.0125、Adapter 学习率 4e-4、共享 b32 Image-token Adapter、Adapter scale 0.1、joint gradient、逐视图辅助损失 0.1、固定三视图权重、strict reliable Face mask、AMP/TF32；不访问 test，不保存完整 checkpoint。

## 预检查

1. shared 模式保持现有 P0 输出、参数名和 compact state 兼容。
2. view-specific 初始化的三路 Selector 数值完全相同，且未改变全局 RNG 状态。
3. view-specific 未训练前的三路输出与 shared 基线一致。
4. 当前 task 的三路 Selector 梯度有限且非零，旧 task Selector 梯度为零并保持不变。
5. S1 和 S2 的 Selector 参数量一致；三组真实 task0 smoke 均为 84 updates、0 skipped、loss 有限。

## 结果与晋级规则

主报告包含 fused final/average mAP、逐 task mAP、forgetting、Full/Person/Face/reliable Face 单路指标、固定 R1 融合、逐类 AP 和 2000 次原图组 bootstrap。

S1 只有同时满足以下条件才进入第二阶段：fused final 至少高于 S0 0.10，average 不下降超过 0.05，Full 单路至少提高 0.10，且 Person、Face、reliable Face 没有超过 0.10 的退化。若要把收益归因于视图专用性，S1 还应相对 S2 至少提高 0.05，或 S1 通过而 S2 未通过。

第一批不运行动态 Router、不解冻 ViT、不补 seed1/2、不访问 test。若 S1 通过，第二阶段再做 Full-only 和两两共享结构，并加入跨视图 Selector 交换诊断、Selector 参数分化和梯度路径审计。
