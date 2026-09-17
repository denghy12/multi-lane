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

## 2026-09-17：seed0 validation 结果

- batch `selector_validation_seed0_20260916` 的 S0/S1/S2 均完整结束：每组 240 epochs、13,950 updates、0 skipped；loss 有限，日志无 OOM/NaN/traceback，严格 val-only、checkpoint off。
- S0/S1/S2 fused final/average validation mAP 分别为 `42.7525/49.6763`、`41.3856/47.7380`、`41.7839/48.9822`。S1 相对 S0 为 `-1.3669/-1.9383`，相对同 Selector 参数量 S2 为 `-0.3983/-1.2442`。
- S1 对 S0 的 8 个累计 task mAP 全负；final Full/Person/Face/reliable-Face 分别为 `-1.3876/-3.0040/-2.6409/-3.2301`，forgetting 增加 `0.1440`。S2 对 S0 也全 task 负，故扩大共享 bank 容量亦不成立。
- S1 未通过全部预注册晋级条件：不补 seed1/2、不访问 test、不进入 Full-only / 两两共享、Selector 交换审计、动态 Router 或 ViT 解冻。结果和逐文件 SHA-256 同步核验见 `output/emotic_track_a_view_specialized_selector/selector_validation_seed0_20260916/analysis.md`。
- 本批未保存逐样本 score，无法补做原计划的 image-group bootstrap；不影响基于已预注册点指标作出的停止判断。

## 工程审计补充

- 225/225 服务器单测通过；额外的 Person bank 单独扰动检查显示 Full/Face 输出逐元素不变，排除 view index 串线。
- 三组 config 除 Selector 模式/数量及参数统计外一致；三组均为 13,950 updates、0 skipped、无 OOM/NaN/traceback，固定融合权重一致。
- S1 task0 最终训练 loss `0.6390` 低于 S0 `0.6476`，但 validation mAP `56.2591` 低于 `59.2605`；一轮 smoke 时 S1 高于 S0，30 epochs 后反转，支持独立 bank 过拟合/长期优化失配的解释。
- S0 的跨视图 representation path cosine 明显为正，S1 因 bank 参数隔离接近 0；这符合设计语义，说明 S1 去除了共享 Selector 的跨视图梯度平均与正则化。
- S2（共享 30 个 Selector）也低于 S0，说明单纯增加容量无益；S1 还低于同容量 S2，说明视图拆分存在额外的独立优化与固定融合失配代价。
