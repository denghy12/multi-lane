# ParaX P-10 稳定性 test-only 结果

批次：`parax_p10_stability_test_20260924_111500`

结果已同步到 `output/emotic_track_a_parax_p10_test/parax_p10_stability_test_20260924_111500/`，日志在 `logs/emotic_track_a_parax_p10_test/parax_p10_stability_test_20260924_111500/`。

## 完成状态与总体指标

| 方法 | 状态 | final mAP | average mAP | final cF1 | final oF1 | forgetting |
|---|---|---:|---:|---:|---:|---:|
| P10-small | 完成 8 tasks | 31.2761 | 38.5367 | 33.4257 | 47.9315 | 5.8130 |
| P10-router | 完成 8 tasks | 31.9207 | 39.3952 | 33.0528 | 49.2863 | 4.9013 |
| P10-experts | 完成 8 tasks | 31.7548 | 38.9873 | 33.9078 | 49.0916 | 5.3338 |
| P10-small-distill | task0 后 OOM | task0 53.8050 | — | — | — | 不完整 |

四组均跳过 validation evaluation，test 只作为用户指定的 exploratory 输出，没有在 test 上搜索权重或选择结构。三组完整组均 240 epochs、13,950 updates、0 skipped；蒸馏组在 task0 完成后进入 task1 时 OOM。

与已同步的冻结三视图 test 锚点 `32.5365/39.1445`（final/average mAP）相比：P10-router 为 `-0.6158/+0.2507`，P10-experts 为 `-0.7817/-0.1572`，P10-small 为 `-1.2604/-0.6078`。average 的轻微上升不能抵消最终 task 的下降和 forgetting；test-only 结果也不改变 validation 阶段的失败结论。

## 任务级结果

逐 task test mAP（task0→task7）：

- P10-small：`53.8050, 44.9809, 35.2087, 37.5582, 36.3332, 35.0950, 34.0364, 31.2761`
- P10-router：`54.2573, 45.9748, 36.0259, 38.5522, 37.7228, 36.0628, 34.6447, 31.9207`
- P10-experts：`53.3870, 45.5266, 35.8979, 38.0767, 37.0688, 35.6803, 34.5061, 31.7548`

P10-router 是三组中最稳的：task0–7 均高于 P10-small，最终 mAP 也最高，且 forgetting 最低。但它仍低于冻结 test 锚点，不能称为有效改进。

## 单视图与路由诊断

task7 单视图 mAP：

| 方法 | fused | Full | Person | Face | reliable-Face |
|---|---:|---:|---:|---:|---:|
| P10-small | 31.2761 | 29.8682 | 28.9954 | 24.6152 | 28.1297 |
| P10-router | 31.9207 | 30.4516 | 29.3402 | 25.0380 | 28.0542 |
| P10-experts | 31.7548 | 30.2315 | 29.1220 | 25.1460 | 28.1501 |

P10-router 的融合、Full、Person 和 Face 均高于 P10-small；但 reliable-Face 略低于 P10-small。三组的融合收益主要仍来自固定三视图互补，并非 ParaX 产生了更强的单路表征。

task7 layer10 的最终诊断：

- P10-small：Full/Person/Face residual ratio `0.0942/0.0921/0.0881`，view gate L1 `0.1799`。
- P10-router：residual ratio 为 `0/0/0`，因为 expert center 冻结且 small initialization 的专家残差为零；gate entropy `0.9700/0.9423/0.8717`，view gate L1 `0.1115`。
- P10-experts：residual ratio `0.0932/0.0879/0.0917`，但 Full/Person/Face gate 几乎相同，view gate L1 只有 `0.0132`，说明冻结 router 后基本失去按 level 区分的能力。

因此三组清楚分离了两个因素：router-only 可以产生轻微 view-conditioned gate 差异，但没有可训练的有效残差；experts-only 可以改变共享参数，但 gate 几乎不随 level 变化；all 组同时更新二者，残差受控后仍然出现最终 task 退化。

## 蒸馏组 OOM

P10-small-distill 完成 task0 test mAP `53.8050`，进入 task1 时在第二个完整 CLIP teacher/student 三视图前向中 OOM。GPU 总显存 23.52 GiB，当时仅剩约 13 MiB，额外申请 74 MiB 失败。原因是当前实现用 `copy.deepcopy(model)` 保存完整 teacher，并在同一 batch 同时执行 teacher 与 student 的三视图 image stream；这会把 CLIP 激活和 teacher 参数/缓存叠加到显存峰值。

蒸馏结果不完整，不能与三组完整 test 横向比较，也不能据 task0 单点判断有效性。

## 结论与下一步

小残差和 ratio cap 成功解决了上一轮的“残差爆炸”问题：ratio 从 `0.46–0.58` 降到约 `0.09`，连续层 one-hot 路由问题也不再出现。但性能仍低于冻结 baseline，说明主要瓶颈不只是残差尺度，还包括 ParaX 在 block10 改写图像表示后对后续 Selector/任务路径的校准问题，以及共享参数在 task 间持续更新造成的遗忘。

下一步不建议继续扩大 ParaX 或恢复 level embedding。若保留路线，应先修正蒸馏实现为低显存形式：teacher 固定在 CPU 或只缓存 task 开始时的旧 logits/feature，训练 batch 不同时跑完整 teacher；然后只验证 P10-router 与 P10-experts 的短程 validation，避免再次直接读取 test 选择结构。若稳定版本仍低于 B0，应停止 ParaX image-stream 方案，转向冻结 backbone 上的后置小模块或显式视图专家。
