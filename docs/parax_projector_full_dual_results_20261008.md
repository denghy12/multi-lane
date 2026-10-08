# ParaX 后接 Projector：八任务 Validation/Test 结果与下一步

## 实验条件

批次 `parax_projector_full_dual_seed0_20261008_01`，代码 `1e39814`。五组均使用 EMOTIC Track-A、seed0、task0–7、每任务 30 epochs、batch64、Adam、主参数学习率 0.0125、ParaX/Projector 学习率 0.0004、余弦调度、冻结 CLIP ViT-B/16、共享 Selector/Prompt/classifier、固定可靠 Face 三视图特征融合；不启用 b32 Image-token Adapter。每组仅训练一次，在同一套逐任务权重上分别评估 validation 和 held-out test。全部五组完成 240 epochs、13,950 optimizer updates、0 skipped，无 NaN、OOM 或 traceback。结果文件保存在 `output/emotic_track_a_parax_projector_full_dual/parax_projector_full_dual_seed0_20261008_01/`。

ParaX 位于 Frozen Forward 第 11 个 Transformer block（索引 10）之后、第 12 个 block 之前；三组共享 parameter-matrix experts 和动态 Router 在每个任务持续更新。Projector 是接在 ParaX 后的共享 768→64→768 残差 MLP，其末层零初始化，三视图和所有任务共用，并在每个任务持续更新。只有 patch tokens 经此路径，`[CLS]` 旁路。`Projector-only` 将 ParaX 输出比例固定为零，仍保留可训练的 Projector；该组的 expert/router 不起功能作用。

## 八任务结果

| 方法 | Validation final mAP | Validation average mAP | Validation forgetting | Test final mAP | Test average mAP | Test forgetting |
|---|---:|---:|---:|---:|---:|---:|
| 不加 ParaX/Projector | 42.9847 | 50.0234 | 1.0785 | 32.1939 | 39.5336 | 4.9683 |
| 仅 Frozen Forward ParaX | 36.8990 | 44.9009 | 6.8457 | 26.5642 | 35.2209 | 11.0752 |
| 仅 Projector，ParaX 输出固定为零 | 32.3468 | 41.7086 | 10.9985 | 22.9780 | 32.2171 | 13.5319 |
| ParaX + Projector | 31.7357 | 40.5669 | 10.9283 | 22.0367 | 31.1619 | 13.7723 |
| ParaX + Projector + 特征方向对齐 | 32.0006 | 40.4798 | 9.2492 | 22.7443 | 31.2364 | 11.9849 |

与同批基线相比，原 ParaX 的 final mAP 在 validation/test 分别低 6.0858/5.6296；加 Projector 后分别低 11.2490/10.1572。对齐约束相对无约束组合只恢复 0.2649/0.7076，仍分别低于基线 10.9841/9.4496。Projector-only 同样显著退步，表明新增 Projector 本身的跨任务更新已足以破坏稳定性；不能把组合下降全归因于 ParaX Router。

task0 的 validation final mAP 依次是 60.5225、58.5343、57.5298、58.1436、58.2807；也就是说，ParaX/Projector 组在增量学习开始前就没有优于基线。原 ParaX 在 task0 test 为 54.8082，略高于基线 54.4515，但 task7 降到 26.5642，不能把它解释成稳定收益。

task7 的 Validation 单路指标也一致下降：基线 Full/Person/Face/可靠 Face 为 41.4701/39.8515/33.9268/38.1875；ParaX+Projector 为 30.5535/28.0088/26.3857/29.9308。固定融合不是唯一失败点，输入融合前的各视图表征已退化。

## 同一样本上的旧任务漂移

以 task0 validation 的 2,285 个样本和 test 的 3,142 个样本为固定 anchor，在 task0 与 task7 的分数文件中按 sample ID 对齐，比较旧五类 logits 和 mAP：

| 方法 | Validation 旧五类 mAP 变化 | Validation 平均绝对 logit 变化 | Test 旧五类 mAP 变化 | Test 平均绝对 logit 变化 |
|---|---:|---:|---:|---:|
| 不加 ParaX/Projector | +0.014 | 0.004 | +0.007 | 0.004 |
| 仅 Frozen Forward ParaX | -11.598 | 1.898 | -12.578 | 1.899 |
| 仅 Projector | -24.942 | 2.541 | -23.766 | 2.575 |
| ParaX + Projector | -24.535 | 2.842 | -25.289 | 2.900 |
| ParaX + Projector + 对齐 | -17.677 | 2.079 | -16.404 | 2.107 |

这组固定样本分析比仅看 final mAP 更直接：基线的旧任务输出几乎不漂移；持续更新共享 ParaX/Projector 时，同一图像的旧类别分数与排序明显变化。它支持“共享图像/特征变换在后续任务持续更新，旧任务路径接收到改变的特征”这一主要机制。这里未保存逐任务中间视觉特征，不能仅凭 logits 精确量化每层 feature drift。

训练日志中 task7 最后一个 epoch 的 Full/Person/Face 总 residual/token ratio：原 ParaX 为约 0.60/0.91/0.73；ParaX+Projector 为约 1.33/1.80/1.56，其中 Projector 校正量约 1.31/1.71/1.51，明显主导总扰动。方向对齐将总比例压到约 0.08/0.08/0.07，但 task0 性能和旧任务稳定性仍未回到基线。该比例来自训练 batch，不是 held-out feature drift；“残差较小”不足以保证旧任务 logits 稳定。

## 结论与下一步

本次 Projector 没有修复 Frozen Forward ParaX：无约束 Projector 主导并放大表征偏移，对齐约束虽抑制幅度，却没有恢复基线性能。继续在相同内部插入点扩大 Projector、加专家、调 Router 或直接看 test 选 scale，缺少依据。

下一步改为**保持 Frozen Forward 的 CLIP 图像 tokens 完全不变**。在 Task Forward 完成 patch summarization、输出每个 task lane 的最终特征后，使用一个跨 Full/Person/Face 共享的 parameter-matrix expert center；task0 学中心，随后冻结中心。每个 task 使用独立的小 Router，训练结束即冻结该 Router，评估旧类别时用其所属 task 的 Router。这样仍由一套冻结 CLIP 编码器和一套共享专家中心处理三个 level，但后续 task 无法修改旧任务使用的路由。先用严格零输出、小固定残差做 task0–2 validation，配对比较相同 B0，记录单路 mAP、固定 anchor 旧类 logits 和 gate 差异。只有 task0 精度与旧任务稳定性同时接近或优于基线，才进入完整八任务。
