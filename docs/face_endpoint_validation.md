# Face endpoint seed0 validation

## 目标与边界

本阶段只回答一个问题：在已锁定的 Full/Person 概率融合锚点上，可靠 Face 是否提供额外的
validation 信息。它不搜索 Face 网络结构，不读取 test，也不启动三路动态 Router。

固定锚点为：

```text
FP = 0.80 * p_full + 0.20 * p_person
FPH = (1 - beta) * FP + beta * p_face
beta in {0, 0.05, 0.10, 0.20}
```

仅当 `valid_face && !ambiguous_match && face_short_side >= 24 &&
face_detection_score >= 0.6` 时使用 Face。其他样本的输出逐元素精确等于 FP，不能以 Face
占位图改变结果。`beta=0` 是同批离线重建的 FP 锚点。

## Face expert

- 数据：EMOTIC train/val；Face 框来自阶段0冻结的 `emotic-face-manifest-v1`。
- 训练样本：当前 task 标签相交且 `valid_face && !ambiguous_match`；过滤发生在 DataLoader 之前，
  因此无脸和歧义样本不会进入任何 BCE/ASL loss 归约。
- 输入：manifest 中 margin=0.15 的 Face crop，pad/letterbox 成正方形后 Resize 224；train 只加
  RandomHorizontalFlip；CLIP normalization；无二次 RandomResizedCrop、无 ColorJitter。
- 模型：MULTI-LANE，Image-token Adapter zero-based layer1，bottleneck32，Adapter LR4e-4，
  residual scale0.1，ReLU，independent。
- 优化：30 epochs/task，batch64，main LR0.0125，Adam，per-task cosine，eta_min0，no warmup，
  main BCE + Adapter ASL 9.8/0/0.05，AMP/TF32 on。
- 评估：seed0、完整8 tasks、reporting split=val，保存每task实际概率/logits/targets/stable IDs，
  不保存 checkpoint。

内部训练 epoch 的 validation loss/mAP只在有效且非歧义 Face 子集上计算；最终 task 指标与 score
dump仍覆盖原始完整 validation 样本池。无效样本使用 CLIP 均值占位图以保持 batch 形状稳定，
但只用于生成会被融合 mask 丢弃的占位分数。

## 来源审计与选择规则

Full/Person 复用 `20260903_134720` 的 seed0 完整8-task validation score。融合入口逐字段验证
两者与 Face 的训练预算、loss、Adapter、精度模式和 split，并校验 Face audit 与两个 manifest
的 SHA-256。任何不完整任务、ID集合不一致、target错位或 score 无法复现原指标都会失败关闭。

输出比较全量 final/average mAP，并记录各 beta 的逐task/per-class AP、可靠样本率、可靠/不可靠
Face 分组指标。赢家依次按 final mAP、average mAP、较小 beta 排序。只有全量 final validation
mAP 严格超过 FP 锚点，才允许进入下一阶段动态三路 Router；本阶段无论结果如何都不运行 test。

## 入口与产物

- 单次训练：`scripts/emotic/run_multilane_track_a_face_endpoint_val.sh`
- 安全排队及融合：`scripts/emotic/launch_multilane_track_a_face_endpoint_val.sh`
- 训练产物：服务器外部目录
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/<batch>/`
- 控制与融合结果：`output/emotic_track_a_face_endpoint_val/<batch>/`
- 日志：`logs/emotic_track_a_face_endpoint_val/`
