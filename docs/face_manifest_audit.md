# EMOTIC Face manifest 与目标人物匹配审计

本工具是 Full/Person/Face 动态路由计划的阶段 0。它只生成离线人脸检测结果、目标人物匹配、覆盖率统计和人工复核图，不训练模型，也不读取 test split。

## 固定协议

- 数据：EMOTIC `train` 与 `val`，使用原图、人体标注框和稳定 `sample_id`。
- 检测器：InsightFace SCRFD `det_10g.onnx`，`640×640`，阈值 `0.5`。
- 执行后端：`CPUExecutionProvider`。manifest 会记录 checkpoint SHA-256、依赖版本和全部检测参数。
- 匹配：脸中心位于目标人体框内且脸框至少 50% 被人体框包含；随后以检测置信度、包含率、头部相对位置和相对面积评分，对同图全部人物与脸做一对一全局分配。
- 歧义：同一人物前两名候选分差不超过 `0.08` 时标记为风险样本。该值是“错配风险代理”，不是有 Face GT 时才能计算的真实错配率。
- Face 裁剪定义：匹配脸框四周扩大 15%，限制在原图内；未来输入采用 pad/letterbox 成正方形后 Resize 224，不再二次随机裁掉脸。
- 无脸或无法可靠匹配时保持 `valid_face=false`，未来 Face 分支必须 fail closed，不能用整图冒充脸。

## 运行

```bash
RUN_ID=face_manifest_v1 \
  bash scripts/emotic/run_emotic_face_manifest_audit.sh
```

真实数据 smoke 可增加 `MAX_IMAGES=8 VISUAL_SAMPLES=8`。完整运行支持相同 `RUN_ID` 断点续跑，检测缓存按图像逐行刷新。

完整审计结束后，可运行以下安全等待器验证真实Face crop经过letterbox、CLIP normalization和冻结
ViT-B/16 GPU forward后形状/数值有限。它要求任一卡至少6GB空闲且利用率不高于10%，连续两次满足
才启动；只做4个样本的前向，不训练：

```bash
RUN_ID=face_manifest_v1 bash scripts/emotic/wait_and_run_face_manifest_gpu_smoke.sh
```

## 产物

所有结果位于 `output/emotic_face_manifest/<RUN_ID>/`：

- `detector_config.json`：检测器版本、权重哈希和协议；
- `detections/{train,val}.jsonl`：每张图只检测一次的原始脸框缓存；
- `manifests/{train,val}.jsonl`：逐目标人物匹配结果；
- `summaries/{train,val}.json`：覆盖率、尺寸、多人物歧义及 8 个 task 当前类有效正例；
- `visual_audit/*.jpg`：优先展示歧义、多人、失败和小脸样本的人工复核图；
- `audit_summary.json`：总入口，并明确 `training_started=false`。

日志位于 `logs/emotic_face_manifest/<RUN_ID>.log`。

阶段 0 的决策重点不是追求一个总覆盖率，而是确认：每个 task、尤其 task6/7 的每个当前类是否仍有足够有效正例；小脸与多人场景是否集中在特定任务；一对一匹配的人工抽查能否支持后续 Face 分支。结论通过后才进入完整训练。
