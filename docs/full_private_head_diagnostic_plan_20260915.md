# Full分类头独立诊断方案

状态：已实现，服务器221项测试与H0/HF真实task0 smoke通过；正式batch
`full_private_head_seed0_20260915_231942`已启动等待GPU0/1。目标是在预测来源诊断指向Full后，判断
Full与Body/Face共用分类头是否是性能损失的重要机制。

## 配对配置

只运行seed0 validation，不访问test：

- H0：Full/Body/Face使用同一个共享线性head；各路先产生logit，再按固定权重融合。
- HF：Full使用独立线性head；Body/Face继续共用原head；各路logit按同一固定权重融合。
- HF的Full head从共享head逐元素复制初始化，且创建过程不改变全局RNG。两组初始logits一致。
- 两组均共享冻结CLIP ViT-B/16、Selector、Prompt、Image-token Adapter b32；保持P0全部数据、Face mask、
  BCE/ASL目标、aux0.1、optimizer、LR、scheduler、batch与增强协议。

完整训练：EMOTIC B5-C3，8 tasks，30 epochs/task，train/eval batch64，workers2，seed0；Adam每task重置，
主LR0.0125、Adapter LR4e-4、cosine到0、无warmup/weight decay；Adapter layer1/scale0.1/ReLU/independent；
AMP/TF32。可靠Face权重0.64/0.16/0.20，否则0.80/0.20/0。记录早中晚各3 batch路径梯度、逐路validation
scores、普通融合scores及每task compact状态。离线同时报告logit与概率融合和2000次原图组bootstrap。

入口：`scripts/emotic/launch_multilane_track_a_full_private_head_val.sh`。输出到
`./output/emotic_track_a_full_private_head/<batch>/`，日志到`./logs/emotic_track_a_full_private_head/<batch>/`。

## 判断规则

首先要求H0与历史P0 final mAP差值不超过0.10，确认逐路logit实现没有产生明显协议漂移。Full-head机制获得
初步支持需同时满足：HF final至少高于H0 0.10、average不下降、Full单路不下降、到R1的差距缩小。
满足后再补H0/HF seed1/2；不满足则停止该方向，按已有诊断转向Full Adapter或Selector/Prompt。

HF同时改变Full分类自由度和共享head接收的Full梯度。即使有效，也只能说明“Full分类头绑定”这一干预有价值，
还不能区分是Full需要不同决策边界，还是Body/Face共享head从移除Full梯度中获益。后续需按结果细分。
