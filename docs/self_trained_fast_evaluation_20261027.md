# 自训练四组评估吞吐优化（2026-10-27）

本次是原已授权实验的只读评估接续，不新增训练或调参。原4进程各72线程/workers0，合计约130核且GPU多数空闲。小张量预处理线程调度可能是瓶颈，需计时验证。

## 完整配置

- 数据EMOTIC原Val/Test，seed1，8阶段，历史samples_intersect_seen_classes人群及算术平均。
- 四学生SELF_BASE、SELF_NO_CONSTRAINT、SELF_MAIN_CONSTRAINT、SELF_MAIN_ADAPTER_CONSTRAINT，原32阶段compact checkpoint；推理不载入教师。
- 来源../multi-lane-main-self-trained-audit-recovery-20261027/output/self_trained_view_transfer/resumed_seed1_20261027_03；训练/封装Git fb5f6cc745ae6c103990876e7bc0e8370a1d155f。
- 入口python -m multi_lane.track_a.self_trained_fast_evaluation benchmark|run。
- B64、模型72线程、AMP/TF32、threshold.5、原逐batch CPU sigmoid和完整诊断不变；0训练更新，optimizer/LR/scheduler不适用。
- DataLoader改为2 worker，每worker1 Torch线程，spawn/prefetch1/persistent_workers/pin_memory。4worker仅作相同预测吞吐预检候选。4评估并发，仅GPU0；新启动需GPU至少4GB及SHM至少1GB。
- 数据../multi-lane-main/datasets/EMOTIC；CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
- 新输出output/self_trained_fast_evaluation/seed1_20261027_01；日志logs/self_trained_fast_evaluation/seed1_20261027_01；原输出不覆盖。

## 复用与正确性门控

新增文件外所有multi_lane源码必须与训练Git字节一致。task0/7 Val各128人比较workers0/2/4，逐batch像素/标签/ID/mask SHA及fused/full/person/face logits要求精确一致。分别计时冷启动及复用worker加载，加载倍数不能当整网倍数。

原controller/子进程可逆暂停供隔离benchmark，不通过则恢复。通过后只停止已核对cwd/cmdline的原评估及controller，保留checkpoint和所有产物。旧完整pass须JSON、fused NPZ、view NPZ齐全，核验人群/标签/mask/类别/B64长度/四路指标后按SHA复制；未落盘当前pass重算。原训练完成和跨组数据流验证保留，另记录training_git/evaluation_git，不改写权重来源。32学生checkpoint评估前后SHA一致、模型参数digest不变，完成所有完整Val/Test/ON-OFF后导出summary.csv/completion.json。

本次维护三个上下文文件，不改AGENTS。
