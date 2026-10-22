# 三组目标人物空间摘要实验：继承进度迁至GPU0

用户2026-10-22要求暂停原三组，将它们迁到GPU0并行，继承当前进度。源实验是`formal_seed0_20261022_01`，原执行b0d8263；迁移分支`fix/target-summary-gpu0-resume-20261022`。原本地分析文件已保留在提交历史；服务器主树与test-only不改。

## 迁移边界与训练配置

源程序只保存每任务训练结束、reporting之前的完整恢复状态，尚无任意batch中间保存。为避免丢弃task3已经执行的更新，先暂停总控制器，各组各自完成task3后，在原子快照落盘时SIGSTOP。不会回退到task2并重跑当前任务，也不把任意compact checkpoint当完整恢复点。

三组baseline/absolute/target继承task0–3全部权重、120epoch/8760次成功更新，继续task4–7的120epoch/3900次更新，最终仍为240epoch/12660次更新。上述是按已锁定fit人数计算的预期预算，迁移门槛以真实快照核验为准。

EMOTIC、seed0、任务5/3/3/3/3/3/3/3，B64、每任务30epoch、TRAIN原图90/10 fit/holdout、融合BCE+aux.1、原参数LR.0125/空间LR.0004、Adam每任务重置/WD0/余弦退火、AMP1024/固定growth、TF32、workers/eval-workers0均保持。原Full/Person/Face输入、Selectors/Prompt、固定融合、分类头及空间参数形式不变。不添加MoE、Adapter、绑定损失或新的准确率筛选。

输入仍来自源launch的原始路径：`../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP/pretrained/clip/ViT-B-16.pt`、`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

## 怎样保留状态

1. `pause_spatial_boundary`验证每个PID的命令、工作目录和出生时间，先暂停旧控制器，防止后续生成旧GPU队列；只暂停本批次三个训练进程，外部作业不处理。
2. 继承原task3完整恢复文件：包含模型、requires_grad、Python/NumPy/Torch CPU/CUDA RNG、sampler generator、Adam/scheduler/scaler及历史/配对审计。原文件保留并记SHA。
3. 恢复task3训练结束的模型后，只重新执行其validation/holdout报告，不重复训练更新；恢复到报告之前的随机数状态，使后续随机流与原任务流程一致。
4. task4按原协议复制前一任务初始化并重置Adam。这里保留的是原本每任务重置优化器的行为，不能把task3的Adam错误地跨任务继续使用。旧优化器等状态仍随原恢复文件完整保留。
5. 原task0–2 compact/预测及task0–3恢复文件按字节复制到新的输出目录，并核验SHA；其checkpoint来源仍是原Git。边界task3新报告compact及后续任务记录新执行Git，通过`resumed_from`明确继承前缀，评价器逐任务核对来源，不修改旧checkpoint来源字段。
6. GPU0三组均重新保存task3 compact，确认与各自源task3恢复模型逐tensor一致后，才结束旧已暂停PID并释放GPU4/5/6。若恢复检查失败，旧暂停进程和全部原始文件保留。

## GPU0资源与后续评估

三组训练都在GPU0并行，每进程Torch分配器上限16%，启动三组前要求至少14000MiB余量，单进程启动继续要求5600MiB余量。真实八条任务路径资源探针（未训练的未来槽位，仅资源、不报告精度）B64×3视图峰值分配749.95MiB/保留1252MiB，六倍保留约7512MiB；另需CUDA上下文开销。

训练后自动执行跨组三方预算、原图隔离、冻结、恢复/compact一致性核验，然后三组完整validation2397/test5368共六个进程全部在GPU0并行。最后target task0/2/7的位置正常/关闭/同图换框归因也在GPU0，validation-only。不得再调用原GPU1–6评估安排。

新入口：

```bash
OMP_NUM_THREADS=1 /opt/conda/envs/ddp/bin/python -u -m multi_lane.track_a.continue_spatial_gpu0 \
  --source-root ../multi-lane-main-target-anchored-summary-20261022 \
  --source-run-id formal_seed0_20261022_01 \
  --run-id formal_gpu0_resume_seed0_20261022_01 \
  --boundary-task 3 \
  --pause-record output/target_summary_gpu0_migration/implementation_20261022/pause.json
```

新日志：`logs/target_anchored_summarization/formal_gpu0_resume_seed0_20261022_01/`；新结果：`output/target_anchored_summarization/formal_gpu0_resume_seed0_20261022_01/`。旧来源目录、checkpoint、日志均保留。控制器日志和预检在`logs/target_summary_gpu0_migration/implementation_20261022/`，暂停记录/资源检查在同名`output/`路径。

## 正式迁移前核验

420项完整测试已通过，包含配置漂移/不完整RNG拒绝、只读复制与来源映射、恢复后下个任务初始化、PID身份拒绝、原命令训练参数保持。第一次资源探针因读取错误的Face配置字段在前向前退出；已按归档嵌套schema修正，第二次资源探针成功。未改源权重。

GPU0三组真实恢复smoke已启动：从旧2task×2update短跑的task0边界恢复，只新增task1两次更新，要求其task0/1六份方法状态与未中断短跑逐值一致，再跑六进程评价和位置归因。正式迁移须等完整smoke通过，不把短跑精度视为方法效果。

当前本文写入时尚不声称正式三组已迁移完成；后续追加真实暂停、恢复、旧PID退出及GPU0首轮状态。

## 恢复短跑通过

三组恢复后的task0/1六份模型状态与未中断来源逐值相同；task1完整rng、sampler_generator、training_state、requires_grad也逐值一致。三训练、六评估及位置归因全部exit0，正常预测重放及旧参数/恢复检查通过。只新增负向保护回归，运行代码未再变。原三组task3约第21–22轮，暂停器继续等待保存边界；正式接续控制器将在该边界后自动于GPU0启动。
