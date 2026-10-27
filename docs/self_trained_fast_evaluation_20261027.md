# 自训练四组评估吞吐优化（2026-10-27）

完整结果更新：恢复run已完成四组8任务训练，fast评估全部Val/Test及ON/OFF已完成。最新结论与基线RNG复现疏漏见[self_trained_view_transfer_results_20261027.md](self_trained_view_transfer_results_20261027.md)。下方未完成/启动/故障为历史记录。

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

## 已完成预检与切换

运行Git a61ffc2，已Git-only部署独立服务器worktree multi-lane-main-self-trained-fast-evaluation-20261027，源码clean。4新增测试和522项全回归均通过（39.236秒）。task0/7 Val各128人，workers0/2/4的所有输入SHA及fused/Full/Person/Face logits均逐值一致，未更新模型重复差也均0。

原单进程72线程加载128人2.514/2.758秒；2worker复用加载1.135/1.495秒（2.21/1.84倍）；4worker1.142/1.597秒，无收益且启动更慢，正式采用2worker。spawn冷启动4.275/5.138秒，按阶段/ON-OFF复用worker。以上为隔离加载吞吐，不等于整网加速。

原controller2879990与4评估2893591/2894118/2894645/2895172核对cwd/cmdline后暂停供测试，通过后SIGTERM并CONT有序终止，原日志/所有落盘阶段/32权重保留。原queue.json会保留旧running快照，以新original_evaluation_handoff.json为准。未落盘pass需重放。

已启动tmux self_trained_fast_eval_seed1_20261027；首4评估PID3002571/3003123/3003989/3004732。BASE Val task0–6、BASE Test task0–3及其他已完成pass逐样本/指标验证后复用，新前向在继续。最终耗时与GPU利用率待完整pass实测。

## 切换后完整阶段吞吐

实测加速（2026-10-27）：GPU0连续10秒均100%利用率、约6–7GB显存；各评估主进程CPU约1–2核（另有2个单线程加载worker），替代此前每进程30余核。切换339.96秒已新增29个完整pass/95004前向人次，另23旧pass/68654人次核验并SHA复用；BASE Val/Test和NO_CONSTRAINT Val已完成，其他自动接续，无报错。实际总量406854人次，剩243196；当前279.46人/秒线性约14.5分钟，考虑后段任务/尾部并发建议估剩余15–25分钟，不再沿用10–13小时。输入与四路logits精确重放、4专项/522回归通过。8份预检日志/协议/吞吐与handoff已回传output/self_trained_fast_evaluation/preflight_snapshot_20261027_01逐SHA核验，归档6e46021af5e6e6ac50f1236a76baa25e70e0466afe9249ec1fb5be4339175e42。正式结果仍在运行，以fast新队列为准，不误读原已停止queue的running快照；remote a61ffc2 clean固定，执行笔记仅本地追加。

实际Val每阶段人数[2285,2375,2381,2397,2397,2397,2397,2397]；Test[3142,4600,4691,5232,5342,5354,5367,5368]，7份ON/OFF模型pass之和为406854。上述吞吐含加载/模型/导出/完整诊断，区别于隔离loader的约2倍提升。此次大幅加速还来自消除四进程同时72线程预处理的CPU争用；不建议继续提高进程数、batch或workers，GPU现已满载且现有B64逐batch输出口径保持。
