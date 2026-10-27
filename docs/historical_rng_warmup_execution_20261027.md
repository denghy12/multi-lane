# 历史随机流修复与本任务额外适配归因

2026-10-27用户授权执行。新实验分支codex/historical-rng-warmup-attribution-20261027。保留原四组全部训练/评估、历史32.6800强底座和之前未提交分析，不覆盖或合并main。

## 1. 精确重放基线

原self训练将全部评估后移，却遗漏每任务报告+视图诊断迭代器的两次全局CPU int64随机种子消耗。修复采用replay_reporting_rng，不加载Val/Test图片或标签。warmup/trained-before-audit恢复边界保存消耗前状态；final恢复边界保存消耗后状态，已完成任务恢复时直接跳过，避免重复。

仅重跑SELF_BASE：seed1、8任务、30epoch/任务，学生只加载标准CLIP视觉权重，任务参数和head fresh初始化；不从其他模型组装参数，也不重新训练三个额外适配学生。每任务warmup保存后，将模型张量、30轮原损失/ASL损失/更新数、Python/NumPy/CPU/CUDA RNG与历史before_reporting checkpoint逐值比较，只作比较，不向学生加载参考参数。任一不一致立即停止，保留该warmup恢复文件，不能放宽门槛当作复现。完成后自动完整8任务Val/Test，与历史两split指标比较容差1e-8。

## 2. 现有学生的只读成对归因

SELF_BASE及三个适配学生的task0/2/7，完整Validation与Test，固定原人群；每个比较WARMUP_ON、WARMUP_OFF、FINAL_ON、FINAL_OFF。warmup ON包括此前任务已训练Attention，但当前新Attention应为初始恒等；明确验证当前类warmup ON/OFF logits差0。当前学生head和旧任务参数须逐值相同；BASE所有参数都须相同，作阴性控制。

所有来源最终模型先在同B64前缀完整前向，和已保存final logits逐视图精确重放。来源final完整JSON+两NPZ经人群/标签/Face mask/类别/指标验证后SHA复制复用；BASE warmup若所有参数相同可复用final。其他warmup需真实完整前向，推理无教师、0训练更新，所有源权重SHA保护。

成对差分：当前ASP变化=FINAL_OFF−WARMUP_OFF；当前Attention新增条件效应=(FINAL_ON−FINAL_OFF)−(WARMUP_ON−WARMUP_OFF)；本任务净适配=FINAL_ON−WARMUP_ON。同时记录当前类AP与旧类logits变化，不能把旧Attention贡献算作当前新增模块。warmup相对原四组BASE已有的差异包含前序封存表征和初始化传递，不能全部断言为单一S/P因果；原同轨迹控制32.1748仍只用于这一归因，不与新修复BASE混作同批对照。

## 3. 完整超参数与路径

- EMOTIC，seed1，GPU0，历史samples_intersect_seen_classes标准，8阶段算术平均。Test参与既有研究探索，不称未用于选择的独立验证。
- B64，OMP/MKL72，AMP/TF32；Adam按任务重置，cosine，无warmup、WD0，30epoch；主LR.0125，Adapter.0004。无新的5epoch适配、无Teacher梯度、无beta变化。
- 原b32 Image-token Adapter/index1（第2层）/ReLU/scale.03；ASL gamma_neg9.8/gamma_pos0/clip.05；原融合BCE/ASL和有效视图aux.1；Person颜色扰动关闭；固定可靠性融合.64/.16/.20或.8/.2/0，原共享头位置不变。
- 训练workers2/prefetch1；评估2个单线程spawn worker/prefetch1/persistent/pin_memory。最多BASE训练+两评估3进程；新启动至少GPU空闲6GB及SHM1GB，间隔8秒。
- 数据../multi-lane-main/datasets/EMOTIC；CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
- 历史参考../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train；其父目录historical_val/test保存原评估。
- 旧学生../multi-lane-main-self-trained-audit-recovery-20261027/output/self_trained_view_transfer/resumed_seed1_20261027_03。
- 旧完整评估../multi-lane-main-self-trained-fast-evaluation-20261027/output/self_trained_fast_evaluation/seed1_20261027_01。
- BASE不运行教师；原入口必须携带的person-run/dual-run/reference-root沿用旧来源，仅配置溯源，不加载到学生。
- 基线输出output/historical_rng_replay/seed1_20261027_01；归因output/self_trained_warmup_attribution/seed1_20261027_01；队列/日志logs/historical_rng_replay/seed1_20261027_01。

## 4. 入口与验收

python -m multi_lane.track_a.run_historical_rng_warmup，source-run/person-run/dual-run/reference-root/data-root/clip-checkpoint/face-manifest-root/student-root/existing-eval-root/run-id均显式给定。先新增专项和全回归，再两任务各2更新BASE smoke（带种子修复、不与30epoch参考比较）和Val首64；再四来源task0/2/7 Val首64归因smoke，通过才正式。

预期产物：修复后BASE的8任务compact/recovery/逐任务历史精确证明、完整Val/Test；原学生24个warmup/final边界的状态保护与两split四条件预测/视图指标、summary.csv差分及源SHA，最终completion.json。所有checkpoint保留，不删除原产物，不改main/test-only。三个上下文文档同步维护，不改AGENTS。
