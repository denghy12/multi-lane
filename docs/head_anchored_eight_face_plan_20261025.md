# 完整八任务续跑与可靠Face摘要Attention实验

2026-10-26更新：全部实验完成，结果与机制分析见`docs/head_anchored_eight_face_results_20261025.md`。正式产物已回传核验，后文启动状态按历史记录保留。

执行更新：运行源码d1322ab91e3ebbcf7c4c6c03d856a8bfbb065b16，本地/远程独立树branch、HEAD、upstream一致。488项回归通过（19.002秒），新增6项专项通过。smoke_seed1_20261025_01完成13作业（5训练＋8评估），验证四组采样一致/原prefix保留、MOnly OFF=B0、Face OFF不改Full/Person/无可靠Face logits，原Person-only换头task7精确重放Val43.07406218346569/Test32.886421356337166。smoke完成门槛后正式队列已自动启动。

正式output/head_anchored_eight_face/formal_seed1_20261025_01，tmux head_anchored_eight_face_seed1_20261025。GPU0当前Face共训PID1873009、固定头无M1873585、固定头有M1874348、B0全固定只训M1875111四并行；允许头更新的第四学生组排队。三学生task3首2epoch各156更新/0跳步，Face task0前4epoch各84更新/0跳步，GPU约12413MiB/100%，尚无正式最终指标。队列自动排完整Val/Test及OFF/换头对照，最多4并行，不需再次启动。不更新运行中源码，不修改原pilot/worktree。

原40历史＋15 pilot checkpoint共55份SHA已登记并核验；主树/test-only HEAD与状态SHA未变。256份完成的pilot/smoke/单测产物已同步本地output/head_anchored_eight_face/startup_snapshot_20261025_01并逐SHA核验，归档SHA100949809c3a8def6cee7fe065e69dc0173379ac059f47681bdd6b5827d57a08。当前追加执行记录仅本地未提交，运行源码保持d1322ab clean。完成后核验8阶段最终/阶段平均、逐视图/Face可靠性与关闭模块贡献，明确30epoch共训与额外5epoch适配属于两条不同训练路线。

2026-10-25用户授权执行。新增分支codex/head-anchored-eight-task-face-attention-20261025。当前为实现/待测试状态；不凭pilot Val阻止完整Test核验。

## 1. 两条实验线分开解释

四组5epoch适配继续原seed1 pilot。复用原task0–2权重，不重新训练、不改变三阶段结果；新增task3–7，每任务5epoch/全部可训练参数LR0.0004、Adam每任务重置/cosine/WD0。B0当前任务ASP/H来自原对应阶段checkpoint，旧学生ASP/M/H行保持冻结，独立采样和完整RNG隔离。全冻结组的OFF仍须精确回到B0，其余OFF保留学生共训ASP/H。四组共用完全相同实际TRAIN batch。

新Face版本使用原受限Attention的30epoch共训流程。原Person-only换B0头Val43.0741/Test32.8864仅代表task7；不能把它写成8任务平均。新增Face时主LR0.0125、Adapter/Attention LR0.0004，沿用b32/index1/scale0.03/ReLU/ASL(9.8,0,0.05)/aux0.1/Person颜色扰动0、72线程初始化、AMP65536/growth2000/TF32、Adam reset/cosine/WD0。新增Face模块不消耗原公共初始化/采样随机流；完整训练后应对照旧Person-only的初始公共SHA及各任务CPU RNG摘要。

## 2. Face分支数据流

Full/Person/Face仍分别读取同一组冻结CLIP骨干；各任务共享A/S/P和线性分类头，三路Task Forward保留。

```
Full Summarize ──→ 原Person Attention ──→ 限幅残差 ──→ Person摘要 → Person Task Forward
       └────────→ 新Face Attention ────→ 可靠性屏蔽＋限幅残差 → Face摘要 → Face Task Forward
```

两条Attention都用任务Selector作Query、Full摘要作K/V，位置为第6层摘要之后/任务MSA之前。Face另配独立任务参数，避免同一Query和K/V强制产生与Person相同的修正；不是改成Face内容Query。各自inner64/heads4/scale0.1，平滑RMS上限0.1，输出零初始化。每任务Person196608＋Face196608＝393216参数，8任务新增Face参数1572864。Face可靠性沿用已有valid/nonambiguous/min24/minscore0.6，不新调阈值。只对可靠Face行运行新Attention；不可靠行逐值保持原摘要，整batch不可靠时Face梯度None，Adam跳过Face更新。

原固定融合不变：可靠Face用0.64Full+0.16Person+0.20Face，否则0.80Full+0.20Person；三路各自归一化，加权后不再次归一化。训练ASL只路由Adapter，BCE+原aux用于S/P/Attention/H。任务结束冻结当前ASP与两套Attention，旧任务输入及参数保护仍按原MULTI-LANE。

## 3. 评估配置

- 全部seed1，原EMOTIC划分、历史samples_intersect_seen_classes；按8阶段mAP算术平均。每阶段使用其当时checkpoint及B0 H，不使用未来阶段参数。
- 四学生各完整8阶段Val/Test，native头ON及适用的全Attention OFF。prefix Val逐值匹配原pilot，MOnly OFF逐值匹配原B0对应split。
- 原Person-only共训ASP/M＋B0 H：补完整8阶段Val/Test，ON/OFF；task7必须精确重放43.07406218346569/32.886421356337166。
- 新Person＋可靠Face：B0 H下ON、仅Face OFF、全部Attention OFF；原共训H下ON另作读出控制。仅Face OFF时Full/Person logits逐值一致，不可靠Face及不可靠人群融合logits逐值一致。
- 均保存fused/Full/Person/Face logits、probabilities、ID、class顺序、可靠性mask、每类AP、mAP/cF1/oF1。Test已参与方法探索，不能称独立保留测试。

## 4. 路径与入口

源B0：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
源pilot：`../multi-lane-main-historical-b32-head-anchored-pilot-20261025/output/historical_b32_head_anchored_pilot/formal_seed1_tasks012_20261025_01`。
数据：`../multi-lane-main/datasets/EMOTIC`；CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face清单：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

主入口`python -m multi_lane.track_a.run_head_anchored_eight_face`，学生续跑/评估`multi_lane.track_a.head_anchored_eight`，Face训练复用原`multi_lane.track_a.runner`及historical_b32_modules ATTENTION_BOUND flags，新增`--cross-view-face-attention`。输出`output/head_anchored_eight_face/<run_id>`，日志`logs/head_anchored_eight_face/<run_id>`。GPU0统一最多4作业，显存空闲6000MiB/shm1024MiB、12秒启动间隔；不足自动后排。

新独立worktree Git-only部署，先单测/旧回归与同源码GPU smoke，全部通过后自动正式启动；不修改正在执行/已完成的原pilot源码。学生每阶段先存完整optimizer/scheduler/scaler/RNG恢复状态，训练完自动排Val/Test；Face继续原评估前恢复保存机制。所有源checkpoint只读，保留源git和SHA，prefix不复制成伪新训练来源。产物结束后同步本地核验，再分析两个split和条件贡献。
