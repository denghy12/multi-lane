# 完整八任务续跑与可靠Face摘要Attention实验

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
