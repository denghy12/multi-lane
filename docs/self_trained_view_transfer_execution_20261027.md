# 自训练单一三视图模型：执行配置

2026-10-27用户授权执行，并明确最终模型须由事先定义的流程自己训练得到，不接受把别的模型参数组装成最终模型。该要求取代旧方案的“A已训练参数初始化＋B0头复制”。新分支 `codex/self-trained-three-view-transfer-20261027`。

## 1. 哪些参数来自哪里

学生只加载标准预训练CLIP ViT-B/16视觉权重；Task Selectors、Prefix Prompts、Image-token Adapter、当前实验Person Attention、分类头均在学生构造时初始化。训练过程中不向学生复制A/B/B0的任何已训练任务参数或分类头，后续任务也只用自身前一任务的参数按原MULTI-LANE `activate_task`规则初始化。

A/B/B0已存checkpoint只用于构建冻结教师的输出目标。教师head仍为原研究中共同B0头，学生head始终是自己训练的head。两者不是同一对象。最终学生checkpoint只含自己学得的任务参数，推理入口不构造教师、不读取教师参数文件，不进行参数拼接、来源路径选择或推理模型均值。它属于有教师监督的自训练流程，不是“完全不使用外部教师”；用户排除的是外部已训练参数组装。

因此新学生不预称为32.8864的换头模型。32.8864/39.8687、历史B0 32.6800/39.5292和双模型R5 33.6248/40.5129继续单列为既有参照；本轮另有从统一初值自训练的匹配基线。

## 2. 事先固定的训练流程

每任务按顺序完成：

1. **硬标签学习30epoch。** 学生使用原三视图Image-token Adapter-ASL、原Summarize与Task Forward、原固定特征融合；当前新增Person Attention关闭且不训练。自己的当前类别head与任务表示一起学习。
2. **小步学习5epoch。** 冻结上阶段由学生自己学出的head，启用学生从零开始的Person Attention；其他当前任务参数可以适配。比较原硬目标、硬目标＋主参数功能项、硬目标＋主参数和Adapter功能项。
3. 封存当前任务的全部已训练任务参数与head旧类别行。下一任务从学生自己的前一任务Selectors/Prompts继承，绝不重装外部教师checkpoint覆盖旧学生路径。适配结果会进入下一任务自身初始化，是预先定义的新训练轨迹。

前30epoch的DataLoader使用历史global RNG流；额外5epoch、教师构造和TRAIN探针在独立RNG上下文中运行，退出后恢复前30epoch的随机流。四组实际消费的前两批IDs、targets、增强像素逐epoch跨组核验，避免“同seed却数据不同”。参数经过不同适配后，从task1起可以不同，这是方法作用，不虚称各任务参数初值完全相同。

保留Full学习阶段并不数学保证后续Full能力不变。Full/Person/可靠Face与融合全部输出，专门判断新增约束是否保住Full并利用辅助信息。旧任务参数哈希和固定TRAIN探针预测则逐值保护。

## 3. 训练组与目标

| 名称 | 前30epoch | 后5epoch | 学生分类头来源 |
|---|---|---|---|
| SELF_BASE | 原三视图硬标签监督 | 无 | 自己训练 |
| SELF_NO_CONSTRAINT | 同上 | 硬标签监督，Person Attention开启 | 自己训练后冻结 |
| SELF_MAIN_CONSTRAINT | 同上 | 硬标签＋主参数融合功能约束 | 同上 |
| SELF_MAIN_ADAPTER_CONSTRAINT | 同上 | 硬标签＋主参数与Adapter融合功能约束 | 同上 |

后三组更新预算、模块、采样和增强相同；SELF_BASE是无额外阶段参照，不冒充35epoch等预算控制。没有新增Face Attention或MoE。

教师和学生每次读取同一个增强后的Full/Person/Face tensor对象及同一个Face可靠性mask。教师只运行当前任务路径，使用对应任务边界checkpoint，不用task7教师提前指导task0。教师eval/no-grad：

`zT = 0.5*zA + 0.5*zB`；`D = mean((zStudent_current − stopgrad(zT_current))^2)`。

D只约束当前任务类别；不对不同教师Selector槽位强行逐元素对齐，也不额外规定三个学生分路分别等于三个不同来源。主参数（Selectors/Prompts/Person Attention）目标为原融合BCE＋0.1有效视图辅助BCE＋beta_M D；Adapter目标为原融合ASL＋0.1辅助ASL，最后一组再加beta_A D。原硬标签目标权重始终为1，不使用互斥类别softmax或稀释硬标签的概率混合。

beta在每任务后5epoch的首个实际增强TRAIN batch按梯度范数定标一次：`min(1, 0.25*原目标梯度范数/D梯度范数)`。主参数和Adapter分别定标，零信号组beta=0并记录；校准梯度使用固定1024尺度减少AMP中间下溢。所有值随后固定，不根据Val/Test搜索。目标非有限值立即失败，不静默跳过。

## 4. 数据和超参数

- EMOTIC，seed1，完整8任务；Validation与Test均沿用历史`samples_intersect_seen_classes`人群，8阶段mAP算术平均；Face可靠性和固定融合权重保持原配方。
- 可靠Face：Full/Person/Face=.64/.16/.20；无可靠Face=.80/.20/0。最终共享线性head位于融合后，融合后不再L2归一化。
- 只用GPU0。batch64，训练workers2、prefetch_factor1，正式评估workers0；最多4作业并行，每次新起作业至少空闲6000MiB GPU显存及1024MiB共享内存，间隔12秒。不改变训练batch去追求吞吐。
- OMP/MKL72线程；AMP/TF32；Adam每任务/每阶段重置，WD0，cosine，无warmup。阶段一主LR.0125、Adapter LR.0004；阶段二所有可训练组LR.0004。
- Adapter b32、第2层index1、scale.03、ReLU；ASL gamma_neg9.8/gamma_pos0/clip.05。Person颜色扰动关闭。Person Attention位于第6层摘要入口，task Selector Query读取Full摘要，inner64、4heads、scale.1、平滑cap.1。
- 预训练CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- 数据 `../multi-lane-main/datasets/EMOTIC`。
- Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 教师B0 `../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`；教师A同来源根的`ATTENTION_BOUND/train`。
- 教师B `../multi-lane-main-head-anchored-eight-face-20261025/output/head_anchored_eight_face/formal_seed1_20261025_01/FACE_JOINT/train`。
- 教师来源校验参考根 `../multi-lane-main-selector-face-controls-20261026/output/selector_face_controls/formal_seed1_20261026_01`。

## 5. 入口、输出与自动执行

训练/评估/初始化审计入口 `python -m multi_lane.track_a.self_trained_view_transfer {train,eval,audit}`；GPU0队列入口 `python -m multi_lane.track_a.run_self_trained_view_transfer`，接受上述source-run/person-run/dual-run/reference-root/data-root/clip-checkpoint/face-manifest-root和run-id。

先执行专项和完整回归、fresh初始化/CPU-CUDA RNG/初始TRAIN预测审计，再做2任务、每阶段2更新smoke，只评Val首64人，不打开Test。smoke验收旧参数/旧预测保护、自己的head冻结、教师不更新、跨组增强像素和标签逐值一致、无外部学生参数加载；通过后才自动启动正式8任务。

smoke输出 `output/self_trained_view_transfer/smoke_seed1_20261027_01`，正式输出 `output/self_trained_view_transfer/formal_seed1_20261027_01`，日志用对应`logs/`前缀。服务器独立worktree `multi-lane-main-self-trained-view-transfer-20261027`，Git-only同步。主树/test-only及所有旧实验产物不修改。

每任务先保存自己前30epoch的compact和可恢复边界，再保存后5epoch完成的compact/optimizer/scheduler/scaler/RNG边界；完整训练后自动排队各组Val/Test。各组有Person Attention时，同时保存完整8阶段Attention ON/OFF，OFF仍保留共同训练后的其他参数，不当作独立重训基线。所有数据流检查通过前不打开正式Test。

最终产物包括各任务compact/recovery、两阶段history、beta与梯度定标、批次追踪、旧任务保护审计、全部视图与融合NPZ/指标、ON/OFF、最终与阶段平均mAP/cF1/oF1、来源SHA及完成标记。仅使用学生进行正式评估。教师源码/参数依赖只存在训练阶段，其训练成本须单列；完整来源包含B0/A/B各30epoch，再加学生35epoch约125epoch/任务，不能声称和无教师30epoch基线等总训练预算。

## 6. 结论边界

这是强教师辅助的一套自训练流程，不是已证实的新Attention机制，也不保证比32.8864或R5更高。此前固定头后5epoch和OOF多视图→Full蒸馏都失败过；本轮的可检验差异是学生保留三视图、使用同增强、目标为完整双来源融合功能、显式区分ASL梯度路径，且学生不装配任何外部已训练任务参数。

首要成功条件：功能约束组同时优于相同35epoch无约束组和自训练30epoch组的两个Test mAP，并能解释Full/辅助/可靠性人群变化。只拟合TRAIN的D或提高Face单路不算成功。Test已参与过研究探索，完整报告全部固定组；不扩新seed或额外参数网格，不合并main。
