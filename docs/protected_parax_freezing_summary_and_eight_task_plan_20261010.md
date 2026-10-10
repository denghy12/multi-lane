# 专家冻结与复用：现有结论及八任务验证计划

本会话只研究 ParaX 在增量任务中的专家冻结和复用，不研究其他会话的三路全量学习、视图增删或融合参数搜索。本轮四组始终使用原固定 Full/Person/Face 特征融合，仅改变专家的可训练性与访问范围。

## 已有结果说明什么

最新统一修正版、seed0、task0–2 validation：

| 实际方法 | task0 mAP | task1 mAP | task2最终mAP | 三任务平均mAP |
|---|---:|---:|---:|---:|
| 不加ParaX的共享基线 | 58.8817 | 57.4995 | 44.3281 | 53.5698 |
| task0学专家池，之后冻结专家，只学当前任务Router和小投影 | 60.7111 | 58.7637 | 45.2055 | 54.8934 |
| 每任务只学、使用自己的新专家 | 60.7111 | 58.5565 | 45.1656 | 54.8111 |
| 冻结旧专家，组合旧专家并学习新专家 | 60.7111 | 58.8070 | 45.0411 | 54.8531 |

1. 早期持续更新共享中心时，后续任务会改变旧任务依赖的有效Adapter，导致严重漂移。现在不仅冻结旧专家，也冻结旧Router、小投影、访问集合和原任务路径。旧参数逐值不变，固定2285个旧人物的五类mAP到task2仅变化约−0.0023点，说明保护旧路径有效。
2. 冻结池最终高于基线0.8773，平均高1.3236；它后续只训练约9千个ParaX参数，另两组约7.5万。收益有希望，但只有前三任务、一个seed，不能外推至八任务。
3. 复用旧专家相对只学新专家，task1新增三类+0.6679，task2新增三类−0.9869；最终−0.1245。旧知识被访问不等于产生正迁移。不能断言不断扩池优于冻结小池。
4. 同模型关闭ParaX残差后，再开启带来约+0.26–0.27最终mAP；关闭后的模型也高于独立基线。因此本轮不支持“联合训练必然拖累原模块”。分阶段学习仅作为条件式后续方案，不能机械进入。
5. 当前任务Router的三视图均值较接近，尚无有效level专门化证据；这是后续机制研究，当前八任务先回答冻结/复用能否保持收益。不加入静态Router新因素。
6. 常规forgetting包含评估人群扩大，需同时报告固定旧人物AP与扩展人群旧类AP，不把所有下降都解释为旧参数漂移。

## 本轮要回答的问题

- 冻结两位参数专家、只更新新任务Router与小投影，能否支持后续全部七个增量任务？
- 只学新专家的收益能否超越小冻结池，是否值得持续增加参数？
- 在相同新增专家预算下，读取冻结旧专家是否比仅使用新专家更好？
- 这些结论是否在三个seed、validation和test方向一致？test只作固定配置评估，不用于选方法、epoch或超参数。

## 架构位置和冻结含义

共享冻结CLIP → 原Frozen Forward/Summarize → 各已见任务Task Forward最终归一化CLS → 该任务Router组合共享池中的低秩参数矩阵 → 任务专用32→32投影与受控残差 → 原固定三视图特征融合 → 分类头。

ParaX在 **Task Forward输出后**，不在Frozen Forward内部。CLIP、Image-token Adapter关闭状态以及此前patch Projector关闭状态均保持不变。“小投影”是ParaX低秩内部的任务专用零初始化32→32层，不是修复CLIP patch特征的Projector。每个专家是一对E_A/E_B参数矩阵，三种level共享池和同一任务Router，不是三套视图独立专家。旧任务不得访问未来专家。

| 实际方法 | task0 | task1–7 |
|---|---|---|
| 原基线 | 学原任务模块 | 学当前原任务模块；旧任务冻结 |
| 学完专家池后冻结 | 学专家0、1和任务Router/小投影 | 两个专家始终冻结；只学当前Router/小投影及原任务模块 |
| 只使用新专家 | 学专家0、1 | 第t任务只用、学习专家2t和2t+1 |
| 复用旧专家并学习新专家 | 学专家0、1 | 旧专家冻结，组合0至2t+1；仅当前两位新专家和当前Router/小投影可学 |

每任务结束封存旧专家、Router、投影、访问mask、Selector、Prompt、分类头。固定旧样本审计逐任务进行，不能用浮点阈值检查代替参数逐值检查。

## 完整正式配置（启动前声明）

- 数据集：EMOTIC Track-A增量协议，task0–7，类别数5+7×3=26；训练只使用train，逐任务评估validation和test。
- 方法：以上四组，seed0/1/2，共12次训练；同一次训练的固定epoch终点权重分别评估两个split，不分别重训validation/test，不按任何split选最优checkpoint。
- epoch：30/task，240/组；预期13,950有效optimizer updates/组，zero skips。不从旧三任务compact checkpoint续跑。
- train/eval batch64，workers2，OMP/MKL线程1。
- Adam每任务重置；原任务LR0.0125（0.05×64/256），ParaX LR0.0004，weight decay0；cosine，最低LR0，warmup0。
- AMP初始scale1024、growth interval1e9；identity/旧anchor用FP32关闭TF32，训练TF32沿用原设置。
- CLIP ViT-B/16冻结；共享Selector10、Prompt10、前5层prefix；任务类别按原MULTI-LANE归属汇集，推理不使用真实任务ID。
- rank32，Router hidden16，image-only；16个预分配参数专家矩阵对，两位/任务；冻结池组只激活最初两位。所有ParaX组相同槽位和初始化流程，避免池大小不同造成不公平初始化。
- task-local零初始化32→32投影，fixed scale1，实际原始残差可微比例界0.02；不启用hard cap。
- legacy_full_zero/joint_bce；融合BCE+0.1可靠视图辅助BCE。
- 固定特征融合：可靠Face[0.64,0.16,0.20]；不可靠[0.80,0.20,0]。可靠条件保持原有效检测/无歧义/短边≥24/score≥0.6。
- 不加入Image-token Adapter、patch Projector、level embedding、蒸馏、分阶段训练、动态融合或视图增删。
- GPU：只使用核实空闲的GPU0–7，每GPU同时一组；最多8组并行，其余组自动分配到完成后重新核实空闲的卡。不终止其他进程。
- 相对输入路径（相对于服务器新worktree）：数据`../multi-lane-main/datasets/EMOTIC`；权重`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；完整Face清单`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 完整Face清单的train/val文件与旧三任务清单SHA相同，新增test输入；CLIP实际解析权重SHA沿用已核验5806e77c…。
- 分支：`exp/parax-freezing-eight-task-locked`。独立本地managed worktree避免混入主工作树另一条路线的未提交代码；服务器smoke worktree名`multi-lane-main-parax-freezing-eight-task-locked`，正式队列使用独立的`multi-lane-main-parax-freezing-eight-task-formal`，通过Git-only同步。
- 正式batch：`protected_parax_freezing_8task_seed012_valtest_20261010_01`。
- 输出`./output/emotic_protected_parax_full/<batch>/seed<seed>/<method>/`；日志`./logs/emotic_protected_parax_full/<batch>/seed<seed>/`；control记录配置、GPU映射、退出码和最终汇总。
- 产物：每task compact checkpoint；val/test融合和单路分数、AP；两split残差开关对照；旧参数/访问范围/CLIP/hash及固定anchor审计；全部旧task同人物AP；三seed平均/标准差JSON及CSV。

16槽位相对原三任务6槽位会改变ParaX随机参数流，因此不要求新task0精度严格等于旧三任务值。只在本轮相同槽位、相同seed、配对初始化和采样协议内比较，不混用历史绝对基线。

## 运行入口与验证

```bash
export DATA_ROOT=../multi-lane-main/datasets/EMOTIC
export CLIP_CHECKPOINT=../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt
export FACE_MANIFEST_ROOT=../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
export OUTPUT_ROOT=./output/emotic_protected_parax_full
export LOG_ROOT=./logs/emotic_protected_parax_full
export BATCH_ID=protected_parax_freezing_8task_seed012_valtest_20261010_01
export GPU_POOL=0,1,2,3,4,5,6,7
export SEEDS=0,1,2
export EPOCHS=30
export UPDATES_PER_TASK=0
/opt/conda/envs/ddp/bin/python scripts/emotic/launch_protected_parax_full_evaluation.py
```

先本地和服务器完整单测，再真实ViT四组八任务smoke：seed0、每task1epoch/2有效更新，共16updates/组，同时验证val/test导出、所有八任务identity/旧参数访问保护、三任务到八任务汇总逻辑。smoke只验证实现，不作为精度结论。通过后启动正式12组，做一次有限启动检查后停止实时监督，结束再统一分析。

## 实验结束后的判断

主要比较同seed、同split的Final/Average mAP；旧类稳定性用固定人物和原forgetting共同判断；拆分当次新增类AP，防止总体增益只来自早期任务。复用组必须优于只学新专家且旧路径稳定，才能支持知识复用。如果冻结池稳定领先或接近，优先选择其较小增量预算。如果只有seed0提升，不能宣称收益稳定。任何test结果都不反向用于本轮配置或后续权重搜索。

## 预检与后台执行状态

业务实现eb04972已通过本地和服务器291项完整单测。真实四组八任务smoke已启动（smoke batch protected_parax_freezing_8task_smoke_20261010_01），当前GPU0–5由其他任务使用，调度器只分配空闲GPU6、7，其他组自动排队。smoke尚未完成，不把其小步精度当作正式结果。

正式队列使用另一个独立checkout运行，不更新正在预检的工作树。增加WAIT_FOR_SMOKE门禁：必须看到smoke completed与完整比较产物，才开始正式GPU调度；smoke失败或控制器提前退出则记录prerequisite_failed并停止。正式控制流程放入tmux后台，预检和训练无需持续人工监督。尚未开始30epoch训练时，必须称“已排队等待预检”，不能称12组已经训练。
