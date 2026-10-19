# Frozen Forward MoE：与Task Forward调用3／冻结3的对比

2026-10-19用户授权，仅GPU0、seed0、8任务validation/test。独立分支codex/moe-frozen-forward-20261019，从ce70645创建，保留原工作树。

## 假设与唯一新训练配置

对照旧L1_K3_F3（3410546）：均第2视觉块[1]，22专家，768→64→768，ReLU、dropout0.1、up零初始化、residual0.1；Top3 noisy gating（softplus噪声＋.01，推理关闭），任务后按Full+Person+可靠Face训练累计float64 gate mass取Top3，冻结累计且可重复入选，不补足三名新专家。8任务各3个Router/noise（24套），共享专家池参数2,992,000不变。

新插入：Frozen Forward本块MSA及其残差后的全部197图像tokens进入专家，Router读取这批图像tokens的CLS。MoE与该块冻结FFN并联，相加结果写回图像token流，后续Frozen blocks及Summarize读取更新结果。Task Forward不再加MoE残差。骨干权重始终requires_grad=False，但需保持从Summarize穿过后续冻结运算到MoE的梯度，不能detach图像tokens或把整个主干包进no_grad。

原MULTI-LANE在每层Task Forward读该层Frozen block的输入，因此[1]的图像MoE首先影响下一层[2]的Summarize；不是把MoE输出绕过接口直接塞入同层Summarize。禁止末层插入（对任务路径无后续读入）及混用ParaX/Selector conditioning。

保留任务×视图Router意味着每task/view有独立图像流；推理遍历所有已见task，不用真标签选任务、不引入DDAS。各lane按顺序计算限制显存，图像骨干权重共用、不是复制8份可训练骨干；计算量增加，记录耗时/显存。不声称仍仅一条共享Frozen Forward或全旧任务函数不变。

## 完整训练/评估参数

EMOTIC，26类按5/3/3/3/3/3/3/3，seed0，train各30epoch，batch64，预期13950成功更新0跳步；Adam每任务重建，主LR .0125（.05×64/256）、MoE .0004，WD0、cosine无warmup/minLR0。legacy_full_zero/joint_bce、监督系数1、辅助视图BCE .1，无新增负载/蒸馏/正则。
冻结CLIP ViT-B/16；Selector10/Prompt10/5prompt层；同任务三视图shared selectors/prompts/head，pre-head归一化，固定融合可靠Face .64/.16/.20，否则.8/.2/0。Full legacy crop .05–1、Person letterbox margin.15/无jitter、原Face manifest、CLIP mean/std。AMP initial1024/growth1e9、TF32、无梯度裁剪；train workers2、内外eval workers0。

每任务验证前先原子保存恢复状态和最终compact模型；8任务完成后审计/核验，再GPU0双进程并行评估固定全人群val2397人/1705原图和test5368人/3682原图。test不选专家/系数/checkpoint。评估batch64，workers0；显存不足时先停止记录并报告，不能默默改batch/训练预算。报告最终和阶段均值、三视图/可靠Face、旧类遗忘、路由/冻结覆盖、耗时/显存。

输入相对远程独立worktree：data ../multi-lane-main/datasets/EMOTIC；权重 ../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
配对基线 ../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_baseline_seed0_20261017_01/train。
入口 bash scripts/emotic/run_moe_frozen_forward.sh；正式RUN_ID=formal_seed0_20261019_01，smoke RUN_ID=smoke_seed0_20261019_01 SMOKE_UPDATES=2（两任务各2步，无test）；日志logs/moe_frozen_forward/<id>/，结果output/moe_frozen_forward/<id>/。
预训练从相同原始CLIP开始，不从旧MoE checkpoint续训；旧L1_K3_F3只用作已完成对照，不重复训练。

## 启动前检查

完整Track-A测试；新测试覆盖图像而非任务tokens、CLS来源、下一层图像流写回、跨冻结层非零梯度、骨干无梯度、可靠Face统计、task各自Router、恢复重放、初始零输出identity。两任务真实smoke通过冻结及B0配对/保存重放、检查显存和13950预算后才正式。正式前进一步用同smoke权重检查全部8lane的eval形状/数值和成本，不用smoke判断精度。源main/test-only不动，受控代码Git-only同步，运行中服务器HEAD锁定。

## 预检进展

883868f在服务器独立worktree通过全部364项Track-A测试（30.623s）。真实GPU0两任务各2更新smoke已启动，首任务2步/0跳步，恢复状态保存及累计gate-mass冻3完成。smoke结束后执行probe_frozen_moe，只用val前64人物做zero-up FP32 identity和8lane资源检查；后六Router仍未训练，不保存其预测为实验精度。不读取test或改写checkpoint。
