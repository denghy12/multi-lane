# 固定真实基线，组合完整专家残差输出：实验计划

日期2026-10-12。用户授权修改并在空闲GPU运行。只研究增量冻结/复用，不纳入全量类别学习。正式seed0、task0–7 Validation与Test，不用前三任务替代八任务，不按Test选择配置。

## 动机和对照

上一轮固定真实基线后，参数矩阵混合的八任务val/test：基线42.1912/32.0610，只新专家41.9417/31.8895，旧专家复用＋新专家41.9687/31.8734。旧路径稳定，却尚无泛化/复用收益。本轮验证完整残差函数输出组合，不再把旧A/B经过新任务P后称为完整旧功能。

| 对照 | 做什么 |
| --- | --- |
| 固定真实基线 | 直接恢复已训练的逐任务原checkpoint，不重训 |
| 只学新完整专家 | 每任务增加两个低秩残差函数，仅组合自己的两个 |
| 复用旧完整专家＋新完整专家 | 同样新增两个，旧完整函数不更新，当前任务组合旧函数并学习新函数 |

三个视图共享一个CLIP和同一专家池/当前任务Router，没有三套独立专家。位置：Task Forward最终归一化CLS之后→完整残差输出组合→保持identity的归一化→固定三视图特征融合→冻结分类头。Frozen Forward内部不加模块；Image-token Adapter及旧patch Projector关闭。

## 函数定义、冻结及严格零起点

任务s新生的两个专家各有独立A_i/B_i，二者共享归属任务的小投影P_s：

```text
E_i(x) = fixed_scale · B_i GELU(P_owner(i) A_i LN(x))
new(x) = Σ_{i∈当前两个} softmax(z_current)[i] E_i(x)
old(x) = Σ_{i∈全部可访问旧专家} softmax(z_old)[i] E_i(x)
r_t(x) = new(x) + tanh(beta_t) old(x)
y = x + smooth_bound(r_t(x), .02 ||x||)
```

- 任务结束封存A/B、归属P、Router、beta及访问范围；全局LN、固定scale从一开始冻结。旧任务不能访问未来专家。每个旧专家依赖的完整计算不经过新任务P。
- P_t权重/偏置严格零初始化，A/B随机；beta_t=0。新函数初始输出0，非零旧函数的贡献初始也为0，因此原基线严格identity。
- P_t首先有有效梯度，后续A/B和Router可学习；beta_t在0处导数非零，能开启旧功能。不能同时零初始化所有因子。
- beta通过tanh限制在(-1,1)，旧贡献可为负，是有界的带符号残差组合，不是非负凸MoE。若要求平滑非负门在内部零起点，其导数会为0；本轮明确选择可学习的带符号方案，不做隐蔽直通梯度。
- 旧/新组各自softmax，保持新专家组权重总和1，避免旧专家数量增加自动稀释新贡献。它和零起点beta是本轮必要的明确结构变化，不能将结果完全归因于“仅换求和顺序”。本轮两组新增预算一致；与历史参数混合并非所有初始化/有效自由度精确相同。
- 保存的gate是全体可访问专家softmax的选择分布，位于组内重新归一化和beta之前，不是实际带符号输出系数。分析时须同时读beta，不能用旧gate总量当功能贡献。
- 16槽位，owner(i)=i//2，归属在checkpoint恢复时验证。当前requires_grad参数量两组均75073：A/B65536＋Router8480＋P1056＋beta1；复用组更多Router输出行可生效，不能说有效自由度完全相等。完整bank总602665参数，不含冻结原模型。
- 这是从task0开始学习的独立函数输出组合，没有把上一轮共同参数混合中学到的矩阵直接拆开冒充完整功能。它是MoE式新对照，不是原论文ParaX参数路由。

## 完整配置和路径

- EMOTIC train/val/test，seed0，任务大小[5,3,3,3,3,3,3,3]，CLIP ViT-B/16完全冻结。
- 原基线：`../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed0/baseline`；已训练30epochs/task、13950updates，本轮只恢复。
- TRAIN缓存：`../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed0/train_cache/task{t}.pth`；逐任务核验split、样本ID/标签、维度、来源checkpoint SHA及完整缓存SHA。确定性图像裁剪，AMP原前向转FP32缓存，不混入val/test。
- 数据：`../multi-lane-main/datasets/EMOTIC`；预训练：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。全部沿用原模型和检测协议/SHA。
- rank32、Router hidden16、16专家槽位、2新专家/task、总2%可微残差界、fixed scale1；无level条件/蒸馏/动态融合/熵正则。
- 拟合5epoch/task，batch64，FP32，Adam lr.0004/WD0，cosine每任务重置，clip1；固定融合BCE＋.1可靠单路辅助BCE，按当前类数/26缩放；consistency0；两组使用相同seed+1009*task的缓存排列。
- 固定融合：可靠Face [.64,.16,.20]，否则[.8,.2,0]。可靠为有效/无匹配歧义/短边≥24/检测分≥.6。
- 所有8任务拟合和Validation完成后才创建Test数据集；恢复16份已封存checkpoint只读Test，沿用同源基线Test，不依Test选择模型/轮次或改变配置。
- 入口：`python -u -m multi_lane.track_a.fixed_baseline_functional_experts`；Shell：`scripts/emotic/run_fixed_baseline_functional_experts.sh`。
- 正式batch：`functional_output_reuse_seed0_val_test_20261012_01`；预检batch：`functional_output_reuse_smoke_20261012_02`。
- 产物：`./output/emotic_functional_output_experts/<batch>/`；日志：`./logs/emotic_functional_output_experts/<batch>/`；保存逐任务/单路val-test概率、compact checkpoint、旧函数与旧路径审计、严格identity和梯度/门系数记录。
- 分支：`codex/frozen-functional-expert-output-reuse-20261012`；服务器独立worktree：`../multi-lane-main-functional-output-experts`。Git-only同步；不触碰主/test-only及其他运行中工作树。
- GPU：准备预检期间GPU1–7已被其他并行实验占用；初次smoke_01被idle资源检查拦截，未发生训练，日志保留。用户明确选择共享GPU0显存立即运行，不排队；GPU_POLICY=shared且空余≥8192MiB，现有任务不停止或修改。一个进程依任务顺序拟合两组，共用CLIP及TRAIN缓存。idle模式仍支持完全空闲卡检查。

```bash
GPU=0 GPU_POLICY=shared BATCH=functional_output_reuse_seed0_val_test_20261012_01 \
 bash scripts/emotic/run_fixed_baseline_functional_experts.sh
```

## 启动前验证与结果判断

1. 完整Track-A回归及bash语法检查；新增覆盖真实独立输出组合、owner投影、零起点有效梯度、八任务旧函数保护、2%界及非法checkpoint/访问恢复。
2. 真实ViT预检同样遍历八任务，两组每任务64个TRAIN样本×5轮、64个val/test样本。不能把预检成绩作为正式结果。检查源文件不变、旧参数和旧完整函数逐值不变、旧FP32预测、当前专家与Router有效梯度、旧贡献beta实际有梯度、所有checkpoint只读恢复。
3. 预检通过再启动正式，不覆盖既有batch；启动后只确认真实缓存/训练开始，不持续监督，结束后用户召回统一分析。
4. 复用必须同时优于同源固定基线和只新专家，且旧路径稳定、val/test方向一致；单seed不能宣称稳定收益。若两种输出组合仍低于基线或复用无独立优势，暂停此处专家扩展，不自动加seed/专家/动态融合。
