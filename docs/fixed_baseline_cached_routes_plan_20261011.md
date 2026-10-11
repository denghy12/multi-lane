# 固定真实基线后学习共享专家：实施方案（2026-10-11）

本会话只研究专家冻结，不改其他会话的三路全量实验。

## 先明确 test 范围

最新 `protected_parax_freezing_8task_seed012_valtest_20261011_01` 已完成四组、三seed、八任务的 validation 与 test，不能重复宣称它未跑 test。

按用户“之前 validation 补 seed0 test”先核查到：`protected_parax_audit_v2_seed0_val_20261010_01` 确为仅 validation 的前三任务四组，12份 compact checkpoint 全在。此次补测它们，只评估不训练；若用户指另一批，需按其澄清调整。它们使用原来的6槽位及原策略，不能改成新批的16槽位。

## 新方案要回答的问题

此前联合训练即使旧参数保护成功，原任务模块也可能走向不同的优化轨迹。此次直接复用三个seed、共24份原共享基线checkpoint，不重训基线。只在冻结模型的训练集三视图特征上拟合额外模块，检验专家能否提供增量收益。

| 对照 | 实际操作 |
|---|---|
| 固定原模型 | 每个任务加载对应训练结束的checkpoint；关闭ParaX |
| 固定原模型＋动态图像路由 | 当前Task Forward归一化CLS后、固定特征融合前添加受保护ParaX；Router输入该视图特征 |
| 固定原模型＋任务固定组合 | 同结构、同初始化及同训练顺序；Router始终输入全1常量，学习每任务固定权重；不是均匀专家混合 |

三视图共享一个专家池和每任务Router，没有level embedding。保持参数矩阵组合：分别对专家A、B矩阵加权，再构建输入对应的低秩残差，未改成完整专家输出求和。task0训练两个专家，完成后整池冻结；后续只训练当前Router和rank32小投影；旧Router/投影/访问mask保持冻结。维持16个预分配槽位初始化协议，但只有最初两个专家可访问，避免与上一八任务批次改变初始化预算。

不在Frozen Forward中加入ParaX，不启用Image-token Adapter或旧patch Projector。原Task Forward、Selector、Prompt、分类头及CLIP全部冻结。小投影是受保护ParaX内部32→32零初始化输出映射，不是此前CLIP特征Projector。

## 完整执行配置

- 数据集：EMOTIC Track A，26类，task sizes=5+7×3，原分类顺序。
- 补测：旧四组seed0，task0–2，仅test，无优化器、无训练、无选模。
- 新拟合：seed0/1/2，task0–7，仅validation；先通过task0–2恢复/identity/冻结/有限梯度审计，再继续至task7，重点保留task6结果。前三任务不按test或短程正收益筛选。
- rank32，router hidden16，两个活跃专家，fixed residual scale=1，逐token可微原始残差范数界2%，零初始化task-local projection。归一化后的特征变化另行报告，不能把2%直接当作logit变化保证。
- train/eval/cache batch64，workers2；确定性评估裁剪用于train-only缓存，保存sample IDs及源checkpoint SHA256。缓存来自原任务训练子集，不含validation/test。
- 缓存拟合5 epoch/task；Adam，lr=0.0004，weight decay=0，CosineAnnealingLR reset/task、Tmax=5、无warmup。复用现有缓存拟合器的梯度范数裁剪1.0，两组相同。
- 当前类监督按原legacy_full_zero的类数比例缩放；fused BCE＋0.1可靠单路辅助BCE；consistency=0，无蒸馏、level条件或额外残差正则。
- 冻结视觉提取和常规评估AMP；缓存转换FP32、拟合FP32；FP32对齐审计关闭TF32。训练/常规评估沿用源配置TF32。
- 固定融合发生于Task Forward特征：可靠Face=[.64,.16,.20]，否则[.8,.2,0]。可靠条件valid且无歧义，短边≥24，检测分≥.6。没有动态融合。
- 输入：`../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。核对train/val manifest哈希及test扩展检测配置。
- 旧checkpoint：`../multi-lane-main-parax-audit-paired-staged/output/emotic_protected_parax_val/protected_parax_audit_v2_seed0_val_20261010_01/{method}/compact_checkpoints/task{t}.pth`。
- 基线checkpoint：`../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed{s}/baseline/compact_checkpoints/task{t}.pth`。
- 入口：`bash scripts/emotic/run_fixed_baseline_routes.sh`，内部 `python -m multi_lane.track_a.fixed_baseline_routes --mode evaluate-old|fit ...`。
- 新batch：`fixed_baseline_cached_routes_20261011_01`；输出 `./output/emotic_fixed_baseline_routes/<batch>/`，日志 `./logs/emotic_fixed_baseline_routes/<batch>/`。
- 只使用启动时空闲的GPU0；两条队列（补test、缓存拟合）并行前先做真实GPU末任务推理显存smoke，显存不足则串行。其他GPU有其他会话任务，不占用、不停止。
- 预期产物：原test分数/指标；新三组validation逐任务分数、单路指标、train cache、拟合历史、梯度分量、gate均值/标准差/熵、残差比例、逐值冻结检查、FP32固定anchor漂移、compact checkpoint、summary及前三任务预检标记。

## 必须通过的恢复与冻结检查

1. checkpoint task/git/schema正确；原模型键集合、形状及加载后值完全匹配；使用restore_task，不能使用会复制旧Selector/Prompt的activate_task。
2. 关闭ParaX后，同源validation样本ID/标签顺序相同。完整AMP前向最大logit/概率差异单独记录；概率恢复容忍0.002，超出则停止，不能以此放宽严格identity。
3. 同一FP32缓存输入上的零残差、归一化后特征及初始logit必须严格相等。
4. 拟合前后原模型参数hash一致，无原模型梯度；旧专家、旧Router/投影及访问mask逐值不变。
5. 相邻原基线checkpoint旧任务参数必须逐值一致；固定validation anchor只用于审计，不参与拟合。记录各旧任务feature/logit/gate漂移。
6. 动态/静态两组每任务重置同一拟合随机种子，使用相同缓存及batch顺序；常量输入不意味着有效容量等同，只能说明名义结构与训练预算一致。

## 判读与停止

先比较各自固定基线的逐seed差值，再看动态相对静态。保留原计划标准：完整validation最终均值至少+0.20、至少两个seed同方向、平均任务性能不下降、旧路径稳定，才考虑扩大。若两者相近，不把收益归为图像路由；若都无稳定收益，停止当前后置ParaX扩展。既有test仅用于报告，不据其选新结构或超参数。

实现：新增`fixed_baseline_routes.py`、专用启动脚本与回归测试；只对protected bank添加可复核的常量Router输入选项，旧默认行为不变。未修改原联合训练入口与比较器。

## 2026-10-11 真实smoke通过，正式任务等待GPU0

- 本地309项unittest通过；远程308项全套与最新5项新入口回归通过。真实ViT task7 smoke完成动态图像/任务常量两组5缓存epoch，初始差异严格0、原模型与旧专家hash一致；旧四组中复用专家task2 checkpoint的test smoke完成。保留前两次配置恢复失败日志，正式训练未受影响。
- 正式启动前GPU0被其他进程占用（约5.4GB，util83%，NVML PID2585416/2585413），GPU1–7亦忙；安全检查拒绝启动，未停止/改变其他任务。启动脚本改为后台等待GPU0完全空闲，30秒资源检查、最长24h；不做训练实时分析。
- batch仍 `fixed_baseline_cached_routes_20261011_01`；等待状态在 `./logs/emotic_fixed_baseline_routes/<batch>/resource_wait.json`，空闲后自动并行旧四组seed0 test队列与三seed新缓存拟合队列。正式指标尚未产生，不能表述为已经训练中。
