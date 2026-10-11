# 冻结与复用实验：流程修复与正式重启

用户于2026-10-11明确授权执行修复、重审和原定正式实验。本会话只研究ParaX专家冻结/复用，保持原三视图固定特征融合，不加入其他会话的三路全量实验改动。

## 修复及验证

- 比较器使用runner真实产物字段`parameter_group_loss_routing`。缺字段明确失败，四组不一致仍拒绝。
- `--report-root`将重审派生报告保存到新目录，不修改旧预检的日志/失败标记或伪造完成标记。
- 控制器比较失败记录阶段、退出码、具体错误、命令和日志路径；下游传递具体失败原因。
- `REAUDIT_SMOKE`允许重审完整旧预检：核对四组来源commit一致，Git检查`multi_lane/`及原训练Shell与当前HEAD除比较器外完全一致，模型训练有变则拒绝复用预检。
- 预检重审记录训练来源commit和审核器commit，记录通过后才调度正式训练。`dispatch_ready.txt`只表示可调度，首次实际启动worker才写`started.txt`。
- 新增11项回归测试：真实runner完整配置schema、八任务双split比较与独立报告、缺字段/配置不一致/缺终点/缺分数/预算不符、epoch预算路径、比较具体失败、审核来源追踪、模型改动拒绝旧预检。
- fixture来自真实20261010预检的baseline完整config，仅替换三个机器输入路径；分数/audit小fixture是人工配对记录，不作为研究结果。
- 本地完整304项unittest通过；原Shell语法检查、diff检查通过。修正版比较器对已同步的真实四组八任务预检直接重审通过，不使用上轮内存替换方案。

## 正式配置

| 项目 | 锁定配置 |
|---|---|
| 数据/协议 | EMOTIC Track-A incremental task0–7，5+7×3=26类 |
| 方法 | 原共享基线；task0学习两位参数专家后冻结池；每任务仅用两个新专家；冻结旧专家并复用、学习两个新专家 |
| seed/训练预算 | seed0/1/2，12组；30epoch/task，240epoch/run，预期13,950有效updates/run |
| 优化 | Adam reset/task；原任务LR0.0125；ParaX LR0.0004；WD0；cosine/min0/warmup0 |
| 批量/精度 | train/eval batch64，workers2，OMP/MKL1；AMP scale1024/growth1e9 |
| 骨干 | 预训练CLIP ViT-B/16始终冻结 |
| ParaX位置 | Task Forward最终归一化CLS之后、原固定三视图特征融合之前；不是Frozen Forward内部 |
| 参数专家 | 同一共享池，16对低秩E_A/E_B矩阵；rank32；每任务最多两位新专家；任务Router hidden16、image-only、三视图共享 |
| 残差与冻结 | task-local32→32零初始化投影；fixed scale1；可微原始残差比例界0.02；封存旧专家/Router/投影/访问mask及原任务路径 |
| 融合 | 可靠Face权重[.64,.16,.20]；不可靠[.8,.2,0]；可靠条件valid且无匹配歧义、短边≥24、检测分≥.6 |
| 损失 | legacy_full_zero、joint_bce，融合BCE+.1可靠视图辅助BCE |
| 排除 | Image-token Adapter、旧patch Projector、level embedding、蒸馏、分阶段训练、动态融合 |
| 评估 | 同次训练固定epoch终点权重同时评估validation/test；推理汇集所有已见任务路径，不使用真实任务ID；test不用于选择 |
| GPU策略 | 只用空闲0–7，每GPU一组，最多8组同时；其余等待，绝不终止其他任务 |

## 路径与产物

所有相对路径以服务器新worktree为起点。

- 源码分支`fix/parax-full-comparison-relaunch-20261011`；提交后通过Git-only fetch/worktree同步。精确源码commit记录在control/manifest.json。
- 服务器独立worktree：`/mnt/haoyuan/workspace/multi-lane-main-parax-freezing-eight-task-relaunch`。
- 数据`../multi-lane-main/datasets/EMOTIC`。
- CLIP权重`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；实际权重SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。
- Face清单`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 重审源`../multi-lane-main-parax-freezing-eight-task-locked/output/emotic_protected_parax_full/protected_parax_freezing_8task_smoke_20261010_01`，训练commit eb049726bb053e3101e0d0259002a91fc054d64b；重审不继续此批训练、不改变它的失败状态。
- 新batch `protected_parax_freezing_8task_seed012_valtest_20261011_01`。
- 结果`./output/emotic_protected_parax_full/<batch>/seed<seed>/<method>/`。
- 日志`./logs/emotic_protected_parax_full/<batch>/seed<seed>/`；控制器日志在对应日志batch目录。
- 预检重审报告`./output/emotic_protected_parax_full/<batch>/control/prerequisite_audit/reports/`。
- 产物：compact checkpoints、val/test融合及单路分数、同模型残差开关、旧参数/identity/固定anchor审计、三seedJSON/CSV。

## 后台入口

```bash
DATA_ROOT=../multi-lane-main/datasets/EMOTIC \
CLIP_CHECKPOINT=../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
FACE_MANIFEST_ROOT=../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
OUTPUT_ROOT=./output/emotic_protected_parax_full \
LOG_ROOT=./logs/emotic_protected_parax_full \
REAUDIT_SMOKE=../multi-lane-main-parax-freezing-eight-task-locked/output/emotic_protected_parax_full/protected_parax_freezing_8task_smoke_20261010_01 \
BATCH_ID=protected_parax_freezing_8task_seed012_valtest_20261011_01 \
SEEDS=0,1,2 GPU_POOL=0,1,2,3,4,5,6,7 EPOCHS=30 UPDATES_PER_TASK=0 \
/opt/conda/envs/ddp/bin/python scripts/emotic/launch_protected_parax_full_evaluation.py
```

实际后台启动在Git-only服务器同步、完整单测、Shell语法和输入校验通过后执行。若所有GPU忙，记录为队列已启动但训练未启动；不把dispatch-ready当成已跑训练。原控制器无人值守等待空闲卡，核验队列/首次启动状态后停止监督。
