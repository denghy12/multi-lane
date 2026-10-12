# 固定真实基线后，冻结旧专家并学习新专家（2026-10-12）

用户授权开始本方向修改，随后明确首轮做seed0、task0–7的validation和test，不做前三任务正式validation。沿用当前三视图增量协议，不纳入另一个会话的全量学习，不修改正在补齐历史test的工作树。

## 本轮回答的问题

此前“旧专家复用＋新专家”与当前原任务模块联合训练，旧路径稳定但收益不稳定。当前冻结小池实验又没有新增专家。本轮固定已有逐任务真实基线，对比相同新增参数预算下的：

| 组别 | 当前任务可以读取什么 | 当前任务可以更新什么 |
| --- | --- | --- |
| 原真实基线 | 不加ParaX，使用现有checkpoint | 不训练 |
| 固定基线＋仅新专家 | 仅本任务新增的两个参数专家 | 两个新专家、当前Router和当前零初始化小投影 |
| 固定基线＋复用旧专家并学新专家 | 所有已见专家，包括本任务新增两个 | 同上；旧专家及旧任务Router/小投影/访问范围全部封存 |

本轮保留ParaX的A/B参数矩阵分别组合，不同时改成MoE输出组合。下一步“保留完整旧变换后组合输出”尚未实现/启动，应与本轮独立比较。

## 数据流和固定项

Full/Person/Face → 同一个冻结CLIP及对应已训练MULTI-LANE Task Forward → 最终归一化任务CLS → 当前任务的受保护ParaX残差 → 原固定三视图特征融合 → 原冻结分类头。

加载每个task对应的基线checkpoint，CLIP、Selector、Prompt、Task Forward、分类头全部不参与额外拟合。Image-token Adapter、旧patch Projector关闭；三视图共享专家池和当前任务Router。不插入Frozen Forward内部。

## 完整实验配置

- 数据集：EMOTIC；增量5+7×3类，seed0、8任务；每任务使用对应基线checkpoint，不用未来task7基线替代早期状态。
- 来源基线：`../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed0/baseline`；已有30epoch/task、13,950次原模型更新及完整val/test。此次不重训基线。
- 复用TRAIN缓存：`../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed0/train_cache/task{t}.pth`。来自确定性裁剪，AMP原前向转FP32缓存；原配置、源checkpoint SHA、完整当前任务TRAIN ID顺序/标签及三视图维度必须匹配，禁止用val/test缓存拟合。
- 参数专家rank32，16个预分配槽位，每任务激活两个新专家；Router hidden16，每任务一个Router和32→32小投影。两组全部初始参数相同，只改变访问mask；新专家/Router/投影初始化和实际可训练参数量逐任务校验一致。
- 小投影零初始化，A/B随机初始化；严格零残差起点，fixed scale1、2%可微范数界。不是所有因子同时置零。
- 拟合：5epoch/task、batch64、FP32、Adam lr0.0004/WD0，cosine每任务重置，gradient clip1。融合BCE按当前类数/26缩放，可靠单路BCE权重0.1；consistency0，无蒸馏/level条件/新增熵正则。每组八任务约2325次额外小模块更新，实际按缓存行数记录。
- 可靠Face固定特征融合[Full,Person,Face]=[.64,.16,.20]，否则[.8,.2,0]。可靠条件：有效、无匹配歧义、短边≥24、检测分≥.6。
- 数据 `../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。核对CLIP及Face检测协议/SHA。
- Validation：每任务保存原基线与两组分数、单路诊断。Test：所有八任务拟合并封存完成后才创建test dataset，恢复每任务checkpoint只读评价两组；基线test沿用同源既有评价。全部终点都评价，不根据val/test筛组、挑epoch或续训练；最终只按task7/26类Test报告效果。
- 训练入口 `python -u -m multi_lane.track_a.fixed_baseline_expert_reuse`；Shell入口 `scripts/emotic/run_fixed_baseline_expert_reuse.sh`。
- 正式batch `fixed_baseline_expanding_experts_seed0_val_test_20261012_01`；输出 `./output/emotic_fixed_baseline_expert_reuse/<batch>/`，日志 `./logs/emotic_fixed_baseline_expert_reuse/<batch>/`。
- 分支 `codex/fixed-baseline-new-expert-reuse-20261012`；服务器新独立worktree `../multi-lane-main-fixed-baseline-expert-reuse`，Git-only同步，不触碰主/test-only或历史补齐进程。
- GPU0目前仍有本会话的历史补齐工作；按此前持续授权共享显存，启动前至少8GiB空余，单个进程按任务顺序拟合两组、复用同一缓存。无需两份CLIP同时训练。其他GPU任务不干预。

```bash
GPU=0 BATCH=fixed_baseline_expanding_experts_seed0_val_test_20261012_01 \
  bash scripts/emotic/run_fixed_baseline_expert_reuse.sh
```

## 验收和运行方式

1. 完整Track-A回归；新增测试覆盖八任务预算配对、新专家更新、旧函数严格不变、TRAIN缓存来源、错误访问mask/封存状态及checkpoint恢复。
2. 真实ViT预检同样经过全部八任务，每任务只取64个TRAIN/val样本、拟合5个小缓存epoch，并测试只读checkpoint恢复；预检Test只取64人，不作为性能结果，不作前三任务正式实验。
3. 预检成功后启动上述完整八任务正式实验。禁止旧参数改变、访问未来专家、损坏源checkpoint/cache；出现故障保留失败日志，不覆盖原batch。
4. 初始化零残差必须严格identity；关闭ParaX恢复真实基线；源基线预测复核、基线原参数/旧路径hash、新专家实际更新、有限梯度、固定旧图像的logit/特征/gate漂移全部记录。
5. 启动后不实时监督；用户召回后同步并统一分析。单seed正向只作为观察，不能直接宣称稳定复用收益；“复用”必须同时与相同新增预算的“仅新专家”及真实基线比较。
