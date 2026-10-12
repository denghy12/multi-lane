# 补齐导师报告：完整八任务 Test 执行方案

用户2026-10-12授权补齐缺项，第一轮优先八任务test，最终只认八任务test；随后授权使用全部空闲GPU。仍排除全量类别联合学习，不按test改超参数。

## 执行范围与顺序

1. 已有八任务权重优先推理：固定真实基线后拟合的dynamic/static，seed0/1/2，每组task0–7各8份checkpoint，48份权重不重新训练。
2. 原本只有前三任务的历史方法统一重新完成八任务：seed0基线、Task Forward后置Router、冻结随机中心动态校正、冻结随机中心均匀校正。原训练入口与学习规则保留，旧短程不续跑。当前统一恢复/归一化实现上的八任务复核不能伪称原短程checkpoint的直接延续。
3. 融合支线：seed0按原图分组留出20% TRAIN，80%训练一个八任务基线；缓存拟合任务固定BCE、动态图像BCE、动态排序。随后复用它补历史背景校准，更新预算仍与原池相同。先完成所有train校准，再只读导出test预测，应用已经封存的gate，禁止用test拟合。
4. 初版数值审计失败属于流程记录，不复活旧错误检查；正确冻结规则的四组八任务三seed已经完整测试，无需重复。Projector五组八任务test也已齐全。

旧三任务test不再额外补跑。它们保留为历史记录，不替代task7的26类结果。已有baseline test直接保留，不把新增方法test替换成baseline或另一训练策略的数据。

## 完整配置

- EMOTIC Track-A，增量协议5+7×3类；train学习，validation辅助报告，test固定逐任务权重评估。最终标准为task7的26类test mAP；同时保存八个终点，便于average/旧样本诊断，不根据中途test选模型。
- 模型：冻结CLIP ViT-B/16；三视图共享原Selector/Prompt/分类器；Image-token Adapter、旧patch Projector关闭，无level embedding/蒸馏。
- 原模型训练30epoch/task，train/eval batch64，workers2，Adam每任务重置，主lr0.0125（source0.05×64/256）、weight_decay0、cosine/min0/warmup0；legacy_full_zero/joint_bce、可靠单视图辅助0.1；AMP/TF32沿用原配置。
- 早期后置Router：Task Forward视觉投影之后/最终归一化之前，3对rank32参数专家、hidden16、zero_output、fixedscale0.001、task0中心学习后冻结，旧task Router封存，当前模块联合学习，lr0.0004。
- 随机中心分步校正：最终归一化CLS之后/融合之前；3对rank32随机矩阵从开始冻结；每任务Router/32→32零初始化投影，任务后冻结；动态与均匀对照；2%可微残差界、fixedscale1，train-only确定性特征缓存，5epoch、FP32、Adam0.0004/cosine/WD0/batch64/clip1，可靠view BCE0.1、baseline logit consistency0.1。
- 最新固定真实基线的直接test：2个活跃专家task0学习后冻结、rank32/hidden16、2%界、每task Router/小投影；动态图像/固定常量的选择从checkpoint恢复，不改变任何训练配置。
- 融合：30epoch、batch256、Adam0.001/cosine/WD0/clip1、FP32/TF32off，hidden8、末层零初始化；Person/Face偏移各≤0.05，MSE0.1、偏移惩罚1；旧gate冻结。补背景属于历史校准记忆机制对照，非无回放。
- 固定特征融合可靠Face[.64,.16,.20]，否则[.8,.2,0]；Face有效、匹配无歧义、短边≥24、score≥.6。
- 数据 `../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。核对CLIP SHA、train/val Face SHA与检测协议，test扩展不改变train/val。
- 既有源基线 `../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed{s}/baseline`。
- 既有拟合权重 `../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed{s}/{dynamic,static}/compact_checkpoints`。
- 分支 `codex/parax-eight-task-test-completion-20261012`，本地沿用managed工作树；远程新独立worktree `/mnt/haoyuan/workspace/multi-lane-main-parax-eight-task-test-completion`，Git-only同步，旧工作树/test-only不改。
- 正式batch `parax_eight_task_test_completion_20261012_01`，tmux `ml_parax_test8_20261012`；入口 `/opt/conda/envs/ddp/bin/python -u scripts/emotic/launch_eight_task_completion.py --batch parax_eight_task_test_completion_20261012_01`。
- 输出 `./output/emotic_eight_task_completion/<batch>/`，日志 `./logs/emotic_eight_task_completion/<batch>/`。保存源SHA、恢复审计、逐任务test及各视图分数、metric/summary、模型参数不变检查、每job退出码和控制器完成/失败原因。
- 资源：启动时选择7张真正空闲卡（used<512MiB/free≥18000MiB/util≤5%），三卡先做seed012 test，各卡dynamic→static；另四卡并行四组八任务训练。seed0 test结束后该卡运行80%训练源及融合/补背景，其他卡有任务不清理。若空闲不足，显式报错，不私自抢占。

## 验证与缺项处理

- 完整unit、shell/Python语法检查；实际runner schema回归，防止epochs_per_task等字段误读。
- 真实GPU预检batch单列：最新checkpoint task7/64样本恢复，另八任务各2updates/1epoch验证四组与融合全链路。该预检不是充分训练结果，不纳入导师性能表。
- Test无optimizer、无训练集缓存拟合；动态图像/常量Router属性必须恢复。每个拟合checkpoint的原参数须与对应真实基线逐值相同；原state/源checkpoint在推理前后不变。
- 融合拟合只读excluded TRAIN，test仅在全部校准结束后打开；旧gate访问范围与类别切片校验，Face可靠性mask保留。
- 未结束的八任务组标“运行中”，不填前三任务数值。失败明确记录，不把worker退出0当总控制器成功。
- 启动后不实时监督，不新增自动提醒；结束后用户召回再同步、复算并将八任务test写入导师报告。既有小幅负结果不阻止用户授权的固定配置test报告，但不能据此开展test调参。
