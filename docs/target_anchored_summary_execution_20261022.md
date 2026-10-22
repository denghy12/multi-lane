# 目标人物条件Patch Summarization：seed0完整8任务执行配置

用户2026-10-22要求按计划实现，并直接运行seed0、8task、validation和test；覆盖原三任务先导/val-only建议。保留原图90/10 TRAIN fit/holdout，不启用后续绑定损失。

三组：baseline（disabled）、absolute（固定整图参考位置、相同真实可见性门控）、target（真实人物位置）。Full Summarize softmax前加入v*tanh(w·phi)，原10 Selectors、3条Task Forward、MSA/Prefix-Tuning/D&R、末端特征融合及head保持。每任务60个跨12层共享空间参数，全8任务480；新任务独立零初始化，旧任务全部依赖冻结。

EMOTIC、seed0、5/3/3/3/3/3/3/3类；8×30ep，batch64，Adam按任务重置，WD0，cosine到0，主LR.0125/空间LR.0004，AMP初始scale1024/growth1e9、TF32；TRAIN 90/10按原图hash隔离，fit分别4854/3972/782/8932/3912/2312/576/1394，每epoch76/63/13/140/62/37/9/22步，各组12660更新。workers/eval-workers均0，避免共享内存预取超额。融合BCE+0.1×原可用三视图辅助BCE，legacy_full_zero归一化；无绑定/蒸馏/额外损失。

CLIP ViT-B/16；Full legacy crop .05–1，Person margin.15/letterbox，Face原manifest可靠性；所有MoE、ParaX、image-token adapter、旧selector conditioner、cross-view residual、joint summary关闭。
数据相对路径../multi-lane-main/datasets/EMOTIC；权重../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。

入口bash scripts/emotic/run_target_anchored_summary.sh；结果output/target_anchored_summarization/<run>/<arm>，日志logs/target_anchored_summarization/<run>。先保存每边界完整恢复状态再报告；compact checkpoint、TRAIN holdout、配对初始化/数据/sampler/RNG及旧参数SHA、完整val2397/test5368的逐任务预测与分支诊断均保存。

资源：GPU0启动前仅余约0.7GB，其余卡已有外部作业但约7GB余量。先测真实峰值；训练拟GPU4/5/6每卡1进程；评估拟GPU1–6每卡1进程，三组val/test并行。每次启动须重新检查显存余量，足够才用；不停止/修改外部作业。若资源下降则等待，不冒险并发。

先CPU完整单元测试及CUDA机制检查；真实2task×2updates smoke，核验默认/零初始化与旧预测、几何元数据不改变像素/RNG、真实梯度、旧参数冻结、参数覆盖和恢复重放。通过后才启动正式。完整主运行不使用test选择参数。额外只读target val开启/关闭/同图框交换归因在主训练/评估后运行，用固定几何规则，保留本人可见性、Person/Face及标签；不重新训练或覆盖checkpoint。

本分支基于e81f2f2，原工作树与test-only保留，代码只经Git同步。当前处于实现阶段，正式尚未启动。具体提交、短跑、资源实测和正式状态将追加到本文及三个上下文。

## 首轮验证

9f9f6de新增10项机制测试及远程413项全量回归通过。控制器和真实CLIP/历史重放检查已备好：每个子进程Torch分配器上限16%，启动需5600MiB余量，防止本轮无界占用显存；仍不能阻止外部作业之后自行增长。smoke为三组各2任务×2更新，训练后完整val与原reporting cohort恢复重放，并做前2batch框交换核验，不访问test。正式尚未启动。

真实CLIP预检cad0b96通过：零参数三组逐值等价，旧task0/2/7正常预测逐值重放，几何交换仅改Full；源checkpoint SHA未变。峰值分配760.50MiB、保留1044MiB。报告output/target_anchored_summarization/implementation_20261022/real_initialization_01.json。

smoke_01训练三组均exit0/0跳步，验证器因weights_only=True拒绝完整训练恢复元数据而失败（非模型训练失败）。修复可信快照读取并补回归；无效框覆盖统计归零但不改预测公式。失败产物保留，使用新run smoke_seed0_20261022_02重跑，正式未启动。

## 正式启动门槛已通过

最终415项全测通过。smoke_seed0_20261022_02全batch exit0，三组各4成功更新/0跳步，六评估exit0，恢复重放逐值相同；空间关/框交换固定Person/Face，可交换33人中task0/1的Full预测31/33、33/33真实改变。smoke01与02六份方法状态逐值一致。峰值分配2575.41/2576.23/2576.23MiB，保留约2710–2714MiB。准备formal_seed0_20261022_01，源码随后锁定提交。
本次用户完整8任务要求覆盖原三任务先导的性能晋级门槛：不根据中间val/test终止某臂或修改超参，仅遇正确性/资源故障停止并保留产物。训练完成后先跨组三方审计，再三组完整val/test六进程评估，最后target task0/2/7完整val做空间关闭与同图框交换归因。
