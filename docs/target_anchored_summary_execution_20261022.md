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

## 正式运行状态

formal_seed0_20261022_01已启动，源码锁定b0d826342a55165f805ff25eec846df5688296ea；两端同分支codex/target-anchored-summary-20261022/upstream一致，服务器启动前clean。tmux为spatial_summary_formal_seed0_20261022；训练baseline/absolute/target位于GPU4/5/6，PID776820/776823/776826，均进入task0且初始底座SHA相同。启动时GPU4/5/6余量7201/7221/6641MiB，均超过5600MiB门槛。显存分配器上限16%，每卡一个本轮作业。

总入口日志logs/target_anchored_summarization/implementation_20261022/formal_driver.log；训练日志logs/target_anchored_summarization/formal_seed0_20261022_01/<arm>/train.log；输出output/target_anchored_summarization/formal_seed0_20261022_01。训练完成后自动审计、并行三组validation/test，再做target task0/2/7位置归因；暂无最终结果。预检与短跑155份静态产物已同步本地并SHA全部一致，checkpoint仅保留服务器。本地启动记录不修改运行中的远程代码。

启动健康核验（2026-10-22 12:42 CST）：三组均完成task0第1个epoch，各76次成功更新、0跳步、无异常栈/OOM；三个训练进程存活。首轮loss仅用于数值健康检查，不作性能判断。正式launch.json和startup_health_01.json已同步本地。

## 2026-10-22 13:56 CST：核实实验尚未结束，完成前三任务只读阶段分析

用户要求查看结束结果；实查formal_seed0_20261022_01的三组进程仍存活，task0/1/2各90epoch、4560成功更新/0跳步完成，已进入task3第2轮；完成标记、三组正式完整val/test及位置归因尚未生成。源码仍锁定b0d8263/远程clean，没有重启、调参或额外GPU评估。

阶段reporting validation（人数2285/2375/2381，类别5/8/11）：baseline58.1525/57.8075/44.4069，absolute59.2844/57.9388/44.3322，target58.4303/58.3855/44.9024。目标相对基线三阶段平均+0.4505，但相对整图控制仅+0.0543；500次原图配对bootstrap各阶段区间均跨0。TRAIN留出target−baseline为+.4720/−.3055/−1.1915，最后仅79人。不能把这些当固定2397人的完整正式val，更没有最终test数值。

空间参数非零/梯度有限，训练期偏置与attention KL较小、无饱和；这些支持分支在学习，不证明正确人物绑定或最终推理净收益。9份compact CPU核验旧Selector/Prompt/head行/空间参数逐值冻结且SHA不变；同人旧类AP近持平，累计阶段mAP下降不能直接解读为遗忘。

27份已完成预测/划分源文件同步并SHA一致，18个NPZ逐类AP重算差0；无新模型前向/训练/梯度/test访问。阶段报告docs/target_anchored_summary_partial_results_20261022.md，ignored快照/脚本/JSON在output/target_anchored_summarization/analysis_20261022_partial_01。原主树153/test-only5行及状态SHA不变。三个上下文与执行记录同步更新；本轮仅文档和只读分析，不重跑模型测试、不提交推送、不改运行中的远程代码。

下一步仍是原控制器完成全部8任务、六路完整val/test及target task0/2/7空间关闭/同图换框，再同步最终产物判断人物对应及互补性。当前不根据中间指标停止某组、修改bias/LR或启用绑定损失。
