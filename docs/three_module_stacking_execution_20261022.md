# 三模块叠加消融执行记录

2026-10-22开始，2026-10-23继续。用户已要求实现并在GPU0自动并行/分轮执行完整8任务validation和test。当前状态（2026-10-23 14:28 CST）：**八组训练、16路完整validation/test、F111四路关闭归因全部完成，completion为complete、无failure；完整产物已同步本地并SHA核验**。

最终分析见`docs/three_module_stacking_complete_results_20261023.md`。formal_seed0_20261023_01已完整结束：八组各240epoch/12660更新/0跳步，16路完整validation/test及F111 task0/2/7四路关闭归因全部exit0，completion为complete、无failure。远程执行树clean、HEAD0fb85f1，原primary153行/test-only5行及SHA保持。598份非checkpoint产物同步本地并SHA全等；128份checkpoint仅记录SHA未下载。独立CPU复算652项逐类AP最大误差3.98e-13，八组task7 val存档逐值重放；本轮无新模型前向/训练。

按基线/A/B/C/AB/AC/BC/ABC，最终val为42.5650/42.4311/42.4323/42.6478/42.9530/41.8702/42.4916/42.3712；最终test为31.9391/31.8358/32.2804/32.1179/32.4286/31.8614/32.2555/32.1765。AB最佳最终val/test，较本轮基线+.3880/+.4896，阶段平均48.5646/35.7581（基线48.5026/35.4273）；C单独val平均48.8647最高，ABC test平均35.8087最高。不能称AB四指标全面最优，也不支持三模块普遍叠加。

ABC−AB的task6三类对26类mAP贡献val−.6332/test−.3875，其余23类分别+.0515/+.1354；两split都集中小样本task6。F111 task7关A/B/C/全关val为42.3373/42.2222/42.3755/42.2100（全开42.3712），关C未恢复AB42.9530或task6三类。主要差异留在共同训练的任务参数，不能断言即时位置bias过强或旧任务漂移。AB的Full两split变强，Person端点未同步增强，融合−Full只小增；不能宣称人物对应/互补已解决。

1000次原图配对bootstrap：AB−基线val95%[−.3395,+1.0274]、test[+.0793,+.8921]；AB−Attention的test区间跨0，ABC−AB与全部F111开关区间均跨0。仅固定seed0的样本不确定性、无多重校正，不代表训练seed稳定或机制已证实。不要用历史全TRAIN B0替代本轮90%fit匹配基线；历史test阶段平均36.2781仍高于本轮所有组。

以下“启动/等待”段落保留为历史执行记录，不能作为当前状态。下一步AB开关仅为建议，未新启动训练或GPU归因。

本地实验分支`exp/three-module-stacking-20261022`，从`1e28bfb`创建，保留此前未提交分析文档。远程拟使用独立工作树`/mnt/haoyuan/workspace/multi-lane-main-three-module-stacking-20261022`，不改原主树/test-only/已完成来源实验。

## 锁定配置

EMOTIC、seed0、八任务5/3/3/3/3/3/3/3类、30epoch/task、B64、原图90/10 fit/holdout，预期12660更新/组。主Adam LR0.0125、模块LR0.0004、WD0、余弦到0；joint BCE＋aux0.1，无ASL；AMP/TF32、scale1024/growth1e9、workers0。原三视图输入/可靠Face规则、固定归一化特征融合和共享分类头不变。

A=shared Image-token Adapter，索引1/b32/ReLU/scale0.1；B=task Selector Query读取Full摘要并写回Person，索引5/总宽64/4头/scale0.1；C=target原六维几何、12层共享60参数/任务。新增读取统一收缩路径；A不写回Frozen Forward；B源缓存包含当前A/C影响。八组F000/100/010/001/110/101/011/111，不据中途成绩删除格子。

输入沿用已确认资源，通过环境/命令参数传入：

- `DATA_ROOT=../multi-lane-main/datasets/EMOTIC`
- `CLIP_CHECKPOINT=../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- `FACE_MANIFEST_ROOT=../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`
- 初始化只读参考：`../multi-lane-main-target-summary-gpu0-resume-20261022/output/target_anchored_summarization/formal_gpu0_resume_seed0_20261022_01/baseline/train`；不复用其训练权重。

输出：`output/three_module_stacking/<run_id>/`；日志：`logs/three_module_stacking/<run_id>/`。入口`bash scripts/emotic/run_three_module_stacking_ablation.sh`；controller自动启动全部训练、训练完整性核验、16路完整val/test作业队列和F111 task0/2/7的四组validation关闭归因（全开重放正式val）。最后输出summary.csv、interactions.json及1000次原图bootstrap的validation最终差值。

GPU0初始24GiB空闲。拟训练/评估最多4进程、每进程Torch分配器上限18%、启动需5600MiB余量，最终并发值以真实最重配置smoke为准。队列容量不足自动等待、多轮启动；不降低batch，不终止外部进程。失败记录未运行作业及错误，不能标记完成或静默删实验。

## 保护与预检

服务器原主树状态153行，SHA256 `ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf`；test-only状态5行，SHA256 `e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152`。不修改这些工作树。

最初本地语法预检及diff检查通过；后续远程回归与真实短跑结果见下文。测试覆盖：八组零初始化等价、优化器与梯度、A+C偏置生效、C经Full缓存影响B、冻结图像流不变、旧任务冻结、compact/完整恢复、严格协议与交互公式。smoke仅validation，不读取test。

后续核验：初轮新增8项机制测试发现旧空间分支绕过跨视图缓存，已增加stacking独立三视图调度并修复测试中reporting-split缺省值。远程`d4a4238`完整429项回归通过；真实CLIP、B64、固定validation首批的八组初始化、共同原参数/RNG和输出逐值一致，冻结视觉不变，初始化检查峰值分配742.1MiB/保留1042MiB。另补两个自动队列测试：显存不足等待后多轮排满八组、失败时保留未运行清单，待最新回归。

## 2026-10-23中断后接续

本地/服务器独立工作树均锁定0fb85f1，431项完整回归通过，八组实际启动参数均通过严格profile检查。GPU0初始35MiB占用/24047MiB余量，未发现重复本批作业。已启动tmux stacking_smoke_seed0_20261023_01，短跑批次smoke_seed0_20261023_01，前三个单模块/基线与三个双模块均complete，F111在进行；前四组并发显存约13015MiB，0跳步，评估前完整快照已保存。

真实恢复验证另由tmux stacking_recovery_probe_20261023_01等待F111来源训练完成：新输出recovery_probe_F111/train，只从task0完整边界继续到task1，比较model/requires_grad/training_state/rng/sampler及两个任务的val预测；来源SHA前后核对。未访问test，正式尚未启动。

## 正式启动与后续自动执行

正式批次formal_seed0_20261023_01，tmux stacking_formal_seed0_20261023_01；两端exp/three-module-stacking-20261022、HEAD0fb85f1ed8c277c48ca4267d2dbe7ec1be8fcbf6/upstream一致，远程执行工作树clean。F000/F100/F010/F001四组GPU0并行，PID916442/916468/916494/916520均完成task0第1轮各76更新、0跳步，初始化底座SHA相同。F110/F101/F011及F111按队列自动接续；每组seed0、完整8任务×30epoch、B64、12660更新，统一原图90/10与aux.1 joint BCE，未访问新test或按指标筛组。当前只有启动健康状态，尚无正式最终mAP。

431项完整回归、真实CLIP B64八组初始化逐值一致、八组各2task×2updates短跑、16路validation/重放、4路F111关闭归因全部通过。真实task0边界恢复至task1后model/requires_grad/optimizer/scheduler/scaler/RNG/sampler及val存档逐值相同，来源SHA不变。最重F111短跑峰值分配2658.84MiB/保留2806MiB；GPU0最多4并发/单进程分配器18%/启动余量5600MiB，容量不足自动等待、多轮排队，不减batch、不干预外部任务。

训练结束后控制器自动核验八组预算/冻结/恢复，再在GPU0队列并行完整8阶段val/test共16作业，最后F111 task0/2/7做关闭A/B/C/全关validation，输出交互与原图bootstrap。正式入口日志logs/three_module_stacking/implementation_20261022_01/formal_driver_01.log；组日志logs/three_module_stacking/formal_seed0_20261023_01；结果output/three_module_stacking/formal_seed0_20261023_01。283份静态预检/启动产物已同步本地并SHA全等，36份checkpoint仅存SHA未下载。原服务器主树153行/test-only5行及状态SHA保持；不合并main，不修改已锁定远程代码。本条及执行记录启动状态仅本地未提交更新。

12:07补记：F010正式val最终42.4323、阶段平均48.3613，新增17份产物SHA核对，累计309份；task7归档逐值重放一致。其余正式评估继续自动排队，未新增GPU实验。
