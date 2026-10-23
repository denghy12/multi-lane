# 强Full ASL三视图执行记录

状态：实现与预检准备，正式实验尚未启动。

用户授权所有显卡按空闲容量并行。七组新配置＋R历史Full复现；seed0，EMOTIC，全TRAIN，8任务×30epoch/B64，预期13950更新/组；主Adam LR.0125，Adapter/Attention/位置LR.0004，WD0/cosine，aux0。BCE→Selectors/Prompt/head/Attention/位置，ASL9.8/0/.05→Image-token Adapter。新七组AMP1024/growth1e9；R保留f1ada22旧数值与全局采样并单列，不要求新S-ASL精确32.5365。

输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、既有face_manifest_train_val_test_v1_20260909。入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，日志logs/strong_full_asl_three_view/<run_id>，结果output/strong_full_asl_three_view/<run_id>。训练保存边界恢复/compact，之后完整固定人群validation/test及预声明task0/2/7关闭归因；无按test选模型或改配置。

来源远程primary153行SHA ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf；test-only5行SHA e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152。保护这些及其他既有工作树。旧f1ada22工作树clean，不改其源码，包装器通过Git受控新代码加载旧模块，保存不消耗RNG的任务边界及明确历史导出元数据。

## 实现与第一轮预检

432项回归通过，真实CLIP B64初始化7组公共参数/RNG/同视图输出逐值一致；冻结视觉不变。首次smoke因单Full误传Face参数被入口拦截（0更新），修正后smoke_seed0_20261023_02八组训练均complete/0跳步。最重2818.77MiB/保留3034，单Full ASL1357.87MiB。验证器补区分短跑updates和正式epoch预算；新调度按训练→逐组完整性核验→该组val/test及预声明归因的依赖流水执行，评估只用本批已释放卡，每卡最多2路，不与本批活动训练抢同卡。正式未启动。

2026-10-23预检补记：3868f61完整433项回归通过；新流水smoke03八组训练均0跳步，16份同checkpoint validation重放逐值一致，历史Full导出/单视图评估也通过。恢复probe从cd1489d task0接续至task1，model/requires_grad/optimizer-scheduler-scaler/RNG/sampler及两阶段val logits全部逐值一致。当前最后6路关闭归因短跑仍在执行，正式未启动。补充首批各参数组梯度范数只读日志，并实现仅新三视图协议单前向导出融合及各视图score，待数值等价核验；没有更改训练损失或模型公式。


2026-10-23预检全部完成补记：3868f61 smoke03完整30作业（8训练/16validation与重放/6关闭归因）全部exit0，0跳步，16份scores重放逐值一致；恢复probe所有状态与logits逐值一致。历史原始config进一步核对workers=2，严格R保留2/旧随机流；此前包装器smoke用0仅作接口验证，新七组仍固定workers0。5d56449增加单前向融合+视图导出及首batch参数组梯度只读日志，待最后回归/真实评估等价核验后锁定正式，尚未启动正式。

## 2026-10-23：全部预检通过，准备正式八组多卡运行

新协议7组＋旧Full严格复现R，正式全TRAIN/seed0/8task×30epoch/B64/13950updates每组/aux0/主LR.0125/模块LR4e-4/WD0/cosine。BCE→原任务及Attention/位置，ASL9.8/0/.05→Adapter；新7组scaler1024/growth1e9/workers0，R原f1ada22/全局随机流/workers2。R最终另记是否数值匹配32.5365/39.1445，不把预算verified冒充历史得分已复现。

最终434项回归通过，真实CLIP B64公共初始化/RNG/同视图输出逐值一致；完整smoke03共30作业全部成功、0跳步，16份重复预测逐值相同。最后19项真实检查全true：5个三视图模型×2边界单前向导出与原评估逐值相同，历史Full当前/旧代码2边界重放相同，新增梯度日志后恢复model/requires_grad/optimizer-scheduler-scaler/RNG/sampler全等，六参数组梯度范数已记录，R workers2短跑成功。无新超参搜索。

多卡调度每卡1训练，完成核验后释放卡接续每卡最多2路val/test，6路预声明validation关闭归因自动执行；GPU0–7按实际显存准入/不足等待，不改B64或其他作业。正式run_id拟formal_seed0_20261023_01，入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，输出output/strong_full_asl_three_view/formal_seed0_20261023_01、同名logs；本条为启动前锁定记录，不能作已启动证据。远程primary/test-only和历史代码树继续保持。
