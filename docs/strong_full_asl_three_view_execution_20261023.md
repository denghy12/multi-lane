# 强Full ASL三视图执行记录

状态：实现与预检准备，正式实验尚未启动。

用户授权所有显卡按空闲容量并行。七组新配置＋R历史Full复现；seed0，EMOTIC，全TRAIN，8任务×30epoch/B64，预期13950更新/组；主Adam LR.0125，Adapter/Attention/位置LR.0004，WD0/cosine，aux0。BCE→Selectors/Prompt/head/Attention/位置，ASL9.8/0/.05→Image-token Adapter。新七组AMP1024/growth1e9；R保留f1ada22旧数值与全局采样并单列，不要求新S-ASL精确32.5365。

输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、既有face_manifest_train_val_test_v1_20260909。入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，日志logs/strong_full_asl_three_view/<run_id>，结果output/strong_full_asl_three_view/<run_id>。训练保存边界恢复/compact，之后完整固定人群validation/test及预声明task0/2/7关闭归因；无按test选模型或改配置。

来源远程primary153行SHA ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf；test-only5行SHA e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152。保护这些及其他既有工作树。旧f1ada22工作树clean，不改其源码，包装器通过Git受控新代码加载旧模块，保存不消耗RNG的任务边界及明确历史导出元数据。

## 实现与第一轮预检

432项回归通过，真实CLIP B64初始化7组公共参数/RNG/同视图输出逐值一致；冻结视觉不变。首次smoke因单Full误传Face参数被入口拦截（0更新），修正后smoke_seed0_20261023_02八组训练均complete/0跳步。最重2818.77MiB/保留3034，单Full ASL1357.87MiB。验证器补区分短跑updates和正式epoch预算；新调度按训练→逐组完整性核验→该组val/test及预声明归因的依赖流水执行，评估只用本批已释放卡，每卡最多2路，不与本批活动训练抢同卡。正式未启动。
