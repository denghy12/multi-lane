# 自训练四组：本轮未完成的原因与恢复方案

## 2026-10-27：数值审计修复并继承自身进度，正式Val/Test仍待完成

运行源码fb5f6cc745ae6c103990876e7bc0e8370a1d155f，fix/self-trained-audit-recovery-20261027，本地/服务器HEAD/upstream一致；远程新独立worktree multi-lane-main-self-trained-audit-recovery-20261027 clean。11项专项与518项完整测试（1跳过）通过。四组真实已完整训练task0及task1warmup的A/A/B/A审计全部在数学SDP下四视图重复/更新后旧路径/回放差0，旧参数SHA一致；smoke_seed1_20261027_04全部成功。

已恢复运行output/self_trained_view_transfer/resumed_seed1_20261027_03，tmux self_trained_recovery_seed1_20261027，GPU0并行4组PID BASE2880903/NO2881441/MAIN2882175/MAIN_ADAPTER2882919。每组6份完整本组自身状态导入，模型Tensor SHA与CPU/CUDA RNG不变；超参数相同、核心训练代码逐字节及训练调用/初始化AST相同，未从其他模型拼装分路/head。task0及task1前30epoch跳过不重训，后三组task1遗失5epoch从自身warmup重放；NoD重放前2epoch日志值与原日志一致（不声称找回丢失权重）。现在四组均跨过task1原故障点：BASE/NO已保存task2并到task3 epoch5/2，MAIN在task2第30epoch，MAIN_ADAPTER第29，均0跳步/无Traceback。task1实际数学SDP保护审计四组所有差值0。全8阶段Val/Test与ON/OFF仍将在完整训练后自动执行，当前尚无正式mAP。

103份原失败日志/进度/完整回归/真实checkpoint审计/smoke产物已回传output/self_trained_view_transfer/failure_snapshot_20261027_01并逐SHA核验；归档03e4bce424550644d31e4f84cd42dd10d2871ed8586b3a6335d10c1aa37ff534，24份原学生checkpoint SHA未变、pth保留服务器。主树/test-only原HEAD/status SHA保持。运行源码固定fb5f6cc，不在运行中同步本地执行记录。不能用原实验进程结束或训练loss判断三视图方法精度；现只确认工程审计原因并恢复，待完整Val/Test。


2026-10-27用户要求分析完整validation/test。检查发现正式`formal_seed1_20261027_02`没有完成：四组均在task1末的旧任务预测审计处退出，没有进入task2–7，也没有启动正式Val/Test。因此目前不能提供本轮8阶段mAP或判断学习方法有效/无效。下述故障属于程序审计，并非mAP评估失败。

## 1. 实际进度

原运行源码6613c5610deccda09102f606b02fe6791485823c，远程worktree`multi-lane-main-self-trained-view-transfer-20261027`。

| 组 | task0 | task1硬监督30epoch | task1额外5epoch | task2–7 | 正式Val/Test |
|---|---|---|---|---|---|
| 原三视图自训练 | 完整保存 | 完整恢复状态已保存 | 无此阶段 | 未运行 | 未启动 |
| 无功能约束 | 完整保存 | 完整恢复状态已保存 | 日志完成，审计前未存权重 | 未运行 | 未启动 |
| 仅主参数约束 | 完整保存 | 完整恢复状态已保存 | 同上 | 未运行 | 未启动 |
| 主参数＋Adapter约束 | 完整保存 | 完整恢复状态已保存 | 同上 | 未运行 | 未启动 |

task1硬监督每epoch69次更新，30epoch正常完成；后三组额外5epoch也正常完成，所示日志均0跳步。退出点全部是`compare_probes`的`uncontrolled_probe_noise`。这项判断比较的是**没有任何参数更新的两次前向**，因此它越界本身不能证明旧任务遗忘或教师梯度破坏了Full。

## 2. 故障证据

旧审计已禁用TF32，但没有禁用优化SDP注意力计算核。两更新smoke的重复差较小，而完整训练的checkpoint上超过固定绝对1e-5阈值：

| 组 | 未更新参数的重复前向最大差 |
|---|---:|
| 原三视图自训练 | 1.3232e-5 |
| 无约束 | 1.2785e-5 |
| 仅主参数约束 | 1.3709e-5 |
| 主参数＋Adapter约束 | 1.1444e-5 |

在同一份完整训练后的SELF_BASE task0、自己的task1前30epoch checkpoint、同一固定TRAIN32人物上，执行A→A→B→A：

- 默认优化SDP：未更新A的重复差最大约1.3351e-5；旧任务A/B差最大约1.6212e-5，二者同量级。
- 仅审计改用标准数学SDP（禁Flash和memory-efficient、FP32、关闭TF32）：四个输出（Full/Person/Face/融合）的重复、更新后旧路径、回放差均为0。
- 对应旧任务参数的SHA完全相同。

这给出了比扩大容差更直接的修复：**为功能保持审计使用可重放的计算路径**。训练和正式评估继续沿用原AMP/TF32/SDP设置；审计上下文退出后恢复全部后端设置。

随后已逐组用四组真实已训练checkpoint完成数学SDP下的A/A/B/A检查，四组全部精确重放，证据已同步到本地快照。不能把这次故障写成Face或蒸馏导致退步。

## 3. 从MULTI-LANE数据流解释检查哪里出了问题

MULTI-LANE以Frozen Forward提供图像证据，旧Task Forward的Selectors、Prompts、Adapter及读出封存后，旧任务函数应保持。原审计试图用相同TRAIN输入比较旧类别logits，这是合理的检查对象。

问题在于，检查把GPU计算核自身的重复噪声也当成任务函数变化。原三视图参照组没有新功能约束，却触发同一错误；而且错误先发生在“不更新模型”的阴性控制中。因此当前证据定位的是审计不稳定，尚未触及“教师是否能帮助学生保住Full、融合是否增加互补”的研究问题。

训练损失、MSE下降或0跳步只能说明优化正常执行；完整Val/Test尚未启动，不能从这些训练量推断泛化收益，也不能用旧32.6800/32.8864/R5的数值充当本轮结果。

## 4. 恢复范围与完整配置

修复分支`fix/self-trained-audit-recovery-20261027`，新独立服务器worktree`multi-lane-main-self-trained-audit-recovery-20261027`，Git-only部署；旧run与所有checkpoint只读保留。

训练目标/架构/数据完全不改：seed1、EMOTIC、GPU0、8任务、原三视图b32 Image-token Adapter-ASL、主LR.0125/Adapter.0004训练30epoch，自己学得分类头冻结后LR.0004训练5epoch；Adam/reset/cosine/WD0、aux.1、beta ratio.25/max1、相同增强、72线程、B64、workers2/prefetch1、正式evalworkers0、历史seen-positive人群。A/B教师仍只提供训练目标，推理仅学生。

所有输入数据/预训练/教师相对路径沿用`self_trained_view_transfer_execution_20261027.md`；新增恢复源：

`../multi-lane-main-self-trained-view-transfer-20261027/output/self_trained_view_transfer/formal_seed1_20261027_02`。

队列新增`--resume-root`，每组只恢复**本轮同一学生自身的完整checkpoint**及RNG/optimizer/scheduler/scaler，不导入其他模型的层、分路或head。逐项检查超参数相同、核心训练文件字节相同、模型初始化与train_task调用AST相同，并保存原权重训练Git、原文件SHA和新打包Git。

- task0直接沿用本组完整恢复边界，不重训。
- task1直接恢复本组30epoch结束的边界，不重训这30epoch。
- 后三组task1已运行但未保存的5epoch必须从自身warmup边界重放；不能声称从日志找回最终权重或断言与丢失状态位级一致。
- task2–7按原预定义流程继续，最后自动8阶段Val/Test及Attention ON/OFF。

新代码在训练完成后、保护审计前额外保存完整`trained_recovery`及compact，后续审计或报告故障也不再丢失已完成优化状态；恢复该状态时跳过已完成训练，使用已存的前向参考继续审计。

新输出`output/self_trained_view_transfer/resumed_seed1_20261027_03`，日志同`logs/`前缀。先新增专项和完整测试、四组真实checkpoint只读审计、两任务两更新smoke，再从上述进度接续。尚未产生完整恢复后的Val/Test。

## 5. 这次能够回答和不能回答的事

可以确认：原四组没有跑完，原因是训练保护审计误停；保存了足够状态避免从task0重跑；旧任务冻结参数与非确定GPU计算误差必须分别核查。

不能确认：功能约束是否优于无约束，是否保住当前任务Full、是否提高两项Test mAP，或单模型能否吸收双模型均值收益。必须等完整模型实际评估。恢复工作不改变学习方法，也不构成新方法实验。

## 6. 已有训练日志的有限信息

只看保存完整的task0：四组前30epoch均2520次更新、0跳步，最后训练目标约.647879。后三组额外5epoch各420次更新、0跳步。两个约束组task0的平均teacher MSE首末epoch分别为.4639→.5839、.4369→.5098，没有给出越来越接近教师的平均趋势；但每epoch增强和batch不同，且缺少同输入下NoD教师距离控制，不能据此判定蒸馏或mAP失败。不同目标组的总loss包含不同功能项，也不能直接横比数值高低。

从结构看，本轮D只约束融合logits，没有逐项要求Full不变。共享的Adapter/Selectors/Prompts可能在Full、Person和Face之间产生补偿，因此旧任务参数保持与当前Full分路能力保持是两件事。固定首batch的梯度范数比例也不保证梯度同向或之后的教师距离持续下降。这些是完整评估需要检验的限制，不能由此次审计误停来证明。恢复后保持原配置，先取得同预算无约束/主参数约束/主参数＋Adapter约束的完整Val/Test，再决定是否有研究收益。
