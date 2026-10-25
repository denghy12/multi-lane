# 共同训练如何改变历史b32强底座：只读归因方案

2026-10-25，用户授权执行检查，并根据结果分析下一阶段适配。当前不训练新模型。

## 为什么要检查

前轮四模型的Test关闭新模块仍低于B0。新模块即时注入并不是全部损失来源；需定位原Adapter、Selectors、Prompt与分类头如何共同变化。仅按参数距离挑“变化最大”的部件，或者换一个部件变差就归咎于它，都不可靠：表征和分类头可能共同适配。

MULTI-LANE的activate_task还会复制上个任务的Selector/Prompt初始化新任务，需区分继承改变与新任务训练改变。B0和各变体的全局初始状态相同，不代表每一阶段训练起点相同。

## 预声明的诊断

1. CPU只读比较全部8阶段，四变体相对同阶段B0。按A=Image-token Adapter、S=Selectors、P=Prompt、H=分类头，分别统计当前任务及累计已见坐标的L2、相对L2和cosine。检查下一任务初始化的复制来源，以及未来Adapter/head坐标是否相同。原始权重距离不能等同于性能损害。
2. 固定TRAIN原图分组样本，task0/2/7各两批、人数上限64；四变体及16种A/S/P/H来源组合使用相同RGB和Face可靠性。用evaluation transform锁定像素、确定性FP32记录三视图/融合特征cosine、相对变化、logits和BCE；可靠Face才计Face指标。TRAIN探针的BCE仅是诊断损失，不冒充历史训练原始目标或正式mAP，不用于选择Test配置。
3. 最终task7完整validation/test，固定七种状态：none（共训模型、关闭新模块）；A/S/P/H分别回到B0；ASP（全部表征回到B0，保留共训head）；ASPH（全部回B0）。4模型×2split×7=56结果。none和ASPH必须逐预测匹配已有OFF和独立B0，才运行其余五种恢复。H、ASP、none、ASPH形成表征/分类头2×2控制，防止把配对失配误解为单一坏部件。

每个部件包含截至该阶段全部已见任务的对应坐标和bias；使用同阶段B0，不访问未来训练任务参数。为保持完整状态装载，未激活坐标也来自相同阶段，但不参与推理。新Attention/位置模块全部关闭，Image-token Adapter保持开启，Frozen Forward不写回；所有替换在内存发生并自动复原，不保存混合模型checkpoint。

## 资源、入口、数据、产物

- 来源模型：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01`，源码ab15571。
- 已完成Test开关来源：`../multi-lane-main-historical-b32-test-attribution-20261025/output/historical_b32_test_attribution/seed1_tasks027_20261025_01`，评估源码54af61b。
- 数据：`../multi-lane-main/datasets/EMOTIC`；CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- seed1、旧samples_intersect_seen_classes、B64/workers2/OMP-MKL72/AMP-TF32；TRAIN probe单独deterministic FP32/workers0，无optimizer/learning rate/epoch。CLIP权重与原实验相同。
- 仅GPU0最多4作业，显存余量6000MiB、SHM余量1024MiB、12秒启动间隔。4 probe、8 anchor、8组中间恢复作业，共20 GPU作业，另一个CPU审计。56结果均为task7，不报告8阶段平均。
- 入口`python -m multi_lane.track_a.run_historical_b32_backbone_attribution`。
- 输出`output/historical_b32_backbone_attribution/seed1_20261025_01`，日志对应`logs/`。

## 可靠性与解释边界

先5项针对性测试与完整回归。逐组校验非恢复参数不变、恢复发生且退出后完整还原，冻结骨干未变化，源40份checkpoint前后SHA一致。混合checkpoint只能回答有条件的替换效果，不是独立训练的可部署方法，不能按Test最优混合状态宣称新方法成立。Test已获用户授权用于探索，仍标记参与选择。

结果优先回答：原表征是否与分类头错配？Adapter/Selectors/Prompt各自在当前共同适配中有什么条件作用？上个任务的初始化改变是否传播？据此再规划保护B0、局部解冻、监督适配或初始化隔离的训练控制，不提前选择方案。

独立诊断分支`codex/historical-b32-backbone-attribution-20261025`。当前本地实现/语法检查完成，待远程验收与运行。

## 启动验收

诊断源码4289fc3f51186da5e5886dc300970bf8bcd2b09c已部署独立树multi-lane-main-historical-b32-backbone-attribution-20261025；新增5项检查及478项完整回归通过。tmux historical_b32_backbone_attribution_seed1_20261025已启动，执行seed1_20261025_01。启动时新分支两端HEAD/upstream一致且clean；原模型ab15571/旧评估54af61b保持clean，主树153行/test-only5行及状态SHA不变。当前源码锁定不再改动，执行记录仅本地文档更新。

## 根据中间Val证据追加的兼容性检查

两组Attention的H恢复提高Val，位置两组H恢复反降。补充仅两组Attention、task7、Val/Test：新Attention开启，none/H/ASPH三状态，共12份评估。none重放原ON为数值门槛；H检验B0分类头能否保留新分支贡献；ASPH检验共训Attention能否直接接到完整B0。它们是内存混合模型，不作为新训练方法。规则使用Val H恢复改善，但此前Test已被查看，继续标记探索。

独立补充分支codex/historical-b32-head-transfer-20261025，入口multi_lane.track_a.historical_b32_head_transfer，输出output/historical_b32_head_transfer/seed1_20261025_01，日志对应logs。新独立worktree等待主诊断完成才启动，GPU0总并行保持最多4，原4289fc3源码不变。仍B64/workers2/OMP-MKL72/AMP-TF32、相同原数据/CLIP/Face、无optimizer无训练。新增GPU作业4个、12结果；加原诊断共68个task7完整评估。补充真实原ON重放为实际数值验收，原恢复工具5单测/478完整回归已有通过记录。

补充源码131cca7c2ef47a927cc45a71950ddbd4b635b19d已Git-only部署到独立树multi-lane-main-historical-b32-head-transfer-20261025，10项针对性回归及导入通过。tmux historical_b32_head_transfer_seed1_20261025等待主诊断成功后自动运行，原4289fc3诊断源码保持不动。当前没有新训练。

## 完成状态

主20作业/56结果和补充4作业/12结果全部成功。294文件已同步并逐SHA验证，源40checkpoint、受保护工作树及四个执行源码树保持。最佳受限Attention混合状态为共训ASP+B0 H+M，Val43.0741/Test32.8864；关闭M仍43.0278/32.8651，不能把全部恢复归为模块。该状态仅task7探索性混合，非新训练方法，无8阶段平均，改进对B0的bootstrap区间跨0。两种固定Face可靠性切换也未形成稳定收益。详见结果报告及head_anchored_adaptation计划；新的五组训练控制尚未启动。
