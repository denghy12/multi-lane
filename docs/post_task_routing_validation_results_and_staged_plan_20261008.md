# Task Forward 后置路由：三任务 Validation 结果与下一步

## 本轮结论

把 ParaX-style 参数路由从 Frozen Forward 移到 Task Forward 的最终特征之后，并冻结旧任务的共享中心和 Router，已经让旧任务预测基本保持稳定。但它仍没有提高识别精度：首个任务就低于基线，最终三任务 mAP 低约 1 点。当前主要问题是初始学习和校正方式，不能继续把下降全部归因于增量遗忘。

本报告只比较同一批次、seed0、task0–2 的 Validation。它不是八任务结果，也没有评估 Test。两组均关闭 Image-token Adapter，所以这里的基线不等同于历史 bottleneck=32 的 Image-token Adapter 版本。

## 改了什么

两组使用同一个冻结 CLIP ViT-B/16，Full、Person、Face 在每个任务内共享 Patch Selectors、prefix prompts 和分类头。Task Forward 的三个视图最终 CLS 特征分别归一化后，按 Face 可靠性进行固定权重的特征融合，再送入共享分类头。

后置路由组在 Task Forward 的最后一层 CLS 特征经过 CLIP 的 LayerNorm 和 visual projection 后、特征归一化与三视图融合之前，增加 token-only ParaX-style 变换。这里没有修改 Frozen Forward 内部的 patch tokens。

共享 expert center 包含三组上下投影参数矩阵 `E_A/E_B`，rank=32；Router 根据输入特征生成 routing coefficients，将参数矩阵混合为样本对应的有效投影。三个视图共享这套参数矩阵，同一任务共享 Router；每个增量任务有自己的 Router。task0 更新共享中心与 task0 Router，task1 起共享中心冻结，各任务只更新自己的 Router，旧 Router 保持冻结。输出 projection 零初始化，残差系数固定为 0.001；没有 Projector 或 level embedding。

## 完成情况与指标

批次：`post_task_router_seed0_val_gpu0_parallel_20261008_01`。代码：`2fa5177`，分支：`exp/post-task-routed-parax`。GPU0 两组并行；每任务 30 epochs、batch64、Adam、主学习率 0.0125、ParaX 学习率 0.0004、cosine、seed0。

两组均完成 90 epochs、5,010 次更新，task0/task1/task2 分别 2,520/2,070/420 次更新，0 skipped；退出码均为 0，日志有完整完成标记。运行时间约 31.6–31.7 分钟。

| 指标（mAP 为百分点） | 原基线，无 Image-token Adapter | Task Forward 后置动态参数路由 | 后置路由减去基线 |
| --- | ---: | ---: | ---: |
| task0，已见 5 类 | 60.5225 | 59.1245 | -1.3981 |
| task1，已见 8 类 | 59.0402 | 57.6210 | -1.4192 |
| task2，已见 11 类，最终 mAP | 45.2395 | 44.2407 | -0.9987 |
| 三任务平均 mAP | 54.9341 | 53.6620 | -1.2720 |
| 协议 forgetting，越低越好 | 2.7760 | 2.4242 | -0.3518 |
| task2 cF1 | 36.8870 | 36.1998 | -0.6872 |
| task2 oF1 | 61.6314 | 61.1496 | -0.4818 |

| task2 单视图 mAP | 原基线 | 后置路由 | 差值 |
| --- | ---: | ---: | ---: |
| Full | 43.7058 | 42.3563 | -1.3495 |
| Person | 42.5883 | 42.2797 | -0.3086 |
| Face，含不可用占位图 | 33.9794 | 32.9977 | -0.9816 |
| 可靠 Face 子集 | 39.4689 | 38.0084 | -1.4605 |

可靠 Face 使用 `valid_face && !ambiguous_match && short_side>=24 && detection_score>=0.6`。可靠样本融合权重为 `[0.64,0.16,0.20]`，其余为 `[0.80,0.20,0]`。可靠 Face 子集与全体视图的评估样本不同，不能直接横向比较绝对 mAP。

task1 新引入三类的平均 AP：基线 61.9468，后置路由 59.8845；task2 新引入三类：基线 10.4613，后置路由 10.2552。后置路由既没有改善 task0，也没有改善后两次新类别学习。

## 问题定位

### 1. 本轮旧任务预测基本没有漂移

将 task0 的 2,285 个 Validation sample IDs 与 task2 输出逐一匹配，只比较相同图像的旧五类 logits，并在原训练环境复算 AP：基线平均绝对变化约 0.000392，后置路由约 0.000388。基线旧五类 mAP 为 60.5225→60.4982（-0.0243），后置路由为 59.1245→59.1240（-0.0005）。两组的旧任务预测基本保持住了；应区分这些极小浮点与排序变化和历史内部 ParaX 的大幅遗忘。

协议 forgetting 从 2.7760 降到 2.4242，不能直接解释成模型明显减少了遗忘。本项目每阶段评估“标签与已见类别相交”的样本；task0/task1/task2 分别评估 2,285/2,375/2,381 个样本。已见类别增加后，旧类 AP 的评估样本也在变化。固定样本审计比单独看这个 forgetting 数字更能说明旧预测是否被改写。

本轮没有保存旧任务原始特征与完整 checkpoint，因此不能补出同一批图像的 feature cosine drift。logit 审计支持旧预测稳定，不能替代完整的表征审计。

### 2. 精度损失已经在 task0 出现

task0 没有“学习后续任务导致遗忘”的条件，但后置路由融合 mAP 已下降 1.3981 点，可靠 Face 下降 2.6789 点。训练 BCE 略低并没有换来 Validation mAP 提升。

这支持“联合训练改变了原有任务学习方式、校正没有改善泛化”的解释；不能只归因于 Frozen Forward 特征被破坏，因为这一轮 Frozen Forward 没有被 ParaX 改写。具体是联合优化、专家表达还是泛化不足，需要下一轮拆分实验，当前结果不能单独判定。

### 3. 系数小，不代表实际特征改变量小

| 每任务最后一个训练 epoch，实际 residual/token ratio | Full | Person | Face |
| --- | ---: | ---: | ---: |
| task0 | 0.0481 | 0.0613 | 0.0489 |
| task1 | 0.0295 | 0.0675 | 0.0403 |
| task2 | 0.0314 | 0.0352 | 0.0396 |

固定 `0.001` 只乘在输出上，共享中心和投影仍可通过增大自身幅值抵消它。训练过程中实测改变量最终达到约 3%–7%；不能称它为“0.1% 的特征扰动”。这里是训练增强样本的 epoch 平均，不是 Validation 上的逐样本最大值。

### 4. 路由有差异，但没有有效专门化的证据

task0/task1/task2 最后训练 epoch 的三视图 gate 均值 L1 距离分别约 0.2293/0.3501/0.0634，说明输入确实产生不同 routing coefficients。但三路单视图指标全部退步。task0 expert1 平均权重仅约 2%–6%，几乎从不成为最大权重；三组参数矩阵没有均衡使用。这可以提示偏置，不能仅凭 top-expert 频率断言 soft routing 完全塌缩。

task1 起只能重组 task0 学到并冻结的变换，不能新增适合新类别的变换。task0 中心是否限制后续表达仍是待验证假设。下一轮固定随机中心，避免把“task0 学到的中心”和“共享参数本身”混在一起。

### 5. 当前不能把固定融合认定为唯一瓶颈

融合输出仍高于 Full 单路，task2 后置路由的融合增益约 1.8844 点。但路由后的 Full、Person、Face 本身都低于基线，融合无法自动恢复损失。动态融合继续推迟，先验证校正是否能稳定改善单路。

## 下一步：先保留基线学习，再训练受限后置校正

每个任务先按原基线训练 30 epochs，此阶段完全关闭路由。训练完后固定 Patch Selectors、prompts、分类头和 CLIP，用同一训练集的确定性视图缓存当前任务的最终归一化特征。缓存只含训练样本和当前类别标签。

在最终归一化的 Task Forward CLS 特征之后、固定三视图融合之前，应用共享参数矩阵和每任务 Router。共享 expert center 从随机初始化开始就固定，避免 task0 偏置与跨任务中心更新；每任务另有一个 rank32→rank32 小投影（1,056 参数），零初始化，训练完冻结。三个视图共享 center、当前任务小投影和 Router；只有 routing coefficients 随输入变化。

实际残差使用可微的范数边界：`delta_bounded = delta * radius / sqrt(radius² + ||delta||²)`，其中 `radius=0.02*||feature||`。这保证归一化前新增残差比例小于 2%，不会通过增大投影权重逃逸；初始输出严格等于输入。校正后再做一次特征归一化。

校正使用 5 epochs、FP32、Adam lr=0.0004、cosine、batch64、gradient clip=1.0。损失是当前类别 BCE、可靠视图辅助 BCE（0.1）与相对基线的 fused/view logit MSE（0.1）。只用训练集，固定训练预算，不用 Validation 选校正 epoch。

| 对照 | 训练方式 | 回答的问题 |
| --- | --- | --- |
| 原基线 | 30 epochs，无校正 | 重新核对配对一致性 |
| 基线后动态参数路由校正 | 基线 30 epochs + 固定中心、每任务投影及 Router 校正 5 epochs | 隔离基线后，小幅动态校正能否提高精度 |
| 基线后均匀参数混合校正 | 同上，routing coefficients 固定为 1/3 | 提升是否需要动态 Router |

每个任务额外保存校正前的 baseline scores、训练特征缓存、校正历史、基线/共享中心/旧路由参数 hash、逐视图分数和 compact checkpoint。校正期间 hash 必须完全不变；额外校正与缓存过程保存恢复 Python/NumPy/PyTorch RNG，防止改变下一任务的数据和参数初始化。

第一道门槛是校正前输出与独立基线对齐；若不对齐，先修实现。通过后才看校正是否改善 task0、新类和可靠 Face。晋级八任务需三任务 final mAP 至少 +0.10，average 不下降，固定样本旧预测稳定；若动态组不优于均匀组，不能宣称动态路由有效。若两组均无收益，停止继续增加这一后置低秩变换，回到保持特征不变的输出校准或融合研究。新实验增加了校正预算，其训练时间与更新数必须单独报告，不能声称与基线预算相同。

## 结果位置

本地完整结果：`output/emotic_track_a_post_task_router_val/post_task_router_seed0_val_gpu0_parallel_20261008_01/`。

日志：`logs/emotic_track_a_post_task_router_val/post_task_router_seed0_val_gpu0_parallel_20261008_01/`。

本轮结论仅来自一个 seed、三个任务；没有得到跨 seed 或八任务有效性的证据。
