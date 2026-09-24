# ParaX 当前问题定位与下一步实验方案

日期：2026-09-24

## 当前结论

ParaX 首轮没有改善三视图共享编码器。更准确的结论是：当前实现把一个输入条件化残差分支放进了共享 CLIP 图像 residual stream，并让同一份 ParaX 参数在 8 个增量 task 中持续更新；这会同时改变旧 task 的图像输入和当前 task 的 Selector/Prompt 优化轨迹。现有结果不能说明“动态参数路由本身无效”，但已经说明当前训练耦合方式不稳定。

首轮 validation 的证据如下：B0 为 `42.9847/50.0234`（final/average mAP），P-10 为 `37.5118/45.6598`，P-8:10 为 `31.3970/42.0353`，P-8:10-level 为 `33.0375/43.7033`，Static-control 为 `33.9727/45.4505`；forgetting 分别为 `1.0785/6.5905/12.0309/11.9386/10.9068`。所有 ParaX 结构均低于 B0，连续插入最差，Static-control 也退化。

第二轮把 P-10 残差压到约 `0.09` 后，P10-small、P10-router、P10-experts 的 test final mAP 为 `31.2761/31.9207/31.7548`，冻结三视图锚点为 `32.5365`。P10-router forgetting 最低（`4.9013`），但仍高于锚点（`4.7308`）；因此小残差改善了稳定性，却没有产生有效表征收益。

## 根因按证据强度排序

### 1. B0 与 ParaX 不是严格 paired 初始化

`MultiLaneModel.__init__()` 在 Selector、Prompt 和 classifier 之前创建 ParaX bank。ParaX 的 expert、router 和 projection 初始化没有放入 `torch.random.fork_rng()`，而原有 Image-token Adapter、Selector conditioner 和 view fusion 已经使用了 RNG 隔离。因此同一个 seed 下，B0 和 P10 的 Selector/Prompt/head 初值不同，模型构造结束后的全局 RNG 状态也不同，后续 DataLoader shuffle 和随机增强可能不同。

这会污染退化幅度的估计，必须先修复并用同一初始状态重做最小 paired validation。它不能解释 Static-control、P10 单视图全部下降的方向性现象，但在修复前不应把具体 mAP 差值归因给 ParaX。

### 2. `small` 不是 identity，且 ratio cap 会改变梯度

当前 `small` 只把输出 scale 设为 `0.001`，expert A/B 仍是随机非零参数。因此初始输出不是严格等于 CLIP 原输出。`residual_ratio_cap` 在 forward 中对输出做硬裁剪，达到 cap 后梯度也会被缩放；它不是单纯的监控项。首轮 stability test 中 P10-small 的 ratio 接近 cap，说明 cap 已经参与了优化。

严格 identity 应采用一种可学习的零输出初始化：要么保留随机 residual path、只把可学习 output gate 设为 0；要么把最终 projection 设为 0、把 output gate 设为 1。不能把 gate、projection、expert B 同时置零，否则残差为零且相关参数没有梯度。稳定性诊断阶段不应同时使用随机残差、硬 cap 和 level embedding。

### 3. 共享 ParaX 参数跨 task 持续漂移

Selector、Prompt 和普通 Adapter 在 `activate_task()` 时会复制/切换 task-local 参数；ParaX bank 没有 task-local center，`activate_task()` 只重新设置 trainability。同一份 expert center、router 和 output scale 在 task0 到 task7 中连续更新，旧 task 再次前向时得到的 image tokens 已经变化。

这正好解释了为什么冻结 CLIP 仍然会遗忘：冻结的是 CLIP 参数，不是送进 Selector 和 head 的图像表征。第一轮 P-10 的 forgetting 从 `1.0785` 增到 `6.5905`；即便第二轮把单次 residual ratio 限制到约 `0.09`，最终 task 仍低于锚点。需要把“当前 task 适应”和“旧 task 表征不漂移”作为独立实验因素。

### 4. 训练目标没有推动 level specialization

ParaX 目前只从最终 fixed-three-view 融合损失（以及较小的 view auxiliary loss）得到训练信号。没有目标要求同一原图的 Full、Person、Face 使用不同专家，也没有目标要求某一路的单视图性能提升。因而 gate 差异不等于有效的 level-specific feature。

P10-router 的 gate 有一定 view 差异，但 expert center 被冻结且 residual ratio 约 `5e-6`，所以基本没有有效特征变换；P10-experts 能产生约 `0.09` residual，却只有 `0.0132` 的 view gate L1，三个 level 基本使用相同路由。这说明“路由器”和“专家变换”必须联动，且需要明确的 level 条件或辅助目标；但在稳定性未验证前不应直接加 level embedding。

### 5. 当前残差分支与 AMP/优化尺度存在额外混淆

ParaX 低秩乘法把 expert 参数转换到 token dtype 后执行，small scale 下有效残差可能非常小；同时 ParaX 与普通模型参数共用训练函数，使用 `adapter_learning_rate` 这一隐含参数组。现有 config 对 ParaX-only 实验记录的 `adapter_learning_rate` 可能为 null，虽然实际 optimizer 使用了默认 `4e-4`。下一轮必须单独记录 ParaX learning rate、weight decay、梯度范数和每层输出 drift，避免把优化尺度误判成结构结论。

### 6. 蒸馏组的 OOM 是实现问题，不是方法证据

当前蒸馏在一个 batch 同时保留完整 teacher/student，并执行三视图 CLIP forward，task1 OOM。因此不能据 task0 单点结果判断蒸馏是否有效。需要改成 task 开始时离线缓存旧 logits/features，训练时不再运行完整 teacher。

## 下一步方案

### 阶段 0：只修正可比性和诊断，不跑正式长实验

在新的实验分支上做以下实现修改：

1. 用 `torch.random.fork_rng(devices=[])` 包住 ParaX bank 初始化。
2. 为 ParaX 增加严格 identity 模式，选择“随机 residual path + 可学习 output gate=0”或“最终 projection zero + output gate=1”；不要同时把 gate、projection、expert B 置零。初始前向与 B0 的 token/logit 差异应接近机器误差。
3. 增加 paired 初始化检查：B0/P10 的 Selector、Prompt、classifier 参数 hash 相同；构造后 global RNG state 相同；第一批 DataLoader index 相同。
4. 将 ParaX learning rate、weight decay、trainable component 和 output scale 写入 config；记录 expert/router/gate 的梯度范数。
5. 先关闭硬 ratio cap，改为监控 residual ratio；如果需要约束，用可微的 bounded gate 或 residual penalty，而不是达到阈值后硬缩放整个梯度。

只运行 py_compile、单元测试和一个真实 task0 paired smoke，不启动 8-task test。

### 阶段 1：最小 paired validation，回答“ParaX 是否破坏 B0”

固定 seed0、同一数据顺序、同一 CLIP 权重、同一 fixed-three-view fusion、rank32、单 P-10、关闭 level embedding。只比较：

| 组 | ParaX | 目的 |
|---|---|---|
| B0-paired | disabled | 严格锚点 |
| P10-identity | P-10，zero-output identity | 检查加入模块但不改变表示时是否保持 B0 |
| P10-small | P-10，固定 `1e-3` 小 gate | 检查小残差的早期 drift |
| P10-zeroB | P-10，最终投影/Expert-B zero init | 检查真正从 identity 学习的残差 |

先做 task0、task1、task2 的短 validation，再决定是否跑完整 8 task。每组都要保存初始 logits、task0 结束 logits、task2 结束 logits，以及 Full/Person/Face 的 token cosine drift。

阶段 1 的判断规则：

- P10-identity 的初始 logits 与 B0 差异接近 0；否则先修实现，不看 mAP。
- P10-identity 的 task0 结果与 B0 接近；否则是 forward/optimizer/RNG 仍不等价。
- P10-small/P10-zeroB 若 residual ratio 受控但 feature drift 和 forgetting 仍明显升高，问题就是共享参数跨 task 漂移，而不是初始化。

### 阶段 2：拆开“共享中心”和“持续学习”

在阶段 1 通过后，仍只用 P-10、rank32、无 level embedding，做三种明确的 stability 对照：

1. **Shared-live**：当前 ParaX center/router 跨 task 更新，作为现状组。
2. **Frozen-center**：task0 学到的 shared expert center 固定，后续 task 只允许 router 或 task-local tiny gate 更新。
3. **Task-local delta**：保留共享 frozen center，每个 task 只新增很小的 task-local residual/gate；统计参数增长和旧 task drift。

所有组都用相同的 zero-init、相同 ParaX learning rate 和相同 task budget。每个 task 结束后在固定 anchor batch 上测旧 task logits、features、gate 和 residual。这样可以直接判断遗忘来自 center 更新，还是来自 router/task head 共同训练。

### 阶段 3：低显存旧任务蒸馏

如果阶段 2 仍有明显遗忘，再加入蒸馏；不要复制完整 teacher 在训练 batch 上同步前向。推荐流程：

1. task `t` 开始时，冻结当前模型快照。
2. 用 eval transform、按 Full/Person/Face 分路、`inference_mode()` 分批导出旧类 logits，按稳定 sample id 保存 CPU `float16/float32` 缓存。
3. 训练 student 时读取对应旧 logits，只对旧类计算 BCE/KL；不再保留 GPU teacher。
4. 先只蒸馏 Full，再比较 Full-only 与三视图蒸馏，避免一次引入过多显存和目标。

蒸馏应作为稳定性辅助，不应替代 paired B0/P10。需记录蒸馏损失、旧类 mAP、feature/logit drift 和显存峰值。

### 阶段 4：只有稳定后才重新验证 level routing

若阶段 2/3 至少能把 P-10 恢复到 B0 附近，再按以下顺序验证“一个共享 encoder 产生三个 level 的不同有效参数”：

1. image-only router；
2. image router + 很小的 level bias（只加到 gate logits，不拼接完整高维 embedding）；
3. level-conditioned router + gate entropy/usage balance 正则；
4. 固定 hard level routing 仅作为机制上界，不作为主方法。

每一步都必须报告同一原图三视图的 gate JS/L1 距离、同一 level 内不同图像的 gate 方差、single-view mAP、fixed fusion mAP、residual ratio 和旧 task forgetting。只有 gate 差异、对应单路性能和融合收益同时出现，才能声称 level-specific routing 有效。

## 预注册停止条件

- 如果 P10-identity 不能复现 B0 的初始 forward，停止所有性能实验，先修 autograd/RNG/插入时序。
- 如果 identity 正常，但 zero-init/小残差在短程 validation 中仍明显增加 drift 或 forgetting，停止在 CLIP image stream 内继续扩大 ParaX，优先转到冻结 backbone 后置 adapter 或显式 view-specific 小专家。
- 如果 frozen-center/task-local delta 能稳定但 shared-live 不能稳定，结论应是“ParaX 需要增量学习保护机制”，而不是“需要更多专家/更多层”。
- 只有在稳定性和严格 paired 对照通过后，才允许补 level embedding、连续层、rank64 或 seed1/2；当前不启动这些实验，也不继续用 test 选择结构。

## 当前不应做的修改

暂不恢复 P-8:10、P-8:10-level、rank64、seed1/2 或新的 test 搜索；暂不把 ParaX 放进 task-lane Adapter；暂不同时启用现有 image-token Adapter 与 ParaX。否则无法判断收益来自路由、参数量、插入位置还是多个适配器的叠加。
