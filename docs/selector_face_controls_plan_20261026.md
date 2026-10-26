# 独立Selector与冻结强Person模型的Face适配实验

完成更新（2026-10-26）：formal_seed1_20261026_01的12作业全部exit0，两组8任务Val/Test/OFF及参考重放已结束。576非权重产物回传并逐SHA验证，24源与16新checkpoint SHA一致。CPU复算208阶段融合/832视图最大误差7.11e-14，80旧参照重放、32完整保护及16Face mask逐值核验通过。新分析分支codex/selector-face-analysis-20261026，远程执行f6ef130保持clean。

独立S+B0头：Val38.6066/46.9717，Test29.2649/37.7092；原共训头40.9621/48.1452，30.7724/38.2750，按匹配原头也退步。全Attention OFF Test29.2269/37.5399，主要损失是共同训练后的表征/读出；A/P和Full K/V梯度仍共享，不是完整隔离。冻结强Person只训Face：Val43.0787/50.2760、Test32.8792/39.8493，原强Person43.0741/50.2828、32.8864/39.8687。保护与OFF恢复全部精确通过，Face非零更新，但无整体净收益。保留强Person和B0，不继续扩本次两结构。完整八阶段/分路/机制/局限见selector_face_controls_results_20261026.md；以下为启动时计划及历史执行记录。

执行更新：源码f6ef130已Git-only部署独立remote worktree multi-lane-main-selector-face-controls-20261026，branch/HEAD/upstream一致，旧主树/test-only的HEAD/status SHA不变。7新增专项及495项完整回归通过（21.979秒）。GPU初始化审计确认三套Selector精确克隆、其余初值/CPU-CUDA RNG/初始预测等价，新增122880参数；smoke_seed1_20261026_01全部成功，冻结组更新后Full/Person/无可靠Face预测逐值保持，Face OFF回到强Person，原Person/双Attention参考逐值重放。

正式formal_seed1_20261026_01已自动启动于GPU0，tmux selector_face_controls_seed1_20261026。PRIVATE_SELECTORS训练PID1972432，FROZEN_PERSON_FACE训练PID1972995，参考Val/Test也排入同一4并行队列。两组task0前4epoch各84成功更新/0跳步；前者30epoch/任务，后者额外5epoch/任务。GPU约8537MiB/100%。正式8任务及所有Val/Test/OFF评估将自动接续，当前尚无最终成绩。源24份checkpoint SHA保持，旧实验/主树/test-only未修改。

106份smoke/单测/预测文件已回传本地output/selector_face_controls/smoke_snapshot_20261026_01并逐SHA核验，归档SHA5b3a5a3761a67eba3397c3c1f7b361f00eced32b92be3dcf9d0063569c0e3c65。运行源码保持remote f6ef130 clean；当前仅本地追加执行记录。完成后分析独立Selector的共享/跨K-V通路差别，以及冻结强Person后Face条件收益能否保留，不预先宣称提升。

2026-10-26用户授权，两组均seed1、GPU0、完整8任务validation/test。实验分支codex/private-selectors-frozen-face-20261026，当前实现完成，待远程测试/smoke。

## 第一组：三个分路各10个独立Task Selector

参照前轮Person＋可靠Face Attention共训表征＋B0头（Val43.3394/50.5522，Test32.7573/39.7004），只把Selector模式改为view_specific。每任务Full/Person/Face各10个，Tensor从[8,10,768]变为[8,3,10,768]。三个独立参数切片初值完全相同，不重新抽样；后续任务对应视图复制上个任务的同视图Selector。与旧方案相比新增122880参数。

Image-token Adapter及Prefix Prompts仍共享。Attention仍以接收视图的任务Selector作Query、Full摘要作K/V，Person/Face各有独立模块。Face损失仍可能沿Full K/V回传Full Selector；本组不加detach、不拆Adapter/Prompts，因此不是完整梯度隔离。若Full仍下降，不能据此证明所有共享路径均已排除。

仍30epoch/任务，主LR.0125、Adapter/Attention LR.0004，Adam逐任务重置、cosine/WD0；b32/index1/scale.03/ReLU，Adapter ASL(9.8/0/.05)，其他可训练参数BCE+aux.1，Person颜色扰动0。Attention第6层/index5、inner64/heads4/scale.1/cap.1，可靠Face规则不变。训练原共用分类头，推理换对应阶段B0头；保留原共训头ON的读出控制及B0头下ON/Face OFF/全部OFF。

验收：GPU初始化审计三个Selector拷贝、其余参数、CPU/CUDA RNG和初始预测逐值等价；训练后三路Selector应分化，旧任务参数冻结；完整阶段CPU RNG摘要对照旧双Attention源。每epoch首batch额外记录三视图Selector梯度范数。

## 第二组：冻结Person-only强候选，只训练新增Face Attention

各任务从原Person-only对应阶段checkpoint装载Adapter、共享Selectors/Prompts、原Person Attention，分类头替换为对应阶段已经学会当前类别的B0 H，全部冻结。保留ASL Image-token Adapter运行，不再更新Adapter。仅当前任务新增Face Attention q/k/v/out四个投影可训练，旧Face模块冻结；新任务只装载Teacher原参数，不覆盖已训练Face模块。

额外5epoch/任务、LR.0004、Adam重置/cosine/WD0，融合BCE+原aux.1，仅Face模块接收梯度。使用全部当前任务TRAIN正例人群与原三视图增强，独立sampler317001+t、额外阶段RNG20261025+1009t并恢复。整批无可靠Face时无可训练梯度，省略该batch的反向和Adam步骤，单独记录no-signal，不当作AMP skip；不改变Face判定阈值。

每阶段训练前当前类别预测重放原强候选，参数保护覆盖视觉骨干、全部Adapter/Selector/Prompt、原Person M、分类头及旧Face M。评估Full/Person所有logits/probabilities逐值不变，无可靠Face人群的Face/融合预测逐值不变；Face OFF所有输出逐值回到原Person强候选。只关闭新Face，不关闭原Person Attention。该组与旧“冻结B0只训Person M”锚点、更新分支都不同。

每任务先存compact及optimizer/scheduler/scaler/RNG恢复状态再评估。当前H按阶段来自B0，未来类别在其任务完成学习后再装载，不永久冻结随机未来头。

## 公共配置及调度

两组均B64/workers2/OMP-MKL72、AMP65536/growth2000/TF32、EMOTIC原划分、历史samples_intersect_seen_classes、8阶段算术平均。Test已参与探索，继续如实标记。

主入口`multi_lane.track_a.run_selector_face_controls`，Face-only训练/初始审计`multi_lane.track_a.selector_face_controls`，评估复用`multi_lane.track_a.head_anchored_eight`。输出`output/selector_face_controls/<run_id>`，日志`logs/selector_face_controls/<run_id>`。

输入均为既有相对路径：
- 数据`../multi-lane-main/datasets/EMOTIC`。
- CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- Face清单`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- B0 `../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- Person-only共训源为上述同级`ATTENTION_BOUND/train`。
- 上轮study `../multi-lane-main-head-anchored-eight-face-20261025/output/head_anchored_eight_face/formal_seed1_20261025_01`；双Attention训练源在其中`FACE_JOINT/train`，强Person逐值参考在`PERSON_REFERENCE/b0_val/on`及`b0_test/on`。

只用GPU0，最多4个训练/评估作业，空闲显存6000MiB/shm1024MiB才放行，启动间隔12秒；两个新训练并行，原Person/双Attention参考重放也入同一队列。先单测及GPU smoke（两任务各两更新、新模型只Val；已完成旧参考允许Val/Test重放），同源码gate通过后自动完整训练/Val/Test。

首要结果是独立Selector是否优于旧双Attention，以及固定强底座的Face-only是否超过32.8864/39.8687并维持逐值保护。所有控制与完整指标都报告，不依据最好单点省略不利结果。

## 文件改动

- `multi_lane/track_a/model.py`：限定开放历史双Attention的独立Selector组合。
- `multi_lane/track_a/historical_b32_modules.py`、`runner.py`：显式配置/校验开关、实际记录及分视图Selector梯度日志。
- `multi_lane/track_a/head_anchored_eight.py`：不同Selector形状时只替换分类头，新增冻结Face组的逐值预测验收。
- `multi_lane/track_a/selector_face_controls.py`、`run_selector_face_controls.py`：初始等价审计、仅Face训练、原参考重放及GPU0自动队列。
- `tests/test_selector_face_controls.py`：新增7项真实梯度、冻结、初值、无信号及预测保护检查；完整495项回归通过。
