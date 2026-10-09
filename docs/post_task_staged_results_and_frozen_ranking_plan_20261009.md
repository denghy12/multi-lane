# 后置 ParaX 分步校正结果与下一步融合方案

这轮回答了两个问题：能否在不改写旧任务的情况下增加 ParaX，以及这种动态路由能否带来识别收益。结果是：**旧预测保持稳定，但动态路由没有超过原固定融合，也没有超过均匀路由。** 因此停止继续扩展这条后置低秩变换，下一步只改变输出融合，保留视觉特征。

## 本轮实际改变了什么

CLIP ViT-B/16 权重完全冻结。每个任务先按原来的 MULTI-LANE 流程训练30轮，三种视图共用模型参数，原可靠 Face 固定融合不变；不启用 Image-token Adapter。

训练完成后，缓存训练图像在确定性裁剪下的三视图特征。在 **Task Forward 的最终归一化 lane [CLS] 特征之后、三视图固定融合与共享分类器之前** 加入 ParaX-style 残差。此处不属于 Frozen Forward 的 Transformer block。

所有任务和视图共用一套从开始就冻结的随机 parameter-matrix expert center，采用3个低秩参数矩阵专家、秩32。每个任务增加零初始化的32→32投影；动态组还训练输入条件 Router，均匀组使用相同的均匀专家权重。旧任务的投影及 Router 均冻结。可微边界将每个 token 的残差/原特征范数比例限制在2%以内。校正另外训练5轮，只用当前类标签，并约束校正前后 logits 的差异。

这不是原始 ParaX 全部训练流程的复现，而是针对本项目稳定性问题设计的受限对照。

## task0–2 validation 结果（seed0）

以下 mAP 是百分数，差值单位为百分点；没有运行 test。

| 实际修改 | task0 mAP | task1 mAP | task2最终mAP | 三阶段平均mAP | 协议forgetting |
|---|---:|---:|---:|---:|---:|
| 原共享模型，可靠性固定融合 |60.5225|59.0402|45.2395|54.9341|2.7760|
| 基线训练后，加入受限后置ParaX动态路由 |60.4657|59.0050|45.2254|54.8987|2.7550|
| 基线训练后，相同专家池使用均匀权重 |60.4731|59.0055|45.2258|54.9015|2.7588|

动态组最终低基线0.0141点、平均低0.0354点；均匀组最终低0.0137点。动态和均匀组差距仅0.0005点。单seed如此小的差异不支持方法优劣或收益声明。

| task2单视图mAP | 原共享模型 | 受限动态路由 | 均匀路由 |
|---|---:|---:|---:|
| Full |43.7058|43.6838|43.7003|
| Person |42.5883|42.5676|42.5657|
| Face |33.9794|33.9868|34.0354|
| 可靠Face子集 |39.4689|39.4584|39.4571|

最终 cF1/oF1：基线36.8870/61.6314，动态36.9456/61.6499，均匀36.9782/61.6879。阈值指标的微小增加并没有转化为 mAP 增加。

## 是否正常训练，以及问题在哪里

三组均完成90轮基础训练、5010次更新、0跳过；两种校正各增加15轮、835次更新。无NaN/OOM。校正前每个任务的 logits 对独立基线最大差都是0；compact checkpoint 的基础参数全部逐值相等。校正初始输出差0，基础模型、共享中心、旧路由的hash全部不变。

实际残差比例约1.1%–1.35%，梯度有限且非零。因此不能把没有收益解释为校正模块未生效。

动态gate熵约1.09，接近3专家均匀分布的最大熵1.0986；不同视图的均值非常接近，样本间标准差多为0.002–0.012。它没有塌缩成单专家，但近似一个固定混合，没有产生有用的视图专门化。这与动态、均匀组几乎相同的结果一致。

固定task0的2285张人物样本，比较task0与task2的旧五类预测：

| 方法 | 平均绝对logit变化 | task0旧五类mAP | task2同样本旧五类mAP |
|---|---:|---:|---:|
| 原共享模型 |0.000392|60.5225|60.4982|
| 受限动态路由 |0.000392|60.4657|60.4677|
| 均匀路由 |0.000386|60.4731|60.4727|

旧预测基本不变，协议forgetting约2.75主要不能解释为这些固定旧样本发生大幅漂移，因为协议逐阶段评估样本池会变化。现在主要是**新增模块没有提供额外有效信息**，而非旧共享参数被持续改写。受限随机中心也可能限制表达能力；这些结果不能推出所有ParaX形式均无效，但不支持继续扩大该路线。

## 固定融合附近还有多大空间

用已有validation预测作诊断：Person、Face权重偏移取{-0.05,0,+0.05}，Full保持权重和为1；不可靠Face始终为0，共9个候选组合。task2基于raw logits的固定mAP45.2410；用validation标签挑选最佳统一组合45.4331；每类挑选45.5114；逐样本/类别利用真标签挑选正例最高、负例最低得到48.1914。

这些都是**使用validation标签的有限网格参考值**，不是可部署模型、不是已经实现的收益、也不是任意融合方法的严格上界。raw-logit AP与正式sigmoid AP存在饱和/并列处理差异，不能直接混成一张正式方法对比表。它们只说明：附近有少量统一校准空间，样本互补更大的空间是否可学习尚未证明。

## 下一步：冻结预测，仅校准融合权重

已有动态特征融合、概率融合及OOF stacking失败或仅有很小收益，不能重复以增加复杂度为目标的搜索。这次独立检验三个变化：共享原模型的冻结输出、真正按原始图像分组留出的训练校准集、直接排序目标。

- 一套原共享模型：20%训练图像组在所有任务中均排除出基础训练，只用于融合校准；其余约80%训练原模型。使用现有固定SHA256 image-group划分，避免同图多人/不同视图泄漏。
- 对比来自同一模型的固定融合、两参数静态BCE权重、82参数动态BCE权重、相同82参数动态排序权重。每任务拥有一个融合gate，任务内跨类别共享；旧gate冻结。
- 输入为三个按校准训练集统计标准化的view logits、三个分数差异、Face可靠mask；隐藏层8、tanh、末层零初始化。允许样本及类别分数不同而产生不同权重，但没有独立类别参数或类别embedding。
- Person/Face各用0.05×tanh给出小偏移，Full抵消两者之和；可靠Face的Full最大偏移可达0.10，并非所有权重仅改变0.05。不可靠Face强制权重0。所有权重保持正值且和为1。
- 直接在保存的基线融合logits上加偏移：`z=z_fixed+deltaP*(zP-zFull)+deltaF*(zFace-zFull)`，避免AMP舍入导致初始不等价。后续单路特征和分类器不变。
- 排序组对每类正负样本分数差使用softplus，类间等权；温度为校准训练集该类固定logit标准差、下限1。无正负对的batch/class不计排序项，不用BCE替代；记录支持数量。
- 三校准组均加0.1×基线logit MSE和1.0×权重偏移平方惩罚；30轮、batch256、Adam0.001、cosine、FP32、关闭TF32、梯度裁剪1。不用validation选epoch或参数。

基础训练沿用seed0、task0–2、30轮/task、batch64、Adam主lr0.0125、cosine、weight decay0、AMP+TF32、当前类标签、辅助单视图BCE权重0.1。可靠Face权重[0.64,0.16,0.20]，否则[0.80,0.20,0]。CLIP checkpoint `./models/clip/ViT-B-16.pt`，数据 `./datasets/EMOTIC`，Face manifest `./output/emotic_face_manifest/face_manifest_audit_v1_20260908`。不加入ParaX、Image-token Adapter、Projector、level embedding或蒸馏，不访问test。

一套基础模型在GPU0训练，三种校准只处理缓存logits，避免重复训练四套CLIP。公平归因以同一80%训练源的固定融合为基线；另外报告与已有100%训练基线的差距，不能把不同数据预算混作配对收益。

晋级条件：动态排序相对同源固定融合和动态BCE的final mAP都至少+0.10，average不下降，收益不是单类驱动，旧固定样本预测稳定；若只有静态校准有效，只声称校准收益。只有超过完整训练基线才具备直接替换价值。不满足则停止继续这轮参数搜索，不追加test调参。

入口 `scripts/emotic/launch_multilane_frozen_view_ranking_val.sh`；输出 `output/emotic_frozen_view_ranking_val/<batch>/`，日志 `logs/emotic_frozen_view_ranking_val/<batch>/`。本轮先完成单测与真实两任务短程smoke，再启动正式批次；运行中不持续监控，结束后统一分析。

## 结果来源

完成批次 `post_task_staged_bounded_seed0_val_gpu0_20261008_01`；训练代码028d29c，文档追加8763ffe；三组日志、指标、checkpoint及缓存已同步本地。严格配对审计和融合网格诊断见该批次根目录 `paired_audit_and_fusion_ceilings.json`。

## 实施与正式运行状态

最终训练提交 `df5443e`。服务器完整259项单测通过；真实两任务各1epoch基础训练共124updates、0skipped，留出TRAIN样本task0/1为1031/817，三种缓存校准各1epoch，identity、Face mask、跨任务image-group exclusion、source不变检查通过。Face屏蔽修正额外复测通过。不对smoke精度作方法结论。

正式batch `frozen_view_ranking_seed0_val_gpu0_20261009_01` 已在GPU0启动，tmux `ml_frozen_rank_gpu0_20261009`，服务器worktree `/mnt/haoyuan/workspace/multi-lane-main-frozen-view-ranking-fusion`，Git状态clean。训练代码启动后不更新。实际输入以相对路径指向已有文件：`../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`、`../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`。没有复制数据或覆盖旧产物。待完成后统一同步分析，不持续监督。
