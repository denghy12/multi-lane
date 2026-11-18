# Full／Person／Face 样本级融合训练

EMOTIC 历史划分、seed1、8任务、30轮、batch64。两个训练模型从既有 CLIP ViT-B/16 预训练权重出发，重新初始化各任务的 Adapter、Selectors、Prompts、Person 和 Face 的 Frozen9 交互、分类头及融合模块。分支和门控各自训练并随任务封存，两个协议分别保存完整状态。输入沿用 run_r1_frozen_cross_view.input_paths() 中的 EMOTIC、Face manifest 和原分支配置。

| 模型与数据分工 | 门控输入 | 融合训练损失 |
|---|---|---|
| 全 TRAIN | Frozen9 跨视图关联，Task10 Selectors 在接收端空间位置加权 | BCE |
| 全 TRAIN | 三路最终 Task CLS | BCE |
| 90% 分支／10% 融合 | 同一中间关联 | BCE |
| 90% 分支／10% 融合 | 三路最终 Task CLS | BCE |
| 同一 90% 分支／10% 融合轨迹 | 与中间 BCE 相同的特征和初始读出 | ASL |

每个样本指原图中的一个目标人物。每个已见任务对应独立的预测器，按 Face 是否可靠选择零初始化的截距，按三路得分在有效视图上做 softmax。各分类头输出原始 logits 之后按比例求和并逐类 sigmoid。可靠 Face 从三等份开始，不可靠 Face 从两等份开始。固定比例与等权仅作评价参照。

中间读出在 Person 和 Face 各读取与 Full 关联的四项统计量。用第10层 Summarize 的接收端空间权重汇总第9层逐位置统计。末层对归一化后、送入分类头的独立 Task CLS 作确定的低维随机投影，Full／Person／Face 分别保留3／3／2维，均使用8维输入的线性读出，并遮蔽无效 Face。一次低维投影的比较只能说明这些实际接口的效果。

完整 TRAIN 和永久留出组均用当前任务合法 TRAIN 人物。后一组按原图固定分工，同图所有人物及视图跨任务归于同一侧。两组每个任务的融合训练预算均为300次更新；适配与分类的损失只更新分支。融合特征和分支 logits 在融合监督入口断开梯度。融合 ASL 使用仓库已有公式，对当前任务类别计算，gamma_neg=9.8、gamma_pos=0、clip=0.05；第五组与第三组配对。采用相同分支预算和数据覆盖之间的区别，以实际人数及更新数报告。分支 Adam／cosine 主学习率0.0125，Adapter、交互和门控0.0004，weight decay 0，BF16。详细配置随输出保存。

运行 bash scripts/emotic/run_d9_sample_fusion.sh <smoke或formal> <sample_full或sample_holdout> <GPU>。输出位于 output/d9_sample_fusion/seed1_20261118，日志位于 logs/d9_sample_fusion/seed1_20261118。正式输出包括逐任务 checkpoint、训练人群覆盖、门控学习记录、八阶段逐人 logits 和权重、Validation 与 Test 的最终及阶段平均 mAP、同任务同可靠性内交换权重的比较。全 TRAIN 的固定读出核对受保护历史参照。比较各自同轨迹的固定读出与绝对参照，Test 最终33.2019、阶段平均40.2847，Validation 同批对照43.6272／51.1094。结果是否达标以正式评估为准。
