# Person融合下一阶段：证据、实现审计与实验优先级

本次是结果分析与方案，不修改业务代码或启动新实验。`selector_patch_2x2_test_20260906`四组已经
完整结束、同步并核验：各240 epochs/13,950 updates/skipped0，源码固定`f1ada22`。

## 2026-09-06 2×2最终结果与路线收敛

| Full crop | Person条件 | final mAP | average mAP | cF1 | oF1 | forgetting |
|---|---|---:|---:|---:|---:|---:|
| legacy | disabled | **32.5365** | **39.1445** | **32.5756** | **49.3827** | **4.7308** |
| legacy | selector-specific patches | 31.6826 | 38.6537 | 32.2015 | 49.0288 | 4.7496 |
| target-aware | disabled | 30.5018 | 37.5976 | 32.3379 | 48.0792 | 4.7669 |
| target-aware | selector-specific patches | 30.2505 | 37.4263 | 31.2644 | 47.5866 | 5.0422 |

- legacy下patch使final mAP下降`0.8540`，仅task0提高`0.2132`，task1--7全部下降；最终26类
  5升21降，task6 Sadness/Sensitivity/Suffering变化为`+0.6527/-0.0981/-4.8180`。
- disabled下target-aware使final mAP下降`2.0348`；patch模式下下降`1.4321`。它将test bbox可见率
  从`0.8032`提高到`0.9778`，却更差，说明“框更完整”不等于情绪识别更好，原中心构图和上下文重要。
- 真实train Person mask最少28个有效patch，空mask频率`0/16,001`；已知空mask bug仍应修，
  但不是本次负收益原因。target-aware训练loss更低而test更差，更符合泛化/分布问题而非未收敛。
- 两条路线到此关闭：不补seed1/2，不搜索当前target-aware参数，也不继续搜索query-only的层数/容量。
  完整结果见`output/emotic_track_a_person_conditioned_selector/selector_patch_2x2_test_20260906_analysis/analysis.md`。

## 已确认的证据

- 单Full seed0 test 32.5365；同seed Full+Person seed0 33.2672。三seed单Full
  32.1562±0.3701、双视图32.8263±0.4025，不能混用单seed与均值比较。
- 匹配循环辅助seed：Full+Full 32.6551，Full+Person 32.7957，差0.1406；说明人物有额外
  贡献，但整体双视图收益也包含集成效应。循环配对共享模型，不作独立重复显著性结论。
- 共享Person CLS条件：31.1695；bbox+Person 31.3557；均差于32.5365，且task0已下降。
- 强约束静态门控三seed32.8380，仅比固定融合高0.0117；样本级MLP的8组validation均值
  全部未超过同协议固定融合。门控source使用90%fit，不能直接当100%train的新锚点。
- 容量/LR、ASL局部、层组合、训练步数、输出正则、可学习scale、epochs、scheduler都已广泛
  尝试；没有证据支持再重复大网格。

## 本次代码审计的关键结论

1. 当前Person只修改Selector query；`selected`仍完全由Full tokens加权求和。它可以改变
   从Full选择什么，但无法直接将高分辨率人物crop中独有的信息作为value送入分类路径。
   最值得验证的是保留Full汇总，再加入Person内容残差。
2. patch条件layer1使用经过block0的Person tokens；同深度对齐成立，但这是浅层信息，
   不等价于旧完整Person模型的情绪表征。Adapter最优层也不能自动视为融合最优层。
3. padding只在Selector到Person的softmax前屏蔽；Person block0内部自注意力未屏蔽padding。
   patch中心规则也允许边缘patch含部分padding，因此目前不能声称完全隔离补边。
4. 可复现边界bug：all-false patch mask被safe_mask变成all-true，但最终valid不包含
   mask.any()；将输出bias置1、scale0.1，空mask仍得到0.1增量。3×200细长crop的14×14
   中心mask可全空。尚未统计真实数据触发率，不能认定这是当前mAP的主要原因。
5. target-aware eval固定Resize256后截224，超大目标仍会被截断；训练拒绝采样失败时把
   整张图非等比例resize到224，可能改变人物比例。当前2×2只区分两种完整预处理策略，
   无法拆开train crop与eval crop各自贡献。
6. 旧分类头持续破坏旧类的历史解释需要纠正：training_loss_view把非当前logits置常数零，
   其梯度为零；Adam按task重建，head weight decay为零。小模型复核旧head梯度/更新均为0。
   当前全量selectors/prompts张量在优化器中也不等于旧slice一定更新。

历史disabled的真实test scores按sample ID对齐，检查相同旧类与相同样本：

| 检查的类别集合 | 引入时mAP | task7相同ID mAP | task7全部样本mAP |
|---|---:|---:|---:|
| task0五类 | 52.7473 | 52.7182 | 36.3269 |
| 截至task5的20类 | 36.2052 | 36.1814 | 36.0551 |
| 截至task6的23类 | 35.1618 | 35.1541 | 35.1531 |

task0样本池从3142扩大到5368；旧类正例已在早期池中，后来增加大量旧类负例。这说明至少
task0的AP大幅下降主要与评估池扩张有关，不能解释为参数灾难性遗忘。相同ID仍有小幅数值差，
其来源（AMP/批次/多lane计算路径等）需固定输入FP32复评估确认，不能说严格零遗忘。
历史forgetting指标继续原样报告，另外增加固定ID的预测漂移/固定样本AP。

## 优先级与边界

| 优先级 | 方向 | 可行性 | 收益判断 |
|---|---|---|---|
| 0 | 完成当前2×2与mask/评估队列审计 | 高 | 保证方向正确；修复本身未必涨mAP |
| 1 | Person summary直接残差进入Full Summarize | 高 | 中等潜力，直接解除query-only信息限制 |
| 2 | 统一模型的深层Full/Person特征残差融合 | 中高 | 更接近已验证有效的双视图语义互补，计算较贵 |
| 3 | bbox映射/ROI人物对照 | 高 | 低至中等潜力，能区分人物定位与crop分辨率收益 |
| 4 | 当前任务辅助监督或同task教师蒸馏 | 中 | 条件性收益；需先确认人物分支确有独立判别性 |
| 5 | 新负例覆盖/旧类适应 | 需另立协议 | 潜力不确定；不能把未来任务数据提前作为训练负例 |

这些是研究优先判断，不是成功概率或mAP数值承诺。无需因个别类、F1或历史forgetting下降
直接否决final mAP更高的候选，但要完整报告。

## 最小分阶段实验

阶段0已完成：四组均240 epochs/13950 updates/skipped0，全部score与来源通过逐文件哈希审计。
下一批固定原legacy crop；当前target-aware不再升为默认。
修复空mask后统计train/val的有效patch数、Full可见率、crop回退率、Selector间注意力相似性及
残差范数；若需完全屏蔽padding，采用Person专用masked frozen-block前向，保持Full分支原样。

阶段1：先四组seed0完整8-task validation，固定原冠军Adapter和legacy crop：

1. Full disabled锚点（这里仅关闭Person融合，Image-token Adapter保留）。
2. value residual：保留原Full summary，额外加入每Selector的Person summary经过零输出
   MLP的残差。读取同layer1 Person tokens，参数预算与query-only相同。
3. frozen深层特征残差：Full task CLS在最终归一化前加零输出MLP(Person最后层CLS)，然后沿用
   原投影、归一化与分类路径。Person暂冻结，hidden32、scale0.1、LR4e-4。额外计算如实记录。
4. task-adapted深层Person lane：Person视图增加当前task独立的轻量Adapter/投影并输出深层语义，
   再经task独立零初始化门写入Full lane。第一轮不叠加aux loss，避免无法归因。

只固定一个位置验证机制，不展开层数/容量搜索。query-only已有完整负结果，不再重复。浅层value组
验证query/value信息瓶颈；两种深层组筛选冻结语义与task-adapted语义，不能将整体差异归因于单一因素。
采用完整8-task final validation mAP优先，average及task5--7为辅助；类别表现不是硬门槛。

阶段2：若有赢家，追加有针对性的四组，而非自动填满GPU：同配置复测、同预算Full/ROI内容残差
对照、+当前task Person辅助BCE、+同task双视图教师蒸馏。ROI对照区分定位和独有Person信息。
辅助BCE必须作用于能影响主预测的Person特征投影，否则独立aux head训练没有主路径收益。
蒸馏teacher只能用到task t状态与t时允许的数据；最终task7教师不能倒灌早期任务，旧test/val
预测不能做训练目标。没有相应teacher状态/训练预测时需要重新生成，计入成本。

阶段3：seed0胜出后锁定结构与超参数，补seed1/2 validation及同预算对照；随后一次锁定test。
单模型32.5365、双视图seed0 33.2672分别是不同成本参照；统一双视图仅超过单Full说明有收益，
要说优于已存在的固定双视图方案，还需超过匹配seed/预算的Full+Person。

持续增量约束：每任务融合参数独立，旧任务冻结；训练只监督当前允许标签；推理按原lane类别
拼接，不使用真实task标签挑lane。若改变共享模块更新或负例数据可见范围，需要单列协议及锚点。

## 依据

- 本项目`model.py`的_lane_block、encode_lanes及runner.py的training_loss_view、dataset_view。
- CocoER本地`models_sw.py`约888行起，body/head坐标映射到context位置嵌入，跨视图attention
  将value写入特征；约1195行将refined特征连接后分类。启发是空间定位和内容级交互，不直接
  移植其全26类监督、逐样本竞争流程或教师到当前增量协议。
- 冻结主干、残差跨注意力与零初始化接入可参考Flamingo原论文及补充材料
  https://arxiv.org/abs/2204.14198 。这只支持结构可实现，不证明EMOTIC上会涨mAP。
