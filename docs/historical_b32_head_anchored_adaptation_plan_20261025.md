# 下一阶段：以B0分类头为锚点的任务表征适配

2026-10-25用户已授权执行，新入口`multi_lane.track_a.head_anchored_pilot`、GPU0队列`multi_lane.track_a.run_head_anchored_pilot`。当前实现中，待远程测试及smoke。正式范围仅seed1/task0–2/5epoch/Val。复用当时阶段B0 checkpoint，省去重复30epoch Teacher训练；总训练成本仍包含原B0训练和新增适配。

输出`output/historical_b32_head_anchored_pilot/<run_id>`、日志`logs/historical_b32_head_anchored_pilot/<run_id>`。源B0 `../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`；数据`../multi-lane-main/datasets/EMOTIC`；预训练`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face清单`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

共四学生+BASE重放5作业，最多4并行，显存空闲≥6000MiB、共享内存≥1024MiB才放行，间隔12秒。Teacher SHA对照既有training_verification，各阶段初始当前类别预测逐值等同Teacher，BASE及MOnly OFF重放完整Val。每epoch实际消费前两batch的ID/像素/目标摘要跨四组一致。任务边界评估前保存optimizer/scheduler/scaler/RNG和参数；中断可在同源码/同配置下补评估，不重复已完成任务更新。

## 1. 根据当前证据调整方向

现有受限Attention的共训ASP换回B0 H后，关闭模块已有Val43.0278/Test32.8651；开启为43.0741/32.8864。它提示当前表征中仍有可用信息，应优先检验读出约束，而非把所有表征冻结当作唯一方案。新Attention的即时贡献只有约0.0462/0.0213，需要专门无Attention控制。

位置模块没有类似换头收益，直接贡献又很弱，当前不作为主候选。Face可靠性硬切换在两split未形成可靠改善，当前不加新Router或可靠性开关。以上是探索性seed1信号，不是新方法正式提升证据。

## 2. 保留MULTI-LANE两条清楚的逻辑

每任务t先按原B0配方学习当前任务：三个视图的Frozen Forward共享同一预训练视觉骨干，Image-token Adapter只改Selector读取证据，Selectors完成Summarize，Prompt参与Task Forward，三视图任务特征按原方式融合，分类头学习当前类别。旧任务参数照常冻结。

完成该任务B0学习后，才进入额外适配阶段：保留一份原始B0参数供基线传递与验收，复制当前任务的Adapter/Selector/Prompt给学生适配路径。学生分类头取该阶段已经学好的B0分类头并冻结；不永久冻结还没有学过未来类别的随机head。

```text
任务t B0训练：按原配方学A_t、S_t、P_t、H_t
           ├→ 保留B0参数/RNG，供下一任务原始初始化传递和B0验收
           └→ 当前任务ASP副本＋固定H_t → 少量额外适配（可含限幅Attention）
```

下一任务B0的Selector/Prompt初始化来自上一任务B0，不能来自学生适配结果。学生新任务只装载Teacher当前任务坐标；已训练的旧学生ASP/Attention必须保留冻结，不能被新Teacher整份checkpoint覆盖。学生H更新时只复制Teacher新阶段的H，旧类别行须保持一致。

额外阶段必须使用独立DataLoader generator/随机流，并在结束后恢复Python/NumPy/Torch CPU/CUDA RNG。新模块初始化不消耗B0随机流；B0原训练/报告的调用顺序和72线程环境保持。仅冻结参数并不足以保证32.6800轨迹，因为额外迭代本身也可能改变下一任务的全局采样RNG。

先用已有对应阶段B0 checkpoint做pilot，可省去重复B0训练；实现算法时只能使用当时任务训练完成的checkpoint，不能使用未来阶段权重。

## 3. 四个匹配预算的适配控制＋B0参照

| 组别 | 当前任务ASP | 当前阶段H | 新Attention | 回答的问题 |
|---|---|---|---|---|
| B0 | 不增加更新 | 原B0 | 无 | 保留强参照 |
| 只训练新模块 | 冻结B0 ASP | 冻结B0 H | 训练限幅Attention | 新分支能否独立增加证据 |
| 固定头，不加Attention | 从B0复制后微调 | 冻结B0 H | 无 | 约束读出及额外微调本身是否有效 |
| 固定头，加Attention | 从B0复制后微调 | 冻结B0 H | 训练限幅Attention | 主候选；模块是否带来额外净收益 |
| 同预算允许头更新 | 从B0复制后微调 | 也可训练 | 训练限幅Attention | 隔离冻结分类头是否必要 |

除明确列出的H/ASP/Attention开关，各学生组初始化、样本、更新预算、AMP、优化器与监督完全一致。不能把额外训练带来的改善都归给Attention。

## 4. 首轮建议锁定的配置

- seed1，先task0–2 validation pilot，再决定是否完整8任务。仍原EMOTIC划分、旧seen-positive人群、B64、workers2、OMP/MKL72、AMP/TF32。
- B0使用历史完整配方：b32、index[1]、scale0.03、Person颜色扰动0、aux0.1、主LR0.0125、Adapter LR0.0004、30epoch、Adam/reset/cosine/WD0。B0不因新阶段改变。
- 学生额外阶段先固定5epoch，不做LR/温度网格。建议ASP和新Attention LR0.0004；可训练head控制也用0.0004，比较冻结与更新而非同时改多个学习率。此处是待验证配置，不是已确认最优值。
- 学生中可训练Adapter沿用ASL（9.8/0/0.05）及原辅助ASL；S/P/Attention/可训练head接受原融合BCE＋0.1视图辅助BCE，可靠Face掩码不变。完整冻结ASP的组只优化Attention的BCE，冻结Adapter不再接受ASL更新，不把ASL扩大到所有参数。
- Attention沿用index5任务Selector Query读取Full、写回Person摘要的版本，inner64/heads4、scale0.1、平滑RMS上限0.1。先固定旧模块检验优化约束，不同时换Query、增加层或引入位置偏置。
- 所有学生组使用相同额外TRAIN人群/更新预算；不改变原B0训练人群。先按validation选择，不凭Test最高值回头改配置。既有Test已参与探索，应始终保留该事实。

## 5. 必须分清两个“关闭”

1. 只关闭Attention：仍保留适配后的学生ASP，用来测Attention的条件贡献，不能称为重新训练的B0。
2. 关闭整条适配、回到Teacher参数路径：才应逐阶段复现B0。

完整冻结ASP的“只训练新模块”组，这两个开关可重合；ASP也微调的组不能混淆。保存原B0 checkpoint作为回退和初始化来源，而非要求仅关闭Attention就回到32.6800。

## 6. 验收与继续条件

- Teacher基线路径每任务参数与RNG保持预期，回退到Teacher后原B0预测精确重放；Frozen Forward始终不变；旧学生任务ASP、M、head旧类别行冻结。
- “固定头＋Attention”应优于同预算“固定头、不加Attention”，并在自身开/关Attention时显示可重复的validation贡献；否则不把收益归为新融合模块。
- 同预算允许head更新控制用于判断H约束是否有必要，不把本轮内存换头效果直接当成冻结训练必然有效。
- 同时报Full/Person/可靠Face、无Face回退人群、多人/单人和类别变化。候选无Face Test仍下降，不能只看全体mAP；但不据此直接硬拼两套模型分数。
- pilot通过后才完整seed1八阶段Val/Test；最终mAP和八阶段平均都要与32.6800/39.5292比较。再验证seed0/3，保留既有seed0强Full32.5365/39.1445。当前不声称32.8864代表八阶段或跨seed成功。
- 本轮混合候选cF1提高但oF1下降，F1与mAP分别报告；不通过阈值搜索制造mAP改善。

## 7. 成本与限制

本方案增加额外训练阶段；需把B0训练和学生适配成本都计入，匹配控制也获得同样额外预算。训练/审计需保留Teacher和Student当前/旧任务参数，不需要复制预训练视觉骨干权重。完整ASP为每任务134432参数、8任务1075456；限幅Attention每任务196608、8任务1572864。保留双ASP与Attention时，相对B0约多2648320参数，若另存学生head再加13338；实际部署可按回退需求决定是否常驻Teacher参数，显存/吞吐实测后再汇报。

这是一套优化与参数传递约束的候选，不是已经成立的模块创新。若不加Attention的固定头控制同样好，应如实把改进归为表征/读出训练策略。本节方案已经用户授权进入实现；当前执行状态见文首及上下文文档，不能把先前只读换头结果写成这次训练的结果。

## 8. 当前三路分类的数据流

三视图是同一目标人物的Full整图、Person裁剪、Face裁剪。三路图像分别通过Frozen Forward，但共享同一组冻结CLIP视觉骨干权重，并非三套独立训练的backbone。同一任务的A/S/P也由三视图共享，输入内容不同。

在各层，Task Selectors通过Summarize读取当前视图图像证据；Image-token Adapter位于第2层的Selector证据分支，只改读取的图像token，不写回Frozen Forward。Task Forward结合摘要和Prefix Prompt做任务注意力、Drop & Replace及FFN。新增限幅Attention位于第6层摘要入口：任务Selector Query读取Full摘要K/V，残差写入Person摘要，然后继续原Person Task Forward；Full与Face分支结构保留。

三路任务CLS经ln_post与CLIP projection，得到512维特征，各自L2归一化。可靠Face时融合0.64Full+0.16Person+0.20Face；不可靠Face时0.80Full+0.20Person，Face为0。代码在加权和后不再次L2归一化。

融合特征送入一个共享线性分类头H，得到多标签logits，独立sigmoid为各情绪概率。训练还把每路特征各自送入同一个H以提供aux监督；不是三个独立head，也不是从三路里挑效果最好的一路。主目标为融合BCE，加0.1乘有效视图BCE的均值；Face可靠性掩码用于Face辅助监督。Adapter对应路径接受融合ASL+0.1视图ASL，其余可训练参数接受BCE。冻结H只禁止H更新，不阻断梯度流向前面的可训练参数。

增量推理对全部已见Task Forward计算各任务对应类别logits，按task_class_mask取出各自拥有的类别并拼成累计类别预测，不挑一个最优任务。pilot到task2只含11个已见类别，不能和task7的26类别结果直接比大小。
