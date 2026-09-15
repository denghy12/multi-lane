# P0与独立R1性能差距诊断方案

状态：第一批已实现，等待服务器测试、smoke及正式seed0运行。目标是定位历史seed0 final validation差距0.505786，
而不是筛选新冠军。

## 1. 已核对的事实

来源：`output/emotic_track_a_face_endpoint_val/face_endpoint_val_seed0_20260909_1320/control/fusion_summary.json`
及P0 batch的`control/validation_summary.json`。

| 指标 | 独立专家/R1 | P0 | P0减独立 |
| --- | ---: | ---: | ---: |
| Full | 42.3799 | 41.6817 | -0.6982 |
| Person | 38.3044 | 39.7203 | +1.4158 |
| Face全量（含占位输入） | 33.3521 | 34.7389 | +1.3868 |
| 可靠Face（1460样本） | 36.3096 | 38.2278 | +1.9182 |
| 三路final mAP | 43.5812 | 43.0754 | -0.5058 |
| 三路average mAP | 50.7509 | 49.9807 | -0.7702 |

P0并非所有分支都变差。P2相对P0的分支退化不能用来解释P0相对独立R1的机制。
上表来自历史产物，不是同实现/训练样本流的因果对照；正式复评需验证样本ID、标签、Face mask和预处理。

代码证据：`MultiLaneModel.head`为线性层；各路lane特征归一化后加权求和，融合后不再归一化。
权重和为1，所以共享head下 `W(sum w_v*h_v)+b = sum w_v*(W*h_v+b)`。
因此P0特征融合等价于logit融合（浮点误差除外），而R1为sigmoid后的概率融合。
独立专家head不同时不具有此特征融合恒等式，不直接平均独立特征再任意套某一路head。

另一个真实混杂：独立Face基线用valid且非ambiguous样本训练；P0的Face融合/辅助损失只用更严格的可靠Face。
独立Face过滤后重建batch，P0在完整人物batch内使用mask，样本覆盖、每步有效样本数和更新预算不同。

## 2. 第一批：一个P0复现训练，完成融合差异和预测来源定位

### D0：带诊断产物的P0复现

锁定EMOTIC B5-C3，8 tasks，seed0，30 epochs/task，train/eval batch64，workers2；冻结OpenAI CLIP ViT-B/16。
Adam每task重置，主LR0.0125，Adapter LR4e-4，cosine到0，无warmup/weight decay；AMP/TF32。
共享Selector/Prompt/head，Image-token block1/b32/scale0.1/ReLU/independent；主BCE、Adapter ASL9.8/0/0.05。
训练仍为固定logit等价融合损失加0.1倍有效单路损失均值。Full/Person/Face预处理和可靠阈值完全复现P0。
可靠Face权重0.64/0.16/0.20，其余0.80/0.20/0。不改变训练融合以保持历史锚点。

训练入口为`scripts/emotic/run_multilane_track_a_p0_gap_reproduction_val.sh`，整批入口为
`scripts/emotic/launch_multilane_track_a_p0_r1_gap_diagnostic.sh`；离线A/B分析器为
`python -m multi_lane.track_a.p0_r1_gap_diagnostic`。
数据使用服务器已有EMOTIC、ViT-B-16.pt和face_manifest_audit_v1_20260908；实施时由配置指定输入路径并校验SHA。
输出`./output/emotic_track_a_p0_r1_gap/<batch>/`，日志`./logs/emotic_track_a_p0_r1_gap/<batch>/`。
新增产物：每task当前时点的compact可训练参数快照、validation三路原始FP32 logits/概率、标签、sample ID、
Face有效标记、原融合输出；必要时保存lane表示以支持冻结表示probe。冻结CLIP不重复保存。
快照必须涵盖该时点head与全部已见task状态，不能仅保存最终head代替历史task评估。
梯度审计保持早中晚采样；诊断不改变optimizer梯度、随机数状态或数据顺序。先复核P0指标/训练历史再归因。

### D1：同一预测上的2×2融合运算表

| 参数/训练产物来源 | 固定logit融合 | 固定概率融合 |
| --- | --- | --- |
| 独立专家 | I-logit | I-prob = 历史R1 |
| P0共享模型 | S-logit = 历史P0 | S-prob |

全部使用原固定权重与同一Face mask。优先复用旧独立专家scores；若只有概率，需审查截断/饱和，
不能把clamp后反推logit当作精确原值。缺原始logit时先完成I-prob/S-prob/S-logit三格，第四格标为缺失。
缺少P0checkpoint与分路逐样本scores，所以D0至少需复现一次；不凭现有mAP反推概率融合结果。

按指定路径作恒等式拆分：
`0.505786 = [mAP(I-prob)-mAP(S-prob)] + [mAP(S-prob)-mAP(S-logit)]`。
后一项量化相同P0表示上的评估融合运算效应；前一项仍包含参数、目标、数据协议等差异，不能称为纯共享损失。
四格齐全时报告另一分解路径和差分之差，揭示融合运算与模型来源的交互。负项也照实报告，不强行分配正百分比。
这只研究评估运算；训练改为概率融合是后续单独干预，不混入该表。

### D2：固定概率融合下的八种预测来源替换

三路分别从独立专家I或P0共享模型S取预测，共8种组合。权重固定、样本对齐，不重训练、不按结果选新模型。
重点观察(S,S,S)替换Full为(I,S,S)后能否恢复差距，以及其余Body/Face替换的变化。
汇总各路在4种其他分支背景下的边际效应；可算三路Shapley，精确归摊I-prob与S-prob端点差距。
这属于预测来源归因，不能把贡献解释成某个训练参数模块的因果效应；分类头置信度和互补性仍包含其中。
评估只混合概率，不混合不同坐标系的独立特征。

### 同步报告

- 每task累计及当前新类、历史旧类mAP，最终逐类AP、cF1/oF1、forgetting。
- 同一可靠Face子集同时报告Full/Person/Face及融合，另报不可靠子集；子集AP不做加权相加来解释全量AP。
- 比较分支相对Full的纠错/破坏、概率分布和分歧；总pair count不替代宏平均AP。
- validation按原始image ID成组配对bootstrap，预设2000次，报告95%区间；这不是训练seed方差。
- 不搜索alpha/beta、温度、阈值，不用test或用validation标签构造可部署oracle路由。

## 3. 第二批：若共同概率融合下仍有差距，再定位参数共享

保持D0样本、增强、目标、optimizer和步数；所有候选均记录两种融合评估。用加权分路logit定义训练融合，
使独立head时目标仍明确，且全共享时与P0数值一致。

| 配置 | Selector/Prompt | Adapter | head | 回答的问题 |
| --- | --- | --- | --- | --- |
| S0 | 共享 | 共享b32 | 共享 | D0基线 |
| SH | 共享 | 共享b32 | 各视图独立 | 是否主要来自共同分类规则 |
| SA | 共享 | 各视图独立b32 | 共享 | 是否来自Adapter参数绑定 |
| SQ | 各视图独立 | 共享b32 | 共享 | 是否来自共同证据查询 |
| SI | 各视图独立 | 各视图独立b32 | 各视图独立 | 所有可训练模块解绑的联合训练端点 |

P2不是SA：P2保留共享b32再加b4；SA是三路各自完整b32，二者回答不同问题。
新增模块从同一基线初始张量复制，独立RNG用于模块构造，公共数据流和增强按(seed,task,epoch,sample,view)对齐。
若这种实现改变历史随机序列，应在新受控实现内重跑S0，保留D0作历史锚点，不把两者混为精确配对。
本轮解绑同时改变可用自由度/参数量，发现正效应后必须补同预算全共享容量对照；P1 b45不能替代所有候选的预算对照。
若仅SI有效，提示模块交互，再补H+A、H+Q、A+Q组合，构成完整2^3设计；先不一口气跑所有组合。

## 4. 第三批：训练目标和Face协议的条件诊断

### 4.1 目标与共享的2×2

仅在第一/二批不能解释差距，或SI提示共享与联合目标交互时启动。

| | 纯单视图监督U | 融合监督加原aux0.1的J |
| --- | --- | --- |
| 可训练参数共享S | S-U（新增） | S-J（S0已有） |
| 可训练参数独立I | I-U（新增） | I-J（SI已有） |

U取三路有效单视图目标均值；主组使用BCE、Adapter使用ASL，head也由U训练。共享时各路梯度仍会相加，
这不是三个独立专家。保持LR、batch、更新预算，记录梯度尺度和实际Adam更新范数；干预包含目标方向和尺度，
若需区分尺度，应进一步单列匹配更新量/延长固定预算诊断，不能仅由原始范数宣称欠训练。
G2的head只接收融合目标，与本U方案不同，不能复用G2作为S-U的严格对照。
比较I-J与I-U定位联合目标效应，S-U与I-U定位该目标下的参数绑定效应，报告交互，避免单因素解释全部差距。

### 4.2 Face训练样本协议

历史Face用valid且非ambiguous，当前P0用strict reliable；按需在同结构同目标下仅改变Face监督mask，
验证两种mask。融合启用mask始终固定为R1严格可靠规则。新增实验必须记录每类正例、每步有效样本和累计曝光。
过滤后重组Face batch与在公共batch内mask也不同，如要严格复现历史独立专家需另设桥接对照。
历史独立Face锚点不可未经对齐就作为I-U，否则混入数据覆盖、增强随机序列与更新数量差异。

## 5. 决策与研究边界

第一批优先级最高：一次P0复现及离线诊断，先量化融合运算和分支来源；先不启动SH/SA/SQ/SI。
seed0用于定位该历史差距；只对明确的主效应与其基线补seed1/2配对验证，不把0.5058当作已知总体显著差异。
多指标/替换组合按预声明全报告，bootstrap区间用于描述不确定性，不按未经校正的多重检验宣称机制成立。
参数释放、训练目标、数据mask对照分别回答不同问题。特征CKA/梯度余弦可辅助解释，但相似度高或低本身不是机制证明。
在关键层保存表示时，冻结表示probe用train-only拟合并控制样本/容量；旧task数据回看只能标为离线诊断，
不能把诊断probe算作遵循原增量协议的新模型结果。

本方案没有引入Router或新共享/私有损失。下一项实现应先让D0/D1/D2成为可复现的小批次，之后由结果选择诊断分支。

## 6. 第一批实现清单

- `runner`新增`--save-view-evaluation-scores`，复用既有`view_evaluation_diagnostics`验证前向，按原batch保存
  fused及Full/Person/Face的FP32 logits和实际sigmoid概率、标签、sample ID、Face可靠mask与batch边界。
- 分析器强制审计P0结构、8个task普通score与分路score的一致性、Face manifest mask及8份compact状态；
  同时验证固定权重分路logit与模型原始fused logit的最大绝对误差。
- A完整计算I-logit、I-prob、S-logit、S-prob四格；B固定概率规则计算八种I/S来源组合。
- 输出累计/当前/旧类指标、逐类AP、遗忘、Face可靠/不可靠子集、置信分布与分支分歧、单路替换排序纠错/破坏、
  三路边际效应和Shapley，以及最终task四项预声明差值的2000次原图组配对bootstrap。
- 没有alpha/beta、温度或阈值搜索，没有访问test；没有实现或启动第二批模块解绑。
