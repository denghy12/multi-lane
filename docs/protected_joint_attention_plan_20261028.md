# 从完整模型互补到单模型：受保护的三视图交互摘要方案

日期：2026-10-28。状态：研究方案，未修改模型或训练入口、未启动实验。

## 1. 研究目标与证据边界

目标是在预先定义的一套训练流程中，由一个模型自行学习三视图融合；只使用标准预训练CLIP和当前任务真实标签，不导入旧A/B的任务参数、分类头或预测答案。最终同时提高历史标准下的Test最终mAP与8阶段平均mAP。

| 已有结果，seed1，8任务 | Val最终/平均 | Test最终/平均 |
|---|---:|---:|
| 原三视图强B0 | 42.4628 / 49.4763 | 32.6800 / 39.5292 |
| A：Person-only共训表示＋B0头 | 43.0741 / 50.2828 | 32.8864 / 39.8687 |
| B：Person＋可靠Face共训表示＋B0头 | 43.3394 / 50.5522 | 32.7573 / 39.7004 |
| A/B完整模型特征固定均值，真实前向 | 44.1550 / 51.2964 | 33.6248 / 40.5129 |

完整均值说明两套预测函数存在可利用的互补，并不唯一证明Attention、Face或某一种梯度冲突是收益来源。均值同时改变Full、Person和无可靠Face人群的预测。旧离线控制中，B关闭全部Attention仍有33.5854 / 40.4390，多数均值收益仍在；此数不是本轮真实均值前向新跑的OFF结果。

完整均值实测推理约2.045倍，来源训练含B0/A/B各30轮，不是单次30轮公平对照。33.6248适合保留为有成本标注的经验参照，不作为新模型教师或必须复现的理论上界。

本方案的假设是：任务相关摘要中存在末端固定融合尚未充分利用的联合证据；一个具有目标人物条件、三视图联合访问权限的轻量读出器，可能在不改坏原任务表示的前提下利用这些证据。该假设尚待验证。

## 2. 从MULTI-LANE看当前缺口

MULTI-LANE用Patch Selectors从Frozen Forward的patch tokens产生Summarized Patches，再用任务专属Prefix Prompts及Task Forward得到任务表示。Drop & Replace让任务路径在后续层重新读取摘要证据；增量阶段结束后封存任务参数，测试运行已见任务路径。

当前三视图扩展让同一任务的Selectors/Prompts/Adapter作用于Full、Person、Face，分别得到任务CLS，最后固定特征融合。三视图及其辅助监督是本项目扩展，不是原MULTI-LANE论文的原生设计。

旧Attention在第6层读取Full摘要，向Person或可靠Face摘要直接写残差。虽然Frozen Forward冻结，新增损失仍能经共享Image-token Adapter、Selectors和Prompts改变三路任务表示。近期2×2读出诊断说明，共享表示变化不能单靠换分类头解释；局部TRAIN BCE/Adam步保护也没有保住Test排序。

因此需要区分两个职责：原Task Forward学习可用的任务表示；新增交互读出器学习如何利用三视图的联合证据。首版让两种职责具有明确的梯度边界，而不是继续让新增融合目标自由改写原三路表示。

## 3. 建议结构：原任务路径＋并行交互摘要读出

```text
同一目标人物的 Full / Person / Face
                  │
       同一组冻结ViT权重，三路Frozen Forward
                  │
       当前任务共享Image-token Adapter / Patch Selectors
                  │
        三路Summarize、Summarized Patches
             ┌────┴────────────────────┐
             │                         │ 停止新增分支反传
   原三路Task Forward                  │
   Prefix / MSA / Drop & Replace / FFN  │
             │                         │
       hFull、hPerson、hFace             │
             │                         │
       原固定融合 h0 ────────→ 交互Query │
             │                    │    │
             │           三视图联合摘要Attention
             │           目标区域约束 / Face有效mask
             │                    │
             │           内容门控、有界交互残差 δ
             └─────────── 加法 ←───┘
                         │
                     hFinal=h0+δ
                         │
               同一个本任务分类头 Ht
```

分类头的位置仍在最终融合特征之后。不是再建一套完整三视图模型，也不把三条Task Forward压成一条。交互分支不增加另一套Frozen Forward或完整Task Forward，只复用已有中间证据与任务CLS。

## 4. 三个前向模块分别解决什么

### 4.1 任务对齐、目标人物约束的摘要证据库

原三路Summarize保持原样。在第6层额外构建只供交互分支使用的证据库：

- Person/Face使用该任务原有Summarized Patches的停止梯度副本。
- Full额外做一次目标区域限定的Summarize：使用同一组任务Selectors的停止梯度副本，读取目标人物框覆盖的Frozen Forward patch tokens。首版不新增三套独立Selectors。
- 人物框必须随Full训练增强的crop、resize、flip准确变换，不能直接使用原图坐标。Full增强后目标不可见时，关闭此证据库；Person证据仍可用。
- 目标限定的Full证据库不包含全局CLS，否则可能绕过区域限制。原Full Task Forward仍保留其全局证据。
- Face不可靠时，在Attention softmax之前屏蔽整个Face证据库；门控输入也不读取该Face特征。仅置零向量不足以排除归一化、bias及视图类型编码的影响。

这一步让联合读取具有共同的任务语义锚点和可检查的人物对应条件。冻结ViT的patch token已经包含全局上下文，因此区域mask只限制读取位置，不能保证特征完全不含其他人或背景，仍须人物交换验证。

本模块设有无区域约束的配对版本，避免将过去位置摘要失败的配置直接当作已证实改进重新堆入。

### 4.2 当前人物与当前任务条件的联合摘要Attention

将三组摘要加上各自的视图类型编码，经私有归一化和K/V投影，组成联合memory。交互Query由当前任务的融合CLS与Person CLS共同条件化，全部输入对原主路停止梯度：

`q = Qφ([sg(h0), sg(hPerson)])`

`c = Attention(q, [GFull,target, GPerson, GFace], valid_mask)`

`δraw = Oφ(c)`

任务条件来自本任务CLS与本任务Selectors；人物条件来自Person内容和目标框。三视图不同patch索引不要求物理对齐，先摘要后联合访问。

首版可用单个64维交互Query、4头、单层Attention；第6层摘要的768维和最终任务CLS的512维通过各自投影进入64维空间，输出再投到512维最终特征空间。先缓存摘要，待任务CLS生成后读出，数据流无循环。具体维度需实现时核对模型配置，不作为通用硬编码。

旧结构问“Full的什么信息可以写进Person/Face”；新结构问“在已有三路判断基础上，当前人物还需要怎样的联合修正”。输出只进入最终融合，不覆盖原Summarized Patches或三路任务CLS。

MulT可借鉴的是跨流证据交互的Attention接口，而不是其时间序列问题设定；它本身不解决多人场景的目标人物归属。

### 4.3 可拒绝补充的内容门控与有界残差

可靠Face时原固定融合为`h0=.64hFull+.16hPerson+.20hFace`；不可靠Face时为`.80hFull+.20hPerson`，均沿用原特征与归一化顺序。

新输出为：

`hFinal = h0 + γφ(sg(hFull,hPerson,hFace), reliability, visibility) · Bρ(δraw; sg(h0))`

其中γ是图片内容相关的门控，Bρ限制残差RMS相对原融合特征的幅度。原三路固定权重不先行重搜，最终加法后也不额外归一化。

有效性mask解决“Face能否参加”，可学习门控解决“本图是否需要交互修正”，两者不能混为一谈。门控可以趋近零；但如果全程近零且无融合收益，应判定分支未显示价值，不能将“不伤害”当成功。

建议以输出投影零初始化、门控初值0.1开始，避免同时把输出和乘法门控置零而使分支无法起步。ρ=0.1只作首个稳定性候选，不意味着mAP得到保证。旧Attention已经有零输出初始化、LayerNorm及限幅，因此这些部件本身不是新贡献。

本方案不强制新特征与Full正交；相关的情绪证据也可能有用。是否互补必须通过预测收益及内容干预判断，不能用特征夹角代替。

## 5. 最关键的训练边界：原配方继续学，新增目标不改写主路

### 5.1 原三路共同学习完整保留

令θ表示原Image-token Adapter、Selectors、Prompts，H为本任务头，φ为新增交互参数。

- Selectors/Prompts/H保留原融合BCE＋0.1×有效视图BCE均值。
- Image-token Adapter保留原融合ASL＋0.1×有效视图ASL均值，参数组损失分工与历史代码完全一致。
- 原主路损失必须从原固定融合`H(h0)`计算，不能改成`H(hFinal)`后仅把δ detach；后者依然因最终logits变化而改变主路梯度。

### 5.2 新交互分支直接从真实标签学习

额外目标仅更新φ：

`Linteraction = BCE(Hsg(sg(h0) + δφ(sg(evidence))), current_task_labels)`

Hsg使用同一次训练中当前H的数值，仅切断额外目标对H的梯度。H仍在原目标下全30轮学习，不是从task0固定随机头，不是加载B0头，也不是训练完再换头。旧类别的mask按历史实现处理，不能产生旧类伪负标签更新。

主路和交互分支在同一批增强数据、同一30轮流程内学习。推理只有同一套当前任务参数及同一个H，无A/B来源、蒸馏答案或checkpoint拼装。训练时对同一个头做两种读出不等于训练两套头。

### 5.3 保护的范围与实现条件

额外目标不向θ/H反传，且使用独立optimizer/GradScaler、独立初始化随机流，不混用全局gradient clipping。私有Norm、bias、scale均归φ。主路初始化、数据增强、历史报告RNG消耗、Adam状态及参数冻结/继承必须与原B0相同。

满足这些条件时，原三路函数有机会逐步精确重放B0，而不是仅满足某一训练batch的局部损失约束。必须用张量、loss、RNG和输出审计验证；不能只由stop-gradient这行代码宣称保真。

额外前向的H数值应与所用证据来自同一更新时刻；先完成两种反向再更新参数，或显式保存H常量，避免in-place版本错误和隐含优化时序差异。

这保护的是本模型内部原三路函数，**不保证新增融合输出在每张图或Test总体上不降**。本方案首版也不要求原Face分路单独AP再提高；可学习能力新增在交互分支，研究指标是Person/Face对最终融合是否有条件收益。

### 5.4 增量机制

每任务有一个交互模块φt，随原任务参数结束后冻结；下一任务的原Selectors/Prompts仍按B0继承，交互结果不覆盖其初始化。旧任务所用证据与访问规则均固定。测试对所有已见任务路径计算各自类别，不做“尝试Router后取最好任务”。

首版不引入22专家、动态专家池或跨任务交互参数复用，避免重新改变旧任务依赖的输入。

## 6. 与旧方案的实质区别和不能夸大的地方

| 旧方案 | 本方案改变 | 仍需防止的问题 |
|---|---|---|
| Full摘要→Person/Face摘要残差，共享参数共训 | 独立联合证据读出；新增梯度不改原表示 | 单独换Query不构成新机制 |
| 三路独立Selectors | 保留原共享10 Selectors；只增加私有交互投影 | 失去主路任务语义共享已被验证有风险 |
| 训练好B0后额外5轮只训Attention | 从开始就定义双目标，原三路与交互共同30轮 | 旧只训Attention近乎零收益，保护本身不够 |
| 近期排序＋Adam局部保护 | 结构上隔离新增目标的主路梯度 | 不再用TRAIN一阶BCE改善推断Test排序受保护 |
| 早期B0保真logit residual校准 | 访问任务摘要和目标区域，学习内容相关交互 | 保留B0、后置残差不是新发明，必须胜过简单校准/MLP |
| 完整A/B固定均值 | 一套原任务表示＋小交互模块，自身真实标签训练 | 均值收益可能来自额外Full表示或方差降低，未必可被小模块复制 |

2026-09-25的旧logit residual控制仅seed0任务0–2 validation：45.2395→45.3251、阶段均值54.9341→55.0368；当时无Image-token Adapter且未达到预设晋级线。不能与本轮8任务Test混表比较，但它说明保护和后置校准早已存在，新方案的价值必须来自联合摘要信息与人物条件选择。

本方法仍是三视图共同学习：原共享参数继续接受原三路监督；新增交互目标受限于私有参数。它没有让所有损失自由更新所有共享参数。若最终论文要求新增Face目标直接提高原共享表征，则本首版只解决融合贡献的保护性验证，不能声称已经解决全部共享优化问题。

## 7. 建议实验顺序

首先只做可验证的最小实现，不同时加MoE、排序loss、去相关loss、教师或更多Attention层。

| 组别 | 在相同原B0之外新增什么 | 要回答的问题 |
|---|---|---|
| C0 | 无 | 精确配对的强B0 |
| C1 | 读取三路最终CLS的受保护MLP残差 | 是否只是增加后置非线性容量就够了 |
| C2 | 受保护联合摘要Attention，固定相同残差额度，无目标区域限制 | 摘要级交互相对CLS融合是否有价值 |
| C3 | C2＋目标区域限定的Full Summarize | 目标人物对应约束是否有价值 |
| C4 | C3＋内容相关残差门控 | 可选择地接受修正是否比固定额度更好 |

C1/C2尽量匹配新增参数与预算，固定门控数值和门控初始化需预先写入执行协议。另补Full-source-only交互控制：原三视图主路不变，新增分支只读Full，且不再读取Person内容作Query。这样才能区分三视图交互与再次适配Full，不能把三视图主路关闭误称此控制。

先task0检查机制与泛化，再task0–2筛选；候选通过后完整seed1八任务Validation/Test。首要目标是同时超过配对B0的Test最终32.6800与阶段平均39.5292，再比较原A及高成本均值参照。不能用task0–2的分数代替26类最终和8阶段平均。

现有Test已经参与研究，必须记录全部预声明配置及结果，不把选出的最高Test称为独立无选择验证，不按Test挑某一epoch代替完整训练结果。

## 8. 必须通过的归因和停止条件

1. **原函数保真**：关闭新增交互后，原三路及固定融合logits、任务参数、训练loss/RNG精确重放匹配C0；新分支开启也不改变原三路函数。若只在AMP实现差异范围内一致，须报告数值容差及原因，不能写成逐值相同。
2. **即时净贡献**：同一checkpoint关闭交互残差回到C0；开启有实际融合收益。因为本方案主路经审计一致，OFF比旧共训Attention的OFF更有明确基线含义。
3. **内容有用**：移除/打乱Person或可靠Face memory，比较同人群差值；无效Face像素替换不得改变最终输出。Attention权重或门控非零不是成功指标。
4. **目标人物对应**：同图多人且双方可见时，正确Person/框/Face组合应优于交换成其他人的组合；分别只交换Query、框、Face，避免一次改变全部输入后无法定位。
5. **额外计算控制**：新增参数、训练时间、task7推理时间、显存与Full-only/MLP控制一起报告。不能按Frozen ViT权重共享就宣称没有额外计算。
6. **增量保真**：新增任务后，固定TRAIN审计集上旧任务的原路径与交互路径预测均保持不变；只保护原θ、不封存φ或访问范围仍会漂移。

如果C1与C2无差别，不能将收益归为Attention；如果Full-source-only同样好，三视图机制证据不足；若门控趋零或关Face不降，不能说Face互补已被利用。若保护后仍无收益，应承认该交互接口没有证明价值，不自动升级复杂模块或扩大专家池。

## 9. 未来执行配置草案（本轮不执行）

- 数据：原EMOTIC三视图TRAIN/validation/test及原Face manifest，沿用已锁定路径，不新建划分。
- 标准：历史`samples_intersect_seen_classes`，每阶段所见类内至少一正标签的人物；8阶段算术均值；原CPU分batch sigmoid、增强/人群及mask不变。
- 原模型：冻结CLIP ViT-B/16，Image-token Adapter b32/index1/scale0.03/ReLU，10共享Selectors、原Prefix Prompts，Person颜色扰动关闭。
- 原训练：seed1，30epoch/task，batch64，Adam/WD0/按任务重置、cosine，主LR0.0125、Adapter LR0.0004，原BCE/ASL路由与aux0.1，ASL gamma负9.8/正0/clip0.05，AMP/TF32，历史72线程初始化与报告RNG。
- 新增：第6层证据；Attention宽64/4头/1层，交互LR0.0004/cosine/WD0，新增部分随机流与GradScaler独立；上文ρ0.1、门控初值0.1作为首配置，非已确认最佳值。
- 资源：后续GPU工作仅GPU0，按显存排队；原加载/评估优化设置继承，训练workers2/prefetch1，评估2个单线程spawnworkers；不为并发改变原初始化线程数。
- 产物建议：`./output/protected_joint_attention/<run_id>/`、`./logs/protected_joint_attention/<run_id>/`；训练入口尚未实现，不提供假执行命令。
- 预检：关闭分支历史重放、主/新梯度隔离、无效Face与目标不可见mask、零初始化可训练性、旧任务封存、恢复完整RNG/optimizer/GradScaler；再真实短smoke。未满足预检不启动正式训练。

## 10. 原始论文与本地依据

- MULTI-LANE：<https://arxiv.org/html/2405.15633v1#S3.SS3>。借鉴Patch Selectors、Summarize、任务专属路径及封存；三视图交互为本项目新候选。
- MulT：<https://aclanthology.org/P19-1656/>。借鉴跨流Attention读取，不声称其已解决RGB三视图目标归属。
- Flamingo：<https://arxiv.org/html/2204.14198v2#S2.SS2>。借鉴门控交互与保留原函数的设计原则；本方案原任务参数仍正常训练，不等同其冻结语言模型设置。
- GMU：<https://arxiv.org/abs/1702.01992>。借鉴输入相关门控，首版实现不是原GMU复现。
- 本地：`fixed_model_mean_results_20261027.md`、`protected_view_forward_mechanism_analysis_20261026.md`、`selector_face_controls_results_20261026.md`、`head_anchored_eight_face_results_20261025.md`、`teacher_free_shared_learning_results_20261028.md`、`teacher_free_epoch_head_results_20261028.md`、`historical_rng_warmup_results_20261027.md`、`logit_residual_control_results_20260925.md`。

本轮仅检查已有代码/结果和原始论文并记录方案，未增加训练结果，未提交/推送或合并main。
