# 从32.6800强底座继续：无外部教师的三视图共同学习方案

日期：2026-10-27。状态：研究与实施方案，未实现训练机制、未启动训练。
本地规划分支：`codex/teacher-free-shared-view-plan-20261027`；起点源码：`28a1cbb87052ecce8d95114153a736882bb70119`。
本方案中所有新增系数都是首轮预声明假设，不是已验证最优值。

## 1. 要解决的问题和方法边界

目标是在同一个模型、预先定义的训练流程中，让Full/Person/Face共享任务参数接受三路监督，保留Full的有效判别能力，并让Person/可靠Face改善最终融合的排序。正式目标是同时超过seed1强底座的Test最终mAP32.6800和8阶段平均mAP39.5292。

模型只从标准CLIP预训练权重及新建任务参数开始。没有A/B教师、外部融合答案、历史任务参数移植、训练后换头或多个训练来源拼装。每任务仍为原30轮共同训练；分类头在30轮内持续学习。常规CLIP预训练和MULTI-LANE跨任务复制Selectors/Prompts属于原配方，不等于引入A/B教师。

建议首版保留三路Task Forward、共享Adapter/Selectors/Prompts及原固定特征融合，新增“Full条件的互补排序监督”和“新增共享更新的Full/融合保护”。先研究训练机制，不增加Attention/MoE/新分类头。它是一项有待验证的假设，不能提前称为有效方法。

## 2. 现有证据约束了哪些选择

- 历史强底座已独立精确重放：Val42.4628/49.4763，Test32.6800/39.5292；8任务权重、30轮loss和全部RNG一致。先前32.1748来自任务边界报告迭代器的CPU取种子遗漏，不是需要改变评估标准。
- 同一强底座的Full单路Test31.2373，融合32.6800，说明三视图本身已有互补。该Full是同模型分路，不是独立训练单Full控制。
- 首任务额外固定头适配就出现下降；最新联合约束组task0 Val/Test损失分别1.7430/0.4272，主要保留在共享任务参数变化中。首任务没有旧任务可遗忘，不能主要归因于增量遗忘。
- 30+5、固定头、冻结B0只训Attention、独立Selectors等都没有构成满足目标的解法。参数拆分也可能损失已有对齐关系。
- 历史DGL截断融合到表征的梯度并未改善；旧MoE梯度审计也没有证明跨任务持续的三视图梯度冲突。因此不能直接宣称“PCGrad一定解决”，或把原融合监督全部切断。
- 旧OOF排序蒸馏曾使训练排序loss下降、Validation变差。其硬筛教师优势对、类别重采样、不同增强输入和把多视图优势迁移给单Full，均不能照搬。新排序假设必须与普通排序对照，仍可能过拟合。
- Face全人群AP上升不等于可靠Face证据增强；占位Face不参与融合。所有新Face监督严格限制在实际可靠Face样本。

相关记录：`docs/historical_rng_warmup_results_20261027.md`、
`docs/shared_multiview_dgl_validation.md`、
`docs/moe_v4_gradient_attribution_results_20261017.md`、
`docs/selector_face_controls_results_20261026.md`、
`docs/oof_advantage_ranking_results_20260913.md`。

## 3. 可以借鉴什么，哪些不能直接搬

| 研究 | 值得借鉴的思想 | 本项目的适配边界 |
|---|---|---|
| Adapting Auxiliary Losses Using Gradient Similarity | 辅助目标是否有益，应相对于主目标判断 | 非对称地保护Full/融合；不能从训练梯度推出Test mAP保证 |
| PCGrad | 检查和处理共享参数上不利的更新方向 | 只作为优化参考/后续对照；不预设稳定冲突，也不对三视图一律对称处理 |
| ReconBoost | 辅助能力应修正其他证据的错误，服务整体预测 | 借鉴互补目标，不引入其历史模型、KL答案或多套交替训练来源 |
| Gradient Blending | 不同信息源的泛化/过拟合速度可能不同 | 即使梯度同向仍需Val/Test检验；不能把所有失败都解释成负内积 |

原始论文：
- [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224)
- [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [ReconBoost](https://proceedings.mlr.press/v235/hua24a.html)
- [What Makes Training Multi-Modal Classification Networks Hard?](https://arxiv.org/abs/1905.12681)

这些文献支持设计思路，不等于已经发表了下面的具体算法；这里的组合也不能未经相关工作比较就宣称新颖性。

## 4. MULTI-LANE数据流：首版前向结构保持

```text
Full / Person / Face（同一个人物的三份输入）
             ↓
同一套冻结CLIP视觉权重，分别计算三路Frozen Forward
             ↓
当前任务共享Image-token Adapter → 供Selectors读取的图像证据
             ↓
当前任务共享的10个Task Selectors，分别执行三路Summarize
             ↓
三路Task Forward：
各自MSA → Drop & Replace → FFN，使用当前任务共享Prefix Prompts
             ↓
hFull / hPerson / hFace，各自在分类头之前归一化
             ↓
原固定特征加权融合（不在融合后新增L2归一化）
             ↓
原位置的一个共享分类头 → 当前任务类别预测
```

可用Face：Full/Person/Face权重0.64/0.16/0.20；
不可靠Face：0.80/0.20/0。
Image-token Adapter仍在索引[1]的Selector图像证据路径，不写回原Frozen Forward视觉流。

三路单视图logits来自同一个分类头的诊断/辅助读出，不新增三个独立分类器。测试仍执行累计任务路径，并取各任务拥有类别的输出；没有按效果选择任务路径。

新增机制只改变训练目标和共享参数更新规则，正式推理参数量和前向结构不增加。训练反向及候选更新有额外成本，需实测。

## 5. 保留的32.6800完整配方

| 项目 | 固定配置 |
|---|---|
| 数据/顺序 | EMOTIC，历史类别顺序，任务大小[5,3,3,3,3,3,3,3] |
| 首轮seed | seed1；32.6800对应seed1，不能与seed0 Full32.5365混用 |
| 初始化 | 原CLIP ViT-B/16预训练；新建任务参数与分类头；原72线程初始化环境 |
| 任务参数 | 10个共享Selectors，10个Prefix Prompts、5层；原跨任务复制与任务结束冻结 |
| Image-token Adapter | 每任务共享b32，索引[1]，ReLU，固定residual scale0.03；49,952参数/任务 |
| 训练 | 每任务30epoch，batch64，Adam每任务重置，cosine按epoch衰减，WD0 |
| 学习率 | 主参数0.0125，Adapter0.0004；原Adam默认betas/eps和实现保持 |
| 主参数原监督 | 融合BCE + 0.1×有效视图辅助BCE的均值 |
| Adapter原监督 | 融合ASL + 0.1×有效视图辅助ASL的均值 |
| ASL | gamma_neg9.8，gamma_pos0，clip0.05，原eps；BCE温度1 |
| Face | 原manifest、可靠性、占位和辅助mask/reduction代码不变；新增监督采用同一严格可靠mask |
| 增强 | 原Full/Person/Face变换；Person颜色扰动关闭；原crop scale[0.05,1.0] |
| 数值 | AMP/TF32、原GradScaler初值65536；模型OMP/MKL72 |
| 加载 | 训练workers2/prefetch1；评估2个单线程spawn worker，B64 |
| 新增结构 | Attention、位置摘要、MoE、额外分类头均关闭 |
| 评估 | 历史samples_intersect_seen_classes；8阶段mAP算术平均；Val/Test都报告 |
| 任务边界RNG | 使用已验证的历史报告取种子重放；仍inline报告的入口不能重复补种子 |

原任务学习、head旧类别屏蔽和old task冻结逻辑全部保留。新增排序只涉及当前任务类别，不向未来类别或旧类制造伪负标签；原legacy_full_zero损失函数不重写。

输入沿用已归档manifest/CLIP/data配置及SHA。首版实现时用协议文件解析已有路径，不新写数据绝对路径。
当前本地参照配置：`output/self_trained_fast_evaluation/complete_snapshot_20261027_01/snapshot/historical_base/config.json`。

## 6. 新监督：让辅助视图为融合的类别排序负责

### 6.1 排什么：同一个情绪类别，正人物与负人物

对当前batch、当前类别c，收集所有真实标签y(i,c)=1、y(j,c)=0的人物对(i,j)。希望融合分数z(i,c)>z(j,c)。

这是同一类别跨人物排序，贴近AP的评价问题；不是多标签任务中互斥类别的softmax，也不是同一个人物不同情绪之间的排序。pairwise logistic只是一种可微代理，不等价于直接优化mAP。

不新增样本loader、类别重采样、历史教师筛对或单独确定性增强。使用原B0这一批已经增强好的三视图和自然标签分布。没有正负对的类别贡献0，不合成负例。

### 6.2 Full只决定软难度，真实标签决定方向

记当前Full正负分数差为mF=zF(i,c)-zF(j,c)，使用停止梯度后的值：

```text
a(i,j,c) = 1 + sigmoid(-stop_gradient(mF) / T)
```

初版T=1，因此权重在1到2之间。Full分错或难分的对权重稍高，已经分对的对仍保留。避免旧方案那样只训练硬选“Full错、教师对”的小子集。

这里没有要求学生模仿Full概率，也不要求Full给出正确标签；Full即使错了，排序方向仍由人工标签决定。Full与三路模型一起更新，不存在冻结的外部答案。

### 6.3 训练值与原融合相同，额外梯度只从辅助路径进入

令H为当前共享分类头，sg表示只对新增目标停止梯度。对Person构造：

```text
zC_P = H_detached(wF*sg(hF) + wP*hP + wR*sg(hR))
```

对Face构造：

```text
zC_R = H_detached(wF*sg(hF) + wP*sg(hP) + wR*hR)
```

H_detached使用当前头相同的权重/bias，不建立额外头；只在新增监督中不更新H。
原B0的融合BCE、三路辅助BCE、Adapter-ASL仍按原图反传，分类头仍正常学习。

两式数值上对应原特征融合后过头的结果；新图的区别在反向路径。实现必须复用原特征融合顺序与数值精度，不能把原主前向改写成“加权logits”，避免历史AMP舍入轨迹漂移。

使用softplus(-(zC(i,c)-zC(j,c))/T)，加权a后求和。
Person使用全部有效正负对；Face只使用两端Face都可靠的对。不可靠Face相关对的Face项分子为0，不把占位图计为有效Face监督；分母统一采用下述自然对总数，使Face信号随可靠覆盖率缩小。

具体归一化：Z为当前batch自然正负对总数；每个视图均以Z归一化，Face不可靠对的分子为0；两项再除以2。这样可靠Face少时不会因为单独归一化被反向放大。Z=0时新增loss=0。统一排序控制也使用完全相同的mask和分母，只令a=1。

“停止Full边的额外梯度”不等于Full受到保护：Person/Face反向仍经过共享Adapter/Selectors/Prompts，参数更新之后Full函数仍可能改变。因此还必须做下一步更新限制。

### 6.4 TRAIN元数据可行性检查

2026-10-27已做无GPU、无图像/模型前向、0更新的元数据检查。batch64，独立NumPy模拟shuffle；不是历史训练sampler精确重放，不代表实际训练梯度或方法有效。

| 任务 | TRAIN人物数 | 批数 | 每批平均可靠Face人物 | 每批自然正负对 | 每批可靠Face正负对 | 无Face对批数 | 每批同原图人物对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| task0 | 5353 | 84 | 36.18 | 1577.68 | 612.67 | 0 | 0.095 |
| task2 | 861 | 14 | 44.07 | 2391.64 | 1186.07 | 0 | 0.500 |
| task7 | 1526 | 24 | 43.92 | 2565.50 | 1227.54 | 0 | 0.292 |

自然batch已有Face排序信号，不需要首版改变sampler。相反，同图人物很少共同出现，不能假定普通batch支持充分的同图人物对比；此方向如以后研究，必须另做同预算采样控制。

本地原始计数：`output/teacher_free_shared_plan/train_pair_feasibility_20261027.json`。
这次检查没有读取Val/Test标签用于配对选择。

## 7. 共享更新保护：以本模型此刻的原B0更新作参照

### 7.1 不是保护一个已训练B0教师

在当前模型参数θ、当前batch和当前Adam状态上，计算原B0本来要做的更新δ0，再计算加入新增监督的候选更新δα。两者共享这一个当前模型的起点和优化器状态；δ0不来自历史checkpoint，也不需要另训一个模型。

这只约束“本步新监督额外带来的变化”，不是保证改进模型在整个30轮后与历史B0轨迹一致。只要过去接受过新更新，当前θ和Adam状态便可能已不同于标准B0。

### 7.2 原B0梯度不切断，新增梯度分组限幅

记R为Selectors/Prompts等主表征参数，A为Image-token Adapter，H为当前共享头：

```text
g0_R = 原融合/辅助BCE对R的梯度
g0_A = 原融合/辅助ASL对A的梯度
g0_H = 原融合/辅助BCE对H的梯度
q_R, q_A = 新互补排序目标对R、A的梯度；q_H=0
```

对R、A分别将q范数限制为本组g0范数的0.25倍，不能将head的大范数混入主表征分母。基础信号或新增信号为0时该组不添加新更新，记录缺失，不伪造梯度。

ASL梯度可能很小，不能BCE/ASL共用一个未经尺度核对的排序系数；监控unscaled FP32梯度，沿用实际GradScaler或FP32核验，不能固定1024倍就假定无下溢。

新增权重在原30轮内开启：第1轮为0，第2–5轮线性增至1，第5轮后保持1。原head和B0监督30轮全程不变；不重置新阶段优化器，不延长5轮。

### 7.3 检查Adam实际候选步，而不是只检查原始梯度角度

从同一组Adam一阶/二阶矩和step构造候选：
α∈{1,0.5,0.25,0.125,0}，采用g0+α*lambda*q；
头始终采用原g0_H。

令u=δα-δ0，δ表示“加到θ上的参数步”。主候选要求：

```text
grad(L_Full_BCE) dot u <= 0
grad(L_fused_BCE) dot u <= 0
||u_R|| <= 0.1 ||δ0_R||
||u_A|| <= 0.1 ||δ0_A||
```

比较的是额外步在Full/融合目标上的一阶预测影响。选择满足条件的最大α；没有非零候选就采用α=0，即本步原B0更新。全局内积包含所有本次可训练坐标，不遗漏Norm/bias/scale；原old/future mask保持。头额外步为0，但不能据此推断未来头轨迹永不改变。

必须在同一Adam状态上纯计算候选，接受后只提交一次对应参数及矩状态；不能先optimizer.step再仅恢复参数，不恢复矩；GradScaler和scheduler也只能推进一次。基线α=0必须与真实原Adam更新逐值核验。

U1/U2没有方向保护时，仍使用相同梯度限幅和实际步长比例限制，避免U4仅因步子更小占优。候选反向图复用，不为每个α重新读取/增强图片。

### 7.4 保护能力和局限

- 这是本batch、一阶、相对于当前B0步的保护；不保证Full概率精确不变、不保证下一步loss下降，更不保证Val/Test mAP不降。
- Full/fusion BCE是可微保护指标；Adapter原ASL保留并监控，不能宣称所有ASL/排序目标都获得数学保证。
- Adam非线性、曲率、过拟合和后续head轨迹仍可能破坏最终功能。
- 每约100步用固定TRAIN探针对同一θ下δ0与选定δ做短暂反事实loss检查，恢复参数且不污染优化器/RNG；不引入旧模型或额外训练来源。若一阶预测安全而实际Full频繁恶化，需要报告近似失败。
- 若几乎全是α=0、Face没有有效梯度或输出回到B0，属于机制未能利用辅助信号，不算成功。
- 若guards从不约束，不能把收益归功于保护；若只靠减少剂量获益，要做TRAIN确定的等剂量控制。

## 8. 按因果问题设置消融

所有组相同模型、参数量、原30轮、head学习方式、初始化/数据流/增强及新增梯度与步长限幅。没有教师、Attention或训练后换头。

| 组别 | 新增排序 | Full软难度 | Full额外步保护 | 融合额外步保护 | 要回答的问题 |
|---|---|---|---|---|---|
| U0 强底座 | 无 | 无 | 无 | 无 | 精确重放32.6800参照 |
| U1 普通排序控制 | 有 | 无，a=1 | 无 | 无 | 普通排序/额外监督是否已经足够 |
| U2 互补排序 | 有 | 有 | 无 | 无 | Full条件是否比普通排序有用 |
| U3 Full保护 | 有 | 有 | 有 | 无 | 保Full是否避免主证据损失 |
| U4 Full与融合共同保护 | 有 | 有 | 有 | 有 | 保Full的同时，能否让补充服务融合 |

若U4出现收益，增加一个从TRAIN日志预先确定剂量的控制，排除“只是新梯度更小”。
不在首版同时改视图loss权重、去Face、换融合权重或加新Attention，否则无法定位来源。
先验证两机制，不把“论文贡献”预先等同于某个新模块。

## 9. 执行顺序、晋级与停止条件

本次仅制定方案，以下入口尚未实现、未运行。后续所有GPU实验只用GPU0，显存允许可并行，完整恢复状态在评估之前保存。

### 阶段A：实现正确性和短程代价

先实现关闭新机制时零分支开销/零额外RNG的B0路径；初始化、任务边界补种子、原更新逐值核验。
20–50步实测额外反向、候选Adam更新、吞吐和峰值显存，不能先承诺与B0等训练时间。
专项检查：旧参数/未来类别保护，Face无效对0梯度，head只接原目标，原B0融合/辅助梯度保留，候选状态不污染、AMP overflow/skip恢复完整、α=0逐值一致。
这是实际实现验收，不能用形式上镜像代码的测试代替真实前向/更新比较。

### 阶段B：先过task0

先做U0/U2/U4各自完整task0×30轮，同初始化/同输入，不采用30+5。优先task0，因为过去显著损失从首任务就出现。

完整Val/Test都报告，同时看Full、Person、可靠Face、融合及类别AP；Test已经参与本研究的既有探索，明确属于开发性结果，不称独立未参与选择的测试。

初筛原则：U4的Full和融合在两split没有明显系统下降，融合排序相对U2/U0出现信号；若Full/融合任一连续多类下降或task0损失达到既有适配量级，停止晋级，不靠最终task7掩盖早期损失。0.1mAP仅作为预声明实用差异参考，不当统计定理。
同时要求非零辅助更新被接受、可靠Face确有信号，不能“全部拒绝→等于B0”就晋级。

### 阶段C：task0–2与完整控制

通过task0的组从其完整恢复状态续到task1/2，补U1/U3从task0全程训练。
每个控制有自己的Adam/RNG/GradScaler，不从别组中途复制参数；不能拿compact checkpoint伪装完整训练恢复。
若U2不优于U1，不能称Full条件设计有效；若U4不优于U2或U3，不能称双保护有价值。优先淘汰无贡献机制，不扩超参数网格。

### 阶段D：seed1完整8任务

只将有证据的候选、匹配U0及必要控制推进完整8任务Val/Test。
正式成功条件：Test最终和8阶段平均同时超过32.6800/39.5292，并相对同批U0也成立；0.1点/项为优先值得复核的幅度参考，不能以微小单seed差异宣称稳健。Val需报告并解释与Test的差异，不只展示获益split。

各阶段Full功能应无明显系统损伤；可靠Face/Person的条件贡献需比匹配B0更强，接受率与梯度不是空转。只有此时才扩其他seed；每个seed重新配套其基线，不用seed1最高值替代seed0或挑选最好seed。

如另做TRAIN原图分组holdout，所有组都要同样排除该部分数据；其基线不能标为原全TRAIN的32.6800。正式参照保持原完整TRAIN，不在此次方案中默默减少训练数据。

## 10. 如何证明是三视图互补，而不只是Face单路变强

所有8阶段两split保存原四路logits、标签、人物ID、类别和可靠mask；逐类/按原图聚类配对bootstrap估计，不能忽略同图人物相关性。阶段均值的bootstrap要保留跨阶段相同原图的耦合。单seed仍不能估跨seed方差。

新增只读对照：
1. 将Person特征替换为同人物Full特征，保留原Person权重；将可靠Face特征替换为Full特征，保留原Face权重。保持总权重、分类头与可靠性mask，不通过置零/重新归一化引入额外尺度变化。
2. 无Face人群的“Face替换”预测必须逐值不变，是阴性控制。
3. 对同图双方都可靠Face的人物，只交换Face输入、固定Full/Person，比较原正确Face与他人Face；辅助证据应有内容/人物对应价值。此对照不是唯一成功条件，也不能代替正式精度。
4. TRAIN统计的常量Face特征与真实Face比较，固定可靠性mask，检查收益是否只来自可用性线索。
5. 统计Full错序被融合纠正与Full正确排序被融合损坏的比例；按类别归一化，与U0比较。该统计是解释，不是mAP恒等分解。

若Face单路AP升而融合不升，说明额外能力尚未转为互补；若Full下降抵消收益，说明保护不足；若Full不降但关闭辅助内容无影响，说明辅助信号未被使用；若训练更好但Val/Test更差，优先判为泛化问题，不继续仅调梯度角度。

## 11. 首版实现范围和产物约定

后续预计新增独立的互补监督/候选更新模块、机制测试和实验入口；只在runner的可选分支接入，model.py原B0前向保持。关闭模式不导入新参数初始化，不改变历史随机流。具体CLI需实现后核验，不在此提供尚不存在的可执行命令。

计划输出：
- `./output/teacher_free_shared_learning/<run>/<arm>/`：配置、初始SHA、训练日志JSON、参数/RNG恢复状态、8任务checkpoint、Val/Test预测、保护/互补诊断、速度显存。
- `./logs/teacher_free_shared_learning/<run>/`：测试、smoke、训练、评估、队列日志。
- 先提交可审阅实现，再按Git-only同步到新的远程worktree；保留主树和test-only；不覆盖任何旧结果。
- 训练结束先回传非checkpoint产物逐SHA核验，再解释结果；checkpoint保留服务器，不删除来源。

本次实际只完成文献/源码研究、TRAIN元数据可行性计数、此方案与上下文更新；无训练业务代码改动、无GPU前向、无训练更新、未提交/推送/合并。

## 12. 当前辅助监督与拟议机制的数据流辨析（2026-10-27补充）

MULTI-LANE原论文§3.3的分类监督位于Task Forward最终CLS之后，利用当前任务类别的BCE学习Patch Selectors/Prefix Prompts及分类器；原论文没有我们当前Full/Person/Face三路辅助损失或Adapter-ASL。原文：https://arxiv.org/html/2405.15633v1#S3.SS3 。

当前强B0先分别获得三路Task Forward的最终归一化CLS特征，再固定特征融合、应用共享头；同一头也分别读取三路特征，产生辅助预测。以H_t表示共享线性头中当前任务类别的行，而非新建三份分类器。每层Summarize读取对应层Frozen Forward证据，与该层MSA/Drop & Replace/FFN配合；示意图中的Summarize→Task Forward不表示摘要只在网络入口计算一次。

现有辅助监督是同一人物的真实当前任务标签分别监督Full/Person/可靠Face的单路预测。主参数目标为融合BCE+.1×mean(有效视图BCE)，Adapter目标为融合ASL+.1×mean(有效视图ASL)。可靠Face先在本batch可靠子集内平均，再与Full/Person的batch平均等权进入视图均值；不是按可靠Face人数占比自动缩小原Face辅助项，没有可靠Face时该视图从均值中省略。.1是辅助均值的整体权重，不是三个分路各.1相加。

代码核对：runner.py的compute_view_objective_losses、supervised_bce_loss/asl_loss及backward_routed_training_losses；model.py的_lane_logits_with_views和_lane_block/_task_attention_ffn。原B0中共享S/P与当前分类头接BCE；共享Image-token Adapter接ASL；Frozen ViT不更新，旧任务参数及head旧类别按原规则保护。冻结Task Forward使用的ViT权重不等于禁止经过这些固定算子的梯度传回当前可训练tokens。

拟议机制保留上述全部原监督和路由。新增排序以真实标签规定同类别正负人物的正确次序，用当前Full停止梯度后的分数差作软难度权重；计算值是原融合值，但额外反向只经过Person或可靠Face的特征边，分类头在额外项中detach、在原B0项中照常学习。Full不是教师，也不是被固定的目标概率。

这并非原来没有融合监督、现在才引入融合监督；改动是新增排序目标、更有针对性的样本权重与额外反向责任分配。额外loss不直接走Full特征边，仍可能经共享S/P/Adapter改变Full，所以另用同一当前模型/Adam状态的原配方候选步与新增候选步比较。初版保护的是新增机制相对当前原配方步的额外一阶损害，不重新路由B0原有三路辅助项，也不保证原辅助梯度或最终Val/Test AP从此无害。目标是以最小改动检验互补学习和共享更新保护是否值得继续。


