# 从Full＋Image-token Adapter＋Adapter-ASL迁移到三视图：消融实验计划

2026-10-23。状态：仅规划，未修改业务代码、未新建执行分支、未提交推送、未运行任何训练、模型前向或远程作业。当前文档位于既有实验工作树，实施时另建实验分支，保护所有现有工作树和未提交改动。

## 1. 本次要检验的命题

用户的目标是保留已经有效的Full＋Image-token Adapter＋Adapter-ASL，将同一训练思想迁到Full／Person／Face三视图，然后检验摘要Attention、目标人物位置摘要是否还带来额外收益。它不是再做上一轮Adapter开关的八组实验，也不是把整个模型改用ASL。

本轮按三个层次回答：

1. 在相同TRAIN、初始化、训练预算、评估人群下，强Full训练配方是否在新执行路径中成立？
2. 将Full单视图改为共享参数三视图，在BCE及Adapter-ASL两种监督下，结果如何？Adapter-ASL能否改善三视图迁移，而不是只改善一个较弱对照？
3. 在固定Adapter-ASL三视图底座上，Attention、位置及其组合是否具有整体增益和推理条件贡献？

历史32.5365继续保留为外部强参照；新实验中要用同期单Full强对照做因果比较，不用不同协议的历史分数替代。ASL用于多标签不均衡的已有训练机制，但其在三视图上的收益仍待验证，不预先承诺提升。

## 2. 首版推荐的结构：保持三条Task Forward，Adapter必开

```text
                         相同冻结CLIP视觉权重
                  ┌──────────┼──────────┐
                 Full       Person      Face
                  │            │          │
      各层Frozen Forward原始图像tokens，保持不被Adapter写回
                  │            │          │
      第2层LN后的图像tokens，经过同一个本任务Image-token Adapter
                  │            │          │
      同一组本任务Patch Selectors分别执行Summarize
                  │            │          │
      [位置组：仅Full选择softmax加入目标框几何项]
                  │            │          │
           Full摘要 ──[第6层Attention]──→ Person摘要
                      任务Selectors作Q
                      Full摘要作K/V
                  │            │          │
           各自原任务MSA → Drop & Replace → FFN → 任务CLS
                  └──────────┼──────────┘
              原固定可靠性加权特征融合 → 原共享分类头
                             │
                   融合预测同时计算BCE与ASL
              BCE梯度→Selectors／Prompt／head及Attention／位置
              ASL梯度→Image-token Adapter
```

原MULTI-LANE术语边界：Frozen Forward提供图像tokens；Patch Selectors／Summarize读取任务证据；Task Forward完成任务注意力、Drop & Replace、FFN和CLS更新。Adapter只修改供Selector读取的证据，不改变Frozen Forward的后续原始图像流。保持三条Task Forward及末端分类位置，不做联合摘要单路径，不加MoE、ParaX、Router、专家冻结或跨任务共享。

**三视图共享的含义：**同一任务的Full／Person／Face共用一套Image-token Adapter、Selectors、Prompt和分类头；三份图像输入仍产生三条不同特征路径。不是三个独立训练模型的预测集成，也不是三路分别增加一套Adapter。每任务Adapter为49,952参数，三视图共享时不乘3。任务结束后新增模块和原任务依赖整体冻结；推理仍运行所有已见task lanes并拼接对应类别logits，不按真实任务或最高置信度挑一条路。

## 3. Adapter-ASL必须按参数组定义清楚

设θ_A为本任务Image-token Adapter参数，θ_M为本任务Selectors／Prompt／head及本组新增Attention、位置参数，z为融合预测。本轮主矩阵：

    L_M = BCE(z, y)
    L_A = ASL(z, y; gamma_neg=9.8, gamma_pos=0, clip=0.05)
    g_M = ∂L_M / ∂θ_M
    g_A = ∂L_A / ∂θ_A

BCE／ASL经过同一三视图融合计算图，但只把相应梯度写入指定参数组。**不是对BCE＋ASL求一次普通backward；不是所有新增模块都按“Adapter”名字归入ASL组；也不是对ASL分支detach中间Full或Person特征。**

Image-token Adapter在三个视图上的梯度通过共享参数累积；融合时Face不可靠的样本，Face权重为0，不应对本次主损失形成有效Face路径梯度。仍不能保证三视图都获得同等训练：原可靠Face权重Full/Person/Face=.64/.16/.20，不可靠为.80/.20/0，首版不同时更改这些权重。

BCE对照中θ_M和θ_A均由同一融合BCE更新。ASL开启时只改变θ_A的训练目标，θ_M仍受BCE监督；因此可以解释监督差异。Attention和位置使用BCE，LR4e−4；它们参与前向并影响ASL对θ_A的导数，但不接收ASL自己的参数更新。

主矩阵先设**辅助视图损失权重0**：

- 旧Full配方没有三视图辅助项，先做最小迁移。
- 三视图先分别形成任务表示，融合预测提供BCE/ASL两种分组监督。
- 不能把融合ASL偷换成三个独立视图ASL；后者是另一种监督假设。
- 若后续测试aux.1，必须单列：L_M=BCE_fused+.1×mean(BCE_available_views)，L_A=ASL_fused+.1×mean(ASL_available_views)。可靠Face才参与，视图loss先各自平均再对可用视图平均，不按三路直接求和导致监督量翻倍。

采用aux0不代表已证明它优于aux.1；它只是保留原Full监督思想、避免本轮同时更改多个因素。新aux0结果不能直接与上一轮aux.1、90%fit结果归因比较。

训练标签继续使用原增量legacy_full_zero处理和原任务样本筛选，不把未来任务标签加入损失。ASL focal weight detach、eps1e−8及对26维loss view的reduction按历史实现保留，不能顺便改为current-only均值或重新缩放。

## 4. 七组核心配置：先验证迁移，再消融两个模块

所有七组均开启相同Image-token Adapter，seed0、完整8任务、全TRAIN、相同更新和固定评估人群，aux0。S/T名称仅供配置管理，汇报表用中文。

| 编号 | 配置 | θ_M监督 | θ_A监督 | 摘要Attention | 人物位置 | 回答的问题 |
|---|---|---|---|---|---|---|
| S-BCE | 单Full＋Adapter，BCE控制 | BCE | BCE | 关 | 关 | 单Full取消Adapter-ASL后怎样 |
| S-ASL | 单Full＋Adapter-ASL，强匹配对照 | BCE | ASL | 关 | 关 | 原强Full配方在本次统一执行协议下的表现 |
| T-BCE | 三视图＋共享Adapter，BCE控制 | BCE | BCE | 关 | 关 | 仅把单Full改为共享三视图怎样 |
| T-ASL | 三视图＋共享Adapter-ASL，主底座 | BCE | ASL | 关 | 关 | Adapter-ASL是否改善三视图迁移 |
| T-ASL-Attn | 主底座＋摘要Attention | BCE | ASL | 开 | 关 | 改善图像证据后跨视图摘要交互是否再增益 |
| T-ASL-Pos | 主底座＋目标人物位置摘要 | BCE | ASL | 关 | 开 | 保留强底座后目标位置读取是否再增益 |
| T-ASL-Both | 主底座＋Attention＋人物位置 | BCE | ASL | 开 | 开 | 两种读取／交互机制能否在强底座上叠加 |

前四组构成“视图结构×Adapter监督”四格，后三组与T-ASL构成“Attention×位置”四格。共七组，T-ASL只训练一次复用，不重复计算。全部独立从同一预训练CLIP与配对初始化出发，不把T-ASL训练完再给另三组追加30epoch。

关键差值：

- 单视图ASL增量=S-ASL−S-BCE；三视图ASL增量=T-ASL−T-BCE。
- BCE下三视图迁移=T-BCE−S-BCE；ASL下迁移=T-ASL−S-ASL。
- 两者差值 `[T-ASL−T-BCE]−[S-ASL−S-BCE]` 描述ASL与视图迁移的交互；它不是梯度机制证明。
- Attention净训练增量=T-ASL-Attn−T-ASL；位置同理。
- 组合必须比较T-ASL-Both与两个单模块组，不能只看它高于T-ASL。
- 注意力/位置的二阶交互=`Both−Attn−Pos+Base`，与组合是否超越最好的子组分别报告。

首版不额外加“无Adapter三视图”作为主锚点，因为用户已明确要固定保留它；此前无Adapter结果只作背景。若需要回答Adapter结构本身是否有益，应另设相同新协议的no-Adapter组，不能用旧F000代替；它不属于本轮七组必要问题。

## 5. 历史32.5365如何保留，避免又换了一个基线

必须区分**历史严格复现**和**新矩阵中配对训练**。旧代码使用全局DataLoader随机流；最近矩阵使用按任务独立sampler、额外输入与统一张量路径。仅写seed0并不能保证逐更新轨迹一样。

启动前先用旧commit/config的初始化、首批Full增强、梯度和若干更新作数值回放核验；为新七组匹配公共初始化、样本顺序和Full增强。三视图额外Person/Face变换应有隔离随机流，不能改变Full增强或下一批样本的随机状态。

历史原始记录不保存checkpoint，预计无法直接给全部早期阶段补完整人群预测。如果找不到可复用八阶段权重，建议另排一组**R：历史单Full严格复现**，用于保存八阶段checkpoint、复现旧口径32.5365/39.1445并重新计算固定完整人群validation/test。这是可追溯性运行，不与S-ASL混成同一配置，不因复现得分换采样种子。

因此执行预算为：**七组核心新配置；必要时加一组历史复现，合计八组。**实施前检查归档决定R是否可由已有权重只读重放替代。若旧完整权重不可得且不做R，必须明确S-ASL是保留旧数学配方的同批新对照，不能承诺它数值精确等于32.5365。

R不作为新矩阵的ASL因果差值来源；单/三视图比较仍用同一新执行协议下S-ASL。若R无法复现旧行为，或新S-ASL明显弱于R，应先检查执行路径、初始化、增强与数值设置，不以弱S-ASL证明新融合成功。

## 6. 三个模块的超参数均不重新搜索

| 部件 | 固定设置 |
|---|---|
| Adapter | 第2层，zero-based[1]；768→32→768，ReLU，固定残差.1；Up零初始化、bias按旧实现；每任务独立、三视图共享；LR4e−4，WD0 |
| 摘要Attention | 第6层，index5；task_full：任务Patch Selectors Q，Full10个摘要K/V；4头，总内部宽64；输出零初始化；残差.1；LR4e−4 |
| 目标人物位置 | target模式；真实Full裁剪/翻转后未截断目标框＋可见性；6维几何φ(dx,dy,absdx,absdy,overlap,const)；visibility×tanh(w·φ)在Full softmax前加入；每任务10×6参数，12层共享；零初始化、LR4e−4 |
| 模块依赖 | 位置项进入Adapter使用的同一次Summarize softmax；Attention读取本次前向、当前位置状态下的Full摘要，禁止旧缓存或额外关闭位置的源 |

参数增量：底座49,952/任务，+Attention为246,560/任务，+位置为50,012/任务，两个都加246,620/任务。三视图共享不会把Adapter数乘3。不会顺便修改人物Query、位置层数、ASL γ、温度、残差scale或融合权重。

机制边界要保留：任务Selectors Q并不直接核验Person外观；位置项是几何条件，不等于人物身份绑定；压缩后的摘要交互可能带来有效补充，也可能重复Full证据。即使结果提高，也只按对应实验支持的程度解释。

## 7. 统一数据、训练与评估配置

| 项目 | 推荐锁定配置 |
|---|---|
| 数据 | EMOTIC，原8任务顺序5/3/3/3/3/3/3/3，26类，seed0 |
| TRAIN | 全TRAIN，保留原任务样本筛选；calibration_fraction0，禁止某一组偷偷扣10% |
| 为何不用90/10 | 本轮所有模块和超参预先固定，没有训练内路由拟合；官方validation足以承担预先规定的结构比较。若后续拟合门控/校准或调超参，应另定全组统一TRAIN原图holdout协议，不根据test选择 |
| 预算 | 每任务30epoch，每组240epoch；B64/drop_lastFalse；按旧数据预期每组13,950次更新，实施时核对manifest和loader长度 |
| 优化 | Adam，每任务重置；主LR.0125、Adapter/Attention/位置4e−4；WD0；cosine到0，无warmup；梯度裁剪维持旧默认关闭 |
| 数值 | AMP/TF32开启；新七组统一初始scale1024、growth interval1e9，沿近期稳定执行配置。历史R保留旧GradScaler设置并补录。数值配置差异必须记录，S-ASL不冒充严格历史回放 |
| 初始化/随机性 | 公共参数配对，附加模块构造不推进公共RNG；固定task sampler及增强协议；新七组均走同一归一化/张量收缩路径 |
| 原结构 | Selectors10、Prompt10、5个prompt层、pre-head归一化、共享分类头和原冻结策略保留 |
| 视图变换 | Full legacy crop scale[.05,1]、CLIP归一化；Person margin.15/letterbox/no color jitter；原Face manifest及可靠性；缺Face不删除该训练样本 |
| 固定融合 | 可靠Face Full/Person/Face=.64/.16/.20；不可靠=.8/.2/0；相同feature融合函数，不混用历史独立模型的概率融合 |
| 监督 | 主矩阵aux0；legacy_full_zero、温度1、系数1；ASL9.8/0/.05、eps1e−8、detached focal weight；按参数组路由 |
| DataLoader | train/eval workers0，保持B64；为共享内存安全不恢复高预取 |
| 评估 | 每个task边界固定validation2397人、test5368人，累计已见类；保留无已见正类的人物作为负例；最终26类 |
| 权重 | 每任务固定epoch30结束权重；先原子保存可恢复状态再做评估，不选best-test/best-epoch |
| 推理 | 全部已见任务路径，对应类别logits拼接；不给模型真实任务标签选择最优lane |

历史均值39.1445不能直接填到新表的固定人群阶段平均栏。R若有八阶段权重，按新固定人群重新评估；旧口径仅另列追溯。train fit人数及每任务更新由实际数据核验，预期全TRAIN人数5353/4394/861/9931/4352/2536/627/1526，对应每epoch84/69/14/156/68/40/10/24次更新。

## 8. 运行顺序及资源预算

1. 实施前独立分支和worktree核验，保护所有来源实验，完成下面的梯度/配对/恢复预检。
2. 必要的历史R回放或可复用权重确认，与新S-ASL行为核验；不把精确复现要求转嫁给不同RNG的新协议。
3. 先运行S-BCE/S-ASL/T-BCE/T-ASL，得到监督与视图迁移的四格。完整任务结果优先，task0或task2短期排名不作为删除后续配置的依据。
4. T-ASL-Attn/T-ASL-Pos/T-ASL-Both按同预算独立训练，补齐完整四格；即使两个单模块无收益也保留组合，以便判别交互。异常数值或实现验证不通过时暂停排程修复，不能让失败组缺失后报告残余赢家。
5. 完成预声明七组固定人群validation；按最终validation主要、阶段平均辅助的预声明标准解释候选。所有七组完整test只做锁定配置的一次评估，不根据test换参数或筛掉失败配置。
6. 对新三视图模块组的已有checkpoint做预声明validation运行时关闭归因。

本计划不执行以上步骤。继续执行时默认沿用户最近约束使用GPU0。可以在内存允许时并行训练、并行评估或自动分轮，具体并发必须用B64最重ASL组合实测；ASL分组autograd保留图，不能把旧BCE的约2.6GiB峰值当作新成本。初始并发不超过4，显存不足自动等待/分轮、不改batch、不杀其他进程。若完整validation/test安排并行，必须已锁定结构和评价规则，且不让test结果反向控制训练队列。

七组预期97,650更新；若加R共111,600更新。不预估未经实测的小时数。八任务每组保存8份compact和8份恢复状态，实施前核算磁盘；冻结视觉可由SHA引用，不重复保存相同CLIP权重，但恢复语义必须验证。日志./logs/strong_full_asl_three_view/<run_id>/，结果./output/strong_full_asl_three_view/<run_id>/。

## 9. 必须补的实现支持（本轮尚未改）

已经存在的backward_routed_training_losses可以将两种目标精确分配给model/Adapter两组，模型的Attention和位置参数当前归入fusion/base参数，理论上适合保留BCE。但最近module_stacking硬编码joint_bce、calibration.1、aux.1和fixed_three_view，其他cross_view及paired检查也限制BCE；**不是只把命令行loss-routing改为adapter_asl就能执行**。

建议新增独立的strong_full_asl_three_view协议和验证器，保留旧八组配置的锁定规则：

- 单Full/三视图均支持，Adapter固定开启，loss-routing可选joint_bce或adapter_asl；全TRAIN/aux0。
- 仅允许预声明的task_full及target组合，模型数据流沿现有组合实现，不解禁任意MoE/ParaX。
- 显式核对model参数组含Selectors/Prompt/head/Attention/位置，Adapter组仅Image-token Adapter；集合不重叠、不遗漏。
- full val/test、compact载入、完整恢复和开关归因识别新的监督与单/三视图配置。
- 训练完整性和汇总器支持七组，保留原图/样本/标签/CLIP SHA与训练预算，固定人群评估不由reporting_split过滤出另一批人。
- 控制器分别管理历史R和新矩阵身份，禁止用同一个“基线”标签覆盖。

拟触及runner.py的协议校验/元数据/分组监督校验、独立协议与控制器、评估/恢复配置读取及相关测试；模型模块数学首版不改。实施时先说明实际文件再动代码，不修改已经完成批次的源码快照。

## 10. 预检：先确认ASL确实更新了正确参数

- 用同一固定TRAIN batch比较显式autograd参考：model组等于BCE导数，Adapter组等于ASL导数；改ASL γ不应改变同次更新的model组梯度，改变BCE项不应改变Adapter目标梯度。后续训练状态会间接耦合，不能把单步分工说成完全隔离学习。
- BCE控制确保两组都只接收BCE，禁止隐藏的额外ASL或aux项。Attention/位置参与前向但其参数不接收ASL更新；梯度集合无遗漏/重复。
- 同视图数量的BCE/ASL模型初始输出相同；三视图base/+Attn/+Pos/+Both在零初始化时输出一致。单Full与三视图融合输出本来可不同，不要求二者全logits相同；只核对同一Full输入下公共Full特征/参数初始化。
- Full输入增强、样本顺序、共同模块初始化在配对组一致；Person/Face加载、模型附加模块构造或额外评估不推进公共训练RNG。
- 检验位置影响进入Adapter同一次softmax，组合Attention使用当前Full源；没有空间模块接入后绕过Attention的旧分支问题。
- 无效Face不污染融合和损失；不能通过丢弃无Face人物提高表观mAP。
- 旧任务/未来任务依赖保持冻结，任务内活动参数无遗漏；非有限loss/grad/skip均显式失败或标注，不默默降预算。
- 八组代表设置（七组＋R如有）的真实CLIP B64短跑；单元梯度测试用FP32/TF32-off隔离舍入，正式仍按锁定数值配置。
- 任务边界完整恢复继承optimizer/scheduler/scaler/RNG/sampler，并与不中断轨迹对比预测；不能只恢复compact后继续训练冒充原轨迹。

不要求零初始化第一步所有Attention内部投影梯度都非零：输出投影零初始化时Q/K/V首步零梯度可由链式法则产生，应观察后续更新而非误判未接入。

## 11. 随训练自动保留的机制证据

每组保存8阶段融合和Full/Person/Face端点scores、sample IDs、targets、Face可靠性、原图ID、参数SHA；同一共享head端点不当作重新训练的单视图基线。输出validation/test最终和8阶段平均，附cF1/oF1；固定人群的同类AP变化与新增类别造成的mAP变化分开解释。

训练日志最少包括：BCE/ASL各自loss；每组参数grad norm；Adapter在三视图的残差能量比；Attention的L2残差比/夹角/熵；位置bias/KL/饱和率/目标重叠注意力；各视图可用率；成功更新与skip；allocated/reserved显存。不同定义的Adapter能量比和Attention L2比不能混用。

可以在task0/2/6/7固定TRAIN batch额外记录分路径梯度作诊断，需隔离RNG且不额外更新参数；不做PCGrad/梯度均衡等训练改动。BCE与ASL数值尺度不同，不能拿loss变小直接判断学习更好。

重点关注：Full是否被三视图共享训练削弱；融合相对Full是否增加；Person是否提供有用条件贡献；可靠/不可靠Face、多人物标注原图子集；task6的Sadness/Sensitivity/Suffering。多人物子集提升不等于已验证目标绑定。

## 12. 预声明的validation推理归因

固定task0/2/7边界，在对应模型存档上做：

| 模型 | 对照 |
|---|---|
| T-ASL底座 | 全开 vs 关Adapter残差（确认其条件作用，不当作无Adapter重训） |
| T-ASL-Attn | 全开 vs 关Attention |
| T-ASL-Pos | 全开 vs 关位置 |
| T-ASL-Both | 全开 vs 关Attention／关位置／两项全关 |

正常全开直接复用正式validation。归因固定完整人群，按融合/端点/类别/原图配对bootstrap报告；不使用test选择开关或温度。关模块后仍保留共同训练的Selectors/Prompt/head，不能当作另一个重训配置。

如果+模块整体提高但关闭后不降，收益可能来自共同训练，不能包装成推理中有效的跨视图选择。若关模块有损失、但整体仍不如S-ASL，也只能说它对一个三视图模型有条件作用，不能证明迁移整体值得。

## 13. 结果如何判定，后续如何收敛

- 首先看T-ASL与同期S-ASL；只超过T-BCE、不超过强Full时，结论是恢复训练配方的一部分损失，不是三视图已胜出。
- 新模块分别对T-ASL比较，并检查最佳组合对两个子组；最终validation为主，完整阶段平均与任务/类别代价共同报告，不用test排名决定继续哪个配置。
- 固定seed0下按原图为簇1000次配对bootstrap，仅报告样本不确定性，不能代替训练seed复现；探索多组比较应注明无多重校正，不能用一条区间替代整个机制论证。历史ASL超参及当前研究方向已受既有test观察影响，本轮锁定后只评估一次并不能使同一test重新成为从未见过的确认集，结果仍需按探索性研究说明。
- 不预设一个微小数值提高就成功；同时要求训练整体收益和开关条件贡献有支持，视图互补/人物绑定主张另需对应证据。最终mAP提高但阶段平均明显下降时如实报告取舍。
- 七组预声明矩阵应完整执行，不能因T-ASL早期不佳取消另三组。完成本轮后，若T-ASL及最好的预声明模块组合仍明显不如S-ASL，停止继续堆模块，下一阶段才选择共享训练的独立对照：三视图独立任务参数/独立模型固定融合，或保留Full参数不动再训练补充分支。它们会改变参数量、融合层级或优化方式，需单独报告成本，不属于当前四格能唯一归因的“共享参数效应”。
- 若T-ASL成立但+Attention/+位置无净收益，保留强三视图底座，不继续温度/scale/层数大网格。
- 若新增模块有整体及条件贡献，才锁定候选考虑seed1/2及相应控制；本轮seed0不自动扩种子。
- 只有确有独立理由时再加aux.1对照；不把它只加给胜者。若研究aux与Attention/位置交互，应补完整监督匹配控制，不拿不同aux结果相减。

本轮不默认冻结原Full再训练其他路。原Full＋Adapter-ASL提供的是数学配方强参照，三视图版仍联合更新共享任务参数，因此它不保证原Full预测逐值不变。若“绝不削弱原Full”成为硬要求，必须另立保护式训练结构，不能从分组BCE/ASL推出这种保证。

## 14. 提案产物与执行前确认范围

本次已形成该方案并同步三个上下文；没有实现训练入口或更改模型。后续执行入口建议新增`bash scripts/emotic/run_strong_full_asl_three_view_ablation.sh`（当前尚不存在），参数/日志/输出完整声明后再启动。实施使用新实验分支、Git-only同步独立远程worktree，保留原primary/test-only和所有现有实验；完整产物同步后SHA核验。

预计正式表为七组各validation/test最终和阶段平均，历史R单列；附同checkpoint关闭归因、分支/任务/子集、参数/显存与完整恢复证据。历史四指标32.5365/39.1445/32.5756/49.3827按原含义保留，禁止把cF1/oF1放入validation列。
