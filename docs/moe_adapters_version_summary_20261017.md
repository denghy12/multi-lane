# 用 MoE-Adapters 术语理解 Task-Forward v1–v4

依据：各版本锁定配置、结果报告，以及论文
[Section 3.3](https://arxiv.org/html/2403.11549v2#S3.SS3)。本小结未新增实验。

## 共同设计及论文对应

沿用论文的 Adapters as Experts、task-specific Router、Top-K gating 和 gated
summation：以小型瓶颈Adapter作为专家，由Router决定专家选择及混合权重。
专家“处理什么”与Router“根据什么选专家”是两件独立的设计问题。

v1–v4均独立于ParaX，冻结CLIP视觉权重及Frozen Forward图像流。在视觉块索引1
的Task Forward内，MSA、Drop & Replace及该分支残差相加之后，将MoE残差与FFN并联。
每任务独立专家池，同任务三视图共享专家权重、各自使用Router；任务结束整池冻结，
新任务不访问旧池。v4改为证据到CLS的受限接口，已不是逐token MoE同形输出。

## 版本演进

| 版本 | 改动 | 为什么改 | 结果指向 |
|---|---|---|---|
| v1：直接迁移稀疏路由 | 每任务22专家，瓶颈64；raw task CLS驱动三视图独立Router；训练noisy Top-2、推理确定性Top-2；专家逐token处理原task-lane状态 | 先验证Adapter专家与task-specific routing能否在该位置建立三视图共享 | 各任务/视图推理固定选择同一专家对，Top-1近乎占满；同池不等于实际共享 |
| v2：证据路由与强制共享 | Router改读Selector聚合的归一化图片证据；缩为1共享+6专门；共享必经，另hard Top-1专门，0.5/0.5混合；退火噪声、直通估计、轻量负载均衡 | 修复共同CLS成分主导的路由坍缩，并结构性保证三视图共享 | 路由能随图片变化，但专门输出明显弱于共享输出，作用不足 |
| v3：小池与soft Top-2 | 缩为1共享+3专门；3专门中取2，选中权重softmax归一化；取消路由噪声和直通估计；重配固定/动态控制的活跃专家数、系数及scale | 让每名专门专家得到更多样本，训练/推理采用相同门控规则，隔离动态选择的独立价值 | 专门输出变强，但至少99.9821%能量集中在恢复后的Selector位置；专家没有直接读取图片证据 |
| v4：图片证据专家与CLS残差 | 保留4专家和soft Top-2；专家也读本块归一化图片证据；输出只加到task CLS，Selector位置残差为0 | 使专家实际变换图片内容，避免主要做图片条件化的Selector改写 | 专家内容变化、幅度和局部路由作用均存在，整体泛化仍未超过B0/控制 |

v2/v3的归一化图片证据只进入Router，专家仍处理原task-lane token；这正是v4
与前两版的核心区别。v4同时改变专家输入与输出范围，现有结果不能分离两项改动的因果效果。

v3/v4可概括为：

`delta = 0.1 × [0.5 E_shared(input) + 0.5 sum_{i in Top2(R_view(e))} g_i E_i(input)]`

v3的input为原task-lane状态；v4为图片证据e，并将delta仅写入CLS。
shared-only保留0.5共享，view-fixed与dynamic均激活共享加两名专门专家，区别是
专门组合固定还是由图片内容决定；不是将三路视图融合权重改成MoE门控。

## 论文原法与本项目迁移的边界

论文的incremental activate–freeze冻结旧任务高频激活专家，并允许新Router复用
历史专家，同时训练其余专家，目标含inter-task collaboration。我们采用每任务独立池
及整池冻结，未实现该跨任务复用。这里的强制shared expert是跨视图共享的新增约束，
不能等同于论文中跨任务协作形成的共享专家角色。

本项目还保留MULTI-LANE多标签分类、全部已见任务路径推理；没有迁移论文的全层
视觉/文本双塔MoE与DDAS分布选择。因此属于MoE-Adapters启发的三视图适配，
不能将本项目结果称为原论文完整方法的复现结果。

## 最后的梯度归因不是v5

既有checkpoint上的96快照诊断无参数更新。共享Expert及主参数未达到预声明的
跨任务稳定冲突门槛；task7的Prompt有局部Full/Face冲突，但不足以支持全面改梯度。
因此当前保留B0，停止该插入位置路线；否定范围限于上述迁移组合。

更完整结果：[v4报告](moe_task_ffn_v4_results_analysis_20261017.md)、
[梯度归因](moe_v4_gradient_attribution_results_20261017.md)。

## 2026-10-18：v1与论文的路由层次澄清

论文的task-specific Router并不等于每任务固定gate：任务索引t决定Router参数R_t，
当前图像CLS决定G_t(x)=softmax(TopK(R_t(c(x))))，因此同任务图片也可不同路由。
v1是在此基础上增加已知视图条件，使用R_(t,v)，不是以view直接删除/替代task维度。

MULTI-LANE Task Forward已有任务专属的Selector/Prompt/CLS路径，并在推理时遍历
全部已见lane；这提供任务适配和隔离，并非自动完成跨任务专家复用或内容相关专家选择。
迁移目标是任务内、跨Full/Person/Face共享专家权重，并根据各视图图片内容组合专家。
视图身份已知，不学习识别输入属于哪个视图；专家gate也不等于最终视图融合权重。

v1准确接口（已核对model._lane_block与TaskForwardMoEBank）：
1. 冻结图像tokens供Selector执行Summarize；task CLS与summarized patches经带Prompt的MSA。
2. Drop & Replace保留CLS更新、恢复Selectors，再与原lane residual相加，得到u。
3. R_(t,v)读取u的第0个token，即post-Drop-&-Replace task CLS；训练加入可学习尺度
   的Gaussian noise，取22专家中的Top-2、选中分数softmax；推理关闭路由noise。
4. 两名Adapter均逐token处理整个u（task CLS及恢复后的Selectors），同一张图的gate
   对所有token共用。每专家768→64→768，ReLU/dropout0.1/up零初始化。
5. 输出u+FFN(LN2(u))+0.1*sum_i g_(t,v,i)(x_v)*E_(t,i)(u)。仅视觉块索引1插入，
   不写回Frozen Forward；后续Task Forward照常运行，最后task CLS用于原多标签预测。
6. 每任务预分配22专家、3套gate及noise矩阵；同task跨视图共享专家，不强制共同调用
   某名专家。旧任务整池冻结，新任务不访问；不同于论文共享专家池的activate–freeze复用。

结构允许不同图片选不同专家，不意味着必须让每两张图片都不同；v1实测每task/view
确定性Top-2固定且Top-1近乎满权，这是学到的路由坍缩，不是代码按view写死专家。
本次仅解释与文档更新，未改业务代码或运行新实验。

## 2026-10-18：冻结、DDAS与三视图样本身份

当前v1–v4未实现论文按频率冻结最常用两专家的策略；旧任务整池（专家、Router、
noise等）冻结，新任务只训练自己的池。v1路由计数用于诊断，不驱动冻结名单。

论文MTIL的DDAS推断任务/分布并选择Router或原CLIP；当前不使用DDAS。Full/Person/
Face由数据构建时显式命名，image_view静态选择R_(t,v)，但其Top-K权重依然由当前
图片CLS产生。推理遍历全部已见任务lane，不通过已知测试标签指定唯一真实任务。

同一个人物样本的三视图是不同输入tensor，分别通过共享权重的冻结视觉编码器；
编码器单次前向不读取sample_id、不接收另外两个视图，也不自动识别其同源关系。
三视图在数据字典/批次位置、共同target及最终融合中保持配对。当前没有显式跨视图
对比学习/身份一致性约束；联合fusion与辅助监督、共享Expert形成学习耦合。
特征不同属于预期，既不意味着拆成三个不相关标签样本，也不保证Top-K不同。
目标允许三视图调用相同专家以共享，也允许因内容不同选择不同专家；不以强制完全
不重叠或任意两图路由不同作为成功标准。文档澄清不涉及结构变更/新实验。

## 2026-10-18：测试原图如何进入视图Router

“固定视图划分”应表述为固定视图生成规则，不是将train/test文件分别分到三个子集。
已核对evaluate_moe_task_ffn调用evaluate_joint26_checkpoint.make_source：test同样使用
ThreeViewTransform(train=False)，在getitem内从原图生成Full、Person、Face输入字典。
model.encode_lanes_with_views逐视图传image_view，分别调用R_(t,full/person/face)，
并遍历已见任务lane；不是对一张原图先判定唯一视图再只运行一名Router。

当前EMOTIC评估的人物单位来自测试注释的body_bbox，Face裁剪来自已有测试Face
manifest，失效Face由可靠性mask排除融合。因此当前报告的test以给定人物框/Face
预处理为前提，不是仅凭任意RGB自动检测目标人物的端到端部署接口。新原图若无元数据，
需先指定/检测人物框、检测并匹配Face，才能复用三视图流程；本轮未新增该接口。
路由选择不读取情绪标签，标签仅用于既定监督/评价；视图身份由生成过程显式命名。
本次仅澄清已有推理流程，无代码或实验变更。
