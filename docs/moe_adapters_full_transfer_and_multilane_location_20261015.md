# MULTI-LANE 论文位置说明与 MoE-Adapters 整体迁移分析

日期：2026-10-15，承接用户的第二轮问题。本轮按独立的 MoE-Adapters 方法分析，以 CVPR 2024 官方 MTIL 实现为参照，不以 ParaX 为设计基础；只更新文档，无代码实现或实验。

## 1. 修正上轮方案的定位

上轮推荐的“索引1、Summarize输入侧、每任务独立专家池”是为了沿用既有静态MoE接口的保守变体，**不是MoE-Adapters原法的完整迁移**。它可以回答任务内的视图路由问题，但放弃了原法的重要组成：跨任务专家复用与activate-freeze。

用户本轮强调参照官方代码尽量完整迁移，因此当前主要候选改为：**Task Forward各Transformer block的FFN并联完整MoE-Adapters；每层一个跨视图/跨任务共享专家池；task/view相关Router；保留并专门审查原activate-freeze策略。** 原图像证据侧方案保留为另一种位置消融，不再作为“完整迁移”的默认结构。

MoE-Adapters机制与此前ParaX参数混合方法分别讨论。历史静态Adapter结果只能约束其已测位置/路由，不能证明Task Forward全层MoE无效；此前ParaX的结果不能代替本独立方法的实验结论。

## 2. “Selector图像证据路径”对应论文什么地方

“Selector图像证据路径”是上轮为解释本地接口使用的描述，不是MULTI-LANE论文中新定义的模块。它对应[论文Figure 2、第3.3节](https://arxiv.org/html/2405.15633v1#S3.SS3)中的 **Frozen Forward→Summarize** 连线，即Patch Selectors读取冻结图像token的位置。

结合本地原版 `multi_lane/blocks.py::PreT_Attention.forward` 与Track-A `multi_lane/track_a/model.py::_lane_block`，一次block的顺序为：

1. **Frozen Forward** 提供当前层的图像表示：patch embeddings及原图像CLS；原始图像流继续通过冻结ViT。
2. **Task Forward** 的task-specific Patch Selectors与这些图像tokens计算相似性，经softmax得到权重，再加权汇总为少量 **summarized patches**。这里是软加权汇总，不是硬选几个原patch。
3. summarized patches和该任务自己的CLS一起进入带task-specific prefix prompts的 **MSA**。
4. **Drop & Replace** 丢弃MSA输出的summarized patch部分，恢复Patch Selectors；任务CLS及恢复后的Selectors经残差继续进入FFN和下一block。
5. 最后一层任务CLS用于分类；推理对所有已见Task Forward运行并按任务类别汇总。

第2步可用本地代码的等价表达说明（省略task/batch维与LN）：

```text
attention = softmax(Selectors @ ImageTokens.T / sqrt(d))
SummarizedPatches = attention @ ImageTokens
```

上轮的Image-token Adapter接法是在上式之前令 `AdaptedTokens = ImageTokens + MoE(ImageTokens)`，两处ImageTokens都换成AdaptedTokens，因而同时改变Selector匹配依据与汇总内容。这里的key/value是上述汇总运算中的角色，不是修改Frozen Forward的Q/K/V投影权重。

```text
Frozen Forward 当前层图像tokens ───────────────────→ 后续 Frozen Forward
               │
               └→ [上轮建议的 Image-token MoE 位置]
                          │
Patch Selectors ─────────→ Summarize
                          │ summarized patches + task CLS
                          ↓
                    MSA + prefix prompts
                          ↓
                     Drop & Replace
                          ↓ task CLS + restored Selectors（含残差）
                         FFN
                          ↓
                    下一层 Task Forward
```

“索引1”只是本地Python零基编号，表示第2个block读取图像tokens时插入：此时输入来自第1个Frozen Forward block的输出，经第2个block的ln1后供Summarize使用。**不是第2个block完整输出之后，也不是论文规定MoE必须放在第2层。**

当前代码定位：`model.py:725`准备冻结图像证据，`:747`适配，`:750`匹配、`:753`加权汇总，`:769`拼接task CLS，`:770`MSA，`:779`Drop & Replace，`:781`FFN。原版对应`blocks.py:77–96`汇总、`:136`恢复、`:245/:248`任务残差与FFN。

## 3. 官方 MoE-Adapters 的实际计算

审计固定官方CVPR分支commit `1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885`，不混用MoE-Adapters++。

来源：[Adapter定义](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885/mtil/clip/adapter.py)、[Transformer及Router](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885/mtil/clip/model.py)、[MTIL训练及冻结](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885/mtil/src/models/finetune.py)。

其block先完成MSA残差，再由当前CLS生成Top-2 gate，完整专家输出作为FFN的并联残差加入。每层有自己的专家池，跨层不共享专家参数；同层专家跨任务复用，路由按任务分配。图像与文本两套编码器均有MoE，参数各自独立。训练后按每层激活次数冻结最常用的两位专家，名单累积。

官方Adapter实际为down→ReLU→dropout→up→scale，up零初始化；`init_option='lora'`指初始化选项，不能据此把它当成对Q/K/V权重直接做低秩更新。`SparseDispatcher`对选中的整张样本token序列分发、按gate加权求和，并非逐patch重新路由。

## 4. 全部迁移的兼容性

**核心MoE机制可完整迁移；官方全部训练/推理流程无法原封不动保留。** 主要原因是MULTI-LANE的双前向、多标签和任务并行接口与官方图文单标签识别不同。

| 官方组成 | 可行性 | 本项目需要处理的问题 |
| --- | --- | --- |
| 完整瓶颈Adapter专家 | 高 | 使用独立MoE模块；宽度/初始化/scale明确，不复用参数混合模块 |
| CLS驱动noisy Top-2 + sparse dispatch | 高 | 从Task Forward任务CLS取描述符；扩展task/view索引和多lane张量 |
| 同层跨任务共享专家池 | 高 | 三视图共用参数，但分别处理各路tokens；区别于每任务独立池 |
| activate-freeze | 可实现，但不保证严格隔离 | 旧任务可能访问未冻结专家；三视图统计与容量消耗需重新定义 |
| 图像12层全部插入 | 工程可行 | 首选Task Forward FFN并联；插入位置因MULTI-LANE结构改变，属于方法迁移而非逐行复现 |
| 文本侧12层MoE及图文相似度分类 | 需新增结构/训练目标 | 当前Track-A只有图像编码器及分类头；加文本塔和文本模板会改变分类器与比较范围 |
| DDAS | 可编写模块，但原任务选择假设不匹配 | 同图多任务标签、同域任务分布重叠；不能靠单个task ID覆盖全部情绪标签 |
| 原cross-entropy/argmax/accuracy评价 | 必须改 | 使用多标签BCE/sigmoid/mAP与当前类别监督mask；不能沿用单标签softmax |

### 为什么不直接替换 Frozen Forward 为官方 MoE-CLIP

普通MULTI-LANE每个视图只需一份公共Frozen Forward结果，所有Task Forward都读取它。若在这条公共流里加入task-dependent MoE，task0和task1需要的图像tokens会不同：

- 只运行一个当前/预测任务Router，会把该任务的适配结果提供给其他lane，改变原来的任务隔离语义。
- 为全部任务分别运行适配后的完整图像ViT，能保持多任务预测，但Frozen Forward的公共图像计算不再共享，主干计算可能随3个视图×T个任务增长。
- 只冻结CLIP原始参数不能让整条流保持不变；专家和Router仍能改变中间图像表示。

所以Task Forward FFN并联是较自然的结构适配。另加一整套官方图文MoE分支再融合也能实现，但属于双模型系统，还需额外融合训练与成本对照，不能说是无代价的完整移植。

## 5. 更贴近原法的三视图 MULTI-LANE 结构

保留Frozen Forward、Summarize、prefix MSA和Drop & Replace。每层l配置一个专家池 `E[l,1..N]`，在任务及视图间共享；每任务t、视图v、层l保存Router及训练噪声参数。

令u为Task Forward在MSA、Drop & Replace和对应残差之后、FFN之前的持久token状态（任务CLS+恢复的Selectors）：

```text
g[t,v,l] = noisy_top2(u.CLS @ W_gate[t,v,l], W_noise[t,v,l])
moe[t,v,l](u) = sum_e g[t,v,l,e] * Adapter[l,e](u)
next_state = u + FrozenFFN(LN2(u)) + moe[t,v,l](u)
```

推理关闭噪声。官方专家读取注意力残差后的raw状态u；现有task_lane Adapter接口读LN2(u)，不能直接复用接口后声称输入完全相同。必须缓存正确的FFN前状态，避免错误地从FFN之后计算MoE。

这保留了完整专家、样本级Top-2、跨任务池和冻结复用。相对官方的必要改动是：MoE处理MULTI-LANE的压缩任务流；Drop & Replace位置明确；Router扩展view维；分类训练保持多标签。专家参数在三路共享，专家输入、输出及gate各自独立；它不直接融合三路patch或特征，最终三视图融合仍另行保留。

需要区分任务t（类别增量阶段）、视图v（Full/Person/Face）、图片x（内容）、层l。用一个视图编号代替任务Router，会丢掉原法的任务适配；三个视图共用同一Router也可能因输入不同产生不同gate，但缺少显式的视图专属决策函数，应设共享Router控制。

### 冻结策略是需要实验回答的取舍

可先把原频率Top-2策略作为“原法式冻结”对照，并测量旧路径变化，不预先声称零遗忘。对三视图存在两个选择：合并激活统计后Top-2，可能漏掉较弱视图依赖的专家；各视图Top-2取并集，一次最多新增冻结6位，每层有限池可能更早用尽可学习容量。这里是容量上界风险，不是实际一定每任务新增6位。

若要求保留MULTI-LANE的严格任务隔离，需要单独设计并标记保护版：旧Router、旧路径可访问专家、Norm/bias/scale和旧mask一起冻结；新增专家不能进入旧softmax范围。仅保护有限观测样本曾激活的专家也不保证所有未来图片稳定；限制旧访问集合可能改变封存当下的预测，需在冻结前后审计。保护版是对原法的修改，不能与原频率Top-2混称。

多层池还有上游依赖：哪怕某层所选旧专家冻结，上游未冻结专家改变u后，旧gate和后续输出仍会变化。应检查整条旧lane，而非只检查专家张量hash。

### 文本分支和DDAS不能用名词替换接口

文本分支可以新增，但要加载完整CLIP文本塔、定义情绪文本模板和图文相似度分类，并重新训练/校准多标签输出；当前线性分类头不能靠“加一个文本MoE”保持完全相同。只移植视觉MoE核心是合理方法，但应准确称为视觉侧迁移。

DDAS原实现以AlexNet特征重建误差选择唯一task/原CLIP分支。MULTI-LANE的task是同一数据域里的类别分组，同一人物图像可同时含task0和task3情绪，重建图像分布不等于判断标签所属任务。若改成多任务soft gates或只负责OOD拒识，需要新协议/校准，属于扩展；原论文阈值不可直接用于Full/Person/Face裁剪。

## 6. 官方代码中已发现的具体移植注意点

以下是固定版本静态代码审计发现，尚未运行移植后的模型，不将静态风险描述成已发生的实验故障：

- `mtil/clip/adapter.py:35`降维层写死64，上投影却使用配置bottleneck。默认64兼容；若直接配置16/32而不修改down层会维度不一致。应参数化两端，而非只改CLI。
- `model.py:336`预分配11个任务Router，`finetune.py:196`遍历12层，`evaluation.py:63`固定遍历12个AE槽位；需要从实际8任务/模型深度解析，不能保留原数据集数量。
- 官方通过模块全局`val_task_id`切换路由；多lane推理应显式传task/view，防止错误复用一个任务状态。
- `self.is_train`来自启动参数，不自动等价于PyTorch的`.training`；移植时需保证eval关闭noise。计数在forward内更新，也要隔离训练/验证并正确处理无效Face和多GPU汇总。
- block计算了importance/load的CV项，但核对的MTIL训练入口损失是cross-entropy，未看到该项并入训练loss。不能默认“复制Router即包含负载均衡训练”；若新增此损失需单列为方法变化。
- 原DDAS评价把重建误差归约成标量并选择一个task ID。对batch>1会给整个batch同一路由；即便改成逐样本也无法解决单图多任务标签的根本问题。
- MTIL训练通过冻结文件维护专家，审计的CIL训练入口没有同等冻结名单处理，且其forward路由调用不同；不能任选一个入口后宣称完整实现Figure 3。
- 当前compact checkpoint排除所有`visual_encoder.*`。若把新专家直接放到visual_encoder内部，会一起漏存；应显式保存MoE bank、routers/noise、冻结集合、eligible masks及协议，不以requires_grad过滤旧专家。

## 7. 成本与验证边界

如果保持原视觉侧规模22专家、b64、12层、d768，且专家各带down/up bias，专家池为26,171,904参数。8任务×3视图、每层各自gate/noise线性矩阵再加9,732,096，总计**35,904,000新增视觉侧参数**（官方gate/noise无bias；不含新增文本塔/DDAS）。Top-2只减少激活专家的计算，不减少全部专家权重存储。这个预算远高于历史单层小池，不能用小池资源需求预估完整迁移。

先验证原法式完整视觉核心的函数正确性和资源，再决定规模，不能为了降低成本无记录地改层数、专家数或冻结策略。对原法式频率冻结与保护版分别记录：学习新类的收益、旧lane输出漂移、可训练专家余量、三视图路由/专家使用率、Face有效性、残差强度和总/活跃参数。路由不同不等于语义分工，冻结成功不等于专家具有可迁移判别力。

最小机制对照应围绕新位置重新建立：无MoE基线、同位置普通Adapter、无冻结MoE、原频率冻结MoE、需要时的严格保护版；对共享Router/视图Router作单变量比较。容量/冻结/视图条件化不能全部同时改变后只比较两个端点。最终按用户确定的完整多标签实验协议评价，test不用于反复选择结构。

本轮结论是可行性与风险判断，不是实验配置：没有宣布学习率、epoch、数据/权重实际路径或启动命令，也没有运行smoke/训练/test。后续若用户要求实施与执行，再声明完整配置并按Git-only远程流程验证。

## 8. 本轮交付状态

继续在独立分支 `codex/moe-adapters-three-view-20261015`（HEAD `25fc861`）维护分析。新增本报告，给上轮报告增加本轮修正入口，更新PROJECT_CONTEXT/WORK_LOG/TODO_NEXT；原主工作区和其他worktree不修改，无commit/push/服务器操作。

源码审计通过公开官方固定commit读取8份文件（Adapter、block、MTIL训练/评价、AutoEncoder、通用评价、CIL入口、原训练脚本），只在系统临时目录保存只读参考副本。本地模型接口按实际代码行核对；文档表格/算式及diff检查，不运行业务单元测试。上述静态分析不能代替后续移植测试。
