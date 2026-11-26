# Summarized Patches 决定 logit 融合比例：两种 TRAIN 来源的八组对照

2026-11-26：已实现，尚未在 Linux 运行新单元测试、真实 smoke 或正式实验，尚无新成绩。
分支 `exp/d9-summary-fusion-coverage`，从已核验的 `626d812` 建立独立工作树。
本文件覆盖历史六组方案，不恢复任何分类输出输入或 Task CLS 输入。

## 数据与八组

| 方法 | 原90%来源 | 新增全部TRAIN来源 | 可训练参数 |
|---|---|---|---:|
| 原固定比例 | 90%分支，同源固定读出 | 全TRAIN分支，同源固定读出 | 0 |
| 当前任务学习一套共用比例 | 另10%融合TRAIN | 全部当期合法TRAIN人物 | 2 |
| 根据Summarized Patches平均相似度学习比例 | 另10%融合TRAIN | 全部当期合法TRAIN人物 | 14 |
| 根据Summarized Patches对应情况学习比例 | 另10%融合TRAIN | 全部当期合法TRAIN人物 | 54 |

90%来源是原随机原图划分、补足13,950更新的冻结三路：11,623图训练分支，另外1,292图作为融合池。
全部TRAIN来源复用已有 `sample_full` 冻结三路，覆盖12,915原图；融合使用当期合法的训练人物。
**全部TRAIN不表示读取所有26类未来标签，也不表示重新训练三路。** 两来源的模型结构、seed1、30 epochs、精度和初始化CLIP相同；来源自身的历史融合模块不用于本轮的比例计算。

全部TRAIN人物已经参加分支训练，可能有较强的训练集预测偏差。三折只隔离新增比例模块的拟合数据，不能把它描述为“三路也未见过的折外数据”。
所有组保留同一自然图片预处理、原合法监督人物、逐人Face规则；本轮不补负例、不做模糊增强、不增加TCP或收益监督。

本批为task0/3/6分别初始化和拟合的当前头诊断。第二组是当前任务的常量，另外两组每张原图产生一套基础分数；均不是已实现的正式八任务公共比例训练轨迹。
验证与测试使用同批完整评价人群，但只计算这些阶段的新类别AP。报告的三任务当前头平均不能与八任务最终/阶段平均直接相比。
原目标仍为Val44.0329/51.1335、Test33.1770/40.2083；达到目标须后续单独授权完整八任务实验。

## 结构与输入

三路原分类头和原Frozen9跨视图路径保留。第10层（索引9）Task Forward的Summarize输出为未平均的Summarized Patches，捕获位置在Task MSA之前。
捕获接口默认关闭；启用时只旁读、detach，不改变正常计算路径。输入统计没有新的可学习投影。

对每个已见任务头t、每个人物p，两对集合分别计算：

\[
C^{(v,t,p)}_{ij}=\frac{\bar g^{(Full,t,p)}_i\cdot\bar g^{(v,t,p)}_j}
 {\|\bar g^{(Full,t,p)}_i\|\|\bar g^{(v,t,p)}_j\|},\quad v\in\{Person,Face\}.
\]

零范数按归一化的epsilon处理；Selector编号不硬配对。每对取矩阵均值、行最大值均值/标准差、列最大值均值/标准差，以及温度0.1的行softmax归一化熵均值。只有一个Selector时，熵及单元素标准差为零。
这些是“集合对应情况”的描述，不是真实空间坐标或可靠性标签。

先在已见头间等权平均，再在完整原图人物名册中汇总。名册由人物框和Face可用性确定，不按情绪标签筛选；Face相关量只汇总可靠Face人物，无可靠Face时填零。

- 平均相似度组：两对均值的图内均值/标准差，共4维；追加可靠Face比例、n/(n+1)人数编码，共6维。
- 对应情况组：两对各6项统计的图内均值/标准差，共24维；追加相同2维元数据，共26维。

两个逐图组的输入维度和参数量不同，不宣称严格等容量；同一种方法在两种TRAIN来源之间的结构和参数量完全一致。

分类logits、分类概率、分类置信度/熵、分类分差和最终Task CLS都不进入比例模块。相似度矩阵的对应熵与分类预测熵含义不同。

## 比例和监督

令原权重w0=(0.64,0.16,0.20)，每张原图I的分数为：

\[
a_F(I)=1,\qquad
a_v(I)=1+\log\frac{w^0_v}{w^0_F}+b_v+
0.25\tanh\big([W^T(\phi(I)-\mu_{TRAIN})]_v\big).
\]

固定组令b=W=0；共用比例组只学b；两组逐图方法共同学习b和W，从零开始。
这里b是**本次诊断任务唯一的两项共用偏置**，不读取第二组训练结果、不按旧任务头分别加载常量。
均值中心只用本次拟合TRAIN；三折时分别用该折训练图，Val/Test不重新中心化，也不用z-score放大小方差。
0.25限制图像动态项，不限制b，不保证Full永远最大或精度不降。

同图每个人p共用a(I)，逐人遮蔽不可用Face，再softmax：

\[
w_{I,p}=\operatorname{MaskedSoftmax}(a(I),m_{I,p}),\qquad
z_{I,p,c}=\sum_v w_{I,p,v}\,\operatorname{stopgrad}(z^v_{I,p,c}).
\]

零初始化恢复64/16/20；无可靠Face的人恢复80/20/0。混合缺脸图共享基础分数，但最终归一化权重可以不同。
最终预测逐类sigmoid(z)，不把三路特征融合后再分类。

三种学习组使用同一真实情绪标签监督：

\[
L=\frac1{|\mathcal I|}\sum_{I\in\mathcal I}\frac1{|P_I^{sup}|}
\sum_{p\in P_I^{sup}}\frac1{|Y^t|}\sum_{c\in Y^t}
\operatorname{BCEWithLogits}(z_{I,p,c},y_{I,p,c})
+0.001\operatorname{mean}(W^2).
\]

没有W的组省略正则。监督来自当前合法TRAIN标签，经最终logit加权和softmax更新新增参数；相似度只是输入，没有“相似度越大越可靠”的监督规定。
三路权重、中间摘要、原logits均冻结。图内先平均有监督人物，再对图平均，避免多人图片占据更多优化权重。

## 相同配置和验证顺序

- EMOTIC CVPR17、seed1，task0/3/6独立拟合；全部TRAIN对照与90%对照共用一个配置对象，没有逐来源超参数覆盖接口。
- CPU float32单线程，全批Adam300步；b学习率0.01，W学习率0.001，weight decay0，无scheduler，W的L2系数0.001。
- GPU只用于冻结模型提取，batch64、workers2、确定性设置与原生评价重放一致；模型checkpoint来自两个只读来源。
- 原图固定三折，跨任务同图同折；保存逐折正负支持、人数、Face覆盖、TRAIN与折外指标。折外结果只作诊断，不选择收缩系数或超参数。
- 每个来源先拟合全部TRAIN，再完整Val；两来源Val结果、100次同Face可用性类别内跨图交换、组均值和1000次按原图配对bootstrap封存后，才进入两来源Test。无论Val胜负都按协议报告Test。
- 组均值指无Face/混合/全部Face三类原图内的**基础分数均值**；从Val无标签输入计算，封存用于Test，仅作归因。缺少某类Val图时使用全Val分数均值，不能用Test重算。
- 跨图交换同组随机循环，尽量无自身对应；组内只有一图时无法交换，报告组大小。bootstrap在跨任务重复原图上共用采样次数，比较本图与共用比例、本图与100次交换AP均值；缺正/负支持的重复作废并报告数量。
- 完整原生首批logits要求捕获开关前后及相对归档逐值一致；正式评分始终使用归档原生logits，避免按图重排造成AMP批边界差异。
- 记录模型/config/roles/原始标注/Face manifest/缓存/code SHA-256，原图和人物ID，以及完整名册与监督人数。两来源Val/Test人物、标签及Face标记逐值配对。

逐图候选需在Val平均优于固定、共用、组均值、交换且至少两个任务改善；对应情况组还须优于平均相似度组。相对共用和交换的bootstrap下界须大于0。
未满足记为“未确认”，不自动增大MLP或新增损失。即使通过，也仅支持申请独立的八任务计划，不宣称达到原目标。

## 入口与产物

配置：`scripts/emotic/configs/d9_summary_fusion_20261126.json`。
Linux独立工作树在同步同一Git commit、激活项目现有PyTorch环境后使用：

```bash
bash scripts/emotic/run_d9_summary_fusion.sh check
# 先核实可用GPU，再设置实际编号；不预设GPU0空闲。
CUDA_VISIBLE_DEVICES=<可用GPU编号> bash scripts/emotic/run_d9_summary_fusion.sh preflight
CUDA_VISIBLE_DEVICES=<可用GPU编号> bash scripts/emotic/run_d9_summary_fusion.sh pilot
```

输出 `output/d9_summary_fusion/seed1_20261126/{preflight,pilot}/{equal90,full}/`：
`features/`为中间统计，`native_train/`为全TRAIN原生logits，`readout/`保存三折/拟合参数、原图基础分数、逐人权重、预测、逐类AP、分组结果、交换、bootstrap及封存文件。
顶层 `comparison.json` 汇总八行，明确标注不是正式八任务分数。日志在 `logs/d9_summary_fusion/seed1_20261126/`。
每阶段拟合独立；真实smoke用task0/6、每可用性类别至多12图、5步、2次交换、10次bootstrap，两个来源同步降预算，只验证流程。全TRAIN原生logits缓存仍按完整合法人群提取，不能当作正式效果。
pilot要求同代码、同配置、同来源已通过preflight，且工作树干净。已有不兼容产物不会静默覆盖。

实现涉及四个原模型文件的可选摘要捕获、新比例模块、提取/拟合/诊断/运行入口、统一配置和Linux launcher。
新增12项单元测试覆盖捕获不改logits、输入/梯度边界、Selector/人物顺序、零初始化与Face遮蔽、三折、Val封存及完整小型合成流程。

本地已通过Python AST语法、JSON、bash语法和diff空白检查；本地缺PyTorch且项目运行约定在Linux，故未将新增单元测试、原生数值重放或真实smoke列为已通过。
服务器只读核对已确认两来源目录、全部TRAIN角色表及task0/3/6 checkpoint存在；原90%来源结构与全部TRAIN来源匹配，预算13,950更新。
未提交、推送或运行新实验。远程实验结束后需同步 `logs/`、`output/` 回本地，核对哈希后分析结果。
