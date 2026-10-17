# v4 既有 checkpoint 梯度归因结果

## 结论与决策

本次只读归因**没有达到预先声明的跨任务稳定冲突门槛，不支持立即训练梯度隔离版 v5**。
这不是证明所有训练阶段都没有冲突，而是现有终点 checkpoint、固定训练批次中的证据
不足以把“共享优化冲突”确认为可修复的主要失败原因。结合 v4 泛化及视图消融结果，
当前建议保留 B0，结束当前 Task Forward、Drop & Replace 后与 FFN 并联的 MoE 训练路线。

确实存在 task7 Prompt 的局部重复冲突和一个 MoE-on/off 方向反转 batch，须保留这些
事实；但它们不能外推为跨任务、全模型的稳定负迁移，也不足以自动启动新训练。

## 执行与只读审计

- 独立分支：`codex/moe-v4-gradient-audit-20261017`。
- 源训练：clean `bfd8ef99f5a13d2c223b069522cf7fe2a099b6e7`，seed0。
  正式诊断：clean `272b768`；成功 smoke：clean `b8107df`。
- EMOTIC **train**，B0 / shared_fixed / view_fixed / dynamic；task0、task2、task7。
- 每任务4个固定batch，每批64人，同一原图的 eligible 人物保持同批；四组、两数值
  配置复用相同像素及标签。共12个输入batch、768个人物槽位；任务之间可能有重合，
  不视为768个独立样本。
- deterministic_fp32：Expert dropout off、TF32 off；training_amp：Expert dropout0.1、
  FP16/TF32 on、梯度先乘1024再还原。两配置是敏感性对照，不是两个随机训练seed。
- 4×3×4×2=**96**个正式快照；GPU0，约139.5秒，峰值reserved **6138 MiB**。
- **312项测试通过**；真实smoke8快照通过。前两次smoke在配置/数据入口阶段退出，
  分别修正归档字段名及三视图Face参数null的读取，没有产生梯度或参数更新。
- 12个源checkpoint SHA、模型state SHA全部不变；`.grad`全部None；像素及标签SHA不变。
  梯度分解最大相对重构误差 **0.000284864（0.0285%）**，在声明容差内。
- optimizer/step/训练更新数均0；不加载validation/test样本。只读取既有val结果中的
  checkpoint哈希作来源校验。120份远程日志/结果同步本地后逐项SHA校验通过；
  本地离线重新汇总与同步文件逐项一致。

输入审计（每行依次为四个batch）：

| task | 原图数 | 可靠Face人数/64 |
|---|---|---|
| 0 | 56 / 58 / 58 / 61 | 38 / 41 / 38 / 35 |
| 2 | 59 / 58 / 57 / 60 | 50 / 43 / 49 / 52 |
| 7 | 56 / 59 / 58 / 59 | 45 / 47 / 45 / 53 |

## 1. 共享 Expert 没有出现跨任务稳定的视图冲突

预声明的描述性门槛：cosine<−0.1，且每任务至少3/4 batch、至少2/3任务、
两数值配置一致。对所有已汇总参数组/视图对，没有组合满足跨任务门槛。
零梯度/缺失组标为N/A；B0没有Expert，shared_fixed没有活跃专门Expert，不能算作无冲突。
独立视图Router块不用于证明共享参数的冲突或无冲突。

下表为 dynamic **shared Expert** 上 cosine<−0.1 的batch数，每格为FP32/AMP，分母均4：

| 梯度对 | task0 | task2 | task7 |
|---|---:|---:|---:|
| 单路Full vs Person BCE | 0/0 | 1/0 | 1/1 |
| 同可靠人群Full vs Face BCE | 0/0 | 0/0 | 1/2 |
| 同可靠人群Person vs Face BCE | 1/1 | 0/0 | 0/1 |
| fused内部Full vs Person路径 | 0/0 | 0/0 | 1/1 |
| fused内部Full vs Face路径 | 0/0 | 0/0 | 1/2 |
| fused内部Person vs Face路径 | 1/1 | 0/0 | 0/0 |

这里区分了“分别要求每路分类”的BCE梯度与原fused loss内部按链式法则拆出的路径梯度。
后者相加能够重构fused表示梯度；classifier有直接梯度，不能按endpoint路径分解，
其对应路径格保留N/A。并非用各视图独立BCE去近似融合目标。

B0本身也有负夹角。例如Selector+Prompt合并坐标上，Full/Person单路梯度在task7
两数值配置均2/4 batch低于−0.1；fused Full/Face路径也在task7均2/4。
因此“观察到任意负夹角”不能归因为新增MoE。

## 2. 局部例外：task7 Prompt 的 Full/Face 冲突真实存在

dynamic在同一可靠Face人群的Prompt坐标上，Full/Face单路BCE梯度余弦：

| 配置 | batch0 | batch1 | batch2 | batch3 |
|---|---:|---:|---:|---:|
| FP32 | −0.1829 | −0.3921 | −0.2881 | −0.2091 |
| AMP/dropout | −0.1776 | −0.3622 | −0.2683 | −0.2536 |

这是单任务4/4重复的局部信号；对应task0/task2均0/4，不满足跨任务要求。
shared_fixed在task7也有3/4，B0是1/4；view_fixed在task7为0/4。
所以它值得作为病例记录，不能简单称为“B0已有的一切冲突”。但它没有表现为
实际整体auxiliary反向，dynamic该task的Prompt辅助梯度沿fused方向仍增加约3.6%–9.3%。
fused内部Prompt Full/Face路径在task7是FP32 3/4、AMP 2/4，稳定性也弱于单路BCE。

## 3. 实际加权辅助监督没有大幅抵消 fused 监督

使用原训练目标：

`L_total = L_fused + 0.1 × mean(L_Full, L_Person, L_Face可靠) + 0.001 × L_balance(dynamic)`。

记 `g_A = ∇L_auxiliary`、`g_F = ∇L_fused`，检查
`p = <g_A,g_F>/||g_F||²`。p<−0.1才表示辅助监督反向抵消至少10%的fused方向分量。
下表合并两数值配置24个测量，比例中位数仅作描述，不把它们视为24个独立batch：

| 方法 | Selector+Prompt的 aux/fused 范数比中位数 | 最小p | shared Expert范数比中位数 | shared Expert最小p |
|---|---:|---:|---:|---:|
| B0 | 0.1255 | +0.0428 | N/A | N/A |
| shared_fixed | 0.1285 | +0.0479 | 0.1195 | +0.0345 |
| view_fixed | 0.1323 | +0.0660 | 0.1117 | +0.0585 |
| dynamic | 0.1068 | +0.0558 | 0.1109 | +0.0432 |

逐组检查而非只看合并坐标：dynamic的Selector、Prompt、classifier、共享及专门
Expert均无负p。shared_fixed在task0/batch2的Prompt有负p：FP32 −0.0141、
AMP −0.0172，即沿fused抵消约1.4%/1.7%；没有达到10%门槛。
其余被测主参数/活跃专家组未发现负p。

dynamic的 `cos(g_total,g_fused)` 最小值：Selector0.9897、Prompt0.9873、
classifier0.9944、共享Expert0.9920、专门Expert0.9843。
总raw梯度主要仍沿fused方向，不能据此主张立即做PCGrad或移除全部视图辅助监督。
这也不证明当前fused目标足以改善泛化。

dynamic FP32四batch的梯度范数中位数如下；Full/Person是全批单路BCE，Face是可靠
人群单路BCE，aux是已乘0.1并取视图均值的实际梯度：

| task | 参数组 | Full | Person | Face可靠 | fused | auxiliary |
|---|---|---:|---:|---:|---:|---:|
| 0 | shared Expert | .01962 | .02755 | .04799 | .01266 | .002213 |
| 0 | Selector | .02169 | .02237 | .03004 | .01656 | .002375 |
| 0 | Prompt | .001446 | .001625 | .003444 | .0009113 | .0001798 |
| 0 | classifier | .003366 | .003386 | .006389 | .002595 | .0003798 |
| 2 | shared Expert | .03030 | .02904 | .02681 | .02448 | .001984 |
| 2 | Selector | .01617 | .01170 | .01770 | .01262 | .001136 |
| 2 | Prompt | .002201 | .002667 | .002408 | .001796 | .0001590 |
| 2 | classifier | .004425 | .003829 | .004667 | .002962 | .0002827 |
| 7 | shared Expert | .02994 | .03197 | .03041 | .01639 | .002012 |
| 7 | Selector | .01501 | .01739 | .01479 | .009516 | .001023 |
| 7 | Prompt | .005206 | .006997 | .005197 | .003597 | .0003294 |
| 7 | classifier | .004503 | .004531 | .004360 | .003190 | .0003364 |

参数组维数、原训练LR不同，且原Adam moment未保存；不能把跨组范数直接当作实际更新
或泛化影响大小。全部四组逐batch范数/余弦见机器可读JSON/CSV，未只保留中位数。

## 4. 不可靠 Face 没有泄漏进真实监督

96个快照中，fused loss对不可靠Face endpoint的梯度最大绝对值均严格0；
其实际Face辅助loss系数也为0。`face_unreliable_shadow`单独保存为假设性诊断，
不属于原训练损失，不能把它的梯度解释成实际训练污染。

在dynamic共享Expert上，FP32的Full/Person条件梯度：可靠人群1/12 batch低于−0.1，
不可靠人群0/12；两类均未形成跨任务稳定信号。更多条件比较和其他三组均保留在CSV。

## 5. 同参数点 MoE-off 锚点存在局部偏转，未形成稳定系统性反向

在MoE模型相同Selector/Prompt/classifier参数处关闭MoE，再计算B0形式的原监督梯度。
这比把fresh B0另一参数点的梯度直接当作因果对照更可解释，但仍只是局部反事实。

- dynamic：Selector+Prompt合并梯度与off锚点的余弦中位数0.5979，最小−0.3032。
  两数值配置的显著反向均只在 **task7/batch1**，分别−0.3032/−0.2803；该任务另外3批
  未低于−0.1，task0/task2也未达到，因此没有满足3/4及跨任务条件。
- view_fixed同一task7/batch1也反向：−0.2442/−0.2303；shared_fixed没有该合并组反向。
- dynamic的Prompt锚点比较同样在该batch反向，classifier则全部正向。
- 与fresh B0不同已学参数点的同坐标对齐只作描述，原始数值保留，不用于因果判定。

这支持“部分图片上新增残差明显改变局部优化方向”，不支持“共享Expert持续破坏B0监督”
的总体结论。按当前少量终点batch直接设计锚点梯度修正，有针对这些病例过拟合的风险。

## 下一步

1. **本轮归因结束；不训练v5，不扩大专家/改温度/改scale，不扩seed，不合并main。**
   继续使用B0作为当前结果。没有实施任何模型或训练目标变更。
2. 当前Task-Forward MoE路线暂作负结果归档：v4已解决弱专家/无效路由机制问题，
   但未获得整体泛化优势；本轮也未找到达到预声明门槛的共享梯度冲突修复依据。
3. 若未来有独立研究理由重开梯度路线，先取得早/中训练阶段同配对样本及Adam状态，
   再验证task7局部模式能否跨训练阶段/任务成立；需单独声明新实验，不能把本次终点诊断
   外推为完整训练轨迹。随后才可能讨论以B0监督为锚、固定路由控制及原图分组holdout。
4. 本次不自动转向Face后置校准。已有可靠Face单路局部收益未转化为融合净收益，
   需要新的独立证据与假设，不能用重新命名继续试同一方案。

## 复现与产物

- 协议：[诊断配置](moe_v4_gradient_attribution_plan_20261017.md)。
- 诊断入口：`scripts/emotic/run_moe_v4_gradient_audit.sh`；汇总入口：
  `python multi_lane/track_a/summarize_gradient_attribution.py output/moe_v4_gradient_audit/formal_20261017_01`。
  汇总只读已产生JSON，不加载模型/数据。
- `output/moe_v4_gradient_audit/formal_20261017_01/`：config、96份梯度JSON、result、
  summary、metrics.csv、aggregates.csv。
- `output/moe_v4_gradient_audit/transfer_sha256_20261017.json`：120份同步文件哈希清单。
- `logs/moe_v4_gradient_audit/`：312项测试、成功/早停smoke及正式运行日志。
- 代码经Git-only在新worktree运行，源v4/main/test-only工作树未改动。
