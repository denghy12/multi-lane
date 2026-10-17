# Task-Forward MoE v3：完整 validation/test 与机制定位

日期：2026-10-17。mAP/AP及差值均为百分数/百分点。结论限于seed0，
bootstrap仅反映固定权重下的样本波动，不包含训练seed不确定性。

## 1. 结论

v3解决了v2提出的“专门专家输出太弱”，但没有得到可靠的动态MoE收益。
动态组最终val/test为42.2628/31.6540，B0为42.1912/32.0610；
val阶段平均+0.5892，test阶段平均却−0.9542。动态组最终也低于匹配
shared-only控制的42.3141/31.7327。没有证据支持扩大专家数、提高
specialist LR、继续增加路由噪声、跨任务复用或合并main。

本次只读诊断进一步定位：专家接收Drop & Replace后的任务CLS和恢复的
Selector，而归一化图片证据仅供Router使用。所有任务/视图中，专门
残差至少99.9821%的能量在Selector位置；task7约99.99999%。
这些Selector输入在不同图片间不变，专门输出的变化主要由图片Router
混合不同专家产生。该方法实际上主要在做图片条件化的Selector改写。

因此，上轮“specialist/shared输出比≥0.25”机制门槛不充分：它被
Selector大范数主导。下一步应停止当前raw-task-lane专家形式；若继续
研究MoE，优先测试的结构候选是让专家直接处理本块图片
证据并只输出CLS残差，保留当前FFN并联加法位置，再用普通共享Adapter
控制判断是否需要MoE。这是新的待验证假设，不是已有改善结果。

## 2. 训练、评估、同步与基线可比性

- 三组训练固定clean提交`7eb1f6400fe9ddc68d81eba8d7741fdea9e7f264`；
  各8任务×30epochs、13950成功更新、0 skipped，训练期间未加载test。
- 每组训练结束后val/test两个进程并行遍历同一套8个compact checkpoint；
  三组两路退出码均0，批次完成标记齐全。两专门组val specialist-off也完整。
- val每阶段固定2397人（1705原图），test固定5368人（3682原图）。
  所有阶段、方法及B0的sample ID/targets/class indices逐项对齐，Face
  mask一致；val/test及消融的每任务checkpoint SHA一致。
- B0使用已归档`v2_evaluation/full_val|full_test/BASELINE`，没有使用
  训练期按已见正例筛选的`val_scores`。B0与v3的初始主参数SHA、CLIP
  SHA、8任务训练样本SHA、首批ID、采样种子及主要超参数逐项一致，
  Face manifest hashes一致。基线与v3业务代码提交不同，公共协议已审计。
- 三组CLIP前后SHA不变；峰值reserved显存dynamic2940MiB、
  shared-only2922MiB、view-fixed2934MiB；训练含原有阶段评估耗时约
  80.9/80.7/79.4分钟。每任务403456专家/Router参数，总计3227648。
- 已同步148份正式分数/summary/config/history/审计文件并逐文件SHA-256
  对照服务器。日志与额外诊断也已同步，compact checkpoint留服务器。

## 3. 整体与阶段结果

| 方法 | 最终val | val阶段平均 | 最终test | test阶段平均 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 42.1912 | 48.4382 | **32.0610** | **36.2781** |
| v3 shared-only | **42.3141** | 48.7763 | 31.7327 | 35.5893 |
| v3 view-fixed | 42.1996 | 48.6707 | 31.5408 | 35.4995 |
| v3 dynamic | 42.2628 | **49.0274** | 31.6540 | 35.3239 |

| 阶段 | val B0 | val dynamic | test B0 | test dynamic | test差值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 55.2676 | 56.2322 | 37.7303 | 36.2049 | −1.5254 |
| 1 | 56.6820 | 58.1029 | 41.4107 | 39.6654 | −1.7453 |
| 2 | 44.1527 | 44.9775 | 33.2027 | 31.8049 | −1.3978 |
| 3 | 49.3111 | 49.8075 | 37.6984 | 36.6842 | −1.0141 |
| 4 | 48.3659 | 48.8033 | 37.3636 | 36.6985 | −0.6651 |
| 5 | 46.7316 | 47.1223 | 35.8557 | 35.5076 | −0.3481 |
| 6 | 44.8036 | 44.9109 | 34.9020 | 34.3717 | −0.5303 |
| 7 | 42.1912 | 42.2628 | 32.0610 | 31.6540 | −0.4071 |

三组test的8个阶段全部低于B0，动态组test阶段平均在三候选中最低。
dynamic比v2的val阶段平均提高0.7756，但最终val反低0.0249；v2没有
本轮配对test，不能构造v2→v3的test改善。首版test最终31.7789，v3
仍低0.1249，阶段平均从35.1268升至35.3239，整体仍未超过B0。

task0 test已经低1.5254，此时不存在后续任务遗忘。固定人群forgetting
也很小：B0 val/test为0.00647/0.00710，dynamic为0.00687/0.00534。
主要问题是任务学习与泛化，不能用增加旧任务蒸馏解释或修复本轮结果。

## 4. 不确定性

500次原图聚类配对bootstrap，同图所有人物一起重采样，8阶段使用相同
原图抽样后再取阶段均值。种子val20261017、test20261018。未做多重比较
校正；以下区间不能替代多训练seed复现。

| 差值 | 点估计 | 95%区间 |
| --- | ---: | --- |
| dynamic−B0，val最终 | +0.0715 | [−0.639,+0.796] |
| dynamic−B0，val阶段平均 | +0.5892 | [−0.310,+1.612] |
| dynamic−B0，test task0 | −1.5254 | [−3.077,−0.155] |
| dynamic−B0，test最终 | −0.4071 | [−0.947,+0.131] |
| dynamic−B0，test阶段平均 | −0.9542 | [−1.772,−0.264] |
| shared-only−B0，test阶段平均 | −0.6887 | [−1.355,−0.078] |
| dynamic−shared-only，test最终 | −0.0787 | [−0.471,+0.389] |
| dynamic−shared-only，test阶段平均 | −0.2654 | [−0.774,+0.279] |

最终task7的单个差值跨0，不能断言总体真值必然下降；但当前seed的test
阶段平均下降有样本层面的证据。dynamic与控制组的区间跨0，表示没有
证明动态优势，不表示已证明等价。

## 5. 三视图、可靠性与类别

最终dynamic−B0：

| 视图 | val差值 | test差值 |
| --- | ---: | ---: |
| fused | +0.0715 | −0.4071 |
| Full | +0.2761 | −0.4646 |
| Person | −0.5424 | −0.2550 |
| Face | −0.1163 | +0.4003 |
| 可靠Face单路 | −0.8422 | +0.3012 |

Person在两个split均下降。test Face单路改善没有抵消Full/Person损失，
不能据此在test上调高Face融合权重。可靠Face人群的fused test差值
为−0.2824，不可靠人群为−0.3172；问题并非只来自低质量Face。

按来源描述性分组，dynamic−B0 fused val/test差值分别为：MSCOCO
+0.6219/+0.3035，emodb −0.5405/−0.8214，ADE20K −2.0016/−4.7005。
ADE仅74/160人，类别支持不一致，不能单凭这些数推定某种域偏差，更不能
按人数线性加权还原总体mAP；这些结果提示收益分布不均，需要保留来源
维度审计，不能只看全体平均。

dynamic最终test有12/26类AP超过B0。下降最多包括Suffering −5.019
（186正例）、Annoyance −3.499（253）、Confidence −3.217（2855）、
Disconnection −2.781（809）、Aversion −2.189（145）；提高最多包括
Pleasure +3.978（1657）、Happiness +2.729（2380）、Fatigue +2.384
（314）。损失不只发生在少正例类，不能用稀有类波动解释全部退步。

## 6. 专门专家变强了，但主要在改写Selector

全部任务/视图都实际调用过三名专门专家，task7三个视图最常见专家对
占比约41.9%/42.8%/41.3%；部分旧任务仍偏斜，例如task1 Full有92.1%
选择同一对。Soft Top-2熵为0.645–0.692（两名均匀权重上限ln2=0.693），
多数被选专家权重接近各0.5。它具备跨图片路由变化，但不能据此证明有效。

24个动态池/视图的specialist/shared范数比为0.715–4.121，全部达到
预设0.25门槛；shared与specialist输出余弦为0.680–0.934，方向较相似。
task7 Full/Person/Face范数比分别0.976/0.985/0.984，CLS比分别
0.673/0.703/0.682。

只读前向同时记录专家输入与输出的跨图片均值/方差、CLS与Selector能量。
所有任务的Selector专家输入跨图片能量变化约0（最大7.8e−16）；
这是当前实现的结构结果：Drop & Replace恢复Selector，FFN逐token
运算，当前层的图片`selected`仅提供给Router，不直接进入专家。

| 当前task7池，val | Full | Person | Face |
| --- | ---: | ---: | ---: |
| 共享输出CLS RMS范数 | 0.2993 | 0.3018 | 0.3040 |
| 专门输出CLS RMS范数 | 0.2052 | 0.2149 | 0.2109 |
| 专门输出在Selector的能量占比 | 99.9999906% | 99.9999900% | 99.9999903% |

在所有8任务/3视图，专门Selector能量占比至少99.9821%。这不意味着
Selector更新天然无用；Selector会影响后续图像证据读取。但是全token
范数不能被解释为对图像证据或CLS的有效适配。v3的强专家主要相当于
由图片Router选择不同的Selector变换，专门CLS残差依然很小。

## 7. 专门专家关闭与同checkpoint干预

关闭specialist保留`0.5*shared`以及联合训练后的Selector/Prompt/head，
因此该消融不是重新训练shared-only，也不是B0。

| validation | on最终 | specialist-off最终 | on−off | on−off阶段平均 |
| --- | ---: | ---: | ---: | ---: |
| dynamic | 42.2628 | 42.5179 | −0.2552 | +0.4265 |
| view-fixed | 42.1996 | 42.5822 | −0.3827 | +0.0182 |

dynamic的on−off曲线为+1.5330/+0.8394/+0.6570/+0.4636/+0.3603/
+0.0145/−0.2006/−0.2552；早期有直接收益，到最后变为负。动态最终
融合差值95%区间[−0.732,+0.298]跨0，固定视图最终区间
[−0.764,−0.029]为负。

动态最终分支on−off：Full −0.3366（区间[−0.871,+0.372]），Person
−0.5304（[−0.931,−0.127]），Face +0.6565（[+0.112,+0.978]）。
这支持任务/视图作用不一致，尤其Person受损；不能把幅度门槛通过当作
泛化通过，也不能从某个单路改善推导融合收益。

另在同一task7 checkpoint、相同2397人validation做以下只读干预，
所有8个已见任务池一起干预。原始logits逐值重放差0，模型state与源
checkpoint SHA前后不变，无训练、无test干预。

| 前向 | 最终mAP | 相对原始 | 95%区间（干预−原始） |
| --- | ---: | ---: | --- |
| 原始dynamic | 42.2628 | 0 | — |
| 打乱每batch Router图片证据 | 42.1665 | −0.0963 | [−0.379,+0.180] |
| 每任务/视图固定最常用专家对 | 41.7190 | −0.5438 | [−1.202,−0.025] |
| 专门残差只写CLS | 42.5112 | +0.2484 | [−0.283,+0.718] |
| 专门残差只写Selector | 42.2568 | −0.0060 | [−0.132,+0.110] |

固定专家对由原val负载确定，属于描述性干预。原路由优于这个同权重
固定替代，说明路由与专家有共同适配，不能说Router完全无用；打乱
图片证据的差值较小且区间跨0，没有足够证据说明图片匹配本身带来
稳定的净收益。只写CLS近似specialist-off、只写Selector近似原始，
支持当前专门分量作用主要发生在Selector位置；它不是重训CLS-only
方法成功的证据。

## 8. 下一步：停止raw-lane形式，改变专家处理对象

按上一轮退出条件，“输出足够而mAP无收益”已经触发。当前交付继续
保留B0，暂停v3扩seed、扩池、升LR/输出scale和跨任务旧专家复用。
本轮不再用test搜索关闭分支或融合系数。

若研究目标仍是三视图共享参数与内容路由，建议下一次只验证一个新的
结构假设：**图片证据进入专家，专家仅写任务CLS**。保持当前块、
MSA/Drop & Replace后FFN并联加法位置、Frozen Forward和旧池冻结。

```
e = LayerNorm(mean_over_selectors(selected_image_evidence))
g = view_router(e)                       # 沿用当前soft Top-2
delta_cls = 0.1 * (0.5 * shared(e) + 0.5 * sum_i g_i * specialist_i(e))
task_lane_CLS = task_lane_CLS + delta_cls
```

Selector token保留MULTI-LANE原路径，MoE分支不写它们。与当前v3的
区别是专家输入从raw task lane变为该块实际图片证据，输出范围从11个
任务token缩为CLS；它也不同于历史改图像key/value的静态MoE或最终
特征之后的ParaX校准。现有静态value-only诊断及多次后置校准未形成
稳定收益，不建议原样重启这些缺少推进证据的配置。

执行顺序建议：

1. 先做零输出identity、RNG/输入样本对齐、专家输入跨图片变化检查，
   只记录`delta_CLS/input_CLS`、CLS输出变化与分支消融；废除全token
   范数比≥0.25作为性能晋级条件，不主动用归一化或放大系数追这个比值。
2. 用相同图片证据和CLS输出接口比较B0、普通shared-only、固定视图
   Top-2和dynamic；主超参数及训练预算相同。普通共享控制决定新增
   图片Adapter是否有价值，动态与固定控制决定Router是否有独立收益。
3. 首轮只做task0–2固定人群validation，不自动再跑8任务/test。
   fused、Person及阶段平均应不退化，specialist-off应造成可重复下降，
   dynamic应优于普通共享与固定路由；同时查看来源与可靠性子群。
4. 通过后才扩大完整8任务与多seedvalidation，再锁定test报告。
   若普通共享已解释全部收益，就采用普通共享；若新接口仍没有收益，
   结束当前MoE方向，回到B0，不把“再调一次Router”作为默认下一步。

新接口目前未实现或训练。它是本轮诊断支持的可检验方向，不能预先
承诺有效，也不应把现有checkpoint上关闭Selector的分数当作新模型结果。

## 9. 复核产物

- 正式批次：`./output/moe_task_ffn_v3/formal_<mode>_seed0_20261016_01/`，
  包括full_val/full_test及两专门组full_val_specialist_off；日志在同名logs。
- 本地汇总与完整曲线/逐类/子群/区间：
  `./output/moe_task_ffn_v3/analysis/analysis.json`、`analyze.py`、`analyze.log`。
- 只读干预、token能量分解、精确重放与bootstrap：
  `./output/moe_task_ffn_v3/analysis_task7_val_20261017/`；诊断脚本
  `./output/moe_task_ffn_v3/analysis/probe_task7_val.py`。
- 图：诊断目录`v3_results.png`/`v3_results.svg`，绘图脚本在本地analysis。
- 148份正式产物SHA已核对；所有AP由保存的actual probabilities按仓库
  原排序实现独立复算，与正式JSON差低于1e−10。未修改模型业务代码，
  未运行新训练；本轮仅维护分析报告与三个根上下文文档。
