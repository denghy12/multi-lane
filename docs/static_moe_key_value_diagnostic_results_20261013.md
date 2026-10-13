# 静态 MoE Selector key/value 分离：validation 定位结果

## 结论

现有 checkpoint 的固定 key、只适配 value 前向没有超过原始 key+value 前向，因此不启动 value-only 重训。v1 重叠 MoE 的 value-only 最终融合 mAP 为 42.7546，比原始 42.7858 低 0.0311；v2 为 42.0580，比原始 42.0629 低 0.0049。预先声明的推进条件要求两个版本都至少提高 0.1 点，且 value-only 优于 none、key-only 不优于 none；两版均未通过，控制器记录 `training_started=false`。

v1 的 key-only 比 none 低 0.0184，但原始 key+value 又比 value-only 高 0.0311。这个组合说明“单独改 key 有轻微代价”不能推出“去掉 key 改动有收益”：在已适配 value 时，key 改动与 value 有正向交互。v2 四种前向相差很小，不能定位匹配路径为主要损失来源。

## 范围与可比性

- 实现分支 `exp/static-moe-key-value-diagnostic-20261013`，服务器运行固定提交 `b415322dd68f6165e77f465aa41e838d6502f4ec`。独立服务器 worktree：`/mnt/haoyuan/workspace/multi-lane-main-static-moe-kv-20261013`。
- 固定 seed0、第 30 轮 compact checkpoint；v1 SHARED、MOE_UNIFORM、MOE_OVERLAP 与 v2 SHARED、MOE_OVERLAP，共五来源。每来源八任务，四种前向 `both/value_only/key_only/none`；每次完整 EMOTIC validation 2397 人，共 160 次任务级前向。全程 eval/no-grad/零训练更新，不读取 test。
- 干预仅在原 Selector image-token Adapter 的 key/value 证据路径；所有已训练 Selector、Prompt、分类头、专家和固定融合参数保持同一 checkpoint。四种前向作用于所有已见任务 lane 的三个视图；下面逐样本注意力与残差诊断记录的是**当前任务 lane 的 layer 1**。因此不能用这组当前 lane 诊断直接解释全部旧类 AP。
- `none` 仍保留联合训练出的其他参数，不是独立训练的原始无 Adapter 基线。这里的 key/value 也不是 CLIP 主干注意力投影。
- 配置与预声明门槛见 `docs/static_moe_key_value_diagnostic_plan_20261013.md`。batch：`static_moe_kv_validation_seed0_20261013_01`；服务器/本地输出同为 `output/emotic_static_moe_kv/<batch>/`，日志为 `logs/emotic_static_moe_kv/<batch>/`。

## AP 结果

融合 mAP；平均为八阶段 mAP 算术平均。

| checkpoint 来源 | both 最终 | value-only 最终 | key-only 最终 | none 最终 | both 阶段平均 | value-only 阶段平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 SHARED | 42.2920 | 42.2805 | 42.3470 | 42.3648 | 48.7659 | 48.7446 |
| v1 MOE_UNIFORM | 41.9965 | 41.9955 | 41.9457 | 41.9316 | 48.5228 | 48.5153 |
| v1 MOE_OVERLAP | **42.7858** | 42.7546 | 42.4871 | 42.5055 | **49.3401** | 49.3145 |
| v2 SHARED | **42.6181** | 42.6159 | 42.5915 | 42.5917 | 48.8846 | 48.8852 |
| v2 MOE_OVERLAP | **42.0629** | 42.0580 | 42.0386 | 42.0446 | **48.6191** | 48.6186 |

重叠 MoE 的预声明差值：

| 来源 | 最终 V−both | 最终 V−none | 最终 K−none | 平均 V−both | 平均 V−none | 平均 K−none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 MOE_OVERLAP | −0.0311 | +0.2491 | −0.0184 | −0.0256 | +0.1905 | −0.0084 |
| v2 MOE_OVERLAP | −0.0049 | +0.0133 | −0.0060 | −0.0005 | −0.0536 | −0.0095 |

这里 V 是 value-only、K 是 key-only。v1 value-only−both 八阶段依次为 −0.0469、+0.0161、+0.0008、−0.0262、−0.0409、−0.0420、−0.0347、−0.0311，六个阶段为负。v2 变化约在 ±0.007 点以内。v1 最终四格 AP 交互量 `both−V−K+none=+0.0495`，v2 为 +0.0109；它只是该固定模型 AP 的描述性差分，不等价于重训后的交互效应。

v1 重叠 MoE 去掉 key 改动后，Full −0.0058、Person −0.0342、Face +0.0672、可靠 Face +0.0435、融合 −0.0311 点。v2 对应 Full +0.0114、Person +0.0023、Face +0.0233、可靠 Face +0.0040、融合 −0.0049 点。Face 的局部改善没有变成融合收益。

按原图聚类、1000 次配对 bootstrap 的最终融合 mAP 差值，条件于现有 seed0 模型：

| 来源与比较 | 点估计 | 95% 百分位区间 |
| --- | ---: | ---: |
| v1 V−both | −0.0311 | [−0.0708, −0.0017] |
| v1 V−none | +0.2491 | [−0.0257, +0.5559] |
| v1 K−none | −0.0184 | [−0.0515, +0.0011] |
| v2 V−both | −0.0049 | [−0.0221, +0.0097] |
| v2 V−none | +0.0133 | [−0.1174, +0.1484] |
| v2 K−none | −0.0060 | [−0.0329, +0.0065] |

这些区间只描述固定权重、固定 validation 人群的抽样变化，不包含训练 seed 不确定性，也未做多重比较校正。

## 注意力和残差定位

下表为任务 7 原始 `both`、当前任务 lane 的逐人均值。KL 是改变 key 后的注意力相对冻结 key 的 `KL(current||frozen)`；top10 是选中 token 集合交集占 10 的比例。去均值占比为 `sum(centered residual energy)/sum(full residual energy)`，即 token 均值之外的残差份额。

| 来源 | 视图 | 注意力 KL | top10 重合 | 完整残差/输入能量 | 去均值残差份额 |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 OVERLAP | Full | 0.001395 | 0.9537 | 0.03820 | 7.78% |
| v1 OVERLAP | Person | 0.000550 | 0.9659 | 0.00893 | 11.75% |
| v1 OVERLAP | Face | 0.000332 | 0.9762 | 0.01051 | 6.68% |
| v2 OVERLAP | Full | 0.000100 | 0.9841 | 0.00492 | 4.37% |
| v2 OVERLAP | Person | 0.000212 | 0.9800 | 0.00258 | 15.13% |
| v2 OVERLAP | Face | 0.000131 | 0.9683 | 0.00376 | 6.48% |

注意力分布改变很小，top10 大多保留；残差能量主要在 token 均值分量。前者不证明 key 没有作用，后者也不证明均值分量是多余的：LayerNorm、query 和后续分类器都可能对它敏感。残差统计记录完整候选残差，即使某模式暂时断开它。

## 核验与下一步

服务器 324 项 Track-A 单测全部通过。五来源的真实 64 人、八任务、四模式 smoke 均通过，再进入全量；五个正式评价进程 exit 0。独立检查 160 份完整逐样本诊断与分数：样本 ID、标签、可靠 Face mask 配对一致；原始 logits 与历史 full-val 逐值重放最大差 0；预期相同的注意力模式逐样本差 0；残差能量分解最大相对误差 `3.55e-7`。来源 SHA 前后不变。677 份原始非 checkpoint 输出和日志从服务器同步到本地，路径/SHA 清单哈希两端同为 `6ebee60978092e67f7637c58b56ba36aedb6d40178cfcea7303456fe9b577c9c`。

本轮停在诊断阶段，不实现或训练 value-only。若继续研究结构原因，先对**现有 checkpoint 的 validation**补旧任务 lane 注意力记录，并在同一残差上分离 token 均值与去均值分量；这能检验当前任务诊断是否遗漏旧类及主要能量是否来自均值。任何新训练应另立预声明配置，保留原路由/专家/融合和 B0、普通 Adapter 对照。当前部署或模型选择仍以既有基线及完整 test 证据为准，不能据本轮 validation 重新选 test。

复算脚本与结果：`output/static_moe_kv_analysis_20261013/analyze.py`、`audit_full.py`、`analysis.json`、`full_acceptance.json`；原始 `comparison.json`、`decision.json` 与逐样本 NPZ 在上述 batch 输出。
