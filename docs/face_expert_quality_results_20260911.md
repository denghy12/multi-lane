# 独立Face expert质量validation结果（2026-09-11）

## 完整性

- batch：`face_quality_seed0_20260911_001`
- 三组均完成seed0、8 tasks×30 epochs，退出码全0；每组8,610 optimizer updates、0 skipped，
  无OOM、Traceback或非有限值。
- `selection_split=val`、`test_accessed=false`；未保存checkpoint。
- Full、Person和融合规则完全冻结。可靠Face使用`[0.64,0.16,0.20]`，无效Face严格使用
  `[0.80,0.20,0]`，没有搜索alpha/beta或threshold。

## 结果

| Face source | Face全量 final mAP | 可靠Face final mAP | 固定R1 final mAP | R1相对锚点 |
|---|---:|---:|---:|---:|
| 旧baseline m15 | 33.3521 | 36.3096 | **43.5812** | — |
| reliable m15 | **33.4910** | **37.3942** | 43.4933 | -0.0879 |
| reliable m05 | 32.7851 | 36.2327 | 43.3646 | -0.2166 |
| reliable m05 + jitter | 33.2895 | 36.4197 | 43.3363 | -0.2449 |

三组均未通过“可靠Face和固定R1 final mAP各至少+0.05”的双门槛，不补seed1/2、不运行test。

`reliable_m15`是Face自身唯一明确提升的候选：可靠子集`+1.0846`，全量Face `+0.1389`，但固定
R1反而`-0.0879`。它的R1 cF1/oF1相对锚点提高约`+0.4929/+0.2412`，说明质量过滤改善了0.5
阈值附近的分类，却没有改善融合后的跨样本排序。

`reliable_m15`的R1逐task mAP相对锚点为
`-0.1677/+0.0572/+0.0046/-0.0001/-0.0272/-0.0696/-0.0946/-0.0879`。最终类别中，Annoyance/
Suffering/Disapproval约`+1.1415/+0.8472/+0.5448`，但Sadness/Aversion/Anger约
`-1.6888/-1.0735/-0.8381`，类别交换抵消了Face自身平均增益。

## 解释

1. 训练过滤从12,442个valid非歧义样本收紧到9,696个融合可靠样本，去除了噪声并明显提高可靠
   Face自身指标；但训练更新从旧锚点10,980降到8,610（少21.6%），当前比较同时改变了样本质量
   和优化量。
2. Face自身AP提高不保证固定概率融合提高。新Face在部分样本/类别上的排序和置信尺度发生变化；
   它可能修正Face单路错误，却扰动Full+Person已经正确的排序。
3. margin从0.15缩到0.05后Face与融合都下降，说明EMOTIC情绪不仅依赖眼口局部；头发、头姿和脸周
   上半身语境是有效信息。停止继续缩紧crop。
4. ColorJitter使Face全量oF1从53.07提高到56.04，但可靠mAP只提高0.11、R1下降0.245，说明它主要
   改变阈值校准而非产生稳定排序信息。停止继续调基础颜色增强。

## 下一步

先做一个单组、低成本公平性控制：保持`reliable_m15`所有设置不变，把每task实际optimizer updates
对齐旧baseline的`[1920,1620,360,3570,1710,960,240,600]`，scheduler也按对应更新量完成一次
cosine。它能区分“可靠过滤无助于融合”和“过滤后少21.6%训练量导致欠拟合”。仍只跑seed0
validation，固定R1；只有可靠Face和R1 final mAP同时超过锚点至少0.05才补seed1/2。

如果等更新量仍不能改善R1，应结束crop/filter/基础增强路线。下一项应换Face专用表征：优先采用
AffectNet/FERPlus等面部表情预训练encoder（冻结主干），利用manifest五点landmark做对齐，接
MULTI-LANE task-specific轻量投影/Adapter；margin0.15保留头部上下文。该方案能提供不同于通用
CLIP Full/Person的表情特征，理论上比继续微调相同CLIP输入更可能增加互补信息。

## 产物

- 本地：`output/emotic_track_a_face_quality/face_quality_seed0_20260911_001/`
- 压缩包：`output/emotic_track_a_face_quality/face_quality_seed0_20260911_001.tar.gz`
- SHA-256：`8c869470cfe7d1241ffd2acd668304bfd471fd4a14ba593e71cda74895121757`

