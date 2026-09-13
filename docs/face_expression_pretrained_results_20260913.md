# Face 表情预训练专用表征：seed0 validation 结果

## 执行完整性

唯一batch为`face_expression_seed0_20260912_1405`，代码HEAD `733171b`。projection与
bottleneck Adapter均完成8 tasks×30 epochs=240 epochs、10,980次optimizer updates、0 skipped，
退出码均为0；无OOM、NaN、checkpoint或test访问。运行前187项单测、官方权重加载、真实数据与GPU
smoke均通过。

五点landmark审计中，train/val的`valid && !ambiguous`样本为12,442/1,861，有效五点覆盖率为
99.968%/100%，相似变换失败0；4个train样本按协议回退margin0.15 letterbox。官方EmotiEffLib
checkpoint SHA-256为`47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`。

## 结果

| Face source | Face all final mAP | reliable Face final mAP | 固定R1 final mAP | R1相对锚点 | R1 average mAP | R1 forgetting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧CLIP Face锚点 | 33.3521 | 36.3096 | **43.5812** | — | **50.7509** | **0.9039** |
| AffectNet encoder + task投影 | 32.2915 | **35.7820** | **43.3234** | **-0.2578** | 50.5487 | 0.9073 |
| AffectNet encoder + task Adapter | 31.9417 | 35.1329 | 43.2659 | -0.3153 | 50.4695 | 0.9258 |

projection是两个新候选中较好者，但其可靠Face相对锚点`-0.5276`、固定R1 `-0.2578`；Adapter分别
为`-1.1767/-0.3153`。两者均未达到两个指标各`+0.05`的门槛，因此不补seed1/2、不运行test。

projection的固定R1在task0--7相对锚点依次为
`-0.2059/-0.2173/-0.1472/-0.0941/-0.1400/-0.2609/-0.2940/-0.2578`，没有任何task获益。
最终类别中Suffering、Annoyance、Affection分别提高`+0.5615/+1.3060/+1.1896`，但Pain、Aversion、
Anger、Sadness、Fear、Peace、Sensitivity分别下降`-1.9010/-1.8591/-1.3881/-1.3084/-0.8866/
-0.8026/-0.7679`，少数表情相关收益不足以抵消更广泛的语义损失。

## 判断

这不是预处理规范、checkpoint损坏或数值训练失败。官方EmotiEffLib接口对该EfficientNet-B0明确使用
224输入和ImageNet mean/std，本实现与之相同；encoder输出1280维有限特征且完全冻结，两组均无跳步。

主要限制是表示与任务语义不一致：AffectNet 8类表情特征强调Anger/Contempt/Disgust/Fear/Happiness/
Neutral/Sadness/Surprise，而EMOTIC是26类、多标签、强依赖身体和场景的情绪。当前候选又用表情特征
完全替换旧CLIP Face表示，因而丢掉CLIP对Pain、Aversion、Peace、Sensitivity等更宽语义的建模。
feature-level Adapter位于已经池化的1280维特征之后，只能重映射摘要，不能恢复被encoder压掉的信息；
Adapter ASL还比单纯BCE投影更差，说明继续扩大该Adapter不合理。

若继续Face路线，最有依据的下一步不是换Router或再调margin，而是保留旧CLIP Face主路径，仅把
冻结AffectNet embedding/8类logits作为零初始化、task-specific的小残差补入CLIP Face表征。这样可
验证Suffering/Annoyance等新增表情信息能否保留，同时不牺牲旧CLIP的宽语义。仍只做一个seed0
validation候选并锁定R1 beta0.20；若仍不能同时提高可靠Face和R1各0.05，应停止该Face表征路线。
