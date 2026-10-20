# 联合融合残差只读归因：seed0 task0/2/7 validation

## 做了什么

使用原dense和moe checkpoint，在同一批完整validation2397人、1705张原图上，比较原预测与关闭联合融合残差。
关闭方式为全部已见任务、全部12层的联合残差scale在内存设0；Selector、Prompt、分类头和固定摘要融合部分保留。
GPU5/6，B64、workers0、AMP/TF32；无训练、无test。原预测逐数组逐值重放，六个源checkpoint的SHA开关前后相同。
执行代码3ac8fbf；离线500次原图分组配对bootstrap。区间只反映这些固定seed0模型的样本不确定性，不反映训练随机性。

## 结果

单位为mAP百分数，变化为“关闭−开启”；三个边界不是八阶段平均。

| 融合方式 | 任务边界 | 开启残差 | 关闭残差 | 变化 | 变化的95%区间 |
|---|---:|---:|---:|---:|---|
| 普通融合器 | 0 | 49.9872 | 38.7399 | -11.2472 | [-14.3211, -7.9551] |
| 普通融合器 | 2 | 40.3464 | 32.6391 | -7.7074 | [-9.3688, -5.8344] |
| 普通融合器 | 7 | 38.5627 | 36.0553 | -2.5074 | [-3.6834, -1.4767] |
| 联合融合MoE | 0 | 48.3830 | 38.7092 | -9.6738 | [-12.6462, -6.7273] |
| 联合融合MoE | 2 | 39.8763 | 35.4139 | -4.4624 | [-6.2228, -2.9350] |
| 联合融合MoE | 7 | 38.6142 | 38.3423 | -0.2719 | [-1.3254, +0.6584] |

## 这些结果说明什么

联合残差确实参与了学习，而且早期任务路径已经依赖它。不能再说专家分支完全没起作用。
但这并不等于它改善了三视图融合：此前相同监督、独立重训的固定融合组，validation/test仍优于这两个可学习融合组。
关闭残差保留了与残差共同训练的Selector、Prompt和head。突然拿掉它后下降，只能说明当前解对它有条件依赖，不能证明“加入它比从头不加入更好”。

MoE最后总体仅下降0.2719，区间跨0，也不能理解为它最后自然失效。
在task7 checkpoint按类别所属训练任务拆开：task0类别关闭后−9.6350，task5类别+4.6942，task6类别+8.0022。
这些是描述性差值，尚未给子类别组区间；它们说明总体均值存在正负抵消，不能推断每条任务路径都无收益。
普通融合器也有类似方向差异，task0−11.2018，task6+5.6276，整体仍下降2.5074。

可靠Face与不可靠Face两组中，早期关闭残差都下降；MoE task7两组的区间也都跨0。
因此不能把此次现象统一归因于无效Face，也不能用更改专家调用数或冻结数代替结构验证。

## 机制层面的解释与边界

当前方法先把三组Summarized Patches压成一组，再通过自由残差改写这份证据，交给冻结的任务注意力读取。
Selector、Prompt与head会适应这种新证据，专家产生的内容遂成为当前预测依赖的一部分。
这些结果与“共同适应后依赖新增改写、却泛化较差”的解释相容；没有直接证明具体哪一维证据丢失、或哪一层语义几何被破坏。
冻结旧任务能保住已学的函数，无法保证这个函数利用三路信息的方式本来就好。
路由随图片改变也仅说明选择机制运作，不保证被选专家的组合比普通注意力更合适。

## 已执行的下一步

已实现不增加Expert的三组摘要保留版本：同一任务Selectors分别读取三视图，保留各10个摘要，由唯一Task CLS直接注意力读取。
无效Face在softmax前屏蔽；随后保留原Drop & Replace和FFN，每任务持久状态仍为1个CLS+10个Selectors。
这一版取消自由融合残差，同时改变摘要压缩及注意力隐式视图先验，不能将效果单独归因于一个因素。

先导三臂：原三路结构aux0、单路径固定摘要、单路径保留三组摘要；都重新训练seed0 task0–2，各30epoch/B64/Adam LR.0125/cosine/WD0。
共同按TRAIN原图固定90/10 fit/holdout：task0 4854/499，task1 3972/422，task2 782/79；每epoch76/63/13步，正式每臂4560成功更新。
只使用validation；GPU5顺序两个控制，GPU6候选。379项测试通过；三臂2task×2更新真实smoke、旧参数/CLIP保护、TRAIN原图无泄漏与跨臂配对、checkpoint重放全部通过，批次exit0。
2026-10-21已启动正式先导pilot_seed0_20261021_01，tmux ml_retained_summary_seed0_20261021。
GPU5当前baseline_noaux，完成后自动fixed；GPU6当前retained。完整先导指标尚未产生，不报告精度收益。
运行源22d1ae0，服务器pilot worktree clean detached该commit；后续本地文档提交不修改运行服务器。
43份归因产物、86份smoke/预检产物已同步本地且逐文件SHA核验。
本轮不自动扩8task、test或seed，不合并main。

## 产物与代码

- 结果与逐实例分数：`output/joint_residual_attribution/formal_seed0_20261020_01/`
- 主分析：同目录`analysis.json`；类别组拆分：`task7_class_groups.json`。
- 日志：`logs/joint_residual_attribution/`。
- 新结构代码：`multi_lane/track_a/{joint_summary,model,runner}.py`；新机制测试`tests/test_track_a_retained_summary.py`。
- 启动器：`scripts/emotic/run_retained_view_summaries_batch.sh`；完整预锁配置见`docs/retained_view_summaries_plan_20261020.md`。
