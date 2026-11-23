# 在补足更新的90%分支上重算四种CARAN式融合

## 目的和范围

用户要求将旧12,570更新90%分支上的四种融合读出，迁到已完成13,950更新的同名单90%分支，重新用该模型自己的10% TRAIN留出预测训练融合，并完成八阶段Validation和Test。复用分支，不移用旧融合参数，不重训视觉网络。比较任务常量、类别常量、人物共享类别和人物逐类别四种结构；保留同源固定/等权读出。

## 配置

- EMOTIC历史划分、seed1、8任务，原随机原图名单：11,623张分支TRAIN、1,292张融合留出，同图人物及视图不跨组。
- 源分支：`../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal`。源训练batch64、30轮/任务、Adam/cosine、主学习率0.0125、Adapter及Frozen9交互0.0004，实际13,950有效更新、零跳步。
- 融合只用每任务引入时合法、至少一个当前类别为正的留出TRAIN人物；使用自然图，不加入模糊版本，不扩大人群。
- 两个人物读出均使用分类头实际接收的原生最终Task CLS，每路512维、拼接1536维；无压缩、无额外相似度计算。
- 四组全零初始化，即可用视图等权；每任务全批Adam、学习率0.01、BCE、300步，无scheduler/额外正则；沿用历史`train_head`数学。
- Face无效时输入置零、权重屏蔽。分支特征和logits固定，旧任务融合头封存；不使用Validation/Test标签更新参数。
- 按原样本次序、batch64和AMP评价口径提取；逐值检查sample IDs、标签、Face mask及三路logits。检查源checkpoint、config、cohort、历史有效更新及哈希。
- GPU0两条独立任务提取流共享四种方法的特征；拟合和排序评估使用CPU。Validation完成并锁定后运行Test。保留人物权重同组均值及100次错配诊断，均不是新训练候选。

输入数据沿用`input_paths()`：`../multi-lane-main/datasets/EMOTIC`、原Face清单和`../CODE_DDP/pretrained/clip/ViT-B-16.pt`。checkpoint只读。

## 入口与产物

独立分支`exp/d9-caran-equal-updates`。入口`bash scripts/emotic/run_d9_caran_equal.sh smoke`，通过后运行同入口`formal`。输出`output/d9_caran_equal_updates/seed1_20261123/`，日志`logs/d9_caran_equal_updates/seed1_20261123/`。预期24份共享原生特征、八个任务的四组融合参数、TRAIN诊断、八阶段Val/Test结果与来源核验。

同源固定参照：Val43.7028/51.2535，Test32.7557/39.9983。全TRAIN固定33.2019/40.2847仅另列强参照。正式表不能混入旧90%32.4573/39.9684。

当前为已声明执行协议，待预检及正式结果；不根据Test更改设置。

## 预检通过

代码c90ce6e：8项历史CARAN回归通过，task0/task7各64人TRAIN与Val真实前向逐值重放。TRAIN源缓存不可用Face分数已置零，提取器显式遵循相同规则；有效分数完全一致，源模型没有改动。正式过程将复核完整8任务来源并封存四组权重。
