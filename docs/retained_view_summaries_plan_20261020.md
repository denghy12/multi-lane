# 联合残差条件贡献与三组摘要保留

## 已有模型只读验证
目的：检验融合残差对共同训练的任务路径的作用。EMOTIC完整validation2397人，边界task0/2/7。
来源joint_view_summary/formal_seed0_20261020_01的dense/moe，f057e71，CLIP ViT-B/16。
原训练seed0、8task各30epoch、B64、Adam、主LR.0125、融合LR.0004、weight decay0、cosine、aux0。
Dense宽32；MoE每任务每层4名宽16专家softTop2；12层全部融合，scale.1。
本轮无优化器、无训练、无test；B64、workers0、AMP/TF32。将所有已见任务层的scale在内存设0，其余权重不变。
GPU5运行dense，GPU6运行moe，分别on→原预测逐数组精确重放→off。先SMOKE=1仅task0完整val，再正式三边界。
入口GPU=5 ARM=dense bash scripts/emotic/run_joint_residual_attribution.sh，GPU6对应moe。
输入相邻worktree原checkpoint/EMOTIC数据/既有Face manifest，CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt。
产物output/joint_residual_attribution/<run>/<arm>，日志logs同名；含逐实例分数、路由和mAP、来源SHA。
离线报告可靠Face分组、原图分组500次配对bootstrap；不把三边界均值叫8阶段均值。
关闭残差模型仍保留共同训练的Selector/Prompt/head，不等同重训固定融合或B0。

## 后续候选
同一组任务Selectors分别读取Full/Person/Face，各10个摘要保持独立；同一个Task CLS读取30个摘要。
无效Face的Key/Value必须在softmax前屏蔽；Drop & Replace放回原10个Selectors，FFN仍处理11个状态token。
无Expert/Router/可学习视图embedding，无跨任务可变共享模块，Frozen Forward及分类头位置保留。
注意力序列11→31，不能预先宣称提速；取消提前平均，也改变注意力中三视图的隐式先验。
先task0–2的TRAIN原图分组holdout与监督匹配对照；不自动开启8task或test。正式训练配置启动前补齐。

## 三组摘要保留的先导配置（启动前锁定）

三臂均重新训练：baseline_noaux保留三路Task Forward及末端固定融合；fixed单路径提前固定平均摘要；retained单路径保留三组摘要供注意力读取。
三臂均禁用Expert、Router、ParaX和image-token Adapter，aux0，仅联合BCE；不与含aux.1的历史B0作单因素归因。
数据EMOTIC，seed0，task0–2（类别数5/3/3），每任务30epoch，B64；三臂共同使用固定TRAIN原图分组约90/10 fit/holdout。
盐emotic-reliability-calibration-v1，按原图SHA256固定划分，所有任务/视图相同规则；holdout图片任何person都不参与任何任务更新。
--calibration-fraction 0.1 --save-calibration-scores，逐任务保存fit/holdout完整ID与跨任务无泄漏核验。
与历史全TRAIN实验人群不同，三臂都重跑；不将历史全TRAIN曲线当新候选的严格匹配控制。
Adam，主LR.0125（source .05×64/256），weight decay0，每任务cosine无warmup/min0，30epoch；AMP初始scale1024/growth1e9，TF32。
每层10个Selectors，前5层10个prefix prompts，共12层；CLIP ViT-B/16冻结；head为共享post-fusion，pre-head归一化。
retained无额外融合参数，不使用expert数/融合LR/scale等保留CLI占位参数，config中effective专家数0/residual false。
Full使用原随机crop(.05,1)，Person letterbox margin.15，无Person jitter；Face来自既有manifest可靠性，Face无效则softmax前mask。
workers训练2、评估0；内部逐epoch validation关闭，任务结束保存完整可恢复状态后再报告seen-positive val及TRAIN holdout。
每臂完成后外置完整validation2397人、三个边界均保存预测；不加载test。固定30epoch，不用val选最佳epoch。
GPU5 baseline_noaux→fixed；GPU6 retained，两条队列并行；先三臂各2task×2更新smoke并做checkpoint精确重放，再正式。
入口RUN_ID=... bash scripts/emotic/run_retained_view_summaries_batch.sh；smoke加SMOKE_UPDATES=2。
输入../multi-lane-main/datasets/EMOTIC，CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt，Face manifest同旧实验。
日志logs/retained_view_summaries/<run>，结果output/retained_view_summaries/<run>/<arm>/{train,full_val}。
产物含3份compact、3份验证前完整恢复、holdout/val预测、训练history、初始化/样本配对、冻结CLIP及旧任务保护核验。
停止规则：不自动扩8task/test；仅当候选在holdout与完整validation对匹配三路/单路控制显示一致改善，再讨论完整训练。
可用性与端点信息保留是这次变量；无新自由残差，注意力token数变化及取消固定视图先验都要一并报告。

## 执行记录：2026-10-21

只读两臂三边界完成，on逐值重放、off残差范数全为0、六源SHA及用户主树/test-only状态SHA不变。
结果见docs/joint_residual_attribution_results_20261021.md。
379项全测与smoke三臂各2task×2更新/0跳步、所有配对/holdout/重放核验通过，批次exit0。
正式pilot_seed0_20261021_01已在GPU5/6启动，源22d1ae0，tmux ml_retained_summary_seed0_20261021。
服务器独立pilot树detached该commit且clean，后续文档更新不修改运行树。
三臂各3task×30epoch、4560更新；只holdout/validation，正式结果尚未产生。
