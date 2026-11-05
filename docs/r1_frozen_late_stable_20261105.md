**第6层数值恢复与第9层跨视图Attention对照**

日期2026-11-05；状态实现，待远程测试与smoke。目标是补齐图像CLS Person-only和Person+Face两配置，再验证消息后移能否改善最终融合；不重训R1／Pool／第6层已完成的三条路径。

**诊断及恢复边界**

原CLS Person在task3 epoch19跳3步；epoch18有完整模型/Adam/scheduler/scaler/RNG，0skip、2808次当前任务更新。固定该模型权重，用保存RNG重建epoch19原TRAIN增强前8和末16批，比较AMP loss scale131072/4096/1及FP32。24×4试验均有限，未重现动态训练失败，4096与FP32余弦最低约0.9985；这不能唯一定位故障算子，也不能保证后续每批安全。

从epoch18开始固定损失缩放4096、growth_interval=1e9；保留AMP前向、模型、监督、LR、优化器、调度器、batch64。此前任务和前18轮保持原数值策略，后续采用新策略，明确标为数值稳定续训而非原动态scaler轨迹。任何新skip仍停止该路径。严格核对外部配置和优化器参数顺序，拒绝epoch19或任何含skip恢复状态。复用task0–2的模型和预测文件保留原来源；不改源checkpoint。

**第9层位置对照**

| 条件 | 原位置 | 新位置 |
|---|---|---|
| 注入 | 第6完整Frozen block后，index5 | 第9完整Frozen block后，index8 |
| 首次读取 | Task block7的Summarize | Task block10的Summarize |
| 其余任务读取层数 | 6层 | 3层 |
| Query | 接收方图像CLS或196patch | 同接口，取第9层特征 |
| K/V | 自身图像CLS+另两路原冻结patch | 同接口，发送特征也取第9层 |
| 输出及预算 | 图像CLS或各patch残差；总范数≤.05×原图像CLS范数 | 不变 |

第12层后无后续Task读取点，不作为候选。第9层既缩短消息经过的固定图像变换，也保留三层Summarize/Task MSA吸收信息；语义更深和传播更短可能有利，也可能使CLS消息来不及转化为patch证据。本轮只检验该候选，不预设后移有效。

两种粒度各有Face-only、Person-only、Person+Face，六配置按预定开关组成；实际各训练Person/Face两条，共四路径。发送证据始终来自交互前冻结patch，无效Face在Person K/V softmax前屏蔽。Full保持原R1。原Selectors、Prompts、Drop & Replace、每路head和固定logits融合保留，新增模块任务私有并冻结旧模块，Frozen后缀保留输入梯度。

新第9层四路径全程固定scale4096。旧第6层有效组三路径没有跳步，继续复用，但scaler策略不同；不能把新旧差异全部归于位置。若第9层出现清楚收益，再补匹配数值策略的第6层控制，不立即重复所有旧训练。

**统一正式配方**

- EMOTIC历史train/val/test、samples_intersect_seen_classes；seed1，8任务5+7×3，30epoch/task，B64。
- CLIP ViT-B/16冻结权重；10 Selectors／10 Prompts／5层Prefix；image-token Adapter index1／b32／scale0.1／ReLU，原各视图增强与Face筛样。
- Adam／WD0，主LR.0125，Adapter/交互.0004，cosine无warmup；本视图BCE，Adapter-ASL9.8/0/.05，不加融合BCE。
- 交互768→64／4头，投影后无仿射QK LN、零out、相同总修正额度；loss scale4096，禁止增长，skip仍判失败。
- 可靠Face logit权重.64/.16/.20，否则.8/.2/0；统一sigmoid。完整8阶段Val/Test，不按Test选checkpoint或scale。

**执行与产物**

仅GPU0，DAG最多2个训练/评估进程并按显存准入；恢复训练和第9层训练可重叠。第6层恢复成功后优先验证并生成completed_layer6_results.json；某个训练失败只阻断自己的验证，其他成功组继续，不伪造六组完整。

源目录../multi-lane-main-r1-frozen-cross-view-20261105/output/r1_frozen_cross_view/seed1_20261105，参考../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02。数据../multi-lane-main/datasets/EMOTIC，CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt，manifest沿用../emotic_benchmark_runs历史来源。新输出output/r1_frozen_late_stable/seed1_20261105，日志logs/r1_frozen_late_stable/seed1_20261105，不覆盖旧结果。

入口bash scripts/emotic/run_r1_frozen_late_stable_gpu0.sh。先新专项及Frozen/R1回归，再真实epoch18两更新续训smoke、四条第9层路径各2task×2update的Val64 smoke及零恢复/旧路径/梯度/八lane检查，全部通过后正式启动。恢复smoke的截短epoch19仅用于预检，绝不载入正式恢复。

正式报告各组8阶段mAP、平均与可靠性子群，保留异常/来源SHA和完整恢复状态。第9层若提高两个Test mAP，再核对Val、实际任务表示扰动、时延和同输入Pool对照；不以注意力图或某一路AP替代融合指标。
