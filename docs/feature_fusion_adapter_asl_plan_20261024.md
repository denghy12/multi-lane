# 原特征融合与Image-token Adapter-ASL的匹配实验

## 为什么重做这条对照

用户提供的八组叠加表来自90%TRAIN fit、视图辅助BCE=0.1、所有参数BCE的实验。Adapter开启不等于Adapter-ASL开启。后续ASL迁移已经做过特征融合，但改成全TRAIN、aux=0、独立采样器和低固定AMP scale；不能把它当作原融合结构、原监督只替换Adapter目标的唯一结论。

目前旧标准Full已经在三个seed完整复现：seed0 test32.536519/39.144491/cF132.575617/oF149.382716，val42.379911/49.265161；seed1 test32.134682/39.020979，seed2为31.797277/38.611898。该事实来自远程正式completion与summary，完整88份非checkpoint产物已同步并逐SHA核验；不能继续称三组运行中。

本轮目标是保留原Task Forward、原固定特征融合，直接验证Image-token Adapter使用ASL时，恢复三路直接监督能否提高融合效果。没有Attention、人物位置、MoE、ParaX、额外路由或新融合网络。

## 四组配置

| 组别 | 视图与融合 | Selectors/Prefix-Tuning/head目标 | Image-token Adapter目标 |
|---|---|---|---|
| FULL_ASL | 单Full | BCE(Full) | ASL(Full) |
| THREE_BCE_AUX | 原固定三视图特征融合 | BCE(fused)+0.1 mean(BCE(views)) | 同左 |
| THREE_ASL_NO_AUX | 同上 | BCE(fused) | ASL(fused) |
| THREE_ASL_AUX | 同上 | BCE(fused)+0.1 mean(BCE(views)) | ASL(fused)+0.1 mean(ASL(views)) |

Face辅助损失只用face_reliable=True的样本，无可靠Face的batch只对可用视图取均值。固定融合仍为可靠0.64/0.16/0.20，不可靠0.8/0.2/0。每个任务一套Selectors、Prefix-Tuning和Adapter，三视图共享；按原MULTI-LANE任务冻结机制保护旧任务。

Full强参照不是被较弱共享模型替代的对照。FULL_ASL新入口短跑必须与不可修改的f1ada22原生旧runner模型所有张量逐值相同，之后正式8任务再与完整归档比较。旧独立三视图33.3392/40.2855另作精度与成本参照，不把预测重组33.5724当作新模型成绩。

## 共同训练条件

- EMOTIC全TRAIN，seed0，顺序8任务，每任务30epoch，B64，共13950成功更新；不拿八组旧表的90%fit模型填入匹配控制。
- CLIP ViT-B/16原权重，Frozen Forward不改。Adapter在索引[1]的图像证据读取支路，瓶颈32、ReLU、residual scale0.1，任务独立初始化。不是Frozen骨干连续改写。
- Adam每任务重置，主LR0.0125、Adapter LR0.0004，weight decay0，cosine，旧global RNG采样，workers2，AMP scale65536/growth2000，TF32与deterministic设置沿旧入口；OMP/MKL固定72。原配对增强margin0.15、person jitter0.1/概率0.2、Full legacy crop0.05–1。
- 保存compact及评估前完整优化器/scheduler/scaler/CPU-CUDA/Python/NumPy随机状态。历史采样器使用global RNG，sampler_generator=None是预期配置。
- 先进行四组2任务×2更新短跑，旧原生Full独立对照，检查所有参数等价、路由损失梯度、冻结、完整恢复保存、validation重复评估；不以短跑mAP筛组。
- 正式GPU0至多4进程，显存不足自动排队；所有子进程独立session。完成训练后自动排旧标准8阶段validation与test，导出Full/Person/Face及融合预测；测试不用于挑权重、系数或种子。

## 入口、产物与判断

入口python -m multi_lane.track_a.run_historical_feature_fusion；输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt与原Face manifest。输出output/feature_fusion_adapter_asl/<run-id>，日志logs/feature_fusion_adapter_asl/<run-id>。

正式run-id预定formal_seed0_20261024_01；短跑先01/02，再以smoke_seed0_20261024_03只读接续02评估。主表四列validation最终/阶段平均、test最终/阶段平均，附test cF1/oF1、每阶段人数、各视图端点及更新/冻结审计。全部使用samples_intersect_seen_classes，禁止full-cohort。

先看THREE_ASL_AUX对THREE_BCE_AUX，判断Adapter目标替换；再对THREE_ASL_NO_AUX，判断局部直接监督；最后对FULL_ASL与历史强Full，判断三视图是否有净收益。若Full端点仍削弱而融合不超过强Full，不能称目标实现；若只有val提高而test下降，不能称稳定泛化。aux0.1已是历史系数，首轮不搜索权重。

当前是实现/预检记录，未宣称正式已启动或已有提升。

## 预检进展

447项完整回归通过（1项skip），第一轮四组短跑训练均4更新/0跳步。控制器配置审计误将已解析的eval_workers=2当成None，已修正并补测试；原产物保留，重新完整短跑才允许正式。历史Full三个seed已同步88份非checkpoint产物，三seed全部8阶段logits/人群/标签与历史逐值相同，四指标精确匹配。

第二轮短跑Full与原生f1ada的两任务所有方法参数逐值等价；其余三组训练/冻结/完整边界检查通过。CPU72与eval workers0组合开销过大，已停止02评估并保留训练，新增03仅复用02权重做workers2评估重放；正式也使用旧workers2。没有改变训练或预测规则。正式Full四指标增加精确锚点守卫。
