# 共享B0的Face CLS跨视图读取：三组配对实验

2026-11-05。用户已授权执行。分支`exp/shared-b0-face-cross-view-20261105`，基于58db7f7创建独立工作树，不覆盖现有用户改动。本文件记录预先定义的训练流程，此时尚未得到正式结果。

## 研究问题与三组配置

比较B0、B0+Face-only Pool、B0+Face-only Attention，首先回答能否改善共享B0，不将R1/B0不同配方的差值全归于共享。三个模型均自行学习当前任务的Adapter、Selectors、Prompts和共同分类头；不安装A/B权重、不换入历史B0头、不蒸馏。

三路分别进行Frozen Forward和Task Forward，共用冻结CLIP及本任务适配参数。在第6完整Task Forward块之后，只有可靠Face的任务CLS接收修正，随后运行第7—12块。Full/Person提供同一次前向第6完整Frozen块之后的空间patch，不增加另一套证据裁剪。Attention以Face任务CLS作Q，自身原CLS+Full/Person各196patch作K/V。Pool分别对同样两组外部patch做Value投影和平均，与归一化Face CLS拼接，经GELU-MLP产生修正；Pool没有QK匹配。

两个读出均768→64、来源嵌入2×768、零输出投影、统一平滑范数上限0.05×停止梯度原Face任务CLS范数。Attention4头，投影后Q/K逐head无仿射LayerNorm，FP32读取；Pool相同FP32数值路径及相近参数预算。Attention198144参数/任务，Pool198784参数/任务，差640；公共Value与来源嵌入初始化一致。初始化用CPU fork_rng，不消耗原B0随机流。模块按任务保存，旧模块冻结；全图骨干权重始终冻结。

Full/Person不接收直接修正，但Face及融合监督仍可更新共享参数，因此不要求它们在共训后重放原B0。单个checkpoint上开关模块时，应保证Full/Person及不可靠Face的融合预测保持；不能将OFF模型当成重新训练的B0。

## 完整训练配置

| 项目 | 配置 |
|---|---|
| 数据 | EMOTIC历史train/val/test划分，samples_intersect_seen_classes历史评估人群 |
| seed/任务 | seed1，8任务，类别5+7×3 |
| 训练 | 每任务30epoch、batch64、固定最后一轮；预计13950有效更新/组 |
| 冻结权重 | 原CLIP ViT-B/16；没有其他已训练教师或路径供体 |
| Adapter | image_token，index1，b32/ReLU/scale0.03；独立任务初始化；只用于摘要证据，不写Frozen Forward |
| 任务适配 | 三视图共享10 Selectors、10 Prompts、前5层Prefix；按原B0传递下一任务起点 |
| 优化 | Adam每任务重建，主LR0.0125，Adapter及读出LR0.0004，WD0；cosine/no warmup/min LR0 |
| 监督 | 主参数/读出：融合BCE+0.1有效视图辅助BCE；Adapter：对应ASL+0.1辅助ASL，neg9.8/pos0/clip0.05；Face辅助仅可靠Face |
| 增强 | Full原legacy裁剪scale0.05—1；Person margin0.15/letterbox；关闭Person颜色扰动 |
| 数值随机流 | AMP初始65536、TF32、初始化与训练72CPU线程；原sampler/增强随机流 |
| 融合 | 各路归一化任务特征，可靠Face权重0.64/0.16/0.20，否则0.8/0.2/0；加权后共同线性head，无再次归一化 |
| 评估 | 完整8阶段Validation/Test；B64/workers2；独立评估4CPU线程，避免72线程多进程争用 |
| GPU | 仅GPU0；三组显存准入并行，空间不足自动排队，保持B64 |

## 路径、入口和预期产物

全部相对服务器独立工作树根目录解析。

- 输入数据：`../multi-lane-main/datasets/EMOTIC`。
- 权重：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 历史参照仅供只读核验：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- 入口：`bash scripts/emotic/run_shared_b0_face_cross_view_gpu0.sh`。
- 输出：`output/shared_b0_face_cross_view/seed1_20261105`；预检同级`seed1_20261105_smoke`。
- 日志：`logs/shared_b0_face_cross_view/seed1_20261105`。
- 产物：每任务compact及report前完整恢复状态；新增组每任务latest epoch模型/Adam/scheduler/scaler/RNG；完整Val/Test各阶段预测、逐视图及可靠性人群指标；summary.csv；任务0/2/7完整Val/Test关闭交互归因。

## 启动门槛与结果解释

先跑新增单测及历史B0/摘要Attention回归；再顺序跑三组各2任务×2更新的真实数据预检，只用Val。B0须逐值重放历史已验证smoke的loss、所有任务参数和预测；检查同初始化、同sampler RNG、ViT无梯度/无变更、旧任务参数冻结。两新增组用固定TRAIN64测Q/K/V/out有效有限梯度，检查Full/Person与不可靠Face开关预测一致。预检实测reserved显存×1.25+1024MiB准入，另留1024MiB；最多3本队列进程，不操作其他任务。

正式B0重训后对照历史完整8任务Test预测，失配必须记录并暂停新评估调度，不能填回32.6800。三组主结果必须使用自身训练head，报告Val及Test最终和阶段平均；Attention还须与同证据Pool比较。OFF仅用于区分即时消息与共同训练作用，三个边界的均值不冒充八阶段平均。未满足条件不自动扩层、增加Query、换K/V或启动其他seed。
