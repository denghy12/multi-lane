# Face-only Attention 的 Query 来源对照

用户明确要求只训练新三组，原R1与Face-only Pool不重跑。本批seed1，GPU0，EMOTIC历史划分，8任务(5+7×3)，每任务30epoch，batch64，Adam/WD0/cosine，主LR0.0125、Adapter及交互LR0.0004。CLIP ViT-B/16冻结；原生Face的Adapter b32/index1/scale0.1，10 Selectors、10 Prompts、前5层Prefix，Face本视图BCE与Adapter-ASL(9.8/0/0.05)，沿用valid&&!ambiguous训练筛样。没有新增融合训练loss、教师、换头、位置约束或门控。

仅训练三条Face路径，均从标准CLIP和任务参数初始化起点开始：

| 模式 | Query | K/V | 读出 |
|---|---|---|---|
| cls | 本路第6个完整任务块后的Face CLS | 原Face CLS＋Full/Person第6层冻结空间patch | 1份context→共享out→加回CLS |
| selector_mean | 同位置10个Face任务Selectors先平均 | 完全相同 | 同上 |
| selectors | 同位置10个Face任务Selectors分别作Q | 完全相同 | 10份context平均→同一个out→加回CLS |

中层768维状态、64维4头，统一使用原CLSAgent的投影前无仿射LayerNorm，QK按sqrt(16)缩放。三组均为198144交互参数/任务、八任务1585152；同一独立RNG初始化，零out起点，不加投影后QK归一化、额外限幅或额外MLP。Query正常接受Face BCE梯度；外部tokens停止梯度。图像输入不能改变原Selectors的同层状态，但K随图像变化，输出仍可内容相关。Selectors不会被context替换。Face CLS修正后继续原第7–12层及本分支head。

Full/Person既有原生R1预测逐阶段复用，不新训、不重复评估其模型；原R1和Pool基线也从既有分数复算固定logits口径。可靠Face时权重0.64/0.16/0.20，不可靠时0.8/0.2/0，最后sigmoid。检查不可靠人群融合逐值不变；CLS控制完整重放既有Face Attention。既有源预测启动/结束SHA不变。这里是按授权复用独立不受Face训练影响的基础路径，不根据最终分数选择来源参数。

参考 `../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02`；数据 `../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

入口 `bash scripts/emotic/run_r1_selector_query_gpu0.sh`。输出 `output/r1_selector_query/seed1_20261104`，日志 `logs/r1_selector_query/seed1_20261104`。最多三条Face训练并行，GPU空余至少5000MiB才准入；不足自动排队。训练线程72保持历史初始化/训练流，验证线程2、workers2/prefetch1，最多两个验证作业并发。所有正式训练保留原来每任务Test报告和随机流，训练后补完整seen-class Validation。每epoch保存优化器/scheduler/scaler/全部RNG，任务边界存compact与任务恢复状态；失败不取消其他组。

正式前先跑新专项与既有回归，然后每组三个模式各2任务、每任务2更新、Val最多64人的真实smoke，无Test。检查无skip、Q/K/V/out的Adam动量非零、旧任务/冻结骨干保护，确认query数量1/1/10，随后自动启动三组8任务。正式末尾保存task0/2/7固定Val首64人的查询范数、batch/slot差异、来源权重、熵与残差/CLS比例，用于后续归因，不按探针挑结果。

预期产物：三组各八任务完整训练/Val/Test，固定logits融合及可靠/不可靠Face人群mAP、逐类AP和阶段平均；原R1/Pool同口径对照，来源保护与CLS重放审计，查询探针。只有融合最终与阶段平均提高并优于已有Pool，才支持新Q产生方法收益；本轮不自动扩其他KV或Full写回。
