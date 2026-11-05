# Frozen Forward中的三视图交互：六组配对实验

本轮固定Full，只研究交互粒度及辅助接收方向。图像CLS和空间patch各比较Face-only、Person-only、Person+Face。日期2026-11-05，分支exp/frozen-cross-view-six-20261105，源于ee5dcbb；实现中，尚无正式效果。

| 组别 | Full | Person | Face |
|---|---|---|---|
| 图像CLS／Face-only | 原R1 | 原R1 | 图像CLS交互 |
| 图像CLS／Person-only | 原R1 | 图像CLS交互 | 原R1 |
| 图像CLS／Person+Face | 原R1 | 图像CLS交互 | 图像CLS交互 |
| patch／Face-only | 原R1 | 原R1 | patch交互 |
| patch／Person-only | 原R1 | patch交互 | 原R1 |
| patch／Person+Face | 原R1 | patch交互 | patch交互 |

**结构假设。** 不再直接把外部context加到Task CLS。先在第6个完整Frozen block之后更新接收方图像tokens，由第7层开始的Selectors/Summarize及后续Task Forward筛选新证据。图像CLS组借鉴CrossViT的CLS代理；patch组是196个位置分别读取的稠密扩展。位置、Q、写回与保护预算相对旧方法均有变化，不能称为纯位置消融。

**Q/K/V与更新。** 图像CLS组Q=接收方图像CLS，patch组Q=接收方196个空间patch。共同memory为接收方自身图像CLS+另外两个视图各196个原冻结patch，共393tokens；不加入自身196patch作为KV。Face读取Full/Person，Person读取Full/Face，Full不接收。Person读取时无效Face在softmax前屏蔽。patch组保留196份输出，不平均为一份CLS修正，也不做相同索引patch直接相加。

**共同尺度。** 768→64，4头，每头16维；输入无仿射LN，投影后Q/K再分head无仿射LN，除sqrt16。无bias投影，来源嵌入2×768，out零初始化。每任务198144参数，每接收路径8任务1585152；双接收共3170304。两种粒度参数初值配对，初始化不消耗原随机流。总修正Frobenius范数上限为0.05×停止梯度的原接收方图像CLS范数；patch与CLS共用整图额度，记录每token与总范数。该保守额度尚无最优性依据，负结果只针对本预算。

**梯度与持续学习。** ViT/LN/MSA/FFN权重冻结，但注入后图像流和Summarize保留输入梯度，前缀与发送memory停止梯度。CA随任务私有并冻结旧模块，公共前缀只读；旧任务head块/Selectors/Prompts/Adapter保持原保护。后半图像流按任务分流，因此损失公共Frozen Forward只编码一次的效率优势，必须报告真实成本。

**四条训练路径与六组模型。** 各路径只有本视图BCE，Adapter仅接受ASL；没有融合BCE，且发送端只提供交互前冻结patch。故同一Person模型不会因Face是否作为接收端而改变，反之亦然。只训练image_cls/person、image_cls/face、patch/person、patch/face四条路径，按事先指定的六种开关构成六组完整配置。没有根据分数挑选头、表征或checkpoint。原R1和Face Pool不重训；Full及未增强路径复用同一原R1来源。

| 项目 | 正式配置 |
|---|---|
| 数据与权重 | EMOTIC历史train/val/test，CLIP ViT-B/16原预训练 |
| 任务与seed | seed1，8任务，5+7×3类 |
| 预算 | 每任务30epoch，batch64，固定最后一轮 |
| 原任务参数 | Adam，主LR0.0125，weight decay0，每任务重建 |
| Adapter/交互 | LR0.0004，cosine，无warmup，最低LR0 |
| 原Adapter | image-token，b32，index1，scale0.1，ReLU，任务独立初始化 |
| 任务路径 | 10 Selectors、10 Prompts、前5层Prefix、pre-head归一化 |
| 监督 | 原接收分支BCE；Adapter ASL neg9.8/pos0/clip0.05 |
| 增强 | 各视图历史原配方；外部证据为同一TRAIN人物的确定性eval裁剪 |
| Face训练筛样 | 原valid且非ambiguous，不改为仅可靠Face训练 |
| 融合 | 可靠Face固定0.64/0.16/0.20，否则0.8/0.2/0；logits融合后sigmoid |
| GPU与并发 | GPU0；显存准入，最多2训练/评估进程；空间不足保持B64排队 |
| 评估 | 历史samples_intersect_seen_classes，8阶段完整Val/Test |
| 最终人群 | Val2397/Test5368；可靠Face1460/3261 |

**输入与产物。** 路径按服务器独立worktree根目录解析：数据../multi-lane-main/datasets/EMOTIC；CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt；manifest ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。配方来源../emotic_benchmark_runs，参照预测../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02。输出output/r1_frozen_cross_view/seed1_20261105，日志logs/r1_frozen_cross_view/seed1_20261105。不覆盖旧输出。入口：bash scripts/emotic/run_r1_frozen_cross_view_gpu0.sh。

**启动检查。** 单元测试、原R1/Selector回归；每条路径2任务×2更新（Val64，不使用Test）。真实CLIP检查零修正恢复、task1后task0重放、Q/K/V/out梯度、冻结权重无梯度、发送端detach、无效Face屏蔽、8lane形状。按smoke实测reserved显存×1.2+1024MiB准入，留余量；失败不取消其他独立路径，不跳步冒充完整训练。每epoch保存优化器/scheduler/scaler/RNG恢复状态，每任务保留compact状态。

**评估与归因。** 训练后完整8阶段Val/Test；输出六组最终及阶段平均mAP、F1、可靠/不可靠子群和逐类AP。自动对task0/2/7做完整Val/Test关闭交互归因，关闭保留共训参数，不能叫重训R1；不把三边界平均算作八阶段平均。另记录固定TRAIN batch的QK得分、来源权重、熵、修正预算和范数。Full预测只读复用；Face-only不可靠人群重放R1，Person接收组不能要求该子群重放，因为Person本身已变化。

**成本和判断。** 两组参数相同，patch的QK/AV读取约196倍，不能说整网196倍；任务条件后缀带来额外训练激活和随任务增长的推理成本。原R1固定logits的Val44.0900/51.2467、Test32.9566/40.0703；Face Pool为44.2812/51.4608、33.2490/40.3140。主目标Test最终及阶段平均同时超过Pool，并核查Val与OFF方向。Face更强但融合不升，仍不支持互补机制。单seed探索不宣称稳定优势，不自动扩层/温度/预算。
