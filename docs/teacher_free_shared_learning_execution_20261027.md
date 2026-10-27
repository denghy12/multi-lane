# 无教师三视图互补学习：实现与GPU0执行协议

2026-10-27用户要求按已讨论计划修改并在GPU0高效运行。分支`codex/teacher-free-shared-learning-20261027`，基于28a1cbb；不覆盖此前未提交分析报告，不改主树或test-only。设计详见`teacher_free_shared_view_learning_plan_20261027.md`。本文件首先记录运行前声明，执行状态在完成预检后追加。

## 配方与任务范围

原EMOTIC历史seed1强底座32.6800/39.5292；标准CLIP ViT-B/16 +fresh任务参数/head，无A/B教师/旧头交换。三路Task Forward，共享Selectors10、Prompts10/5层、b32 Image-token Adapter索引[1]/ReLU/scale.03；原固定特征融合与共享头。原融合BCE+aux.1，Adapter融合ASL+aux.1，gamma_neg9.8/gamma_pos0/clip.05；Person颜色扰动0。每任务30epochs/B64、Adam按任务重置/WD0/cosine、主LR.0125/Adapter.0004、AMP65536/TF32、OMP/MKL72。训练workers2/prefetch1；评估2个单线程spawnworker/B64。历史samples_intersect_seen_classes与阶段算术均值不变，Test标为参与开发。

第一阶段task0：U0原配方、U2Full条件互补排序（无方向保护）、U4互补排序+Full/融合一阶保护。新增梯度限原组范数.25倍，新增实际步限原组B0步.1倍，候选α=1/.5/.25/.125/0。原30轮内第1轮λ0，第2–5轮线性至1。Head始终接受原B0监督，全30轮学习；新增排序不更新head/不直接走Full特征边。U0不导入或调用新增目标/候选选择，历史RNG两次CPU取种子仅在完成当前任务边界重放一次。

完成task0全部两split评估后，自动晋级条件：U4两split Full下降不超过.10，融合均不低U0/U2，至少一个split融合较U0提高.02；接受过辅助更新且实际浮点参数步确实非零。这只是实用筛选门槛，不是统计显著性。通过后U0/U2/U4从自身完整恢复继续task1–2，并从fresh初始化补U1普通排序/U3Full保护的task0–2；未通过则记录原因后自然结束，不自动调参。不会自动进入8任务，完整8任务须先看控制结论，符合既定阶段计划。

## 实现范围

- complementary_ranking.py：同batch自然同类别正负人物对，Full软难度1..2，Person/可靠Face额外反向，Face两端可靠与统一分母。数值复用特征stack/sum+共享head，不重写主前向。
- complementary_update.py：安装版本PyTorch functional Adam在私有Tensor/矩副本中构造候选，真实参数/优化器状态不变；只选择R/A的额外梯度。原head梯度不动。α0保留原grad，非有限额外梯度回退；记录真正非零参数步。
- runner.py：可选目标/反向接入、分组unscaled梯度、一次GradScaler step、候选日志、逐epoch恢复与续跑；None关闭机制保持原路径。
- complementary_training_probe.py：每100更新，用同一已增强TRAIN batch比较原B0候选与实际选定步，2次no_grad前向，恢复参数/RNG/training flags且不改Adam。它是同批局部反事实，不能当独立泛化探针；为避免新增加载/随机流，采用同批检查而非另建固定图片loader。
- teacher_free_shared_learning.py：clean源配置读取/fresh构建、历史U0精确核验、旧参数/CLIP及旧TRAIN预测保护、epoch/task完整恢复、两split阶段前缀评估，无教师参数/输出导入。
- run_teacher_free_shared_learning.py及shell入口：GPU0显存/共享内存感知队列、上限3任务、启动间隔8秒、锁与孤儿进程检查、自动评估/预声明晋级。

恢复状态在验证前保存，包含模型/Adam/调度器/GradScaler/全部RNG/任务与epoch/实际增强batch审计；已完成任务直接跳过，避免重复报告取种子。保存每个epoch，不删除来源或中间checkpoint。该保护只限制新增更新的本batch一阶影响，不保证Full预测不变或Val/Test AP不下降。

## 输入、输出和入口

从新的remote独立worktree执行：
- 配置及U0比较参考：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`，SHA与历史配置核对。
- Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`，原协议。
- 输出：`output/teacher_free_shared_learning/seed1_20261027_01`。
- 日志：`logs/teacher_free_shared_learning/seed1_20261027_01`；预检同目录前缀preflight_20261027_01。
- 入口：`bash scripts/emotic/run_teacher_free_shared_learning_gpu0.sh seed1_20261027_01`；Python默认服务器ddp环境，也可ML_PYTHON明确指定。

预期产物：三组task0训练/恢复/完整Val-Test四路预测与指标、配对batch及初始SHA、U0历史精确证明、实际新增更新/Full-fusion反事实loss与显存耗时、晋级决策；若通过，附五组task0–2完整结果。prefix1/3均值明确标为阶段前缀，不能称8任务均值。

## 预检与部署

本地py_compile/bash-n/git diff --check；指定排序/候选Adam/runner集成/队列恢复机制测试，以及全部tests/test_track_a*.py回归和相关独立tests；再真实CLIP三组2任务×24更新smoke/Val首64。Smoke强制λ1以实际覆盖新增机制，明确不改变正式渐入；只读来源比较，不报告smoke精度为效果。核对原主树153条/test-only5条dirty的HEAD和status SHA保持，任何源码仅Git提交推送/fetch到新worktree。

实际GPU0 4090约24GB空闲，其余1–7忙，禁止使用或中断。正式最大3并发，每新增训练需至少6GB余量，评估至少4GB，/dev/shm至少768MB；预检如显存不足按实测降低并发并记录，不改batch或原初始化线程。源码提交固定后才正式运行。

