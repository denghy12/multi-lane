# 全TRAIN控制恢复与融合监督人群：执行协议

日期：2026-11-13。用户授权执行。当前阶段：实现和预检；正式训练尚未启动。独立分支codex/r1-fusion-population-recovery-20261113，基于a6a5140，不修改旧实验checkpoint。

## 目标与范围

1. 从full_extend的task3 epoch13完整无跳步状态，先做AMP／FP32无更新梯度诊断，再决定数值稳定恢复并补齐8任务Validation／Test。
2. 从头执行一条holdout_return参数轨迹，比较当前任务阳性TRAIN H与已见任务TRAIN H（补当前类全阴性人物）两种融合监督。固定融合为第三个同期读出。禁止外部模型、分类头或比例导入。

不新增Attention、门控、排序目标或seed。固定原强参照Val44.0329/51.1335、Test33.1770/40.2083。

## 训练与数据配置

GPU0；seed1，8任务，batch64，原EMOTIC划分，Frozen9 patch Person＋Face交互，Full原路径。三路独立Task Selectors、Summarize、Task Forward和MLP Head。Image-token Adapter b32/index1/scale0.1；分支原BCE/Adapter-ASL、Face原筛样及可靠性mask。

主LR0.0125，Adapter／交互LR0.0004，Adam，cosine，WD0。新候选按原图固定A/H约80/20：A训练30轮；同一模型在H学习融合100step；融合封存，全TRAIN回填5轮，峰值LR为原0.1倍，保留Adam、scaler、模型和随机状态。A/H各曝光35/5轮，不称每张图完整训练30轮。

两学习读出各两个标量a,b，等权初始化，FP32 BCE，Adam LR0.02，WD0，100次更新。当前任务只监督当前类，旧任务参数封存，融合梯度不进入分支。固定比例可靠Face .64/.16/.20，不可靠 .80/.20/0；两学习组用[a,1-a,0]和[(1-b)a,(1-b)(1-a),b]。

输入相对工作树：EMOTIC `../multi-lane-main/datasets/EMOTIC`；预训练 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。沿历史配置读取预处理参数，不导入历史训练权重。

## 融合监督人群的唯一区别

- current_positive：原H中至少一个当前任务标签为正的人物。
- seen_population：同一原图H分配中至少一个已见类标签为正的人物，包含上述current_positive及当前任务全部类别为负的历史已见人物。
- 排除仅具有未来类别的TRAIN人物；Validation／Test绝不拟合。
- 原H的forward批次与输出保留，新增负例单独预测，再对齐合并，避免把旧H输出变化混入比较。
- 不对融合监督套用Face训练筛样；无可靠Face仍可学习Full／Person的两路比例。
- 新读出访问历史TRAIN仅用于融合监督，不回放到三路分支损失。报告每任务新增人物、原图、与当前任务图像重叠数、历史曝光以及缓存字节；这增加历史数据访问，不能冒充原无回放协议。
- H可能在旧任务回填过，继续报告其历史曝光，不声称整个参数轨迹严格未见。

## 全TRAIN控制的诊断与恢复边界

旧状态来自 `../multi-lane-main-r1-fusion-return-train-20261113/output/r1_fusion_return_train/seed1_20261113/full_extend/recovery/task3/epoch13.pth`，SHA256 `15d570f6d928854312d473ccdf0a41de3110ce111fc964581feeadcbcceedfd8`。

诊断固定该模型，重放同一训练数据批次，比较AMP4096／低scale与FP32梯度有限性，不调用optimizer.step，不写回源checkpoint。固定模型的无更新诊断不等于完整复现实际epoch14逐batch参数变化；恢复试验仍须逐epoch守零跳步。

安全恢复在独立输出中从三路共同epoch13开始；保留task0–2已完成结果、task3前13轮及Adam/RNG，不使用含7次跳步的epoch14。数值修复必须由诊断决定并记录。源产物保持只读；新旧控制的数值策略边界公开报告。

## 入口、产物与验收

新增人口对照入口 `python -m scripts.emotic.run_r1_fusion_population_train`；恢复／诊断入口以完成后的实现补记。

新对照输出 `output/r1_fusion_population_train/seed1_20261113`，日志 `logs/r1_fusion_population_train/seed1_20261113`。控制恢复输出 `output/r1_full_extend_recovery/seed1_20261113`，日志同名logs目录。预检另用smoke目录，禁止覆盖正式产物。

正式同批报告8阶段Validation／Test、最终与阶段平均、可靠／不可靠Face子群和task6/7新类AP。固定和两学习读出来自每阶段同一个完整模型；输出各bank与来源元数据。预检应验证TRAIN/已见类/原图边界，负例与Face mask，100步与封存，观察不改变分支参数或RNG，恢复状态完整；smoke含2任务及两split真实前向。

仅通过实现检查和真实smoke后启动正式实验。候选须优于同流程固定且达到原强参照Test33.1770/40.2083，才考虑扩seed；不以Test选步数或补入人群。
