# TRAIN内部80/20学习三路表征与融合：第二轮执行

2026-11-11，用户授权按计划执行。第一轮已完成，源eaae912：固定Val44.0329/51.1335、Test33.1770/40.2083；全TRAIN Smooth-AP Val40.5209/47.5329、Test29.9943/36.9996。八任务TRAIN平滑目标与真实AP均改善，比例偏向Person/Face，未泛化。按既定计划进入分支未见TRAIN人物对照。

## 固定结构与训练预算

EMOTIC历史数据划分；seed1、8任务、每任务30epoch、batch64，原三路独立MLP Head；Full原路径，Person/Face均在Frozen9后进行patch Cross-view Attention，Person读Full/可靠Face、Face读Full/Person，64维4头、投影后QK归一化、5%上限、AMP固定scale4096（P/F）。Full沿原数值配方。主LR0.0125、Adapter/交互LR0.0004、Adam/cosine/WD0、b32/索引1/scale0.1与原Adapter-ASL。初始化/训练沿原72线程，评估2线程，DataLoader workers2。

源输入仍为`../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP/pretrained/clip/ViT-B-16.pt`和历史Face manifest。只读取历史配方与预训练CLIP，三条分支从头训练，不加载原实验的可训练模块。

官方TRAIN按原图稳定SHA-256分组，salt=`emotic-reliability-calibration-v1`，20%作为融合参数学习数据，80%作为原分支参数学习数据；同图所有人物及所有视图同组，跨任务相同，不用未来类别标签选择划分。Face在80%内仍执行原有效样本筛选。30轮不补步数，较全TRAIN更新数减少；所有本批读出共用相同新三路，避免把分支差异误作融合差异。

新增显式CLI`--train-only-fusion-holdout --calibration-fraction .2 --save-calibration-scores`，只对该R1研究开放Face 80/20设置，原历史Face入口规则保留。补齐交互模型的TRAIN留出图像证据输入和恢复时已存在的calibration目录支持。源代码均在新实验分支，不影响其它工作树。

## 五种读出

1. fixed：原R1可靠.64/.16/.20、不可靠.8/.2/0。
2. equal：可靠三路等权、不可靠两路等权。
3. seen_smooth_ap：从80%中按原图确定性抽取与留出20%近似等人数的人物，反传Smooth-AP。
4. heldout_bce：留出20%反传BCE。
5. heldout_smooth_ap：留出20%反传Smooth-AP。

三个可训练组均每任务2参数、a=.5/b=1/3等权初始化、FP32 Adam .02/300步/WD0；Smooth-AP温度.15。每任务训练完冻结比例，旧类读出不更新。D/F原图数量与逐类支持可能不同，报告支持人数，不能把差值完全归因见过/未见过。

模型三路参数只从80% TRAIN更新，融合参数仅从其声明的TRAIN子集更新。官方Val只选配置，完整Val不重拟合；锁定融合checkpoint和Val决策后导出并评估Test，报告所有组8阶段全人群/可靠/不可靠mAP、AP与F1。

## 执行、门禁与产物

GPU0，smoke三个分支串行，两任务各2有效更新，Val每阶段64样本；P/F额外做旧函数保护、梯度、有效Face屏蔽、零残差与8路径显存预检，随后两任务10步学习融合、完整流程双split验证。正式依smoke显存准入最多并行3条路径；显存不足会排队，训练完自动融合拟合/Val/Test。

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_train_holdout_fusion --smoke
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_train_holdout_fusion
```

输出`output/r1_train_holdout_fusion/seed1_20261111`，预检为同名`_smoke`目录；日志`logs/r1_train_holdout_fusion/seed1_20261111`。包括三路30轮/恢复状态与compact checkpoint、calibration原图计数、训练预算/零跳步检查、三路对齐logits、各组融合模型/优化器、锁定SHA、八阶段Val/Test预测与结果。磁盘已检查约1.5TB可用，预计新增十余GiB，不删除历史产物。

第一轮缓存结果是已完成证据，第二轮在门禁通过并启动后才记作运行。两轮都不依赖Validation梯度，也没有教师、换头或三折模型集成。
