# Frozen Forward 第8＋9层和第9＋10层 patch 双接收

单层对照没有在两个 split 上共同胜出，按约定复用第9层为单次交互参照。相邻第8层的 Validation 43.8602／51.2670 高于第10层的43.7029／51.1904，因此主比较为第8＋9层。用户另要求加入第9＋10层，按同样的共享与独立模块预算比较。

每个任务的 Person 和 Face 各自训练。前一块后写回接收方空间 patch，下一 Task Forward block 首次读取。后一 Frozen block 完成后以更新后的 patch 做第二次 Q，再由后续 Task Forward 读取。每个发送端一次冻结前向提取两层交互前 patch，Person 的 K/V 屏蔽不可靠 Face。每处含自身图像 CLS，64维4头、逐头 Q/K 归一化、零输出初始化。两处分别以本处写回前图像 CLS 范数的2.5%限幅。

| GPU | 训练路径 |
|---|---|
| 0 | 两对层位各自的共享模块 Person、独立模块 Face |
| 1 | 两对层位各自的独立模块 Person、共享模块 Face |

共享是每任务每接收端跨两层复用一套交互参数。独立是每任务每接收端两套，原有旧任务参数保护不变。Full 沿用 R1 预测。可靠 Face 融合 logits 权重0.64/0.16/0.20，其他样本0.80/0.20/0，再过 sigmoid。不可靠 Face 人群仍可能受新 Person 路径影响。

EMOTIC 历史划分，seed1、8任务、30 epoch/任务、batch64。Adam，主LR0.0125，Adapter 和交互0.0004，cosine，WD0。Image-token Adapter b32/index1/scale0.1，原分支 BCE＋Adapter-ASL（9.8/0/0.05）。AMP固定4096，TF32开，跳步即停并保留恢复状态。数据 `../multi-lane-main/datasets/EMOTIC`，预训练权重 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，历史Face manifest。新路径从头训练，不载入其他任务模型权重。

入口 `PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_patch_receiver_depth --two-layer`。八条路径按显存并行预检，各2任务×2更新。按实测峰值加余量启动正式训练，给可能同时发生的八路任务推理保留显存。保留历史训练线程和数据随机流。Test 逐任务产出，完成后自动验证全8阶段。task0/2/7在两split做全开（复用正式预测）、只开前层、只开后层、全关。三个开关前向复用每批输入，保留2.5%单处限额。分析只覆盖三个边界，不计算八阶段开关均值。

日志 `logs/r1_patch_two_layer/seed1_20261106`，结果 `output/r1_patch_two_layer/seed1_20261106`。包括恢复状态、八阶段预测、`two_layer_patch_person_face_results.json` 和 `attribution/`。单层、R1、Face-only Pool 来源保留并记录SHA。
