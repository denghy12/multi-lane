# Frozen Forward 单层 patch 双接收层位对照

固定原 R1 Full，分别训练 Face 与 Person 的接收路径。Face patch 作 Q，读取自身图像 CLS 与 Full、Person 冻结 patch。Person patch 作 Q，读取自身图像 CLS 与 Full、可靠 Face 冻结 patch。发送方取交互前的同层空间证据。写回接收方 Frozen Forward，下一块 Task Forward 通过 Summarize 开始读取。每条路径独立训练，可靠 Face 的固定 logits 权重为 0.64/0.16/0.20，否则为 0.80/0.20/0，融合后过 sigmoid。

| GPU | 位置 | 训练路径 |
|---|---|---|
| 0 | 第6/11块后，索引5/10 | Person 与 Face 各两条 |
| 1 | 第8/10块后，索引7/9 | Person 与 Face 各两条 |
| 复用 | 第9块后，索引8 | 同精度、同配方的原双接收路径 |

EMOTIC 历史划分，seed1，8任务，每任务30轮，batch64。Adam 主学习率0.0125，Adapter 与交互0.0004，权重衰减0，cosine。Image-token Adapter b32/index1/scale0.1，原各路 BCE 和 Adapter-ASL（9.8/0/0.05）。交互64维4头，Q/K 投影后归一化，零输出初值，每图5%上限。AMP 固定4096且 TF32 开，检测到跳步即停止该路径并保留恢复状态。Person 期望13950有效更新，Face 10980。先各2任务×2更新预检，核对原路径保护、梯度和八路输出，再按实测峰值加进程余量准入。每条正式训练逐任务保存 Test，训练完成自动补完整 Validation。

服务器入口从独立工作树根目录运行：

```sh
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_patch_receiver_depth
```

数据 `../multi-lane-main/datasets/EMOTIC`，CLIP ViT-B/16 权重 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，沿用历史 Face manifest。参考 logits 来自原 R1、Face-only Pool 和第9层结果，哈希记录在批次输出。新训练不载入这些路径的任务参数。日志 `logs/r1_patch_receiver_depth/seed1_20261106`，结果 `output/r1_patch_receiver_depth/seed1_20261106`，包括逐阶段预测、指标、可恢复 checkpoint、队列与 `single_layer_patch_person_face_results.json`。按同一 split、人群、可靠性和 logits 口径汇总。层位对照包含同层外部证据的变化。
