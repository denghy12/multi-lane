# Frozen9 patch → Task10 Pool 双接收

Person 与 Face 各自独立训练。Frozen 第9块后，接收方空间 patch 作 Q，读取自身图像 CLS 和另外两视图的冻结 patch，再把修正写回接收方图像流。Task 第10块经 Selectors、Summarize、任务 MSA、Drop & Replace 和 FFN 更新 Task CLS。该块后，Pool 分别聚合两发送方原 Frozen9 patch，和当前 Task CLS 拼接，经 MLP 产生任务 CLS 残差。两阶段共用每个发送端的一次冻结前向，Person 屏蔽不可靠 Face。Full 使用原 R1。

Frozen patch 每图上限为图像 CLS 范数5%，Task Pool 上限为写回前 Task CLS 范数2.5%。各模块零输出初始化，任务私有。Pool-only 使用相同 Task10 位置、Frozen9 发送证据、Pool 初始化和上限。历史 Face-only Pool 的 Task6 位置、无显式上限，保留作参考。训练用 EMOTIC 历史划分、seed1、8任务、每任务30轮、B64。Adam、cosine、WD0，主LR0.0125，Adapter 与新增模块0.0004。b32/index1/scale0.1、原分支 BCE＋Adapter-ASL（9.8/0/0.05）、AMP固定4096与TF32。预训练 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，数据 `../multi-lane-main/datasets/EMOTIC` 与历史 Face manifest。可靠 Face 固定权重0.64/0.16/0.20，否则0.80/0.20/0，融合 logits 后过 sigmoid。

| 正式比较 | Person 来源 | Face 来源 |
|---|---|---|
| Frozen9 双接收 | 已有 patch | 已有 patch |
| Frozen9 ＋ Face Task10 Pool | 已有 patch | 新 patch＋Pool |
| Frozen9 ＋ Person、Face Task10 Pool | 新 patch＋Pool | 与上一行同一条 Face |
| 仅 Task10 Pool 双接收 | 新 Pool-only | 新 Pool-only |

GPU0跑 patch＋Pool Face 和 Pool-only Person，GPU1跑 patch＋Pool Person 和 Pool-only Face。各2任务×2更新真实预检，按峰值加余量并行准入。训练完成后补完整8阶段Validation，逐任务Test由训练入口输出。task0/2/7两split的两条新组合路径各做只开Pool、只开patch、全关，复用正式全开预测；Face加Pool组在这些归因中保留已有Person patch路径。

入口 `PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_frozen9_task10_pool`。日志 `logs/r1_frozen9_task10_pool/seed1_20261106`，输出 `output/r1_frozen9_task10_pool/seed1_20261106`，包括逐阶段预测、可恢复checkpoint、`frozen9_task10_pool_results.json` 和开关汇总。
