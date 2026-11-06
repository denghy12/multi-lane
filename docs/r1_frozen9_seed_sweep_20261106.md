# Frozen9 patch · Person＋Face 五种子对照

用户要求测seed0、2和更多seed，额外在看结果前固定seed3、4，复用完整seed1。每个新seed从CLIP开始分别训练Full、Person、Face。Full使用原R1路径；Person与Face使用原第9块后空间patch交互，任务私有，64维4头、QK逐头归一化、零输出初始化、5%图像CLS限幅。Person屏蔽不可靠Face。可靠时固定logits融合0.64/0.16/0.20，否则0.80/0.20/0，再过sigmoid。

统一EMOTIC历史划分，8任务每任务30轮，train/eval batch64，Adam、cosine、WD0，主LR0.0125、Adapter和交互0.0004。原BCE＋Adapter-ASL（9.8/0/0.05），Image-token Adapter b32/index1/scale0.1。辅助路径固定AMP4096，Full保留历史动态AMP，均TF32开。沿用历史有效Face训练筛样和Val/Test可靠性规则。输入 `../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP/pretrained/clip/ViT-B-16.pt`、历史Face manifest。所有新模型独立初始化。

seed0/2历史Full Test存档的训练配方与seed1相同，但另一套Val存档排除了校准人群。此次重训Full以保证同一权重输出Val和Test，额外逐阶段核对历史Full Test logits，记录逐值相等及最大差值。seed3/4采用相同Full配方。seed1融合参照为原44.0329/51.1335、33.1770/40.2083。

GPU0/1优先派seed0/2，其余随资源并行或排队。12路径各2任务×2更新，实际预检峰值后准入，保留运行中Task10的峰值预算。每个新seed训练完成后计算完整八阶段Val/Test并复用已确定的融合读出。汇总列四项mAP排名、各项的五seed均值与样本标准差。Val优选按阶段平均排名，Test最高分单列为这批观察值。

入口 `PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_frozen9_seed_sweep`。输出与日志分别位于 `output/r1_frozen9_seed_sweep/20261106`、`logs/r1_frozen9_seed_sweep/20261106`。结果含分支checkpoint、逐阶段Val/Test预测、历史Full Test核验和 `frozen9_seed_sweep_results.json`。
