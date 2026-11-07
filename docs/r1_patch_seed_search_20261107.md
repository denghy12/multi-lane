# Frozen9 与 Frozen10 单层 patch 双接收种子对照

本批按用户的8组要求，运行第9层单层、第10层单层与 seed3047、1024、42、123 的笛卡尔积。用户引用的 44.0329/51.1335、33.1770/40.2083 是已完成的第9层 seed1 结果。两层配置分别从头训练 Person 与 Face，共16条辅助训练。四个 seed 各自训练原生 R1 Full，并供同 seed 的两个层位共用，总计20条路径。

统一 EMOTIC 历史划分，8任务、每任务30轮、train/eval batch64。Adam、cosine、WD0，主LR0.0125，Image-token Adapter及交互LR0.0004，b32/index1/scale0.1，原分支BCE与Adapter-ASL（9.8/0/0.05）。64维4头的patch交互、Q/K逐头归一化、零输出初始化、5%图像CLS范数上限。Person与Face固定AMP4096，Full保持历史动态AMP。任务Selectors与Prompts延续原初始化传递方式，训练保留72线程随机流，评估使用2线程。只使用GPU0和1。

Person patch作Q读取Full与可靠Face的原冻结patch，Face patch作Q读取Full与Person的原冻结patch，两者都附加自己的图像CLS作为K/V。发送与写回都取同一深度，第9层对应索引8，第10层对应索引9。下一Task Forward块的Summarize开始读取修正。最后各自分类头输出logits，可靠Face权重0.64/0.16/0.20，否则0.80/0.20/0，融合后过sigmoid。

数据 `../multi-lane-main/datasets/EMOTIC`，预训练 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，Face manifest沿用 `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。配方只读取历史config，新路径的可学习参数自行训练。

每条先跑2任务、各2次更新的真实预检，核对梯度、可靠性mask、旧任务保护与8路推理。按每条实测峰值加1024MiB预留准入，资源释放后自动补后续路径。训练每阶段保存Test分数、完整恢复状态和compact checkpoint，完成后自动补8阶段Val。正式更新预算Full/Person为13950，Face为10980，跳步即停止并保留状态。

入口

```sh
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_patch_seed_search
```

结果在 `output/r1_patch_seed_search/20261107`，日志在 `logs/r1_patch_seed_search/20261107`。完成后汇总 `patch_seed_search_results.json`，包含八组两split的最终/阶段平均mAP、逐阶段与可靠性子群指标、四项独立排名和各层四seed均值/样本标准差。Val预选按阶段平均，Test最高结果单列为本批观察值。
