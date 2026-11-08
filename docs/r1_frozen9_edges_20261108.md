# 第9层seed1辅助视图边的四组对照

以原Frozen9 patch Person＋Face seed1为D组，Val44.0329/51.1335、Test33.1770/40.2083。只研究Person与Face之间的两条证据边，Full继续独立原R1路径。

| 组 | Person读取 | Face读取 |
|---|---|---|
| A | Full | Full |
| B | Full | Full＋Person |
| C | Full＋可靠Face | Full |
| D，原基准 | Full＋可靠Face | Full＋Person |

所有接收端仍用自己的空间patch作Q，K/V仍包括自身图像CLS与两个外部证据槽位，写回仍在Frozen第9块后。Full-only通过softmax前的mask排除另一辅助视图，保留相同模块、参数、初始化随机流和两个证据槽位。自图像CLS始终有效，外部Full始终有效。

先用原P/F checkpoint做两种证据模式的真实前向，完整8阶段Val/Test形成四组。这一部分只改变推理可访问的证据，衡量已训练权重中的条件贡献。Full、Person双源和Face双源ON输出须按样本ID与历史预测逐值一致，不可靠Face人群的A/B与C/D融合logits须逐值相同。

随后从头训练Full-only Person与Full-only Face两条路径。已有双源P/F不依赖另一接收路径的更新后特征，因此两新两旧四条路径即可形成上表四种预定结构。每条都连同自己的原分类头使用，证据来自原冻结CLIP流。新训练完成后以完整模型前向生成四组八阶段两split结果，并逐值重放每条路径自己的来源输出。

训练用EMOTIC历史划分、seed1、8任务、每任务30轮、B64，Adam/cosine、WD0、主LR0.0125、Adapter/交互LR0.0004。b32/index1/scale0.1、原分支BCE与Adapter-ASL（9.8/0/0.05），64维4头、Q/K逐头归一化、零输出初值、每图5%图像CLS范数上限、固定AMP4096与TF32。Person沿历史letterbox、颜色强度0.1/概率0.2，Face沿原有效训练筛样。Person预期13950次有效更新，Face10980，跳步停止并保留恢复状态。

原参考为 `../multi-lane-main-r1-frozen-late-stable-20261105/output/r1_frozen_late_stable/seed1_20261105/layer9/patch/attention`，Full为原R1 `../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02/native/full`。数据 `../multi-lane-main/datasets/EMOTIC`，CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，Face manifest沿历史 `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

固定logits权重可靠Face为0.64/0.16/0.20，其余为0.80/0.20/0，融合后sigmoid。GPU0先做只读预检和全量归因，再GPU0/1分别跑新Person/Face的两任务各两更新预检，通过后两路正式并行。完成后GPU0自动做训练结构的完整四组前向。磁盘启动前要求至少15GiB，预期新增约9GiB恢复状态，既有产物保留。

入口

```sh
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_frozen9_edges
```

结果 `output/r1_frozen9_edges/seed1_20261108`，日志 `logs/r1_frozen9_edges/seed1_20261108`。分别提供 `evaluation/readonly` 与 `evaluation/trained` 两类四组预测/指标，`four_group_results.json` 为最终汇总，源checkpoint与历史预测前后SHA核验。
