# Person10＋Face9 完整前向、开关及同种子基线

覆盖seed3047、1024、42、123。现有同seed Full、Person10、Face9在一个批次级推理模块中共同运行，先按各路原头得到logits，再按固定R1权重融合并过sigmoid。所有评价使用原EMOTIC历史划分、seen-class人群、batch64、确定性预处理与可靠性规则。原路径来源是 `../multi-lane-main-r1-patch-seed-search-20261107/output/r1_patch_seed_search/20261107`。

完整前向从Person评价数据入口同时获得该人物的Person、Full、Face确定性裁剪，映射到各分支原输入接口。Person10与Face9使用原checkpoint，来源分支的ON logits必须与原存档按ID、标签逐值重放。八阶段两个split均计算非对称主组。在task0/2/7额外运行Person9与Face10，分别对双9、双10、非对称三种结构做全开、关Person、关Face、全关。OFF保留本路径共训的Adapter、Selectors、Prompts和分类头。切换Face时不可靠Face人群的融合输出必须逐值一致。

为支持归因，评价进程驻留5条模型路径，非对称主组只使用其中3条。显存预检包含5模型和开关工作量，不能直接作为3模型部署显存。各模型使用独立autocast上下文，融合保留历史CPU float32运算次序，来源checkpoint与预测均记录SHA并在结束后核对。

同seed补两组训练对照。四个Full复用本批完整原生Full，每seed训练原生Person、原生Face、Task6 Pool Face，共12条。原R1用原生Person/Face，Face-only Pool用同一新原生Person与新Pool Face。三条可学习路径均独立从原初始化训练，seed对应配对，不更换分类头。Pool保留历史Task Forward第6块后位置、原Frozen6发送patch、64维4头、无显式cap，使用原动态AMP。

训练用8任务、每任务30轮、batch64、Adam、cosine、WD0、主LR0.0125、Adapter/Pool LR0.0004、b32/index1/scale0.1、原分支BCE与Adapter-ASL（9.8/0/0.05）。Person为历史letterbox、颜色强度0.1/概率0.2，Face保留历史有效训练筛样。可靠Face权重0.64/0.16/0.20，否则0.80/0.20/0，先融合logits再过sigmoid。训练保留历史72线程随机流，所有R1路径评估临时限制为2线程并在评估后恢复。

数据 `../multi-lane-main/datasets/EMOTIC`，CLIP权重 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，Face manifest沿用 `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。源checkpoint、预测与数据保留。12条训练预计需要约35–40GiB新增恢复状态空间，启动前检查可用空间至少50GiB。

四条完整前向预检用task0/7两split各64人，包含8任务推理和开关，必须逐值重放来源。12条控制训练先2任务各2更新，检查有效预算、有限梯度与八任务推理。所有预检通过后，GPU0/1按峰值加1024MiB预留同时准入正式评估和训练。新训练遇跳步即停止并保留恢复状态。逐阶段Test由训练入口输出，完整训练结束自动补八阶段Val。

入口

```sh
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_asymmetric_verify
```

结果 `output/r1_asymmetric_verify/20261107`，日志 `logs/r1_asymmetric_verify/20261107`。包括来源重放、完整前向各路与融合预测、开关、可恢复控制checkpoint、R1/Face-only Pool八阶段两split汇总和 `asymmetric_paired_results.json`。最终汇总报告非对称相对同seed两种控制的配对差及跨seed均值/样本标准差。
