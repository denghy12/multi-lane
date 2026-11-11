# TRAIN-only任务融合第一轮执行协议

用户于2026-11-11要求按计划执行。当前范围为计划第一轮A/B及等权读出，尚不进行80/20新分支训练。目标是固定已有Frozen9三路能力，仅用TRAIN监督反传学习当前任务融合层。

- EMOTIC历史划分、seed1、task0–7；源三路为Frozen9 patch Person＋Face，同一原MLP Head，不更换头或使用教师。
- 源三路原训练为30epoch/task、batch64、Adam/cosine、主LR0.0125、Adapter/交互LR0.0004；b32/第2层/scale0.1，沿原Adapter-ASL、增强、Face筛样、Frozen9/64维4头/QK归一化/5%上限。此次不重训源三路。
- TRAIN缓存：`../multi-lane-main-r1-frozen9-learned-fusion-20261110/output/r1_frozen9_learned_fusion/seed1_20261110/train_logits/task{t}.npz`，当前任务原始类别、全部合法TRAIN人物，共29580条任务/人物记录。
- 每任务两个nn.Parameter，a=sigmoid(u)、b=sigmoid(v)；等权初始化a=.5,b=1/3，可靠组((1-b)a,(1-b)(1-a),b)，不可靠(a,1-a,0)。新任务独立初始化，旧任务封存。
- 仅TRAIN优化：FP32 Adam、LR0.02、300updates/task、温度0.15、WD0、无原固定比例锚点。全拟合列表计算Smooth-AP，阳性锚点128分块仅用于显存，不改变比较列表。
- 对照：原固定比例、无训练等权、TRAIN Smooth-AP。全部同源三路，不重复三路训练。
- 不运行Val三折，不在完整Val重拟合；训练完保存并锁定模型checkpoint、优化器、权重及SHA，再评Val，按两mAP均高于固定判断候选，再报告Test全部对照。
- 原checkpoint/pretrained为源协议记录路径，源预测在`../multi-lane-main-r1-frozen9-edges-20261108/output/r1_frozen9_edges/seed1_20261108/evaluation/readonly`。
- GPU0；torch/OMP/MKL/OpenBLAS各2线程（OpenBLAS1），TF32矩阵乘关闭；无需DataLoader重新全量解码图像。真实图像smoke仍用64人、workers2/preload1。
- 输出`output/r1_train_only_fusion/seed1_20261111`，日志`logs/r1_train_only_fusion/seed1_20261111`。预期8个融合checkpoint（含优化器及源引用）、逐任务训练轨迹、锁定参数、八阶段Val/Test全人群与可靠/不可靠子群、逐类AP/F1、原固定逐值重放与源SHA审计。

新增文件：`multi_lane/track_a/task_logit_fusion.py`、`scripts/emotic/run_r1_train_only_fusion.py`、`tests/test_r1_train_only_fusion.py`。未改历史校准入口，避免继续误用Val拟合。

执行入口：

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_train_only_fusion --smoke
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_train_only_fusion
```

仅在回归、两任务10步真实缓存预检、真实图像六路重放与学习融合对齐通过后执行完整300步。此轮是缓存上的模型融合参数学习，不冒充新的三路从头训练。80/20阶段需按第一轮结果判断是否值得付出源分支重训成本。
