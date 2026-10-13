# EMOTIC PAMI：三组全量学习 seed0 直接 test

## 目的与可比性

将此前 CVPR17 数据集上的三组全量学习配置原样移到用户提供的官方 PAMI 扩充版，判断 Full、Person、Face 三路与共享 ParaX 在新数据版本上的表现。此前表中的 38.2098、39.2278、39.6299 是 CVPR17 **三个 seed 的 test 均值**；本次按用户要求只运行 **seed0**，不能把单次结果当作三 seed 均值复现。

PAMI 文件位于服务器 `../multi-lane-main/datasets/EMOTIC_PAMI/`，标注为 `Annotations/Annotations.mat`，图片在 `emotic/`。已核验 34,320 人物记录、23,171 个被引用的不同图像路径均存在。官方标注中 train 与 val 有 383 张相同图片且人物框重合；本次固定训练 30 轮，不用 val 选模型，训练结束才评估 test。该重叠仍需在论文比较中说明。PAMI 官方 train/val/test 划分不改动；最终 test 只有一个固定配置，没有按 test 搜索参数。

## 三组

| 组别 | 输入与模型修改 |
| --- | --- |
| Full 单路 | 仅完整场景；冻结共享 CLIP；不加 Image-token Adapter 或 ParaX。 |
| 三路，无 ParaX | Full、Person、Face 复用同一冻结 CLIP、Selector、Prompt 和分类头；固定可靠性特征融合。 |
| 三路，共享 ParaX | 在 Frozen Forward 的 ViT block 索引 10 完整输出之后、最后一个 block 之前加入三视图共享的 ParaX；三个 rank32 低秩矩阵专家对、router hidden16、官方随机初始化、可学习输出 scale 初值 0.1；只作用于 patch tokens。最终三路融合仍固定。 |

三组均只使用一个 Task Forward，26 类同时监督。可靠 Face 时固定权重为 Full/Person/Face=0.64/0.16/0.20；不可靠时为 0.80/0.20/0。Face 检测使用 SCRFD `det_10g.onnx`、CPU、640、threshold 0.5、margin 0.15、ambiguity gap 0.08；可靠性仍要求有效、无歧义、短边至少 24 像素且检测分数至少 0.6。PAMI 专属 Face manifest 覆盖 train/val/test，不能复用 CVPR17 manifest。

## 固定执行配置

- 训练入口：`scripts/emotic/launch_joint26_pami_three_way.sh`，单组入口 `scripts/emotic/run_joint26_parax_view_ablation.sh`；数据集参数 `--dataset-release pami`。
- seed=0；30 个完整 train epochs；train/eval batch 64；workers 2；Adam，主要参数学习率 0.0125，ParaX 学习率 0.0004，weight decay 0；cosine scheduler、无 warmup、末端比例 0。
- AMP 开启，初始 loss scale=1024、growth interval=1,000,000,000；CLIP 原权重冻结；Image-token Adapter 关闭；10 个 Selectors；Full random resized crop scale 0.05–1；Person margin 0.15 并 letterbox；CLIP normalization。
- 三路损失为融合 BCE + 0.1×可用视图 BCE 均值；Full 单路为 1.1×BCE。固定第 30 轮直接评估 test，不进行每轮 validation 评估或用 validation 选 checkpoint。正式训练使用 `--reporting-split test --skip-validation-eval`，导出逐人 26 类 test scores。
- 预训练权重：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；数据：`../multi-lane-main/datasets/EMOTIC_PAMI`；Face 检测权重：`../baseline_sources/cocoer_insightface/models/buffalo_l/det_10g.onnx`。
- 服务器独立工作树：`/mnt/haoyuan/workspace/multi-lane-main-pami-joint26`；日志 `./logs/emotic_pami_joint26/`，训练与 test 结果 `./output/emotic_pami_joint26/`，Face manifest `./output/emotic_pami_face_manifest/`。GPU0 空闲时三组并行，若显存条件不满足则等待或调整执行并行度，不改变单组配置。
- 预期产物：Face audit/manifest、各组配置、逐轮训练史、固定末轮 checkpoint、test logits/targets/sample IDs、视图诊断、配对审计、test 对照表。检查三组初始任务参数 hash、首批样本/完整 train IDs、test IDs 与标签一致，CLIP hash 不变，ParaX 梯度有限，正式更新无跳步。

先运行 Face manifest 与真实 ViT 短程 smoke。smoke 只用于接线检查，使用 validation 而不访问 test；成功后用新的 batch ID 启动上述正式 test 队列。训练期间不主动持续监督，结束后统一分析。当前 loader 保留 CVPR17 作为默认，PAMI 需要显式参数，避免混用两个数据版本。
