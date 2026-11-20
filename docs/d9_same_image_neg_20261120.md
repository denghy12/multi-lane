# 原随机留出原图内补充当前新类全阴性人物

## 训练与评价协议

EMOTIC seed1，13,950 次更新的原随机 90% Frozen9 三路及八份原生任务 checkpoint，Face 和 Person 在 Frozen9 接收跨视图信息。融合使用另 10% TRAIN 原图。固定三路、旧任务封存、可用视图掩码与自然推理均沿用原版。A 以原人群、双模糊 BCE 学比例，B 在 A 的监督形式上增加自然图排序损失。C、D 复用相同两种目标，唯一改动为补进下述人物。两种损失均从等权起步，保持全批 Adam、恒定学习率 0.01、每任务 300 步。排序只在自然图当前情绪内配对，所有合法正负人物组合先对类内求均值，再对可比较类等权平均，系数和温度均为 1。

只补同一当前任务原本正在访问的留出原图中的其他已标注人物。本人至少有一项旧类标注，且对当前全部新类均已标为负。原图级分组，去重人物 ID，不用未来标签；按任务预计新增 0、58、16、70、74、91、6、53 人次，可靠 Face 为 0、28、13、39、44、55、4、34。当期模型对新增人物前向，使用与原档一致的自然、身体 224→112→224、可靠人脸 224→112→224 输入，保留原人群的全部既有预测。每个人有效版本等权，两个可用性组均学习比例。

选择标准执行前锁定。在全体自然 Validation 上比较最终 mAP 和八阶段平均，允许 1e-10 浮点容差。只有两项均达到且一项更高才算通过。C 须通过与 A 的比较，D 须分别通过与 A 和 C 的比较；优先通过双重比较的 D，其次为通过的 C，否则保留 A。B 和同组三路手设固定读出完整重放。先做 task0、1 真实小跑；八任务权重与 Validation 选择封存后，一次报告全部自然 Test。以 33.2019／40.2847 为全 TRAIN 固定 Test 参照。同一人物在多个任务可重复出现，新人次不能当作不同原图。前列正负例指标与各 Face 人群只用于解释结果。

入口 `bash scripts/emotic/run_d9_same_image_neg.sh 0`。图像和人脸清单沿用 `../multi-lane-main/datasets/EMOTIC` 及 `../emotic_benchmark_runs`；冻结 ViT 使用已有 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`。预训练分支与 TRAIN／Val／Test 预测来自 `../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal`，既有 A/B 来自 `../multi-lane-main-d9-natural-rank/output/d9_natural_rank/seed1_20261120/formal`。GPU0 提取，原 batch64，2 个数据加载进程，CPU 拟合与评价。输出 `output/d9_same_image_neg/seed1_20261120`，日志 `logs/d9_same_image_neg/seed1_20261120`。
