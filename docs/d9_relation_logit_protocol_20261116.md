# 从第 9 层跨视图关联学习最终融合比例

EMOTIC 原划分，seed1、8 个增量任务。Full 保留原图像路径；Person 与可靠 Face 在 Frozen Forward 第 9 块后做空间 patch 交互。三路各自的 Task Selectors、Task Forward 和分类头输出当前任务 logits。按同一原图和目标人物取三个裁剪，发送端用历史确定性变换，接收端用对应分支训练增强。Face 训练有效性和推理可靠性分别沿用原 manifest。

三组都从预训练 CLIP ViT-B/16 完整训练，使用每个当前任务全部合规 TRAIN 人物。每组同一条轨迹中更新分支和融合。八阶段验证集与测试集只做评价。共同目标保留各分支 BCE 和针对图像 Adapter 的 ASL 梯度，再把当前类别融合 BCE 反传到本任务分支和可学习的比例。与此前独立训练的 D9 做外部参照；从三组内部的直接对照归因学习比例的作用。

| 组别 | 决定三个分类头的 logit 比例 |
| --- | --- |
| fixed | Face 可靠时 0.64／0.16／0.20，其余 0.80／0.20／0 |
| static | 每个任务两个可学习相对分数，对本任务人物共用 |
| person | 每个任务根据同一目标人物第 9 层 Person–Full、Face–Full 的关联产生两个相对分数 |

Full 的相对分数为零。两个可学习组从相同固定比例出发，经可用视图 softmax 归一化。Face 无效时在 softmax 前屏蔽。比例网络只使用中间跨视图信号：Full 得到的注意力总量、Full 内匹配的集中程度、接收 patch 与对齐 Full 的 Q/K 及 V 空间相似度。各组使用独立分类头的未激活分数加权，最后应用 sigmoid。进入后续任务后封存旧任务的参数与融合规则。

8×30 轮，batch 64（同一批人物可拆为多个微批累计），Adam，cosine，衰减 0；主参数学习率 0.0125，Adapter、跨视图交互、比例网络学习率 0.0004。微批内 FP32 计算关联及损失，主前向 BF16 autocast。完整协议和实际状态以各组 config、日志和 checkpoint 为准。固定评估两 split 的 8 阶段最终 mAP 和阶段平均 mAP、可靠／不可靠人群、单路表现和同群人物的比例置换。历史参考为 Validation 44.0329／51.1335、Test 33.1770／40.2083。

Linux 入口为 `bash scripts/emotic/run_d9_relation_batch.sh smoke|formal GPU MODE...`，各模式见表。输入为 `../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP/pretrained/clip/ViT-B-16.pt` 和 `../emotic_benchmark_runs` 中原配方及人脸 manifest。输出为 `output/d9_relation_logit/seed1_20261116`，日志为 `logs/d9_relation_logit/seed1_20261116`。训练产物按阶段保存，先验收两任务快速运行及真实数据读出。

GPU0 真实 64 人训练批的预检中，微批 16、32、64 的第二次更新时间约 0.88、0.50、0.35 秒。正式使用整批 64，三组同时运行；该测速没有包含整轮读取数据、存盘和两进程同时占用同一卡的时间。两接收分支复用同次前向的 Full 冻结发送 patch。
