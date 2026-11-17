# 第9层跨视图关联能否预测哪一路能帮助当前人物

EMOTIC 历史划分，seed1，8 个任务各 30 轮、batch/microbatch 64、2 workers、GPU0。三路从历史 CLIP ViT-B/16 开始训练，沿用受保护固定组的每批人物、训练增强、Face 筛样、原 BCE / Adapter-ASL 和 Adam、cosine 日程（主 LR 0.0125、Adapter/交互 LR 0.0004、weight decay 0），确定性 BF16。在 Frozen Forward 第9块后，Person 和 Face 进行 patch 交互。各自的 Task Forward 和分类头保持原监督。来自 TRAIN 的门控融合/方向损失对中间关联与分类头 logits 停止梯度。完整 TRAIN 指当前任务至少一项新类为正的原协议人物；同一模型同期学习，未额外划留出。

| 读出 | 中心 | 训练目标 |
|---|---|---|
| fixed | 历史固定可靠 64/16/20、不可靠 80/20/0 | 不训练 |
| a | 历史固定 | 直接最小化融合 logits 的当前类别 BCE |
| b | 同 a | 在本人当前任务三路输出和标签上生成两个方向标签，预测单独从 Full 转移 5 个百分点到 Person 或可靠 Face 时，BCE 是否严格降低 |
| c | 同 a | 同逻辑 batch、任务、Face 可靠性内错配 b 的整组标签，再预测方向 |
| center | 可用视图等权起步 | 仅以 TRAIN 融合 BCE 学任务私有两个相对分数 |
| center_b | 与 center 共用同一组权重 | 一套独立的方向读出以该训练批次更新前的中心生成标签，与 b 同法学习。Full 不足 5 个百分点时，转移其现有份额 |

A/B/C 使用相同的每视图4→1无偏线性读出，合计8参数；center_b 同形状且独立。每个方向的未限幅分数用于二元方向监督；部署时用 tanh 限至 ±1 相对 log-odds 并加到中心后做掩码 softmax。同一任务多人配比可不同，同一人的任务内类别共用该任务配比。无效 Face 的融合权重为零。方向标签为零含无改善或并列；置换按组随机无自配，单人组在 B/C 方向监督中同步屏蔽。对每个任务均匀选 300 个逻辑批次（每轮 10 个），所有门控使用同批次、各自的 Adam、学习率 0.0004，300 步余弦日程。正式训练人数和分支预算保持不变。center_b 使用同一轨迹自身中心，训练和推理均不依赖 64/16/20。5 个百分点是方向标签的探测额度，±1 是固定的 log-odds 调整上限，均为预定义超参数。独立探测一种小调整的方向正确，未必保证同时改变两路后 mAP 提升。

预先固定六种读出及评估顺序。在 Validation 对每个任务头和 Face 可靠性组核对 B vs C 的方向 AUC、实际转移收益及本人 vs 同组均值／置换的 mAP。AUC 有利类或无利类缺失时记缺失。分别比较 B vs A/C、center_b vs center；随后看相对受保护固定组 Val 43.6272/51.1094、Test 33.2019/40.2847 的最终和八阶段平均。所有六种读出用同一组 checkpoint。Validation 同组平均仅使用无标签权重，再锁定到 Test。两套数据只评价，Test 不选择设置。

真实两任务预检和正式训练入口分别为 `bash scripts/emotic/run_d9_direction.sh smoke 0` 及 `bash scripts/emotic/run_d9_direction.sh formal 0`。输入沿用 `run_r1_frozen_cross_view.input_paths()` 的 EMOTIC、Face manifest、CLIP。每个任务保存完整可恢复训练状态及 compact checkpoint。输出在 `output/d9_direction/seed1_20261118/`，日志在 `logs/d9_direction/seed1_20261118/`。在评估前与受保护固定组的训练人群、更新数及各分支状态比较，真实前向再逐人核对三路 logits 与固定读出。归档每个读出的 logits、权重、方向标签与诊断。
