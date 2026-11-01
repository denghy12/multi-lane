# 基于历史独立三视图 R1 的 CrossViT 式改进（2026-11-01）

## 实验起点

本轮直接回答用户的最新要求：从独立 Full、Person、Face 专家固定 R1 融合继续改进，而不是从共享参数 B0 或上一轮联合损失版本继续。

历史三种子 Test 最终 mAP 是 33.0119 ± 0.3038，阶段平均是 40.0608 ± 0.2564（样本标准差）。seed0/1/2 的最终/阶段平均分别为 33.3392/40.2855、32.9575/40.1154、32.7390/39.7815。本轮先沿用 GPU0、seed1，配对参照是 32.9575/40.1154，不能把单种子与三种子均值直接作为配对差值。

需要校正名称：历史代码融合的是各分支 sigmoid 后的概率，而非原始 logits。可靠 Face 的固定权重是 Full/Person/Face=0.64/0.16/0.20，不可靠 Face 回退 0.80/0.20/0。主结果严格保留这个规则；相同权重的 logit 融合仅作独立读出对照。

## 为什么重新建立对照

上一轮“独立分支”实验同时引入融合 BCE、辅助权重 0.1、scale 0.03 和 logit 融合，没有严格承接历史 R1。它不能否定历史独立专家的有效性，也不能作为本轮 CrossViT 的公平参照。

新分支从历史 Face 源提交 acf4634 建立；Full/Person 源提交为 d2442ee，禁用的条件查询之外两者原模型运算一致。历史源没有保存任务 checkpoint，因此只读取原配置和预测用于核验，不加载历史学习参数。三路均从标准冻结 CLIP 自行训练。

## 三组结构与问题

| 组别 | Task Forward 中的新增结构 | 回答的问题 |
| --- | --- | --- |
| native | 无交互，逐分支重放历史训练 | 历史强底座是否真正恢复 |
| attention | 第6个 block 后，任务 CLS 作为 agent，以其他两视图冻结空间 patch tokens 为证据做 cross-attention | 独立专家能否利用跨视图内容改善自身判断 |
| pool | 同位置、同证据来源的池化 MLP，参数量与 Attention 接近 | 收益是否只是多看到其他视图，而非动态注意力读取 |

三个接收分支各有独立 Adapter、Task Selectors、Prefix Prompts、分类头、任务私有交互模块；不共享可训练参数。其他视图提供同一 TRAIN 人物的确定性原生评估变换，只读取冻结骨干前6层 spatial patch tokens，不读取其他模型的可训练 Task Forward 输出。接收分支保留历史随机增强、筛样和每个任务更新数量。与原 CrossViT 的双向可训练分支交换不同，这是冻结证据的三接收分支改造，不宣称完整复现 CrossViT。

接收路径：本视图冻结图像流 → 原 Summarize/Task Forward 前6层 → CLS agent 加性更新 → 原后6层 Task Forward → 本视图分类头。外部 Full/Person/Face 的冻结 patch tokens 作为 K/V；agent 自身 CLS 也进入 Attention 的 K/V 集合。原 Selector/Prompt token 不被交互直接改写。新 readout 使用 FP32 和零输出初始化，无额外 FFN；原 Task Forward 保持历史 AMP，不暗中改变原路径精度。

每个分支只使用原本的本视图 BCE 与 Adapter-ASL。无融合训练 loss，无辅助 0.1，无教师、旧 A/B 参数拼装或换头。交互参数接受接收分支 BCE；Adapter 继续接受 ASL。旧任务的整个任务状态及交互模块冻结；新的冻结图像证据不随任务变化。

可靠性 mask 在 Attention softmax 前应用。Face 训练依旧仅要求 valid_face && !ambiguous_match；可靠 Face 参与融合和外部读取的门槛是短边≥24px、检测分≥0.6，不能混成同一种筛样。

## 固定训练与评估配方

- EMOTIC，历史 train/val/test 划分，8任务，类别数 5+7×3；seed1，每任务30 epoch，batch64，原训练 workers2。
- 冻结 OpenAI CLIP ViT-B/16；10 Task Selectors、10 Prefix Prompts、5个前缀层。
- Adam、weight decay0、每任务重置；主 LR0.0125、Adapter及交互 LR0.0004；cosine、无 warmup、最终LR0。
- Image-token Adapter：索引1、b32、scale0.1、ReLU、任务独立初始化；Adapter-ASL γneg9.8/γpos0/clip0.05。
- Full：历史随机裁剪0.05–1；Person：margin0.15、letterbox、裁剪0.7–1、颜色扰动strength0.1/probability0.2；Face：历史 pad-square/resize/flip。
- 原72线程初始化和训练随机流；新增模块初始化隔离随机流，新增确定性证据不消耗增强随机数。
- 交互索引5（第6层后）、inner64、4 heads；每任务/每视图独立；无区域约束、门控或专家池。
- 训练中保持原 current-class Validation 与 seen-class Test 的调用顺序；完整 seen-class Validation 在独立进程补评，避免额外 DataLoader 改变下一任务随机流。
- 历史评估人群 samples_intersect_seen_classes；报告最终 mAP 与8阶段算术平均、逐视图指标。关闭交互推理是同一共训模型的条件贡献，不是重新训练的 native。

入口：`bash scripts/emotic/run_r1_crossvit_gpu0.sh`。GPU0最多3进程，显存及4GB共享内存门槛自动排队，评估最多2并发，评估线程降至2。原训练线程保持72以承接历史配方。

默认相对输入为 `../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP/pretrained/clip/ViT-B-16.pt`、`../emotic_benchmark_runs/` 下原配置及 Face manifests。输出 `output/r1_crossvit/seed1_20261101_01/`，日志 `logs/r1_crossvit/seed1_20261101_01/`。每 epoch 保存含 optimizer/scheduler/scaler/RNG 的恢复状态，每任务保存 compact checkpoint、预测、指标。历史 checkpoint 不覆盖。

## 执行顺序与判读

1. 单元测试和9个真实GPU短 smoke（每任务2更新、2任务、仅Val）全部成功，才能进入正式训练。
2. 先并行重放3个 native 分支。逐任务 Test mAP 和 seed1 R1 最终/平均须在1e-4点内重放历史结果；同时记录对齐预测差值。未通过则停止新交互组，先查复现差异，不降低标准继续比较。
3. native 重放通过后自动排队 Attention、Pool 各3个分支，完整8任务与 Validation/Test。所有结果均报告，不按 Test 挑 checkpoint 或融合权重。
4. Attention 必须同时优于配对 native 和同输入 Pool 的 Test 最终/阶段平均，才支持注意力读取有额外价值。验证集方向、各视图变化及可靠性人群一并解释，三种子稳定性仍须后续验证。

截至本记录建立：本地实现和静态检查完成，尚未启动远程正式训练。后续以同目录 protocol/completion/reproduction_gate 文件为准。
