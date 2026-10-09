# Full+Person / Full+Face 的独立贡献：锁定实验配置

## 目标与对照

稳定 AMP 三 seed 表明三路优于 Full，但无法分开两个辅助视图的独立贡献。只新增 Full+Person、Full+Face 的从头训练。对照为同 seed 已完成的 Full 与三路无 ParaX 模型，不重新训练这两个参照。

## 完整配置（运行前声明）

- EMOTIC train 16,001 人物、validation 2,397 人物；26 类同时训练，一个 Task Forward pathway；最终第 30 轮 checkpoint 再做固定 test 5,368 人物。训练不加载 test、不用 test 选模型或参数。
- 共享且冻结 OpenAI CLIP ViT-B/16；Selector 10、任务提示 10、提示层数 5、共享线性分类头，归一化模式 pre-head。关闭 ParaX、Image-token Adapter、视图私有参数和动态融合。
- Full+Person 固定特征融合 0.80/0.20；Full+Face 从原 0.64/0.16/0.20 先验删除 Person 后归一为 0.76190476/0.23809524；Face 不可靠时为 1/0。可靠性沿用有效、非歧义、短边≥24、检测分数≥0.6。
- 固定融合发生于 Task Forward 各视图归一化任务特征之后、共享分类头之前。损失为 BCE(fused)+0.1×可用单路 BCE 均值；不可靠 Face 不参加 Face 辅助损失。未启用视图不经过编码器，也不参加损失。仍使用相同三视图数据变换，以保持数据与随机增强协议一致。
- seed0、1、2；每组30 epochs、batch64、eval batch64、workers2，7,530 次更新。Adam、主LR0.0125、WD0、cosine、无warmup、末LR0；temperature1、threshold0.5。
- AMP/TF32 开启；初始 loss scale1024、growth interval1e9；任何跳步导致严格比较失败，保留失败产物，不自动调参补跑。
- Full 为原 legacy 随机裁剪（scale0.05–1.0），Person margin0.15、letterbox、无颜色抖动；Face 沿用 manifest 裁剪。确定性 validation/test 变换与原模型相同。
- 输入相对路径：`../multi-lane-main/datasets/EMOTIC`、`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`。训练 Face root=`../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`；test root=`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。test扩展清单必须与训练train/val hash及检测器配置一致（仅splits可扩展）。
- 分支`exp/joint26-auxiliary-view-contribution`，Git-only独立worktree；保留main和test-only工作树。训练入口`scripts/emotic/launch_joint26_auxiliary_view_contribution.sh`。
- GPU0同时训练两组，按seed0→1→2；每seed两组结束后GPU1做固定最终checkpoint test。没有GPU teacher或离线蒸馏，不修改原结果。
- 输出`./output/emotic_joint26_auxiliary_view_contribution/<batch>/seed<seed>/<method>/`；test在同seed的`test/<method>/`；日志`./logs/emotic_joint26_auxiliary_view_contribution/<batch>/`；保存配置、审计、history、最终compact checkpoint、validation/test及单路分数、同样本比较表。
- 先运行全套unittest、shell语法检查、真实ViT两组各4updates smoke；smoke不做test。确认两组与原三路对照的初始Selector/Prompt/classifier、首批训练ID、全量train/val ID、CLIP hash一致后再启动正式队列。

## 判定与限制

validation配对均值与各seed方向为主，test仅锁定报告，不据其搜索权重。报告Full+Person减Full、Full+Face减Full、三路减两个两路模型；不将mAP增益简单相加解释交互。

若两个辅助视图都独立有效，保留三路，后续关注稳定融合；若只有一个稳定有效，优先保留有效的两路简化模型；若两路都弱但三路有效，保留互补性解释并进一步做错误分析。这些是下一轮决策条件，不预先自动扩展实验。即使全量两路有效，也需之后单独验证增量任务；本轮不提供增量遗忘结论。

源码版本发生变化，但参照模型不重训；比较必须核验关键超参数、初始参数、样本ID和冻结权重的实际hash，明确记录各自Git来源，不能声称相同Git HEAD。
