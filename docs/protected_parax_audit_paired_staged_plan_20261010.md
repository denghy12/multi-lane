# ParaX 对齐修正、残差消融与分阶段训练计划

用户已授权按上一轮分析开始修改。分支 `exp/parax-audit-paired-staged`，起点 `de9b0f8`。保留既有未提交文档和所有历史实验，不覆盖或恢复原失败批次。

## 本轮做什么

1. 修正 task2 被数值误差误停的 identity 审计，并以统一修正版从头进行四组配对短程 validation。
2. 每个已训练 ParaX 模型额外关闭全部后置残差做验证推理，保存同样人物的融合/单路分数；不增加优化更新。
3. 实现并测试“先训练原 Task Forward，再冻结它、学习新专家”的能力。正式分阶段训练仍等待四组及残差消融结果，不自动扩大实验。

## 实现位置与准确含义

- `model.py`：受保护 ParaX 仍在最终归一化 Task Forward CLS 后、固定三视图特征融合前。共享专家池/任务专用 Router、投影不变。对零残差采用 `h + (normalize(h+delta) - normalize(h))`，使浮点输入精确保持，同时保留投影和上游梯度；原非受保护模式保持历史实现。非零时与直接再归一化的差异只是输入单位化舍入偏差，不引入新参数。0.02 约束仍针对再归一化前实际原始残差。
- `protected_parax_audit.py`：分别严格检查当前投影零值、相同缓存输入的原始残差为零、受保护后置路径的缓存特征/logits 精确对齐、gate 有限且未来专家权重为零。完整 FP32 前向关闭/开启各重复三次，独立记录误差。全前向差异界为 `4*max(关闭重复误差,开启重复误差)+8*FP32_eps*max(1,logit绝对值最大值)`；它不能替代前面的严格检查。旧参数/mask逐值检查不放宽。失败时也保存 identity JSON 并恢复模式/TF32/RNG。
- `runner.py`：增加可选残差关闭推理，输出到独立目录，保持原参数、训练配置和下一任务 RNG；增加显式 protected staged 开关，默认关闭。
- `post_task_calibration.py`：保留旧随机冻结中心校准；新增受保护池校准，允许当前任务的新专家矩阵学习，冻结所有 B0 参数及旧路径，结束逐值 hash 检查。只缓存 train split，当前 task 标签和固定可靠性融合。新专家/Router/投影分项梯度及新专家变化均记录。
- `protected_parax_compare.py`：只接受新分层审计结果；拒绝把分阶段方法混进联合训练四组。汇总新类/旧类 AP、固定 task0 人物 AP、残差开启/关闭差，以及关闭后相对独立基线的差。

## 正式四组配置

| 组别 | 新任务允许使用的专家 | 新任务可训练参数 |
|---|---|---|
| 原共享三视图基线 | 无 ParaX | 原当前任务 Selector/Prompt/分类参数 |
| 学完专家池后冻结 | 始终使用 task0 两对矩阵 | 原任务参数＋当前 Router/零初始化投影 |
| 仅新专家 | 每任务自己的两对新矩阵 | 原任务参数＋当前新专家/Router/投影 |
| 旧专家复用＋新专家 | 旧任务冻结矩阵＋当前两对新矩阵 | 原任务参数＋当前新专家/Router/投影 |

EMOTIC Track-A；seed0；task0–2；train 优化、validation 评估；不加载 test。每任务30epochs；train/eval batch64；workers2；Adam每任务重置；主LR0.0125（source0.05×64/256），ParaX LR0.0004；WD0；cosine最低0、warmup0。AMP初值1024、增长间隔1e9；审计FP32关闭TF32，训练TF32沿用原设置。

CLIP ViT-B/16完全冻结；Selector10、Prompt10、前5层prefix；两种额外Adapter/Projector关闭，无level embedding、蒸馏。rank32、Router hidden16；6个预分配参数矩阵对，task0启用2个、扩展组每task新增2个；fixed scale1，原始残差可微比例界0.02。legacy_full_zero/joint_bce；融合BCE＋0.1可靠视图辅助BCE。固定特征融合：可靠Face[0.64,0.16,0.20]；不可靠[0.8,0.2,0]。

仅GPU0，四组并行，启动前至少14500MiB空闲，有限显存等待，不停止其他任务、不使用其他GPU。

服务器专用worktree：`/mnt/haoyuan/workspace/multi-lane-main-parax-audit-paired-staged`。输入沿用已核验相对路径：数据`../multi-lane-main/datasets/EMOTIC`，权重`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，Face清单`../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`，仅train/val。

入口：`scripts/emotic/run_protected_parax_validation.sh`，启动器`launch_protected_parax_validation.sh`。正式 batch 预定 `protected_parax_audit_v2_seed0_val_20261010_01`；输出`./output/emotic_protected_parax_val/<batch>`，日志`./logs/emotic_protected_parax_val/<batch>`。每组预期90epochs、5010有效更新、0 skips；保存checkpoint、审计、task/单路AP、固定样本和残差消融分数、四组比较JSON。

先完整单测和bash检查，再四组真实ViT三任务smoke（每task4updates，每组12updates）；另做一组受保护可学习新专家的分阶段真实smoke（B0每task4updates、缓存train特征、每task1轮校准、LR0.0004、consistency0.1、辅助0.1、同样0.02界）。smoke不作为精度证据。

## 后续分阶段方案的边界

显式`--protected-parax-staged-training`开启时，B0阶段关闭ParaX并将其从优化器参数组排除；训练完当前任务后，从train split确定性裁剪缓存三视图特征，冻结Task Forward/head，恢复仅当前专家/Router/投影可训练性。受保护版本的BCE梯度按当前类数/26缩放，匹配legacy_full_zero当前类项；不活跃类常数不参与梯度。校准采用FP32 Adam/cosine、gradient clip1，当前任务logit一致性权重0.1，Face辅助只用可靠样本。默认5轮，当前仅做1轮真实smoke，不启动正式校准搜索。

新专家矩阵允许学习，区别于之前冻结随机center仅训练Router/投影的方法。额外训练预算、确定性裁剪、FP32及校准损失都必须单列，不能宣称与联合训练优化过程完全等价。缓存/校准/额外推理隔离Python、NumPy、CPU/CUDA RNG，保留下一任务B0可比性。

## 结果判据

先确认四组真正完成、无skip/NaN/OOM、初始参数/样本一致、CLIP和旧参数不变。旧AP使用固定同人群，避免常规forgetting混入扩展评估人群效应。

- 关闭残差恢复：优先检查残差方向及当前任务适配。
- 关闭残差仍低：提示原任务模块在联合训练中学习轨迹发生变化，不能仅归咎输出残差。
- 复用仅超过新增但不超过基线：保留弱复用信号，不升级八任务。
- 全部仍低：再正式比较相同专家预算的联合训练/分阶段学习，不同时改动态融合或扩大专家。

启动后停止监督，结束后统一分析。不会根据test调结构或融合权重。
