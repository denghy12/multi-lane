# ParaX 旧路径保护与跨任务专家复用：三任务配对实验

用户已授权实现并在 GPU0 并行执行。当前分支 `exp/parax-protected-expert-reuse`，起点 `8b6475d`。本轮不重复以前的全八任务实验，不加载 test。

## 实验回答的问题

以前持续更新共享专家会改变旧任务使用的参数；仅冻结专家而继续共享 Router 或投影，也不能保证旧路径不变。本轮同时保护旧专家、旧 Router、旧输出投影及专家访问集合，拆分“新参数容量”和“跨任务复用”。

| 方法 | task0 | task1 | task2 | 新任务是否读取旧专家 |
|---|---|---|---|---|
| 原共享基线 | 不使用 ParaX | 不使用 ParaX | 不使用 ParaX | 不适用 |
| 学完中心后整池冻结 | 学专家0、1 | 专家0、1冻结，只学当前 Router/投影 | 同左 | 是，没有新增专家 |
| 新任务只用新专家 | 学专家0、1 | 只学/用专家2、3 | 只学/用专家4、5 | 否 |
| 复用旧专家并学新专家 | 学专家0、1 | 组合冻结0、1与可训练2、3 | 组合冻结0–3与可训练4、5 | 是 |

“专家”指 ParaX 的一对低秩参数矩阵 E_A、E_B。Router 分别对矩阵进行加权组合，形成样本对应的有效矩阵，再计算残差；不是分别计算专家输出再加权。两个扩展组每个新任务同样增加两对矩阵，排除新增专家数量不同。三个 ParaX 组都预分配六个专家槽，但未来槽不参与计算；整池冻结组的四个备用槽始终不启用。可训练参数、有效专家数分别记录，不能把预分配总数当作有效容量。

## 数据流与冻结边界

三视图 → 共享冻结 CLIP 与 MULTI-LANE Task Forward → 每条任务 lane 的最终归一化 CLS → ParaX 参数路由与任务专用零初始化投影 → 再归一化 → 固定三视图**特征融合** → 共享分类头，按任务类别归属汇总预测。

ParaX 在 Task Forward 完成后，不在 Frozen Forward 的第11个 Transformer block 内。这里的逻辑 bank key `0` 不是 ViT block 索引。三视图共享专家池与同一任务 Router，没有三套视图专家。每个任务有自己的 Router 和 rank32→rank32 小投影；投影不是此前 CLIP patch-token Projector。该投影零初始化，专家矩阵随机非零，第一步有投影梯度，随后专家/Router 可以获得梯度。

每个任务结束立即封存当前 Router、投影和该任务新增专家。旧任务 softmax 前的硬访问 mask 永久不变，未来专家的 gate 精确为零。LayerNorm、固定 scale 与未使用共享 projection 从一开始不训练。独立 ParameterList 保护冻结专家，避免对可训练大张量只做切片梯度掩码导致 Adam 动量或 weight decay 修改旧槽。推理遍历全部已见 task lanes，不使用测试样本的真实 task ID。

## 完整配置

- EMOTIC Track-A，train 优化，validation 报告；seed0，task0–2；无 test。
- 30 epochs/task；train/eval batch64；workers2；Adam 每任务重置；主学习率0.0125（source0.05×64/256），ParaX0.0004；weight decay0；cosine min0、warmup0。
- CLIP ViT-B/16 完全冻结；shared Selector10、Prompt10、前5层 prefix；Image-token Adapter与CLIP patch Projector关闭；无蒸馏/level embedding。
- rank32，Router hidden16，6个预分配参数矩阵对；task0使用2个，扩展组每任务新增2个。
- task-local projection 严格零输出起点，fixed scale1，逐token可微残差比例界0.02。实际送入再归一化前的残差受约束；无硬裁剪。
- legacy_full_zero/joint_bce；融合BCE + 0.1可靠视图辅助BCE；固定特征融合权重可靠Face[0.64,0.16,0.20]，不可靠[0.80,0.20,0]。
- AMP初始scale1024、growth interval1e9；其余TF32设置沿用 runner。
- 新远程独立 worktree：`/mnt/haoyuan/workspace/multi-lane-main-parax-protected-expert-reuse`。
- 输入：`../multi-lane-main/datasets/EMOTIC`；CLIP `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face manifest `../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`，只读取train/val。
- 入口：`scripts/emotic/run_protected_parax_validation.sh`；四组启动：`scripts/emotic/launch_protected_parax_validation.sh`。
- 输出：`./output/emotic_protected_parax_val/<batch>/<method>`；日志：`./logs/emotic_protected_parax_val/<batch>`。
- 产物：task/per-view mAP、forgetting、训练历史与梯度、固定anchor原始输入/特征/logits/gates、旧参数逐值审计、paired初始化/CLIP/sampler hash、compact checkpoints、四组比较JSON。

## 审计与判断

四组同seed初始化与独立task sampler。ParaX初始化不推进全局RNG。固定16个validation样本仅用于评估，不参与优化，额外审计隔离Python/NumPy/CPU/CUDA RNG。

每任务结束逐值检查所有旧路径ParaX参数及mask；不变条件失败直接中止。报告旧类别logit漂移、三视图feature cosine和gate漂移；FP32评估仍有CUDA数值误差，须与同批基线漂移比较，不用“绝对零”误判正常浮点差异。当前任务ParaX初始logits须与关闭ParaX路径对齐。

- 旧参数不变但旧预测明显变化：检查上游Selector/Prompt/head及访问路径，不能宣称论文策略失效。
- 旧预测稳定但低于基线：增量保护有效，当前任务适配收益不足。
- 复用组不优于仅新增组：没有知识复用收益证据。
- 复用组优于仅新增和基线且旧预测稳定：才支持继续八任务validation，仍需后续多seed确认。

首次先顺序运行四组真实三任务smoke，每任务4次优化更新。确认配对、无skip/NaN/OOM、梯度和旧状态保护，再按显存预算在GPU0四组并行。若已有任务占用导致不足，启动器有限等待显存，不终止现有进程，不使用GPU1–7。正式实验启动后不持续监督。
