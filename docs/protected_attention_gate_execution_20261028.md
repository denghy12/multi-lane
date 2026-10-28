# Attention修正幅度与内容门控：现有checkpoint真实前向归因

2026-10-28用户要求执行诊断并据此研究Attention改进。先完成现有checkpoint归因，再决定条件性的重训；本轮没有optimizer、梯度或新增训练预算。

## 目标和输入

- EMOTIC，seed1，现有C4完整门控版task0/2/7 compact checkpoints；原三路强B0轨迹精确重放的3950b35来源。
- 目标一：确认TRAIN锁定缩放的实际特征前向能否重现离线logits诊断。目标二：在相同最终修正上限下区分内容门控、固定/均值门控及幅度变化。目标三：保持门控不变，仅屏蔽新增Face摘要，判断其条件价值。
- 原10共享Selectors/Prompts、Image-token Adapter b32/index1/scale.03、三路Task Forward、同一分类头均只读。原训练30epoch、B64、Adam/cosine/BCE-Adapter-ASL配方不变，但本轮训练更新为0，学习率/优化器不适用。
- 原checkpoint仅从`../multi-lane-main-protected-joint-attention-20261028/output/protected_joint_attention/seed1_20261028_01/C4/train`读取，配对RMS日志来自同run的C3。
- 数据`../multi-lane-main/datasets/EMOTIC`；标准CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

## TRAIN锁定系数

对每个任务t，记原C4的输出投影为raw，`b=B_0.1(raw;h0)`，原门控为γ(x)。在读取新Val/Test之前保存：

- s_t = C3最后5个TRAIN epoch输出RMS比均值 / C4相同统计；与上一轮离线TRAIN_ratio算法相同。
- k_t = C4最后5个TRAIN epoch门控均值。
- 来源history/config及三份checkpoint的SHA，不读取Val/Test选择常量。

这些是TRAIN日志均值，不意味着新Val/Test的实际平均RMS完全相同；新评估会逐任务报告均值和最大值。固定/动态的严格共同条件是最终RMS上界相同。

## 十个真实特征读出条件

| 条件 | 最终写入 | 目的 |
|---|---|---|
| native | γ·B_0.1(raw) | 精确重放原C4 |
| off | 0 | 精确重放B0 |
| train_scaled_dynamic | s_t·γ·B_0.1(raw) | 确认离线TRAIN缩放的真实前向 |
| train_scaled_constant | s_t·k_t·B_0.1(raw) | 日志均值匹配的常量控制，非严格1%上限组 |
| cap1_dynamic | γ·B_0.01(raw) | 共同1%上限下的原内容门控 |
| cap1_constant | k_t·B_0.01(raw) | 同上限、TRAIN均值常量门控 |
| cap1_open | B_0.01(raw) | 同上限固定全开 |
| cap1_permuted_gate | γ_donor·B_0.01(raw) | 保留门控分布但破坏与图片内容对应 |
| cap1_dynamic_no_face_fixed_gate | 原γ·B_0.01(raw_noFace) | 保持原门控，单独去掉新增Face读取 |
| cap1_constant_no_face | k_t·B_0.01(raw_noFace) | 保持常量门控，单独去掉新增Face读取 |

门控错配在每个batch内按Face可靠性/目标Full可见性分组，使用固定独立CPU RNG，保留每组门控值分布，不读取标签。Face干预保留h0及Person Query，调用原no_face读取获得raw后丢弃其改变过的门控，严格使用原接收者门控或同一k_t。

同一batch只编码一次三视图和原任务路径；用临时hooks获取原raw/gate输出，不改训练或模型实现；各条件通过真实h0+delta接原分类头。不得把AMP舍入logits插值当本轮特征前向。

## 执行与产物

- GPU仅0，seed1，B64、OMP/MKL/model线程72、AMP/TF32，历史CPU FP32每batch sigmoid；2个单线程spawn workers/prefetch1。
- 专项测试和相关回归后，先task0/2/7 Val各首64 smoke；原动态、OFF、三路原输出必须逐batch精确重放；参数/源SHA不变，统一cap条件不得超过1%RMS。
- 通过后GPU0并行完整Val/Test两进程，启动前空闲至少4500MiB及共享内存768MiB，间隔8秒；不得中断其他GPU作业。
- 入口：`bash scripts/emotic/run_protected_attention_gate_gpu0.sh seed1_task027_20261028_01`。
- 结果`./output/protected_attention_gate/seed1_task027_20261028_01/`，日志`./logs/protected_attention_gate/seed1_task027_20261028_01/`。
- 每条件每阶段NPZ含完整人物预测、标签、ID、mask、历史batch边界、native/permuted gate与实际feature_ratio；JSON含mAP/F1/子群、门控均值/标准差/接近全开比例、写入RMS及BCE效用相关。
- task7为26类最终结果，但仅0/2/7三个阶段，均值只能称所选阶段均值，不能称8任务阶段平均。Test参与诊断，全部十组保留，无Test选比例或早停。
- 本轮不重训；若内容门控没有优于常量/错配，不自动启动第三项门控训练或扩大Attention容量。下一训练必须先统一最终写入上限并保留MLP控制。

原服务器主/test-only状态SHA为ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf、e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152。独立Git-only诊断worktree，原模型3950b35保持clean；所有源码变更限新增诊断文件，不改原model/runner/readout。
