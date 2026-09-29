# 共享 b32 Image-token Adapter + P-10 ParaX 组合版本

## 代码位置与问题

`MultiLaneModel` 已允许 `adapter_mode=image_token` 与 `parax_mode=image` 同时启用；此前 ParaX 入口显式设置 `--adapter-mode disabled`，因此没有检验在稳定的共享 b32 Adapter 上额外加入 ParaX 是否有收益。本次新增一组严格配对的 test-only 入口，不修改模型前向。

数据流为：同一个冻结 CLIP ViT-B/16 按顺序处理 Full、Person、Face；共享 b32 Image-token Adapter 在视觉块索引 1 的 Selector 读取位置生成临时适配 tokens，不写回图像 residual stream；共享 ParaX 在视觉块索引 10 之后、11 之前仅修改主流 patch tokens，CLS 旁路。Task Forward 与图像流逐块并行。三路最终 task 特征按可靠 Face 固定权重融合，再经过共享线性分类头。固定权重为可靠 `[0.64, 0.16, 0.20]`、不可靠 `[0.80, 0.20, 0]`。

## 配对运行臂

| 方法 | Image-token Adapter | ParaX |
| --- | --- | --- |
| `B32` | shared、bottleneck 32、层索引 1、固定 residual scale 0.03 | 关闭 |
| `B32_P10` | 与 `B32` 完全相同 | image-stream、层索引 10、rank 32、3 experts、router hidden 16、无 level embedding、zero-output 初始化、固定 scale 0.001、task0 学 center，随后冻结 center 只训练 router |

两臂其余配置相同：EMOTIC Track A，train split 上增量训练 8 tasks，30 epochs/task，batch size 64，seed 1，冻结 CLIP，shared Selector/Prompt/head，`legacy_full_zero`，主参数 BCE、Adapter 和 ParaX 参数 ASL（`gamma_neg=9.8`、`gamma_pos=0`、`clip=0.05`），三视图辅助损失权重 0.1，Adam（主学习率 0.0125、Adapter/ParaX 学习率 0.0004，weight decay 0），cosine scheduler，AMP/TF32。跳过 validation evaluation，只在锁定的 held-out test 逐 task 评估；不保存 checkpoint。注意在当前优化器中 ParaX 与 b32 Adapter 分属两个参数组，但两者都按 Adapter 的 ASL 目标和 0.0004 学习率更新。

入口为 `scripts/emotic/run_multilane_track_a_b32_parax_p10_test.sh`。`DATA_ROOT`、`CLIP_CHECKPOINT`、`FACE_MANIFEST_ROOT` 默认使用项目相对路径；服务器实际资产若不在默认位置，应通过同名环境变量指定。两臂共享一个 `RUN_ID`，各自写入：

- 结果：`./output/emotic_track_a_b32_parax_p10_test/<RUN_ID>/<METHOD>/`
- 日志：`./logs/emotic_track_a_b32_parax_p10_test/<RUN_ID>/<METHOD>.log`

运行示例（在服务器 clean worktree、激活 PyTorch 环境后，按真实资产位置设置路径）：

```bash
RUN_ID=b32_parax_p10_seed1_locked SEED=1 GPU=0 METHOD=B32 \
  DATA_ROOT=./datasets/EMOTIC CLIP_CHECKPOINT=./models/clip/ViT-B-16.pt \
  FACE_MANIFEST_ROOT=./output/emotic_face_manifest/face_manifest_audit_v1_20260908 \
  bash scripts/emotic/run_multilane_track_a_b32_parax_p10_test.sh

RUN_ID=b32_parax_p10_seed1_locked SEED=1 GPU=1 METHOD=B32_P10 \
  DATA_ROOT=./datasets/EMOTIC CLIP_CHECKPOINT=./models/clip/ViT-B-16.pt \
  FACE_MANIFEST_ROOT=./output/emotic_face_manifest/face_manifest_audit_v1_20260908 \
  bash scripts/emotic/run_multilane_track_a_b32_parax_p10_test.sh
```

## 运行前检查和解释边界

服务器先运行完整 unittest、`bash -n` 和两组真实 ViT smoke。组合 smoke 必须核对：ParaX 严格零输出时初始 logits 与 `B32` 对齐；Image-token Adapter 与 ParaX 均有有限非零梯度；CLIP 参数梯度为零；task1 后 center 冻结而 router 与该 task 的 b32 Adapter 可训练。若上述任一项失败，不启动正式训练。

正式结果同步后，核对两臂配置、8 tasks、240 epochs、update/skipped 数、单路和融合 mAP、forgetting、每个 view 的 gate 与 residual/token ratio。此前 ParaX 单独运行和 b32 单独运行的指标不能替代本批配对对照。ParaX 小残差配置曾在 held-out test 上表现较稳，故这组组合实验应作为**探索性验证**；不使用本批 test 结果继续搜索 scale、rank、层位或融合权重。如需确认泛化，后续另锁定独立数据或协议。

截至本文件创建时，仅完成本地脚本语法和测试文件编译检查；未提交、推送或启动服务器 smoke/full 训练。本机 Python 没有 PyTorch，集成测试需在服务器环境运行。
