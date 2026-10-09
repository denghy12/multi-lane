# 全量学习四组最终 checkpoint 的 test 补评估

用户已授权：确认 ParaX 梯度日志去除 AMP loss scale、AMP 初始 scale1024且不主动增长，并补跑这几组实验的test。

前两项已由训练代码 `f068414` 实现，当前稳定AMP队列仍在GPU0训练，不修改运行工作树。**AMP loss scale只影响训练反向传播，test推理没有GradScaler，也无法事后改变旧模型的训练轨迹。**

## 覆盖范围

| 模型训练来源 | test安排 |
| --- | --- |
| 原始seed0四组：Full/三路 × 有无Frozen Forward内部ParaX | 最终checkpoint已存在，立即评估 |
| 稳定AMP seed0四组 | 同一seed四组全部训练结束后自动评估 |
| 稳定AMP seed1四组 | 完成后自动评估 |
| 稳定AMP seed2四组 | 完成后自动评估 |

共16个已训练/待完成模型的推理评估。原seed0与新seed0/1/2分开保存，不混算均值。所有组均使用第30个epoch的最终`compact_checkpoints/task0.pth`，不按validation/test成绩选择轮次，不重训模型。

## 完整配置

- 数据集EMOTIC原始test，全部26类、全部有标签的人物样本；数据相对路径`../multi-lane-main/datasets/EMOTIC`。
- 同一冻结CLIP ViT-B/16预训练权重`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，按官方SHA和训练时CLIP参数hash核验。
- Full-only或Full/Person/Face按源配置恢复。Image-token Adapter关闭；ParaX有无、Frozen Forward代码索引10、rank32、3个参数矩阵专家等均从源config还原；保留学习后的expert、router和output scale数值。
- 固定融合先验可靠Face为0.64/0.16/0.20，否则0.80/0.20/0；可靠性为valid、非ambiguous、短边≥24、检测分数≥0.6。Face manifest使用`../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908`中的test清单。
- eval batch64、workers2、threshold0.5、AMP/TF32与源模型一致；使用原确定性验证变换，不做训练增强。
- **训练epochs=0、optimizer updates=0、无learning rate/optimizer/scheduler**；`eval()`、`no_grad()`，全部参数冻结。
- GPU1同一来源四组并行，至少20,000MiB空闲才启动；四组完成后等待下一来源最终checkpoint。每个来源最长等待12小时，失败状态落盘；GPU0训练继续运行。
- 入口`scripts/emotic/launch_joint26_checkpoint_test.sh`，内部调用`run_joint26_locked_test`和`evaluate_joint26_checkpoint`；运行环境`/opt/conda/envs/ddp/bin/python`。
- 新分支`exp/joint26-checkpoint-locked-test`，独立服务器worktree`/mnt/haoyuan/workspace/multi-lane-main-joint26-checkpoint-locked-test`，Git-only同步，不改旧训练/test-only worktree。
- 输出`./output/emotic_joint26_locked_test/<run_id>/<来源>/<方法>/`；日志`./logs/emotic_joint26_locked_test/<run_id>/`。

## 恢复与不可变检查

先检查源训练30轮完成、全部26类、一个任务路径、冻结CLIP不变，校验compact checkpoint的source Git。恢复后用源validation第一批样本与保存logits对齐（容差0.01），通过才打开test样本。该replay只检查恢复，不选模型或调参。

比较前后方法参数、CLIP参数、源config/summary/checkpoint/scores文件hash，确认没有变化。保存完整test分数、每类AP、mAP/F1、三路单路与可靠Face指标、验证恢复误差和来源hash；同一来源四组test sample ID和targets必须完全一致。每个来源自动生成一张并列Validation/Test表。

原seed0训练各有2次AMP跳步，仍明确记录为探索性来源；补跑test不会将其升级成零跳步训练。稳定AMP来源的跳步数也如实记录。test结果只用于报告，不据此更改本批配置或重新选模型。

## 状态

已实现专用推理入口，等待远程单测后启动。后续只确认启动与原始四组完成情况，不持续监督等待中的训练。
