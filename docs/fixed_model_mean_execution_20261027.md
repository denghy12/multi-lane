# 固定整模型均值：完整前向验证与后续研究边界

完成更新（2026-10-27）：R5已用b017389源码完成GPU0/seed1/旧标准完整8任务Val/Test与隔离测速，无训练。Val44.1550/51.2964，Test33.6248/40.5129；对A Test+.7384/+.6442，两个split均8/8阶段升。对既有R2 Test+.5718/+.4658。与当批logits均值Test最终差−.000934/均值+.000461，最大logits差.0078125。全部来源预测精确、24源SHA及参数保持；507测试（1跳过）通过。A/R5隔离B64耗时241.06/492.98ms，约2.045倍，非视觉参数2661658/6882842，峰值925.71/946.22MiB。266产物已回传并逐SHA核验，归档f330a43266eadd7b08d4bb1250796e5d415f82cb8174d3bff73372d2158b9f1e；本地64融合指标/128来源视图/64logit均值关系通过，误差7.11e-14，日志已入logs/。主树/test-only状态保持，GPU作业已结束。

固定均值完整函数组合现可作为更强实测参照，但仍为双来源，未证明新Attention因果收益或单模型训练成功。后续单模型方案docs/single_model_function_transfer_plan_20261027.md，保留三视图输入、同增强教师、融合功能约束、ASL/BCE分组及同预算U0–U3；仅为研究方案，未新增训练。报告docs/fixed_model_mean_results_20261027.md。以下启动/等待描述均为历史记录。


用户于2026-10-27授权先验证固定均值强对照，再研究单一推理模型。当前仅执行无训练的固定均值验证，不自动启动尚未定型的蒸馏/统一训练候选。分支 `codex/fixed-model-mean-forward-20261027`。

## 要回答的问题

此前CPU读取历史预测，固定 `.5 z_A + .5 z_B` 得到 Val最终/平均44.1558/51.2957、Test33.6258/40.5125。该数值还不是新模型的特征融合前向。本轮从图片完整运行两个来源，验证相同功能是否能在共同B0头之前实现，明确AMP误差和资源成本。固定系数不搜索，Test按既有授权参与探索；单seed与旧探索选择局限仍保留。

## 数据流与控制

```text
Full / Person / Face图片 + Face可靠性
             │
       ┌─────┴─────┐
 A完整三视图模型   B完整三视图模型
 自己的A/S/P及M   自己的A/S/P及Person/Face M
 自己的Full摘要   自己的Full摘要
       │                 │
 原固定三视图融合   原固定三视图融合
       h_A               h_B
       └───── 0.5 / 0.5 ──┘
                  │
        同阶段共同B0线性分类头
                  │
       所有已见任务的类别归属输出
```

A=Person-only共训完整三视图表征；B=Person+可靠Face共训完整三视图表征。两者都换对应任务边界的B0头。冻结ViT权重和头对象共享，但两个来源仍各运行三路图像流；没有tokens缓存，不声称单模型或低成本新融合。保留B内部Attention读取B自己的Full摘要，最终平均不改内部依赖。

R5先分别得到各来源已融合任务特征，再计算`.5 h_A + .5 h_B`，不加L2归一化、额外Norm、门控或拟合分类头。每个阶段同时保存A/B原来源与LOGIT_MEAN控制，不重复骨干执行。源视图固定权重可靠Face为.64/.16/.20，无可靠Face为.8/.2/0。每个任务输出本任务类别，不挑选效果最好的任务路径。

## 完整运行配置

- EMOTIC；seed1；完整8阶段validation和test；沿用`samples_intersect_seen_classes`，阶段mAP取算术平均，分类阈值概率>0.5。
- GPU0；batch64；评估workers2，隔离测速workers0；OMP/MKL72线程；AMP/TF32。无训练，无优化器、LR、scheduler、训练epoch和参数更新，无新训练checkpoint。
- 冻结CLIP ViT-B/16；来源保留Image-token Adapter b32，第2层、scale.03/ReLU和ASL学得参数；保留第6层摘要Attention（inner64/heads4/scale.1/cap.1）。
- 数据 `../multi-lane-main/datasets/EMOTIC`。
- CLIP `../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- B0 `../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- A：同来源根的 `ATTENTION_BOUND/train`。
- B `../multi-lane-main-head-anchored-eight-face-20261025/output/head_anchored_eight_face/formal_seed1_20261025_01/FACE_JOINT/train`。
- 逐值来源参考 `../multi-lane-main-selector-face-controls-20261026/output/selector_face_controls/formal_seed1_20261026_01`。
- 输出 `output/fixed_model_mean_forward/formal_seed1_20261027_01`，日志 `logs/fixed_model_mean_forward/formal_seed1_20261027_01`。

## 验收与执行次序

1. 语法/差异检查，远程新增专项及现有完整单测。
2. 只读验证24份来源checkpoint的SHA。task0/2/7各Val首64人，用完整模型前向精确重放两个来源的全部视图和融合logits、历史batch sigmoid概率、IDs/labels/可靠性。
3. A和R5各自独立进程，在task7首64人隔离测速：3次warmup/10次测量，CUDA事件、wall time、参数数、峰值allocated/reserved、profiler覆盖FLOPs。不会与并发评估混测。
4. Val/Test两进程并发，完整8阶段。每batch来源逐值复核，R5与当批LOGIT_MEAN逐视图记录最大绝对logits误差，预声明AMP容差.03125。容差只检查数值实现异常，不代替mAP对比；最终报告实际mAP差。
5. 完成后再次核对24源SHA、任务参数与冻结ViT摘要。导出每阶段每组NPZ、指标、summary、资源表、审计与完成标识；回传并校验本地产物。

入口：

```bash
python -u -m multi_lane.track_a.run_fixed_model_mean_forward \
  --run-id formal_seed1_20261027_01 \
  --source-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train \
  --person-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/ATTENTION_BOUND/train \
  --dual-run ../multi-lane-main-head-anchored-eight-face-20261025/output/head_anchored_eight_face/formal_seed1_20261025_01/FACE_JOINT/train \
  --reference-root ../multi-lane-main-selector-face-controls-20261026/output/selector_face_controls/formal_seed1_20261026_01 \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
```

服务器独立worktree通过Git-only部署，不改远程主树、test-only或旧实验。既有主树/test-only的HEAD与status SHA须保持。当前实现阶段尚无R5正式指标。
