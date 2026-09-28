# ParaX validation 候选的 held-out test 重跑方案

## 目的

此前 ParaX 结构和稳定性实验主要只在 validation 上运行。为满足导师要求，本批把历史 ParaX validation 候选按原训练配置迁移到统一的 held-out test-only 协议，直接检验它们在正式 test 上的表现。已有 `parax_p10_stability_test_20260924_111500` 四组 P-10 小残差实验不重复启动。

## 统一协议

- EMOTIC Track-A，seed0，train split 完成全部 8 个增量 task。
- 每个 task 训练 30 epochs，batch size 64，Adam 每 task 重置，main learning rate `0.0125`，ParaX learning rate `0.0004`，cosine scheduler，AMP/TF32。
- 冻结 CLIP ViT-B/16；共享 Selector、Prompt、classifier；现有 image-token Adapter 关闭。
- 固定可靠 Face 三视图融合、joint BCE、view auxiliary loss `0.1`；rank `32`、3 experts、router hidden `16`。
- 不保存 checkpoint；设置 `--skip-validation-eval`，只在 held-out test 逐 task 导出 score 和指标。
- test 不用于调阈值、融合权重、超参数或选择结构；所有候选在启动前一次性锁定，结果作为 exploratory horizontal comparison。

## 重跑候选

| suite | test 运行臂 | 与历史 validation 的对应关系 |
|---|---|---|
| level | B0、P-post、P-10、P-8:10、P-8:10-level、Static-control | 原始六组共享 level routing 结构对照 |
| paired | B0-paired、P10-identity、P10-small、P10-zeroB | P-10 初始化与零输出稳定性对照 |
| center | B0、Shared-live、Frozen-center、Task-local-delta | shared expert center 跨 task 稳定性对照 |
| frozen | B0、Frozen-center-small、Frozen-center-penalty | Frozen-center 残差尺度和 penalty 对照 |
| post | B0、P-post-zero、P-post-small、P-post-tiny、P-post-static | 最终 lane feature 后置 ParaX 对照 |

共 22 个运行臂。每个 suite 内的 B0 用于同协议对照；不同 suite 的 B0 不用于跨 suite 的严格配对检验。

## 产物

服务器结果根目录为 `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_all_test_v0.1/<batch>/`，代码工作树的日志和控制文件分别位于 `logs/emotic_track_a_parax_test/<batch>/` 与 `output/emotic_track_a_parax_test/<batch>/`。完成后统一同步并分析 final/average mAP、逐 task mAP、Full/Person/Face/reliable-Face、forgetting、gate、residual/token ratio 及任何 OOM/AMP 异常。
