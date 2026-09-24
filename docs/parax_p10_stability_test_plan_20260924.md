# ParaX P-10 稳定性 test-only 实验

## 目的

首轮 ParaX validation 中，官方初始化和 0.1 残差导致 final mAP 大幅下降、residual/token ratio 达到 0.46–1.02，并造成严重旧任务遗忘。本轮只保留 P-10 单插入点，验证小残差、参数冻结和旧任务 logit 蒸馏能否恢复稳定性。

## 固定配置

- EMOTIC Track-A，8 tasks，每 task 30 epochs，batch size 64。
- Frozen OpenAI CLIP ViT-B/16；shared Selector/Prompt/classifier。
- Existing image-token Adapter disabled。
- Fixed reliable-Face three-view fusion，joint BCE，view auxiliary loss 0.1。
- ParaX token-only，block 10 后、block 11 前，rank 32，3 experts，router hidden 16。
- small initialization，output scale fixed 0.001，residual/token ratio cap 0.10。
- main learning rate 0.0125，ParaX learning rate 0.0004，cosine scheduler，AMP/TF32。
- seed0，checkpoint disabled。

## Test arms

| arm | trainable ParaX components | distillation |
|---|---|---:|
| P10-small | experts + routers | 0 |
| P10-router | router only | 0 |
| P10-experts | expert center only | 0 |
| P10-small-distill | experts + routers | 0.2 |

蒸馏只约束当前 task 开始前快照模型的已见类别 logits；不访问 test 标签构造目标。训练完成后只在 held-out test 上导出分数，不运行 validation evaluation，也不在 test 上搜索权重或选择超参数。由于用户明确要求跳过 validation，这四组 test 结果属于 exploratory comparison，不用于声称 validation-locked 优势。

## 产物

- 服务器结果：`/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_p10_stability_test_v0.1/<batch>/`
- 服务器日志：项目 worktree 下 `logs/emotic_track_a_parax_p10_test/<batch>/`
- 本地同步后：`output/emotic_track_a_parax_p10_test/<batch>/` 与对应 `logs/`
- 每组保存 test score dump、task metrics、training history 和 config；不保存完整 checkpoint。
