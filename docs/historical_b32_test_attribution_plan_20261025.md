# 历史b32摘要模块Test关闭归因

用户2026-10-25授权执行。目的：区分新增Attention/位置模块在Test推理时的直接作用，与共同训练后的其余参数变化。不训练、不改变原checkpoint或八任务成绩。

## 配置

- 来源：旧独立工作树`multi-lane-main-historical-b32-modules-seed1-20261025`中`output/historical_b32_modules/formal_seed1_20261025_01`；模型源码`ab1557142c95d0d8574ea172a968c55d78466a25`。
- 模型：ATTENTION、ATTENTION_BOUND、POSITION、POSITION_CALIBRATED，全部seed1且完成8任务。原B0最终Test32.6800，沿用其旧评估标准。
- 边界：task0/2/7；对应Test人物3142/4691/5368，仍为已见类别至少一项正标签的人物。
- 每组先开启重放，再关闭自己的新增模块（Attention关闭B，位置关闭C）。Image-token Adapter、Selectors、Prompt、head及固定融合保持原checkpoint状态。共8作业、24阶段。
- seed1、B64、workers2、OMP/MKL72、AMP/TF32；无optimizer、learning rate、epoch或训练更新。继承b32[1]/scale.03/ASL历史已训练权重。
- 数据`../multi-lane-main/datasets/EMOTIC`；CLIP`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 仅GPU0，最多4并行；GPU空闲≥6000MiB且SHM空闲≥1024MiB才派发，12秒间隔。其他GPU不使用。
- 入口`python -m multi_lane.track_a.run_historical_b32_test_attribution`。
- 输出`output/historical_b32_test_attribution/seed1_tasks027_20261025_01`；日志对应`logs/`。

## 最小代码改动和验收

评估入口原先只允许validation关模块。新增显式`--historical-b32-test-attribution`，仅允许上述四个完整历史b32 profile及固定边界；不放开其他Test干预，不改变训练/模型数据流。

先运行新增请求校验单测和完整回归。开启重放是数值验收：与原Test每个指定边界的IDs/标签/类别/logits/probabilities及全部视图逐值相同，才派发该组关闭作业。关闭后Attention的Full/Face、位置的Person/Face应逐值不变。每个子评估对checkpoint前后SHA核验，调度器还在整个作业前后重算原40份compact SHA。

产出各边界基线/开启/关闭mAP、F1、分视图与逐类别AP变化。开启−关闭描述当前checkpoint的推理边际；关闭−独立基线描述剩余性能差，不把其视为新训练无模块模型。仅三个边界，不能把其平均与完整八阶段平均比较。Test已被用户授权用于探索，保留相关标记，不冒充未参与选择的独立Test验证。

源码Git-only部署到新独立评估worktree；原模型树、primary和test-only保留。当前尚未远程验收和启动。
