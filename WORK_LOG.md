# 工作日志

最后一次更新：2026-08-11。

## 2026-08-11：启动 seed0 完整8-task warm-start validation

- 在服务器主工作树clean `58886f1` 上启动唯一候选：seed0、EMOTIC train→val、完整8 tasks、
  30 epochs/task、batch64、base LR0.0125、Adapter LR4e-4、b64、layer11、ReLU、scale0.1、
  `copy_previous`、隔离初始化RNG、AMP/TF32；不构造或读取test split。
- run ID为
  `task_lane_adapter_full_val_seed0_b64_lr0.0004_copy_previous_20260811_174404`，tmux为
  `mla_warm_val_s0_174404`，使用GPU1。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_warmstart_val_v0.2/`，
  日志位于 `./logs/emotic_track_a_adapter_warmstart_val/`。
- 启动核验已进入task0并完成至少3个epoch；每epoch 84 updates、skipped0，暂未发现OOM、NaN、
  inf或traceback。等待8 tasks完成后只分析validation曲线，重点比较task6的Sadness、
  Sensitivity、Suffering；不根据正式test调参。
- 为形成同代码、同seed、同随机轨迹的配对对照，在GPU2启动 Adapter disabled 的完整8-task
  val-only运行 `multi_lane_disabled_full_val_seed0_paired_clean_20260811_174847`，tmux为
  `mla_disabled_val_s0_174847`。首次对照因Automatic Upload使config记录dirty而在task0早期
  主动终止并保留原目录；clean重启版本config记录 `dirty=false`，已完成至少3个epoch且无错误。
- warm-start与clean disabled均完成240 epochs/8 tasks并输出completion marker。逐task val mAP
  差值为 `+2.041904/+2.367560/+1.808337/+1.307971/+0.890832/+0.539336/-0.334822/-0.448594`，
  明确在task6/7转负；final cF1虽提高 `+0.482067`，但final oF1下降 `-0.770218`。
- warm-start相对disabled的average mAP提高 `+1.021565`、forgetting改善 `0.079821`，但final
  mAP下降 `0.448594`。task6新三类引入时平均AP下降 `6.165031`：Sadness `-8.788894`、
  Sensitivity `+0.997110`、Suffering `-10.703310`；final三类平均差进一步为 `-6.248597`。
- 新增 `multi_lane/track_a/compare_validation.py`，严格检查同commit/seed/protocol/val-only与
  clean状态，自动生成8-task曲线、task6逐类AP及停止判定；结果写入本地
  `./output/emotic_track_a_adapter_warmstart_val/paired_validation_comparison.json`。按预先规则
  三项继续条件全部失败，结论为停止task-lane Adapter容量扩张，不再增加bottleneck或层数。

## 2026-08-11：实现 Adapter RNG 隔离与跨任务 warm-start

- 当前结构明确为“每任务独立 Adapter”，不是所有任务共享一个：8个 Adapter 预分配，训练
  task t 时只解冻第t个，推理按lane id路由。上一版本每个task独立随机初始化且不复制。
- 按用户确认的计划增加 `adapter_task_initialization=independent|copy_previous`；默认
  `independent` 保留上一轮复现。`copy_previous` 在task>0激活时把上一task Adapter完整复制
  到当前task，然后只训练当前副本，旧task仍冻结。
- Adapter Bank 初始化被包在 `torch.random.fork_rng(devices=[])` 中；Adapter仍获得确定性的
  seed相关初始化，但构造结束后全局torch RNG恢复到与disabled模型相同的状态，避免改变
  DataLoader shuffle和随机增强序列。
- runner/config/smoke新增 `--adapter-task-init`；task0脚本默认保持independent。新增
  `scripts/emotic/run_multilane_track_a_adapter_full_val.sh`，用于seed0完整8-task、val-only、
  b64/lr4e-4 warm-start验证，不构造test dataset。
- 单元测试新增disabled/adapter构造后RNG state完全一致，以及warm-start参数逐tensor复制、
  旧task冻结和当前task解冻检查。本地py_compile、bash -n和git diff --check均已通过；
  实现以 `313618c` 提交并推送，服务器主工作树fast-forward到同一提交，test-only未触碰。
- 服务器完整单元测试10/10通过；GPU1真实CLIP `copy_previous` Adapter smoke通过，可训练参数
  `788314`，零初始化Adapter与disabled logits最大差 `9.1552734375e-05`，在既定AMP容差
  `1e-4` 内。smoke仅执行batch2 forward/backward，未加载EMOTIC、未生成checkpoint、未启动
  validation或正式训练。

## 2026-08-11：启动 Task-lane Adapter task0 validation screen

- 用户确认 Automatic Upload 已关闭，并确认可以启动 task0 validation screen。启动前服务器
  主工作树干净，local/origin/server 均为
  `exp/emotic-multilane-transformer-adapter@6e8b15d`；test-only worktree 未触碰。
- 完整配置：EMOTIC full image、seed0、task0/5 classes、train→val、30 epochs、batch64、
  eval batch64、Adam base LR0.0125、weight decay0、temperature1、每任务 cosine、AMP/TF32、
  threshold0.5、layer11、ReLU、scale0.1；不构造 test dataset。
- batch ID：`task_lane_adapter_task0_seed0_20260811_153033`。b32/lr1e-4、b32/lr4e-4、
  b64/lr1e-4、b64/lr4e-4 分别运行在 GPU1/2/3/4，对应 tmux：
  `mla_t0_b32_lr1e4_153033`、`mla_t0_b32_lr4e4_153033`、
  `mla_t0_b64_lr1e4_153033`、`mla_t0_b64_lr4e4_153033`。
- 初始检查四组均约进入 epoch3/30，每 epoch 84 optimizer updates、skipped 0，无 OOM、NaN
  或 traceback。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_screen_v0.1/`，日志位于
  `./logs/emotic_track_a_adapter/`。本地上下文改动待实验完成、补齐结果后再确认提交。
- 四组均以 exit code0 完成 30 epochs、2520 optimizer updates、skipped 0。b32/lr1e-4、
  b32/lr4e-4、b64/lr1e-4、b64/lr4e-4 的 seed0 task0 val mAP 分别为
  `56.792729/58.974777/57.720757/59.697275`；相对严格复现 seed0 `57.339154` 的增益分别为
  `-0.546425/+1.635623/+0.381603/+2.358121`，只晋级 b64/lr4e-4。
- b64/lr4e-4 的 task0 seed1/2 确认于 15:53 启动，run ID 为
  `task_lane_adapter_task0_confirm_b64_lr4e4_20260811_155343_seed1/seed2`，tmux 为
  `mla_t0_confirm_s1_155343` / `mla_t0_confirm_s2_155343`，分别使用 GPU1/2。复用已有 seed0，
  晋级规则为三种子平均增益≥+0.5、至少2/3 seeds提升、任一seed降幅不超过1.0；仍不读取test。
- 老师随后明确不需要额外多 seed 确认。停止信号到达前 seed1/2 已正常完成，val mAP 为
  `59.329355/56.178599`，相对各自严格复现基线 `56.932921/54.973777` 仍分别提升
  `+2.396434/+1.204822`；产物保留，但不再作为必须前置步骤。
- 按用户要求新增 b128/lr4e-4 seed0 task0 val screen：run ID
  `task_lane_adapter_task0_seed0_b128_lr4e4_20260811_155954`，tmux
  `mla_t0_b128_lr4e4_155954`，GPU3。每任务 Adapter 参数 `197504`、当前总可训练参数
  `886682`。最终 val mAP `56.824802`，比 b64/lr4e-4 的 `59.697275` 低 `2.872474`，因此
  淘汰 b128，正式配置确定为 b64/lr4e-4。
- 正式 run ID：`multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927`。seed0/1/2
  分别在 GPU1/2/3、tmux `mla_adapter_formal_s0_160927`、
  `mla_adapter_formal_s1_160927`、`mla_adapter_formal_s2_160927` 启动。完整配置为 8 tasks、
  30 epochs/task、batch64、Adam base LR0.0125、Adapter LR4e-4、b64/layer11/ReLU/scale0.1、
  cosine per task、AMP/TF32、test reporting、threshold0.5；运行提交 clean `6e8b15d`。
- 正式输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_formal_v0.1/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`，日志位于
  `./logs/emotic_track_a_adapter_formal/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`。
  启动检查三组均完成模型构建并进入 task0，GPU 显存约1.9GB，Git仍为clean。
- 正式三组均以 completion marker 和 tmux exit code0 完成；每 seed 240 epochs、8 tasks、
  skipped0，config 均为 test reporting、b64/lr4e-4、Git clean `6e8b15d`。
- 聚合五项 mean±sample std：final mAP `31.128461±0.082307`、final cF1
  `31.994054±0.322333`、final oF1 `48.878881±0.080051`、average mAP
  `38.599579±0.360451`、forgetting `4.869595±0.112483`。相对无 Adapter 基线均值变化为
  `-0.171025/+0.182944/-0.230322/+0.601003/+0.081138`。
- 使用显式白名单打包 formal summary、每 seed 四个 JSON、三份日志和 SHA manifest，共17个
  普通文件，确认无 checkpoint。服务器压缩包大小61813 bytes，SHA-256
  `16e0a9fcf9dc7d66dbf8ac5ba0a681539a400fdcf0f53916bd122071818946e1`；已通过 SSH key 下载到
  本地 `./output/emotic_track_a_adapter/<run_id>/`，校验一致并解压到 `synced_files/`。
- 逐 task 对齐发现 Adapter 的 seen-class test mAP 在 task0–5 分别提升
  `+1.665288/+1.280988/+0.912148/+0.433352/+0.510573/+0.410443`，但 task6/7 变为
  `-0.233741/-0.171025`。task6 新类平均 AP 下降 `4.514579`，Sadness/Suffering 分别下降
  `6.289815/7.842116`；到 final 仍保持类似差值，说明是 task6 获取不足而非后续遗忘。
- task6 train view 仅627样本、10 updates/epoch、总计300 updates；其 Adapter up-weight norm
  `2.494252` 为8个 task最低。全部 task 的最终 BCE 均低于基线，但 task3/5/6/7 val mAP
  更差，支持低数据任务欠适应/目标错配诊断。参数全部 finite，无爆炸或未训练迹象。
- 发现严格控制缺陷：Adapter Bank 的 down projection 随机初始化会在 DataLoader 前消耗
  全局 torch RNG，使相同 seed 的数据 shuffle/augmentation 不再与 disabled 基线相同，
  不同 bottleneck 也不完全配对。下一轮必须先隔离初始化 RNG，并优先测试上一 task Adapter
  warm-start；不再根据已观察的正式 test 反复调参。

## 2026-08-11：开始 Task-lane Transformer Adapter 阶段 0

- 用户确认按分析方案开始修改；从严格复现分支 `exp/emotic-multilane-track-a-repro` 的
  `ce7d9a0` 创建本地分支 `exp/emotic-multilane-transformer-adapter`。服务器尚未切换，
  当前改动尚未 commit/push。
- 新增 `multi_lane/track_a/adapter.py`：实现 task-specific、layer-specific bottleneck
  Adapter bank；down 使用 Xavier 初始化，up 权重/偏置全零，按 lane id 路由，激活新任务
  时只解冻当前 task Adapter，不复制或覆盖旧任务参数。
- `multi_lane/track_a/model.py` 在 lane block 冻结 MLP 旁路接入 Adapter；默认
  `adapter_mode=disabled`，因此原模型不会创建 Adapter 或额外消耗初始化随机数。新增运行时
  开关及基础/Adapter 参数分组接口。
- `multi_lane/track_a/runner.py` 增加独立 Adapter LR、bottleneck/layer/scale/activation、
  `max_tasks` 和 `reporting_split`；默认仍是 disabled、8 tasks、test。val screen 模式不创建
  test dataset；单 task summary 的 forgetting 明确定义为 0。
- 新增 `scripts/emotic/run_multilane_track_a_adapter_task0.sh`，固定 task0、val-only、
  layer11、ReLU、scale0.1，并允许通过环境变量筛选 bottleneck 与 Adapter LR；脚本尚未运行。
- 单元测试扩展为覆盖零初始化基线等价、运行时旁路、lane 路由、任务冻结、无 Adapter 复制、
  参数计数和单任务 forgetting。
- GPU smoke 新增显式 `--adapter-mode task_lane` 路径，检查真实 CLIP 上零初始化 logits
  等价、Adapter 梯度、冻结视觉塔和当前任务参数量；默认 disabled 路径仍执行原基线检查。
- 本机系统 Python 3.9 已通过 `py_compile`，新 shell 入口通过 `bash -n`，`git diff --check`
  通过；本机缺少 PyTorch，张量级单元测试与 GPU smoke 等待提交推送后在服务器执行。
- 阶段 0 主实现以 `531f3f3` 提交并推送；服务器主工作树检查为空后通过 `fetch` 和新建跟踪
  分支同步到同一提交。test-only worktree 的历史修改和未跟踪文件保持原样。
- 服务器 CPU 单元测试 8/8 通过；原基线 GPU1 smoke 通过，可训练参数仍严格为 `689178`。
- 首次 Adapter GPU smoke 的逐 bit logits 检查失败。诊断确认 Adapter delta、up weight 和
  up bias 的非零元素均为 0；重复 forward 的 AMP 最大差为 `6.1035e-5`，FP32 最大差为
  `1.4901e-8`，属于 GPU 重复计算数值差，而非 Adapter 非零输出。GPU smoke 因此改为直接
  检查 Adapter residual 严格为零，并对两次 AMP logits 使用 `atol/rtol=1e-4`；CPU 单测的
  逐 bit 等价检查保持不变。
- AMP smoke 修正以 `fc32af7` 提交推送并同步。最终提交上服务器 8/8 单元测试再次通过；
  GPU1 Adapter smoke 通过，可训练参数 `788314`，其中新增当前任务 Adapter `99136`，
  零初始化 AMP logits 最大差 `3.0518e-5`。全程未读取 EMOTIC、未生成 checkpoint、未启动
  task0 或正式训练。

## 2026-08-11：开始在当前仓库移植 MULTI-LANE Track-A 严格复现

- 用户确认在当前 `multi-lane-main` 中复现注册的 EMOTIC B5-C3 三种子结果，随后从该
  结果继续修改，而不是切换到 DDP 仓库开发。
- 从 `fix/independent-git-worktrees@1b1dcd2` 创建本地分支
  `exp/emotic-multilane-track-a-repro`；服务器分支尚未切换。
- 对照确认旧 `clip_vit_b16_patch` 与目标协议存在权重来源、768/512 维分类头、
  val+test/test、0.8/0.5 阈值、batch 256/64、AMP 和三种子口径等差异。
- 新增 `multi_lane/track_a/`：固定 OpenAI CLIP checkpoint 加载、目标 MULTI-LANE
  模型、严格训练/评估 runner、GPU smoke 和三种子汇总器。
- 新增 `scripts/emotic/` worker、直接 launcher 和 tmux launcher；旧入口保持不变。
- 新增单元测试，覆盖 task-slice 复制、concat inference、冻结视觉塔、固定阈值指标和
  逐类 forgetting。本地系统 Python 缺少 PyTorch/NumPy，已完成 AST/compile 与 shell
  语法检查，完整测试待服务器执行。
- 首次服务器单元测试中 3 个模型/forgetting 测试通过；AP 测试因旧公式的 `1e-8`
  epsilon 返回 `99.99999925` 而非数学上的 100，已将断言调整为 5 位小数容差。
- 容差修正后服务器 4/4 单元测试通过。
- 当前仓库 checkpoint loader 与目标 benchmark loader 的 152 个 CLIP visual state
  tensor 完全相同，最大权重误差和同输入输出误差均为 `0.0`，输出为 512 维。
- GPU0 forward/backward smoke 通过，可训练参数为 `689178`，冻结视觉塔没有梯度。
- 本地、远端和服务器同步后，以 `8cff911` 启动正式三种子训练：run ID
  `multi_lane_main_track_a_seed012_20260811_132448`，tmux
  `emotic_multilane_track_a_seed012`，seed 0/1/2 对应 GPU 0/1/2。
- task 0 三个 seed 的 mAP/cF1/oF1 与注册实验逐项完全相同；停止持续轮询，让 tmux 后台
  独立完成。最后一次快照位于 task 1 epoch 14–17/30，错误、NaN、OOM、skip 计数均为 0。
- 三种子正式实验已完成并输出 completion marker。最终 final mAP/cF1/oF1/average
  mAP/forgetting 为 `31.2994855/31.8111103/49.1092038/37.9985757/4.7884572`，所有
  mean/std 与原注册实验完全一致。
- 在服务器生成 `multi_lane_main_track_a_seed012_20260811_132448_download.tar.gz`，包含
  13 个 JSON 和 4 个日志、共 17 个文件，明确无 `checkpoints/`；大小约 80K，SHA-256
  `d3a2f23080da5ff6bd5d0fde99c6ebb9eae1916db31b1e09e9ebb6ed5fb15dcc`。
- 压缩包已通过 SSH 下载到本地项目 `./output/emotic_track_a/<run_id>/`，本地校验一致并
  解压到 `synced_files/`，可直接进行后续分析。

## 当前 Git 状态

权威 Git 仓库位置：

```text
/mnt/haoyuan/workspace/multi-lane-main
```

当前工作分支：

```text
feature/clip-taskCLS-posneg-text-head
```

当前服务器本地提交：

```text
Add CLIP DDP prompt head
```

状态说明：

- 该提交只存在于服务器本地分支，尚未 push 到远端。
- `train_c100.sh` 仍有一个未提交的末尾换行变化，未纳入 DDP 提交。

干净起点：

```text
52cf6b1 Add CLIP task CLS text head
```

该提交同时对应：

```text
origin/feature/clip-taskCLS-text-head
origin/feature/clip-taskCLS-text-siglip-scale
feature/clip-taskCLS-text-head
feature/clip-taskCLS-text-siglip-scale
```

当前 DDP 半成品历史：

```text
stash@{0}: On clip-taskCLS-posneg-text-head: wip clip_ddp draft before clean restart
stash@{1}: On clip-taskCLS-text-siglip-scale: wip siglip scale calibration experiment
stash@{2}: On clip-task-text-head: wip clip_patch_text experiment
```

## 本轮已完成修改

从干净分支重新实现 DDP v1 / PCD 结构，未直接应用 `stash@{0}`。

2026-06-22 已按用户要求在服务器上创建本地提交 `Add CLIP DDP prompt head`，未同步到远端。

2026-06-22 开始下一阶段修改：

- 新增 DDP eval diagnostics：评估时打印并保存 predicted positives、micro precision/recall/F1、prob/logit 分布和阈值 sweep。
- 新增 eval-only checkpoint 复评估入口：`--eval --eval_checkpoint <path>`，也兼容 `--eval_dir` 或 `output_dir/checkpoints.pth`。
- 新增 CLI 参数：`--ddp_diagnostics`、`--ddp_diagnostic_thresholds`、`--eval_checkpoint`。
- 当前这批下一阶段修改尚未提交。

业务代码改动：

- `multi_lane/clip_vit.py`
  - 新增 `clip_ddp` 独立 forward 分支。
  - 新增 class-specific positive/negative text prompts。
  - 新增 class-specific positive/negative visual prompts。
  - 新增 late-layer DDP prompt attention 注入。
  - 新增 token-wise similarity aggregation，支持 `mean`、`max`、`cls`。
  - 新增 PCD temperature 计算。
  - DDP final logit 使用 `CLIP logit_scale * (s_pos - s_neg) / tau`，避免纯 cosine margin 在固定阈值 F1 下不可达。
  - DDP 模式跳过 MULTI-LANE selectors/task CLS 分类路径。
- `multi_lane/engine.py`
  - DDP 训练 loss 只在当前 task classes 上计算。
  - DDP 评估 loss 与 mAP 使用已见类别 clean logits/targets。
  - DDP loss 不再额外使用项目通用 `args.temperature`，温度只由 DDP 内部训练 `tau=1` / 评估 PCD 控制。
- `multi_lane/utils.py`
  - DDP 模式只训练 `ddp_text_prompts` 和 `ddp_visual_prompts`。
  - DDP optimizer 参数组关闭 weight decay，降低旧类 prompt 漂移风险。
- `multi_lane/configs/*.py`
  - 增加 `head_mode=clip_ddp`。
  - 增加 DDP/PCD CLI 参数。

上下文文档：

- 重建 `AGENTS.md`。
- 重建 `PROJECT_CONTEXT.md`、`WORK_LOG.md`、`TODO_NEXT.md`。

## 已运行验证

本地已运行：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器 `multilane` 环境已运行：

```text
git diff --check
python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器配置解析检查：

```text
REMOTE_PARSER clip_ddp True 3.0 0.7
```

服务器参数冻结检查：

```text
TRAINABLE ['ddp_text_prompts', 'ddp_visual_prompts']
PROMPTS (8, 2, 16, 512) (8, 2, 16, 768)
MASK_T0 [0, 1, 2]
OPT_GROUPS 1 0.0 327680
MASK_T1 1 [3, 4]
```

服务器 dummy forward/backward smoke：

```text
TRAIN_LOGITS (1, 4)
TRAIN_DEBUG ... 'class_ids': (2,), 'tau': 1.0, 'final_logits': (1, 4)
TRAIN_ACTIVE_GRAD [0, 1]
EVAL_LOGITS (1, 4)
EVAL_DEBUG ... 'class_ids': (4,), 'tau': 3.0, 'final_logits': (1, 4)
MASK_T1 [2, 3]
```

## 风险点

- `clip_ddp` 已完成 dummy forward/backward smoke，但新增 logit scale 修正后需要重新做一次服务器静态检查和 VOC smoke。
- 旧类 prompt 冻结逻辑已通过 dummy 梯度检查确认，正式训练前仍建议在首个真实 batch 再打印一次 learnable/gradient 状态。
- DDP binary softmax 当前用 `CLIP logit_scale * (s_pos - s_neg) / tau` 形式交给 `BCEWithLogitsLoss`；该尺度修正用于让 PCD 后的 score 与项目固定阈值 F1 兼容。
- EMOTIC 的 `tau_max/gamma` 没有论文指定值，当前只是采用 COCO 风格默认。

## VOC DDP smoke 分析

2026-06-22 的 `voc_clip_ddp_smoke` 结果：

```text
task0 mAP 96.1067, oF1 0.0, cF1 0.0
task1 mAP 71.9124, oF1 0.0, cF1 0.0
task2 mAP 64.1040, oF1 0.0, cF1 0.0
task3 mAP 51.0589, oF1 0.0, cF1 0.0
task4 mAP 50.4372, amAP 66.7238, oF1 0.0, cF1 0.0
```

明细报告显示每个 task 的 `predicted_positive` 总数均为 0。原因是 F1 使用项目固定规则 `sigmoid(logits) > 0.8`，而原实现 DDP logit 为纯 cosine margin `(s_pos - s_neg) / tau`；随着 PCD `tau` 增大，阈值所需 margin 不可达。已修正为带 CLIP `logit_scale` 的 DDP logit，需要重新运行 VOC smoke 验证。

2026-06-22 重新运行 `voc_clip_ddp_smoke_scale`，使用 `CLIP logit_scale * (s_pos - s_neg) / tau`：

```text
task0 mAP 98.7263, amAP 98.7263, oF1 93.9921, cF1 93.7758, loss 0.0862
task1 mAP 90.7199, amAP 94.7231, oF1 80.1021, cF1 78.4673, loss 0.2587
task2 mAP 80.4492, amAP 89.9651, oF1 69.1805, cF1 69.8794, loss 0.3413
task3 mAP 74.8414, amAP 86.1842, oF1 67.4487, cF1 65.8933, loss 0.3784
task4 mAP 71.5174, amAP 83.2508, oF1 64.0643, cF1 62.1179, loss 0.3756
```

该结果确认 scale 修正有效：最终 task 的 predicted positives 总数为 7175，20 个 seen classes 全部有正预测。当前只是 1 epoch VOC B4-C4 smoke，不能直接对齐论文 VOC B4-C2/B5-C3/B0-C4/B10-C2 协议；但相对旧 smoke，final mAP 从 50.44 提升到 71.52，OF1/CF1 从 0 恢复到 64.06/62.12。

## VOC 论文协议实验

2026-06-22 已为严格对齐 VOC B0-C4 修正 `multi_lane/datasets.py` 的 split 逻辑：

- `base_classes=0,num_tasks=5` 现在切分为 `[0-3],[4-7],[8-11],[12-15],[16-19]`。
- `base_classes=10,num_tasks=6` 保持 B10-C2：`[0-9],[10-11],[12-13],[14-15],[16-17],[18-19]`。

已启动 tmux：

```text
voc_b0c4_ddp_gpu1
  GPU: 1
  config: VOC B0-C4, epochs=20, batch_size=16, tau_max=7.0, gamma=0.2
  log: ./logs/voc_ddp_b0c4_20ep_tau7_g02.log
  output: ./output/voc_ddp_b0c4_20ep_tau7_g02

voc_b10c2_ddp_gpu0
  GPU: 0
  config: VOC B10-C2, epochs=20, tau_max=7.0, gamma=0.2
  initial batch_size=16/chunk=2 and batch_size=16/chunk=1 both OOM on GPU0 because other processes already occupied memory
  current retry: batch_size=8, ddp_class_chunk_size=1
  log: ./logs/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1.log
  output: ./output/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1
```

用户明确要求：如果显存 OOM 退出，就等待，不要动别人服务器上正在跑的实验。

### 2026-06-22 结果分析

本地已同步 `voc_ddp_b0c4_20ep_tau7_g02` 完整结果：

```text
task0 mAP 99.2469, amAP 99.2469, oF1 95.3933, cF1 95.2475, loss 0.0536
task1 mAP 90.0728, amAP 94.6598, oF1 52.7571, cF1 49.3263, loss 0.3966
task2 mAP 82.8824, amAP 90.7340, oF1 33.8562, cF1 31.9064, loss 0.4603
task3 mAP 79.5050, amAP 87.9267, oF1 23.5792, cF1 23.1313, loss 0.4751
task4 mAP 76.9800, amAP 85.7374, oF1 21.5977, cF1 22.8244, loss 0.4872
```

关键症状：

- final mAP 仍有 76.98，说明排序信号没有完全坏掉。
- final F1 崩塌来自 recall 极低：20 类总 support 为 7632，只预测 943 个正样本，TP 926、FP 17、FN 6706。
- final precision 约 0.982，recall 约 0.121，说明 PCD/阈值过强，模型过度保守。
- 对比 1 epoch `voc_clip_ddp_smoke_scale`：final precision/recall 约 0.661/0.621；本次 VOC B0-C4 使用 `tau_max=7,gamma=0.2` 后 precision/recall 变为约 0.982/0.121。

当前判断：

- `tau_max=7,gamma=0.2` 来自论文附录 VOC B4-C2 的超参分析，不应直接假定适合 VOC B0-C4/B10-C2。
- 当前实现中的 PCD 对 VOC B0-C4 明显抑制过强，是 final CF1/OF1 远低于论文值的第一嫌疑。
- B10-C2 本地同步日志尚未出现最终 `Average performances` 或 detail JSON；当前只能确认其训练阶段 mAP/oF1 较高但 cF1 偏低，不能作为完整最终结果分析。

### 2026-06-22 DDP 诊断实验

本地已同步三组 VOC B0-C4 诊断结果：

```text
voc_ddp_b0c4_diag_tau3_g07_3ep
voc_ddp_b0c4_diag_eval_tau3_g07
voc_ddp_b0c4_diag_eval_pcd_off
```

关键结果：

```text
3ep tau3/gamma0.7 train final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval tau3/gamma0.7 final:     mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval PCD off final:           mAP 76.1609, amAP 85.3464, oF1 55.9951, cF1 59.0601, loss 0.4391
```

诊断结论：

- `train_tau3` 与 `eval_tau3` 每个 task 指标一致，说明 `--eval --eval_checkpoint` 路径可复现训练末尾评估。
- `tau_max=3,gamma=0.7` 不会出现 tau7/gamma0.2 的 recall 崩塌；final predicted positives/support 为 6544/7632，micro precision/recall/F1 为 0.695/0.596/0.642。
- `pcd_off` 的 final predicted positives/support 为 15170/7632，micro precision/recall/F1 为 0.421/0.837/0.560；关闭 PCD recall 更高，但误报太多，F1 反而低。
- mAP 在 tau3 与 PCD off 下基本一致，因为 PCD 是正尺度变换，主要影响固定阈值下的 F1，而不是排序。
- 当前 `tau_max=3,gamma=0.7` 是比 `pcd_off` 和 `tau_max=7,gamma=0.2` 更好的默认候选。

### 2026-06-23 PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
tau2/gamma0.7 final: mAP 76.1613, amAP 85.3464, oF1 62.9751, cF1 63.1456, loss 0.3584
tau5/gamma0.5 final: mAP 76.1613, amAP 85.3464, oF1 46.4039, cF1 45.9509, loss 0.4374
tau7/gamma0.2 final: mAP 76.1613, amAP 85.3464, oF1 20.3382, cF1 23.7697, loss 0.4850
```

与已有结果合并比较：

```text
pcd_off:       OF1 55.9951, CF1 59.0601, pred/support 15170/7632, P/R/F1 0.421/0.837/0.560
tau2/gamma0.7: OF1 62.9751, CF1 63.1456, pred/support 10102/7632, P/R/F1 0.553/0.732/0.630
tau3/gamma0.7: OF1 64.2071, CF1 62.1797, pred/support 6544/7632,  P/R/F1 0.695/0.596/0.642
tau5/gamma0.5: OF1 46.4039, CF1 45.9509, pred/support 2643/7632,  P/R/F1 0.902/0.312/0.464
tau7/gamma0.2: OF1 20.3382, CF1 23.7697, pred/support 884/7632,   P/R/F1 0.980/0.113/0.203
```

结论：

- tau7/gamma0.2 明确过强，复现 recall collapse；该配置不适合作为 VOC B0-C4 默认。
- tau5/gamma0.5 也偏保守，precision 高但 recall 明显不足。
- tau2/gamma0.7 与 tau3/gamma0.7 接近：tau2 提高 recall 和 CF1，tau3 提高 precision、micro F1 和 OF1。
- 若按论文同时看 OF1/CF1，tau3/gamma0.7 当前更均衡；tau2/gamma0.7 可作为 macro-F1 候选保留。
- 下一步不应继续调更强 PCD，应进入 aggregation 对照，优先用 tau3/gamma0.7，同时可保留 tau2/gamma0.7 做备选复核。

### 2026-06-23 aggregation 对照

本地已同步 VOC B0-C4 3 epoch aggregation 对照，固定 `tau_max=3.0,gamma=0.7`：

```text
mean final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
max final:  mAP 71.6560, amAP 81.9845, oF1 59.6344, cF1 59.1139, loss 0.3544
cls final:  mAP 77.7732, amAP 86.6709, oF1 64.0544, cF1 63.8777, loss 0.3535
```

诊断对比：

```text
mean: pred/support 6544/7632, P/R/F1 0.695/0.596/0.642
max:  pred/support 6920/7632, P/R/F1 0.627/0.569/0.596
cls:  pred/support 9569/7632, P/R/F1 0.576/0.722/0.641
```

类别层面：

- `cls` 明显改善了部分 mean 下偏保守的局部/物体类：diningtable F1 0.270 -> 0.529，pottedplant 0.329 -> 0.552，motorbike 0.645 -> 0.853，bus 0.583 -> 0.775，chair 0.372 -> 0.441。
- `cls` 也带来一些误报：sheep 0.521 -> 0.291，bicycle 0.615 -> 0.445，bird 0.671 -> 0.466，train 0.690 -> 0.590。
- `max` 没有达到“局部 token 最大值更好”的预期，final mAP、OF1、CF1 均低于 mean/cls，暂停作为主路线。

当前判断：

- `cls` 是当前 aggregation 的最好候选，mAP 和 CF1 均优于 mean，OF1 与 mean 基本持平。
- `cls` 的分数分布更偏召回，后续建议在 `cls` checkpoint 上复评估 tau2/tau3/tau5，确认是否能通过 PCD 强度把 precision/recall 再平衡。
- 如果 `cls + tau3` 或 `cls + tau5` 复评估稳定优于 mean，再用该配置跑 20 epoch B0-C4。

### 2026-06-23 cls PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_cls_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
cls tau2/gamma0.7 final: mAP 77.7733, amAP 86.6713, oF1 60.3633, cF1 62.0513, loss 0.3796
cls tau5/gamma0.5 final: mAP 77.7733, amAP 86.6708, oF1 55.3263, cF1 55.4358, loss 0.3792
cls tau7/gamma0.2 final: mAP 77.7732, amAP 86.6708, oF1 36.0121, cF1 39.6079, loss 0.4183
```

与已有 `cls tau3/gamma0.7` 合并比较：

```text
cls tau2/gamma0.7: pred/support 13013/7632, P/R/F1 0.479/0.816/0.604
cls tau3/gamma0.7: pred/support  9569/7632, P/R/F1 0.576/0.722/0.641
cls tau5/gamma0.5: pred/support  4825/7632, P/R/F1 0.714/0.452/0.553
cls tau7/gamma0.2: pred/support  2298/7632, P/R/F1 0.778/0.234/0.360
```

结论：

- 固定项目/论文主阈值 `sigmoid(logits)>0.8` 下，`cls + tau3/gamma0.7` 最均衡，OF1/CF1 最高。
- `cls + tau2/gamma0.7` recall 过强，误报变多，OF1/CF1 均低于 tau3。
- `cls + tau5/gamma0.5` 与 `cls + tau7/gamma0.2` 均偏保守，尤其 task3/task4 的 recall 明显下降。
- `cls + tau5` 在诊断阈值 0.7 下 micro-F1 接近 tau3，但主实验不能改阈值，因此不作为当前主配置。
- 已按 `head_mode=clip_ddp, ddp_similarity_aggregation=cls, ddp_tau_max=3.0, ddp_gamma=0.7` 跑完 VOC B0-C4 20 epoch，并保留 checkpoint。

### 2026-06-23 VOC B0-C4 cls 20 epoch

本地已同步并分析：

```text
log:    ./logs/voc_ddp_b0c4_cls_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_cls_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_cls_tau3_g07_20ep/checkpoints.pth
```

实验配置：

```text
dataset Split-VOC, num_tasks=5, base_classes=0
backbone clip_vit_b16_patch
head_mode clip_ddp
epochs=20, batch_size=16, optimizer=adam, lr=0.05 scaled to 0.003125
ddp_prompt_length=16, ddp_prompt_layers=5
ddp_similarity_aggregation=cls
ddp_pcd=true, ddp_tau_max=3.0, ddp_gamma=0.7
ddp_class_chunk_size=2, store_model=true
```

逐 task 结果：

```text
task0 mAP 99.5071, amAP 99.5071, oF1 97.0199, cF1 96.9398, loss 0.0418
task1 mAP 91.8224, amAP 95.6648, oF1 76.2983, cF1 79.7620, loss 0.3046
task2 mAP 84.4233, amAP 91.9176, oF1 66.9186, cF1 71.0141, loss 0.3542
task3 mAP 81.4017, amAP 89.2886, oF1 68.3450, cF1 69.1338, loss 0.3587
task4 mAP 79.7877, amAP 87.3884, oF1 65.7037, cF1 64.7468, loss 0.3690
```

最终诊断：

```text
pred/support 8568/7632
TP/FP/FN 5322/3246/2310
micro precision/recall/F1 0.621/0.697/0.657
threshold sweep F1: th0.5 0.465, th0.7 0.613, th0.8 0.657, th0.9 0.589
```

对比：

```text
vs cls 3ep tau3:   +2.01 mAP, +0.72 amAP, +1.65 OF1, +0.87 CF1
vs mean 3ep tau3:  +3.63 mAP, +2.04 amAP, +1.50 OF1, +2.57 CF1
vs mean 20ep tau7: +2.81 mAP, +1.65 amAP, +44.11 OF1, +41.92 CF1
```

类别层面：

- 最差 F1 类别：chair 0.327，bicycle 0.368，sheep 0.415，diningtable 0.477，sofa 0.499，pottedplant 0.508，bottle 0.520，bird 0.547。
- 20 epoch 相比 3 epoch 改善较大：sheep、dog、tvmonitor、bird、bottle、cat、train、cow。
- 20 epoch 相比 3 epoch 下降较大：chair、sofa、motorbike、bicycle、diningtable、pottedplant、bus。

结论：

- `cls + tau3/gamma0.7` 20 epoch 是当前 VOC B0-C4 最好结果，且已经解决 tau7/gamma0.2 的 recall collapse。
- 20 epoch 让整体 mAP 和 micro F1 上升，precision 从 0.576 提到 0.621，recall 从 0.722 降到 0.697，说明训练更稳但更保守。
- 当前 gap 不再主要是 PCD 强度一眼错配，而是 per-class calibration 和 prompt 表达不足；chair/diningtable/sofa/pottedplant 这类小目标/上下文依赖类偏低，bicycle/sheep/bird 有明显误报。
- 下一步优先使用该 checkpoint 做 eval-only PCD 微调，不先重训：复评估 `pcd_off`、`tau2/gamma0.7`、`tau2.5/gamma0.7`，必要时再看 `tau3.5/gamma0.7`。

### 2026-06-23 DDP v1/PCD 结构核对

已核对 DDP v1 论文源码 `arxiv:2509.23335v1` 和当前 `clip_ddp` 实现。

确认匹配：

- 一类一套 positive/negative text prompts 与 positive/negative visual prompts。
- 训练时只训练当前 task classes prompts，推理时对 seen classes 计算。
- 冻结 CLIP encoders，不使用 MULTI-LANE selectors/task CLS 作为 DDP 分类路径。
- PCD v1 公式与当前实现一致：训练 `tau=1`，推理按 `tau_max/gamma` 随 seen classes 增长。
- visual prompt 插入最后 5 个 CLIP visual layers，并在 MSA 输出后切掉 prompt token；interlayer shared prompts 是论文写法，不需要每层一套 prompt。
- optimizer 每 task 重建，只接收 DDP prompts 且 weight decay 为 0，配合 gradient mask 能避免旧类 prompt 被 Adam 动量或 weight decay 继续更新。

高风险差异/疑点：

- 论文公式是 `softmax(s+ / tau, s- / tau)`，`s` 为 cosine similarity；当前实现输出 `CLIP logit_scale * (s+ - s-) / tau`。该改动解决了项目固定 `sigmoid(logits)>0.8` 下全 0 的问题，但会改变 PCD 温度含义。需要把严格 DDP probability/logit 路径和项目阈值兼容路径显式分开。
- visual prompt 注入时，当前代码对原始 visual tokens 做 `block.norm1(x)`，但 prompts 本身没有经过同一个 LayerNorm 后再进入 Q/K/V。更严格的 pre-norm ViT 实现应考虑先拼接 `[P; x]` 再对整段序列做 `norm1` 和 attention，最后切掉 prompt tokens。
- 论文 v1 正文明确描述了 interlayer prompting 内部的 token 维度：`P^c` 为 `L_P x d`，image hidden states 为 `L x d`，prompt 参与 MSA 后切掉 prompt tokens 并保持 `L x d`。但公式中的 `E_V(P_V^c, x)` 被抽象为可直接与 text feature 做 cosine 的 visual feature，正文没有明确要求 token-wise similarity 再聚合。因此，当前 `mean/max/cls` 以及后续 `patch_mean/patch_max/topk` 应表述为实现级 visual evidence aggregation 诊断，而不是论文明确规定。
- 论文 v1 implementation details 写 prompt length 为 16；当前匹配。论文后续 DeCLIP/AST 稿改成 prompt length 4 和 `lambda=16`，这属于版本差异，不应混到当前 DDP v1/PCD 目标中。
- 论文附录提到 VOC DDP learnable params 约 `0.49M`，当前 VOC 为 `0.819M = text prompts + visual prompts`。这说明论文参数统计或实际实现可能没有把 text prompts 作为同等可训练参数计入，需优先核对；也应做 text-only/visual-only/pos-only/neg-only ablation。
- text side 当前是 `[SOT] + learnable prompt + class name + '.' + [EOT]`，positive/negative 只靠不同 learnable embeddings 区分；论文未给自然语言正负模板，所以这不是明确 bug，但需要通过 prompt 初始化/模板 ablation 验证。

结构排查结论：

- 当前实现已经是 DDP 大框架，但还不能说严格复现论文实现。
- 最值得先改/查的是 logit scale 与 PCD 的关系、visual prompt LayerNorm 注入位置、CLIP 常规 CLS/pooled visual feature 路径，以及 learnable parameter 统计/文本 prompt 是否该完全可训练；token-wise aggregation 只作为补充实验。

### 2026-06-23 DDP 结构开关实现

本地已完成第一批结构修正开关，默认值保持旧实验可复现：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_logit_scale_mode clip|none`。
  - 新增 `--ddp_prompt_norm_mode legacy|prompted`。
  - 扩展 `--ddp_similarity_aggregation`：`pooled_cls`、`patch_mean`、`patch_max`、`cls_plus_patch_max`、`topk_mean`。
  - 新增 `--ddp_similarity_topk`。
  - 新增 `--ddp_train_text_prompts`、`--ddp_train_visual_prompts`、`--ddp_prompt_polarity`。
- `multi_lane/clip_vit.py`
  - `ddp_logit_scale_mode=none` 使用严格 raw cosine margin `(s+ - s-) / tau`；`clip` 保留当前 `CLIP logit_scale * (s+ - s-) / tau`。
  - `ddp_prompt_norm_mode=prompted` 将 `[prompt; image tokens]` 一起送入 `block.norm1` 再做 MSA，之后切掉 prompt tokens；`legacy` 保留旧路径。
  - `pooled_cls` 与 `cls` 都使用 class-conditioned CLS/pooled feature 和 text feature 直接 cosine，作为最贴近论文公式抽象 `cos(E_T(...), E_V(...))` 的路径命名。
  - `patch_mean/patch_max/cls_plus_patch_max/topk_mean` 作为实现级 visual evidence aggregation 诊断项。
  - gradient mask 支持 text/visual prompt 分支和 positive/negative 分支训练开关。
- `multi_lane/utils.py`
  - DDP optimizer 参数选择遵守 `ddp_train_text_prompts` 和 `ddp_train_visual_prompts`，避免被冻结分支进入 optimizer。

论文对照：

- 对应公式 1/2：新增 `ddp_logit_scale_mode=none`，可回到论文写法中的 cosine similarity binary softmax；`clip` 是项目固定阈值兼容分支。
- 对应 interlayer prompting 公式 4：新增 `prompted` pre-norm 路径，让 prompt 和 image tokens 在 MSA 前共同归一化，再切片回 `L x d`。
- 对应 `E_V(P_V^c,x)` 抽象：新增 `pooled_cls` 命名，明确表示单个 class-conditioned visual feature，而不是声称论文要求 token-wise 聚合。

已验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。本地环境缺少 `open_clip`，forward smoke 需要在服务器 `multilane` 环境执行。

### 2026-06-23 VOC B0-C4 结构对照 3 epoch

已同步并分析三组 VOC B0-C4 结构对照实验，配置均为 `num_tasks=5, base_classes=0, epochs=3, prompt_length=16, prompt_layers=5, PCD tau3/gamma0.7, batch_size=8, chunk_size=1, store_model=true`。

结果：

```text
pooled_cls + prompted + raw cosine:
  final mAP/amAP/OF1/CF1 = 81.7107 / 89.1806 / 0.0000 / 0.0000
  pred/support = 0 / 7632 at threshold 0.8
  threshold 0.5 micro P/R/F1 = 0.361 / 0.910 / 0.517

pooled_cls + prompted + CLIP logit_scale:
  final mAP/amAP/OF1/CF1 = 79.8608 / 87.8689 / 61.5493 / 62.9698
  pred/support = 10906 / 7632
  micro P/R/F1 = 0.523 / 0.748 / 0.615
  threshold 0.9 micro P/R/F1 = 0.635 / 0.562 / 0.596

cls + legacy + CLIP logit_scale control:
  final mAP/amAP/OF1/CF1 = 77.1030 / 85.8240 / 61.5543 / 64.1565
  pred/support = 10846 / 7632
  micro P/R/F1 = 0.524 / 0.745 / 0.616
  threshold 0.9 micro P/R/F1 = 0.661 / 0.549 / 0.600
```

结论：

- `raw cosine` 更接近 DDP 论文公式，但在当前项目主指标 `sigmoid(logits)>0.8` 下完全不可用；它不是排序崩了，因为 mAP 达到 `81.71`，而是 logit 尺度太小，固定阈值不兼容。
- `pooled_cls + prompted + clip` 相比旧 control 提高 `+2.76 mAP`、`+2.04 amAP`，OF1 基本持平，CF1 低 `1.19`。这说明更贴近论文的 visual prompt pre-norm 和 pooled visual feature 路径是正向结构改动。
- F1 没明显变好，主要因为两组 `clip` 配置在固定阈值下的 precision/recall 形态几乎一样：predicted positives 都约 `1.09e4`，micro F1 都约 `0.615`。
- 下一步应把 `pooled_cls + prompted + clip + tau3/gamma0.7` 作为新的 20 epoch 候选，而不是继续围绕旧 `cls + legacy` 做长实验。
- `raw cosine` 保留为论文公式对照/排序诊断，不作为当前项目主表配置，除非后续把评估指标从固定 sigmoid 阈值改成 DDP binary probability 或按验证集校准阈值。

### 2026-06-23 DDP gap 分析补充

用户提供了 GPT5.5Thinking 对当前 DDP gap 的独立分析。其判断与当前实验结论基本一致：当前 `clip_ddp` 已接近 DDP 主框架，但论文差距更可能来自 score/probability 语义、阈值协议、CLIP preprocessing、text prompt 初始化、visual pooling 和 VOC protocol，而不是缺少正负 prompt 或 PCD 这类大模块。

更新后的高优先级排查顺序：

- 增加严格 DDP probability 评估路径：保留 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`p_pos=sigmoid(margin/tau)`，并支持 probability mode 下不再二次 sigmoid。
- 增加更密集 threshold sweep 和 oracle threshold 诊断，用来区分结构排序能力与固定阈值 calibration 问题。
- 保存每个 task/class/sample 的 `y_true/s_pos/s_neg/margin/p_pos/scaled_logit/class_id`，定位哪些类是过度预测、哪些类过度保守。
- 增加 CLIP mean/std 输入归一化选项，和当前 no-normalize/ImageNet normalize 对照。
- 打印 VOC task class order/seen class order，确认是否严格符合论文 lexicographic protocol。
- 在上述诊断后，再跑 `pooled_cls + prompted + clip` 20 epoch；如果 strict probability/oracle F1 仍差，再做 text/visual/polarity prompt ablation 与 semantic text initialization。

当前最强证据仍是 raw cosine 3 epoch `mAP=81.71` 但主阈值 F1 为 0：排序能力和固定阈值 calibration 已经明显解耦。

### 2026-06-23 DDP probability / calibration 诊断实现

已实现 GPT5.5Thinking 建议的第一批诊断改动：

- `multi_lane/clip_vit.py`
  - DDP forward 现在除主 logits 外，会缓存完整 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`ddp_prob=sigmoid(margin/tau)`、`scaled_logit=CLIP logit_scale*margin/tau`。
  - 主训练/默认评估仍返回 scaled logits，不破坏旧实验复现。
- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_eval_score_mode logits|probability`。
  - 新增 `--ddp_eval_threshold`，默认 `0.8`。
  - 新增 `--ddp_score_dump true|false`。
  - 默认 DDP diagnostic thresholds 扩展为 `0.1 ... 0.9`。
  - 新增 `--clip_normalize_input true|false`。
- `multi_lane/engine.py`
  - DDP eval metrics 可选择 `logits` 模式：`sigmoid(scaled_logit)>threshold`，或 `probability` 模式：`ddp_prob>threshold`，后者不再二次 sigmoid。
  - DDP diagnostics 增加 dense threshold sweep、global oracle threshold、per-class oracle F1 诊断。
  - detail report 增加 `s_pos/s_neg/margin/ddp_prob/scaled_logit` 分布字段。
  - `--ddp_score_dump true` 时保存 `detail/{run_name}_task{t}_ddp_scores.npz`，包含 `y_true/eval_score/s_pos/s_neg/margin/ddp_prob/scaled_logit/class_ids`。
- `multi_lane/datasets.py`
  - 新增 CLIP mean/std normalization：`mean=(0.48145466,0.4578275,0.40821073)`，`std=(0.26862954,0.26130258,0.27577711)`。
  - `--clip_normalize_input true` 优先于旧 `--normalize_input`；默认仍保持旧行为。
- `main.py`
  - 每次运行打印并保存 `class_order.json`，用于核对 VOC lexicographic task protocol。

验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

下一步实验目的：

- 用同一 checkpoint 做 eval-only 对照，比较 `logits` 与 `probability` 评估语义。
- 开启 `ddp_score_dump`，检查 strict DDP probability/oracle threshold 下 F1 是否接近论文。
- 如果 strict probability + oracle 仍差，再继续查 text prompt 初始化、CLIP normalize 和 visual pooling。

### 2026-06-23 DDP calibration 诊断实验启动

已按用户要求在服务器 `/mnt/haoyuan/workspace/multi-lane-main` 启动三个 tmux 实验：

```text
tmux: voc_ddp_eval_logits_diag
gpu: 0
log: ./logs/voc_ddp_eval_logits_diag.log
output: ./output/voc_ddp_eval_logits_diag
purpose: 同一 pooled_cls+prompted checkpoint，使用当前 scaled-logit 语义复评估，并保存 dense threshold/oracle/score dump。

tmux: voc_ddp_eval_probability_diag
gpu: 1
log: ./logs/voc_ddp_eval_probability_diag.log
output: ./output/voc_ddp_eval_probability_diag
purpose: 同一 checkpoint，使用严格 DDP probability 语义复评估，检查 raw DDP probability 下阈值/校准问题。

tmux: voc_ddp_3ep_clipnorm
gpu: 4
log: ./logs/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm.log
output: ./output/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm
purpose: 训练 3 epoch CLIP mean/std normalized 输入，对照 no-normalize 下的 pooled_cls+prompted+clip。
```

启动后确认：

- 三个 tmux session 均存在。
- `voc_ddp_eval_logits_diag` 已输出 task2 diagnostics。
- `voc_ddp_eval_probability_diag` 已输出 task1 diagnostics。
- `voc_ddp_3ep_clipnorm` 已进入训练并打印配置。

### 2026-06-23 DDP calibration 诊断结果

三组实验已同步并分析。

1. `voc_ddp_eval_logits_diag`

使用同一 `pooled_cls + prompted + clip` 3 epoch checkpoint，评估语义为当前工程 scaled logits：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8689 / 61.5493 / 62.9698
pred/support = 10906 / 7632
micro P/R/F1 = 0.523 / 0.748 / 0.615
global oracle threshold = 0.8, oracle micro-F1 = 0.615
per-class oracle F1 = 0.696
```

结论：dense threshold sweep 证明 final task 的全局最佳阈值仍是 `0.8`，不是简单调全局阈值就能接近论文。per-class oracle 从 `62.97` 提到约 `69.6`，说明存在明显 class-wise calibration 问题，但 oracle 后仍低于论文主表。

2. `voc_ddp_eval_probability_diag`

同一 checkpoint，评估语义为严格 DDP probability：`ddp_prob=sigmoid((s_pos-s_neg)/tau)`，主阈值 `0.5`：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8687 / 46.2331 / 51.5324
pred/support = 22459 / 7632
micro P/R/F1 = 0.310 / 0.911 / 0.462
global oracle threshold = 0.5, oracle micro-F1 = 0.462
per-class oracle F1 = 0.515
```

结论：严格 DDP probability 没有改善 F1，反而明显更差。`ddp_prob` 分布集中在 0.5 附近，低于 0.5 基本全预测，高于 0.6 基本无预测，说明 raw margin 尺度太小；当前论文公式路径排序仍可用，但 probability calibration 不适合项目固定阈值。

3. `voc_ddp_3ep_pooledcls_prompted_clip_clipnorm`

在 `pooled_cls + prompted + clip` 基础上启用 CLIP mean/std 输入归一化训练 3 epoch：

```text
final mAP/amAP/OF1/CF1 = 81.5703 / 88.7328 / 65.0564 / 66.6065
pred/support = 10792 / 7632
micro P/R/F1 = 0.555 / 0.785 / 0.651
global oracle threshold = 0.9, oracle micro-F1 = 0.669
per-class oracle F1 = 0.713
```

对比 no-normalize 的同结构 3 epoch：

```text
no-normalize: mAP/OF1/CF1 = 79.8607 / 61.5493 / 62.9698
CLIP normalize: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
gain: +1.71 mAP, +3.51 OF1, +3.64 CF1
```

对比旧 `cls + legacy + clip` 20 epoch：

```text
old 20ep: mAP/OF1/CF1 = 79.7877 / 65.7037 / 64.7468
clipnorm 3ep: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
```

结论：CLIP preprocessing 是重要 gap。仅 3 epoch 的 `pooled_cls + prompted + clip + CLIP normalize` 已经超过旧 20 epoch 的 mAP/CF1，OF1 基本接近。下一步应优先跑该配置 20 epoch。

当前 updated 判断：

- 严格 probability 评估说明 gap 不是“把评估改回 DDP probability 就解决”，raw margin 尺度本身过小。
- CLIP mean/std normalize 明确有效，应作为后续 VOC 主配置。
- 当前剩余问题更偏 class-wise calibration 和 prompt/类别语义，尤其 sheep/bicycle/bottle 等误报仍多，diningtable/chair 等仍偏难。

### 2026-06-23 VOC B0-C4 pooled_cls+prompted+clipnorm 20 epoch

本地已同步主候选 20 epoch 结果：

```text
log:    ./logs/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep/checkpoints.pth
```

配置：

```text
dataset: VOC B0-C4
head: clip_ddp
aggregation: pooled_cls
prompt_norm_mode: prompted
logit_scale_mode: clip
clip_normalize_input: true
PCD: tau_max=3.0, gamma=0.7
epochs: 20
batch_size: 8
chunk_size: 1
```

最终结果：

```text
task1: mAP/OF1/CF1 = 99.6331 / 97.8536 / 97.8136
task2: mAP/OF1/CF1 = 94.1302 / 76.4492 / 81.0003
task3: mAP/OF1/CF1 = 86.0975 / 67.8669 / 72.8785
task4: mAP/OF1/CF1 = 83.5890 / 69.1588 / 72.1785
task5: mAP/OF1/CF1 = 82.9480 / 67.3165 / 68.9025
```

final diagnostics:

```text
threshold 0.8: pred/support=10527/7632, P/R/F1=0.581/0.801/0.673
threshold 0.9: pred=6730, P/R/F1=0.736/0.649/0.690
global oracle threshold=0.9, oracle micro-F1=0.690
per-class oracle F1=0.748
```

对比：

```text
vs old cls+legacy+clip 20ep:
  +3.16 mAP, +1.61 OF1, +4.16 CF1

vs clipnorm 3ep:
  +1.38 mAP, +2.26 OF1, +2.30 CF1

vs no-normalize pooled_cls+prompted 3ep:
  +3.09 mAP, +5.77 OF1, +5.93 CF1
```

结论：

- 这是当前 VOC B0-C4 最佳结果，证明 `pooled_cls + prompted + CLIP normalize` 是正确主线。
- 20 epoch 没有过拟合崩塌，较 3 epoch 继续提升。
- 但当前 final task 全局最佳阈值已经偏向 `0.9`，说明模型仍偏误报；下一步优先做 eval-only PCD/threshold 复评估，而不是马上改结构。
- 相对 DDP 论文 VOC B0-C4 主表，当前约低 `7.3 mAP / 13.5 OF1 / 8.0 CF1`，gap 明显缩小但还没到论文水平。
- 最差类别仍集中在 `bicycle/sheep/bottle/chair/diningtable/car`。其中 bicycle/sheep/bottle 是高 recall 低 precision 的误报问题； diningtable 是高 precision 低 recall 的漏检问题；chair AP/F1 都偏低。

下一步建议：

- 用该 checkpoint 做 eval-only：`tau2.5/tau3/tau3.5`，并比较 `ddp_eval_threshold=0.8/0.9`。
- 若 threshold 0.9 + 合理 PCD 明显提升 OF1/CF1，再跑 B10-C2 或 20 epoch 正式论文协议。
- 若仍明显低于论文，再进入 text prompt semantic initialization 和 class-wise calibration。

### 2026-06-23 下一阶段 margin-first 计划

用户提供 GPT5.5Thinking 新判断：当前核心问题应从 PCD/threshold 转向 positive/negative prompt margin separation。结合最新 `pooled_cls+prompted+clipnorm` 20 epoch 结果，当前判断如下：

- CLIP normalize 和 prompted pooled feature 已确认有效，主结构路线正确。
- strict DDP probability final OF1/CF1 只有 `46.23/51.53`，且 `ddp_prob` 大量挤在 0.5 附近，说明 raw margin 本身不够大。
- scaled-logit 分支能得到 `82.95/67.32/68.90`，说明排序信息存在，但依赖 CLIP scale/threshold 做工程校准。
- final threshold 0.9 比 0.8 的 micro-F1 更高，说明误报仍偏多，尤其 `bicycle/sheep/bottle`；而 `diningtable/chair` 仍有漏检/表达弱问题。

下一阶段计划：

1. 先做 checkpoint eval-only calibration sweep，但目标是读 margin 而不是继续盲调 PCD：
   - `tau2.5/tau3/tau3.5` × `threshold0.8/0.9`
   - 重点看 `s_pos/s_neg/margin/ddp_prob` 分布、per-class margin gap、oracle threshold。
2. 增加 margin summary report：
   - 每类正样本 margin mean/p10/p50/p90。
   - 每类负样本 margin mean/p10/p50/p90。
   - `pos_margin_mean - neg_margin_mean`。
   - 正样本 raw prob 大于 0.5/0.6 的比例、负样本 raw prob 大于 0.5/0.6 的比例。
3. 做 prompt branch ablation：
   - full text+visual pos+neg。
   - visual-only。
   - text-only。
   - text frozen vs text trainable。
   - visual frozen vs visual trainable。
   - positive-only / negative ablation。
4. 做 semantic text initialization：
   - positive templates: `a photo containing a {class}.`, `a photo of a {class}.`, `a photo with a {class}.`
   - negative templates: `a photo without a {class}.`, `a photo not containing a {class}.`, `a photo with no {class}.`
   - 观察是否提升 positive sample 上的 `s_pos`、negative sample 上的 `s_neg`，并扩大 margin gap。
5. 如果 ablation 证明 DDP loss 本身不足以拉开 margin，再考虑 margin regularizer；该步骤属于超出 DDP v1 的改进，需要单独标记为 extension，而不是论文复现配置。

### 2026-06-23 margin-first 代码改动

本轮按用户要求进入 positive/negative prompt margin separation 排查，已完成以下未提交改动：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_text_init random|same|semantic`。
  - 新增 `--ddp_positive_text_template`、`--ddp_negative_text_template`。
- `multi_lane/clip_vit.py`
  - `random` 保持旧 DDP text prompt 随机初始化，保证已有实验可复现。
  - `same` 让 positive/negative text prompt 从相同随机值开始，用于诊断正负分支是否真的学出分离。
  - `semantic` 用正/负模板前缀分别初始化 learnable text prompt，例如 positive 默认来自 `a photo containing a`，negative 默认来自 `a photo without a`。
  - semantic 初始化只在 `set_class_names()` 后执行一次；eval-only 加载 checkpoint 时，checkpoint 参数会覆盖初始化值，不污染旧模型复评估。
- `multi_lane/engine.py`
  - detail report 新增 per-class positive/negative margin 分布。
  - overall diagnostics 新增 `diag_margin_gap_mean`、正/负样本 margin 大于 0 的比例、正/负样本 raw `ddp_prob` 大于 0.5/0.6 的比例。
  - 评估日志新增 `[DDP margin task ...]` 行，便于直接观察 raw margin 是否拉开。
- `PROJECT_CONTEXT.md`、`TODO_NEXT.md`、`WORK_LOG.md`
  - 已同步记录本轮改动、目的和下一轮实验方向。

本地验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

本地 `python3 main.py voc --help` 因本机 Python 缺少 `numpy` 无法运行；已用直接 parser 检查确认新增参数可见。服务器 `multilane` 环境仍需运行完整 `main.py voc --help` 和 smoke。

### 2026-06-23 VOC B0-C4 margin init 对照结果

用户已同步两组 3 epoch 对照：

```text
random:
  log: ./logs/voc_ddp_margin_random_3ep.log
  output: ./output/voc_ddp_margin_random_3ep
semantic:
  log: ./logs/voc_ddp_margin_semantic_3ep.log
  output: ./output/voc_ddp_margin_semantic_3ep
```

共同配置：

```text
VOC B0-C4, num_tasks=5, base_classes=0, epochs=3
head_mode=clip_ddp
aggregation=pooled_cls
prompt_norm_mode=prompted
logit_scale_mode=clip
clip_normalize_input=true
tau_max=3.0, gamma=0.7
ddp_score_dump=true
```

final 结果：

```text
random:   mAP 82.3370, amAP 89.1434, OF1 66.7329, CF1 67.8947
semantic: mAP 83.8680, amAP 90.1785, OF1 70.2598, CF1 70.6409
delta:   +1.5310 mAP, +1.0351 amAP, +3.5269 OF1, +2.7462 CF1
```

final threshold 0.8 诊断：

```text
random:   pred/support=10491/7632, P=0.5764, R=0.7923, F1=0.6673
semantic: pred/support= 8417/7632, P=0.6698, R=0.7387, F1=0.7026
```

margin 诊断：

```text
random:   pos_margin_mean=0.0940, neg_margin_mean=-0.0747, gap=0.1688
semantic: pos_margin_mean=0.0711, neg_margin_mean=-0.0645, gap=0.1356
```

结论：

- `semantic` 明显提升主指标，主要原因是减少过度预测，precision 从 `0.5764` 提升到 `0.6698`。
- `semantic` 没有直接扩大 raw margin gap；gap 反而从 `0.1688` 降到 `0.1356`。
- 两组 final `pos_ddp_prob>0.6` 和 `neg_ddp_prob>0.6` 都是 0，说明 raw DDP probability 仍然挤在 0.5 附近，raw margin 绝对尺度仍不足。
- 因此 semantic init 可以作为下一轮主配置候选，但它解决的是 scaled-logit 校准/误报问题，不是严格意义上直接把 raw DDP margin 拉大。
- 下一步应在 semantic 配置上继续做 branch-level 排查：冻结 text prompts / 冻结 visual prompts / positive-only / negative-only，以定位 margin gap 主要受哪一支限制。

### 2026-06-23 semantic 20ep 与 branch ablation 结果

用户已同步五组结果：

```text
semantic 20ep:
  log: ./logs/voc_ddp_margin_semantic_20ep.log
  output: ./output/voc_ddp_margin_semantic_20ep

branch ablation 3ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_visual_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_posonly_3ep.log
  ./logs/voc_ddp_ablate_sem_negonly_3ep.log
```

final summary：

```text
random 3ep:        mAP 82.3370, OF1 66.7329, CF1 67.8947, gap 0.1688, pred 10491
full semantic 3ep: mAP 83.8680, OF1 70.2598, CF1 70.6409, gap 0.1356, pred  8417
semantic 20ep:     mAP 84.2737, OF1 70.5397, CF1 72.5721, gap 0.1252, pred  9323
text frozen 3ep:   mAP 82.3217, OF1 71.1562, CF1 72.7656, gap 0.1298, pred  9044
visual frozen 3ep: mAP 82.4743, OF1 68.7695, CF1 69.7780, gap 0.1270, pred  8695
positive-only 3ep: mAP 81.2522, OF1 70.9906, CF1 71.5840, gap 0.1186, pred  8156
negative-only 3ep: mAP 81.3779, OF1 68.4138, CF1 72.4098, gap 0.1355, pred  9768
old best 20ep:     mAP 82.9480, OF1 67.3165, CF1 68.9025
```

关键结论：

- `semantic 20ep` 达到当前最高 mAP：`84.27`，相对 old best `82.95` 提升 `+1.33 mAP`，相对 `semantic 3ep` 只提升 `+0.41 mAP`。
- 固定阈值 F1 的当前最好不是 full semantic 20ep，而是 `text_frozen 3ep`：`OF1 71.16 / CF1 72.77`。
- `text_frozen` 表示固定 semantic text prompts，只训练 visual prompts。它比 `visual_frozen` 明显好，说明当前 F1 主要依赖 visual prompt 学到 class-conditioned evidence。
- full semantic 从 3ep 到 20ep 主要增加 recall：`0.739 -> 0.784`，但 precision 从 `0.670 -> 0.641` 降低，predicted positives 从 `8417 -> 9323` 增加，因此 OF1 只小幅提升。
- raw margin 绝对尺度仍未解决：所有组 `raw_prob_pos_mean` 约 `0.505-0.508`，`raw_prob_neg_mean` 约 `0.494-0.496`，`raw_prob>0.55` 几乎为 0。当前 F1 仍依赖 CLIP logit scale 放大 margin。
- `random` 的 raw margin gap 最大，但 F1 更差，说明“gap 大”本身不够；还需要正负样本分布位置和阈值校准合理。
- `positive-only` 更保守，precision 较高、recall 较低；`negative-only` 更容易过预测，OF1 较低但 CF1 接近。这说明 presence branch 对抑制 false positive/固定阈值 precision 很重要，absence branch 单独训练不能充分压住误报。

下一步建议：

- 优先跑 `semantic text frozen 20ep`：固定语义 text anchor，只训练 visual prompts，验证 3ep 最佳 F1 是否能随训练拉长继续保持/提升。
- 同时可跑 `semantic positive-only 20ep` 作为 precision-oriented 对照。
- 若目标是严格 DDP raw probability，需要考虑 margin regularizer 或 raw binary-softmax loss 加强；这属于 DDP extension，不能混入论文复现主配置。

### 2026-06-24 semantic 长训消融与 seed1 稳定性复验

用户同步四组完整结果：

```text
text frozen 20ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_20ep.log
  ./output/voc_ddp_ablate_sem_text_frozen_20ep

positive-only 20ep:
  ./logs/voc_ddp_ablate_sem_posonly_20ep.log
  ./output/voc_ddp_ablate_sem_posonly_20ep

text frozen 3ep seed1:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep_seed1.log
  ./output/voc_ddp_ablate_sem_text_frozen_3ep_seed1

full semantic 3ep seed1:
  ./logs/voc_ddp_margin_semantic_3ep_seed1.log
  ./output/voc_ddp_margin_semantic_3ep_seed1
```

所有实验均为 VOC B0-C4、`pooled_cls + prompted + clip + CLIP normalize`、
PCD `tau_max=3.0,gamma=0.7`、batch size 16、seed0/seed1，并保存了 detail/score dump；
两个 20 epoch 实验保存了 checkpoint。

最终结果：

```text
full semantic 3ep seed0:  mAP/amAP/OF1/CF1 = 83.8680/90.1785/70.2598/70.6409
full semantic 3ep seed1:  mAP/amAP/OF1/CF1 = 83.6778/90.1794/70.5977/70.8508
text frozen 3ep seed0:    mAP/amAP/OF1/CF1 = 82.3217/89.1572/71.1562/72.7656
text frozen 3ep seed1:    mAP/amAP/OF1/CF1 = 81.5404/88.5455/69.2907/71.3656
text frozen 20ep seed0:   mAP/amAP/OF1/CF1 = 80.8415/87.9108/68.2167/71.2956
positive-only 3ep seed0:  mAP/amAP/OF1/CF1 = 81.2522/88.3632/70.9906/71.5840
positive-only 20ep seed0: mAP/amAP/OF1/CF1 = 83.0647/89.3826/70.0879/72.6094
full semantic 20ep seed0: mAP/amAP/OF1/CF1 = 84.2737/90.3961/70.5397/72.5721
```

结论：

- full semantic 3ep 两个 seed 几乎重合，mAP 波动仅 `0.19`，是当前最稳定的结构。
- text-frozen 3ep 的 seed0 高 F1 未稳定复现；seed1 回落 `0.78 mAP/1.87 OF1/1.40 CF1`。
- text-frozen 拉长到 20 epoch 后全面退化。pred/support 从 3ep seed0 的 `9044/7632`
  增到 `10475/7632`，precision `0.656 -> 0.590`、recall `0.777 -> 0.809`，
  说明 visual prompts 长训后过预测，而不是 margin 完全没有学习。
- text-frozen 20ep 的 global oracle threshold 为 `0.9`，oracle micro-F1 `0.698`，
  仍低于 3ep seed0 主阈值 F1 `0.712`；问题不能只靠换阈值补回。
- positive-only 20ep 相对 3ep 提升 `+1.81 mAP/+1.03 CF1`，但 OF1 下降 `0.90`；
  相对 full semantic 20ep 仍低 `1.21 mAP/0.45 OF1`，没有证明应删除 negative branch。
- raw DDP probability 仍集中在 0.5：四组 final positive mean 约 `0.506`，
  negative mean 约 `0.495`，`>0.6` 比例仍为 0。
- 主要困难类别继续是 `chair/bicycle/bottle/car/diningtable`。text-frozen 20ep 的
  bicycle 出现 `P=0.202,R=0.940`，是典型严重误报；chair 的 margin gap 仅 `0.043`，
  表达分离仍弱。
- text-frozen learnable 参数 `491,520` 与论文约 `0.49M` 很接近，是重要结构线索；
  但其效果和长训趋势表明还需核对论文参数统计口径、训练调度及 prompt 更新方式，
  不能把参数量相等直接当作结构已严格复现。

下一步：

- 主线恢复为 full semantic text+visual prompts；text-frozen/positive-only 作为消融保留。
- 优先做 full semantic 的学习率/训练时长对照，避免当前 20 epoch 出现 precision 下降。
- 基于已有 20ep checkpoints 做 threshold/PCD eval-only，不重训地确认 calibration 上限。
- 在继续增加非论文 loss 前，先核对论文官方实现的 optimizer、LR、scheduler、warmup、
  visual prompt 初始化和 `0.49M` 参数统计口径。

### 2026-06-24 Git 提交与远端推送

- 服务器分支：`feature/clip-taskCLS-posneg-text-head`。
- 功能提交：`19417f1 Add DDP semantic prompt diagnostics`。
- 提交包含 DDP semantic text initialization、prompt branch ablation 开关、
  CLIP normalization、eval-only/checkpoint、score/margin diagnostics 及上下文文档。
- `python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py
  multi_lane/continual_datasets/*.py` 已在服务器 `multilane` 环境通过。
- 已首次推送并设置上游：
  `origin/feature/clip-taskCLS-posneg-text-head`。
- 推送后本地与远端完整哈希一致：
  `19417f1500dd3e1046e79afe18711628f835eb84`。
- 无关的 `train_c100.sh` 文件末尾换行改动未纳入功能提交，仍保留在服务器工作区。

### 2026-06-24 margin 与 mAP 阶段结论

- full semantic 20ep final margin：
  `pos_mean=0.0731`、`neg_mean=-0.0521`、gap `0.1252`。
- `93.4%` 正样本的 margin 大于 0，说明多数正样本方向正确；
  `15.0%` 负样本的 margin 仍大于 0，说明 false-positive 排序仍明显存在。
- task5 PCD `tau=3` 后 strict DDP probability 仍为
  positive `0.5061`、negative `0.4957`，绝对尺度没有充分拉开。
- random 3ep gap `0.1688` 大于 semantic 20ep 的 `0.1252`，但 mAP 只有 `82.34`
  而非 `84.27`。因此 mAP 需要改善每类正负样本排序，不能只优化 overall mean gap。
- full semantic 20ep 最低 AP 类别为：
  `chair 38.51`、`bottle 51.73`、`tvmonitor 68.62`、`pottedplant 73.80`、
  `diningtable 75.33`、`sofa 76.15`、`car 78.05`。
- 代码核对发现配置中的 warmup/min-lr 参数未被实际 scheduler 使用；
  当前每 task 使用裸 `CosineAnnealingLR(T_max=epochs)`，并无条件重建 optimizer。
  这是下一阶段论文协议核对和 mAP 优化的高优先级风险点。

### 2026-06-24 官方 supplementary 对齐改造

从 OpenReview supplementary material 取得官方代码包 `CODE_DDP`，核对后确认：

```text
visual token similarity scale: 20
token attention: softmax(positive token similarities)
same token attention weights applied to positive and negative branches
weighted pair-logit scale: 5
effective final margin scale: 20 * 5 = 100
loss: binary pair softmax BCE, reduction=sum, loss weight=0.03
optimizer: Adam, lr=5.9e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0
scheduler: MultiStepLR, milestones=[0,20], gamma=0.1
optimizer lifetime: one optimizer across all tasks
epochs: 20
batch size: 8
train transform: Resize(224), CutoutPIL(0.2), RandAugment, CLIP normalize
val transform: Resize(224), CLIP normalize
visual prompt layers: [7,8,9,10,11]
text/visual prompt length: 16
```

因此此前计划中的“raw DDP train 是论文主路径”不成立。论文公式省略了实现级 scale，
但 supplementary code 明确使用固定 100 的强尺度。raw margin train 只保留为诊断
ablation，不再标记为严格论文配置。

本轮修改：

- `multi_lane/configs/custom_types.py`
  - 新增 `paper_attention` aggregation。
  - 新增 train/eval logit scale 分离和固定 paper scale。
  - 新增 `paper_sum` loss、持续 optimizer、paper scheduler 和 paper transform 参数。
- `multi_lane/clip_vit.py`
  - 实现官方 positive-token attention：positive similarities 经 `softmax(20*s)`，
    同一权重用于正负分支 token 汇总。
  - train/eval 可分别选择 `clip/none/paper` scale。
- `multi_lane/engine.py`
  - 实现官方 sum binary-softmax 等价 loss：margin BCE sum × `0.03`。
  - 支持 optimizer 跨 task 持续存在和全局 MultiStepLR。
- `multi_lane/utils.py`
  - 清理非当前类别 Adam state，确保旧类 prompt 在持续 optimizer 下不受历史动量漂移。
- `multi_lane/datasets.py`
  - 新增官方 Resize + CutoutPIL + RandAugment + CLIP normalize 路径。
- `main.py`
  - 支持 DDP unscaled optimizer LR override，并打印实际 LR。
- `requirements.txt`
  - 新增 `randaugment==1.0.2`。

服务器验证：

```text
py_compile: passed
paper attention formula equality: passed
pair-softmax BCE == margin BCE(sum): passed
full CPU forward/backward: passed
current task prompts have gradients, future prompts zero: passed
continual Adam frozen-state test: passed
paper scheduler: 5.9e-4 at init, 5.9e-5 after global epoch 20
paper transform with isolated randaugment 1.0.2: passed
```

注意：

- supplementary 的公开 `DDP.py` 硬编码示例是 VOC B4-C2，论文主表还包含 B0-C4；
  方法级参数可以对齐，但 B0-C4 的 PCD 超参仍需用新 attention 路径重新验证。
- 官方代码没有显式清理旧类 Adam 动量；本实现的 state 清理是为了严格满足论文
  “past prompts are frozen as knowledge anchors”的方法描述。

### 2026-06-24 official random 首次启动失败修复

- `voc_ddp_paperattn_semantic_3ep` 已在 GPU0 正常运行。
- `voc_ddp_official_random_3ep` 首次启动后 tmux 立即消失。
- 日志确认不是 OOM；GPU1 为空闲。失败发生在 DataLoader worker：
  `randaugment==1.0.2` 使用了 NumPy 1.24 后删除的 `np.int`。
- 已在 `_paper_randaugment()` 中对该官方旧依赖增加 `np.int = int` 兼容映射，
  不改变 RandAugment 策略。
- 服务器 NumPy 1.26.4 下已验证：
  `Resize -> CutoutPIL -> RandAugment -> ToTensor -> CLIP Normalize`
  可正常输出 `[3,224,224]` tensor。
- 当前 GPU1 空闲，可重启 official random；GPU2-7 均被其他任务占用，不启动 semantic
  official 对照，也不触碰其他用户进程。

### 2026-06-24 official-path 三组 3ep 结果

三组实验均完整结束并同步日志、detail JSON 和 score dumps：

```text
A semantic + paper attention:
  mAP/amAP/OF1/CF1 = 84.4317/90.5850/74.5444/70.9225
  P/R/F1 = 0.849/0.665/0.745
  margin gap = 0.1039

B official random:
  mAP/amAP/OF1/CF1 = 86.0550/91.9571/59.0481/51.9978
  P/R/F1 = 0.954/0.428/0.590
  margin gap = 0.1482
  oracle global threshold 0.7, F1 = 0.745

C official semantic:
  mAP/amAP/OF1/CF1 = 88.0259/93.3177/37.9391/28.7707
  P/R/F1 = 0.985/0.235/0.379
  margin gap = 0.1250
  oracle global threshold 0.6, F1 = 0.796
```

对照结论：

- A 对旧 semantic 3ep：`+0.56 mAP/+0.41 amAP/+4.28 OF1/+0.28 CF1`。
  token attention 显著提高 precision，并确认官方聚合不是无效细节。
- B 对 A：`+1.62 mAP/+1.37 amAP`，说明官方 loss、augmentation、optimizer/scheduler
  组合继续改善排序和 margin。
- C 对 B：`+1.97 mAP/+1.36 amAP`。semantic 初始化在官方路径上仍然有效，
  尽管 overall mean margin gap 更小，再次证明 AP 不能只由 mean gap 判断。
- C 对旧 semantic 3ep：`+4.16 mAP/+3.14 amAP`；对旧 semantic 20ep：
  `+3.75 mAP/+2.92 amAP`。仅 3 epoch 已明显超过此前所有 mAP。
- C 距 DDP 论文 VOC B0-C4：last mAP `-2.17`，average mAP `-1.48`。

official semantic 对旧 semantic 3ep 的关键 per-class AP 变化：

```text
chair       34.95 -> 61.61 (+26.66)
car         80.11 -> 92.69 (+12.57)
tvmonitor   75.01 -> 86.01 (+11.00)
pottedplant 68.26 -> 78.13 (+9.87)
diningtable 77.75 -> 84.77 (+7.02)
sofa        71.77 -> 77.44 (+5.67)
person      89.94 -> 95.74 (+5.79)
```

固定阈值解释：

- B/C 使用 official VOC `tau_max=7,gamma=0.2`。在 B0-C4 final task，tau=7 将
  confidence 大幅压向中间，threshold 0.8 导致 recall collapse。
- C 在 threshold 0.6 时已有 `P=0.763,R=0.832,F1=0.796`，说明排序和可校准上限明显
  高于当前主表 F1。
- 对固定 0.8 阈值，`tau≈2.0` 与 tau7/threshold0.6 的 margin 边界近似等价；
  因此下一步优先 eval-only `tau2/gamma0.7` 和 `tau3/gamma0.7`。

### 2026-06-24 official semantic tau2 优化运行脚本

按用户要求，已基于当前最高 mAP 的 `official semantic` 配置新增两个可直接运行的
VOC B0-C4 脚本：

```text
run_voc_ddp_official_semantic_tau2_3ep.sh
run_voc_ddp_official_semantic_tau2_20ep.sh
```

配置策略：

- 保持 official semantic recipe 不变：`paper_attention`、`paper` logit scale、
  `paper_sum` loss、continual optimizer、paper MultiStepLR、paper transform、
  semantic text initialization。
- 只把 PCD 从官方 VOC 风格的 `tau_max=7.0,gamma=0.2` 改为
  `tau_max=2.0,gamma=0.7`。
- 目的不是改排序，而是先修复 fixed-threshold 0.8 下的 confidence 过度压缩问题。

脚本行为：

- 日志写入：
  - `./logs/voc_ddp_official_semantic_tau2_g07_3ep.log`
  - `./logs/voc_ddp_official_semantic_tau2_g07_20ep.log`
- 输出写入：
  - `./output/voc_ddp_official_semantic_tau2_g07_3ep`
  - `./output/voc_ddp_official_semantic_tau2_g07_20ep`
- 均开启 `--store_model` 和 `--ddp_score_dump true`。
- 可通过 `GPU=<id> ./script.sh` 指定显卡。

已完成本地检查：

```text
bash -n run_voc_ddp_official_semantic_tau2_3ep.sh run_voc_ddp_official_semantic_tau2_20ep.sh
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

2026-06-24 追加修复：

- 远程服务器用 `bash run_voc_ddp_official_semantic_tau2_3ep.sh` 启动时，
  `/root/.bashrc` 在 `set -u` 下引用未定义 `PS1`，导致脚本提前退出。
- 已在两个 tau2 脚本中对 `source ~/.bashrc` 前后临时切换 `set +u` / `set -u`，
  仅规避 shell 初始化兼容问题，不改变任何实验配置。
- 已重新运行 `bash -n`，结果通过。

注意：当前本地目录不是 Git 仓库，无法在本地创建实验分支或提交；Git 状态仍以服务器
`/mnt/haoyuan/workspace/multi-lane-main` 为准。

### 2026-06-24 official semantic tau2 3ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_3ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_3ep
ckpt:   ./output/voc_ddp_official_semantic_tau2_g07_3ep/checkpoints.pth
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.0259/93.3177/79.4418/78.6858
pred/support = 8420/7632
micro precision/recall/F1 = 0.7572/0.8354/0.7944
oracle global threshold = 0.80, F1 = 0.794
per-class oracle F1 = 0.824, threshold mean = 0.790
```

对比：

```text
official semantic tau7 3ep: 88.0259/93.3177/37.9391/28.7707
official semantic tau2 3ep: 88.0259/93.3177/79.4418/78.6858
delta:                         +0.00/+0.00/+41.50/+49.92

official random tau7 3ep:      86.0550/91.9571/59.0481/51.9978
delta tau2 semantic:            +1.97/+1.36/+20.39/+26.69

paper-attention semantic 3ep:   84.4317/90.5850/74.5444/70.9225
delta tau2 semantic:            +3.59/+2.73/+4.90/+7.76
```

结论：

- `tau2/gamma0.7` 没改变排序，mAP 与 tau7 official semantic 完全一致；
  它只改变 fixed-threshold calibration。
- tau2 主阈值 0.8 的结果几乎复现了 tau7 score dump 中 threshold 0.6 的 oracle 行，
  验证此前“tau≈2 对齐主阈值 0.8”的判断成立。
- 当前已接近 DDP 论文 VOC B0-C4 OF1 `80.8`，但 CF1 仍低约 `5` 点，剩余问题主要在
  类别级 calibration，特别是 `bottle/chair/bicycle/diningtable/sheep/sofa`。

### 2026-06-24 official semantic tau2 20ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_20ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_20ep
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.5226/93.4123/72.7760/76.3618
pred/support = 10286/7632
micro precision/recall/F1 = 0.6339/0.8543/0.7278
margin: pos=0.0813, neg=-0.0598, gap=0.1411
global oracle threshold = 0.90, F1 = 0.7584
per-class oracle F1 = 0.805, threshold mean = 0.795
```

关键对比：

```text
official semantic tau2 3ep:  88.0259/93.3177/79.4418/78.6858
official semantic tau2 20ep: 88.5226/93.4123/72.7760/76.3618
delta:                        +0.4967/+0.0946/-6.6658/-2.3240

paper B0-C4:                  90.2/94.8/80.8/76.9
20ep gap:                     -1.68/-1.39/-8.02/-0.54
```

结论：

- 20ep 让 ranking 和 margin 继续改善，但收益不大：mAP 只增加 `0.50`，
  margin gap 从 `0.1250` 增加到 `0.1411`。
- tau2 在 3ep 时校准正好，在 20ep 时偏激进。预测正类从 `8420` 增到 `10286`，
  precision 从 `0.757` 降到 `0.634`，而 recall 仅从 `0.835` 升到 `0.854`。
- 20ep 最佳扫描阈值已经移到 `0.9`，因此下一步先使用现有 checkpoint 做
  eval-only tau `2.5/3/4` 对照，不能把 OF1 下降误判成结构退化。
- mAP 剩余问题集中在困难类和跨任务退化。最终弱类：
  `bottle 61.96`、`chair 61.45`、`sofa 77.13`、`pottedplant 79.00`、
  `diningtable 82.03`。
- 从类别刚学完到 task5，主要 AP 下降：
  `bottle -29.97`、`chair -14.96`、`bicycle -10.53`、`cow -9.03`、
  `diningtable -5.92`。这不是旧 prompt 参数漂移的直接证据，因为 prompts 已冻结；
  更可能是后续 seen-class 评估中类别间分数竞争、任务数据分布和负样本覆盖不足。

### 2026-06-24 最佳 20ep tau2 版本提交与推送

- 分支：`feature/clip-taskCLS-posneg-text-head`
- 提交：`5ac25a8 Align CLIP DDP with official training recipe`
- 已推送：`origin/feature/clip-taskCLS-posneg-text-head`
- 提交包含 paper-aligned DDP 实现、official semantic tau2 3ep/20ep 运行脚本及实验记录。
- `logs/`、`output/`、`checkpoints.pth` 均未进入 Git。
- 服务器工作区仍有无关的 `train_c100.sh` 末尾换行变化，未纳入提交。

### 2026-06-24 创建 EMOTIC DDP tau2 分支

- 从 `feature/clip-taskCLS-posneg-text-head` 的 `01c10fc` 创建
  `feature/emotic-ddp-semantic-tau2`。
- 数据集确认存在于服务器 `./datasets/EMOTIC`，加载参数使用 `--data_path ./datasets`。
- 采用 Split-EMOTIC 默认 alphabetical B5-C3 协议：
  `base_classes=5,num_tasks=8`，共 26 个类别。
- `multi_lane/configs/emotic.py` 的 DDP semantic prompt 默认值改为情感专用前缀：
  positive `a photo of a person clearly feeling`，negative
  `a photo of a person not feeling`。
- 新增 `run_emotic_ddp_official_semantic_tau2_20ep.sh`，完整迁移 VOC 当前最佳
  paper-aligned DDP 20ep tau2 配置，输出独立 log/output/checkpoint/score dump，
  detail HTML/JSON 继续复用同一评估引擎。
- 服务器验证结果：
  - Python 静态编译和脚本 `bash -n` 通过；
  - parser 默认提示与脚本提示一致；
  - train/eval 人物样本数为 `16001/7765`，26 类任务切分为
    `[5,3,3,3,3,3,3,3]`；
  - DDP text/visual prompt shapes 为 `(26,2,16,512)` / `(26,2,16,768)`；
  - positive/negative semantic seeds 不相同；
  - 报告 smoke 成功生成 `.html` 和 `.json` 双文件。

### 2026-06-24 EMOTIC B5-C3 real-data smoke

- 用户确认在 GPU0 运行 1ep smoke。
- 完整配置保持正式 20ep recipe 不变，仅使用 `epochs=1`；B5-C3、batch8、full image、
  paper attention/loss/optimizer/scheduler/transform、semantic emotion prompts、
  PCD tau2/gamma0.7，并保留 checkpoint、score dump、HTML/JSON。
- 独立 tmux：`ddp_emotic_tau2_smoke_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke`
- 用户确认 GPU0 剩余显存足够后，取消等待并立即并行启动；未触碰
  `/mnt/haoyuan/workspace/CODE_DDP` 的现有实验。
- 启动后 GPU0 总占用约 `9.4/24.6GB`，本 smoke 进程约 `5.2GB`；
  task0 首个训练 batch 正常完成，无 OOM。

### 2026-06-24 EMOTIC B5-C3 30ep 正式配置

- 参考对照：
  `emotic_b5c3_alphabetical_valtest_clip_taskCLS_text_bias_lr001563_v1`
- 对齐项：Split-EMOTIC、26 类字母序、B5-C3、full image、val+test、seed0、
  224 输入、8 tasks、每 task 30 epochs、Adam、weight decay 0、相同 detail
  HTML/JSON 指标口径。
- batch 对齐采用 physical batch8、gradient accumulation32、effective batch256；
  避免 class-wise positive/negative visual branches 在 physical batch256 下 OOM。
- DDP 方法专属项保留当前最佳配置：drop_last、paper attention、
  paper summed BCE `*0.03`、continual Adam 配置 LR `0.0059`、paper multistep、
  semantic emotion prompts、PCD tau2/gamma0.7。
- 旧 taskCLS 对照的 batch256、cosine scheduler 和 LR `0.001563` 不直接迁移，
  其中 batch256 通过梯度累积等效实现；cosine scheduler 和 LR `0.001563` 不迁移，
  因为当前最佳结果来自 official DDP 优化协议。
- 修复 `train_one_epoch` 的 accumulation step 边界，并区分 mean loss 与
  official `paper_sum` 的梯度缩放；accumulation1 行为保持不变。
- 正式脚本：`run_emotic_ddp_official_semantic_tau2_30ep.sh`
- tmux：`ddp_emotic_tau2_30ep_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep`
- 首次 tmux 启动在 Python 前退出，launcher log 确认非交互 shell 中
  `/root/.bashrc` 提前 return，`conda` 未定义。EMOTIC 20ep/30ep 脚本已改为直接
  source `/opt/conda/etc/profile.d/conda.sh`，不影响训练参数。
- 第二次启动发现 tmux 继承的 `csc` 环境 deactivate hook 与 `set -u` 不兼容；
  已仅在 `conda activate multilane` 前后临时切换 `set +u/set -u`。
- 修复 shell 初始化后正式实验成功进入 task0 epoch1：
  - accumulation32、pin_mem false、effective optimizer LR `0.0059` 配置均已解析；
  - scheduler 初始化后的实际 LR 为 `0.000590`；
  - 正式实验自身显存峰值约 `4.3GB`；
  - GPU0 三个并行实验总占用约 `15.6/24.6GB`；
  - 未发现 OOM、Traceback、RuntimeError 或 loss NaN。

### 2026-06-25 EMOTIC DDP smoke/30ep 结果分析

两组均完整跑完 8 tasks，HTML/JSON、8 个 score dump 和服务器 checkpoint 均生成。

```text
1ep smoke: mAP/amAP/OF1/CF1 = 29.8272/36.6230/21.8038/9.5456
30ep run:  mAP/amAP/OF1/CF1 = 31.3195/37.8293/25.9253/10.1560
delta:                              +1.4923/+1.2063/+4.1215/+0.6104
```

正式 30ep 与历史对照：

```text
DDP semantic tau2: 31.3195/37.8293/25.9253/10.1560
taskCLS text bias: 31.1848/37.2170/57.2128/22.7483
CLIP patch concat: 32.8635/39.8831/47.0667/20.2515
```

关键判断：

- 30ep effective batch256 与 1ep batch8 的 optimizer update 数接近：
  每 task 约 `30*ceil(num_batches/32)` 对比 smoke 的 `num_batches`，因此两者结果接近
  是合理现象；30ep 的梯度统计更接近历史 batch256 对照。
- DDP 对 taskCLS-text-bias 的 final mAP 仅提高 `0.13`，amAP 提高 `0.61`；
  对普通 CLIP patch baseline 则低 `1.54/2.05`。排序收益不足。
- final threshold0.8 precision/recall/F1 为 `0.754/0.157/0.259`；
  threshold0.5 为 `0.432/0.646/0.518`。tau2 在 EMOTIC 上明显过保守。
- 即使使用 global oracle threshold，OF1 仍比 taskCLS-text-bias 低约 `5.45` 点，
  因此不能把差距全部归因于阈值。
- final raw margin：positive `0.0075`、negative `-0.0210`、gap `0.0285`；
  positive margin >0 rate `64.6%`，negative margin >0 rate `20.3%`。
  26 类中 17 类 positive margin mean <=0，说明 macro/class-wise 分离不足。
- 最弱 AP 包括：Embarrassment `2.12`、Sensitivity `4.47`、Fear `8.49`、
  Surprise `8.57`、Aversion `8.95`、Yearning `11.21`。
- 相对 taskCLS-text-bias，DDP 大幅改善 Affection `+19.84 AP`，并改善
  Confidence `+4.51`、Pleasure `+2.98`；主要下降为 Suffering `-9.08`、
  Disquietment `-6.15`、Doubt/Confusion `-4.89`、Sadness `-3.57`。
- 从类别刚学完到最终评估，早期类 AP 下降主要来自评估样本/负样本分布扩展，
  不是旧 prompt 参数更新：Anticipation `-26.20`、Affection `-13.38`、
  Annoyance `-8.98`。模型参数冻结不保证在新增负样本上的 AP 不变。
### 2026-07-01 EMOTIC test-only 评估入口

- 发现历史 `emotic_b5c3_alphabetical_valtest_clip_vit_b16_patch_v1` 报告使用
  `val+test`。该实验没有保存 checkpoint，因此无法只重跑测试，必须重训才能得到
  对应模型的纯 test 结果。
- 在 EMOTIC CLI 增加 `--emotic_eval_splits`，默认 `val test`，可显式设为 `test`。
- 修复数据构建入口未传递 `emotic_input_mode` 的问题，并把选择的 eval splits 传入
  EMOTIC dataset。
- 新增历史 CLIP patch concat baseline 的纯 test 30ep 脚本，保持 B5-C3、26 类字母序、
  batch256、seed0、full image 等训练配置不变，新增 checkpoint 保存。
- 预期产物为独立 log、checkpoint、class order JSON、逐类逐任务 HTML/JSON。
- 用户确认历史代码位于 `feature/clip-vit-b16`。已从 `0b59138` 建立独立 worktree
  `/mnt/haoyuan/workspace/multi-lane-main-test-only` 和分支
  `fix/emotic-test-only-clip-vit-b16`，没有切换或污染当前 DDP 工作区。
- 实测 eval 样本数为 val `2397`、test `5368`、val+test `7765`；test-only 仍为
  26 类字母序及 B5-C3 八任务。
- 纯 test 30ep 正式实验已在 GPU1、tmux `emotic_clip_b16_testonly_30ep` 启动。
  task0 前两轮训练指标与历史 val+test 日志逐项一致，验证训练协议未改变；GPU1
  峰值约 8.2GB，无 OOM。
- B5-C3 纯 test 已完成，final `29.9174/36.6938/45.5877/19.5187`，耗时 `36:01`。
- 旧 upper bound 配置核对为 joint training：`dataset=EMOTIC`、1 task、26 base classes、
  30 epochs、batch256、CLIP patch concat；旧实验 `store_model=False`，因此纯 test 必须
  重训。
- 新增 upper-bound test-only 脚本，在 GPU0/tmux `emotic_upper_testonly_gpu0` 启动。
  Namespace 已确认 `emotic_eval_splits=['test']`，首轮训练数值与旧 val+test upper
  bound 一致，预计约 14–16 分钟完成。

### 2026-08-11 本地/服务器独立 Git 工作树迁移

- 只读审计确认服务器主工作树原分支为 `fix/emotic-test-only-eval@b984688`；该分支没有
  upstream，但 HEAD 已存在于 `origin/feature/emotic-ddp-semantic-tau2`，没有未推送的
  普通提交。服务器原有三个 stash 均为仅存于服务器的 WIP。
- 审计确认额外 worktree `/mnt/haoyuan/workspace/multi-lane-main-test-only` 位于
  `fix/emotic-test-only-clip-vit-b16@0b59138`，包含两处代码修改、两个未跟踪运行脚本和
  指向主数据集目录的符号链接；迁移期间未修改该 worktree。
- 迁移前完整备份位于
  `/mnt/haoyuan/workspace/git-migration-backups/multi-lane-main-20260811-1145/`：包含经
  `git bundle verify` 验证的全引用 bundle、两个 worktree 的 binary patch、状态清单、
  stash 清单、未跟踪 PNG/脚本副本及 SHA-256 校验清单。大型数据、日志、output 和权重
  未复制或纳入 Git。
- 创建并推送安全分支 `backup/pre-local-git-migration-20260811-1145@b984688`，随后将
  服务器主工作树迁移前改动保存到命名 stash `7d0366b`，并正常切换到
  `feature/clip-vit-b16@0b59138`。
- 本地原目录已执行 `git init`，添加相同 origin 并 fetch；使用 mixed 基线识别出的修改
  与服务器备份 patch 及 PNG SHA-256 完全一致。本地迁移前改动保存到命名 stash
  `a9a4ef5`，`.idea/` 和 `.DS_Store` 写入本地 `.git/info/exclude`。
- 从 `feature/clip-vit-b16@0b59138` 创建并推送
  `fix/independent-git-worktrees`。本地和服务器主工作树均跟踪该远端分支，Git 元数据
  完全独立；PyCharm Automatic Upload 已关闭。
- 本次迁移未运行训练或模型测试；验证范围为 Git 引用、远端提交、stash/patch 备份、
  两端 HEAD/upstream、工作树状态和文件校验和。
- 用户已确认将 `.gitignore` 和四份迁移上下文文档提交并推送到
  `fix/independent-git-worktrees`，再由服务器使用 `git pull --ff-only` 同步。
