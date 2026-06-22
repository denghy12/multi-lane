# 工作日志

最后一次更新：2026-06-22。

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
