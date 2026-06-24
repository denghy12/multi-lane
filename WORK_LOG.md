nvi# 工作日志

最后一次更新：2026-06-23。

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
