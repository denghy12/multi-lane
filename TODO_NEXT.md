# 下一步任务

最后一次更新：2026-06-22。

## 最高优先级

1. 先补齐 DDP 诊断/复评估能力，再启动下一批长实验。

   目标：让同一个 checkpoint 能反复评估不同 PCD 参数、aggregation 和阈值诊断，避免每改一个 `tau_max/gamma` 都重新训练 20 epoch。

   建议代码改动：

   - 增加 DDP eval diagnostic 输出：每个 task 记录 `predicted_positive/support`、micro precision、micro recall、score/logit 的均值、分位数、正负样本分布。
   - detail report 增加最终 task 的 score 分布字段，方便判断是排序问题、校准问题还是阈值问题。
   - 增加 checkpoint 复评估入口或最小脚本：读取 `--store_model` 保存的 `checkpoints.pth`，只跑 validation/test，不重新训练。
   - 保持论文主指标仍使用项目默认 `sigmoid(logits) > 0.8`，但允许诊断模式打印不同阈值下的 precision/recall/F1 曲线；诊断结果不作为论文对齐主表。
   - 后续所有 20 epoch 论文协议实验默认加 `--store_model`，保证训练完成后可重评估 PCD。

2. 基于 VOC B0-C4 结果修正下一轮实验设计。

   当前 B0-C4 final：

   ```text
   mAP 76.9800, amAP 85.7374, oF1 21.5977, cF1 22.8244
   ```

   主要问题不是 precision，而是 recall 被 PCD/阈值压得过低。优先做 PCD 强度诊断：

   - `ddp_pcd=false`
   - `ddp_tau_max=2.0, ddp_gamma=0.7`
   - `ddp_tau_max=3.0, ddp_gamma=0.7`
   - `ddp_tau_max=5.0, ddp_gamma=0.5`
   - 保留 `tau_max=7,gamma=0.2` 作为强抑制对照

   推荐先用短实验筛参数：

   - VOC B0-C4，`epochs=3` 或 `epochs=5`，快速看 final predicted positives、precision/recall 和 mAP。
   - 固定其余参数，只改 PCD，避免混合变量。
   - 找到 recall 不再崩塌的范围后，再跑 20 epoch。

3. 监控当前 VOC 论文协议 tmux 实验，不要动别人服务器上的进程。

   ```text
   voc_b0c4_ddp_gpu1
   voc_b10c2_ddp_gpu0
   ```

   如果 B10-C2 因 GPU0 显存 OOM 再次退出，按用户要求等待，不继续调整或触碰其他人的任务。

4. 评估 visual similarity aggregation。

   当前默认 `ddp_similarity_aggregation=mean`，可能稀释 class-specific visual signal。下一轮建议按同一 PCD 配置比较：

   - `mean`
   - `max`
   - `cls`

   先用 VOC 短实验判断方向，再决定是否用于 EMOTIC。

5. 在真实 EMOTIC batch 上做 smoke 前，向用户确认完整实验配置。

   建议初始配置：

   ```text
   dataset: Split-EMOTIC
   backbone: clip_vit_b16_patch
   head_mode: clip_ddp
   ddp_prompt_length: 16
   ddp_prompt_layers: 5
   ddp_pcd: true
   ddp_tau_max: 3.0
   ddp_gamma: 0.7
   ddp_similarity_aggregation: mean
   ddp_class_chunk_size: 4
   output: ./output/emotic_clip_ddp_smoke
   log: ./logs/emotic_clip_ddp_smoke.log
   ```

6. EMOTIC smoke 通过后，再决定是否推送当前 DDP v1/PCD 本地提交。

## 已完成检查

- 服务器 `multilane` 环境静态检查通过；新增 DDP logit scale 修正后需要重新运行。
- `head_mode=clip_ddp` 与 `ddp_*` 参数解析通过。
- 结构实例化检查通过：仅 `ddp_text_prompts` 和 `ddp_visual_prompts` 可训练，optimizer 参数组 weight decay 为 `0.0`。
- dummy forward/backward smoke 通过：训练 `tau=1.0`，eval PCD `tau=3.0`，task0 只有 class `[0,1]` prompts 有梯度，切到 task1 后可训练 mask 为 `[2,3]`。
- VOC smoke 已暴露 CF1/OF1 全 0 的阈值尺度问题，已修正 DDP final logit 为 `CLIP logit_scale * (s_pos - s_neg) / tau`；重新运行 `voc_clip_ddp_smoke_scale` 后 final mAP/OF1/CF1 为 71.52/64.06/62.12。

## 后续对照实验

- `clip_ddp` + PCD on/off。
- `ddp_similarity_aggregation` 的 `mean`、`max`、`cls` 对比。
- `tau_max/gamma` 使用 COCO 风格和 VOC 风格的对照。
- 与已有 `clip_vit_b16_patch`、`clip_taskCLS_text`、SigLIP-style global scale 结果对比。

## 推送前检查

- 服务器 `git status --short --branch`。
- `git diff --stat`。
- Python 语法检查。
- forward smoke 和梯度冻结检查。
- 更新 `PROJECT_CONTEXT.md`、`WORK_LOG.md`、`TODO_NEXT.md`。
- 经用户确认后再考虑 push 到远端。
