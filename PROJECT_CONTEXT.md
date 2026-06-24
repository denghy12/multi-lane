# 项目上下文

最后一次更新：2026-06-24；当前分支为 `feature/clip-taskCLS-posneg-text-head`。

## 项目目标

本仓库基于 MULTI-LANE 官方实现：

> "Less is more: Summarizing Patch Tokens for efficient Multi-Label Class-Incremental Learning (MULTI-LANE)", CoLLAs 2024。

当前项目方向是在 MULTI-LANE 基础上做 CLIP 相关多标签类增量学习实验，重点数据集为 EMOTIC。当前已从 `clip_taskCLS_text` 基线新增服务器本地提交版 DDP v1 / Progressive Confidence Decoupling (PCD) 风格结构，尚未 push 到远端。

## 开发与运行环境

- 本地负责代码编辑，远程 Linux 服务器负责 Git、训练和实验运行。
- 服务器项目路径：`/mnt/haoyuan/workspace/multi-lane-main`。
- 服务器 conda 环境：`multilane`。
- 数据集优先使用 `./datasets/`。
- 实验日志写入 `./logs/`。
- 实验输出写入 `./output/`。
- 本地目录不是权威 Git 仓库；Git 状态以服务器为准。

## 当前技术路线

已提交基线：

- `clip_vit_b16_patch` 使用 `open_clip` 加载 CLIP ViT-B/16 visual transformer。
- `clip_taskCLS_text` 使用 MULTI-LANE task CLS，经 `visual_proj` 映射后与固定 CLIP text prototypes 计算 similarity。
- `main.py` 已从 dataloaders 提取类别名并传入支持 `set_class_names()` 的模型。

当前服务器本地已提交的 DDP v1 / PCD 改造：

- 新增独立 `head_mode=clip_ddp`，保留旧 head 不变。
- DDP 路径不使用 MULTI-LANE selectors/task CLS 作为分类依据，而是以 class-specific prompts 为基本单位。
- 每个类别拥有 learnable positive/negative text prompts 与 positive/negative visual prompts。
- 文本侧通过 CLIP text transformer 编码 `[SOT] + learnable prompts + class name + [EOT]`。
- 视觉侧在最后 `ddp_prompt_layers` 个 CLIP visual blocks 中注入 class-specific visual prompts。
- 输出使用 `CLIP logit_scale * (s_pos - s_neg) / tau` 作为 binary softmax 的 logit；训练阶段 `tau=1`，评估阶段可启用 PCD。
- 当前 EMOTIC PCD 默认采用 `ddp_tau_max=3.0`、`ddp_gamma=0.7`。

当前本地/服务器同步中的下一阶段未提交改动：

- `--eval` 开始支持通过 `--eval_checkpoint` / `--eval_dir` / `output_dir/checkpoints.pth` 加载 checkpoint 并只跑评估。
- DDP 评估会输出并写入 detail report：`predicted_positive/support`、micro precision/recall/F1、score/logit 分布和多阈值诊断。
- 新增 `--ddp_diagnostics` 与 `--ddp_diagnostic_thresholds`，默认诊断阈值为 `0.5 0.7 0.8 0.9`，主指标仍保持项目默认 `sigmoid(logits) > 0.8`。
- 结构核对后新增 DDP 实验开关，默认保持旧结果可复现：`--ddp_logit_scale_mode clip|none`、`--ddp_prompt_norm_mode legacy|prompted`、`--ddp_similarity_aggregation pooled_cls/patch_mean/patch_max/cls_plus_patch_max/topk_mean`、`--ddp_similarity_topk`、`--ddp_train_text_prompts`、`--ddp_train_visual_prompts`、`--ddp_prompt_polarity`。
- 2026-06-23 margin-first 排查新增 `--ddp_text_init random|same|semantic`、`--ddp_positive_text_template`、`--ddp_negative_text_template`；`semantic` 会用正/负模板前缀初始化 learnable text prompt，默认 `random` 保持旧实验可复现。
- detail report 与 DDP diagnostics 新增正/负样本拆分后的 `margin`、`ddp_prob`、`margin_gap` 和过阈值比例，用于直接判断 positive/negative prompt margin 是否真正拉开。

## 当前已知方法表现

从已同步日志确认：

- Baseline `clip_vit_b16_patch` 完整增量实验：
  - `mAP 32.8635`，`amAP 39.8831`，`oF1 47.0667`，`cF1 20.2515`，`Loss 0.4933`。
- `clip_taskCLS_text` 使用 CLIP scale 加 per-class bias：
  - `mAP 31.1848`，`amAP 37.2170`，`oF1 57.2128`，`cF1 22.7483`，`Loss 0.5715`。
- SigLIP-style global scale 完整增量实验：
  - `mAP 30.5178`，`amAP 36.3047`，`oF1 22.8700`，`cF1 11.4511`，`Loss 0.4596`。

## 重要注意事项

- `clip_ddp` 目标严格按 DDP v1 / PCD，而不是 DeCLIP v2 / AST。
- 半成品 DDP 代码已保存到 `stash@{0}: wip clip_ddp draft before clean restart`，本轮实现没有直接恢复该 stash。
- 当前实现已在服务器环境完成静态检查、配置解析、dummy forward/backward smoke 和梯度冻结检查；尚未运行 EMOTIC smoke 实验。
- 2026-06-22 VOC smoke 暴露出纯 cosine DDP logit 在 `sigmoid > 0.8` 阈值下会导致 CF1/OF1 全 0，因此已恢复 CLIP `logit_scale` 到 DDP similarity logit；重新 smoke 后 final mAP/OF1/CF1 为 71.52/64.06/62.12。
- VOC B0-C4 需要 `base_classes=0,num_tasks=5` 特殊切分，当前已修正为空 base 不再产生空 task；B0-C4 `cls + tau3/gamma0.7` 20 epoch 已完成，B10-C2 需在 B0-C4 checkpoint 复评估后按选定 PCD 配置继续跑。
- 服务器本地提交 `Add CLIP DDP prompt head` 已创建，按用户要求尚未 push 到远端。
- VOC B0-C4 20 epoch `tau_max=7,gamma=0.2` 的 final mAP/OF1/CF1 为 76.98/21.60/22.82；问题主要是 PCD 抑制过强导致 recall 极低。
- VOC B0-C4 3 epoch `tau_max=3,gamma=0.7` 诊断实验 final mAP/OF1/CF1 为 76.16/64.21/62.18；checkpoint 复评估结果与训练末尾完全一致，说明 eval-only 路径可用。
- 同一 checkpoint 关闭 PCD 后 final mAP/OF1/CF1 为 76.16/56.00/59.06；PCD off 提高 recall 但 precision 掉太多，当前 `tau_max=3,gamma=0.7` 更优。
- 同一 checkpoint PCD sweep 已完成：tau2/gamma0.7 final OF1/CF1 为 62.98/63.15，tau3/gamma0.7 为 64.21/62.18，tau5/gamma0.5 为 46.40/45.95，tau7/gamma0.2 为 20.34/23.77；当前 tau3 最均衡，tau2 可作为 macro-F1 候选。
- VOC B0-C4 3 epoch aggregation 对照已完成：mean final mAP/OF1/CF1 为 76.16/64.21/62.18，max 为 71.66/59.63/59.11，cls 为 77.77/64.05/63.88；当前 `cls` 的 mAP 和 CF1 最好，`max` 明显不如 mean/cls。
- VOC B0-C4 `cls` checkpoint PCD sweep 已完成：tau2/gamma0.7 final OF1/CF1 为 60.36/62.05，tau3/gamma0.7 为 64.05/63.88，tau5/gamma0.5 为 55.33/55.44，tau7/gamma0.2 为 36.01/39.61；固定阈值 0.8 下 `cls + tau3/gamma0.7` 是当前最佳主配置。
- VOC B0-C4 20 epoch `cls + tau3/gamma0.7` 已完成并保存 checkpoint：final mAP/amAP/OF1/CF1 为 79.79/87.39/65.70/64.75，pred/support 为 8568/7632，micro P/R/F1 为 0.621/0.697/0.657；这是当前 VOC B0-C4 最好结果。
- 当前主要差距：相对 DDP 论文 VOC B0-C4 主表仍低约 10 mAP、15 OF1、12 CF1。问题不再是 tau7 那种 recall collapse，而是 class-level calibration 和 per-class prompt 表达仍不足，尤其 chair、bicycle、sheep、diningtable、sofa、pottedplant、bottle、bird 等类别。
- 2026-06-23 结构修正已开始：`pooled_cls` 用于显式表示论文抽象中的单个 class-conditioned visual feature；`prompted` pre-norm 可让 prompt 与 image tokens 一起过 LayerNorm；`ddp_logit_scale_mode=none` 可检查严格 raw cosine DDP/PCD，`clip` 保留项目阈值兼容路径。
- VOC B0-C4 结构对照 3 epoch 已完成：`pooled_cls+prompted+raw` final mAP/OF1/CF1 为 `81.71/0/0`，证明 raw cosine 排序不差但固定阈值不兼容；`pooled_cls+prompted+clip` 为 `79.86/61.55/62.97`，高于旧 control `cls+legacy+clip` 的 `77.10/61.55/64.16`。当前结构候选更新为 `pooled_cls + prompted + clip + tau3/gamma0.7`。
- GPT5.5Thinking 独立分析已纳入当前判断：下一步优先补严格 DDP probability 评估、score dump、dense threshold/oracle threshold、CLIP mean/std normalize 和 VOC class-order 日志，再决定是否直接启动新结构 20 epoch。原因是当前 gap 更像 calibration/protocol/preprocessing 与 prompt 细节叠加，而不是 DDP 主框架缺失。
- 2026-06-23 已实现第一批 calibration 诊断：`--ddp_eval_score_mode logits|probability`、`--ddp_eval_threshold`、`--ddp_score_dump`、dense threshold/oracle threshold、`s_pos/s_neg/margin/ddp_prob/scaled_logit` 分布、`--clip_normalize_input` 和 class order JSON。默认仍是 logits 模式，旧实验行为保持。
- 2026-06-23 已在服务器启动三个 tmux 诊断实验：`voc_ddp_eval_logits_diag` on GPU0、`voc_ddp_eval_probability_diag` on GPU1、`voc_ddp_3ep_clipnorm` on GPU4。等待用户同步完整日志/输出后分析。
- 三组诊断结果已同步：strict probability final mAP/OF1/CF1 为 `79.86/46.23/51.53`，明显差于 scaled-logit `79.86/61.55/62.97`，说明 raw DDP probability 分布过度集中在 0.5 附近；CLIP normalize 3 epoch final 为 `81.57/65.06/66.61`，较 no-normalize 同结构提升 `+1.71 mAP/+3.51 OF1/+3.64 CF1`。下一步主配置应改为 `pooled_cls + prompted + clip + CLIP normalize`，优先跑 20 epoch。
- VOC B0-C4 `pooled_cls + prompted + clip + CLIP normalize` 20 epoch 已完成并保存 checkpoint：final mAP/OF1/CF1 为 `82.95/67.32/68.90`，相对旧 `cls+legacy+clip` 20 epoch 提升 `+3.16/+1.61/+4.16`。final threshold 0.9 的 micro-F1 为 `0.690`，高于主阈值 0.8 的 `0.673`，下一步应基于该 checkpoint 做 `tau/threshold` eval-only sweep。
- 2026-06-23 已开始 margin separation 结构排查：新增 semantic text prompt initialization 和 per-class positive/negative margin diagnostics；下一轮优先跑 `random` vs `semantic` 的短实验，并检查 `pos_margin_mean`、`neg_margin_mean`、`margin_gap_mean` 是否扩大。
- VOC B0-C4 3 epoch `random` vs `semantic` text init 对照已完成：semantic final mAP/OF1/CF1 为 `83.87/70.26/70.64`，优于 random 的 `82.34/66.73/67.89`；但 semantic 的 raw margin gap 从 `0.1688` 降到 `0.1356`，说明收益主要来自减少过度预测和改善 scaled-logit 阈值表现，而不是直接扩大 raw DDP margin。
- VOC B0-C4 semantic 20ep 与 branch ablation 已完成：semantic 20ep 达到当前最高 mAP `84.27`，OF1/CF1 为 `70.54/72.57`；3ep `text_frozen` 达到当前最高 OF1/CF1 `71.16/72.77`。这说明固定 semantic text anchors、主要训练 visual prompts 可能比 full text+visual 长训更适合固定阈值 F1。
- 所有 semantic/ablation 组的 raw DDP probability 仍集中在 0.5 附近，`raw_prob_pos_mean≈0.506`、`raw_prob_neg_mean≈0.495`，说明 raw margin 尺度问题仍未解决；当前好结果主要来自 CLIP logit scale 后的 calibrated logits。
- 2026-06-24 已完成 `text_frozen 20ep`、`positive-only 20ep` 及 seed1 稳定性复验：
  - `text_frozen 20ep` final `mAP/amAP/OF1/CF1=80.84/87.91/68.22/71.30`，明显低于同配置 3ep seed0 的 `82.32/89.16/71.16/72.77`；predicted positives 从 `9044` 增到 `10475`，precision 从 `0.656` 降到 `0.590`，说明长训导致过预测。
  - `positive-only 20ep` final `83.06/89.38/70.09/72.61`；相对 3ep 提升 mAP/CF1，但 OF1 下降，仍未超过 full semantic 20ep 的 mAP/OF1。
  - full semantic 3ep seed1 为 `83.68/90.18/70.60/70.85`，与 seed0 的 `83.87/90.18/70.26/70.64` 高度一致；这是当前最稳定的结构。
  - text-frozen 3ep seed1 为 `81.54/88.55/69.29/71.37`，较 seed0 回落 `0.78 mAP/1.87 OF1/1.40 CF1`，此前 seed0 的最高 F1 有明显随机性。
  - text-frozen 的 learnable 参数为 `491,520`，与论文约 `0.49M` 参数规模高度吻合，但实验不支持仅凭参数量认定该分支就是论文完整训练结构。
- 当前主结论更新为：full semantic text+visual prompts 是稳定复现主线；text-frozen 和 positive-only 保留为消融，不再作为下一轮默认主结构。下一步优先排查 20 epoch 调度与类别级过预测，尤其 `bicycle/chair/bottle/car`。
- 运行任何训练前必须先向用户说明完整实验配置并等待确认。
