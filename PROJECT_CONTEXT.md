# 项目上下文

最后一次更新：2026-08-11；本地和服务器主工作树当前分支均为
`fix/independent-git-worktrees`，基线为 `feature/clip-vit-b16@0b59138`。

## 项目目标

本仓库基于 MULTI-LANE 官方实现：

> "Less is more: Summarizing Patch Tokens for efficient Multi-Label Class-Incremental Learning (MULTI-LANE)", CoLLAs 2024。

当前项目方向是在 MULTI-LANE 基础上做 CLIP 相关多标签类增量学习实验，重点数据集为
EMOTIC。当前工作分支以最初的 `feature/clip-vit-b16` 代码为基线；后续
`clip_taskCLS_text`、DDP v1 / Progressive Confidence Decoupling (PCD) 改造仍完整保留在
对应远端功能分支和迁移安全备份中，并未丢失。

## 开发与运行环境

- 本地 macOS 项目路径：`/Users/denghaoyuan/workspace/MyCode/multi-lane-main`。
- 服务器项目路径：`/mnt/haoyuan/workspace/multi-lane-main`。
- 本地和服务器分别维护独立 `.git`，共同跟踪
  `git@github.com:denghy12/multi-lane.git`；Git 负责受控代码同步。
- PyCharm Automatic Upload 已关闭；`.git/` 永远不得通过 SFTP 同步。
- 服务器 conda 环境：`multilane`。
- 数据集优先使用 `./datasets/`。
- 实验日志写入 `./logs/`。
- 实验输出写入 `./output/`。

## Git 架构与迁移状态

- 2026-08-11 已完成从“服务器单边 Git + SFTP 代码同步”到“两端独立 Git 工作树”的迁移。
- 两端主工作树共同跟踪 `origin/fix/independent-git-worktrees`；该分支从
  `feature/clip-vit-b16@0b591388da2f9efa2e9157a00c64d1b3d68d5c38` 创建，迁移文档
  提交不改变该业务代码基线。
- 迁移前服务器 HEAD `b984688` 已由远端分支
  `backup/pre-local-git-migration-20260811-1145` 保护；同一提交也仍存在于
  `origin/feature/emotic-ddp-semantic-tau2`。
- 完整 Git bundle、stash、patch、状态清单和未跟踪业务文件备份位于服务器：
  `/mnt/haoyuan/workspace/git-migration-backups/multi-lane-main-20260811-1145/`。
- 恢复完整仓库时应在新目录执行
  `git clone repository-all-refs.bundle <new-directory>`；恢复 patch 时先在其原始基线分支
  执行 `git apply --check <patch>`，确认无冲突后再 `git apply <patch>`。未跟踪文件从
  `main-untracked/` 或 `test-only-untracked/` 手动复制，禁止覆盖现有文件。
- 服务器主工作树迁移前改动保存在命名 stash（提交 `7d0366b`），本地对应改动保存在
  本地 stash（提交 `a9a4ef5`）；恢复前必须先创建专用分支并检查 patch。
- 服务器额外 worktree `/mnt/haoyuan/workspace/multi-lane-main-test-only` 保持原样，分支为
  `fix/emotic-test-only-clip-vit-b16`，其中仍有 test-only 业务改动和两个未跟踪运行脚本。
- 标准开发流程为：本地分支修改、检查、提交并 push；服务器 fetch 后切换同一分支，
  仅允许 fast-forward 更新。标准回滚流程为从旧 commit 创建恢复分支，而不是硬重置。

## 当前技术路线

已提交基线：

- `clip_vit_b16_patch` 使用 `open_clip` 加载 CLIP ViT-B/16 visual transformer。
- `clip_taskCLS_text` 使用 MULTI-LANE task CLS，经 `visual_proj` 映射后与固定 CLIP text prototypes 计算 similarity。
- `main.py` 已从 dataloaders 提取类别名并传入支持 `set_class_names()` 的模型。

后续功能分支中已提交的 DDP v1 / PCD 改造（当前基线分支不包含这些实现）：

- 新增独立 `head_mode=clip_ddp`，保留旧 head 不变。
- DDP 路径不使用 MULTI-LANE selectors/task CLS 作为分类依据，而是以 class-specific prompts 为基本单位。
- 每个类别拥有 learnable positive/negative text prompts 与 positive/negative visual prompts。
- 文本侧通过 CLIP text transformer 编码 `[SOT] + learnable prompts + class name + [EOT]`。
- 视觉侧在最后 `ddp_prompt_layers` 个 CLIP visual blocks 中注入 class-specific visual prompts。
- 输出使用 `CLIP logit_scale * (s_pos - s_neg) / tau` 作为 binary softmax 的 logit；训练阶段 `tau=1`，评估阶段可启用 PCD。
- 当前 EMOTIC PCD 默认采用 `ddp_tau_max=3.0`、`ddp_gamma=0.7`。

迁移前后续分支及工作记录中的评估/诊断改动：

- `--eval` 开始支持通过 `--eval_checkpoint` / `--eval_dir` / `output_dir/checkpoints.pth` 加载 checkpoint 并只跑评估。
- DDP 评估会输出并写入 detail report：`predicted_positive/support`、micro precision/recall/F1、score/logit 分布和多阈值诊断。
- 新增 `--ddp_diagnostics` 与 `--ddp_diagnostic_thresholds`，默认诊断阈值为 `0.5 0.7 0.8 0.9`，主指标仍保持项目默认 `sigmoid(logits) > 0.8`。
- 结构核对后新增 DDP 实验开关，默认保持旧结果可复现：`--ddp_logit_scale_mode clip|none`、`--ddp_prompt_norm_mode legacy|prompted`、`--ddp_similarity_aggregation pooled_cls/patch_mean/patch_max/cls_plus_patch_max/topk_mean`、`--ddp_similarity_topk`、`--ddp_train_text_prompts`、`--ddp_train_visual_prompts`、`--ddp_prompt_polarity`。
- 2026-06-23 margin-first 排查新增 `--ddp_text_init random|same|semantic`、`--ddp_positive_text_template`、`--ddp_negative_text_template`；`semantic` 会用正/负模板前缀初始化 learnable text prompt，默认 `random` 保持旧实验可复现。
- detail report 与 DDP diagnostics 新增正/负样本拆分后的 `margin`、`ddp_prob`、`margin_gap` 和过阈值比例，用于直接判断 positive/negative prompt margin 是否真正拉开。

## 当前已知方法表现

### EMOTIC 纯 test 评估修正

- 已确认历史实验 `emotic_b5c3_alphabetical_valtest_clip_vit_b16_patch_v1` 的评估集是
  `val+test`，不是纯 test；其日志配置为 `store_model=False`，本地和服务器均未找到
  对应 checkpoint，因此这组 baseline 若要得到纯 test 指标必须重新训练。
- 新增 `--emotic_eval_splits val test|test`，默认仍为 `val test`，保证旧命令可复现；
  `--emotic_eval_splits test` 时训练集不变，只把每个 task 的评估样本限制为 test split。
- 数据构建入口现在会实际传递 `--emotic_input_mode` 和 `--emotic_eval_splits`。
- 新增 `run_emotic_b5c3_clip_vit_b16_patch_test_only_30ep.sh`：复现历史 B5-C3、
  30 epochs/task、batch256、concat head、seed0 配置，仅将评估改成纯 test，并保存
  checkpoint；现有引擎继续生成同格式 log、逐类逐任务 HTML 和 JSON。
- 历史 baseline 的权威代码起点是 `feature/clip-vit-b16@0b59138`。为避免混入 DDP
  改动，服务器使用独立 worktree `/mnt/haoyuan/workspace/multi-lane-main-test-only` 和
  分支 `fix/emotic-test-only-clip-vit-b16` 运行纯 test 重训。
- 对已经保存 checkpoint 的实验，无需重新训练；使用相同训练参数加 `--eval`、
  `--eval_checkpoint` 和 `--emotic_eval_splits test` 即可重新生成纯 test 报告。
- 数据核验：val `2397`、test `5368`、val+test `7765` 人物样本；三者均保持 26 类
  字母序，B5-C3 task sizes 为 `[5,3,3,3,3,3,3,3]`。
- B5-C3 纯 test 重训已完成：final `mAP/amAP/oF1/cF1 =
  29.9174/36.6938/45.5877/19.5187`，总耗时 `36:01`，服务器已打印 checkpoint
  保存信息；待核验并同步 HTML/JSON 后与历史 val+test 逐类对比。
- 历史 joint-training upper bound 同样没有 checkpoint，已在 GPU0 启动纯 test 重训：
  `dataset=EMOTIC,num_tasks=1,base_classes=26,epochs=30,batch_size=256`，其余训练协议
  与旧实验一致，输出名为 `emotic_upper_bound_test_clip_vit_b16_patch`。

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
- DDP 核心提交 `0f0db5f Add CLIP DDP prompt head` 和语义 prompt/诊断提交
  `19417f1 Add DDP semantic prompt diagnostics` 已推送至
  `origin/feature/clip-taskCLS-posneg-text-head`；尚未合并到 `main`。
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
- 当前 margin 已实现“方向性分离”但没有实现“大尺度分离”：full semantic 20ep final
  `pos_margin_mean=0.0731`、`neg_margin_mean=-0.0521`、gap `0.1252`，
  正样本 margin 为正的比例 `93.4%`，负样本 margin 错误为正的比例仍有 `15.0%`；
  PCD task5 `tau=3` 后 raw probability 均值仅为 `0.5061/0.4957`。此外 random 3ep
  的 gap 更大但 mAP 更低，证明仅扩大 mean gap 不能直接提高 AP。
- 训练调度存在已确认的实现差异：配置中的 `warmup_epochs`、`warmup_lr` 和 `min_lr`
  当前没有接入实际 scheduler；`engine.py` 只创建
  `CosineAnnealingLR(optimizer, T_max=args.epochs)`，并且每个 task 无条件重建 optimizer。
  下一阶段应先与 DDP 论文/官方实现核对这些训练协议，再增加论文外 margin loss。
- 2026-06-24 已取得并核对 OpenReview supplementary 中的官方 DDP 代码。此前“严格
  DDP 应使用 raw margin 训练”的推断需要纠正：官方实现先计算每个 token 的 cosine，
  用 positive branch 的 `softmax(20 * similarity)` 作为共享 token attention，再对
  positive/negative 分支加权汇总并乘 `5`，最终 pair-logit 尺度等效为固定 `100`。
- 已新增可选官方路径：
  - `ddp_similarity_aggregation=paper_attention`；
  - train/eval scale 分离，`paper` 为固定 `100`；
  - `ddp_loss_mode=paper_sum`，等价于官方 pair-softmax BCE 求和后乘 `0.03`；
  - `ddp_optimizer_scope=continual`、`ddp_optimizer_lr=5.9e-3` 和
    `ddp_scheduler_mode=paper_multistep`；
  - `ddp_train_transform=paper`，使用 square resize、Cutout、RandAugment 和 CLIP normalize。
- 官方补充代码使用一个持续 Adam optimizer 和 `[0,20]` MultiStepLR；在当前 PyTorch
  中 scheduler 初始化后实际 LR 为 `5.9e-4`，全局 epoch 20 后为 `5.9e-5`。本项目额外
  清理非当前类别的 Adam state，以严格满足论文“旧类 prompts 冻结”的方法描述，避免
  官方代码中整张 prompt tensor 可能受到历史动量影响。
- 新官方路径已通过服务器 CPU 静态检查、attention/loss 公式等价检查、完整
  forward/backward、当前类梯度 mask、持续 optimizer 旧类不漂移和 scheduler 检查。
  `randaugment==1.0.2` 已加入 requirements，并已安装到服务器 `multilane` 环境。
- 首次启动 official random 3ep 时，`randaugment==1.0.2` 因 NumPy 1.26 删除
  `np.int` 在 DataLoader worker 中退出。`datasets.py` 已增加兼容映射并在服务器实际
  NumPy 1.26.4 环境通过完整 paper transform 测试。
- 三组 official-path 3ep 实验已完成：
  - semantic control + paper attention：`mAP/amAP/OF1/CF1 =
    84.43/90.59/74.54/70.92`；
  - official random：`86.05/91.96/59.05/52.00`；
  - official semantic：`88.03/93.32/37.94/28.77`。
- `paper_attention` 单独替换 pooled CLS 后，较旧 semantic 3ep 提升
  `+0.56 mAP/+4.28 OF1/+0.28 CF1`，证明官方 token attention 有效。
- 完整 official random 进一步将 mAP 提升至 `86.05`，margin gap 从 attention control
  的 `0.104` 提升至 `0.148`。official semantic 达到当前最高 3ep mAP `88.03`，
  距论文 B0-C4 last mAP `90.2` 仅 `2.17`，average mAP 距离 `1.48`。
- official semantic 相对旧 semantic 3ep 的主要 AP 增益包括：
  `chair +26.66`、`car +12.57`、`tvmonitor +11.00`、`pottedplant +9.87`、
  `diningtable +7.02`、`sofa +5.67`。当前主要剩余弱类为 `bottle 50.47` 和
  `chair 61.61`。
- official random/semantic 的固定阈值 F1 低不是排序崩溃，而是 B0-C4 下
  `tau_max=7,gamma=0.2` 过度压缩 confidence。official semantic threshold 0.8
  precision/recall 为 `0.985/0.235`，但同一 score dump 在 threshold 0.6 的
  micro-F1 已达 `0.796`，接近论文 OF1 `0.808`。下一步应基于服务器 checkpoint
  eval-only 复评估 `tau2/tau3`，不重新训练。
- 2026-06-24 已新增两个 official semantic 优化版运行脚本：
  `run_voc_ddp_official_semantic_tau2_3ep.sh` 和
  `run_voc_ddp_official_semantic_tau2_20ep.sh`。两者保持 official semantic recipe
  不变，只将 PCD 改为 `tau_max=2.0,gamma=0.7`，用于先恢复 fixed-threshold 0.8 下
  的 recall/F1，再观察 mAP 是否保持。脚本默认写入
  `./logs/voc_ddp_official_semantic_tau2_g07_*.log` 与
  `./output/voc_ddp_official_semantic_tau2_g07_*`，并开启 `--store_model`、
  `--ddp_score_dump true`。
- VOC B0-C4 official semantic tau2 3ep 已同步并分析：final
  `mAP/amAP/OF1/CF1 = 88.03/93.32/79.44/78.69`，pred/support `8420/7632`，
  micro P/R/F1 `0.757/0.835/0.794`。该结果与 tau7 official semantic 的 mAP 完全一致，
  但 fixed-threshold F1 从 `37.94/28.77` 恢复到 `79.44/78.69`，验证问题主要是
  PCD calibration。当前 3ep 结果已接近论文 VOC B0-C4 OF1 `80.8`，CF1 仍约低 `5` 点。
- VOC B0-C4 official semantic tau2 20ep 已同步并分析：final
  `mAP/amAP/OF1/CF1 = 88.52/93.41/72.78/76.36`。相对同配置 3ep，
  mAP/amAP 提升 `+0.50/+0.09`，raw margin gap 从 `0.1250` 提升到 `0.1411`，
  但 threshold 0.8 下 precision/recall 变为 `0.634/0.854`，出现过预测，OF1 下降
  `6.67` 点。20ep 的 global oracle 位于 threshold `0.9`，F1 `75.84`，说明 tau2
  适合 3ep，但对训练后更大的 margin 偏小；下一步应优先用现有 checkpoint
  eval-only 对照 tau `2.5/3/4`，无需重训。
- 当前 20ep 排名指标距论文 B0-C4 `last mAP 90.2 / average mAP 94.8` 分别为
  `-1.68/-1.39`；CF1 `76.36` 距论文 `76.9` 仅 `-0.54`，OF1 `72.78` 距论文
  `80.8` 仍为 `-8.02`。主要弱类仍是 `bottle 61.96`、`chair 61.45`、
  `sofa 77.13`、`pottedplant 79.00`、`diningtable 82.03`。跨任务下降尤其集中在
  `bottle -29.97 AP`、`chair -14.96`、`bicycle -10.53`、`cow -9.03`，
  表明剩余 mAP 差距还包含旧类在新任务数据分布上的泛化/校准退化。
- 2026-06-24 已将产生当前最佳 B0-C4 20ep tau2 结果的 paper-aligned DDP 实现、
  可复现实验脚本和结果记录提交并推送到
  `origin/feature/clip-taskCLS-posneg-text-head`，主提交为
  `5ac25a8 Align CLIP DDP with official training recipe`。日志、output 和 checkpoint
  未纳入 Git；无关的 `train_c100.sh` 末尾换行变化仍保留在服务器工作区。
- 2026-06-24 从父分支提交 `01c10fc` 创建
  `feature/emotic-ddp-semantic-tau2`，开始将当前最佳 official semantic tau2 20ep
  配置迁移到 Split-EMOTIC B5-C3。EMOTIC semantic prompt 前缀为：
  positive `a photo of a person clearly feeling`，negative
  `a photo of a person not feeling`；具体情感类别名由 DDP 文本编码路径追加。
- EMOTIC 正式脚本为 `run_emotic_ddp_official_semantic_tau2_20ep.sh`，使用
  `num_tasks=8,base_classes=5,epochs=20,batch_size=8`、full image 输入、
  paper attention/loss/optimizer/scheduler/transform、PCD tau2/gamma0.7，并开启
  checkpoint、score dump、class order、逐类 HTML 和 JSON detail report。
- 服务器迁移验证已通过：EMOTIC train/eval 人物样本数为 `16001/7765`，类别数为 26，
  task sizes 为 `[5,3,3,3,3,3,3,3]`；text/visual prompt shapes 分别为
  `(26,2,16,512)` 和 `(26,2,16,768)`。情感正负 prompt seed 不相同，detail report
  smoke 同时生成了同名 HTML/JSON。
- EMOTIC B5-C3 1ep real-data smoke 已在 GPU0 并行启动，tmux 为
  `ddp_emotic_tau2_smoke_gpu0`。smoke 使用完整 official semantic tau2 配置，仅将
  epochs 改为 1；启动后 GPU0 总显存约 `9.4/24.6GB`，本实验进程约 `5.2GB`，
  task0 首个 batch 已正常完成，无 OOM。
- EMOTIC 正式 30ep 配置新增脚本
  `run_emotic_ddp_official_semantic_tau2_30ep.sh`。它与历史
  `emotic_b5c3_alphabetical_valtest_clip_taskCLS_text_bias_lr001563_v1` 对齐
  Split-EMOTIC、字母序 B5-C3、full image、val+test、seed0、224 输入和每 task
  30 epochs。为对齐参考实验 batch256 且控制 DDP 显存，使用 physical batch8 与
  accumulation32，effective batch 为 256；LR/scheduler/loss 保留当前最佳 official
  DDP recipe。
- 修正了 `train_one_epoch` 的 gradient accumulation 边界：从原先第 0 个
  micro-batch 错误 step，改为每满 N 个或 epoch 最后一个 batch 才 step。普通 mean
  loss 按 N 缩放，official DDP `paper_sum` 保留求和梯度，因此 batch8×32 与一次
  batch256 的 summed loss 对齐。accumulation=1 的已有实验行为不变。
- EMOTIC B5-C3 30ep 正式实验已在 GPU0 启动，tmux 为
  `ddp_emotic_tau2_30ep_gpu0`。运行参数确认 accumulation32、pin_mem false 生效；
  task0 epoch1 已正常推进，无 OOM/NaN。启动观察时 GPU0 三个实验合计约
  `15.6/24.6GB`，本正式实验显存峰值约 `4.3GB`。
- EMOTIC B5-C3 official semantic tau2 30ep 已完成并同步：final
  `mAP/amAP/OF1/CF1 = 31.32/37.83/25.93/10.16`。相对历史 taskCLS-text-bias
  为 `+0.13/+0.61/-31.29/-12.59`，相对普通 CLIP patch baseline 为
  `-1.54/-2.05/-21.14/-10.10`。DDP 在多数任务上略高于 taskCLS-text-bias 的
  mAP，但没有超过普通 patch baseline。
- EMOTIC 最终 tau2/threshold0.8 precision/recall 为 `0.754/0.157`，明显过保守；
  同一模型 threshold0.5 的 micro-F1 为 `0.518`，per-class oracle F1 为 `0.364`，
  说明类别级 calibration 问题严重，但校准仍不足以完全追平 taskCLS-bias OF1。
- EMOTIC raw margin 分离远弱于 VOC：final positive/negative margin mean 为
  `0.0075/-0.0210`，gap `0.0285`；仅 `64.6%` 正样本 margin 为正，负样本误为正
  比例 `20.3%`。26 类中有 17 类 positive margin mean 不大于 0，`Sensitivity`
  的 class-wise margin gap 甚至为负。当前主要问题是稀有/细粒度情感判别和类别不平衡，
  不只是 PCD 阈值。
- 运行任何训练前必须先向用户说明完整实验配置并等待确认。
