# 下一步任务

最后一次更新：2026-09-03。

## 当前最高优先级：实现统计修复并启动固定权重validation归因对照

- 用户已授权执行前两步骤；新分支`exp/emotic-fusion-numerics-ensemble-control`保留上轮未提交结果记录。
- 当前实现：schema2保存实际评估probabilities与batch元数据；schema1按原eval_batch_size重建；
  fixed test/validation离线anchor都严格核验全部指标与逐类AP，不增大容差。
- 新对照固定0.8/0.2 probability/threshold0.5，比较full_s+full_next与full_s+person_next，
  next为(1,2,0)，另报告full_s+person_s。循环配对复用模型，统计只做描述，不当作独立重复显著性检验。
- 待提交同步后完成完整单测、已有score数值回归、真实data/score smoke与GPU smoke；通过后
  GPU0/1运行seed1 full/person，GPU2/3运行seed2 full/person的完整8-task validation；复用seed0。
- 配置不变：30 epochs/task、batch64/main LR0.0125、cosine/min0/no warmup、Image-token
  layer1/b32/LR4e-4/scale0.1/ReLU/independent、BCE+Adapter ASL9.8/0/0.05、AMP/TF32 on。
- 新launcher不会搜索alpha、读取test或自动启动test；完成后生成固定对照ensemble_comparison.json。

## 已完成的三种子正式结果与本轮来源

1. 锁定probability融合full0.80/person0.20已完成正式test：final mAP`33.2672`，超过同批full
   `32.5365`共`0.7307`。8个task mAP/F1均提高，当前不需要重跑seed0。
2. 结果、test scores、日志与固定融合摘要已同步本地，完整性和哈希核验通过；详见
   `output/emotic_track_a_dual_view_formal_test/20260903_145816/analysis.md`。
3. 用户已要求固定现有结构、训练与融合规则补seed1/2配对确认，不再使用test调alpha/mode/threshold。
   已核查171份config：旧full seed1/2没有逐样本scores/checkpoint，不能复用融合；person尚无结果。
   新分支`exp/emotic-dual-view-seed12`参数化seed并编排GPU0/1的seed1 full/person及GPU2/3的seed2。
   实现`d2442ee`已推送并建立独立clean工作树；75项完整单测和GPU smoke通过，四个训练进程
   已在batch`dual_view_locked_formal_seed12_20260903_174938`启动，config除seed/Git外与seed0完全一致。
4. 论文结论区分seed0最佳与三种子均值，计入双模型双视图计算成本。若做方法归因，考虑后续等计算量
   full+full集成对照，不能把双视图收益全部归因于Adapter或letterbox。
5. 四组训练现已完整成功；原自动融合因sigmoid全表/batch64的float32差异触发anchor严格校验而失败。
   已通过原batch64精确重建指标并保持锁定融合路径不变，离线完成seed0/1/2汇总，未放宽容差或重训。
   70个结果文件已同步并校验；汇总位于该batch本地`analysis_recovery/formal_seed_summary.json`。
6. 当前正式报告：融合`32.8263 ± 0.4025`，full`32.1562 ± 0.3701`，配对提高
   `+0.6701 ± 0.0670`；三个seed全部8-task mAP均改善。forgetting平均`+0.0519`，仍略差。
7. 下一次开发先修复通用离线anchor重建并加浮点并列分数回归测试；未来score dump保存实际probabilities。
   本次仅产物分析恢复，业务源码尚未修复，不应原样重跑launcher或覆盖历史失败记录。
8. 下一轮实验建议固定0.8/0.2在validation比较等预算full+full和full+person；可复用seed0两视图val，
   补seed1/2两视图四组val，配对辅助seed一致以控制集成随机性。随后按框质量/遮挡/人物大小诊断，
   再考虑轻量person-guided Selector。以上仅计划，尚未开始，不继续扩容Adapter或在test调参。
9. final validation mAP优先，average/task5--7其次；类别AP/F1/forgetting仅作为解释指标，不恢复
   用户已经撤销的逐类硬否决门槛。注意双模型计算成本及仅三种子的统计局限。

## 历史记录：阶段二ASL局部8组正式test（已完成）

1. layer1已以正式seed0 final mAP`32.5365`超过layer8旧冠军`32.4193`，差值`+0.1172`；
   average mAP、cF1、oF1和forgetting也全部改善，8个task mAP均未下降。当前最优配置只把
   Adapter zero-based层索引从8改为1，其余保持原冠军参数。
2. 首批15组已经全部完成且skipped0。layer1局部网格最高仅b32/LR3e-4的`32.1281`；
   single7 FP32为`31.7931`，均未超过当前冠军。
3. 归一化多层最高为`[1,2]` b16/scale0.05的`32.3233`，低冠军`0.2132`；已有`[1,2]`
   b32/scale0.05仅`31.7764`，说明该组合扩大容量明显不利。
4. 追加5组每层b32上界正式test已全部完成并同步。最高`[1,4]` b32/s0.05仅`32.2213`；
   其余为`31.6235--32.0605`，没有配置超过`32.5365`。
5. 20组统一结论已经满足停止条件：固定single1 b32/LR4e-4/scale0.1，停止所有多层、
   bottleneck和Adapter LR扩张。下一步只能转ASL局部微调或稳定性复核。
6. 首批15组已经独立同步并分析，结果包SHA-256为
   `b6eb43e335ec0031f36d022ac701ff80aba0f78c29be27af393e906221169cbb`；无需再次下载。待追加5组
   已另包同步，SHA-256为`72995ac436e75d0f9d51eff3da2c0c1c0823cbc8051b47c4b13d3b7e8eb37698`。
7. 阶段一`image_token_asl_layer1_lr_scale_formal_seed0_20260901_171236`已完成并同步。同批
   LR4e-4/scale0.1精确复现`32.5365`并保持第一；次优LR5e-4/scale0.075为`32.3851`，
   LR6e-4不占优，因此不补7e-4。
8. 阶段二`image_token_asl_layer1_asl_local_formal_seed0_20260901_182542`已启动：固定layer1/
   b32/LR4e-4/scale0.1，运行ASL锚点、gamma-neg8/12、clip0.0375/0.075及gamma-pos
   0.25/0.5/1.0共8组正式test，GPU0--7各一组、无checkpoint。
9. 阶段二只按完整运行后的final mAP判断是否超过`32.5365`；若gamma-pos边界1.0成为冠军，
   再考虑向1.5扩展，否则ASL局部搜索收口。average mAP、F1和类别AP只用于解释。

## 已完成：容量/LR局部正式搜索

1. 用户已将目标改为尽快最大化正式test final mAP，允许个别类别下降。按final/average
   validation mAP选出b16/LR4e-4和b64/LR4e-4两个候选；原类别/F1/forgetting
   硬门槛仅保留为观察指标。
2. 两组seed0完整8-task held-out test已于2026-08-31 00:35在GPU0/1并行启动并完成，batch为
   `image_token_asl_capacity_formal_seed0_20260831_003546`。两组固定layer8、LR4e-4、scale0.1、
   ReLU、independent、主模型BCE + Adapter ASL 9.8/0/0.05、FP32且TF32 off。
3. b16/b64均完成240 epochs、13950 updates、skipped0，无OOM、NaN或写盘异常。final mAP
   分别为`32.0269/31.3193`，均未超过历史b32冠军`32.4193`；b16明显优于b64。
4. 结果包已同步本地并通过SHA-256校验，不含checkpoint。历史冠军的数值模式为
   `AMP=true, TF32=true`，新两组为`AMP=false, TF32=false`，容量与精度模式同时改变。
5. 用户已确认同时并行执行两阶段。b16/LR4e-4的AMP/TF32-on正式test与以b32/LR4e-4为
   基线的scale`{0.025,0.05,0.1,0.2}` × activation`{ReLU,GELU}` 8组validation已一次启动。
6. 新批次`20260831_114209`已完成并同步。b16 AMP/TF32-on正式test final mAP为
   `31.3796`，低于历史b32冠军`1.0397`；不再保留b16为正式候选。
7. 8组scale×activation中5组完整成功；`0.1/GELU`、`0.2/ReLU`、`0.2/GELU`在task3因
   非有限logits失败。有效结果中原`0.1/ReLU`在final、average和task5--7平均mAP三项均第一，
   final mAP为`42.6673`；停止scale和activation搜索。
8. 用户考虑validation与test耗时接近后，明确授权跳过validation，直接将bottleneck
   `{24,28,32,40}` × Adapter LR`{3e-4,4e-4}` 8组放入seed0完整held-out test。
9. 批次`image_token_asl_local_capacity_lr_amp_formal_seed0_20260831_124751`已全部完成并同步；
   8组均240 epochs、13950 updates、skipped0，无OOM、NaN或写盘异常。
10. 没有新冠军：同批b32/LR4e-4精确复现`32.4193`并仍排名第一。最接近的b24/LR3e-4
    为`32.2492`，低`0.1702`；它的average mAP高`0.2668`，但task6类别组均AP低`3.5030`。
11. 统一容量/LR搜索已基本收口。若层位置正式test无新冠军，下一步二选一：保持
    b32/LR4e-4进入ASL局部微调；或实现task-dependent Adapter，task0--5候选b24/LR3e-4、
    task6--7保留b32/LR4e-4。
12. 已经用户批准删除8个明确失败/中止实验checkpoint，释放约6.48 GiB；无进一步
   删除授权。若需使非root `df` 恢复可用空间，必须另行列出已完成实验的可归档
   checkpoint并再次获得用户批准。

## 已完成：ASL gamma 搜索收口

1. FP32稳定版13组已全部完成并严格核验，同commit、240 epochs、13950 updates、skipped0且无
   数值错误。结果没有eligible winner：12个ASL的final cF1全部下降、forgetting全部恶化；
   `gn9.8/clip0.05`虽final mAP提高`0.425109`，但Sadness下降`2.088062`、cF1下降
   `0.635873`。按规则停止gamma-neg/clip扩边界，不进入gamma-pos，不启动test。
2. 当前稳定loss基线保留FP32 Image-token joint-BCE；结构搜索仍保留同层Adapter-BCE对照，
   并将9.8/0/0.05 Adapter-ASL作为主要候选，但不会放宽Sadness/Suffering门槛。

## 已完成：Image-token Adapter × ASL seed0 正式对照

1. 原Image-token BCE与`model_asl/adapter_asl/both_asl`四组均已完成；每组240 epochs、
   13950 updates、skipped0、exit code0，运行配置clean且除loss路由外完全对齐。
2. 四组final mAP为`31.676786/30.462064/32.419329/32.199410`。当前探索性最优是
   `model BCE + Adapter ASL`，相对Image-token BCE提高`0.742543`；task6/7分别提高
   `0.735797/0.742543`，task6 Sadness与Suffering同时提高`7.524682/8.239046`。
3. 不采用主模型ASL：`model_asl/both_asl`的final oRecall接近100%，oF1相对BCE下降
   `20.004362/19.356518`，表明gamma-neg9.8导致固定阈值校准坍塌。
4. 四组24个JSON/日志小文件已无checkpoint打包并同步到本地；压缩包SHA-256为
   `60cb61ebe7f64d639f351821834c95539bad149ad0abe6e1d3f4796a2d8df884`，详细分析见本地
   `output/emotic_track_a_image_token_loss_routing_formal/image_token_bce_asl_seed0_results_20260813/comparison_analysis.md`。
5. 当前没有活动的本轮实验，不启动新训练。由于本轮只有seed0且已直接观察held-out test，
   不继续基于该test调gamma、阈值、层数或容量。若要确认Adapter-ASL，应另行预注册未用于本轮
   选择的validation/独立复现实验，再决定是否补seed，而不是把本次`+0.7425`当作显著结论。

## 已收口：Task-lane Transformer Adapter 阶段 0

0. `exp/emotic-adapter-position-diagnostics` 的实现提交 `5766860` 已推送并同步；服务器20/20
   单测及GPU2/3/4 zero-based layer5/8/11 Adapter smoke均通过。loss二选一已完成：current-only
   的task6/7 mAP下降`0.757291/0.645109`，task6新类均值下降`4.452897`，Suffering下降
   `13.552041`；第二阶段已固定legacy并在GPU2/3/4/7启动normalization `none/clip` × crop
   minimum `0.05/0.50`的2×2 seed0完整8-task val-only对照已完成。crop0.50在8个task均退化；
   clip+crop0.05使task6/final mAP提升`0.134184/0.211756`、task6三类均值提升`0.484693`，
   因此第三阶段已固定`legacy_full_zero + clip normalization + crop(0.05,1.0)`并在GPU2/3/4
   启动的independent layer5/8/11均已完成。三层task6新类均值分别下降
   `0.525884/1.296133/8.770622`，Suffering分别下降`0.528539/2.105999/14.799475`；没有一层
   通过四项条件。按预定规则停止task-lane Adapter路线，不进行多层或容量扩张；下一方法方向
   需在用户确认新假设后另开分支，不再用当前正式test反复调参。
   每一步均只用seed0完整8-task validation，不能并行跳过前置选择。

   当前保留基线：Adapter disabled、`legacy_full_zero`、CLIP normalization、crop
   `(0.05,1.0)`。当前没有活动诊断实验；下一步不是继续调Adapter层数/容量，而是先提出能直接
   改善后期新类获取、稀有类排序或类别不平衡的新机制，再经用户确认后另建分支。

1. 阶段 0 主实现 `531f3f3` 已提交推送，本地和服务器主工作树已同步；test-only worktree
   未触碰。
2. 服务器 8/8 单元测试、原基线 GPU smoke 和 Adapter GPU smoke 均已通过；Adapter
   bottleneck64 单层每任务参数为 `99136`，当前总可训练参数为 `788314`。
3. 用户确认完整配置后，seed0 task0 val screen 已在 GPU 1/2/3/4 启动：layer11、ReLU、
   scale0.1，组合 bottleneck `32/64` 与 Adapter LR `1e-4/4e-4`。四组均已完成，b64/lr4e-4
   以 `59.697275` 胜出，相对 seed0 基线提升 `+2.358121`；不读取 test。
4. b64/lr4e-4 task0 seed1/2 在停止信号前已正常完成，val mAP 为
   `59.329355/56.178599`，均高于各自基线；按老师意见不再增加 task0 多 seed 确认。
5. b128/lr4e-4 seed0 val mAP `56.824802`，比 b64 低 `2.872474`，已淘汰；正式配置固定为
   b64/lr4e-4。
6. 完整 8-task seed0/1/2 正式实验已在 GPU1/2/3 启动，run ID
   `multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927`。三组均已完成并通过
   240 epochs/8 tasks/skipped0/exit code0 检查；五项均值为
   `31.128461/31.994054/48.878881/38.599579/4.869595`。
7. 当前 Adapter 相对基线的五项变化为
   `-0.171025/+0.182944/-0.230322/+0.601003/+0.081138`。下一步先分析逐 task/逐类曲线，
   重点解释 average mAP 提升但 final mAP 和 forgetting 变差；在找到后期退化原因前不合并
   main，也不盲目扩大 Adapter。
8. 无 checkpoint 的17文件包已下载并解压到本地；正式三种子结果与诊断已写入上下文。
9. RNG隔离与 `copy_previous` Adapter warm-start 已由提交 `313618c` 推送并同步；默认
   `independent` 保留上一轮可复现性。服务器10/10单元测试与GPU1 Adapter smoke通过，
   smoke可训练参数 `788314`、初始logits最大差 `9.1552734375e-05`；未启动训练。
10. seed0完整8-task val-only warm-start已在GPU1完成：b64/lr4e-4、layer11/ReLU/scale0.1，
    run ID为 `task_lane_adapter_full_val_seed0_b64_lr0.0004_copy_previous_20260811_174404`。
    GPU2 clean disabled配对对照 `multi_lane_disabled_full_val_seed0_paired_clean_20260811_174847`
    同样完成。task6/7 mAP下降 `0.334822/0.448594`，task6 Sadness/Suffering AP下降
    `8.788894/10.703310`，三项继续条件全部失败；停止继续堆叠task-lane Adapter容量，不跑
    新的test调参。分析脚本提交 `3d7c5aa` 已同步且服务器13/13测试通过。下一步应转向
    非task-lane容量扩张方向，先由用户确认新的方法假设。

## 已完成：当前仓库 Track-A 三种子复现

1. 已完成复现实现提交推送、服务器同步、4/4 单元测试、checkpoint/输出零误差核验和
   GPU smoke；test-only worktree 保持原状。
2. 正式三种子实验、结果汇总、无 checkpoint 归档、SSH 下载和本地解压均已完成；五项
   mean/std 与注册结果完全一致。
3. 如需进一步审计，检查本地 `output/emotic_track_a/<run_id>/synced_files/` 中的逐 task
   指标、逐 epoch 历史和四个日志；服务器 checkpoint 继续保留，不同步到本地。
4. 后续方法改动已进入独立 Adapter 实验分支，当前复现分支保持冻结。

## Git 迁移后续

- 本地和服务器主工作树已共同跟踪 `origin/fix/independent-git-worktrees`，当前基线为
  `feature/clip-vit-b16@0b59138`。
- 迁移文档和 `.gitignore` 已纳入 `fix/independent-git-worktrees`；后续两端同步该分支时
  继续坚持本地 push、服务器 `git pull --ff-only` 的流程。
- PyCharm Automatic Upload 保持关闭；确认部署配置永久排除 `.git/` 后，SFTP 也仅用于
  数据、日志、输出和 checkpoint 等未纳入 Git 的产物。
- test-only worktree 保持未提交状态，后续应在其现有分支中单独审查并保护两处代码修改、
  两个运行脚本和数据集符号链接，不与主工作树迁移提交混合。
- 如需恢复迁移前 DDP 或 test-only 工作，优先从安全分支、命名 stash 或备份 patch 创建
  专用恢复分支；不要在当前工作树直接 `stash pop`，也不要使用硬重置。

## 最高优先级

0. 验证并运行 EMOTIC B5-C3 CLIP patch baseline 的纯 test 重训。

   - 服务器额外 worktree 分支：`fix/emotic-test-only-clip-vit-b16`；主工作树当前为
     `fix/independent-git-worktrees`，不要混用两者的未提交内容。
   - 入口：`run_emotic_b5c3_clip_vit_b16_patch_test_only_30ep.sh`。
   - 保持历史训练配置，只将 eval split 从 `val+test` 改为 `test`，并保存 checkpoint。
   - 启动前检查 GPU 状态；不得停止或改动服务器上其他用户的实验。
   - 完成后同步 log 和 output 下的 detail HTML/JSON；checkpoint 较大，可留在服务器。
   - 对已有 checkpoint 的其他模型，使用 eval-only 纯 test，不重复训练。
   - 已在独立 worktree 分支 `fix/emotic-test-only-clip-vit-b16` 的 GPU1/tmux
     `emotic_clip_b16_testonly_30ep` 启动；等待 8 tasks 全部完成后核对 checkpoint、
     HTML/JSON，并与历史 val+test 指标逐类比较。
   - B5-C3 已完成；下一步核验/同步产物并分析纯 test 与 val+test 的指标差异。
   - joint-training upper bound 纯 test 已在 GPU0/tmux `emotic_upper_testonly_gpu0`
     启动；完成后同样核验 checkpoint、HTML/JSON，并计算新的 incremental-to-upper-bound
     gap。

1. 在 `feature/emotic-ddp-semantic-tau2` 验证并运行 EMOTIC B5-C3 迁移实验。

   - parser、数据集类别数/任务划分、情感 prompt、静态编译和模型/报告 smoke 已完成。
   - 正式配置：Split-EMOTIC、full image、B5-C3、20ep、batch8、
     official semantic paper recipe、PCD tau2/gamma0.7。
   - 正式入口：`run_emotic_ddp_official_semantic_tau2_20ep.sh`。
   - 预期产物：训练日志、checkpoint、8 个 task 的 score dump、class order JSON、
     per-class/task HTML 和 JSON。
   - 1ep real-data smoke 已通过 `ddp_emotic_tau2_smoke_gpu0` 在 GPU0 启动；
     当前无 OOM，完成后检查 NaN、8-task 流程和产物完整性。
   - smoke 通过后，再向用户汇报完整配置并确认是否启动正式 20ep 训练。
   - 用户已确认将正式协议改为与历史 EMOTIC 对照一致的 30 epochs/task；正式脚本为
     `run_emotic_ddp_official_semantic_tau2_30ep.sh`，计划在 GPU0 的
     `ddp_emotic_tau2_30ep_gpu0` 运行。batch 使用 8×32 accumulation 等效 256。
   - 正式 30ep 已启动并稳定进入 task0；后续等待 1ep smoke 和正式实验完成，
     同步 log/output 后分析 mAP、amAP、OF1、CF1、margin 与逐类指标。
   - smoke 和正式 30ep 已完成。下一步先用服务器 30ep checkpoint 做 eval-only
     PCD sweep：`pcd off/tau1.0/tau1.25/tau1.5/tau2.0`，固定 threshold0.8；
     同时保留 threshold0.5 作为诊断。该步骤只改善 calibration，不改变 mAP。
   - ranking/margin 的下一轮短对照优先级：
     1. 每 task 重置 paper scheduler/LR，验证后续情感 prompts 是否因持续 scheduler
        在 task1 后降到 `5.9e-5` 而欠拟合；
     2. 保持 full image 主流，增加 full+person-crop 双视图/特征融合诊断；已有纯
        person-crop baseline mAP `30.75`，不支持直接改成 crop-only；
     3. 针对 rare classes 对照 class-balanced loss 或 positive reweighting，这属于
        EMOTIC extension，需与严格 DDP recipe 分开报告。
   - 重点观察 class-wise margin 近零或反向的类别：
     `Embarrassment/Sensitivity/Disquietment/Doubt/Surprise/Esteem/Yearning`。

1. 当前已完成 margin-first 代码改动，下一步应先验证这些改动是否真的扩大正负 margin。

   已实现：

   - 新增 `--ddp_text_init random|same|semantic`，默认 `random` 保持旧实验可复现。
   - 新增 `--ddp_positive_text_template` 与 `--ddp_negative_text_template`，`semantic` 初始化时用模板前缀分别 seed positive/negative learnable text prompts。
   - 新增 per-class 和 overall margin 分解诊断：正样本 margin 分布、负样本 margin 分布、`margin_gap_mean`、正/负样本 raw `ddp_prob` 大于 0.5/0.6 的比例。

   下一步实验优先级：

   - 先用 VOC B0-C4 跑 3 epoch：`random` control vs `semantic` text init。已完成。
   - 两组都使用当前最佳结构：`pooled_cls + prompted + clip + CLIP normalize + tau3/gamma0.7`。
   - 结果：`semantic` final mAP/OF1/CF1 为 `83.87/70.26/70.64`，优于 `random` 的 `82.34/66.73/67.89`。
   - 但 `semantic` 没有提高 `margin_gap_mean`；final gap 从 `random=0.1688` 降到 `semantic=0.1356`，两组 `pos_ddp_prob_gt_0_6` / `neg_ddp_prob_gt_0_6` 都为 0。
   - 因此 semantic init 暂时作为更好的主配置候选，但它没有解决 raw margin 尺度不足；下一步继续做 text-only/visual-only branch ablation。已完成。

   branch ablation 结果：

   - `semantic 20ep`: `84.27/70.54/72.57`，当前最高 mAP。
   - `text_frozen 3ep`: `82.32/71.16/72.77`，当前最高 OF1/CF1，说明固定 semantic text anchors、主要训练 visual prompts 是下一条重点路线。
   - `visual_frozen 3ep`: `82.47/68.77/69.78`，低于 text_frozen，说明 visual prompts 对 fixed-threshold F1 更关键。
   - `positive-only 3ep`: `81.25/70.99/71.58`，更偏 precision。
   - `negative-only 3ep`: `81.38/68.41/72.41`，更容易过预测。

   最新验证：

   - `semantic text frozen 20ep` 已完成，final `80.84/68.22/71.30`，较 3ep 全面退化；
     predicted positives 墠到 `10475`，主要问题是 precision 下降和过预测。
   - `semantic positive-only 20ep` 已完成，final `83.06/70.09/72.61`；
     mAP/CF1 较 3ep 提升，但 OF1 下降，且没有超过 full semantic 20ep 主线。
   - seed1 复验显示 full semantic 3ep 高度稳定，text-frozen 3ep 的高 F1 有明显随机性。

   下一步优先级更新：

   - full semantic text+visual prompts 恢复为默认主线。
   - 先用已有 full semantic/text-frozen/positive-only 20ep checkpoint 做 eval-only
     threshold/PCD sweep，确认 calibration 可恢复的上限。
   - 做训练调度短对照：优先比较 `3/5/10/20 epoch` 或降低 prompt LR，重点监控
     predicted positives、precision、recall 和 `bicycle/chair/bottle/car`。
   - 对照论文官方实现核对 optimizer、实际 prompt LR、scheduler/warmup、visual prompt
     初始化、text prompt 是否更新，以及论文约 `0.49M` 参数统计口径。
   - 已确认当前实现没有实际使用 `warmup_epochs/warmup_lr/min_lr`，只使用裸
     `CosineAnnealingLR(T_max=epochs)`；下一轮修改前应先查明论文官方调度，并补成
     可切换配置，保留当前行为用于 control。
   - mAP 优化以 class-wise ranking 为目标，优先观察
     `chair/bottle/tvmonitor/pottedplant/diningtable/sofa/car` 的 AP；
     threshold/PCD sweep 主要改善 F1/calibration，不能作为提升 mAP 的主手段。
   - 如果严格论文配置核对后 raw margin 仍不足，再考虑 margin regularizer；
     该项必须标记为 DDP extension。

   官方 supplementary 核对后的修正：

   - 官方实现没有 warmup，也不是 raw margin train；它使用
     `paper_attention + fixed scale 100 + summed binary-softmax BCE * 0.03`。
   - 官方 optimizer/scheduler 路径已实现为可选配置：
     `continual Adam(lr=5.9e-3) + MultiStepLR([0,20], gamma=0.1)`。
   - 官方数据增强路径已实现；服务器正式运行前需安装
     `randaugment==1.0.2`。
   - `randaugment==1.0.2` 已安装；NumPy 1.26 的 `np.int` 兼容问题已修复并验证。

   下一轮三组短实验应按单变量逻辑组织：

   - A：当前 full semantic control，只把 aggregation 改成 `paper_attention`，
     判断官方 token attention 是否直接提高困难类别 AP。
   - B：完整官方 recipe，使用 random class-specific text prompts、
     paper loss/optimizer/scheduler/transform 和官方 PCD。
   - C：完整官方 recipe + semantic text init，判断 semantic 初始化是否仍能在
     官方 aggregation 下提供额外收益。

   三组首先跑 3 epoch 做结构筛选，重点比较 last/average mAP 以及
   `chair/bottle/tvmonitor/pottedplant/diningtable/sofa/car` 的 AP；选出候选后再跑
   20 epoch。raw train 只作为后续诊断，不进入论文复现主表。

   三组已完成，当前结论：

   - `official semantic 3ep` 是明确主候选：last/average mAP `88.03/93.32`。
   - 按用户当前要求，优先直接跑优化版 official semantic：
     `run_voc_ddp_official_semantic_tau2_3ep.sh`，随后立刻运行
     `run_voc_ddp_official_semantic_tau2_20ep.sh`。
   - 该优化版保留 official semantic recipe，只把 PCD 改为
     `tau_max=2.0,gamma=0.7`，保持主指标 threshold 0.8。
   - 根据 score dump，tau2 最接近当前 threshold0.6 的 oracle 决策边界；预计主要恢复
     recall，同时尽量保持 official semantic 的高 mAP。
   - 如果 3ep 出现明显过预测，再补 `tau2.5/gamma0.7` 或 `tau3/gamma0.7`。
   - tau2 3ep 已完成，final `mAP/amAP/OF1/CF1=88.03/93.32/79.44/78.69`；
     主阈值 0.8 已是 global oracle threshold，说明当前无需先补 tau2.5/tau3。
   - tau2 20ep 已完成，final `mAP/amAP/OF1/CF1=88.52/93.41/72.78/76.36`。
     ranking 较 3ep 小幅提高，但 tau2 在 20ep 出现过预测；global oracle threshold
     移到 `0.9`。
   - 下一步第一优先级不是重训，而是基于 20ep checkpoint 做 eval-only
     `tau_max=2.5/3/4, gamma=0.7` 对照，固定 threshold 0.8，选择 precision/recall
     更平衡的 PCD。该实验只改变校准，不会改变 mAP。
   - mAP 优化随后聚焦跨任务退化最严重的 `bottle/chair/bicycle/cow`。先从 score dump
     分解这些类在后续任务中的正负样本 margin 漂移，再决定是否需要类别级 PCD、
     current-task hard-negative 采样或 replay；后两项属于 DDP extension，不能混入
     严格论文复现主结果。
   - 为严格论文对照，仍需单独完成 official random-init 20ep B0-C4；semantic init
     是当前增强版本，不能代替官方默认初始化的复现结果。
   - 当前最佳 tau2 20ep 实现及脚本已通过提交 `5ac25a8` 推送到远端功能分支；
     后续实验应以该提交为基线，避免混入服务器工作区的无关修改。

1. 增加严格 DDP probability / score calibration 诊断，先回答当前 gap 是结构问题还是评估/校准问题。

   背景：

   - `pooled_cls + prompted + raw cosine` 3 epoch final mAP 为 `81.71`，但在项目主阈值 `sigmoid(logits)>0.8` 下 OF1/CF1 为 0。
   - 这说明排序能力和固定阈值 F1 已明显解耦，继续只看 scaled logits 的 F1 不足以判断 DDP 结构是否正确。

   已实现：

   - `clip_ddp` forward/diagnostics 保留 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`ddp_prob=sigmoid(margin/tau)`、`scaled_logit=logit_scale*margin/tau`。
   - 新增 `--ddp_eval_score_mode logits|probability`，其中 probability 模式直接对 `ddp_prob` 阈值化，不再额外 sigmoid。
   - 新增 dense threshold sweep `0.1 ... 0.9`，并输出 global oracle threshold 和 per-class oracle F1 诊断。
   - 新增 `--ddp_score_dump true`，保存每个 task 的 `y_true/eval_score/s_pos/s_neg/margin/ddp_prob/scaled_logit/class_ids`。
   - 新增 VOC task class order / seen class order 日志，核对论文 lexicographic protocol。
   - 新增 `--clip_normalize_input true`，使用 CLIP mean/std，并可与当前 no-normalize/ImageNet normalize 对照。

   待运行：

   - 服务器静态检查和 parser smoke。已完成。
   - 用已有 checkpoint 做 eval-only：同一模型分别跑 `logits` 与 `probability` score mode。已完成。
   - 开启 score dump，分析 oracle threshold 下 F1 是否接近论文。已完成初步分析：strict probability 不改善 F1，raw margin 尺度过小。
   - CLIP normalize 3 epoch 对照已完成，结果显著优于 no-normalize。

   当前下一步：

   - 跑 VOC B0-C4 20 epoch 主候选：`pooled_cls + prompted + clip + CLIP normalize + tau3/gamma0.7`，保存 checkpoint。已完成，final mAP/OF1/CF1 为 `82.95/67.32/68.90`。
   - 当前最高优先级改为 margin-first 排查：先跑短实验比较 `ddp_text_init=random` 与 `semantic`，同时用新 margin summary report 读取 per-class positive/negative margin gap。
   - 增加 margin summary report：per-class positive/negative margin 分布、margin gap、raw probability 分布和过阈值比例。已完成代码实现，等待服务器检查和短实验验证。
   - 然后做 text/visual/polarity prompt ablation，判断 margin 拉不开来自 text prompt、visual prompt 还是 negative branch。
   - semantic positive/negative text initialization 已完成代码实现；如果仍不足，再考虑 margin regularizer，并明确标为 DDP extension。

2. 使用已补齐的 DDP 诊断/复评估能力，先完成 checkpoint 复评估再启动下一批长实验。

   目标：让同一个 checkpoint 反复评估不同 PCD 参数、aggregation 和阈值诊断，避免每改一个 `tau_max/gamma` 都重新训练 20 epoch。

   当前实现进度：

   - 已增加 DDP eval diagnostic 输出：每个 task 记录 `predicted_positive/support`、micro precision、micro recall、score/logit 的均值、分位数、正负样本分布。
   - 已让 detail report 增加 score 分布和 overall diagnostic 字段，方便判断是排序问题、校准问题还是阈值问题。
   - 已增加 checkpoint 复评估入口：读取 `--store_model` 保存的 `checkpoints.pth`，只跑 validation/test，不重新训练。
   - 已保持论文主指标使用项目默认 `sigmoid(logits) > 0.8`，诊断模式额外打印不同阈值下的 precision/recall/F1 曲线；诊断结果不作为论文对齐主表。
   - 后续所有 20 epoch 论文协议实验默认加 `--store_model`，保证训练完成后可重评估 PCD。

   已验证：

   - 已完成服务器 `multilane` 环境静态检查。
   - 已用 VOC B0-C4 3 epoch + checkpoint 复评估验证 `--eval --eval_checkpoint`。

3. 先做 DDP 结构级修正/核对，再继续长实验。

   最高优先级修改顺序：

   已完成：

   - 已增加 `ddp_logit_scale_mode`，支持 `none` 与 `clip`，把严格 DDP 的 raw cosine binary-softmax logit 和当前项目阈值兼容的 scaled logit 分开记录。
   - 已增加 visual prompt strict pre-norm 开关 `ddp_prompt_norm_mode=prompted`：先拼接 `[P; x]`，再对完整序列做 `block.norm1` 和 attention，attention 输出后切掉 prompt tokens。
   - 已增加更贴近论文抽象的 CLIP pooled visual feature 命名路径 `ddp_similarity_aggregation=pooled_cls`，即用 prompted CLS/pooled representation 与 text feature 直接 cosine。
   - 已将 `patch_mean`、`patch_max`、`cls_plus_patch_max`、`topk_mean` 加为实现级 visual evidence aggregation 诊断项，不再把它们表述为论文明确要求。
   - 已增加 prompt 训练分支开关：`ddp_train_text_prompts`、`ddp_train_visual_prompts`、`ddp_prompt_polarity`，用于解释当前 `0.819M` learnable params 与论文 VOC `0.49M` 统计不一致的问题。

   已完成结构对照：

   - `pooled_cls + prompted + raw cosine` 3 epoch final mAP/OF1/CF1：`81.71/0.00/0.00`。排序能力较好，但 raw cosine 在项目固定 `sigmoid(logits)>0.8` 主指标下预测正类为 0。
   - `pooled_cls + prompted + CLIP logit_scale` 3 epoch final mAP/OF1/CF1：`79.86/61.55/62.97`。
   - `cls + legacy + CLIP logit_scale` control 3 epoch final mAP/OF1/CF1：`77.10/61.55/64.16`。
   - 结论：`pooled_cls + prompted + clip` 是新的 20 epoch 候选；`raw cosine` 只作为论文公式/排序诊断，不作为当前固定阈值主配置。

   待做：

   - 启动 VOC B0-C4 20 epoch：`pooled_cls + prompted + clip + tau3/gamma0.7`，并保存 checkpoint。
   - 完成后用 eval-only 复评估 `tau2.5/tau3/tau3.5` 和必要的 threshold sweep。

4. 基于 VOC B0-C4 20 epoch 最佳 checkpoint 做 eval-only 校准。

   当前最佳 checkpoint：

   ```text
   ./output/voc_ddp_b0c4_cls_tau3_g07_20ep/checkpoints.pth
   ```

   当前最佳结果：

   ```text
   VOC B0-C4, cls aggregation, tau3/gamma0.7, 20 epoch
   final mAP 79.7877, amAP 87.3884, oF1 65.7037, cF1 64.7468
   pred/support 8568/7632, micro P/R/F1 0.621/0.697/0.657
   ```

   下一步先不重训，直接复评估同一个 checkpoint：

   - `ddp_pcd=false`，确认 20 epoch 后关闭 PCD 是否仍误报过多。
   - `ddp_tau_max=2.0, ddp_gamma=0.7`，看能否追回一些 recall 且不明显牺牲 precision。
   - `ddp_tau_max=2.5, ddp_gamma=0.7`，在 tau2 与 tau3 中间找固定阈值下的最优点。
   - 必要时再评估 `ddp_tau_max=3.5, ddp_gamma=0.7`，验证 tau3 附近是否已经过了峰值。

   已完成背景：

   - `tau_max=3.0,gamma=0.7` 3 epoch final：mAP 76.1613，oF1 64.2071，cF1 62.1797。
   - 同 checkpoint `ddp_pcd=false` final：mAP 76.1609，oF1 55.9951，cF1 59.0601。
   - 同 checkpoint PCD sweep：tau2/gamma0.7 final oF1/cF1 62.98/63.15，tau5/gamma0.5 final 46.40/45.95，tau7/gamma0.2 final 20.34/23.77。
   - 3 epoch aggregation 对照显示 `cls` 优于 `mean/max`；20 epoch `cls + tau3/gamma0.7` 已成为当前 B0-C4 最佳。

   当前结论：

   - `tau_max=3.0,gamma=0.7` 目前仍是主配置。
   - 20 epoch 后模型比 3 epoch 更保守，tau2/tau2.5 可能改善 recall，需要用 eval-only 快速验证。
   - `tau_max=5.0,gamma=0.5` 和 `tau_max=7.0,gamma=0.2` 明显偏保守，暂停使用。

5. 复评估和结构修正后启动 VOC B10-C2 论文协议实验。

   ```text
   dataset: Split-VOC
   protocol: B10-C2, num_tasks=6, base_classes=10
   epochs: 20
   aggregation: pooled_cls
   prompt_norm_mode: prompted
   logit_scale_mode: clip
   PCD: 先使用 B0-C4 复评估后最优 tau/gamma；若无更优则用 tau3/gamma0.7
   store_model: true
   ```

   如果 GPU0 显存 OOM，按用户要求等待，不继续调整或触碰其他人的任务。

6. 评估 visual similarity aggregation。

   当前默认 `ddp_similarity_aggregation=mean`，可能稀释 class-specific visual signal。已按同一 PCD 配置比较：

   - `mean`
   - `max`
   - `cls`

   结果：

   - `mean` final mAP/oF1/cF1：76.16/64.21/62.18。
   - `max` final mAP/oF1/cF1：71.66/59.63/59.11。
   - `cls` final mAP/oF1/cF1：77.77/64.05/63.88。

   已完成：

   - 已完成 `cls` checkpoint 的 PCD 复评估：tau2/gamma0.7、tau5/gamma0.5、tau7/gamma0.2。
   - 固定阈值 0.8 下，`cls + tau3/gamma0.7` 当前最好。
   - 已用 `cls + tau3/gamma0.7` 跑完 VOC B0-C4 20 epoch，并开启 `--store_model`。

   后续如继续拉近论文差距，再考虑结构级改动：

   - 对照 DDP 原文检查 text prompt 初始化、positive/negative class name 模板和 class token 位置。
   - 检查 visual prompt 输出聚合是否应使用原始 CLS、prompt-conditioned CLS 或 token pooling 的特定组合。
   - 分析旧类冻结后不同 task 的 prompt score drift，尤其 chair/diningtable/sofa/pottedplant 等类别。

7. 在真实 EMOTIC batch 上做 smoke 前，向用户确认完整实验配置。

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

8. EMOTIC smoke 通过后，再决定是否推送当前 DDP v1/PCD 本地提交。

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

当前状态：已于 2026-06-24 将 `19417f1 Add DDP semantic prompt diagnostics`
推送至 `origin/feature/clip-taskCLS-posneg-text-head`。后续仍需完成论文实现细节核对和
VOC 对照实验，不应在现阶段合并到 `main`。

## 2026-09-01 当前Image-token Adapter下一步

1. 阶段二ASL局部8组正式test已完成；原9.8/0/0.05锚点以final mAP`32.5365`保持第一，
   且8个task mAP逐项第一。停止gamma-neg、gamma-pos和clip扩展。
2. 下一批将阶段三和阶段四合并并行：阶段三比较uniform b32与task0 b24/b28、其余task b32
   的按任务bottleneck；阶段四在uniform b32上联合比较主模型实际LR
   `{0.010,0.0125,0.015}`与Adapter专属weight decay`{0,1e-5,1e-4}`。
3. 两阶段共用原配置作为同批锚点，去重后共11组；固定layer1、Adapter LR4e-4、scale0.1、
   ReLU、ASL 9.8/0/0.05、AMP/TF32 on、seed0完整8-task test、无checkpoint。
4. 实现前需新建实验分支，增加per-task bottleneck和Adapter-only weight decay CLI、单元测试、
   GPU smoke。提交推送与服务器Git同步仍需用户明确确认；当前没有启动阶段三/四。
5. 用户现已确认运行阶段三/四；实现和11组launcher已经完成静态检查。下一步提交推送实验分支，
   服务器fetch并新建/切换独立实验worktree，运行完整单元测试和代表性task0-b24 GPU smoke。
6. 测试通过后启动batch `image_token_asl_layer1_stage34_formal_seed0_<timestamp>`。启动核验必须确认
   11份config逐字段正确、GPU0--2双进程仍有安全余量、所有组epoch1 skipped0且无非有限值。
7. 上述检查已完成，正式batch为`..._20260901_203100`，当前11组均正常运行。完成后按final mAP
   排名，并分别回答：task0小容量是否超过uniform b32；主模型LR与Adapter weight decay是否存在
   稳定交互；只有超过`32.5365`的配置才成为新冠军。
8. 11组已完成并同步；没有超过`32.5365`。停止task0 b24/b28。下一批若继续，固定其他结构，
   运行实际主模型LR`{0.014,0.0145,0.015,0.0155}`×Adapter WD`{3e-6,1e-5}`八组，仍以同批
   `0.0125/WD0`作锚点；只有final mAP超过锚点才升级冠军。
9. 用户纠正未知未来任务约束后，取消task-dependent LR建议。下一阶段改为9组全局统一参数
    直接test：LR`{0.014,0.0145,0.015,0.0155}`×WD`{3e-6,1e-5}`加原冠军锚点。
10. 该批完成后只选择统一全局配置。随后才进入固定optimizer update预算实验；任何动态规则
    必须对所有task使用同一预定义公式，不能按已知task编号或test难度手调。
11. 9组launcher已在服务器安全等待GPU释放，当前没有训练进程。自动启动后需核对9份config、
    GPU0双进程余量、epoch1的84 steps/skipped0以及错误日志；等待期间不要重复启动第二批。
12. 8组已完整结束并同步；第9组LR0.0155/WD1e-5因外部显存占用失败后已在GPU1从头安全补跑。
    等补跑产生完整`seed_summary.json`后，补同步该run及日志，验证240 epochs、13950 updates、
    skipped0，再生成9组最终排名与正式小结果包。
13. 当前临时结论是锚点`32.5365`仍领先，LR0.015/WD1e-5以`32.5023`接近且average mAP/F1更好。
    在第9组完成前不启动固定更新预算或一致性正则实验，避免并行阶段混淆结论。
14. 第9组已完成，final mAP`31.8789`，9组均未超过`32.5365`；全局LR×Adapter WD搜索停止。
15. 下一阶段固定冠军LR0.0125/WD0以及既有Image-token Adapter结构，新增按optimizer updates终止和
    按updates调度cosine的入口。使用同一预声明预算作用于全部task，不读取未来task难度。
16. 计划8组为原30-epoch control，加每task updates`900/1200/1500/1800/2100/2400/2700`。
    实现后先补单元测试和GPU smoke；按照当前用户要求，正式对比直接运行seed0完整8-task test，
    继续标记exploratory。只有final mAP超过`32.5365`才升级冠军。
17. 用户要求三个机制阶段并行一次跑完。实现已完成：阶段一8组；阶段二residual/cosine各
    `1%/3%/10%`共6组；阶段三learnable gate init`0.025/0.05/0.1/0.2`共4组，总计18组。
18. 下一步提交推送`exp/emotic-image-token-training-mechanisms`，服务器审计全部worktree后创建新的
    clean独立worktree；Automatic Upload写入primary的副本必须逐文件备份并恢复，不触碰test-only。
19. 在服务器`ddp`环境运行完整单元测试，再分别运行control、10% residual regularization、
    learnable gate0.1三种真实CLIP AMP/TF32-on GPU smoke。任何梯度非有限、旧gate变化或update预算
    不精确都必须先修复，不能启动正式实验。
20. 测试通过后启动18组batch。启动核验包括18份config全部记录clean同一commit、control精确保持
    epochs模式、7个预算值正确、6个正则配置含30-update calibration、4个gate配置各多8个总参数，
    每卡2--3进程仍有安全显存、首轮无OOM/non-finite/skipped。
21. 服务器50项单元测试和四种真实CLIP GPU smoke已全部通过。下一步将本次验证上下文commit推送并让
    独立实验worktree`git pull --ff-only`，然后启动batch
    `image_token_asl_layer1_training_mechanisms_formal_seed0_20260902_150731`。
22. 18组已全部安全启动且配置/显存/首轮数值核验通过。当前只需等待完成；不要重复启动、不修改运行
    worktree。结束后验证18份complete summary、各预算组总updates=`目标×8`、epoch组13950 updates、
    skipped0和无错误，再打包同步小结果。
23. 分析时分阶段排名：阶段一先比较final mAP并以average mAP破平；阶段二分别判断residual/cosine；
    阶段三同时报告final mAP和8个learned gate终值。若不同阶段各有赢家，再只补一组赢家组合确认。
24. 18组已完成并同步；三阶段均未超过`32.5365`，因此取消组合确认。固定updates、Adapter输出正则、
    learnable gate三条路线停止，继续保留原30-epoch/fixed-scale0.1/no-reg冠军。
25. 若继续优化，优先使用对所有task统一的exposure-based epoch/scheduler或仅依据当前task validation的
    同一early-stop规则，不按task编号或未来难度手调。正式论文结论前必须锁定配置并做paired 3 seeds；
    已查看test的所有seed0结果仍标记exploratory。
26. 用户已确认先执行统一epoch搜索。新分支`exp/emotic-image-token-epoch-search`实现8组
    `18/22/26/30/34/38/42/48` epochs一组一卡launcher、固定冠军单组runner和严格自动汇总器。
27. 提交前运行Python编译、两个shell语法、epoch汇总器单测与完整现有测试；提交推送后必须审计服务器
    所有worktree，并使用新的clean独立worktree，不能直接污染primary或test-only。
28. 服务器测试通过后启动8组batch。核对每组config仅epochs不同、8份clean同commit/tree、AMP/TF32
    on、no-checkpoint；首轮应分别形成对应run目录且task0 skipped0，无OOM/non-finite错误。
29. 完成后只按final test mAP排名、average mAP破平。内部赢家补上下3个整数epoch；48赢家补54/60；
    30保持第一则停止epoch调参并转scheduler。所有本批结果继续标记exploratory test-tuning。
30. 实现`7449b30`已推送，服务器独立worktree clean；53项完整单元测试和代表性真实CLIP
    Adapter-ASL AMP/TF32 smoke均通过。下一步提交推送本次验证上下文、服务器ff-only更新后立即启动
    8组正式batch，并核验8份config、首轮steps/skipped与显存。
31. 正式batch`..._20260902_171919`已一组一卡启动，8份clean config和task0首轮随机轨迹/84 steps/
    skipped0已核验，显存与错误扫描正常。当前只等待完成，不重复启动；结束后运行严格汇总器，并只
    同步JSON、manifest、launcher status与日志的小结果包，不同步checkpoint。
32. 8组已全部完成并同步。30 epochs以final mAP`32.5365`保持第一；26 epochs虽把average提高
    `0.0495`，但final低`0.1684`且task5--7均下降。34以上训练loss继续下降而test全面退化，停止
    epoch搜索，不补相邻整数epoch。
33. 下一阶段固定30 epochs和其余冠军参数，只实现对所有task相同的scheduler网格；仍以同批30-epoch
    current cosine为锚点。scheduler若不能稳定超过`32.5365`，结束单视图参数搜索并开始
    full image + person crop双视图低参数融合诊断。
34. 阶段B代码已实现：8组为cosine anchor、相对min1%/10%、warmup5%/10%、linear、constant和
    multistep18/26 gamma0.1；主模型和Adapter共享相对LR曲线，历史anchor路径保持原样。
35. 下一步提交推送`exp/emotic-image-token-scheduler-search`，服务器审计worktree并创建新的clean
    scheduler工作树；运行完整单元测试和scheduler LR轨迹/GPU smoke，通过后8卡一组一卡启动。
36. 启动核验必须检查8份config仅scheduler字段不同、共同clean commit、30 epochs/13950 updates目标、
    task0首轮LR符合预期且skipped0。完成后按final mAP/average mAP排名，不因单类变化否决。
37. 实现`58eb233`已推送，服务器scheduler工作树clean；60项完整测试、8种LR轨迹审计和真实CLIP
    Adapter-ASL AMP/TF32 smoke均通过。下一步提交本次验证记录、服务器ff-only同步后立即8卡启动，
    并核验8份config及各自首轮/关键epoch LR。
38. 正式batch`..._20260902_185259`已一组一卡启动；8份clean config、首轮LR/loss/84 steps/
    skipped0和显存均核验通过。当前只等待完成，不重复启动；结束后严格核验240 epochs、13950 updates、
    全LR轨迹和错误日志，再打包同步JSON/日志且不含checkpoint。
39. 8组已全部完成并同步。原cosine以`32.5365`保持第一；multistep虽把average提高`0.4074`，但
    task6/7回落使final低`0.2196`。其余scheduler final低`0.2867--0.9827`，阶段B无冠军。
40. 固定原30-epoch per-task cosine并关闭单视图scheduler路线。下一阶段开始full image + person crop
    双视图低参数融合；第一轮必须含full冠军锚点、同协议person-only、固定logit融合、归一化feature
    融合和全局标量gate，并保持所有task使用同一规则。
41. `exp/emotic-person-only-layer1`已实现person-only第一步：bbox margin0.15、person训练crop
    scale0.70--1.0，其他参数严格复现layer1冠军，正式test且不保存checkpoint。`d0f8e81`已推送，
    服务器63项单测、真实数据检查和GPU smoke均通过；run`..._20260903_121738`已在GPU0稳定启动。
42. 等待person-only完成后同步config/task_metrics/training_history/seed_summary和日志，不同步checkpoint；
    对比full冠军的完整8-task、逐类AP及错误互补性。只有person分支与full确有互补，才实现validation
    score dump和全局标量logit/probability融合，不直接从person-only单项高低推断融合是否有效。
43. person-only已完成：final mAP`28.7815`，低full冠军`3.7550`；8个task与最终26类全部负增益，
    当前版本停止。若继续双视图，下一步先实现body-preserving pad-to-square/letterbox、稳定sample ID和
    validation score dump；同批运行full anchor与校正person分支，validation融合不超过full则结束路线。
44. `exp/emotic-dual-view-validation-fusion`已实现letterbox person transform、稳定ID、validation NPZ
    score dump、全局logit/probability alpha融合器和两卡启动脚本。用户已确认直接commit/push；下一步
    在服务器新独立worktree运行完整单测、真实数据/score dump smoke和GPU smoke，全部通过后GPU0/1
    并行运行full/person完整8-task validation并自动离线融合，禁止读取test。
45. 自2026-09-03起，用户已明确要求开始的实验，其实验分支commit/push、服务器Git-only独立worktree
    同步、单测/smoke和已声明的非破坏性实验启动不再重复审批；main合并、删除数据、覆盖未提交改动或
    触碰test-only工作树仍必须单独确认。
46. 双视图validation已完成并同步：两组均240 epochs、13,950 updates、skipped0，score dump完整且
    对齐，无checkpoint/错误。probability alpha0.20的final mAP`43.3035`比full `42.3799`高`0.9236`，
    并在8个task上全部提高，因此通过进入正式test的预声明门槛。
47. 已完成锁定配置的seed0完整8-task held-out test：full/person训练配置不变，只应用probability
    alpha0.20和threshold0.5。final mAP`33.2672`超过旧单视图冠军`32.5365`；不在test继续搜索参数。
48. 固定test实现`c9fa74c`已推送；服务器新clean工作树的72项单测、固定选择SHA/无搜索CLI检查及
    真实Adapter GPU smoke全部通过。
49. 正式batch`image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816`已自动完成；
    等待器正常退出，所有GPU现空闲。两组和固定融合产物齐全，不重跑、不再启动第二批seed0。
50. 两份config均为clean`c9fa74c`、reporting=test、score purpose固定、无checkpoint；240 epochs/
    13,950 updates/skipped0核验通过。固定规则复算与`fixed_test_fusion_summary.json`完全一致。
44. Image-token冠军seed1/2已完成并与既有seed0汇总：final mAP `32.1562 ± 0.3701`，相对注册
    baseline平均提高`0.8567`且三个配对seed均为正。结果已同步、哈希核对并打包；本配置到此锁定，
    不再依据seed1/2调参。论文报告时同时给出三种子均值/标准差，并说明此前seed0 test探索的边界。
