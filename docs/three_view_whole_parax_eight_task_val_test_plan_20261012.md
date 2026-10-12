# seed0：task0后完整冻结内部ParaX的八任务validation与固定test

授权日期2026-10-12：用户明确跳过其他seed，直接完整八任务validation和固定配置test。本方案不混入其他会话Task Forward后的受保护专家。

## 目标与锁定架构

验证task0训练得到的单一共享ParaX能否稳定支持全部26类，重点检查后期新类；旧函数稳定性和新类精度分别判断。Full/Person/Face共享冻结CLIP ViT-B/16；Frozen Forward索引10（第11个block完整输出后、第12个block前）插入ParaX-style模块，patch tokens残差，CLS旁路；rank32、三对低秩专家矩阵、Router hidden16、official初始化、初始output scale0.1且仅task0可学习。task1起完整冻结整个ParaX，含专家、Router、Norm、投影、scale及buffers。仍保留输入相关路由，不添加level条件。原Image-token Adapter关闭。Selector/Prompt/分类参数三视图共享、按任务分配，只更新当前任务。

归一化任务CLS特征固定融合，可靠Face权重.64/.16/.20，不可靠.80/.20/0；可用视图辅助BCE权重.1，joint BCE，无新增正则/蒸馏/专家/动态融合。

## 数据、优化与来源

- EMOTIC：`../multi-lane-main/datasets/EMOTIC`，输入标注CVPR17_Annotations.mat。
- CLIP：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`，沿用原权重SHA256 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`；核对train/val/test和detector hash。
- seed0、8-task原参数分配5/3/3/3/3/3/3/3，从task0重新训练；不从缺少optimizer/RNG的compact checkpoint续训。
- 每task30epochs、train/eval batch64、workers2、OMP1、13950有效updates；Adam每task重置，任务参数初始lr.0125，ParaX task0初始lr.0004；weight decay0、cosine/minlr0/warmup0。
- AMP初始1024，growth interval1e9，沿用TF32；冻结审计使用FP32/TF32-off。全量任务loss与预处理保持之前配置。
- 原baseline/live来源：`../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seed0/{THREE_VIEW,THREE_VIEW_PARAX}`。只恢复已训练checkpoint，不重训。
- pilot来源：`../multi-lane-main-whole-parax-freeze/output/emotic_whole_parax_control/three_view_whole_parax_freeze_seed0_val_20261011_01/frozen`。要求前三任务预测及task0 ParaX状态与pilot对齐。

## 入口、输出和评价

分支`exp/three-view-whole-parax-freeze-eight-task`；本地复用干净的three-view-whole-parax-freeze工作树并创建新分支，远程新独立worktree `../multi-lane-main-whole-parax-eight-task`，Git-only同步。保留主/test-only及其他会话改动。

正式batch：`three_view_whole_parax_eight_task_seed0_val_test_20261012_01`。

```bash
BATCH_ID=three_view_whole_parax_eight_task_seed0_val_test_20261012_01 GPU=0 PYTHON=/opt/conda/envs/ddp/bin/python bash scripts/emotic/launch_whole_parax_eight_task.sh
```

只一个seed0新训练进程，GPU0；不需要占满其他空闲GPU。启动要求空闲>=14000MiB/util<=15%，最多等待12h，不干预他人进程。训练只validation，随后自动审计、完整validation，再评估固定配置test，测试不以validation胜负为条件，不挑checkpoint/epoch/权重。

- `./output/emotic_whole_parax_eight_task/<batch>/frozen`：8个compact checkpoint、training history、原任务末val scores、完整冻结/配对/FP32 anchor审计。
- `.../full_val`：三组已有/新模型每任务全部2397人、5/8/.../26已见类，NPZ、单路/可靠Face指标、新旧类AP、固定人群forgetting、汇总。
- `.../full_test`：三组每任务全部5368人，固定checkpoint对应已见类，同样逐任务产物；评估开始前强制检查完整validation已完成、所有来源hash一致。
- 同时报告原“至少一个已见类”筛选口径的指标，不能与完整人群口径混用。验证已有score重放。每checkpoint先重放来源validation再加载test；新冻结模型没有已有test score，不虚报test重放。
- `./logs/emotic_whole_parax_eight_task/<batch>/`：训练及val/test评估日志。
- control/state、各phase exit_code、manifest、完成时间；失败保留产物，不改参数自动重试。

## 启动前验证

本轮不改变模型/冻结算法，只扩展任务数入口、完整配对审计和独立评价。先compile/bash/diff检查、全量unittest，然后真实ViT八任务各4updates smoke，共32updates，正式参数预算分配；审计晚期task7全bank冻结与旧函数稳定，再跑val preflight三方法task0/task7各64人。不进行smoke test。smoke不是精度实验。

smoke batch `whole_parax_eight_task_smoke_20261012_01`，输出和日志均在相应output/logs子目录。通过后直接启动已授权正式配置；不连续监控，结束后统一同步分析。

## 判读与限制

旧特征/logits不稳定先报实现失败；稳定但晚期新类低于baseline，说明固定task0变换迁移不足；稳定且整体提升才支持此路线。尤其单独列task6 Sadness/Sensitivity/Suffering，但不利用test分组结果搜索超参数。仅seed0不能证明跨seed稳定。此test属于新训练整体冻结模型，不等同旧模型按类别恢复历史状态。无需执行seed1/2。


### 2026-10-12 八任务启动检查通过

实现cc71e77。298/298 unittest、compile/bash/diff通过。真实ViT smoke whole_parax_eight_task_smoke_20261012_01八任务各4updates/共32updates，0AMP skips；task1-7整bank冻结，task7 trainable0，固定旧FP32 logits最大差1.1920928955078125e-6，reserved3342MiB。新增评价入口六次来源val重放（baseline/live/frozen task0/task7）最大logit差全部0，64人val preflight通过，无test加载，不作精度证据。正式batch three_view_whole_parax_eight_task_seed0_val_test_20261012_01待启动，GPU0，8×30epochs/13950updates，val-only训练后自动完整val和固定test；原三个任务pilot作为前三任务函数配对对照，源baseline/live不重训。preflight包SHA256 a31445e0d0d375f9e0cf8e3352362f8769a5ba37490d599025e97bdb7b7a3a29，产物同步主本地logs/output，checkpoint留远端。验证后只同步文档提交再启动，不改变模型代码，不占用其他会话/test-only。


### 2026-10-12 正式八任务整体冻结seed0已启动

GPU0确认state=training、runner PID2923566、tmux ml_whole_parax_eight_task_20261012。正式干净运行HEAD7e1daccfabc3e90b7a114f01c696b5ee199ad46f，分支exp/three-view-whole-parax-freeze-eight-task；远程新worktree /mnt/haoyuan/workspace/multi-lane-main-whole-parax-eight-task。batch three_view_whole_parax_eight_task_seed0_val_test_20261012_01。seed0/8×30epochs/13950updates，task0原配置学习，task1-7整个ParaX冻结；原Image-token Adapter关闭，固定三视图融合。训练val-only，完成后自动完整2397人val三方法比较、固定5368人test三方法比较（复用baseline/live检查点），保存各任务全人群及原筛选口径、新旧类/单路/可靠Face指标和同人群forgetting。强制验证pilot前三任务配对、所有后期整bank状态与FP32旧函数稳定；不挑checkpoint，不按test调参，不训练其他seed。原来源worktree、主/test-only和其他GPU任务未修改。

298项测试/真实32update八任务smoke/六次val replay0通过；preflight产物包SHA256 a31445e0d0d375f9e0cf8e3352362f8769a5ba37490d599025e97bdb7b7a3a29两端一致并已同步本地。正式output/logs按emotic_whole_parax_eight_task/<batch>保存。完整计划docs/three_view_whole_parax_eight_task_val_test_plan_20261012.md。启动后仅提交本地文档记录，不拉取到运行工作树；停止常规监控，等待用户要求结束后统一同步分析。


## 2026-10-12 重放修复验证通过，完整val→test评价队列已启动

新分支fix/whole-parax-evaluation-batch-replay-20261012，代码提交/干净运行HEAD b8660514679338eb32a7efb39336091ddd8bd663。服务器独立worktree /mnt/haoyuan/workspace/multi-lane-main-whole-parax-eval-repair，原训练及主/test-only/其他工作树保留。修复完整分数校验使用原筛选人群和原batch边界，人物/标签/类别/batch/有限值均检查，阈值0.01不变；完整人群输出仍独立计算。真实三方法task0/task7六次首批及完整原人群重放差全部0，确认原baseline task0最大差0.015625源于换batch后的AMP舍入；301/301单测与compile/bash/diff通过，无smoke训练或test加载。

正式batch three_view_whole_parax_eight_task_eval_replayfix_seed0_20261012_01，GPU0，tmux ml_whole_parax_evalfix_20261012，确认state=evaluating_val、主PID3120242、baseline task0完整校验已通过；尚未进入test。仅评价现有seed0三方法各8checkpoint，完整2397人val通过后自动完整5368人test；batch64/workers2/AMP/TF32按原config，固定融合，不训练或选择模型。输出/日志在emotic_whole_parax_eight_task/<batch>的新目录，不覆盖原failed批次。完整配置docs/whole_parax_eight_task_evaluation_repair_plan_20261012.md。启动后不持续监控，结束后同步结果并补导师总览。

预检NPZ/JSON/单测与真实重放日志、正式manifest已同步主本地；包logs/emotic_whole_parax_eight_task/verification/whole_parax_eval_repair_preflight_20261012.tar.gz两端SHA2561472c5a684c60ba4d39dad54815007b3d480f161f4966aa12076d8197a1c0de9。checkpoint留远端。导师报告已注明修复和队列实际状态，43.3248仍是训练末val记录，test等待自动阶段。提交推送获本次用户执行授权，仅实验修复分支；后续文档提交不拉取到正在评价的远程工作树。
