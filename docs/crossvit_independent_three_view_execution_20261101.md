# 独立三视图 CrossViT 式 MULTI-LANE：执行协议

2026-11-01用户授权修改并在GPU0高效运行完整8任务Validation／Test。本文件为实现与执行记录；当前先准备专项测试和真实预检，未报告任何正式精度。

## 执行更新：正式五组队列已经启动

运行源码`1e2125efaa12187364ef9b8e7d56e3d71447792b`，本地／远程新worktree branch、HEAD、upstream一致，远程clean。tmux `crossvit_three_view_seed1_20261101`。14项专项全部通过无skip，包含真实CUDA AMP；41项历史回归通过，总55项。五组真实task0/1各4更新，共40更新／0skip，10项Val首64完成；配对审计确认公共初始化、所有记录增强batch一致，旧参数／冻结骨干／固定TRAIN旧类预测受保护。无预检Test。

预检训练峰值：I_W/I_M各3524.6MiB；C_W/C_M各4204.9MiB；P_M4190.8MiB。CA参数594432／任务、Pool596352／任务，差约0.32%。正式队列据此设最多3并发；截至启动核验，I_W PID188483、I_M189152、C_W189764运行，C_M/P_M排队。前两组已完成task0首epoch各84更新／0skip；C_W首epoch同为84更新／0skip。GPU0约13460MiB、利用率100%；其余GPU未操作。后续Val/Test及task0/2/6/7 ON/OFF自动排队。

72份预检非权重产物已同步本地`output/crossvit_independent_three_view/preflight_snapshot_20261101_01/snapshot`，归档SHA256 `379ec3aad4b46214d050182684de0358563a5adbcb21e5040da4606e96cb3595`，核验通过并保存逐文件清单。主树／test-only状态SHA与启动前一致。正式训练无最终mAP，本地本段和三上下文为启动记录，不更新正在运行的源码或配置。

研究方案：`crossvit_independent_three_view_plan_20261101.md`。源码分支`codex/crossvit-independent-three-view-20261101`。不修改历史工作树、原共享模型的model.py／runner.py，不安装来源A/B/B0任务参数。三份冻结CLIP与三套当前任务ASP/head独立，index5本地块后3个接收模块同步用Task CLS读取对方Frozen Forward空间patch，继续后续Task Forward。

## 配置和分组

五组`I_W/I_M/C_W/C_M/P_M`：独立分支无交互的历史权重／均值控制；CLS-Attention的历史权重／均值；相同证据输入的Pool-MLP均值控制。各组公共初始三分支参数逐值匹配，新模块独立RNG；非共享训练参数无跨视图存储别名。

EMOTIC原26类8任务、seed1、每任务30epoch、batch64、Adam每任务重置／WD0／cosine，主LR0.0125、Adapter／交互LR0.0004；b32/index1/scale0.03/ReLU、10 Selectors、原5层Prefix、Person颜色扰动off、原aux BCE0.1与Adapter-ASL(neg9.8,pos0,clip0.05)、legacy_full_zero、AMP／TF32／65536 scaler、72线程历史初始化／训练流。第6层块后CA内部64／4头，无额外FFN或1%后置限制，输出零初始化。

可靠Face的Q接收模块仅对可靠行运行；不可靠Face在其他模块的softmax之前屏蔽，aux和最终fusion同步mask。整batch不可靠时将Face当前参数grad置None，避免Adam已有momentum在零梯度下继续改变Face参数。

## 路径与入口

远程新独立工作树拟`/mnt/haoyuan/workspace/multi-lane-main-crossvit-independent-20261101`，只通过Git fetch/worktree部署。

- 配方来源：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train/config.json`，只读取配置和SHA，不加载任务参数。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- 标准预训练：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- Face清单：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 总入口：`bash scripts/emotic/run_crossvit_three_view_gpu0.sh`。
- 训练／评估：`python -m multi_lane.track_a.crossvit_experiment train|eval`。
- 队列：`python -m multi_lane.track_a.run_crossvit_three_view`。
- 输出：`output/crossvit_independent_three_view/seed1_20261101_01/{I_W,I_M,C_W,C_M,P_M}/{train,val,test}`。
- 日志：`logs/crossvit_independent_three_view/seed1_20261101_01`。

入口先运行专项unittest，再顺序完成五组task0/1各4更新＋Val首64预检；只在同源码预检完整通过后自动启动五组正式8任务，不需要再次询问。预检不读取Test。

正式并发上限由4090实测24GB和预检各组峰值决定，最多5进程；按峰值×1.3＋768MiB进行显存准入，保留1536MiB余量，只用GPU0，最多2个评估进程。训练优先，完成一组自动排Val/Test；资源不足自动后排。评估在72线程初始化后降为2线程，2workers／prefetch1；开关归因在task0/2/6/7同一batch内复用冻结三流，避免重复读取和骨干计算。

## 保护、恢复与产物

每epoch原子保存任务参数＋optimizer／scheduler／scaler／完整RNG／训练历史／增强batch轨迹，可从最新完成epoch恢复。任务边界保存前先核对旧参数和冻结ViT SHA、固定TRAIN旧类预测，并按历史规则重放两次CPU reporting seed抽样。所有旧任务交互和视图嵌入整体冻结。

正式两split每阶段输出融合与Full/Person/Face、可靠Face指标、逐类AP／cF1／oF1及logits／probabilities／IDs／mask。历史samples_intersect_seen_classes，8阶段mAP算术平均；开关为条件贡献，不作为重训基线。所有组结果合成summary.csv并审计公共初始化与增强batch匹配。

原32.6800／39.5292 seed1 B0作为既有强参照，不用新独立模型OFF冒充精确B0。若CA在同权重下不超过独立无交互对照，不能将独立heads或融合权重收益归给Attention。

## 初始安全快照

GPU0启动前仅35MiB、利用率0%；GPU1–7有其他作业，不操作。共享内存4GB。主远程树状态摘要`ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf`；test-only状态摘要`e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152`，HEAD`0b591388da2f9efa2e9157a00c64d1b3d68d5c38`。仅新增独立工作树，禁止清理／覆盖上述工作。

本地语法编译与shell语法检查通过。实际Torch/CUDA专项与真实图片预检待远程执行；不能把语法检查当成模型验证。
