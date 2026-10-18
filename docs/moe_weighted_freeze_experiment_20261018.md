# WAF-v1：累计 gate 权重的冻结排名

用户2026-10-18要求新增带权重激活统计版本，GPU1、seed0、8task validation/test并行。
从AF-v1运行提交`07fd35cf97a2efc416d56348a1e82a2f00572291`创建独立分支
`codex/moe-weighted-freeze-20261018`，保留GPU0硬计数作业及其工作树不变。
仅复制前一分支本会话未提交的文档，原始文件不移走、不覆盖其他用户工作。

## 单一方法改动

AF-v1按硬次数`C_i=sum 1[g_i>0]`冻结；WAF-v1按
`S_i=sum_{当前任务训练前向, 样本, Full/Person/可靠Face} g_i`冻结。
g为训练期实际noisy Top-2后softmax权重，非选中项为0。例：0.999/0.001分别
贡献0.999/0.001，不再都贡献1。每个有效视图样本总gate mass约1（硬次数约2）。

- 仍按视图分别保存硬计数、有效样本数，新增float64累计gate mass及其三视图合计。
  不乘0.64/0.16/0.20视图融合系数，不作三路频率等权归一化，不乘专家范数或梯度。
  这是门控分配量，不是已证明的预测贡献或因果重要性。
- 只统计当前任务训练、梯度启用的实际前向；Face仅可靠样本，invalid placeholder排除。
  eval和旧lane只读探针不得污染硬计数及gate mass；每任务计数窗口独立。
- 全22专家按S_i取Top-2，包括已经冻结的专家；并入累计mask，不补足两名新专家。
  同分按expert ID升序，排名在CPU float64上计算以便独立checkpoint重算。
- 冻结参数/旧Router、输入梯度、dropout语义与AF-v1相同。gate统计detach/no_grad，
  不引入新损失、不参与反向，也不改变冻结边界之前的Router或Expert计算。
- 新配置`--moe-freeze-metric gate_mass`；默认hard_count不增加mass buffer，保持原AF
  checkpoint兼容。evaluator从config恢复统计策略；checkpoint保存训练gate mass。
  验证器核对sum_i mass约等于有效样本数、0<=mass<=hard_count、记录与tensor一致，
  重算加权Top-2/累计mask并检查先前冻结权重及旧Router不变。
- 该加权排名属于本项目新变体，不称为论文原始硬频率冻结的完全复现。

## 锁定配置（开始前已向用户声明）

- EMOTIC，8tasks：5/3/3/3/3/3/3/3，seed0，每任务30epochs，预计13950更新。
- 仅GPU1（RTX4090）；各task顺序训练，训练完成后val/test两进程同GPU1并行。
  固定全人群val2397、test5368，每阶段读取对应共享池checkpoint；test不用于选择策略。
- 冻结CLIP ViT-B/16；视觉Task Forward索引1，MSA/Drop & Replace后与FFN并联；
  raw task CLS驱动task×view独立Router，跨任务/三视图共享22专家，768→64→768，
  noisy Top-2、dropout0.1、up零初始化、residual scale0.1；无v4接口、ParaX、文本塔、DDAS。
- 新增可训练参数仍2,992,000；额外mass只是buffer，8task×1层×3视图×22×8=4224字节。
- Adam每task重建，主LR0.0125（source0.05×64/256）、MoE LR0.0004，weight decay0。
  CosineAnnealingLR，无warmup/min LR；train/eval batch64、workers2。
- legacy_full_zero BCE、joint_bce、auxiliary view BCE0.1、supervised scale1，无负载loss/
  蒸馏/新增正则；AMP scale1024、growth interval1e9、TF32，无梯度裁剪。
- shared Selector/Prompt/head，Selector10、Prompt10、Prompt层5、pre-head归一化。
  Full legacy随机crop0.05–1，Person letterbox/margin0.15，无jitter，CLIP输入归一化，
  Face沿用既有manifest/可靠性mask，固定融合0.64/0.16/0.2或0.8/0.2/0。
- 数据`../multi-lane-main/datasets/EMOTIC`；CLIP权重
  `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 入口`bash scripts/emotic/run_moe_weighted_freeze.sh`；正式run_id
  `formal_seed0_20261018_01`；日志`./logs/moe_weighted_freeze/<run_id>/`；
  输出`./output/moe_weighted_freeze/<run_id>/`。
- 预期产物：config/history、paired/freeze audits、硬计数/gate mass/冻结历史、8 checkpoint、
  full_val/full_test逐阶段及单视图分数、路由统计、verification和退出状态。

## 运行门槛

1. 本地语法/diff及GPU1 launcher的val/test并发与失败处理测试。
2. Git-only服务器独立worktree；全Track-A测试，新增小数排名与硬次数产生不同名单、
   前向/反向等同、可靠Face与eval隔离、加权checkpoint恢复及篡改拒绝测试。
3. GPU1两任务smoke，各2更新；独立checkpoint重放、B0来源核验、冻结权重SHA及gate
   mass守恒检查；对照硬AF既有smoke验证task0参数和训练记录在首次冻结前保持一致。
4. GPU1空闲且>=12GiB后启动正式8task。GPU0代码/进程不动，其他工作树与test-only保留。
5. 运行后固定服务器checkout；正式结束再同步日志/输出分析，以最终和阶段平均mAP、
   旧任务漂移、冻结名单差别及实际专家复用判断，不以名单不同本身判断成功。

当前已实现统计/配置/恢复/核验接线，两个launcher测试及语法/diff检查通过；准备
提交/Git同步、全测、真实smoke。未启动GPU1正式训练。

## 首轮smoke与修复

7d79318全Track-A 330项通过。smoke_seed0_20261018_01完成两任务训练，但在
checkpoint核验阶段退出1：入口model.float()将诊断buffer降为float32。保留失败产物；
正式八任务未启动。新增_apply保护mass原精度和设备迁移，并补dtype转换回归测试，
将用smoke_seed0_20261018_02从头复验。首轮task0与硬AF smoke的145个浮点参数
完全相同；task1加权Top2[4,18]而硬计数[4,20]，仅用于机制核验。

## 实际检查与启动记录

- 修复后运行提交`ceded63a0f58a27729845c32a78ff95f15b53fd8`，本地/服务器/
  upstream一致且启动工作树clean；仅本地文档在启动后追加状态，服务器checkout不动。
- 全Track-A **331项通过**（13.573s）。`smoke_seed0_20261018_02`退出0，
  verification=verified、TWO_TASK_CHECKPOINT_REPLAY_PASSED。每task仅2更新，
  smoke不用于评估方法性能。
- task0有效样本Full/Person/Face=128/128/73，gate mass约128/128/73，
  硬激活256/256/146；task1有效样本128/128/66，gate mass约128/128/66，
  硬激活256/256/132。Face无效占位不进入统计，双精度保存、重算排序一致。
- 两任务各145个浮点参数张量均与原硬AF smoke完全一致；唯一新增state key为
  `adapter_bank.training_gate_mass`。首次冻结名单相同[9,19]，task1加权Top2[4,18]
  与硬Top2[4,20]不同；下一任务起由不同冻结集合影响训练。
- Full-cohort smoke val两阶段25.228568/27.608866；seen-positive replay val
  26.760875/27.958062，与训练记录相同。核验B0初始化/样本来源、旧专家和Router
  不变、CLIP不变、skipped=0。task1旧lane probe relative L2=0.00295064，
  说明冻结部分专家不保证整条旧lane特征不变。
- 正式已启动：`tmux ml_moe_waf_seed0_20261018`，
  `RUN_ID=formal_seed0_20261018_01 GPU=1 SEED=0`，8task×30epochs。
  初始PID514567位于GPU1 UUID `GPU-7d793ed1-ce16-d011-ad9e-aaa15a9eb7f5`；
  task0 epoch1已完成84更新、0跳步、loss0.67717795，尚无正式最终指标。
- 训练后自动核验，再full_val/full_test同GPU1双进程并行；非八任务训练并行。
  日志`logs/moe_weighted_freeze/formal_seed0_20261018_01/`，
  外层日志`logs/moe_weighted_freeze/formal_seed0_20261018_01.launch.log`；
  输出`output/moe_weighted_freeze/formal_seed0_20261018_01/`。
- GPU0原AF session仍在运行，服务器工作树保持07fd35c clean；主树和test-only不变。
  smoke/preflight共37个非checkpoint文件已同步此本地工作树，SHA256全部相符。
  首轮失败产物保留。等待正式完成后再同步分析；未合并main。

正式命令（已执行，不要重复启动同run ID）：

```bash
env -u SMOKE_UPDATES GPU=1 SEED=0 RUN_ID=formal_seed0_20261018_01 \
  bash scripts/emotic/run_moe_weighted_freeze.sh
```

## 2026-10-18正式完成（替代前文运行中状态）

8task×30epochs、13950更新、0跳步，整体/validation/test退出0。最终WAF
val42.2952/test31.6815，阶段均值47.9071/35.0797；AF最终42.4007/31.7904。
已同步结果并与B0/v1核对来源/固定cohort，新增只读Router probe与离线bootstrap。
完整分析及下一步建议见[结果报告](moe_weighted_freeze_results_analysis_20261018.md)。
当前保留B0，不自动新训练或扩seed，不合并main。
