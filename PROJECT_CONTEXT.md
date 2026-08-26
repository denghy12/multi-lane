# 项目上下文

最后一次更新：2026-08-26；当前本地分支为
`exp/emotic-image-token-asl-capacity-lr`。服务器四组pair8/9 BCE同批确认已在clean`e50b4a3`
完成并退出；新的Image-token Adapter-ASL容量/LR阶段一实现待提交同步。额外test-only worktree
保持原分支和原有未提交状态，未被本轮同步、分析或实验修改。

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

### Task-lane Transformer Adapter 实验

- 新的诊断分支为 `exp/emotic-adapter-position-diagnostics`，从已完成warm-start结论的
  `5b2f791` 创建；诊断实现提交 `5766860` 已推送，本地与服务器主工作树均切到该分支。
- runner新增可选 `legacy_full_zero/current_only` loss；默认继续使用legacy，保证历史严格复现
  不变。current-only直接对当前任务5/3类logits计算BCE，去除26维零填充常数，并把原始有效
  梯度分别放大约 `26/5` 与 `26/3`；但Adam矩归一化会抵消大部分纯常数缩放，必须用完整
  seed0 validation实证选择，不能预设current-only一定提升。
- 输入预处理新增独立 `none/clip` normalization和可配置RandomResizedCrop scale；旧默认仍为
  无normalization、scale `(0.05,1.0)`。CLIP normalization使用OpenAI mean/std，crop诊断候选
  为最小面积 `0.05/0.50`，两项以2×2 disabled对照分开评估。
- 诊断流程固定为三阶段：先disabled比较legacy/current-only loss；再用选中loss做normalization
  × crop 2×2；最后以选中的disabled配置为基线，运行independent Adapter的单层零基索引
  `5/8/11`（即第6/9/12 block）。每阶段均限seed0、完整8-task、val-only、clean Git；只有中层
  改善task6 mAP、task7 final mAP和task6新类平均AP，才考虑小范围多层。
- 新增通用validation diagnostics汇总器，用于loss和预处理阶段的disabled多运行对照；它严格
  校验seed0/8-task/val-only/clean Git并输出完整曲线、汇总指标和task6三类AP，但不使用单一
  scalar自动选优，避免average mAP掩盖后期退化。
- GPU smoke入口支持显式指定零基Adapter layer indices；本轮验证计划仅在GPU2/3/4分别对
  layer `5/8/11`执行真实CLIP batch2 forward/backward、零初始化等价、梯度/冻结与路由检查，
  不读取EMOTIC、不写checkpoint，也不启动任何诊断训练。
- 服务器完整Track-A单元测试20/20通过。GPU2/3/4上的layer `5/8/11` smoke均通过：每组
  可训练参数 `788314`，零初始化Adapter与disabled路径的AMP logits最大差均为
  `6.103515625e-05`；本轮未启动loss、预处理或层位置实验。
- 第一阶段loss诊断已于2026-08-12启动：同一clean `001d8a0`、seed0、完整8-task val-only、
  Adapter disabled配置下，GPU2运行`legacy_full_zero`，GPU3运行`current_only`；batch64、
  30 epochs/task、LR0.0125、none normalization、crop `(0.05,1.0)`。启动后两组各新增约
  1.9GB显存，仍各有约14.1GB空闲；task0前5个epoch均steps84/skipped0且无OOM/NaN/异常。
- 两组均以240 epochs、13950 optimizer updates、exit code 0完成。current-only相对legacy的
  final/average mAP为`-0.645109/-0.096314`，final cF1/oF1为`-0.875881/-0.078401`；
  forgetting数值改善`0.095630`，但task6/7 mAP下降`0.757291/0.645109`。task6新类引入时
  Sadness/Sensitivity/Suffering AP变化为`-0.557096/+0.750446/-13.552041`，均值下降
  `4.452897`，因此loss选择为保留`legacy_full_zero`。
- legacy结果与上一轮clean disabled对照的8-task mAP及五项summary逐项完全一致，确认随机轨迹
  与历史行为未漂移。current-only与legacy在数学上仅相差常数项和每task整体梯度缩放；Adam
  基本抵消该缩放，所以它没有修复监督逻辑，剩余数值轨迹扰动反而损害后期Suffering。
- 第二阶段预处理诊断已启动：固定`legacy_full_zero`、Adapter disabled和其余协议，GPU2/3/4/7
  分别运行`none+crop0.05`、`clip+crop0.05`、`none+crop0.50`、`clip+crop0.50`。四组均为
  clean `001d8a0`、seed0、完整8-task val-only；首个epoch均84 steps、skipped0，每组约占
  1.94GB显存且仍有约22.1GB空闲，无OOM/NaN/异常。
- 四组均完成240 epochs/13950 updates并自动汇总。相对旧默认`none+crop0.05`，
  `clip+crop0.05`的task6/final mAP为`+0.134184/+0.211756`，average mAP`+0.098297`，
  task6新三类均值`+0.484693`；但final cF1/oF1为`-0.090485/-0.285273`，且Sadness
  `+7.019647`的同时Suffering`-5.928491`。按预先三项排序指标规则，下一阶段预处理固定为
  CLIP normalization与crop `(0.05,1.0)`，同时将Suffering列为层位置诊断硬性观察项。
- crop minimum提高至`0.50`在none/clip下分别使final mAP下降`1.318310/1.003698`，并且
  8个task mAP全部下降；虽然末轮训练loss更低，validation更差说明较弱裁剪增强导致泛化下降，
  因此淘汰crop0.50。四组21个JSON/日志文件已同步本地并逐项通过SHA-256校验。
- 第三阶段单层位置诊断已启动：固定`legacy_full_zero + clip normalization + crop(0.05,1.0)`，
  independent Adapter b64/LR4e-4/ReLU/scale0.1，GPU2/3/4分别运行zero-based layer `5/8/11`。
  三组均为clean `001d8a0`、seed0、完整8-task val-only；首两轮84 steps、skipped0，每组约
  1.94GB显存且仍有约22.1GB空闲。最终除task6/final mAP和task6三类均值外，增加Suffering
  AP不得低于同预处理disabled基线的硬性条件。
- 三层均完成240 epochs/13950 updates且无运行异常。相对同预处理disabled基线，layer5/8/11
  的task6 mAP为`+0.031871/+0.540776/-0.340454`，final mAP为
  `-0.093808/+0.380562/-0.439175`，task6新类均值为
  `-0.525884/-1.296133/-8.770622`，Suffering AP为
  `-0.528539/-2.105999/-14.799475`；没有一层通过四项硬条件，停止多层与容量扩张。
- layer8虽是aggregate最优位置，average/final mAP提升`1.013492/0.380562`，但task6旧类平均
  AP提升`0.816313`的同时新类均值下降`1.296133`；layer11更是旧类`+0.924071`、新类
  `-8.770622`。位置变化主要强化已学lane而损害后期新类可塑性，不能解决当前task-lane
  Adapter的结构问题。18个JSON/日志文件已同步本地并逐项通过SHA-256校验。
- 2026-08-12已正式收口该路线：停止layer位置扩展、多层Adapter、bottleneck扩容和Adapter深度
  增加，不再使用正式test调参。当前保留的validation基线为Adapter disabled、
  `legacy_full_zero + CLIP normalization + crop(0.05,1.0)`；三个已完成诊断tmux已关闭，活动
  诊断runner为0，日志与结果目录完整保留。后续若提出针对新类可塑性/类别不平衡的新假设，
  应另建实验分支，而不是继续扩展当前task-lane Adapter。
- 2026-08-13按用户确认开始新的Image-token Adapter方向，实验分支为
  `exp/emotic-image-token-adapter`。该模式在zero-based CLIP block 8读取冻结并经`ln_1`
  归一化的全部`CLS + patch tokens`，为每个task训练独立bottleneck32 Adapter；适配后的图像
  tokens只供对应lane的selector相似度计算与patch汇聚，不回写冻结CLIP残差流。现有
  `disabled`与`task_lane`路径保持不变。
- 用户确认时间紧，跳过validation筛选，直接运行一个seed（seed0）的完整8-task held-out test
  正式实验。固定配置为`legacy_full_zero + CLIP normalization + crop(0.05,1.0)`、30 epochs/task、
  batch64、Adam、base LR0.0125、Adapter LR4e-4、scale0.1、ReLU、independent初始化。该单seed
  结果只能作为探索性对照，不能报告三seed均值或统计显著性。
- 新实现已以`e10324f`提交并推送，服务器主工作树与本地同HEAD且clean；服务器`ddp`环境
  22/22完整单元测试通过。GPU2真实CLIP smoke通过：可训练参数`739130`，zero-up Adapter与
  runtime-disabled输出最大差`3.4332e-05`（AMP容差内），forward/backward、Adapter梯度、
  frozen visual无梯度和concat inference均正常。
- seed0 held-out test正式实验已于2026-08-13 00:45在GPU2启动，tmux为
  `mla_image_token_seed0_004500`，run ID为
  `image_token_adapter_b32_layer8_seed0_20260813_004500`，运行代码提交为`a322419`。
  task0 epoch1耗时17.7秒、loss `0.61481267`、84步且skipped0；GPU2占用约2.0GB，未出现
  OOM、NaN、Traceback或RuntimeError。实验仍在运行，不启动seed1/2。
- 用户随后要求参照本地`CODE_DDP`的ASL，在Image-token Adapter基础上直接做三组seed0正式
  test：仅selectors/prompts/head使用ASL、仅Adapter使用ASL、两者都使用ASL。新分支为
  `exp/emotic-image-token-asl-routing`，其余Image-token架构与训练协议保持不变。
- ASL公式锁定为`CODE_DDP`的`gamma_neg=9.8, gamma_pos=0, clip=0.05, eps=1e-8`、detach
  focal weight，并保持MULTI-LANE `legacy_full_zero`的26维mean reduction。混合loss采用一次
  forward后对两个不相交参数组分别求`autograd.grad`：model组为selectors/prompts/head，
  Adapter组只包含当前task的Image-token Adapter，从而保证model-ASL与adapter-ASL不是同一
  梯度的不同命名。三组均为seed0、完整8-task、30 epochs/task、held-out test，不做validation
  筛选；单seed/test直跑只作探索性结果。
- ASL实现提交`6d5430a`已推送；为避免切换GPU2正在运行的原Image-token BCE主工作树，服务器
  新建独立worktree `/mnt/haoyuan/workspace/multi-lane-main-asl-routing`并检出同一clean提交，
  test-only worktree保持原状。服务器25/25完整单元测试通过；GPU3/4/7上的
  `model_asl/adapter_asl/both_asl`真实CLIP smoke全部通过，初始等价最大差分别为
  `3.0518e-05/4.1962e-05/3.0518e-05`，可训练参数均为`739130`，无OOM或梯度/冻结异常。
- 验证记录提交`8840e96`同步后，三组正式实验于2026-08-13 01:16在tmux
  `mla_image_token_asl_seed0_011602`启动：GPU3=`model_asl`、GPU4=`adapter_asl`、
  GPU7=`both_asl`，共同时间戳`20260813_011602`。task0 epoch1均完成84步/skipped0，耗时
  16.4--16.8秒；model/Adapter objective loss分别为`0.01573/0.76665`、
  `0.61468/0.02818`、`0.01578/0.01578`，与三种路由定义一致。单卡占用约2.0--2.4GB，
  无OOM、NaN、Traceback或RuntimeError。
- 原Image-token BCE及三组ASL均已正常完成：每组8 tasks × 30 epochs，共240 epochs、13950
  optimizer updates、skipped0、launcher exit code0，config均记录clean Git。四组除loss路由
  外的数据、seed、预处理与Adapter配置逐项一致。
- Image-token BCE的final/average mAP、cF1、oF1、forgetting为
  `31.676786/38.564709/31.911654/49.231766/4.801635`。相对它，`adapter_asl`为
  `+0.742543/+0.314189/+0.247229/-0.130171/+0.022592`，是四组中final mAP最佳配置；
  task6/7 mAP分别提高`0.735797/0.742543`。
- `adapter_asl`在task6的Sadness/Sensitivity/Suffering AP分别提高
  `7.524682/0.652493/8.239046`，三类均值提高`5.472074`，且提升在task7仍保留；它通过
  Sadness与Suffering必须同时改善的方向性检查。但只有seed0且直接观察held-out test，不能
  宣称统计显著或继续据此在test上调参。
- `model_asl`使final mAP下降`1.214722`；`both_asl`虽使final mAP提高`0.522624`，两者的
  final oRecall均接近100%、oPrecision约17%，oF1分别下降`20.004362/19.356518`。根因是
  gamma-neg9.8与negative clip过度压低主模型易负样本梯度，导致logit校准在固定0.5阈值下
  偏向全正预测；mAP排序能力不能掩盖该F1失效。仅Adapter使用ASL时主模型BCE仍维持校准，
  因而没有该坍塌。
- 四组结果与日志已按24文件白名单打包，不含checkpoint。本地目录为
  `output/emotic_track_a_image_token_loss_routing_formal/image_token_bce_asl_seed0_results_20260813/`，
  本地压缩包SHA-256为`60cb61ebe7f64d639f351821834c95539bad149ad0abe6e1d3f4796a2d8df884`；
  详细对照见该目录`comparison_analysis.md`。本轮未启动任何后续实验。
- 2026-08-20开始Image-token Adapter-only ASL调参阶段，新分支为
  `exp/emotic-image-token-asl-hparam-search`。第一阶段预注册为seed0、完整8-task、30
  epochs/task、validation-only；固定主模型BCE、legacy监督、CLIP normalization、crop0.05、
  image-token layer8/b32/LR4e-4/scale0.1/ReLU/independent，仅联合搜索
  `gamma_neg={1,2,4,6,9.8}` × `clip={0,0.025,0.05,0.1}`、`gamma_pos=0`，并加入一组
  joint-BCE对照，共21组。
- 新增通用Image-token validation worker、8-GPU队列启动器和专用汇总器。汇总器严格校验
  clean Git、8 tasks、240 epochs、13950 updates、skipped0、完整预注册网格与配置一致性；
  只有task6/final mAP、Sadness、Suffering不下降且final cF1/oF1下降不超过0.5的候选才进入
  排名。新增`--no-save-checkpoints`，调参输出只保留小型JSON和日志，旧正式流程默认保存
  checkpoint的行为不变。
- 8-GPU启动器加入资源门控：每张卡必须连续两次满足空闲显存至少5000MiB且利用率不高于
  10%，检查间隔60秒，才允许执行；指定smoke卡先运行真实CLIP Adapter-ASL smoke，通过后
  各GPU独立等待并顺序执行自己的2--3组队列。这样可以在当前任务结束后自动启动，同时避免
  把调参任务叠加到繁忙GPU。
- 2026-08-20服务器GPU0--7预检时全部已有高负载进程：每卡使用约20.4--22.2GB、仅余
  1.9--3.7GB，利用率57%--91%。为避免OOM和干扰现有任务，第一阶段尚未启动，也未终止任何
  进程。Automatic Upload产生的服务器主worktree漂移已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-hparam-upload-20260820`并恢复clean；
  ASL独立worktree同样保持clean。
- 第一阶段队列已进入tmux等待态，session为
  `mla_image_token_asl_hparam_s0_20260820_191924`，batch ID为
  `image_token_asl_loss_seed0_20260820_191924`。首次门控快照GPU5仅余3520MiB且利用率99%，
  因而正确保持等待；此时尚未运行GPU smoke、尚未创建任何训练run，也没有占用新增显存。
- 第一阶段最终完成18/21组；所有完整组均为240 epochs、13950 updates、skipped0，另有
  `gn4/clip0`、`gn6/clip0.025`、`gn9.8/clip0.025`在task3因非有限ASL loss主动终止，
  判为数值无效而不进入排名。不是OOM，当前GPU0--7均已释放。
- 严格汇总按预先硬门槛选出`adapter_asl_gn9p8_clip0p05`。相对同轨迹joint-BCE validation，
  其final/average mAP提高`0.709193/0.922159`，final cF1/oF1提高
  `0.686412/0.430169`，task6 mAP提高`0.695724`；8个task的mAP均有改善。
- 最优配置task6 Sadness/Sensitivity/Suffering变化为
  `+1.404916/-0.530427/+8.675063`，三类均值提高`3.183184`。它解决了本阶段硬性关注的
  Suffering方向，但未改善Sensitivity；forgetting由`0.940729`变为`1.017189`，恶化
  `0.076460`，因此不能宣称已解决遗忘或所有稀有新类。
- 唯一另一组全门槛合格配置为`gn9.8/clip0`，其final mAP提高`0.526062`，但cF1、forgetting
  与Suffering均弱于赢家。此前held-out test探索恰好使用同一赢家参数，final mAP与Suffering
  同方向改善；由于test已被观察，仍不能把它当作调参后的未触碰最终检验。
- 结果已按107文件白名单打包并同步本地，含18组JSON/日志和失败日志，不含checkpoint；
  压缩包SHA-256为`90e0ea0817ec0ea266e72f3961a58e4f2abe89603c31f4755ef91c8ad0ab7684`，
  详细报告位于本地结果目录`analysis.md`。服务器failure-aware汇总实现完整单测29/29通过。
- 当前没有活动实验。由于赢家`gamma_neg=9.8`仍位于预注册搜索上边界，下一阶段若获确认，
  应先用seed0完整8-task val-only做`gamma_neg={8,9.8,12,16}` ×
  `clip={0.0375,0.05,0.075}`局部边界确认（复用已有9.8/0.05，共11个新运行），得到内部最优
  后才搜索gamma-pos；不得直接启动新test。
- 失败日志进一步表明三组均在task3约2000次连续更新后开始出现AMP GradScaler跳步，恰好接近
  默认loss scale增长区间；随后跳步逐渐增多并出现非有限logits/loss。因此稳定版不用特殊LR
  掩盖失败，而是让joint-BCE与全部12个局部ASL组合统一使用FP32，重新生成同提交对照。
- 新分支`exp/emotic-image-token-asl-stable-refine`已准备FP32 smoke、13组8-GPU启动器和
  `stable_refine_fp32`严格汇总profile。局部网格为`gamma_neg={8,9.8,12,16}` ×
  `clip={0.0375,0.05,0.075}`、gamma-pos0；由于精度协议变化，旧9.8/0.05不再复用，需重跑
  12个ASL加1个joint-BCE。runner额外在ASL入口显式检查logits/targets有限性，便于区分上游
  NaN与loss公式异常；历史AMP入口默认行为保持不变。
- 稳定版实现提交`3564e12`已推送并安全同步到服务器ASL独立worktree；完整Track-A单元测试
  30/30通过。GPU0真实CLIP FP32 Adapter-ASL smoke通过，可训练参数739130，零初始化Adapter
  与disabled logits最大差`1.70e-08`，梯度路由与冻结visual正常。
- 13组正式validation队列已于2026-08-21 23:14启动，batch ID为
  `image_token_asl_stable_refine_seed0_20260821_231444`，tmux为
  `mla_asl_stable_refine_s0_20260821_231444`。第一轮GPU0--7分别运行joint-BCE、gn8的三个
  clip、gn9.8的三个clip和gn12/clip0.0375；第二轮GPU0--4自动运行gn12剩余两组与gn16三组。
- 第一轮8组config均记录clean`3564e12`与`amp=false`；task0 epoch1均为84 steps、skipped0，
  单卡显存约2.8--3.0GB且仍余约21GB，无OOM、NaN、Traceback或RuntimeError。服务器ASL
  worktree在13组全部结束前不得fast-forward到后续文档提交，否则第二轮Git commit会不一致。
- FP32稳定版13/13组已完成，每组8 tasks、240 epochs、13950 updates、skipped0、exit code0；
  总计3120 epochs、181350 updates。严格汇总确认同一clean`3564e12`、`amp=false`，无OOM、
  NaN、Inf或异常，说明上一轮三个非有限loss确由AMP数值轨迹触发，FP32已解决运行稳定性。
- 但严格硬门槛没有合格赢家。FP32 joint-BCE的final/average mAP、cF1、oF1、forgetting为
  `42.164859/48.795463/37.501026/58.083137/0.832476`；12个ASL组合的final cF1全部下降、
  forgetting全部恶化，不能宣称ASL整体优于BCE。
- final mAP最高仍为`gn9.8/clip0.05`，相对FP32 BCE为
  `final mAP +0.425109 / average mAP +0.660861 / cF1 -0.635873 / oF1 +0.443694 /
  forgetting +0.058057`；task6 Sadness/Sensitivity/Suffering为
  `-2.088062/-1.408021/+6.074236`，失败Sadness与cF1门槛。
- 最接近全部门槛的`gn16/clip0.075`仅失败Sadness门槛，final/average mAP提高
  `0.362468/0.850003`，Suffering提高`6.925002`，但Sadness下降`2.326867`、forgetting
  恶化`0.213311`，按预先规则仍不得选中或据此放宽门槛。
- 81文件无checkpoint结果包已同步本地并逐文件SHA-256通过；压缩包SHA为
  `5e917f92033491bf6d281a6e9f281167434f40194dd6188cd624ac6a1f302e22`，详细报告见本地
  `image_token_asl_stable_refine_seed0_20260821_231444_results/analysis.md`。当前不启动
  gamma-pos或test，建议收口无目标的gamma扩展。
- 2026-08-22按新的结构调参计划创建`exp/emotic-image-token-structure-tuning`。第一阶段只
  诊断Image-token Adapter的单层位置，不复用Task-lane Adapter的位置结论：zero-based
  layer `0--11`分别成对运行Adapter-BCE与“主模型BCE + Adapter ASL”，并加入同轨迹
  Adapter-disabled BCE锚点，共25组。ASL固定`gamma_neg=9.8/gamma_pos=0/clip=0.05`。
- 25组统一使用FP32、seed0、完整8-task validation-only、30 epochs/task、batch64、
  `legacy_full_zero + CLIP normalization + crop(0.05,1.0)`、Image-token b32、Adapter
  LR4e-4、scale0.1、ReLU、independent初始化；不读test、不保存checkpoint。只有层位置确定后
  才允许进入bottleneck/LR搜索，后续依赖阶段不会预先排队。
- 新增8卡独立资源门控启动器：全局真实CLIP FP32 Adapter-ASL smoke使用最先空闲的任意GPU；
  每个GPU lane在每组训练前都必须连续两次满足空闲显存至少8000MiB、利用率不高于10%，默认
  间隔60秒。现有GPU0--7均被其他高负载任务占用，启动器必须保持等待且不得终止现有进程。
- 新增严格层搜索汇总器：强制25组同一clean commit/tree、完整240 epochs/13950 updates、
  skipped0、无checkpoint和配置完全一致；每层先比较BCE与disabled，再比较ASL与同层BCE。
  选择除final/task6 mAP外还要求Sadness与Suffering不降、Sensitivity/cF1/oF1下降不超过
  0.5点且forgetting不恶化；ASL候选需要同时相对同层BCE与disabled锚点通过这些门槛，防止
  只靠aggregate mAP或较弱的同层BCE掩盖task6新类退化。
- 实现提交`d63da6b`已推送并由服务器ASL独立worktree切换到同一clean HEAD；服务器完整
  Track-A单元测试33/33通过。Automatic Upload产生的服务器主worktree漂移已完整备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-layer-search-upload-20260822`并恢复clean，
  test-only worktree保持原有未提交状态且未被修改。
- 8卡门控队列已于2026-08-22 23:52放入tmux
  `mla_image_token_layer_s0_20260822_235242`，batch ID为
  `image_token_layer_search_seed0_20260822_235242`。启动审计时GPU0--7空闲显存仅
  1906--4604MiB，全部低于8000MiB门槛；launcher log显示8卡均在等待，输出目录只有
  `search_manifest.tsv`，尚未执行GPU smoke、创建训练config或启动任何本批runner。
- 服务器实验worktree必须固定在`d63da6b`直至25组全部结束并汇总；后续文档提交只推送远端，
  此期间不对该worktree fast-forward。某张GPU率先连续两次满足门槛后会先执行全局smoke，
  smoke通过才创建8个独立lane，其他繁忙GPU继续各自等待。
- 25组已全部正常结束并自动汇总：每组8 tasks、240 epochs、13950 updates、skipped0、exit
  code0，总计6000 epochs与348750 updates；同一clean`d63da6b`，无OOM、非有限数值或异常，
  无checkpoint。当前tmux已退出、无残留runner，GPU0--7均空闲。
- disabled锚点final/average mAP、cF1、oF1、forgetting为
  `41.955741/48.656124/37.493415/58.151120/0.806163`。BCE final mAP最高为layer9，
  相对disabled提高`0.402611`，但Sadness下降`1.836305`且forgetting恶化`0.216600`；
  12个BCE层的forgetting均恶化，因此没有BCE层通过硬门槛。
- ASL final mAP最高为layer3，相对disabled的final/average/task6 mAP提高
  `0.976536/0.956794/1.048385`，cF1/oF1提高`0.655264/1.155307`，但Suffering下降
  `2.695190`且forgetting恶化`0.043960`。layer7以Sadness下降换取Suffering上升；layer11
  虽同时改善Sadness/Suffering，却使Sensitivity下降`1.126123`、forgetting恶化`0.300807`。
- 严格汇总结果`eligible_bce_layers=[]`、`eligible_asl_layers=[]`，两种loss均无winner和continue
  标志。层位置没有消除task6类别间交换或遗忘，按预注册依赖关系不进入bottleneck/LR、scale、
  activation或ASL微调，也不启动新test。
- 结果已按153文件白名单打包同步本地并逐项SHA-256通过，不含checkpoint；压缩包SHA为
  `8f6d4dd4f6fa0cb9aa0f917dd6f7f971c0acde77b5742e01608142ab356e7c05`，详细报告见本地
  `output/emotic_image_token_tuning/layer_search/image_token_layer_search_seed0_20260822_235242_results/analysis.md`。
- 当前layer8 BCE与上一批相同训练数学和seed的FP32 layer8 joint-BCE相比，final/average mAP
  仍出现`-0.307112/-0.719150`的重复运行差异，且从task0 epoch1已有微小数值分叉；两次位于
  不同物理GPU。因此本批约0.1--0.4点的小幅位置收益不能视为可靠结构优势。
- 2026-08-25按用户要求继续做一次受控多层诊断，但不直接把失败硬门槛的layer3/11送入正式
  held-out test：layer3牺牲Suffering，layer11牺牲Sensitivity且forgetting显著恶化，尚不能
  判断为“预期正式效果好”。新分支为`exp/emotic-image-token-multilayer-screen`。
- 多层screen共15组、8卡两轮：同批重跑disabled锚点与single3/single11的BCE/ASL作为数值
  可重复性控制；多层结构为zero-based`[2,3]`、`[3,7]`、`[3,11]`、`[8,9]`与物理block
  8--12对应的zero-based`[7,8,9,10,11]`，每种均成对运行BCE和Adapter-ASL 9.8/0/0.05。
- 多层Adapter不是级联改写CLIP残差流：每个选定block拥有独立task-specific Adapter，只将
  该层`LN1(CLS+patch)`的适配结果用于selector匹配与patch汇聚；冻结CLIP image-token残差流
  继续原样传到下一block。5层配置每task新增`5*49952=249760`个当前可训练Adapter参数，远低于
  4090显存容量，但仍需真实CLIP多层FP32 smoke后才启动训练。
- 15组继续固定seed0、完整8-task validation-only、FP32、30 epochs/task、batch64、
  legacy+CLIP normalization+crop0.05、b32/LR4e-4/scale0.1/ReLU/independent，不读test、不存
  checkpoint。多层除原Sadness/Sensitivity/Suffering/F1/forgetting门槛外，final与task6 mAP
  还不得低于同批single3/single11中同loss最佳值，避免仅靠增加参数得到无意义小幅波动。
- 多层实现提交`976f2a1`已推送并安全同步到服务器独立实验worktree；服务器完整Track-A测试
  36/36通过。GPU0真实CLIP layers`[3,11]` FP32 Adapter-ASL smoke通过，可训练参数789082，
  zero-init与disabled logits最大差`1.7695e-08`，梯度路由和冻结visual正常。
- 15组队列已于2026-08-25 11:16启动，batch ID为
  `image_token_multilayer_screen_seed0_20260825_111638`，tmux为
  `mla_image_token_multilayer_s0_20260825_111638`。第一轮GPU0--7依次运行disabled、single3
  BCE/ASL、single11 BCE/ASL、pair2_3 BCE/ASL和pair3_7 BCE；8组task0 epoch1均完成84
  steps、skipped0，约16.0--18.0秒，无异常。
- 第一轮单卡总显存占用约2.4--3.3GB，仍余20.8GB以上，无OOM风险信号。第二轮由各GPU lane
  自动接续pair3_7 ASL、pair3_11 BCE/ASL、late8_9 BCE/ASL与late_blocks8_12 BCE/ASL；
  每组开跑前继续执行8GB/10%资源门控。实验worktree固定`976f2a1`直至15组全部结束并汇总。
- 15/15组已全部完成：每组8 tasks、240 epochs、13950 updates、skipped0、exit code0，总计
  3600 epochs与209250 updates；同一clean`976f2a1`，无OOM、NaN、Inf或异常，无checkpoint。
  tmux已退出、无残留runner，GPU0--7均空闲。
- 同批disabled锚点final/average mAP为`41.532346/48.189057`。single3 ASL仍以Suffering
  `-4.979032`换取final mAP`+0.823415`，确认不能正式test；single11 BCE本批通过全部门槛，
  final/task6 mAP`+0.753962/+0.817999`且三类均不降，但其跨批方向仍有明显波动。
- 多层唯一合格结构为zero-based`[8,9]` BCE：相对disabled的final/average/task6 mAP为
  `+0.939906/+0.844164/+1.007650`，cF1`+1.052756`，Sadness/Sensitivity/Suffering为
  `+3.473313/+2.236642/+3.889481`，forgetting改善`0.048449`。严格汇总
  `winner_bce_structure=late8_9`且`continue_with_bce=true`。
- 所有5个多层ASL相对同结构BCE的final mAP均下降`0.252--1.369`点，且没有候选通过类别/F1/
  forgetting门槛；`eligible_asl_structures=[]`。dense zero-based`[7,8,9,10,11]`也使Suffering
  下降`4.981710`、forgetting恶化`0.194592`，说明增加层数不是单调收益，停止多层ASL与dense
  后段扩展。
- `[8,9]` BCE相对同批single11 BCE仅提高final/task6 mAP`0.185944/0.189651`；上一批最佳
  single9 BCE绝对final mAP为`42.358352`，本批pair为`42.472252`，跨批只差`0.113899`且两批
  disabled自身相差`0.423396`。由于本批没有fresh single9，尚不能证明双层优于layer9单层，
  暂不启动held-out test。
- 93文件白名单结果包已同步本地并逐项SHA-256通过，无checkpoint；压缩包SHA为
  `ddd5c6c0a07702a70094f4b89cec2cf0a2ef47db5be5a2006d60630688ce2f44`。详细分析位于
  `output/emotic_image_token_tuning/multilayer_screen/image_token_multilayer_screen_seed0_20260825_111638_results/analysis.md`。
- 2026-08-25按用户确认进入最小BCE-only pair8/9复核分支
  `exp/emotic-image-token-pair89-confirmation`。只运行fresh disabled、single8、single9和
  pair`[8,9]`四组，同一提交在GPU0--3一轮完成；不再运行ASL、layer3/11或其他多层结构。
- 四组继续固定seed0、完整8-task validation-only、FP32、30 epochs/task、batch64、
  legacy+CLIP normalization+crop0.05、b32/LR4e-4/scale0.1/ReLU/independent，不读test、不存
  checkpoint。启动前使用pair`[8,9]` BCE做真实CLIP FP32 smoke，每卡连续两次满足8GB/10%
  资源门槛后才开跑。
- 严格确认规则预先固定：pair先通过相对disabled的原硬门槛；相对fresh single9还必须同时满足
  final/average/task6 mAP不降、Sadness/Sensitivity/Suffering不降、forgetting不增，final
  cF1/oF1下降不超过0.5。任一失败都不进入正式test，并推荐参数更少的fresh最佳单层。
- 确认实现提交`e50b4a3`已推送，服务器39/39完整Track-A测试通过；pair`[8,9]`真实CLIP FP32
  BCE smoke通过，可训练参数789082，zero-init最大差`1.4901161e-08`。
- 四组于2026-08-25 13:21在GPU0--3同批启动，batch ID为
  `image_token_pair89_confirmation_seed0_20260825_132139`，tmux为
  `mla_image_token_pair89_s0_20260825_132139`。disabled、single8、single9、pair8_9的task0
  epoch1均为84 steps、skipped0、12.5--14.6秒；每卡占用约2.4--2.8GB、剩余21.2GB以上，
  无OOM、NaN、Inf或异常。GPU4--7未占用，本轮没有ASL、其他多层或held-out test。
- 四组最终均完成240 epochs、13950 updates、skipped0、exit code0，无异常或checkpoint。
  disabled/single8/single9/pair的final mAP为`41.863114/42.422405/41.615440/42.350681`。
- pair相对single9的final/average/task6 mAP虽提高`0.735241/1.187716/1.020656`，但Sensitivity
  下降`1.210430`、forgetting恶化`0.216919`，且相对disabled forgetting也恶化`0.193886`；
  因此`confirmed_for_formal_test=false`，停止pair、多层、扩容和ASL，不运行正式test。
- single8相对disabled在八个task阶段的mAP均提高，final/average提高`0.559291/0.902963`且
  forgetting改善`0.093862`，但Sadness下降`5.071147`，只能作为下一轮validation探索锚点，
  不能宣称已通过方法门槛。task6仅627个训练样本而pair每task参数99904，下一方向应优先测试
  single8小容量/低Adapter LR来降低方差；若仍有类别交换，再转双视图上下文融合。
- 本批虽关闭AMP，但四组`config.json`仍记录`tf32=true`，所以此前“FP32”应准确表述为
  AMP off而非严格IEEE FP32。该设置不破坏本批同配置比较；下一轮筛选小效应前应显式
  `--no-tf32`并重跑同批fresh anchors。
- 2026-08-26用户明确后续方法固定为“主模型BCE + Image-token Adapter ASL”，不再以BCE
  Adapter作为主候选。新的阶段一分支为`exp/emotic-image-token-asl-capacity-lr`，固定zero-based
  layer8、scale0.1、ReLU、independent及ASL 9.8/0/0.05，联合搜索
  bottleneck`{8,16,32,64}` × Adapter LR`{1e-4,2e-4,4e-4}`，另加fresh disabled BCE，
  共13组、8卡自动两轮。
- 阶段一统一seed0完整8-task validation-only、30 epochs/task、batch64、legacy、CLIP
  normalization、crop0.05、AMP off、TF32 off、无checkpoint，不读取held-out test。b32/LR4e-4
  是同数值协议下的当前冠军参数锚点；候选必须同时相对disabled和该锚点保护final/average/task6
  mAP、task6三类、F1和forgetting，并至少提高0.5 final mAP才进入scale/activation阶段。
- worker新增显式`NO_TF32`路由，GPU smoke新增`--no-tf32`并报告TF32状态；新增13组资源门控
  launcher、严格汇总器和3项隔离测试。启动前先在服务器运行完整Track-A测试及b64真实CLIP
  Adapter-ASL严格FP32 smoke。
- 阶段一实现提交`94f3327`已推送并安全同步至服务器独立实验worktree，完整Track-A测试42/42
  通过。GPU1 b64最大容量真实CLIP Adapter-ASL smoke通过，明确记录`precision=fp32 tf32=off`、
  可训练参数788314、zero-init最大差`1.8626451e-08`，无冻结、梯度或数值异常。
- 13组总队列已于2026-08-26 16:49启动，batch ID为
  `image_token_asl_capacity_lr_seed0_20260826_164945`，tmux为
  `mla_image_token_asl_caplr_s0_20260826_164945`。首轮8组已全部运行到task0 epoch4--5，均为
  84 steps、skipped0，ASL loss有限；每卡约占2.8--3.0GB、仍余21GB以上，无OOM、NaN、Inf或
  异常。GPU0--4将在首组结束后自动接续剩余5组，实验worktree固定`94f3327`直至自动汇总。

- 当前实验分支为 `exp/emotic-multilane-transformer-adapter`，起点是已严格复现注册结果的
  `ce7d9a0`；严格复现分支保持冻结。
- Adapter 只作用于 MULTI-LANE 的 task lane tokens，不修改共享 CLIP image tokens；默认
  放在第 12 个视觉 block（Python 索引 11），与冻结 MLP 并联：
  `MLP(LN2(x)) + 0.1 * Up(ReLU(Down(LN2(x))))`。
- 每个 task/layer 使用独立、预分配的 Adapter；进入新任务时只解冻当前任务 Adapter，
  旧任务与未来任务 Adapter 均冻结，concat inference 按 lane id 路由，因此不需要 task
  oracle，也不会覆盖旧任务 Adapter。
- 上投影权重和偏置全零初始化，启用 Adapter 的初始 forward 应与原 MULTI-LANE 完全一致；
  `--adapter-mode disabled` 是默认值，原正式 launcher 不传任何新参数，保持旧训练计算路径。
- 阶段 0 新增 bottleneck、layer、residual scale、activation 和独立 Adapter LR 配置，并增加
  `--max-tasks 1 --reporting-split val` 的 task0 验证集筛选能力；该模式不会构造或读取 test
  split，避免用测试集选超参数。
- 首轮候选固定为 layer 11、ReLU、scale 0.1，对 bottleneck `32/64` 与 Adapter LR
  `1e-4/4e-4` 做 seed0 task0 validation screen；尚未启动任何训练。
- 服务器 8/8 单元测试通过；原基线 GPU smoke 仍为 `689178` 个可训练参数。真实 CLIP
  Adapter 诊断确认零初始化残差非零元素严格为 0；两次独立 GPU forward 的 AMP 最大数值差
  为 `6.1035e-5`、FP32 最大差为 `1.4901e-8`，因此 GPU smoke 使用 `1e-4` 容差，同时 CPU
  单测继续要求逐 bit 相等。
- 修正后的 Adapter GPU1 smoke 已通过：bottleneck64 单层当前任务可训练参数为
  `788314 = 689178 + 99136`，零初始化真实 CLIP AMP logits 最大差为 `3.0518e-5`。该验证
  只执行 batch2 forward/backward，没有加载 EMOTIC 或启动训练。
- task0 validation screen 已于 2026-08-11 15:30 启动，batch ID 为
  `task_lane_adapter_task0_seed0_20260811_153033`。四组 bottleneck `32/64` × Adapter LR
  `1e-4/4e-4` 分别运行在 GPU 1/2/3/4；均使用 seed0、30 epochs、batch64、base LR
  0.0125、val-only。四组均完成 30 epochs/2520 updates、skipped 0，val mAP 依次为
  `56.792729/58.974777/57.720757/59.697275`；相对 seed0 基线 `57.339154` 的增益为
  `-0.546425/+1.635623/+0.381603/+2.358121`，因此只晋级 b64/lr4e-4。
- b64/lr4e-4 的 task0 三种子确认已启动：复用 screen 的 seed0 结果，seed1/2 分别在 GPU1/2
  运行，run ID 前缀为 `task_lane_adapter_task0_confirm_b64_lr4e4_20260811_155343`。两组在停止
  信号到达前已正常完成，seed1/2 val mAP 为 `59.329355/56.178599`，相对各自基线均提升。
  按老师意见，后续不再把额外 task0 多 seed 确认作为正式实验前置条件。
- 新增 b128/lr4e-4 seed0 task0 val screen，GPU3、30 epochs，其每任务 Adapter 参数为
  `197504`、当前总可训练参数 `886682`。其 val mAP 为 `56.824802`，比 b64 的
  `59.697275` 低 `2.872474`，因此淘汰 b128，正式配置确定为 b64/lr4e-4。
- 正式三种子实验 `multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927` 已在
  GPU1/2/3 启动，对应 seed0/1/2。配置为完整 8 tasks、30 epochs/task、batch64、base
  LR0.0125、Adapter LR4e-4、b64/layer11/ReLU/scale0.1、AMP/TF32、held-out test、阈值0.5；
  运行提交固定为 clean `6e8b15d`。三组均完成 240 epochs/8 tasks、skipped0、exit code0。
- Adapter 三种子均值为 final mAP `31.128461`、final cF1 `31.994054`、final oF1
  `48.878881`、average mAP `38.599579`、forgetting `4.869595`。相对无 Adapter 严格复现
  分别变化 `-0.171025/+0.182944/-0.230322/+0.601003/+0.081138`；forgetting 越低越好，
  因而当前 Adapter 提高了阶段平均 mAP 和 cF1，但没有提高最终 mAP/oF1/forgetting。
- 正式结果已生成无 checkpoint 的17文件白名单压缩包，SHA-256
  `16e0a9fcf9dc7d66dbf8ac5ba0a681539a400fdcf0f53916bd122071818946e1`，并下载、校验、解压到
  本地 `./output/emotic_track_a_adapter/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`。
- 逐 task 诊断表明不是运行失败：test mAP 在 task0–5 均高于基线，但 task6/7 转为
  `-0.233741/-0.171025`；task6 新引入三类的平均 AP 在引入时下降 `4.514579`，主要来自
  Sadness `-6.289815` 和 Suffering `-7.842116`，而不是训练后覆盖旧 lane。
- Adapter 每个新任务独立随机初始化且不继承上一任务；task6 训练 view 只有627样本、
  10 updates/epoch、总计300 updates，其最终 up-weight norm `2.494252` 也是8个 task中最低。
  所有 task 的训练 BCE 都更低，但 task3/5/6/7 val mAP 更低，说明当前 BCE/零起点 Adapter
  在低数据后期任务上出现目标错配或欠适应，而非数值爆炸。
- 当前实现还存在严格对照混杂：Adapter Bank 在 DataLoader 创建前消耗全局 PyTorch RNG，
  导致同 seed 的 Adapter 与基线、不同 bottleneck 之间使用不同 shuffle/augmentation 随机
  序列。三种子同向结果仍说明权衡具有一致性，但下一轮必须先隔离 Adapter 初始化 RNG，
  再评估方法本身。
- 当前版本已增加 `independent/copy_previous` 两种 task 初始化策略；warm-start 模式在新
  task 激活时复制上一 task Adapter 的完整参数，再仅解冻当前副本，旧 Adapter 继续冻结。
  默认仍为 `independent`，保证上一轮实验配置可复现。
- Adapter Bank 构造已放入独立 torch RNG context，构造后恢复全局 RNG state；新增单元测试
  对比 disabled/adapter 模型构造后的 RNG state。新增 seed0 完整8-task val-only入口
  `run_multilane_track_a_adapter_full_val.sh`，固定不读取test。实现提交 `313618c` 已推送并在
  服务器 fast-forward 同步；服务器10/10单元测试及GPU1 `copy_previous` Adapter smoke均
  通过。seed0完整8-task val-only warm-start已在GPU1完成，run ID为
  `task_lane_adapter_full_val_seed0_b64_lr0.0004_copy_previous_20260811_174404`；不读取test，
  GPU2的同seed/同配置clean disabled val-only配对对照
  `multi_lane_disabled_full_val_seed0_paired_clean_20260811_174847`也已完成。尚未启动新的正式
  test实验。
- 配对validation表明warm-start在task0–5 mAP均提升，但task6/7分别下降
  `0.334822/0.448594`；task6新类平均AP下降 `6.165031`，Sadness/Suffering分别下降
  `8.788894/10.703310`。average mAP虽提升 `1.021565`，final cF1提升 `0.482067`且forgetting
  改善 `0.079821`，但final mAP和oF1分别下降 `0.448594/0.770218`。
- 按实验前固定判定规则，task6 mAP、task7/final mAP、task6新类平均AP三项继续条件均失败。
  当前task-lane Adapter不适合解决MULTI-LANE后期低数据任务问题，停止继续扩大bottleneck或
  增加层数；结果保留为方法消融，不再使用正式test调参。
- 配对分析实现以 `3d7c5aa` 提交推送并在服务器同步，13/13单元测试通过。训练history表明
  warm-start在所有task的最终BCE更低，但新类平均AP从task3开始多数下降，task6降幅最大。
  task6仅300次更新，最终Adapter与task5余弦相似度 `0.9841`；连续复制使Adapter范数从task0
  `12.3918`累积至task6 `17.8840`。结合selectors/prompts本已复制，问题是重复继承造成的
  prior-task偏置和低数据可塑性不足，而非初始化RNG或单纯容量不足。

### 当前仓库内的 EMOTIC Track-A 严格复现

- 新增 `multi_lane/track_a/`，在当前仓库内实现目标 benchmark 使用的冻结 OpenAI
  CLIP ViT-B/16 MULTI-LANE 路径；历史 `clip_vit_b16_patch` 和 `main.py` 默认行为不变。
- OpenAI CLIP checkpoint 固定为 SHA-256
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；加载器不依赖
  DDP worktree 或运行时网络下载。
- Track-A 模型保留 10 selectors、10 组逐层 K/V prompts、前 5 层 prompt、
  drop-and-replace、上一任务 slice 初始化、共享 512 维 projection 后分类器和 concat
  inference；可训练参数应严格为 `689178`。
- 冻结协议为 EMOTIC B5-C3、8 tasks、30 epochs/task、seed 0/1/2、batch 64、
  Adam LR 0.0125、每任务重置 cosine scheduler、AMP/TF32、held-out test、固定阈值 0.5。
- 三种子 launcher 先运行 GPU smoke，再在 GPU 0/1/2 并行运行；日志进入
  `./logs/emotic_track_a/`，项目汇总进入 `./output/emotic_track_a/`，完整产物进入服务器
  `emotic_benchmark_runs`。
- 目标注册结果 final mAP/cF1/oF1/average mAP/forgetting 为
  `31.2995/31.8111/49.1092/37.9986/4.7885`。
- 服务器预检已通过：4/4 单元测试通过；当前 loader 与目标 loader 的 152 个视觉权重及
  同输入输出最大误差均为 `0.0`；GPU forward/backward smoke 通过，可训练参数严格为
  `689178`，冻结视觉塔没有梯度。
- 正式实验 `multi_lane_main_track_a_seed012_20260811_132448` 已在 tmux
  `emotic_multilane_track_a_seed012` 完成，seed 0/1/2 分别使用 GPU 0/1/2。最终五项
  三种子均值/标准差与注册实验完全一致：final mAP `31.2994855 ± 0.1410457`、final cF1
  `31.8111103 ± 0.2026319`、final oF1 `49.1092038 ± 0.1330790`、average mAP
  `37.9985757 ± 0.4824589`、forgetting `4.7884572 ± 0.0199469`。
- 已生成不含 checkpoint 的 17 文件下载包，SHA-256 为
  `d3a2f23080da5ff6bd5d0fde99c6ebb9eae1916db31b1e09e9ebb6ed5fb15dcc`，并下载、解压到
  本地 `./output/emotic_track_a/multi_lane_main_track_a_seed012_20260811_132448/`。

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
