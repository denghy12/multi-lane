# 项目上下文

最后一次更新：2026-09-01；当前本地分支为
`exp/emotic-image-token-asl-capacity-lr`。服务器独立实验worktree保持clean`94f3327`；
Image-token Adapter-ASL层位置正式test已完成并同步，zero-based layer1以final mAP
`32.5365`成为新的探索性冠军；首批layer1局部容量/LR、single7 FP32与归一化多层结构15组
及追加5组每层b32多层容量上界正式test均已完成并同步；layer1 Adapter LR×scale阶段一
8组已完成并同步，阶段二ASL局部8组正式test正在并行运行。额外
test-only worktree保持原分支和原有未提交状态，未被本轮同步、分析或实验修改。

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
- 用户确认单卡显存允许并行多进程后，没有中止已运行到task4的首轮8组；从同一`94f3327`
  创建临时detached clean worktree`/mnt/haoyuan/workspace/multi-lane-main-asl-capacity-lr-parallel`，
  将剩余5组立即放到GPU0--4。现在13组全部同时运行，GPU0--4双进程总占用约5.9--6.1GB、
  仍余约18GB；新增5组task0 epoch1均为84 steps、skipped0、ASL loss有限且无异常。
- 为后续可复现，launcher已改为默认一次启动13个进程：GPU0--4各两组、GPU5--7各一组，并以
  8GB显存门槛保护启动。当前批次仍全部记录同一实现commit/tree`94f3327`；原总launcher随后
  对已存在的第二轮目录会安全失败退出，因此本批完成后需基于13份完整产物手动运行严格汇总器。

- 阶段一13/13组现已全部完成并同步本地：每组均为seed0完整8-task validation-only、240 epochs、
  13950 updates、skipped0，AMP/TF32均关闭，无OOM、NaN、Inf、Traceback或RuntimeError；未保存
  checkpoint、未读取held-out test。结果包位于
  `output/emotic_image_token_tuning/asl_capacity_lr/image_token_asl_capacity_lr_seed0_20260826_164945_results_package_v2/`，
  压缩包为`image_token_asl_capacity_lr_seed0_20260826_164945_results_v2.tar.gz`。
- fresh disabled BCE的final/average mAP为`42.0628/48.5032`。绝对最高为bottleneck16、LR4e-4的
  Adapter-ASL：`42.5718/49.5165`，但相对同批b32/LR4e-4锚点虽有final mAP `+1.1516`，
  Sensitivity下降`2.0188`且forgetting恶化`0.0678`，不能作为合格候选；b64/LR2e-4虽有
  `42.3616` final mAP且forgetting较锚点改善，但Sensitivity仍下降`1.5427`。
- 唯一通过锚点全部单项门槛的是b32/LR1e-4，但相对fresh disabled的task6 mAP、Sadness、
  Suffering仍分别下降`0.0235/7.7775/1.1387`；严格汇总`winner_label=null`、
  `eligible_labels=[]`，不进入scale×activation，也不运行正式test。
- 结论是容量和Adapter LR没有解决ASL的类别收益交换：较大容量/较高LR偏向提高aggregate或
  Suffering，却会牺牲Sadness、Sensitivity或遗忘；低LR仅减轻部分遗忘，未稳定改善task6。
  若继续Adapter，应先做同一DataLoader/worker随机轨迹下的paired复现（disabled、b32/LR4e-4、
  b32/LR1e-4），量化独立运行约`0.64` final mAP差异；若仍不能同时保护三类与forgetting，
  则结束容量/LR调参，转向双视图融合或类别感知校准。
- 2026-08-30重新审计并同步同一batch：13组结果、严格汇总与2026-08-27分析一致。
  新增白名单归档`image_token_asl_capacity_lr_seed0_20260826_164945_results_20260830.tar.gz`，
  SHA-256为`d35c62f6f73e3ba08eadeaa6c6c171a794f54204d1c7e44151347917485863dd`；不含checkpoint。
- 2026-08-31用户将优化目标改为尽快超过历史正式test final mAP`32.4193`，允许个别
  类别、F1或forgetting退化。因此validation final/average mAP最高的b16/LR4e-4与次高
  b64/LR4e-4被选为一次性正式test候选；原严格类别门槛仅作观察指标。
- 两组于2026-08-31 00:35在GPU0/1并行启动，batch ID为
  `image_token_asl_capacity_formal_seed0_20260831_003546`，使用服务器clean`94f3327`。固定seed0、
  8 tasks×30 epochs、batch64、layer8、LR4e-4、scale0.1、ReLU、independent、主模型BCE +
  Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、AMP/TF32 off、reporting=test。
- b64真实CLIP smoke通过；正式运行的b16/b64 task0 epoch1均为84 steps、skipped0，无数值
  或显存异常，GPU占用约3.4/3.0GB。服务器磁盘已满，本轮显式`--no-save-checkpoints`，
  避免重复产生历史每组约2.7GB的checkpoint；小型JSON与日志仍按正常路径保存。
- b16/b64正式test均已完成8 tasks、240 epochs、13950 updates、skipped0，无OOM、NaN、Inf、
  Traceback、RuntimeError或写盘错误。b16的final/average mAP为`32.0269/38.6674`，b64为
  `31.3193/38.4295`，相对历史b32冠军`32.4193/38.8789`分别下降`0.3924/0.2115`和
  `1.1000/0.4494`；本轮没有新冠军，b16明显优于b64。
- 8-task轨迹显示b16仅task0超过历史b32，task1--7均略低；b64从task4开始差距扩大，
  task6/7均低约`1.10` mAP，进一步否定扩大容量。但历史冠军是AMP/TF32 on，新运行是
  AMP/TF32 off，所以不能将所有差异只归因于bottleneck。
- 新结果已不含checkpoint打包同步本地，archive SHA-256为
  `66ef7139375d2913830bc5683f26c378506c3e2430fc7f18b47d0b9bd74d1484`。若继续冲击最高mAP，
  下一个最小对照是只b16/LR4e-4完全对齐历史冠军的AMP/TF32模式；当时尚未启动。
- 用户随后明确要求利用8张4090和单卡多进程容量，将b16 AMP/TF32-on正式test与
  b32 scale×activation 8组validation同时并行，不再等待第一步结果。为对齐目标冠军，
  8组validation也统一AMP/TF32 on，其余固定seed0、8 tasks×30 epochs、batch64、layer8、
  b32、LR4e-4、independent、主模型BCE + Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05。
- 9组于2026-08-31 11:42启动：formal batch为
  `image_token_asl_b16_amp_formal_seed0_20260831_114209`，validation batch为
  `image_token_asl_scale_activation_amp_seed0_20260831_114209`。GPU0同时运行b16 formal与scale0.025/ReLU，
  GPU1--7各一组validation；实际GPU0约5.2GB，其他卡约2.4GB，余量充足。
- 启动核验时9组均已进入task0 epoch2，每轮84 steps、skipped0，loss和Adapter loss有限，
  无OOM、NaN、Inf、Traceback、RuntimeError或写盘错误。全部显式关闭checkpoint；服务器实验
  worktree保持clean`94f3327`。
- 完成审计显示b16 AMP/TF32-on正式test完整完成240 epochs、13950 updates、skipped0，final/
  average mAP为`31.3796/38.8129`。相对历史b32冠军分别下降`1.0397/0.0660`；相对自己
  FP32运行，final mAP也下降`0.6473`。b16在task0--4略高于冠军，但task6/7分别低
  `1.0856/1.0397`，表明关键问题是后期任务容量不足，不是AMP/TF32未对齐。
- scale×activation的8组validation中5组完成全部任务；`0.1/GELU`、`0.2/ReLU`、
  `0.2/GELU`均在task3出现ASL非有限logits并主动终止，失败前已有明显skipped updates。
  三组不纳入指标排名，也不建议为追求高scale改成FP32补跑。
- 有效5组中原始`scale0.1/ReLU`同时获得最高final mAP`42.6673`、average mAP`49.2378`和
  task5--7平均mAP`44.8851`。其他完成组的final mAP低`0.5031--0.7962`；因此保留
  scale0.1/ReLU，停止这两个维度的搜索。
- 43个结果/日志文件已打包同步本地，压缩包160KB，SHA-256为
  `29c87d99ddcc868bd0c1989ec921d5c22c20a36c9a6a6c77c9d8220254241461`，不含checkpoint。下一建议是
  固定scale0.1/ReLU，在b32附近做`{24,28,32,40}` × LR`{3e-4,4e-4}` 8组validation。
- 用户随后明确以最快获得最高test mAP为目标，由于val与test单组均约50分钟，授权跳过
  validation并直接并行8组seed0 held-out test。新批次为
  `image_token_asl_local_capacity_lr_amp_formal_seed0_20260831_124751`，搜索bottleneck
  `{24,28,32,40}` × LR`{3e-4,4e-4}`，GPU0--7各一组。
- 8组固定layer8、scale0.1、ReLU、independent、主模型BCE + Adapter ASL 9.8/0/0.05、
  seed0、8 tasks×30 epochs、batch64、CLIP normalization、crop0.05、AMP/TF32 on、reporting=test、
  无checkpoint；b32/LR4e-4是同批冠军锚点。
- 启动时GPU已有其他任务，每卡仍余8.6--11.8GB；本实验另占约2.4--3.0GB，没有停止或
  改动其他进程。启动后每卡仍余6.2GB以上；8组均已到task0 epoch3--4、84 steps、skipped0，
  loss有限，无OOM、NaN、Inf、Traceback、RuntimeError或写盘错误；实验worktree保持clean`94f3327`。
- 8组局部容量/LR正式test均已完成8 tasks、240 epochs、13950 updates、skipped0，无数值、显存或
  写盘异常。同批b32/LR4e-4的final/average/cF1/oF1/forgetting及8-task曲线与2026-08-13历史
  冠军完全一致，final mAP仍为`32.4193`；这证明管线精确可复现。
- 最接近候选b24/LR3e-4的final/average mAP为`32.2492/39.1457`，相对冠军为
  `-0.1702/+0.2668`，task5--7平均mAP也略高`0.0104`。但它在task6类别组均AP低`3.5030`，
  其中Suffering/Sadness低`8.2892/2.8459`，因而最终mAP未胜出。
- 其余按final mAP依次为b28/LR4e-4`32.1666`、b32/LR3e-4`32.1172`、b40/LR4e-4
  `31.9602`、b28/LR3e-4`31.8790`、b24/LR4e-4`31.2931`、b40/LR3e-4`31.1764`。
  容量与LR有强交互，统一超参网格已没有稳定超过b32/LR4e-4的趋势。
- 41个结果/日志文件已打包同步本地，压缩包180KB，SHA-256为
  `acc4e0682ecc6ff170dc92e1663dafb4d94e97cc5aa7d56c03696b440dc1a978`，不含checkpoint。下一方向应在
  b32结构上局部微调ASL，或利用Adapter按任务独立的特性实现task-dependent容量/LR。
- 2026-09-01完成此前只做过validation的Image-token层位置正式test补测。候选为单层
  zero-based`1/2/3/4/5/7/11`与多层`[2,3]`，其余固定b32/LR4e-4/scale0.1/ReLU、
  主模型BCE + Adapter ASL 9.8/0/0.05、AMP/TF32 on。single1/2/3/4/5/11完整完成且
  skipped0；single7在task3因非有限ASL logits失败；pair`[2,3]`虽完成但task3跳过91次更新。
- single1的final/average mAP为`32.5365/39.1445`，相对精确复现的layer8冠军提高
  `0.1172/0.2656`；cF1/oF1提高`0.4167/0.2811`，forgetting降低`0.0934`。8个task mAP
  均为正增益，主要来自task4/5的`+0.5355/+0.6315`，因此layer1成为当前探索性正式冠军。
- 其余完整候选final mAP依次为single4`32.0577`、single5`31.9281`、single2`31.9014`、
  single3`31.3633`、single11`31.0015`、pair`[2,3]``30.8747`，均未超过旧冠军。
  single1仍存在类别交换，最终Suffering/Sadness分别低约`7.0468/4.9255` AP；当前只追总mAP，
  因而不据此否决。
- 旧层搜索为AMP off，而本轮正式test为AMP on；6个可配对单层的validation/test final mAP
  Pearson相关仅`-0.273`。layer3从validation第一变为test较差，说明旧validation排序不能
  直接外推AMP test。新冠军只高`0.1172`且从多组test筛选，属于探索性小效应，不能宣称显著。
- 39个JSON/日志文件已同步本地且无checkpoint；192KB压缩包SHA-256为
  `d8e22c7c8cab09af697ce9c705530af5446c7de51b168d2b3049037cf46380c4`。详细分析见
  `output/emotic_track_a_image_token_asl_layer_formal/image_token_asl_layer_formal_seed0_20260831_154234/analysis.md`。
- 用户随后要求并行推进三条正式test路线，批次为
  `image_token_asl_layer1_local_multilayer_formal_seed0_20260901_100337`。共同固定EMOTIC seed0、
  8 tasks×30 epochs、batch64、主模型BCE、Adapter ASL 9.8/0/0.05、ReLU、independent、
  CLIP normalization、crop0.05、TF32 on、reporting=test、无checkpoint。
- single7补测固定layer7/b32/LR4e-4/scale0.1并使用FP32。layer1局部搜索使用AMP，运行
  b24/b28/b32/b40与LR3e-4/4e-4的7个尚未完成组合，复用已完成的b32/LR4e-4作为冠军锚点。
- 多层7组均使用FP32：容量与残差强度双归一化的`[0,1]/[1,2]/[1,4]/[1,8]`
  （每层b16/scale0.05）；`[1,2]` b32/scale0.05容量控制；`[1,2]` b16/scale0.1强度控制；
  `[0,1,2]` b12/scale1/30三层归一化。该设计区分分布到多层、参数翻倍和残差介入增强。
- FP32/TF32-on smoke因TF32重复前向差异约`1.1e-5--2.2e-5`超过纯FP32严格容差；改用
  FP32/TF32-off做结构不变量验证后，single7及所有代表性双层/三层均通过，最大初始差约
  `6.52e-8`。正式训练仍按计划保持TF32 on并由非有限检查兜底。
- 15组在clean`94f3327`上同时启动；GPU0--6各两组、GPU7一组。全部已生成config并进入task0，
  每卡总显存约2.6--5.5GB、最低仍余18.6GB，启动核验无OOM、非有限值或Traceback。
- 首批15组随后全部完成240 epochs、13950 updates且无异常。layer1局部7组均未超过single1
  b32/LR4e-4冠军；最佳为b32/LR3e-4的`32.1281`。single7 FP32为`31.7931`，补测未取胜。
- 首批多层最佳为`[1,2]` b16/scale0.05，final mAP`32.3233`，仍低冠军`0.2132`；其次
  `[1,4]` b16/scale0.05为`32.2490`。已有`[1,2]` b32/scale0.05仅`31.7764`，初步显示
  多层增大容量反而退化。
- 按用户要求继续补齐每层b32多层容量上界：`[0,1]/[1,4]/[1,8]` scale0.05、
  `[0,1,2]` scale1/30，以及`[1,2]` scale0.1，共5组FP32/TF32-on正式test。三层b32
  FP32/TF32-off严格smoke通过，初始最大差`1.49e-8`；5组已在GPU0--4启动，仍无checkpoint。
- 首批15组的75个JSON/日志文件已单独打包同步本地，328KB archive在两端SHA-256均为
  `b6eb43e335ec0031f36d022ac701ff80aba0f78c29be27af393e906221169cbb`，不含正在运行的5组
  b32目录和任何checkpoint。完整分析位于
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_initial15/analysis.md`。
- 首批结论已收口：layer1 b32/LR4e-4是尖锐局部最优；归一化多层最佳`[1,2]` b16/s0.05
  在8个task上仍全部低于single1；single7 FP32在task6/7下降`0.7912/0.7434`。追加5组只作为
  每层b32容量上界证据，不改变当前冠军或继续扩大结构的优先级。
- 追加5组每层b32全部完成240 epochs、13950 updates、skipped0。最佳`[1,4]` b32/s0.05
  final mAP`32.2213`，低冠军`0.3152`；其余`[0,1]`、三层`[0,1,2]`、`[1,8]`、
  `[1,2]` b32/s0.1分别为`32.0605/31.9493/31.7306/31.6235`。
- 与小容量对应组相比，b32对`[0,1]`和三层分别提高`0.1816/0.1629`，对`[1,4]`近似持平，
  对`[1,8]`和`[1,2]` s0.1分别下降`0.3211/0.1307`。容量效应无稳定方向，且没有一组
  超过多层最佳`[1,2]` b16/s0.05或single1冠军；多层与容量扩张正式收口。
- 新增5组25个小文件已同步本地，无checkpoint；112KB archive两端SHA-256为
  `72995ac436e75d0f9d51eff3da2c0c1c0823cbc8051b47c4b13d3b7e8eb37698`。分析位于
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_b32_extension5/analysis.md`。
- 用户因validation与test耗时相近，明确要求layer1阶段一8组直接运行seed0完整held-out test。
  批次`image_token_asl_layer1_lr_scale_formal_seed0_20260901_171236`固定layer1/b32/ReLU/
  independent、主模型BCE + Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、
  AMP/TF32 on、无checkpoint；联合测试LR4e-4×scale0.075/0.1/0.125、LR5e-4×三种scale，
  以及LR6e-4×scale0.075/0.1，共8组。
- 启动前GPU0--7全空闲、无残留runner，服务器实验worktree为clean`94f3327`；GPU0--7各一组。
  8组均已生成config并进入初始化，每卡约1.8--2.5GB、仍余21.6GB以上，无启动异常。
- 阶段一8组均完成240 epochs、13950 updates、skipped0。同批LR4e-4/scale0.1精确复现
  历史single1冠军`32.5365`并保持final mAP第一；次优LR5e-4/scale0.075为`32.3851`，虽
  average mAP高`0.1017`，但task5--7转负、final低`0.1514`。LR6e-4两组至少低`0.4977`，
  不向7e-4扩展；阶段二固定LR4e-4/scale0.1。
- 阶段一40个JSON/日志小文件已同步且无checkpoint；180KB archive SHA-256为
  `403004fba2f2244dc9c1f630c1f1490f89cc256dfe734d808a9cda64a941bc89`。分析位于
  `output/emotic_track_a_image_token_asl_layer1_lr_scale_formal/20260901_171236/analysis.md`。
- 阶段二批次`image_token_asl_layer1_asl_local_formal_seed0_20260901_182542`直接运行8组正式
  test：锚点9.8/0/0.05；gamma-neg 8/12；clip 0.0375/0.075；gamma-pos 0.25/0.5/1.0。
  其余固定layer1/b32/LR4e-4/scale0.1/ReLU/AMP+TF32、无checkpoint，GPU0--7各一组。
- 2026-08-31经用户明确批准，已删除8个明确失败/中止实验的`checkpoints/`目录，保留其配置、
  日志、指标和父目录，实际释放`6796956 KiB`（约6.48 GiB）。删除前再次验证与当前
  Python训练进程无关；删除后8个目标均不存在，b16/b64正式test tmux仍正常。ext4普通
  `df`仍显示100%/非root可用为0，包含预留块的实际空闲约139.3 GiB。

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

## 2026-09-01 Image-token Adapter layer1 ASL局部搜索结果

- 阶段二8组seed0完整held-out test已经结束并同步；固定layer1/b32/Adapter LR4e-4/
  scale0.1/ReLU，局部比较gamma-neg、gamma-pos和clip。7组为240 epochs、13950 updates、
  skipped0；gamma-pos1在task3由AMP GradScaler跳过11次update，但仍完整产出结果。
- 原ASL `gamma_neg=9.8,gamma_pos=0,clip=0.05`在final mAP`32.5365`、average mAP
  `39.1445`及8个逐task mAP上全部排名第一。第二名gamma-pos1的final mAP为`32.1476`，
  仍低`0.3889`；其他变体低`0.4191--0.8605`。
- gamma-pos0.5改善最终Sadness/Sensitivity/Suffering约`1.02/0.95/0.72 AP`，但其他类别
  损失使final mAP下降`0.5561`；gamma-neg12虽提高F1，final mAP仍下降`0.4191`。
  ASL局部搜索收口，当前冠军参数不变。
- 小结果包SHA-256为
  `c8258d3215747c4dfd7abc30c35c004e6b9196ea230d394e1a8c2ae71f665691`，不含checkpoint。
  详细分析见`output/emotic_track_a_image_token_asl_layer1_asl_local_formal/20260901_182542/analysis.md`。

## 2026-09-01 Image-token Adapter阶段三/四实现

- 新实验分支为`exp/emotic-image-token-stage34`。阶段三新增每个task独立的bottleneck配置，
  阶段四新增Adapter专属weight decay；未指定新参数时保持旧行为。
- 非统一bottleneck使用`adapter_bottleneck_dim=32`作为初始化随机轨迹参考。改变task0容量时，
  task1--7的b32 Adapter仍与uniform b32锚点获得完全相同的初始权重，避免容量与初始化漂移混杂。
- 非统一bottleneck只允许`independent`初始化；`copy_previous`因模块形状不同会在构造时明确拒绝。
  config新增逐task bottleneck与参数量列表，以及Adapter专属weight decay记录。
- 阶段三/四合并为11组seed0完整held-out test：两组task0 b24/b28其余b32，加uniform b32下
  主模型实际LR`0.010/0.0125/0.015`×Adapter weight decay`0/1e-5/1e-4`九组。
  uniform b32/LR0.0125/WD0为共用锚点；固定layer1、Adapter LR4e-4、scale0.1、ReLU、
  ASL 9.8/0/0.05、AMP/TF32 on，无checkpoint。
- 实现已提交推送为`exp/emotic-image-token-stage34@8e56b8a`，服务器独立worktree为
  `/mnt/haoyuan/workspace/multi-lane-main-stage34`。服务器21项单元测试、uniform b32与
  task0-b24两种真实CLIP Adapter-ASL GPU smoke均通过。
- 正式batch为`image_token_asl_layer1_stage34_formal_seed0_20260901_203100`。11组config均记录
  clean`8e56b8a`及正确逐task容量、实际主模型LR和Adapter weight decay；GPU0--2各两组、
  GPU3--7各一组。启动后每卡仍余约19.5--22.0GB，11组均到task0 epoch4以上、84 steps/epoch、
  skipped0，无OOM、非有限值或Traceback。
- 11组随后全部完成240 epochs、13950 updates、skipped0，没有新冠军。uniform b32、实际主模型
  LR0.0125、Adapter WD0精确复现`32.5365/39.1445`并保持final mAP第一。
- 阶段三task0 b24/b28分别把task0提高`0.5923/0.2732`，但final mAP降低`0.5427/0.4997`。
  后续Adapter初始化已与锚点对齐，退化来自task0对selectors/prompts复制轨迹的路径依赖；
  task0-only容量路线停止。
- 阶段四最接近候选为实际主模型LR0.015 + Adapter WD1e-5，final mAP`32.5023`，仅低冠军
  `0.0343`；average mAP/cF1/oF1分别高`0.5252/0.5704/0.1764`。它在task0--6逐项提高，
  仅task7低`0.0343`，成为下一轮唯一值得局部微调的方向。
- 结果小包SHA-256为`bde0e9746985cb74751cc946332b8ada3e823401a7ea5bb1db732def30fcecfd`，
  无checkpoint。详细分析见`output/emotic_track_a_image_token_stage34_formal/20260901_203100/analysis.md`。

## 2026-09-01 全局主模型LR×Adapter WD局部直接test

- 用户因validation与test耗时相近，明确要求将下一轮统一超参数局部网格直接运行完整test；
  所有结果继续标记为exploratory test-tuning，不作为无偏最终评估。
- 新分支`exp/emotic-image-token-global-lr-wd`固定Image-token layer1/b32、Adapter LR4e-4、
  scale0.1/ReLU/independent、主模型BCE + Adapter ASL 9.8/0/0.05、AMP/TF32 on。
- 搜索对所有task完全相同的实际主模型LR`0.014/0.0145/0.015/0.0155`与Adapter WD
  `3e-6/1e-5`八组，增加LR0.0125/WD0同批锚点，共9组seed0完整8-task test，无checkpoint。
- 固定更新预算与全局Adapter正则化暂不并行；它们必须等本批选定统一LR/WD后再逐阶段运行，
  避免优化器与训练预算/方法改动同时变化而无法归因。
- 实现已提交推送为`exp/emotic-image-token-global-lr-wd@4b45dd9`，服务器独立worktree为
  `/mnt/haoyuan/workspace/multi-lane-main-global-lr-wd`，21项单元测试通过。
- batch`image_token_asl_layer1_global_lr_wd_formal_seed0_20260901_224909`已排入tmux
  `image_token_global_lr_wd_20260901_224909`。启动时8卡被其他任务占用、仅余4.5--5.1GB；
  launcher要求8卡连续两次至少12GB空闲后才原子启动9组，因此当前安全等待且未创建训练进程。
- 该batch随后自动启动。8组已完整结束，均为240 epochs、13950 optimizer updates、
  skipped0。原锚点LR0.0125/WD0仍以final mAP`32.5365`暂列第一；最接近候选
  LR0.015/WD1e-5为`32.5023`，仅低`0.0343`，但average mAP/cF1/oF1分别提高
  `0.5252/0.5704/0.1764`。
- LR0.0155/WD1e-5原进程在task6开始时因GPU0被外部进程突然占用而OOM；这是显存资源冲突，
  不是ASL数值故障。已在GPU1从头补跑为`..._lr0155_wd1e5_retry1`，当前进行中。
- 已先同步8组完整结果及全部原始日志到本地；partial archive SHA-256为
  `4be9b24748df8dc32f3a4759b31157b1e44b2384b1e3c690f43b6372234356b4`，不含checkpoint。
  在补跑完成前，9组最终排名仍未封口；所有结果均属于exploratory test-tuning。
- LR0.0155/WD1e-5补跑已完整结束：240 epochs、13950 updates、skipped0，final/average mAP为
  `31.8789/39.2960`，未超过冠军且final排名第8。原锚点LR0.0125/WD0继续以`32.5365`保持第一，
  LR0.015/WD1e-5以`32.5023`保持第二。
- 9组最终小结果包SHA-256为
  `724611f09c4df5501257b92238790cb4309074571828c692eec09862767a2fba`，包含完整JSON、原始日志、
  失败诊断和clean retry日志，不含checkpoint。LR/WD局部搜索正式收口；下一优先级是固定每task
  optimizer updates的全局统一预算实验。

## 2026-09-02 Image-token Adapter训练机制三阶段并行实现

- 新实验分支为`exp/emotic-image-token-training-mechanisms`。用户要求将固定updates、Adapter输出
  正则和learnable residual gate三阶段同批并行；为保持可归因性，阶段二/三均以当前30-epoch
  冠军为锚点，不预先假设阶段一未知的最佳预算。若阶段一产生新预算，只对各阶段赢家补组合确认。
- 阶段一共8组：原30 epochs control，以及所有task统一使用成功optimizer updates
  `900/1200/1500/1800/2100/2400/2700`。update模式循环当前task数据直到精确成功步数，cosine
  scheduler按成功step推进；相同预声明规则用于全部task，不读取未来task难度。
- 阶段二共6组：冻结原结构和30-epoch预算，分别测试normalized Adapter residual ratio与adapted/
  frozen token cosine distance，目标贡献比例为Adapter ASL的`1%/3%/10%`。每task统一先用30个成功
  updates校准量级，再固定该task正则权重；规则、warm-up和目标比例对所有task相同。
- 阶段三共4组：每task独立learnable sigmoid gate，统一初始输出比例
  `0.025/0.05/0.1/0.2`，新task使用相同初始化；旧task gate随Adapter一起冻结，不会被后续task
  的optimizer动量改写。
- 三阶段去重后18组，固定seed0、Image-token layer1/b32、main LR0.0125、Adapter LR4e-4、WD0、
  ReLU/independent、main BCE + Adapter ASL9.8/0/0.05、AMP/TF32 on、完整8-task exploratory test、
  无checkpoint。8卡一次启动，每卡2--3组，启动前要求每卡连续两次至少18GB空闲。
- 本地静态编译、shell语法和diff检查已通过；本地系统Python缺少Torch/NumPy，完整单元测试与
  三种真实CLIP GPU smoke需在服务器`ddp`环境执行后才能启动正式batch。
- 实现提交已推送为`exp/emotic-image-token-training-mechanisms@2cafb04`，服务器独立worktree为
  `/mnt/haoyuan/workspace/multi-lane-main-training-mechanisms`。服务器`ddp`环境50项完整单元测试
  全部通过；control、residual10%、cosine10%与learnable-gate0.1四种真实CLIP AMP/TF32 smoke
  全部通过，trainable参数分别为739130/739130/739130/739131。
- 计划正式batch为`image_token_asl_layer1_training_mechanisms_formal_seed0_20260902_150731`；结果写
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_training_mechanisms_formal_v0.1/`，
  日志写独立实验worktree的`logs/emotic_track_a_image_token_training_mechanisms_formal/`。
- 验证上下文已follow-up提交为`5505a36`，服务器实验worktree已fast-forward并保持clean。正式batch
  已在tmux`image_token_training_mechanisms_20260902_150731`一次启动18组；全部config记录clean
  `5505a36`且矩阵字段核验正确。
- GPU0/1各3进程、GPU2--7各2进程，启动后每卡约占4.2--6.7GB，仍余17.4--19.9GB，无OOM、
  non-finite或Traceback。control task0前5个cycle的loss/LR/Adapter loss与历史冠军逐项一致。
- 六个正则组均在task0前30个成功updates校准后产生非零固定权重；1%/3%/10%的权重严格成比例，
  首cycle正则loss亦随目标比例单调增加。当前让18组继续运行，不启动其他实验。
- 18组随后全部完成，launcher退出码18/18为0；所有配置status complete、skipped0，无错误日志或
  checkpoint。固定预算组实际总updates精确等于目标×8，其他11组均为13950 updates。
- 三阶段均未超过control`32.5365`。固定updates最佳1800仅`30.8757`（-1.6608）；输出正则最佳
  residual1%为`32.0751`（-0.4614）；learnable gate最佳init0.05为`32.3397`（-0.1969）。
- 固定updates失败的根因是等epoch已经等化每样本曝光；等updates反而让1800步对应task0--7约
  `21/26/129/12/26/45/180/75`次数据遍历，同时欠训练大task3并过度重复小task2/6/7。
- 正则确实压低Adapter扰动，但所有强度均降低final，说明当前residual主要承载有效判别信号；gate0.05
  最终学习到`0.055--0.086`，仍低于固定0.1并损失后期mAP。没有阶段赢家，不运行组合实验。
- 小结果包SHA-256为`fc051087a60a293ae06824396c7981c8cbeaeb9f04a24d0cd1226981a81c003a`，
  不含checkpoint。详细分析见
  `output/emotic_track_a_image_token_training_mechanisms_formal/20260902_150731/analysis.md`。

## 2026-09-02 Image-token Adapter统一epoch搜索

- 用户确认把统一数据暴露作为最后一轮单视图参数优化，固定现有冠军全部结构、loss、优化器与
  预处理，只搜索所有task共同的`epochs/task={18,22,26,30,34,38,42,48}`。该规则不使用task
  编号、未来数据规模或已知难度，符合未知增量任务设定。
- 新分支`exp/emotic-image-token-epoch-search`固定seed0、完整8-task exploratory test、full image、
  CLIP normalization、crop0.05、Image-token layer1/b32/LR4e-4/scale0.1/ReLU/independent、main
  LR0.0125/WD0、main BCE + Adapter ASL9.8/0/0.05、batch64、per-task cosine、AMP/TF32 on、
  no-checkpoint。30 epochs作为同批锚点，只按final mAP排名并用average mAP破平。
- 新增单组epoch runner、8-GPU一组一卡原子launcher和严格汇总器。汇总器要求8组同一clean
  commit/tree、完整task0--7、每task历史长度等于声明epoch、总epochs=`8×epoch`、总updates=
  `465×epoch`、skipped0、协议字段完全一致且无checkpoint；完成后自动生成JSON和Markdown排名。
- 预声明下一步：内部epoch获胜则补其上下3个整数epoch；48获胜则扩展54/60；30保持第一则停止
  epoch搜索并进入scheduler诊断。所有直接test选择继续标记exploratory，最终需paired多seed确认。
- 实现提交`7449b30`已推送。服务器创建clean独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-epoch-search`并跟踪同名远端分支；服务器ddp环境完整53项
  单元测试通过。真实ViT-B/16 Image-token layer1/b32、Adapter-ASL、AMP/TF32 GPU smoke通过，
  trainable参数739130、初始零残差logit最大差`7.63e-05`且所有梯度有限。
- 验证上下文提交`7091da6`已推送并在服务器实验worktree ff-only同步后启动正式batch
  `image_token_asl_layer1_epoch_search_formal_seed0_20260902_171919`，tmux为
  `image_token_epoch_search_20260902_171919`。启动时8卡均余23.6GB以上，连续两次通过18GB门槛，
  一卡一组原子启动；结果写入`/mnt/haoyuan/workspace/emotic_benchmark_runs/`
  `multi_lane_image_token_epoch_search_formal_v0.1/<batch>`，日志写实验worktree的专用logs目录。
- 8份config全部记录clean`7091da6`且除epoch外固定协议一致。八组task0 cycle1均为84 steps、
  skipped0，loss`0.61478711`、Adapter loss`0.02839343`逐项相同；每组约占2.0GB，无OOM、
  non-finite、RuntimeError或Traceback。让本批继续运行，不启动重复batch。
- 8组随后全部完成，exit code全0、skipped0、无错误或checkpoint，严格自动汇总通过。30 epochs
  精确复现并以final/average mAP`32.5365/39.1445`保持第一；26 epochs为第二，final低`0.1684`，
  虽使average提高`0.0495`，但task5/6/7分别下降`0.0619/0.1539/0.1684`。
- 34和48 epochs相对30 epochs在8个task全部下降，final分别低`0.5990/0.9864`；同时末轮训练loss
  继续下降，证明延长训练主要造成泛化退化而非弥补欠拟合。按预声明规则停止epoch搜索，不补相邻
  整数epoch；下一阶段固定30 epochs进入全局统一scheduler诊断。
- 小结果包SHA-256为`460890c5b44eb2788f2254b333a3d3376e3df6a310cbc18431065537d2632f30`，
  不含checkpoint。详细分析见
  `output/emotic_track_a_image_token_epoch_search_formal/20260902_171919/analysis.md`。

## 2026-09-02 Image-token Adapter统一scheduler搜索

- 新分支`exp/emotic-image-token-scheduler-search`固定30 epochs冠军的全部数据、结构、loss、优化器、
  精度与报告协议，只比较8种对所有task统一且每task重置的scheduler：历史cosine锚点、cosine相对
  min LR 1%/10%、cosine线性warmup 5%/10%、linear、constant、multistep 60%/85%。
- scheduler同时对主模型和Adapter参数组使用同一相对倍率，避免将主模型绝对min LR错误用于Adapter。
  30 epochs下5%/10% warmup向上取整为2/3 epochs；multistep在epoch18/26以gamma0.1衰减。
  历史cosine锚点继续使用原生`CosineAnnealingLR(T_max=30)`，保持旧冠军数学路径不变。
- runner新增scheduler CLI、严格参数校验和完整config元数据；固定update预算仍只允许旧cosine，避免改变
  已有实验语义。新增LR曲线单测、单组正式runner、8-GPU原子launcher和严格汇总器；汇总器除完整
  协议外逐task逐epoch核验主模型/Adapter的当前及下一LR轨迹。
- 本批仍为seed0完整8-task held-out test、AMP/TF32 on、no-checkpoint，final mAP优先、average mAP
  破平，继续标记exploratory。若历史cosine保持第一，停止单视图scheduler路线并转双视图融合。
- 实现提交`58eb233`已推送，服务器新建clean独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-scheduler-search`。ddp环境完整60项单元测试全部通过；8种
  30-epoch主模型/Adapter LR关键点审计与预期一致。真实ViT-B/16 Image-token layer1/b32、
  Adapter-ASL、AMP/TF32 GPU smoke通过，trainable参数739130且梯度有限。
- 验证提交`7337f96`已推送并在服务器实验worktree ff-only同步。正式batch
  `image_token_asl_layer1_scheduler_search_formal_seed0_20260902_185259`已在tmux
  `image_token_scheduler_search_20260902_185259`一组一卡启动，结果写benchmark外部目录，日志写
  scheduler工作树专用logs目录。
- 8份config均记录clean`7337f96`、30 epochs、冠军固定协议和正确scheduler字段。非warmup六组
  task0 cycle1的loss/Adapter loss均为`0.61478711/0.02839343`且LR0.0125；warmup5%/10%首轮
  LR为`0.00625/0.00416667`。八组均84 steps、skipped0，每卡约2GB，无错误，继续运行。
- 8组随后全部完成并通过严格LR轨迹汇总：每组240 epochs、13950 updates、skipped0、exit code0，
  无错误或checkpoint。历史cosine以final mAP`32.5365`保持第一；multistep第二为`32.3170`，虽使
  average mAP提高`0.4074`，但final低`0.2196`。
- multistep在task0--5均提高`0.3190--1.0186`，到task6/7转为`-0.0747/-0.2196`；epoch18/26
  的强衰减对每epoch只有10/24 batches的小任务过早。constant/min-LR使尾部更新过强并过拟合，warmup
  减少早期有效更新，均明显退化。固定原cosine，按预声明规则结束单视图scheduler搜索并转双视图。
- 小结果包SHA-256为`d9d78f47ebfe5a74d1866a1d8e795043da070fb55a3940b16637803ea88e90c9`，
  不含checkpoint。详细分析见
  `output/emotic_track_a_image_token_scheduler_search_formal/20260902_185259/analysis.md`。
- 2026-09-03按用户确认进入person-only阶段，在新分支`exp/emotic-person-only-layer1`实现
  `--person-crop-margin`并将其传入train/val/test EMOTIC数据集及写入config；默认0保持旧实验兼容，
  本轮计划使用0.15。新增person-only正式脚本，固定当前冠军zero-based layer1/b32/Adapter LR4e-4、
  scale0.1/ReLU/independent、main BCE + Adapter ASL 9.8/0/0.05、main LR0.0125、30 epochs/task、
  per-task cosine eta_min0/no warmup、AMP/TF32、seed0、test reporting、无checkpoint，只改变为
  person crop并采用bbox margin0.15与人体训练crop scale0.70--1.0。实现提交`d0f8e81`已推送；
  服务器独立worktree`/mnt/haoyuan/workspace/multi-lane-main-person-only`已通过63项单测、真实
  EMOTIC数据构建和GPU0真实CLIP Adapter-ASL smoke。正式run
  `person_only_layer1_b32_asl_formal_seed0_20260903_121738`已在GPU0启动，clean commit、config和
  前4个epoch的84 steps/skipped0均核验通过，当前继续运行。
- person-only正式run现已完成并同步：240 epochs、13,950 updates、skipped0、exit0且无错误。
  final/average mAP为`28.7815/35.2444`，相对full冠军下降`3.7550/3.9001`；8个task mAP全部
  下降，最终26类AP也全部下降，task6 Sadness/Sensitivity/Suffering分别下降
  `14.5821/1.1381/11.5128`。当前person-only不能替代full，也没有类别级oracle增益。
- 数据诊断确认目标人物歧义真实存在，但当前人物预处理会破坏人体：test 5,368人物样本来自3,682
  张图，2,923样本属于多人物图；人物框绝对长宽比中位数1.561，54.14%超过1.5、27.96%超过2。
  `Resize(short=256)+CenterCrop(224)`及方形RandomResizedCrop会再次截断细长人物框。下一步若继续
  双视图，应先改为pad-to-square/letterbox保全人体并保存validation逐样本score，再判断融合。
- 本地32KB结果包`output/emotic_track_a_person_only_formal/person_only_layer1_b32_asl_formal_seed0_20260903_121738_results.tar.gz`
  包含4个JSON、日志和analysis，不含checkpoint，SHA-256为
  `15e79e5f9dc3f33500b09d65e115b96e21c85608fd420ba16547d611cc150f6a`。
- 2026-09-03从`f2b9fdf`创建`exp/emotic-dual-view-validation-fusion`，开始修正person视图并建立
  validation-only融合诊断。新增显式`letterbox`模式：bbox margin之后使用CLIP均值居中pad-to-square、
  bicubic resize 224、水平翻转和0.10强度/0.20概率ColorJitter，完全移除person分支的
  RandomResizedCrop；`legacy_crop`默认值保留历史复现。
- EMOTIC数据集新增稳定`split:path#person=index` sample ID。runner新增仅允许validation使用的逐task
  `.npz` score dump，包含sample IDs、seen-class logits和targets；离线融合器严格对齐ID/targets并验证
  dump可重算原指标，然后在全局alpha 0--1、步长0.05上比较logit与probability融合。
- 新增两卡validation脚本：GPU0为完整full冠军锚点，GPU1为letterbox person；两者固定seed0、8 tasks、
  30 epochs/task、per-task cosine eta_min0/no warmup、main LR0.0125、Image-token layer1/b32/LR4e-4、
  scale0.1/ReLU/independent、main BCE + Adapter ASL 9.8/0/0.05、AMP/TF32、无test、无checkpoint。
  Python静态编译、shell语法和diff检查已通过；当前代码尚未提交，服务器实验尚未启动。
- Automatic Upload再次把本轮9个文件写入服务器主工作树；SHA-256逐项与本地一致后已完整备份到
  `/mnt/haoyuan/workspace/git-sync-backup-dual-view-code-8ULK5BSO`并恢复服务器主工作树clean。
- 用户于2026-09-03授予后续实验执行的持续授权：用户已明确要求开始某项实验后，可直接完成实验分支
  commit/push、服务器Git-only独立worktree同步、单测/smoke及已声明的非破坏性实验启动，不再重复
  请求审批。合并main、删除产物、覆盖未提交改动、触碰test-only工作树等仍须单独确认。
- 双视图实现已作为`6ab53c8`推送。服务器独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-dual-view`保持clean并通过70项完整单测、2,397条真实val样本
  的稳定ID/letterbox/score-dump smoke及真实ViT-B/16 Adapter-ASL GPU smoke。
- validation batch`image_token_layer1_full_person_letterbox_val_seed0_20260903_134720`已在GPU0/1并行
  启动；两份config均为clean`6ab53c8`、seed0完整8-task val-only、score dump开启、checkpoint关闭。
  full与person的task0首轮均84 steps/skipped0，显存及数值状态正常；两run成功结束后launcher自动
  搜索logit/probability全局alpha并生成`fusion_summary.json`。
- 双视图validation已完成并同步。full/person均240 epochs、13,950 updates、skipped0，8份score dump
  的ID/targets逐task完全对齐。full与person final mAP为`42.3799/38.3044`；最佳为probability
  fusion alpha0.20，final/average mAP为`43.3035/50.4283`，相对full提高`0.9236/1.1631`。
- 融合在task0--7均提高mAP，最终26类有21类提高；Sadness/Suffering分别提高`3.3176/2.9992`，
  Sensitivity下降`1.0548`。alpha0.15--0.35均超过full，结果不是单点尖峰。下一步应锁定alpha0.20
  做一次seed0 held-out test，不在test上重新搜索融合权重。
- 本地完整小结果位于`output/emotic_track_a_dual_view_val/20260903_134720/`，另有不含checkpoint的
  1.6MB压缩包；其SHA-256为`c988359a9b572c0799f3b1953beadfc7ccfabafae085d126dab19561748c2d0c`。
- 2026-09-03按用户要求从validation结果提交创建`exp/emotic-dual-view-formal-test`。正式test规则在
  代码中锁定为probability fusion、full权重0.80、person权重0.20、threshold0.5；固定融合程序必须
  读取并验证原validation `fusion_summary.json`，test端没有alpha/mode/threshold搜索入口。
- runner新增显式`evaluation_score_purpose=fixed_test_fusion`门禁，只有该用途才允许held-out test
  score export；输出到独立`test_scores/`。正式脚本固定seed0、8 tasks、冠军训练协议、person
  letterbox、AMP/TF32和no-checkpoint，完成后只计算一次锁定融合并比较`32.53651921448899`冠军。
- 实现提交`c9fa74c`已推送；服务器新建clean独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-dual-view-formal-test`并通过72项完整单测、固定validation
  selection SHA/无搜索CLI检查和真实ViT-B/16 Adapter-ASL GPU smoke。
- 正式batch`image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816`已挂入tmux
  `multilane_dual_view_formal_20260903_145816`。GPU0/1当前被外部任务占用，launcher按连续两次每卡
  至少18GB空闲且利用率不高于10%的门槛自动等待；不会抢占、重复启动或覆盖结果。

## 2026-09-03 Image-token layer1冠军的seed1/2确认

- 用户要求对当前单视图冠军补跑seed1/2，并同时检查注册MULTI-LANE baseline的seed1/2；存在且
  完整的seed必须直接复用，不能重复训练。
- 历史baseline batch`multi_lane_main_track_a_seed012_20260811_132448`已经包含完整seed0/1/2，
  三种子final mAP分别为`31.3621035/31.3983807/31.1379723`，聚合为
  `31.2994855 ± 0.1410457`，因此baseline不启动任何新run。
- 对服务器`emotic_benchmark_runs`现有165份config进行精确协议匹配后，没有找到冠军配置的
  seed1或seed2完成结果。本次复用seed0 scheduler anchor，使用同一clean commit`7337f96`、
  同一CLIP checkpoint SHA-256 `5806e77c...df416f`，只补跑seed1/2。
- 两组固定完整8-task held-out test、30 epochs/task、batch64、main LR0.0125/WD0、per-task
  cosine eta_min0/no warmup、full image/CLIP normalization/crop0.05、Image-token layer1/b32/
  LR4e-4/scale0.1/ReLU/independent、main BCE + Adapter ASL9.8/0/0.05、AMP/TF32、no-checkpoint。
- batch为`image_token_asl_layer1_formal_seed12_20260903_123014`，seed1/2分别在GPU1/2的tmux
  `multilane_imgtok_l1_s1_20260903_123014`和`multilane_imgtok_l1_s2_20260903_123014`运行；
  GPU0上的person-only保持独立。两份config全字段核验通过，task0前三轮均84 steps、skipped0，
  每卡约2.0GB且loss有限，无OOM或启动异常。
- seed1/2现已完成，exit code均为0；每组240 epochs、13,950 updates、skipped0、完整task0--7，
  配置除seed外一致，日志无OOM/non-finite/Traceback/RuntimeError且没有checkpoint。seed1/2的
  final mAP分别为`32.1347/31.7973`。
- 与既有seed0 `32.5365`聚合后，当前方案三种子final mAP为`32.1562 ± 0.3701`，average mAP
  `38.9260 ± 0.2789`，cF1 `32.3979 ± 0.1642`，oF1 `49.4041 ± 0.0504`，forgetting
  `4.7720 ± 0.0793`。
- 相对注册baseline三种子，final mAP同seed分别提高`+1.1744/+0.7363/+0.6593`，平均提高
  `+0.8567`；average mAP/cF1/oF1分别提高`+0.9274/+0.5867/+0.2949`，forgetting平均改善
  `0.0164`。8个task的平均mAP变化全部为正，说明收益不只来自seed0或某一个task。
- 类别交换仍存在：final Sadness平均提高`+3.8635`，Sensitivity近似持平`+0.0818`，Suffering
  平均下降`-2.7971`且三个seed变化差异较大。整体配置可以作为稳定超过baseline的三种子结果，
  但完整增益同时包含CLIP normalization、crop和Image-token Adapter ASL，不能全归因于Adapter。
- 同步结果、日志、三种子汇总和analysis已保存到
  `output/emotic_track_a_image_token_seed_confirmation_formal/20260903_123014/`；144KB小结果包
  不含checkpoint，SHA-256为`480838060a634d242776a38f698f027cbf1560527e32a24b573f3a9fbe48a330`。
