# 统一1%最终修正上限：四组自行训练的完整八任务控制

2026-10-28用户授权执行公平训练控制，并要求解释并行Summarize查询候选。本轮只实现前者，不把后者混入四组。

## 要回答的问题

原C4把最终幅度从约1%放大到最多10%；现有checkpoint缩幅恢复损失，但常量门控与动态同样好。新实验把最终限幅和初始化统一，重新自行训练，判断Attention是否优于近等参数MLP，以及门控自由度是否必要。只读缩放结果不能直接作为本轮重训成绩。

| 组别 | 修正来源raw | gain | 最终写入 |
|---|---|---|---|
| DOSE_MLP | Full/Person/有效Face最终CLS拼接，隐藏96 MLP | 固定1 | B_.01(raw;h0) |
| DOSE_FIXED | 第6层三视图原全局Summarized Patches联合Attention | 固定1 | B_.01(raw;h0) |
| DOSE_TASK | 同上 | 每任务2sigmoid(s_t)，初始1 | B_.01(gain·raw;h0) |
| DOSE_CONTENT | 同上 | [h0,hPerson,reliable·hFace,r,1]→16→1，2sigmoid，初始1 | B_.01(gain·raw;h0) |

B为原平滑RMS限幅函数，最后一次操作统一保证每个样本最终修正/原融合特征RMS≤1%。gain在限幅前，不能再获得10倍写入空间；这不是所有人的实际修正范数被强行设为相同。三组Attention每任务Q/K/V/视图编码按928100+task独立种子初始化，内容门控额外初始化不影响后续任务的共同参数。所有输出投影零初始化，初始delta=0，gain=1。门控标量可能被投影缩放吸收，须报告实际输出范数、near-cap比例及gate分布，不把gate变化视为成功。

MLP和Attention读入不同（CLS/摘要），是架构对照，不是严格同输入算子对照。MLP约197216/任务，固定Attention198912、任务标量198913、图片门控223553；实际参数数目写入config，不掩盖内容门控多出的容量。均使用相同数据、更新预算、初始化函数、最终上限和原分类头训练策略。

## 完整训练配置

- 数据EMOTIC，seed1，8任务[5,3,3,3,3,3,3,3]。历史samples_intersect_seen_classes，每阶段只评估累计已见类至少一正标签人物，八阶段mAP算术平均；最后Val2397/Test5368人。
- 强底座32.6800/39.5292配方：CLIP ViT-B/16 Frozen Forward冻结；Image-token Adapter b32/index1/scale.03/ReLU，仅原图像证据读取支路；10个任务Selectors、共享三视图Selectors与Prefix Prompts，原三路Task Forward及固定特征融合保留，Face可靠性mask保留，Person颜色扰动关闭。
- 三路原目标：BCE/Adapter-ASL分工，view auxiliary权重.1，ASL负gamma9.8/正0/clip.05；原融合权重可靠Face .64/.16/.20，不可靠Face沿用原Full/Person规则。
- 每任务30epoch、B64、Adam/WD0、cosine/每任务重置，主LR.0125、Adapter LR.0004；新增交互私有Adam/WD0/cosine LR.0004、独立GradScaler/RNG。
- 任务分类头从fresh初始化，由本模型原目标完整训练30轮；新交互目标经同一head的detach权重计算，只更新新分支，不改原Selectors、Prompts、Adapter或head梯度。没有教师、A/B安装、换头或额外适配轮。
- 原每任务训练13950更新/8task，240epoch/组，四组共55800主更新；新分支有对应私有更新。主和私有0跳步为机制门槛，不因精度低跳过后期任务。
- OMP/MKL/model72线程、原AMP初始scale65536/TF32、train workers2/prefetch1，保留历史两个CPU报告seed draw。评估CPU FP32按原batch sigmoid、2单线程spawn workers/prefetch1。
- 交互第6层index5、宽64、4头、单层；不加目标区域约束、额外查询、专家或其他新模块。
- 每任务结束封存私有参数，旧任务参数/旧路径probe审计；原共享函数逐任务与历史参数、30轮loss及完整RNG逐值核对，历史权重只读比较，不加载到学生。

## 路径和入口

以下相对独立服务器实验worktree：

- 配方和历史只读比对：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- 标准视觉预训练：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`，核验历史SHA。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 输出：`output/protected_attention_dose/<run_id>/<arm>/{train,val,test}`；日志`logs/protected_attention_dose/<run_id>`。
- 训练各epoch包含两个优化器、scheduler、scaler、完整RNG的恢复checkpoint；最终紧凑checkpoint与验证前恢复状态；保留前轮来源，不覆盖任何历史结果。

```bash
bash scripts/emotic/run_protected_attention_dose_gpu0.sh smoke_seed1_20261028_01 --smoke --smoke-updates 4
bash scripts/emotic/run_protected_attention_dose_gpu0.sh seed1_20261028_01
```

## 预检、排队与产物

仅GPU0，最大4作业/2评估；启动需余量GPU6000MiB、共享内存1024MiB，间隔12秒。每组训练结束自动并行完整Val/Test，输出ON/OFF及移除新增Face/Person入口。后两种为复合入口干预：会同时改变图片门控输入，不将其直接等同“只删Face memory且固定gate”。残差OFF才是核心原函数对照。每模式记录实际gain/范数/近上限比例。

预声明：新8项机制测试、完整Track-A回归和相关protected/historical RNG/recovery测试；再四组task0/1×4更新真实smoke及Val首64。smoke不跑Test，不以精度筛选正式组。四组原参数/loss/记录的增强batch和OFF/三路预测必须一致，私有梯度非零、初始恒等、无效Face阴性控制、最大写入1%、两优化器恢复通过才启动。

正式跨组用DOSE_MLP的OFF作为配对锚点，四组各任务仍独立核对历史强B0。无需额外重训第五个B0；OFF必须准确回到历史轨迹。预期产物四组完整8阶段Val/Test mAP/F1/逐视图/可靠Face与开关归因、训练/保真审计、资源与参数计数。用户目标是Test最终和阶段平均都改善，并判断Attention相对MLP是否有净优势；不以门控非零、loss下降或大于单个弱控制作为成功。Test参与研究开发，完整报告所有组，不选择最好epoch冒充固定预算结果。

分支`codex/protected-attention-dose-controls-20261028`。仅Git提交/push/fetch/独立worktree同步源码。原main/test-only状态SHA ea2e261e…/e9d1479f…保留；不合并main、不删除数据，不改其他GPU作业。

## 第二件事：并行Summarize查询候选（仅解释，未实现）

MULTI-LANE原论文第3节将Frozen Forward提供的图像tokens记为g，任务Patch Selectors记为s_t，以点积softmax加权和得到Summarized Patches，然后连同任务CLS进入MSA/Prefix Tuning、Drop & Replace和FFN。参见[论文](https://raw.githubusercontent.com/mlresearch/v274/main/assets/min25a/min25a.pdf)。

原三视图使用同一组任务Selectors，分别对Full/Person/可靠Face的tokens执行Summarize，得到三份内容不同的摘要，送入原三路Task Forward，最终固定融合h0。共享的是查询参数，并不意味着三视图摘要相同。

现有新增Attention读取`sg(G_F),sg(G_P),sg(G_R)`，新目标无法改变原摘要的提取。候选新增一组任务私有查询（首版可为10个，与原同数），专用于交互证据提取：

```
同一套冻结ViT权重，分别编码Full/Person/Face
      ↓ 当前层图像tokens
      ├→ 原Selectors → 原Summarize → 原三路Task Forward → hF/hP/hR → 固定融合h0
      │
      └→ sg(tokens) + 新查询sI_t → 并行Summarize → GF_I/GP_I/GR_I
                                                       ↓
                                                 联合摘要Attention
                                                       ↓
                                             最终限幅delta → h0+delta → 同一head
```

冻结ViT只是一套权重，对三种输入分别前向；不新建三套backbone。现有Image-token Adapter在配置索引1的图像证据读取中工作，不应画成改写Frozen Forward的共享图像流。新查询首次只在index5读取该层tokens，不是另建12层Task Forward。原三路MSA、Drop & Replace、FFN及分类位置均保留；新增摘要不会被替换回原Task Forward。

若原层查询可复用，可定义`sI_t=sg(s_t^(6))+U_t`，U_t为该任务新参数、零初始化；必须同时对原查询和图像tokens停止梯度。也可单独初始化私有查询，但不能在同一次消融中混用初始化配方。对视图v：

`GI_(t,v)=softmax(sI_t · sg(g_v)^T / sqrt(D)) · sg(g_v)`。

这里sg固定的是上游梯度，不是Summarize输出：GI必须对新查询可导。不得把GI再次整体detach，否则新查询学不到东西。Face mask在归一化前处理；同编号Selector只是同一查询对不同画面的读出，不是跨裁剪patch对齐保证。区域约束不自动叠加，保留全局Full上下文。

原损失L_base继续更新原Adapter、Selectors、Prompts和当前分类头；新增L_interaction计算`head_detach(sg(h0)+delta)`，只更新U_t及交互投影。U_t→GI→Attention→delta的梯度保留，不能流回原tokens、原Selectors、h0或head。原模型没有被冻结为预训练教师，而是在同一30轮流程内按原目标自己学习；新支路在旁边接受同批真实标签的监督。新增目标不更新原Face Task Forward，所以应准确表述为“增加可学习的Face补充取证”，不能预先宣称原Face单路一定变强。

本轮四个控制不含U_t。若后续验证候选，至少固定原摘要/新可学习并行摘要Attention及同输入池化MLP控制，匹配参数/预算/最终上限；10×768×8约614400额外查询参数需计入成本。任务结束封存私有查询和交互分支，测试仍跑已见任务路径、使用各任务类别读出并拼接，不是挑最好的任务Router。是否成功需由Face入口、人物交换、内容错配及最终/阶段平均收益判断。
