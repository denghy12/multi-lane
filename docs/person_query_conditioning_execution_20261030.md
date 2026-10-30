# 人物条件查询修复：代码与预检协议

日期2026-10-30，用户授权“开始修改”。新分支codex/person-query-conditioning-20261030从131919e创建，保留所有已有dirty分析。此次范围为实现、单测、GPU短smoke及TRAIN人物交换检查；完整8任务入口准备好，但本次不自动开始正式训练，不生成新正式mAP结论。

## 要分开的两个因素

旧诊断显示无定位f太小，定位版投影后tanh饱和。第一版提供互斥arm，没有联合修复开关：

| arm | 匹配旧控制 | 唯一修改 |
| --- | --- | --- |
| QPROJ_GROUND_POOL | PERSON_GROUND_POOL | tanh前投影RMS上限1 |
| QPROJ_GROUND_ATTN | PERSON_GROUND_ATTN | 同上 |
| QGAIN_CONTENT_POOL | PERSON_CONTENT_POOL | 人物偏移相对Query的有界增益 |
| QGAIN_CONTENT_ATTN | PERSON_CONTENT_ATTN | 同上 |

原Person CLS已有LayerNorm，不再增加输入LN。PROJ对z=C(LN(hP))按每个样本16维latent做无参数RMS上限：

    z_used = z / max(1, RMS(z) / 1.0)
    c = tanh(z_used)

低于上限时恒等，不改变方向，不添加可学习affine参数。GROUND原定位loss0.01及5轮warmup保留，f输出不放大、不加新的Query偏移cap。该约束可防整体尺度增长造成普遍饱和，不能保证投影方向保留人物信息或分类改善。

GAIN保留原z和tanh，且仍无定位loss；只对原低秩raw=f(hP)做：

    shift = Bound(raw * 32, anchor=S_last2, RMS ratio cap=0.01)

复用已测试的smooth bounded_delta。小信号增益最大32，零输入仍零且梯度有限；每个主体槽位相对自身原Query的RMS偏移不超过1%。这是新增“Query偏移预算”，与一直保留的“最终融合Δh不超过h0的1%”是两个不同位置。GAIN没有启用投影RMS约束，PROJ没有启用偏移增益。两项均为事先锁定的工程起点，不称最优参数，不根据本次Val/Test挑选。

两者仍按NormMatch保持原查询模长；主体U不变。量级约束不保证人物对应，强制增大f不等于获得有用信息，因此保留正确/错配人物及zero-condition阴性对照。

## 原模型与训练预算保持

仍为原强底座32.6800／39.5292配方：seed1、EMOTIC、CLIP ViT-B/16、b32 Adapter index1 scale0.03、10原共享Selectors、原Prefix Prompts/三路Task Forward/head、Person颜色扰动off、原aux BCE0.1和Adapter-ASL（neg9.8/pos0/clip0.05）、legacy_full_zero。

正式入口仍30epoch/task、8任务[5,3,3,3,3,3,3,3]、B64、Adam/WD0/任务cosine，主LR0.0125、Adapter与新私有支路LR0.0004；72线程/AMP/TF32/65536 scaler和历史报告RNG保留。新模块初始化仍930100+task独立随机流，原B0正常自训练；新增目标不更新原参数和head，不安装A/B教师或来源任务参数。

第6层、Full8全局+2主体、Person10/Face10、64维4头/单读出Query、原可靠性mask与最终1%限幅保持。Pool/Attention每任务分别221789/221728参数，预期与对应控制逐个参数名/初始化一致，无新增可训练参数。

## 代码及记录

新增person_query_conditioning.py独立子类，不改旧person_grounded_summary.py；通用build/config/参数统计/CLI/evaluation及TRAIN校准仅按新kind分派。新trace_person_query读取实际condition_components，不使用旧硬编码公式推测归一化后的查询。probe读取自有checkpoint的固定TRAIN多人batch，保持自己的晚期Query/其他memory/head，检查early swap、zero-condition、几何frame、参数和SHA。

日志增加raw/used投影RMS、收缩比例、raw/used饱和通道、tanh导数、raw/used人物偏移RMS及相对原Query上限。查询/取证/最终输出变化仍通过人物交换衡量，数值不同不当作mAP成功。

## 本次预检完整配置与路径

- 仅GPU0；当前约17GB已有任务，不动这些进程，不使用GPU1–7；max jobs1，free GPU>=6000MiB/shm>=1024MiB时运行。
- 先新机制单测与已有相关回归，真实CUDA AMP不可跳过。
- 再四candidate real smoke：每组task0/1、各4个更新，B64，原优化配置；每组8次主更新，结束自动TRAIN方向标定及Val首64和原归因；无Test选择。
- 再每组自有smoke checkpoint的固定TRAIN多人batch、0次更新，只验证来源/路径/信号和交换隔离，不以小批次分类分数宣布方案有效。
- 新远程独立worktree multi-lane-main-person-query-conditioning-20261030，经Git-only同步；旧source/main/test-only及全部权重保持。
- 输入由历史source配置读取：../multi-lane-main/datasets/EMOTIC，../CODE_DDP/pretrained/clip/ViT-B-16.pt，../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
- source recipe ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train。
- 复用已核验TRAIN几何统计 ../multi-lane-main-person-grounded-summary-20261029/output/person_grounded_summary/train_geometry_census_20261029/result.json，只作配置SHA及非空监督核对。
- 预检日志logs/person_query_conditioning/preflight_20261030/。
- smoke输出output/person_query_conditioning/smoke_seed1_20261030_01/，探针在各arm/train_person_probe/。

## 入口

本次smoke拟执行：

    bash scripts/emotic/run_person_query_conditioning_gpu0.sh smoke_seed1_20261030_01 \
      --family both --maximum-jobs 1 --smoke --smoke-updates 4

将来若明确执行正式训练，可分别使用--family projection或--family gain保持实验顺序。默认只选projection族；不将两个修复叠加，不为已有四组旧控制默认重训。完整8任务结果必须同时比较B0/CLS-MLP、各自匹配旧控制和同输入Pool，并复核原函数。不能仅看前三任务正差或解除饱和就扩大训练。

创建本协议时仅已完成本地静态检查，尚未宣称远程测试/真实smoke通过。通过后追加真实证据，三上下文同步。

## 实际完成的预检

执行源码ffab151d0f540cf38e310dcd70e2b3d6e2d94088，经Git-only部署上述独立worktree。新机制11项、既有条件取证/私有摘要/梯度保护/队列44项测试均通过，无skip，含真实CUDA AMP。静态编译、bash -n及git diff --check通过。

GPU0最大1并发，四组各完成task0/1各4次主更新及4次私有更新，共32主/32私有更新、0skip。完成4份TRAIN标定及4份Val首64人预检。跨组8份arm-task审计确认原参数、原监督loss、增强batch流、OFF融合及三路独立预测逐值一致；新支路均有非零梯度。OFF参照是本次同短预算底座，不把它称为历史30轮正式B0精度。

TRAIN探针各组使用相同task0首62人batch（55名双方可见供体）、task1首64人batch（58名供体），四组共452次有效人物条件交换；task1同时检查两条已见lane。Full图坐标逐像素一致、native修正精确重放、晚期Query不变、zero-condition交换阴性控制精确通过。探针0更新，模型/视觉参数、自有checkpoint SHA保持。

| 检查 | 实际观察 | 结论边界 |
| --- | --- | --- |
| 投影尺度版 | 每轮日志used RMS最大约1.00000009；used abs(z)>3为0；平均tanh导数0.5661–0.6028 | 约束确实位于投影后，短训练未普遍饱和；不证明30轮后人物方向有效 |
| 偏移增益版 | raw/used投影RMS完全一致，Query偏移低于1%上界 | 没有混入投影修复；1%是上界，不强制拉到1% |
| 交换人物后的主体Query相对RMS差异 | 投影版0.000252%–0.000399%；增益版0.000468%–0.001389% | 人物信号非零，但仍很小，不能称人物对应检验通过 |
| 框归属和分类修正 | 变化很小，部分binding mass gain仍为负 | 未证明正确人物优于错配或Attention有效 |

完整8任务训练及Test未启动，无新正式mAP结果；不能将4步与旧30轮数值直接解释为机制提升。后续仍先检查充分学习后的投影→Query→取证→分类链条，再判断Attention与同输入Pool的相对收益，不自动叠加修复、扩大容量或搜索参数。

旧48份源checkpoint SHA逐一匹配；main/test-only status SHA仍ea2e261e…/e9d1479f…，原model.py、runner.py、person_grounded_summary.py与131919e无差异。全部327份非权重产物已同步本地output/person_query_conditioning/preflight_snapshot_20261030/snapshot并逐SHA核验，服务器权重保留。归档SHA256 af94f8f36ad4b53363cfae7b9e8adc6a0844f65dd835418cc61e9edc957e0fcf；核心证据为snapshot/output/person_query_conditioning/smoke_seed1_20261030_01/preflight_audit.json及cross_arm_audit.json，日志在snapshot/logs/person_query_conditioning/。
