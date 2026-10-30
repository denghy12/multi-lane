# 下一步任务

## 2026-10-30：人物条件修复预检完成，正式效果待验证

ffab151已实现互斥投影RMS约束/相对Query偏移增益及两套匹配Pool。55测试全过无skip、四组task0/1各4步，共32主/32私有更新0skip；4 TRAIN标定/Val首64和TRAIN多人探针完成，原参数/监督/增强/OFF逐值不变。48旧checkpoint保持，327份产物本地逐SHA，三上下文与执行文档均更新。完整8任务/Test未启动。

不要把预检tanh不饱和当作人物绑定通过。换人Query差异仅0.000252%–0.001389%，部分binding gain为负，且与旧30轮不具相同步数。若继续，先分别检验充分学习后的raw/used z、人物偏移/Query方向、正确/错配取证及分类贡献；通过后再判断Attention与同输入Pool。不要直接叠加修复或盲增gain/容量。入口scripts/emotic/run_person_query_conditioning_gpu0.sh默认projection族，正式完整训练需明确执行，不自动续跑。保留B0/CLS-MLP正式参照。

预检证据output/person_query_conditioning/preflight_snapshot_20261030/snapshot/output/person_query_conditioning/smoke_seed1_20261030_01/preflight_audit.json；详细配置/边界见docs/person_query_conditioning_execution_20261030.md。

## 2026-10-30：开始修改人物条件传递，两个互斥单因素版本

用户授权开始修改，本次先代码/单测/真实短smoke，不自动启动完整8任务。新codex/person-query-conditioning-20261030从131919e，保留既有dirty。新增PersonQueryReadout/Model：QPROJ_GROUND_POOL/ATTN只在tanh前对C(LN(hP))做无affine RMS上限1，保留原loc .01/5轮warmup且f不放大；QGAIN_CONTENT_POOL/ATTN保留原投影/tanh且无loc，只把raw f以最大32小信号增益送入相对anchor RMS1%平滑上界。两项互斥，最终h0修正1%仍原样，不添加参数/改原U或原三路训练。模式参数为预先工程起点，非最优或提升保证。

原person_grounded_summary.py/model.py/runner.py保持；通用build/config/eval/参数统计/CLI和TRAIN校准仅新kind分派，新增actual-path trace及TRAIN own/wrong/zero-condition probe。候选初始化逐参匹配旧控制，原B0仍自训练，无教师/任务参数安装。Pool221789/Attn221728每任务不变，seed1历史b32/scale.03/aux/ASL/Person颜色off/72线程配方保持。正式入口准备projection/gain两族单独运行，暂不正式训练。

GPU0现有任务约17GB/99%，其他卡也满载不动。本次相关单测/CUDA AMP后，四组task0/1各4更新+B64 smoke在GPU0 max1自动等free>=6GB/shm>=1GB；之后自有checkpoint固定TRAIN多人probe0更新。日志/输出logs/output下person_query_conditioning，协议docs/person_query_conditioning_execution_20261030.md。当前仅实现与本地静态检查，远程未测、不声称效果。main/test-only/既有source权重不改，结果将回传核验，三上下文已更新。

## 2026-10-30：48项断点诊断全部完成，定位两类条件传递瓶颈

诊断seed1_20261029_02／源码131919e，六arm×task0/2/6/7×Val/Test共48case及CPU分析均complete，无新训练；源训练模型仍4349449。230份产物已同步本地逐SHA，归档b610657cf74822a73df355efb3a7fe5e01278b718cdc36db573f0e6b13e11bbc；48源compact SHA均与训练完成记录相同。原模型/诊断remote worktree clean，main/test-only status SHA仍ea2e261e…/e9d1479f…，无修改。

主要新证据（stage7 lane7、Val1178/Test2829有效配对）：Person LN输入换人相对RMS差83.84%/83.80%；无定位Attn tanh输出仍差86.82%/88.68%，但f RMS只有.000383/.000378，对Query RMS1.3516调制太弱，Query只变.01835%/.01854%。定位Attn投影z仍差18.03%/18.16%，tanh后只.01059%/.01597%；abs(z)>3通道99.9416%/99.9249%，平均导数5.42e−5/6.86e−5，Query只变.000835%/.001185%。当前task0/2/6/7和Pool也出现定位版99.25%–99.94%饱和。不同模型有不同断点：不宜说所有模型都饱和或Person输入一样。近常量的是人物条件偏移，Full原查询/摘要仍可随图片变化。

空间证据：定位Attn任务7配对人群，Val自己框/他人框质量均约39.64%，Test均约40.19%，无定位约29.86%/30.54%，两框一起提升。当前人物框确实监督loss；减少背景并同时读几个人可改善loss，但不足以说明指定当前目标。高可见+面积匹配+足够独占区域Val198/Test438人，自己相对他人密度、正确vs换人binding gain区间均跨0。CONTENT Attn Test全可见有微小正gain约7e−5，但Val不稳/控制后跨0，不能忽略该小信号，也不能升级有效人物读取结论。

排序证据：GROUND Attn最终Sadness/Sensitivity/Suffering AP差，Val−.3329/−.0172/+.3721，Test−.2202/+.0255/−.4913。Suffering全体正负投影均值分离+.0179，但固定B0邻近负例修正方向−.008435，修复19.89%/破坏27.42%；Sadness修复16.24%/破坏20.25%。task6自身checkpoint已有损失（Test Suffering约−.4971），不能主要解释为后续遗忘。原Full/head未受新梯度损伤；人物绑定没通过，不能进一步唯一断言分类头接口是根因。

正式完整8任务评分仍沿用原模型：B0 Val42.4628/49.4763、Test32.6800/39.5292；CLS-MLP42.5495/49.6120、32.7139/39.5923；CONST Pool42.5180/49.6220、32.6726/39.5909；CONST Attn42.4484/49.5074、32.7007/39.6201；CONTENT Pool42.4986/49.5905、32.6849/39.6294；CONTENT Attn42.5259/49.6281、32.6783/39.5889；GROUND Pool42.4995/49.6346、32.6870/39.6158；GROUND Attn42.5114/49.5745、32.6589/39.5575。没有新版本或新最好结果，不将4个诊断阶段平均当成8阶段平均。

本地独立复算96指标/780逐类/16416配对/2736查询/2736空间/192区间；mAP/AP/AUC、邻近排序和区间与remote一致，少量float32分位等版本差<=5.8e−6。新增只读audit_person_grounded_breakpoint_results.py核验240历史文件和3648条人物×lane原始张量（RMS/tanh/导数/alpha/框覆盖），全部通过。16常量阴性case通过。正式全人物保留，小尾仅不做张量归因：Val投影最多缺29人/阶段，Test task6/7全部5367/5368人覆盖，核心排序损伤非尾样本导致。较严格标签差异子群可能只有29/38人，报告功效边界。

报告docs/person_grounded_breakpoint_results_20261030.md；完整回传output/person_grounded_breakpoints/complete_snapshot_20261030_01/snapshot；独立分析analysis_20261030_01与raw_audit_20261030_01。下一步保留B0/CLS-MLP，停止本版Attention扩容；若继续，先单独检验C(LN(hP))投影后的尺度/饱和保护，再单独检验f相对Query的有效量级。输入hP已LN，不能重复加输入LN当新修复；正确人物取证须先稳定优于错配，再比较Attention/Pool或Task CLS语义对齐。去tanh/加LR/放大偏移不保证mAP。当前只更新本地报告/复算脚本和三上下文，保留旧dirty；未训练、未提交/推送/合并/删除。

## 2026-10-29：三个断点正式02已启动，GPU0六组并行且首7case通过

运行源码131919ed90e914244c7c7abf4daa0a1b80bd048c，local/upstream/remote已Git-only一致，独立multi-lane-main-person-grounded-breakpoint-audit-20261029 clean。23项相关+4项NumPy测试全通过、CUDA AMP实际执行无skip。CONST_ATTN/GROUND_ATTN的task0 Val全量预检通过：正式源2285人全保留且本阶段2285人新forward精确，paired原1139人中1135做新trace（4人尾batch不计张量统计），有效供体1109；原native/early/late概率精确匹配，所有参数/源checkpoint未变。19份完整修复预检产物已本地同步逐SHA，archive1c2a190e4a2624925682f52b739f048b9a07374a557e37120ab5e05453a88813；此前19首batch预检也已回传，旧01失败产物仍保留。

正式run seed1_20261029_02、tmux person_grounded_breakpoints_seed1_20261029_02；六组实际全已启动，PID CONST_POOL4180948、CONST_ATTN4181472、CONTENT_POOL4182066、CONTENT_ATTN4182654、GROUND_POOL4183314、GROUND_ATTN4183975。健康检查6active／pending空／0failure，已有7/48case完成。仅GPU0约9760MiB，原48份compact SHA逐一重算与训练完成记录完全匹配，source4349449 clean、main/test-only status SHA仍ea2e261e…/e9d1479f…。startup_health.json已回传output/person_grounded_breakpoints/startup_20261029。

剩余case自动按每arm task0/2/6/7的Val/Test完成；六arm完成后自动CPU生成analysis目录（各断点/框区域/分类排序及500次聚类统计），分别forward completion与analysis completion。没有新训练、没有组装模型、没有改旧评估标准。全历史mAP/AP/AUC仍用全部源人物；新Wdelta/张量只统计原batch>=32且精确重放人群，缺失尾样本名单与mask/NaN显式保存，不当作零值。尚无完整正式断点结论，不提前宣称tanh饱和或空间绑定失败。

本地本次只补运行文档与三上下文，不再改/pull运行源码；已有dirty报告/分析均保留，不合并main、不删数据。不需用户手动启动或同步代码。完整协议docs/person_grounded_breakpoint_execution_20261029.md，正式产物/日志output/logs下person_grounded_breakpoints/seed1_20261029_02。

## 2026-10-29：诊断01严格回放中止；固定小尾batch策略后准备重跑02

首次六组诊断seed1_20261029_01均在task0 Val配对尾batch中止；各组完整2285人的ON/OFF/三路已逐值重放，但最后4人batch在未插入观察器的原native前向就无法匹配。单独同checkpoint/同input连续前向，72→4→72线程均有变化（相对历史max abs .00390625；完整诊断首次尾batch曾.01171875），不是简单恢复72线程可解决。原参数和checkpoint未变，失败输出/日志/队列保留，不用其计算结果下机制结论，也不放宽容差。

修复只限定诊断覆盖：全部原人物及历史正式ON/OFF概率/评分保留；新的张量前向固定只做原batch>=32的部分，不重排/补人/改变原batch。小尾batch名单明确保存，完整population NPZ中projection_available=False且Wdelta用NaN，分析仅在available人群计算投影分离/相邻负例；AP/AUC仍来自全部历史原预测。配对trace同样明确source人数与traced人数，排除小尾后所有条件在人群/batch上严格配对。不会将浮点波动计成人物条件效果，不混用诊断子集与正式全人群mAP。

新增保留所有尾batch正式score且不伪造Wdelta的机制测试，待远程23相关+4 NumPy测试及两Attention组task0 Val完整full/paired预检。正式重跑ID改seed1_20261029_02，仍6arm×task0/2/6/7×Val/Test、GPU0最多6并行、0训练；诊断commit待推送。原source4349449和main/test-only未改，旧dirty分析保留，三上下文同步修正。执行文档已记录此覆盖限制，不能再声称所有人物都做了新张量前向。

## 2026-10-29：三个断点诊断预检通过，准备六组正式只读前向

模型观察代码2622b73已Git-only部署独立multi-lane-main-person-grounded-breakpoint-audit-20261029，原source4349449全部原模块无修改。远程22项相关测试全部通过（包含7新trace机制、真实CUDA AMP及既有Summarize/调度），无skip；CONST_ATTN/GROUND_ATTN的task0 Val原full首64与原paired首batch真实smoke通过，每组62名有效供体，native/early/late logits与CPU sigmoid逐值重放、原alpha/summary/delta精确、method/视觉/源checkpoint未变。推理4线程/worker1不改变已检查预测。

增加独立NumPy分析脚本，4项排序/区域分层/聚类测试本地通过；预计同一正式队列后自动CPU分析，无需等待新训练。分析保存full完整历史mAP/逐类AP与全体AUC、Wdelta正负分离及B0相邻负样本修复/损伤比例；paired按原人物标签范围和面积/重叠/可见性分层输出各断点相对变化、空间binding gain、分类条件贡献与500次原图聚类区间。FP64仅用于线性头解析，不取代历史正式概率。

待将这次仅新增分析/协议字段的提交ff同步，复核4项NumPy测试及source无差异后，启动6arm×4阶段×2split共48case。此时未报告正式断点或改进结论，未启动训练；输出/日志仍person_grounded_breakpoints/seed1_20261029_01。全部三上下文与执行文档已更新，旧dirty分析保留。

## 2026-10-29：开始执行人物条件Summarize三个断点的只读诊断

用户明确授权开始；新分支codex/person-grounded-breakpoint-audit-20261029从daaacef建立，保留全部既有dirty分析。新增独立person_grounded_breakpoints入口及队列/7机制测试，不修改原模型/训练代码。计划6个原arm×task0/2/6/7×Val/Test共48case；先完整历史人群native/OFF/三视图逐值重放及解析W·delta，再同图人物对追踪LN输入→pre/tanh→低秩shift→query→alpha→summary/LN/value→lateQuery/hidden/raw/delta，并比较自己/他人框及独占区域质量/面积密度。标签差异只分层，不挑供体；原Full crop不同则拒绝跨框比较。

GPU0当前空闲24GB，其他卡满载不动。新worktree经Git-only部署，原source4349449所有原multi_lane文件要求无差异。源模型seed1历史b32/scale.03/30轮终点不变，无优化器、0更新；batch64、72线程初始化/推理4线程、worker1/prefetch1，以严格历史logits/CPU sigmoid逐值重放控制数值。最多6arm并行，3.5GB GPU/512MB shm启动余量。输出logs/output下person_grounded_breakpoints/seed1_20261029_01，完整路径/模块/诊断指标见docs/person_grounded_breakpoint_execution_20261029.md。

本条创建时仅本地编译/diff检查通过，尚未远程测试/smoke/正式，不提前断言tanh饱和或人物条件恒定。模型source clean、main/test-only status --short SHA ea2e261e…/e9d1479f…保持。先7新测试+既有相关测试，再CONST_ATTN/GROUND_ATTN task0 Val首个原full/paired batch smoke，通过才启动；不改旧checkpoint、不开新训练、不合并main。

## 2026-10-29：人物条件Summarize六组完整结束与只读归因完成

GPU0／seed1／seed1_20261029_01，6×8任务、各240epoch／13950更新、主私有均0skip。原模型源码434944950677935c26d9484e4a05b319bd866f2a clean；实际原四并发队列自动完成，六并发接管未应用，不记为六组同时训练。6份TRAIN标定、12份完整Val／Test及全部预定干预完成，无failure；没有新训练、换头、A/B参数组装或教师。

Val最终／平均；Test最终／平均：
- B0 42.4628／49.4763；32.6800／39.5292。
- 已有CLS-MLP 42.5495／49.6120；32.7139／39.5923。
- 常量Pool 42.5180／49.6220；32.6726／39.5909。
- 常量Attention 42.4484／49.5074；32.7007／39.6201。
- Person条件Pool 42.4986／49.5905；32.6849／39.6294。
- Person条件Attention 42.5259／49.6281；32.6783／39.5889。
- Person＋定位Pool 42.4995／49.6346；32.6870／39.6158。
- Person＋定位Attention 42.5114／49.5745；32.6589／39.5575。

没有一组通过同时超过B0／CLS-MLP及同输入Pool的两项Test标准。六组最终相对B0的图片聚类bootstrap500次95%区间全部含0；常量Attention、内容Pool、定位Pool平均正差区间为正，限固定checkpoint、未校正多比较，不覆盖seed方差。

48历史参数／loss／RNG边界精确，96份OFF与全部原三路输出逐值恢复B0。因此本轮不能解释成Face又损伤Full，也没有提高原Face单路。定位Attention的task7八lane平均主体框内质量Val31.21%→61.00%、Test31.56%→60.93%，但换错早期Person条件几乎不改预测；主体/Face入口没有稳定分类收益。定位Attention关闭新增Face后Test32.6887／39.5865，好于开启32.6589／39.5575，Val也提高。固定同范数错配方向六组最终Val/Test均不降，仅作归因，不作新正式最优。

结构解释：框只参与定位loss与有效性，没有以坐标进入主体query；正确人物条件的端到端依赖仍弱。平均空间取证改善不等于区分当前人物／提供互补情绪证据。中层摘要到最终CLS语义对齐可能也是瓶颈，尚未证明唯一根因。不可断言tanh饱和、梯度冲突或所有Attention无效。

原产物2924份已回传output/person_grounded_summary/complete_snapshot_20261029_01/snapshot，归档SHA f6c7f4240a48df57417ebd229f9df0342724a7cc5b43090e393c6d817aad23b9；全部逐SHA。独立NumPy复算7104组／133224项、最大误差4.84e-13，32常量交换／96无效Face／96同范数／288配对文件控制通过，48任务checkpoint训练/eval SHA匹配。1584训练/恢复pth仅留服务器，snapshot只含SHA清单。分析output/person_grounded_summary/analysis_20261029_02，复算脚本scripts/emotic/analyze_person_grounded_summary.py；analysis_01为空的失败初次目录，不是正式分析。

完整报告docs/person_grounded_summary_results_20261029.md。建议保留B0与CLS-MLP，不扩seed/层数/query/scale；若继续，先已有checkpoint记录换人从条件偏移→query→alpha→summary→hidden→W·delta的传递与自己/他人框质量，区分人物取证和语义读出断点，未授权/未启动下一轮新训练。本轮只新增本地分析脚本/报告并更新三上下文，保留此前全部dirty文件，未提交/推送分析。main/test-only的git status --short SHA仍ea2e261e…/e9d1479f…，源模型checkout未改。


## 2026-10-29：中断后复核，前四组训练已结束，GROUND两组继续；六并发接管未应用

此前四train进程在中断期间已各自完成8task/240epoch/13950更新，GROUND两组已由原队列自动启动。重新进行六并发dry-run时，active已包含两train和两eval，detached_trainers全训练前提不成立；检查在任何信号、接管文件写入前中止，未杀controller/任何trainer、未重启或重复启动已完成组。原run控制器仍正常管理，无failure。六并发脚本在单独scheduler分支/worktree daaacef提交，4项本地和远程调度测试通过，但尚未应用；不要误记已实现六组同时训练，也不要为了凑并发重跑已完成模型。

最新健康核对：CONST/CONTENT四组均完整8task训练完成，各0主私有skip、8历史参数/loss/RNG边界精确；四份Val均complete，两个Attention的Test已complete，其余两个Pool的Test当时运行/排队并已自动推进。GROUND_POOL/ATTN均task3 epoch9、累计6414成功更新、0skip，task0/1/2历史边界精确。六组均已启动，当前没有六个未完成训练需要同时启动，剩余两组训练与评估继续在GPU0并行；全部训练完成后依旧TRAIN标定→完整Val/Test/归因，不使用其他GPU。

模型源worktree4349449仍clean，main/test-only状态SHA仍ea2e261e…/e9d1479f…不变。读取原队列、进度与审计后保存startup_health_after_interruption.json，已回传output/person_grounded_summary/startup_20261029。没有修改运行源码、超参数、batch或已有进度，没有新增训练/测试组。475预检产物已回传逐SHA，正式结果尚不完整，不提前作六组效果结论。用户“显存足够六组并发”偏好保留；下一次新运行应在初始状态就使用经验证的6并发政策。


## 2026-10-29：六组正式已启动；按用户要求升至六并发（调度迁移预检）

模型源码434944950677935c26d9484e4a05b319bd866f2a已Git-only部署独立multi-lane-main-person-grounded-summary-20261029 clean，source/upstream一致。40专项/相关（含11新机制与真实CUDA AMP）和434 Track-A回归全通过。TRAIN单增强pass8任务均有定位监督，task6为127/627=20.255%，10batch全非空；不据此保证统计功效。六组两任务×4更新smoke、TRAIN标定、Val首64与归因均complete、主/私有0skip；12阶段跨臂原参数/关键loss/增强batch/OFF和三路预测逐值一致，U有梯度、FP32/1%cap/旧任务冻结/恢复通过。每stage61名早期条件交换、8名可靠Face交换；常量条件early_swap逐值不变。参数Pool221789/任务、Attention221728。475份非pth产物已回传逐SHA，archive3e52bae439ada02ebc1ad66b118844e66b6b23596ebb7ab19c11d4546f565a82。

正式run seed1_20261029_01、tmux person_grounded_seed1_20261029已启动，首四组在GPU0并行、GROUND两组初始排队。用户随后明确要求显存够时六组同时启动，覆盖原max4计划。将只迁移调度器，保留原训练进程/PID/开始时间和进度；模型worktree不pull/不修改、batch64/72线程和配方不变。新调度分支codex/person-grounded-six-way-20261029，独立scheduler worktree运行，采用已有detached PID接管逻辑，max6、非训练最多2、free GPU>=6GB/shm>=1GB/12秒间隔。TRAIN校准完成后才放行Val/Test，避免重复启动活进程。

当前新调度4项本地测试/编译通过，尚未接管或声称六组全部在跑；实际接管后补PID与显存证据。新脚本scripts/emotic/expand_person_grounded_queue.py，不改正在运行模型/训练源码。main/test-only和既有dirty分析保留，三上下文已更新。后续队列自动完整8task Validation/Test及归因，不需用户手动执行。


## 2026-10-29：开始执行人物条件/空间锚定六组8任务（实现预检中）

用户明确授权执行，task按上下文为Test。新分支codex/person-grounded-summary-20261029从b4015bc建立，保留所有既有dirty分析。新增person_grounded_summary模型（任务常量/Person内容/Person+定位 ×Pool/Attention），Full8全局+2主体、Person/Face各10；低秩条件16、主体U独立、查询模长匹配、主体不读Frozen CLS、全局保留197tokens。主体TRAIN框内质量loss .01前5轮warmup且>=90%可见，仅更新新主体查询/条件器。修复旧私有Summarize缺失autocast禁用；新增完整评估/早期条件交换、TRAIN几何统计、队列入口与机制测试。原model.py/runner.py不改，通用build/config/eval和私有目标仅按新kind分派。

计划seed1/8task/30epoch/B64，主LR.0125、新模块和Adapter.0004，历史b32/index1/scale.03/aux.1/ASL/Person颜色扰动off/72线程及原RNG/历史人群口径。GPU0约24GB空闲、shm4GB，其他卡满载且不使用；最多4训练/2非训练，保留6GB GPU和1GB shm余量排队。先Git-only同步独立multi-lane-main-person-grounded-summary-20261029，完整TRAIN单增强pass统计有效定位监督，再相关单测/CUDA AMP、完整Track-A、六组2任务×4更新真实smoke与TRAIN标定/Val首64。通过历史原函数、RNG、恢复及归因检查后才启动六组完整8任务训练/Val/Test，不依据早期精度筛选。

完整配置/路径/命令docs/person_grounded_summary_execution_20261029.md，机制方案docs/person_grounded_parallel_summary_plan_20261029.md；输出output/person_grounded_summary、日志logs同前缀。本条创建时仅本地编译/bash语法检查通过，远程测试/几何统计/正式尚未执行，效果未知。main/test-only状态SHA仍ea2e261e…/e9d1479f…保留，不合并、不删数据、不开其他GPU。


## 2026-10-29：人物条件与空间锚定Summarize的下一阶段方案（仅计划）

用户要求详细方案，本轮只读核对私有Summarize结果、旧selector_conditioner、旧空间摘要与受保护Attention结果，以及MULTI-LANE/DSCT/Conditional DETR原论文；新增docs/person_grounded_parallel_summary_plan_20261029.md。明确“Person→MLP→query delta”已有实现，不能换名当新突破。研究候选保留原三路自训练和梯度保护，在新增Full十摘要中用8个全局候选＋2个主体候选，Person/Face各10，总memory仍30。主体2查询由sg当前原Person最终CLS条件化低秩调制，模长匹配；先原Task Forward取hP并缓存第6层，后旁路先条件查询再压缩，不把最终CLS循环送回原第6层。

主体只读196patch不读Frozen CLS；全局8仍可读197token/全局场景。仅主体2用TRAIN实际变换框的注意力框内质量loss，建议lambda.01前5轮warmup、>=90%可见才监督；不做推理硬ROI，不约束所有场景摘要，不添加情绪互斥负对/排序loss。Lloc只更新主体私有U/条件器，新增BCE更新新旁路，原BCE/ASL照旧更新原三路和head。原三路端点不变，目标是补充取证/融合，不承诺原Face单路改善。

提议六组=任务常量/当前Person条件/Person条件+锚定 ×同输入Pool/Attention；常量条件用固定非零buffer输入同一条件器，后置Query仍保留自身Person信息。四组memory类型各自pool，322→127→64匹配64维4头单QueryAttention；估算221789/221728参数每任务，差61，落地后核验。不把名义参数匹配称有效容量严格相同。保留B0及CLS-MLP32.6800/39.5292、32.7139/39.5923。

未来执行沿seed1/8task/30epoch/B64、主LR.0125/Adapter新模块.0004、b32/scale.03/Person颜色扰动off/原aux与ASL/72线程历史RNG；只GPU0自动排队，先修autocast CUDA验证，不根据前三任务效果截断。TRAIN人数5353/4394/861/9931/4352/2536/627/1526，task6仅300总主更新；须先核对>=90%可见有效定位供给，不自动改裁剪/采样/epoch补数据。正确早期Person条件交换应保持晚期Query、h0、原memory/head；正确目标优势、主体入口净贡献、两项Test超过强参照、Attention超过同输入Pool分层验收。若仅定位或Pool改善不称Attention成功；若全失败停止后置1%接口。

本轮没有修改业务代码/训练配置，没有创建实现分支、运行数据统计/GPU/新实验、提交或push；新模块和lambda只是待验证方案。旧dirty分析全部保留，三上下文同步更新，未来需新codex/分支和Git-only部署，不合并main或删除数据。


## 2026-10-29：私有Summarize四组8任务结果及FP32补评已核验

本节为最新状态，后面的“预检／已启动／待评估”为历史过程。运行b4015bc、seed1_20261029_01全部complete：四组各240epoch／13950主更新／主私有0skip，32历史原参数／30轮loss／RNG及跨臂审计通过；64份OFF和全部原Full／Person／Face逐值恢复历史强B0，28旧TRAIN探针最大差0。原Val最终／平均42.4628／49.4763、Test32.6800／39.5292。

原正式Val最终／平均、Test最终／平均：原查询Pool42.4565／49.5508、32.6372／39.5783；原查询Attention42.5033／49.5460、32.6716／39.5803；新查询Pool42.4860／49.6078、32.6704／39.6075；新查询Attention42.4957／49.5188、32.6607／39.5576。四组Test前6阶段正、task6/7负；task6 Sadness/Sensitivity/Suffering合计最终贡献−.0551/−.0254/−.0584/−.0235。没有两项Test同时超过B0，也未超过已有CLS-MLP32.7139／39.5923；新查询改善Pool点估计但不改善Attention，不声称稳定算子优势。

U有梯度/实际偏移，查询夹角约.21–1.11度、模长匹配；首64复核摘要RMS变化约Pool3.32%、Attention2.53%，不能说完全没学。U归零Test最终/平均条件贡献Pool+.00150/+.00324，Attention+.00095/+.00185，直接净作用小；不撤销读出共适配。新Attention Face与Person入口在双split最终/平均均负。最终Test同图双方可见2842人Query/Person交换、双方可靠1321人Face交换；新Attention原Query优势−.0106、Face−.0568、Person+.0259，多阶段无稳定人物绑定证据。质量组内等范数错配方向最终Test四组都优于native，仅是诊断，不可当部署方法。修正几乎总饱和1%，原Full未变，损失发生于新delta读出。

发现实现偏差：summarize的.float未关闭外层AMP，训练私有backward为FP32，原评估矩阵运算为FP16。GPU0额外0/2/7首64及完整8任务只读FP32补评，保持历史head AMP、人物/batch/CPU每batch sigmoid；原AMP64份逐值重放，OFF64份概率精确重放，32checkpoint SHA未变。补评原查询Pool42.4562／49.5507、32.6371／39.5784；原查询Attention42.5032／49.5458、32.6727／39.5818；新查询Pool42.4845／49.6071、32.6704／39.6074；新查询Attention42.4971／49.5192、32.6606／39.5578。主表最大变化.001495，不改变结论；原训练/模型源码保持，不把精度偏差当根因。 补评NumPy320份预测/17760值误差≤3.84e−13，OFF64/U0固定32/无效Face64阴性控制均精确通过。

原1505产物回传逐SHA，archive a691477d7780ae258933f82b71b65812ad654bfef4f3c6e318824680e57186a7；补评390份archive0ceaab5603f0c1b2b563ca56406b2f15f73c9cd336bda8c2d27e87965ebe7c0c。原NumPy2944组/54992值误差≤4.13e−13，500次原图聚类/阶段耦合bootstrap仅learn Pool平均相对B0区间正，最终和主要组间比较跨0，非seed方差/未多重校正。报告docs/private_summarize_attention_results_20261029.md，复算scripts/emotic/analyze_private_summarize_attention.py及audit_private_summary_precision.py，完整回传与analysis在output/private_summarize_attention。

本轮仅分析和只读精度补评，无新训练、无组装/教师、不合并main、不删checkpoint。remote实验源码clean，main/test-only状态SHA仍ea2e261e…/e9d1479f…；本地保留既有dirty分析并更新三上下文。原函数保护成功，当前Attention候选未通过推进条件；保留B0及CLS-MLP参照，不直接扩seed/查询/层数/scale。下次使用需修正Summarize autocast边界并补CUDA AMP回归；若继续研究，先验证摘要压缩前的人物条件取证和修正对应性，不把已试过的区域mask自动叠回。


## 2026-10-29：交互专用Summarize四组完整8任务已启动

运行b4015bc03def0fce29e4abcfca0b46d285878c73，本地/remote/upstream一致，独立multi-lane-main-private-summarize-attention-20261029 clean。33专项/相关含11新机制全部通过，434完整Track-A和15门控/恢复相关通过。四组2任务×4更新smoke、TRAIN标定、Val首64及开关/人物交换/同范数方向归因全部complete，0skip；8阶段跨臂原参数/loss/记录增强batch/OFF/三路逐值一致，query梯度/模长/原路径隔离/恢复/1%cap通过。每个smoke阶段61名Person、8名可靠Face实际交换，非空归因。参数/任务固定Pool198768、固定Attn198912、learn Pool206448、learn Attn206592。

正式run seed1_20261029_01，tmux private_summary_seed1_20261029，四组GPU0同时训练；约14257MiB/100%，queue无失败。首次健康检查固定Pool task0已3epoch、另外三组2epoch，各84更新/epoch/0skip，原关键loss与历史精确一致；learn Pool/Attn新query梯度3.45e−5/8.36e−5、偏移范数.498/.483，持续更新。每组完整8任务后自动先TRAIN锁方向，再完整Val/Test和0/2/6/7人物/方向诊断；无早期效果门槛。当前没有正式精度结论，不要重复启动或修改运行源码。

254份预检/smoke产物已回传逐SHA，archive7e40b4173557ba2aa2cc6cd5bb914ca4ba612224cded12ab30bb47b09f4e2d59，本地output/private_summarize_attention/preflight_snapshot_20261029_01。原main/test-only状态仍ea2e261e…/e9d1479f…保留。完整配置与实际记录docs/private_summarize_attention_execution_20261029.md；本地仅补记录，不在运行中pull/改源码、不合并main，不需要用户手动执行。


## 2026-10-29：执行交互专用Summarize 2×2完整8任务（实现预检中）

用户明确授权开始。新分支codex/private-summarize-attention-20261029从0e366d9，保留既有dirty分析。四组SUMMARY_FIXED_POOL/FIXED_ATTN/LEARN_POOL/LEARN_ATTN；原第6层LN tokens/查询sg缓存、共同FP32 Summarize重算，U零初始化且原模长匹配，10查询/任务共享三视图。Pool257→152→64，对照Attention64/4头，Q/V/O共同初始化，固定gain1/最终1%限幅。U在interaction_bank下，固定组为buffer、learn组为参数；私有目标只更新U/读出。原model/runner不改，通用build/config/eval/queue支持新方法与TRAIN标定前置依赖；新源模块/eval/launcher/11项机制测试已添加。

seed1/8task/30epoch/B64，历史强B0主LR.0125/Adapter-U-读出.0004、BCE/ASL aux.1、b32/index1/scale.03/Person颜色扰动关闭/72线程及原报告RNG保持。仅GPU0，最多4作业/2非训练，6GB/1GB余量自动排队。训练后每组先固定TRAIN每任务4批标定修正均值方向，再完整Val/Test：ON/OFF/U0/固定Query单memory屏蔽；0/2/6/7补同范数TRAIN均值/错配方向，以及同图双方可见Query/证据交换，配对native单独保存。无外部任务参数或教师，不按早期精度筛选。

完整协议docs/private_summarize_attention_execution_20261029.md，输出output/private_summarize_attention、日志logs同前缀。当前编译/bash/diff检查通过，尚未远程测试/smoke/正式训练，不声称有效。原main/test-only状态与其他GPU任务保留，后续Git-only同步独立worktree，检查通过才启动；三上下文已同步更新。


## 2026-10-29：并行私有Summarize查询的下一阶段计划，仅分析未实现

用户要求依据统一1%四组结果制定详细方案，明确不改代码。本次只读核对结果与MULTI-LANE §3.3/式3/图2/3，新增docs/parallel_private_summarize_attention_plan_20261029.md。研究问题为交互专用摘要取证与组合的分工，不把原摘要遗漏证据当已证实事实。主方案2×2：沿原查询/新增可学习查询 × 同输入近等参数池化MLP/Attention；保留B0及已有CLS-MLP强参照。两种读出统一token源、FP32重算Summarize、10摘要/视图、mask、Query条件/共同初始化/1%固定限幅，无学习gate或ROI。Pool使用共同64维q和Value，257→152→64 MLP配平Key投影，约198768/任务，对照Attention198912。共同模块按任务/组件配对初始化；可学习摘要经不同读出训练后不要求逐值相同，不夸称唯一算子因果效应。

新查询每任务U为10×768，三视图共享，建议以原第6层查询的sg副本+零偏移、保持原模长学习方向，限制温度混杂。sg原tokens和原Selectors，但保留新摘要对U的梯度；原路径在同一30轮内照旧自训，新增目标只更新U/交互，head在新增目标中detach。保留独立optimizer/scaler/RNG、逐边界B0保真、旧任务封存，推理所有已见任务类别拼接。任务私有不是人物私有，Full多人归属仍需交换验证；不承诺原Face独立AP上涨。

拟议正式seed1完整8task/30epoch/B64、历史B0配方/LR/72线程不变，仅GPU0，先实现smoke再完整双split，不用task0–2效果筛选绕过task6/7；此处没有执行命令或新入口。必要归因含OFF、U归零、固定Query的单memory屏蔽、人物Query/证据交换、TRAIN常量方向/同范数错配方向，及task6逐类排序。若两新摘要组均失败停止扩容；仅Pool改善则围绕取证报告，只有Attention优于同输入Pool/强参照且两项Test正向才推进。效果尚未知，未启动新训练/测试/提交/push，原模型/脚本源码未改。

纠正文档参数量：每任务10×768=7680，8任务共61440（6.144万），此前execution误写614400，多算10倍；已完成四组未启用这些查询，实验参数和指标不受影响。对应旧说明已更正，三上下文同步记录，保留所有既有dirty文件。


## 2026-10-28：统一1%上限四组8任务结果已核验，未见Attention特有净收益

运行0e366d9四组训练/8份Val-Test均complete，各240epoch/13950主更新/0主私有skip，32历史原参数/30轮loss/RNG审计及跨臂审计通过；64完整OFF预测/所有原三路逐值重放历史B0。Val最终/平均、Test最终/平均：B0 42.4628/49.4763、32.6800/39.5292；MLP42.5495/49.6120、32.7139/39.5923；固定Attention42.4797/49.5224、32.6684/39.5668；任务标量42.5722/49.6820、32.6507/39.5451；图片门控42.4973/49.5523、32.6756/39.5989。Val四组8/8正；Test MLP8/8、Attention各6/8，task6/7负。三Attention未同时提高两项Test，MLP小正但条件区间跨0。

结构发现：原Full/Person/Face不变，损失发生于新增delta读出，非Face共训损伤Full/主要增量遗忘。最终Attention平均RMS均约.99989%，所有人/所有lane> .95%；内容gain各任务均值1.0129–1.0421，图间std .000023–.000152，近常量。gain位于已饱和平滑Bound之前，正gain近似被归一化抵消；预算公平但拒绝能力未充分激活，不能普遍否定内容门控。Task6 Sadness/Sensitivity/Suffering Test的三Attention合计最终贡献−.0287/−.0380/−.0372，Suffering各约−.57；MLP该组近保持。Test Face入口最终/平均：MLP+.0099/+.0270，固定−.0111/−.0283，标量−.0015/+.0094，内容−.0004/+.0090；后者入口开关含gate输入变化。去Face仍不能消除Suffering损失，不归为单一路问题。

889非pth产物已回传逐SHA，archive64c31b9c91bbfeb9ecdb8cc41f2ce414856a2952e6849fc5390932561a80874e；32compact checkpoint实测SHA同完成记录，原main/test-only状态保持。独立NumPy1792视图/人群组、33152数值误差3.98e−13；500次原图聚类/八阶段耦合bootstrap：MLP Test最终/平均区间[-.0072,.0842]/[-.0082,.1333]；图片门控平均[.0040,.1380]但最终[-.0565,.0447]，相对MLP平均差区间跨0。固定checkpoint/未多重校正/非seed方差。报告docs/protected_attention_dose_results_20261028.md，复算scripts/emotic/analyze_protected_attention_dose.py，回传/分析output/protected_attention_dose/{complete_snapshot_20261028_01,analysis_20261028_01}。

本轮仅分析，无新训练/GPU前向；GPU0空闲。保留强B0和MLP控制，不扩seed/层数/gate/区域。若沿Attention继续，先以同输入池化MLP和固定摘要控制检验并行私有Summarize查询，既有候选仍未实现且效果未知，不将本轮自动作为有效性证明。不合并main、不删数据；本地更新报告与三上下文，原运行源码clean保持。


## 2026-10-28：统一1%上限四组8task训练已启动

运行0e366d9d5623ecd65500e4cc16ac72915b419e8d，服务器独立multi-lane-main-protected-attention-dose-20261028 clean且与本地/upstream一致。29项专项/相关（含8项新机制）全通过，434完整Track-A通过，另38项训练/恢复回归通过（有重叠，不累计独立测试数）。四组2任务×4更新smoke全部完成、每组8更新/0skip，8阶段原参数/loss/记录增强batch/OFF及三路预测精确一致，新梯度非零、最大写入≤1%、峰值allocated2819–2821MiB。

正式seed1_20261028_01已启动，tmux attention_dose_seed1_20261028，DOSE_MLP/FIXED/TASK/CONTENT四组GPU0并行，约14259MiB/100%，queue无失败。首次健康核验MLP task0前4epoch、三Attention前3epoch，各84更新/epoch/0skip，原关键loss与历史逐值相同，实际上限通过。各组完整8任务后自动Val/Test及ON/OFF/入口归因，最多2评估；尚无正式精度结论。150预检/smoke产物回传逐SHA，archive2727b17250409b9275fb397f08ad04752a27420ee989e2ed4fd99a6b780f43b9。原主/test-only状态保持；本地仅补启动记录，不改正在运行源码、不合并main。协议docs/protected_attention_dose_execution_20261028.md；并行私有Summarize查询仍仅解释、未实现。


## 2026-10-28：执行统一1%上限的四组Attention训练控制（预检中）

用户授权第一件事完整执行，第二件事仅要求MULTI-LANE术语解释。新分支codex/protected-attention-dose-controls-20261028，保留全部原dirty分析。四组DOSE_MLP/DOSE_FIXED/DOSE_TASK/DOSE_CONTENT，seed1/8task/30epoch/B64，强B0配方与同一学习分类头不变；最终delta=Bound_.01(alpha·raw)，固定alpha1、另两门控2sigmoid初值1，三Attention每task QKV初始化相同。无区域/新查询/教师/A-B安装；新增目标独立优化，只原目标更新原参数。增加独立dose模块/入口/测试，原readout仅提取可覆盖的gate/correction方法、保留默认原运算；通用audit/queue支持四组锚点，评估增加真实gain/范数profile。

GPU0空闲约24GB、shm4GB；上限4作业/2评估，余量6GB/1GB自动排队。单测/真实两任务×4更新smoke及OFF/原参数/loss/RNG审计通过后才启动完整训练，训练后自动Val/Test及OFF/入口开关。输出output/protected_attention_dose、日志logs同前缀；完整路径/命令/协议docs/protected_attention_dose_execution_20261028.md。原主/test-only状态SHA仍ea2e261e…/e9d1479f…，保留。当前尚未远程测试或启动正式，不声称提升。

并行Summarize查询候选仅说明：保护原Selectors/原三路Task Forward，新增任务私有查询从sg(image tokens)可导取证，再交互修正；如用原查询初始化/偏移也要sg原查询，不能detach新摘要输出。原模型在同一30轮自己训练，新目标只更新私有取证与交互；不是拆分原Selectors、教师或组装。候选本轮未实现，效果未知。


## 2026-10-28：Attention幅度/内容门控真实前向归因完成

最终源码bd9ca07，run seed1_task027_20261028_02；24项相关测试、三阶段首64/真实尾batch smoke通过，GPU0完整Val/Test task0/2/7十条件全部完成（76.74/125.63秒，各峰值allocated952.08MiB）。01尾batch中间分量检查失败已保留，02恢复原每lane二维计算后完整重跑，未放宽native/OFF逐值检查。无训练更新，原model/runner/readout实现3950b35不变。

真实TRAIN缩放task7 Val42.4741/Test32.7131，原C4为41.8629/32.1570、OFF精确B0为42.4628/32.6800；真实Test对旧离线32.713263差−.000150。统一1%上限后，动态/常量/全开/错配门控Test最终32.7108/32.7113/32.7138/32.7125，所选三个Test阶段动态均未优于其余门控；Val动态微弱正差未转化为Test优势。控制仅检验最终gamma，不是否定Attention内部内容权重。实际task7 Test平均RMS动态/常量.824355%/.828841%，错配门控值集合逐batch相同。保留gamma只移除新增Face，Test贡献task0−.0216/task2−.0081/task7+.0043、Val三阶段均负；没有稳定Face互补证据。相对B0及动态−常量/错配的task7聚类bootstrap区间均跨0，不宣称稳定增益。

153份产物已同步逐SHA，归档98b79592883c69e54f386f0dbe8cd50086eb8b56169d8645269916b555e39878；NumPy1920指标误差≤4.27e−13、12源预测精确、320门控集合/12无效Face阴性控制通过。3源checkpoint SHA与前后模型摘要不变，主/test-only状态仍ea2e261e…/e9d1479f…；模型worktree clean。全部仅0/2/7，不能称8任务阶段平均。报告docs/protected_attention_gate_results_20261028.md，回传/分析output/protected_attention_gate/{complete_snapshot_20261028_02,analysis_20261028_02}。

本轮不自动启动更大门控/Attention训练。若重训，先将最终Bound放在写入末端统一1%上界，比较MLP/固定/任务标量/内容门控及同初始化，不自动叠区域；后续证据提取候选是保护原Selectors、另设仅供新增交互学习的并行Summarize查询，不是重新拆分原三路Selectors。上述均未实现/训练。所有分析仅本地追加，无main合并或删除。


2026-10-28归因预检补充：925d357相关24项通过、首64三阶段smoke通过、完整Test已结束；完整Val在尾batch函数式中间分量重算逐值检查停止，尚无完整双split结论。保留01失败/已成产物，不训练/改权重。改为严格保持原逐任务[batch,feature]的sigmoid/RMS形状，新增三个真实Val尾batch smoke；使用新run_id02完整重跑双split，原native/OFF预测仍必须逐值一致，不放宽其门槛。

2026-10-28门控归因预检：9a4e6d6已Git-only部署独立multi-lane-main-protected-attention-gate-20261028，6项新机制测试通过。增加多任务系数/类别所有权重放测试及reference NPZ/CLIP SHA核验。新增模式为临时hooks＋函数式特征读出，不改既有模型；继续相关测试后首64 smoke与正式双split。尚无新指标或训练。

## 2026-10-28：执行Attention幅度/内容只读归因（实现预检中）

用户授权先执行前述诊断再判断改进。本轮先现有C4 task0/2/7完整Val/Test，0训练，不自动展开第三项重训。新分支codex/protected-attention-gate-attribution-20261028，保留所有既有dirty分析。新增独立protected_attention_gate_attribution/run入口、GPU0 shell和6项机制测试；不改原模型/runner/Attention文件。

TRAIN常量取各任务最后5轮C3/C4输出RMS比和C4门控均值，在新Val/Test读取前锁定。10条件：原动态/关闭、TRAIN缩放动态/常量、统一最终1%RMS上限的动态/常量/全开/质量分组内打乱门控，以及固定原动态或常量门控时屏蔽Face摘要。Face干预不再同时改变门控，门控错配保留质量标记和分布。原三路编码共享一次，原ON/OFF与独立视图逐batch精确重放；记录最终写入RMS/门控分布/相关BCE效用，报告AMP插值差，不把3阶段均值称8阶段平均。

seed1/GPU0/B64/72线程/AMP-TF32/2单线程spawn workers/历史CPU每batch sigmoid，source3950b35只读，新源码独立Git-only worktree。GPU0约24GB空闲，其他卡忙且不使用。输出output/protected_attention_gate/seed1_task027_20261028_01、日志同前缀。协议docs/protected_attention_gate_execution_20261028.md；当前仅编译/语法预检，远程测试/首64smoke/正式待启动，不声称结果。main/test-only原状态SHA保留。

## 2026-10-28：受保护联合摘要六组完整8任务结果已核验

模型3950b35、调度8f8ab80全部complete；六组各240epoch/13950主更新/0主私有skip，48历史参数/loss/RNG及跨臂完整OFF/三路预测逐值通过。Val/Test原B0精确42.4628/49.4763、32.6800/39.5292。MLP42.5606/49.6355、32.7220/39.5811；普通联合Attention42.5156/49.6015、32.7075/39.5952；区域42.5427/49.6253、32.6746/39.5489；门控41.8629/48.8527、32.1570/38.9161；新增分支仅Full42.3024/49.2331、32.4328/39.1922。

原函数保护成功，下降在新增读出而非Face共训损害原Full。MLP/普通Attention两split各8/8阶段小正，Attention未明确胜MLP；区域Test5/8正，门控/Full-only两split0/8。Task0门控Test即-0.9464，非主要遗忘。C4 TRAIN末轮gate约.405–.986、输出RMS原特征4.0%–9.9%，C3固定约1%；原消融混合内容选择与最多10倍剂量。C2 Face入口Test条件贡献最终+.0093/平均-.0065，C4关闭Face有所恢复但不回B0；无可靠Face人群仍明显损伤。

本轮只读CPU已存logits固定0/.1/.25/.5/1缩放及TRAIN最后5轮RMS比例诊断（非新模型前向/部署成绩）：C4×.1 Val42.4882/49.5889、Test32.7108/39.6038；TRAIN等剂量Val42.4758/49.5873、Test32.7133/39.6085；支持写入过强为重要因素，恢复仍与MLP同量级。0/1端点概率逐值重放。原图分组500次bootstrap：C1 Test最终差[.0010,.0869]、C2[-.0253,.0846]、C2-C1[-.0746,.0377]，仅checkpoint条件区间/未多重校正/非多seed。

1235非pth回传逐SHA，归档b019f48542d2ae64afc3c26943d428f2409d5420225656d7c75471bddd1881ac，48源compact SHA核验。独立NumPy2464项指标最大误差3.98e-13；分析output/protected_joint_attention/analysis_20261028_01，报告docs/protected_joint_attention_results_20261028.md，脚本scripts/emotic/analyze_protected_joint_attention.py（可选CPU Torch dose-only）。main/test-only状态SHA及源模型HEAD保持，GPU0空闲。分析分支codex/protected-joint-results-20261028，当前未提交/推送报告或分析，不改模型/不重训/不合并。

下一步建议仅现有task0/2/7真实前向确认TRAIN锁定剂量，并比较同剂量动态/任务均值门控，分离内容与放大量；未执行。若再训，统一最终1%RMS上限，固定/任务标量/内容门控和MLP匹配对照，区域暂不自动叠加，不扩seed/层数/专家。Test参与开发全部披露；不可把离线最佳缩放当正式新结果。

2026-10-28结构说明补记：后续解释以实际C4的第6层缓存、终层[h0,hPerson]Query、[h0,hPerson,reliable·hFace,r,v]门控、同head及新增目标SG为准。不能把当前区域mask称作原Full摘要改写、逐patch跨视图对齐或完全人物特征隔离；C2/C3固定门控与C4学习门控有幅度自由度差，精度变化不能单凭门控非零认定内容选择有效。原Face辅助监督和原共享更新照常，新增交互不保证原Face AP提高或最终Test必涨。本轮仅说明/文档，六组GPU0训练与后续评估保持现有队列，不新起实验。

## 2026-10-28：GPU0六组并行已接管完成，原训练未重启

调度源码8f8ab80已Git-only部署multi-lane-main-protected-joint-scheduling-20261028，5项专项远程通过（本地连同既有8项共13）；dry-run确认detached sessions与无tty。新tmux protected_joint_six_20261028只终止原controller3339902，原四训练PID3339982/3340572/3341343/3342106及工作目录完全不变，任务0继续至25/24/24/23轮。C4新PID3347791、FULL_ONLY3348364已各完成2轮。六组均0主/私有跳步，已完成轮原关键loss全部与历史精确一致，队列0失败。

GPU0实测21358MiB占用/2724MiB空闲/100%；六组都活跃，pending为空。原模型源码HEAD/upstream仍3950b35且clean，cfg/训练过程不改；后续最多2评估自动跑8task Val/Test/OFF/证据移除和跨臂审计。runtime_scheduling_policy及handoff记录6并发，初始protocol保留4并发历史。原main/test-only状态SHA保持ea2e261e…/e9d1479f…。

接管8份产物已同步本地output/protected_joint_scheduling/handoff_20261028_01并逐SHA核验，归档f44bb06280b61fd36dde0b656fbfb01a6cd0a723517417f4c5a677450bec764a。本地补执行笔记，运行源码固定不动。北京时间12:01 ETA：总剩余约7.4万主更新，六并发每组84步约81秒，合计约6.2 updates/s；训练约3–3.5h，完整双split/开关对照预留30–50min且部分重叠，总约3.5–4.5h，预计今日15:30–16:30完成，不保证绝对时间。

## 2026-10-28：受保护联合摘要六组并行调度（预检中）

用户明确要求将C4内容门控和FULL_ONLY两组从排队改为GPU0并行。GPU0剩余9837MiB、原四训练占14245MiB；两组smoke峰值allocated约2821MiB，预计六组可容纳。新增独立调度分支codex/protected-joint-six-way-scheduling-20261028，原模型运行worktree/HEAD3950b35必须保持不变，以免破坏当前cfg/git一致性。

新增standalone expand_protected_joint_queue.py和5项测试（加已有8项调度测试，共13项本地通过）；使用独立Git-only调度worktree，接管原4个detached训练PID，只结束原controller PID、不发送信号给训练/worker、不重启训练，新增C4/FULL_ONLY。上限6训练/2评估、启动前空闲>=6GB和共享内存>=1GB，原B64/seed1/30epoch/8task/损失/随机流/验证标准均不改；自动续Val/Test/OFF/证据移除及完整跨臂审计。当前待远程dry-run/接管，未声称两新增组已启动。方案docs/protected_joint_six_way_scheduling_20261028.md。保留全部既有dirty分析及main/test-only。

## 2026-10-28：六组完整8任务已在GPU0启动

运行源码3950b35a734ccde4a9430e1ceb66ad45c840a042，local/remote HEAD/upstream一致，remote独立multi-lane-main-protected-joint-attention-20261028 clean。434项Track-A＋52项相关/新增测试通过。六臂各2任务×4更新smoke全部0skip，新梯度非零/有限；12个跨臂阶段原B0张量/关键loss/增强batch摘要及完整Val首64 OFF/三路预测逐值一致，旧任务参数/预测保护通过。单组峰值2808–2821MiB。

正式tmux protected_joint_seed1_20261028，batch seed1_20261028_01；C0/C1/C2/C3已并行训练（PID详见queue.json；前两组3339982/3340572），C4/FULL_ONLY自动排队。GPU0瞬时100%/14151MiB，其他GPU未使用。启动复核：C0/C1/C2已完成task0前3轮、C3前2轮，全部0skip，已完成轮的原关键loss均与历史逐值相同；新增梯度/输出非零，队列0失败。尚无正式效果结论。六组均跑seed1完整8任务30epoch/B64，结束后自动双split及共享前向ON/OFF/移除新增Person或Face证据，不按task0效果截断。

207份非pth预检产物已回传output/protected_joint_attention/preflight_snapshot_20261028_01并逐SHA核验，归档9454b2290832e64436560ee3219b1e88f5c3a1c2b897916dfeea4c3fc4d14dba。主树/test-only状态SHA仍ea2e261e…/e9d1479f…保持。完整协议docs/protected_joint_attention_execution_20261028.md；当前只本地补执行记录，不改运行源码、不合并main、不删除checkpoint。

2026-10-28预检补充：2297dbe远程独立worktree已Git-only建立，5项新增机制测试、434项Track-A与52项相关回归全部通过。增加跨臂原参数/关键loss/增强batch摘要与完整OFF/单路NPZ逐值审计，评估仍共享同次三视图编码；清理新入口未用import，不改旧前向。下一步六组两任务×4更新真实smoke，尚未正式训练，无效果结论。

## 2026-10-28：执行受保护联合摘要Attention六组8任务（实现预检中）

用户授权按方案修改并完整跑8task Val/Test，明确本轮不使用旧task0效果晋级门槛。新分支codex/protected-joint-attention-20261028，起点09a6729；保留此前dirty分析及其他工作树。新增protected_joint_attention/experiment/evaluation/run入口、GPU0队列与机制测试；原model.py不改，runner仅默认关闭的私有优化接口，历史build增加可选模型类型。

C0原B0、C1 CLS-MLP、C2联合摘要、C3目标区域、C4内容门控、FULL_ONLY新增分支仅Full；各seed1/8task/30epoch/B64，原主LR.0125/Adapter .0004/b32/index1/scale.03/aux.1/BCE-ASL，72线程/历史报告RNG，私有Adam-AMP/RNG隔离。第6层摘要，宽64/4头，固定门控.1/上界.1；新门控初值.1。真实标签自训、无A/B参数/答案、同一学习分类头。全部正式臂每任务要求原参数/loss/RNG精确匹配历史边界，否则报机制错误；新增分支不以效果决定是否继续8task。

GPU0空闲约24GB，GPU1–7忙且不使用；最多4作业、最多2评估、最低余量6GB/共享内存1GB自动排队。一次图像前向同时得到ON/OFF/移除新增Face和Person证据，评估2个单线程spawn workers，原CPU每batch sigmoid。epoch恢复包含两个优化器/调度器/scaler与完整RNG，验证前保存。输出output/protected_joint_attention、日志logs同前缀。当前编译/bash/diff检查通过，尚未远程测试/smoke/正式训练；不声称有效。

## 2026-10-28：下一候选为单模型的受保护联合摘要Attention（仅方案）

- 先讨论docs/protected_joint_attention_plan_20261028.md，不自动启动实验。完整均值是高成本函数互补参照，不作为A/B教师答案，也不视作Attention净收益已证实。
- 若授权执行，新实验分支实现原B0三路不变＋私有联合摘要Attention：并行目标区域Full Summarize、可靠Face硬mask、人物/任务条件Query、内容门控和末端有界特征残差；分类头仍原位置、原30轮学习。
- 新增loss只更新新增参数，原融合BCE/Adapter-ASL及aux0.1完整保留；原梯度必须从h0而非hFinal计算。初始化/RNG/optimizer/GradScaler分离，主路逐步重放32.6800配方，不能仅凭detach宣称保真。
- C0强B0、C1 CLS-MLP、C2联合摘要、C3目标约束、C4门控，并补新增分支Full-only source控制。先task0→0–2，再考虑8任务；历史标准、完整30轮、不Test挑epoch，GPU仅0。
- 成功需Test最终/阶段平均双提升及内容/人物对应干预成立；原Face单路提高不是必需，原三路应保持匹配基线。保护旧功能、非零Attention或门控不能代替收益。无效Face替换、OFF回B0、旧任务封存、成本均须审计。
- 本轮未修改业务源码/训练、未新建分支或提交推送；三份上下文同步更新，既有未提交分析保留。

## 2026-10-28：epoch轨迹与表征×head两项只读归因完整完成

运行09a6729源码clean，510新机制/相关回归通过；前64人全部5epoch×7条件smoke通过。GPU0并行完整Val/Test（157.52/194.44秒），共70完整读出条件/350视图含可靠Face指标，无optimizer/0训练。epoch1七条件恒等、epoch30原生完整四路logits/概率及mask/batch/人群精确重放，源权重/预测SHA与主树153/test-only5条状态不变。160产物已回传逐SHA，归档e8636acf7e65aab047394a8ae054cf97de60a3e570367d0a500909f5e47cde1c；独立指标误差3.41061e−13、BCE2.15383e−14。

结果：Test融合e5原48.8319/U2 49.1840/U4 49.5933短暂正差；U4在5–10轮间转负，U2在10–20轮间转负，e30为53.5004/52.9827/52.5870。不能按Test选e5替代完整30轮基线。e30 Test固定原头H0时R2融合52.9161、R4 52.7418仍大部下降；固定原表征R0换H2/H4为53.5274/53.5162近乎不变，差异不能仅归head。U4 Val原生59.5352掩盖R4H0 58.5729的表示损失与head/交互补偿；2×2是条件读出干预，不是唯一训练因果分解或部署组装。

报告docs/teacher_free_epoch_head_results_20261028.md，完整配置execution同前缀；分析output/teacher_free_epoch_head/analysis_20261028_01含轨迹、全交叉、逐类AP/BCE、漂移与核验。原模型/训练源码不改，无新训练、不扩8任务、不Test挑epoch；三个上下文已更新，执行记录和报告只在本地追加，源运行HEAD保持09a6729。

归因预检补充：8项新机制测试已通过；新增显式Face可靠mask、历史batch边界及完整人群长度的重放检查，不改变任何前向数值。完整既有回归进行中，随后Git-only同步该审计增强，执行GPU0前64人smoke后双split并行。

## 2026-10-28：执行现有epoch轨迹与表征×head只读归因（预检中）

用户授权两项归因。新分支codex/teacher-free-epoch-head-attribution-20261028从701e8df，保留此前dirty分析。仅新增teacher_free_epoch_head_attribution/run_teacher_free_epoch_head、GPU0shell和机制测试；原model/runner/训练源码不改，无optimizer/0训练。读取U0/U2/U4各epoch1/5/10/20/30，共15份；7个同epoch条件覆盖两套2×2，三份表征各编码一次复用head读出，Val/Test两进程GPU0。B64/72线程/AMP-TF32/2单线程spawnworkers/原CPU分batch sigmoid/历史人群不变；当前5类BCE/逐类AP/可靠Face/Full融合轨迹与交互完整保留，不按Test选epoch。

预检核心：epoch1全部条件恒等、原生缓存读出重放、epoch30对已有四路logits/概率逐值一致、源SHA/参数/无梯度保护；先专项/既有回归，再前64人smoke后完整双split。source原worktree701clean、15checkpoint均存在、GPU0空闲；原主树153/test-only5条状态SHA保持。配置/输入/输出/完整命令见docs/teacher_free_epoch_head_execution_20261028.md，输出output/teacher_free_epoch_head/seed1_20261028_01，日志同前缀。当前未正式启动，无新结论。三个上下文同步更新。

## 2026-10-28：无教师三视图互补学习止于task0，完整Val/Test核验结束

701e8df三组各30epoch/2520更新/0skip，队列正常complete但门槛stop_after_task0；未运行0–2控制或8任务。U0历史task0模型/30轮loss/RNG精确一致，Val59.5117/Test53.5004；U2 Val59.7144(+.2028)/Test52.9827(−.5177)；U4 Val59.5352(+.0236)/Test52.5870(−.9134)。本轮是5类task0，不能充当最终26类/8阶段平均。Test U4 Full51.1206/Person50.4471/可靠Face47.2882，均低原52.0252/51.3258/47.8107，不支持Face变强拖累Full的旧解释。Val小涨由类别/人群补偿，bootstrap融合差均含0。

U4实际额外步2124/2436生效、312拒绝(87.192%)；TRAIN25probe Full17负/3零/5正、融合21负/3零/1正，平均额外BCE负。训练BCE/ASL/rank更低但Test BCE/AP更差；局部一阶相对步约束不等于长期Full函数或AP保护，head/Adam轨迹仍可变，不能认定稳定梯度冲突或单一实现bug。无新增前向/训练。本版不自动8task/扩seed/收紧guard；下一候选先已有epoch轨迹与表征×head只读2×2归因，仅诊断而非部署拼装，尚未执行。

61份非pth已回传逐SHA，归档03da750b50c15d007311e10145d63ee7379d362563d0a5580f1856df83421811。CPU独立6融合/30视图指标最大误差2.55795e−13，人物/标签/mask/记录的30轮前2批一致，原8参考checkpoint及主树/test-only状态保持。输出output/teacher_free_shared_learning/analysis_20261028_01，报告docs/teacher_free_shared_learning_results_20261028.md，脚本scripts/emotic/analyze_teacher_free_shared_learning.py。分析分支codex/teacher-free-shared-results-20261028/HEAD701e8df；运行源码不改，报告未提交，不合并main。所有三个上下文更新，GPU0已空闲。

启动后核验：U0/U2已完成task0前3轮、U4前2轮，均0跳步；三组正式第1轮全部模型张量、实际增强batch和Python/NumPy/CPU/CUDA RNG逐值一致，证明λ0不改变原路径。第2轮起实际新增更新和每100步同批TRAIN反事实检查运行正常，GPU0瞬时100%/10684MiB。正式epoch1_identity_audit.json已生成；当前仅运行健康检查，没有正式Val/Test结论。

## 2026-10-27：无教师互补学习预检通过，GPU0正式三组运行

源码701e8dfa已Git-only同步clean远程multi-lane-main-teacher-free-shared-learning-20261027，local/remote HEAD/upstream一致。502项机制/Track-A/RNG/相关回归通过；U0/U2/U4真实2任务各24更新共48步/组、0skip，初始与实际增强batch相同，额外图与原融合logits差0，旧参数/CLIP/参考8checkpoint SHA保持，Val首64两阶段完成。单组峰值分配约2.82GB；U4实际非零更新率task0/1为37.5%/50%，仅验证机制，不代表收益。

正式tmux teacher_free_shared_seed1_20261027，U0/U2/U4训练PID3185093/3185652/3186398在GPU0并行；初始GPU99%/7.8GB瞬时。seed1task0×30轮，之后自动完整Val/Test，符合既定门槛才续0–2并补U1/U3，不自动8任务。不要重复启动；完整配置/输出/日志见docs/teacher_free_shared_learning_execution_20261027.md。53份预检产物已回传output/teacher_free_shared_learning/preflight_snapshot_20261027_03逐SHA，归档a5eb0203a625ab3fcaef2d79a2d6541dd36f06350532726d22f848a856591ea4。主树153/test-only5 dirty状态SHA保持；执行笔记仅本地追加，运行源码固定，不合并main。


二轮预检：排序/Adam24项已通过，恢复/关闭机制精确更新已通过；集成暴露原move_model_inputs将可靠mask转float，新增目标接入现按原辅助监督语义显式转bool，不改原输入流。新增记录额外图与原融合logits差异，便于真实AMP重放核验。继续标准模块入口回归，未正式训练。

预检补记：e750818首轮502项发现Torch2.0 any(tuple)兼容、完整own恢复读取weights_only、集成测试task属性及stdin驱动spawn测试问题；仅在预检，未正式训练。已改用逐维any、仅own恢复显式完整加载、正确current_task_id，并以python -m unittest标准入口重跑；新增文件EOF空行同步清理。不改目标/配方/旧权重。

## 2026-10-27：执行无教师三视图互补学习（实现完成，预检中）

用户授权按方案修改并GPU0高效运行。新分支codex/teacher-free-shared-learning-20261027，基于28a1cbb，保留旧dirty分析。新增自然正负对互补排序、Full/融合实际Adam候选步保护、同批TRAIN反事实probe、runner可选接入/完整epoch恢复、无教师训练评估与GPU0显存队列；原model.py/B0前向不改，head原30轮全程学习，原BCE/ASL保留。初始U0/U2/U4 seed1task0×30轮、B64、主LR.0125/Adapter.0004、b32/index1/scale.03、72线程、历史报告RNG；Test/Val完整评估。符合门槛才自动续0–2并补U1/U3，不自动8任务。无A/B任务参数或输出导入。

GPU0空闲24GB，其余忙，仅0；上限3并发、训练余量6GB/评估4GB、共享内存保护。正式前指定机制/回归测试及真实三组2任务×24更新smoke，关闭机制与U0历史精确重放必须核验。执行协议docs/teacher_free_shared_learning_execution_20261027.md，设计/细节docs/teacher_free_shared_view_learning_plan_20261027.md。当前本地编译检查通过，远程预检/正式待启动，不声称方法有效。主树HEADa322419/status SHA ea2e261e…153条，test-only0b59138/e9d1479f…5条均保留；不合并main，不删除checkpoint。


2026-10-27说明补充：再次核对原论文§3.3与当前runner/model，明确三路辅助监督是扩展而非原论文机制；原B0已监督融合和各路，.1作用于有效视图均值，Face子集先平均。新机制只增互补排序/额外反向分配和新增Adam步保护，不切断原融合梯度、不冻结原head、不保证完整Full功能不变。详见方案第12节；本轮仅说明/文档，无模型训练或新实验。

## 2026-10-27：下一候选是无教师的互补监督与共享更新保护

- 本轮仅方案；不自动训练。先审阅`docs/teacher_free_shared_view_learning_plan_20261027.md`，不是重启A/B教师或额外5轮固定头。
- 若用户要求执行：从修复后seed1强B0完整配方开始，同一模型30轮/head正常学习；保持三路Task Forward、共享任务参数、原融合BCE/Adapter-ASL和历史RNG，新增Full条件真实标签排序及新增Adam步的Full/融合保护。
- 先实现可选训练模块/测试/相对路径入口；关闭路径必须精确重放B0。真实20–50步验证Adam候选状态/AMP/old-mask/Face零梯度/成本，不先承诺训练速度。
- U0/U2/U4先完整task0，两split及各视图检查；通过后续0–2并补U1/U3，再完整8任务Val/Test。按原完整TRAIN与samples_intersect_seen_classes；若另设TRAINholdout，匹配基线不可标32.6800。
- 不以全部拒绝新增更新、Face单路AP升或单独Test最终升作为成功。目标Test最终32.6800和均值39.5292均提升；验证互补内容的条件贡献，必要时补TRAIN确定的等剂量控制。Test已参与开发必须披露。
- GPU后续只用0。当前规划分支codex/teacher-free-shared-view-plan-20261027，HEAD28a1cbb，未实现/提交/推送。三份上下文已更新；现有结果、主树/test-only不修改。


## 2026-10-27：历史底座精确恢复，warmup/final归因完整结束

运行28a1cbb87052ecce8d95114153a736882bb70119全部完成，GPU作业已退出。fresh SELF_BASE 8任务×30epoch、0跳步，每阶段权重/30轮loss/Python-NumPy-CPU-CUDA RNG精确匹配历史；独立CPU再比8×41非视觉Tensor原始字节SHA均相同。Val42.4627720833/49.4762827230、Test32.6800218743/39.5291640411，cF1/oF132.8226918781/49.5458931279，历史配方已准确恢复；先前32.1748来自报告RNG遗漏，不是评估标准变化。原72源checkpoint SHA保持。基线同模型Full单路Test31.2373→融合32.6800，融合收益仍在，不能代替独立训练单Full消融。

原四组task0/2/7四条件完整Val/Test归因全部完成，不重训适配组，不把修复基线混入旧匹配差分。联合约束组当前5轮Val task0/2/7变化−1.743021/+.017516/+.012691，Test−.427226/+.007488/+.019311。task0损失主要来自ASP变化Val−1.740831/Test−.397587，当前Attention−.002190/−.029638；Face全人群AP升但可靠Face降，不能当有效面部证据提高。旧类别各视图logits和概率在warmup/final间精确不变，当前head不变，初始当前Attention恒等，BASE阴性控制通过。

联合约束task7相对原匹配BASE：Val适配前−.486082、当前ASP+.012794/Attention−.000102、最终−.473390；Test适配前+.082832、当前ASP+.019638/Attention−.000327、最终+.102143。约81%的Test最终优势早于当前5轮。当前新3类Test AP+.167362，但仅3/26类能改变，整体+.019311；Val旧23类贡献warmup差−.426227，当前5轮无法修复已冻结旧类。失败模式是首任务排序受损后封存、后续起点已变，不是每阶段5轮都持续损害或已证明单一S/P梯度冲突。

523份非pth产物已回传逐SHA核验，归档290e0d89c9e470283751457e688b2e74b5ea554c588a817979b5fda9b95b1861；另回传独立Tensor比对JSON。CPU独立复算112融合/560视图含可靠Face，最大误差6.394884621840902e-14，分析output/historical_rng_replay/analysis_20261027_02，日志同前缀，脚本scripts/emotic/analyze_historical_rng_warmup.py。报告docs/historical_rng_warmup_results_20261027.md。

当前分析分支codex/historical-rng-attribution-results-20261027，未提交本次分析/文档，remote运行源码保持clean。下一步保留恢复的历史强底座，暂不把3适配组全重跑/扩seed/beta/Attention，先围绕首任务原排序保护与原30轮共同学习设计同预算小控制。本次仅分析、同步产物和文档，没有另起训练/前向或合并；三个上下文均更新，旧主树/test-only保留。


执行更新：28a1cbb87052ecce8d95114153a736882bb70119已Git-only部署独立remote multi-lane-main-historical-rng-warmup-20261027，本地/remote HEAD/upstream一致，remote clean。4 RNG专项+2边界归因专项、528全回归通过（46.186秒）；两任务各2更新BASE smoke、Val首64，以及四来源task0/2/7 Val首64四条件smoke全部完成，来源final前缀四路logits精确重放。165份smoke/测试/协议已回传output/historical_rng_replay/preflight_snapshot_20261027_01逐SHA核验，归档5ff6553ae437ac823523cd3701250878d12280343f0916a895a5eaee8685cc55。

正式tmux historical_rng_warmup_seed1_20261027运行中；BASE训练PID3123323、归因Val3123850、Test3124588，仅GPU0。BASE task0前8epoch正常、每轮84更新/0跳步；Val归因已完成BASE/NO/MAIN三来源并进入最后来源，Test已完成BASE/NO、进入MAIN。无故障。BASE仍须30轮权重/loss/所有RNG逐值验历史后进入下一任务，全部8任务完毕才自动Val/Test，不预称32.6800已恢复。旧源权重72份SHA协议锁定，原主树/test-only保留。运行源码28a1cbb固定，执行记录仅本地追加。


## 2026-10-27：执行历史随机流修复与现有warmup/final归因

用户授权执行两项：修复并验证历史随机流；只读比较现有四组task0/2/7 warmup30与final35及ON/OFF。新分支codex/historical-rng-warmup-attribution-20261027，保留前轮未提交报告。仅添加两次CPU int64取种子、任务边界验证和独立归因/队列；不改模型/目标/超参数，不增加Attention容量/beta/seed。将只重跑SELF_BASE完整8任务×30epoch，标准CLIP+任务参数fresh初始化，不复制教师任务参数；每阶段逐值比对历史全部权重、30轮loss和Python/NumPy/CPU-CUDA RNG，失败立即停。额外3学生不重训。源72 checkpoint SHA前后检查。

原四组checkpoint只读任务0/2/7 Val/Test，WARMUP_ON/OFF与FINAL_ON/OFF；检查本学生head和旧参数相同，完整final来源前64 logits精确重放，已完成final预测验证复用。ON指旧Attention路径也开启，不能简单把所有ON-OFF都叫当前新增残差；记录当前类AP和成对差分隔离。原四组归因仍用原匹配32.1748控制，不混入修复后历史32.6800轨迹。

GPU0、B64、72线程、AMP/TF32、训练workers2/prefetch1；评估2单线程spawn workers。BASE主LR.0125、Adapter.0004、Adam/reset/cosine/WD0、原ASL-BCE/aux.1、b32/index1/scale.03、关闭Person颜色扰动。先专项/全测、2任务2更新BASE smoke及完整来源Val首64归因smoke，再并发BASE训练与只读Val/Test，BASE结束自动8任务Val/Test。输出output/historical_rng_replay/seed1_20261027_01与output/self_trained_warmup_attribution/seed1_20261027_01，日志同前缀；配置docs/historical_rng_warmup_execution_20261027.md。当前实现/检查中，尚未正式启动。


## 2026-10-27：自训练四组完整8任务结果已核验，联合约束未达成目标

fast评估seed1_20261027_01全部完成，a61ffc2 clean；训练/恢复fb5f6cc。加速后20.01分钟结束；112完整阶段/开关pass，32源compact SHA不变。435份非pth产物已回传逐SHA核验，归档1732162b225e56c1cefcaa63d9a793ad8040a51364c6a7b350473c85e727c0c8。独立CPU重算112融合/560视图含可靠Face，最大误差6.3949e-14；批次/旧参数/教师保护与0跳步均通过。报告docs/self_trained_view_transfer_results_20261027.md；独立分析output/self_trained_fast_evaluation/analysis_20261027_01。

Val最终/均值、Test最终/均值：SELF_BASE42.5058/49.5676、32.1748/39.2752；无约束41.6845/48.4685、31.9945/39.0460；主参数约束41.9213/48.5290、32.0159/38.8470；主参数+Adapter42.0324/48.7930、32.2770/39.1375。最后一组比等预算无约束Val+.3479/+.3245、Test+.2825/+.0916，但比BASE Val−.4734/−.7747、Test+.1021/−.1377；Val8/8低、Test仅task6/7高。关闭Attention后最后组Test32.2808/39.1470，8/8阶段反而更高；Full/Face开关logits全部精确不变，Person输出有变化但无稳定排序收益。不能归为新Attention成功；自训练单推理模型/无外部参数拼装工程要求满足，精度目标不满足。

基线复现疏漏已定位：历史每任务报告预测+视图诊断两次DataLoader迭代器各消耗全局CPU RNG int64一次；新入口全部后评估且未补取种子。task0全部权重/CPU-CUDA RNG/30轮loss与历史精确一致，task1起分叉；八任务新warmup CPU RNG补2*t次int64后都等于历史对应before_reporting状态，CUDA RNG仍相同；本轮task1实际64人IDs已用无报告种子轨迹精确重放。评估标准未变，加速发生训练后不改变权重。32.1748不能取代历史强B0 32.6800/39.5292；四组内部实际增强批次仍匹配，保留其内部比较。随机流只读诊断已回传，不声称已修复/重训。

下一步优先修复并验历史随机流重放，再考虑训练；若继续归因，可只读比较本学生task0/2/7已保存warmup30与最终35及ON/OFF，拆本任务适配与前任务S/P传递。当前未自动启动新训练/前向、未扩seed/调beta/合并main。本地新分析分支codex/self-trained-transfer-results-20261027，分析脚本和报告未提交；原主树/test-only保留。


实测加速（2026-10-27）：GPU0连续10秒均100%利用率、约6–7GB显存；各评估主进程CPU约1–2核（另有2个单线程加载worker），替代此前每进程30余核。切换339.96秒已新增29个完整pass/95004前向人次，另23旧pass/68654人次核验并SHA复用；BASE Val/Test和NO_CONSTRAINT Val已完成，其他自动接续，无报错。实际总量406854人次，剩243196；当前279.46人/秒线性约14.5分钟，考虑后段任务/尾部并发建议估剩余15–25分钟，不再沿用10–13小时。输入与四路logits精确重放、4专项/522回归通过。8份预检日志/协议/吞吐与handoff已回传output/self_trained_fast_evaluation/preflight_snapshot_20261027_01逐SHA核验，归档6e46021af5e6e6ac50f1236a76baa25e70e0466afe9249ec1fb5be4339175e42。正式结果仍在运行，以fast新队列为准，不误读原已停止queue的running快照；remote a61ffc2 clean固定，执行笔记仅本地追加。


评估接续更新：新入口a61ffc2已Git-only部署并启动tmux self_trained_fast_eval_seed1_20261027；4新增/522全回归通过。task0/7 Val共256人，workers0/2/4的输入及四路logits精确一致。2worker稳定加载128人1.135/1.495秒，原2.514/2.758秒；4worker无收益，采用2worker。原controller及4评估已按PID/cwd核对后有序停止，已完成完整pass均验证并SHA复用，原权重/结果保留；新4评估3002571/3003123/3003989/3004732在GPU0继续。尚无整网实测倍数或新完整mAP。


## 2026-10-27：评估CPU争用优化（仅评估，预检中）

四组8任务训练已结束。原4评估workers0各72线程、合计约130核而GPU0空闲，旧吞吐预计剩余10–13小时。新分支codex/self-trained-evaluation-throughput-20261027仅新增独立fast评估入口，已有multi_lane源码与训练Git字节一致。B64/模型72线程/AMP-TF32/旧seen-positive不变，图像预处理移至2个单线程spawn worker、prefetch1。先task0/7 Val各128人核对像素/标签/ID/mask及四路logits逐值一致，再接续32个原学生checkpoint的完整Val/Test和ON/OFF，0训练。已完成pass经JSON+两NPZ、人群及指标验证后按SHA复制复用，原输出保留。新输出output/self_trained_fast_evaluation/seed1_20261027_01，配置docs/self_trained_fast_evaluation_20261027.md。当前预检准备，尚无实测加速比。


时间估算更新（2026-10-27）：恢复run resumed_seed1_20261027_03的四组8任务训练现已全部结束、8份最终checkpoint各组齐全、无Traceback。当前在只读评估：BASE Val/Test及NO_CONSTRAINT Val/Test四进程运行，MAIN两组四评估待队列。已完成19个阶段/开关pass，约56231/406854前向人次，评估已运行111.1分钟；按已完成pass吞吐外推剩余约11.55小时，考虑当前未完成pass建议报10–13小时（未优化吞吐）。4评估各72线程、workers0，单进程约30余CPU核占用而GPU0大部分空闲，存在明显CPU预处理/线程争用瓶颈；本次仅只读查询和估算，未改线程、加载配置、权重或重启进程。正式完整Val/Test仍未完成。


## 2026-10-27：数值审计修复并继承自身进度，正式Val/Test仍待完成

运行源码fb5f6cc745ae6c103990876e7bc0e8370a1d155f，fix/self-trained-audit-recovery-20261027，本地/服务器HEAD/upstream一致；远程新独立worktree multi-lane-main-self-trained-audit-recovery-20261027 clean。11项专项与518项完整测试（1跳过）通过。四组真实已完整训练task0及task1warmup的A/A/B/A审计全部在数学SDP下四视图重复/更新后旧路径/回放差0，旧参数SHA一致；smoke_seed1_20261027_04全部成功。

已恢复运行output/self_trained_view_transfer/resumed_seed1_20261027_03，tmux self_trained_recovery_seed1_20261027，GPU0并行4组PID BASE2880903/NO2881441/MAIN2882175/MAIN_ADAPTER2882919。每组6份完整本组自身状态导入，模型Tensor SHA与CPU/CUDA RNG不变；超参数相同、核心训练代码逐字节及训练调用/初始化AST相同，未从其他模型拼装分路/head。task0及task1前30epoch跳过不重训，后三组task1遗失5epoch从自身warmup重放；NoD重放前2epoch日志值与原日志一致（不声称找回丢失权重）。现在四组均跨过task1原故障点：BASE/NO已保存task2并到task3 epoch5/2，MAIN在task2第30epoch，MAIN_ADAPTER第29，均0跳步/无Traceback。task1实际数学SDP保护审计四组所有差值0。全8阶段Val/Test与ON/OFF仍将在完整训练后自动执行，当前尚无正式mAP。

103份原失败日志/进度/完整回归/真实checkpoint审计/smoke产物已回传output/self_trained_view_transfer/failure_snapshot_20261027_01并逐SHA核验；归档03e4bce424550644d31e4f84cd42dd10d2871ed8586b3a6335d10c1aa37ff534，24份原学生checkpoint SHA未变、pth保留服务器。主树/test-only原HEAD/status SHA保持。运行源码固定fb5f6cc，不在运行中同步本地执行记录。不能用原实验进程结束或训练loss判断三视图方法精度；现只确认工程审计原因并恢复，待完整Val/Test。


## 2026-10-27：正式四组实际task1审计误停，尚无8任务Val/Test

用户要求完整结果复核。remote6613c56 formal_seed1_20261027_02四组均只有task0最终/完整恢复和task1前30epoch完整恢复；后三组task1额外5epoch在日志中完成但因审计前未保存而丢失RAM权重。task2–7未运行，Val/Test未启动，不能判断方法成功/失败。错误均uncontrolled_probe_noise，未更新模型重复前向最大1.14e-5–1.37e-5超过旧1e-5阈值。只读完整已训练BASE的A/A/B/A诊断：优化SDP重复约1.34e-5，FP32禁TF32且强制数学SDP后四视图重复/更新后旧路径/回放差均0，旧参数SHA相同，定位为审计计算核非确定性，不是已证明的遗忘或Face冲突。

新修复分支fix/self-trained-audit-recovery-20261027，保留本地未提交运行记录。只改TRAIN审计使用数学SDP、增加同一学生完整恢复导入与审计前保存，不改训练/评估AMP、目标或超参数；恢复源原formal02，目标resumed_seed1_20261027_03，GPU0四组。task0及task1前30epoch不重训，后三组task1丢失5epoch需从自己的warmup重放，明确记录Git/SHA，不声称找回丢失权重。将先专项/全测、四组真实checkpoint审计及smoke再自动接续；当前实现中，尚无完整mAP。报告docs/self_trained_view_transfer_failure_recovery_20261027.md。


正式启动核验补充：四组均已完成task0首个epoch，各84成功更新/0跳步，四组loss .67947638和Adapter loss .03236856完全一致并匹配历史B0首轮；BASE已到第3epoch，NO/MAIN到第2，MAIN_ADAPTER第1。GPU0约14243MiB/100%，无Traceback。此时尚无正式8任务Val/Test结果。所有运行源码继续保持remote6613c56 clean；本地仅追加执行记录，避免运行中变更评估Git。


## 2026-10-27：自训练四组正式8任务已在GPU0并行启动

最终运行源码6613c5610deccda09102f606b02fe6791485823c，本地/服务器branch、HEAD、upstream一致；独立远程worktree multi-lane-main-self-trained-view-transfer-20261027 clean，运行期间不更新其源码。8新增专项、515完整测试（1跳过）通过。smoke_seed1_20261027_03四组两任务/每阶段2更新和Val首64全部成功，自己head与无外部学生参数加载、旧参数SHA、FP32重复控制、跨组实际batch像素/标签/ID均通过；MAIN_ADAPTER两任务beta_A约.00013736/.00025232，真实约束非零，教师未更新。

正式formal_seed1_20261027_02已自动启动，tmux self_trained_view_transfer_seed1_20261027，GPU0四并行PID BASE2778683/NO_CONSTRAINT2779210/MAIN2779933/MAIN_ADAPTER2780656；当前约14175MiB/100%。BASE和NO_CONSTRAINT首epoch各84更新/0跳步，loss .67947638/Adapter .03236856精确匹配历史B0首轮，尚不代表8任务完整复现。其他组正在进入训练，尚无正式Val/Test成绩。四组训练结束后自动完整8阶段Val/Test及Attention ON/OFF，eval仅加载自己学生，所有源24checkpoint只读。

179份三轮smoke/初始化审计/批次和数值诊断/完整测试日志已回传output/self_trained_view_transfer/smoke_snapshot_20261027_03并逐SHA核验，归档ad30756b8688f782234c295868fcccc1ab73e4ab37da8e01ee10ae84e5f0959d；模型pth保留服务器，不在轻量快照内。旧主树/test-only的原HEAD/status SHA保持。smoke01与formal01预启动中断记录完整保留，正式不复用它们。以下失败/等待条目为历史记录，以本节为准。


正式接续门控已加强：smoke必须同时实际启用主参数和Adapter功能项（beta>0），并验证教师参数/梯度未改变；不再仅依据进程exit0就启动正式。真实正式任务若ASL确实无梯度仍按预声明零信号策略记录，不用Test改系数。


smoke02数值检查与再次门控：513测试（1跳过）、smoke02初始化/四组两任务训练/Val/像素标签流/旧参数与FP32重复探针全部通过。但Adapter-ASL实际训练范数约1.6e-7，而1024校准尺度在task1误读为0，令MAIN_ADAPTER的beta_A=0。正式自动接续只预启动BASE（PID2689011），已在核对cmdline后SIGINT中断；其他正式训练未启动，formal01产物保留不复用。修复校准为匹配初始训练scaler的65536，只在梯度实际溢出时依次降至8192/1024/128/1，逐组记录尺度；不改变训练scaler或ratio.25/maxbeta1，新增半精度下溢和溢出单测。将运行smoke_seed1_20261027_03，通过后正式使用formal_seed1_20261027_02全新统一启动。


smoke审计修正：ac4b5a5的smoke_seed1_20261027_01在BASE/NO_CONSTRAINT task1旧预测逐值断言失败，正式未启动，旧产物保留。只读ABA诊断发现旧参数SHA完全一致，连同一模型A重复A也有TF32约8.1e-5的最大差；关闭TF32后AA/AB均约2e-7，属于CUDA舍入而非参数漂移。改为现有audit_precision关闭TF32的FP32探针、增加未更新模型重复控制，预测固定atol1e-5且逐项记录，参数SHA仍严格；新增审计单测，不改变训练/评估标准。下一smoke run_id改为smoke_seed1_20261027_02，formal01尚未创建。


远程测试更新：30df39b已Git-only部署独立worktree，5新增专项及512完整测试（1跳过）通过。正式前额外清理阶段切换时残留grad，避免冻结head的旧梯度污染诊断；待最终源码再验证后启动smoke/正式。原主树/test-only状态SHA与HEAD保持。教师总预算90epoch/任务加本轮学生35epoch，约125epoch/任务，须与单模型30epoch的训练成本区分。


## 2026-10-27：用户要求自训练成型，取消学生外部参数装配（实现中）

用户授权执行单模型路线，并明确最终模型必须自己训练，不能组装其他已训练模型参数。新分支codex/self-trained-three-view-transfer-20261027。旧计划的A参数初始化/B0头复制不再采用：学生仅加载标准预训练CLIP，任务参数和head全新初始化；每任务先原三视图硬监督30epoch，再冻结自己学得head、启用自己的Person Attention训练5epoch；下一任务继承自身适配后的S/P。A/B/B0只用于冻结教师目标，训练约束组三视图同增强读入，推理仅学生，不复制教师任务参数或head。

四组SELF_BASE（30epoch）、SELF_NO_CONSTRAINT（30+5）、SELF_MAIN_CONSTRAINT、SELF_MAIN_ADAPTER_CONSTRAINT（后两者同35epoch）。B64/训练workers2/prefetch1/evalworkers0/72线程/AMP-TF32/GPU0，前30epoch主LR.0125、Adapter.0004，后5epoch均.0004，Adam/reset/cosine/WD0/原BCE-ASL+aux.1。beta由首个实际TRAIN batch分组梯度定标，ratio.25/max1，不按Test搜索。独立额外阶段RNG恢复历史warmup流，实际ID/标签/像素跨组核验。旧参数/固定TRAIN旧预测、自己的head、教师无更新和24源SHA均验收。先单测/GPU初始化/两任务两更新smoke，再自动完整8任务Val/Test/Attention ON-OFF。

配置docs/self_trained_view_transfer_execution_20261027.md；输出output/self_trained_view_transfer/{smoke,formal}_seed1_20261027_01，日志logs/同前缀。当前实现和语法检查完成，尚未跑远程测试/训练。旧32.8864换头候选与R5仍保留为不同来源参照，不声称新学生天然等于其指标。PROJECT_CONTEXT/WORK_LOG/TODO_NEXT已同步，本次不改AGENTS。


完成更新（2026-10-27）：R5已用b017389源码完成GPU0/seed1/旧标准完整8任务Val/Test与隔离测速，无训练。Val44.1550/51.2964，Test33.6248/40.5129；对A Test+.7384/+.6442，两个split均8/8阶段升。对既有R2 Test+.5718/+.4658。与当批logits均值Test最终差−.000934/均值+.000461，最大logits差.0078125。全部来源预测精确、24源SHA及参数保持；507测试（1跳过）通过。A/R5隔离B64耗时241.06/492.98ms，约2.045倍，非视觉参数2661658/6882842，峰值925.71/946.22MiB。266产物已回传并逐SHA核验，归档f330a43266eadd7b08d4bb1250796e5d415f82cb8174d3bff73372d2158b9f1e；本地64融合指标/128来源视图/64logit均值关系通过，误差7.11e-14，日志已入logs/。主树/test-only状态保持，GPU作业已结束。

固定均值完整函数组合现可作为更强实测参照，但仍为双来源，未证明新Attention因果收益或单模型训练成功。后续单模型方案docs/single_model_function_transfer_plan_20261027.md，保留三视图输入、同增强教师、融合功能约束、ASL/BCE分组及同预算U0–U3；仅为研究方案，未新增训练。报告docs/fixed_model_mean_results_20261027.md。以下启动/等待描述均为历史记录。


执行更新：源码b017389已通过Git-only部署远程独立worktree，branch/HEAD/upstream一致且clean；4新增专项通过，完整507测试（1跳过）通过。formal_seed1_20261027_01已启动，tmux fixed_model_mean_seed1_20261027。task0/2/7首64预检来源全预测精确、最大均值logit误差.005859375；隔离测速A241.06ms/R5 492.98ms，非视觉参数2,661,658/6,882,842，峰值925.71/946.22MiB。Val/Test PID2431901/2431902正并发完整8任务，尚无最终成绩。无训练更新。新单模型功能迁移候选文档docs/single_model_function_transfer_plan_20261027.md只作方案，明确同三视图增强、融合logit功能约束、ASL/BCE梯度分工、U0–U3同预算控制；没有启动其训练。主树/test-only原状态SHA保持。

## 2026-10-27：固定整模型均值的真实前向验证（已授权，执行中）

新分支 `codex/fixed-model-mean-forward-20261027`，保留所有既有未提交分析。新增 fixed_model_mean_forward 与队列入口，复用来源装载、历史逐 batch sigmoid 导出；旧导出函数仅增加可选来源标签，原 R0–R4 行为不变。A/B 分别保留完整三视图、各自 Full 摘要依赖与任务参数，先各自固定三视图融合，再将融合特征 0.5/0.5 平均后接同阶段共同 B0 线性头，不做额外归一化、门控或训练。并导出 A、B 与当批 logits 固定均值，不能把离线控制当作真实 R5。

seed1 / GPU0 / EMOTIC / 8任务 Val-Test / 历史 samples_intersect_seen_classes / B64 / workers2 / 72线程 / AMP-TF32；先单测及 task0/2/7 Val 首64重放，两个隔离资源测试，然后 Val/Test 并发2。所有来源输出（含序列化概率）逐值一致、24源 checkpoint SHA 及参数不变为验收；R5 与 logits 均值的 AMP 差记录，预声明最大绝对 logits 容差 .03125，不按结果搜索系数。无优化器/LR/epoch/新训练 checkpoint。

输出 `output/fixed_model_mean_forward/formal_seed1_20261027_01`，日志同前缀 `logs/`；远程独立 worktree 拟为 multi-lane-main-fixed-model-mean-forward-20261027。完整配方和相对来源路径见 docs/fixed_model_mean_execution_20261027.md。预检/正式尚未完成，无新正式指标。后续研究单模型时必须保留同预算无蒸馏控制，不能复提已失败的冻结底座只训新 Attention 或弱辅助教师蒸馏 Full；当前不启动新的训练路线。



## 2026-10-26：R0–R4进一步机制分析与更强的简单均值控制

本地新分析分支codex/protected-view-mechanism-analysis-20261026，保留既有未提交报告；仅读取预测，未训练/运行新GPU前向/改远程模型源码。R2 Val/Test均8/8阶段优于R0；R1 Test均值增益约71.4%来自task0，其他7阶段平均仅+.0179。R3相对R2 Val8/8升、Test6/8降，最终+.0073/均值−.0060；其Test边际CI最终[-.0966,.1841]、均值[-.1346,.2016]均含0。R2单人Test+.2936、多人仅+.0462，可靠Face多人交集略降，不证明人物对应已解决。R3对R2虽16类提高，Pleasure/Disconnection−1.156/−1.072抵消多类小收益。

新增固定0.5整模型logit均值（离线CPU控制、非新增真实模型前向）：Val44.1558/51.2957，Test33.6258/40.5125，比R2+.5727/+.4653；两split均8/8阶段升、各22/26类最终AP升。服务器原Torch/历史batch sigmoid独立CPU复核相同，IDs/labels/classes/masks核验，源预测SHA不变。500原图配对，均值对R2 Test最终CI[.3588,.8744]、平均[.2379,.8114]，不含seed方差/不校正Test选择。固定均值中B全部Attention关闭仍Val44.0965/51.1880、Test33.5854/40.4390；全部开启再+.0404/+.0735 Test，Face自身条件+.0284/+.0530。大部分组合差值在B OFF时仍保留，不能将全部收益归于推理残差；OFF共训参数仍保留，不是重训无模块基线。

下一步优先建议固定0.5均值的真实特征融合/B0头8任务Val/Test，尚未启动；然后比较同预算两个Person-only模型平均、缓存/裁剪，再谈复杂门控/单库蒸馏/人物绑定。R2仍是R0–R4实测平均最强，但视图硬选择未优于新离线均值，不再优先继续扩大R2结构。报告docs/protected_view_forward_mechanism_analysis_20261026.md，原结果报告加链接并修订后续优先级；CPU脚本scripts/emotic/analyze_protected_view_mechanisms.py，产物output/protected_view_forward/mechanism_analysis_20261026_02，日志同名。成本与单seed局限仍保持。



完成更新（2026-10-26）：R0–R4完整8阶段Val/Test及隔离资源测速均结束。前向780732e、CPU导出修复0c66113；503项回归通过。formal_02正确导出保持原GPU logits，按历史batch sigmoid，R0全部16阶段精确重放；R1无可靠Face融合保持。CPU独立复算80融合/240来源分路逐值检查，最大误差7.11e-14，16 R1保护通过。24源checkpoint及80原始前向NPZ SHA保持；415产物已回传并逐SHA核验，归档7d96d1228840b4a663d7fe8f03d5fc8f2814cdfa2d2ea0689551e5e359871256。远程0c66113 clean，主树/test-only的原HEAD/status SHA未变，无训练/新checkpoint。

Val最终/平均、Test最终/平均：R0 43.0741/50.2828、32.8864/39.8687；R1 43.1793/50.4647、32.9426/39.9234；R2 43.4216/50.5635、33.0530/40.0471；R3 43.4727/50.7186、33.0603/40.0411；R4 43.4768/50.5177、33.0489/39.9378。R2相对R0 Test+.1666/+.1785，R3+.1739/+.1724；R2平均最高、R3最终最高，不称Face稳定叠加。500原图配对阶段耦合bootstrap：R2 Test最终CI[.0381,.2955]、平均[.0330,.3080]，R3平均[-.0339,.4165]，R1两区间均含0，未估seed方差/未校正Test筛选。

当前参考实现非视觉参数2,661,658→6,882,842（+4,221,184）、训练参数0，峰值allocated925.71→946.22MiB，B64/task7 CUDA耗时R0 241.91ms、其他489–508ms；共享冻结权重但两套来源各三视图前向，算力约2倍，尚未裁剪/缓存。保留R2和R3，先考虑预测保持的缓存/裁剪或更多seed，不自动新训练。完整报告docs/protected_view_forward_results_20261026.md；分析output/protected_view_forward/analysis_20261026_01；正确快照formal_02，formal_01原GPU/bench保留。以下启动/待完成均为历史记录。



前向完成与导出修复：780732e预检、5组隔离测速及Val/Test完整8任务均exit0，A/B每batch来源及R0/R1保护通过。后验发现将CPU logits拼接后整体sigmoid与历史逐batch sigmoid在Test R0最终仅3个概率位置相差ULP，logits全部相同、mAP约差1e-6；已确认按原batch_lengths分批sigmoid可精确重放。现修复仅CPU导出，新增第8项专项测试，不重训、不重复GPU前向，不改来源checkpoint。将保留formal_01原始产物，生成formal_seed1_20261026_02修正导出及概率逐值审计；bench资源记录仍来自原真实前向源码780732e，export Git单独记录。



执行更新：780732e已Git-only部署remote独立worktree multi-lane-main-protected-view-forward-20261026，branch/HEAD/upstream一致且clean；主树153行及test-only5行dirty状态SHA、HEAD均保持。完整502测试通过（50.496s）。formal_seed1_20261026_01已在tmux protected_view_forward_seed1_20261026运行，真实CLIP task0/2/7各Val首64人预检22.77s成功，A/B全部来源预测、所选分路、R0融合和R1无可靠Face融合均逐值一致，零训练更新。隔离R0–R4测速正在自动执行，随后Val/Test两进程并发完成8阶段。来源24checkpoint SHA预检保持；本地仅追加执行记录，不改变remote 780732e运行源码。首64人mAP不代表完整指标，不用于方案选择。



## 2026-10-26：用户授权完整模型视图来源组合验证

新实验分支codex/protected-view-forward-20261026，新增protected_view_forward/run_protected_view_forward及7项专项测试，保留此前未提交分析文档。R0 AAA、R1 AAB、R2 ABA、R3 ABB、R4 BAA，真实特征融合后接同阶段B0头，A/B各自Full摘要来源保留，冻结ViT权重/头共享但不缓存tokens。无训练/更新，seed1/GPU0/B64/workers2/72线程/AMP-TF32/旧seen-positive。先远程单测及task0/2/7 Val首64精确重放，再隔离5组资源测速，之后Val/Test并发每批复用来源计算并导出5组8任务全部预测。源24checkpoint SHA、参数/来源预测、R0及R1无Face保护均验收。配置和相对输入/输出/日志详见docs/protected_view_forward_execution_20261026.md；当前本地实现完成，待远程测试和执行，尚无正式模型结果。主树/test-only及旧实验不修改。


## 2026-10-26：下一研究候选——复用已经学好的整条辅助任务路径

- 只换Person的Test最终/平均配对CI[.0446,.2994]/[.0369,.3115]均正，优先真实前向R2并保留R1/R3控制；双换平均CI仍跨0，不能只追最终最高。来源组合正式成本含B0头+Person+Dual三次30epoch训练，不能声称与单次基线等预算。没有seed方差/选择校正，尚未执行完整模型。

- 本次只读组合完成，无新训练或模型前向。A强Person为32.8864/39.8687，A底换B双Attention Face的离线Val43.1784/50.4692、Test32.9419/39.9239；换Person为43.4224/50.5673、33.0549/40.0482；双换43.4681/50.7165、33.0608/40.0413。Face有正信号，Person来源替换更强，双换未同时胜过单Person。单独换Full也升，避免仅凭Full单路AP归因融合损失。
- 下一步如用户授权，优先完整模型无训练R0来源重放、R1 Face替换、R2 Person替换、R3双替换，保留R4 Full替换控制；同seed1/8任务/旧标准Val/Test，报告成本。先task0/2/7逐值来源功能核验；B辅助Attention必须读取B来源Full摘要。R3 Person/Face保留原B共享库，非从随机三独立S重新训练。
- 不把离线结果升级为真实模型提升：AMP融合/head舍入有.0036最终AP重构差，Face替换bootstrap最终/平均CI均跨0，未估seed方差。当前正式参照不变；后续须真实前向和更多seed证据。
- 完整路径复用成立后，再考虑单库的双教师Full/Person保护+Face能力保持，或受限融合门控；这些尚未改代码/训练，不复提旧弱辅助教师蒸馏Full。若拟用TRAIN holdout/OOF，教师也须匹配划分，不虚称事后holdout未见。
- 方案docs/face_protected_fusion_research_plan_20261026.md；CPU脚本scripts/emotic/analyze_protected_face_swap.py；产物output/face_protected_fusion/readonly_20261026_03，源48NPZ SHA不变。以下原失败两组的停止结论仍有效，本次整路复用是不同候选。

## 2026-10-26：当前状态——两组完整结束，均不作为新提升版本

- 已完成：独立三路Selectors与冻结强Person只训Face的seed1/GPU0/8任务Val/Test/OFF；12作业全部exit0，576产物回传校验，24源及16新checkpoint SHA一致，208融合/832视图CPU复算及逐值保护通过。
- 独立S+B0头Val38.6066/46.9717、Test29.2649/37.7092；原共训头40.9621/48.1452、30.7724/38.2750，按匹配原头对照仍输旧共享双Attention。不要把换头恢复误算模块收益；不要把此组视为完全视图隔离，Adapter/Prompts与Full K/V反传仍存在。暂不进一步拆共享参数或扩此版。
- 冻结强Person只训FaceVal43.0787/50.2760、Test32.8792/39.8493，对比强Person43.0741/50.2828、32.8864/39.8687无净增益；Full/Person和无可靠Face预测全阶段精确保持，Face OFF回到强候选。保护机制保留为工具，本版Full→Face模块不作为有效增强。
- 当前保留原Person-only共训ASP/Person Attention＋B0头，以及正式B0 32.6800/39.5292；均seed1历史seen-positive，不与其他seed/Full指标混合。现有Test已参与探索，未估计多seed稳定性。
- 若用户继续授权研究Face，先用现有预测检查Face是否存在Val/Test一致的错误互补以及目标人物对应；新结构必须解决此接口缺口，不直接延长5epoch、增scale、拆所有共享参数或重跑已失败后置残差。当前不启动新训练、不合并main。
- 详细结果docs/selector_face_controls_results_20261026.md；分析output/selector_face_controls/analysis_20261026_01，脚本scripts/emotic/analyze_selector_face_controls.py；本地分析分支codex/selector-face-analysis-20261026，远程执行f6ef130保持clean。下文所有“待完成/启动中”为历史记录，以本节为准。

## 2026-10-26：用户授权独立Selector与冻结强Person只训Face两组8任务

执行更新：源码f6ef130已Git-only部署独立remote worktree multi-lane-main-selector-face-controls-20261026，branch/HEAD/upstream一致，旧主树/test-only的HEAD/status SHA不变。7新增专项及495项完整回归通过（21.979秒）。GPU初始化审计确认三套Selector精确克隆、其余初值/CPU-CUDA RNG/初始预测等价，新增122880参数；smoke_seed1_20261026_01全部成功，冻结组更新后Full/Person/无可靠Face预测逐值保持，Face OFF回到强Person，原Person/双Attention参考逐值重放。

正式formal_seed1_20261026_01已自动启动于GPU0，tmux selector_face_controls_seed1_20261026。PRIVATE_SELECTORS训练PID1972432，FROZEN_PERSON_FACE训练PID1972995，参考Val/Test也排入同一4并行队列。两组task0前4epoch各84成功更新/0跳步；前者30epoch/任务，后者额外5epoch/任务。GPU约8537MiB/100%。正式8任务及所有Val/Test/OFF评估将自动接续，当前尚无最终成绩。源24份checkpoint SHA保持，旧实验/主树/test-only未修改。

106份smoke/单测/预测文件已回传本地output/selector_face_controls/smoke_snapshot_20261026_01并逐SHA核验，归档SHA5b3a5a3761a67eba3397c3c1f7b361f00eced32b92be3dcf9d0063569c0e3c65。运行源码保持remote f6ef130 clean；当前仅本地追加执行记录。完成后分析独立Selector的共享/跨K-V通路差别，以及冻结强Person后Face条件收益能否保留，不预先宣称提升。

新分支codex/private-selectors-frozen-face-20261026。PRIVATE_SELECTORS：基于双Attention共训+B0头方案，三视图每任务各10个独立Selector，共30个；从同一共享初值克隆，保持全局RNG与其余初始参数不变，训练30epoch/任务、主LR.0125、A/M LR.0004，仍共享Image-token Adapter与Prefix Prompts，保留Face/Person Attention读取Full K/V的反传，不额外detach。该组只检验Selector共享，不宣称完整视图梯度隔离。增加Selector122880参数。

FROZEN_PERSON_FACE：按任务装载原Person-only共训ASP/M及对应阶段B0 H（32.8864/39.8687强候选），全冻结，只训练当前新增可靠Face M，额外5epoch/LR.0004；Adapter保留ASL学得参数但无更新，M用融合BCE+原aux.1。旧Face模块保留冻结，下一任务不继承学生ASP。无可靠Face整批不进backward/Adam并单独计数，与AMP跳步区分。所有源checkpoint只读；评估逐值检查Full/Person和无可靠Face预测不变，Face OFF精确回到强候选。

全部seed1/GPU0/B64/workers2/OMP-MKL72/AMP-TF32/旧seen-positive，跑完整8阶段Val/Test。两新训练作业并行，最多4作业含原Person/双Attention参考重放；不足显存6000MiB/shm1024MiB自动排队，间隔12秒。同源码smoke通过后自动正式启动。新增selector_face_controls和run_selector_face_controls、独立Selector profile限定、不同Selector形状的纯head替换与冻结预测验收；新增真实梯度/冻结/初始等价/Face无信号测试。当前实现完成待远程测试，计划docs/selector_face_controls_plan_20261026.md。原d1322ab/ab15571及主树/test-only保持不改，前轮结果分析文件一并保留在新实验分支。

## 2026-10-26：八任务续跑与Face Attention完成，CPU复核和分析完成

远程执行源码d1322ab保持clean。正式19作业全部exit0，四适配组各40额外epoch/2325更新（前3任务保留，新增1490），Face共训240epoch/13950更新，全0跳步。55源checkpoint SHA保持；694份完整产物回传并逐SHA核验，归档SHA7c9fd4a795f5e224333ca78874b6887bcc5d777552934be13821ab822144a0dc。CPU重算240融合/960视图指标，最大误差7.82e-14；12 prefix、16 MOnly OFF=B0、8 Face原头Test、16 Face mask逐值检查通过。新增结果分支codex/eight-task-face-results-20261025，仅本地CPU脚本/文档，不改远程执行树。

seed1 Val最终/平均、Test最终/平均：B0 42.4628/49.4763、32.6800/39.5292；固定头无M41.8698/48.4461、32.3631/39.1494；固定头有M41.8739/48.4413、32.3625/39.1555；全B0固定只训Person M42.4763/49.4915、32.6829/39.5283；头可训42.0249/48.6800、32.4597/39.2871。额外5epoch固定头路线未达标，mAP下降并非Attention独有，task0已发生，不能归为旧任务遗忘。Test oF1改善来自召回增加，不能代替mAP收益。

原Person-only共训表征+B0 H：Val43.0741/50.2828，Test32.8864/39.8687，为当前seed1最好探索候选。新Person+可靠Face+B0 H：Val43.3394/50.5522，Test32.7573/39.7004；Val8/8更高、Test8/8更低于旧候选。仅Face OFF为Val43.2889/50.4403、Test32.7194/39.6213；全部OFF为43.2304/50.3430、32.6920/39.5698。新Face原共训头为Val43.1264/50.3256、Test31.9459/39.1951，换头收益不可算Face模块收益。

Test新Face相比旧Person：两模型全OFF表征/固定H组合−0.1730最终/−0.2615平均，Person条件作用变化+0.0061/+0.0141，Face条件作用+0.0378/+0.0790，合计−0.1292/−0.1683。新模型Person和可靠Face AP升，Full31.4947→31.3667，融合32.8864→32.7573；融合减自身Full几乎不变1.3917→1.3906。Face写入只在Face，但共同训练改变共享A/S/P，Frozen Forward不动不等于Full任务表征不动。Face分支非死分支，主要损失表现为共同训练后的表示/读出泛化，不将梯度冲突当已证实。

Test可靠Face子集新旧−0.1963、同图多人−0.4294，单人+0.0732、不可靠Face+0.2785；Val可靠Face/多人反而改善。可靠Face比例Val60.91%/Test60.75%近似，平均正标签6.266/4.445，标准一致而分布不同。任务Selector Query不显式读取Face内容、同Full/同任务原始修正缺少人物区分；Full→Face可能强化共有上下文，不能把单路提升等同互补改善。这是结构风险与结果相符，不声称已完成目标错配归因。

500次成对原图bootstrap、8阶段耦合：Test Face ON−Face OFF最终CI[0.0108,0.0630]、平均[0.0297,0.1265]，有条件贡献；旧Person−B0、新Face−旧Person总差CI均含0，未估计seed方差/未修正Test筛选，不称稳定多seed收益。报告docs/head_anchored_eight_face_results_20261025.md含8阶段、分组、机制分解、统计及后续建议；CPU脚本scripts/emotic/analyze_head_anchored_eight_face.py，产物output/head_anchored_eight_face/analysis_20261025_01。

下一步仅建议：保留旧Person+B0头强候选与正式B0；不继续额外ASP固定头适配。若用户授权，可冻结旧Person强候选全部ASP/原Person M/B0 H，只训练新增可靠Face M，强制Full/Person及无可靠Face预测不变；区别于已跑的B0固定只训Person M。此控制未启动；若无收益则停止本版Full→Face路线，不重启已有负结果的后置残差等。不自动扩seed/新训练/合并main。

## 2026-10-25：用户授权续完8任务Val/Test，并新增可靠Face摘要Attention

执行更新：运行源码d1322ab91e3ebbcf7c4c6c03d856a8bfbb065b16，本地/远程独立树branch、HEAD、upstream一致。488项回归通过（19.002秒），新增6项专项通过。smoke_seed1_20261025_01完成13作业（5训练＋8评估），验证四组采样一致/原prefix保留、MOnly OFF=B0、Face OFF不改Full/Person/无可靠Face logits，原Person-only换头task7精确重放Val43.07406218346569/Test32.886421356337166。smoke完成门槛后正式队列已自动启动。

正式output/head_anchored_eight_face/formal_seed1_20261025_01，tmux head_anchored_eight_face_seed1_20261025。GPU0当前Face共训PID1873009、固定头无M1873585、固定头有M1874348、B0全固定只训M1875111四并行；允许头更新的第四学生组排队。三学生task3首2epoch各156更新/0跳步，Face task0前4epoch各84更新/0跳步，GPU约12413MiB/100%，尚无正式最终指标。队列自动排完整Val/Test及OFF/换头对照，最多4并行，不需再次启动。不更新运行中源码，不修改原pilot/worktree。

原40历史＋15 pilot checkpoint共55份SHA已登记并核验；主树/test-only HEAD与状态SHA未变。256份完成的pilot/smoke/单测产物已同步本地output/head_anchored_eight_face/startup_snapshot_20261025_01并逐SHA核验，归档SHA100949809c3a8def6cee7fe065e69dc0173379ac059f47681bdd6b5827d57a08。当前追加执行记录仅本地未提交，运行源码保持d1322ab clean。完成后核验8阶段最终/阶段平均、逐视图/Face可靠性与关闭模块贡献，明确30epoch共训与额外5epoch适配属于两条不同训练路线。

新分支codex/head-anchored-eight-task-face-attention-20261025。原466ce26的seed1三任务pilot已全部完成：task2 Val/三阶段均值，B0 44.7316/54.0934，固定头无M43.9498/52.7115，固定头有M43.9420/52.6928，完整B0固定只训M44.7462/54.1164，允许头更新44.0892/52.9801。用户要求不据此停止，保留task0–2原checkpoint逐值不变，四组只新增task3–7各5epoch/LR.0004，并评估8阶段Val/Test（ON与适用的OFF）。不重复前3任务，不以3阶段均值代替8阶段。

新增Face版本沿用原ATTENTION_BOUND的30epoch共训配方，并非5epoch学生方案：Person与可靠Face各有任务专属、独立同规格Full摘要Attention；Task Selector Query、layer5/inner64/heads4/scale.1/cap.1，Face不可靠时残差严格为0、该batch无Face样本时Face参数无梯度并跳过Adam动量更新。旧Person参数及初始化随机流不变，Face用独立CPU fork RNG。原任务冻结机制覆盖Person/Face两套模块。训练保留共享头共训，推理内存换回对应阶段B0 H，ASP/两Attention均保留；另补Face单独OFF/全部OFF及原共训头ON。43.0741/32.8864是原Person-only换头的task7结果，追加完整8阶段旧Person-only换头Val/Test对照。

GPU0统一最多4并行，显存≥6000MiB/shm≥1024MiB及12秒放行；B64/workers2/OMP-MKL72/AMP-TF32/旧seen-positive，Test参与探索如实记录。新runner入口run_head_anchored_eight_face，输出output/head_anchored_eight_face，日志logs/head_anchored_eight_face。代码已实现待远程单测/smoke及正式启动；保持原pilot及主树/test-only不修改。详细计划docs/head_anchored_eight_face_plan_20261025.md。

## 2026-10-25：分类头锚定的额外适配pilot实现中

执行状态更新：最终运行源码466ce2666479f941a5ad6473cafb761b761516cb（本地/远程branch、HEAD、upstream一致）。最终4项专项通过，smoke_seed1_20261025_02五作业/24完整Val ON-OFF结果全部完成，四组各task0–2两更新均0跳步，batch完全一致，B0及MOnly OFF三个阶段逐值重放；此前482项完整回归通过。最终104份smoke产物已同步本地smoke_snapshot_20261025_02并逐SHA核验，归档SHA b81e5cc2d7024c72df35fc53a904f4900c89487265d6752f80511da8efb00bb8。

正式formal_seed1_tasks012_20261025_01已在GPU0四并行启动（tmux head_anchored_pilot_seed1_20261025，worker PID1841374/1841954/1842714/1843477）。四组task0首epoch各84成功更新/0跳步，无M组已完成2epoch，GPU约12475MiB/100%。原B0三个阶段重放为第5作业自动排队。正式当前运行中，尚无最终结果，不新跑Test。源码运行中保持远程树clean/466ce26不更新；当前本地仅追加执行记录。原40份checkpoint SHA全部保持，主树/test-only的HEAD/status SHA与既有记录一致。

482项远程回归、smoke_seed1_20261025_01五作业全部完成：四学生各task0–2两更新0跳步，实际batch跨组逐值一致，B0及MOnly OFF的三个阶段融合预测精确重放。正式前完善学生实际架构/参数量、AMP及适配RNG元数据，需在最终源码重新通过smoke后才启动5epoch；不改变损失或训练数据流。

远程Git-only独立树已建立，源码34a13e37174b33cc919aac391f498f9fb4785c27与本地分支一致；482项unittest通过（20.52秒）。GPU0的smoke_seed1_20261025_01已通过tmux启动：四组各task0–2两更新和完整Val、Attention OFF及B0重放，当前运行中，正式5epoch尚未启动。主树/test-only HEAD及status SHA与既有记录一致，未修改。

用户授权执行，分支codex/historical-b32-head-anchored-pilot-20261025。四学生组：固定头无Attention、固定头加限幅Attention、完整B0冻结只训Attention、允许头一起更新。seed1 task0–2，仅Val，额外5epoch、全部可训练参数LR.0004、Adam/reset/cosine/WD0、B64/workers2/72线程/AMP/TF32。GPU0最多4并行，空闲显存≥6000MiB、shm≥1024MiB放行，间隔12秒。先同源码smoke gate再正式。

复用ab15571的BASE各当时阶段checkpoint，禁止未来阶段权重；只覆盖学生当前ASP及H当前类别行，保留旧学生任务冻结，不调用activate_task继承学生S/P。额外随机流隔离并恢复，实际batch跨四组核对。Adapter ASL，其余可训练参数融合BCE+aux.1；MOnly只优化M的BCE。每阶段评估前先存compact及optimizer/scheduler/scaler/RNG恢复状态。原B0三阶段重放、MOnly OFF必须精确匹配既有预测。原Attention OFF仍保留学生ASP/H，不是B0。仅3阶段均值，不新跑Test/扩8任务/扩seed。

新增head_anchored_pilot、run_head_anchored_pilot及真实梯度/冻结/Teacher切片/RNG测试；runner仅加可选optimizer分组注入，旧默认不变。当前本地语法/diff检查通过，待远程单测、GPU smoke和正式启动。待完成后同步结果核验。此前归因报告/分析脚本和上下文改动一起保留在新分支，以下为历史时点记录。

## 2026-10-25：共同训练归因及兼容性检查全部完成，下一方向为B0分类头约束的表示适配

主诊断4289fc3（独立backbone-attribution树）20 GPU作业/56份task7完整Val/Test成功；补充131cca7（独立head-transfer树）4作业/12份task7结果成功。无failure/无训练/无新checkpoint/main合并。原40份compact SHA全程不变；主树153行/test-only5行及状态SHA保持；原ab15571模型树、54af61b评估树、4289fc3诊断树、131cca7补充树均clean。主5单测+478完整回归、补充10针对性回归/导入通过。294份产物已同步本地逐SHA核验，归档SHA176ca05dd3f300e3b4e3cf77b8fb1f809d1cbde2f96645d08a2e0fb49c0bef95。

CPU参数审计256项、初始化继承112项：后续A/H起点28/28相同，S/P复制来源各28/28不同；差异通过新任务初始化传递，非旧任务解冻。固定TRAIN16组合×task0/2/7×2批×4模型384记录，像素/IDs跨组相同，Head不改特征768项检查通过。raw参数距离及feature cosine不等于损害。独立复算68融合/68视图结果、20端点逐值重放，最大误差4.27e-14。

关键受限Attention结果（Val/Test）：原版42.5177/32.2423；关闭模块共训ASP/H42.4725/32.2152；保留共训ASP换B0 H、关模块43.0278/32.8651；同样换头并开模块43.0741/32.8864；完整B0+共训模块42.4881/32.6870；B0本身42.4628/32.6800。主要恢复信号在表示/读出配对，新模块直接只+0.0462/+0.0213。原Attention换头同样改善但Test32.6167仍未过B0；位置两组换头反而变差，不泛化为所有方法head坏。候选Test三路Full31.4947/Person30.9738/Face25.7673均高B0，但oF1降至48.8001，cF1升至33.3119。

混合候选相对B0+0.2064 Test的500原图bootstrap CI[−0.2711,0.6437]，Val CI[−0.4011,1.6101]；非稳定跨seed证明。只task7，无混合模型八阶段平均，既有Test已参与探索。候选Test可靠Face+0.3126、不可靠Face−0.4911；用保存预测做两种固定可靠性切换：仅可靠用候选Val42.0142/Test32.2616，反向42.5459/32.2282，均未形成可靠整体收益；分组AP不保证全体排序、不可据此直接做Face开关。

报告docs/historical_b32_backbone_attribution_results_20261025.md；下一阶段建议docs/historical_b32_head_anchored_adaptation_plan_20261025.md。建议每任务先按B0学习当前类别H，再在额外学生阶段冻结H；当前ASP从B0复制，比较无M/有限幅M，配全Base固定只训M及同预算头可训练控制。建议先seed1 task0–2 Val、额外5epoch/LR.0004，保持b32[1]/scale.03/aux.1/ASL及原数据/人群。B0 S/P继承和RNG不受学生阶段影响，旧学生任务保留冻结。关闭M不等于B0，完整回退Teacher才精确B0。四学生组匹配额外预算，未来通过后再完整8任务/seed0、3。本轮仅制定，未修改或执行新训练。

CPU脚本scripts/emotic/analyze_historical_b32_backbone_attribution.py，最终分析output/historical_b32_backbone_attribution/analysis_20261025_02（01为加可靠性回退前版本，公共数值相同），合并snapshot在complete_snapshot_20261025_01/snapshot。最新分析脚本、报告、执行上下文仅本地未提交，远程实际执行源码保持4289fc3/131cca7不变。


## 2026-10-25：共同训练诊断已有中间证据，追加Attention/分类头兼容性只读检查

主诊断继续运行于独立树源码4289fc3，原始20作业/56结果配置不改变。CPU审计完成256参数比较、112初始化继承比较：未来Adapter/head28/28起点逐值相同，S/P复制来源28/28不同；固定TRAIN16组合×3边界×2批×4模型共384记录完成。task7未恢复的融合特征与B0平均cos约0.57–0.62，但原始参数/特征距离不等同于性能损害。所有已完成恢复端点均匹配既有OFF/B0。

中间完整Val结果：只恢复H时原Attention42.8981（OFF原42.6576）、受限43.0278（OFF原42.4725），位置两组反降；原Attention H恢复Test32.5703比OFF32.3058回升但低B0。依据两组Attention的Val H恢复改善，新增独立分支codex/historical-b32-head-transfer-20261025：继续只读task7，保留新Attention开启，比较none/H/ASPH三状态，两组×Val/Test共12结果；none必须逐值匹配原ON。该筛选规则使用Val，但Test此前已被查看，保持探索性标记。不训练，不保存混合权重。

补充12结果放到独立树、等待原主诊断completion通过后才启动，避免超过GPU0四并行总资源。补充源码131cca7c2ef47a927cc45a71950ddbd4b635b19d已Git-only部署独立树multi-lane-main-historical-b32-head-transfer-20261025，10项针对性回归及导入检查通过。tmux historical_b32_head_transfer_seed1_20261025已设置完成门槛，等待主诊断成功后自动执行；当前只等待、不与主诊断抢占额外GPU作业。原ON逐值数值门槛仍保留。不得更新或干扰正在运行的4289fc3源码树。最终分析区分表征/分类头配对、初始化传递和新分支直接作用；后续训练方案尚未启动。


## 2026-10-25：用户授权继续追查共同训练，参数/特征/受控恢复诊断实现中

新分支codex/historical-b32-backbone-attribution-20261025，仅只读checkpoint与内存恢复，不训练。初步代码核对：activate_task(t)会把上个任务的Selectors及Prompts复制到新任务，Adapter为独立初始化；因此全局初始SHA/随机流相同不意味着后续任务的S/P起点相同，需量化继承差异。单部件换回B0可能造成配对失配，不能据此单独判定坏部件。

预声明：CPU比较4模型×8阶段各A(Adapter)/S(Selector)/P(Prompt)/H(head)参数，按当前/已见分片测量norm/方向；重建下一任务S/P复制初始化差异，检查未来A/H起点。GPU固定TRAIN task0/2/7各两批、cap64、原图分组固定种子，16组参数组合，确定性FP32观察三视图/融合特征和logits，不作为正式泛化成绩。最终task7完整旧人群Val/Test做none/A/S/P/H/ASP/ASPH七状态×4模型×2split，共56结果；none匹配既有OFF、ASPH匹配B0全预测为端点门槛，通过后才派发5种中间恢复。

新增模块B/C全关，Image-token Adapter开启，恢复至对应阶段B0，仅内存替换全套相应部件（包括bias），后自动复原。仅GPU0最多4作业，B64/workers2/OMP-MKL72/AMP-TF32，显存6000MiB/SHM1024MiB余量、12秒启动间隔；TRAIN探针workers0和单独确定性FP32。输入仍原formal_seed1_20261025_01与已完成seed1_tasks027_20261025_01，CLIP/Data/Face manifest不变。输出output/historical_b32_backbone_attribution/seed1_20261025_01，日志对应logs。没有新增8任务训练或新checkpoint。下一步适配仅在结果明确后提出，不在此时假设某模块必然有害。

本地语法检查通过，新增5项恢复正确性/异常复原/状态覆盖/Head不改表征检查及远程478项完整回归全部通过。诊断源码4289fc3f51186da5e5886dc300970bf8bcd2b09c已Git-only部署到独立树multi-lane-main-historical-b32-backbone-attribution-20261025；两端branch/HEAD/upstream启动时一致且clean，tmux historical_b32_backbone_attribution_seed1_20261025已启动，先CPU审计和固定TRAIN探针、再端点与混合恢复。尚无完整归因结论。原主树153行/test-only5行及SHA保持，旧模型ab15571与旧评估54af61b工作树clean，执行记录仅本地更新。旧模型ab15571、旧评估54af61b都不更新；主树/test-only不修改。


## 2026-10-25：Test关闭模块归因完成，关闭不能恢复32.6800

独立评估树multi-lane-main-historical-b32-test-attribution-20261025、源码54af61bf5d1df6993cfd0f0ce9425501aac5888a，seed1_tasks027_20261025_01 completion=complete、8作业exit0/24阶段成功、无failure。四组ON task0/2/7的融合及分视图预测与原Test逐值一致；OFF后Attention的Full/Face、位置的Person/Face逐值不变；40份源compact SHA全程相同。原模型树仍clean于ab15571，原主树153行/test-only5行及状态SHA保持。新增5项单测、473项完整回归通过，无训练/optimizer/权重改写/main合并，GPU0评估已结束。

Test task0/2/7开−关：原Attention+0.2882/+0.1380/+0.1033；受限+0.0946/+0.0394/+0.0271；原位置+0.0018/+0.0142/+0.0032；校准位置+0.0193/+0.0183/+0.0022。task7开启/关闭依次32.4091/32.3058、32.2423/32.2152、32.4352/32.4321、32.2928/32.2906；关闭后相对B0仍−0.3743/−0.4648/−0.2480/−0.3894。两个Attention当前checkpoint有小正向补偿；当前Test劣势并非仅由推理新增残差直接造成，更应检查共同训练后的底座。位置即时贡献很小。原Attention开使Person单路Test下降0.5092而融合增加0.1033，不能将融合改善叫作Person增强。

83份产物已同步本地并逐SHA核验，归档SHA2b3ab30e01cbea91be2eef5ba65c1fb3dacc2a56a13daab28bb7fa29697918d3。独立CPU复算24融合/24分视图文件，最大误差6.40e-14。500次原图bootstrap最终Test开−关CI：原Attention[0.0075,0.2014]、受限[0.0018,0.0533]、原位置[−0.0075,0.0140]、校准位置[−0.0106,0.0191]；不含seed方差或选择修正，非整方案优于B0的证明。只三个边界，未计算或冒充完整八阶段平均。

报告docs/historical_b32_test_attribution_results_20261025.md；原结果报告已补充完成状态，产物output/historical_b32_test_attribution/complete_snapshot_20261025_01/snapshot，CPU复算output/historical_b32_test_attribution/analysis_20261025_01。后续仅建议冻结对应阶段B0原参数、只训新模块做独立贡献控制，先TRAIN原图holdout和少量任务；此项未启动。最新执行上下文/补充报告仅本地未提交，实际评估源码保持54af61b。


## 2026-10-25：用户授权补Test关闭模块，独立评估分支实现中

新建codex/historical-b32-test-bypass-20261025，保留前轮分析报告/脚本及上下文未提交记录。仅增加historical_b32_test_attribution限定开关，允许四个完整seed1历史b32模型的Test task0/2/7开启重放及只关闭各自B或C，拒绝关闭Adapter、改变人群、smoke或其他模型。原默认validation限制不变，不修改训练/model代码。新增GPU0四并行只读调度与ON逐值重放、未受影响视图、40份checkpoint SHA校验；8评估作业、24阶段，三边界均值不作为8阶段平均。

输入为旧独立模型工作树formal_seed1_20261025_01，模型源码ab15571；EMOTIC Test/CLIP ViT-B16/原Face manifest，seed1/B64/workers2/OMP-MKL72/AMP-TF32，Adapter b32[1]/scale.03保持启用；不设优化器不训练。结果output/historical_b32_test_attribution/seed1_tasks027_20261025_01，日志对应logs。GPU0空闲约24GB，GPU1-7忙不使用，/dev/shm4GB空闲；门槛GPU6000MiB/SHM1024MiB，启动间隔12秒。

本地python无Torch，测试移到独立远程ddp环境执行；新增5项检查及473项完整回归全部通过。评估源码54af61bf5d1df6993cfd0f0ce9425501aac5888a已Git-only部署独立树multi-lane-main-historical-b32-test-attribution-20261025，两端branch/HEAD/upstream启动时一致且clean。tmux historical_b32_test_attribution_seed1_20261025已启动；先ON逐值重放，匹配后才自动派发该组OFF。模型源码仍ab15571、旧模型树clean，原主树153行/test-only5行及SHA保持。最新执行记录仅本地更新，尚无关闭Test结果。持续授权已覆盖Git-only提交推送/独立worktree部署及此次非破坏性评估，无需重复确认。不修改旧源码树、主树或test-only，不合并main。


## 2026-10-25：历史b32底座叠加摘要模块五组完整结束，本轮未超过32.6800

formal_seed1_20261025_01 completion=complete、failure不存在，19作业exit0/无待执行/无错误；5组240epoch/13950更新/0skip。基线历史四指标与8阶段完整预测逐值复现；40compact重算SHA均与训练/评估一致，旧任务参数、新模块、Frozen Forward冻结及恢复一致通过。源码ab1557142c95d0d8574ea172a968c55d78466a25仍clean；原主树153行/test-only5行及SHA保持。本次只读CPU分析，无新GPU前向/训练/权重改写/提交推送合并。

Val最终/Val平均/Test最终/Test平均：BASE42.4628/49.4763/32.6800/39.5292；ATTENTION42.7411/49.9141/32.4091/39.1801；受限ATTENTION42.5177/50.2792/32.2423/39.3917；POSITION42.5951/49.6903/32.4352/39.5144；尺度校准POSITION42.8344/50.0303/32.2928/39.5217。四组val mAP均提高，Test最终/平均均未超过基线。位置校准Test cF1+0.3672，但mAP−0.3872，不混淆指标。原Attention Test八阶段全负；原位置前5阶段正、校准位置前4正，后期转负。旧人群旧类重放mAP变动≤0.0042，不将后期下降称灾难性遗忘；AMP logits并非逐值恒定（最大差0.0831以内，最大平均差0.00094）。

结构证据：原Attention写回范数比0.180–2.161，受限版实际0.066–0.089，幅度机制生效仍未提升。Query来自任务Selector，跨图std约1e-7，权重仍随Full图片变化，但同图不同人物没有额外Query身份信息。Full/Person Test秩相关0.6883降到0.6387却精度下降，多样性不等于互补。原Attention Test Full31.2373→30.6947、Person30.4315→29.8938，融合相对自身Full补偿虽更大，净效果仍下降；受限版Adapter能量比更大，限新分支不等于保护整个底座。未测梯度方向，不宣称冲突已证实。

Val关模块task0/2/7：原Attention开−关+0.4252/+0.2297/+0.0835；受限+0.1416/+0.0427/+0.0452；原位置+0.0041/+0.0095/+0.0065；校准位置−0.0127/+0.0009/+0.0016。校准位置最终关后42.8327，几乎保留其相对B0的val增益；不能将表面提升直接归为推理位置作用。位置确让同图Full端点产生差异，Test跨度中位数0→0.0115/0.0303，非坐标未接入。位置KL小且无饱和记录。关模块保留共训参数，不是重新训练基线。Test关模块尚未执行，不外推Val开关结论。

304份正式非checkpoint产物已同步本地并逐SHA核验，归档SHAed1690345cc0b62e6e037e8a7e09c0f71ea7ad1f6402896de8bf958c8268a5fd。独立复算80常规/80视图/12关闭预测，最大误差5.69e-14，40原生Test重放逐值一致。500次原图bootstrap全部最终相对B0区间跨0；受限Attention最终Val开−关区间[0.0076,0.0890]为小正信号，非整方案优于B0。Test曾获授权参与搜索，不当作独立未选择验证。

报告docs/historical_b32_modules_seed1_results_20261025.md，CPU脚本scripts/emotic/analyze_historical_b32_modules.py，分析CSV/审计/bootstrap位于output/historical_b32_modules/analysis_20261025_01。下一步仅建议：保留强B0，不立即叠加/扩网格；先现有checkpoint补Test task0/2/7关闭模块，若继续设计则先冻结对应阶段B0的Adapter/Selectors/Prompt/head，只训新增模块，在TRAIN按原图分组holdout和少量任务检验独立贡献。未启动额外实验。新增报告/脚本/执行上下文仅本地未提交，结束实验源码保持ab15571。


## 2026-10-25约13:20：五组训练更新结束，正在收尾报告与完整评估，预计再20–30分钟

只读核查formal_seed1_20261025_01：五组均240轮训练日志已齐；BASE/ATTENTION/ATTENTION_BOUND训练与验收完成，POSITION两组已完成task7第30轮，尚在原生task7 Test报告。BASE历史八阶段预测及四项指标已精确匹配（baseline_reproduction.status=exact）；BASE和ATTENTION完整val已完成，test及其余评估/四组关闭模块val继续自动排队，queue.failed=false，仍仅GPU0。实测完整8阶段val约4.2–4.4分钟，test每阶段约45–80秒，最多4评估并行。据此估计从13:20再20–30分钟，保守预留40分钟；这只是进度估计，未开展本轮精度/结构归因分析，无代码或调度变更。执行记录仅本地更新。


## 2026-10-25：五组摘要模块smoke全部通过，正式seed1八任务已在GPU0启动

模型源码ab1557142c95d0d8574ea172a968c55d78466a25，两端branch/HEAD/upstream在启动时一致且clean。独立远程工作树multi-lane-main-historical-b32-modules-seed1-20261025。smoke_seed1_20261025_04 completion=complete：五组真实CLIP FP32/AMP确定性初始化与LEGACY逐值相同，五组初始主参数SHA及CPU采样RNG一致；2task×2update均0skip，旧参数和新增模块冻结/Frozen Forward/恢复与compact一致；BASE与原36b7753的全部方法参数、loss、原生val预测逐值相同；10份常规val重放逐值匹配，4份关闭新增模块val成功。最新完整468项回归全部通过。

正式formal_seed1_20261025_01已自动启动，tmux historical_b32_modules_formal_seed1_20261025；五组已全部在GPU0并行，PID BASE1646912/ATTENTION1647479/ATTENTION_BOUND1648259/POSITION1649021/POSITION_CALIBRATED1649784，首轮均84更新/0skip；BASE主loss.67947638及Adapter loss.03236856与历史首轮精确匹配。训练进程均未设置额外CUBLAS_WORKSPACE_CONFIG，保持原历史CUDA设置。五组为BASE、ATTENTION、ATTENTION_BOUND、POSITION、POSITION_CALIBRATED，seed1/8task/30epoch/B64/旧标准完整val+test；Adapter b32[1]/scale.03/LR.0004/ASL9.8/0/.05/aux.1，主LR.0125，新增模块BCE、LR.0004。前轮两路线不相互叠加。正式BASE最终仍必须精确匹配历史32.6800/39.5292和八阶段score；当前不预称已完整复现或精度提升。新增组另外自动val task0/2/7关闭模块归因，保留共同训练的其余参数，不视为独立基线。

输出output/historical_b32_modules/formal_seed1_20261025_01，日志同名logs目录，driver位于logs/historical_b32_modules/implementation_20261025_01/formal_driver.log。startup_audit.json已核验五组config/source/进程/GPU及受保护工作树；238份非checkpoint启动/短程记录已同步本地output/historical_b32_modules/startup_snapshot_20261025_01并逐SHA验证，归档SHAb39591161f2c96f2d96b9aaaed7e2a5fd71ccc1f82440ef055807d91a9fd0ec1。运行源码保持ab15571不再更新，最新执行记录仅本地上下文更新。smoke01/02为仅初始化审计拦停，03的五训练成功但验收profile误用被拦停；均保留而非冒充成功。04为最终通过门槛。原主树153行/test-only5行及SHA保持，旧00fbf90/a30120c工作树clean；仅新增独立工作树，无main合并。


## 2026-10-25：五组摘要模块远程检查，GPU初始化审计定位为旧小批量GEMM非确定性

新分支已Git-only部署独立工作树multi-lane-main-historical-b32-modules-seed1-20261025。新增5项真实梯度/配置/初始化/适配检查通过；完整468项回归两次通过，1项跳过。smoke01/02在初始化检查阶段拦停，没有发生正式训练。保留原BASE执行路径后仍出现约1e-5 FP32/约6e-5 AMP差异；单独原始LEGACY重复前向也出现同量级，关闭AMP cache不能消除。设置仅审计进程的CUBLAS_WORKSPACE_CONFIG=:4096:8与torch.use_deterministic_algorithms(True)后，原模型FP32/AMP三次重复逐值一致，故恢复全部零输出跨组逐值检查，不以容差替代。此设置不传给历史训练子进程。

smoke03的五组真实CLIP初始化FP32/AMP输出与原始模型逐值相同，现已开始GPU0短程训练，尚无正式实验。实际五组config均通过严格新profile检查。代码复核发现复用的训练验收函数默认只接受旧profile，改为显式传入新profile验证器，旧调用默认行为不变；待本轮子进程退出后再Git-only更新工作树并重跑最终smoke。全部训练loss/方法张量/原生预测及评估重放通过后才正式5组seed1/8task/完整val+test。


## 2026-10-25：用户授权在32.6800底座分别加Attention/目标人物位置，五组实现中

新建codex/historical-b32-summary-modules-seed1-20261025，保留前轮分析未提交改动。计划seed1 BASE、原Attention、RMS受限Attention、原位置偏置、内容尺度校准位置偏置，共5组完整8task val/test，另自动task0/2/7 val关闭新增模块。保留历史b32[1]/scale.03/Adapter-ASL/aux.1/Person jitter0、全TRAIN/B64/30epoch/主LR.0125/模块与Adapter LR.0004/旧AMP65536-growth2000/全局采样/CPU72；仅GPU0最多5作业/4评估，显存不足自动排队。

新模块仍BCE更新，仅Adapter走ASL，原版和适配版独立比较不同时叠加。受限Attention平滑RMS界.1且零点梯度不截断；位置偏置乘detach内容std截断[.25,4]并FP32softmax。独立historical_b32_modules profile避免启用旧strongFull标志改变AMP/随机流；模型内部兼容原三路执行路径。新增真实CLIP/AMP初始化审计、native36b7753短跑、实际辅助梯度测试、冻结/恢复/旧预测锚点守卫。尚仅本地实现和语法检查，无新正式成绩，待远程回归/smoke后运行。

方案docs/historical_b32_modules_seed1_plan_20261025.md，入口multi_lane.track_a.run_historical_b32_modules，输出output/historical_b32_modules，日志对应logs。原主树153行/test-only5行及SHA保持，旧00fbf90/a30120c工作树clean。用户已授权执行，提交推送与Git-only部署不再重复确认。

## 2026-10-25：18组历史b32实验完整结束，精确复现但最终收益未跨seed稳定

formal_seed013_20261024_01 completion=complete、failure不存在，18训练和36份8阶段val/test均成功，队列54作业无待执行/活动/错误。18组均240epoch/13950更新/0skip；144份compact SHA重算与训练及评估记录一致，冻结旧参数/Frozen Forward/恢复model等于compact审计通过。模型00fbf90、调度器a30120c工作树clean，原主树153行/test-only5行及状态SHA保持。原4接管作业依成功日志和完整产物核验，其他新子进程保留实际退出码。此次仅CPU分析，无GPU前向/训练/权重更改/提交推送合并。

历史seed1精确复现32.6800218743/39.5291640411/cF132.8226918781/oF149.5458931279及8阶段全部预测。三视图基准/同seed Full四项val最终/val平均/test最终/test平均：seed0三视图42.0575/48.8372/32.2177/39.2517，Full42.4351/48.8234/32.3836/38.8570；seed1三视图42.4628/49.4763/32.6800/39.5292，Full42.1535/49.0777/32.1389/38.6729；seed3三视图42.4591/49.4636/31.8730/38.7950，Full41.7897/48.0306/31.8920/38.4180。最终Test差−.1659/+.5411/−.0190，阶段平均三个seed均正+.3948/+.8562/+.3770；平均最终+.1187但1/3为正，非稳定最终提升。

参数结果：seed0最终Test最好aux.2=32.2935，仍低匹配Full32.3836与强Full32.5365；seed3最终最好LR.0002=32.2026，但缺同LR Full，不能把全部差值归给融合。scale.01的seed0 val43.1869/50.6019大增，Test32.1519/39.1674却低基准。scale.05两seed四项均低。aux0两seed四项均低；aux.2两seed最终val/test均提高，但seed0阶段均下降，Test仅末阶段高基准。过去scale.1/带颜色扰动时辅助监督不优，不能推广为本配方aux无益。Test按授权参与选择，完整报告全部结果。

结构归因：共享模型Full Test30.5997/31.2373/30.6061，融合补回1.6180/1.4427/1.2669；seed1成功并非补回最多，而是Full损失较少。同图多人Test1237图中1185图标签不同，Full几乎给所有人物相同logits，局部视图提供主体区分；Test多人/单人子集差：seed0−.0189/−.7763，seed1+.7091/+.3993，seed3+.6800/−.8931。可靠Face子集三个seed val/test均正，不可靠回退更不稳；Face已屏蔽不能归为坏Face直接混入。

必须准确解释Adapter日志：ratio为E[||delta||²/||x||²]的能量比，不是L2范数比或百分比。LR.0008的task7 Full能量比9.2928/7.4832，scale.05为3.4795/3.6748；小名义系数不能保证弱适配，但能量不直接证明失败。固定旧人群旧类36项重放中，三视图mAP差在−.0030到+.0028，不能把阶段累计下降称灾难性遗忘。500次原图bootstrap全部关键最终Test差值区间跨0，不含训练seed方差或搜索校正。

906份正式非checkpoint结果和日志已同步到output/historical_b32_search/complete_snapshot_20261025_01并逐SHA核验，归档SHA775f5b54b3d7893212bd4c42cde3ed8246021067dc4311a16dadbe04829b4995。独立复算288常规/240视图预测，最大误差1.21e-13，144原生Test重放全等。报告docs/historical_b32_seed03_results_20261025.md，脚本scripts/emotic/analyze_historical_b32_search.py，CSV/审计/bootstrap位于output/historical_b32_search/analysis_20261025_01。

下一步仅建议未执行：本轮checkpoint在val task0/2/7关闭Adapter归因，不能以上轮scale.1关闭结果替代；若保留LR.0002或scale.01候选，补对应参数Full后再称融合收益。暂不扩大网格、叠加Attention/位置或把旧失败Full锚定方案换名。保留强Full32.5365/39.1445；未经新授权不启动训练/额外GPU推理。

## 2026-10-24：GPU0已成功6组并行，原4进程和训练进度完整保留

新控制器a30120c19c0181a0c95f9e883e93d9ba11ee8312已Git-only部署并接管formal_seed013_20261024_01，tmux historical_b32_six_slots_20261024_02、控制器PID1442482。先核实四训练SID=PGID=各自PID、TTY=0，再仅SIGTERM旧调度器1430541；没有给训练发信号或重启。原四PID1430615/1430769/1431304/1431843及start ticks均保持；新增seed1 Full PID1442557、seed3 Full PID1443124，二者前5epoch均0skip；原三视图已到task1 epoch13、seed0 Full到task3 epoch16，0skip。当前6训练/其余12训练自动排队，后续完整36份8阶段val/test仍自动执行。

B64、原训练CLI、模型源码00fbf90和所有seed/epoch/优化器/AMP/RNG保持，6组config均核验。GPU0约17056MiB使用/7026MiB空闲、计算100%，/dev/shm约38MiB。原seed1 task0完整score的IDs/labels/classes/logits/probabilities与历史逐值一致。新增8项本地调度测试、远程22项相关检查及实际Linux zombie退出码预检/dry-run通过。第一次暂停式接管失败的driver.log保留，不当作训练失败；成功driver02.log为当前入口。

模型工作树multi-lane-main-historical-b32-seed03-20261024仍clean在00fbf90；独立控制工作树multi-lane-main-b32-gpu0-controller-20261024在a30120c。队列继续原logs/historical_b32_search/formal_seed013_20261024_01/queue.json，带controller=six_slot_handoff；控制日志在控制树logs/historical_b32_search/concurrency_20261024_01/driver02.log。原run的runtime_scheduling_policy.json可动态设训练上限（当前6）、评估上限4、GPU余量6000MiB/共享内存余量1024MiB、启动间隔12秒；变更历史jsonl保存。

资源空闲不等于计算空闲，4组时计算已100%；此次增加同时训练组数，不承诺线性提速，也不在历史复现中改batch。接管审计concurrency_after_audit.json和concurrency_handoff.json保留原PID身份及训练未重启证据。待全部完成再分析精度；禁止重复启动或更新活动模型/控制源码。最新执行记录仅本地上下文更新。

调度接管26份非checkpoint审计/日志已同步本地并逐SHA核验，归档SHA ea80ee565ebf0e2ec33cab00bd610b02fdbb3ce7b93611a5cb87792ddd62d394。模型00fbf90与控制器a30120c工作树均clean；原主树153行/test-only5行及状态SHA保持。

## 2026-10-24：6槽首次暂停式接管中止，改用独立会话验证后退出旧调度器

控制器7fa561d已Git-only部署multi-lane-main-b32-gpu0-controller-20261024，远程21项相关测试和dry-run通过。但首次SIGSTOP后原调度器未保持T状态，接管在写handoff/启动新训练之前中止；原4组继续、队列仍正常，已SIGCONT回滚。日志concurrency_20261024_01/driver.log保留，原训练从未重启。不能把首次尝试说成成功，也不猜测外部SIGCONT原因。

已核实4个训练各自SID=PGID=PID、TTY=0（与原Popen start_new_session=True一致）。改为先严格验证独立会话，再只SIGTERM旧调度器PID，不发信号给训练PID/组，接管原队列。增加会话不独立即拒绝的回归，本地8项通过；待更新独立控制工作树后再dry-run与正式接管。模型00fbf90、B64、8任务/18组不改。4槽吞吐参考约1158 view-images/s（末3epoch近似，跨任务/评估会影响比较），记录于concurrency_before_audit.json；seed1最近3轮loss与历史仍匹配。

## 2026-10-24：用户要求利用空闲显存，准备GPU0从4组扩到6组并行

实测GPU0 12785MiB已用/11297MiB空闲、计算100%/显存带宽68%，4组正在运行且0跳步，seed1原轨迹前13轮仍匹配；/dev/shm仅185MiB。选择保留B64与完整历史配方，增加并行，不在当前训练中改batch破坏复现。新分支codex/b32-gpu0-adaptive-concurrency-20261024；新增独立接管调度器和动态资源策略，模型工作树00fbf90不改。最多6训练、最多4评估、GPU余量6000MiB/共享内存余量1024MiB、12秒启动间隔。

接管只SIGSTOP旧调度器1430541并保持其tmux，不给原4训练发送信号；新作业仍原工作树/原CLI/CPU72启动，不重训、不改epoch/随机流，全部仍旧标准8任务val/test。只有全部作业完成后才清理暂停的旧调度器，防止旧面板关闭带走训练。现有7项本地调度测试通过，待Git-only部署、远程检查和dry-run；尚未称6并行已生效。方案docs/historical_b32_gpu0_scheduling_20261024.md。

## 2026-10-24：历史b32正式18组已在GPU0启动，seed1前两轮精确匹配历史

formal_seed013_20261024_01已启动，tmux historical_b32_formal_seed013_20261024，运行源码锁定00fbf90c2b86176cc03a2ac1b75a7fec31a5e71e，两端branch/HEAD/upstream在启动时一致且clean。首批四进程均GPU0：seed1 THREE_BASE PID1430615、seed0 THREE_BASE PID1430769、seed3 THREE_BASE PID1431304、seed0 FULL_BASE PID1431843；其余14训练自动排队，完成后自动36份完整8阶段val/test。三路已完成前2epoch各84更新/0跳步，Full前5epoch正常；seed1两轮主loss .67947638/.66449360与Adapter .03236856/.02941815均与历史逐字匹配，不能预称最终32.6800已复现。GPU0约12785MiB/100%利用，其他GPU不使用。

smoke_seed013_20261024_01完成，新旧36b7753 seed1 Full/三视图全部方法张量、训练loss与完整预测逐值等价；seed3 aux0正常。12份常规重放/4对视图重放全等，冻结/恢复/初始化/全局采样器核验通过。454项完整回归（1skip）和最终14项相关检查通过；本地轻量导入因无Torch不能执行，实际验证均使用远程ddp环境。本条覆盖此前“正式待启动”状态。

正式输出output/historical_b32_search/formal_seed013_20261024_01，日志对应logs，控制日志logs/historical_b32_search/implementation_20261024_01/formal_driver.log。18组均全TRAIN/seed0,1,3/8×30epoch/B64/Adam-cosine/旧标准，基准b32 scale.03/Adapter LR.0004/aux.1/Person颜色扰动0；seed0/3各六个单因素变体。允许Test参与选择，探索排名和全部结果会保留，不用最好seed替代强参照。源代码运行中不更改；最新启动记录仅本地上下文未提交。

95份短跑/正式启动非checkpoint产物已同步本地并逐SHA核验，archive SHA8a432f1a957c5a705d992b14b82f189a535f78ce7173af7a5b20a70c3ad90b41。原主树153行/test-only5行及状态SHA未变，历史36b7753源码clean。等待完整completion后同步并分析，禁止重复启动；方案docs/historical_b32_seed03_search_plan_20261024.md。

## 2026-10-24：历史b32新旧两基准短跑逐值等价，准备18组正式队列

smoke_seed013_20261024_01 completion=complete。seed1 THREE_BASE/FULL_BASE相对不可修改36b7753的两任务四更新，所有方法张量、loss/LR/步数及全部阶段预测逐值等价。seed3 THREE_AUX_0正常；三组新训练0skip、旧参数与Frozen视觉不变、恢复model等于compact、同seed初始化和三视图采样RNG核验通过。12份常规重评估与训练预测一致，4对视图NPZ逐字段重放一致。454项完整回归通过（1skip）。

修正队列summary对第二份smoke重评估也执行训练预测比较的校验条件，补故意损坏第二份logits即拒绝的回归；只改变验证，不改变训练或模型。将提交这批验证记录，通过Git ff-only同步后启动formal_seed013_20261024_01（18组/8task/完整val-test/GPU0四槽）。此条尚未确认正式PID；不能把短跑mAP当正式提升。方案和旧标准保持。

## 2026-10-24：新历史b32分支已部署，454项回归通过，GPU0短跑进行中

源码74b4c62已提交并push，远程Git-only创建multi-lane-main-historical-b32-seed03-20261024，两端branch/HEAD/upstream一致且clean。13项相关测试、454项完整回归通过（1skip）。短跑smoke_seed013_20261024_01、tmux historical_b32_smoke_seed013_20261024已启动，最多4作业GPU0；新入口seed1Full/三视图和seed3 aux0，旧36b7753 native seed1Full/三视图。已观察新旧三视图task0短跑loss .75824943／Adapter .02955740相同，Full两任务训练完成0skip并通过边界核验；尚未完成全部参数/预测等价及重复val，不称正式启动。

运行日志logs/historical_b32_search/implementation_20261024_01/smoke_driver.log，队列对应logs/historical_b32_search/smoke_seed013_20261024_01/queue.json；产物output/historical_b32_search/smoke_seed013_20261024_01。保持旧源码和上轮工作树不变，当前不在远程运行中更新源代码；待完整smoke通过才正式18组。此前本地“未远程验证”由本条覆盖。

## 2026-10-24：启动历史b32 seed0/1/3复核与18组参数探索实现

用户授权GPU0并行，随后明确允许根据Test调参；已在AGENTS.md记录此次探索例外，旧评估标准与强Full参照保持。新分支codex/historical-b32-seed03-search-20261024继承0073a2f及未提交分析，未覆盖原改动。计划seed0/1/3各三视图基准和匹配Full，共6组；seed0/3各scale.01/.05、Adapter LR.0002/.0008、aux0/.2单因素6组，共12组；合计18组完整8task val/test。历史基准scale.03、无Person颜色扰动、aux.1、Adapter-ASL、B64/30epoch/Adam-cosine/主LR.0125/Adapter LR.0004/CPU72/旧全局RNG/旧AMP，仅GPU0最多4并发，内存不足排队。Test参与选择如实标探索，不改旧强参照。

新增historical_b32_search配方、GPU0队列和不可修改旧36b7753 native短跑；训练入口支持seed3，并增加独立profile校验，不改模型结构。先2task×2update的新旧seed1Full/三视图逐值等价、seed3 aux0、完整重复val和冻结/恢复审计；通过才跑正式，正式seed1完整Test必须精确匹配历史，否则暂停发出新作业查因。方案docs/historical_b32_seed03_search_plan_20261024.md；当前本地实现、未远程验证或正式启动。

只读确认GPU0约24GB空闲、其他卡忙；原主树153行及test-only5行和状态SHA不变，旧源36b7753/上轮0073a2f clean。训练输出output/historical_b32_search，日志对应logs，代码Git-only同步，旧工作树不修改。

## 2026-10-24：追溯历史32.6800，纠正遗漏的共享三视图Adapter-ASL正面参照

用户给出的Test32.6800／39.5292、task0 53.5004、Full31.2373／Person30.4315／Face25.5217来自2026-09-29 seed1 B32臂，源36b7753。它确为共享三视图、Adapter-ASL、aux.1、固定特征融合、ParaX关闭；融合对自身Full+1.4427。旧Adapter scale .03、Person颜色扰动0，与本轮seed0/scale.1/jitter.1概率.2不同。两者旧标准相同，8阶段ID/targets/classes逐值全等；独立复算8份Test三指标最大误差4.27e-14，240epoch/13950updates/0skip。历史跳过validation、不保存checkpoint，无法直接补原权重validation。

上一轮分析遗漏这一正面共享ASL记录，现已补报告；只可说本轮seed0配方未超强Full，不可泛化成共享三视图ASL从未有提升。32.6800数值超seed0独立Full32.5365，但跨seed/配方非严格消融，也不能把差距单归因scale或seed。若以后执行复核应先完整恢复历史seed1/.03/无颜色扰动，再做匹配单Full；本次未授权启动、不运行新实验。追溯audit在output/feature_fusion_adapter_asl/historical_b32_seed1_trace_20261024_01/audit.json，详细补充在docs/feature_fusion_adapter_asl_results_20261024.md。

## 2026-10-24：四组原特征融合／Adapter-ASL完整结束，结果已同步并完成只读分析

本条覆盖下面历史“运行中／等待completion”状态。formal_seed0_20261024_01四组训练和8份完整validation／test均完成、全部exit0；每组8×30轮、13950成功更新、0跳步。源0073a2f856686cff8d8c764370b33209139cc00b不变，远程独立worktree clean。32份compact SHA与评估元数据一致，旧任务参数和Frozen Forward不变；32个任务Adapter up.weight均非零。Full八阶段预测精确匹配历史，四Test指标32.536519／39.144491／32.575617／49.382716；后续继续唯一历史samples_intersect_seen_classes标准。

| 配置 | Val最终 | Val阶段均 | Test最终 | Test阶段均 |
|---|---|---|---|---|
| 单Full＋Adapter-ASL |42.3799|49.2652|32.5365|39.1445|
| 三视图Adapter-BCE＋辅助0.1 |43.0455|49.7375|32.2818|38.9566|
| 三视图Adapter-ASL无辅助 |43.1640|50.8093|32.2871|39.3312|
| 三视图Adapter-ASL＋辅助0.1 |43.0185|49.7887|32.1929|38.9920|

三视图validation正增益，但均未验证最终Test超强Full。无辅助ASL有部分价值：val 8/8阶段更高、Test阶段均+.1867、最终cF1=33.6608；不是全面失败。主要候选恢复辅助后val/test全部8阶段均低于无辅助；Person/Face Test端点29.4957/24.7122→30.4744/25.3926，融合反降32.2871→32.1929。共享模型Full端点约30.92低独立32.5365，局部互补未覆盖Full削弱。线性共享head下固定特征融合等价固定logit加权，无内容选择。辅助提高跨视图预测相关性；这与互补性受限相容，不是梯度冲突因果证明。

Test可靠Face子集无辅助ASL+.1937、不可靠−1.0489；后者Face权重已0，不能归因坏Face直接混入。Val两子集均正，Face可靠率两集合几乎相同，不能按Test选回退规则。无辅助组最后三类贡献最终差值−.1897，其余23类净−.0597，不以累计mAP下跌推断遗忘。500次原图成组配对bootstrap关键最终差值区间均跨0，仅seed0，不宣称显著胜负。缺少BCE无辅助格，辅助同时作用主参数和Adapter，因果分解有限。

195份正式非checkpoint产物同步本地并逐SHA核验，archive SHA d00484a0d8278c9ca32248bb35467e0b5becf16c2d089945021b107a627ba34e。独立NumPy复算64份常规阶段指标及48份视图mAP，常规指标最大误差5.69e-14；32份原生Test与重评估预测逐值全等。未下载权重，额外Adapter权重审计JSON SHA 2d835e69e7225fc000bef65e5d8b5b3424a01ec4522e8c5c7f37abd9aa79b2f4。主树153行／test-only5行及状态SHA不变。

详细报告docs/feature_fusion_adapter_asl_results_20261024.md，分析脚本scripts/emotic/analyze_historical_feature_fusion.py，CSV/JSON在output/feature_fusion_adapter_asl/analysis_20261024_01。此轮仅同步、CPU审计和报告，无新训练／GPU前向，无提交推送合并。下一步候选是现有无辅助／辅助task0/2/7 validation关闭Adapter残差归因（尚未执行）；先保强Full并明确局部证据纠错机制，不重叠加Attention/位置，不将已失败方案换名重训。

## 2026-10-24：正式四组GPU0并行启动，首轮84更新/0跳步，Full轨迹匹配历史

formal_seed0_20261024_01已启动，tmux feature_asl_formal_seed0_20261024；远程独立工作树multi-lane-main-feature-fusion-asl-20261024锁定0073a2f856686cff8d8c764370b33209139cc00b，启动时两端branch/HEAD/upstream一致、工作树clean。训练PID FULL_ASL1301745、THREE_BCE_AUX1301828、THREE_ASL_NO_AUX1302304、THREE_ASL_AUX1302839，全部CUDA0/OMP72/MKL72，各首轮84updates/0skips并进入后续轮；Full首轮.61478711/Adapter.02839343与历史精确匹配。正式未完成，无新最终成绩；禁止重复启动或提前称提升。

smoke03以只读链接复用02训练，8个workers2评估全部exit0；14份常规/视图NPZ重放字段逐值相同，8份训练阶段预测与评估重放的IDs/classes/labels/logits/probabilities亦逐值相同。原生f1ada与新入口Full两任务所有方法参数/训练loss/LR/步数全等；三个三视图组边界CPU RNG相同。447项完整回归通过(1skip)，最终14项相关检查通过。恢复状态model等于compact、旧任务参数与Frozen视觉不变、0跳步已查。

正式四组全TRAIN/seed0/8×30epoch/B64/13950更新、原全局采样/旧AMP65536-growth2000/旧增强；原固定feature融合、共享Selectors/Prompt/head/Image Adapter，不启用Attention/位置/MoE/ParaX。主要候选主参数BCE(fused)+.1meanBCE(views)，Adapter ASL(fused)+.1meanASL(views)，Face仅可靠样本。训练后自动GPU0最多4并发旧标准val/test；显存准入6000MiB，workers2预计预取低于4GB。Full正式四指标必须精确重现32.5365/39.1445/32.5756/49.3827，否则守卫报错。

日志logs/feature_fusion_adapter_asl/formal_seed0_20261024_01，输出对应output；入口日志logs/feature_fusion_adapter_asl/implementation_20261024_01/formal_driver.log。源代码锁定不随本地后续执行记录变化；本条实际启动状态仅本地上下文更新，不在运行中更改远程代码。原主树/test-only状态153/5行及SHA保持。正式与短跑共154份非checkpoint产物已同步本地并逐SHA核验，archive SHAb2c9a5831a640e5bc17055e3b4b433f98c0a274f678323d96eafa82ac314c88d；权重留服务器。

下一步等待正式completion，核对四组预算/冻结/Full锚点及每阶段旧人群，再同步分析主候选对BCE辅助对照、ASL无辅助与强Full；同时检查Full端点是否保住。初步短跑不作性能筛选，test不用于改系数。

## 2026-10-24：Full新旧短跑全部方法参数等价，保留权重仅接续workers2评估

smoke_seed0_20261024_02四组及原生f1ada Full均完成，Full两任务全部方法张量、loss/LR/步数逐值等价；四组初始化base SHA2fb4071185ac375fef174e18e066c3eca800db733aa8197f8a116f17350cc2b0一致，训练0skips、旧参数冻结、完整边界保存核验通过。13项相关回归通过；此前447项全回归通过(1skip)。

发现CPU72配eval workers0导致四主进程合计约135核用于逐图预处理，GPU利用率低；已计划停止02控制器及4个评估PID1290562/1291109/1291188/1292518，所有训练早已结束且权重保留，不重训。补evaluation-only smoke复用模式，03仅以只读链接复用02完整训练，改回旧workers2核验8路validation重放；当前正在部署，尚未称03完成或正式启动。第二轮native等价及planned_evaluation_restart.json保留。正式评估也workers2，最多4并发预取预计低于4GiB共享内存。

正式Full完成时增加四指标精确锚点守卫，防止较弱桥接结果被当强Full。新增测试检查workers2/历史人群。方案与3上下文同步；原始failed01及planned02都保留。

## 2026-10-24：447项回归通过，短跑训练完成，修正评估workers审计后重验

远程独立工作树multi-lane-main-feature-fusion-asl-20261024已Git-only部署d6f10aa，四组smoke_seed0_20261024_01训练均完成2task×2updates、0skips；调度器在Full结束后的配置检查因CLI eval_workers=None保存为resolved2而失败，尚未开始完整smoke评估及原生Full对照，不是训练失败。其余子进程独立session均正常完成，源产物保留；已修正validate_config预期2并补回归。将用新run-id完整验证后才正式启动，不把失败的控制器当正式成功。

447项完整回归通过（1项skip），12项相关检查通过；梯度测试覆盖实际train_task参数组路由和Face掩码。上一轮三个seed完整Full已同步88份非checkpoint产物并逐SHA核验；8阶段三seed IDs/targets/classes/logits及完整test指标与各自历史全等，每组13950更新/240epoch/0skips。同步archive SHA0f44e23e2e9ed865fab83aa865288e7ef936c9b7cd381b121a73e92ec4bbf4b5。原主树/test-only状态153/5行及SHA保持。

## 2026-10-24：按用户要求只验证原特征融合＋Adapter-ASL

用户要求暂停Attention/位置，目标是强Full上的三视图融合。八组旧表是90%fit、aux0.1、BCE；后续三视图ASL用aux0，配方非仅替换Adapter目标。新建exp/feature-fusion-adapter-asl-20261024，从ab84674创建；计划FULL_ASL、THREE_BCE_AUX、THREE_ASL_NO_AUX、THREE_ASL_AUX四组，原固定feature融合不改，ASL只给Adapter，主要候选恢复0.1视图辅助BCE/ASL；全TRAIN/seed0/8task/30epoch/B64/13950更新/旧AMP65536-growth2000/workers2/global RNG/CPU72，仅GPU0。

修改runner和评估器增加严格historical_feature_fusion入口及预报告恢复保存，原模型结构/融合公式不改；新增配方、四并发队列、梯度与Face屏蔽测试。先四组2task×2update smoke，原生f1ada Full所有方法张量等价通过才启动正式。正式/短跑输出output/feature_fusion_adapter_asl，日志对应logs。当前只本地实现，尚未远程测试或启动，不预称成功。方案docs/feature_fusion_adapter_asl_plan_20261024.md。

刚只读确认上一轮三Full完整结束：seed0精确复现test32.536519/39.144491/cF132.575617/oF149.382716、val42.379911/49.265161；seed1 test32.134682/39.020979，seed2为31.797277/38.611898，均complete。完整产物待同步核验，旧“运行中”状态由本条覆盖。原主树/test-only不触碰。

## 2026-10-23：独立/共享端点全八阶段归因完成，Full是主要可观察损失来源

旧概率融合规则及Face掩码固定，8阶段×3视图IDs/labels/classes核验通过。旧三套独立模型33.3392/40.2855的四指标复现；当前共享原特征融合32.1638/39.1273，单改概率融合31.9319/38.8677，故公式更换不能修复。保持当前P/Face，仅换旧独立Full后33.5724/40.2616；所有4种P/Face背景下换独立Full均增加1.3349–1.6406。此为只读预测重组归因，不是新训练模型的成绩，不能据test选组合。旧/新配置还不同；同期现代单Full32.2519对共享Full30.5910的1.6609差距也支持Full削弱，尚未分离哪个共享组件导致。

原生seed0 task0–2共90轮训练字段（除耗时）、完整阶段指标和score全部字段与历史逐值相同；task3–7未完成，seed1/2训练仍运行GPU0。seed1/2前21轮主loss与Adapter loss亦各自匹配历史到8位小数、0跳步；该额外审计已同步本地，SHA8a9c0410da67489395f9a6ed463ae132031cb0f82d2aed2c560745e2ad5f0416。14份线程/端点/启动产物已同步本地并逐SHA核验，archive SHA190f1ba5e19a3727c6cd7cf0f8ef9e16cd3bf08150a570657bb60687b9e3da5f。报告docs/historical_standard_alignment_20261023.md及output/strong_full_asl_three_view/historical_three_view_endpoint_audit_20261023_01/summary.csv。主树/test-only状态153/5行及SHA不变。

下一步等待三Full完整旧标准test+val，核对四指标和每阶段人群；不重复启动。未来既保留Full32.5365/39.1445锚点，也保留独立三视图33.3392/40.2855强参照，先在validation设计有实质差异的完整Full保护方案，不能重包装旧阴性控制。

## 2026-10-23：三种子旧标准Full核验已在GPU0启动；补独立/共享端点归因

最新执行：seed0原生CPU72 PID1139164继续，historical_exact_seed0_20261023_02控制器已接管后续validation；seed1/2控制器historical_exact_seed1_20261023_01、historical_exact_seed2_20261023_01已启动，训练PID1146811/1146812，各自首更新前归档配置differences={}。9项相关回归通过（首次模块测试路径未含tests而导入失败，补PYTHONPATH后全通过）。CPU1包装诊断PID1137560已SIGTERM+SIGCONT计划收尾，原日志/产物保留，不再遗留挂起进程。尚未称完整四指标重现；只GPU0，GPU1–7不动。

确认旧独立三模型概率融合三seed均有收益：Full32.5365/32.1347/31.7973→三视图33.3392/32.9575/32.7390。当前共享三视图Full30.5910→融合32.1638仍增1.5728，局部端点未整体变差；共享任务参数、融合监督与特征融合不等于原强Full加两路。新增只读scripts/emotic/audit_historical_three_view_endpoints.py，以旧锁定权重、旧人群、相同ID/标签/Face掩码，完整2^3替换端点，验证独立Full与公式影响。不训练、不改模型、不按test选规则。报告docs/historical_standard_alignment_20261023.md更新最新执行及旧标准唯一政策；过去全人群记录保留。

## 2026-10-23：定位复现偏差的CPU线程根因；后续旧标准唯一，补三种子Full参照

用户明确后续全部旧标准，不再用新全人群标准；固定seed0 Full锚点32.5365/39.1445/32.5756/49.3827。AGENTS.md已同步写入规则。归档config与固定三视图总表双重确认该组确为seed0；旧seed1/2 Full为32.1347/31.7973（test最终），不是把其他seed误标成seed0。将按相同旧配置补seed1/2 Full完整8任务val/test并分别报告，不按test挑seed。

关键根因已受控验证：原生f1ada22 runner/相同seed0和完整参数，OMP/MKL1的前两epoch精确复现前次偏离轨迹（.61478234/.60360414）；原启动环境默认72线程则精确复现历史（.61478711/.60355623），Adapter loss亦对应匹配。相同seed/相同RNG消耗下，CPU orthogonal_检查证实5个Prompt矩阵浮点值及部分FP16舍入不同（最大约1e−8，5组各1–14个FP16元素变化），Selectors该诊断中相同。不是仅有paired配置遗漏；补齐paired但仍CPU1时轨迹不变。因此修正此前将配对参数视作主要解释的推测，记录初始化环境为复现必要条件。

当前原生seed0 PID1139164正在GPU0继续完整8任务（historical_native_probe_seed0_20261023_01），前两轮历史loss匹配；CPU1诊断PID1141026仅观察2轮后计划终止，非完整实验；包装CPU1 PID1137560临时STOP保留，待收尾停止并记诊断，不能忘记挂起。下一步独立控制器接管原生seed0完成后的验证，并用CPU72启动seed1/2；不改原生运行中的旧源码。

已核对旧三视图正式表：seed0独立Full32.5365，独立Full+Person33.2672，独立三视图概率融合33.3392；seed1三视图32.9575，seed2为32.7390，三seed均优于各自Full。旧融合是3套私有任务参数、各自监督后做概率融合；当前三视图共享任务参数并以融合目标训练，做特征融合，不能当同一个方法“突然失效”。当前Full端点30.5910→融合32.1638仍有+1.5728补偿，但低于旧独立Full。继续只读核验公式/端点来源，不重新训练Attention/位置组合。

## 2026-10-23：用户要求恢复旧标准，已完成全矩阵历史人群复算并修正复现入口

主报告恢复samples_intersect_seen_classes：每阶段累计已见类至少1正标签；val人数2285/2375/2381/2397/2397/2397/2397/2397，test3142/4600/4691/5232/5342/5354/5367/5368。8模型×8阶段test IDs/labels/classes与旧存档64项全等，CPU只读复算。旧口径test平均单FullASL38.9072、三视图ASL39.1273、+Attention38.9613、+位置38.6970、+两者38.8396。最终分数不变；三视图ASL相对独立Full的阶段平均改为+.2201，而非全人群−.0334。旧Full39.1445与原35.x主要是人群差异；当前最终仍未超过旧32.5365，叠加失败结论不变。

已新建fix/historical-protocol-alignment-20261023，保留全部原产物和未提交分析。run_strong_full_asl默认历史人群，全人群需显式full；legacy_full_reference补回旧paired/margin/jitter，并在首个更新前与归档config逐字段校验（仅放行两项compact保存字段）。计划GPU0唯一新历史Full校准重跑，seed0/8task/30epoch/B64/workers2/旧Adam-cosine-ASL及随机流，完成后旧标准val；原主树/test-only/旧源码不改。当前仅预检，尚未称已启动或已复现。详见docs/historical_standard_alignment_20261023.md，复算输出output/strong_full_asl_three_view/historical_standard_20261023_01。

## 2026-10-23：强Full ASL三视图与叠加实验全部结束，完整分析完成

recovered_gpu0_seed0_20261023_01 completion=complete，21个恢复批次作业全部exit0；合并原3组Full已完成结果，八组训练/16份完整val-test/6项关闭归因齐全。7个新模型均13950updates/0skips/旧任务依赖冻结/恢复与compact模型一致。479份产物已同步并逐SHA核验，独立CPU复算128份常规阶段score与18份运行时score，人群/标签/类别逐值相同，最大mAP误差4.98e−14，模型源码86a8c1c clean。保留前次tmux故障与任务边界重算事实，不将恢复说成逐epoch无损。

四列val最终/val阶段均/test最终/test阶段均：单Full BCE42.3467/48.1833/31.7109/35.1143，单Full ASL42.1754/47.7626/32.2519/35.6855，三视图BCE41.9839/48.1421/31.1035/34.7656，三视图ASL42.5578/49.1965/32.1638/35.6521，+Attention42.5144/48.8822/31.8968/35.5416，+位置42.8656/49.2204/31.8404/35.2431，+两者42.4552/49.2248/31.8472/35.2948。ASL改善三视图BCE但迁移到三视图未超过同批Full test；新增两模块和组合没有验证出叠加收益。位置与组合test全部8阶段低于ASL底座，Attention val全部8阶段低；val最终bootstrap关键差值区间均跨0，只有seed0。

关闭条件贡献（开启−关闭）task7：ASL底座Adapter−.4199、Attention单独−.0412、位置+.0107、组合关Attention+.2081/关位置+.0043/关两者+.2177。不能把关闭版本当独立重训，也没有test关闭数据。Attention实际残差范数足够大，未形成稳定净收益；位置bias小、softmax KL约1e−5–1e−4，val提升集中Sadness等类别，不支持已建立可靠人物对应。三视图ASL test Full端点30.5910→融合32.1638，仍低独立Full32.2519，互补补偿未超过共享训练代价；未做跨视图梯度夹角，不能宣称已证明冲突。

必须纠正R“严格复现”称呼：旧源码Full重跑固定人群41.8703/48.0657/31.9535/35.2819，旧人群test均38.5130，未达历史32.5365/39.1445。原config paired_full_person=true/margin.15/jitter.1/.2，R为false/0/0，匹配不完整，差值因果未分解；保留旧32.5365外部强参照，不以低R替代。详细报告docs/strong_full_asl_three_view_results_20261023.md，CPU复算脚本scripts/emotic/analyze_strong_full_asl_results.py，分析表/审计output/strong_full_asl_three_view/analysis_20261023_01。

当前只完成结果同步/CPU分析与文档，不再运行GPU前向、训练或test搜索。暂停扩展Attention/位置叠加，不合并main。先修正历史Full复现、审计独立专家补救Full的净空间；同时回查既有Full-anchored残差A0/A1/A2 42.3799/41.8802/42.1556以及后置logit+.0857的失败记录，不能将旧路线换名后再训。

本批无待跑实验；停止旧进度/ETA监控。若用户要求下一阶段，再制定与旧保护Full/后置校准有实质差异的控制，不直接启动新训练。

2026-10-23 18:48 CST：五组剩余训练已全部GPU0并行恢复，PID T_BCE1121327/T_ASL1121349/ATTN1121367/POS1121389/BOTH1121407，均独立session且已进入新的训练epoch，首轮0跳步。当前运行模型HEAD86a8c1c clean，调度脚本独立Git工作树HEAD6eb1e3e；新run_id=recovered_gpu0_seed0_20261023_01，日志/输出均在对应strong_full_asl_three_view目录；原三组已完成Full产物只读链接复用。五组边界validation的全部npz字段（含logits/概率/人群/标签）与原输出逐值相同，已重算的前2/5/4/1/1epoch训练loss/LR/步数均与原轨迹日志一致（耗时除外）。这支持当前恢复正确，不等于未完成部分已验证。

恢复前完整状态校验5/5、调度测试4/4、真实tmux父会话关闭后独立子进程存活探针均通过。原源恢复checkpoint SHA保持，primary/test-only状态仍153/5行且SHA不变。19份故障/预检/恢复启动元数据已同步本地逐SHA核验，归档1001319ab44360837744f1631cb58be0b9886dc480822284881c9438619abf6a。全部后续训练/val/test/6项关闭归因只GPU0，至多5总进程/2评估，显存不足自动等位；旧多卡ETA不再适用。此条覆盖前次“尚未恢复”状态，不掩盖旧tmux接管造成的中断与重算。

接下来只跟踪recovered_gpu0_seed0_20261023_01的新队列与completion；原formal批次failure作为故障记录保留。全部完成后同步新树结果并核对统一预算13950/组和8阶段人群，结合原3组结果汇总分析。

2026-10-23 18:42 CST纠正前次接管状态并记录故障：旧派发器SIGTERM/CONT退出导致旧tmux会话关闭，五个原训练子进程收到会话退出影响而终止，原“训练不重启”目标未达到，已向用户明确说明并致歉。原始日志/全部任务边界恢复/模型权重保留，不删除旧failure。最新用户要求剩余工作全部GPU0并行，覆盖GPU1–7续完原进程安排。恢复边界T_BCE/task6、T_ASL/task5、ATTN/task5、POS/task4、BOTH/task6；只需重算各自未完成任务（原日志已做26/10/15/24/1epoch），不能声称epoch无损恢复。

新增scripts/emotic/recover_strong_full_asl_gpu0.py，原模型HEAD86a8c1c clean环境下从全状态边界续到8task，保存至新recovered_gpu0_seed0_20261023_01输出树；3组已完成Full训练及val/test只读链接复用，原输出不覆盖。GPU0至多5训练/2评估且总5进程、显存准入6500MiB，所有子进程新建独立session，防止调度tmux退出连带终止。修改旧接管脚本为保留暂停旧派发器到全部子进程结束后才退出；当前按最新要求不再使用该接管方案。正在预检/部署，尚未宣称恢复已启动。

2026-10-23用户要求GPU1–7当前训练跑完即释放，后续完整validation/test与关闭归因仅GPU0。已仅SIGSTOP旧派发器1108543，五个训练PID1108598/1108624/1108650/1108670/1108697继续不重启；新建exp/strong-full-asl-gpu0-eval-20261023，实现只接管原训练PID的独立调度脚本scripts/emotic/resume_strong_full_asl_gpu0.py。GPU0本身训练完成前不派发评估，之后GPU0最多2路，固定B64/workers0/原模型HEAD86a8c1c与全部数据监督不变。新脚本从独立Git工作树运行，原实验代码树不切换；当前正验证接管，不能视为已完成迁移。

2026-10-23 18:28 CST最新进度：历史R及S_BCE/S_ASL均训练与完整val/test exit0；仍有五组训练，其中用户所指四个ASL核心组task5进度分别13/15/20/5epoch（T_ASL/ATTN/BOTH/POS），T_BCE为task6第16epoch。总预算13950updates，各组剩1700/1620/1420/2020/860；最近每update约.75–.83秒，四ASL纯训练还19–28分钟，计任务边界报告/保存预留20–35分钟。后接完整val/test与关闭归因，整批粗估还50–80分钟、北京时间约19:20–19:50（保守20:00左右）；此为早期耗时更新，非保证。所有已记录epoch均0skips，无failure；未改配置或进程。快照output/strong_full_asl_three_view/startup_audit_20261023_01/progress_eta_1828.json。

2026-10-23 15:43 CST耗时快照：正式task0历史R已16epoch，单Full BCE13/ASL12，五组三视图5epoch；全部当前0跳步。最近Full每epoch约28–31秒，历史R20–22秒，三视图63–71秒（84updates）。以每组13950总更新折算，三视图纯训练约3.0–3.3小时，另含任务边界报告/保存和最终8阶段全量评估。粗估从本次查询起全部含validation/test/关闭归因还需4–5小时（北京时间约19:45–20:45）；这是task0早期速率外推，后续任务/I/O会影响，非承诺完成时间。未改配置或调度。

## 2026-10-23 15:42 CST：正式八组已同时启动，首轮健康检查通过

覆盖此前“尚未启动/准备启动”状态。运行分支exp/strong-full-asl-three-view-20261023，Git-only同步并锁定HEAD 86a8c1c36a6cdbbe7fac5a584aaba1b7f00aadf4；远程独立worktree multi-lane-main-strong-full-asl-20261023 clean。正式run_id=formal_seed0_20261023_01，tmux=strong_asl_formal_seed0_20261023_01，8组训练均已运行，无待启动组。

GPU0三视图ASL＋Attention＋目标位置；GPU1严格历史Full；GPU2单Full BCE；GPU3单Full Adapter-ASL；GPU4三视图BCE；GPU5三视图Adapter-ASL；GPU6三视图ASL＋Attention；GPU7三视图ASL＋目标位置。各组均seed0/8task/30epoch/B64，内部task0→7顺序继承增量状态，不能把阶段并行当同一实验。GPU1既有外部作业保留，本批小型历史Full使用剩余容量。

正式首轮八组全部84steps/0skips，并已完成后续多个epoch，未发现Traceback/OOM/failure.json。每组训练完成且通过预算、冻结及恢复核验后，自动接续完整固定人群validation/test；释放卡每卡最多并行2路评估，6项预声明validation关闭归因及最终汇总自动执行。尚无正式最终结果，不以启动健康当性能提升。

434项回归、30作业短跑、19项最终真实检查与断点恢复/预测等价预检均通过。已同步416份预检与启动配置文件并逐SHA核验，另存44个checkpoint SHA而未下载权重；归档SHA 0a299b321264e96e52fa69d7473cfb19b9045bf853a5ccd8fa5c5819f6e40870。审计output/strong_full_asl_three_view/startup_audit_20261023_01/{remote_manifest,local_sync_verification,formal_health}.json。primary/test-only原153/5行状态SHA及历史源clean保持，未改用户树。启动后仅补本地执行与上下文记录，不再变动远程运行代码。

当前待办：等待本批8组完成，先核对正式completion/预算/冻结/评估人群与历史R匹配标志，再按固定validation比较监督与视图迁移、Attention/位置及叠加收益，最后完整报告test；不要重启或重复提交同批任务。正式产物完成后同步回本地并核验SHA。

## 2026-10-23：全部预检通过，准备正式八组多卡运行

新协议7组＋旧Full严格复现R，正式全TRAIN/seed0/8task×30epoch/B64/13950updates每组/aux0/主LR.0125/模块LR4e-4/WD0/cosine。BCE→原任务及Attention/位置，ASL9.8/0/.05→Adapter；新7组scaler1024/growth1e9/workers0，R原f1ada22/全局随机流/workers2。R最终另记是否数值匹配32.5365/39.1445，不把预算verified冒充历史得分已复现。

最终434项回归通过，真实CLIP B64公共初始化/RNG/同视图输出逐值一致；完整smoke03共30作业全部成功、0跳步，16份重复预测逐值相同。最后19项真实检查全true：5个三视图模型×2边界单前向导出与原评估逐值相同，历史Full当前/旧代码2边界重放相同，新增梯度日志后恢复model/requires_grad/optimizer-scheduler-scaler/RNG/sampler全等，六参数组梯度范数已记录，R workers2短跑成功。无新超参搜索。

多卡调度每卡1训练，完成核验后释放卡接续每卡最多2路val/test，6路预声明validation关闭归因自动执行；GPU0–7按实际显存准入/不足等待，不改B64或其他作业。正式run_id拟formal_seed0_20261023_01，入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，输出output/strong_full_asl_three_view/formal_seed0_20261023_01、同名logs；本条为启动前锁定记录，不能作已启动证据。远程primary/test-only和历史代码树继续保持。

2026-10-23预检全部完成补记：3868f61 smoke03完整30作业（8训练/16validation与重放/6关闭归因）全部exit0，0跳步，16份scores重放逐值一致；恢复probe所有状态与logits逐值一致。历史原始config进一步核对workers=2，严格R保留2/旧随机流；此前包装器smoke用0仅作接口验证，新七组仍固定workers0。5d56449增加单前向融合+视图导出及首batch参数组梯度只读日志，待最后回归/真实评估等价核验后锁定正式，尚未启动正式。

2026-10-23预检补记：3868f61完整433项回归通过；新流水smoke03八组训练均0跳步，16份同checkpoint validation重放逐值一致，历史Full导出/单视图评估也通过。恢复probe从cd1489d task0接续至task1，model/requires_grad/optimizer-scheduler-scaler/RNG/sampler及两阶段val logits全部逐值一致。当前最后6路关闭归因短跑仍在执行，正式未启动。补充首批各参数组梯度范数只读日志，并实现仅新三视图协议单前向导出融合及各视图score，待数值等价核验；没有更改训练损失或模型公式。

2026-10-23短跑更新核验：cd1489d八组真实B64×2task×2updates训练全结束/0跳步，历史包装也成功；最重ASL Both峰值2818.77MiB/保留3034，单Full ASL1357.87MiB。正式尚未启动。短跑汇总需识别updates预算的scheduler/epochs元数据，已修正；新增训练完成即完整性核验并接续各组val/test的多卡流水排程，释放的单视图卡不空等，其余训练继续。接着验证新调度、评估和恢复。

2026-10-23预检进展：432项回归及真实CLIP B64七组初始化通过。首轮smoke在首个单Full启动前被Face manifest参数合法性检查拦截（无更新），已让单视图命令不传三视图专属参数；保留失败记录并用新run_id重跑。另保持旧快照缺省strong标志兼容。正式尚未启动。

2026-10-23实现预检补记：2b698b2已Git-only同步远程独立strong-full-asl工作树；首轮4项新增测试中3项通过，单Full构造限制发现一处条件作用域错误，已修正并安排含旧功能的完整回归。正式训练尚未启动。

## 2026-10-23：开始实现并执行强Full ASL三视图七组及历史复现

用户授权开始修改、全空闲显卡高效并行完整8task validation/test，覆盖此前GPU0-only限制。已新建exp/strong-full-asl-three-view-20261023（基于0fb85f1），保留此前全部未提交分析文档；计划七组全TRAIN/aux0/seed0/30epoch/B64/主LR.0125/模块LR4e-4，BCE更新原任务及Attention/位置、ASL只更新Adapter。历史f1ada22工作树clean，旧32.5365输出无checkpoint，增加只读旧源码驱动的R历史Full复现及快照导出。

远程primary153行/test-only5行及SHA保持；GPU0、2–7约24GB空闲，GPU1有其他作业/约10.7GB可用，144CPU/528GB磁盘可用。新多卡队列按实际显存准入，不减少B64或干预外部作业。当前正在实现独立协议、评估/恢复兼容、分组梯度测试及真实CLIP初始化核验，尚未运行预检或正式训练。后续Git-only同步新远程worktree，检查通过再正式启动；本条不会把实现阶段当实验已开始。

## 2026-10-23：强Full Adapter-ASL迁移三视图的详细计划（未执行）

用户明确要保留Full＋Image-token Adapter＋Adapter-ASL迁到三视图，再加摘要Attention/目标人物位置；本轮只规划。方案docs/strong_full_asl_three_view_ablation_plan_20261023.md：Adapter始终开启，主矩阵全TRAIN/calibration0/aux0，model参数含Selectors/Prompt/head/Attention/位置用BCE，Image-token Adapter仅用ASL9.8/0/.05，ASL读取融合预测而非独立各视图loss。

七组新配置：单Full BCE/ASL、三视图 BCE/ASL，以及三视图ASL+Attention/+位置/+两者；前四格分离视图迁移和ASL监督，后四格（复用三视图ASL）分离两个新模块。seed0/8task/30epoch/B64/主LR.0125/模块4e-4/WD0/cosine，无新专家或任务复用。每组预期13950updates、七组97650；必要时另加历史严格Full复现R（预计无旧checkpoint可复用，执行前核验），总八组111600updates。R与新配对采样/AMP设置区分，不承诺新S-ASL精确32.5365；固定完整val2397/test5368，历史39.1445另列旧口径。

新七组统一近期稳定AMP初始1024/growth1e9，严格R保留旧数值/全局sampler；共享初始化、Full增强/RNG、分组梯度、缓存、冻结及恢复需预检。现有module_stacking/runner硬限制jointBCE/.1holdout/.1aux，不是只改loss命令即可；需独立协议，不破坏旧矩阵。GPU0沿最近约束，ASL最重B64测峰值后定并发/自动分轮，当前不查询或操作GPU。

预声明task0/2/7 validation开关：ASL底座offAdapter，Attention组offAttention，位置组off位置，组合offAttention/off位置/off两项。先与同期强Full比较，再看对三视图ASL底座增量；不能只以战胜弱BCE为成功，也不保证旧Full预测不变。aux.1、独立三视图/保护Full/seed扩展均后置有条件，不混入主矩阵。本轮仅文档及三个上下文更新；无业务代码、训练/前向、提交推送或新执行分支，不触碰其他工作树。

## 2026-10-23 14:40 CST：旧Full32.5365基线澄清，修正跨轮阶段平均比较

用户指出旧单Full为32.5365/39.1445/32.5756/49.3827。已核对本地主树历史config/summary/scores：四列为test最终mAP/阶段平均/cF1/oF1。旧Full有第2层Image-token Adapter b32/s.1/LR4e-4，任务参数BCE、Adapter专属ASL9.8/0/.05，全TRAIN13950updates；disabled仅关闭Person条件。本轮F000关闭Adapter，三视图共享任务参数，joint融合BCE+aux.1，90%fit12660updates；不是同一基线复现，不能只由数据减少解释。

只读CPU按旧各阶段sample IDs筛当前存档并核对classes/targets：旧Full阶段平均39.1445；当前F000固定完整人群35.4273→旧人群38.7443；AB35.7581→旧人群39.0871。旧人数3142/4600/4691/5232/5342/5354/5367/5368，新每阶段5368；平均差主要有人群口径因素，不能直接解释成性能下降。最终两边5368人/26类IDs及targets全等，旧Full32.5365、当前F000 Full端点30.5598/融合31.9391、AB融合32.4286；AB尚低旧Full.1079，不能宣布超越强Full。

建议优先保留并解释旧强Full训练配方，未来以统一数据/损失/评估的桥接控制研究三视图，不能只对较弱本轮对照汇报增益。ASL/数据量/共享训练的各自因果未隔离；未启动新训练/GPU前向/提交推送。详见docs/historical_full_baseline_clarification_20261023.md及output/three_module_stacking/analysis_20261023_historical_full_01。三个上下文及完整报告已同步更新，原主树只读。

## 2026-10-23 14:28 CST：完整分析已结束；下一步只保留AB归因建议

八组训练、完整16路val/test及F111四路开关已全部完成，不要重复启动既有队列。按基线/A/B/C/AB/AC/BC/ABC，最终val为42.5650/42.4311/42.4323/42.6478/42.9530/41.8702/42.4916/42.3712；最终test为31.9391/31.8358/32.2804/32.1179/32.4286/31.8614/32.2555/32.1765。AB最佳最终val/test，较本轮基线+.3880/+.4896，阶段平均48.5646/35.7581（基线48.5026/35.4273）；C单独val平均48.8647最高，ABC test平均35.8087最高。不能称AB四指标全面最优，也不支持三模块普遍叠加。

1. 建议先对现有Adapter＋Attention（AB）task0/2/7做固定完整validation全开/关A/关B/全关；这是建议，当前未启动。现有归因仅针对ABC，不能直接用于证明最佳AB内的模块贡献。
2. 分别报告融合、Full、Person、可靠Face及task6三类；Person端点不升与融合贡献不能混为一谈。若B关闭无重复下降，不将其包装为有效推理交互；若确有净收益，再单独规划小任务训练控制与固定候选复现。
3. 暂不默认三模块、调scale/LR、加层、扩MoE或用test选择。ABC−AB的task6三类对26类mAP贡献val−.6332/test−.3875，其余23类分别+.0515/+.1354；两split都集中小样本task6。F111 task7关A/B/C/全关val为42.3373/42.2222/42.3755/42.2100（全开42.3712），关C未恢复AB42.9530或task6三类。主要差异留在共同训练的任务参数，不能断言即时位置bias过强或旧任务漂移。AB的Full两split变强，Person端点未同步增强，融合−Full只小增；不能宣称人物对应/互补已解决。
4. 1000次原图配对bootstrap：AB−基线val95%[−.3395,+1.0274]、test[+.0793,+.8921]；AB−Attention的test区间跨0，ABC−AB与全部F111开关区间均跨0。仅固定seed0的样本不确定性、无多重校正，不代表训练seed稳定或机制已证实。不要用历史全TRAIN B0替代本轮90%fit匹配基线；历史test阶段平均36.2781仍高于本轮所有组。

formal_seed0_20261023_01已完整结束：八组各240epoch/12660更新/0跳步，16路完整validation/test及F111 task0/2/7四路关闭归因全部exit0，completion为complete、无failure。远程执行树clean、HEAD0fb85f1，原primary153行/test-only5行及SHA保持。598份非checkpoint产物同步本地并SHA全等；128份checkpoint仅记录SHA未下载。独立CPU复算652项逐类AP最大误差3.98e-13，八组task7 val存档逐值重放；本轮无新模型前向/训练。

完整报告docs/three_module_stacking_complete_results_20261023.md，完整CPU统计/SHA在output/three_module_stacking/analysis_20261023_complete_01。本轮只更新本地文档，不改远程锁定代码、primary/test-only或用户本地主树；无新训练/GPU诊断、提交推送或合并。

## 2026-10-23：等待既有完整评估，暂不训练下一版

12:07补记：F010完整validation也已结束，最终42.4323、阶段平均48.3613；新增17份正式val产物同步本地并SHA全等（合计309份），task7与训练归档逐值一致。当前3/16完整评估作业结束，test及关闭归因仍未齐全。

八组训练全部240epoch/12660更新/0跳步完成；截至12:02 F000/F100正式val完成，16路完整val/test及F111 task0/2/7关闭归因整体仍运行，无failure/completion。GPU0自动分轮授权有效，不改锁定源代码/超参、不干预外部作业、不重复启动。

1. 后续读取既有队列完整结果后同步本地并核对SHA，补齐八组固定人群val/test最终和8阶段平均。train早期报告人群不同，不能用其mean代替；当前没有完整test结论。
2. 最终val AB42.9530比本轮基线42.5650高.3880，但原图bootstrap区间跨0；C加入AB后回落.5817也跨0。保留AB候选，不能宣称三模块叠加或直接扩seed/调scale。
3. ABC-AB回落集中task6的Sadness/Sensitivity/Suffering，三类贡献−.6332、其他类+.0515；从已排task7关闭预测拆类分析。关C若恢复，才支持推理条件影响；否则更应查共同训练轨迹。F111关C不等同重训F110，当前开关不能证明F110内各模块作用。
4. 完整validation最终/阶段平均、分支诊断和推理净贡献支持后再考虑AB独立归因或有限重复；本轮不新增训练、不据test调参、不合并main。历史单模块收益不能与本轮不同数值路径/协议绝对分数相加。

阶段报告docs/three_module_stacking_partial_results_20261023.md，292份完成非checkpoint产物SHA全等及CPU统计在output/three_module_stacking/analysis_20261023_partial_01。两端HEAD0fb85f1，远程执行树clean、primary/test-only状态保持。此前“首批刚启动”状态仅历史记录。

## 2026-10-23：三模块八组正式队列已在GPU0启动

正式批次formal_seed0_20261023_01，tmux stacking_formal_seed0_20261023_01；两端exp/three-module-stacking-20261022、HEAD0fb85f1ed8c277c48ca4267d2dbe7ec1be8fcbf6/upstream一致，远程执行工作树clean。F000/F100/F010/F001四组GPU0并行，PID916442/916468/916494/916520均完成task0第1轮各76更新、0跳步，初始化底座SHA相同。F110/F101/F011及F111按队列自动接续；每组seed0、完整8任务×30epoch、B64、12660更新，统一原图90/10与aux.1 joint BCE，未访问新test或按指标筛组。当前只有启动健康状态，尚无正式最终mAP。

431项完整回归、真实CLIP B64八组初始化逐值一致、八组各2task×2updates短跑、16路validation/重放、4路F111关闭归因全部通过。真实task0边界恢复至task1后model/requires_grad/optimizer/scheduler/scaler/RNG/sampler及val存档逐值相同，来源SHA不变。最重F111短跑峰值分配2658.84MiB/保留2806MiB；GPU0最多4并发/单进程分配器18%/启动余量5600MiB，容量不足自动等待、多轮排队，不减batch、不干预外部任务。

训练结束后控制器自动核验八组预算/冻结/恢复，再在GPU0队列并行完整8阶段val/test共16作业，最后F111 task0/2/7做关闭A/B/C/全关validation，输出交互与原图bootstrap。正式入口日志logs/three_module_stacking/implementation_20261022_01/formal_driver_01.log；组日志logs/three_module_stacking/formal_seed0_20261023_01；结果output/three_module_stacking/formal_seed0_20261023_01。283份静态预检/启动产物已同步本地并SHA全等，36份checkpoint仅存SHA未下载。原服务器主树153行/test-only5行及状态SHA保持；不合并main，不修改已锁定远程代码。本条及执行记录启动状态仅本地未提交更新。

## 2026-10-23：中断后继续，431项回归通过，GPU0八组短跑进行中

两端分支exp/three-module-stacking-20261022锁定0fb85f1，工作树均clean；431项完整回归通过，真实CLIP B64八组共同参数/RNG与初始输出逐值一致。GPU0空闲核验后，通过tmux stacking_smoke_seed0_20261023_01启动smoke_seed0_20261023_01，最多4并发、每组2task×2updates、仅validation。七组已complete，F111进行中，前两轮自动分队正常，0跳步；尚未启动正式8任务训练。

另调度stacking_recovery_probe_20261023_01：等待F111短跑训练完成，独立输出继承task0完整边界再训练task1，核对模型/optimizer/scheduler/scaler/RNG/sampler及val存档逐值一致，来源checkpoint不改。日志在logs/three_module_stacking/implementation_20261022_01。待全部短跑评价及恢复核验通过，再锁定执行版本启动正式GPU0队列；不会重复启动中断前作业。

2026-10-23预检进展：exp/three-module-stacking-20261022已Git-only创建远程独立工作树，d4a4238完整429项回归通过，真实CLIP B64八组共同参数/RNG和零初始化预测逐值相同，冻结视觉不变，初始化峰值742.1MiB。修复了旧空间分支绕过B缓存的组合调度问题；另补显存等待/多轮排满八组及失败未运行清单两项队列测试。下一步八组2task×2update smoke、validation并行重放、全恢复一致性后正式启动；目前正式未启动，不把短跑作为性能结果。GPU0限定和自动多轮排队持续有效，数据/超参不改。

## 2026-10-22：开始实现三模块叠加与GPU0自动队列

用户已授权开始，并补充显存不足可自动分轮确保八组均排队。新分支exp/three-module-stacking-20261022从1e28bfb创建，保留原分析文档。实现显式module_stacking协议、同一读取路径中A+C组合、原task_full摘要B缓存、aux.1统一监督、image_token的训练/评估/完整恢复支持；新增八组分阶段内存队列、严格训练/冻结/holdout核验及汇总/validation关闭归因。锁定seed0/8task/30epoch/B64，原图90/10，主LR.0125/模块LR.0004，joint BCE+aux.1，无ASL。

新增8项组合机制回归与真实CLIP八组初始化预检入口，当前仅本地语法和diff检查通过，尚未运行远程测试/短跑或正式实验。GPU0初始空闲约24GiB，拟最多4并发/单进程18%分配器/5600MiB余量，按实测峰值调整并发，资源不足自动等待且不减batch；训练之后自动排完整val/test和F111关闭归因。执行记录docs/three_module_stacking_execution_20261022.md。原主树153/test-only5行及状态SHA已记录，不触碰；下一步提交实验分支、Git-only创建远程独立worktree并验证。

## 2026-10-22：待实施的现有三模块叠加方案（本轮只规划）

详细方案docs/three_module_stacking_ablation_plan_20261022.md已完成。用户先验证既有Image-token Adapter＋摘要Attention＋人物位置三模块叠加，优先于上轮新Query结构建议。仅规划，不启动训练或只读GPU诊断。

后续若执行：锁定2^3八格F000/100/010/001/110/101/011/111，先单模块、再两两、最后三者；不能只对比F000与F111或因两两无收益删除格子。A shared索引1/b32/scale.1，B任务Selector Query索引5/内宽64/4头/scale.1，C原target几何12层共享。共同90/10原图manifest、aux.1 joint BCE、无ASL、seed0/8task/30epoch/B64、主LR.0125/模块LR.0004；当前八格重训配对。

实现前需明确支持目前被禁止的三模块组合、A支路softmax消费C偏置、B源缓存依赖A/C状态、B允许统一aux.1、全部模块optimizer/冻结和完整恢复。不要顺便切人物Query、增层、改变scale或改Frozen Forward。先保存完整恢复状态再评估；沿最近GPU0约束时以最重配置探针确定并发，不承诺八组全并行。

评价保留全部8阶段val/test、F111 task0/2/7 val去一模块和全关；报告组合对子组合增量及二/三阶交互，明确off非重训baseline、bootstrap非训练seed。方案入口尚未实现。

## 2026-10-22：两条正向候选的结合讨论，尚未授权实施

先保留既定只读整图位置组task0/2/7 validation开关诊断。不要直接叠加摘要Attention和位置bias，或用两轮绝对分数预期加法收益：前者全TRAIN/aux0/13950更新，后者90% fit/aux0.1/12660更新，且两者关闭新增项后仍保留收益。

后续可选研究方向需区分目标：摘要后Attention是跨视图表示传递，需保护Person私有证据和约束目标对应；Full位置摘要是人物条件证据选择，需让几何与人物内容形成有用对应；训练时使用、推理移除是独立训练策略假设，需从训练协议和匹配普通扰动控制重新验证，不能把当前off当独立训练成功。

若用户授权结合方案，优先研究在Full Patch Summarization中联合任务语义、Person内容和人物几何调整读取权重，保持原Full Value、Person/Face Task Forward及旧任务冻结；不默认继承旧Full→Person残差。先统一监督/数据/预算并做无条件、仅位置、仅内容、联合四组，在TRAIN原图holdout与validation验证推理净贡献及同图人物条件对应，再决定是否扩完整训练。当前仅方案讨论，未改模型或运行实验。

## 2026-10-22：目标人物空间摘要完整8任务分析结束

formal_gpu0_resume_seed0_20261022_01三组训练、六路val/test、位置归因及总batch全exit0，全部8任务/240epoch/12660更新/0跳步，恢复/旧参数/原图隔离核验通过。执行1e28bfb远程clean；原primary153/test-only5行及SHA保持。233份正式非checkpoint产物已同步本地并SHA一致，48份checkpoint/恢复SHA复核，未下载/修改权重。

完整固定人群val2397/test5368：匹配原方法最终42.1351/31.8525、阶段平均48.3591/35.3942；整图位置42.5486/32.1509、48.5989/35.9578；目标人物位置42.9717/32.3127、48.9116/35.7182。目标对匹配原方法val/test均8/8阶段为正，最终+.8367/+.4601；相对整图位置最终+.4231/+.1618，但test阶段平均−.2396，不能说真实人物位置稳定优于一般空间偏好。

target同checkpoint task0/2/7正常−关闭validation融合为+.00212/+.00279/−.00051，最终关闭42.9723仍保留对基线+.8372。373人/175图换框均真实改变Full，PFace及不可交换者固定；最终本人/换框融合41.3038/41.2804，仅+.0234，500次原图bootstrap区间[−.0164,+.1276]；Full39.3664/39.7758，差−.4094且区间跨0。标签不同369人分层也没有明确正确框优势。不能把框敏感当人物对应成功，也不能把关闭当重训原方法。

目标最终test Full/Person/可靠Face/融合30.6782/29.7310/28.3097/32.3127，相比基线+.1811/+.3834/+.5106/+.4601；性能改善保留在共同训练参数中更符合证据，尚不能确定为正则化或跨seed稳定收益。目标对原方法最终原图bootstrap val[+.0202,+1.4962]/test[+.0309,+.9295]，只条件固定seed0，非训练seed证明；对整图位置区间跨0。历史全TRAIN B0仅参考，本轮90% fit不能混为匹配控制。

独立复算276个指标最大逐类差4.41e−13；首次派生CPU sigmoid线程未匹配产生两处末位差异，按OMP1重新导出后归档位置干预逐类完全一致，原结果无改动。分析仅NumPy/CPU数学及读权重，无新模型前向、训练、梯度或额外test推理。

完整报告docs/target_anchored_summary_complete_results_20261022.md；分析JSON/脚本/manifest在output/target_anchored_summarization/analysis_20261022_complete_01。结论：保留性能候选与历史B0，当前人物条件推理净收益门槛未过，不直接加绑定loss、放大bias、扩MoE或合并main。建议下一步现有整图位置控制task0/2/7做只读关闭归因，区分一般空间先验/训练扰动；本轮未启动，后续若追训练过程收益需另立匹配扰动控制与validation复验。

三个上下文及执行/迁移记录同步更新，只有本地分析文档与ignored统计变化，不重跑模型单元测试、不提交推送。此前“运行中/等待完整结果/前三任务阶段分析”均由本条完整结论覆盖。


## 2026-10-22 14:56 CST：GPU0迁移已完成，三组正式接续健康

formal_gpu0_resume_seed0_20261022_01在远程/mnt/haoyuan/workspace/multi-lane-main-target-summary-gpu0-resume-20261022运行，HEAD1e28bfbb6be898df79d933352a4c5a13e6cfccc1、clean，分支fix/target-summary-gpu0-resume-20261022。baseline/absolute/target三个训练PID805547/805550/805553全部GPU0，均完成task4第6轮，每轮62更新/0跳步；GPU0总使用9908MiB。

三组task0–3全部120epoch/8760更新完整继承，只在新进程重做task3报告，没有重复训练更新；接续task4–7新增3900更新，最终仍12660。恢复后的task3模型与源完整边界逐tensor一致，17份继承文件/组共51份SHA保持，旧训练PID776820/776823/776826及旧控制器776747均退出，仅释放本批次GPU4/5/6占用，不触碰其他作业。原主树153行/test-only5行及状态SHA与开始一致。

421项全测与两轮GPU0真实恢复smoke通过，六份方法状态/后续RNG/采样/优化器等逐值一致，旧前缀来源兼容检查及六评估/位置归因均通过。210份静态验证产物已同步并SHA一致；正式launch、恢复manifest、暂停/迁移验证与启动健康记录同步本地。后续全部完整8task validation/test六进程和target task0/2/7位置归因自动在GPU0执行。

当前训练仍在进行，尚无最终8任务结果。后续查询应优先读取新worktree及formal_gpu0_resume_seed0_20261022_01；旧formal_seed0_20261022_01在task3训练结束处因用户迁移要求停止，其目录保留，不等待其完成标记。报告docs/target_summary_gpu0_migration_20261022.md。启动状态仅本地更新，远程代码锁定不改；不合并main。


正式GPU0接续控制器已启动并等待task3保存边界：运行HEAD1e28bfb，tmux spatial_summary_gpu0_formal_20261022，目标批次formal_gpu0_resume_seed0_20261022_01，入口日志logs/target_summary_gpu0_migration/implementation_20261022/formal_driver.log。源baseline/absolute/target最后核验到task3第28/29/28轮、零跳步，旧控制器仍暂停，未提前结束训练。两轮GPU0恢复smoke全链路通过，421项测试通过；无需用户手动接续。当前仍在等待真实迁移，不把排队状态写成GPU0已训练。


GPU0正式迁移门槛全部通过：421项默认全测；从task0恢复继续更新及从task1继承旧compact前缀的两轮三组真实smoke均完整exit0，六进程GPU0评估与位置归因通过，六份方法状态逐值对齐来源。task1结束的RNG/sampler/optimizer/scheduler/scaler/requires_grad全等。将源码锁定随后文档提交，调度formal_gpu0_resume_seed0_20261022_01等待task3三份完整快照；当下原任务尚在最后数轮，不能提前声称GPU0正式训练已开始。模型、超参、预算及测试协议不变，后续全部训练/评价GPU0。


GPU0恢复短跑已全链路完成：三组训练exit0、六路GPU0并行validation及位置归因exit0。六份task0/1模型状态与未中断来源逐值相同；三组task1恢复状态的Python/NumPy/Torch CPU/CUDA RNG、sampler generator、Adam/scheduler/scaler/参数组及requires_grad全部逐值一致。原420项全测通过；另补“恢复模型不等则禁止结束旧PID”的负向保护回归，准备421项全测。八路径资源探针B64三视图保留1252MiB，支持后续GPU0六评估。真实三组当前task3约21–22轮，边界暂停等待中，旧控制器已T停止；即将调度GPU0正式接续控制器等待task3快照，继承8760更新后只新增3900更新，不改变总12660预算。


GPU0迁移预检：Track-A模式416项测试通过；资源探针首次读取错误的face_manifest_root字段而在模型前向前退出，已按归档face_manifest.root核对，并让控制器沿用源launch原始Face路径；不改任何训练配置或源进程。准备完整默认测试发现与GPU0真实恢复smoke，失败探针未改权重或产生预测。


GPU0迁移实现预检更新：新增spatial_resume严格配置/来源/完成预算校验、原文件SHA复制、完整模型与requires_grad恢复、原随机数及sampler恢复；runner跳过已训练任务更新，只重做边界报告再激活下一任务。原任务按协议重置Adam，旧optimizer/scaler/scheduler快照保留不伪装连续使用。checkpoint来源按继承前缀与新执行Git分别核验，原模型结构/训练损失不改。新增GPU0接续控制器：三组16%显存分配上限/启动总余量14GB，恢复当前任务compact与源边界模型逐值核验后才结束旧已暂停PID，后续六路val/test及位置归因全部GPU0。已调度task3边界暂停器，旧控制器为T状态，训练尚在等待task3结束；准备4项新增回归、全测与三组真实恢复短跑（对比未中断权重逐值一致）。当前不声称迁移完成。


## 2026-10-22：用户要求继承进度迁到GPU0，准备任务边界暂停

新分支fix/target-summary-gpu0-resume-20261022，保留本地未提交分析/上下文；原运行源码b0d8263及服务器工作树不改。GPU0当前空闲约24GiB，原三组在task3约第5–6轮；当前仅支持任务结束前报告恢复快照，没有任意batch保存。为避免丢更新，计划先验证PID/工作目录身份，暂停旧总控制器，各组完成task3原子快照后立即SIGSTOP；再在新独立worktree从task3完整模型/RNG状态继续报告并训练task4–7。原任务每次重置Adam，所以不将旧任务优化器错误用于新任务；快照中的优化器/调度器/scaler全部保留。三组移到GPU0并行，8任务、seed0及val/test配置不变。

准备新增边界暂停器、严格配置/来源校验的恢复入口及GPU0接续控制器，先做恢复一致性测试和真实短跑后才释放旧进程；不覆盖原checkpoint/未提交内容、不修改main/test-only。当前尚未声称已经迁移或暂停完成。


## 2026-10-22 13:56 CST：核实实验尚未结束，完成前三任务只读阶段分析

用户要求查看结束结果；实查formal_seed0_20261022_01的三组进程仍存活，task0/1/2各90epoch、4560成功更新/0跳步完成，已进入task3第2轮；完成标记、三组正式完整val/test及位置归因尚未生成。源码仍锁定b0d8263/远程clean，没有重启、调参或额外GPU评估。

阶段reporting validation（人数2285/2375/2381，类别5/8/11）：baseline58.1525/57.8075/44.4069，absolute59.2844/57.9388/44.3322，target58.4303/58.3855/44.9024。目标相对基线三阶段平均+0.4505，但相对整图控制仅+0.0543；500次原图配对bootstrap各阶段区间均跨0。TRAIN留出target−baseline为+.4720/−.3055/−1.1915，最后仅79人。不能把这些当固定2397人的完整正式val，更没有最终test数值。

空间参数非零/梯度有限，训练期偏置与attention KL较小、无饱和；这些支持分支在学习，不证明正确人物绑定或最终推理净收益。9份compact CPU核验旧Selector/Prompt/head行/空间参数逐值冻结且SHA不变；同人旧类AP近持平，累计阶段mAP下降不能直接解读为遗忘。

27份已完成预测/划分源文件同步并SHA一致，18个NPZ逐类AP重算差0；无新模型前向/训练/梯度/test访问。阶段报告docs/target_anchored_summary_partial_results_20261022.md，ignored快照/脚本/JSON在output/target_anchored_summarization/analysis_20261022_partial_01。原主树153/test-only5行及状态SHA不变。三个上下文与执行记录同步更新；本轮仅文档和只读分析，不重跑模型测试、不提交推送、不改运行中的远程代码。

下一步仍是原控制器完成全部8任务、六路完整val/test及target task0/2/7空间关闭/同图换框，再同步最终产物判断人物对应及互补性。当前不根据中间指标停止某组、修改bias/LR或启用绑定损失。


启动健康核验（2026-10-22 12:42 CST）：三组均完成task0第1个epoch，各76次成功更新、0跳步、无异常栈/OOM；三个训练进程存活。首轮loss仅用于数值健康检查，不作性能判断。正式launch.json和startup_health_01.json已同步本地。

## 2026-10-22当前状态：正式运行中，无需重复启动

- [x] b0d8263两端Git-only同步、415项回归、真实CLIP检查及全链路smoke通过；155份静态产物同步并SHA一致。
- [x] formal_seed0_20261022_01三组在GPU4/5/6进入task0，tmux spatial_summary_formal_seed0_20261022；seed0、8任务、30epoch/task、B64、90/10原图留出、aux.1。
- [ ] 等待三组12660更新完成和配对/冻结/恢复审计通过；六路完整validation2397/test5368自动启动，target task0/2/7位置关闭/框交换归因随后自动运行。
- [ ] 完成后同步全部正式结果/日志并核验SHA；逐任务、分支、多人/可靠Face与位置干预联合分析，不能把预测改变等同于正确人物对应，不能把历史full-TRAIN基线当同监督/预算控制。

当前无最终指标，不根据中途val/test改变条件；只有正确性/资源故障才停止并保留产物。运行版本锁定b0d8263，后续本地文档状态更新不改变远程代码。原main/test-only及来源checkpoint保留，不合并main。

2026-10-22正式启动前检查全部通过：415项完整回归；真实CLIP零参数/历史task0/2/7逐值重放；smoke_seed0_20261022_02三组各4更新/0跳步、原图隔离/初始化和采样配对/旧参数与恢复核验；六评估进程及总batch exit0，正常预测与训练时存档逐值重放，框交换/关闭只改Full而Person/Face固定。33可交换者的Full在task0/1分别31/33、33/33人变化。smoke01与02的六份方法状态逐值相同，确认修复未改训练轨迹。实测训练峰值baseline2575.41、两空间组2576.23MiB，保留约2710–2714MiB。
准备formal_seed0_20261022_01，GPU4/5/6三组并行完整8任务，每组12660更新，训练后六split评估GPU1–6，每卡单进程/内存上限16%/余量门槛5600MiB；原图90/10留出及aux.1保持，无绑定loss。用户8任务要求覆盖先导性能晋级门槛，不因中途指标变更条件。当前正式待调度，尚无最终结果。


2026-10-22短跑_01：三组各2task×2updates均完成、0跳步；空间参数梯度有限非零。验证器用compact的weights_only读取完整恢复文件，因TorchVersion/RNG元数据被严格拒绝，批次exit1且未开始评估；已修复为仅读取本实验可信完整快照的显式加载并补回归，失败产物保留。顺便将无效/不可见框的目标覆盖诊断归零，预测及优化公式不变。拟smoke_02重新跑全链路，通过才正式启动。


2026-10-22真实预检通过：CLIP/真实val首batch64上absolute/target零参数与原方法逐值相同；默认路径对历史task0/2/7预测逐值重放，源SHA与冻结视觉不变；非零空间参数交换框只改变Full，Person/Face不变。真实检查峰值分配760.5MiB/保留1044MiB。补充配对Dataset仅换框且保留本人可见性/像素/标签回归，准备最新414项全测与三组2任务短跑；正式尚未启动。


2026-10-22实现验证更新：新增10项空间机制测试及远程413项完整回归通过；源码9f9f6de已通过Git-only同步新远程worktree。原主树和test-only保持原样。新增三组批次控制器、恢复/旧参数/原图holdout核验、只读空间关闭/框交换评估，以及真实CLIP零初始化与历史task0/2/7重放检查；正式尚未启动，下一步真实预检和2任务×2更新短跑。训练GPU4/5/6、评估GPU1–6拟每卡1进程，每进程Torch分配器上限16%（约3.9GiB），启动要求至少5600MiB余量；不足等待，不干预外部进程。


## 2026-10-22：开始实现目标人物空间摘要，用户要求完整8任务双split

新隔离分支codex/target-anchored-summary-20261022，起点e81f2f2；旧分析/计划及原主/test-only工作树保留。用户要求8task覆盖原三任务先导，保留原图90/10 fit/holdout；三组disabled/absolute/target，seed0、8×30ep、B64、12660更新，主LR.0125/空间LR.0004、aux.1，禁用MoE/ParaX/adapter/旧cross-view。未来训练后并行完整val2397/test5368，不启用绑定损失。
已实现未验证：未截断真实crop/flip框元数据、Full原汇聚logits几何偏置、60参数/task跨层共享、零初始化/无效回退、优化器/保存恢复/旧任务保护；新增机制测试。当前尚未正式运行，先Git-only远程完整单测/真实smoke。完整配置docs/target_anchored_summary_execution_20261022.md。


## 2026-10-22当前待讨论方案：任务语义与目标位置共同决定Full汇聚

用户本轮只要求研究计划，明确先不修改代码。计划已写docs/target_anchored_patch_summarization_plan_20261022.md；现无新GPU任务、实现代码或自动8task/test队列。

已完成：论文/旧负结果复盘；已有val预测核对Full同图人物完全相同（527多人图，519图标签有差异）；只读统计保存并检查源SHA；三个上下文更新。

建议第一步假设：Full Summarize的原semantic logits加目标–patch相对几何偏置（60参数/任务、跨12层共享），保持Frozen Forward/三条Task Forward/原MSA与D&R/末端融合和head。不是重启旧bbox query-delta或Person残差，也不是已经证明的三视图交互模块。

未来获执行指令后才做：先预检TRAIN当前任务fit框/可见性/配对量，独立实验分支实现；零参数与默认关闭等价/输入及RNG/几何与旧依赖保护smoke；P0原三视图/P1无目标位置同可见性/P2目标位置的seed0 task0–2、原图holdout+validation，统一aux.1监督和预算，不直接跑test。必须检验P2超过两控制且运行时有净贡献，正确框优于同图交换框；否则停止。

仅第一阶段机制通过后，才考虑同图当前类标签差异的绑定监督，保持同一Full像素及相同配对visibility、监督/成本匹配；不把所有人物当负样本，不使用未来类标签。再经先导重复性验证才决定完整8任务及锁定test；历史test已有多轮观察须如实报告。

---


## 2026-10-22最新判断：本轮已完成，当前无待运行归因作业

- [x] task_full/person_full全部8task validation/test正常开启及残差关闭。
- [x] 人物Query同图可见多人交换；固定几何配对，完整及可交换/Face可靠性子集。
- [x] 四进程/batch exit0，历史正常重放、本人Query控制、固定端点及源参数不变核验。
- [x] 100份正式日志/结果同步本地、SHA匹配；320个mAP复算、500次validation原图配对bootstrap；报告与三个上下文更新。

结果：人物Query关闭残差最终val+1.0083/test+.5753，8阶段全人群均改善；任务Query关闭差值+.0157/+.0468，未显示可靠净收益。正确Query在373/811可交换人物子集未优于同图另一人Query，预测真实改变不能当关联正确的证明。个别类别和早期test子集有收益，不泛化为所有样本或所有attention都无效。

当前建议（未启动新训练）：保留B0；停止本版人物摘要Query→Full摘要→Person残差路线；任务Query记录为共同训练可能带来的间接作用，不宣称推理融合成功。关闭结果非重训B0，其相对aux0优势区间跨0，不作为已验证新方法。若继续，应先提出真实目标人物关系证据/约束的新假设，而不是重复调scale/头数/冻结或旧Selector条件。

本轮报告docs/cross_view_attention_attribution_results_20261022.md，原始output/cross_view_attribution/formal_seed0_20261021_01，复算output/cross_view_attribution/analysis_20261022。分析文档待用户审阅，业务源码未改、没有新实验待批准或自动排队；远程checkpoint保留。此前“等待运行结束/拟执行task0/2/7归因”条目均为历史状态，已被本次完整8任务归因覆盖。

---

以下为历史记录，运行状态以顶部最新条目为准。

启动后静态产物54份（测试、失败与修正短跑、配对manifest、源SHA和健康快照）已同步本地，逐文件SHA256匹配。最新进度：任务Query validation已完成全部8边界；其test完成task0–2，人物Query validation完成task0–4、test完成task0–1，其余继续。此为运行进度，不是完整归因结论。

## 2026-10-22：完整8任务只读归因已启动，四进程运行正常

formal_seed0_20261021_01（保留启动run ID）在GPU0四进程并行：task_full/person_full各完整validation和test。评估执行提交ae56624，分支codex/cross-view-attention-attribution-20261021；原训练源码/16份checkpoint保留并逐份匹配历史SHA。无训练更新。
启动核验四进程均完成task0正常/关闭残差对照，正常预测与历史逐值相同；任务Query val已完成task0–3，人物Query val完成task0–1，两个test完成task0；仍在运行，不是最终结论。GPU0约6275MiB/100%。
固定90%几何规则：validation2397人中373人/175图可交换，test5368人中811人/372图。报告全人群和可交换/不可交换、可靠Face子集；Query替换只改变Q，不改Person底座/Full K/V/Face/head。
403项全测及CUDA补测、smoke_02双来源task0/2/7历史重放/同人Query/端点与不可交换样本不变核验通过。源checkpoint SHA、模型和冻结视觉参数不变；主树/test-only/原来源树状态SHA与开始一致。
下一步等全部8边界/两split及batch exit0，完整同步后比较on−off条件贡献与人物Query交换效果。本轮输出output/cross_view_attribution、日志logs/cross_view_attribution；配置见docs/cross_view_attention_attribution_protocol_20261021.md。不根据中途test更改条件或阈值。

2026-10-21归因启动前核验完成：修正后403项全测通过（1 CUDA RNG在CPU时跳过，另10项cross-view CUDA补测通过），smoke_seed0_20261021_02双来源task0/2/7各2batch/总批次exit0。历史正常预测、同人Query均逐值一致；Full/Face、不交换样本不变；原权重/源SHA不变。每边界33名可交换者均真实改变Person logits。
准备formal_seed0_20261021_01在GPU0四进程并行8task val/test；不重训。validation共373人/175原图符合预锁90%规则。初次smoke失败保留并已定位mask搬运类型修复。正式尚未产出完整结果。

2026-10-21归因实现验证：402项全量测试通过（1项CUDA RNG因CPU环境跳过），新增10项机制测试通过。smoke_01任务Query task0/2/7正常/关闭及历史重放成功；人物Query在交换条件因legacy mover把mask转float而被严格拒绝。已修复归因入口并补类型回归；失败产物保留，拟smoke_02重跑。完整validation固定90%几何规则得到373人/175原图。正式尚未启动。

## 2026-10-21：开始跨视图Attention只读归因，用户要求完整8task val/test

新隔离分支codex/cross-view-attention-attribution-20261021，基于74ece12；旧源码/分析文档及工作树保留。来源为formal_seed0_20261021_01任务Query/人物Query的8份seed0 checkpoint，不重新训练。
两来源on/off残差；人物Query另做同图可见多人仅换Q。90%人体框可见+中心可见、不同身体框，按sample ID循环，标签/分数不参与配对；保留本人Person底座/Full K/V/Face/head，不可交换者保持本人Query。
评估B64/workers0/AMP/TF32，完整val2397/test5368，GPU0四进程并行；先单元测试、历史正常预测重放与同人Query等价、task0/2/7两batch真实短跑，通过后启动正式8任务。
协议docs/cross_view_attention_attribution_protocol_20261021.md；logs/output使用独立cross_view_attribution目录，记录源SHA、完整/可交换子集指标与状态不变，原checkpoint不覆盖。当前实现中，未启动正式评估。

## 2026-10-21：人物条件跨视图注意力8任务完整结束，主方案未获提升

formal_seed0_20261021_01三臂及batch均exit0；每臂240epoch/13950成功更新/0跳步，完整val2397/test5368×8边界结束；恢复/compact/配对/旧依赖保护核验通过。训练代码8fef770，remote clean74ece12，未改训练源码。
匹配aux0基线val最终/平均42.7618/48.4159，test32.0492/35.4368；Person自注意力41.4584/46.9362、30.4556/33.2333；任务Selector读Full42.8976/48.8781、32.4993/35.9290；Person摘要读Full42.1003/47.8098、31.7499/35.1945。
任务Query是小幅正向候选，人物Query主假设未获支持。任务Query test阶段平均仍低于原B0的36.2781。validation最终提升.1357的原图分组bootstrap95%区间[-.6825,.9042]跨0，不能称稳定提升。
完整结果184份非checkpoint产物同步本地且SHA逐文件一致；只读离线统计/500次val原图分组bootstrap完成，没有新前向或训练。结果详见docs/person_conditioned_cross_view_results_20261021.md，分析JSON/脚本在output/person_conditioned_cross_view/analysis_20261021。

建议下一步（尚未执行、未获新启动指令）：已有任务Query/人物Query task0/2/7 checkpoint的validation残差关闭对照；同图多人且双方可见时仅交换Person Query，检验人物关联。若任务Query确有净贡献，再做B0 aux.1和任务Query aux.1的监督匹配控制，先训练内原图holdout/小任务预算；若无条件收益则结束本版结构，不自动扩seed/搜索层数。

## 2026-10-21：人物条件跨视图注意力完整8任务已正式运行

formal_seed0_20261021_01已在GPU0启动person_self/task_full/person_full三组并行，tmux=cross_view_full8_seed0_20261021。训练源码8fef770，分支codex/person-conditioned-cross-view-20261021，两端Git-only同步；后续仅更新文档。
启动健康快照：三组均完成task0前4epoch，每epoch84成功更新、0跳步，GPU0约10014MiB/100%利用率。48项关键配置与历史fullTRAIN aux0基线逐项匹配；基线复用源完整8任务/13950更新/全人群val-test已核验，默认关闭路径93预测数组完全一致。
正式每臂8×30epoch/B64、主LR.0125/模块LR.0004、scale.1、4头/宽64、索引5、每任务196608新增参数；每边界先恢复状态再报告，训练及核验后自动并行完整validation2397/test5368人。MoE/ParaX/image-token adapter关闭。
391项全量测试、修正后10项机制/CUDA测试、三臂两任务真实短跑和六评估并行/重放已通过。远程主工作树和test-only的状态SHA均与开始时相同。
下一步等待完整训练/验证/测试及批次exit0，同步产物后比较三个控制与基线；目前没有最终精度结论，不依据中途loss改超参。日志/结果位于logs和output/person_conditioned_cross_view；执行配置见docs/person_conditioned_cross_view_execution_20261021.md。

2026-10-21正式启动前核验完成：391项全量测试+修正后10项机制/CUDA检查通过；基线task0/2/7共93预测数组完全一致；真实CLIP三模式零输出与基线逐位相同。
smoke_seed0_20261021_01三臂/批次exit0，每臂4更新0跳步、恢复与checkpoint重放/初始化采样配对/旧参数保护通过；六评估并行完成，单进程峰值2593.64MiB、观察总峰值9904MiB。
固定checkpoint Person query显著随样本变化，task query仅约9e-5数值差异；训练累计Q变化不可当图片条件证据。旧参数精确不变，AMP跨任务批形状旧logits差异约.0012–.0022，避免逐位不变承诺。
准备正式formal_seed0_20261021_01，GPU0三臂并行，完整8task×30epoch及自动并行val/test，匹配aux0基线只读复用。正式未产出完整结果。

2026-10-21实现核验：新9项机制测试及391项全量测试通过。检查新增初始化发现torch.manual_seed会同时重置CUDA RNG，已改成仅CPU默认generator初始化并增加CUDA RNG测试；待远程验证该修正。旧默认关闭路径task0/2/7全validation重放进行中；尚未启动正式训练。

## 2026-10-21：开始实现人物条件摘要注意力，按用户要求完整8任务

新隔离分支codex/person-conditioned-cross-view-20261021，基于83f6282；原主工作树和test-only保留。
新增第6层Summarize后Full→Person注意力，保留三路Task Forward；person_self/task_full/person_full同容量控制。
seed0、fullTRAIN、8×30epoch、B64、主LR.0125/新增LR.0004、aux0、每任务完整冻结；关闭MoE/ParaX/image-token adapter。
用户本次8task val/test要求覆盖此前3task/holdout建议；不通过test挑参数。GPU5/6外部占用，GPU0空闲，拟三臂并行。
匹配baseline_noaux优先只读核验复用；新旧默认路径预测需验证。当前仅实现中，尚未启动正式训练。
执行协议docs/person_conditioned_cross_view_execution_20261021.md，下一步远程单元测试、并行真实短跑、通过后正式队列。

## 2026-10-21：用户要求Full→Person摘要交互详细计划，暂不改代码

新增docs/person_conditioned_cross_view_feasibility_plan_20261021.md；仅文档，不改业务代码/配置/脚本、不启动训练或测试。
MulT方向澄清：Person提供Q、Full提供K/V，信息方向Full→Person。保留三路任务状态，索引[5]一次摘要交互，原MSA/Drop & Replace/FFN/末端融合及head保留。
建议每任务独立四头宽64、固定scale.1、零输出初始化，无新FFN；196608参数/任务、8任务1572864参数。
核心四组：aux0匹配B0、Person自注意力、任务Selector Query读Full、Person Query读Full；后三组同参数量。
先导拟seed0 task0–2×30ep/B64/主LR.0125/新增LR.0004、相同90/10原图holdout/4560更新，仅validation。
完整说明Full摘要压缩限制、Full非纯背景、共享梯度仍可改当前Full、Person增强分支非单路、loss语义与旧依赖完整冻结。
只建议预先锁定晋级条件，尚未执行/获准实施；不将普通attention或冻结机制本身宣称创新，不复用历史test调参。
三个上下文已同步，业务源码diff保持空；无需模型测试。


## 2026-10-21：按模块筛选三视图融合论文，仅研究建议

核对MulT(ACL2019)定向crossmodal attention、MBT(NeurIPS2021)持久融合tokens、CCIM(CVPR2023)背景原型、
EmotiCon(CVPR2020)场景/邻近交互、MISA(MM2020)共享私有表示，原论文方法与本项目迁移假设分开记录。
当前retained的临时三组摘要在Drop & Replace后丢弃，没有持久三视图状态/多融合token；与MBT不同，不能据此断言失败根因。
若优先融合模块创新，建议从人物条件定向交互或受限持久token中只选一个主机制；CCIM须先有背景偏置证据，MISA为备选。
已有Face硬mask不算新贡献；不原样重启历史query-delta、后置残差、私有参数或梯度修正路线。
文档docs/three_view_fusion_module_literature_20261021.md含来源、接入点、限制和最小消融；未启动实验或改业务代码，无需模型测试。


## 2026-10-21：三任务先导已完整结束，暂不扩大；后续建议改为证据层诊断

pilot_seed0_20261021_01三臂和批次exit0，各90epoch/4560成功更新/0跳步，三份checkpoint/恢复与跨臂配对核验通过。
完整validation最终/三阶段均值：三路aux0 44.3003/52.0944，固定摘要44.6137/52.5977，三组摘要43.8444/52.1276。
三组摘要未同时胜过匹配控制；TRAIN holdout也未一致改善（task2仅79人），不能用该局部提升晋级8task/test。
78份正式非checkpoint产物已同步本地并逐文件SHA匹配；没有新增模型前向、训练或test。
用户询问后续方向；复盘排除原样重启类别权重OOF、Face表情替换/后置残差、排序蒸馏、框条件Selector及MoE超参路线。
优先候选：在原三视图框架中审计并补充目标人物姿态/交互关系证据；备选先诊断背景依赖，再考虑CCIM启发的偏置控制。
这些均是未验证假设，未获新执行指令、未启动新实验；保留B0。现有分支仅更新分析/上下文，不需模型测试。


2026-10-21启动健康核验：两组task0已完成前6epoch，每epoch76成功更新、0跳步；配置逐项符合预锁。
服务器pilot源码仍clean 22d1ae0，fixed在GPU5后续队列；健康快照已同步并SHA匹配，正式尚无最终结果。

## 2026-10-21：三组摘要无专家正式task0–2先导已启动

379项全测通过；smoke_seed0_20261020_01三臂各2task×2更新、全部0跳步，三个训练/重放及批次exit0，
完整可恢复状态、旧参数与CLIP保护、原图holdout跨任务无泄漏、初始化/采样配对、checkpoint重放均通过。
正式run_id=pilot_seed0_20261021_01，tmux=ml_retained_summary_seed0_20261021，源22d1ae0c70f16d45ce0f41c830dc9869711d3270。
远程/mnt/haoyuan/workspace/multi-lane-main-retained-view-summary-pilot-20261020，clean detached运行源；本地分支codex/retained-view-summaries-20261020。
GPU5 baseline_noaux→fixed，GPU6 retained；当前两组task0开始，formal尚无最终结果；每臂3task×30epoch/B64/Adam LR.0125/WD0/cosine/aux0，
TRAIN原图90/10 holdout，fit4854/3972/782、每epoch76/63/13步、总4560更新；只holdout+完整validation，不自动test或8task。
运行记录logs/retained_view_summaries/preflight/launch_pilot_seed0_20261021_01.json，日志/结果retained_view_summaries/<run>。
归因44文件、smoke/预检87文件（含正式启动健康快照）已同步本地并SHA匹配；模型运行与历史checkpoint均保留。
完成后同步新formal日志/分数，比较监督和holdout匹配的两控制，分别报告三阶段完整val与逐任务holdout，不能用train/seed_summary筛选人群混算。


## 2026-10-21：联合残差三边界只读归因完成；无专家先导真实smoke中

来源3ac8fbf，GPU5/6两臂正式均exit0，原动态预测三边界全部逐数组重放相同，六源checkpoint SHA不变。
Dense on/off：task0 49.9872/38.7399，task2 40.3464/32.6391，task7 38.5627/36.0553。
MoE on/off：48.3830/38.7092，39.8763/35.4139，38.6142/38.3423；最后差−.2719，500原图bootstrap95%[-1.3254,.6584]。
MoE最后类别所属task0关闭−9.6350、task5+4.6942、task6+8.0022，存在抵消，不可声称每条路径无依赖。
条件依赖不等于相对重训控制有净收益；结果支持先移除自由改写、保留三组证据的结构验证。
正式分数/日志已同步本地；分析analysis.json及报告docs/joint_residual_attribution_results_20261021.md。
先导代码22d1ae0的379项全测通过，远程独立pilot worktree clean detached该commit，不改运行归因源树。
smoke_seed0_20261020_01已在GPU5/6启动三臂2task×2更新、val-only，当前初始更新有限且0跳步，尚待完整重放/跨臂核验。
正式先导尚未启动；预算tasks0–2 30epoch，fit数4854/3972/782，每epoch76/63/13更新，共4560更新/臂。


379项测试在e57b3bb全部通过；复查补正旧allow_smoke单边界入口的checkpoint范围，新增同项测试覆盖。
运行中的3ac8fbf归因未更新；先导仍须新版本全测及真实smoke后启动。

2026-10-20执行更新：375项旧测试通过。3ac8fbf只读task0 smoke两臂均exit0，on与历史预测逐数组相同；off明显下降。
正式task0/2/7已在GPU5/6启动，服务器运行3ac8fbf保持不更新。
本地新增retained三组摘要、softmax前Key/Value有效性mask、无新增融合参数；无效Face、全Full退化、梯度、旧任务和重载测试待远程执行。
先导锁定三臂baseline_noaux/fixed/retained重新训练、seed0 tasks0–2各30epoch/B64/Adam LR.0125/WD0/cosine/aux0，
共同90/10 TRAIN原图holdout，val-only；GPU5前两臂顺序，GPU6候选。需完整测试和三臂2task×2更新真实smoke通过才启动。
所有新代码通过Git-only独立worktree，源和用户脏树保留；协议已补齐。


## 2026-10-20：联合残差归因与三组摘要保留候选

用户已授权开始执行。新分支codex/retained-view-summaries-20261020，从f057e71创建，旧脏树保留。
先dense/moe已有seed0 checkpoint的task0/2/7完整validation2397人，比较原预测与全部任务融合残差scale=0。
GPU5=dense、GPU6=moe，B64、AMP/TF32、workers0，无训练无test，源SHA保护；先task0 smoke逐数组重放，再正式。
输出output/joint_residual_attribution，日志logs同名。off仍保留共同训练的Selector/Prompt/head，不能当重训fixed或B0。
候选保留三组各10个摘要，由同一个Task CLS注意力读取；无效Face在softmax前mask，原Drop & Replace/FFN保留。
无Expert/Router/新增融合参数；先task0–2、TRAIN原图holdout与监督匹配控制，不自动8task/test。
当前尚未执行GPU实验，协议docs/retained_view_summaries_plan_20261020.md。


## 2026-10-20：联合摘要四组完整结束，结果已同步并完成只读分析

正式formal_seed0_20261020_01、运行f057e71，GPU5/6批次和四组train/val/test全部exit0；
每组240epochs/13950成功更新/0跳步，8compact+8完整恢复及旧模块冻结/CLIP保护核验均通过。
184份正式与历史B0产物同步并SHA匹配；80份全人群预测mAP复算差严格0，40源checkpoint SHA分析前后不变。
原服务器主树153项/test-only5项状态SHA未变，服务器源码仍clean f057e71；没有额外训练或模型前向。
统一full_val/full_test全人群（2397人1705原图／5368人3682原图），不能用train/seed_summary的筛选人群均值。
最终val/val均值/test/test均值：B0 42.1912/48.4382/32.0610/36.2781；
noaux 42.7618/48.4159/32.0492/35.4368；fixed 42.3356/48.7787/30.9878/34.8224；
dense 38.5627/44.4665/28.4476/31.6720；moe 38.6142/43.9184/28.4245/31.8450。
同监督fixed最终test比noaux−1.0614，原图配对bootstrap500 CI[-1.7843,-.4502]；
dense/moe比fixed−2.5402/−2.5633，CI[-3.2631,-1.8455]/[-3.2253,-1.9054]；
MoE−dense−.0231 CI跨0。均只固定seed0样本不确定性，不代表训练seed稳定性。
Dense/MoE在8/8任务训练BCE更低而8/8阶段val/test更低，task0已退化；冻结保住旧表示，但没有保证新表示泛化。
联合残差/固定摘要范数比val中位.1184/.0840，96位置最大平均.4211/.3252，分支非零。
固定checkpoint条件于相同Face可靠性，96/96位置仍有门控内容变化；可靠组95/96、不可靠组96/96位置Top2集合确有变化。
不能再把主因默认归结为无动态路由、分支未参与或旧共享Expert漂移；具体信息损失/改写过拟合尚待干预归因。
fixed训练峰值1804.20MiB vs同监督2575.68，约省29.95%；dense/moe总方法参数10.21M/23.43M。
报告docs/joint_view_summary_results_analysis_20261020.md；离线JSON/CSV/两张图及复算脚本
output/joint_view_summary/analysis_seed0_20261020_01/，源保护source_unchanged_after_analysis.json。
下一步仅建议：现有task0/2/7 validation关闭联合残差、在fixed做Full/FP/FA/三路输入条件诊断；尚未执行。
如继续训练，优先单CLS读取三组分开的摘要、Face在attention softmax前mask、不先加Expert；先task0–2的TRAIN原图holdout/val。
当前保留B0，不扩大Top-k/专家池/冻结数、不扩seed、不合并main。源码不改，仅本地追加文档和离线分析产物。


## 2026-10-20 18:22：联合摘要正式四组seed0八任务队列已启动

运行提交f057e7125787cf83dad1c7a34b31ff067d3a9a30，本地/服务器/upstream启动时一致，服务器clean。
正式run_id=formal_seed0_20261020_01，tmux=ml_joint_summary_seed0_20261020。
GPU5正在baseline_noaux（3路末端融合、aux0），随后dense；GPU6正在fixed（联合摘要、单路径），随后moe。
当前两组task0已完成前3epoch，每epoch84成功更新、均0跳步；max_tasks8/每任务30epoch/seed0/B64/
主LR.0125/融合LR.0004/aux0配置已逐项核对。后两组为自动队列，尚未开始正式训练。
每组训练、8checkpoint与恢复/冻结核验完成后，在该GPU并行完整validation2397与test5368、workers0。
原B0复用历史，不用test调参。正式尚无最终mAP，不能把机制smoke作为精度结论。
151份预检/smoke日志及分数已同步本地并逐文件SHA匹配，另3份正式启动快照/配置SHA匹配。
记录logs/joint_view_summary/formal_seed0_20261020_01/launch_snapshot.json，原主树/test-only状态SHA未变。
本地仅追加PROJECT_CONTEXT/WORK_LOG/TODO_NEXT/协议的运行状态；源码HEAD保持f057e71，运行服务器不更新。
后续等待四组完整完成，同步正式logs/output、核对8阶段完整性、对比监督匹配基线与原B0，先不扩seed或合并main。


## 2026-10-20：联合摘要四组预检全部通过，准备正式seed0八任务

375项Track-A全测通过；8项新机制测试覆盖单路径、三路读取、无效Face、旧任务保护、重载、RNG、
硬调用与零权重区分及AMP。四组均在clean 9055ebc完成2task×2成功更新/0跳步、完整val及seen-positive
val重放，mAP重放差严格0；baseline_noaux/fixed/dense来自smoke02，MoE来自smoke03，均同一source HEAD。
原失败smoke01（AMP）/smoke02队列guard exit1保留；缺失MoE在GPU6单独补齐，没有覆盖原状态。
smoke训练峰值MiB：2575.65/1805.83/1948.04/2145.92，均非正式8task资源结论。
MoE固定checkpoint的24个任务层gate均存在样本差异，仅证明机制活跃，不能宣称精度收益。
聚合审计logs/joint_view_summary/preflight/smoke_verified.json，8个checkpoint前后SHA相同，旧主树和
测试树状态SHA未变。启动器延迟利用率/外部占用两项模拟检查通过，bf0ad7d只改guard与文档。
准备formal_seed0_20261020_01：GPU5 baseline_noaux→dense，GPU6 fixed→moe，四组独立8task×30epochs、
seed0/B64/主LR.0125/融合LR.0004/aux0；每组训练及8份checkpoint核验完成后，在同GPU并行完整val/test，
workers0，禁止以test筛选。原B0历史结果复用。正式尚未启动；协议docs/joint_view_summary_experiment_20261020.md。


2026-10-20 smoke02：fixed双validation并发/精确重放全部通过；切换下组时GPU短暂残留利用率读数，
旧单次idle guard误拒绝MoE启动（未加载/训练MoE），实际GPU6随后确认全空。启动器改为最多60秒等待
前作业context/利用率窗口清空，不终止任何外部进程。模型核心2faeb52及375项测试不变，控制组已通过
不因启动器检查重复训练；剩余MoE须独立新run ID补齐双validation短跑后方能启动正式。


2026-10-20修复预检：2faeb52在独立只读测试worktree完成375项Track-A测试（15.746s）全部通过，
包括8项联合摘要测试。smoke01已结束：baseline_noaux/fixed/dense来源、保护、恢复及val重放全部通过，
MoE首次AMP累加dtype错误导致该组exit1、批次exit1；失败产物全部保留。新dtype修复及gate日志已完成。
准备在同步后的clean版本运行smoke_seed0_20261020_02：四组2任务各2步，双validation并发、无test；
只有全部通过才正式8task。旧服务器主树/test-only仍保留，不能把短跑指标当精度证据。


2026-10-20 smoke01预检拦截：fixed两任务训练/来源审计/val重放全部通过，MoE在首次AMP稀疏dispatch时
出现index_add Float/Half不一致，未进入正式训练。已修复source按累加张量dtype转换，新增CPU混合精度
反向测试；失败日志与目录保留。需完整375项测试及新版四组smoke02通过，禁止据失败短跑报告效果。


2026-10-20真实smoke01：GPU5 baseline_noaux/GPU6 fixed两组已完成2task×2步，均0跳步，
初始化/采样/CLIP保护/完整恢复审计通过，正在只读val重放；dense/moe按队列继续。
本地补齐gate方差/熵及可靠Face分组质量统计，硬负载改为按实际dispatch计数，覆盖soft权重下溢为0；
新增极端门控机制测试，运行中的服务器7b20c70不变。补齐新结构元数据及双validation并发smoke检查，待核验。


2026-10-20预检：d618fb8已Git-only同步服务器新worktree，373项Track-A测试全部通过（16.580s），包括6项新机制测试。
服务器主树153项/test-only5项脏状态SHA与预检前一致，未改动；空闲GPU5/6各24GiB，磁盘537GiB。
准备启动四组2task×2更新真实smoke，seed0、GPU5/6、validation-only；正式仍待smoke与重放核验。


## 2026-10-20：用户授权三视图联合摘要＋单Task Forward完整对照

新分支codex/joint-view-summary-20261020从635f729创建，旧脏树均保留。用户已确认GPU5/6。
原B0复用历史；新增baseline_noaux/fixed/dense/moe四组，各seed0、8task×30epochs、B64、主LR.0125、
融合LR.0004、aux0。每层同一组Selectors分别读取三路冻结图像证据，联合摘要进入单任务注意力/FFN；
分类头位置保留；无跨任务可变共享模块。dense宽32，MoE每任务每层4名宽16专家softTop2，完整任务融合冻结。
新TaskJointSummaryBank/model/runner/evaluator/verification/双GPU launcher及机制测试已编写。
需远程全测、四组各2任务×2步真实smoke及checkpoint重放、SHA保护通过，才能正式启动；尚未启动。
完整配置docs/joint_view_summary_experiment_20261020.md；日志/产物均joint_view_summary。


## 2026-10-19：用户授权Frozen Forward MoE对比实验

新分支codex/moe-frozen-forward-20261019从ce70645创建，保留旧脏树。仅GPU0，seed0、8task
validation/test，22专家Top3/按训练累计gate mass冻3，索引[1]；对照旧L1_K3_F3不重训。
MoE从Task Forward移到Frozen block MSA后、与FFN并联，写回197图像tokens；Router读图像CLS。
任务×视图Router仍24套，因此推理各lane必须有对应的图像流，逐lane计算限制显存；骨干参数
保持冻结但图像梯度不可detach。[1]首先影响[2]Summarize，不引入DDAS/ParaX，不保证旧函数不变。
核心接入/CLI/eval/冻结核验/恢复launcher已实现；新测试与真实smoke待执行，尚未正式启动。
完整配置docs/moe_frozen_forward_experiment_20261019.md；所有新日志/输出使用moe_frozen_forward。
当前GPU0空闲，其他卡有既有任务；服务器主树/测试树有用户改动，保持不动，只用新独立worktree。

## 2026-10-18：用户授权Late只读路由对照与Middle完整重跑

新分支exp/moe-late-route-middle-recovery-20261018从f17a256创建，继承5完整/1缺失
分析及文档，保留旧脏工作树。GPU0计划Middle[5]/22expert/Top2/冻2，seed0、8task×30ep，
原WAF预算不变，训练workers2、内部/外部评估0；验证前原子保存任务边界完整恢复信息。
GPU1原Late task0/2/7，先以固定TRAIN原图分组4批校准每lane/view固定Top2和归一权重，
再锁定完整val2397上dynamic/fixed/off三条件，workers0、B64、500原图配对bootstrap，
无训练或test；动态重放逐值验证、源/模型SHA保护。全部TRAIN表在val读取前锁定。
先全测及双方真实smoke；正式尚未启动。服务器107工作树只主树153项/test-only5项脏，
GPU0/1均空闲，/dev/shm=4GiB空；只使用GPU0/1，不动其他卡或已有作业。
详细配置与预锁门槛docs/moe_late_route_middle_recovery_20261018.md。

## 2026-10-18：六臂消融完整性核验、val补齐与结果分析完成（5完整/1缺失）

原3410546批次failed，不能称六臂全部完成。Middle[5]/k2/冻2虽完成task7 cycle30，
在内嵌val Bus error后未保存最后checkpoint/summary，只有task0–6的7份checkpoint；
其他5臂均240epochs、13950成功更新、0跳步，完整test已存在。
原双split并行漏算共享内存：12loader预取图像约5.17GiB>容器/dev/shm 4GiB；
两臂外置val同样失败。f17a256独立repair worktree新增评估--workers=0，351项全测
通过（15.625s），GPU0/1各1进程只读补L1_K2_F2/L1_K3_F3完整8阶段val，均exit0。
7份重叠NPZ全部数组逐值相同；225原产物+45补评估/预检/绘图产物及清单同步并核验，
B0另39份新SHA匹配。47个源checkpoint SHA前后不变，原运行启动核验clean，补评估结束Git记录clean，
没有新训练/扩seed/test调参/合并main，Middle未重训，原failed状态与目录保留。

最终val/test（val均值/test均值）：B0 42.1912/32.0610（48.4382/36.2781）；
Early2/2 42.2952/31.6815（47.9071/35.0797）；Late2/2 42.2600/31.6347
（48.3586/34.7766）；Early2/3 41.8437/31.7328（47.7562/34.9600）；
Early3/2 42.0640/31.5780（48.2216/35.1649）；Early3/3 42.0622/31.8150
（48.1568/35.1551）。后者虽然MoE最终test数值最高，val低原2/2，不据test选它。
五臂test全部8阶段点估计低B0；500次原图配对bootstrap最终差CI跨0、阶段平均差CI
全部为负，仅seed0样本不确定性。保留B0，不晋级3/3、不扩容量/温度或seed。

冻结3改善test forgetting：k2 .3918→.0829，k3 .3417→.0446；最终冻结22/21名，
训练task7时仍各有3名可更新（不能称task7已无容量）。低遗忘不等于更高最终旧类AP；
3/3相对2/2的+.1335最终test主要来自最后3类+1.3475，旧23类−.0249。
Late只有1/24任务视图集合固定（Early23/24），Full/Person也有15/16动态（Early0/16），
但test全程无收益；不能再用Early的弱图片路由解释全部层。Top3第三权重中位约.27%/.31%，
仅6/24≥1%，集合固定不代表gate权重恒定，Face路由汇总含占位，精度单列可靠Face。
首任务test各MoE已低B0，不能归因后续遗忘；增加冻结名额没修复前置泛化差距。

完整报告docs/moe_layer_routing_ablation_results_20261018.md；离线JSON/CSV/复算脚本
output/moe_ablation_repair/analysis_20261018/；补评估目录val_workers0_20261018_01。
建议研究上先用Late现checkpoint在task0/2/7做只读动态vsTRAIN确定固定Top2vs关闭
MoE残差对照，validation看路由净收益再决定是否训练；这些新干预尚未执行。
工程上评估workers0/共享内存预算、验证前保存完整可恢复状态；要完成三层消融，
Middle需唯一配置独立完整重跑，当前未启动，不能用缺失结果宣称所有层均无效。

## 2026-10-18：六臂结束核验发现共享内存故障，先补只读validation

原batch实际failed。5臂训练完整（13950更新/0跳步、8 checkpoint）且test完整；
L1_K2_F2/L1_K3_F3的val Bus error失败。Middle L5_K2_F2虽完成task7 30epochs及
冻结，在内嵌val失败，未保存task7/最终summary，仅7 checkpoint，不能报完整结果。
4GiB /dev/shm小于正式12评估进程预取图像估算约5.17GiB；原smoke每臂串行两类
val未覆盖双split峰值。属执行流程故障，不能计为模型效果，保留全部失败产物。
新独立分支fix/moe-ablation-eval-workers-20261018补--workers=0评估覆盖与运行来源
记录；计划GPU0/1各1进程补两个已训练模型val，不训练/不动原checkpoint。
原主树/test-only与旧实验工作树保留；协议docs/moe_ablation_results_repair_20261018.md。
随后同步结果并分析5个完整模型、B0和有效层比较，Middle完整结果明确缺失。

## 2026-10-18：六臂正式运行健康检查与本地快照

已逐臂确认task0完成3–4个epochs、每epoch84次成功更新、所有skipped=0，未见错误。
GPU0/1各3训练进程、各约10GB显存；运行worktree clean、HEAD/upstream=3410546。
启动快照分开记录容器PID与nvidia-smi宿主机PID，防止命名空间混淆。
6份正式config、批次映射与运行快照共8文件及SHA清单同步本地并核验一致。
本地快照位于output/moe_layer_routing_ablation/formal_seed0_20261018_01/，仅启动
元数据，不代表完整正式结果；val/test均待各自训练与审计完成后自动执行。
原主树153项/test-only5项未提交改动数量未变；未操作其他GPU作业。
PROJECT_CONTEXT/WORK_LOG/TODO_NEXT与实验协议均已同步本次状态，源码运行提交不动。

## 2026-10-18：两组消融六臂正式GPU0/1已启动

正式run_id=formal_seed0_20261018_01，tmux=ml_moe_ablation_seed0_20261018，
代码34105463a64e642d74f8f154ddbe3455734ae5c9；启动时本地/服务器/upstream同HEAD，
服务器工作树clean。GPU0并行L1_K2_F2、L5_K2_F2、L11_K2_F2；GPU1并行
L1_K2_F3、L1_K3_F2、L1_K3_F3。每臂8task×30epochs、seed0、batch64，
训练完成并核验8个checkpoint后在其GPU上并行完整val/test。两组共享新训练的
L1_K2_F2对照，总计6次训练，不是7次或12个全组合；不据smoke指标筛选配置。

349项Track-A测试通过；真实六臂smoke01全部exit0，均2任务各2成功更新、0跳步，
B0初始化/采样配对、加权冻结重算、可靠Face计数、float64权重守恒、已冻结Expert/
旧Router/CLIP保护、checkpoint验证集逐阶段重放全部通过。共同对照两个checkpoint
各152个状态张量与旧WAF smoke逐值相同；同k不同f的task0各145个浮点模型张量
逐值相同，证实冻结差异从任务边界开始生效。六臂参数分配相同2992000。
147份smoke/preflight日志与结果及SHA清单同步本地，逐文件SHA256相同；checkpoint
留服务器。正式六进程已加载正确配置，首个配置task0 cycle1=84步、0跳步；尚无
正式八任务结果，不能报告精度收益。正式日志/输出根均为moe_layer_routing_ablation。
运行协议docs/moe_layer_routing_ablation_20261018.md。本地仅追加状态文档；运行
服务器固定3410546，不在长任务中更新源码；保留原主树及test-only用户未提交改动。
实验结束后同步正式logs/output、核验完整性，再作两组对照分析，不自动扩seed/合并main。

## 2026-10-18：六臂消融服务器全测通过，开始真实smoke

运行代码34105463a64e642d74f8f154ddbe3455734ae5c9已Git-only同步新服务器工作树，
与本地/upstream一致且clean。349项Track-A测试全部通过（18.785s）；本地启动器
3项通过。服务器内存可用133GiB、磁盘539GiB，GPU0/1空闲，各约24GiB。
六臂真实smoke预设run_id=smoke_seed0_20261018_01、各2任务×2更新，仅validation，
tmux=ml_moe_ablation_smoke_20261018；需全部机制／重放检查及旧WAF默认兼容后
才启动正式八任务。协议docs/moe_layer_routing_ablation_20261018.md。正式尚未启动。

## 2026-10-18：用户授权插入层与Top-k／冻结数量两组完整消融

在独立分支codex/moe-layer-routing-ablation-20261018从41567eb实现；保留此前回滚
与WAF分析文档，原停止建议作为历史结果，不覆盖用户此次明确实验授权。
预设六臂：GPU0为L1/L5/L11、K2/F2；GPU1为L1的K2/F3、K3/F2、K3/F3。
共同L1/K2/F2在新代码重新训练一次供两组共享；沿用gate_mass加权冻结、22专家、
宽度64、noisy Top-k、30epochs/task、8task、seed0、B0配对和原WAF全部其他配置。
新加独立--moe-freeze-count、f3 checkpoint恢复／审计、推理门控权重日志、六臂启动器；
累计冻结仍可重选、不补足，并不保证旧函数完整保护。3项启动器本地测试通过；
计划服务器全测与六臂两任务smoke通过后，再在GPU0/1并行正式训练和val/test。
完整协议docs/moe_layer_routing_ablation_20261018.md。当前未启动远程实验。
已检查服务器105工作树，仅原主树153项/test-only5项脏，保留；GPU0/1均空闲。

## 2026-10-18：冻结统计含义澄清（无新实验）

再次核对现有task0审计：AF/WAF首次冻结前模型相同，Expert6/9/13硬激活分别
164710/144382/96664；gate mass分别150217.5084/76701.9608/88206.2159，故AF选
[6,9]、WAF选[6,13]均符合各自统计规则。Full专家9全训练Full mass75842.4733/
160590≈47.23%，不同于训练结束只读推理平均gate90.47%；历史累计/训练噪声及
视图合并造成使用量与最终依赖的差别。该解释基于已有产物，无新前向/训练或代码修改。

## 2026-10-18：旧Expert回滚归因完成，未通过保护设计门槛

GPU1只读正式run formal_seed0_20261018_01（41567eb）exit0/complete；AF/WAF各
边界1/7，4批TRAIN各64人、全val2397，五条件，前后相同旧lane数，FP32主归因及
TRAIN AMP/TF32复核。338项全测、7项最终针对测试、真实smoke04通过。正式88份
日志/JSON/NPZ已同步本地并与服务器新SHA逐项相同；8个源checkpoint前后SHA相同，
原AF/WAF与诊断服务器tracked工作树clean，GPU1已空闲。没有test前向/选择或训练。

机制因果已确认：访问可变Expert回滚在4个方法边界、两精度全部4/4 TRAIN批恢复
旧特征/logits至训练前；完整val fused/Full/Person/Face旧类logits也逐元素恢复。
冻结Expert阴性在两精度TRAIN及完整val各分支逐值不变；旧非Expert坐标SHA一致。
AF边界1/7旧类val恢复+.2678/-.0153，CI[-.2621,+.7502]/[-.0983,+.0627]；
WAF为+.2580/+.0346，CI[-.4480,+.9577]/[-.1449,+.2000]。500次原图配对bootstrap，
只含seed0样本不确定性。WAF两边界正、特征全恢复，但至少一个CI>0的门槛未通过。

按预锁协议，不进入完整保护方案设计或新训练；保留B0，停止当前Task Forward MoE
扩展，不扩seed/专家/温度/LR，不合并main。区间跨0并非证明真实收益严格为0；
已证实漂移机制也不能替代泛化收益。完整报告：docs/moe_expert_rollback_results_20261018.md。
原始输出output/moe_expert_rollback/formal_seed0_20261018_01/，离线结果与复算脚本
位于output/moe_expert_rollback/analysis_20261018/。仅本地追加结果文档，服务器
运行提交及源模型不动；未覆盖原WAF/主树/test-only未提交改动。

当前本轮无待运行训练/诊断任务；下面“正式在运行/等待”的记录由本条替代。
不重复回滚、不为跨0区间自动扩seed，不按结果更换边界或专家集。未来若重新立项，
先提出独立任务学习假设，并要求task0与匹配固定路由控制的TRAIN留出/validation收益。

## 2026-10-18：只读Expert回滚正式运行已启动

独立运行提交41567eb3bf320b39395c30bdfcdaefdfdf987a8d，本地/服务器/upstream一致、
启动时clean。338项全测通过（13.962s），之后七项针对测试通过；真实smoke尾号04
四个方法边界五条件全部完成，阴性TRAIN两精度/validation逐值相同，内存patch恢复、
源checkpoint/视觉权重SHA保护通过。前3次失败smoke保留（元数据root、稀疏标签、
FP32非确定性），用确定性CUDA算法和workspace修正，未放宽逐值门槛。
正式run_id=formal_seed0_20261018_01，tmux=ml_moe_rollback_seed0_20261018，GPU1，
4批TRAIN（上限64/批）及全val2397、AF/WAF边界1/7、五条件、无optimizer/test。
源路径固定原AF/WAF训练输出，日志logs/moe_expert_rollback，输出output同名子目录。
当前正式在运行，等待完整validation和原图配对bootstrap后按预锁门槛判断；不新训练。

## 2026-10-18：回滚阴性控制发现FP32重复前向数值非确定性

第三轮smoke在严格逐值阴性门槛停止。独立同输入复放确认相同state SHA、不做回滚的
after replay也有FP32 feature约1e-6/logit约1e-5差，AMP复放为0；不把该差解释成
模型改变。开启确定性算法与CUBLAS_WORKSPACE_CONFIG=:4096:8，保留逐值阴性门槛，
准备smoke尾号04。没有修改源checkpoint或放宽参数/预测不变性检查，正式尚未启动。

## 2026-10-18：修正回滚TRAIN稀疏标签筛选

第二次smoke读取了源模型并通过旧非Expert坐标核验，但在TRAIN抽样时发现dataset.targets
是类别ID列表而非one-hot。改用项目既有_intersects，补类0/未来类排除回归测试；
原失败产物保留，尚无正式运行/参数更新。下一次smoke使用run_id尾号03。

## 2026-10-18：只读回滚全测通过，修复smoke元数据入口

ca82541服务器337项Track-A测试通过。首轮smoke在git_metadata缺少root参数时退出，
尚未加载checkpoint/模型前向/更新；失败输出保留。补充显式项目root，下一轮使用
smoke_seed0_20261018_02重跑完整4个方法边界及五条件；正式尚未启动。

## 2026-10-18：用户授权执行旧Expert只读回滚归因

新分支codex/moe-expert-rollback-20261018从ceded63创建；原WAF脏文档复制保留，
不修改旧工作树。新增独立expert_rollback模块、六项机制测试、GPU1启动器及协议
文档docs/moe_expert_rollback_plan_20261018.md。锁定AF/WAF seed0、边界1/7，
4批按原图分组TRAIN（每批≤64），前后相同旧lane数，五条件，FP32主归因、TRAIN
AMP/TF32复核；全val2397旧类mAP，不读取test。零优化器更新，内存patch必恢复、
源checkpoint/state SHA审计。WAF两个边界val收益均正且至少一个CI>0、两配置两
边界≥3/4批特征漂移降低≥90%才进入完整保护设计；否则不新训练。先测试/smoke。
已只读检查服务器104工作树，仅原主树153项/test-only5项脏，均保留；GPU1空闲。

## 2026-10-18：WAF/AF已完成，当前不启动新训练

结果与证据以docs/moe_weighted_freeze_results_analysis_20261018.md为准，替代下面
“等待运行结束”的旧状态。WAF最终val/test42.2952/31.6815，AF42.4007/31.7904；
B0 test32.0610。加权未提供晋级证据，保留B0；不自动扩seed/专家/层数或调冻结分数。

若用户决定继续归因：
1. 预锁task1/task7边界与固定TRAIN图片、相同batch/lane/精度，比较旧函数变化。
2. 在内存副本中仅回滚旧lane仍访问的未冻结Expert；已冻结Expert回滚为阴性控制，
   检查参数SHA与数值底噪。无optimizer、无test选择，之后才报validation结果。
3. 仅若回滚证明有可恢复旧函数/validation收益，再设计完整访问依赖封存控制。
   旧Router仍读可变Expert会破坏稳定性；按视图Top2并集最多6名，只保护观察支持，
   训练noisy支持并集可能task0冻满22名。快照/任务子池/增长库均要明报存储与容量，
   不声称仍是原始每任务冻2名。单修保护不能修复task0已低B0 2.8205的问题。
4. 若仅消除漂移却无validation收益，应结束当前Task Forward MoE路线；v2–v4的
   图片证据/共享池/softTop2已试过，不重复包装旧改动，也不据v4局部冲突直接上PCGrad。

本轮只读Router probe和离线分析已完成，所有正式结果已同步并校验；没有待运行新
实验/训练队列，没有合并main。服务器训练代码和原工作树均保留。

## 2026-10-18：WAF-v1正式GPU1八任务已启动

代码分支codex/moe-weighted-freeze-20261018，本地/服务器/upstream运行提交均为
ceded63a0f58a27729845c32a78ff95f15b53fd8；启动时工作树clean。服务器独立checkout：
/mnt/haoyuan/workspace/multi-lane-main-moe-weighted-freeze-20261018。
331项Track-A测试通过。修复入口model.float()降低统计buffer精度的问题后，真实
smoke_seed0_20261018_02两任务各2更新全部通过：B0来源配对、加权Top2重算、
gate mass守恒、冻结专家与历史Router不变、CLIP不变、无跳步、checkpoint重放一致。
两任务各145个浮点参数与硬AF对应smoke完全相同，唯一新增key为training_gate_mass；
task0冻结[9,19]相同，task1加权[4,18]而硬计数[4,20]，机制差异成立，非性能结论。

正式run_id=formal_seed0_20261018_01，tmux=ml_moe_waf_seed0_20261018，GPU1，
seed0，8任务各30epochs；训练顺序执行，checkpoint核验后full_val/full_test双进程
同GPU1并行。已观察task0 epoch1：84更新、skipped=0、loss=0.67717795；当前仍在
训练，不能报告八任务最终结果。GPU1训练PID启动观测为514567，GPU0 AF原作业
PID199069/session和07fd35c clean checkout保留，未操作原主树/test-only用户改动。
日志/输出根分别logs/moe_weighted_freeze与output/moe_weighted_freeze。
两轮smoke/preflight的37个非checkpoint产物已同步本地并核验SHA256。首次失败
smoke保留用于追溯。仅在本地补写运行状态文档；服务器运行代码固定ceded63，
不在长任务中更新checkout。详见docs/moe_weighted_freeze_experiment_20261018.md。

待正式结束后：先检查exit_status、verification及evaluation_status双split退出0，
再同步全部日志/结果及checkpoint SHA清单。与GPU0硬AF按相同seed、预算、全人群
val/test逐阶段对照，分析最终与阶段均值mAP、三视图指标、每task冻结名单和gate mass/
硬次数排名差别、冻结集合增长与旧lane漂移。测试集仅作已锁定版本报告，不据此调参；
不以名单变化本身判定改善，未授权合并main。

## 2026-10-18：WAF smoke发现并修复入口dtype转换

首轮全测330项通过；GPU1两任务smoke训练完成，但核验发现model.float()将新增
float64统计buffer转成float32，正式运行未启动。新增bank._apply保护累计值原精度，
跟随模型迁移设备；补充float/half/to转换回归测试，下一步重测并用新run_id重跑smoke。
首轮task0的145个浮点参数张量与硬AF smoke完全一致，新增key仅training_gate_mass。
task1加权Top2为[4,18]，硬AF为[4,20]；这仅证明排序不同，不代表性能改善。
保留首轮失败产物与GPU0原运行；通过后再启动GPU1八任务。

## 2026-10-18：用户授权GPU1带权重激活冻结版本

新分支codex/moe-weighted-freeze-20261018从AF-v1运行提交07fd35c创建；保留GPU0
硬计数作业及其服务器checkout。复制本会话未提交文档，原树与主工作区用户改动不动。
WAF-v1唯一方法改动为按当前任务训练noisy Top-2的gate权重总和排名冻结，三视图汇总
Full/Person/可靠Face，不乘视图融合系数/输出范数，不归一化视图票数。原硬计数仍保存，
新增float64 buffer与checkpoint/核验支持；默认hard_count保持原checkpoint schema。
共享池22专家、单层v1接口、超参数均不变；GPU1/seed0，8task各30epochs，训练后
val/test双进程GPU1并行。完整配置见docs/moe_weighted_freeze_experiment_20261018.md。
已新增统计、model/runner/evaluator参数、boundary/验证器及机制测试和GPU1 launcher；
本地两项launcher测试与语法/diff通过。下一步Git-only远程新worktree、全测及两任务smoke。
本加权排名是本项目新变体，门控权重不是预测贡献的因果量，不宣称等同原论文硬频率。

## 2026-10-18：AF-v1与论文异同对照

核对论文与固定官方MTIL实现，补充AF实验文档：核心跨任务共享/CLS Top-2/频率累计
冻结复用已对齐；仍保留MULTI-LANE单层Task Forward、task×view Router、多标签
BCE与全部已见lane推理，无文本塔或DDAS。说明两冻结专家时基础任务参数仍更新、
三视图原始硬计数汇总总选2，以及AF-v1相对旧v1同时改变共享范围与冻结策略的归因边界。
仅本地文档，未修改正在运行的07fd35c服务器checkout，不需新测试/实验。

## 2026-10-18：AF-v1正式8任务已启动

真实两任务smoke和checkpoint/B0配对核验全部通过；322项Track-A测试通过，checkpoint
与原阶段mAP逐值重放一致，CLIP未变，peak reserved2826MiB；22份smoke日志/结果已同步
本地并逐项SHA验证（checkpoint留服务器）。
正式run_id=formal_seed0_20261018_01，tmux=ml_moe_af_seed0_20261018，GPU0/seed0，
8任务各30epochs，batch64。服务器独立worktree
/mnt/haoyuan/workspace/multi-lane-main-moe-activate-freeze-20261018，clean运行HEAD
07fd35cf97a2efc416d56348a1e82a2f00572291，branch/upstream一致；本地同分支HEAD一致。
已确认进入task0真实训练，使用配对初始化SHA35af16da...，test_loaded=False。
训练后先自动核验8个checkpoint与冻结历史/旧权重，再GPU0双进程并行固定全人群
full_val/full_test。日志./logs/moe_activate_freeze/formal_seed0_20261018_01/；
输出./output/moe_activate_freeze/formal_seed0_20261018_01/；外层launch.log在同日志父目录。
运行期间服务器checkout固定，不拉入后续文档；此次状态仅追加本地文档。未合并main。
正式结果尚未产生，不能用smoke mAP推断效果；完成后需同步正式日志/结果再作配对分析。

## 2026-10-18：AF-v1全测与两任务smoke通过

服务器独立worktree clean1070854，322项Track-A测试通过。GPU0 smoke_seed0_20261018_01
已完成task0/1各2成功更新、0跳步；task0冻结9/19，task1加入4/20，旧冻结专家及旧Router
SHA均不变；旧lane0的固定train特征relative L2漂移0.00296444，符合未冻结专家仍更新的语义。
源CLIP保持冻结。独立checkpoint重放两任务mAP26.760875/27.958062与训练评估逐值相同，
前者也匹配原v1 smoke。新增离线验证器直接核对checkpoint计数/累计mask/旧权重和B0
配对元数据，launcher将在正式test前先过该门槛。正式8task尚未启动，下一步运行验证器。

## 2026-10-18：用户授权AF-v1跨任务共享池与频率冻结实验

独立分支codex/moe-activate-freeze-20261018从d81b7e2创建；前一分析工作树的本会话
未提交报告/上下文已复制保留，原树和主工作区未动。用户明确要求新版本seed0完整8task，
仅GPU0，训练后val/test并行。方案以v1接口为对照，22名专家跨任务/视图共享，任务后按
训练Full/Person/可靠Face硬激活总次数选Top-2加入累计冻结集合，含已冻结者、不补足。
新Router可调用所有专家，只更新未冻结专家；冻结参数保留输入梯度，checkpoint保存完整
计数/mask/历史；记录冻结专家与旧Router SHA及旧lane输出漂移。新增参数2992000。
完整配置/边界见docs/moe_activate_freeze_experiment_20261018.md；新增bank、boundary
审计、model/runner/evaluator接线、launcher及机制/并发测试。launcher两测与语法/diff
检查通过；下一步Git-only服务器新worktree、Track-A全测、task0/1各2更新smoke再正式。
本次用户授权重开的是频率冻结迁移，非默认延续v4梯度隔离或结构搜索；不合并main。

## 2026-10-18：测试三视图生成与输入前提澄清

核对test入口也使用ThreeViewTransform(train=False)：同一原图在线生成三视图，
各自进入对应view Router再融合，不是仅训练有三视图或测试只选一个Router。
当前人物级EMOTIC test依赖注释body_bbox与预计算Face manifest；任意RGB端到端
找人/匹配Face并非已实现的测试入口。版本说明已记录该边界，业务/实验均未变，
不需要新测试；不以测试情绪标签决定视图或唯一任务。

## 2026-10-18：冻结与三视图身份接口澄清

确认v1–v4采用旧任务整池冻结，并非频率Top-2冻结；v1使用率统计只作诊断。
无DDAS；已知image_view选择对应Router，专家gate仍由内容动态生成。三视图分别
编码但保留同一人物样本的target/配对/融合关系，未引入显式跨视图身份对齐loss。
详细说明补入版本小结。仅文档更新，无业务变更/新结果，不需新实验或测试。

## 2026-10-18：澄清v1路由结构与论文差别

核对论文3.3及v1代码：论文同时具有task-specific Router和CLS条件的image-dependent
gating；不能概括成同任务图片使用固定专家。v1保留任务索引并增加view-specific
Router，同task三视图共享22专家，按图像CLS产生Top-2；实测固定路由属于坍缩。
明确Task Forward的任务路径隔离不等于专家复用，专家路由不等于三视图融合门控。
详细结构补入版本小结；未修改业务代码/配置/实验，不需新测试，原有未提交分析保留。

## 2026-10-17：按MoE-Adapters术语整理版本演进

新增`docs/moe_adapters_version_summary_20261017.md`，核对v1–v4配置与论文3.3：
v1迁移Adapters as Experts/CLS noisy Top-2；v2图片证据Router及必经共享；
v3小池、soft Top-2与匹配控制；v4专家也读取图片证据且仅写task CLS。
明确跨视图共享与论文跨任务activate–freeze协作的差别，最后梯度诊断不构成v5。
本次仅文档小结，无新增结果/业务代码/训练；未改当前保留B0的决定，无需新测试。

## 2026-10-17：再次复核归因结果与后续边界

120份同步文件SHA再次核对一致，无新训练/推理。补充报告解释：total/fused高余弦
只排除本次终点上aux大幅扭转当前fused，不能证明B0梯度被保护。task7 Prompt
Full/Face局部冲突不跨task；同参数点off锚点只有一个task7 batch重复反向。
固定训练batch fused BCE显示dynamic task7较B0更差，不能无完整曲线就认定单纯过拟合。

首选方案仍是保留B0、结束当前插入位置，不启动v5。若未来重开梯度假设，需独立
声明早/中/末轨迹及Adam状态的诊断；若换新方法，训练前按原图划定并排除holdout，
不能事后把已训练过的train子集当未见数据。Face校准/锚点残差不可换名重启。
本轮仅补充归因报告与三份上下文文档，无业务代码修改，不需新模型测试或远程执行。

## 2026-10-17：v4只读梯度归因完成，不启动v5

GPU0、seed0，四组B0/shared_fixed/view_fixed/dynamic，task0/2/7各4批64人，
FP32/dropout-off与AMP/dropout-on，共96快照完成；正式运行clean `272b768`。
312项测试通过，真实smoke8快照通过。12个checkpoint及模型state SHA均不变，
所有.grad为None，缓存像素/标签未变，0更新；分解最大相对误差0.0285%，
峰值reserved6138MiB。120份远程日志/结果已同步本地并逐项SHA复核，本地重汇总一致。

所有被测参数组/视图对均未达到预声明的跨任务冲突门槛（cos<−0.1，至少2任务、
每任务3/4批且两配置一致）。dynamic task7可靠人群Full/Face对Prompt的单路BCE
梯度确有4/4局部冲突，但task0/2均0/4；不能称为全局稳定干扰。dynamic实际加权
aux在各主要组均与fused同向；shared_fixed仅task0/batch2 Prompt抵消1.4%/1.7%，
未达10%。dynamic total/fused在Selector/Prompt/classifier/共享和专门Expert的
最小cos均>0.98。task7/batch1的同参数点MoE-on/off表示梯度反向在fixed与dynamic
重复，但其他批次/任务未重复。不可把终点raw梯度当Adam更新或全程泛化因果证据。

决定：当前证据不足以支持梯度隔离v5；保留B0，结束当前Task-Forward FFN并联
MoE路线，不扩seed/改温度/scale/专家数，不合并main，不自动转Face后置校准。
仅新增诊断/汇总/测试和报告，训练模型未变。未来重开需新的独立证据和明确实验。
完整报告 `docs/moe_v4_gradient_attribution_results_20261017.md`；机器结果与日志在
`./output/moe_v4_gradient_audit/`、`./logs/moe_v4_gradient_audit/`。
代码Git-only同步，服务器v4/main/test-only与原有用户改动均保留；运行完GPU0已空闲。

## 2026-10-17：执行v4既有checkpoint只读梯度归因

用户授权开始task0/2/7、四组固定训练batch梯度诊断，不训练v5，继续只用GPU0。
新分支`codex/moe-v4-gradient-audit-20261017`从`bc5f572`创建，保留v4未提交分析。
新增独立gradient_attribution模块、launcher、机制单测及完整协议，模型/训练入口不改。
每task4个原图组不跨batch的至多64人batch，缓存像素跨四组/两数值配置复用；
共96 snapshot，FP32/Expert dropout off与训练AMP/dropout敏感性对照。源checkpoint
位于原v4服务器worktree，均只读；不创建optimizer，无参数更新，验证state/checkpoint
与像素SHA不变。详见`docs/moe_v4_gradient_attribution_plan_20261017.md`。
下一步提交/Git-only远程新worktree、全测、8-snapshot smoke，通过后运行正式诊断。


## 2026-10-17：v4结束并完成结果分析，当前保留B0

GPU0、seed0四组各8tasks×30epochs、13950成功更新/0跳步，正式batch与
所有val/test/validation消融退出0；运行提交`bfd8ef9`。270份正式日志/JSON/
NPZ已同步并逐项SHA核对，32 checkpoint实测SHA与评估记录一致；另12份
只读诊断日志/分数/科研图也逐项SHA核对通过。fresh B0
与历史B0的val/test八阶段概率数组逐元素相同。

最终val/test（阶段平均val/test）：B0 42.1912/32.0610（48.4382/36.2781）；
shared-only42.6217/31.9368（49.2331/35.8074）；view-fixed42.8794/32.2326
（49.3266/36.0570）；dynamic42.5036/31.8392（49.0768/35.6642）。
dynamic test最终/阶段平均差−0.2219/−0.6139，500次原图聚类配对95%区间
分别[−0.849,+0.339]/[−1.514,+0.117]；全部关键总体比较区间跨0，无晋级证据。

v4专家确实处理图片证据，CLS残差/输入比4.5%–26.5%，专门/共享范数比
0.987–4.357；不能继续按弱专家调LR/scale。Person单路点估计改善，但
Full test−0.3613。dynamic specialist-off val42.5625，高于全开42.5036；
专门净贡献最终区间跨0。两次只读Router置换val42.3099/42.2838，原始
42.5036，支持局部匹配作用，不代表整体动态收益。

追加只读同checkpoint val视图屏蔽：Full-only42.6344、Face-only42.3050；
Full-only相对普通共享仅+0.0127且区间跨0，Face-only也无融合净收益。
可靠Face单路专门贡献+0.6632不能推导后置Face校准有效。上述诊断无训练/
无test干预，源checkpoint与模型state均不变，原始logits逐值重放差0。

决定：保留B0；不自动训练v5/扩seed/复用旧专家/合并main，也不把固定
最终小正差当成功。若继续，先在固定训练batch做共享梯度范数/冲突及fused/
auxiliary监督归因，证明可修复的负迁移后才考虑梯度隔离/平衡；否则结束
当前插入位置MoE。该下一步尚未运行。不原样重启已有失败的Face/post校准。
完整分析`docs/moe_task_ffn_v4_results_analysis_20261017.md`；表图/区间/只读
探针见`./output/moe_task_ffn_v4/analysis_20261017/`。本轮仅修复审计工具
`epochs_per_task`字段并更新报告与三上下文；模型/训练代码未改，未提交分析改动。


## 2026-10-17：v4正式8任务已在GPU0并行启动

最终用户范围：仅GPU0、seed0、8任务，各训练30epochs；四组baseline、
shared_fixed、view_fixed、dynamic并行。运行代码固定clean
`bfd8ef99f5a13d2c223b069522cf7fe2a099b6e7`，远程worktree
`/mnt/haoyuan/workspace/multi-lane-main-moe-task-ffn-v4-20261017`，tmux
`ml_moe_ffn_v4_20261017`。四组已进入task0，初始主参数SHA一致，
训练未加载test。巡检四个主进程均在GPU0，总显存约13GiB。

305项单测、256人真实图片输入/零残差探针、四组2-update smoke及实际
产物配对审计全部通过。smoke各0跳步、CLIP不变、checkpoint重放mAP
差0，峰值reserved为baseline2710/shared2802/fixed2806/dynamic2808MiB。
预检日志/分数已通过SSH归档传输同步本地，未传受控代码或.git。

正式run_id：`formal_<baseline|shared_fixed|view_fixed|dynamic>_seed0_20261017_01`。
输出`./output/moe_task_ffn_v4/<run_id>/`，日志`./logs/moe_task_ffn_v4/<run_id>/`。
每组训练结束后自动在GPU0以两个进程并行评估同一套8个checkpoint的
固定全人群val/test，结束后再做预声明validation消融。当前尚无正式
完整结果，test不用于调参；不扩seed、不合并main。后续文档提交不在
运行中更新远程checkout。


## 2026-10-17：v3诊断后的v4图片证据→CLS接口

用户要求执行修改并允许所有空闲GPU。独立分支
`codex/moe-task-ffn-v4-20261017`，起点`44c40fe`；主工作树与v3未提交
分析文档保留。v3动态最终val/test42.2628/31.6540；专门输出足够，
但至少99.9821%残差能量在Selector，详见归档v3分析。
v4保留索引1 Task Forward FFN并联位置和Frozen Forward，Router及专家
都读取归一化selected图片证据，残差仅写CLS。1共享+3专门、soft Top-2、
0.5/0.5、scale0.1沿用；旧任务整池冻结。四组B0/shared_fixed/view_fixed/
dynamic×seed0，各8tasks×30epochs；**仅GPU0四组并行训练，每组训练后
val/test两个进程并行评估**。这是用户后续最新明确指令，覆盖初拟pilot。完整配置与晋级条件见
`docs/moe_task_ffn_v4_experiment_20261017.md`。首版实现`7ebfb9e`远程303项Track-A单测通过。真实图片探针发现TF32
重复前向约1e-4波动，禁用TF32约3e-7；恒等审计改用关闭TF32、1e-6
容差并保留严格零残差检查。待新并行入口复测、探针与4组2-update smoke
及配对审计后启动正式批次。
远程最终305项全测与256人探针已过，FP32 identity最大差2.7474e-7；
四组GPU0 smoke已并行开始，等待完整checkpoint重放与配对审计。


## 2026-10-16：执行已授权v3

296项远程全测与三组smoke已过，checkpoint重放差0；两项并行启动
模拟测试本地已过。同步新测试并远程复测后启动正式8任务批次。
正式批次已在clean `7eb1f64`启动，tmux `ml_moe_ffn_v3_20261016`；
远程298项全测通过。等待自然完成，保持checkout固定；结束后同步
`./logs/moe_task_ffn_v3/`与`./output/moe_task_ffn_v3/`对应批次产物，
核查三组各13950更新/0跳步与val/test退出码，按预声明指标分析。

- v3分支`codex/moe-task-ffn-v3-20261016`；先远程相关/全套Track-A单测，
  再dynamic/shared_fixed/view_fixed各task0两更新smoke与同人群checkpoint重放。
- 通过后GPU0三组顺序跑8tasks×30epochs；每组训练后两个进程并行评估
  全8阶段固定人群validation/test，随后两专门组validation specialist-off。
- 用户最新明确授权覆盖旧的3task门槛及暂缓test建议；test只作预锁定报告。
- 完成后同步日志/输出、核对SHA、更新/跳步、冻结CLIP与范数/负载，按mAP
  和specialist消融判断；不自动扩seed/调LR或输出归一化，不合并main。

## 2026-10-16：MoE v2 结果后的 v3 候选

- 当前不运行v2 test、不扩seed、不做跨任务旧专家复用，也不合并main。
- v3保留当前Task Forward位置和归一化图片证据；缩为每任务1共享+3专门
  专家，专门路由使用训练/推理一致的soft Top-2，取消hard-ST与噪声。
- 修正控制：shared-only使用`0.5*shared`，view-fixed和dynamic都使用
  `0.5*shared+0.5*specialist`；固定/动态都激活两名专门专家并采用相同
  residual scale，避免当前共享系数和活跃容量混杂。
- 记录shared/specialist/combined输出范数、夹角、软硬负载；先以task0–2
  validation验证专门/共享输出比≥0.25及specialist消融有实际收益。
- 只有dynamic超过两控制且不伤Person，再做完整8任务seed0；再同时超过
  B0最终和阶段平均后才补seed1/2与锁定test。详见
  `docs/moe_task_ffn_v2_results_analysis_20261016.md`。

## 2026-10-16：等待 MoE v2 第三组完成

- dynamic 与 shared_fixed 训练/固定人群 val 已完成；view_fixed 巡检时
  在 task6 第7/30 epoch。保持 tmux 自然运行，不改动远程工作树。
- 第三组完成后核查三组相同代码提交、更新/跳步、CLIP 哈希、固定样本
  人群与逐视图/路由指标，再决定是否做 MoE-off validation 和 test。

## 2026-10-16：MoE v2 执行顺序

训练前条件已满足：服务器独立 v2 worktree clean、292项单测通过、
固定 val 前256人三视图归一化图片证据变化均过阈值；三组 task0
2-update smoke 已完成。
三组 smoke 已完成，正式阶段应顺序运行三组 8任务×30epochs，保持
GPU0、固定全人群 validation、同预算与不同 run_id；结束后优先比较
路由负载、融合/分支收益及 MoE-on/off，再决定 test。
正式三组已在 tmux `ml_moe_ffn_v2_20261016` 顺序启动；等待自然完成后
核查 8任务完整性、总更新/跳步、固定人群 val 与逐视图/路由统计，
然后同步远程 `./logs/`、`./output/` 的相关产物回本地。

- 本地提交并推送 v2 分支，经 Git 在服务器新增独立 worktree；保留远程
  主工作树、test-only 和 v1 worktree 的现状。
- GPU0 远程运行 Track-A 单测及新 MoE v2 单测，再用固定 val 前256人
  探测索引1的归一化图片证据；先复测修正的类别数断言，三视图均达
  变化门槛才进入训练。
- 先各做 2 更新 task0 smoke，检查 MoE 非零梯度、旧任务冻结、
  compact checkpoint 重放和实际路由统计。
- smoke 后以同配置顺序完成 dynamic/shared_fixed/view_fixed 三组
  8任务×30epochs，固定全人群 validation 优先；按 mAP、分支收益、
  专家使用判断后再决定是否读取 test。路线详见
  `docs/moe_task_ffn_v2_experiment_20261016.md`。
- 实验产物同步回本地后核查日志、配置 SHA、checkpoint、指标与路由。

## 2026-10-15：MoE 首版完成后的推荐方向

- 不把首版合并到 main，也不直接扩张为跨任务旧专家复用；
  test 阶段平均较 B0 低1.1513 mAP，Router 推理近乎单专家。
- 若启动 v2，保持 Task Forward 位置；先做图片证据 Router
  输入/小池/显式共享专家方案与同位置固定路由控制组，固定全人群
  validation 比较，再决定是否做完整 8 任务训练与最终 test。
- 用户需按需将远程 `./logs/moe_task_ffn_v1/<run_id>/` 与
  `./output/moe_task_ffn_v1/<run_id>/` 同步回本地，保留原 checkpoint
  和全部只读重评估/消融产物。

## 2026-10-15：MoE 首版分析后待办

- 用同一批冻结 checkpoint 在全人群 validation 上关闭 MoE 残差，
  输出独立 `full_val_moe_off/`，不重新训练、不覆盖原评估；以此区分
  MoE 的即时贡献与训练共同适应。
- 核查 CLS 路由输入的跨图片变化与 Top-2 权重变化；若确定性路由
  仍固定，先在验证集设计小专家池/训练推理一致的门控及负载约束，
  再考虑新正式训练。继续保持任务池冻结，不贸然复用旧专家。
- 测试集只用于报告；新方案先定验证集成功门槛与同版 B0 对照。

## 2026-10-15：正式 GPU0 批次结果诊断

- tmux `ml_moe_ffn_v1_20261015`；远程 worktree
  `/mnt/haoyuan/workspace/multi-lane-main-moe-task-ffn-v1`，固定提交
  `17a9b84`，批次 `moe_task_ffn_v1_seed0_20261015_01`。
- 8任务训练及并行 val/test 均已完成。原阶段评估人群随任务改变，
  先用冻结的 8 checkpoint 对 2397 val / 5368 test 全人群重评估，
  仅用重评估曲线和旧 B0 比阶段均值与遗忘。
- Router 在全部任务/视图的正式测试中均坍缩到固定 Top-2 专家。
  查门控分数与训练噪声、路由负载/均衡约束；先做不训练的验证集
  MoE-off 与路由消融，再决定是否缩池/调整门控、优化三视图互补。
- 在任何新训练前锁定同版 B0、初始化/采样/样本 ID、验证集指标及
  每任务负载分布，避免将随机差异当成方法收益。
- 提醒用户按需同步 `./logs/moe_task_ffn_v1/<run_id>/` 与
  `./output/moe_task_ffn_v1/<run_id>/` 回本地，之后核查可复现结果。


## 2026-10-15：执行已授权的 Task-Forward MoE 首版

- `smoke_03` 已完成 2 updates、0 skips、冻结 CLIP 不变、MoE 有有限非零
  梯度、峰值 reserved2902MiB。先独立重放 compact checkpoint 的 val，
  batch64 与 smoke 训练末尾 val mAP26.760875 逐值一致；可正式启动。
- 首次 GPU0 smoke 因 AMP index_add dtype 不匹配而在首批前向失败；
  第二次 smoke 完成 2 次更新后在验证 Router scatter 发现另一 AMP dtype
  问题；两处修复须先通过 autocast 单测并用 `smoke_03` 重跑，
  不继续使用失败的 `smoke_01/02` 输出。
- 完成 `docs/moe_task_ffn_v1_experiment_20261015.md` 所列代码/配置检查，
  在本独立分支提交并通过 Git 同步远程新 worktree，不碰远程主与 test-only。
- GPU0 跑 task0 两成功更新 smoke，审计初始 identity、实际专家/Router
  梯度、旧池冻结、显存、Face 三视图与 compact checkpoint 重放。
- smoke 通过后同 GPU0 跑 seed0 8 tasks × 30epochs；训练后自动启动
  validation/test 两个进程并行遍历固定 8 checkpoint，记录完整 run_id、
  日志/输出、各任务/视图/路由指标；完成后提醒同步远程产物并分析。
- 先以 validation 评估内容及视图相关路由是否改善泛化与三路互补性，
  核对旧 B0 的协议/hash 后才比较其 test 值；本版不加跨任务专家复用。


## 2026-10-15：整体迁移问题取代上一轮默认插入建议

- 先按 `docs/moe_adapters_full_transfer_and_multilane_location_20261015.md` 与用户讨论新方法。上轮索引1/任务独立池是另一保守变体，不作为“原MoE-Adapters完整迁移”默认实现。
- 若用户要求实施，优先独立MoE bank接Task Forward FFN并联，保留每层跨任务/视图共享池、task/view CLS Router、noisy Top-2、原法式activate-freeze；明确Drop & Replace后的raw任务状态与现有LN2 Adapter接口差异。
- 声明原频率冻结的遗忘风险与三视图计数策略，保护版另标；文本塔、DDAS、多标签损失及最终融合分别定义边界，不能用唯一DDAS任务选择替换all-lane推理。
- 参数化官方64/11/12等硬编码，消除全局task状态、审计noise eval开关/统计/冻结/checkpoint保存，再做声明过的远程验证。全规模新增视觉侧约35.904M参数，需资源估算。
- 当前仅分析；未启动或授权自动启动新实验，下文旧计划不自动执行。

## 2026-10-15：当前 MoE-Adapters 方向仅完成可行性分析

- 当前分支 `codex/moe-adapters-three-view-20261015`；详见 `docs/moe_adapters_three_view_transfer_feasibility_20261015.md`。本轮用户要求先分析，未实施或启动实验；下文旧计划不自动恢复执行。
- 如用户要求推进，先审阅并迁移已有静态MoE支线的最小依赖（本分支尚无该源码），实现同位置完整输出专家池及task/view Router；保留原Frozen Forward、损失、可靠Face mask与固定融合。
- 先声明完整配置，比较B0、普通Adapter、同池静态重叠和动态视图Router；必要时用任务常量路由/共享动态图像Router区分权重学习、内容条件化和视图专属规则。逐组报告总/活跃参数，避免b48与激活b32混淆。
- 预检零输出identity、Router探索及非零梯度、RNG隔离、旧函数/mask保护、checkpoint重放；首轮不同时扩跨任务复用、插入位置或DDAS。先有可重复validation收益，再锁定正式评价协议；不据已有test微小正差宣布有效。
- 当前没有可直接执行的新训练命令；未commit/push、未同步服务器。用户要求执行后依AGENTS.md持续授权和Git-only流程推进。

## 2026-10-10：全套preflight通过，正式四组启动后停止监督

业务c440321：287项本地/远程全测、四组真实三task smoke每组12updates/zero skips、原已训练task1→task2新版审计及可学习新专家分阶段smoke全部通过；79个静态产物已同步SHA一致。四组smoke配对、旧保护、残差关闭导出和比较器均通过。下一步GPU0正式 `protected_parax_audit_v2_seed0_val_20261010_01`（task0–2 seed0，30epochs，固定融合，OMP1），显存不足有限排队，不干预其他任务。启动后停止监督。分阶段正式实验未启动；本轮先收齐四组及残差开关新/旧类AP，再决定是否比较分阶段，不升级8-task/test/更多专家/level embedding。计划与检查证据见 `docs/protected_parax_audit_paired_staged_plan_20261010.md`。

## 2026-10-10：执行统一修正版四组，暂不启动正式分阶段

分支 `exp/parax-audit-paired-staged`；完整计划 `docs/protected_parax_audit_paired_staged_plan_20261010.md`。代码实现/本地287项单测完成，下一步远程全测、四组真实task0–2×4updates smoke与一组可学习新专家分阶段smoke，再GPU0四组30epochs/task正式validation。新batch `protected_parax_audit_v2_seed0_val_20261010_01`；先确认是否已存在，避免重复。旧失败batch不恢复，旧参数/mask逐值保护不能取消。新审计拆分缓存exact identity与完整前向数值底噪。每任务保存同模型关闭全部残差的分数以区分残差输出与联合训练轨迹影响，固定同人群旧AP。分阶段仅实现和smoke，不自动正式启动；待四组仍低B0且消融定位后再比较相同专家预算的联合/分阶段学习。仅GPU0，不干预已有任务，不持续监督，不访问test、不升级8-task/level embedding/更多专家。

## 下一阶段：shared Adapter 容量 seed1/2 复现（2026-09-26）

seed0 residual-scale validation 已同步并通过SHA核验。b97提高mean task mAP `+1.3077`但final只`+0.2654`，低于注册`+0.5`门槛；收益主要来自Full/Person，reliable-Face无提升，forgetting略差。先不扩结构，以seed1、seed2复现b32/b97配对8-task validation。固定scale0.03、冻结ViT-B/16、layer1 shared Image-token Adapter、30 epochs/task、batch64、BCE+Adapter ASL、auxiliary loss0.1、AMP/TF32，其余与seed0完全一致；validation-only、无checkpoint/test。

执行顺序为seed1在GPU0/1配对运行b32/b97，完成后seed2复用GPU0/1；不要占用seed0 held-out test当前使用的GPU2/3。若三seed final mAP平均增益未达`+0.5`或稳定性/遗忘明显变差，停止b97扩容；若通过，再按validation指标设计最小的视图残差强度对照。详细分析见`docs/shared_capacity_residual_scale_validation_results_20260926.md`。

正式 batch `shared_capacity_residual_scale_multiseed_seed12_20260926` 已启动于服务器 clean worktree `/mnt/haoyuan/workspace/multi-lane-main-shared-capacity-multiseed-validation`，tmux `ml_caps12_20260926`。seed1 b32/b97 已进入 GPU0/1；脚本会在两组完成后自动运行 seed2。运行期间不访问test、不添加新参数搜索。

## 当前运行：共享 Adapter 容量 residual-scale held-out test（2026-09-26）

batch `shared_capacity_residual_scale_test_20260925_160954` 已在 tmux `ml_captest_160954`、GPU2/3 运行 A0 shared b32 与 A-cap shared b97。两组均以 seed0 在 train split 从头训练 8 tasks × 30 epochs，再按 task 评估 held-out test；validation 未保存 checkpoint。Residual scale 固定为0.03，其他训练配置与 validation 配对一致，不做 test-side 搜索。

等两臂自然结束后，同步服务器 `logs/emotic_track_a_shared_capacity_residual_scale/shared_capacity_residual_scale_test_20260925_160954/` 与外部结果目录 `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_capacity_residual_scale_test_v0.1/shared_capacity_residual_scale_test_20260925_160954/`，核对状态、8 tasks、updates、skip、异常和SHA-256，再分析总体及 Full/Person/Face/reliable-Face 指标。详细锁定配置见 `docs/shared_capacity_residual_scale_test_plan_20260926.md`。

## 当前运行：后置特征校准对照（2026-09-25）

P-post-zero 已与 B0 完全一致，闭合了阶段一；tiny 的非零残差确定会损失精度。ParaX image-stream、蒸馏和 level routing 停止。

下一步 `V-post-residual`：ParaX/Adapter/CLIP 微调全部关闭，Full 作为 anchor，使用现有 `residual_three_view` 融合模块对 Person/Face 训练 task-local zero-initialized residual，scale=`0.1`、hidden=`16`、fusion LR=`4e-4`，只跑 task0--2 validation，不保存 checkpoint，不运行 test。若仍低于 B0，说明当前辅助视图本身需要重新建模，不再继续 image-stream 路由。

## 下一步：P-post-zero strict identity（2026-09-25）

P-post-tiny final mAP `44.7075`，低 B0 `0.5319`；其 residual ratio 已约 `1.7e-7`。下一步只运行 B0-paired 与 P-post-zero：结构、optimizer 参数组、router/expert 保留，fixed output scale=`0`，task0--2 validation-only，无 checkpoint，禁止 test。

判断规则：zero 若仍低于 B0，优先修复完整训练可比性；zero 若与 B0 对齐，确认 tiny 残差仍有负作用并结束 ParaX image-stream 路线，转向冻结 B0 feature 上的 view-specific calibration 或 logit-level residual。

当前批次：`parax_post_zero_control_20260925_020000`，GPU0 B0-paired、GPU1 P-post-zero。等待结束后同步统一分析，不实时监督。

## 当前运行：ParaX P-post-tiny（2026-09-25）

P-post-static 已完成：final mAP `43.9191`、forgetting `2.2440`；dynamic P-post-small 为 `44.2960/2.5475`，均低于 B0 `45.2395/2.7760`。静态方案稳定但 task0 损失更大，动态路由只部分恢复精度。

下一步运行 `P-post-tiny`：最终 lane feature 后置 dynamic ParaX，rank32/3 experts，strict zero-output，fixed scale `0.0001`，task0 训练 center、后续 task 冻结 center，无 level embedding，task0--2 validation-only，无 checkpoint，禁止 test。若 tiny 仍低 B0，停止 ParaX image-stream 扩展，改做 view-specific post-feature calibration 或 logit-level residual。

## 当前运行：ParaX post static control（2026-09-25）

P-post-small 已完成但 final mAP `44.2960` 低于 B0 `45.2395`，尽管 forgetting `2.5475` 略低于 B0 `2.7760`。task0 已低 `1.3981`，因此阶段四 Full-only CPU-cache 蒸馏暂缓；蒸馏不能修复 task0 表征损失。

下一步已固定为参数量匹配的 `P-post-static`：P-post 后置位置、rank32、3 experts、fixed scale `0.001`、task0 后冻结 center，改用 uniform gate，不使用动态 router。仅 task0--2 validation、seed0、无 checkpoint、禁止 test。若 static 仍低 B0，则停止继续扩展 ParaX image stream，改做 strict identity 或 view-specific 受限后置专家。

实际批次 `parax_post_static_control_20260925_003631` 已启动（GPU0 B0-paired、GPU1 P-post-static）。`post_static` 是独立模式，确保静态对照与 P-post 位于同一最终 lane feature 位置。

## 当前运行：ParaX post-encoder control（2026-09-25）

批次 `parax_post_control_20260925_230104` 正在 GPU0/1 运行 B0-paired 与 P-post-small。该实验用于区分“ParaX 机制无效”和“CLIP 中间插入位置破坏后续编码”；等待完成后统一分析，不访问 test。

## 当前运行：Frozen-center controlled residual 阶段（2026-09-24）

唯一 batch `parax_frozen_controlled_20260924_220444` 正在 GPU0/1/2 运行 B0-paired、Frozen-center-small、Frozen-center-penalty；等待完成后同步和统一分析，训练期间不做实时监督，不访问 test。

## ParaX shared-center stability 阶段三已完成（2026-09-24）

Shared-live/Frozen-center/Task-local-delta 已完成。Shared-live final `39.2991`、forgetting `8.1971`；Frozen-center `43.0695/3.3804`；Task-local-delta `42.6452/6.7104`；B0 `45.2395/2.7760`。Shared-live 与 Frozen-center task0 一致、task1 起分化，确认跨 task shared center 漂移是主要遗忘来源。Frozen-center 仍低 B0 `2.1700`，因此还存在 task0 表征改写和固定融合失配；Task-local gate 残差过大，停止该组。

下一步：

1. 只保留 Frozen-center，加入更小固定 output scale或可微 residual penalty，目标 residual/token ratio `0.05--0.10`。
2. 做严格 task0 paired 对照，记录 residual/logit drift 和单路 mAP，先解决 task0 相对 B0 的 `-3.7219`。
3. 若受控 Frozen-center 仍有遗忘，再做 Full-only CPU cache 蒸馏；不在训练 batch 中运行 teacher。
4. 只有接近 B0 后才恢复 level routing；否则转向冻结 CLIP 后置 adapter或显式 view-specific 小专家。

## 当前运行：ParaX shared-center stability 阶段三（2026-09-24）

唯一 batch `parax_center_stability_20260924_173712` 正在 GPU0/1/2 运行 `Shared-live`、`Frozen-center`、`Task-local-delta`。等待完成后同步 output/log 并统一分析；训练期间不做实时监督，不访问 test。

## ParaX paired stability validation 已结束（2026-09-24）

批次 `parax_paired_stability_20260924_161643` 的 B0/P10-identity/P10-small/P10-zeroB task0--2 validation 已完成。P10-small final mAP `44.3357`，低 B0 `0.9037`，但 forgetting `2.6773` 与 B0 `2.7760` 接近；P10-zeroB 和 P10-identity 分别低 `6.3187/6.6329`，forgetting 升至 `9.6834/10.9558`。当前结论是“小残差可稳定但无收益，identity/zeroB 仍会漂移”。

下一步顺序：

1. 补完整模型级 initial token/logit diff、Selector/Prompt/head hash、global RNG 和第一批 DataLoader index 对齐；修正 ParaX-only smoke 对 `max_initial_difference` 的漏检。
2. 将 identity 改为真正的零输出对照，或使用有界 output gate；不要让当前无界 identity gate 继续作为稳定性证据。
3. 只保留 P-10/rank32/无 level embedding，做 Shared-live、Frozen-center、Task-local delta 三组短 validation，并在固定 anchor batch 测旧 task feature/logit drift。
4. 只有 frozen-center 仍有明显遗忘时，才实现 CPU cache 的低显存旧 task 蒸馏；先 Full-only。
5. 只有稳定版本达到或超过 B0 后，才恢复 level routing；否则转向冻结 CLIP 后置 ParaX-style adapter 或显式 view-specific 小专家。

## 当前 ParaX 决策：先做严格 paired 稳定性诊断（2026-09-24）

根因定位与实验协议见 `docs/parax_root_cause_and_next_plan_20260924.md`。下一步顺序固定为：

1. ParaX 初始化使用 `torch.random.fork_rng()`，增加 B0/P10 参数 hash、global RNG、第一批 DataLoader index 对齐检查。
2. 增加严格 identity/zero-output 初始化，先做真实 task0 paired smoke。
3. 只做 B0、P10-identity、P10-small、P10-zeroB 的 task0-2 短 validation；关闭 level embedding、连续层、rank64 和 test。
4. 稳定后再比较 shared-live、frozen-center、task-local delta；如仍遗忘，再实现 CPU logits/features cache 的低显存蒸馏。
5. 只有稳定且接近 B0 后，才恢复 image-only level routing、level-bias 和 gate 正则。

当前仅完成分析文档，未修改业务代码、未启动实验。

## 当前方向调整：ParaX seed0 已失败，先做稳定性诊断

用户认为当前个性化 Selector 路线实验效果不支持继续，停止该路线；不要再推进其 selector 参数扩展、Router 或额外 seed。下一研究候选为本地 `/Users/denghaoyuan/workspace/MyCode/ParaX-main` 的 ParaX 动态参数路由。

ParaX shared level routing 的六组 seed0 validation 已完成，但全部低于 B0；结果和逐视图、gate、residual 诊断见 `docs/parax_shared_level_routing_analysis_20260923.md`。当前不要启动连续层、rank64、seed1/2 或 test。

下一步仅保留 P-10 单插入点做短程稳定性实验：identity/1e-3 输出初始化或固定小 residual scale（目标 ratio 0.05–0.10），并比较冻结 shared center、只训练 router、旧 task feature/logit distillation。首轮关闭 level embedding，先确认能否控制表征漂移和 forgetting；task0/2-task validation 通过后再决定是否恢复动态 level routing。

完整结构比较、配置建议、推进门槛与风险见 `docs/parax_shared_level_routing_analysis_20260923.md`。当前没有实验执行授权；实施前需按项目实验流程向用户声明完整配置。

## 已停止：视图专用 Selector 路线

1. 历史分支 `exp/view-specialized-selector` 已实现 Selector 视图专用模式，但用户确认当前实验证据不足以继续。
2. 不再推进该路线的额外 Selector 容量、Selector-aware Router、seed1/2 或 test；保留历史代码、文档和结果用于审计。
3. 下一候选统一转入上方 ParaX 动态参数路由分析；实现前重新创建实验分支并声明完整配置。

## 导师讨论材料

当前关键实验的通俗版总结见`docs/three_view_shared_fusion_experiment_summary_for_advisor_20260916.md`。文档用于
讨论现有证据、结论边界和下一步Full完整私有Adapter方案；不包含新的实验授权或启动状态。

## 已完成：Full分类头独立seed0对照

按`docs/full_private_head_diagnostic_plan_20260915.md`先运行H0共享逐路head与HF Full独立head。服务器221项
单元测试和真实配对smoke已通过，正式两组8-task validation也已完成。H0/HF logit final为
`42.6627/41.1145`，HF低`1.5482`；Full单路低`1.3932`且其他两路也下降，所有advance检查失败。
不补seed1/2、不test。H0比历史P0低0.4127，说明等价的逐路head改写在AMP长期训练中产生基线漂移；
后续实验必须保留历史P0原前向。结果见`docs/full_private_head_results_20260916.md`。

## 下一候选：保留P0原前向的Full完整私有Adapter

若用户确认继续，先设计一组最小seed0 validation对照：基线严格复用历史P0“固定特征融合后过共享head”；
实验组只让Full使用完整b32私有Image-token Adapter，Body/Face继续共享b32，Selector/Prompt/head、损失、
Face mask、数据与优化协议不变。第一步只判断Full单路及融合是否同时恢复；出现明确正效应后再补等参数量
全共享容量对照和seed1/2。若失败，再转向Full Selector/Prompt解绑。本项尚未实现或启动。

## P0第一批已完成：预测来源已定位到Full

先阅读`docs/p0_r1_gap_results_20260915.md`。一次P0复现+A/B全部完成并同步；最清晰的来源差异是Full，
共享Face有正向收益，Body与Full来源有交互。Full分类头独立对照现已完成且明显退化，排除该最小候选；
下一步条件候选为保留P0原前向的Full完整私有Adapter，再根据结果决定是否解绑Selector/Prompt。

## 已完成：P0与R1差距第一批诊断

依据`docs/p0_r1_gap_diagnostic_plan.md`，服务器完整单元测试与真实smoke均已通过；D0 P0 seed0 validation
batch`p0_r1_gap_seed0_20260915_1640`及自动D1四格/D2八种来源替换已完成。融合运算已排除，预测来源差异
主要落在Full；随后Full分类头解绑也已失败。当前按上节条件决定是否进入Full完整私有Adapter诊断。

## 待讨论：文献支持的共享与动态融合设计

参考`docs/shared_view_fusion_literature_20260915.md`，优先讨论AdapterFusion两阶段训练、VLMo模块级共享与
MBT瓶颈交互。先分离融合运算和参数共享影响，再比较静态/动态组合。下文已结束实验的晋级规则保持为历史记录；
“所有单路先增强”不是动态融合成立的普遍必要条件。尚未锁定或启动新实验。

## 当前执行：共享主体加视图专用Image-token Adapter

1. P0/P1/P2分别锁定为共享b32、参数量匹配共享b45、共享b32加Full/Person/Face各b4；固定三路
   特征融合、joint gradient和0.1逐视图辅助监督，不引入Router或DGL。
2. 完成本地静态检查后只提交本实验文件并推送；服务器先审计全部worktree并创建新的独立clean
   worktree，不修改主工作树或`multi-lane-main-test-only`。
3. 在`ddp`环境运行完整单测，再运行P0/P1/P2真实EMOTIC task0一轮GPU smoke；核对参数计数、
   view-specific梯度、路径审计9个采样点、有限loss、0 skipped、Face mask和view diagnostics。
   首轮已训练通过但捕获Adapter路径审计AMP下溢；先提交诊断缩放修复，再重跑全测和smoke。
   修复后213项全测与三组重试smoke已通过；提交smoke记录并ff-only同步后即可检查GPU并启动正式批次。
4. smoke通过后用三张安全空闲GPU启动唯一seed0完整8-task validation；日志写入`./logs/`，控制文件
   写入`./output/`，大结果写入服务器外部结果目录，不保存checkpoint、不访问test。
   用户确认约8GiB空闲足够后，唯一batch `view_specialized_adapter_seed0_20260915_104759`已在tmux
   `ml_view_adapter_104759`放宽资源门并于GPU0/1/2正式启动；三份config和首个epoch均通过审计，
   84 updates/0 skipped且无OOM。只监测该批次，不重复启动、不更新运行worktree。
5. 只有P2 final相对P0/P1均至少+0.10、average与Standalone Full不低P0且缩小到独立R1的差距，
   才进入Selector-aware动态Router阶段；失败则根据路径诊断重新判断机制，不搜索bottleneck或LR。
   正式结果P2 final=`42.7961`，比P0低`0.2793`，五项全部失败；本阶段已结束，不运行Router、seed1/2
   或test。若继续共享路线，下一阶段先在相同结构对照下增强并平衡单视图监督，且分开记录共享Adapter
   与三路delta的梯度和残差强度；只有单路专家确实增强后才训练Router。

## 已完成：OOF优势限定的排序蒸馏

1. D3直接软概率BCE蒸馏已失败；不补seed1/2、不test、不搜索更多mix。
2. 如果继续OOF教师路线，hard BCE恢复权重1.0，Adapter hard-label ASL不变；跨视图教师
   不再提供绝对概率目标。
3. 仅对13个R1 OOF AP优势在3 folds一致的类别，使用“Full OOF排错且R1纠正”的
   正负样本对优化student logit排序。
4. 排序分支使用确定性Full视图，hard BCE/ASL使用现有随机增强视图。E1已锁定pair batch16、
   weight0.05；实现提交`0c0b753`已推送，服务器独立clean worktree的203项全测和真实
   EMOTIC/OOF GPU smoke均通过。
5. 唯一batch `oof_advantage_ranking_seed0_20260913_203305`已完成；Full final、Full average、
   固定R1 final相对D0为`-2.6785/-3.2720/-2.0639`，三项门槛全部失败。
6. 结果已同步并校验；不补seed1/2、不运行test，不再搜索ranking weight、pair batch、筛选阈值或
   distillation mix。OOF蒸馏路线正式结束。

## 下一候选：独立姿态/上半身Person专家

1. 固定现有Full、Person、Face和R1权重，先不修改融合器。
2. 阶段0只审计目标人物关键点覆盖、置信度、多人错配、上半身裁剪有效分辨率和各task正例覆盖。
3. 审计通过后，seed0 validation比较旧Person锚点与一个轻量pose/上半身独立专家；优先使用
   bbox归一化关键点或冻结pose-pretrained表示，避免再次重复通用CLIP Face路线。
4. 新专家必须同时提高自身final/average mAP以及固定融合final mAP，才补seed1/2。失败则不扩网络，
   回到固定R1正式冠军`33.0119 ± 0.3038`整理消融与论文结论。

## 已完成：更强的OOF R1集成教师

1. D1/D2 seed0 validation已完成但未通过门槛；不补seed1/2、不运行test。
2. 不直接搜索distillation mix。Person和Person+Face OOF在8 tasks均弱于Full OOF，
   固定0.20权重对早期类别造成广泛损失。
3. 已新建实验分支并实现D3：无泄漏固定R1 OOF教师，mix0.20、Full主路、Adapter ASL、
   seed0完整8-task validation和选择门槛均不变。实现已推送，服务器独立worktree的
   199项全测和真实OOF/GPU smoke已通过。D3 batch `oof_r1_teacher_seed0_20260913_192759`
   已完成，三项门槛分别为`-0.1689/-0.4322/-0.3271`，禁止test。
4. D3失败后已停止全样本直接概率BCE蒸馏，不搜索更多mix。仅保留上方受约束
   排序蒸馏作为最后候选。

## 已完成：OOF跨视图教师蒸馏Full lane

1. Face表情预训练projection、Adapter和CLIP+AffectNet残差均未提高可靠Face或R1；停止该路线，
   不补seed1/2、不test、不再调rank/scale/LR。
2. 保持旧CLIP Face和固定R1不变；已审计现有image-group OOF为9个完整source，不重训教师。
3. 只对每task当前新类构造软教师，固定硬BCE/软BCE=`0.80/0.20`；Adapter ASL不变，旧task
   selectors/prompts/head/Adapter与蒸馏目标都不再改动。
4. seed0完整8-task validation最小比较：D0 fresh Full champion、D1 Person OOF teacher、
   D2 Person+Face OOF teacher。
5. 选择门槛固定为Full final至少`+0.05`、Full average不降、锁定R1 final至少`+0.05`。
   实现与preflight记录已推送，服务器独立clean worktree的197项全测和真实
   OOF/GPU smoke已通过。batch `oof_distillation_seed0_20260913_230500`已完成；D1/D2均未
   通过门槛，不补seed1/2、不test。

## 当前执行：CLIP Face + AffectNet补充残差（2026-09-13）

1. 新分支只实现一个混合候选：旧CLIP Face主路径不变，冻结AffectNet embedding+8类logits通过
   task-specific零初始化rank32、scale0.1残差补入最终Face表征。
2. 本地静态检查后提交推送；服务器必须创建新独立clean worktree，不触碰主工作树和test-only。
3. 191项完整单测与真实双encoder GPU smoke已通过；零残差误差`1.55e-7`，两条loss梯度有限且路由
   正确，冻结主干无梯度。
4. 失败首启动未执行优化；dict输入路由修复后，干净batch
   `face_expression_residual_seed0_20260913_143719_retry1`已在GPU5/tmux
   `ml_face_expr_residual_retry1`正常运行完整8-task validation。仍固定R1 beta0.20、不存checkpoint、不访问test。
5. 已完成：可靠Face/固定R1相对锚点`-0.9660/-0.2765`，双门槛失败。不补seed1/2、
   不test，Face专用表征路线结束。

## 当前执行：Face表情预训练专用表征（2026-09-12）

1. 官方EmotiEffLib AffectNet EfficientNet-B0权重已锁定，SHA-256为
   `47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`；冻结encoder和BatchNorm统计。
2. 已提交推送并创建服务器独立clean worktree；187项完整单测与五点审计通过。
3. 官方checkpoint、真实五点对齐dataset及projection/Adapter GPU smoke均通过，0非有限值，峰值不足
   200MiB。
4. 唯一batch已完成：projection/Adapter固定R1为`43.3234/43.2659`，均低旧CLIP Face锚点
   `43.5812`；可靠Face也分别下降0.5276/1.1767。双门槛失败，不补seed1/2、不test。
5. 不扩大pooled-feature Adapter。若继续Face路线，只运行一个“旧CLIP Face主路径+AffectNet冻结特征
   零初始化task-specific残差”seed0 validation候选；仍须可靠Face和固定R1各至少+0.05，否则结束
   Face专用表征路线。

## 下一阶段：Face表情预训练专用表征（2026-09-12）

1. reliable_m15等更新量控制已失败：可靠Face/R1分别相对旧锚点`-0.1384/-0.1137`；不补seed1/2、
   不运行test，正式停止margin、过滤阈值和基础增强路线。
2. 新建独立实验分支，复用manifest已有`face_keypoints`审计五点覆盖、顺序、有限性和对齐边界；无需
   重跑Face detector。
3. 引入AffectNet/FERPlus等表情预训练encoder，冻结主干；恢复旧锚点`valid && !ambiguous`训练池，
   用五点相似变换+margin0.15上下文，无效关键点回退letterbox并保留mask。
4. seed0 validation最小比较旧CLIP Face锚点、表情encoder+task-specific轻量投影、表情encoder+
   task-specific bottleneck Adapter；Full、Person和R1 beta0.20固定。
5. 仍执行Face可靠子集与固定R1 final mAP各至少`+0.05`的双门槛；通过后才补seed1/2并考虑一次锁定
   test。开始编码前先锁定预训练权重来源、许可、输入分辨率/normalization和权重SHA。

## 当前执行：reliable_m15等更新量公平实验（2026-09-11）

1. 已实现按task成功updates预算：`[1920,1620,360,3570,1710,960,240,600]`，总计10,980；
   每task cosine在自己的更新终点降至0。
2. 只运行seed0完整8-task validation；锁定Face reliable_m15、Full/Person来源、R1 Face权重0.20、
   threshold0.5，不保存checkpoint、不访问test。
3. 提交推送后，在服务器独立worktree完成完整单测、真实数据和Adapter GPU smoke，再启动唯一正式批次。
4. 只有可靠Face和固定R1 final validation mAP相对旧锚点均至少`+0.05`才继续；否则停止margin、
   过滤和基础增强路线，转向landmark对齐的AffectNet/FERPlus表情预训练Face encoder。

前置验收已完成：179项全测和真实GPU smoke均通过。唯一正式batch
`face_reliable_equal_updates_seed0_20260911_094039`正在GPU0运行，tmux为
`multilane_face_equal_updates_20260911_094039`。只等待8 tasks完成并由launcher自动执行严格R1比较；
不重复启动、不访问test。

## 独立Face expert质量validation已完成（2026-09-11）

1. 三组均完成；固定R1 final val mAP为m15过滤`43.4933`、m05`43.3646`、m05+jitter
   `43.3363`，均低旧锚点`43.5812`。不补seed1/2、不test、不继续搜索margin/jitter。
2. reliable_m15的Face可靠子集提高1.0846，但总updates因过滤少21.6%。下一步只做一个公平控制：
   按旧锚点逐task updates `[1920,1620,360,3570,1710,960,240,600]`重跑reliable_m15，保持固定R1。
3. 等更新量候选仍须同时提高Face可靠子集与R1 final mAP各至少0.05；失败则结束基础Face输入路线。
4. 后续结构方向为landmark对齐的面部表情预训练encoder+task-specific轻量投影/Adapter，不再优化
   融合器。完整结果见`docs/face_expert_quality_results_20260911.md`。

## 服务器环境恢复完成（2026-09-10）

1. `ddp`与`cocoer-preprocess`均已恢复并通过169项全测、真实Face检测、CLIP/Adapter GPU smoke和
   单task单epoch端到端smoke；当前不需要重跑任何正式实验。
2. 后续运行必须显式使用`/opt/conda/envs/ddp/bin/python`，不得使用重置后的base Python3.11/
   torch2.6。Face离线检测使用`/opt/conda/envs/cocoer-preprocess/bin/python`。
3. SSH免密和GitHub agent forwarding已恢复。若容器再次重置，按
   `docs/server_environment_restore_20260910.md`先恢复授权与两套环境，再运行实验。
4. 环境已就绪后，研究路线仍按下方OOF结果决策推进；不要因为恢复smoke重复启动OOF或正式test。

## class-aware OOF stacking已完成（2026-09-11）

1. 唯一有效run完成且未访问test：C0/C1/C2 final val mAP为
   `43.581193/43.586666/43.600490`。
2. C2相对C0/C1为`+0.019297/+0.013824`，虽收益分散且8个task均略升，仍未通过已放宽的双重
   `+0.05`门槛；不再下调门槛，不补seed1/2、不运行test。
3. 停止当前动态Router/stacking：不搜索更多prior、rank、hidden、描述符维度或类别独立大网络。
4. 下一候选方向为提高独立Face expert的信息质量；保持Full、Person与固定R1权重不变，先在
   validation验证Face自身和R1融合是否同时提升，再决定是否补多seed。详细见
   `docs/class_aware_oof_stacking_results_20260911.md`。

## OOF完成后的下一步（2026-09-10）

1. 当前OOF-R3未同时超过R1和R2，严格停止现有Router：不补seed1/2、不运行test，不扩大hidden、
   descriptor或prior搜索。固定R1仍是正式三种子冠军`33.0119 ± 0.3038`。
2. 下一轮可复用现有22MB OOF产物，不重训9个专家，先做CPU-only seed0 validation：C0固定R1；
   C1为强收缩到R1的class-specific三路bias；C2在C1上增加rank-2共享样本可靠性交互。
3. 无效Face始终权重0；每个类别只使用其增量task结束时的Router快照。禁止每类独立大MLP或test选参。
4. 只有C2同时超过C0/C1至少`0.10` final mAP，且收益不由单类独占，才补seed1/2 validation；
   否则结束动态融合路线，转向提高独立Face expert或保持固定R1。

## 当前执行：三视图image-group OOF/cross-fitting（2026-09-10）

1. 提交推送`codex/three-view-oof-crossfit`，服务器审计全部worktree并创建新的独立clean worktree，
   不修改主工作树和`multi-lane-main-test-only`。
2. 已完成：ddp环境166项完整单测、真实train image-group三折覆盖/三视图ID与target对齐、单task
   Full/Person/Face OOF score-dump与Adapter GPU smoke、完整train/val描述符导出均通过。
3. 使用GPU0--7安全调度9个seed0 OOF专家：3 folds×3 views，每个专家8 tasks×30 epochs，模型、
   loss、Adapter、增强和精度协议全部固定。每条训练预测必须来自未训练该image-group的fold模型。
4. 专家完成后导出完整train/val冻结CLIP描述符，在CPU训练OOF-R2/R3：共享hidden16主干、每task
   3维偏置、prior`0/0.1/1`、80 epochs/task、旧task快照冻结、无效Face权重0。
5. 同一批100% train endpoints上比较固定R1、最佳OOF-R2、最佳OOF-R3。只有R3 final validation
   mAP同时超过R1和R2才补seed1/2；否则停止该Router。当前批次不读取或运行test。

唯一正式batch`three_view_oof_seed0_20260910_160250`已启动，tmux
`multilane_three_view_oof_20260910`。前8组正在GPU0--7运行，第9组自动排队；不重复启动。完成后
同步OOF来源摘要、Router states/selection、control和日志，并执行预设R3双重胜出判断。

## 当前执行：固定R1三种子正式test（2026-09-09）

1. 提交推送当前分支；安全处理Automatic Upload后，在服务器新建独立worktree，不触碰test-only。
2. 在服务器ddp环境运行完整单测；生成独立train/val/test Face manifest，确认test完整且无重复匹配。
3. 运行真实Face test dataset/score-dump smoke和Image-token Adapter GPU smoke。
4. smoke通过后用三张空闲GPU并行seed0/1/2 Face正式test；复用已有同seed Full/Person test dumps。
5. 三组完成后自动只计算一次锁定R1。最终报告三种子final mAP mean±sample std、逐seed相对同seed
   Full+Person差值及方向一致性；禁止根据test结果改变权重或质量门槛。
6. 当前唯一正式batch为`fixed_three_view_seed012_test_20260909_213052`，已通过preflight并在GPU0/1/2
   运行。只等待完成，不重复启动；结束后核验3×240 epochs、skipped0、score对齐和固定融合摘要。

本阶段已完成：固定三视图`33.0119 ± 0.3038`，相对Full+Person三个seed均为正，升级为当前正式
冠军。结果、test scores和日志已同步且哈希一致。

## 下一阶段：先修正增量Router语义，再做特征融合

1. validation-only复用已有90/10三视图scores、CLIP descriptors和R2/R3 task states；不重训专家。
2. 对task t评估时，分别用已冻结Router0...t只融合各自类别组，再拼接全部已见类别。增加回归测试，
   保证训练新task不会改变旧类别的Router权重和融合结果。
3. 比较固定R1、修正后的taskwise R2和R3。只有R3同时超过R1与R2，才补seed1/2 validation；否则
   结束当前小样本离线Router路线。
4. 随后新建端到端三视图特征融合分支，seed0 validation最小比较J0固定三视图、J1样本级软路由、
   J3 Full+Person软路由；保持task独立冻结、分支辅助监督和现有冠军Adapter配置。
5. J1必须超过J0且J3，才补seed1/2并锁定一次新test。不得使用本轮test类别差值选择新权重。

当前已开始第1--3项实现，用户额外要求同时计算三种子探索性test。下一步提交推送
`codex/taskwise-three-view-router`，服务器新建独立worktree并运行完整单测、真实state/validation
reassembly smoke和test descriptor GPU smoke；通过后运行唯一完整批次。结果必须分别标注validation
决策与test诊断，test不能改变validation候选选择。

本阶段已完成。taskwise语义使R2 validation略超R1，但R3仍低于R1/R2；探索性test中R1仍以
`33.0119`第一。按既定规则停止离线小样本Router，不补其seed1/2 validation，不增加hidden、prior或
描述符维度。

下一步新建端到端taskwise三视图特征融合分支。先实现并在seed0完整8-task validation比较：J0固定
三视图特征融合+辅助监督、J1样本级软路由+辅助监督、J3 Full+Person软路由+辅助监督。固定冠军
Adapter、30 epochs/task和相同训练预算；每个task融合lane训练后冻结。J1必须同时超过J0和J3才继续。

已进入`codex/end-to-end-taskwise-three-view-fusion`：完成J0/J1/J3、三视图transform、有效Face
mask、0.1辅助监督、汇总器和launcher后，先运行完整单测与真实数据/GPU smoke。通过后使用三张
空闲GPU并行seed0完整8-task validation。输出进入`logs/emotic_track_a_end_to_end_view_fusion/`和
服务器`emotic_benchmark_runs/multi_lane_end_to_end_view_fusion_v0.1/`；不运行test或seed1/2。

当前159项全测和J0/J1/J3真实task0 GPU smoke均通过。下一动作是提交smoke记录、服务器ff-only同步，
确认三卡空闲后启动唯一完整batch；运行中只监测退出码、OOM、非有限值和任务进度，不重复启动。

唯一batch`end_to_end_view_fusion_seed0_20260910_005421`已启动，tmux
`multilane_e2e_fusion_20260910_005421`。等待三组各完成240 epochs后由launcher自动生成
`validation_summary.json`；未完成前不重复启动、不运行test。完成后同步result/control/logs并按J1同时
超过J0/J3的预设规则决定是否补seed1/2。

本批已完成：J0/J1/J3 final validation mAP为`42.5330/39.8569/38.3318`。J1未超过J0，
`advance=false`；不补seed1/2、不运行test，也不扩大当前Router。三组原始结果、日志和分析已同步。

下一步优先实现Full-anchored complementary residual，而不是继续让三路softmax竞争：

1. Full表征保持系数1和冠军主路径；Person/Face只能通过零初始化、小尺度、有界残差补充。
2. Person与Face使用各自task-specific轻量projection/Adapter，task结束冻结；无效Face残差严格为0。
3. seed0完整8-task validation仅比较A0 fresh Full anchor、A1 Full+Person residual、A2
   Full+Person+Face residual，训练预算和冠军Adapter协议一致。
4. 只有A2同时超过A0/A1才补seed1/2；否则保留独立专家固定R1正式冠军并停止训练级特征融合扩展。

当前已在`codex/full-anchored-complementary-residual`完成核心实现、三组launcher、严格汇总器、测试和
协议文档。下一步提交推送；服务器先审计全部worktree并创建新的独立clean worktree，运行完整单测、
真实A0/A1/A2数据通路和Image-token Adapter GPU smoke。全部通过且三张GPU安全空闲后，启动唯一
seed0 validation batch；运行期间不访问test、不启动重复batch。

当前修复后164项全测与三组真实task0 smoke均通过。下一动作是提交推送smoke记录、服务器实验
worktree ff-only同步，并在GPU连续空闲检查后用三张卡启动唯一A0/A1/A2完整batch。完成后只运行
严格validation汇总器；A2未同时超过A0/A1时必须停止，不得运行test。

唯一batch`full_anchored_residual_seed0_20260910_103042`已在GPU0/1/2启动，tmux为
`multilane_full_anchored_residual_20260910_103042`。当前只等待A0/A1/A2完成240 epochs，不重复启动、
不访问test。结束后核验updates/skipped/error及汇总器的`advance_A2_to_seed1_seed2_validation`。

本批已完成：A0/A1/A2 final validation mAP=`42.3799/41.8802/42.1556`，A2低A0`0.2243`，
`advance=false`。停止该训练级融合，不补seed1/2、不运行test、不搜索residual scale/bottleneck/gate/
auxiliary loss。固定R1继续保持正式冠军。

下一步若继续实现样本级动态三视图，先制定image-group cross-fitting协议而非立即训练：用K-fold独立
专家生成覆盖完整train的无泄漏OOF scores/质量特征，再训练跨task共享、类别无关且正则到R1的小Router；
Router不向专家反传，task lane训练后冻结。先进行3-fold成本与样本覆盖审计，通过后才启动专家训练。


## 当前优先：结束当前Router，验证固定R1稳定性（2026-09-09）

1. R1中心公平性修正已完成：固定R1、最佳R2、最佳R3 final val mAP为
   `42.7974/42.6804/42.6322`；动态Router正式停止，不补R2/R3 seed1/2、不运行Router test。
2. 不再扩大hidden、增加类别独立权重或搜索prior。动态路线只有在构造image-group cross-fitting的
   out-of-fold专家预测、显著增加无泄漏训练样本后才允许重启。
3. 若当前目标优先追求mAP，下一批应把R1作为固定三路候选：复用20260904已有seed1/2 Full/Person
   90/10 sources，只补seed1/2 Face 90/10 sources；锁定可靠Face权重0.20，不训练Router、不搜索权重。
4. 只有R1相对同seed R0在seed0/1/2方向一致，才进入100% fit的锁定三种子正式test准备；否则保留
   Full+Person，不继续Face静态融合。

## 已完成：三视图样本级Router validation（2026-09-09）

1. Face endpoint阶段已通过：固定beta0.20在完整seed0 val将FP final mAP从43.3035提高到43.5812，
   8个task均改善；当前不运行test，也不继续搜索静态比例。
2. `codex/three-view-router-validation`已实现、提交并Git-only同步；服务器146项完整单测、真实数据
   对齐、冻结CLIP描述符GPU smoke和Face source GPU smoke均通过。统一90% fit/10% image-group
   calibration；复用已有20260904 seed0 Full/Person source，只补同协议Face source。
3. 同一批专家比较R0固定FP、R1固定Face beta0.20、R2质量/三路预测统计软路由、R3加入低维视觉
   描述；R2/R3各只比较prior `{0,0.1,1}`，task独立hidden16、masked softmax、无效Face权重0。
4. task6 calibration只有51条：禁止扩大router、类别独立权重或直接用validation标签训练。报告
   calibration/validation gap、权重分布、逐task及质量分组。
5. 只有R3同时超过R1与R2才宣称动态视觉路由有效；再补seed1/2 validation，锁定后一次test。
6. 两GPU seed0完整validation批次`three_view_router_seed0_20260909_163655`已完成；Face source、
   描述符和修复后CPU Router均成功，全程未运行test。
7. 结果已同步：R3超过R2约0.0033但低R1约0.1268，不满足advance；按规则不补seed1/2、不test，
   也不扩大Router。只保留上节所述R1中心公平性修正。

## 已完成：Face endpoint seed0 validation（2026-09-09）

1. `codex/face-endpoint-validation`已实现并Git-only同步服务器独立worktree；142项完整单测、真实
   manifest/transform/score-dump检查与task0 GPU smoke全部通过。
2. Face expert已在GPU0完成：seed0、8 tasks、30 epochs/task、batch64、
   layer1/b32/LR4e-4/scale0.1/ReLU/independent、main BCE+Adapter ASL9.8/0/0.05、AMP/TF32。
3. 已复用既有Full/Person validation scores完成固定FP与可靠Face beta
   `{0,0.05,0.10,0.20}`比较；beta0.20胜出并允许进入动态三路Router，全程未读取test。

## 当前优先：2026-09-08三视图动态路由计划

本节覆盖以下历史“下一步”条目。详细计划：`docs/full_person_face_dynamic_routing_plan_20260908.md`。

1. 阶段0已全部完成：train/val覆盖率84.98%/86.32%，全部task/类别正例非零；完整产物、200张
   train抽检和真实Face→冻结CLIP GPU smoke均通过并同步。
2. 下一步新建`codex/face-endpoint-validation`：训练时只用有效且非歧义Face，Face loss mask；
   短边>=24且分数>=0.6才参与初始融合，其余完整回退既有Full0.8/Person0.2。
3. seed0完整8-task validation比较FP锚点与FPH beta `{0,0.05,0.10,0.20}`，不读取test、不搜索Face
   层数；只有完整样本池final validation mAP超过FP才进入动态三路路由。
4. 再做同协议固定三路、统计软路由、语义软路由；胜出后追加稀疏路由。保留calibration小样本限制。
5. 证据支持后统一task独立三路特征融合，完成辅助监督/Face消融、多seed及一次锁定test。
6. 停止共享Person CLS/query-only方向；原四组Person residual不自动启动。阶段0不读取test、不训练模型。

## 2026-09-07：补Full + 初始Person crop

1. 提交推送`exp/emotic-initial-person-fusion-control`的固定对照入口；服务器审计全部worktree并创建
   新clean独立worktree，禁止触碰`multi-lane-main-test-only`。
2. 服务器ddp环境的完整单测和真实EMOTIC legacy Person transform检查已通过。
   GPU当前繁忙；等待一张卡连续两次满足used<=2GB且util<=10%后自动运行Adapter GPU
   smoke，任一失败则不启动正式test。
3. 只训练seed0初始Person分支：8 tasks×30 epochs、13,950 updates目标、skipped0、无checkpoint。
   复用已有Full scores，固定probability 0.80/0.20一次，不在test搜索权重或阈值。
4. 完成后同步config/task metrics/history/summary、8份test scores、融合JSON和日志；比较初始Person
   与letterbox Person的独立指标，以及两者分别与同一Full融合的final/average mAP、cF1/oF1。

以上全部完成。结论是保留letterbox Person：初始Person固定融合`33.1429`虽比Full高
`0.6064`，但低于letterbox融合`33.2672`，且后者在8个task均领先。不在test上为初始
Person重新调alpha。下一个结构实验优先使用letterbox Person value/deep-feature residual。

## 2026-09-06 2×2完成后的执行优先级

1. 当前2×2已完成并同步：`32.5365/31.6826/30.5018/30.2505`。不补seed1/2，不继续搜索
   target-aware crop或query-only Person patches的层数、hidden、scale、LR。
2. 下次业务修改先修空Person mask fail-closed并补回归；真实train频率为0，因此只作正确性修复。
   同时记录有效patch数、Person残差范数及不同Selector注意力相似性，防止融合退化为共享常数偏移。
3. 固定legacy crop与冠军Image-token Adapter，运行fresh disabled、layer1 Person value residual、
   frozen deep Person CLS residual、task-adapted deep Person lane residual四组seed0完整8-task validation。
4. 按final mAP优先、average及task5--7辅助，不恢复逐类别硬门槛。只有超过同批disabled的赢家才进入
   下一步；不要以2×2的正交互`+0.6027`作为成功依据，因为绝对指标四组均未改善。
5. 对赢家再做当前task Person辅助BCE、等计算量ROI/Full内容残差对照；随后补seed1/2 validation，
   结构和参数锁定后仅做一次test。若三种内容写入均不胜，保留外部0.8/0.2固定融合并结束统一融合路线。

## 2026-09-06分析后的优先级（方案，未执行）

1. 等待当前2×2完整结果；01:52约task0 epoch14--15，不能提前定赢家。
2. 下一次修改优先修复空Person patch mask输出未归零，统计padding/裁剪回退；补固定ID旧类预测
   漂移评估。旧head在当前loss/Adam/WD0下并非持续被新任务梯度改变。
3. 先4组seed0完整val：Full锚点、修正query-only、同层Person summary value残差、深层Person
   特征残差。保持冠军Adapter，默认legacy crop；target-aware需val复核后才能替换。
4. 赢家再补ROI等预算对照及当前任务辅助监督/蒸馏，之后seed1/2与锁定test。
5. 详细参数、证据和协议边界见`docs/person_fusion_next_experiments_20260906.md`。以下为历史状态。

## 当前：selector-specific Person patches 2×2（2026-09-06）

- 新分支`codex/selector-specific-person-patches`已完成代码：`legacy/target_aware Full crop ×
  disabled/person_patches`四组，Person patch条件固定zero-based layer1。
- Person有效内容由14×14 mask定义，letterbox padding在attention softmax前屏蔽；每个Selector
  单独读取同深度冻结Person patches，task独立末层零初始化MLP只修改查询，不写回Full tokens。
- target-aware train在可行时保证bbox完整并随机取场景范围；eval在Resize256后选择包含目标的224区域，
  超大目标使用最大保留裁剪。legacy Full路径保持原像素/RNG。
- 固定实验：EMOTIC/seed0/100%train/8 tasks/30ep/task/batch64/main LR0.0125/cosine；Image-token
  layer1/b32/LR4e-4/scale0.1/ReLU/independent；主BCE+Adapter ASL9.8/0/0.05；AMP/TF32；无checkpoint。
- 因val/test训练耗时相同，用户要求直接test。本轮是结构探索，不依据test继续搜索crop/条件层/MLP参数。
- 本地与服务器127项全测、服务器真实EMOTIC transform smoke及两种GPU smoke均通过；运行源码
  固定`f1ada22`。用户确认可叠加后，四组已在GPU0--3启动并进入task0，tmux为
  `multilane_selector_patch_2x2_test_20260906`；首轮均84 updates/skipped0且显存仍余约9GB。
- 下一步等待四组各完成240 epochs/13,950 successful updates/0 skipped和8份test score，随后只同步
  JSON/NPZ/日志等必要小文件（无checkpoint），对2×2做主效应与交互分析。
- 完成后比较四组final/average/task5--7 mAP和逐task曲线，分离target-aware主效应、patch主效应与交互；
  只有person_patches超过同crop disabled且最好结果超过32.5365，才补seed1/2。

## 当前：Person-conditioned Selector代码实现（2026-09-05）

- 已按用户要求创建`codex/person-conditioned-selector`并实现第一版。
- 四模式disabled/bbox/person/bbox_person，条件hidden32/scale0.1/LR4e-4，仅layer1；
  冻结CLIP Person CLS、每任务独立查询MLP，原Image-token Adapter主模型BCE+ASL路由保持。
- 已添加成对输入/框坐标和可见性、零初始化与RNG隔离、旧任务冻结、优化器与score通路测试。
- 本地完整123项CPU回归通过，新增13项；Python编译、Shell语法与入口参数解析通过。
- 服务器123项、真实数据和三条件GPU smoke通过；代码固定`a8a759f`。四组seed0完整8-task
  test已在GPU0--3启动，tmux`multilane_person_conditioned_test_20260905`，批次
  `person_conditioned_selector_test_20260905`；现已全部完成并同步，240ep/13,950 updates/0 skipped。
- 实验入口为`scripts/emotic/run_multilane_track_a_person_conditioned_val.sh`；Git-only独立
  worktree、完整单测、真实配对数据和GPU smoke已经完成。
- 因val/test完整训练耗时基本相同，按用户明确要求，检查后GPU0--3并行四组seed0完整8-task test；
  对照条件化Full独立表现及原0.8/0.2 probability融合。该test标记为结构探索。
  fixed layer1/b32/LR4e-4、30ep、原100%train；不混入90%fit门控协议。
- 选择final val mAP优先，其次average/task5–7；逐类/F1/forgetting作解释，不恢复旧否决门槛。
  有效候选再补seed1/2。以下2026-09-04段落保留为历史结果。
- 本轮结果：disabled32.5365；bbox32.0791（-0.4575）；person31.1695（-1.3670）；
  bbox_person31.3557（-1.1808）。固定0.8/0.2外部Person融合也均低于原33.2672。
- 不再调整本版shared-delta的hidden/LR/scale。下一结构先实现target-aware Full crop并为其跑
  disabled锚点；再用Person有效patch summary按selector产生独立查询增量，回到seed0完整val。
  若仍不超过同批disabled及现有Full+Person val锚点，停止Person-guided Selector路线。

## 已完成：严格同seed Full第二次训练（2026-09-04）

- 用户已要求对标同seed双视图32.8263±0.4025；真实重训Full seed0/1/2，各自与原Full同global seed、
  同参数、同source提交。不能直接复制同一预测，也不擅自引入不同初始化/增强seed。
- 完整单测和原source Adapter GPU smoke后，在GPU0/1/2启动三组完整8-task正式test。
- 完成后自动汇总0.8/0.2同seed Full+Full并对照旧Full+Person，审计概率差异及除耗时外的训练轨迹。
- 同seed同视图可能退化为重复模型；需要按实际结果报告，不承诺集成增益。此前循环seed对照仍保留。
- 不搜索test权重、配对或新参数，不启动Person条件Selector。
- 批次`same_seed_full_full_test_20260904_122941`已在GPU0/1/2运行，tmux
  `multilane_same_seed_full_full_122941`。110项单测及两版GPU smoke通过，config与原Full完全相同。
- 三组已完整结束且自动汇总成功；240epochs/13950updates/skipped0、99项source哈希均通过。
  结果及日志已同步：同seed Full+Full32.1562±0.3701，全部task预测及训练轨迹（除耗时）精确重复。
  同seed Full+Person32.8263±0.4025；本轮仅验证重复性，不能作为独立随机集成无效的反证。
- 下一阶段建议（未启动）：停止重复Full与权重网格；最小Person条件Selector，seed0完整8-task val，
  比较原固定融合、bbox条件和Person内容条件；固定原100%训练数据/Adapter/ASL/优化协议及输出权重。
  有效候选再补seed1/2，锁定后一次test。报告成本，检验零初始化等价与旧task条件模块冻结。

## 当前执行：Full+Full固定正式test对照（2026-09-04）

- 用户已要求执行。复用三种子完整test scores，新增独立入口；完整单测通过后仅CPU离线评估。
- Full_s+Full_next与Full_s+Person_next使用相同next=(1,2,0)、固定0.8/0.2和阈值0.5。
- 核对原Full、Person、同seed融合锚点，报告final/average mAP、8-task曲线、F1、forgetting和来源哈希。
- 不重训、不搜索test、不启动Person-guided Selector；循环配对不视为独立重复显著性检验。
- 输出与日志：`output/emotic_track_a_ensemble_control_test/`及`logs/`；结果同步后分析。
- 以下旧任务描述按历史记录阅读，不代表仍待执行。
- 本项已完成：107项全测通过；Full+Full正式test32.6551±0.2622，匹配辅助seed Full+Person
  32.7957±0.2378，净增益0.1406且三个配对/24个task均正。完整结果已同步并哈希校验。
- 保留固定双视图，不再依据本次test搜索权重或seed配对；下一结构方向仍为待确认的Person条件Selector，
  本轮没有启动该方向或任何新训练。循环配对共享模型，不对三组数值做独立重复显著性结论。

最后一次更新：2026-09-04。

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
- 已补旧seed0 schema1缺少score-purpose元数据的受限兼容；首轮86项单测及三种子历史固定结果回归
  已通过，需在新增兼容回归测试后重跑全量检查并完成剩余smoke，再启动。
- 最终87项单测、历史结果回归、seed0 val复现、真实data/schema2 score smoke与Adapter GPU smoke
  全通过。clean`ebf49a2`上已启动batch`fixed_ensemble_control_val_seed12_20260903_191515`，
  4组在GPU0--3并行；本地日志镜像位于`output/emotic_track_a_ensemble_control/20260903_191515/`。
- 四份运行config已核对与seed0训练参数一致；初始task0均84 steps/skipped0且无OOM/数值异常。
- 下一步等待四组完成及自动固定对照汇总；核验每组240 epochs/13950 updates/skipped0、schema2
  probabilities与指标完全一致后同步。关注matched auxiliary person-minus-full的final/average mAP，
  同时报告同seed生产方案，不把共享模型的循环配对当作独立种子显著性检验。不自动开启test。
- 四组训练已成功但初次汇总路径失败：新runner保存`val_scores/`、汇总器读取旧
  `validation_scores/`。修复需同时支持新旧且拒绝二义目录，补测试后直接重跑现有scores汇总，
  不重训、不改权重；原launcher exit1必须保留并单独记录恢复汇总。
- 修复`343ddca`已完成并通过88项完整单测；恢复汇总成功。matched auxiliary对照显示Full+Person
  相对Full+Full的final/average mAP平均提高`0.3281/0.4616`，三个配对和24个seed×task mAP均为正。
  统计修复与等计算量归因两步到此完成；不启动新的正式test。
- 下一阶段优先复用已有validation scores和样本ID，补bbox面积/长宽比/多人物metadata诊断，并以
  全局person权重0.20为锚点设计强约束的类别/框质量感知门控。只允许跨三个seed方向一致的少量偏移，
  先在validation锁定规则；锁定后才做一次held-out test评估，禁止在test继续搜索。
- 若低参数门控不能稳定超过固定0.20，再实现person CLS/patch summary辅助selector的单模型结构；
  保留full-image视觉主干，并与双模型方案按相同推理计算量报告，不回到person-only替代路线。
- 当前已进入上述门控阶段：新分支`exp/emotic-constrained-gated-fusion`实现全局alpha 0--1步长0.01、
  bbox面积/长宽比/单多人固定分层、三seed一致类别/质量方向和9个小delta组合。下一步提交推送后在
  新服务器worktree运行完整单测；再执行一次validation选择，先落盘不可变选择JSON，然后只计算该
  锁定规则的一次三种子test。输出不得包含test候选表或权重搜索。
- 本阶段已完成：全局validation赢家为Person0.25；受限门控base0.20/class+0.05/quality±0.025
  进一步通过三seed门槛。锁定test为`32.8380 ± 0.4002`，仅比固定0.20高`0.0117`，不再细搜
  静态权重或bbox阈值。
- 下一阶段建议实现train/calibration监督的共享样本级可靠性MLP：输入bbox几何、两视图置信度/熵/
  disagreement，输出一个限制在0.10--0.35的Person权重并正则到0.20。禁止按test或未来task定规则，
  三种子validation锁定正则后只做一次test；若仍无实质增益，再转Person token辅助selector结构。

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
45. 完成样本级人物可靠性门控实现审查和回归测试；禁止继续搜索静态alpha、bbox阈值或类别独立权重。
46. 提交推送当前实验分支；Automatic Upload若污染服务器primary，先逐文件哈希、备份并恢复，随后
    只在独立`multi-lane-main-constrained-gated-fusion` worktree执行ff-only同步，禁止触碰test-only。
47. 服务器先运行完整单测，再做真实EMOTIC图像组calibration划分/score dump smoke、compact state
    restore与真实CLIP Image-token Adapter GPU smoke；任一失败则不启动完整实验。
48. 测试通过后用GPU0--5并行3 seeds×Full/Person共6组完整8-task validation source。固定基础协议为
    30 epochs/task、main LR0.0125、layer1/b32/Adapter LR4e-4/scale0.1/ReLU/independent、主BCE+
    Adapter ASL9.8/0/0.05、AMP/TF32；train图像组90% fit/10% calibration，无完整checkpoint。
49. 只在calibration训练8个共享门控候选，并在三种子validation选择prior；必须每个seed不低于同批
    固定0.20 final mAP才可锁定。若不满足，记录拒绝并停止，test保持未访问。
50. 若validation锁定赢家，使用per-task compact states对三个seed的Full/Person test各导出一次概率，
    只评估锁定赢家一次。同步JSON/NPZ/日志和compact states，不同步冻结CLIP或完整checkpoint。
51. `2e352e0`的97项单测和全部预运行smoke已经通过。下一步提交推送本次验证记录并让实验worktree
    ff-only更新，然后检查GPU0--5连续空闲两次，启动唯一一批learned reliability gate实验。
52. batch`learned_reliability_gate_seed012_20260904_021636`已正常启动且6份config/首轮通过审计。
    当前只等待，不重复启动、不修改运行worktree。完成后先核验6×240 epochs、每task calibration/
    val score与compact state、skipped0和错误日志，再读取validation selection；未通过门槛不得test。
53. 已结束但非全流程成功：六组source训练完整，门控因train-only geometry缺少val ID中断，test未执行。
    原始产物已同步且哈希一致；固定融合final val均值43.1073，比同批Full高0.7296。不要重训基础模型。
54. 待修：显式分离train/val几何输入，预检ID覆盖；增加不同split ID且禁止test访问的端到端测试。
    然后用新目录补原定8候选×3 seeds门控validation。当前仅完成结果分析，业务修复/补跑尚未执行。
55. 用户已授权补跑，修复分支`codex/fix-learned-gate-split-routing`已实现分split几何路由与5项回归
    检查。先提交推送、服务器新建clean独立worktree并跑完整测试，再用resume脚本只补门控阶段。
56. 补跑后报告8候选三seed完整曲线；尤其task6/7校准与validation的相对固定0.20增益，以及权重std/
    分位数/锚点距离。近常数标记仅作诊断，不新增否决门槛；通过既有规则才执行锁定test。
57. 修复及补跑已完成，102项单测/真实split覆盖/GPU smoke/保存compact的val概率精确恢复通过。
    20260904_113753八候选均不合格，最高均值43.106145低固定43.107346，按原规则未运行test。
    本轮停止，不继续扩大MLP、不放宽门槛。结果/192份gate states/日志已同步并通过哈希审计。
58. 若未来继续研究，另立阶段处理小样本calibration泛化和当前task更新对旧类排序的影响；当前保留
    固定Full0.80/Person0.20。不将未胜出的门控或validation指标描述为正式test新冠军。
-
59. 提交并推送`exp/three-architecture-locked-test`，服务器先审计全部worktree并创建新的独立clean
    worktree；禁止修改`multi-lane-main-test-only`及任何已有实验工作树。
60. 在服务器`ddp`环境运行完整单测；随后分别对J0、A2执行真实EMOTIC task0/1epoch test-score-dump
    GPU smoke，并对C2执行锁定状态/descriptor/R1复现只读smoke。任一失败不得启动正式批次。
61. smoke通过后确认7张卡各至少9GiB空闲，GPU0--5并行J0/A2×seed0/1/2六组完整test，GPU6并行
    C2三种子离线评估。允许与其他任务共卡，但首轮后必须复查显存、skipped steps和错误日志。
62. 完成后统一报告R1、C2、J0、A2的final/average mAP、cF1/oF1、forgetting三种子均值±样本标准差；
    明确这些是validation锁定后的探索性横向test补测，不基于test继续调参。
63. 206项全测、J0/A2真实test-score-dump smoke及C2只读评估已通过；下一步直接启动唯一正式batch，
    完成前不重复启动、不修改运行worktree。C2预结果已确认但仍由正式launcher写入统一控制目录。
64. 唯一batch`locked_architecture_test_20260914_013250`已启动并通过六组config/首轮/显存审计；当前
    只等待J0/A2各三种子完成，不重复启动。结束后运行launcher自动汇总并同步小结果文件，不同步checkpoint。
65. 横向test已完成：保留R1 `33.0119±0.3038`为实践冠军；C2仅高`0.0034`且seed方向不一致，按持平
    报告；J0/A2明确失败。若论文严格要求所有run零skip，仅复跑A2 seed1，否则不再调这三种融合结构。
# Shared multi-view DGL阶段0/1

1. 检查本地diff与新增测试后，仅提交本实验受控文件并推送`exp/shared-multiview-dgl-audit`；不得纳入
   四份用户未跟踪Adapter文档或`tmp/`。
2. 服务器先运行`git worktree list`并检查各worktree状态，创建新的独立DGL worktree；不得修改
   `/mnt/haoyuan/workspace/multi-lane-main-test-only`，不得覆盖主工作树未提交内容。
3. 在`ddp`环境运行完整单元测试；随后分别运行G0/G1/G2真实EMOTIC task0一轮GPU smoke，核对有限
   loss、0 skipped、G0非零梯度审计、G1/G2融合到表示的梯度范数严格为0、Face mask和诊断文件。
4. 全部通过后用三张空闲GPU并行启动唯一seed0完整8-task validation。输出到外部结果目录，日志和控制
   文件进入`logs/output/emotic_track_a_shared_dgl`，不保存checkpoint、不访问test。
5. 完成后按预注册四项门槛判断G2是否进入受约束动态Router；未通过则停止本共享DGL路线，不搜索alpha。
6. 首轮G0/G1 smoke已通过；先提交入口字段修复，服务器ff-only同步后重跑210项全测和G2 smoke，随后
   检查三组审计/诊断内容，再决定是否启动完整batch。
7. 上述复验已完成且通过。提交smoke记录并ff-only同步后，确认GPU0/1/2空闲，在同一clean HEAD启动
   唯一G0/G1/G2正式seed0 validation batch；不启动J1、不读取test。
8. 唯一batch`shared_dgl_seed0_20260914_215006`已启动；等待三组各完成240 epochs/13,950 updates，
   然后同步小结果文件并运行严格汇总。期间不重复启动、不调整alpha、不运行动态Router或test。
9. batch已完整结束并完成SHA-256同步审计。G2 final mAP `42.4147`低同批G0 `43.0754`达
   `0.6607`，average与Full单视图也下降，四项门槛全部失败；当前Shared-DGL实验到此停止。
10. 不补G2 seed1/2或test，不搜索alpha，不启动动态Router。保留独立专家固定R1作为当前三视图
    实践基线；若继续研究参数共享，须另立新阶段验证“冻结共享CLIP主干 + view-specific极小Adapter/
    projection + 固定融合”，先证明部分共享结构本身超过普通共享G0，再讨论DGL或动态路由。

# View-specialized Selector validation

1. `selector_validation_seed0_20260916` 已完成并通过哈希同步。S1 未过预注册门槛：禁止补 seed1/2、test、动态 Router、ViT 解冻及事后调参。
2. 保留 `exp/view-specialized-selector` 作为已验证的负结果分支；如后续研究共享 encoder 下的视图个性化，必须另立假设和预注册协议，不能把 S1 的结果用于选择新参数。
3. 当前三视图实践基线仍为独立专家固定 R1 的正式 test `33.0119±0.3038`；除非用户另行指定，不对已停止的共享多视图训练路线启动新作业。

## 2026-09-17：Selector 结果审计后的约束

1. 已确认 S1 下降没有发现工程索引或梯度接线错误；不要为解释该结果直接修改实现或补跑 seed1/2。
2. 若重新研究视图个性化，应先设计带共享基座的低秩视图残差、显式跨视图一致性正则或可学习可靠性门控，并预注册容量/正则/融合对照；不能把当前 S1 的独立 bank 作为正向结果。

## 视图专用下游组件验证

1. 在`exp/view-private-components`修复Prompt初始化维度与回归测试契约后，服务器`ddp`运行完整单测。
2. 通过后运行P1/P2/H1/H3代表性task0 GPU smoke，核对有限loss、0 skipped、视图路由和诊断输出。
3. smoke通过后启动唯一seed0的P0--P3/H0--H3 validation；结束后同步小结果文件并按预注册门槛分析，
   不访问test、不保存完整checkpoint。
4. 先完成smoke入口变量修复后的四组AMP smoke；若均通过，再启动正式八组validation。
5. 等待batch `view_private_components_seed0_20260917_102434`完成；同步P0--P3/H0--H3结果、日志和控制文件后再做预注册门槛分析。

6. batch已分析：不运行test，也不基于本批继续做后验调参。若用户希望确认“部分缓解”，另立并预注册P2固定配置的seed1/2复现；若研究all-private组合，先增加task3的参数/梯度有限性审计，再讨论学习率、AMP或裁剪策略。

## P3稳定性修正版

1. 已完成：`exp/view-private-components-p3-stable`上的FP32 P3通过服务器全测、task0 smoke及完整8-task validation；240 epochs、13,950 updates、zero skipped，数值稳定。
2. 已完成叠加分析：P3 final mAP `40.9340`，低P0/P1/P2 `0.4516/0.9096/1.3312`，并且Person/reliable-Face相对P0下降超过0.50；不满足预注册推进条件。
3. 不运行P3 seed1/2或test，也不继续Prompt+独立Adapter的all-private组合。P3使用FP32、旧臂使用AMP，不能把本结果作为无混杂的精度控制；若将来重新研究机制，须先另立含FP32 P0/P1/P2对照的预注册协议。当前不自动启动新实验。

## FP32交互项与选择性 Prompt（当前任务）

1. 提交并推送 `exp/view-private-components-fp32-selective-prompt`；服务器先审计 worktree，创建独立 clean worktree，不修改 `multi-lane-main-test-only`。
2. 在 `ddp` 环境运行完整单测与选择性 Prompt/task0 smoke，核对初始化等价、Person 路由和 zero skipped。
3. 通过后启动唯一 `launch_multilane_track_a_fp32_interaction_selective_val.sh`，运行 F0/F1/F2/F3/S1；仅 validation，不访问 test、不保存 full checkpoint。
4. 同步结果后运行 `summarize_fp32_interaction_selective.py`，报告 final/average mAP、cF1/oF1、forgetting、逐 task mAP、交互项及 S1 相对 F2 的变化。
5. batch `fp32_interaction_selective_seed0_20260917_230307` 运行中；完成后同步 runs、logs、status，逐文件校验 SHA-256，并运行交互项汇总器。

6. batch 已完成并同步。F3 final 交互项仅 `-0.1121`、average 交互项 `+0.8438`，不支持简单的“结构负交互”结论；S1 相对 F2 final `+0.1659` 但 average 持平。当前不运行 test；若继续，应先做多 seed 或更小容量/晚层 Prompt 残差的预注册验证，并重点处理 Person 分支。

## 共享基座视图残差与晚层 Prompt（当前任务）

1. batch `shared_residuals_late_prompt_seed0_20260918_103705` 运行中；完成后同步 output/log/status，校验 SHA-256。
2. 汇总 R0/R1/R2/R3 的 final/average mAP、cF1/oF1、forgetting、逐 task 和 Full/Person/Face/reliable-Face 指标。
3. 按预先定义的机制比较判断：Selector 残差、Adapter 残差、晚层 Prompt 残差是否分别改善共享基线；不根据 Person 单独定制门槛，不访问 test。

## 共享基座残差与晚层 Prompt（已完成）

- 已完成并同步 batch `shared_residuals_late_prompt_seed0_20260918_103705`；R0/R1/R2/R3均通过完整 validation，服务器与本地产物 SHA-256 一致。
- seed0 结果不支持 Selector residual、Adapter residual 或晚层 Full/Face Prompt residual 优于共享基线。当前不启动 test，也不直接扩大残差容量。
- 后续候选：先固定 R0/R2 的表示，单独研究可学习融合权重、按样本可靠性校准和三路 logits/概率温度校准；若要声称结构改进，先预注册 seed1/2 复现并保持 FP32/TF32 配置一致。

## R0 与动态三视图融合 test（用户已要求）

1. 已确认动态 `soft_three_view` 机制和当前 R0 其余配置一致：task-local hidden-16 masked-softmax、固定先验初始化、可靠 Face mask。
2. 已新增独立 R0/D1 test runner/launcher；两组仅改变 fixed/soft fusion，保持 shared Selector/Prompt/Adapter、FP32/TF32、seed0、8 tasks、30 epochs/task、batch64、auxiliary0.1、无checkpoint和无test搜索。
3. 历史动态 J1 validation 曾低于固定 J0；本次 test 为用户明确要求的诊断性比较，结果不能用于test后再选模型或调融合权重。

## R0 与动态三视图融合 test（已完成）

- R0/D1 test 已完成并同步。R0 final/average mAP=`31.9883/38.8315`，D1=`29.5224/36.1115`；D1 final 低`2.4658`，average 低`2.7200`，8个task全部落后。
- D1 权重从固定先验明显漂移到 Person 主导，说明当前 joint-gradient Router 受主损失驱动而缺少先验/校准约束。该结果不支持直接采用动态融合，也不支持在 test 上继续调 Router。
- 后续若保留动态方向，只能回到 validation 预注册：image-group OOF/calibration、KL/prior 约束或小幅受限 residual，并补多 seed；当前固定 R0 仍是共享基座下更可靠的融合方案。

## ParaX shared level routing (2026-09-23)

1. Commit and push `exp/parax-level-routing`, preserving unrelated pre-existing local docs/tmp.
2. On the server, inspect worktrees and create a clean Git-only checkout; run full ddp unit tests and six real ViT task0 GPU smoke arms.
3. If smoke passes, launch the declared seed0 validation screen with B0, P-post, P-10, P-8:10, P-8:10-level, and Static-control; use validation only, no test or full checkpoint.
4. Report mAP, per-view performance, gate entropy/top-expert frequency, cross-view gate distance, residual/token norm ratio, gradient/CLIP-freeze checks, memory and forgetting before any rank64 or seed1/2 follow-up.

## ParaX seed0 validation in progress

- Batch `parax_shared_level_seed0_20260923_20260923_2045` is running six arms on GPUs 0--5 (rank32, official initialization, fixed-three-view, 8 tasks × 30 epochs); first epoch completed in all arms with zero skipped updates.
- Do not launch duplicate runs. Wait for all six status codes; then sync run outputs and logs from server to local output/logs, verify SHA-256, and analyze final/average mAP, per-view metrics, gate statistics, drift, speed/memory, and forgetting. Test remains forbidden.

## 2026-09-25：B0-preserving logit residual control

1. 在`exp/fixed-fusion-residual-calibration`只提交本次model/runner/view-fusion、测试、两份入口、协议和三份上下文文档；不得纳入用户已有未提交docs或`tmp/`。
2. 推送后在服务器先运行`git worktree list`并创建新的clean worktree；不得修改`/mnt/haoyuan/workspace/multi-lane-main-test-only`。
3. 已完成：`ddp`完整242项单测、目标23项复验与真实EMOTIC task0短smoke全部通过；zero-init对齐、B0/calibration梯度隔离、52参数/task、finite非零系数和zero skipped成立。
4. 已启动唯一batch`logit_residual_control_20260925_163436`：B0-paired与L-post-logit，seed0、task0--2、30 epochs/task、batch64、auxiliary0.1、AMP/TF32、validation-only、无checkpoint/test；首轮均zero skipped。等待自然完成，不重复启动或持续监控。
5. 晋级要求：L-post-logit task0与average不低于paired B0、final至少`+0.10`、forgetting增加不超过`0.10`，且系数既非全零也不饱和。失败则停止当前residual/calibration路线，不加level embedding、ParaX experts、蒸馏、seed1/2或test。

## 2026-09-25：共享Selector + 三路独立完整Adapter

1. logit residual已结束：final只比B0高`0.0857`，不补seed1/2或test，不继续调校准超参数。
2. 在`exp/shared-selector-independent-adapter`只提交本实验代码、测试、入口、协议、结果报告和三份上下文文档；不得纳入用户既有未提交docs或`tmp/`。
3. 推送后先审计服务器`git worktree list`，创建新的clean worktree；不得修改`/mnt/haoyuan/workspace/multi-lane-main-test-only`或服务器主工作树。
4. 在`ddp`环境运行完整单测，再对A0共享b32、A-cap共享b97、A-view独立三路b32分别运行真实EMOTIC task0一轮smoke。必须核对参数量`49,952/149,857/149,856`、A0/A-view初始对齐、有限非零梯度、CLIP冻结和zero skipped。
5. smoke全部通过后启动唯一seed0 task0--2 validation：30 epochs/task、batch64、main LR0.0125、Adapter LR4e-4、BCE/ASL、auxiliary0.1、AMP/TF32、validation-only、无checkpoint/test。
6. A-view只有在task0不低A0、task2分别至少高A0/A-cap `0.10`、average不低A0、单路与forgetting保护均通过时，才进入完整8-task validation；失败则停止，不加Router、level embedding、更大Adapter、多层、seed1/2或test。
7. 已完成服务器245/245全测和三组真实task0 smoke；下一步在同一clean分支启动唯一正式task0--2 batch，训练期间不持续监控，结束后统一同步分析。
8. 正式batch`shared_selector_independent_adapter_20260925_1803`已启动并通过首轮审计；不要重复启动、不要中途调参或读取test。完成后同步三组JSON/NPZ/log/status并逐项校验SHA-256，再按预注册六项门槛决定是否进入完整8-task validation。

## 共享 Adapter 容量完整 validation（2026-09-25）

- 已创建分支 `exp/shared-capacity-lane-freeze`，新增 `scripts/emotic/run_multilane_track_a_shared_capacity_full_validation.sh` 与双 GPU launcher，以及 `docs/shared_capacity_full_validation_plan_20260925.md`。
- 锁定 A0 shared b32 与 A-cap shared b97，seed0、EMOTIC Track A、8 tasks、30 epochs/task、batch64、BCE+ASL、auxiliary0.1、AMP/TF32、validation-only；不加 private view、Router、level embedding、蒸馏或 test。
- 下一步：提交并 push，服务器创建 clean worktree，运行完整单测和两组 task0 smoke；通过后启动唯一正式 batch，训练结束再同步分析。

## 共享 Adapter 容量完整 validation 已启动（2026-09-25）

- batch `shared_capacity_full_validation_20260925_2015` 已在服务器 GPU0/1启动；不要重复启动或中途调参。
- 完成后同步两组 JSON/NPZ、日志和 status，核对8 tasks、240 epochs、13,950 updates、zero skipped、无OOM/NaN/traceback，再判断 b97 是否跨后期任务保持收益。
- 2026-09-25: shared-capacity full validation ended with A0 complete and A-cap failing at the start of task3 cycle18, after 17 completed cycles and 140 skipped AMP updates. A-cap's task0--2 gain is real but is not a full-sequence result. Full analysis: `docs/shared_capacity_full_validation_results_20260925.md`.
- Active batch `shared_capacity_residual_scale_validation_20260925_222440` on `exp/shared-capacity-residual-control`: seed0, EMOTIC Track-A tasks0--7, validation-only, b32 vs b97, residual scale `0.03`, 30 epochs/task, batch64, main LR0.0125, Adapter LR4e-4, BCE+ASL, auxiliary view loss0.1, AMP/TF32, no checkpoint/test. Full 245-test suite and both real ViT task0 smokes passed; each arm entered cycle1 with 84 updates and zero skips. Wait for natural completion before analysis; do not test or tune during the run.


## 2026-10-10 ParaX 受保护专家复用三任务 pilot（用户授权执行）

- 新分支 `exp/parax-protected-expert-reuse`，起点 `8b6475d`；新增独立专家参数、task-local Router/零输出投影、永久访问 mask、task结束封存与旧路径逐值审计。
- 四组：无ParaX基线、task0学中心后冻结、只用新专家、复用旧专家并学新专家；两个扩展组每任务新增2个rank32参数矩阵对。
- 接入最终归一化 Task Forward CLS 后、固定三视图特征融合前；CLIP冻结、Image-token Adapter/Projector关闭；task0–2 seed0 validation，30epoch/task，batch64，base lr0.0125/ParaX lr0.0004，Adam/cosine/WD0，smooth bound0.02；无test。
- 本地 compile/bash 检查已完成，单测与真实数据smoke待运行；服务器SSH已恢复，GPU0已有其他任务，禁止中断，按实际显存决定启动或有限排队。尚无新实验指标。
- 完整设计与输入/输出/日志见 `docs/protected_parax_expert_reuse_pilot_20261010.md`。保留本地既有未提交文档及远端主/test-only工作树改动。


### 2026-10-10 Protected ParaX preflight and GPU0 launch

- Branch: exp/parax-protected-expert-reuse. Launch commit: 3aa6447; model/audit code fd1cde4. Remote clean worktree: /mnt/haoyuan/workspace/multi-lane-main-parax-protected-expert-reuse.
- Full remote unittest: 282/282 passed; seven new local mechanism tests passed. Four-arm smoke protected_parax_smoke_20261010_02 completed tasks0-2, 12 updates/arm, zero skips, no NaN/OOM. Initialization/sample-ID pairing and frozen CLIP/protected old parameters passed.
- New expert A/B and Router gradients nonzero/finite; initial logit difference about1e-7, fixed old-anchor logit drift below1e-6, comparable to baseline numerical floor. Audits disable TF32 and restore training flags.
- Smoke peak reserved memory2756MiB; four arms fit GPU0 alongside existing work. Formal minimum free memory14500MiB, finite queue if needed, never stop existing processes or useGPU1-7.
- Formal batch protected_parax_expert_reuse_seed0_val_20261010_01; tmux ml_protected_parax_20261010. Baseline/frozen_pool/fresh_only/reuse_old run concurrently onGPU0, seed0 tasks0-2,30epochs/task, expected5010updates/arm, validation only, no test.
- Results: output/emotic_protected_parax_val/<batch>/<method>; logs: logs/emotic_protected_parax_val/<batch>. Smoke scores are implementation checks, not efficacy evidence. Do not continuously monitor formal training.

四组GPU0 smoke已全部退出0，各2成功更新/0跳步，训练人群checkpoint重放
mAP差均0。三个Adapter组梯度非零/有限，dynamic Person CLS残差/输入比
约0.00103；这只验证路径。配对审计脚本初次把tasks字典误当列表，已按
现有schema修正，待真实产物审计通过后启动正式批次；模型/训练代码未变。


### 2026-10-17 梯度诊断预检

独立诊断分支已通过Git-only同步；服务器311项Track-A测试通过。首轮smoke
在配置校验阶段因归档字段名parameter_group_loss_routing误写而退出，未计算梯度、
未更新任何参数。已按真实config修正，并增加缓存标签SHA复核；使用新run ID重试。

第二轮smoke发现三视图归档将Face单路crop参数记为null；诊断数据入口现使用
与原多视图路径一致的EMOTIC默认值，实际Face裁剪/可靠性仍由manifest及ThreeViewTransform决定。
此前两轮均在加载数据/模型前停止，无梯度或权重更新。

第三轮真实smoke（clean b8107df）已完成四组task0×两数值配置共8快照，
无训练更新。新增离线汇总器及缺失梯度/重复冲突门槛测试；正式诊断待汇总器
远程测试后启动，固定配置仍为task0/2/7×每任务4批×四组×两精度。

AF离线核验首次在epoch元数据比较处停止：smoke按updates预算，归档epochs_per_task为null；
已修正仅正式epochs预算与30epoch基线核对，更新预算仅允许少于8task的smoke。
不改变模型/冻结机制，无训练重跑；其余来源、checkpoint核对继续执行。

2026-10-19预检：81b54b1已Git-only同步新服务器worktree；新增测试4通过/1入口错误，
单视图探针误调用要求三视图字典的融合API，修正为单视图内部接口；无模型实现改动、尚未smoke。

2026-10-19：883868f完整364项Track-A测试通过（30.623s），GPU0两任务各2步smoke已启动。
首任务2更新/0跳步、gate-mass冻结3名成功，验证前恢复状态已保存。新增只读真实图像零输出
恒等与8lane形状/资源探针；后6任务Router未训练，明确仅资源检查、不报告准确率。


## 最新执行计划（2026-10-20，Frozen路由归因）
1. 远程独立worktree通过全部Track-A测试及9条件GPU0 smoke，动态重放原val逐值一致，模型/源SHA不变。
2. GPU0正式task0/2/7三条件完整validation，TRAIN锁表在任何val读取前完成；no_grad、0训练更新、无test，bootstrap500。
3. 同步完整日志/分数及SHA，比较dynamic−fixed/off及Full/Person/可靠Face/融合。未通过内容路由收益门槛不追加训练。
4. 独立image_token Adapter当前关闭、与MoE模式互斥；是否叠加在MoE单独有效后另立对照，不能本轮混入。

2026-10-20预检：首次完整367项测试中366通过，新增schema恢复测试的合成训练前向误用no_grad，按设计不累计训练激活，finish_task因此拒绝空计数。已修正测试准备步骤，不改变模型/真实诊断逻辑；未启动真实smoke或正式验证，失败日志保留。
