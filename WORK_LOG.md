# 工作日志

## 2026-10-30：条件传递两个单因素版本完成及预检

ffab151（codex/person-query-conditioning-20261030）Git-only部署独立worktree。新增投影RMS上限1与独立相对Query偏移增益32/上界1%，各有Pool/Attention，无参数增长/组合修复，原三路训练及最终1%修正保持。11新+44旧测试全通过无skip，含实际CUDA AMP。

GPU0最大1并发完成四组task0/1各4主+4私有更新，共32/32、0skip；4 TRAIN标定/Val首64完成。8份arm-task原参数/loss/增强输入/OFF及三路预测精确相同。TRAIN每组62人/55供体及64人/58供体，native/lateQuery/zero-condition控制均通过；人物交换Query差异仍仅0.000252%–0.001389%，尚无人物绑定或分类收益证据。投影used z RMS约<=1.00000009、used饱和0；增益未改投影。短训练不作正式效果依据。

旧48checkpoint SHA不变，main/test-only状态不变，未启动完整8任务/Test。327份非权重结果日志已回传逐SHA核验至output/person_query_conditioning/preflight_snapshot_20261030/snapshot，归档af94f8f36ad4b53363cfae7b9e8adc6a0844f65dd835418cc61e9edc957e0fcf。三上下文和详细执行协议同步，未merge/删改数据/触及用户dirty。

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

2026-10-28：按用户要求逐行核对protected_joint_attention/model/runner数据流，说明第6层摘要缓存→第12层CLS→单Query联合读出→末端同头、任务参数与SG分工。明确Layer6 Selector是该层由共享初始Selectors演化的任务查询，本轮Drop & Replace保留同层共享查询；目标框不直接拼Query、实际C4门控用h0而非单Full；修正方案公式，记录1538→16→1门控与固定.1门控下1%特征RMS上界。原共享三路BCE/ASL未切断，仅新增目标隔离。只更新解释文档和三上下文，无模型/实验/调度/源码提交变化。

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

## 2026-10-28：围绕Attention研究单模型交互摘要与原函数保护

用户以完整A/B均值Val44.1550/51.2964、Test33.6248/40.5129为依据，希望沿Attention增加合理约束与融合且不做多模型组装。检查真实均值成本/旧OFF控制、原Attention零输出/Norm/限幅实现、独立Selectors/冻结底座旧失败、近期表示×head诊断和2026-09-25原B0保真校准，阅读MULTI-LANE/MulT/Flamingo/GMU原始论文。确认均值证据支持函数互补，不足以证明Attention或Face单独贡献；保护与后置残差并非新机制。

新增docs/protected_joint_attention_plan_20261028.md：保留强B0原三路共训任务表示及原损失，新增私有联合摘要读出，目标区域并行Full Summarize/人物与任务条件Query/可靠Face mask/内容门控有界残差；末端同头、原30轮从头自行学习，无A/B任务权重或预测。原参数与新增目标梯度隔离，需RNG/Adam/AMP完整重放；保护原三路不等于保证新融合泛化。与旧零初始化、后置校准、5轮只训Attention均逐项区分，列出C0–C4/Full-source-only、内容和人物交换控制、成本与停止条件。

本轮仅代码/结果阅读、论文核对、方案与三上下文更新；未修改模型/训练代码、无GPU/新指标/提交推送，不触碰其他工作树或源checkpoint。仍在分析分支codex/teacher-free-epoch-head-attribution-20261028（HEAD09a6729），保留全部此前dirty文件。

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

## 2026-10-27：无外部教师的共享三视图方案研究

用户收紧最终目标：自己训练的单模型，在原30轮内保Full并学Person/Face互补；不能依赖A/B输出或参数拼装。检查最新随机流精确恢复、30→35归因、DGL/梯度审计、独立Selector、旧OOF排序失败与当前runner损失/参数路由，阅读辅助梯度相似性、PCGrad、ReconBoost、Gradient Blending原始论文。

新建规划分支codex/teacher-free-shared-view-plan-20261027（HEAD28a1cbb），保留既有未提交结果分析。新增`docs/teacher_free_shared_view_learning_plan_20261027.md`，提出保留原B0前向/共享参数与BCE-ASL；新增Full条件真实标签排序，通过Person/可靠Face路径反传，同时对新增共享更新做分组与实际Adam候选步限制。当前头全30轮学习，单一CLIP预训练+fresh任务参数；无教师、旧头交换、新Attention、额外5轮。明确保护只是本batch一阶相对步约束，不保证泛化/最终AP。U0–U4与阶段门槛、旧失败区别、互补证据/剂量控制已列。

远程在历史RNG独立worktree仅做TRAIN标签/Face元数据统计，无GPU/图像前向/训练更新；B64独立模拟shuffle，task0/2/7可靠Face正负对每批均有，同图人物自然同批稀少。计数JSON回传output/teacher_free_shared_plan/train_pair_feasibility_20261027.json，SHA256 ceea9e58fd7c54b25e74360f498f121dfd7a3b6a10561812d67ecdf319232071。它仅支持采样可行性，不证明优化有效。

本轮没有训练业务代码修改/模型实验/提交推送合并；仅文档与既有context更新，后续用户要求执行后再实施和GPU0验证。此前dirty分析文件原样保留。


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


## 2026-10-26：针对Face共训代价的研究方案与只读分路组合

03补齐500次原图配对结果：Person替换Test最终[.0446,.2994]、平均[.0369,.3115]均正，双换最终[.0088,.3810]、平均[-.0345,.4124]。修订优先级为真实验证R2且保留R1/R3，避免只按最终最高选择；明确正式来源需B0头/Person/Dual共90epoch/任务，推理成本另测。本轮仍无真实模型执行或新训练。

用户仅要求研究解决方案。本地创建codex/face-protected-fusion-plan-20261026保留全部既有未提交文档。读取旧失败的Face残差及OOF跨视图蒸馏结果，查阅MULTI-LANE、Learning without Forgetting、Gradient-Blending等原论文。没有训练或真实模型前向、没有GPU作业/远程修改。

新增scripts/emotic/analyze_protected_face_swap.py；对强Person和双Attention同阶段B0头下48个原NPZ做完整8阶段CPUlogit组合，固定旧融合权重，Face lambda1为预先主比较并报告0/.25/.5/.75/1全网格，原IDs/标签/mask/SHA保持。首轮逐值断言拦截NumPy/torch sigmoid一ULP差，修正零扰动人群直接保留原概率，保留失败日志；02完成Face区间，03补Person/双换配对区间。无权重更新。

Face替换Val43.1784/50.4692、Test32.9419/39.9239（相对强Person+.0554/+.0552）；Person替换43.4224/50.5673、33.0549/40.0482（+.1685/+.1795）；双换43.4681/50.7165、33.0608/40.0413。Full替换控制33.0472/39.9369说明低单路AP不等于负融合贡献。Face原始条件delta移植32.9117/39.9297。Face替换bootstrap最终[-.0676,.1918]、平均[-.0809,.2360]均含0；全换B的AP重构差.0036且sigmoid最大.000368，均明确是离线信号，非新模型成绩。

保存docs/face_protected_fusion_research_plan_20261026.md，包含全网格/所有控制、机制修正、完整来源任务路径复用方案、B Full摘要依赖、R0–R4无训练验证矩阵、成本/预测保护边界、后续双教师软约束及门控，均未启动。output/face_protected_fusion/readonly_20261026_03和logs/face_protected_fusion记录完整产物，三个上下文同步更新。

## 2026-10-26：两组8任务结果回传、CPU复算及机制分析

用户要求检查已完成实验，未要求新训练。本地创建codex/selector-face-analysis-20261026并保留4份既有未提交文档。远程12作业exit0、240epoch/13950更新和额外40epoch/2325更新均无AMP跳步；归档576个非权重文件、SHA cadada4716c97a241e298326d6e1f01eb611e7690d2769c60899ad852022fb12并回传逐项验证。24源和16新checkpoint SHA复查，远程f6ef130 clean，主树/test-only原HEAD和dirty状态SHA一致。

新增CPU分析脚本analyze_selector_face_controls.py及docs/selector_face_controls_results_20261026.md。208阶段融合/832视图独立复算最大误差7.11e-14；80旧参考NPZ全字段逐值、32完整保护与16Face mask检查通过。补入旧双Attention原共训头作为匹配控制，避免将独立S原头与旧B0头混合比较；500原图配对阶段耦合bootstrap，数据/源权重零修改，无GPU新计算/训练。

主结果（Val最终/平均；Test最终/平均）：独立S+B0 38.6066/46.9717；29.2649/37.7092；独立S原头40.9621/48.1452；30.7724/38.2750。旧共享双M原头43.1264/50.3256；31.9459/39.1951，B0头43.3394/50.5522；32.7573/39.7004。新S全OFF Test29.2269，较旧全OFF−3.4651，解释大部分总差−3.4924；三路均退步而训练目标降低，取消共同查询约束/读出兼容性是合理解释但不当作已证实唯一因果。A/P与Full K/V梯度仍共享，task0已下降，不称完整梯度隔离或灾难遗忘。

冻结强Person新Face Val43.0787/50.2760；Test32.8792/39.8493，旧强Person43.0741/50.2828；32.8864/39.8687。完整预测保护和OFF恢复精确通过；Face确实更新，可靠子集Face AP+.0261、融合+.0129，但跨人群排序使全体−.0072，八阶段Test均微降。新增Face的Query不含当前Face内容，Full K/V上下文并不自动提供人物对应和错误互补。Test最终bootstrapCI[-.0333,.0185]不支持正收益。保留强Person及B0，停止扩本轮两结构，后续需新授权再启动研究实验。


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

2026-10-23 18:48 CST：五组剩余训练已全部GPU0并行恢复，PID T_BCE1121327/T_ASL1121349/ATTN1121367/POS1121389/BOTH1121407，均独立session且已进入新的训练epoch，首轮0跳步。当前运行模型HEAD86a8c1c clean，调度脚本独立Git工作树HEAD6eb1e3e；新run_id=recovered_gpu0_seed0_20261023_01，日志/输出均在对应strong_full_asl_three_view目录；原三组已完成Full产物只读链接复用。五组边界validation的全部npz字段（含logits/概率/人群/标签）与原输出逐值相同，已重算的前2/5/4/1/1epoch训练loss/LR/步数均与原轨迹日志一致（耗时除外）。这支持当前恢复正确，不等于未完成部分已验证。

恢复前完整状态校验5/5、调度测试4/4、真实tmux父会话关闭后独立子进程存活探针均通过。原源恢复checkpoint SHA保持，primary/test-only状态仍153/5行且SHA不变。19份故障/预检/恢复启动元数据已同步本地逐SHA核验，归档1001319ab44360837744f1631cb58be0b9886dc480822284881c9438619abf6a。全部后续训练/val/test/6项关闭归因只GPU0，至多5总进程/2评估，显存不足自动等位；旧多卡ETA不再适用。此条覆盖前次“尚未恢复”状态，不掩盖旧tmux接管造成的中断与重算。

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

## 2026-10-23 14:28 CST：完整八组validation/test、运行时归因与CPU复算

用户要求仔细分析已结束实验，包含validation/test及结构原因。formal_seed0_20261023_01已完整结束：八组各240epoch/12660更新/0跳步，16路完整validation/test及F111 task0/2/7四路关闭归因全部exit0，completion为complete、无failure。远程执行树clean、HEAD0fb85f1，原primary153行/test-only5行及SHA保持。598份非checkpoint产物同步本地并SHA全等；128份checkpoint仅记录SHA未下载。独立CPU复算652项逐类AP最大误差3.98e-13，八组task7 val存档逐值重放；本轮无新模型前向/训练。

按基线/A/B/C/AB/AC/BC/ABC，最终val为42.5650/42.4311/42.4323/42.6478/42.9530/41.8702/42.4916/42.3712；最终test为31.9391/31.8358/32.2804/32.1179/32.4286/31.8614/32.2555/32.1765。AB最佳最终val/test，较本轮基线+.3880/+.4896，阶段平均48.5646/35.7581（基线48.5026/35.4273）；C单独val平均48.8647最高，ABC test平均35.8087最高。不能称AB四指标全面最优，也不支持三模块普遍叠加。

ABC−AB的task6三类对26类mAP贡献val−.6332/test−.3875，其余23类分别+.0515/+.1354；两split都集中小样本task6。F111 task7关A/B/C/全关val为42.3373/42.2222/42.3755/42.2100（全开42.3712），关C未恢复AB42.9530或task6三类。主要差异留在共同训练的任务参数，不能断言即时位置bias过强或旧任务漂移。AB的Full两split变强，Person端点未同步增强，融合−Full只小增；不能宣称人物对应/互补已解决。

1000次原图配对bootstrap：AB−基线val95%[−.3395,+1.0274]、test[+.0793,+.8921]；AB−Attention的test区间跨0，ABC−AB与全部F111开关区间均跨0。仅固定seed0的样本不确定性、无多重校正，不代表训练seed稳定或机制已证实。不要用历史全TRAIN B0替代本轮90%fit匹配基线；历史test阶段平均36.2781仍高于本轮所有组。

新增docs/three_module_stacking_complete_results_20261023.md，包含全部8阶段val/test、参数/内存/开关/子集/任务6逐类/置信区间与后续控制。CPU分析脚本和JSON在output/three_module_stacking/analysis_20261023_complete_01；使用本地NumPy运行，不依赖Torch。已同步检查更新三个上下文、执行记录及阶段报告覆盖说明。保留所有历史/用户未提交文档；不改本地主树及远程代码、未运行新的训练或GPU前向，也未提交推送。没有业务实现变化，故未重复模型单元测试；完成存档复算和git diff检查。

## 2026-10-23：三模块训练归档只读核验与阶段性结果分析

12:07补记：F010完整validation也已结束，最终42.4323、阶段平均48.3613；新增17份正式val产物同步本地并SHA全等（合计309份），task7与训练归档逐值一致。当前3/16完整评估作业结束，test及关闭归因仍未齐全。

用户要求分析结束实验。检查确认八组训练均完成，但正式val/test及F111关闭归因仍在运行，已明确告知，未终止/迁移/重启任何作业。12:02远程F000/F100完整val结束，其余仍自动排队，无completion/failure；执行树clean、0fb85f1，原primary153/test-only5行及状态SHA保持。

同步292份已完成非checkpoint产物并全部SHA核对，0份权重下载。八组task7完整val2397人/1705原图对齐，直接使用归档probabilities独立CPU复算，与训练summary差<1e-8；F000/F100正式task7归档逐值一致。八组各240epoch/12660更新/0跳步、冻结/holdout/恢复核验全部通过。分析脚本初次从package导入触发本地缺失Torch，改为runpy加载纯协议并复用AP公式后完成，无模型前向/参数更新。

最终val基线42.5650，A42.4311/B42.4323/C42.6478/AB42.9530/AC41.8702/BC42.4916/ABC42.3712。1000次配对原图bootstrap所有对基线区间均跨0；AB+.3880区间[−.3395,+1.0274]，ABC-AB−.5817区间[−1.3295,+.1306]。二阶AB+.6545/AC−.6438/BC−.0236、三阶+.0028均不确定。三模块回落主要task6：三类贡献−.6332，其余23类贡献+.0515；不直接宣布普遍视图冲突。

核对源实现：A只改Selector证据K/V，C改变Full Summarize softmax，B任务Q读Full摘要写Person。B训练残差已有足够幅度、C未饱和且KL小、A各任务幅度差异大，不重复使用“未训练/弱分支/过强bias”作为未经验证解释。训练期task0/1/2报告人群不同，未把其8阶段均值或forgetting当正式结果。完整test、分支净收益、开关归因仍待现有队列。

新增阶段报告docs/three_module_stacking_partial_results_20261023.md；CPU脚本/逐类统计/区间/SHA清单在output/three_module_stacking/analysis_20261023_partial_01。三个上下文及执行记录同步更新，原有未提交文档保留；没有业务代码改动，未重复跑模型测试、未提交推送、未新增训练。

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

## 2026-10-22：制定现有三模块完整叠加消融方案（未执行）

核对现有目标摘要实验分支model/runner、Adapter/cross-view/spatial模块和历史脚本报告，并查NIST全因子设计及MULTI-LANE原论文。主工作树exp/parax-protected-eight-task-evaluation有用户未提交内容，未修改；方案写在当前target摘要实验工作树。

新增docs/three_module_stacking_ablation_plan_20261022.md，明确完整八格、单→双→三执行次序、固定模块版本/位置/参数、统一90/10+aux.1 joint BCE及预算、全部8阶段val/test、F111条件贡献、交互计算/训练seed局限、保存恢复与GPU调度原则。发现A/C互斥及A分支未消费空间bias、B aux0约束等代码支持缺口，只记录后续实施范围，未改代码。

同步三个上下文。未创建新分支/提交推送、未运行训练或GPU推理、未访问新test。文档修改只做git diff --check，不重复模型测试；组合入口尚未实现，不能当作可执行脚本。

## 2026-10-22：核对Attention与人物位置两轮结果，讨论结合可行性

只读复查cross_view_attention_attribution_results_20261022.md与target_anchored_summary_complete_results_20261022.md，并核对MULTI-LANE/MulT原论文方法。确认两轮TRAIN比例、辅助损失与成功更新预算不同，不能混用匹配基线；两条路线均没有新增分支的明确推理净收益，人物Query旧残差还有整体损害，不能据模型间正向差值推导直接组合有效。

给出条件性方向：摘要Attention研究跨视图表示传递，人物位置方案研究Full原patch的目标条件读取；可在Full摘要权重中组合人物内容与几何，而保留原Value和三路Task Forward。该假设不同于直接叠加旧残差，未实施。共同训练参数改善/训练辅助也是待验证的另一研究方向，不能宣称正则化已经成立。先保留整图位置只读关闭归因；未来任何四组训练比较需统一split/监督/预算并只以holdout/validation筛选。同步更新三个上下文，无模型代码变化或新训练/评估，不提交推送。

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

## 2026-10-22：正式三组完整8任务已调度

源码b0d826342a55165f805ff25eec846df5688296ea已push，并在远程独立worktree执行git pull --ff-only；工作树clean后启动tmux spatial_summary_formal_seed0_20261022。formal_seed0_20261022_01的baseline/absolute/target分别GPU4/5/6、PID776820/776823/776826，已进入task0；初始底座SHA相同。保留原图90/10和aux.1，30epoch/task、B64、8task、seed0；后续审计、六路完整val/test和target三个边界位置归因已写入同一控制器，无需手动启动。

同步静态预检/失败短跑/修复短跑共155文件，SHA256全部匹配，未下载checkpoint或改写源产物。远程原主树和test-only默认status的153/5行及SHA与开始一致。415项测试及全链路真实smoke通过；正式结果尚未完成。启动状态只更新本地三个上下文及执行说明，不修改运行中的模型/脚本/远程HEAD。

2026-10-22正式启动前检查全部通过：415项完整回归；真实CLIP零参数/历史task0/2/7逐值重放；smoke_seed0_20261022_02三组各4更新/0跳步、原图隔离/初始化和采样配对/旧参数与恢复核验；六评估进程及总batch exit0，正常预测与训练时存档逐值重放，框交换/关闭只改Full而Person/Face固定。33可交换者的Full在task0/1分别31/33、33/33人变化。smoke01与02的六份方法状态逐值相同，确认修复未改训练轨迹。实测训练峰值baseline2575.41、两空间组2576.23MiB，保留约2710–2714MiB。
准备formal_seed0_20261022_01，GPU4/5/6三组并行完整8任务，每组12660更新，训练后六split评估GPU1–6，每卡单进程/内存上限16%/余量门槛5600MiB；原图90/10留出及aux.1保持，无绑定loss。用户8任务要求覆盖先导性能晋级门槛，不因中途指标变更条件。当前正式待调度，尚无最终结果。


2026-10-22短跑_01：三组各2task×2updates均完成、0跳步；空间参数梯度有限非零。验证器用compact的weights_only读取完整恢复文件，因TorchVersion/RNG元数据被严格拒绝，批次exit1且未开始评估；已修复为仅读取本实验可信完整快照的显式加载并补回归，失败产物保留。顺便将无效/不可见框的目标覆盖诊断归零，预测及优化公式不变。拟smoke_02重新跑全链路，通过才正式启动。


2026-10-22真实预检通过：CLIP/真实val首batch64上absolute/target零参数与原方法逐值相同；默认路径对历史task0/2/7预测逐值重放，源SHA与冻结视觉不变；非零空间参数交换框只改变Full，Person/Face不变。真实检查峰值分配760.5MiB/保留1044MiB。补充配对Dataset仅换框且保留本人可见性/像素/标签回归，准备最新414项全测与三组2任务短跑；正式尚未启动。


2026-10-22实现验证更新：新增10项空间机制测试及远程413项完整回归通过；源码9f9f6de已通过Git-only同步新远程worktree。原主树和test-only保持原样。新增三组批次控制器、恢复/旧参数/原图holdout核验、只读空间关闭/框交换评估，以及真实CLIP零初始化与历史task0/2/7重放检查；正式尚未启动，下一步真实预检和2任务×2更新短跑。训练GPU4/5/6、评估GPU1–6拟每卡1进程，每进程Torch分配器上限16%（约3.9GiB），启动要求至少5600MiB余量；不足等待，不干预外部进程。


## 2026-10-22：开始实现目标人物空间摘要，用户要求完整8任务双split

新隔离分支codex/target-anchored-summary-20261022，起点e81f2f2；旧分析/计划及原主/test-only工作树保留。用户要求8task覆盖原三任务先导，保留原图90/10 fit/holdout；三组disabled/absolute/target，seed0、8×30ep、B64、12660更新，主LR.0125/空间LR.0004、aux.1，禁用MoE/ParaX/adapter/旧cross-view。未来训练后并行完整val2397/test5368，不启用绑定损失。
已实现未验证：未截断真实crop/flip框元数据、Full原汇聚logits几何偏置、60参数/task跨层共享、零初始化/无效回退、优化器/保存恢复/旧任务保护；新增机制测试。当前尚未正式运行，先Git-only远程完整单测/真实smoke。完整配置docs/target_anchored_summary_execution_20261022.md。


## 2026-10-22：下一阶段文献核对与数据流方案完成，仅文档

读取MULTI-LANE §3.3、DSCT原论文方法及SCG原论文空间条件方法，对照本地Summarize、几何变换和旧Person-conditioned Selector结果。确认旧bbox/Person query-delta、selector-specific Person patches及target-aware裁剪均已有负结果，未把旧路线换名当新机制。

只读已有baseline aux0 task7完整val分支NPZ，527张多人原图中519张标签有差异，Full对同图人物完全相同；2397人/1705图，来源SHA在saved_validation_person_binding_audit.json。只用NumPy读取保存预测，无新前向/训练/梯度、新test或GPU资源操作，源文件SHA前后一致。

新增docs/target_anchored_patch_summarization_plan_20261022.md：提出Full压缩之前显式人物–patch空间偏置，原语义Selectors/值tokens/Person与Face路径/末端融合保持；60参数/task跨12层共享。解释与旧查询残差、摘要后跨注意力、目标裁剪的区别；给出三组先导、同checkpoint框交换/关闭、TRAIN当前类绑定监督的独立后续条件、预算/风险/停止门槛。P1保留同visibility门控，交换保留本人门控，避免位置和可见性混淆；标明全局CLS捷径、部分可见坐标、深层patch全局混合和评估集反复观察的限制。

三个上下文同步更新，未创建实现分支、未提交/推送；业务源码/配置/训练脚本均未改，所以无需运行模型测试。此前未提交归因报告及三上下文修改保留，远程工作树未访问或修改。新方案尚未被当作执行指令启动。

---


## 2026-10-22：完整8任务Attention归因同步、复算与机制分析完成

核验formal_seed0_20261021_01四进程/总batch exit0，32个正常开启边界逐值重放历史预测、本人Query控制/Full和Face不变/不可交换者不变/模型及源SHA保护全部通过。下载100份日志与结果并逐文件SHA验证，未下载或修改checkpoint。原primary默认status153行SHA ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf，test-only5行SHA e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152，来源74ece12及归因e81f2f2远程均clean，未触碰其他工作树。

新增ignored output下纯NumPy分析脚本/JSON，重算80条件边界×4端点=320个mAP，最大误差5.69e-14；500次validation原图配对bootstrap及标签不同人物描述分层，无新模型推理/梯度/训练。未依据test改条件。
人物Query on→off val42.1003→43.1086、test31.7499→32.3252，8阶段全人群均提高，Person恢复5.9295/4.4624；任务Query42.8976→42.9132、32.4993→32.5461整体近持平。本人→同图另一人Query在373/811可交换者最终39.3885→39.6085、27.9853→28.2418；每人预测均发生变化。交换差值抽样区间跨0，不能说另一人Query可靠更好。

新增docs/cross_view_attention_attribution_results_20261022.md，逐任务/分支/子集表及结构解释完整；三个上下文均更新。推理残差无净贡献与训练间接作用分开，人物对应未获支持；实际大残差/重复Full/融合监督缺口为有证据支持的解释而非唯一已证实根因。保留局部类别受益及seed0限制。

业务源码/配置/训练脚本未改，仅分析文档、协议完成标记、ignored统计；无需重跑模型测试。未新增训练、提交/推送或main合并。产物已同步本地，无需用户再次手动同步这100份结果；源checkpoint仍在远程保留。

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

本轮仅读取远程正式结果并同步，新增只读分析脚本（在ignored output下）和结果文档，三个上下文已更新；未新增分支、未提交/推送分析文档、未改业务代码，因此无需模型测试。模型源码与运行分支均保留，checkpoint未同步/修改/删除。离线指标复算与保存mAP一致；近似logit融合重组最大mAP差.0069已标注，不把移除Person项当关闭attention残差。

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

## 2026-10-18：分析结束后的连接状态记录

最后两次仅查Git/进程的SSH复查遇Connection reset，未执行任何服务器修改。
必需结果、47个checkpoint新SHA、7个重叠NPZ、补评估exit0及原产物核验此前已完成
并同步；不影响本地完整分析。当前不根据失败的实时连接声称额外服务器状态。

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

## 2026-10-18：完成WAF对照分析与只读Router探针

用户确认实验结束后，核对GPU1 WAF/GPU0 AF：均exit0、val_exit0/test_exit0、240epochs/
13950成功更新/0跳步。Git运行提交仍ceded63/07fd35c。同步正式日志、四组结果共157份
JSON/NPZ；重算服务器文件SHA逐项匹配，16个AF/WAF checkpoint SHA与两split和核验
记录相同。未同步模型权重、未改主树/test-only或已运行代码。
新增output中的NumPy离线分析，逐阶段AP重算与归档一致；四组cohort/target/class/
Face mask一致，配对初始参数、CLIP、训练IDs/首批/sampler核验通过。500次按原图
配对bootstrap，记录种子和固定tie排序；最终WAF−AF test−.1089 CI[-.5564,+.2786]，
WAF−B0阶段平均test−1.1984 CI[-2.0007,-.4365]，均只对应seed0模型。
GPU1只读4checkpoint探针（task0/7×AF/WAF、固定val前64人/25可靠Face、evalAMP/
TF32、无optimizer）完成，state/源checkpoint SHA前后一致；3份产物同步校验。
首轮face_manifest dict读取错误在数据构建前退出；使用root后成功，失败日志保留。
结果：WAF最终val42.2952/test31.6815，AF42.4007/31.7904，B0 42.1912/32.0610；
两版test八阶段低B0。两版均23/24全cohort任务视图固定专家对；WAF全局两名首次
冻结[6,13]却漏掉Full[4,9]，旧函数漂移增加，不能以冻结参数SHA不变声称旧预测不变。
新增报告docs/moe_weighted_freeze_results_analysis_20261018.md，并更新计划页和三份上下文文档。
下一步只建议旧可变专家回滚归因及完整依赖封存控制；本轮没有新训练/回滚实验，
没有提交/推送文档，没有合并main。

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

## 2026-10-16：MoE v3 实施

新独立v3分支基于`2d6346f`。新增SoftMoE bank和运行/批次脚本，修改
model、runner、evaluator，新增相关单测；本地Python与shell语法及diff
检查通过。v3每任务403456参数，匹配共享系数及固定/动态专门活跃容量，
新增专家输出与路由日志。用户明确要求8任务validation/test并行，已写
完整锁定配置；下一步Git-only远程同步、单测与三组2-update smoke。
服务器v3独立worktree创建于clean `6a4d585`，296项全测通过。三组
smoke均2次成功更新/0 skipped、CLIP SHA未变，主参数初始SHA一致；
同人群checkpoint重放mAP差均0，峰值reserved2818/2798/2804MiB。
dynamic梯度有限非零，三视图专门/共享输出比0.692/0.672/0.701；
每人共享+两专门硬计数和soft权重正确。新增两项launcher模拟测试本地
通过，覆盖val/test并发及失败状态。准备正式三组8task与并行split评估。
新测试经Git同步后，服务器完整298项测试通过。确认GPU0空闲约24047MiB，
正式tmux `ml_moe_ffn_v3_20261016`已启动，使用clean `7eb1f64`；
run_id为`formal_<mode>_seed0_20261016_01`，mode依次dynamic、shared_fixed、
view_fixed，训练每组8×30epochs，随后固定val/test双进程同GPU0并行。
运行期间固定服务器checkout，新增文档提交不拉入运行工作树。

## 2026-10-16：MoE v2 三组完成、MoE-off 与输出规模诊断完成

tmux顺序批次自然结束。dynamic/shared_fixed/view_fixed 均为clean
`eb90978`、8任务、各30epochs、13950更新、0 skipped、固定2397人
validation，CLIP SHA不变；最终mAP 42.2877/41.7671/41.9757，阶段
平均48.2518/48.9959/47.7554。补跑三组冻结checkpoint MoE-off，
无训练、无test；dynamic task7 on−off仅+0.0058。读取旧B0归档的完整
人群validation分数并逐项核对ID/targets，dynamic−B0最终+0.0965的
500次原图分组bootstrap区间[-0.748,+1.007]。task7输出探针显示共享
分量比专门分量大11.6–13.4倍，是“路由多样但作用弱”的主要机制。
本地已同步分析所需结果，checkpoint留服务器。完整报告见
`docs/moe_task_ffn_v2_results_analysis_20261016.md`。

## 2026-10-16：正式批次进度巡检

tmux `ml_moe_ffn_v2_20261016` 仍在运行，GPU0 约3469MiB。
dynamic、shared_fixed 两组的 `train/seed_summary.json` 与
`full_val/result.json` 均完整：各8任务、13950更新、0跳步；
最终/阶段平均全人群 val mAP 分别42.2877/48.2518与
41.7671/48.9959。view_fixed 已完成 task0–5，巡检时 task6
第7/30 epoch，已记录阶段中0跳步。完整三组比较待完成；test 未读。

## 2026-10-16：开始 MoE v2 实施

服务器 v2 独立 worktree 与本地/远端 `3b5612d` 对齐。修正测试期望后
8 项相关测试及 Track-A 全部292项单测通过。GPU0 `evidence_probe_seed0_20261016`
使用真实固定 val 前256人、未训练模型、batch64，无优化更新；
Full/Person/Face 归一化证据相邻距离为0.37471/0.58034/0.56503，
raw CLS 为0.01250/0.01728/0.02331。三视图证据变化坐标均768，
通过预注册阈值。路由计数和 provenance 在
`./output/moe_task_ffn_v2/evidence_probe_seed0_20261016/result.json`，日志在
对应 `./logs/`；随后完成三组 smoke。

三组 `smoke_{dynamic,shared_fixed,view_fixed}_seed0_20261016` 均在 GPU0
完成 task0 两更新、0 skipped、冻结视觉权重 SHA 不变，共同初始主参数
SHA `35af16da...`。固定全人群 val mAP 为
25.16338/25.16258/25.16808；动态组 adapter_grad_total_norm
0.00027468、finite=1，balance metric 0.00135255。动态组筛选人群
checkpoint 重放26.687860与训练末尾完全一致。现进入正式三组
8任务 validation；不读 test。

正式批次已在 GPU0 启动 tmux `ml_moe_ffn_v2_20261016`，使用 clean
`eb90978`，顺序执行 dynamic → shared_fixed → view_fixed，run_id 为
`formal_<mode>_seed0_20261016`；各组 train 与固定全人群 val 独立记录。
启动后已观察 dynamic 进程加载 CLIP，尚无正式指标。test 未启动。

新本地 worktree/分支 `codex/moe-task-ffn-v2-20261016` 基于首版
`b8fe8ea`。新增独立 v2 bank、图片证据预训练探针、三组统一运行脚本和
单测；扩展 model、runner、checkpoint evaluator。已完成 Python 语法、
shell 语法和 diff 检查。远程首轮 8 项 MoE 单测有 7 项通过；唯一失败是
测试将总类别数 3 误写成当前任务类别数 2，已修正断言。下一步重跑远程
相关测试、真实 validation 证据探针与
task0 smoke；仅探针通过后运行训练。实验配置和判定见
`docs/moe_task_ffn_v2_experiment_20261016.md`。

## 2026-10-15：MoE-off 与 Router 原因诊断完成

GPU0 同权重完整人群 validation MoE-off 重放 exit 0：task7
fused on42.0342/off42.5185，off−on +0.4843；Full +0.8786、
Person +2.1521、Face −0.3187。按原图像聚类的500次配对
bootstrap，fused 提升95%区间[+0.0948,+0.9536]；Person
[+1.4751,+2.8277]。8阶段均值只提升0.0665，task1–3
关闭 MoE 反而略降。FP32 Router 探针在 task7 val 前64人
显示各视图只有一组 Top-2，第一名权重0.9976–0.9995，
CLS 图片间变化远小于共同均值。完整分析和下一版建议见
`docs/moe_task_ffn_v1_results_analysis_20261015.md`。

## 2026-10-15：完整人群重评估完成

补丁 `eaac038` 在独立 worktree 经 Git fast-forward 同步，GPU0 双进程
只读重放全部8个 checkpoint，val/test exit均0，输出另存
`full_val/` 与 `full_test/`。val 阶段均值47.9082、最终42.0342、
遗忘0.01636；test 均值35.1268、最终31.7789、遗忘0.00898。
与旧 B0 阶段均值相比 val−0.5300/test−1.1513；test 8阶段
全部低于 B0。原图像聚类配对 bootstrap：task0 test 差值
−2.8205，95%区间[−4.3179,−1.2986]；task7 test 差值
−0.2822，95%区间[−0.8525,+0.2579]。每阶段训练样本 SHA、
首批 ID、初始主参数 SHA 与 B0 一致；task7 test 全部5368个
样本 ID、标签及类序逐项一致。新增 `--disable-moe-runtime`
验证集只读消融入口，尚待重放。

## 2026-10-15：GPU0 正式批次完成及分析

最终训练代码 commit `17a9b84`、本地/服务器同分支同 HEAD 且远程
worktree clean；远程全套 Track-A 288 项单测通过。GPU0 当时空闲约
24047 MiB。已启动 tmux `ml_moe_ffn_v1_20261015`，run_id
`moe_task_ffn_v1_seed0_20261015_01`；观察到 `begin_task=0`、
训练样本5353、validation2397、初始基本权重 SHA
`35af16da90b3bd3dc52ec6756ef6545a06af5936daa577455eefb2648fd83139`，
启动时 GPU0 使用约3431MiB、利用率70%。8任务全部完成、13950
成功更新、0 skipped，冻结 CLIP SHA 不变，峰值 reserved 3022 MiB；
并行 validation/test 退出码均为0。原评估的 task7 val/test fused mAP
为42.0342/31.7789，旧 B0 为42.1912/32.0610；各视图亦未见稳定
优势。该评估的 task0..6 按已见正例筛选人群，不能直接和旧 B0 的
固定完整人群阶段曲线比较。正式路由计数显示每个任务/视图评估时仅
固定使用两名专家，未实现跨图片内容分路；门控和噪声权重均已训练
为非零，仍发生确定性路由坍缩。补充完整人群重评估程序后以独立
`full_val/full_test` 目录保存，不覆盖原评估。


## 2026-10-15：MoE 远程单测与首次真实 smoke

smoke compact checkpoint 经独立评估器重放：eval batch64 的 val mAP
26.76087467369557，与训练末尾逐值相同；batch32 差 -0.01418586，
正式并行评估固定 batch64。route counts 每视图2285人×2专家，
两更新时 Full/Person/Face 各固定用(0,13)/(13,19)/(0,18)，
仅证明路由计算可用，不代表内容相关性已经学出。

第三次 smoke `moe_task_ffn_v1_smoke_20261015_03` 已完整结束：2 成功更新、
0 skipped、val mAP26.760875（仅健全性指标）、CLIP 摘要不变，峰值
allocated2694.4/reserved2902 MiB，MoE 梯度范数0.00011959且finite。
checkpoint 75,984,473 bytes，Git HEAD `396c495` clean。现给独立评估器
加入仅 validation 的单任务 smoke 重放入口，以核对 checkpoint 及路由统计；
通过后才启动正式运行。

第二次 smoke `moe_task_ffn_v1_smoke_20261015_02` task0 成功 2 次更新、
0 skipped，训练 loss 0.75690097；随后 validation 前向在 Router `scatter`
遇到 AMP 下 logits half/softmax float32 混用。将 Router 全程固定 FP32，
新增 eval autocast 回归；`smoke_02` 未完成，指标不得使用。

提交 `25772a3`、旧任务冻结修复 `af4b07a` 已推送；服务器独立 worktree
`/mnt/haoyuan/workspace/multi-lane-main-moe-task-ffn-v1` 与分支同步，
287 项 Track-A 单测通过。首次 GPU0 两更新 smoke
`moe_task_ffn_v1_smoke_20261015_01` 在首批前向失败，尚无优化器更新：
AMP 使专家输出为 fp16，而 task lane 累加张量为 fp32，`index_add`
要求 dtype 一致。现修复加权输出转为累加张量 dtype，并增补 CPU autocast
回归测试；该失败产物保留在 `./output/` 与 `./logs/`，后续用新 run ID 重跑。


## 2026-10-15：按论文定位 Summarize 并审计 MoE-Adapters 整体迁移

- 用户追问位置与原法完整迁移。核对MULTI-LANE第3.3节/Figure2、原版blocks.py与Track-A model.py；明确图像证据路径是Frozen Forward→Summarize输入，索引1是第1冻结block输出供第2块汇总，非block2输出后。
- 重新审计固定CVPR官方commit八份源码。原法是注意力残差后FFN并联完整瓶颈专家、每层跨任务池、task相关noisy Top-2及频率冻结；上轮每任务池/汇总前位置不是完整迁移。新主要候选为Task Forward FFN并联，视图维Router为必要扩展。
- 补充双编码器/文本分类、DDAS唯一任务选择与多标签all-lane冲突、原冻结和严格隔离区别、三视图Top-2并集最多6位带来的容量风险，以及硬编码/噪声/全局task/损失/checkpoint的代码移植问题。计算全层视觉侧22×b64和8任务3视图gate/noise共35,904,000新增参数。
- 新增完整迁移报告，旧报告增加修正链接，三个上下文同步更新。只运行文档/算式/diff静态核验，无业务代码/配置改动，无单测/smoke/训练/test或服务器操作，无commit/push。

## 2026-10-15：新建 MoE-Adapters 三视图分析分支

- 按用户要求先分析可行性。检查主仓库Git状态、remote及九个既有本地worktree；因主工作区有未提交文档，使用managed worktree从25fc861建立独立 `codex/moe-adapters-three-view-20261015`，未切换或覆盖原工作区。
- 只读核对CVPR2024论文第3.3/3.4节、官方commit1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885的MTIL路由和冻结实现、本地model/adapter/view_fusion/runner及另一支线static_moe源码；未将MoE-Adapters++或参数混合ParaX当作同一方法。
- 新增分析报告，给出任务内跨视图共享池+独立动态图像Router、同位置Selector证据路径、严格冻结边界、参数例子和分阶段最小对照；引用主工作区最新静态MoE/旧函数负结果并记录报告SHA。现有结果不支持直接预期精度提升，动态内容路由尚未验证。
- 三上下文均更新；没有业务代码/配置修改，没有运行单元测试、smoke、训练或test推理，没有连接服务器、commit/push/merge。文档检查包括diff空白、表列数、参数算式、Git分支/文件范围与原工作区状态核对。

## 2026-10-10：Git-only同步、远程全测与真实smoke完成

业务提交c440321已推送，服务器新独立worktree同HEAD且clean，主/test-only及运行中其他worktree保持原样。287项本地/远程单测通过。四组smoke各12updates/0skips、peak2712/2754MiB，三task全部identity/旧值和mask保护/paired审计/残差消融/比较器通过。另从原训练30轮task1 checkpoint重建task2，三组新审计通过，源权重SHA前后相同、零训练更新。分阶段真实smoke三task共12次B0更新＋167缓存校准更新，新专家改变且梯度非零有限，B0/旧路径hash不变、初始化及采样与基线一致。79个静态产物同步SHA核验一致。准备新batch正式四组task0–2 validation（从头训练、仅GPU0、OMP1），正式分阶段仅备好实现未启动，不自动扩大8-task/test。

## 2026-10-10：按用户授权修改 ParaX 审计和后置残差诊断

建立 `exp/parax-audit-paired-staged`，保留原本地/服务器脏工作树和test-only。修改model受保护路径精确identity归一化；审计检查零projection、零raw residual、缓存特征/logit精确对齐，完整前向关闭/开启各三次记录重复误差预算，不再使用固定1e-5唯一判据；失败也保存诊断并恢复TF32/RNG/模式。runner新增RNG隔离的后置残差关闭推理与可选protected staged。缓存校准支持学习新专家、冻结B0及旧参数，记录分项梯度/新矩阵变化，保留旧随机center模式。比较器新增固定同人群/新旧类AP/残差开关相对独立B0，不混入分阶段组。新增五项单测，17项针对性与本地287项全测通过；本地缺timm已仅补入临时venv后全测通过，项目依赖未改。完整配置已声明并写plan；接下来Git-only同步新worktree，远程全测与真实smoke通过才启动四组短程validation。正式分阶段等待四组精度诊断，不改动态融合。

## 2026-09-25：P-post-zero 闭环与后置校准实验

- `parax_post_zero_control_20260925_020000` 已完成，B0/P-post-zero 90 epochs、5010 updates，所有指标完全一致，差异为0；smoke 初始 logits diff `2.33e-08`。
- 结论：额外 ParaX 参数组、optimizer group 和 post forward 在 scale=0 时不会破坏可比性；P-post-tiny 的剩余差距由真实非零残差造成。
- ParaX image-stream、阶段四蒸馏和阶段五 level routing 停止。新增 `docs/parax_post_zero_results_20260925.md`。
- 新增 `scripts/emotic/run_multilane_track_a_post_calibration_control.sh`，下一步运行 B0 与 `V-post-residual`：关闭 ParaX，Full anchor，task-local zero-initialized Person/Face post-feature residual，task0--2 validation-only。

## 2026-09-25：P-post-tiny 结果与 P-post-zero

- 已同步批次 `parax_post_tiny_control_20260925_010500`。B0 与 tiny 均完成 task0--2、90 epochs、5010 updates、无 OOM/NaN，未访问 test。
- P-post-tiny final/average mAP `44.7075/54.1956`、forgetting `2.3772`；比 small/static 更好，但 final 仍低 B0 `0.5319`。
- 最终 residual/token ratio 约 `1.7e-7`，gate 近均匀且 view L1 仅 `0.009`。继续单纯减 scale 已无法定位剩余差距。
- 新增 `P-post-zero` 入口，保留结构和参数组但 output scale 固定为0，用于验证 full-training strict identity。
- `P-post-zero` smoke 初始 logits diff `2.33e-08`；批次 `parax_post_zero_control_20260925_020000` 已在 GPU0/1 启动严格 B0/zero 配对。

## 2026-09-25：P-post-static 结果与 P-post-tiny 启动

- 批次 `parax_post_static_control_20260925_003631` 已同步，B0/P-post-static 均完成 90 epochs、5010 updates、无 OOM/NaN、validation-only。
- P-post-static final/average mAP `43.9191/52.8834`、forgetting `2.2440`；动态 P-post-small 为 `44.2960/53.7270`、forgetting `2.5475`。静态方案稳定性最好，但 task0 已比 B0 低 `3.0228`。
- Dynamic 相对 static 在 task0 提升 `1.6248` mAP、最终提升 `0.3769`，说明 routing 有适应作用；两者共同低于 B0，主要问题仍是后置 residual 改写特征坐标与现有融合/head 不匹配。
- 新增 `docs/parax_post_static_control_results_20260925.md`。脚本新增 `P-post-tiny`，scale=`0.0001`，下一步仅比较 B0-paired/P-post-tiny，task0--2 validation-only，不访问 test。

## 2026-09-25：ParaX P-post 结果分析与 static control 启动

- 已从服务器同步批次 `parax_post_control_20260925_230104` 到 `output/emotic_track_a_parax_post_control/` 和 `logs/emotic_track_a_parax_post_control/`。
- 两组均完整结束 task0--2 validation：B0 final/average mAP `45.2395/54.9341`、forgetting `2.7760`；P-post-small `44.2960/53.7270`、forgetting `2.5475`。P-post 遗忘略低但 final mAP 低 `0.9434`，task0 已低 `1.3981`。
- strict zero-output 初始 logits diff 小于 `2e-8`；task2 residual/token ratio 约 Full/Person/Face=`0.035/0.045/0.050`，gate 主要选择 expert2，说明性能损失来自训练后的后置 feature 改写与固定融合/head 失配，不是初始化错位或中间 block 干扰。
- 新增结果报告 `docs/parax_post_control_results_20260925.md`。下一步启动参数量匹配的 `P-post-static`，用 uniform gate 排除动态路由本身的负贡献；仍只做 task0--2 validation，不访问 test。
- 首次启动误将 `static` 解释为中间 block 模式，已在 `a7d83a9` 增加独立 `post_static`，通过真实 ViT smoke（初始 logits diff `1.68e-08`）后重新启动。当前批次为 `parax_post_static_control_20260925_003631`，GPU0/1 正在运行。

## 2026-09-25：启动 ParaX post-encoder control

- controlled residual 结果已完成：Frozen-center-small final mAP `44.3814`，低 B0 `0.8581`；Frozen-center-penalty final `43.5579`，低 B0 `1.6816`。penalty 已将 residual ratio 压到约 `0.006--0.009`，仍未恢复性能，停止继续调 penalty。
- 结论是中间 CLIP block 插入位置本身可能破坏后续视觉编码坐标，因此新分支 `exp/parax-post-adapter-control`、提交 `e12a54b` 修正 P-post：严格保持 B0 block/lane 顺序，在最终 lane feature 后追加 ParaX；strict zero-output 初始 logits diff `1.86e-08`。
- 服务器 239 项单测和真实 P-post smoke 通过。唯一 batch `parax_post_control_20260925_230104` 已启动 B0-paired 与 P-post-small，GPU0/1，task0--2 validation-only、无 checkpoint、禁止 test。

## 2026-09-24：启动 Frozen-center controlled residual 阶段

- 分支 `exp/parax-frozen-center-controlled-residual`，提交 `c71da1d`；服务器独立 worktree `/mnt/haoyuan/workspace/multi-lane-main-parax-frozen-controlled`，239 项单测通过。
- strict zero-output smoke 通过，初始 ParaX logits 最大差异约 `1.9e-08`。
- 唯一 batch `parax_frozen_controlled_20260924_220444` 已在 GPU0/1/2 启动 `B0-paired`、`Frozen-center-small`、`Frozen-center-penalty`。共同配置为 seed0、task0--2、30 epochs/task、P-10/rank32、无 level embedding、validation-only、无 checkpoint、禁止 test；small 使用 fixed scale `0.001`，penalty 使用 scale `0.1` 与 residual penalty `1.0`。
- 用户要求训练期间不做实时监督；完成后再统一同步和分析。

## 2026-09-24：ParaX shared-center stability 阶段三结果

- batch `parax_center_stability_20260924_173712` 已完成并同步；三组均为 task0--2、90 epochs、5010 updates、0 skipped，无 OOM/NaN/traceback，未访问 test。
- final/average mAP：Shared-live `39.2991/49.7908`，Frozen-center `43.0695/52.1703`，Task-local-delta `42.6452/53.8214`；相对同批 B0 `45.2395/54.9341`，final 分别为 `-5.9403/-2.1700/-2.5943`。
- forgetting：Shared-live `8.1971`，Frozen-center `3.3804`，Task-local-delta `6.7104`，B0 `2.7760`。Shared-live 与 Frozen-center 在 task0 完全一致，差异从 task1 开始，直接支持 shared expert center 跨 task 漂移是主要遗忘来源。
- Frozen-center 只训练 router 后显著恢复稳定性，但 task0 就比 B0 低 `3.7219`，说明问题还包括 ParaX center 学习后的表征坐标变化与固定融合失配。Task-local-delta 的 task-local gate 最终仍产生约 `0.26/0.33/0.26` residual ratio，未改善稳定性。
- 详细结果见 `docs/parax_center_stability_results_20260924.md`。五阶段状态：阶段1基本完成，阶段2已完成，阶段3已完成且定位出漂移来源，阶段4尚未执行，阶段5继续暂停。下一步只保留 Frozen-center，测试更小且可微的残差约束，必要时再加入 Full-only CPU-cache 蒸馏；不恢复 level embedding、连续层、rank64、额外 seed 或 test。

## 2026-09-24：启动 ParaX shared-center stability 阶段三

- 分支 `exp/parax-shared-center-stability`，提交 `2282b2a`，服务器独立 worktree `/mnt/haoyuan/workspace/multi-lane-main-parax-center-stability`。
- 新增严格 `zero_output` 初始化、ParaX 初始 logits 对齐诊断、shared center task0 后冻结开关和 task-local gate；服务器 239 项单测通过，真实 ViT smoke 通过，FP32 初始 logits 最大差异 `1.49e-08`。
- 唯一 batch `parax_center_stability_20260924_173712` 已在 GPU0/1/2 启动三组：`Shared-live`、`Frozen-center`、`Task-local-delta`。共同配置为 seed0、task0--2、30 epochs/task、batch64、P-10、rank32、3 experts、zero-output、fixed scale、无 level embedding、validation-only、无 checkpoint、禁止 test。
- 用户要求训练期间不做实时监督；完成后再同步结果并统一分析。

## 2026-09-24：ParaX paired stability validation 结果

- 批次 `parax_paired_stability_20260924_161643` 已完成并同步；分支 `exp/parax-paired-stability`，提交 `6f4853f`。B0、P10-identity、P10-small、P10-zeroB 四组均完成 task0--2、90 epochs、5010 updates、0 skipped，无 OOM/NaN/traceback，未访问 test、未保存 checkpoint。
- final/average mAP：B0 `45.2395/54.9341`；P10-small `44.3357/53.5132`；P10-zeroB `38.9207/49.9796`；P10-identity `38.6066/50.8907`。相对 B0 final 分别为 `-0.9037/-6.3187/-6.6329`，无 ParaX 组超过 B0。
- forgetting：B0 `2.7760`、P10-small `2.6773`、P10-zeroB `9.6834`、P10-identity `10.9558`。P10-small 证明固定极小残差能把稳定性拉回 B0 附近，但没有带来精度收益。
- P10-identity 的 output scale 从0增长到 task2 `0.1247`，residual/token ratio 达 Full/Person/Face `0.2803/0.3472/0.3316`；它不是训练期间保持 identity。P10-zeroB 只将 Expert-B 置零，固定 output scale `0.1`，task2 residual ratio 仍约 `0.2000/0.2426/0.2299`，同样造成明显漂移。
- P10-small task2 residual ratio 为 `0.1076/0.1059/0.1187`，gate L1 为 `0.0570`；gate 差异没有对应单路收益，Full/Person/Face/reliable-Face 相对 B0 均下降。gate 差异不能单独视为有效 level routing。
- smoke 中的 `max_initial_difference` 仍只检查 `adapter_bank`，ParaX-only 时会输出0；完整模型级 initial token/logit diff、Selector/Prompt/head hash、第一批 DataLoader index 和 anchor feature/logit drift 尚未闭环。
- 详细报告见 `docs/parax_paired_stability_results_20260924.md`。五阶段判断：阶段1实现修正完成但诊断未闭环；阶段2已完成且未通过收益门槛；阶段3/4/5尚未执行。下一步先补诊断并做 Shared-live/Frozen-center/Task-local delta 短 validation，不恢复 level embedding、连续层、rank64、seed1/2 或 test。

## 2026-09-24：ParaX 根因定位与下一步方案

- 核对 `MultiLaneModel`、`ParaXImageAdapterBank`、runner optimizer/forward 路径和已同步 output/logs。
- 结论分为四层：B0/P10 存在 RNG 初始化混淆；small 不是严格 identity 且硬 ratio cap 改变梯度；shared ParaX 参数跨 task 更新导致旧 task image feature drift；最终 fixed-fusion loss 没有直接推动 Full/Person/Face 的有效 level specialization。
- P10-router 的 view gate 差异没有对应有效 residual，P10-experts 的 residual 没有对应 level-specific gate，Static-control 仍退化；因此不能继续扩大层数/专家数来搜索。
- 新增 `docs/parax_root_cause_and_next_plan_20260924.md`，规定先做 RNG paired 修复、zero-output identity、task0 smoke 和 task0-2 validation，再拆 shared-live/frozen-center/task-local delta，最后才恢复 level routing。
- 本轮没有修改业务代码、没有启动训练或测试；只更新分析与上下文文档。

## 同步并分析 ParaX P-10 稳定性 test（2026-09-24）

- batch `parax_p10_stability_test_20260924_111500` 已完成：P10-small、P10-router、P10-experts 均 240 epochs、13,950 updates、0 skipped；P10-small-distill 在 task0 后进入 task1 时 CUDA OOM，未形成完整结果。57 个结果/日志文件已同步本地。
- test final/average mAP：P10-small `31.2761/38.5367`，P10-router `31.9207/39.3952`，P10-experts `31.7548/38.9873`；冻结三视图 test 锚点 `32.5365/39.1445`。三组 final 均低于锚点，router-only 最稳定但仍无提升。
- residual ratio cap 生效：P10-small 和 P10-experts task7 layer10 约 0.09；P10-router 因冻结 expert center 实际残差为0。P10-experts view gate L1 仅0.0132，冻结 router 后几乎失去 level 区分；P10-router gate 有轻微 view 差异但无有效残差。
- 蒸馏 OOM 原因是每个新 task 同时保留完整 teacher CLIP 并执行 teacher/student 三视图前向，显存峰值超过23.5GiB。结果报告见 `docs/parax_p10_stability_test_results_20260924.md`。不把不完整蒸馏组用于结论，不继续 test 搜索。

## 开始 ParaX P-10 稳定性 test-only 实验（2026-09-24）

- 用户要求开始修改并直接运行 test，不跑 validation。已从 `exp/parax-level-routing` 新建 `exp/parax-p10-stability-test`，保留既有未提交文档和 `tmp/`。
- 新增 ParaX 稳定性开关：`trainable_components`（all/router/experts）、fixed/learnable output scale、residual ratio cap、旧 task logit distillation；默认旧行为保持兼容。
- 新增 test-only launcher，四组锁定为 P10-small、P10-router、P10-experts、P10-small-distill。共同使用 block10、rank32、3 experts、small init、fixed scale0.001、residual cap0.10；蒸馏权重0.2。
- test 配置与风险边界记录于 `docs/parax_p10_stability_test_plan_20260924.md`。本机 py_compile、bash -n、git diff --check 通过；本机无 torch，完整测试需在服务器 ddp 环境执行。尚未启动服务器 smoke 或正式 test。

## 同步并分析 ParaX shared level routing seed0 结果（2026-09-24）

- 六组正式 validation 已全部结束并同步到本地 `output/emotic_track_a_parax/parax_shared_level_seed0_20260923_20260923_2045/` 与对应 `logs/`；每组 240 epochs、13,950 updates、0 skipped，无 OOM/NaN、未访问 test、未保存完整 checkpoint。
- B0 final/average mAP 为 `42.9847/50.0234`。P-post、P-10、P-8:10、P-8:10-level、Static-control final mAP 相对 B0 分别为 `-6.6444/-5.4729/-11.5877/-9.9472/-9.0120`；所有预注册晋级条件失败。
- 所有 ParaX 单视图均下降。P-10 task7 Full/Person/Face/reliable-Face 相对 B0 为 `-5.741/-6.616/-2.034/-3.770`，Face 虽相对最稳但不足以支撑融合收益。
- P-10 residual/token ratio 约 `0.46–0.58`；P-8:10-level layer10 Person/Face 达 `1.016/0.934`，并出现近乎 one-hot gate。Static-control 仍退化，说明主要问题是残差幅度与跨 task 表征漂移，而非动态路由容量不足。
- forgetting 从 B0 `1.0785` 增至 P-10 `6.5905`、P-8:10 `12.0309`、P-8:10-level `11.9386`。停止连续层、rank64、seed1/2 和 test；下一步只设计 P-10 的 identity/小 scale、残差约束、冻结或蒸馏稳定性小实验，稳定后再恢复 level routing。

## 停止视图专用 Selector 并分析 ParaX 候选

- 用户要求停止个性化 Selector 路线，转向分析 ParaX 对共享 ViT 多 level 表征路由的适用性；本轮遵守“不修改代码”。
- 只读检查了当前 `exp/r0-dynamic-fusion-test` 工作树、MULTI-LANE 图像/任务双流与 Adapter 调用位置，以及本地 ParaX 官方实现和 plug-and-play guide。ParaX 原生 ViT adapter 位于 block 后并只处理 patch tokens；其多个 block 共享专家参数中心，各 block router 根据当前输入动态组合参数。
- 建议内部 late-block 插入作为主路线，冻结 backbone final-token 后置 ParaX 作为低风险对照；不建议直接把 ParaX 塞进 task-lane Adapter。标注了 `no_grad()/detach()` 梯度断路、最后 block 的输出未被现有 Selector 消费、增量 task 遗忘和 gate 路由混淆等风险。
- 记录详细研究判断和预注册式阶段实验方案于 `docs/parax_shared_level_routing_analysis_20260923.md`；未运行测试、smoke、训练或 Git 操作。

## 开始实现视图专用 Selector

- 创建分支 `exp/view-specialized-selector`，保留原工作树中用户未提交的文档和 `tmp/`。
- `MultiLaneModel` 新增 shared/view-specific Selector 模式；view-specific 模式初始化时复制同一 bank 到
  Full/Person/Face，task 激活和 optimizer 只作用于当前 task 的对应参数。
- runner、compact test state builder 和 smoke 新增 Selector 模式与 Selector 数量配置；默认 shared 模式不变。
- 新增 `tests/test_track_a_view_specialized_selector.py` 和预注册实验文档；尚未运行真实 GPU smoke 或正式 batch。
- 服务器首次完整测试为224/225；唯一失败来自测试中对参数切片使用 `copy.deepcopy`，已改为 `detach().clone()`，准备复跑。

## 整理导师讨论版三视图实验总结

- 新增`docs/three_view_shared_fusion_experiment_summary_for_advisor_20260916.md`，去除P0/G0/HF等内部编号，
  按“为什么做、怎么做、结果、说明什么”解释五组关键实验。
- 汇总文档将当前定位表述为Full预测质量及Full-Body互补性问题，明确融合算子、Face和简单分类头拆分不是
  主要原因，同时保留seed0与基线漂移限制。
- 本次未改业务代码、配置或训练脚本，未运行新实验，因而没有新增日志或输出结果。

## 同步并分析Full分类头独立诊断

- 正式batch`full_private_head_seed0_20260915_231942`完成；H0/HF/summary退出码均为0。两组各完成240
  epochs、13,950 updates、0 skipped，无OOM/NaN、checkpoint或test访问，服务器无残留训练进程。
- 结果和日志已同步本地；`validation_summary.json`本地/服务器SHA-256同为
  `73017155294c4ecd71238dd7cd39b2ee1dc9c5e1db920ff9c39cf3096aaf0913`。
- H0/HF logit final为`42.6627/41.1145`，HF低`1.5482`；average低`1.9876`，Full单路低`1.3932`，
  Person/Face/可靠Face也全部下降。HF-H0原图组bootstrap 95%区间`[-2.6199,-0.5365]`。
- HF私有head与共享head最终weight cosine仅0.5853，说明head确实分化但方向有害；不支持“Full共享分类头是
  主要瓶颈”。不补seed1/2或test。
- H0相对历史P0低0.4127，未过0.10复现门槛；记录逐路head/AMP运算次序引起训练轨迹漂移的限制。
  后续若测Full完整私有Adapter，应保留历史P0的特征融合后共享head路径，避免再次引入基线改写。
- 新增`docs/full_private_head_results_20260916.md`，串联P0 A/B、DGL、专用小Adapter和本轮head证据。

## 实现Full分类头独立诊断

- 从`97310fa`新建`exp/full-private-head-diagnostic`，保留用户未跟踪Adapter文档和`tmp/`。
- 模型新增`view_classifier_mode`：默认历史post-fusion共享head；诊断用shared-per-view和Full-private-per-view。
- HF的Full head深拷贝共享head，不消耗额外RNG；固定权重融合三路logits，Body/Face继续使用共享head。
- optimizer、参数统计、config与compact状态覆盖新增head；增加初始化/RNG、梯度隔离和optimizer覆盖测试。
- 新增两组seed0 launcher、汇总器和`docs/full_private_head_diagnostic_plan_20260915.md`。
- 服务器完整测试221/221通过。真实H0/HF task0/1epoch配对smoke各84 updates、0 skipped、2285个validation
  样本；H0/HF分类参数分别13338/26676，compact仅HF含`full_view_head`，链路检查通过。
- 正式batch`full_private_head_seed0_20260915_231942`以实验HEAD`87b406f`启动，分配GPU0/1。启动时8张GPU
  均有既有任务，0/1空闲显存约6.3/6.4GiB，双GPUlauncher正在等待连续两次达到12GiB空闲且利用率不高于10%。

## P0第一批结果同步与分析

- 同步batch `p0_r1_gap_seed0_20260915_1640`的结果、8份compact状态、分路NPZ和日志到本地。
- 训练完整复现43.075407/49.980689；240 epochs、13950更新、0 skipped。
- 修复AMP下先融合/先head数值不严格交换造成的离线分析拦截（`0024129`）；普通/诊断fused分数全相等，
  重建logit的final mAP只差0.001551。相关4项测试通过，离线A/B及2000次bootstrap重跑成功。
- 概率融合使P0下降0.251648；共同概率差距0.757435。Full替换四种背景均提升，共享Face均提升，Body存在交互。
- 新增`docs/p0_r1_gap_results_20260915.md`，给出八组合全表、误差审计、置信区间和下一步受控实验建议。
  本次未新增训练，保留原失败日志与成功retry日志。

## P0与R1诊断实验设计

- 用户将优先级设为定位0.5058差距，新增`docs/p0_r1_gap_diagnostic_plan.md`；未修改训练代码或启动实验。
- 核对历史三路：P0降低Full但提高Body/Face，修正了把P2相对P0的退化用于解释本差距的混淆。
- 确认P0共享线性head下特征/logit融合等价，独立R1为概率融合；Face训练过滤和batch组织也存在混杂。
- 方案先安排一次带scores/compact状态的P0复现及离线融合/来源替换，再条件执行模块解绑与训练目标对照。
- 新建`exp/p0-r1-gap-diagnostic`：`runner`可在既有三视图诊断验证前向导出逐分支FP32 logits/概率、
  fused输出、标签、ID及Face mask；未改变训练前向、loss或optimizer。
- 新增P0复现脚本、单GPU整批launcher和离线A/B分析器。A为I/S×logit/probability四格；B为固定概率
  融合下8种I/S来源组合，附边际/Shapley、子集/排序/分布诊断与2000次原图组bootstrap。
- 新增单元测试覆盖固定mask权重、logit/probability差异、分路score round-trip及bootstrap确定性；
  服务器`/opt/conda/envs/ddp/bin/python -m unittest discover -s tests -v`通过218/218。
- 真实EMOTIC task0/1epoch smoke完成：84 updates、0 skipped、val 2285样本（可靠Face 1382）；普通评估与
  诊断前向的fused logits/概率最大绝对误差均为0，按分路logit重建fused概率最大误差2.5326e-4；
  task0 compact状态含41个张量。smoke仅验证链路，不作为性能结论。
- 正式第一批于2026-09-15在服务器独立worktree启动：batch`p0_r1_gap_seed0_20260915_1640`、GPU0、
  实验HEAD`79b86c6`。只运行一次P0 seed0 validation；launcher将在训练成功后自动执行A、B和2000次bootstrap。

## 2026-09-15：共享视图融合论文检索

- 核对AdapterFusion、VLMo、MBT、MISA及DGL原文，新增文献笔记，补充Cross-Stitch与CD-Net。
- 修正此前机制推断：DGL关注幅度减弱，不以负余弦为必要条件；尚不能认定弱监督或表示同质化是主因。
- 建议按融合运算、共享位置、动态组合分别消融；本次未修改训练代码或运行实验，文档留在本地待审阅。

## 2026-09-15：实现共享主体加视图专用Adapter阶段

- 新建`exp/view-specialized-shared-adapter`，保留既有四份未跟踪Adapter文档和`tmp/`。
- Image-token Adapter新增可选的Full/Person/Face task-specific低秩增量；P2使用共享b32加三路b4，
  输出统一乘固定scale0.1。P1使用共享b45作为参数量匹配控制，P0保持共享b32。
- 模型显式传递当前图像视图到Adapter；单视图和旧配置默认Full且保持原行为。新增参数计数、视图
  隔离和梯度诊断回归测试。
- 新增分路径梯度审计：从融合损失在各view endpoint的梯度做VJP，只回溯对应视图计算图；
  Selector/Prompt使用BCE，Adapter使用ASL，epoch0/14/29各聚合前三batch。
- 新增三卡launcher、严格汇总器和预注册协议。当前本地py_compile、shell语法与diff检查通过；本机
  Python缺少torch/numpy，完整单测和真实GPU smoke将在服务器`ddp`环境执行。
- `e5c52ca`已推送，服务器独立clean worktree完成213项全测。P0/P1/P2真实task0一轮均完成84
  updates、0 skipped，参数计数正确；smoke同时发现AMP下Adapter分路径的二次VJP发生数值下溢，
  审计值为0而训练梯度不受影响。当前修复为先按65536缩放诊断loss完成VJP、再以FP32还原梯度，
  并增加各view×参数组非零回归断言；修复后须重跑全测和三组smoke。
- `32f8169`修复后服务器213项全测通过；P0/P1/P2重试smoke均完成84 updates、0 skipped，三组
  Adapter参数计数为49,952/69,933/70,700，3个审计batch的Selector/Prompt与Adapter三视图路径
  梯度全部非零有限。task0 epoch0显示融合/单视图梯度比例在Full约0.68、Person约0.17、Face约
  0.12--0.13，支持启动完整早中晚诊断，但不提前解释为全程梯度压制。
- 正式启动前GPU0--7均只剩约7.9--8.4GiB且利用率47%--100%，说明其他任务正在占用。三卡launcher
  增加资源等待门：指定三卡均至少12,000MiB空闲且利用率不高于10%，连续两次间隔30秒通过后才启动；
  可在tmux中安全排队，不与当前负载抢占。
- 唯一正式batch为`view_specialized_adapter_seed0_20260915_104759`，tmux为
  `ml_view_adapter_104759`，运行代码固定`10ef477`。当前launcher处于资源等待循环，尚未创建三组
  正式run或占用显存；满足门槛后自动并行启动GPU0/1/2，不重复启动。
- 用户确认约8GiB空闲可运行后，先核验三组config不存在、训练未启动，再停止仅等待资源的tmux并用
  同一batch ID重启，临时覆盖门槛为7,000MiB/允许100%利用率/一次检查。首次重启因入口少了
  `scripts/emotic/`前缀而在训练前安全退出；修正为实际脚本路径后启动，没有产生重复run。
- 三份正式config均记录clean commit `10ef477`、seed0、固定融合、30 epochs/task、8 tasks、val-only、
  checkpoint关闭，Adapter维度分别为32/0、45/0、32/4。P0/P1/P2首个epoch均完成84 updates、
  0 skipped，loss有限，耗时38.2/38.1/39.3秒；每卡约4.5GiB显存余量，无OOM或错误。
- batch已结束，P0/P1/P2及summary四个exit code均为0；每组240 epochs、13,950 updates、0 skipped。
  15个结果、6个控制和8个日志文件已同步，本地与服务器三组组合SHA-256逐组一致。
- final mAP为P0/P1/P2=`43.0754/42.7028/42.7961`，average为
  `49.9807/50.1309/49.7551`。P2只在task0高P0 `0.1097`，task1--7全部负；最终Full、Person、Face和
  可靠Face也全部低于P0，五项advance检查全失败，不补seed1/2或test。
- P2的中后期跨视图负梯度余弦反而少于P0，同视图融合/单路梯度方向仍为正，性能却下降；因此当前证据
  不支持DGL式梯度冲突是主要瓶颈。当前b32+b4设计结束；若继续共享模型，应先以平衡单路监督验证能否
  建立专家，并把共享Adapter与各view delta梯度/残差强度分开记录，再讨论动态Router。

## 2026-09-13：同步并分析OOF优势限定排序蒸馏

- E1完成240 epochs、13,950 updates、0 skipped，无OOM/NaN、checkpoint或test访问。
- 训练后比较器因E1与旧Person/Face的目标来源标签不同而拒绝融合；修复仅放宽
  `model_parameter_objective/adapter_parameter_objective`，其他协议仍严格核对，服务器204项全测通过。
- 修复后Full final/average/固定R1 final相对D0为`-2.6785/-3.2720/-2.0639`，三项门槛全部失败。
- 13个受监督类final AP平均下降`3.8765`，未受监督类下降`1.4805`；ranking loss却在多数task
  降至约`0.02--0.06`，确认是pair-bank泛化失败而非排序目标未收敛。
- 12个run结果文件、控制文件和日志已同步；run文件逐一SHA-256一致，总计约1.7MB，无checkpoint。
- 按预注册规则停止OOF蒸馏：不调weight/pair/mix，不补seed1/2、不test。下一候选应新增真正互补的
  独立专家，优先审计pose/上半身Person信息，而不是继续把多视图蒸馏进Full。

## 2026-09-13：开始OOF优势限定排序蒸馏

- 新建`exp/oof-advantage-ranking-distillation`，四份用户未跟踪Adapter文档保持未修改。
- 训练循环增加可选pairwise ranking loader；ranking梯度仅路由Full主参数，Adapter仍只接收
  hard-label ASL，旧调用默认禁用ranking。
- 新增三折类别优势审计、OOF纠错pair bank、均衡无重复pair dataset、E1 runner/比较器/
  launcher、回归测试和协议文档。
- 锁定hard batch64、pair batch16、ranking weight0.05，只做seed0 validation，失败则结束
  OOF蒸馏路线。
- 实现提交`0c0b753`已推送；服务器新建独立clean worktree，项目203项完整单测通过。
- 真实EMOTIC/OOF GPU smoke确认task0--7合格类别分别为`[2,3,0,3,2,2,0,1]`，真实pair bank、
  hard BCE、pairwise ranking和Adapter ASL均有限；1 update、0 skipped，峰值显存约1587MiB。
- 唯一seed0 validation batch `oof_advantage_ranking_seed0_20260913_203305`已在GPU1/tmux
  `ml_oof_rank_203305`启动；task0前4个epoch正常，每epoch约16.3--17.0秒，无OOM/NaN，
  test与checkpoint保持禁用。

## 2026-09-13：同步并分析D3 OOF R1教师结果

- D3完成240 epochs、13,950 updates、0 skipped，exit0，无OOM/NaN/test访问。
- 19个结果与日志文件已同步到本地，共1.7MB，逐文件SHA-256一致。
- Full final/average/固定R1 final相对D0为`-0.1689/-0.4322/-0.3271`，三项门槛均失败。
  停止raw-probability BCE蒸馏和mix搜索，不补seed1/2、不test。
- 逐折逐类OOF审计确认13/26类的R1优势三折一致，每个合格类有约1万至114万个
  R1纠正Full错序的正负样本对。若继续，仅值得尝试优势限定的排序蒸馏。

## 2026-09-13：开始D3 OOF固定R1集成教师

- 新建`exp/oof-r1-teacher-distillation`，不暂存四份用户未跟踪Adapter文档。
- OOF teacher bank增加锁定R1模式，可靠/无效Face分别使用`[.64,.16,.20]`/
  `[.80,.20,0]`；增加D3专用比较器、单组launcher、回归测试和协议文档。
- D3仅复用已有D0作为锚点，不重跑D0/D1/D2，不搜索mix，禁止test。
- 实现提交`b57f900`已推送；服务器创建新独立worktree，199项全测通过。
- 真实OOF R1/GPU smoke确认task样本数`[5353,4394,861,9931,4352,2536,627,1526]`、
  逐样本权重重建最大误差0，硬BCE/教师BCE/Adapter ASL均有限，1 update、0 skipped，
  峰值显存约572MiB。
- batch `oof_r1_teacher_seed0_20260913_192759`已在GPU3/tmux `ml_oof_r1_192759`启动；
  task0前3 epochs正常，每epoch约13秒，启动后仍余约4.2GiB，无OOM/NaN。

## 2026-09-13：同步并分析OOF跨视图蒸馏结果

- D0/D1/D2均完成240 epochs、13,950 updates、0 skipped，exit0，无OOM/NaN/test访问。
- 49个配置、指标、历史、validation scores、日志和比较文件已同步到本地，逐文件
  SHA-256一致，总计5.0MB，无checkpoint。
- D1的Full final/average/固定R1 final相对D0为`-0.2908/-1.1380/-0.4773`；D2为
  `+0.1928/-0.4150/-0.0754`，两者都未通过锁定门槛，不补seed1/2、不test。
- 额外OOF教师审计显示Person/Person+Face在所有task均弱于Full OOF；固定R1 OOF则在所有
  task均提高AP且降低BCE。下一候选为仅运行D3 R1-OOF教师seed0 validation。

## 2026-09-13：开始OOF跨视图教师蒸馏Full lane

- 新建`exp/oof-cross-view-distillation`；四份用户未跟踪Adapter调参文档保持未修改。
- 服务器旧OOF产物审计为3 folds x 3 views x 8 task scores，9个source完整，可直接复用。
- 训练循环增加仅主参数使用的OOF软BCE，与硬BCE固定`0.20/0.80`混合；Adapter路由保持
  hard-label ASL。历史无蒸馏调用默认mix0，兼容不变。
- 新增OOF teacher bank、Full蒸馏runner、D0/D1/D2三卡launcher、锁定R1比较器、回归测试和
  协议文档；提交`a3f7ee7`已推送。首次全测仅暴露单测将teacher batch误用为val batch，
  修正测试提交`f3f978c`后服务器196项全部通过，业务runner原本即使用标准val batch。
- 真实三折OOF完整性审计与真实CLIP GPU smoke通过：硬/软BCE、Adapter ASL、两参数组梯度
  均有限非零，冻结主干无梯度，峰值显存约572MiB。
- 增加“先通过全部硬门槛、再按指标排序”的比较回归测试，提交`9fe1a58`已推送；项目
  `ddp`环境197项全测通过。
- seed0 batch `oof_distillation_seed0_20260913_230500`已在独立worktree、GPU0/1/2并行启动
  D0/D1/D2；三组均已完成task0前3 epochs，loss有限，无OOM，test仍禁止。

## 2026-09-13：同步并分析CLIP Face + AffectNet补充残差

- batch完成240 epochs/10,980 updates/0 skipped，exit0，无OOM/NaN，test未访问。
- Face all/可靠Face/固定R1 final mAP相对锚点为`-1.5959/-0.9660/-0.2765`；R1八个task
  全部下降，forgetting变差`+0.0661`。
- 17个非checkpoint结果文件已同步至本地`output/emotic_track_a_face_expression_residual/`，三组
  逐文件SHA-256均与服务器一致。
- 按预注册门槛停止Face专用表征，不补seed1/2、不test。下一候选为OOF跨视图教师
  蒸馏Full lane，先做seed0 validation三组最小对照。

## 2026-09-13：开始CLIP Face + AffectNet补充残差

- 创建`codex/face-expression-residual-validation`，不修改或暂存四份用户未跟踪的Adapter调参文档。
- 数据层为同一目标Face提供旧CLIP margin0.15 letterbox和五点对齐AffectNet输入，水平翻转严格同步。
- 模型保留完整CLIP Face主路径；冻结AffectNet embedding+8类logits经task-specific rank32零初始化
  残差补入lane特征，旧task模块冻结，残差归BCE/LR4e-4，CLIP Adapter仍归ASL/LR4e-4。
- 新增单候选runner、launcher、回归测试与协议文档；仍只允许seed0 validation和固定R1 beta0.20。
- 服务器新建独立worktree且191项全测通过。真实64样本双输入GPU smoke确认零残差与纯CLIP最大误差
  `1.55e-7`，FP32和AMP的Adapter ASL梯度均非零，残差BCE梯度非零，两个冻结encoder无梯度。
- 8卡当时均约6.4GiB空闲；为避免OOM，launcher增加8GiB显存门槛的自动排队，不抢占现有训练。
- 代码/排队修正已提交推送至`7840aac`，服务器fast-forward后191项全测再次通过。
- 唯一seed0 validation batch `face_expression_residual_seed0_20260913_143719`已进入tmux
  `ml_face_expr_residual_143719`；当时8卡均不足8GiB空闲，任务正安全排队，未加载模型也未访问test。
- 按用户确认降低至6GiB并在GPU5启动后，首batch在优化前因dict路由误判缺少
  `condition_valid`退出。修复为显式key检测并新增双Face batch训练回归测试；无部分训练产物。
- 修复提交`fe054d5`服务器192项全测通过。干净重试batch
  `face_expression_residual_seed0_20260913_143719_retry1`已在GPU5正常运行；前3 epochs均64 updates、
  0 skipped，loss有限，显存峰值尚有约3.8GiB余量，未OOM。

## 2026-09-12：开始Face表情预训练专用表征

- 新分支`codex/face-expression-pretrained-validation`。
- 增加五点相似变换对齐及覆盖/变换/逐task正例审计；默认数据行为保持不变。
- 增加冻结EmotiEffLib AffectNet EfficientNet-B0的增量Face runner：task-specific Linear投影，或
  task-specific零初始化b32残差Adapter。
- 增加两组并行validation launcher与锁定R1 beta0.20的严格比较器；禁止test和权重搜索。
- 预训练来源选择受官方EmotiEffLib文档与模型仓库约束；权重SHA-256锁定为
  `47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`。
- 服务器首次加载发现官方权重使用新版`timm.layers` pickle路径；加载器增加精确模块别名以兼容项目
  锁定的`timm0.6.7`，不升级或扰动共享训练环境。
- 服务器独立worktree完成187项全测；五点审计train/val覆盖99.968%/100%、变换失败0，仅4个train
  样本回退。官方checkpoint输出1280维有限特征且参数全冻结；真实4样本projection和Adapter单步GPU
  smoke通过，loss有限，峰值显存分别约190/194MiB。
- 正式batch`face_expression_seed0_20260912_1405`已在GPU0/1启动，tmux为
  `multilane_face_expression_20260912_1405`；两组task0 epoch1均完成64 updates、0 skipped，loss
  有限且仍有约11GiB/卡空闲。launcher将在两组完成后自动执行固定beta0.20比较。
- batch已完整结束：两组各240 epochs、10,980 updates、0 skipped。projection/Adapter可靠Face为
  `35.7820/35.1329`，固定R1为`43.3234/43.2659`，均低旧锚点`36.3096/43.5812`；不补seed1/2、
  不test。34个服务器产物已同步，组合SHA-256为`69a7faf5...108f7994`，详见结果文档。

## 2026-09-12：同步并分析Face等更新量结果

- batch完成300个数据循环、10,980次成功updates、0 skipped，8个task实际预算与旧锚点逐项相等；
  launcher/summary均exit0，无OOM/NaN，test未访问。
- 可靠Face final mAP `36.171260`，相对旧锚点`-0.138350`；固定R1 `43.467534`，相对旧锚点
  `-0.113659`。双0.05门槛失败，不补seed1/2、不test。
- R1的8个task mAP全部下降；final cF1/oF1略升但mAP下降，类别上Sadness/Anger/Aversion/Peace的
  损失抵消少数类别收益，定位为阈值校准变化而非排序提升。
- 19个非checkpoint产物已同步到本地并打包，服务器/本地SHA-256均为
  `e0720999...00b8a8`。按协议关闭margin/filter/basic augmentation路线。
- 下一步只做Face专用表征：审计既有五点关键点，冻结表情预训练encoder，接task-specific轻量投影
  或Adapter；Full、Person和固定R1保持不变。

## 2026-09-11：reliable_m15等更新量控制实现

- 从`9fd6b43`创建`codex/face-reliable-equal-updates`，没有修改或暂存四份用户未跟踪的Adapter调参文档。
- Runner新增`--optimizer-updates-by-task`，要求完整8项正整数且不能与统一预算并用；每task仅按成功
  optimizer step计数，独立重建对应T_max的cosine scheduler。
- 新增单候选seed0 launcher和严格汇总审计；Full、Person、旧Face锚点、固定R1及0.05双门槛不变，
  明确禁止test和checkpoint。
- 下一步：本地静态检查后提交推送；服务器创建独立worktree，运行完整单测、真实数据和GPU smoke，
  通过后启动唯一完整validation。
- `b0c4ba2`已推送；服务器新建独立clean worktree，主工作树与test-only的用户改动均未触碰。
- ddp环境179项完整单测通过。真实Face task0两步GPU smoke完成2 updates、0 skipped、终点LR0、val score
  dump成功且无checkpoint；后置断言仅因预期旧目录名而报错，读取实际`val_scores/task0.npz`复核通过。
- 正式batch `face_reliable_equal_updates_seed0_20260911_094039`已在GPU0/tmux
  `multilane_face_equal_updates_20260911_094039`启动；首轮48 updates、0 skipped、约2.1GB显存，运行正常。

## 2026-09-11：同步并分析独立Face expert质量结果

- 三组完整结束且状态全0；每组240 epochs、8,610 updates、0 skipped，无OOM/非有限值，未访问test。
- 旧锚点/reliable_m15/m05/m05+jitter固定R1 final val mAP分别为
  `43.581193/43.493266/43.364586/43.336309`；全部下降，双指标门槛均失败。
- reliable_m15可靠Face自身提高`+1.084585`，但R1降低`-0.087927`；m05丢失脸周语境，jitter主要
  改善oF1而破坏融合排序，均不应继续搜索。
- 50个结果/score/log/status文件已同步；3.6MB归档两端SHA-256为
  `8c869470...5121757`。下一步建议仅补一次reliable_m15等更新量公平控制。

## 2026-09-11：实现独立Face expert质量validation

- 新建`codex/face-expert-quality-validation`，没有改动Full/Person模型或R1融合权重。
- EMOTIC Face输入支持从raw detection bbox重算margin、训练loss独立使用短边/检测分阈值、Face专用
  ColorJitter；训练mask和固定fusion reliability mask分离，避免实验参数意外改变融合样本集合。
- 新增三候选GPU并行launcher与固定beta0.20比较器；比较器同时审计Face可靠子集和R1全量final
  validation mAP，双指标均至少+0.05才advance，禁止test。
- 增加crop margin、Face jitter和锁定beta回归；协议见`docs/face_expert_quality_validation.md`。
- `0972d1a`已推送；服务器创建独立clean worktree。177项全测通过；真实train审计为16,001样本、
  原valid非歧义12,442、融合可靠9,696，m15/m05同一crop尺寸示例107×126/91×107。
- m05+jitter完成task0一轮GPU smoke：48 updates、skipped0、finite loss、val mAP32.9527。正式batch
  `face_quality_seed0_20260911_001`已在GPU0/1/2并行启动，日志/状态/结果路径符合协议。

## 2026-09-11：完成并同步class-aware OOF stacking

- 有效run `class_aware_oof_seed0_20260911_002`完成C1/C2各三档、每档8个task state，退出码0、
  `test_accessed=false`，无OOM/非有限值；174项全测与C1/C2真实GPU smoke通过。
- C0/C1/C2 final val mAP=`43.581193/43.586666/43.600490`；C2相对两锚点仅
  `+0.019297/+0.013824`，未过预注册`+0.05`双门槛，advance=false。
- C2的8个task mAP均小幅高于C0，8类正增益超过0.01且最大类占42.53%；但Sensitivity/Fear增益
  与Suffering/Pain退化并存。最强prior10最好，支持“动态信号不稳、强收缩才有效”的判断。
- 结果、48个小型state、日志和退出码打包同步；68KB归档两端SHA-256均为
  `978e6c72...19e841d8`。按规则结束动态Router，不补seed1/2、不test；详细分析见结果文档。

## 2026-09-11：修复class-aware OOF历史数值复现与汇总字段

- 首次正式run在任何训练前由`validated_run_scores`中止：NumPy1.26.4复算历史schema-v1 Full task0
  mAP比记录值高0.001280；概率哈希完全相同，排除模型、CUDA和数据变化。
- 通过临时隔离的NumPy1.23.5验证Full/Person/Face三组24个validation task全部严格复现，确认
  历史训练指标环境应为1.23.5。保持审计容差不变，修正`requirements.txt`和恢复文档；Face环境
  仍使用1.26.4。
- 修正结果汇总将不存在的`class_ap`当字典读取的问题，改为按`CLASS_ORDER`映射序列化的
  `per_class_ap`，并补充长度和顺序回归测试。首次失败产物保留，新run将在服务器全测/smoke后启动。
- 第二次启动在首个C1 backward后由PyTorch2.0.1 CUDA内部断言停止，定位为二维梯度mask的高级
  索引赋值；改为按第0维`index_fill_`，同时覆盖C1 bias和C2 interaction，不改变优化语义。

## 2026-09-10：同步并分析三视图OOF Router

- 正式batch 11项状态全0，9个专家各240 epochs、0 skipped；Full/Person每fold总updates为
  `9300/9390/9390`，Face为`7410/7380/7320`，无OOM、Traceback或非有限值，test未访问。
- 固定R1/最佳R2/最佳R3 final val mAP=`43.581193/43.587281/43.586777`。R3比R1高
  `0.005584`但比R2低`0.000504`，预设advance=false。R3 average/cF1/oF1更好且forgetting更低，
  但final收益远小于种子方差，不补seed1/2、不test。
- prior0使R2/R3降至`42.9327/42.6948`，prior0.1仍为`43.5523/43.4520`；只有prior1接近R1。
  学习权重的全量均值/std与固定R1由Face-valid mask自然产生的统计几乎一致，动态路由退化为先验。
- task6 R3相对R1：Sadness`-0.9809`、Sensitivity`+0.1050`、Suffering`+0.0187`，暴露同task类别
  共用权重的冲突。下一步候选为复用OOF产物的分层class-aware stacking，先CPU seed0 validation。
- 完整结果、scores、descriptors、states、control和logs共264文件约22MB已同步；20MB原始归档本地
  SHA-256 `c3f67534...529d203`与服务器一致。第一次归档误把目标放入被打包目录并提示self-read，
  该无效包已删除，最终同步包在被归档目录外重新生成。

## 2026-09-10：实现三视图3-fold OOF Router

- 新建`codex/three-view-oof-crossfit`；Runner增加稳定image-group 3-fold协议及严格互斥/validation-only
  参数检查，OOF fold继续输出实际评估概率、完整来源配置和训练排除证明。
- 新增9个专家任务的8卡安全调度器：`3 folds × Full/Person/Face`，每卡同一时刻最多一组，先完成的
  GPU自动接续第9组；异常时停止批次，结果/日志分别写benchmark root与项目output/logs。
- 冻结CLIP描述符导出支持完整train池。新增OOF来源审计、fold去重合并、共享hidden16主干+task偏置
  Router、逐task状态快照、R1/R2/R3完整validation汇总和advance规则；不含test入口。
- 增加image-group完整且互斥覆盖、多人同图同fold、Router固定先验、无效Face精确为0及task bias隔离
  回归。Python编译、shell语法及diff检查通过；本地系统Python缺NumPy，依赖型测试转服务器执行。
- 首次服务器全测165项仅新先验测试因NumPy不广播`(2,3)`与`(3,)`失败，显式tile修正后继续发现并
  修复crossfit接口将类别索引误当样本索引的问题；两处均在正式实验前捕获。最终166项全测通过。
- 真实train三视图16,001条ID/target对齐；task6三折`186/199/242`合并完整覆盖627条，显著大于旧
  calibration 51条。fold0三路task0 1-epoch GPU smoke均0 skipped，留出1,793条概率完全对齐；
  Face训练折正确过滤848条无效/歧义样本。完整train/val描述符smoke覆盖16,001/2,397条，哈希
  `14ee85cb...7ef9`/`30403962...308a`，可靠Face 9,696/1,460，test未访问。
- 服务器主工作树被Automatic Upload提前写入两个新文件；已移动到可恢复备份
  `/mnt/haoyuan/workspace/git-sync-backup-three-view-oof-20260910`并恢复主树原状态。独立实验worktree
  创建于`/mnt/haoyuan/workspace/multi-lane-main-three-view-oof-crossfit`，未触碰test-only。
- 正式batch`three_view_oof_seed0_20260910_160250`在clean `e490da0`启动，tmux
  `multilane_three_view_oof_20260910`；GPU0--7一GPU一任务运行前8组，第9组自动排队。初始训练
  0 skipped、每卡约2.0--2.1GB，无错误；launcher会在9组全部成功后才进入描述符与CPU Router。

## 2026-09-09：完成R1中心Router公平性修正

- `5675a3b`把valid prior严格对齐固定R1`[0.64,0.16,0.20]`，invalid prior不变；测试显式锁定
  两个向量。Automatic Upload文件逐哈希备份后服务器主树恢复clean，实验worktree ff-only同步；
  147项完整单测通过。
- 复用`three_view_router_seed0_20260909_163655`的三专家scores/descriptors，仅CPU重跑6候选；
  新输出`router_selection_r1_centered`，exit0、无test。52个新增状态/结果/日志已同步，组合
  SHA-256 `67edc6d...ee2e1`两端一致。
- 修正后R1/R2/R3 final val mAP为`42.7974/42.6804/42.6322`；R2/R3仍由prior1胜出，R3
  低R1 `0.1652`且低R2 `0.0482`。R2/R3可靠Face权重已约0.20，故失败不再能归因于prior中心。
- 按停止规则结束当前per-task calibration Router：不运行动态Router seed1/2或test，不扩hidden、
  类别独立权重或prior网格。若继续动态方向，必须先以image-group cross-fitting增加out-of-fold
  训练样本；若优先mAP，可将R1固定三路作为独立候选补seed1/2 validation。

## 2026-09-09：同步并分析三视图Router validation

- Face 90/10 source完整完成240 epochs、9,930 updates、0 skipped、耗时1,509.7秒；来源
  `9b5d08c`且clean，8份val/calibration scores与compact states齐全。描述符覆盖calibration
  1,532条、val 2,397条，可靠Face为924/1,460。
- 自动Router首次因无脸manifest的`face_detection_score=null`触发TypeError；专家训练与描述符均
  未受影响。`16c0b87`将nullable尺寸/分数fail-closed为0并补真实格式回归，服务器147项完整单测
  通过；保留失败记录，只复用现有产物重跑CPU Router，三项exit code现均为0，test未访问。
- final val mAP：R0 `42.4101`、R1 `42.7974`、最佳R2/prior1 `42.6674`、最佳R3/prior1
  `42.6707`。R1对R0的8-task mAP均正；R3对R1的8-task mAP均负，决策为不进seed1/2、不test。
- 弱prior使R2/R3 final最低降至41.55/41.39且forgetting升至1.71/1.74。task6仅51条
  calibration，R3无prior虽降低calibration BCE，却提高validation BCE并大幅降低mAP，确认过拟合。
- 当前动态Router正则中心Face0.10与固定赢家Face0.20不一致；强prior赢家在可靠Face上的平均权重只
  约0.095--0.151。下一步先做一次R1中心公平修正，保持所有容量/特征/网格不变并只跑CPU；若仍
  失败则停止当前per-task Router，不扩大网络。
- 85个非compact文件共2.8MB同步至
  `output/emotic_track_a_three_view_router/20260909_163655/`，result/control/logs组合哈希均精确匹配；
  34MB compact checkpoints未下载。完整分析见该目录`analysis.md`。

## 2026-09-09：实现三视图样本级Router validation

- 创建`codex/three-view-router-validation`；Face runner允许锁定的0或0.10 calibration fraction，
  90/10训练仍在DataLoader前过滤无效/歧义Face，calibration保留完整ID池用于三路对齐。
- 新增冻结CLIP低维视觉描述导出：仅train calibration和val，输出Full–Person、Full–Face、
  Person–Face三个余弦；无效Face相关值归零并记录mask，不导出test。
- 新增seed0三路Router：R0固定FP、R1可靠Face beta0.20、R2质量/预测统计、R3追加3维视觉余弦；
  task独立hidden16、masked softmax、Face无效权重0、prior仅0/0.1/1，calibration当前类BCE。
- 新增两GPU安全等待批次入口，可并行补训Face 90/10 source和导出描述符，随后CPU完成6个Router
  候选。实现`906ac10`及断言形状修正`ada5bf1`已推送，服务器独立worktree同步clean。
- 服务器ddp环境146项完整单测通过。真实train/val Full/Person/Face稳定ID和target逐条对齐；稳定
  image-group 90/10索引可复现。GPU0冻结CLIP描述符smoke覆盖2条可靠、2条不可靠Face，三个余弦
  全部有限，后两条Face相关余弦严格为0。
- Face source task0两步GPU smoke完成，2 updates/0 skipped。5,353条eligible中499条隔离为
  calibration，fit侧1,152条无效/歧义Face在DataLoader前剔除，3,702条参与loss；完整val与
  calibration probability v2、compact task state均成功写出。首次自定义smoke的tee目录未预建，
  只导致临时日志tee返回非零，训练产物完整且复核通过；正式launcher会预建日志目录，不受影响。
- 验收记录以`9b5d08c`推送并ff-only同步。正式batch
  `three_view_router_seed0_20260909_163655`已在tmux
  `multilane_three_view_router_seed0_20260909`启动：GPU0补训Face 90/10 source，GPU1并行导出
  三视图描述符。安全门控等待两次连续空闲后启动；task0前三轮各58 updates、skipped0，GPU显存
  充足且日志无错误。两个子进程均成功后才允许CPU Router选择；本批禁止test。

## 2026-09-09：同步并分析Face endpoint validation

- seed0 Face expert完成8 tasks×30 epochs、10,980 updates、0 skipped，用时1,637秒；运行来源
  `285f3e1`、dirty=false、validation-only、无checkpoint。
- launcher自动完成8-task来源/概率/ID/target审计。beta0/0.05/0.10/0.20 final mAP依次为
  `43.3035/43.4777/43.5468/43.5812`；beta0.20相对FP提高0.2777，average提高0.3226，
  但cF1/oF1分别下降0.7834/0.0796，forgetting恶化0.0048。
- task0--7 mAP全部改善；最终26类18升8降。task6 Sadness/Suffering分别+0.2465/+0.4792，
  Sensitivity -0.0877。可靠Face覆盖最终val 1,460/2,397=60.91%。
- 可靠子集mAP提高0.5250，不可靠子集前后完全相等；Face-only明显弱于FP，结论为Face提供互补
  排序信号。18个JSON/NPZ/log/状态产物已同步本地，SHA-256逐项一致，无checkpoint。
- 下一步进入`codex/three-view-router-validation`。复用已有90/10 seed0 Full/Person source，补训
  同分桶Face source，再比较R0/R1/R2/R3；不扩静态beta、不读test。

## 2026-09-09：实现Face endpoint validation

- 在新分支`codex/face-endpoint-validation`增加Face manifest输入、完整人体之外的Face letterbox
  transform及Face expert训练入口。
- loss-bearing train view严格过滤为有效且非歧义Face；内部val同样过滤，最终score dump保留完整
  validation池并使用稳定ID。无效Face占位输入不能参与loss或最终融合。
- 离线融合锁定FP=`0.8F+0.2P`与beta `{0,0.05,0.10,0.20}`；只有可靠Face才改变FP，其他样本
  精确回退。增加manifest SHA、来源配置、8-task score复现和ID/target对齐检查及可靠子集诊断。
- 新增单GPU安全排队脚本。本地语法/diff检查通过；服务器独立worktree固定`56fcf4f`，142项完整
  单测通过。真实数据检查确认train 16,001中12,442条、val 2,397中1,861条满足loss mask，
  其中1,460条val满足严格融合质量线。
- GPU0 task0 1-epoch smoke完成64 updates、0 skipped，实际probability v2 score覆盖完整2,285条；
  与历史Full/Person task0 scores真实对齐后，1,382条可靠Face可改变输出，903条不可靠样本与FP
  逐元素完全相等。无checkpoint、未读取test，可启动seed0完整validation。
- 验收记录提交`285f3e1`后，服务器worktree fast-forward并保持clean；GPU0启动seed0完整8-task
  validation，tmux `multilane_face_endpoint_val_seed0_20260909`，批次
  `face_endpoint_val_seed0_20260909_1320`。task0前5 epochs均64 updates/skipped0；launcher将在训练
  完成后自动执行固定beta融合，任何训练或来源审计失败都会使批次失败而不会启动其他实验。

## 2026-09-09：完成阶段0 Face GPU smoke

- 安全等待器在GPU0连续两次达到空闲条件后执行，退出码0，tmux正常结束；未抢占繁忙GPU。
- 从完整train manifest选取4个非歧义有效Face，覆盖原始短边4.7365、29.0143、47.7905、
  994.2052px；letterbox和CLIP normalization后输入`[4,3,224,224]`，冻结ViT-B/16输出
  `[4,512]`，输入与特征均有限。checkpoint SHA-256与冠军协议一致。
- smoke JSON、日志和状态已同步本地，三文件SHA-256逐项一致；没有训练、反向或test访问。
- 阶段0全部验收完成。下一步允许创建Face endpoint validation分支，但继续禁止直接正式test。

## 2026-09-08：实现阶段0 Face检测与匹配审计

- 创建`codex/face-manifest`；新增`face_manifest.py`、服务器运行脚本、协议文档和6项合成回归。
- 冻结并只加载InsightFace SCRFD检测ONNX，不加载身份embedding；manifest记录权重SHA-256、依赖版本、
  阈值、输入尺寸和CPU provider。同图只检测一次，JSONL逐图刷新并支持断点续跑。
- 目标人物匹配要求脸中心位于body bbox且至少50%包含，再结合检测分数、相对头部位置与面积评分，
  对同图多人/多脸执行全局一对一分配；额外记录候选冲突、前两名分差和歧义风险。
- 审计输出覆盖率、原始脸尺寸分位数、小脸计数、多人有效率/歧义、8个task及每个当前类有效正例；
  固定抽样200个train目标，优先覆盖歧义、多人、失败和小脸。无脸严格标记无效，不以整图回退。
- macOS系统Python缺Pillow，Python语法、shell语法和diff检查通过；完整单测及真实数据smoke转到服务器
  专用环境执行。本阶段不读test、不训练模型。
- 新增真实manifest Face crop到冻结CLIP的GPU smoke及安全等待器；要求任一卡连续两次满足空闲显存
  `>=6GB`、利用率`<=10%`才做4样本前向，输出有限性与形状审计，不进行反向或训练。
- 服务器独立worktree在clean `611d970`完成135项单测和8图真实smoke；随后全量处理train 12,915
  与val 1,705张图，形成16,001/2,397条目标manifest，未读取test且`training_started=false`。
- train/val有效Face为13,598/2,069（84.98%/86.32%）；歧义风险1,156/208，一对一重复分配为0。
  有效脸短边中位数36.50/36.67px，约40%小于32px。task6三类有效正例train为229/274/161，
  val为106/99/91；全部26类均非零。
- 200张train分层抽检已同步并审阅；清晰单人匹配可靠，风险集中在重叠body框、群体照、背景/媒体脸，
  以及低分辨率。结论通过阶段1 validation条件，但后续必须mask歧义并使用尺寸/分数质量条件。
- 完整70MB产物与日志已同步本地，19文件SHA-256清单逐项一致。GPU当时8卡均高负载，`4ff7652`
  的4样本Face→冻结CLIP smoke按6GB/10%连续两次规则安全排队，并于次日通过。

## 2026-09-08：制定Full–Person–Face动态路由实施计划

- 按用户要求新增详细Markdown实施方案并同步三份上下文；保留原未跟踪实验总结文件。
- 收敛为Face数据审计、Face互补性、离线软/稀疏路由、统一任务特征融合、多seed锁定评估五阶段。
- 记录旧门控51条task6 calibration限制，90%/100%训练数据对照边界、旧task路由冻结、缺脸严格
  回退、全量评估及三路计算成本。给出拟新增代码工作包和下一次具体执行清单。
- 方案采用先验证新增信息再验证机制的顺序；未修改业务代码、创建实验分支、提交推送或启动训练。

## 2026-09-07：实现初始Person固定融合复测入口

- 创建`exp/emotic-initial-person-fusion-control`。目标是补齐初始Person-only因未保存scores/checkpoint
  而无法与Full融合的历史缺口，不改动已锁定的融合规则。
- 新增初始Person正式score脚本：legacy crop、margin0.15、train crop0.70--1.0、eval
  Resize256+CenterCrop224、无ColorJitter；其余训练配置与现有Full/letterbox Person冠军一致。
- 新增专用固定融合器：强制验证初始Person transform和完整test来源，读取原validation锁定文件，
  只评估probability Full0.80/Person0.20一次；允许复用不同历史clean commit的Full source但逐项核对
  全部共同协议字段，并记录两端commit。
- 新增单GPU训练后自动融合launcher及2项回归。macOS系统Python缺numpy，已完成Python编译、Shell
  语法和diff静态检查；完整单测、真实数据与GPU smoke待服务器执行，尚未启动训练。

## 2026-09-07：初始Person对照服务器预检

- commit `44ad8ed`已在独立服务器worktree运行129项完整单测，全部通过；
  Automatic Upload的同哈希文件已备份，primary恢复clean，test-only未修改。
- 真实EMOTIC train/test分别检16,001/5,368条，旧版Person流程实际为`bbox+0.15 margin`后
  train `RandomResizedCrop(224, 0.70--1.0)`、test `Resize256+CenterCrop224`；抽样张量尺寸与数值均正常。
- 当时8张GPU均有20--22GB显存占用，不叠加训练。新增单卡安全等待器，要求连续两次
  显存占用不高于2GB、利用率不高于10%，随后先GPU smoke再启动正式test。

## 2026-09-07：同步并分析Full + 初始Person固定融合

- 等待器在GPU0空闲后完成Adapter-ASL smoke和唯一seed0完整8-task test；后台退出码0。
- 初始Person训练240 epochs/13,950 updates/skipped0，clean commit `825f058`，无checkpoint、
  OOM、非有限loss或中断。8份test scores完整并被融合器反算验证。
- initial Person-only final/average mAP为`28.7815/35.2444`，历史final结果精确复现；
  Full+initial Person固定probability0.8/0.2为`33.1429/39.7620`，相对Full提高`0.6064/0.6175`。
- letterbox融合仍为`33.2672/40.0545`，在8个task均超过initial融合，final/average分别高
  `0.1243/0.2925`。initial Person独立final虽高`0.0341`，但average低`0.9803`，说明融合取决于
  与Full的误差互补和概率标定，不是Person-only单一final mAP。
- 已同步融合JSON、训练JSON、8份NPZ和4份日志，本地/服务器逐文件SHA一致，无checkpoint。
  3.5MB小结果包SHA-256为`284c52896c77983070725cfce6df32c2fe45fcaacccc722878e3889ef5a814ba`。

## 2026-09-06：同步并分析selector patch 2×2

- 服务器四组均完整结束：各240 epochs/13,950 successful updates/0 skipped，task0--7与日志完成标记
  齐全，无OOM/non-finite/训练中断。
- 同步四组JSON、8-task NPZ scores和日志共52文件；本地/服务器逐文件SHA一致，manifest aggregate
  `0cc78d2b1f78ada2d59b3403daf630cc128fce82668f4215b00dbfc9286b4670`。
- final mAP：legacy-disabled32.5365、legacy-patches31.6826、target-disabled30.5018、
  target-patches30.2505。patch在legacy/target下分别下降0.8540/0.2513；target-aware在
  disabled/patch下分别下降2.0348/1.4321。
- legacy patch只有task0略升，task1--7全降；最终5类升21类降，task6 Suffering下降4.8180。
  固定外部0.8/0.2 Person融合也无法挽救新端点，原LD+Person仍以33.2672保持seed0双视图最佳。
- 数据审计：target-aware将test bbox可见率0.8032→0.9778但改变71.89%裁剪位置；train有9.87%
  full-image fallback。它训练loss更低但test更差，当前判断为上下文/构图或分布泛化问题。
- 真实16,001条train记录的Person有效patch最少28、空mask0；已知空mask bug仍需修复但不是本轮原因。
- 更新结果报告、方案文档和三份上下文文档；本轮只同步/分析，没有修改业务代码、提交或启动新实验。

## 2026-09-06：审计下一阶段方向（未启动新实验）

- 读取当前模型/transform/loss、历史实验结果及CocoER本地源码；服务器快照四组约task0 epoch14--15。
- 合成复核：旧head梯度与一步更新均0；all-false patch mask在非零bias下错误产生0.1条件残差，
  极细长crop可能生成空mask。未修改正在训练的实现，真实触发率仍待统计。
- 历史disabled预测对齐：task0同ID到task7 mAP只差-0.0291；样本池3142→5368后五类AP下降
  16.4204。修正“共享head持续更新导致遗忘”的过度推断；固定ID对照与标准指标都应保留。
- 新增方案文档：先完成2×2和必要修复，再4组val比较Full/query/Person value residual/深层特征
  residual。后续只给赢家补ROI、辅助监督、合法同task蒸馏及多seed。没有承诺特定mAP收益。
- 三份上下文文档同步更新；本轮未提交推送，也未改动运行中的服务器独立worktree。

## 2026-09-06：target-aware crop × selector-specific Person patches 2×2

- 从结果已提交的`455617a`创建`codex/selector-specific-person-patches`，未修改main。
- Full新增target-aware train/eval裁剪：有效目标可容纳时完整保留bbox并随机保留场景；无法容纳时
  使用最大保留策略。legacy路径的像素和外部PyTorch RNG轨迹保持不变。
- Person letterbox新增14×14内容mask；新`person_patches`条件在layer1读取同深度冻结patch tokens，
  每个Selector单独注意Person内容，再由每task独立零输出MLP产生query residual。padding不进入softmax。
- runner/config/compact score重建/GPU smoke接入新模式与裁剪元数据；脚本支持四组并行：
  legacy/target-aware × disabled/person_patches。固定冠军训练协议，seed0、完整8-task、直接test、无checkpoint。
- 新测试覆盖target-aware目标保留、padding mask、selector-specific差异/屏蔽不变性、冻结Person路径；
  本地完整127项单测、Python编译、Shell语法、CLI参数解析及diff检查通过。
- 实现提交`5166a54`并推送。Automatic Upload写入的13个文件逐SHA与本地一致，已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-selector-specific-person-patches-20260906-preworktree`并
  恢复服务器primary clean；新独立worktree固定同一commit，test-only未改。
- 服务器127项全测通过。真实EMOTIC batch smoke中legacy条件有效率93.75%、target-aware为100%，
  Person有效patch平均74.11%，形状/有限值检查通过。8张GPU仍各有约12.6--13.8GB未知任务占用。
- 新增安全等待脚本：检测到任意4张卡显存占用<=2GB且利用率<=10%后，先跑disabled与
  person_patches GPU smoke，通过才自动4卡启动2×2；避免当前叠加造成OOM。
- 用户确认现有显卡余量允许叠加，故不再等待空闲。第二提交`f1ada22`已推送并ff-only同步实验
  worktree；本轮4个Automatic Upload文件逐SHA一致，备份于
  `/mnt/haoyuan/workspace/git-sync-backup-selector-specific-person-patches-20260906-scheduler`，primary clean。
- GPU0上disabled/person_patches两种真实CLIP smoke通过，trainable参数739130/789082，条件零初始化
  最大差0/9.9182e-5，Adapter最大差均4.5776e-5；AMP容差、梯度路由和冻结检查全部通过。
- tmux`multilane_selector_patch_2x2_test_20260906`已在GPU0--3启动四组seed0 test。四份config齐全，
  每组task0 cycle1为84 updates/skipped0；显存总占用约14.7--15.2GB/24GB，无早期OOM或非有限错误。

## 2026-09-05：Person-conditioned Full Selector第一版

- 用户要求开始方向一代码修改并新开分支；创建`codex/person-conditioned-selector`，基于
  `3f8327a`，原3份未提交实验完成记录原样保留。
- 新增任务隔离Selector条件MLP与成对Full/Person transform；改动EMOTIC、模型、runner、
  compact模型重建和GPU smoke。条件只改查询，保持原Drop & Replace残差语义。
- 四模式disabled/bbox/person/bbox_person；仅layer1、hidden32、scale0.1；新增参数BCE/LR4e-4，
  Adapter继续ASL；Person最终CLS来自冻结CLIP，无额外监督模型。每任务独立零输出初始化。
- 配对保留稳定人物ID，框跟随Full裁剪/resize/翻转，Person完整letterbox并共享flip；
  全不可见/无效目标关闭增量，人物jitter隔离RNG。训练历史增加目标可见比例和条件LR。
- 新增13项回归，首轮定位并修复lane block参数接入问题后全部通过；完整123项CPU回归通过。
  本地原环境无torch，因此在临时隔离venv安装torch/torchvision验证合成数据，不修改项目依赖。
  最终123项CPU回归全部通过（torch2.14/torchvision0.29）；Python3.9编译、Shell语法、
  runner/smoke参数解析与git diff检查通过。真实CLIP/GPU/EMOTIC验证仍待服务器，未启动真实实验。
- 新增val-only脚本及`docs/person_conditioned_selector.md`。本轮未提交推送或服务器同步；
  已知Automatic Upload异常应在下一次Git-only同步前核对，test-only工作树未操作。
- 用户随后要求Git-only同步、服务器检查后运行四组seed0；因两种split训练时间相同，授权直接test。
  脚本已支持REPORTING_SPLIT并自动设置score purpose，待提交推送后执行。
- 初次服务器完整123项单测和真实EMOTIC paired数据smoke通过。三种条件GPU smoke在零初始化
  `torch.equal`处失败；CPU逐bit一致，GPU AMP额外Person前向后相同Full有浮点重算差异。
  改为与原Adapter smoke一致的AMP1e-4/FP321e-6 allclose并打印最大差，需重新提交验证。
- `a8a759f`推送、服务器worktree ff-only后，bbox/person/bbox_person三组GPU smoke通过；最大
  初始差3.05e-5/6.10e-5/4.58e-5，条件与Adapter梯度、冻结CLIP检查均通过。
- Automatic Upload两轮文件都先与本地SHA逐项核验，再备份到
  `/mnt/haoyuan/workspace/git-sync-backup-person-conditioned-selector-20260905-preworktree`和
  `...-smoke-fix`，服务器primary恢复clean；实验只用独立person-conditioned-selector worktree。
- 按用户要求因val/test训练耗时相同直接启动四组seed0 test；tmux
  `multilane_person_conditioned_test_20260905`、batch同名，GPU0--3分别disabled/bbox/person/
  bbox_person。四组进入task0，显存约2.0--2.1GB，无OOM，源码`a8a759f`，不保存checkpoint。
- 四组最终均exit0，240 epochs/13,950 successful updates/0 skipped、32份task scores齐全。
  52个输出/日志已scp同步并用相对路径树SHA复核；本地/服务器均为
  `edb5458b783f1ae319d4910885e8ea4516c0a98973697d4778b47361bead6b1b`。
- final mAP：disabled32.5365、bbox32.0791、person31.1695、bbox_person31.3557；三种条件
  average mAP和forgetting也都差于disabled。disabled与20260903原Full八task logits最大差0。
- 用原seed0 Person固定probability0.8/0.2离线复核：33.2672/32.6246/31.9731/32.0676，
  条件化仍无收益。没有搜索alpha、mode或threshold。
- Person条件训练BCE全任务低于disabled但test明显下降；bbox退化最小。训练有效条件88.31%、
  平均目标可见54.89%。分析认为共享查询增量压缩selector多样性、final CLS与layer1深度失配、
  完整Person与随机Full crop失配及任务小数据过拟合共同导致退化；其中因果需新消融确认。
- 分析写入`output/emotic_track_a_person_conditioned_selector/person_conditioned_selector_test_20260905/analysis.md`。
  建议停止当前shared-delta调参，下一轮先target-aware Full crop控制，再selector-specific Person
  patch summary交互，使用validation；不因本轮test继续调整当前结构。

## 2026-09-04：同seed Full+Full完成、同步与分析

- 用户要求同步分析并提出下一步，未要求启动新实验。批次`same_seed_full_full_test_20260904_122941`
  状态complete、三退出码0，三组各240epochs/13950updates/skipped0，24份test score及日志齐全。
- 全部24个seed×task的重复Full logits/概率差值为0；训练记录除elapsed_seconds外完全相同。
  同seed Full+Full所有task/summary指标均与单Full相同，final32.1562±0.3701；Full+Person旧结果
  32.8263±0.4025精确复现，平均差0.6701。这是重复性结果，不反驳循环seed Full+Full的真实集成收益。
- 新三组输出、scores与日志已通过scp同步（远端无rsync）。本地/远端99个source哈希均匹配，结果JSON
  SHA为f7847c4ebfa158d307c05a650fbdd83f435ed57ab32e44a303accdbc5c76590c；三份训练日志无异常。
- 完整分析位于`output/emotic_track_a_same_seed_full_full/same_seed_full_full_test_20260904_122941/analysis.md`。
- 建议停止重复Full/输出权重搜索，后续转最小Person条件Selector的seed0完整val机制对照，再三seed确认。
  保持原100%训练数据，不混用门控90%fit。没有执行该新计划，未改业务代码；本轮文档只本地更新。

## 2026-09-04：启动严格同seed Full+Full重复对照开发

- 从5da49fd创建`codex/same-seed-full-full-test`。用户要求对标原同seed Full+Person正式结果；
  已说明两个Full使用同global seed可能训练成相同模型，仍按请求真实重训三组，不用复制预测代替。
- 新增同seed比较入口、3卡调度脚本及3项回归；配置及原source提交完全匹配，计算预测/训练轨迹差异。
- 远程GPU0--7均空闲；计划GPU0/1/2。原seed0 c9fa74c及seed1/2 d2442ee工作树clean，
  直接复用其未经修改的正式训练入口；test-only仅只读检查，原改动保留。
- 先推送调度分支、独立worktree完整单测、source审计与两版GPU smoke，通过后启动并自动汇总。
- `20dae80`已推送并创建独立调度工作树`multi-lane-main-same-seed-full-full`，完整110项单测、
  原六组test source审计、原c9fa74c/d2442ee GPU Adapter-ASL smoke全通过。
- 批次`same_seed_full_full_test_20260904_122941`在tmux `multilane_same_seed_full_full_122941`
  启动；GPU0/1/2跑seed0/1/2，三份config与对应历史Full完全相同（含commit/branch/tree/dirty）。
  task0首轮均84steps/skipped0，loss分别0.61478711/0.61898424/0.61532042，无OOM/非有限异常。
- Automatic Upload再次上传6份文件，逐文件SHA一致后备份到
  `/mnt/haoyuan/workspace/git-sync-backup-same-seed-full-full-uD5CDy`并恢复primary clean；test-only未改。
- 运行中不更新训练源或调度worktree；只提交本地运行记录。最终结果尚未产生，不进行完成宣称。

## 2026-09-04：补固定Full+Full正式test对照

- 用户要求将此前validation的等预算Full+Full对照补成test；从clean f1f1308创建
  `codex/full-full-locked-test`。只复用已有六组正式预测，保持0.8/0.2、threshold0.5和循环seed配对。
- 提取固定比较共用逻辑，新增独立test审计/入口及5项回归；validation入口仍禁止test。
- 来源必须完整240epochs/13950updates/skipped0、clean commit，逐task重建原指标并核对split IDs。
- 计划Git-only推送后创建独立服务器worktree，完整单测通过再离线评估，无训练/GPU推理需求。
- 实现`c54e330`已推送，独立`multi-lane-main-full-full-test`全套107项单测通过；六组48份真实score
  审计通过，离线固定test评估成功。Full+Full为32.6551±0.2622；匹配辅助Full+Person为
  32.7957±0.2378，平均净增益0.1406，三个配对和24个task均为正；forgetting平均恶化0.0421。
- 原同seed Full+Person的全部task和summary指标与历史结果逐字段相同，未重训或改写source。
- 本地同步JSON及两份日志，结果JSON SHA-256为
  `c0ab17c2ee9ec21ca675f8f532b2eeda85fc021b5f2de6db6e7c32a44d4e0a25`。详细分析保存于
  `output/emotic_track_a_ensemble_control_test/full_full_locked_test_20260904/analysis.md`。
- Automatic Upload写入6份本轮文件；逐项SHA一致后，备份到
  `/mnt/haoyuan/workspace/git-sync-backup-full-full-test-S0yRpG`并恢复primary clean；test-only未修改。

最后一次更新：2026-09-04。

## 2026-09-03：开发统计一致性修复与等计算量validation对照

- 用户要求执行前两步骤。创建`exp/emotic-fusion-numerics-ensemble-control`，保留上轮三份未提交
  结果文档改动。模型、Adapter、数据增强、loss与训练预算不变。
- score schema2保存实际用于指标计算的CPU float32 probabilities、实际batch长度、操作与torch版本；
  schema1兼容按原eval_batch_size重建。概率必须在各自原样本顺序重建后再按sample ID对齐。
- fixed test与validation分析器统一使用已保存/重建概率，逐字段和全部类别AP以1e-10绝对误差校验；
  不把原数值错误改成宽容差。新增形状相关一ULP模拟回归、schema2不重算sigmoid、batch尾部保存/
  ID对齐/非法元数据测试；后续服务器还将使用本次实际出错scores做数值回归。
- 固定对照模块仅接受完整clean seed0/1/2 validation来源，检查240 epochs/13950 updates/skipped0、
  每视图除seed/Git外配置一致；拒绝test、配置漂移、重复seed与损坏指标。
- 两模型预算一致，权重固定0.8/0.2：主seed0/1/2对应辅助seed1/2/0，比较full+full与full+person，
  另外报告同seed full+person；循环复用模型，只报告描述性均值/std，不伪装独立重复显著性检验。
- 四卡launcher复用已有seed0 full/person validation；GPU0/1为seed1两视图、GPU2/3为seed2两视图。
  固定30 epochs/task、batch64/workers2、Adam_reset_per_task、main LR0.0125/WD0、cosine/min0/
  no warmup、layer1/b32/Adapter LR4e-4/WD0/scale0.1/ReLU/independent、BCE+ASL9.8/0/0.05、
  legacy_full_zero、CLIP normalization、AMP/TF32、无checkpoint；full crop0.05，person margin0.15/
  letterbox/jitter0.10@0.20。数据与ViT-B/16权重沿用原路径，结果与日志在独立worktree的output/logs。
- 本地静态编译、shell语法与diff检查通过；准备提交并通过Git创建独立工作树，先全量单测及smoke，
  再启动四组完整8-task validation。不会新开正式test或改变融合权重。
- 首次服务器86项单测通过；三种子已有固定test scores的修复后anchor与融合均精确复现离线恢复结果，
  未读取新test样本或运行新test推理。seed0旧validation来自6ab53c8，早于score-purpose字段，
  因而新增受限兼容：只允许reporting=val、实际为schema1且通过指标/ID审计的旧文件缺省该字段，
  配置比较只规范化这一元数据，不忽略训练参数；新增回归测试，仍拒绝test和schema2缺失用途。
- 实现`5274d14`及兼容修正`ebf49a2`已提交推送，通过Git-only创建独立服务器工作树
  `/mnt/haoyuan/workspace/multi-lane-main-ensemble-control`。Automatic Upload的完全一致文件
  已逐项SHA校验、备份到`/mnt/haoyuan/workspace/git-sync-backup-ensemble-control-1CWO07Me`
  （兼容修正位于legacy-compatibility子目录），恢复primary clean；test-only未修改。
- 最终87项完整单测通过，实际历史seed0/1/2 fixed test的anchor及融合均精确复现上轮离线恢复结果；
  seed0 validation的固定融合全部task/summary字段完全一致，final43.30347536439419。上述仅对已存
  scores进行回归，没有新test样本读取、test训练/推理或test参数搜索。
- GPU0 Adapter smoke通过：Image-token layer1/b32/independent/scale0.1、Adapter-ASL、AMP/TF32，
  trainable739130、初始化差3.8147e-05。真实EMOTIC full/person各67条validation样本经batch64+3
  完成GPU评估，schema2保存的概率读回后指标逐字段完全一致，sample ID对齐通过。
- batch`fixed_ensemble_control_val_seed12_20260903_191515`及tmux
  `multilane_ensemble_control_20260903_191515`已启动，训练固定clean`ebf49a2`；GPU0/1为seed1
  full/person，GPU2/3为seed2 full/person。输出位于该工作树output/emotic_track_a_ensemble_control/
  对应batch/runs，日志位于logs同名目录；四组完成后自动输出ensemble_comparison.json，不启动test。
- 四份config逐字段核验通过，与对应seed0除seed/Git和旧用途元数据规范化外无差异；task0前几轮
  全部84 steps/skipped0，GPU0--3约21.6--22.1GB空闲，无OOM/非有限值。完整preflight日志及真实
  schema2 smoke产物已同步到本地`output/emotic_track_a_ensemble_control/20260903_191515/`。
- 四组最终均训练exit0、240 epochs/13950 updates/skipped0，日志无OOM/非有限值，每组8份schema2
  scores齐全。总launcher exit1且未生成comparison：新runner按`val_scores/`命名，分析器误沿用旧
  `validation_scores/`。已修改为显式兼容两者且只能出现一个，补目录名回归测试；不重训或调参，
  修复推送后仅对已有scores重新运行固定汇总，保留原失败状态用于审计。
- 启动记录提交后不更新正在运行的实验worktree；Automatic Upload的同内容运行文档备份到上述
  ensemble-control备份目录的launch-records子目录，再从primary自身HEAD恢复clean。

## 2026-09-04：完成等计算量validation归因并同步结果

- 新runner将score写入`val_scores/`而汇总器只识别旧`validation_scores/`，导致四组训练成功但
  原launcher exit1。修复`343ddca`同时接受恰好一种目录命名并新增回归测试；原失败记录保留。
- Automatic Upload写入服务器primary的5份同内容文件逐项SHA-256核验后备份到
  `/mnt/haoyuan/workspace/git-sync-backup-ensemble-control-1CWO07Me/score-folder-compatibility`，
  primary恢复clean；实验worktree经Git ff-only更新到`343ddca`，test-only未修改。
- 服务器ddp环境88项完整单测通过。直接使用既有seed0/1/2 full/person validation score重新汇总，
  没有重训、test访问、alpha/mode/threshold搜索或checkpoint生成。
- 单Full final mAP为`42.3854 ± 0.4964`；同seed Full+Person为`43.2785 ± 0.3954`，配对增益
  `+0.9236/+0.7763/+0.9795`，均值`+0.8931 ± 0.1050`。
- 等计算量循环配对中，Full+Full为`42.9568 ± 0.3957`，Full+Person为`43.2849 ± 0.3656`；
  人物视图净贡献`+0.2939/+0.3183/+0.3722`，均值`+0.3281 ± 0.0401`。average mAP净提高
  `+0.4616 ± 0.0820`，24个seed×task差值全部为正。
- final 26类中人物辅助相对Full辅助平均改善19类；Suffering/Sadness为`+1.9252/+0.5147`，
  Pain/Peace为`-1.1262/-0.6027`。Person单模型仍弱，但其辅助排序信息稳定优于普通第二个Full模型。
- 服务器output与logs已同步到本地`output/emotic_track_a_ensemble_control/20260903_191515/`；输出组和
  日志组的聚合SHA-256分别为`e6d38d896c3e43666128a31339d0d6dd94c460a499f3e3b27b286ff4f0853dd5`
  与`7308c27e4f323e540634ac55ef5fa2d0da76b9812b7e76dc6a0029a4923b3a19`。
- 下一步先用现有score做bbox质量分层和强约束的类别/几何感知门控validation，锁定后只进行一次test；
  若不能超过固定0.20融合，再转person token辅助selector结构。

## 2026-09-04：实现强约束全局权重与几何/类别门控搜索

- 从`343ddca`创建`exp/emotic-constrained-gated-fusion`，保留上一轮尚未提交的结果文档。只复用已有
  seed0/1/2 validation/test逐样本概率，不修改或重训Full/Person模型。
- 新分析器先在validation搜索probability person权重0.00--1.00、步长0.01；目标为三种子final mAP
  均值，average mAP破平，同时报告各seed独立最优但不据此分别选test规则。
- 从EMOTIC稳定sample ID重建原始bbox面积占比、绝对长宽比与同图人物数；固定12个面积×形状×人数
  cell，使用final task全26类报告Person及固定0.20融合相对Full的supported-class mAP。
- 类别方向只在固定0.20相对Full的AP变化三个seed均超过`±0.05`时启用；质量方向只在样本数至少50且
  cell相对总体融合增益三个seed均超过`±0.05`时启用。门控仅搜索class/quality delta各
  `{0,0.025,0.05}`，没有26类自由参数。
- 门控候选必须每个seed的final mAP都不低于全局alpha赢家且三种子均值更高，否则锁定全局赢家。
  validation选择先以独占写入保存，随后test只评估这一条锁定规则一次，输出显式记录候选数1和
  `search_performed_on_test=false`。
- 新增固定服务器路径launcher和回归测试；本地静态编译、shell语法与diff检查通过。本地系统Python
  缺少NumPy，完整测试将在服务器ddp环境执行。通过后按持续授权直接运行，不需再次审批。
- 首次服务器执行在validation选择前因launcher误把config中的内部`data_root`当作EMOTIC构造器根目录
  而退出；实际应传其父目录`.../datasets`。该次没有生成selection、没有加载test，已保留失败日志并
  修正默认路径；使用新batch目录重试，确保test单规则评估仍只发生一次。
- 修复`baf2263`推送并在独立服务器worktree ff-only同步；91项完整单测再次通过。成功batch为
  `constrained_gated_fusion_seed012_20260904_014055`，selection写入后才读取test，test候选数严格为1。
- 全局alpha validation搜索得到Person0.25，final/average mAP`43.3060/50.4747`；各seed独立峰值
  0.24/0.25/0.32。相对固定0.20，统一0.25的final均值仅提高0.0275，属于宽平台。
- bbox cell诊断显示大框+常规长宽比+多人最可靠，固定0.20融合增益`+1.6183`；小框+极端长宽比+
  多人仅`+0.2396`且seed2为负。大框+常规+单人仍提高0.5631，但三seed均低于总体融合增益。
- 15类在三个seed上均有至少+0.05 AP，未发现三seed一致负类。9个强约束门控中，base0.20、
  class delta0.05、quality delta0.025以validation final`43.3420 ± 0.3906`胜出，逐seed均高于0.25。
- 锁定规则唯一test得到final mAP`33.2696/32.7651/32.4793`，均值`32.8380 ± 0.4002`；相对旧
  固定0.20分别`+0.0024/+0.0323/+0.0005`，均值仅`+0.0117`。average mAP/cF1/oF1提高
  `0.0508/0.0159/0.0422`，forgetting恶化0.0129。
- validation类别方向只部分迁移到test：Anger/Suffering提高，Fear/Sadness下降。停止继续静态权重
  微调；下一步若继续，使用train/calibration监督的共享样本级可靠性门控，再无效则做Person token
  辅助selector结构。
- 成功output与全部日志已同步本地`output/emotic_track_a_constrained_gated_fusion/20260904_014055/`；
  聚合SHA-256分别为`881375876478ba886e0d617c7ae22580d42f20bdc875d907fb591d1d812a944b`
  与`5d1c24cd8dcfe53a78c6c02c602b0b91a672c1dd90e15e94f922ee9342603ed5`。

## 2026-09-03：同步双视图三种子并恢复离线融合分析

- 四组seed1/2 full/person训练均exit0、240 epochs/13950 updates/skipped0，8份scores/run齐全、
  无checkpoint或训练错误。六组（含seed0）每视图除seed/Git外配置一致、clean来源审计通过。
- 原自动融合失败、seed1/2与总launcher exit1：全表CPU sigmoid与runner原batch64路径产生极少数
  float32差异，改变AP并列排序。seed1 full task5有1个概率差7.45e-09、mAP差0.00439245；
  seed2 full task6 mAP差-0.00057095。不是训练失败、样本错位或权重参数漂移。
- 在独立analysis_recovery产物中按原batch64严格重建anchor，六组48个task所有指标/逐类AP与原记录
  在1e-12以内一致；融合仍用原全表`fused_scores`，alpha/mode/threshold不变，未放宽容差。
  seed0融合与既有JSON完全相同，batch化/全表的融合mAP在全部24个task也一致。原退出码保留为1。
- 融合seed0/1/2 final mAP为33.2672295/32.7327739/32.4787647，均值32.8262560±0.4024591；
  full均值32.1561593±0.3700888，配对增益0.7307103/0.5980922/0.6814878，均值0.6700968±0.0670388。
- 24个seed×task mAP增益全正；average mAP/cF1/oF1平均提高0.7936236/0.2051686/0.2425607，
  forgetting平均恶化0.0519295，seed1 cF1下降0.1019812。不能宣称所有指标或结果方差都改善。
- task6 Sadness/Sensitivity/Suffering AP平均变化+1.691491/-0.033294/+3.596102；最终task7为
  +1.894352/-0.035151/+3.498238。最终改善类别数21/19/18，按类别三种子均值为19/26。
- 新full seed1/2训练历史除耗时外与上一轮完全一致，final指标一致；average mAP仅有数值路径引起的
  -0.000551692/+0.000030786差异，未改变训练轨迹。
- 70个原始/恢复文件同步到`output/emotic_track_a_dual_view_seed_confirmation/20260903_174938/`；
  runs/logs/recovery及3组control的相对路径SHA清单摘要均与服务器相同，数值见该目录analysis.md。
  结果包含32份新增test scores，原seed0数据复用已有本地目录，不含checkpoint。
- 约6.4MB压缩包`dual_view_seed12_results_and_seed012_summary_20260903_174938.tar.gz`已生成于上述
  output父目录，含完整分析和恢复审计；SHA-256为
  `d2c3a236c6853b16ff289006fdbced6569826913d33a0053fa4d7d308aced822`。
- 三份上下文文档本地未提交；Automatic Upload写入primary的同内容文档按SHA校验后保存在
  `/mnt/haoyuan/workspace/git-sync-backup-dual-view-seed12-DbO8J9xs/results-analysis-records`，
  primary从自身HEAD恢复clean，训练工作树和test-only保持原样。
- 本轮只更新三份上下文文档与输出分析产物，未修改受控业务源码、未启动新实验。下一步先修复
  通用分析脚本及回归测试，再在validation做等成本full+full对照和人物框质量诊断；不继续test调参。

## 2026-09-03：实现锁定双视图seed1/2配对确认

- 用户要求固定全部参数补另外两个正式test，并解释结构与后续改进方向。从`39a30df`创建
  `exp/emotic-dual-view-seed12`；未修改模型、训练数据处理或loss算法。
- 核查服务器171份config及旧full seed1/2产物：其已完成但没有test scores或checkpoint，person
  seed1/2没有历史结果。不能由summary/mAP恢复逐样本概率，需要四个训练进程来产生两个融合seed。
- 正式脚本新增SEED=0/1/2，默认seed0行为保持不变；四卡launcher并行seed1 full/person和seed2
  full/person，完成后固定0.80/0.20 probability融合并自动与现有seed0汇总，不进行任何test搜索。
- 固定EMOTIC、8 tasks×30 epochs、batch64/workers2、Adam_reset_per_task/main LR0.0125/WD0、
  per-task cosine/minLR0/no warmup、CLIP ViT-B/16、Image-token layer1/b32/LR4e-4/scale0.1/
  ReLU/independent、main BCE+Adapter ASL9.8/0/0.05、legacy_full_zero、CLIP normalization、
  AMP/TF32 on、无checkpoint。full维持crop0.05；person维持margin0.15/letterbox/轻量jitter。
- 数据和权重沿用既有服务器路径；新结果放benchmark_runs的multi_lane_dual_view_seed_confirmation_v0.1，
  日志与控制汇总分别写独立工作树logs/output下emotic_track_a_dual_view_seed_confirmation。
- 新增三种子汇总器：强制相同锁定规则/validation SHA、完整训练预算、skipped0及除seed/Git外
  配置一致；报告mean±sample std(ddof1)和同seed fusion-full差值，非显著性检验。
- 新增三项单测覆盖统计值、重复seed拒绝、融合规则漂移拒绝；本地静态编译、shell语法及diff检查通过。
  提交推送后在独立服务器worktree运行完整单测和Adapter GPU smoke，通过后才启动正式训练。
- 实现`d2442ee`已推送，服务器建立clean独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-dual-view-seed12`。Automatic Upload的9个完全一致副本
  已逐文件校验SHA并备份到`/mnt/haoyuan/workspace/git-sync-backup-dual-view-seed12-DbO8J9xs`，
  恢复primary clean；test-only及其他工作树未改动。
- ddp环境75项完整单测通过；GPU0真实ViT-B/16 smoke通过，Image-token layer1/b32、independent、
  fixed scale0.1、Adapter-ASL、AMP/TF32配置下trainable739130、零初始化logits最大差6.1035e-05。
  新汇总器对真实seed0的训练完整性/配置检查通过，并正确拒绝重复seed。模型/runner/Adapter/
  dataset源码与seed0的`c9fa74c`没有差异。
- batch`dual_view_locked_formal_seed12_20260903_174938`及tmux
  `multilane_dual_view_seed12_20260903_174938`已启动，安全空闲检查连续两次通过；GPU0/1为seed1
  full/person，GPU2/3为seed2 full/person。四份config均clean`d2442ee`，与对应seed0逐字段对比
  除seed/Git外无变化；四组已开始task0，单进程初始显存约2GB。
- 启动后task0前几轮均84 steps/skipped0，无OOM或数值错误；GPU0--3分别仍余约21.6/22.1/22.1/
  22.1GB显存。运行记录只提交本地文档，不更新活动服务器工作树，避免改变运行期源码快照。
- 完成后每seed自动固定融合，复用原seed0产生三种子formal_seed_summary.json；目前未获得seed1/2
  新结果，不提前宣称跨seed稳定收益。本次只启动确认实验，没有启动full+full对照、权重搜索或蒸馏。

## 2026-09-03：恢复任务并同步锁定双视图正式test

- 确认之前安全等待器已自动完成，tmux已退出，无需启动或补跑。两组同一clean`c9fa74c`，各240
  epochs、13,950 updates、skipped0，3个退出码均0，无错误、无checkpoint。
- 以原validation选择文件SHA锁定probability alpha0.20，重新计算固定test融合与自动JSON完全相同。
  两组training_history除elapsed_seconds外，全部字段与对应validation历史逐epoch完全一致。
- 同批full以`32.5365194`复现旧冠军；person为`28.7473550`，固定融合为`33.2672295`，超过full
  `0.7307101`。融合average mAP/cF1/oF1=`40.0544613/32.9485639/49.5947888`，forgetting
  `4.7477654`比full略差`0.0169168`。8个task mAP/F1全部提高，21/26最终类别AP提高。
- task6 Sadness/Sensitivity/Suffering AP增益为`0.1103/0.1351/4.4223`；对应final task7增益为
  `0.4832/0.1330/4.3098`，报告时区分引入task与最终task。
- 34个原始文件通过scp同步本地；runs/control/logs聚合SHA-256分别为
  `6fdfd37aa595975c5e070f69b732eab515597a116f7f4807db26f93a0aa14086`、
  `1abab97b8bc8a2b2625184b8ca0a3472595e4202f20e3d9f425187c63baf3a8f`、
  `487a6136c1ba892596df0225ee7f4d37ae6fc4496addc252b36a8fd0886367bc`，与服务器完全一致。
- 详细分析位于`output/emotic_track_a_dual_view_formal_test/20260903_145816/analysis.md`。
  本轮未新开训练；建议下一步锁定配置补seed1/2，不根据test继续改alpha/阈值/类别专用规则。
- 3.2MB小结果包`output/emotic_track_a_dual_view_formal_test/dual_view_formal_test_seed0_20260903_145816_small_results.tar.gz`
  已包含分析、JSON、16份test scores和日志，不含checkpoint；SHA-256为
  `923092d7899029c4af4430c3f707eb228f6dbf3ea982db4e46297c45a0f794a0`。

## 2026-09-01：阶段一完成并启动阶段二ASL局部8组正式test

- 阶段一8组均完成240 epochs、13950 updates、skipped0，无运行异常。同批
  LR4e-4/scale0.1精确复现single1历史冠军final/average mAP`32.5365/39.1445`并排名第一。
- 次优LR5e-4/scale0.075为`32.3851/39.2462`，average高`0.1017`但final低`0.1514`；
  task0--4略高、task5--7下降，Sadness提高约`4.18`而Suffering下降约`4.50` AP。
  LR6e-4两组至少低`0.4977`，不扩展7e-4；阶段二固定LR4e-4/scale0.1。
- 阶段一40个小文件已同步，无checkpoint。180KB archive两端SHA-256均为
  `403004fba2f2244dc9c1f630c1f1490f89cc256dfe734d808a9cda64a941bc89`；分析文件为
  `output/emotic_track_a_image_token_asl_layer1_lr_scale_formal/20260901_171236/analysis.md`。
- 阶段二按用户要求继续直接report test。批次
  `image_token_asl_layer1_asl_local_formal_seed0_20260901_182542`包含锚点9.8/0/0.05、
  gamma-neg 8/12、clip0.0375/0.075及gamma-pos0.25/0.5/1.0共8组；固定layer1/b32/
  LR4e-4/scale0.1/ReLU/independent、AMP/TF32 on、无checkpoint，GPU0--7各一组。

## 2026-09-01：启动layer1 Adapter LR×scale阶段一8组正式test

- 用户明确validation与test耗时相近，要求跳过validation并直接并行全部8组held-out test。
  固定zero-based layer1、b32、ReLU、independent、主模型BCE + Adapter ASL 9.8/0/0.05、
  seed0、8 tasks×30 epochs、batch64、CLIP normalization、crop0.05、AMP/TF32 on、无checkpoint。
- 组合为LR4e-4×scale`0.075/0.1/0.125`、LR5e-4×三种scale、LR6e-4×scale`0.075/0.1`；
  `4e-4/0.1`作为同批冠军锚点，高风险`6e-4/0.125`未运行。
- 批次ID为`image_token_asl_layer1_lr_scale_formal_seed0_20260901_171236`。两次SSH握手被服务器
  reset，端口与密钥复核成功后才执行状态检查，未产生半启动目录。
- GPU0--7启动前均空闲、无runner；clean实验worktree为`94f3327`。8组各占一张GPU，均已生成
  config并开始初始化；每卡约1.8--2.5GB、最低仍余21.6GB，无OOM或启动异常。

## 2026-09-01：同步5组每层b32多层结果，多层路线收口

- 5组均完成240 epochs、13950 updates、skipped0，日志无OOM、非有限值、Traceback或写盘
  错误。最佳`[1,4]` b32/s0.05的final/average mAP为`32.2213/38.9746`，相对single1冠军
  低`0.3152/0.1699`；task1--7均下降，task6/7低`0.2804/0.3152`。
- `[0,1]` b32/s0.05、三层`[0,1,2]` b32/s1/30、`[1,8]` b32/s0.05、`[1,2]`
  b32/s0.1的final mAP依次为`32.0605/31.9493/31.7306/31.6235`，均未超过冠军或首批
  多层最佳`[1,2]` b16/s0.05的`32.3233`。
- b32相对小容量对应组的final变化为`[0,1] +0.1816`、`[1,4] -0.0277`、
  `[1,8] -0.3211`、三层`+0.1629`、`[1,2]` s0.1 `-0.1307`。容量效应没有一致方向，
  无法形成可推广的多层收益。
- 5组25个小文件已单独同步，不含checkpoint；112KB archive两端SHA-256均为
  `72995ac436e75d0f9d51eff3da2c0c1c0823cbc8051b47c4b13d3b7e8eb37698`。详细分析见
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_b32_extension5/analysis.md`。
- 服务器已无本批runner。20组统一结论为固定single1 b32/LR4e-4/scale0.1；停止多层、
  bottleneck和Adapter LR扩展。当前未启动新实验。

## 2026-09-01：首批15组完成并追加5组每层b32多层test

- `image_token_asl_layer1_local_multilayer_formal_seed0_20260901_100337`首批15组均完成240
  epochs、13950 updates、skipped0。layer1局部网格最佳b32/LR3e-4 final mAP`32.1281`，
  其余b24/b28/b40组合为`31.5292--32.0788`，均未超过已完成的single1 b32/LR4e-4
  冠军`32.5365`。single7 FP32补测为`31.7931`，也未取胜。
- 多层归一化最佳为`[1,2]` b16/scale0.05的`32.3233`，其次`[1,4]` b16/scale0.05
  `32.2490`；`[1,8]`为`32.0516`、`[0,1]`为`31.8789`、三层b12为`31.7864`。
  `[1,2]` b32/scale0.05仅`31.7764`，而b16/scale0.1为`31.7542`，说明容量或总残差强度
  放大都没有改善该相邻双层。
- 用户要求多层再补每层b32版本。新增5组：`[0,1]/[1,4]/[1,8]` b32/scale0.05、
  `[0,1,2]` b32/scale1/30、`[1,2]` b32/scale0.1；统一FP32/TF32 on、LR4e-4、正式test、
  无checkpoint。它们用于补齐跨深度组合和容量/强度上界，不改变已完成的归一化结论。
- 三层b32 FP32/TF32-off严格smoke通过，可训练参数839034、零初始化最大差`1.49e-8`。
  5组已在GPU0--4启动，20个总配置目录均已生成，初始核验无OOM、非有限值或Traceback。
- 首批15组已独立同步本地：75个小文件、无checkpoint，压缩包328KB，两端SHA-256均为
  `b6eb43e335ec0031f36d022ac701ff80aba0f78c29be27af393e906221169cbb`。新增5组仍在服务器运行，
  没有混入该包。
- 详细逐task与类别分析确认：最佳多层`[1,2]` b16/s0.05相对single1在8个task全部下降，
  final/average mAP低`0.2132/0.2309`，仅forgetting改善`0.1205`；single7 FP32在task6/7低
  `0.7912/0.7434`。分析文件为
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_initial15/analysis.md`。

## 2026-09-01：并行启动single7 FP32、layer1局部网格和归一化多层test

- 用户要求直接并行三条held-out test路线。批次
  `image_token_asl_layer1_local_multilayer_formal_seed0_20260901_100337`共15组，统一为seed0、
  8 tasks×30 epochs、batch64、主模型BCE + Adapter ASL 9.8/0/0.05、ReLU、independent、
  CLIP normalization、crop0.05、TF32 on、无checkpoint。
- single7使用FP32、layer7/b32/LR4e-4/scale0.1，补跑此前AMP非有限失败的高validation候选。
  layer1局部网格使用AMP，运行bottleneck`24/28/32/40` × LR`3e-4/4e-4`中除已完成
  b32/LR4e-4冠军外的7组。
- 多层使用FP32并围绕layer1设计7组：`[0,1]/[1,2]/[1,4]/[1,8]`采用每层
  b16/scale0.05，使总参数量和总残差强度近似single1 b32/scale0.1；增加`[1,2]`
  b32/scale0.05与b16/scale0.1两个解耦控制，以及`[0,1,2]` b12/scale1/30三层归一化。
- 初始FP32/TF32-on smoke被纯FP32容差拦截，单层/双层/三层最大差约`1.1e-5--2.2e-5`；
  判定为TF32重复前向数值差而非多层特有错误。FP32/TF32-off严格复测中single7、四种双层及
  三层全部通过，最大差约`6.52e-8`；正式训练按用户要求保持TF32 on。
- 服务器主实验worktree为clean`94f3327`，8张GPU启动前均空闲。GPU0--6各两组、GPU7一组；
  15组全部生成config并进入task0。各卡总显存约2.6--5.5GB，最低仍余18.6GB；启动核验没有
  OOM、非有限值或Traceback。当前不持续盯跑，完成后同步JSON与日志分析。

## 2026-09-01：同步层位置正式test，layer1成为探索性新冠军

- 批次`image_token_asl_layer_formal_seed0_20260831_154234`已结束：single1/2/3/4/5/11均完整
  完成240 epochs、13950 updates、skipped0；single7在task3 epoch16后因非有限ASL logits
  终止；pair`[2,3]`完成240 epochs，但task3 epoch17--29共跳过91次AMP optimizer update。
- zero-based single1的final/average mAP为`32.5365/39.1445`，相对layer8旧冠军
  `32.4193/38.8789`提高`0.1172/0.2656`；cF1/oF1提高`0.4167/0.2811`，forgetting降低
  `0.0934`。除层索引外配置逐字段相同，8个task mAP均高于layer8，因此它是当前正式指标的
  探索性新冠军。
- single1的收益集中于task4/5（`+0.5355/+0.6315`），task6/7仅`+0.0329/+0.1172`；最终
  Fear/Fatigue/Aversion等改善，但Suffering/Sadness下降`7.0468/4.9255` AP，仍有类别交换。
- 完整候选其余排名为single4`32.0577`、single5`31.9281`、single2`31.9014`、single3
  `31.3633`、single11`31.0015`、pair`[2,3]``30.8747`。多层既无总mAP收益又有AMP跳步，
  不再增加层数。
- 旧FP32 validation与新AMP test在6个完整单层上的final mAP Pearson相关仅`-0.273`；
  layer3的validation第一没有外推到test，数值模式与split共同造成排序反转。single7的失败
  也表明AMP对ASL层位置并非无关扰动。
- 结果已同步至本地，39个小文件、不含checkpoint。192KB archive在服务器与本地SHA-256均为
  `d8e22c7c8cab09af697ce9c705530af5446c7de51b168d2b3049037cf46380c4`。分析见
  `output/emotic_track_a_image_token_asl_layer_formal/image_token_asl_layer_formal_seed0_20260831_154234/analysis.md`。
- `+0.1172`是从多个held-out test候选中筛出的探索性小效应，存在多重比较乐观偏差；不能作为
  稳定或显著提升。当前未启动新实验。

## 2026-08-31：审计层位置结果并排队8组正式test

- 重新核对历史产物后确认：Image-token Adapter单层`0--11`搜索、15组多层screen以及
  single8/single9/pair8_9 BCE确认均为seed0完整8-task validation-only，此前没有对
  layer3/layer7/layer11或多层结构运行held-out test。
- validation与test使用相同训练协议但不同评估split，并非互不相关；此前不同批次validation
  波动约`0.4--0.6` mAP，而且层搜索使用AMP off/TF32 on、当前正式冠军使用AMP/TF32 on，
  因此validation排序不能保证test排序。
- 按用户当前“只追最终test mAP、个别类别可下降”的目标，取消旧的类别/F1/forgetting否决
  门槛。单层ASL同批超过disabled的候选为zero-based layer`1/2/3/4/5/7/11`；其中layer3、
  layer7相对同批layer8 final validation mAP分别高约`1.1007/0.8227`。多层仅保留各自同批
  超过disabled且最高的`[2,3]`，其余多层ASL不进入test。
- 8组共同配置为EMOTIC seed0、8 tasks×30 epochs、batch64、主模型BCE + Image-token
  Adapter ASL 9.8/0/0.05、b32、Adapter LR4e-4、scale0.1、ReLU、independent、CLIP
  normalization、crop0.05、AMP/TF32 on、reporting test、无checkpoint。
- 批次ID为`image_token_asl_layer_formal_seed0_20260831_154234`，输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_asl_layer_formal_v0.1/`。
  启动时8张卡均被另一批进程占用、仅余约6--10GB；已为GPU0--7各挂一个等待器，要求空闲
  显存至少20GB且连续两次60秒检查通过后才启动。8个等待器均已核验存活，当前未抢占显存。

## 2026-08-31：直接启动 b32 附近容量/LR 8组正式test

- 用户明确validation与test耗时接近时可跳过validation，直接追求最高正式test mAP。因此固定
  当前最优b32/LR4e-4/scale0.1/ReLU协议，将bottleneck`24/28/32/40` × LR`3e-4/4e-4`
  全8组直接运行seed0 held-out test，不先运行validation。
- 批次ID为`image_token_asl_local_capacity_lr_amp_formal_seed0_20260831_124751`，GPU0--7各一组。
  共同配置为8 tasks、30 epochs/task、batch64、layer8、scale0.1、ReLU、independent、主模型BCE +
  Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、AMP/TF32 on、无checkpoint。
- 启动前8卡均有其他高利用率任务，但仍余8.6--11.8GB显存；按用户要求不等待空闲，不停止
  其他进程，每卡追加一组。启动后最低仍余约6.2GB，未接近OOM。
- 8组task0已到epoch3--4，均84 steps、skipped0，单epoch约15--17秒，无OOM、NaN、Inf、
  Traceback、RuntimeError或写盘错误。服务器实验worktree为clean`94f3327`。
- 8组随后均完成240 epochs、13950 updates、skipped0，完成标记各唯一，日志无OOM、NaN、Inf、
  Traceback、RuntimeError或写盘错误。同批b32/LR4e-4精确复现历史冠军全部主指标和曲线，
  final mAP为`32.4193`，仍是第一。
- 排名2--8为b24/LR3e-4`32.2492`、b28/LR4e-4`32.1666`、b32/LR3e-4`32.1172`、
  b40/LR4e-4`31.9602`、b28/LR3e-4`31.8790`、b24/LR4e-4`31.2931`、b40/LR3e-4
  `31.1764`；没有新冠军。
- b24/LR3e-4虽final mAP低`0.1702`，但average mAP高`0.2668`、task5--7平均高`0.0104`。
  组别分解显示它改善多个早期任务，但task6组均低`3.5030`，Suffering低`8.2892`，因而最终落后。
- 41个小文件已同步本地，archive 180KB，SHA-256为
  `acc4e0682ecc6ff170dc92e1663dafb4d94e97cc5aa7d56c03696b440dc1a978`，不含checkpoint。统一容量/LR
  搜索基本收口；下一步应转ASL局部微调或task-dependent Adapter超参。

## 2026-08-31：并行启动 b16 AMP 正式test与 scale×activation 搜索

- 用户要求不等b16正式结果，直接利用8卡同时完成一组b16 AMP/TF32-on held-out test和
  8组b32 scale×activation完整validation。为对齐历史冠军，validation也使用AMP/TF32 on。
- 共同配置为seed0、8 tasks×30 epochs、batch64、layer8、LR4e-4、independent、主模型BCE +
  Image-token Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、无checkpoint。formal使用b16、
  scale0.1/ReLU/reporting test；validation固定b32，搜索scale`0.025/0.05/0.1/0.2`和ReLU/GELU。
- 启动前GPU0--7全空闲、无残留runner，服务器实验worktree为clean`94f3327`。初次SSH握手
  被重置两次，端口和密钥复核后恢复，未产生半启动任务。
- formal batch为`image_token_asl_b16_amp_formal_seed0_20260831_114209`，validation batch为
  `image_token_asl_scale_activation_amp_seed0_20260831_114209`。GPU0同时运行formal和一组validation，
  GPU1--7各一组；共9个tmux会话。
- 9组task0 epoch2均完成84 steps、skipped0，GPU0双进程约5.2GB，其他卡约2.4GB，无
  OOM、NaN、Inf、Traceback、RuntimeError或写盘错误。预计约45--55分钟完成。
- b16 AMP/TF32-on formal完整完成240 epochs、13950 updates、skipped0，final/average mAP为
  `31.3796/38.8129`，未超过历史b32冠军`32.4193/38.8789`。它在task0--4略高，但task6/7
  低`1.0856/1.0397`；因而排除“只对齐AMP/TF32就能让b16取胜”的假设。
- scale×activation中`0.1/GELU`、`0.2/ReLU`、`0.2/GELU`在task3因非有限ASL logits终止，
  其余5组完成240 epochs、13950 updates、skipped0。原`0.1/ReLU`在final/average/task5--7平均mAP
  三项均第一，为`42.6673/49.2378/44.8851`；其他有效组final mAP低`0.5031--0.7962`。
- 结论是保留b32/LR4e-4/scale0.1/ReLU/AMP+TF32，不运行新正式test，不继续扩scale或修复
  GELU。下一局部搜索候选为bottleneck`24/28/32/40` × LR`3e-4/4e-4`。
- 43个小文件已同步本地，archive为160KB、SHA-256
  `29c87d99ddcc868bd0c1989ec921d5c22c20a36c9a6a6c77c9d8220254241461`，不含checkpoint。

## 2026-08-31：启动 b16/b64 LR4e-4 正式test

- 用户明确目标改为尽快最大化正式test final mAP，不再以Sadness、Sensitivity、
  Suffering、F1或forgetting作为候选否决门槛。阶段一validation最高的b16/LR4e-4
  与次高b64/LR4e-4进入一次性seed0 held-out test。
- 服务器GPU0--7启动前均空闲；clean实验worktree为
  `/mnt/haoyuan/workspace/multi-lane-main-asl-routing@94f3327`。b64真实CLIP FP32/TF32-off
  Adapter-ASL smoke通过，可训练参数788314。
- 两组于00:35在GPU0/1并行启动，batch ID为
  `image_token_asl_capacity_formal_seed0_20260831_003546`。共同配置为8 tasks、30 epochs/task、
  batch64、layer8、Adapter LR4e-4、scale0.1、ReLU、independent、主模型BCE + Adapter ASL
  9.8/0/0.05、CLIP normalization、crop0.05、AMP/TF32 off、test reporting；仅bottleneck为16/64。
- 服务器磁盘显示100%；为避免历史正式运行每组约2.7GB checkpoint导致中途写入失败，
  本轮显式关闭checkpoint，预计每组仅约150KB JSON外加日志。根目录在root保留块上的
  16MB写入测试通过。
- b16/b64 task0 epoch1均完成84 steps、skipped0，loss为`0.614753/0.614830`，Adapter loss为
  `0.028270/0.028179`；GPU显存约3.4/3.0GB，无OOM、NaN、Inf、Traceback或RuntimeError。
- 两组后续均完成240 epochs、13950 updates、skipped0，完成标记各唯一且日志无数值、显存或
  写盘错误。b16的final/average mAP为`32.0269/38.6674`，b64为`31.3193/38.4295`；
  两者均未超过历史b32冠军`32.4193/38.8789`。
- b16相对冠军final mAP仅低`0.3924`，cF1/oF1反而高`0.1317/0.2205`；b64的final mAP
  低`1.1000`。b16比b64高`0.7076` final mAP，容量64不再保留。
- 历史b32配置为AMP/TF32 on，本轮b16/b64为AMP/TF32 off；核心代码差异只增加非有限值
  检查和checkpoint开关，正常loss路径未改变。因此下一个最小对照应只b16对齐冠军精度模式。
- 11个小文件与两份日志已同步本地；压缩包48KB，SHA-256为
  `66ef7139375d2913830bc5683f26c378506c3e2430fc7f18b47d0b9bd74d1484`，不含checkpoint。
- 按用户批准对服务器checkpoint做了限定清理：只删除CocoER、旧Task Adapter Bank、
  EMOT-Net+CCIM和DSCT的8个明确失败/中止实验`checkpoints/`目录，保留日志、
  配置与指标。实际释放`6796956 KiB`（约6.48 GiB），8个目标均已复核不存在。
  删除未触及任何成功实验或当前b16/b64运行；两组正式test仍正常。ext4预留块使
  `df`仍显示100%，root可用的实际空间约139.3 GiB。

## 2026-08-26：准备 Image-token Adapter-ASL 容量/LR阶段一

- 用户明确固定主模型BCE、Adapter ASL，不再把Adapter-BCE作为主优化方向。从
  `exp/emotic-image-token-pair89-confirmation@4213310`创建
  `exp/emotic-image-token-asl-capacity-lr`。
- 阶段一预注册12个Adapter-ASL组合：zero-based layer8、scale0.1、ReLU、independent、
  ASL 9.8/0/0.05固定，联合搜索bottleneck`8/16/32/64`和Adapter LR
  `1e-4/2e-4/4e-4`；加入fresh Adapter-disabled joint-BCE锚点，共13组。
- 8卡launcher先并行8组，GPU0--4完成首组后自动接续剩余5组，一次启动即可完成阶段一。
  每组均为seed0完整8-task val-only、30 epochs/task、batch64、legacy、CLIP normalization、
  crop0.05、AMP off、TF32 off、无checkpoint；不运行正式test。
- 汇总器逐组核验同一clean commit/tree、240 epochs、13950 updates、skipped0、无checkpoint、
  精确Adapter参数量和ASL路由。候选需相对disabled与b32/LR4e-4锚点通过全部类别/F1/forgetting
  门槛，且final mAP至少提高0.5，才允许进入scale×activation阶段。
- worker已增加`NO_TF32`开关，smoke增加`--no-tf32`并输出实际TF32状态；新增b64最大容量
  Adapter-ASL smoke、13组自动队列、严格汇总器和3项测试。本地Shell语法、Python3.9编译及
  `git diff --check`通过；本机系统Python缺少PyTorch，完整测试待服务器`ddp`环境执行。
- 实现提交`94f3327`已推送并由服务器独立实验worktree安全切换；服务器完整Track-A测试42/42
  通过。GPU1 b64真实CLIP严格FP32 Adapter-ASL smoke通过，日志确认TF32 off、可训练参数
  788314、zero-init最大差`1.8626451e-08`。
- 13组队列在tmux`mla_image_token_asl_caplr_s0_20260826_164945`启动，batch ID为
  `image_token_asl_capacity_lr_seed0_20260826_164945`。首轮GPU0--7依次运行disabled、冠军
  b32/LR4e-4、b8三种LR及b16三种LR；8组均已完成至少task0 epoch4，84 steps、skipped0，
  ASL loss有限且无异常。
- 启动观察时每卡总显存约2.8--3.0GB、剩余21GB以上。GPU0--4首组完成后会自动接续b32低LR
  和b64三种LR；当前不运行正式test，实验worktree固定clean`94f3327`直至13组完成并汇总。
- 用户确认可以单卡并行多组时，首轮8组已到task4，因此不停止重跑。新增临时detached clean
  worktree`/mnt/haoyuan/workspace/multi-lane-main-asl-capacity-lr-parallel@94f3327`，把原计划
  第二轮5组立即分别加入GPU0--4；13组现已全部同时运行。
- 新增5组task0 epoch1均完成84 steps、skipped0，约24--25秒；GPU0--4双进程实际总占用
  5.9--6.1GB、剩余约18GB，无OOM或非有限ASL loss。launcher本地实现同步改为默认13进程
  同启，便于未来一键复现；当前批次完成后需手动运行严格汇总，避免原串行父launcher的重复
  目录检测影响结果判定。

- 13/13组随后全部完成并已同步到本地结果包；每组均为240 epochs、13950 updates、skipped0，
  无OOM、NaN、Inf、Traceback或RuntimeError。fresh disabled、b16/LR4e-4、b64/LR2e-4的
  final mAP分别为`42.0628/42.5718/42.3616`，但最高mAP候选牺牲Sensitivity并恶化
  forgetting；b32/LR1e-4虽通过锚点门槛，却相对fresh disabled压低task6的Sadness和Suffering。
  严格汇总无合格winner，不进入scale×activation、不扩大容量/LR、不运行正式test。详细结果与
  白名单归档见本地`output/emotic_image_token_tuning/asl_capacity_lr/...results_package_v2/`。
- 2026-08-30再次从服务器核验并同步同一batch，严格汇总内容与原分析完全一致。
  本地新归档为`..._results_20260830.tar.gz`（344 KB），SHA-256校验通过，含68个结果文件和
  19个日志，不含checkpoint。服务器实验worktree仍为clean`94f3327`。

## 2026-08-26：同步并分析 pair8/9 BCE 同批确认

- batch`image_token_pair89_confirmation_seed0_20260825_132139`四组全部完成；每组8 tasks、
  240 epochs、13950 updates、skipped0、exit code0，同一clean`e50b4a3`，无OOM、NaN、Inf、
  异常或checkpoint。
- disabled/single8/single9/pair8_9的final mAP分别为
  `41.863114/42.422405/41.615440/42.350681`，average mAP分别为
  `48.406695/49.309657/47.820751/49.008467`。single8是本批final/average mAP最高且
  forgetting最低的结构。
- pair相对fresh single9虽提高final/average/task6 mAP`0.735241/1.187716/1.020656`，但
  Sensitivity下降`1.210430`、forgetting恶化`0.216919`；相对disabled也使forgetting恶化
  `0.193886`。严格结果为`confirmed_for_formal_test=false`，不运行正式test。
- single8相对disabled在task0--7各阶段mAP均提高，final/average/task6提高
  `0.559291/0.902963/0.652256`且forgetting改善`0.093862`，但Sadness下降`5.071147`，仍未
  通过全部硬门槛。single9整体退化；停止single9、pair、多层、扩容与ASL。
- task6训练视图只有627个样本，三类正例为Sadness260、Sensitivity316、Suffering188；
  Sensitivity并非最稀有却只有约6--7 AP。pair将每task Adapter参数从49952加倍到99904后主要
  产生类别收益交换和更高forgetting，现象更符合小数据下的方差/过拟合而非容量不足。
- 28文件小结果包已同步本地并逐项SHA-256通过，不含checkpoint；archive SHA为
  `3e449853a5520b30d7c413a41a3283ddf332757065f1ac5bd33fbcb386982ad2`，详细分析见结果目录
  `analysis.md`。
- 配置复核发现此前日志中的FP32表示AMP关闭，但`config.json`实际为`tf32=true`；四组相对比较
  仍公平，下一轮若筛选约0.1--0.5点小效应，应先在launcher显式传`--no-tf32`并同批重建
  disabled/single8控制，避免继续混用“AMP off”和“严格FP32”概念。

## 2026-08-25：准备 pair8/9 BCE 最小同批确认

- 用户确认执行fresh disabled、single8、single9、pair`[8,9]`四组最小BCE validation；从
  `exp/emotic-image-token-multilayer-screen@1014831`创建
  `exp/emotic-image-token-pair89-confirmation`。四组使用GPU0--3同批一轮，GPU4--7不占用。
- 固定seed0、完整8-task val-only、FP32、30 epochs/task、batch64、legacy、CLIP normalization、
  crop0.05、b32、Adapter LR4e-4、scale0.1、ReLU、independent、无checkpoint；不运行ASL或
  held-out test。
- 新增4-GPU资源门控启动器和严格确认汇总器。pair除相对disabled的全部硬门槛外，还必须相对
  fresh single9保护final/average/task6 mAP、task6三个新类与forgetting，F1下降不超过0.5；
  只有全部通过才标记`confirmed_for_formal_test=true`。
- 本地Shell语法、Python3.9编译、3项隔离汇总器测试和`git diff --check`通过；完整Track-A测试
  与pair8/9真实CLIP smoke待Git同步服务器后执行。
- 实现提交`e50b4a3`已推送并安全同步到服务器独立实验worktree；服务器完整Track-A测试
  39/39通过。pair`[8,9]`真实CLIP FP32 BCE smoke通过，可训练参数789082，zero-init最大差
  `1.4901161e-08`，未发现冻结、梯度路由或数值异常。
- 四组已于2026-08-25 13:21在tmux`mla_image_token_pair89_s0_20260825_132139`启动，batch ID为
  `image_token_pair89_confirmation_seed0_20260825_132139`。GPU0--3分别运行disabled、single8、
  single9和pair8_9；四组task0 epoch1均完成84 steps、skipped0，耗时12.5--14.6秒，训练loss
  有限且日志中无OOM、NaN、Inf、Traceback或RuntimeError。
- 启动观察时每卡总显存占用约2.4--2.8GB、剩余21.2GB以上；GPU4--7保持空闲。实验worktree
  固定clean`e50b4a3`直至四组完成并自动生成严格确认汇总；当前没有ASL、其他多层或正式test。

## 2026-08-25：准备 Image-token Adapter 多层受控诊断

- 用户提出直接对layer3/layer11运行正式test并探索多层。根据上一阶段预注册规则，两者都不能
  视为预期稳健：layer3 ASL虽final mAP提高`0.976536`但Suffering下降`2.695190`；layer11
  ASL虽改善Sadness/Suffering但Sensitivity下降`1.126123`、forgetting恶化`0.300807`。因此
  暂不消耗held-out test，先做同批validation复核。
- 从`exp/emotic-image-token-structure-tuning@5ba055b`创建
  `exp/emotic-image-token-multilayer-screen`。设计15组8卡两轮：disabled，single3/single11
  的BCE/ASL，以及`[2,3]`、`[3,7]`、`[3,11]`、`[8,9]`、`[7,8,9,10,11]`五种多层的
  BCE/ASL。最后一组表示物理第8--12个block；所有代码参数继续使用zero-based索引。
- 新增多层8卡启动器和严格汇总器。多层必须先通过相对disabled的全部硬门槛，ASL还必须
  相对同结构BCE通过；此外final/task6 mAP不能低于同批single3/single11同loss最佳值。只有
  通过者才允许讨论正式test，不能只按多层final mAP排序。
- 当前实现的多层Image-token Adapter在各指定block独立处理冻结`LN1(CLS+patch)`，仅影响该层
  selector匹配，不把适配token写回CLIP残差流，也不是前一层Adapter输出喂给后一层的级联结构。
  5层配置每task Adapter参数为249760，8卡24GB 4090预计有充足空间，仍先以真实CLIP FP32
  `[3,11]` smoke验证forward/backward、梯度路由和冻结行为。
- 本地Shell语法、Python3.9编译、3项隔离多层汇总测试与`git diff --check`已通过；完整Track-A
  测试和GPU smoke待代码经Git同步到服务器独立实验worktree后执行。
- 实现提交`976f2a1`已推送并安全同步；服务器完整Track-A单测36/36通过。GPU0真实CLIP
  layers3+11 FP32 Adapter-ASL smoke通过，可训练参数789082，zero-init最大差`1.7695e-08`，
  无冻结、梯度或数值异常。
- 15组队列在tmux`mla_image_token_multilayer_s0_20260825_111638`启动，batch ID为
  `image_token_multilayer_screen_seed0_20260825_111638`。第一轮8组均完成task0 epoch1，84
  steps/skipped0、16.0--18.0秒；GPU总占用约2.4--3.3GB且每卡仍余20.8GB以上，无OOM、NaN、
  Traceback或RuntimeError。剩余7组由各lane在第一轮结束后自动接续，不启动正式test。
- 15组最终全部完成，每组240 epochs/13950 updates/skipped0/exit code0，同一clean`976f2a1`，
  无异常且无checkpoint。唯一严格合格多层为`[8,9]` BCE：相对disabled final/average/task6
  mAP提高`0.939906/0.844164/1.007650`，task6三类提高
  `3.473313/2.236642/3.889481`，cF1提高`1.052756`，forgetting改善`0.048449`。
- 5个多层ASL相对各自BCE的final mAP全部下降`0.252--1.369`，没有ASL赢家；五层dense后段
  同样压低Suffering并恶化forgetting，停止ASL和继续堆层。
- `[8,9]`相对fresh single11 BCE的final/task6优势仅`0.185944/0.189651`；本批缺少fresh
  single9对照，而跨批pair相对旧single9绝对final只高`0.113899`，低于disabled跨批
  `0.423396`波动。因此当前只保留候选，不启动正式test，下一步最小验证是BCE-only fresh
  single8/single9/pair8_9。
- 93文件小结果包已同步本地并逐项SHA-256通过，不含checkpoint；archive SHA为
  `ddd5c6c0a07702a70094f4b89cec2cf0a2ef47db5be5a2006d60630688ce2f44`，详细分析已写入本地
  结果目录`analysis.md`。

## 2026-08-25：同步并分析 Image-token Adapter 单层位置搜索

- batch`image_token_layer_search_seed0_20260822_235242`的25/25组全部完成：同一clean
  `d63da6b`、每组8 tasks/240 epochs/13950 updates/skipped0/exit code0，总计6000 epochs与
  348750 updates，无OOM、NaN、Inf或异常，无checkpoint；tmux已退出且无残留runner。
- disabled锚点final/average mAP为`41.955741/48.656124`。BCE的aggregate最佳为layer9，
  final/task6 mAP提高`0.402611/0.469971`，但Sadness下降`1.836305`、forgetting恶化
  `0.216600`；全部12个BCE层都使forgetting恶化，没有合格层。
- ASL aggregate最佳为layer3，final/average/task6 mAP提高`0.976536/0.956794/1.048385`，
  但Suffering下降`2.695190`、forgetting恶化`0.043960`。layer7牺牲Sadness换Suffering，
  layer11同时提高Sadness/Suffering但牺牲Sensitivity并显著恶化forgetting；没有ASL层同时
  通过相对同层BCE和disabled的全部门槛。
- 自动汇总最终为`eligible_bce_layers=[]`、`eligible_asl_layers=[]`、两者winner均null，按预先
  规则不进入bottleneck/LR，不扩大容量或多层，不启动test。位置效应非单调且主要表现为
  task6类别收益交换，不能只按final mAP选择layer3。
- 本批layer8 BCE与上一批同训练数学/seed的FP32 layer8 BCE仍有final/average mAP
  `-0.307112/-0.719150`重复运行差异，提示小于约0.3--0.7点的收益可能落在GPU数值轨迹波动
  范围内；本轮更不能把layer9 BCE的`+0.4026`视作稳健改进。
- 153文件白名单结果包已同步本地，无checkpoint；archive和本地SHA均为
  `8f6d4dd4f6fa0cb9aa0f917dd6f7f971c0acde77b5742e01608142ab356e7c05`，包内153/153逐项
  校验通过。详细分析位于结果目录`analysis.md`。

## 2026-08-22：准备 Image-token Adapter 结构调参第一阶段

- 从`exp/emotic-image-token-asl-stable-refine@fb8037f`创建
  `exp/emotic-image-token-structure-tuning`。本阶段系统搜索Image-token Adapter本身的层位置，
  不使用此前Task-lane layer5/8/11结论，也不同时改变层、容量、LR、scale或activation。
- 第一阶段预注册25组：1个FP32 Adapter-disabled joint-BCE锚点，以及zero-based layer0--11
  各一组Image-token joint-BCE和一组主模型BCE + Adapter ASL。结构固定b32、Adapter LR4e-4、
  scale0.1、ReLU、independent；ASL固定9.8/0/0.05。全部为seed0完整8-task validation-only、
  30 epochs/task、batch64、legacy监督、CLIP normalization、crop0.05，不保存checkpoint。
- validation worker支持显式`disabled/image_token`和独立日志目录。新增8-GPU启动器，把25组轮转
  分配到GPU0--7；全局smoke使用最先空闲卡，随后每张卡在每个任务前独立等待连续两次
  `free>=8000MiB`且`util<=10%`，默认间隔60秒，不抢占当前GPU任务。
- 新增层搜索严格汇总器和3项合成结果测试。汇总器拒绝配置/精度/Git漂移、不完整epoch、更新
  数不符、skipped step与checkpoint；位置选择必须同时保护task6 Sadness和Suffering，并以
  同层BCE为Adapter-ASL的直接对照；ASL还必须相对disabled通过同一组门槛。后续
  bottleneck/LR阶段依赖本阶段赢家，本轮不提前排队。
- 本地Shell语法、Python 3.9编译、`git diff --check`和3项隔离汇总器测试通过；完整Track-A
  unittest需服务器`ddp`环境，因本地Python没有torch而不能按正常包入口运行完整套件。
- 实现提交`d63da6b`已推送；Automatic Upload漂移备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-layer-search-upload-20260822`后，服务器
  ASL独立worktree安全切到该clean提交，完整Track-A测试33/33通过，test-only未触碰。
- 8卡队列进入tmux`mla_image_token_layer_s0_20260822_235242`等待，batch为
  `image_token_layer_search_seed0_20260822_235242`。首轮门控快照8卡仅余1906--4604MiB，
  均未达到8000MiB；输出只有manifest，无config、GPU smoke或训练runner。实验worktree固定
  `d63da6b`直到全部运行结束，避免各轮结果Git元数据不一致。

## 2026-08-22：同步并分析 ASL FP32 稳定版局部搜索

- batch`image_token_asl_stable_refine_seed0_20260821_231444`的13组全部完成：12个ASL
  局部组合与1个joint-BCE均为8 tasks、240 epochs、13950 updates、skipped0、exit code0，
  同一clean`3564e12`且`amp=false`。无OOM、非有限数值或异常，FP32成功消除上一轮AMP失败。
- 严格`stable_refine_fp32`汇总没有合格赢家。12个ASL的final cF1全部低于BCE，forgetting
  全部更差；虽然部分提高final mAP、oF1和Suffering，但没有配置同时保护Sadness与宏观指标。
- FP32 BCE五项为`42.164859/48.795463/37.501026/58.083137/0.832476`。final mAP最高的
  `gn9.8/clip0.05`五项变化为`+0.425109/+0.660861/-0.635873/+0.443694/+0.058057`，
  task6 Sadness/Sensitivity/Suffering变化`-2.088062/-1.408021/+6.074236`。
- `gn16/clip0.075`只失败Sadness硬门槛，average mAP/oF1/Suffering分别提高
  `0.850003/0.841133/6.925002`，但Sadness下降`2.326867`、forgetting恶化`0.213311`，
  不按单项收益放宽预注册规则。
- 同参数9.8/0.05由AMP切到FP32后，cF1与Sadness增益从`+0.6864/+1.4049`变为
  `-0.6359/-2.0881`；Suffering仍提高但幅度从`8.6751`降至`6.0742`。说明小幅收益对精度
  与优化轨迹敏感，单seed不能视为稳健结论。
- 81个JSON、日志、清单文件已打包同步，无checkpoint；服务器/本地archive SHA-256均为
  `5e917f92033491bf6d281a6e9f281167434f40194dd6188cd624ac6a1f302e22`，归档内逐文件校验
  全部通过。详细分析写入本地结果目录`analysis.md`，未启动gamma-pos或新test。

## 2026-08-21：同步并分析 Image-token Adapter-only ASL 第一阶段结果

- 第一阶段batch `image_token_asl_loss_seed0_20260820_191924`已结束。21组预注册配置中18组
  完整完成，均为8 tasks、240 epochs、13950 updates、skipped0；3组在task3因非有限ASL
  loss失败：`gn4/clip0`、`gn6/clip0.025`、`gn9.8/clip0.025`。失败不是OOM，按预注册
  完整性规则判为无效且不重跑。
- 缺失的`gn9.8/clip0`已在原实验提交`b99480e`补齐并正常完成。failure-aware汇总器生成
  `loss_search_summary.json`，服务器完整单元测试29/29通过，当前无本轮训练进程且GPU0--7
  均空闲。
- 相对joint-BCE validation，严格赢家`gn9.8/clip0.05`的final/average mAP为
  `42.667273/49.237821`，提高`0.709193/0.922159`；final cF1/oF1提高
  `0.686412/0.430169`，8个task mAP均提高。forgetting提高到`1.017189`，即恶化
  `0.076460`。
- 赢家task6 mAP提高`0.695724`；Sadness/Sensitivity/Suffering分别变化
  `+1.404916/-0.530427/+8.675063`，三类均值提高`3.183184`。它通过所有硬门槛，但只证明
  Suffering和整体排序获益，不能证明Sensitivity或旧类保持已解决。
- 只有`gn9.8/clip0.05`和`gn9.8/clip0`通过全部门槛；后者final mAP提高`0.526062`，但
  final cF1下降`0.176099`、forgetting恶化`0.108917`，综合列第二。其余若只看final mAP
  会忽略Sadness/Suffering或F1门槛退化。
- 107个结果/日志/清单文件已在服务器按白名单打包，无checkpoint，压缩包512KB；服务器与
  本地archive SHA-256均为
  `90e0ea0817ec0ea266e72f3961a58e4f2abe89603c31f4755ef91c8ad0ab7684`，逐文件SHA校验通过。
  本地已解压至`output/emotic_track_a_image_token_asl_hparam/`并新增`analysis.md`。
- 本轮未启动下一阶段。赢家gamma-neg仍处搜索上边界，建议先局部扩展到
  `gamma_neg={8,9.8,12,16}`、`clip={0.0375,0.05,0.075}`，复用已有赢家后需11个新运行；
  只有确认内部最优后才搜索gamma-pos，并继续只用validation。
- 用户确认使用8张4090并行继续。进一步审计发现三个失败配置都在task3约2000次连续更新后
  开始出现AMP GradScaler跳步，随后跳步增多并产生非有限logits/loss；这与默认GradScaler
  增大loss scale的时间点吻合，不是数据缺失或OOM。
- 从当前分支创建`exp/emotic-image-token-asl-stable-refine`。为避免把不同数值协议混排，
  稳定版统一关闭AMP，以FP32重跑12个局部ASL组合及1个joint-BCE；因此不复用旧9.8/0.05。
  新增8-GPU两轮队列、FP32 GPU smoke入口、稳定版严格汇总profile和有限logit/target诊断。
  实现提交`3564e12`已推送并由服务器ASL独立worktree安全切换；服务器30/30完整单测通过。
- GPU0真实CLIP FP32 Adapter-ASL smoke通过：trainable parameters为739130，zero-up与disabled
  logits最大差`1.70e-08`，无梯度、冻结或数值异常。8张GPU启动前均有23.6GB以上空闲。
- 13组队列在tmux`mla_asl_stable_refine_s0_20260821_231444`启动，batch ID为
  `image_token_asl_stable_refine_seed0_20260821_231444`。第一轮8组全部进入task0 epoch2；
  epoch1均84 steps/skipped0、约16--17秒，单卡占用约2.8--3.0GB，无错误信号。第二轮5组由
  每卡lane在第一组完成后自动接续；不保存checkpoint，不读取held-out test。

## 2026-08-20：Image-token Adapter-only ASL 参数搜索框架

- 从`exp/emotic-image-token-asl-routing@129c73d`创建
  `exp/emotic-image-token-asl-hparam-search`，并保留此前尚未提交的四组结果分析与三份上下文
  文档改动。
- 第一阶段固定seed0、完整8-task validation、30 epochs/task、batch64、legacy监督、CLIP
  normalization、crop0.05、Image-token layer8/b32/LR4e-4/scale0.1/ReLU/independent；主模型
  继续BCE，只联合搜索Adapter ASL的5个gamma-neg与4个clip，并加入joint-BCE对照，共21组。
- 新增通用validation worker与8-GPU队列启动器；每张GPU同一时刻只运行一组，21组按轮转
  队列分配，全部完成后自动生成严格汇总。新增`--no-save-checkpoints`，只影响显式开启该参数
  的调参运行，历史/正式入口继续默认保存checkpoint。
- 队列新增可复现资源门控，默认要求每卡连续两次达到`free>=5000MiB`且`utilization<=10%`，
  间隔60秒；队列启动前在指定卡运行真实CLIP Image-token Adapter-ASL smoke。每条GPU lane
  在开始下一组前都会重新过门槛，防止其他任务临时占卡时继续叠加。
- 新增专用汇总器和3项单测：强制完整21组网格、相同clean commit/tree、240 epochs、13950
  updates、skipped0、无checkpoint，并按预注册task6/final/Sadness/Suffering/F1硬门槛排名。
  本地Python编译、Shell语法、3项独立汇总器单测与`git diff --check`通过；完整torch单测待
  服务器`ddp`环境执行。
- 服务器预检发现GPU0--7均被现有任务占用约20.4--22.2GB，当前不具备安全启动余量，因此未
  启动调参、未终止现有进程。Automatic Upload漂移已备份至
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-hparam-upload-20260820`并恢复主worktree
  clean；ASL独立worktree保持clean，test-only worktree未触碰。
- 首个实现提交`397d74d`已推送并由服务器ASL独立worktree通过Git切换到同一clean HEAD；
  服务器完整Track-A单元测试28/28通过。GPU smoke没有挤入繁忙卡，改由资源门控在队列真正
  启动前自动执行。
- 资源门控补充提交`a0e0aee`已推送并由服务器fast-forward。第一阶段已在tmux
  `mla_image_token_asl_hparam_s0_20260820_191924`排队，batch ID为
  `image_token_asl_loss_seed0_20260820_191924`；首个GPU5快照为free3520MiB/util99%，门控
  正确等待。当前只是CPU侧每60秒轮询，没有GPU smoke或训练进程，也未增加显存占用。

## 2026-08-13：Image-token Adapter 的参数组级 ASL 三组实验

- 审计本地`CODE_DDP`实现与已有结果，沿用其Adapter ASL默认：gamma-neg9.8、gamma-pos0、
  probability clip0.05、eps1e-8、detach focal weight。MULTI-LANE保留legacy 26维监督视图和
  mean reduction，避免同时改变loss形状与历史梯度尺度。
- 从Image-token分支创建`exp/emotic-image-token-asl-routing`，新增参数组loss路由：
  `model_asl`、`adapter_asl`、`both_asl`。混合模式通过`autograd.grad`把model目标只送入
  selectors/prompts/head，把Adapter目标只送入当前task Adapter，消除联合图上的梯度串流。
- 新增ASL数值/梯度路由测试、GPU smoke路由参数、三组seed0正式worker和3-GPU启动器。三组
  固定full image、CLIP normalization、crop0.05、legacy监督、Image-token layer8/b32、
  30 epochs/task、batch64、held-out test。
- 实现提交`6d5430a`已推送。服务器为ASL分支创建独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-asl-routing`，不切换GPU2仍在运行原Image-token BCE的
  主worktree；Automatic Upload文件已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-asl-upload-20260813`并恢复主worktree clean。
- 服务器25/25完整单测通过；GPU3/4/7依次完成`model_asl/adapter_asl/both_asl`真实CLIP
  smoke，三组可训练参数均为739130，初始等价最大差均在`4.2e-05`内，参数组梯度、冻结visual
  和concat inference正常。三组正式训练尚未启动。
- 验证记录提交`8840e96`已推送并由服务器ASL worktree fast-forward。三组seed0 held-out test
  正式实验在tmux `mla_image_token_asl_seed0_011602`启动，GPU3/4/7分别对应
  `model_asl/adapter_asl/both_asl`，run ID共同后缀为`20260813_011602`。
- 三组task0 epoch1均为84 optimizer steps、skipped0、约16.5秒；GPU占用2.0--2.4GB且均有
  21GB以上余量。model-ASL的model/Adapter loss为`0.01572653/0.76665431`，adapter-ASL为
  `0.61468159/0.02817775`，both-ASL为`0.01577781/0.01577781`；路由行为可观测且无异常。
- 原Image-token BCE与三组ASL均完成240 epochs/13950 updates、skipped0、exit code0，运行
  config为clean Git。按白名单打包四组JSON与正式/launcher日志，共24个小文件，不含checkpoint；
  各源文件已通过服务器/本地SHA-256一致性校验，本地统一104KB压缩包SHA-256为
  `60cb61ebe7f64d639f351821834c95539bad149ad0abe6e1d3f4796a2d8df884`。
- 四组final/average mAP依次为：BCE `31.676786/38.564709`、model-ASL
  `30.462064/38.238173`、Adapter-ASL `32.419329/38.878898`、both-ASL
  `32.199410/39.148237`。Adapter-ASL相对BCE的task6/final mAP提高
  `0.735797/0.742543`，task6 Sadness/Sensitivity/Suffering分别提高
  `7.524682/0.652493/8.239046`，三类均值提高`5.472074`。
- model-ASL/both-ASL的final oRecall接近100%、oPrecision约17%，oF1相对BCE下降
  `20.004362/19.356518`。确认是gamma-neg9.8强烈削弱负监督引起的固定阈值校准坍塌，不是
  训练未完成、OOM或梯度路由错误；Adapter-only ASL保留主模型BCE，因此F1基本稳定。
- 完整对照报告写入本地结果目录`comparison_analysis.md`。本轮结果只有seed0且直接使用
  held-out test，记录为探索性证据，不据此继续调gamma、阈值或启动新实验。

## 2026-08-13：实现 Image-token Adapter 并准备 seed0 正式实验

- 将上一阶段task-lane Adapter位置诊断与路线收口文档以提交`ea5f204`推送到
  `origin/exp/emotic-adapter-position-diagnostics`，随后新建
  `exp/emotic-image-token-adapter`，避免两种架构混入同一提交。
- 新增`image_token` Adapter模式：每个task独立路由Adapter，输入为目标CLIP block经`ln_1`
  归一化后的冻结`CLS + patch tokens`；输出仅修改selector读取的图像token视图，不修改冻结
  CLIP主干的后续残差流。zero-up初始化保证初始输出与disabled路径一致，初始化RNG继续隔离。
- runner、GPU smoke、配置元数据和单元测试已扩展到新模式；新增seed0专用正式启动器
  `scripts/emotic/run_multilane_track_a_image_token_adapter_seed0.sh`。
- 用户明确只跑seed0并允许跳过validation。正式配置：Split-EMOTIC B5-C3、full image、held-out
  test、8 tasks、30 epochs/task、batch64、seed0、legacy loss、CLIP normalization、crop0.05、
  image-token Adapter layer8/bottleneck32/LR4e-4/scale0.1/ReLU/independent。
- 本地Python编译与Shell语法通过；本地系统Python缺少torch/numpy，完整单测和GPU smoke改在
  服务器`ddp`环境执行。
- 实现提交`e10324f`已推送到远端并由服务器通过Git安全切换；额外test-only worktree未修改。
  服务器完整22项单元测试全部通过。
- GPU2真实ViT-B/16 Image-token Adapter smoke通过：trainable parameters为`739130`，初始
  等价最大差`3.4332275e-05`，Adapter梯度有限、visual encoder无梯度、无OOM。正式训练尚未
  启动，下一步按确认配置只启动seed0。
- 记录验证结果的`a322419`已推送并由服务器fast-forward到clean同HEAD。seed0正式实验已在
  GPU2/tmux `mla_image_token_seed0_004500`启动，run ID为
  `image_token_adapter_b32_layer8_seed0_20260813_004500`。task0 epoch1正常完成：loss
  `0.61481267`、84 optimizer steps、skipped0、17.7秒；GPU显存约2.0GB，无错误信号。

## 2026-08-12：实现 loss、预处理与 Adapter 层位置分阶段诊断

- 按预定停止规则正式结束当前task-lane Adapter路线：不再扩展layer5/8/11、不做多层、不扩大
  bottleneck、不增加Adapter深度、不使用正式test调参。当前保留的validation基线为Adapter
  disabled、legacy loss、CLIP normalization、crop `(0.05,1.0)`；其seed0五项为
  `42.175807/48.421923/37.198819/58.301308/0.994957`。
- 已确认服务器不存在loss/preprocessing/layer诊断runner，GPU2/3/4已释放；已关闭三个完成后
  停留在shell的tmux `mla_loss_diag_s0_173322`、`mla_prep_diag_s0_182014`、
  `mla_layer_diag_s0_192723`。所有服务器日志、JSON与本地同步副本均保留，未删除实验数据，
  未启动新实验。
- 第三阶段单层位置诊断已在tmux `mla_layer_diag_s0_192723`启动，batch ID为
  `adapter_layer_position_seed0_20260812_192723`。固定legacy loss、CLIP normalization、
  crop `(0.05,1.0)`、seed0、完整8-task val-only、30 epochs/task、batch64、LR0.0125；
  independent Adapter为b64、LR4e-4、ReLU、scale0.1，GPU2/3/4分别运行zero-based layer
  `5/8/11`。基线为已完成的clip+crop0.05 disabled运行。
- 三组config均记录clean `001d8a0`，仅layer index不同；首两轮均84 steps、skipped0，每卡约
  1.94GB且仍空闲约22.1GB，无OOM、CUDA error、Traceback或NaN/Inf。不读取test、不保存
  checkpoint。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_adapter_layer_position_v0.3/adapter_layer_position_seed0_20260812_192723/`。
- 最终判定在原三项task6 mAP、final mAP、task6新类均值改善之外，增加Suffering AP不得低于
  disabled基线的硬性条件；若无单层同时满足，则结束task-lane Adapter路线，不做多层扩容。
- layer5/8/11均以240 epochs、13950 updates、launcher exit code0完成，日志无OOM、NaN或异常。
  相对clip+crop0.05 disabled基线，五项变化分别为：layer5
  `final mAP -0.093808 / average mAP +0.391947 / final cF1 +0.701436 / final oF1 +0.061258 /
  forgetting improvement +0.006156`；layer8为
  `+0.380562/+1.013492/+0.975789/+0.132606/+0.178909`；layer11为
  `-0.439175/+1.389888/+0.425703/-0.124769/+0.195441`。
- 四项硬条件没有任何层通过。layer5的task6新类均值/Suffering为`-0.525884/-0.528539`；
  layer8为`-1.296133/-2.105999`；layer11为`-8.770622/-14.799475`。layer8虽然aggregate
  最优，但task6总mAP`+0.540776`来自旧类均值`+0.816313`掩盖新类下降；layer11同样表现为
  旧类`+0.924071`掩盖新类`-8.770622`，证明层位置不能修复后期可塑性问题。
- 三组12个运行JSON、3个paired report及3份日志已同步到本地
  `./output/emotic_track_a_adapter_layer_diagnostics/adapter_layer_position_seed0_20260812_192723/`；
  服务器与本地18个文件SHA-256逐项一致，没有checkpoint。按预定规则停止task-lane Adapter
  多层与容量扩张，当前未启动新实验。
- 第二阶段2×2预处理诊断已在tmux `mla_prep_diag_s0_182014`启动，batch ID为
  `preprocessing_seed0_20260812_182014`。固定`legacy_full_zero`、seed0、EMOTIC train→val、
  完整8 tasks、30 epochs/task、batch64、LR0.0125、Adapter disabled、full image、AMP/TF32；
  GPU2/3/4/7分别对应`none+crop0.05`、`clip+crop0.05`、`none+crop0.50`、
  `clip+crop0.50`。不读取test、不保存checkpoint、不启动Adapter层实验。
- 四组启动config均记录clean `001d8a0`。首个epoch均84 steps、skipped0；每张卡新增约1.94GB，
  仍空闲约22.1GB，未发现OOM、CUDA error、Traceback或NaN/Inf。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_preprocessing_diagnostics_v0.3/preprocessing_seed0_20260812_182014/`，
  日志位于`./logs/emotic_track_a_adapter_diagnostics/`。
- 四组均以240 epochs、13950 updates、launcher exit code0完成。五项
  `final_mAP/average_mAP/final_cF1/final_oF1/forgetting`分别为：none+0.05
  `41.964051/48.323627/37.289303/58.586581/0.890804`；clip+0.05
  `42.175807/48.421923/37.198819/58.301308/0.994957`；none+0.50
  `40.645741/46.624393/37.102879/57.290124/0.971548`；clip+0.50
  `41.172109/47.777290/37.403789/57.218872/0.800238`。
- crop0.50在两种normalization下均使8个task mAP全部下降；末轮训练loss反而更低，说明训练
  视图更简单但泛化更差，因此淘汰。clip+crop0.05相对旧默认使task6/final mAP提升
  `0.134184/0.211756`、task6新类均值提升`0.484693`，按预先规则选为层位置基线；但其
  Sadness/Sensitivity/Suffering变化为`+7.019647/+0.362923/-5.928491`，后续必须单列
  Suffering，不能只凭aggregate mAP选层。
- 结果与四份日志已同步到本地
  `./output/emotic_track_a_preprocessing_diagnostics/preprocessing_seed0_20260812_182014/`；服务器与
  本地21个文件SHA-256逐项一致，没有checkpoint。当前未启动Adapter层位置实验。
- 第一阶段loss二选一已在tmux `mla_loss_diag_s0_173322`启动，batch ID为
  `training_loss_seed0_20260812_173322`。GPU2为`legacy_full_zero`，GPU3为`current_only`；
  两组均为clean `001d8a0`、seed0、EMOTIC train→val、完整8 tasks、30 epochs/task、batch64、
  LR0.0125、Adapter disabled、full image、none normalization、crop `(0.05,1.0)`、AMP/TF32。
- 启动前GPU2/3已有进程约占7.8/8.3GB，空闲15.8/16.1GB；本轮进程各占约1.9GB，进入稳定
  训练后仍空闲约14.1GB。两组task0前5个epoch均84 steps、skipped0，无OOM、CUDA error、
  Traceback或NaN/Inf。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_loss_diagnostics_v0.3/training_loss_seed0_20260812_173322/`，
  日志位于`./logs/emotic_track_a_adapter_diagnostics/`。
- 两组均完成240 epochs与13950 updates，launcher exit code 0，并自动生成
  `loss_diagnostics_summary.json`。legacy五项为
  `41.964051/48.323627/37.289303/58.586581/0.890804`；current-only为
  `41.318942/48.227313/36.413423/58.508179/0.795174`。current-only虽在task0–3 mAP分别
  微升`0.237138/0.392014/0.198054/0.079564`，但task4开始转负，task6/7下降
  `0.757291/0.645109`。
- task6新三类引入AP：legacy为Sadness `41.064792`、Sensitivity `5.960459`、Suffering
  `51.880423`；current-only为`40.507695/6.710906/38.328382`。均值从`32.968558`降至
  `28.515661`，主要失败点是Suffering `-13.552041`，因此下一阶段固定legacy loss。
- 新legacy运行与上一轮clean disabled运行的8-task mAP及五项summary差值全部为0；结果已连同
  两份日志同步到本地`./output/emotic_track_a_loss_diagnostics/training_loss_seed0_20260812_173322/`，
  11个文件经服务器/本地SHA-256逐项校验完全一致，没有checkpoint。
- 从clean `5b2f791` 创建本地分支 `exp/emotic-adapter-position-diagnostics`；服务器仍停留在
  `exp/emotic-multilane-transformer-adapter`，本轮代码尚未提交推送或运行实验。
- runner新增 `--training-loss-mode legacy_full_zero|current_only`。默认legacy保持历史复现；
  current-only直接在当前task logits/targets上计算BCE。测试将验证有效logit梯度相对legacy
  放大 `26/current_classes`，同时config记录Adam会弱化常数梯度缩放，避免预设结果。
- runner新增 `--input-normalization none|clip` 和 `--train-crop-scale MIN MAX`；默认仍为none和
  `(0.05,1.0)`。CLIP选项使用OpenAI mean/std，normalization与crop可独立组合并写入config。
- 新增统一seed0/8-task/val-only诊断worker，支持disabled或task-lane；所有入口要求clean Git，
  不构造test。新增loss两组launcher、预处理2×2 launcher、以及independent Adapter零基索引
  layer `5/8/11` launcher；后两阶段强制显式传入上一阶段选中的配置，防止混杂。
- paired validation分析器扩展为严格核对loss、input mode、normalization与crop scale，支持
  显式 `independent/copy_previous` candidate，并把输出候选字段改为对应初始化模式。
- 新增通用多运行validation diagnostics汇总器及测试，用于disabled loss/预处理阶段；输出每组
  8-task mAP/cF1/oF1、五项summary和task6三类引入/final AP，不擅自按单一指标自动选优。
- GPU smoke新增 `--adapter-layer-indices`，使同一个真实CLIP forward/backward入口可验证零基
  layer `5/8/11` 的Adapter零初始化、梯度、冻结参数、参数量与路由，不必启动训练。
- 计划顺序固定：disabled legacy/current-only完整validation选loss；选中loss下跑none/clip ×
  crop0.05/0.50；选中预处理下再跑layer5/8/11 independent Adapter。只有中层通过task6、
  task7 final与task6新类平均AP三项条件，才考虑小范围多层；当前未启动任何组。
- 诊断实现已以 `5766860` 提交并推送；服务器主工作树安全切到同一分支和提交。Automatic
  Upload遗留的4个同内容文件已先备份到
  `/mnt/haoyuan/workspace/git-sync-backup-adapter-position-diagnostics-final-20260812`，再恢复旧分支
  clean后通过Git切换；test-only worktree原有改动保持不动。
- 服务器完整Track-A测试20/20通过。GPU2/3/4并行真实CLIP batch2 smoke分别验证zero-based
  layer `5/8/11`，三组均为 `788314` 个可训练参数，初始logits最大差均为
  `6.103515625e-05`，forward/backward、Adapter梯度、视觉塔冻结和concat推理均通过。本轮没有
  读取EMOTIC、没有写checkpoint，也没有启动loss、预处理或层位置实验。

## 2026-08-11：启动 seed0 完整8-task warm-start validation

- 在服务器主工作树clean `58886f1` 上启动唯一候选：seed0、EMOTIC train→val、完整8 tasks、
  30 epochs/task、batch64、base LR0.0125、Adapter LR4e-4、b64、layer11、ReLU、scale0.1、
  `copy_previous`、隔离初始化RNG、AMP/TF32；不构造或读取test split。
- run ID为
  `task_lane_adapter_full_val_seed0_b64_lr0.0004_copy_previous_20260811_174404`，tmux为
  `mla_warm_val_s0_174404`，使用GPU1。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_warmstart_val_v0.2/`，
  日志位于 `./logs/emotic_track_a_adapter_warmstart_val/`。
- 启动核验已进入task0并完成至少3个epoch；每epoch 84 updates、skipped0，暂未发现OOM、NaN、
  inf或traceback。等待8 tasks完成后只分析validation曲线，重点比较task6的Sadness、
  Sensitivity、Suffering；不根据正式test调参。
- 为形成同代码、同seed、同随机轨迹的配对对照，在GPU2启动 Adapter disabled 的完整8-task
  val-only运行 `multi_lane_disabled_full_val_seed0_paired_clean_20260811_174847`，tmux为
  `mla_disabled_val_s0_174847`。首次对照因Automatic Upload使config记录dirty而在task0早期
  主动终止并保留原目录；clean重启版本config记录 `dirty=false`，已完成至少3个epoch且无错误。
- warm-start与clean disabled均完成240 epochs/8 tasks并输出completion marker。逐task val mAP
  差值为 `+2.041904/+2.367560/+1.808337/+1.307971/+0.890832/+0.539336/-0.334822/-0.448594`，
  明确在task6/7转负；final cF1虽提高 `+0.482067`，但final oF1下降 `-0.770218`。
- warm-start相对disabled的average mAP提高 `+1.021565`、forgetting改善 `0.079821`，但final
  mAP下降 `0.448594`。task6新三类引入时平均AP下降 `6.165031`：Sadness `-8.788894`、
  Sensitivity `+0.997110`、Suffering `-10.703310`；final三类平均差进一步为 `-6.248597`。
- 新增 `multi_lane/track_a/compare_validation.py`，严格检查同commit/seed/protocol/val-only与
  clean状态，自动生成8-task曲线、task6逐类AP及停止判定；结果写入本地
  `./output/emotic_track_a_adapter_warmstart_val/paired_validation_comparison.json`。按预先规则
  三项继续条件全部失败，结论为停止task-lane Adapter容量扩张，不再增加bottleneck或层数。
- 配对分析脚本、3项新测试和上下文以 `3d7c5aa` 提交推送并在服务器fast-forward同步；原10项
  Track-A回归测试加3项分析测试共13/13通过，未启动新实验。
- 进一步诊断显示warm-start的当前任务训练BCE在8个task均低于disabled，但新类平均AP差值从
  task3开始转负：task0–7依次为 `+2.041904/+2.073189/+0.601783/-0.703644` 和
  `-0.832849/-1.669440/-6.165031/-1.103683`。总mAP前期的正增益主要由已学旧lane保留，
  而非新task获取。
- task6 train view仅627样本、10 updates/epoch、总计300 updates；Sadness/Sensitivity/
  Suffering训练正例为260/316/188，seen-scope val正例仅113/110/102。最终checkpoint中Adapter
  参数范数从task0 `12.3918`累积到task6 `17.8840`；task6与task5余弦相似度 `0.9841`、参数
  距离 `3.1801`，是最强的相邻继承之一，说明低更新预算无法消除task5的非零残差偏置。
- 原MULTI-LANE已复制selectors/prompts，Adapter再次完整warm-start造成重复继承；同时较低BCE
  未转化为AP排序提升，说明优化更贴合训练概率而非稀疏新类排序。现状属于结构偏置与可塑性
  不足，不是随机轨迹、容量太小或训练崩溃，因此继续堆容量缺乏依据。

## 2026-08-11：实现 Adapter RNG 隔离与跨任务 warm-start

- 当前结构明确为“每任务独立 Adapter”，不是所有任务共享一个：8个 Adapter 预分配，训练
  task t 时只解冻第t个，推理按lane id路由。上一版本每个task独立随机初始化且不复制。
- 按用户确认的计划增加 `adapter_task_initialization=independent|copy_previous`；默认
  `independent` 保留上一轮复现。`copy_previous` 在task>0激活时把上一task Adapter完整复制
  到当前task，然后只训练当前副本，旧task仍冻结。
- Adapter Bank 初始化被包在 `torch.random.fork_rng(devices=[])` 中；Adapter仍获得确定性的
  seed相关初始化，但构造结束后全局torch RNG恢复到与disabled模型相同的状态，避免改变
  DataLoader shuffle和随机增强序列。
- runner/config/smoke新增 `--adapter-task-init`；task0脚本默认保持independent。新增
  `scripts/emotic/run_multilane_track_a_adapter_full_val.sh`，用于seed0完整8-task、val-only、
  b64/lr4e-4 warm-start验证，不构造test dataset。
- 单元测试新增disabled/adapter构造后RNG state完全一致，以及warm-start参数逐tensor复制、
  旧task冻结和当前task解冻检查。本地py_compile、bash -n和git diff --check均已通过；
  实现以 `313618c` 提交并推送，服务器主工作树fast-forward到同一提交，test-only未触碰。
- 服务器完整单元测试10/10通过；GPU1真实CLIP `copy_previous` Adapter smoke通过，可训练参数
  `788314`，零初始化Adapter与disabled logits最大差 `9.1552734375e-05`，在既定AMP容差
  `1e-4` 内。smoke仅执行batch2 forward/backward，未加载EMOTIC、未生成checkpoint、未启动
  validation或正式训练。

## 2026-08-11：启动 Task-lane Adapter task0 validation screen

- 用户确认 Automatic Upload 已关闭，并确认可以启动 task0 validation screen。启动前服务器
  主工作树干净，local/origin/server 均为
  `exp/emotic-multilane-transformer-adapter@6e8b15d`；test-only worktree 未触碰。
- 完整配置：EMOTIC full image、seed0、task0/5 classes、train→val、30 epochs、batch64、
  eval batch64、Adam base LR0.0125、weight decay0、temperature1、每任务 cosine、AMP/TF32、
  threshold0.5、layer11、ReLU、scale0.1；不构造 test dataset。
- batch ID：`task_lane_adapter_task0_seed0_20260811_153033`。b32/lr1e-4、b32/lr4e-4、
  b64/lr1e-4、b64/lr4e-4 分别运行在 GPU1/2/3/4，对应 tmux：
  `mla_t0_b32_lr1e4_153033`、`mla_t0_b32_lr4e4_153033`、
  `mla_t0_b64_lr1e4_153033`、`mla_t0_b64_lr4e4_153033`。
- 初始检查四组均约进入 epoch3/30，每 epoch 84 optimizer updates、skipped 0，无 OOM、NaN
  或 traceback。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_screen_v0.1/`，日志位于
  `./logs/emotic_track_a_adapter/`。本地上下文改动待实验完成、补齐结果后再确认提交。
- 四组均以 exit code0 完成 30 epochs、2520 optimizer updates、skipped 0。b32/lr1e-4、
  b32/lr4e-4、b64/lr1e-4、b64/lr4e-4 的 seed0 task0 val mAP 分别为
  `56.792729/58.974777/57.720757/59.697275`；相对严格复现 seed0 `57.339154` 的增益分别为
  `-0.546425/+1.635623/+0.381603/+2.358121`，只晋级 b64/lr4e-4。
- b64/lr4e-4 的 task0 seed1/2 确认于 15:53 启动，run ID 为
  `task_lane_adapter_task0_confirm_b64_lr4e4_20260811_155343_seed1/seed2`，tmux 为
  `mla_t0_confirm_s1_155343` / `mla_t0_confirm_s2_155343`，分别使用 GPU1/2。复用已有 seed0，
  晋级规则为三种子平均增益≥+0.5、至少2/3 seeds提升、任一seed降幅不超过1.0；仍不读取test。
- 老师随后明确不需要额外多 seed 确认。停止信号到达前 seed1/2 已正常完成，val mAP 为
  `59.329355/56.178599`，相对各自严格复现基线 `56.932921/54.973777` 仍分别提升
  `+2.396434/+1.204822`；产物保留，但不再作为必须前置步骤。
- 按用户要求新增 b128/lr4e-4 seed0 task0 val screen：run ID
  `task_lane_adapter_task0_seed0_b128_lr4e4_20260811_155954`，tmux
  `mla_t0_b128_lr4e4_155954`，GPU3。每任务 Adapter 参数 `197504`、当前总可训练参数
  `886682`。最终 val mAP `56.824802`，比 b64/lr4e-4 的 `59.697275` 低 `2.872474`，因此
  淘汰 b128，正式配置确定为 b64/lr4e-4。
- 正式 run ID：`multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927`。seed0/1/2
  分别在 GPU1/2/3、tmux `mla_adapter_formal_s0_160927`、
  `mla_adapter_formal_s1_160927`、`mla_adapter_formal_s2_160927` 启动。完整配置为 8 tasks、
  30 epochs/task、batch64、Adam base LR0.0125、Adapter LR4e-4、b64/layer11/ReLU/scale0.1、
  cosine per task、AMP/TF32、test reporting、threshold0.5；运行提交 clean `6e8b15d`。
- 正式输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_formal_v0.1/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`，日志位于
  `./logs/emotic_track_a_adapter_formal/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`。
  启动检查三组均完成模型构建并进入 task0，GPU 显存约1.9GB，Git仍为clean。
- 正式三组均以 completion marker 和 tmux exit code0 完成；每 seed 240 epochs、8 tasks、
  skipped0，config 均为 test reporting、b64/lr4e-4、Git clean `6e8b15d`。
- 聚合五项 mean±sample std：final mAP `31.128461±0.082307`、final cF1
  `31.994054±0.322333`、final oF1 `48.878881±0.080051`、average mAP
  `38.599579±0.360451`、forgetting `4.869595±0.112483`。相对无 Adapter 基线均值变化为
  `-0.171025/+0.182944/-0.230322/+0.601003/+0.081138`。
- 使用显式白名单打包 formal summary、每 seed 四个 JSON、三份日志和 SHA manifest，共17个
  普通文件，确认无 checkpoint。服务器压缩包大小61813 bytes，SHA-256
  `16e0a9fcf9dc7d66dbf8ac5ba0a681539a400fdcf0f53916bd122071818946e1`；已通过 SSH key 下载到
  本地 `./output/emotic_track_a_adapter/<run_id>/`，校验一致并解压到 `synced_files/`。
- 逐 task 对齐发现 Adapter 的 seen-class test mAP 在 task0–5 分别提升
  `+1.665288/+1.280988/+0.912148/+0.433352/+0.510573/+0.410443`，但 task6/7 变为
  `-0.233741/-0.171025`。task6 新类平均 AP 下降 `4.514579`，Sadness/Suffering 分别下降
  `6.289815/7.842116`；到 final 仍保持类似差值，说明是 task6 获取不足而非后续遗忘。
- task6 train view 仅627样本、10 updates/epoch、总计300 updates；其 Adapter up-weight norm
  `2.494252` 为8个 task最低。全部 task 的最终 BCE 均低于基线，但 task3/5/6/7 val mAP
  更差，支持低数据任务欠适应/目标错配诊断。参数全部 finite，无爆炸或未训练迹象。
- 发现严格控制缺陷：Adapter Bank 的 down projection 随机初始化会在 DataLoader 前消耗
  全局 torch RNG，使相同 seed 的数据 shuffle/augmentation 不再与 disabled 基线相同，
  不同 bottleneck 也不完全配对。下一轮必须先隔离初始化 RNG，并优先测试上一 task Adapter
  warm-start；不再根据已观察的正式 test 反复调参。

## 2026-08-11：开始 Task-lane Transformer Adapter 阶段 0

- 用户确认按分析方案开始修改；从严格复现分支 `exp/emotic-multilane-track-a-repro` 的
  `ce7d9a0` 创建本地分支 `exp/emotic-multilane-transformer-adapter`。服务器尚未切换，
  当前改动尚未 commit/push。
- 新增 `multi_lane/track_a/adapter.py`：实现 task-specific、layer-specific bottleneck
  Adapter bank；down 使用 Xavier 初始化，up 权重/偏置全零，按 lane id 路由，激活新任务
  时只解冻当前 task Adapter，不复制或覆盖旧任务参数。
- `multi_lane/track_a/model.py` 在 lane block 冻结 MLP 旁路接入 Adapter；默认
  `adapter_mode=disabled`，因此原模型不会创建 Adapter 或额外消耗初始化随机数。新增运行时
  开关及基础/Adapter 参数分组接口。
- `multi_lane/track_a/runner.py` 增加独立 Adapter LR、bottleneck/layer/scale/activation、
  `max_tasks` 和 `reporting_split`；默认仍是 disabled、8 tasks、test。val screen 模式不创建
  test dataset；单 task summary 的 forgetting 明确定义为 0。
- 新增 `scripts/emotic/run_multilane_track_a_adapter_task0.sh`，固定 task0、val-only、
  layer11、ReLU、scale0.1，并允许通过环境变量筛选 bottleneck 与 Adapter LR；脚本尚未运行。
- 单元测试扩展为覆盖零初始化基线等价、运行时旁路、lane 路由、任务冻结、无 Adapter 复制、
  参数计数和单任务 forgetting。
- GPU smoke 新增显式 `--adapter-mode task_lane` 路径，检查真实 CLIP 上零初始化 logits
  等价、Adapter 梯度、冻结视觉塔和当前任务参数量；默认 disabled 路径仍执行原基线检查。
- 本机系统 Python 3.9 已通过 `py_compile`，新 shell 入口通过 `bash -n`，`git diff --check`
  通过；本机缺少 PyTorch，张量级单元测试与 GPU smoke 等待提交推送后在服务器执行。
- 阶段 0 主实现以 `531f3f3` 提交并推送；服务器主工作树检查为空后通过 `fetch` 和新建跟踪
  分支同步到同一提交。test-only worktree 的历史修改和未跟踪文件保持原样。
- 服务器 CPU 单元测试 8/8 通过；原基线 GPU1 smoke 通过，可训练参数仍严格为 `689178`。
- 首次 Adapter GPU smoke 的逐 bit logits 检查失败。诊断确认 Adapter delta、up weight 和
  up bias 的非零元素均为 0；重复 forward 的 AMP 最大差为 `6.1035e-5`，FP32 最大差为
  `1.4901e-8`，属于 GPU 重复计算数值差，而非 Adapter 非零输出。GPU smoke 因此改为直接
  检查 Adapter residual 严格为零，并对两次 AMP logits 使用 `atol/rtol=1e-4`；CPU 单测的
  逐 bit 等价检查保持不变。
- AMP smoke 修正以 `fc32af7` 提交推送并同步。最终提交上服务器 8/8 单元测试再次通过；
  GPU1 Adapter smoke 通过，可训练参数 `788314`，其中新增当前任务 Adapter `99136`，
  零初始化 AMP logits 最大差 `3.0518e-5`。全程未读取 EMOTIC、未生成 checkpoint、未启动
  task0 或正式训练。

## 2026-08-11：开始在当前仓库移植 MULTI-LANE Track-A 严格复现

- 用户确认在当前 `multi-lane-main` 中复现注册的 EMOTIC B5-C3 三种子结果，随后从该
  结果继续修改，而不是切换到 DDP 仓库开发。
- 从 `fix/independent-git-worktrees@1b1dcd2` 创建本地分支
  `exp/emotic-multilane-track-a-repro`；服务器分支尚未切换。
- 对照确认旧 `clip_vit_b16_patch` 与目标协议存在权重来源、768/512 维分类头、
  val+test/test、0.8/0.5 阈值、batch 256/64、AMP 和三种子口径等差异。
- 新增 `multi_lane/track_a/`：固定 OpenAI CLIP checkpoint 加载、目标 MULTI-LANE
  模型、严格训练/评估 runner、GPU smoke 和三种子汇总器。
- 新增 `scripts/emotic/` worker、直接 launcher 和 tmux launcher；旧入口保持不变。
- 新增单元测试，覆盖 task-slice 复制、concat inference、冻结视觉塔、固定阈值指标和
  逐类 forgetting。本地系统 Python 缺少 PyTorch/NumPy，已完成 AST/compile 与 shell
  语法检查，完整测试待服务器执行。
- 首次服务器单元测试中 3 个模型/forgetting 测试通过；AP 测试因旧公式的 `1e-8`
  epsilon 返回 `99.99999925` 而非数学上的 100，已将断言调整为 5 位小数容差。
- 容差修正后服务器 4/4 单元测试通过。
- 当前仓库 checkpoint loader 与目标 benchmark loader 的 152 个 CLIP visual state
  tensor 完全相同，最大权重误差和同输入输出误差均为 `0.0`，输出为 512 维。
- GPU0 forward/backward smoke 通过，可训练参数为 `689178`，冻结视觉塔没有梯度。
- 本地、远端和服务器同步后，以 `8cff911` 启动正式三种子训练：run ID
  `multi_lane_main_track_a_seed012_20260811_132448`，tmux
  `emotic_multilane_track_a_seed012`，seed 0/1/2 对应 GPU 0/1/2。
- task 0 三个 seed 的 mAP/cF1/oF1 与注册实验逐项完全相同；停止持续轮询，让 tmux 后台
  独立完成。最后一次快照位于 task 1 epoch 14–17/30，错误、NaN、OOM、skip 计数均为 0。
- 三种子正式实验已完成并输出 completion marker。最终 final mAP/cF1/oF1/average
  mAP/forgetting 为 `31.2994855/31.8111103/49.1092038/37.9985757/4.7884572`，所有
  mean/std 与原注册实验完全一致。
- 在服务器生成 `multi_lane_main_track_a_seed012_20260811_132448_download.tar.gz`，包含
  13 个 JSON 和 4 个日志、共 17 个文件，明确无 `checkpoints/`；大小约 80K，SHA-256
  `d3a2f23080da5ff6bd5d0fde99c6ebb9eae1916db31b1e09e9ebb6ed5fb15dcc`。
- 压缩包已通过 SSH 下载到本地项目 `./output/emotic_track_a/<run_id>/`，本地校验一致并
  解压到 `synced_files/`，可直接进行后续分析。

## 当前 Git 状态

权威 Git 仓库位置：

```text
/mnt/haoyuan/workspace/multi-lane-main
```

当前工作分支：

```text
feature/clip-taskCLS-posneg-text-head
```

当前服务器本地提交：

```text
Add CLIP DDP prompt head
```

状态说明：

- 该提交只存在于服务器本地分支，尚未 push 到远端。
- `train_c100.sh` 仍有一个未提交的末尾换行变化，未纳入 DDP 提交。

干净起点：

```text
52cf6b1 Add CLIP task CLS text head
```

该提交同时对应：

```text
origin/feature/clip-taskCLS-text-head
origin/feature/clip-taskCLS-text-siglip-scale
feature/clip-taskCLS-text-head
feature/clip-taskCLS-text-siglip-scale
```

当前 DDP 半成品历史：

```text
stash@{0}: On clip-taskCLS-posneg-text-head: wip clip_ddp draft before clean restart
stash@{1}: On clip-taskCLS-text-siglip-scale: wip siglip scale calibration experiment
stash@{2}: On clip-task-text-head: wip clip_patch_text experiment
```

## 本轮已完成修改

从干净分支重新实现 DDP v1 / PCD 结构，未直接应用 `stash@{0}`。

2026-06-22 已按用户要求在服务器上创建本地提交 `Add CLIP DDP prompt head`，未同步到远端。

2026-06-22 开始下一阶段修改：

- 新增 DDP eval diagnostics：评估时打印并保存 predicted positives、micro precision/recall/F1、prob/logit 分布和阈值 sweep。
- 新增 eval-only checkpoint 复评估入口：`--eval --eval_checkpoint <path>`，也兼容 `--eval_dir` 或 `output_dir/checkpoints.pth`。
- 新增 CLI 参数：`--ddp_diagnostics`、`--ddp_diagnostic_thresholds`、`--eval_checkpoint`。
- 当前这批下一阶段修改尚未提交。

业务代码改动：

- `multi_lane/clip_vit.py`
  - 新增 `clip_ddp` 独立 forward 分支。
  - 新增 class-specific positive/negative text prompts。
  - 新增 class-specific positive/negative visual prompts。
  - 新增 late-layer DDP prompt attention 注入。
  - 新增 token-wise similarity aggregation，支持 `mean`、`max`、`cls`。
  - 新增 PCD temperature 计算。
  - DDP final logit 使用 `CLIP logit_scale * (s_pos - s_neg) / tau`，避免纯 cosine margin 在固定阈值 F1 下不可达。
  - DDP 模式跳过 MULTI-LANE selectors/task CLS 分类路径。
- `multi_lane/engine.py`
  - DDP 训练 loss 只在当前 task classes 上计算。
  - DDP 评估 loss 与 mAP 使用已见类别 clean logits/targets。
  - DDP loss 不再额外使用项目通用 `args.temperature`，温度只由 DDP 内部训练 `tau=1` / 评估 PCD 控制。
- `multi_lane/utils.py`
  - DDP 模式只训练 `ddp_text_prompts` 和 `ddp_visual_prompts`。
  - DDP optimizer 参数组关闭 weight decay，降低旧类 prompt 漂移风险。
- `multi_lane/configs/*.py`
  - 增加 `head_mode=clip_ddp`。
  - 增加 DDP/PCD CLI 参数。

上下文文档：

- 重建 `AGENTS.md`。
- 重建 `PROJECT_CONTEXT.md`、`WORK_LOG.md`、`TODO_NEXT.md`。

## 已运行验证

本地已运行：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器 `multilane` 环境已运行：

```text
git diff --check
python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器配置解析检查：

```text
REMOTE_PARSER clip_ddp True 3.0 0.7
```

服务器参数冻结检查：

```text
TRAINABLE ['ddp_text_prompts', 'ddp_visual_prompts']
PROMPTS (8, 2, 16, 512) (8, 2, 16, 768)
MASK_T0 [0, 1, 2]
OPT_GROUPS 1 0.0 327680
MASK_T1 1 [3, 4]
```

服务器 dummy forward/backward smoke：

```text
TRAIN_LOGITS (1, 4)
TRAIN_DEBUG ... 'class_ids': (2,), 'tau': 1.0, 'final_logits': (1, 4)
TRAIN_ACTIVE_GRAD [0, 1]
EVAL_LOGITS (1, 4)
EVAL_DEBUG ... 'class_ids': (4,), 'tau': 3.0, 'final_logits': (1, 4)
MASK_T1 [2, 3]
```

## 风险点

- `clip_ddp` 已完成 dummy forward/backward smoke，但新增 logit scale 修正后需要重新做一次服务器静态检查和 VOC smoke。
- 旧类 prompt 冻结逻辑已通过 dummy 梯度检查确认，正式训练前仍建议在首个真实 batch 再打印一次 learnable/gradient 状态。
- DDP binary softmax 当前用 `CLIP logit_scale * (s_pos - s_neg) / tau` 形式交给 `BCEWithLogitsLoss`；该尺度修正用于让 PCD 后的 score 与项目固定阈值 F1 兼容。
- EMOTIC 的 `tau_max/gamma` 没有论文指定值，当前只是采用 COCO 风格默认。

## VOC DDP smoke 分析

2026-06-22 的 `voc_clip_ddp_smoke` 结果：

```text
task0 mAP 96.1067, oF1 0.0, cF1 0.0
task1 mAP 71.9124, oF1 0.0, cF1 0.0
task2 mAP 64.1040, oF1 0.0, cF1 0.0
task3 mAP 51.0589, oF1 0.0, cF1 0.0
task4 mAP 50.4372, amAP 66.7238, oF1 0.0, cF1 0.0
```

明细报告显示每个 task 的 `predicted_positive` 总数均为 0。原因是 F1 使用项目固定规则 `sigmoid(logits) > 0.8`，而原实现 DDP logit 为纯 cosine margin `(s_pos - s_neg) / tau`；随着 PCD `tau` 增大，阈值所需 margin 不可达。已修正为带 CLIP `logit_scale` 的 DDP logit，需要重新运行 VOC smoke 验证。

2026-06-22 重新运行 `voc_clip_ddp_smoke_scale`，使用 `CLIP logit_scale * (s_pos - s_neg) / tau`：

```text
task0 mAP 98.7263, amAP 98.7263, oF1 93.9921, cF1 93.7758, loss 0.0862
task1 mAP 90.7199, amAP 94.7231, oF1 80.1021, cF1 78.4673, loss 0.2587
task2 mAP 80.4492, amAP 89.9651, oF1 69.1805, cF1 69.8794, loss 0.3413
task3 mAP 74.8414, amAP 86.1842, oF1 67.4487, cF1 65.8933, loss 0.3784
task4 mAP 71.5174, amAP 83.2508, oF1 64.0643, cF1 62.1179, loss 0.3756
```

该结果确认 scale 修正有效：最终 task 的 predicted positives 总数为 7175，20 个 seen classes 全部有正预测。当前只是 1 epoch VOC B4-C4 smoke，不能直接对齐论文 VOC B4-C2/B5-C3/B0-C4/B10-C2 协议；但相对旧 smoke，final mAP 从 50.44 提升到 71.52，OF1/CF1 从 0 恢复到 64.06/62.12。

## VOC 论文协议实验

2026-06-22 已为严格对齐 VOC B0-C4 修正 `multi_lane/datasets.py` 的 split 逻辑：

- `base_classes=0,num_tasks=5` 现在切分为 `[0-3],[4-7],[8-11],[12-15],[16-19]`。
- `base_classes=10,num_tasks=6` 保持 B10-C2：`[0-9],[10-11],[12-13],[14-15],[16-17],[18-19]`。

已启动 tmux：

```text
voc_b0c4_ddp_gpu1
  GPU: 1
  config: VOC B0-C4, epochs=20, batch_size=16, tau_max=7.0, gamma=0.2
  log: ./logs/voc_ddp_b0c4_20ep_tau7_g02.log
  output: ./output/voc_ddp_b0c4_20ep_tau7_g02

voc_b10c2_ddp_gpu0
  GPU: 0
  config: VOC B10-C2, epochs=20, tau_max=7.0, gamma=0.2
  initial batch_size=16/chunk=2 and batch_size=16/chunk=1 both OOM on GPU0 because other processes already occupied memory
  current retry: batch_size=8, ddp_class_chunk_size=1
  log: ./logs/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1.log
  output: ./output/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1
```

用户明确要求：如果显存 OOM 退出，就等待，不要动别人服务器上正在跑的实验。

### 2026-06-22 结果分析

本地已同步 `voc_ddp_b0c4_20ep_tau7_g02` 完整结果：

```text
task0 mAP 99.2469, amAP 99.2469, oF1 95.3933, cF1 95.2475, loss 0.0536
task1 mAP 90.0728, amAP 94.6598, oF1 52.7571, cF1 49.3263, loss 0.3966
task2 mAP 82.8824, amAP 90.7340, oF1 33.8562, cF1 31.9064, loss 0.4603
task3 mAP 79.5050, amAP 87.9267, oF1 23.5792, cF1 23.1313, loss 0.4751
task4 mAP 76.9800, amAP 85.7374, oF1 21.5977, cF1 22.8244, loss 0.4872
```

关键症状：

- final mAP 仍有 76.98，说明排序信号没有完全坏掉。
- final F1 崩塌来自 recall 极低：20 类总 support 为 7632，只预测 943 个正样本，TP 926、FP 17、FN 6706。
- final precision 约 0.982，recall 约 0.121，说明 PCD/阈值过强，模型过度保守。
- 对比 1 epoch `voc_clip_ddp_smoke_scale`：final precision/recall 约 0.661/0.621；本次 VOC B0-C4 使用 `tau_max=7,gamma=0.2` 后 precision/recall 变为约 0.982/0.121。

当前判断：

- `tau_max=7,gamma=0.2` 来自论文附录 VOC B4-C2 的超参分析，不应直接假定适合 VOC B0-C4/B10-C2。
- 当前实现中的 PCD 对 VOC B0-C4 明显抑制过强，是 final CF1/OF1 远低于论文值的第一嫌疑。
- B10-C2 本地同步日志尚未出现最终 `Average performances` 或 detail JSON；当前只能确认其训练阶段 mAP/oF1 较高但 cF1 偏低，不能作为完整最终结果分析。

### 2026-06-22 DDP 诊断实验

本地已同步三组 VOC B0-C4 诊断结果：

```text
voc_ddp_b0c4_diag_tau3_g07_3ep
voc_ddp_b0c4_diag_eval_tau3_g07
voc_ddp_b0c4_diag_eval_pcd_off
```

关键结果：

```text
3ep tau3/gamma0.7 train final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval tau3/gamma0.7 final:     mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval PCD off final:           mAP 76.1609, amAP 85.3464, oF1 55.9951, cF1 59.0601, loss 0.4391
```

诊断结论：

- `train_tau3` 与 `eval_tau3` 每个 task 指标一致，说明 `--eval --eval_checkpoint` 路径可复现训练末尾评估。
- `tau_max=3,gamma=0.7` 不会出现 tau7/gamma0.2 的 recall 崩塌；final predicted positives/support 为 6544/7632，micro precision/recall/F1 为 0.695/0.596/0.642。
- `pcd_off` 的 final predicted positives/support 为 15170/7632，micro precision/recall/F1 为 0.421/0.837/0.560；关闭 PCD recall 更高，但误报太多，F1 反而低。
- mAP 在 tau3 与 PCD off 下基本一致，因为 PCD 是正尺度变换，主要影响固定阈值下的 F1，而不是排序。
- 当前 `tau_max=3,gamma=0.7` 是比 `pcd_off` 和 `tau_max=7,gamma=0.2` 更好的默认候选。

### 2026-06-23 PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
tau2/gamma0.7 final: mAP 76.1613, amAP 85.3464, oF1 62.9751, cF1 63.1456, loss 0.3584
tau5/gamma0.5 final: mAP 76.1613, amAP 85.3464, oF1 46.4039, cF1 45.9509, loss 0.4374
tau7/gamma0.2 final: mAP 76.1613, amAP 85.3464, oF1 20.3382, cF1 23.7697, loss 0.4850
```

与已有结果合并比较：

```text
pcd_off:       OF1 55.9951, CF1 59.0601, pred/support 15170/7632, P/R/F1 0.421/0.837/0.560
tau2/gamma0.7: OF1 62.9751, CF1 63.1456, pred/support 10102/7632, P/R/F1 0.553/0.732/0.630
tau3/gamma0.7: OF1 64.2071, CF1 62.1797, pred/support 6544/7632,  P/R/F1 0.695/0.596/0.642
tau5/gamma0.5: OF1 46.4039, CF1 45.9509, pred/support 2643/7632,  P/R/F1 0.902/0.312/0.464
tau7/gamma0.2: OF1 20.3382, CF1 23.7697, pred/support 884/7632,   P/R/F1 0.980/0.113/0.203
```

结论：

- tau7/gamma0.2 明确过强，复现 recall collapse；该配置不适合作为 VOC B0-C4 默认。
- tau5/gamma0.5 也偏保守，precision 高但 recall 明显不足。
- tau2/gamma0.7 与 tau3/gamma0.7 接近：tau2 提高 recall 和 CF1，tau3 提高 precision、micro F1 和 OF1。
- 若按论文同时看 OF1/CF1，tau3/gamma0.7 当前更均衡；tau2/gamma0.7 可作为 macro-F1 候选保留。
- 下一步不应继续调更强 PCD，应进入 aggregation 对照，优先用 tau3/gamma0.7，同时可保留 tau2/gamma0.7 做备选复核。

### 2026-06-23 aggregation 对照

本地已同步 VOC B0-C4 3 epoch aggregation 对照，固定 `tau_max=3.0,gamma=0.7`：

```text
mean final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
max final:  mAP 71.6560, amAP 81.9845, oF1 59.6344, cF1 59.1139, loss 0.3544
cls final:  mAP 77.7732, amAP 86.6709, oF1 64.0544, cF1 63.8777, loss 0.3535
```

诊断对比：

```text
mean: pred/support 6544/7632, P/R/F1 0.695/0.596/0.642
max:  pred/support 6920/7632, P/R/F1 0.627/0.569/0.596
cls:  pred/support 9569/7632, P/R/F1 0.576/0.722/0.641
```

类别层面：

- `cls` 明显改善了部分 mean 下偏保守的局部/物体类：diningtable F1 0.270 -> 0.529，pottedplant 0.329 -> 0.552，motorbike 0.645 -> 0.853，bus 0.583 -> 0.775，chair 0.372 -> 0.441。
- `cls` 也带来一些误报：sheep 0.521 -> 0.291，bicycle 0.615 -> 0.445，bird 0.671 -> 0.466，train 0.690 -> 0.590。
- `max` 没有达到“局部 token 最大值更好”的预期，final mAP、OF1、CF1 均低于 mean/cls，暂停作为主路线。

当前判断：

- `cls` 是当前 aggregation 的最好候选，mAP 和 CF1 均优于 mean，OF1 与 mean 基本持平。
- `cls` 的分数分布更偏召回，后续建议在 `cls` checkpoint 上复评估 tau2/tau3/tau5，确认是否能通过 PCD 强度把 precision/recall 再平衡。
- 如果 `cls + tau3` 或 `cls + tau5` 复评估稳定优于 mean，再用该配置跑 20 epoch B0-C4。

### 2026-06-23 cls PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_cls_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
cls tau2/gamma0.7 final: mAP 77.7733, amAP 86.6713, oF1 60.3633, cF1 62.0513, loss 0.3796
cls tau5/gamma0.5 final: mAP 77.7733, amAP 86.6708, oF1 55.3263, cF1 55.4358, loss 0.3792
cls tau7/gamma0.2 final: mAP 77.7732, amAP 86.6708, oF1 36.0121, cF1 39.6079, loss 0.4183
```

与已有 `cls tau3/gamma0.7` 合并比较：

```text
cls tau2/gamma0.7: pred/support 13013/7632, P/R/F1 0.479/0.816/0.604
cls tau3/gamma0.7: pred/support  9569/7632, P/R/F1 0.576/0.722/0.641
cls tau5/gamma0.5: pred/support  4825/7632, P/R/F1 0.714/0.452/0.553
cls tau7/gamma0.2: pred/support  2298/7632, P/R/F1 0.778/0.234/0.360
```

结论：

- 固定项目/论文主阈值 `sigmoid(logits)>0.8` 下，`cls + tau3/gamma0.7` 最均衡，OF1/CF1 最高。
- `cls + tau2/gamma0.7` recall 过强，误报变多，OF1/CF1 均低于 tau3。
- `cls + tau5/gamma0.5` 与 `cls + tau7/gamma0.2` 均偏保守，尤其 task3/task4 的 recall 明显下降。
- `cls + tau5` 在诊断阈值 0.7 下 micro-F1 接近 tau3，但主实验不能改阈值，因此不作为当前主配置。
- 已按 `head_mode=clip_ddp, ddp_similarity_aggregation=cls, ddp_tau_max=3.0, ddp_gamma=0.7` 跑完 VOC B0-C4 20 epoch，并保留 checkpoint。

### 2026-06-23 VOC B0-C4 cls 20 epoch

本地已同步并分析：

```text
log:    ./logs/voc_ddp_b0c4_cls_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_cls_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_cls_tau3_g07_20ep/checkpoints.pth
```

实验配置：

```text
dataset Split-VOC, num_tasks=5, base_classes=0
backbone clip_vit_b16_patch
head_mode clip_ddp
epochs=20, batch_size=16, optimizer=adam, lr=0.05 scaled to 0.003125
ddp_prompt_length=16, ddp_prompt_layers=5
ddp_similarity_aggregation=cls
ddp_pcd=true, ddp_tau_max=3.0, ddp_gamma=0.7
ddp_class_chunk_size=2, store_model=true
```

逐 task 结果：

```text
task0 mAP 99.5071, amAP 99.5071, oF1 97.0199, cF1 96.9398, loss 0.0418
task1 mAP 91.8224, amAP 95.6648, oF1 76.2983, cF1 79.7620, loss 0.3046
task2 mAP 84.4233, amAP 91.9176, oF1 66.9186, cF1 71.0141, loss 0.3542
task3 mAP 81.4017, amAP 89.2886, oF1 68.3450, cF1 69.1338, loss 0.3587
task4 mAP 79.7877, amAP 87.3884, oF1 65.7037, cF1 64.7468, loss 0.3690
```

最终诊断：

```text
pred/support 8568/7632
TP/FP/FN 5322/3246/2310
micro precision/recall/F1 0.621/0.697/0.657
threshold sweep F1: th0.5 0.465, th0.7 0.613, th0.8 0.657, th0.9 0.589
```

对比：

```text
vs cls 3ep tau3:   +2.01 mAP, +0.72 amAP, +1.65 OF1, +0.87 CF1
vs mean 3ep tau3:  +3.63 mAP, +2.04 amAP, +1.50 OF1, +2.57 CF1
vs mean 20ep tau7: +2.81 mAP, +1.65 amAP, +44.11 OF1, +41.92 CF1
```

类别层面：

- 最差 F1 类别：chair 0.327，bicycle 0.368，sheep 0.415，diningtable 0.477，sofa 0.499，pottedplant 0.508，bottle 0.520，bird 0.547。
- 20 epoch 相比 3 epoch 改善较大：sheep、dog、tvmonitor、bird、bottle、cat、train、cow。
- 20 epoch 相比 3 epoch 下降较大：chair、sofa、motorbike、bicycle、diningtable、pottedplant、bus。

结论：

- `cls + tau3/gamma0.7` 20 epoch 是当前 VOC B0-C4 最好结果，且已经解决 tau7/gamma0.2 的 recall collapse。
- 20 epoch 让整体 mAP 和 micro F1 上升，precision 从 0.576 提到 0.621，recall 从 0.722 降到 0.697，说明训练更稳但更保守。
- 当前 gap 不再主要是 PCD 强度一眼错配，而是 per-class calibration 和 prompt 表达不足；chair/diningtable/sofa/pottedplant 这类小目标/上下文依赖类偏低，bicycle/sheep/bird 有明显误报。
- 下一步优先使用该 checkpoint 做 eval-only PCD 微调，不先重训：复评估 `pcd_off`、`tau2/gamma0.7`、`tau2.5/gamma0.7`，必要时再看 `tau3.5/gamma0.7`。

### 2026-06-23 DDP v1/PCD 结构核对

已核对 DDP v1 论文源码 `arxiv:2509.23335v1` 和当前 `clip_ddp` 实现。

确认匹配：

- 一类一套 positive/negative text prompts 与 positive/negative visual prompts。
- 训练时只训练当前 task classes prompts，推理时对 seen classes 计算。
- 冻结 CLIP encoders，不使用 MULTI-LANE selectors/task CLS 作为 DDP 分类路径。
- PCD v1 公式与当前实现一致：训练 `tau=1`，推理按 `tau_max/gamma` 随 seen classes 增长。
- visual prompt 插入最后 5 个 CLIP visual layers，并在 MSA 输出后切掉 prompt token；interlayer shared prompts 是论文写法，不需要每层一套 prompt。
- optimizer 每 task 重建，只接收 DDP prompts 且 weight decay 为 0，配合 gradient mask 能避免旧类 prompt 被 Adam 动量或 weight decay 继续更新。

高风险差异/疑点：

- 论文公式是 `softmax(s+ / tau, s- / tau)`，`s` 为 cosine similarity；当前实现输出 `CLIP logit_scale * (s+ - s-) / tau`。该改动解决了项目固定 `sigmoid(logits)>0.8` 下全 0 的问题，但会改变 PCD 温度含义。需要把严格 DDP probability/logit 路径和项目阈值兼容路径显式分开。
- visual prompt 注入时，当前代码对原始 visual tokens 做 `block.norm1(x)`，但 prompts 本身没有经过同一个 LayerNorm 后再进入 Q/K/V。更严格的 pre-norm ViT 实现应考虑先拼接 `[P; x]` 再对整段序列做 `norm1` 和 attention，最后切掉 prompt tokens。
- 论文 v1 正文明确描述了 interlayer prompting 内部的 token 维度：`P^c` 为 `L_P x d`，image hidden states 为 `L x d`，prompt 参与 MSA 后切掉 prompt tokens 并保持 `L x d`。但公式中的 `E_V(P_V^c, x)` 被抽象为可直接与 text feature 做 cosine 的 visual feature，正文没有明确要求 token-wise similarity 再聚合。因此，当前 `mean/max/cls` 以及后续 `patch_mean/patch_max/topk` 应表述为实现级 visual evidence aggregation 诊断，而不是论文明确规定。
- 论文 v1 implementation details 写 prompt length 为 16；当前匹配。论文后续 DeCLIP/AST 稿改成 prompt length 4 和 `lambda=16`，这属于版本差异，不应混到当前 DDP v1/PCD 目标中。
- 论文附录提到 VOC DDP learnable params 约 `0.49M`，当前 VOC 为 `0.819M = text prompts + visual prompts`。这说明论文参数统计或实际实现可能没有把 text prompts 作为同等可训练参数计入，需优先核对；也应做 text-only/visual-only/pos-only/neg-only ablation。
- text side 当前是 `[SOT] + learnable prompt + class name + '.' + [EOT]`，positive/negative 只靠不同 learnable embeddings 区分；论文未给自然语言正负模板，所以这不是明确 bug，但需要通过 prompt 初始化/模板 ablation 验证。

结构排查结论：

- 当前实现已经是 DDP 大框架，但还不能说严格复现论文实现。
- 最值得先改/查的是 logit scale 与 PCD 的关系、visual prompt LayerNorm 注入位置、CLIP 常规 CLS/pooled visual feature 路径，以及 learnable parameter 统计/文本 prompt 是否该完全可训练；token-wise aggregation 只作为补充实验。

### 2026-06-23 DDP 结构开关实现

本地已完成第一批结构修正开关，默认值保持旧实验可复现：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_logit_scale_mode clip|none`。
  - 新增 `--ddp_prompt_norm_mode legacy|prompted`。
  - 扩展 `--ddp_similarity_aggregation`：`pooled_cls`、`patch_mean`、`patch_max`、`cls_plus_patch_max`、`topk_mean`。
  - 新增 `--ddp_similarity_topk`。
  - 新增 `--ddp_train_text_prompts`、`--ddp_train_visual_prompts`、`--ddp_prompt_polarity`。
- `multi_lane/clip_vit.py`
  - `ddp_logit_scale_mode=none` 使用严格 raw cosine margin `(s+ - s-) / tau`；`clip` 保留当前 `CLIP logit_scale * (s+ - s-) / tau`。
  - `ddp_prompt_norm_mode=prompted` 将 `[prompt; image tokens]` 一起送入 `block.norm1` 再做 MSA，之后切掉 prompt tokens；`legacy` 保留旧路径。
  - `pooled_cls` 与 `cls` 都使用 class-conditioned CLS/pooled feature 和 text feature 直接 cosine，作为最贴近论文公式抽象 `cos(E_T(...), E_V(...))` 的路径命名。
  - `patch_mean/patch_max/cls_plus_patch_max/topk_mean` 作为实现级 visual evidence aggregation 诊断项。
  - gradient mask 支持 text/visual prompt 分支和 positive/negative 分支训练开关。
- `multi_lane/utils.py`
  - DDP optimizer 参数选择遵守 `ddp_train_text_prompts` 和 `ddp_train_visual_prompts`，避免被冻结分支进入 optimizer。

论文对照：

- 对应公式 1/2：新增 `ddp_logit_scale_mode=none`，可回到论文写法中的 cosine similarity binary softmax；`clip` 是项目固定阈值兼容分支。
- 对应 interlayer prompting 公式 4：新增 `prompted` pre-norm 路径，让 prompt 和 image tokens 在 MSA 前共同归一化，再切片回 `L x d`。
- 对应 `E_V(P_V^c,x)` 抽象：新增 `pooled_cls` 命名，明确表示单个 class-conditioned visual feature，而不是声称论文要求 token-wise 聚合。

已验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。本地环境缺少 `open_clip`，forward smoke 需要在服务器 `multilane` 环境执行。

### 2026-06-23 VOC B0-C4 结构对照 3 epoch

已同步并分析三组 VOC B0-C4 结构对照实验，配置均为 `num_tasks=5, base_classes=0, epochs=3, prompt_length=16, prompt_layers=5, PCD tau3/gamma0.7, batch_size=8, chunk_size=1, store_model=true`。

结果：

```text
pooled_cls + prompted + raw cosine:
  final mAP/amAP/OF1/CF1 = 81.7107 / 89.1806 / 0.0000 / 0.0000
  pred/support = 0 / 7632 at threshold 0.8
  threshold 0.5 micro P/R/F1 = 0.361 / 0.910 / 0.517

pooled_cls + prompted + CLIP logit_scale:
  final mAP/amAP/OF1/CF1 = 79.8608 / 87.8689 / 61.5493 / 62.9698
  pred/support = 10906 / 7632
  micro P/R/F1 = 0.523 / 0.748 / 0.615
  threshold 0.9 micro P/R/F1 = 0.635 / 0.562 / 0.596

cls + legacy + CLIP logit_scale control:
  final mAP/amAP/OF1/CF1 = 77.1030 / 85.8240 / 61.5543 / 64.1565
  pred/support = 10846 / 7632
  micro P/R/F1 = 0.524 / 0.745 / 0.616
  threshold 0.9 micro P/R/F1 = 0.661 / 0.549 / 0.600
```

结论：

- `raw cosine` 更接近 DDP 论文公式，但在当前项目主指标 `sigmoid(logits)>0.8` 下完全不可用；它不是排序崩了，因为 mAP 达到 `81.71`，而是 logit 尺度太小，固定阈值不兼容。
- `pooled_cls + prompted + clip` 相比旧 control 提高 `+2.76 mAP`、`+2.04 amAP`，OF1 基本持平，CF1 低 `1.19`。这说明更贴近论文的 visual prompt pre-norm 和 pooled visual feature 路径是正向结构改动。
- F1 没明显变好，主要因为两组 `clip` 配置在固定阈值下的 precision/recall 形态几乎一样：predicted positives 都约 `1.09e4`，micro F1 都约 `0.615`。
- 下一步应把 `pooled_cls + prompted + clip + tau3/gamma0.7` 作为新的 20 epoch 候选，而不是继续围绕旧 `cls + legacy` 做长实验。
- `raw cosine` 保留为论文公式对照/排序诊断，不作为当前项目主表配置，除非后续把评估指标从固定 sigmoid 阈值改成 DDP binary probability 或按验证集校准阈值。

### 2026-06-23 DDP gap 分析补充

用户提供了 GPT5.5Thinking 对当前 DDP gap 的独立分析。其判断与当前实验结论基本一致：当前 `clip_ddp` 已接近 DDP 主框架，但论文差距更可能来自 score/probability 语义、阈值协议、CLIP preprocessing、text prompt 初始化、visual pooling 和 VOC protocol，而不是缺少正负 prompt 或 PCD 这类大模块。

更新后的高优先级排查顺序：

- 增加严格 DDP probability 评估路径：保留 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`p_pos=sigmoid(margin/tau)`，并支持 probability mode 下不再二次 sigmoid。
- 增加更密集 threshold sweep 和 oracle threshold 诊断，用来区分结构排序能力与固定阈值 calibration 问题。
- 保存每个 task/class/sample 的 `y_true/s_pos/s_neg/margin/p_pos/scaled_logit/class_id`，定位哪些类是过度预测、哪些类过度保守。
- 增加 CLIP mean/std 输入归一化选项，和当前 no-normalize/ImageNet normalize 对照。
- 打印 VOC task class order/seen class order，确认是否严格符合论文 lexicographic protocol。
- 在上述诊断后，再跑 `pooled_cls + prompted + clip` 20 epoch；如果 strict probability/oracle F1 仍差，再做 text/visual/polarity prompt ablation 与 semantic text initialization。

当前最强证据仍是 raw cosine 3 epoch `mAP=81.71` 但主阈值 F1 为 0：排序能力和固定阈值 calibration 已经明显解耦。

### 2026-06-23 DDP probability / calibration 诊断实现

已实现 GPT5.5Thinking 建议的第一批诊断改动：

- `multi_lane/clip_vit.py`
  - DDP forward 现在除主 logits 外，会缓存完整 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`ddp_prob=sigmoid(margin/tau)`、`scaled_logit=CLIP logit_scale*margin/tau`。
  - 主训练/默认评估仍返回 scaled logits，不破坏旧实验复现。
- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_eval_score_mode logits|probability`。
  - 新增 `--ddp_eval_threshold`，默认 `0.8`。
  - 新增 `--ddp_score_dump true|false`。
  - 默认 DDP diagnostic thresholds 扩展为 `0.1 ... 0.9`。
  - 新增 `--clip_normalize_input true|false`。
- `multi_lane/engine.py`
  - DDP eval metrics 可选择 `logits` 模式：`sigmoid(scaled_logit)>threshold`，或 `probability` 模式：`ddp_prob>threshold`，后者不再二次 sigmoid。
  - DDP diagnostics 增加 dense threshold sweep、global oracle threshold、per-class oracle F1 诊断。
  - detail report 增加 `s_pos/s_neg/margin/ddp_prob/scaled_logit` 分布字段。
  - `--ddp_score_dump true` 时保存 `detail/{run_name}_task{t}_ddp_scores.npz`，包含 `y_true/eval_score/s_pos/s_neg/margin/ddp_prob/scaled_logit/class_ids`。
- `multi_lane/datasets.py`
  - 新增 CLIP mean/std normalization：`mean=(0.48145466,0.4578275,0.40821073)`，`std=(0.26862954,0.26130258,0.27577711)`。
  - `--clip_normalize_input true` 优先于旧 `--normalize_input`；默认仍保持旧行为。
- `main.py`
  - 每次运行打印并保存 `class_order.json`，用于核对 VOC lexicographic task protocol。

验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

下一步实验目的：

- 用同一 checkpoint 做 eval-only 对照，比较 `logits` 与 `probability` 评估语义。
- 开启 `ddp_score_dump`，检查 strict DDP probability/oracle threshold 下 F1 是否接近论文。
- 如果 strict probability + oracle 仍差，再继续查 text prompt 初始化、CLIP normalize 和 visual pooling。

### 2026-06-23 DDP calibration 诊断实验启动

已按用户要求在服务器 `/mnt/haoyuan/workspace/multi-lane-main` 启动三个 tmux 实验：

```text
tmux: voc_ddp_eval_logits_diag
gpu: 0
log: ./logs/voc_ddp_eval_logits_diag.log
output: ./output/voc_ddp_eval_logits_diag
purpose: 同一 pooled_cls+prompted checkpoint，使用当前 scaled-logit 语义复评估，并保存 dense threshold/oracle/score dump。

tmux: voc_ddp_eval_probability_diag
gpu: 1
log: ./logs/voc_ddp_eval_probability_diag.log
output: ./output/voc_ddp_eval_probability_diag
purpose: 同一 checkpoint，使用严格 DDP probability 语义复评估，检查 raw DDP probability 下阈值/校准问题。

tmux: voc_ddp_3ep_clipnorm
gpu: 4
log: ./logs/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm.log
output: ./output/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm
purpose: 训练 3 epoch CLIP mean/std normalized 输入，对照 no-normalize 下的 pooled_cls+prompted+clip。
```

启动后确认：

- 三个 tmux session 均存在。
- `voc_ddp_eval_logits_diag` 已输出 task2 diagnostics。
- `voc_ddp_eval_probability_diag` 已输出 task1 diagnostics。
- `voc_ddp_3ep_clipnorm` 已进入训练并打印配置。

### 2026-06-23 DDP calibration 诊断结果

三组实验已同步并分析。

1. `voc_ddp_eval_logits_diag`

使用同一 `pooled_cls + prompted + clip` 3 epoch checkpoint，评估语义为当前工程 scaled logits：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8689 / 61.5493 / 62.9698
pred/support = 10906 / 7632
micro P/R/F1 = 0.523 / 0.748 / 0.615
global oracle threshold = 0.8, oracle micro-F1 = 0.615
per-class oracle F1 = 0.696
```

结论：dense threshold sweep 证明 final task 的全局最佳阈值仍是 `0.8`，不是简单调全局阈值就能接近论文。per-class oracle 从 `62.97` 提到约 `69.6`，说明存在明显 class-wise calibration 问题，但 oracle 后仍低于论文主表。

2. `voc_ddp_eval_probability_diag`

同一 checkpoint，评估语义为严格 DDP probability：`ddp_prob=sigmoid((s_pos-s_neg)/tau)`，主阈值 `0.5`：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8687 / 46.2331 / 51.5324
pred/support = 22459 / 7632
micro P/R/F1 = 0.310 / 0.911 / 0.462
global oracle threshold = 0.5, oracle micro-F1 = 0.462
per-class oracle F1 = 0.515
```

结论：严格 DDP probability 没有改善 F1，反而明显更差。`ddp_prob` 分布集中在 0.5 附近，低于 0.5 基本全预测，高于 0.6 基本无预测，说明 raw margin 尺度太小；当前论文公式路径排序仍可用，但 probability calibration 不适合项目固定阈值。

3. `voc_ddp_3ep_pooledcls_prompted_clip_clipnorm`

在 `pooled_cls + prompted + clip` 基础上启用 CLIP mean/std 输入归一化训练 3 epoch：

```text
final mAP/amAP/OF1/CF1 = 81.5703 / 88.7328 / 65.0564 / 66.6065
pred/support = 10792 / 7632
micro P/R/F1 = 0.555 / 0.785 / 0.651
global oracle threshold = 0.9, oracle micro-F1 = 0.669
per-class oracle F1 = 0.713
```

对比 no-normalize 的同结构 3 epoch：

```text
no-normalize: mAP/OF1/CF1 = 79.8607 / 61.5493 / 62.9698
CLIP normalize: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
gain: +1.71 mAP, +3.51 OF1, +3.64 CF1
```

对比旧 `cls + legacy + clip` 20 epoch：

```text
old 20ep: mAP/OF1/CF1 = 79.7877 / 65.7037 / 64.7468
clipnorm 3ep: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
```

结论：CLIP preprocessing 是重要 gap。仅 3 epoch 的 `pooled_cls + prompted + clip + CLIP normalize` 已经超过旧 20 epoch 的 mAP/CF1，OF1 基本接近。下一步应优先跑该配置 20 epoch。

当前 updated 判断：

- 严格 probability 评估说明 gap 不是“把评估改回 DDP probability 就解决”，raw margin 尺度本身过小。
- CLIP mean/std normalize 明确有效，应作为后续 VOC 主配置。
- 当前剩余问题更偏 class-wise calibration 和 prompt/类别语义，尤其 sheep/bicycle/bottle 等误报仍多，diningtable/chair 等仍偏难。

### 2026-06-23 VOC B0-C4 pooled_cls+prompted+clipnorm 20 epoch

本地已同步主候选 20 epoch 结果：

```text
log:    ./logs/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep/checkpoints.pth
```

配置：

```text
dataset: VOC B0-C4
head: clip_ddp
aggregation: pooled_cls
prompt_norm_mode: prompted
logit_scale_mode: clip
clip_normalize_input: true
PCD: tau_max=3.0, gamma=0.7
epochs: 20
batch_size: 8
chunk_size: 1
```

最终结果：

```text
task1: mAP/OF1/CF1 = 99.6331 / 97.8536 / 97.8136
task2: mAP/OF1/CF1 = 94.1302 / 76.4492 / 81.0003
task3: mAP/OF1/CF1 = 86.0975 / 67.8669 / 72.8785
task4: mAP/OF1/CF1 = 83.5890 / 69.1588 / 72.1785
task5: mAP/OF1/CF1 = 82.9480 / 67.3165 / 68.9025
```

final diagnostics:

```text
threshold 0.8: pred/support=10527/7632, P/R/F1=0.581/0.801/0.673
threshold 0.9: pred=6730, P/R/F1=0.736/0.649/0.690
global oracle threshold=0.9, oracle micro-F1=0.690
per-class oracle F1=0.748
```

对比：

```text
vs old cls+legacy+clip 20ep:
  +3.16 mAP, +1.61 OF1, +4.16 CF1

vs clipnorm 3ep:
  +1.38 mAP, +2.26 OF1, +2.30 CF1

vs no-normalize pooled_cls+prompted 3ep:
  +3.09 mAP, +5.77 OF1, +5.93 CF1
```

结论：

- 这是当前 VOC B0-C4 最佳结果，证明 `pooled_cls + prompted + CLIP normalize` 是正确主线。
- 20 epoch 没有过拟合崩塌，较 3 epoch 继续提升。
- 但当前 final task 全局最佳阈值已经偏向 `0.9`，说明模型仍偏误报；下一步优先做 eval-only PCD/threshold 复评估，而不是马上改结构。
- 相对 DDP 论文 VOC B0-C4 主表，当前约低 `7.3 mAP / 13.5 OF1 / 8.0 CF1`，gap 明显缩小但还没到论文水平。
- 最差类别仍集中在 `bicycle/sheep/bottle/chair/diningtable/car`。其中 bicycle/sheep/bottle 是高 recall 低 precision 的误报问题； diningtable 是高 precision 低 recall 的漏检问题；chair AP/F1 都偏低。

下一步建议：

- 用该 checkpoint 做 eval-only：`tau2.5/tau3/tau3.5`，并比较 `ddp_eval_threshold=0.8/0.9`。
- 若 threshold 0.9 + 合理 PCD 明显提升 OF1/CF1，再跑 B10-C2 或 20 epoch 正式论文协议。
- 若仍明显低于论文，再进入 text prompt semantic initialization 和 class-wise calibration。

### 2026-06-23 下一阶段 margin-first 计划

用户提供 GPT5.5Thinking 新判断：当前核心问题应从 PCD/threshold 转向 positive/negative prompt margin separation。结合最新 `pooled_cls+prompted+clipnorm` 20 epoch 结果，当前判断如下：

- CLIP normalize 和 prompted pooled feature 已确认有效，主结构路线正确。
- strict DDP probability final OF1/CF1 只有 `46.23/51.53`，且 `ddp_prob` 大量挤在 0.5 附近，说明 raw margin 本身不够大。
- scaled-logit 分支能得到 `82.95/67.32/68.90`，说明排序信息存在，但依赖 CLIP scale/threshold 做工程校准。
- final threshold 0.9 比 0.8 的 micro-F1 更高，说明误报仍偏多，尤其 `bicycle/sheep/bottle`；而 `diningtable/chair` 仍有漏检/表达弱问题。

下一阶段计划：

1. 先做 checkpoint eval-only calibration sweep，但目标是读 margin 而不是继续盲调 PCD：
   - `tau2.5/tau3/tau3.5` × `threshold0.8/0.9`
   - 重点看 `s_pos/s_neg/margin/ddp_prob` 分布、per-class margin gap、oracle threshold。
2. 增加 margin summary report：
   - 每类正样本 margin mean/p10/p50/p90。
   - 每类负样本 margin mean/p10/p50/p90。
   - `pos_margin_mean - neg_margin_mean`。
   - 正样本 raw prob 大于 0.5/0.6 的比例、负样本 raw prob 大于 0.5/0.6 的比例。
3. 做 prompt branch ablation：
   - full text+visual pos+neg。
   - visual-only。
   - text-only。
   - text frozen vs text trainable。
   - visual frozen vs visual trainable。
   - positive-only / negative ablation。
4. 做 semantic text initialization：
   - positive templates: `a photo containing a {class}.`, `a photo of a {class}.`, `a photo with a {class}.`
   - negative templates: `a photo without a {class}.`, `a photo not containing a {class}.`, `a photo with no {class}.`
   - 观察是否提升 positive sample 上的 `s_pos`、negative sample 上的 `s_neg`，并扩大 margin gap。
5. 如果 ablation 证明 DDP loss 本身不足以拉开 margin，再考虑 margin regularizer；该步骤属于超出 DDP v1 的改进，需要单独标记为 extension，而不是论文复现配置。

### 2026-06-23 margin-first 代码改动

本轮按用户要求进入 positive/negative prompt margin separation 排查，已完成以下未提交改动：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_text_init random|same|semantic`。
  - 新增 `--ddp_positive_text_template`、`--ddp_negative_text_template`。
- `multi_lane/clip_vit.py`
  - `random` 保持旧 DDP text prompt 随机初始化，保证已有实验可复现。
  - `same` 让 positive/negative text prompt 从相同随机值开始，用于诊断正负分支是否真的学出分离。
  - `semantic` 用正/负模板前缀分别初始化 learnable text prompt，例如 positive 默认来自 `a photo containing a`，negative 默认来自 `a photo without a`。
  - semantic 初始化只在 `set_class_names()` 后执行一次；eval-only 加载 checkpoint 时，checkpoint 参数会覆盖初始化值，不污染旧模型复评估。
- `multi_lane/engine.py`
  - detail report 新增 per-class positive/negative margin 分布。
  - overall diagnostics 新增 `diag_margin_gap_mean`、正/负样本 margin 大于 0 的比例、正/负样本 raw `ddp_prob` 大于 0.5/0.6 的比例。
  - 评估日志新增 `[DDP margin task ...]` 行，便于直接观察 raw margin 是否拉开。
- `PROJECT_CONTEXT.md`、`TODO_NEXT.md`、`WORK_LOG.md`
  - 已同步记录本轮改动、目的和下一轮实验方向。

本地验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

本地 `python3 main.py voc --help` 因本机 Python 缺少 `numpy` 无法运行；已用直接 parser 检查确认新增参数可见。服务器 `multilane` 环境仍需运行完整 `main.py voc --help` 和 smoke。

### 2026-06-23 VOC B0-C4 margin init 对照结果

用户已同步两组 3 epoch 对照：

```text
random:
  log: ./logs/voc_ddp_margin_random_3ep.log
  output: ./output/voc_ddp_margin_random_3ep
semantic:
  log: ./logs/voc_ddp_margin_semantic_3ep.log
  output: ./output/voc_ddp_margin_semantic_3ep
```

共同配置：

```text
VOC B0-C4, num_tasks=5, base_classes=0, epochs=3
head_mode=clip_ddp
aggregation=pooled_cls
prompt_norm_mode=prompted
logit_scale_mode=clip
clip_normalize_input=true
tau_max=3.0, gamma=0.7
ddp_score_dump=true
```

final 结果：

```text
random:   mAP 82.3370, amAP 89.1434, OF1 66.7329, CF1 67.8947
semantic: mAP 83.8680, amAP 90.1785, OF1 70.2598, CF1 70.6409
delta:   +1.5310 mAP, +1.0351 amAP, +3.5269 OF1, +2.7462 CF1
```

final threshold 0.8 诊断：

```text
random:   pred/support=10491/7632, P=0.5764, R=0.7923, F1=0.6673
semantic: pred/support= 8417/7632, P=0.6698, R=0.7387, F1=0.7026
```

margin 诊断：

```text
random:   pos_margin_mean=0.0940, neg_margin_mean=-0.0747, gap=0.1688
semantic: pos_margin_mean=0.0711, neg_margin_mean=-0.0645, gap=0.1356
```

结论：

- `semantic` 明显提升主指标，主要原因是减少过度预测，precision 从 `0.5764` 提升到 `0.6698`。
- `semantic` 没有直接扩大 raw margin gap；gap 反而从 `0.1688` 降到 `0.1356`。
- 两组 final `pos_ddp_prob>0.6` 和 `neg_ddp_prob>0.6` 都是 0，说明 raw DDP probability 仍然挤在 0.5 附近，raw margin 绝对尺度仍不足。
- 因此 semantic init 可以作为下一轮主配置候选，但它解决的是 scaled-logit 校准/误报问题，不是严格意义上直接把 raw DDP margin 拉大。
- 下一步应在 semantic 配置上继续做 branch-level 排查：冻结 text prompts / 冻结 visual prompts / positive-only / negative-only，以定位 margin gap 主要受哪一支限制。

### 2026-06-23 semantic 20ep 与 branch ablation 结果

用户已同步五组结果：

```text
semantic 20ep:
  log: ./logs/voc_ddp_margin_semantic_20ep.log
  output: ./output/voc_ddp_margin_semantic_20ep

branch ablation 3ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_visual_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_posonly_3ep.log
  ./logs/voc_ddp_ablate_sem_negonly_3ep.log
```

final summary：

```text
random 3ep:        mAP 82.3370, OF1 66.7329, CF1 67.8947, gap 0.1688, pred 10491
full semantic 3ep: mAP 83.8680, OF1 70.2598, CF1 70.6409, gap 0.1356, pred  8417
semantic 20ep:     mAP 84.2737, OF1 70.5397, CF1 72.5721, gap 0.1252, pred  9323
text frozen 3ep:   mAP 82.3217, OF1 71.1562, CF1 72.7656, gap 0.1298, pred  9044
visual frozen 3ep: mAP 82.4743, OF1 68.7695, CF1 69.7780, gap 0.1270, pred  8695
positive-only 3ep: mAP 81.2522, OF1 70.9906, CF1 71.5840, gap 0.1186, pred  8156
negative-only 3ep: mAP 81.3779, OF1 68.4138, CF1 72.4098, gap 0.1355, pred  9768
old best 20ep:     mAP 82.9480, OF1 67.3165, CF1 68.9025
```

关键结论：

- `semantic 20ep` 达到当前最高 mAP：`84.27`，相对 old best `82.95` 提升 `+1.33 mAP`，相对 `semantic 3ep` 只提升 `+0.41 mAP`。
- 固定阈值 F1 的当前最好不是 full semantic 20ep，而是 `text_frozen 3ep`：`OF1 71.16 / CF1 72.77`。
- `text_frozen` 表示固定 semantic text prompts，只训练 visual prompts。它比 `visual_frozen` 明显好，说明当前 F1 主要依赖 visual prompt 学到 class-conditioned evidence。
- full semantic 从 3ep 到 20ep 主要增加 recall：`0.739 -> 0.784`，但 precision 从 `0.670 -> 0.641` 降低，predicted positives 从 `8417 -> 9323` 增加，因此 OF1 只小幅提升。
- raw margin 绝对尺度仍未解决：所有组 `raw_prob_pos_mean` 约 `0.505-0.508`，`raw_prob_neg_mean` 约 `0.494-0.496`，`raw_prob>0.55` 几乎为 0。当前 F1 仍依赖 CLIP logit scale 放大 margin。
- `random` 的 raw margin gap 最大，但 F1 更差，说明“gap 大”本身不够；还需要正负样本分布位置和阈值校准合理。
- `positive-only` 更保守，precision 较高、recall 较低；`negative-only` 更容易过预测，OF1 较低但 CF1 接近。这说明 presence branch 对抑制 false positive/固定阈值 precision 很重要，absence branch 单独训练不能充分压住误报。

下一步建议：

- 优先跑 `semantic text frozen 20ep`：固定语义 text anchor，只训练 visual prompts，验证 3ep 最佳 F1 是否能随训练拉长继续保持/提升。
- 同时可跑 `semantic positive-only 20ep` 作为 precision-oriented 对照。
- 若目标是严格 DDP raw probability，需要考虑 margin regularizer 或 raw binary-softmax loss 加强；这属于 DDP extension，不能混入论文复现主配置。

### 2026-06-24 semantic 长训消融与 seed1 稳定性复验

用户同步四组完整结果：

```text
text frozen 20ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_20ep.log
  ./output/voc_ddp_ablate_sem_text_frozen_20ep

positive-only 20ep:
  ./logs/voc_ddp_ablate_sem_posonly_20ep.log
  ./output/voc_ddp_ablate_sem_posonly_20ep

text frozen 3ep seed1:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep_seed1.log
  ./output/voc_ddp_ablate_sem_text_frozen_3ep_seed1

full semantic 3ep seed1:
  ./logs/voc_ddp_margin_semantic_3ep_seed1.log
  ./output/voc_ddp_margin_semantic_3ep_seed1
```

所有实验均为 VOC B0-C4、`pooled_cls + prompted + clip + CLIP normalize`、
PCD `tau_max=3.0,gamma=0.7`、batch size 16、seed0/seed1，并保存了 detail/score dump；
两个 20 epoch 实验保存了 checkpoint。

最终结果：

```text
full semantic 3ep seed0:  mAP/amAP/OF1/CF1 = 83.8680/90.1785/70.2598/70.6409
full semantic 3ep seed1:  mAP/amAP/OF1/CF1 = 83.6778/90.1794/70.5977/70.8508
text frozen 3ep seed0:    mAP/amAP/OF1/CF1 = 82.3217/89.1572/71.1562/72.7656
text frozen 3ep seed1:    mAP/amAP/OF1/CF1 = 81.5404/88.5455/69.2907/71.3656
text frozen 20ep seed0:   mAP/amAP/OF1/CF1 = 80.8415/87.9108/68.2167/71.2956
positive-only 3ep seed0:  mAP/amAP/OF1/CF1 = 81.2522/88.3632/70.9906/71.5840
positive-only 20ep seed0: mAP/amAP/OF1/CF1 = 83.0647/89.3826/70.0879/72.6094
full semantic 20ep seed0: mAP/amAP/OF1/CF1 = 84.2737/90.3961/70.5397/72.5721
```

结论：

- full semantic 3ep 两个 seed 几乎重合，mAP 波动仅 `0.19`，是当前最稳定的结构。
- text-frozen 3ep 的 seed0 高 F1 未稳定复现；seed1 回落 `0.78 mAP/1.87 OF1/1.40 CF1`。
- text-frozen 拉长到 20 epoch 后全面退化。pred/support 从 3ep seed0 的 `9044/7632`
  增到 `10475/7632`，precision `0.656 -> 0.590`、recall `0.777 -> 0.809`，
  说明 visual prompts 长训后过预测，而不是 margin 完全没有学习。
- text-frozen 20ep 的 global oracle threshold 为 `0.9`，oracle micro-F1 `0.698`，
  仍低于 3ep seed0 主阈值 F1 `0.712`；问题不能只靠换阈值补回。
- positive-only 20ep 相对 3ep 提升 `+1.81 mAP/+1.03 CF1`，但 OF1 下降 `0.90`；
  相对 full semantic 20ep 仍低 `1.21 mAP/0.45 OF1`，没有证明应删除 negative branch。
- raw DDP probability 仍集中在 0.5：四组 final positive mean 约 `0.506`，
  negative mean 约 `0.495`，`>0.6` 比例仍为 0。
- 主要困难类别继续是 `chair/bicycle/bottle/car/diningtable`。text-frozen 20ep 的
  bicycle 出现 `P=0.202,R=0.940`，是典型严重误报；chair 的 margin gap 仅 `0.043`，
  表达分离仍弱。
- text-frozen learnable 参数 `491,520` 与论文约 `0.49M` 很接近，是重要结构线索；
  但其效果和长训趋势表明还需核对论文参数统计口径、训练调度及 prompt 更新方式，
  不能把参数量相等直接当作结构已严格复现。

下一步：

- 主线恢复为 full semantic text+visual prompts；text-frozen/positive-only 作为消融保留。
- 优先做 full semantic 的学习率/训练时长对照，避免当前 20 epoch 出现 precision 下降。
- 基于已有 20ep checkpoints 做 threshold/PCD eval-only，不重训地确认 calibration 上限。
- 在继续增加非论文 loss 前，先核对论文官方实现的 optimizer、LR、scheduler、warmup、
  visual prompt 初始化和 `0.49M` 参数统计口径。

### 2026-06-24 Git 提交与远端推送

- 服务器分支：`feature/clip-taskCLS-posneg-text-head`。
- 功能提交：`19417f1 Add DDP semantic prompt diagnostics`。
- 提交包含 DDP semantic text initialization、prompt branch ablation 开关、
  CLIP normalization、eval-only/checkpoint、score/margin diagnostics 及上下文文档。
- `python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py
  multi_lane/continual_datasets/*.py` 已在服务器 `multilane` 环境通过。
- 已首次推送并设置上游：
  `origin/feature/clip-taskCLS-posneg-text-head`。
- 推送后本地与远端完整哈希一致：
  `19417f1500dd3e1046e79afe18711628f835eb84`。
- 无关的 `train_c100.sh` 文件末尾换行改动未纳入功能提交，仍保留在服务器工作区。

### 2026-06-24 margin 与 mAP 阶段结论

- full semantic 20ep final margin：
  `pos_mean=0.0731`、`neg_mean=-0.0521`、gap `0.1252`。
- `93.4%` 正样本的 margin 大于 0，说明多数正样本方向正确；
  `15.0%` 负样本的 margin 仍大于 0，说明 false-positive 排序仍明显存在。
- task5 PCD `tau=3` 后 strict DDP probability 仍为
  positive `0.5061`、negative `0.4957`，绝对尺度没有充分拉开。
- random 3ep gap `0.1688` 大于 semantic 20ep 的 `0.1252`，但 mAP 只有 `82.34`
  而非 `84.27`。因此 mAP 需要改善每类正负样本排序，不能只优化 overall mean gap。
- full semantic 20ep 最低 AP 类别为：
  `chair 38.51`、`bottle 51.73`、`tvmonitor 68.62`、`pottedplant 73.80`、
  `diningtable 75.33`、`sofa 76.15`、`car 78.05`。
- 代码核对发现配置中的 warmup/min-lr 参数未被实际 scheduler 使用；
  当前每 task 使用裸 `CosineAnnealingLR(T_max=epochs)`，并无条件重建 optimizer。
  这是下一阶段论文协议核对和 mAP 优化的高优先级风险点。

### 2026-06-24 官方 supplementary 对齐改造

从 OpenReview supplementary material 取得官方代码包 `CODE_DDP`，核对后确认：

```text
visual token similarity scale: 20
token attention: softmax(positive token similarities)
same token attention weights applied to positive and negative branches
weighted pair-logit scale: 5
effective final margin scale: 20 * 5 = 100
loss: binary pair softmax BCE, reduction=sum, loss weight=0.03
optimizer: Adam, lr=5.9e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0
scheduler: MultiStepLR, milestones=[0,20], gamma=0.1
optimizer lifetime: one optimizer across all tasks
epochs: 20
batch size: 8
train transform: Resize(224), CutoutPIL(0.2), RandAugment, CLIP normalize
val transform: Resize(224), CLIP normalize
visual prompt layers: [7,8,9,10,11]
text/visual prompt length: 16
```

因此此前计划中的“raw DDP train 是论文主路径”不成立。论文公式省略了实现级 scale，
但 supplementary code 明确使用固定 100 的强尺度。raw margin train 只保留为诊断
ablation，不再标记为严格论文配置。

本轮修改：

- `multi_lane/configs/custom_types.py`
  - 新增 `paper_attention` aggregation。
  - 新增 train/eval logit scale 分离和固定 paper scale。
  - 新增 `paper_sum` loss、持续 optimizer、paper scheduler 和 paper transform 参数。
- `multi_lane/clip_vit.py`
  - 实现官方 positive-token attention：positive similarities 经 `softmax(20*s)`，
    同一权重用于正负分支 token 汇总。
  - train/eval 可分别选择 `clip/none/paper` scale。
- `multi_lane/engine.py`
  - 实现官方 sum binary-softmax 等价 loss：margin BCE sum × `0.03`。
  - 支持 optimizer 跨 task 持续存在和全局 MultiStepLR。
- `multi_lane/utils.py`
  - 清理非当前类别 Adam state，确保旧类 prompt 在持续 optimizer 下不受历史动量漂移。
- `multi_lane/datasets.py`
  - 新增官方 Resize + CutoutPIL + RandAugment + CLIP normalize 路径。
- `main.py`
  - 支持 DDP unscaled optimizer LR override，并打印实际 LR。
- `requirements.txt`
  - 新增 `randaugment==1.0.2`。

服务器验证：

```text
py_compile: passed
paper attention formula equality: passed
pair-softmax BCE == margin BCE(sum): passed
full CPU forward/backward: passed
current task prompts have gradients, future prompts zero: passed
continual Adam frozen-state test: passed
paper scheduler: 5.9e-4 at init, 5.9e-5 after global epoch 20
paper transform with isolated randaugment 1.0.2: passed
```

注意：

- supplementary 的公开 `DDP.py` 硬编码示例是 VOC B4-C2，论文主表还包含 B0-C4；
  方法级参数可以对齐，但 B0-C4 的 PCD 超参仍需用新 attention 路径重新验证。
- 官方代码没有显式清理旧类 Adam 动量；本实现的 state 清理是为了严格满足论文
  “past prompts are frozen as knowledge anchors”的方法描述。

### 2026-06-24 official random 首次启动失败修复

- `voc_ddp_paperattn_semantic_3ep` 已在 GPU0 正常运行。
- `voc_ddp_official_random_3ep` 首次启动后 tmux 立即消失。
- 日志确认不是 OOM；GPU1 为空闲。失败发生在 DataLoader worker：
  `randaugment==1.0.2` 使用了 NumPy 1.24 后删除的 `np.int`。
- 已在 `_paper_randaugment()` 中对该官方旧依赖增加 `np.int = int` 兼容映射，
  不改变 RandAugment 策略。
- 服务器 NumPy 1.26.4 下已验证：
  `Resize -> CutoutPIL -> RandAugment -> ToTensor -> CLIP Normalize`
  可正常输出 `[3,224,224]` tensor。
- 当前 GPU1 空闲，可重启 official random；GPU2-7 均被其他任务占用，不启动 semantic
  official 对照，也不触碰其他用户进程。

### 2026-06-24 official-path 三组 3ep 结果

三组实验均完整结束并同步日志、detail JSON 和 score dumps：

```text
A semantic + paper attention:
  mAP/amAP/OF1/CF1 = 84.4317/90.5850/74.5444/70.9225
  P/R/F1 = 0.849/0.665/0.745
  margin gap = 0.1039

B official random:
  mAP/amAP/OF1/CF1 = 86.0550/91.9571/59.0481/51.9978
  P/R/F1 = 0.954/0.428/0.590
  margin gap = 0.1482
  oracle global threshold 0.7, F1 = 0.745

C official semantic:
  mAP/amAP/OF1/CF1 = 88.0259/93.3177/37.9391/28.7707
  P/R/F1 = 0.985/0.235/0.379
  margin gap = 0.1250
  oracle global threshold 0.6, F1 = 0.796
```

对照结论：

- A 对旧 semantic 3ep：`+0.56 mAP/+0.41 amAP/+4.28 OF1/+0.28 CF1`。
  token attention 显著提高 precision，并确认官方聚合不是无效细节。
- B 对 A：`+1.62 mAP/+1.37 amAP`，说明官方 loss、augmentation、optimizer/scheduler
  组合继续改善排序和 margin。
- C 对 B：`+1.97 mAP/+1.36 amAP`。semantic 初始化在官方路径上仍然有效，
  尽管 overall mean margin gap 更小，再次证明 AP 不能只由 mean gap 判断。
- C 对旧 semantic 3ep：`+4.16 mAP/+3.14 amAP`；对旧 semantic 20ep：
  `+3.75 mAP/+2.92 amAP`。仅 3 epoch 已明显超过此前所有 mAP。
- C 距 DDP 论文 VOC B0-C4：last mAP `-2.17`，average mAP `-1.48`。

official semantic 对旧 semantic 3ep 的关键 per-class AP 变化：

```text
chair       34.95 -> 61.61 (+26.66)
car         80.11 -> 92.69 (+12.57)
tvmonitor   75.01 -> 86.01 (+11.00)
pottedplant 68.26 -> 78.13 (+9.87)
diningtable 77.75 -> 84.77 (+7.02)
sofa        71.77 -> 77.44 (+5.67)
person      89.94 -> 95.74 (+5.79)
```

固定阈值解释：

- B/C 使用 official VOC `tau_max=7,gamma=0.2`。在 B0-C4 final task，tau=7 将
  confidence 大幅压向中间，threshold 0.8 导致 recall collapse。
- C 在 threshold 0.6 时已有 `P=0.763,R=0.832,F1=0.796`，说明排序和可校准上限明显
  高于当前主表 F1。
- 对固定 0.8 阈值，`tau≈2.0` 与 tau7/threshold0.6 的 margin 边界近似等价；
  因此下一步优先 eval-only `tau2/gamma0.7` 和 `tau3/gamma0.7`。

### 2026-06-24 official semantic tau2 优化运行脚本

按用户要求，已基于当前最高 mAP 的 `official semantic` 配置新增两个可直接运行的
VOC B0-C4 脚本：

```text
run_voc_ddp_official_semantic_tau2_3ep.sh
run_voc_ddp_official_semantic_tau2_20ep.sh
```

配置策略：

- 保持 official semantic recipe 不变：`paper_attention`、`paper` logit scale、
  `paper_sum` loss、continual optimizer、paper MultiStepLR、paper transform、
  semantic text initialization。
- 只把 PCD 从官方 VOC 风格的 `tau_max=7.0,gamma=0.2` 改为
  `tau_max=2.0,gamma=0.7`。
- 目的不是改排序，而是先修复 fixed-threshold 0.8 下的 confidence 过度压缩问题。

脚本行为：

- 日志写入：
  - `./logs/voc_ddp_official_semantic_tau2_g07_3ep.log`
  - `./logs/voc_ddp_official_semantic_tau2_g07_20ep.log`
- 输出写入：
  - `./output/voc_ddp_official_semantic_tau2_g07_3ep`
  - `./output/voc_ddp_official_semantic_tau2_g07_20ep`
- 均开启 `--store_model` 和 `--ddp_score_dump true`。
- 可通过 `GPU=<id> ./script.sh` 指定显卡。

已完成本地检查：

```text
bash -n run_voc_ddp_official_semantic_tau2_3ep.sh run_voc_ddp_official_semantic_tau2_20ep.sh
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

2026-06-24 追加修复：

- 远程服务器用 `bash run_voc_ddp_official_semantic_tau2_3ep.sh` 启动时，
  `/root/.bashrc` 在 `set -u` 下引用未定义 `PS1`，导致脚本提前退出。
- 已在两个 tau2 脚本中对 `source ~/.bashrc` 前后临时切换 `set +u` / `set -u`，
  仅规避 shell 初始化兼容问题，不改变任何实验配置。
- 已重新运行 `bash -n`，结果通过。

注意：当前本地目录不是 Git 仓库，无法在本地创建实验分支或提交；Git 状态仍以服务器
`/mnt/haoyuan/workspace/multi-lane-main` 为准。

### 2026-06-24 official semantic tau2 3ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_3ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_3ep
ckpt:   ./output/voc_ddp_official_semantic_tau2_g07_3ep/checkpoints.pth
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.0259/93.3177/79.4418/78.6858
pred/support = 8420/7632
micro precision/recall/F1 = 0.7572/0.8354/0.7944
oracle global threshold = 0.80, F1 = 0.794
per-class oracle F1 = 0.824, threshold mean = 0.790
```

对比：

```text
official semantic tau7 3ep: 88.0259/93.3177/37.9391/28.7707
official semantic tau2 3ep: 88.0259/93.3177/79.4418/78.6858
delta:                         +0.00/+0.00/+41.50/+49.92

official random tau7 3ep:      86.0550/91.9571/59.0481/51.9978
delta tau2 semantic:            +1.97/+1.36/+20.39/+26.69

paper-attention semantic 3ep:   84.4317/90.5850/74.5444/70.9225
delta tau2 semantic:            +3.59/+2.73/+4.90/+7.76
```

结论：

- `tau2/gamma0.7` 没改变排序，mAP 与 tau7 official semantic 完全一致；
  它只改变 fixed-threshold calibration。
- tau2 主阈值 0.8 的结果几乎复现了 tau7 score dump 中 threshold 0.6 的 oracle 行，
  验证此前“tau≈2 对齐主阈值 0.8”的判断成立。
- 当前已接近 DDP 论文 VOC B0-C4 OF1 `80.8`，但 CF1 仍低约 `5` 点，剩余问题主要在
  类别级 calibration，特别是 `bottle/chair/bicycle/diningtable/sheep/sofa`。

### 2026-06-24 official semantic tau2 20ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_20ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_20ep
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.5226/93.4123/72.7760/76.3618
pred/support = 10286/7632
micro precision/recall/F1 = 0.6339/0.8543/0.7278
margin: pos=0.0813, neg=-0.0598, gap=0.1411
global oracle threshold = 0.90, F1 = 0.7584
per-class oracle F1 = 0.805, threshold mean = 0.795
```

关键对比：

```text
official semantic tau2 3ep:  88.0259/93.3177/79.4418/78.6858
official semantic tau2 20ep: 88.5226/93.4123/72.7760/76.3618
delta:                        +0.4967/+0.0946/-6.6658/-2.3240

paper B0-C4:                  90.2/94.8/80.8/76.9
20ep gap:                     -1.68/-1.39/-8.02/-0.54
```

结论：

- 20ep 让 ranking 和 margin 继续改善，但收益不大：mAP 只增加 `0.50`，
  margin gap 从 `0.1250` 增加到 `0.1411`。
- tau2 在 3ep 时校准正好，在 20ep 时偏激进。预测正类从 `8420` 增到 `10286`，
  precision 从 `0.757` 降到 `0.634`，而 recall 仅从 `0.835` 升到 `0.854`。
- 20ep 最佳扫描阈值已经移到 `0.9`，因此下一步先使用现有 checkpoint 做
  eval-only tau `2.5/3/4` 对照，不能把 OF1 下降误判成结构退化。
- mAP 剩余问题集中在困难类和跨任务退化。最终弱类：
  `bottle 61.96`、`chair 61.45`、`sofa 77.13`、`pottedplant 79.00`、
  `diningtable 82.03`。
- 从类别刚学完到 task5，主要 AP 下降：
  `bottle -29.97`、`chair -14.96`、`bicycle -10.53`、`cow -9.03`、
  `diningtable -5.92`。这不是旧 prompt 参数漂移的直接证据，因为 prompts 已冻结；
  更可能是后续 seen-class 评估中类别间分数竞争、任务数据分布和负样本覆盖不足。

### 2026-06-24 最佳 20ep tau2 版本提交与推送

- 分支：`feature/clip-taskCLS-posneg-text-head`
- 提交：`5ac25a8 Align CLIP DDP with official training recipe`
- 已推送：`origin/feature/clip-taskCLS-posneg-text-head`
- 提交包含 paper-aligned DDP 实现、official semantic tau2 3ep/20ep 运行脚本及实验记录。
- `logs/`、`output/`、`checkpoints.pth` 均未进入 Git。
- 服务器工作区仍有无关的 `train_c100.sh` 末尾换行变化，未纳入提交。

### 2026-06-24 创建 EMOTIC DDP tau2 分支

- 从 `feature/clip-taskCLS-posneg-text-head` 的 `01c10fc` 创建
  `feature/emotic-ddp-semantic-tau2`。
- 数据集确认存在于服务器 `./datasets/EMOTIC`，加载参数使用 `--data_path ./datasets`。
- 采用 Split-EMOTIC 默认 alphabetical B5-C3 协议：
  `base_classes=5,num_tasks=8`，共 26 个类别。
- `multi_lane/configs/emotic.py` 的 DDP semantic prompt 默认值改为情感专用前缀：
  positive `a photo of a person clearly feeling`，negative
  `a photo of a person not feeling`。
- 新增 `run_emotic_ddp_official_semantic_tau2_20ep.sh`，完整迁移 VOC 当前最佳
  paper-aligned DDP 20ep tau2 配置，输出独立 log/output/checkpoint/score dump，
  detail HTML/JSON 继续复用同一评估引擎。
- 服务器验证结果：
  - Python 静态编译和脚本 `bash -n` 通过；
  - parser 默认提示与脚本提示一致；
  - train/eval 人物样本数为 `16001/7765`，26 类任务切分为
    `[5,3,3,3,3,3,3,3]`；
  - DDP text/visual prompt shapes 为 `(26,2,16,512)` / `(26,2,16,768)`；
  - positive/negative semantic seeds 不相同；
  - 报告 smoke 成功生成 `.html` 和 `.json` 双文件。

### 2026-06-24 EMOTIC B5-C3 real-data smoke

- 用户确认在 GPU0 运行 1ep smoke。
- 完整配置保持正式 20ep recipe 不变，仅使用 `epochs=1`；B5-C3、batch8、full image、
  paper attention/loss/optimizer/scheduler/transform、semantic emotion prompts、
  PCD tau2/gamma0.7，并保留 checkpoint、score dump、HTML/JSON。
- 独立 tmux：`ddp_emotic_tau2_smoke_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke`
- 用户确认 GPU0 剩余显存足够后，取消等待并立即并行启动；未触碰
  `/mnt/haoyuan/workspace/CODE_DDP` 的现有实验。
- 启动后 GPU0 总占用约 `9.4/24.6GB`，本 smoke 进程约 `5.2GB`；
  task0 首个训练 batch 正常完成，无 OOM。

### 2026-06-24 EMOTIC B5-C3 30ep 正式配置

- 参考对照：
  `emotic_b5c3_alphabetical_valtest_clip_taskCLS_text_bias_lr001563_v1`
- 对齐项：Split-EMOTIC、26 类字母序、B5-C3、full image、val+test、seed0、
  224 输入、8 tasks、每 task 30 epochs、Adam、weight decay 0、相同 detail
  HTML/JSON 指标口径。
- batch 对齐采用 physical batch8、gradient accumulation32、effective batch256；
  避免 class-wise positive/negative visual branches 在 physical batch256 下 OOM。
- DDP 方法专属项保留当前最佳配置：drop_last、paper attention、
  paper summed BCE `*0.03`、continual Adam 配置 LR `0.0059`、paper multistep、
  semantic emotion prompts、PCD tau2/gamma0.7。
- 旧 taskCLS 对照的 batch256、cosine scheduler 和 LR `0.001563` 不直接迁移，
  其中 batch256 通过梯度累积等效实现；cosine scheduler 和 LR `0.001563` 不迁移，
  因为当前最佳结果来自 official DDP 优化协议。
- 修复 `train_one_epoch` 的 accumulation step 边界，并区分 mean loss 与
  official `paper_sum` 的梯度缩放；accumulation1 行为保持不变。
- 正式脚本：`run_emotic_ddp_official_semantic_tau2_30ep.sh`
- tmux：`ddp_emotic_tau2_30ep_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep`
- 首次 tmux 启动在 Python 前退出，launcher log 确认非交互 shell 中
  `/root/.bashrc` 提前 return，`conda` 未定义。EMOTIC 20ep/30ep 脚本已改为直接
  source `/opt/conda/etc/profile.d/conda.sh`，不影响训练参数。
- 第二次启动发现 tmux 继承的 `csc` 环境 deactivate hook 与 `set -u` 不兼容；
  已仅在 `conda activate multilane` 前后临时切换 `set +u/set -u`。
- 修复 shell 初始化后正式实验成功进入 task0 epoch1：
  - accumulation32、pin_mem false、effective optimizer LR `0.0059` 配置均已解析；
  - scheduler 初始化后的实际 LR 为 `0.000590`；
  - 正式实验自身显存峰值约 `4.3GB`；
  - GPU0 三个并行实验总占用约 `15.6/24.6GB`；
  - 未发现 OOM、Traceback、RuntimeError 或 loss NaN。

### 2026-06-25 EMOTIC DDP smoke/30ep 结果分析

两组均完整跑完 8 tasks，HTML/JSON、8 个 score dump 和服务器 checkpoint 均生成。

```text
1ep smoke: mAP/amAP/OF1/CF1 = 29.8272/36.6230/21.8038/9.5456
30ep run:  mAP/amAP/OF1/CF1 = 31.3195/37.8293/25.9253/10.1560
delta:                              +1.4923/+1.2063/+4.1215/+0.6104
```

正式 30ep 与历史对照：

```text
DDP semantic tau2: 31.3195/37.8293/25.9253/10.1560
taskCLS text bias: 31.1848/37.2170/57.2128/22.7483
CLIP patch concat: 32.8635/39.8831/47.0667/20.2515
```

关键判断：

- 30ep effective batch256 与 1ep batch8 的 optimizer update 数接近：
  每 task 约 `30*ceil(num_batches/32)` 对比 smoke 的 `num_batches`，因此两者结果接近
  是合理现象；30ep 的梯度统计更接近历史 batch256 对照。
- DDP 对 taskCLS-text-bias 的 final mAP 仅提高 `0.13`，amAP 提高 `0.61`；
  对普通 CLIP patch baseline 则低 `1.54/2.05`。排序收益不足。
- final threshold0.8 precision/recall/F1 为 `0.754/0.157/0.259`；
  threshold0.5 为 `0.432/0.646/0.518`。tau2 在 EMOTIC 上明显过保守。
- 即使使用 global oracle threshold，OF1 仍比 taskCLS-text-bias 低约 `5.45` 点，
  因此不能把差距全部归因于阈值。
- final raw margin：positive `0.0075`、negative `-0.0210`、gap `0.0285`；
  positive margin >0 rate `64.6%`，negative margin >0 rate `20.3%`。
  26 类中 17 类 positive margin mean <=0，说明 macro/class-wise 分离不足。
- 最弱 AP 包括：Embarrassment `2.12`、Sensitivity `4.47`、Fear `8.49`、
  Surprise `8.57`、Aversion `8.95`、Yearning `11.21`。
- 相对 taskCLS-text-bias，DDP 大幅改善 Affection `+19.84 AP`，并改善
  Confidence `+4.51`、Pleasure `+2.98`；主要下降为 Suffering `-9.08`、
  Disquietment `-6.15`、Doubt/Confusion `-4.89`、Sadness `-3.57`。
- 从类别刚学完到最终评估，早期类 AP 下降主要来自评估样本/负样本分布扩展，
  不是旧 prompt 参数更新：Anticipation `-26.20`、Affection `-13.38`、
  Annoyance `-8.98`。模型参数冻结不保证在新增负样本上的 AP 不变。
### 2026-07-01 EMOTIC test-only 评估入口

- 发现历史 `emotic_b5c3_alphabetical_valtest_clip_vit_b16_patch_v1` 报告使用
  `val+test`。该实验没有保存 checkpoint，因此无法只重跑测试，必须重训才能得到
  对应模型的纯 test 结果。
- 在 EMOTIC CLI 增加 `--emotic_eval_splits`，默认 `val test`，可显式设为 `test`。
- 修复数据构建入口未传递 `emotic_input_mode` 的问题，并把选择的 eval splits 传入
  EMOTIC dataset。
- 新增历史 CLIP patch concat baseline 的纯 test 30ep 脚本，保持 B5-C3、26 类字母序、
  batch256、seed0、full image 等训练配置不变，新增 checkpoint 保存。
- 预期产物为独立 log、checkpoint、class order JSON、逐类逐任务 HTML/JSON。
- 用户确认历史代码位于 `feature/clip-vit-b16`。已从 `0b59138` 建立独立 worktree
  `/mnt/haoyuan/workspace/multi-lane-main-test-only` 和分支
  `fix/emotic-test-only-clip-vit-b16`，没有切换或污染当前 DDP 工作区。
- 实测 eval 样本数为 val `2397`、test `5368`、val+test `7765`；test-only 仍为
  26 类字母序及 B5-C3 八任务。
- 纯 test 30ep 正式实验已在 GPU1、tmux `emotic_clip_b16_testonly_30ep` 启动。
  task0 前两轮训练指标与历史 val+test 日志逐项一致，验证训练协议未改变；GPU1
  峰值约 8.2GB，无 OOM。
- B5-C3 纯 test 已完成，final `29.9174/36.6938/45.5877/19.5187`，耗时 `36:01`。
- 旧 upper bound 配置核对为 joint training：`dataset=EMOTIC`、1 task、26 base classes、
  30 epochs、batch256、CLIP patch concat；旧实验 `store_model=False`，因此纯 test 必须
  重训。
- 新增 upper-bound test-only 脚本，在 GPU0/tmux `emotic_upper_testonly_gpu0` 启动。
  Namespace 已确认 `emotic_eval_splits=['test']`，首轮训练数值与旧 val+test upper
  bound 一致，预计约 14–16 分钟完成。

### 2026-08-11 本地/服务器独立 Git 工作树迁移

- 只读审计确认服务器主工作树原分支为 `fix/emotic-test-only-eval@b984688`；该分支没有
  upstream，但 HEAD 已存在于 `origin/feature/emotic-ddp-semantic-tau2`，没有未推送的
  普通提交。服务器原有三个 stash 均为仅存于服务器的 WIP。
- 审计确认额外 worktree `/mnt/haoyuan/workspace/multi-lane-main-test-only` 位于
  `fix/emotic-test-only-clip-vit-b16@0b59138`，包含两处代码修改、两个未跟踪运行脚本和
  指向主数据集目录的符号链接；迁移期间未修改该 worktree。
- 迁移前完整备份位于
  `/mnt/haoyuan/workspace/git-migration-backups/multi-lane-main-20260811-1145/`：包含经
  `git bundle verify` 验证的全引用 bundle、两个 worktree 的 binary patch、状态清单、
  stash 清单、未跟踪 PNG/脚本副本及 SHA-256 校验清单。大型数据、日志、output 和权重
  未复制或纳入 Git。
- 创建并推送安全分支 `backup/pre-local-git-migration-20260811-1145@b984688`，随后将
  服务器主工作树迁移前改动保存到命名 stash `7d0366b`，并正常切换到
  `feature/clip-vit-b16@0b59138`。
- 本地原目录已执行 `git init`，添加相同 origin 并 fetch；使用 mixed 基线识别出的修改
  与服务器备份 patch 及 PNG SHA-256 完全一致。本地迁移前改动保存到命名 stash
  `a9a4ef5`，`.idea/` 和 `.DS_Store` 写入本地 `.git/info/exclude`。
- 从 `feature/clip-vit-b16@0b59138` 创建并推送
  `fix/independent-git-worktrees`。本地和服务器主工作树均跟踪该远端分支，Git 元数据
  完全独立；PyCharm Automatic Upload 已关闭。
- 本次迁移未运行训练或模型测试；验证范围为 Git 引用、远端提交、stash/patch 备份、
  两端 HEAD/upstream、工作树状态和文件校验和。
- 用户已确认将 `.gitignore` 和四份迁移上下文文档提交并推送到
  `fix/independent-git-worktrees`，再由服务器使用 `git pull --ff-only` 同步。

## 2026-09-01：同步并分析layer1 ASL局部正式test

- 确认服务器已无本批runner，8个目录均包含config、task metrics、training history和
  complete seed summary；服务器实验worktree保持clean`94f3327`。
- 结果和日志已打包同步到本地，不含checkpoint；archive两端SHA-256均为
  `c8258d3215747c4dfd7abc30c35c004e6b9196ea230d394e1a8c2ae71f665691`。
- 锚点ASL 9.8/0/0.05 final/average mAP为`32.5365/39.1445`，且8个task mAP逐项第一。
  其余7组final mAP为`31.6761--32.1476`，没有新冠军。
- gamma-pos1在task3共跳过11次AMP optimizer update；其余7组skipped0。由于该组final
  mAP仍低锚点`0.3889`，不补FP32，也不扩大gamma-pos。
- 阶段二ASL局部搜索正式收口。阶段三/四尚未启动；它们需要新增按task bottleneck和
  Adapter专属weight decay入口，并在用户确认提交推送后才能按Git流程部署服务器。

## 2026-09-01：实现阶段三/四并行实验入口

- 用户确认运行阶段三和阶段四，创建`exp/emotic-image-token-stage34`实验分支。
- `adapter.py/model.py/runner.py/smoke.py`新增每task bottleneck、Adapter-only weight decay、
  逐task参数统计及GPU smoke参数；优化器仍保持selectors/prompts使用全局weight decay、head为0，
  Adapter单独使用新值。
- 修复潜在对照混杂：非统一task0容量构造后，显式推进到uniform b32参考RNG位置，使后续task
  Adapter初始化不受task0形状影响；增加对应相等性测试。
- 新增单组正式入口和8卡launcher。11组同时启动，GPU0--2各两组、GPU3--7各一组；launcher
  启动前要求8卡连续两次至少12GB空闲，全部无checkpoint，结果写外部benchmark目录，日志写`logs/`。
- shell语法、Python静态编译和`git diff --check`通过。本地系统Python缺少NumPy，完整单元测试
  留待服务器`ddp`环境在部署后执行。

## 2026-09-01：部署并启动阶段三/四共11组正式test

- 提交`8e56b8a Add taskwise Adapter stage 3 and 4 search`已推送到
  `origin/exp/emotic-image-token-stage34`。服务器先审计全部worktree，再创建独立clean worktree
  `/mnt/haoyuan/workspace/multi-lane-main-stage34`，未切换ASL、primary或test-only工作树。
- Automatic Upload把本次代码和脚本也写入服务器primary；已把3份文档和7个代码/脚本逐文件
  归档到`/mnt/haoyuan/workspace/git-sync-backup-stage34-deploy-docs-20260901-ooz0O8`，然后恢复primary
  clean。上传代码归档SHA-256为
  `0a843bfc3a5be08372c9006c6c8d72e3d155fef37ce95af6b7694a2033b38522`。
- 服务器`ddp`环境21项完整单元测试通过；uniform b32和task0-b24混合容量的真实CLIP
  AMP/TF32-on Adapter-ASL GPU smoke均通过，trainable参数分别为739130/726834。
- batch`image_token_asl_layer1_stage34_formal_seed0_20260901_203100`已在tmux
  `image_token_stage34_20260901_203100`启动。11组config全部生成且逐字段核验正确，均记录
  clean commit`8e56b8a`。GPU0--2双进程约4.2--4.6GB，GPU3--7单进程约2.1GB。
- 11组已到task0 epoch4--5，每epoch84 steps、skipped0，日志无OOM、non-finite、Traceback或
  RuntimeError。当前让实验继续运行，不持续盯跑；结束后只同步JSON与日志，不下载checkpoint。

## 2026-09-01：同步并分析阶段三/四结果

- 11/11组完成，launcher退出码均为0；每组240 epochs、13950 optimizer updates、skipped0，
  无OOM、non-finite、Traceback、RuntimeError或checkpoint目录。
- 小结果、manifest、状态与日志已同步本地；2.5MB解压目录中的archive两端SHA-256均为
  `bde0e9746985cb74751cc946332b8ada3e823401a7ea5bb1db732def30fcecfd`。
- 阶段三b24/b28在task0有正收益，但从task1开始整体转负，final mAP为`31.9938/32.0368`。
  因后续Adapter RNG已严格对齐锚点，结论指向selectors/prompts复制造成的持续路径差异。
- 阶段四冠军仍为LR0.0125/WD0的`32.5365`。LR0.015/WD1e-5达到`32.5023`，只低`0.0343`，
  同时获得本批最高average mAP`39.6697`和cF1`33.1460`；task0--6均高、task7仅低`0.0343`。
- 下一方向收缩到LR0.015/WD1e-5附近的窄网格；不继续task0小容量或全局weight decay扩展。

## 2026-09-01：准备统一LR×Adapter WD局部直接test

- 用户将下一阶段从validation-only改为直接held-out test，并接受其仅为exploratory结论。
- 创建`exp/emotic-image-token-global-lr-wd`，新增9组8卡launcher；GPU0承载锚点和一组候选，
  GPU1--7各一组。
- 候选实际主模型LR为`0.014/0.0145/0.015/0.0155`，对应source LR
  `0.056/0.058/0.060/0.062`；Adapter WD为`3e-6/1e-5`。所有task使用同一组合，保持未知未来
  增量数据设定，不使用按task手调参数。
- 固定seed0、8 tasks×30 epochs、batch64、Image-token layer1/b32、Adapter LR4e-4、
  scale0.1/ReLU、主模型BCE + Adapter ASL9.8/0/0.05、CLIP normalization、crop0.05、
  AMP/TF32 on、无checkpoint。
- launcher与上下文已提交推送为`4b45dd9`；服务器创建clean独立worktree并通过21项单元测试。
- Automatic Upload副本备份位于
  `/mnt/haoyuan/workspace/git-sync-backup-global-lr-wd-deploy-20260901-RsLHVP`，primary恢复clean。
- 8卡当前各被其他任务占用约19GB，未发现本项目runner。已在tmux
  `image_token_global_lr_wd_20260901_224909`挂起自动等待；batch ID为
  `image_token_asl_layer1_global_lr_wd_formal_seed0_20260901_224909`。每30秒检查一次，要求
  全部GPU连续两次至少12GB空闲后再启动，不停止或抢占现有进程。

## 2026-09-02：同步全局LR×Adapter WD首批8组并补跑资源失败组

- 9组launcher已经运行；其中8组完整产出`seed_summary.json`，每组240 epochs、13950 updates、
  skipped0。LR0.0155/WD1e-5原运行在task6开始时OOM，日志显示GPU0仅剩16.62MiB，属于外部
  GPU进程临时占用引发的资源碰撞，并非loss非有限或模型容量超限。
- GPU释放后，在GPU1以完全相同配置从task0重新补跑，run ID为
  `image_token_asl_layer1_global_lr_wd_formal_seed0_20260901_224909_lr0155_wd1e5_retry1`，tmux为
  `image_token_global_lr_wd_retry1_20260902`。启动后约占2.1GB，尚余约22GB，首4轮正常。
- 已同步8组完整JSON、原launcher日志和失败诊断日志到本地，不含checkpoint。压缩包两端
  SHA-256均为`4be9b24748df8dc32f3a4759b31157b1e44b2384b1e3c690f43b6372234356b4`。
- 暂时排名仍由LR0.0125/WD0锚点`32.5365`领先；LR0.015/WD1e-5为`32.5023`，差`0.0343`，
  但average mAP从`39.1445`升到`39.6697`，cF1从`32.5756`升到`33.1460`，oF1从
  `49.3827`升到`49.5592`。其余6个候选final mAP均不超过`32.3674`。
- 在三个已成对完成的LR点上，WD1e-5相对WD3e-6的final mAP分别提高`0.2662/0.0725/0.1348`，
  average mAP分别提高`0.2588/0.3571/0.5826`，说明较强的候选WD方向一致优于3e-6；但主模型
  LR收益在0.015附近才出现，非简单单调关系。

## 2026-09-02：完成全局LR×Adapter WD最终9组分析

- GPU1 clean retry完整结束，状态complete，240 epochs、13950 optimizer updates、skipped0；tmux与
  runner均已退出。run记录clean`4b45dd9`，配置与原失败组一致。
- LR0.0155/WD1e-5补跑final/average mAP为`31.8789/39.2960`，相对冠军final下降`0.6576`；
  task5/6/7分别下降`0.0223/0.5944/0.6576`，说明LR0.0155已越过有效区域上界。
- 最终冠军仍是LR0.0125/WD0的`32.5365`。LR0.015/WD1e-5以`32.5023`第二，虽提高average
  mAP和F1，但没有满足用户以final mAP为主的替换条件。
- WD1e-5相对3e-6在LR0.014/0.0145/0.015分别提高final mAP`0.2662/0.0725/0.1348`，到
  LR0.0155反而下降`0.1775`，确认LR与WD存在窄区间交互，停止继续微调二者。
- 最终小结果包已同步本地，archive两端SHA-256均为
  `724611f09c4df5501257b92238790cb4309074571828c692eec09862767a2fba`，不含checkpoint。
  最终分析写入`output/emotic_track_a_image_token_global_lr_wd_formal/20260901_224909/analysis.md`。

## 2026-09-02：实现固定预算、Adapter输出正则与可学习gate

- 从`4b45dd9`创建`exp/emotic-image-token-training-mechanisms`，保留已更新但尚未提交的三份上下文
  文档。固定gate/无正则/epoch预算保持旧训练数学路径，作为同批control复现`32.5365`。
- runner新增`--optimizer-updates-per-task`：每task按成功GradScaler/optimizer step精确停止，跳步不计
  入预算；DataLoader按需重新shuffle循环，cosine scheduler逐成功step推进。config记录预算模式、
  每task目标和scheduler语义，summary记录实际data cycles与总成功updates。
- Image-token Adapter新增两种可微输出度量：scaled residual/frozen token二范数比、adapted/frozen
  token cosine distance。正则只加入Adapter ASL分支，主模型selectors/prompts/head仍只接收BCE梯度。
  每task用最初30个成功updates进行无正则量级校准，之后固定权重，避免zero-init早期除以极小量造成
  梯度爆炸。
- 新增按task独立的sigmoid residual gate。gate作为对应task Adapter optimizer参数，激活新task时
  仅当前gate可训练；旧gate及旧Adapter冻结。固定模式不新增参数，learnable模式每task仅增加1参数。
- 新增固定预算精确停止、校准权重、辅助度量、gate初始化/冻结/参数统计单元测试；GPU smoke增加正则
  和gate入口。新增单组正式脚本与18组8卡launcher，外部结果目录不保存checkpoint。
- 本地`py_compile`、两个shell的`bash -n`、18组数组长度和`git diff --check`通过。本地完整测试因系统
  Python缺少Torch/NumPy无法执行，这是环境缺失而非测试失败；部署后必须在服务器`ddp`环境跑完整套件。
- 提交`2cafb04 Add Image-token Adapter training mechanism search`已推送。Automatic Upload写入primary的
  8个受控文件和2个未跟踪脚本已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-training-mechanisms-deploy-20260902-jsBiUR`并逐路径恢复；primary
  clean，test-only原有改动未触碰。
- 服务器从远端创建clean独立worktree`/mnt/haoyuan/workspace/multi-lane-main-training-mechanisms`，HEAD、
  upstream均为`exp/emotic-image-token-training-mechanisms@2cafb04`。
- 服务器`ddp`环境完整50项单元测试全部通过。真实ViT-B/16 AMP/TF32-on GPU smoke覆盖control、
  residual_ratio10%、feature_cosine10%和learnable gate0.1，均确认主模型/Adapter梯度隔离、冻结视觉塔、
  concat inference与有限梯度；固定模式trainable739130，gate模式739131。
- 正式18组batch ID预定为
  `image_token_asl_layer1_training_mechanisms_formal_seed0_20260902_150731`。启动前仍需确认8卡连续两次
  至少18GB空闲；launcher随后一次并行启动全部配置，不保存checkpoint。
- 验证记录follow-up提交`5505a36`已推送并在服务器实验worktree`git pull --ff-only`。Automatic
  Upload再次写入primary的三份文档已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-training-mechanisms-validation-docs-20260902-FMyt8a`并恢复clean。
- 8卡启动前均余23.6GB以上，launcher连续两次通过18GB门槛后在tmux
  `image_token_training_mechanisms_20260902_150731`原子启动18组。run目录与config均为18/18；GPU0/1
  各3组、其余每卡2组，峰值初始占用约6.7GB，无资源冲突。
- 18份config均记录clean`5505a36`、seed0/test/no-checkpoint、main LR0.0125、Adapter LR4e-4/WD0、
  layer1/b32、BCE+Adapter-ASL、AMP/TF32。预算分布为epochs11组/updates7组，updates值精确为
  `900--2700`；正则6组与gate4组字段、每task参数49952/49953均正确。
- control task0 cycle1--5与历史冠军loss、LR和Adapter loss逐项一致。六个正则组在30个成功updates
  后均生成非零固定权重；task0首cycle的1%/3%/10% residual权重为
  `15.7632/47.2897/157.6324`，cosine为`31.5044/94.5132/315.0440`，验证校准比例正确。
- 当前全部日志无OOM、non-finite、FloatingPointError、RuntimeError或Traceback，首轮skipped0。
  让实验继续运行，结束后只同步JSON、manifest、launcher status和日志，不下载checkpoint。

## 2026-09-02：同步并分析训练机制三阶段18组结果

- tmux与18个runner均已退出，batch status complete，18个launcher exit code全部为0；18份config和
  complete seed summary齐全。所有运行skipped0、无OOM/non-finite/Traceback/RuntimeError，且没有
  checkpoint目录。
- 7个固定预算组总成功updates分别精确为`7200/9600/12000/14400/16800/19200/21600`；control、
  6个正则组和4个gate组均为240 cycles、13950 updates，协议完整性通过。
- 阶段一control继续以`32.5365/39.1445`领先。固定updates最佳1800为
  `30.8757/38.2894`，final下降`1.6608`；其他固定预算下降`1.8004--2.1743`。
- 阶段二最佳final为residual1%的`32.0751`，下降`0.4614`；cosine最佳为10%的`32.0490`，
  下降`0.4875`。所有正则都压低task5--7和task6，没有稳定性换取final收益。
- 阶段三gate init0.05最接近，final/average为`32.3397/39.0774`，低control`0.1969/0.0671`；
  cF1提高`0.0271`、forgetting改善`0.0868`，但不符合用户final mAP主目标。最终8个gate为
  `0.0696/0.0720/0.0570/0.0860/0.0688/0.0689/0.0550/0.0633`。
- 三个机制各自最佳均为负收益，因此不补组合实验。保留30 epochs、fixed scale0.1、无输出正则、
  LR0.0125/Adapter WD0冠军；关闭固定updates、输出正则与learnable gate路线。
- 结果JSON、manifest、launcher status和日志已同步本地。860KB压缩包两端SHA-256均为
  `fc051087a60a293ae06824396c7981c8cbeaeb9f04a24d0cd1226981a81c003a`，不含checkpoint。

## 2026-09-02：实现8组统一epoch正式test搜索

- 从`exp/emotic-image-token-training-mechanisms@5505a36`创建
  `exp/emotic-image-token-epoch-search`，保留上一批已更新但未提交的三份上下文文档。
- 新增`run_multilane_track_a_image_token_epoch_search_formal.sh`，只把`--epochs`暴露为严格正整数，
  其余训练参数精确固定为当前single1 b32冠军；每组使用独立输出根和日志，不保存checkpoint。
- 新增`launch_multilane_track_a_image_token_epoch_search_formal_8gpu.sh`，8卡分别运行
  `18/22/26/30/34/38/42/48` epochs，启动前要求8卡连续两次至少18GB空闲，生成manifest与逐组
  exit code，并在全部成功后自动调用严格汇总器。
- 新增`multi_lane.track_a.summarize_image_token_epoch_search`，核验完整网格、clean Git、共同
  commit/tree、固定协议、task/epoch/update/skipped/checkpoint完整性；输出final/average/late-task/
  task6/F1/forgetting排名及相对30-epoch锚点变化，并按预声明规则给出refinement。
- 新增3项单元测试，覆盖内部34-epoch赢家应建议`31/32/33/35/36/37`、30-epoch赢家应停止搜索、
  以及batch size协议漂移必须拒绝。下一步先完成本地静态检查，再提交推送并在服务器ddp环境运行
  完整单元测试；通过后才启动8组正式实验。
- 实现提交`7449b30 Add Image-token Adapter epoch search`已推送。Automatic Upload写入primary的
  7个文件已逐项备份至`/mnt/haoyuan/workspace/git-sync-backup-epoch-search-deploy-20260902-BmmkzK`
  并恢复primary clean；test-only原有修改未触碰。
- 服务器新建独立clean worktree`/mnt/haoyuan/workspace/multi-lane-main-epoch-search`，HEAD/upstream均为
  `exp/emotic-image-token-epoch-search@7449b30`。ddp环境53项完整单元测试通过；GPU0真实CLIP
  Image-token layer1/b32、Adapter-ASL、AMP/TF32 smoke通过，trainable739130、冻结视觉塔无梯度、
  初始零残差max logit差`7.63e-05`。8卡当前均余23.6GB以上，可进入正式启动。
- 验证记录已提交推送为`7091da6`，服务器实验worktree使用`git pull --ff-only`同步并保持clean。
  Automatic Upload写入primary的三份文档已备份至
  `/mnt/haoyuan/workspace/git-sync-backup-epoch-search-validation-docs-20260902-afQHL2`并恢复；test-only未触碰。
- batch`image_token_asl_layer1_epoch_search_formal_seed0_20260902_171919`已在tmux
  `image_token_epoch_search_20260902_171919`启动。8卡连续两次至少23.6GB空闲后，一卡一组运行
  epoch18/22/26/30/34/38/42/48；8份config均为clean`7091da6`、seed0/test/batch64、冠军结构、
  AMP/TF32 on、no-checkpoint，除epochs外字段一致。
- 八组task0 cycle1均完成84 optimizer steps、skipped0，loss`0.61478711`、Adapter ASL
  `0.02839343`完全相同；每组GPU约占1.99--2.07GB，仍余约22GB。日志扫描无OOM/non-finite/
  FloatingPointError/RuntimeError/Traceback，当前继续运行且不启动其他batch。

## 2026-09-02：同步并分析统一epoch搜索

- tmux与runner均已结束；batch status complete、8份seed summary和8个exit code齐全，exit code全0。
  严格汇总确认所有run同一clean`7091da6`、task0--7完整、总epochs和updates与声明epoch精确对应、
  skipped0、无checkpoint；日志无OOM/non-finite/Traceback/RuntimeError。
- 30 epochs精确复现历史冠军final/average mAP`32.5365/39.1445`并排名第一。26 epochs第二，
  final`32.3681`低`0.1684`，average`39.1940`高`0.0495`；其提升集中于task0--4，task5--7
  分别低`0.0619/0.1539/0.1684`，不满足用户final mAP目标。
- 34/38/42/48 epochs final分别为`31.9376/31.8459/31.9648/31.5501`。34与48相对30在8个
  task全部退化；48的task6/final低`1.0287/0.9864`，而多数末轮训练loss更低，属于延长训练后的
  过拟合/泛化下降，不是优化不足。
- 按预声明规则，30保持第一即停止epoch搜索，不运行31--33或27--29。下一步固定30 epochs，进入
  对所有task统一的scheduler模式搜索；若仍无冠军，再转full image + person crop双视图。
- JSON、严格汇总、manifest、launcher status与原始日志已同步本地，压缩包两端SHA-256均为
  `460890c5b44eb2788f2254b333a3d3376e3df6a310cbc18431065537d2632f30`，不含checkpoint。

## 2026-09-02：实现8组统一scheduler搜索

- 从`exp/emotic-image-token-epoch-search@4e8616d`创建
  `exp/emotic-image-token-scheduler-search`，保留已同步epoch结果与三份待提交上下文更新。
- `runner.py`新增`cosine/linear/constant/multistep`模式、相对min LR、warmup比例、multistep比例/
  gamma入口。历史无warmup/min cosine继续走原生PyTorch路径；其他模式用共同relative multiplier
  同步缩放主模型LR0.0125和Adapter LR4e-4。
- warmup定义为30 epochs下前2/3 epochs从`1/W`线性达到基础LR，再进入余下周期的cosine-to-zero；
  multistep里程碑为ceil(30×0.6/0.85)=18/26，gamma0.1。config记录模式、比例、实际warmup epochs、
  milestone比例/epoch、gamma与step unit。
- 新增4项scheduler数学测试，覆盖relative min对所有parameter group、2/3 epoch warmup、linear/
  constant/multistep端点和非法组合拒绝；新增完整scheduler搜索汇总器3项测试，覆盖赢家选择、锚点
  收口以及逐epoch LR漂移拒绝。
- 新增正式单组runner与一组一卡8-GPU launcher；所有run固定30 epochs和single1 b32冠军参数，结束后
  自动严格核验240 epochs、13950 updates、skipped0、clean共同commit、no-checkpoint和四条LR轨迹。
- 本地Python编译、shell语法和diff检查已通过；完整Torch单测需提交推送后在服务器ddp环境执行，
  通过并完成8种代表性scheduler轨迹smoke后才启动正式实验。
- 实现提交`58eb233 Add Image-token Adapter scheduler search`已推送。Automatic Upload写入primary的
  9个受控文件已逐项备份至`/mnt/haoyuan/workspace/git-sync-backup-scheduler-search-deploy-20260902-H8IHWD`
  并恢复primary clean；epoch工作树clean、test-only历史修改未触碰。
- 服务器新建独立clean worktree`/mnt/haoyuan/workspace/multi-lane-main-scheduler-search`，HEAD/upstream
  均为`58eb233`。ddp环境60项完整单元测试全部通过。
- 8种30-epoch实际LR轨迹审计通过：anchor最终主/Adapter LR均为0；min1%最终为
  `1.25e-4/4e-6`，min10%为`1.25e-3/4e-5`；warmup5%首轮为基础LR的1/2，warmup10%前3轮
  为1/3、2/3、1；linear最终0，constant全程不变，multistep在18/26轮后为0.1/0.01倍。
- GPU0真实CLIP Image-token layer1/b32、Adapter-ASL、AMP/TF32 smoke通过，trainable739130、
  frozen视觉塔无梯度、初始零残差logit最大差`3.05e-05`。当前8卡均余23.6GB以上，可正式启动。
- 验证记录提交`7337f96`已推送，服务器scheduler工作树使用`git pull --ff-only`同步并保持clean。
  Automatic Upload写入primary的三份文档备份至
  `/mnt/haoyuan/workspace/git-sync-backup-scheduler-search-validation-docs-20260902-1jhTrQ`并恢复clean。
- batch`image_token_asl_layer1_scheduler_search_formal_seed0_20260902_185259`已在tmux
  `image_token_scheduler_search_20260902_185259`启动。8卡连续两次至少23.6GB空闲后，一卡一组
  原子启动；8份config均为clean`7337f96`，scheduler模式/min/warmup/warmup epochs/milestone/gamma
  字段逐组正确，其他协议一致。
- 非warmup六组task0 cycle1完全复现锚点loss`0.61478711`、Adapter loss`0.02839343`、LR0.0125；
  warmup5%/10%首轮LR分别为0.00625/0.00416667，loss相应为0.62027323/0.62566431。八组均84
  steps、skipped0；每卡约2.0GB，日志无OOM/non-finite/RuntimeError/Traceback，当前继续运行。

## 2026-09-03：同步并分析统一scheduler搜索

- tmux和runner均结束；batch status complete、8份seed summary和8个exit code齐全，exit code全0。
  严格汇总核验clean共同`7337f96`、每组240 epochs/13950 updates/skipped0、逐task逐epoch主模型/
  Adapter current/next LR轨迹和no-checkpoint；日志无OOM/non-finite/Traceback/RuntimeError。
- 历史cosine锚点精确复现final/average mAP`32.5365/39.1445`并保持第一。multistep第二为
  `32.3170/39.5518`，final低`0.2196`但average高`0.4074`；linear第三final`32.2498`。
- multistep相对anchor在task0--5提高`1.0186/0.8246/0.5947/0.4725/0.3237/0.3190`，task6/7
  转为`-0.0747/-0.2196`。同一epoch18/26里程碑让大task获得大量高LR steps，而task6只获得约
  180次高LR更新，解释其后期回落；final Suffering AP低`9.245`是主要类别交换之一。
- cosine warmup5%/10%、min10%/1%、constant分别低anchor
  `0.5607/0.9827/0.5662/0.6372/0.8466`。constant训练loss更低但test更差，尾部高LR造成过拟合；
  warmup压低zero-init Adapter早期更新且剩余30-epoch预算不能补偿。
- 原cosine继续作为冠军scheduler。按预声明规则停止单视图scheduler搜索，下一阶段转full image +
  person crop双视图融合，不再继续微调min LR、warmup或multistep。
- JSON、严格汇总、manifest、launcher status和日志已同步本地；小包两端SHA-256均为
  `d9d78f47ebfe5a74d1866a1d8e795043da070fb55a3940b16637803ea88e90c9`，无checkpoint。

## 2026-09-03：实现layer1冠军的person-only正式对照

- 从`d7e9434`创建`exp/emotic-person-only-layer1`，保留此前尚未提交的scheduler结果文档。
- EMOTIC person crop新增可配置bbox margin，按bbox宽高分别向四边扩展并裁到图像边界；默认0保证
  历史命令不变。本轮固定margin0.15，避免紧框丢失少量邻近上下文。
- runner新增`--person-crop-margin`合法性检查、数据集透传和config记录；新增3项单测覆盖零margin、
  扩展后边界裁切和无效bbox回退。
- 新增`run_multilane_track_a_person_only_formal.sh`：除输入为person crop、margin0.15、训练crop
  scale0.70--1.0外，严格固定当前32.5365冠军训练协议；seed0完整8-task held-out test，关闭checkpoint。
- 本地`git diff --check`、shell语法和Python静态编译通过；macOS系统Python缺少Pillow，完整单测需在
  服务器ddp环境运行。
- 用户确认后提交并推送`d0f8e81`。Automatic Upload同时写入服务器主工作树7个本次文件；逐文件
  SHA-256与本地一致后备份到`/mnt/haoyuan/workspace/git-sync-backup-person-only-IZ5eEXJ0`并恢复
  主工作树clean，没有触碰test-only工作树。
- 新建服务器独立worktree`/mnt/haoyuan/workspace/multi-lane-main-person-only`，HEAD为clean
  `d0f8e81`。服务器ddp环境完整63项单测全部通过；真实数据检查得到train/val/test人物样本数
  `16001/2397/5368`、输入`3x224x224`且有限。
- GPU0真实CLIP Image-token layer1/b32、Adapter-ASL、AMP/TF32 smoke通过：trainable739130，
  max initial difference `6.1035e-05`，冻结视觉塔、梯度路由和concat推理正常。
- 正式run`person_only_layer1_b32_asl_formal_seed0_20260903_121738`在tmux
  `multilane_person_only_s0_20260903_121738`的GPU0启动。输出位于
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_person_only_formal_v0.1/`；config记录clean
  `d0f8e81`及全部预声明参数。前4个epoch每轮84 steps、skipped0，GPU约2.4GB，无异常，继续运行。

## 2026-09-03：同步并分析person-only正式结果

- tmux与runner已结束，外部exit code为0；config/task_metrics/training_history/seed_summary和日志齐全，
  240 epochs、13,950 updates、全部skipped0，日志无OOM/non-finite/Traceback/RuntimeError，无checkpoint。
- 小结果已同步到`output/emotic_track_a_person_only_formal/20260903_121738/`并逐文件SHA-256与服务器一致。
- person-only final/average mAP为`28.7815/35.2444`，比full冠军低`3.7550/3.9001`；final cF1/oF1
  低`1.6732/2.3651`。forgetting仅改善0.0380，但因整条曲线更低，不构成有效收益。
- task0--7 mAP分别低`5.3958/4.6038/3.3542/3.2191/3.4181/3.3758/4.0789/3.7550`。
  最终26类AP无一超过full；task6新类Sadness/Sensitivity/Suffering在引入时分别低
  `14.5821/1.1381/11.5128`，当前crop没有修复task6。
- test数据包含5,368人物样本、3,682唯一图；2,923样本来自1,237张多人物图，其中1,185张具有
  不同人物target。人物身份歧义存在，但bbox绝对长宽比中位数1.561、54.14%>1.5、27.96%>2，
  当前Resize+CenterCrop和方形RandomResizedCrop会严重截断细长人体，是person view的重要混杂因素。
- 结论：拒绝当前person-only替代full；如继续双视图，先做body-preserving letterbox并保存逐样本
  validation scores，同批full锚点与person校正分支只在validation离线融合，超过full才进入test。
- 详细报告：`output/emotic_track_a_person_only_formal/20260903_121738/analysis.md`。
- 32KB小结果包不含checkpoint，SHA-256为
  `15e79e5f9dc3f33500b09d65e115b96e21c85608fd420ba16547d611cc150f6a`。

## 2026-09-03：实现body-preserving双视图validation融合诊断

- 从person-only结果分支创建`exp/emotic-dual-view-validation-fusion`，保留尚未提交的person结果文档。
- EMOTIC为每个标注人物新增与视图无关的稳定sample ID；LabelView可仅在报告阶段携带ID，训练接口仍
  保持`(image,target)`，因此不会改变历史训练路径或DataLoader随机轨迹。
- runner新增`PadToSquare`与显式person transform模式。`legacy_crop`默认保持旧行为；`letterbox`在
  bbox+margin后用CLIP均值padding成正方形再bicubic resize到224，训练只做水平翻转与低强度、低概率
  ColorJitter，不包含RandomResizedCrop，验证完全确定性且保留完整人体。
- 新增`--save-evaluation-scores`，硬性限定`reporting_split=val`；每个task保存压缩NPZ中的稳定ID、
  logits和targets，不导出test scores。离线融合器验证两run公共协议、ID集合、targets、finite数值和
  anchor指标重算，然后搜索logit/probability两种全局融合的alpha 0--1（步长0.05）。
- 新增两卡并行启动脚本，固定GPU0 full anchor、GPU1 corrected person，均为冠军训练协议的seed0完整
  8-task validation-only、无checkpoint；两run成功后自动产生`fusion_summary.json`和是否进入test决策。
- 新增测试覆盖sample ID、LabelView、letterbox像素保留与CLIP fill、无RandomResizedCrop、score dump
  ID重排、融合端点及完整8-task anchor重算。Python静态编译、shell语法和diff检查已通过；完整测试需
  服务器ddp环境。代码未提交，实验未运行。
- Automatic Upload把3份文档、2个源码文件和4个新增文件写入服务器主工作树。9个SHA-256与本地
  完全一致后备份到`/mnt/haoyuan/workspace/git-sync-backup-dual-view-code-8ULK5BSO`；新增文件使用
  可恢复移动、已跟踪文件从HEAD恢复，服务器主工作树重新clean。
- 用户确认提交并推送当前双视图分支，要求服务器创建独立worktree并依次运行完整单测、真实数据/
  score-dump smoke和Adapter GPU smoke；全部通过后在GPU0/1启动full/person两组完整validation并
  自动融合。用户同时授权以后对其已要求启动的实验不再重复审批；授权边界已写入`AGENTS.md`。
- 实现与上下文以`6ab53c8 Add dual-view validation fusion diagnostics`提交并推送。Automatic Upload
  写入服务器主工作树的4份文档已限定范围备份到
  `/mnt/haoyuan/workspace/git-sync-backup-dual-view-authorization-X0hW0zB0`并恢复clean，未触碰test-only。
- 新建clean服务器worktree`/mnt/haoyuan/workspace/multi-lane-main-dual-view@6ab53c8`。ddp环境完整
  70项单元测试通过；真实EMOTIC val smoke确认full/person共2,397个完全对齐且唯一的sample ID，
  两视图张量均为3x224x224且有限，NPZ score dump写入、ID重排和读回通过。
- GPU0真实ViT-B/16 Image-token layer1/b32、fixed scale0.1、independent、main BCE + Adapter ASL、
  AMP/TF32 smoke通过，trainable参数739,130、梯度有限。
- batch`image_token_layer1_full_person_letterbox_val_seed0_20260903_134720`已在tmux
  `multilane_dual_view_val_20260903_134720`启动。GPU0跑full anchor、GPU1跑person letterbox；两份
  clean config逐字段核验通过。task0第1轮均84 steps/skipped0，每卡显存约2.4GB/2.0GB，错误扫描为空。
  launcher将在两组完整8-task validation结束后自动执行alpha步长0.05的logit/probability融合。
- batch现已完整结束：两组exit0、各240 epochs/13,950 updates/skipped0、8个task和8份score dump齐全，
  日志错误扫描为空且没有checkpoint。自动融合成功，probability alpha0.20以final mAP`43.3035`
  超过full anchor `42.3799`共`0.9236`；average mAP提高`1.1631`，cF1/oF1提高`0.0599/0.2464`。
- person独立final mAP仅`38.3044`，但小权重融合使8个task全部提高、最终21/26类提高；Sadness与
  Suffering提高`3.3176/2.9992`，Sensitivity下降`1.0548`。probability alpha0.15--0.35均超过full，
  表明互补收益有稳定区间。
- 33个JSON/NPZ/状态/日志文件已同步到
  `output/emotic_track_a_dual_view_val/20260903_134720/`，共约2.65MB。runs/control/logs聚合SHA-256
  与服务器逐组一致，详细分析见该目录`analysis.md`。1.6MB小结果包
  `output/emotic_track_a_dual_view_val/dual_view_val_seed0_20260903_134720_small_results.tar.gz`
  不含checkpoint，SHA-256为`c988359a9b572c0799f3b1953beadfc7ccfabafae085d126dab19561748c2d0c`。
  下一步只做锁定alpha0.20的正式test，不在test调参。

## 2026-09-03：实现validation锁定的双视图正式test

- 从`5dce5ca`创建`exp/emotic-dual-view-formal-test`。runner将score export用途显式区分为
  `validation_search`与`fixed_test_fusion`；test score只有声明后者时才允许，并写入`test_scores/`。
- 新增固定test融合器：强制读取原validation选择记录并核验winner为probability alpha0.20且已批准
  进入正式test；命令行不暴露alpha、mode或threshold，只运行`0.8*full_prob+0.2*person_prob`一次。
- 新增两卡正式脚本：GPU0 full、GPU1 person letterbox；固定seed0完整8-task、30 epochs/task、
  batch64、main LR0.0125、per-task cosine/no warmup、Image-token layer1/b32/LR4e-4、scale0.1/
  ReLU/independent、main BCE + Adapter ASL9.8/0/0.05、CLIP normalization、AMP/TF32、无checkpoint。
- 新增测试覆盖固定融合只产生一个结果且不含candidate网格，以及validation alpha漂移时拒绝正式融合。
  本地Python静态编译、shell语法与diff检查通过；下一步提交推送并在服务器独立worktree完成全量测试。
- 实现以`c9fa74c Add locked dual-view formal test`提交并推送。Automatic Upload写入服务器primary的
  3份文档和4个新增文件已备份到`/mnt/haoyuan/workspace/git-sync-backup-dual-view-formal-18b3EdUp`
  并恢复primary clean，未触碰test-only。
- 新建服务器clean工作树`/mnt/haoyuan/workspace/multi-lane-main-dual-view-formal-test@c9fa74c`；
  ddp环境72项完整单测通过。固定融合CLI无alpha/mode/threshold选项，原validation summary SHA-256
  `a03ff4da...b90f`核验通过；GPU2真实Image-token Adapter-ASL AMP/TF32 smoke通过。
- 正式batch`image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816`已在tmux
  `multilane_dual_view_formal_20260903_145816`启动安全等待。当时GPU0/1有外部约8GB任务，launcher
  达到18GB空闲/低利用率连续门槛后才启动；后续已完整完成并同步，见本文顶部2026-09-03结果记录。

## 2026-09-03：启动Image-token layer1冠军seed1/2确认

- 先审计本地历史汇总与服务器165份已有config。MULTI-LANE注册baseline的seed0/1/2已于
  `multi_lane_main_track_a_seed012_20260811_132448`完整完成并同步本地，直接复用；冠军配置仅有
  seed0，没有任何完整或部分匹配的seed1/2历史run。
- 为保证与`32.5365` seed0一致，没有从当前person-only代码启动，而是使用服务器clean scheduler
  工作树`/mnt/haoyuan/workspace/multi-lane-main-scheduler-search@7337f96`直接运行两个缺失seed。
  权重SHA-256、数据入口、完整8-task test协议、训练参数、Adapter结构/loss、AMP/TF32及
  no-checkpoint均与seed0 anchor一致，仅seed和GPU不同。
- batch`image_token_asl_layer1_formal_seed12_20260903_123014`已在GPU1/2并行启动；结果位于
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_layer1_seed_confirmation_v0.1/`
  对应batch目录，日志位于scheduler工作树的
  `logs/emotic_track_a_image_token_seed_confirmation/`对应batch目录。
- 启动检查通过：两份config均为clean commit`7337f96`，seed、30 epochs、main LR0.0125、原cosine、
  layer1/b32/Adapter LR4e-4/scale0.1、BCE+ASL9.8/0/0.05、full/CLIP/crop0.05、AMP/TF32、
  8 tasks/test/no-checkpoint逐项正确；task0前三轮均84 steps、skipped0，GPU显存充足且无数值异常。

## 2026-09-03：同步并分析Image-token冠军三种子结果

- seed1/2 tmux均正常退出，exit code0；严格核验每组240 epochs、13,950 updates、skipped0、
  task0--7完整、clean commit`7337f96`、配置除seed外一致、无checkpoint，日志错误扫描为空。
- seed1/2小结果与日志已同步本地，结果树和日志的组合SHA-256分别与服务器一致：
  `f1bffb2fc59b16d73a833c4964db6d872989eaaf298e48d30b83e700e5fe21de`和
  `989c63be18bdc3d81d30f822afefdf3ff1cb0494c10557b35234ea7fed6c2636`。
- 复用scheduler cosine anchor的seed0后生成正式三种子汇总：final mAP
  `32.1562 ± 0.3701`，比注册baseline `31.2995 ± 0.1410`高`0.8567`；三个同seed差值
  `+1.1744/+0.7363/+0.6593`全部为正。average mAP、cF1和oF1也分别提高
  `0.9274/0.5867/0.2949`，forgetting基本持平并平均降低`0.0164`。
- task0--7平均mAP相对baseline全部提高，依次为
  `1.0756/1.3327/0.7886/0.7894/0.9536/0.8432/0.7793/0.8567`。Suffering仍有
  `-2.7971`的平均类别交换且seed敏感，但不改变整体三种子收益结论。
- 本地`analysis.md`和`formal_seed_summary.json`已生成。144KB下载包
  `image_token_asl_layer1_seed012_small_results_20260903_123014.tar.gz`不含checkpoint，SHA-256为
  `480838060a634d242776a38f698f027cbf1560527e32a24b573f3a9fbe48a330`。

## 2026-09-04：实现learned reliability gate（进行中）

- 在`exp/emotic-constrained-gated-fusion`上开始实现下一阶段，尚未提交或启动正式实验。
- `runner.py`新增按图像组SHA-256确定性拆分fit/calibration、calibration逐样本概率保存，以及省略
  `visual_encoder.*`的per-task compact checkpoint。默认fraction0保持旧训练入口兼容。
- 新增`learned_reliability_gate.py`：10维样本特征、共享两层MLP、单一有界人物权重、锚点正则、
  8组正则/hidden组合、三种子validation硬门槛和锁定单次test评估。
- 新增`export_compact_test_scores.py`，强制校验validation selection、source run、checkpoint哈希/
  provenance和clean Git，仅恢复compact method state并输出test概率，不执行训练或test搜索。
- 新增6卡launcher和source/export脚本：先并行3 seeds×2 views validation source，离线选门控；无
  合格赢家则停止，有赢家才并行导出6组test端点并评估一次锁定门控。
- 新增回归测试覆盖图像组切分、Full/Person一致性、compact state完整性、MLP初始0.20/输出边界、
  特征和validation lock。macOS系统Python缺numpy，仅完成py_compile/bash语法/diff静态检查；完整
  单测和真实数据/GPU smoke需提交后在服务器ddp环境执行。
- 代码以`2e352e0`推送；Automatic Upload的10个同哈希文件已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-constrained-gated-fusion-eyRcCTx5/learned-reliability-gate`，
  primary恢复clean，实验worktree ff-only到同一commit，test-only未触碰。
- 服务器ddp环境97项单测全通过。真实EMOTIC task0--7 calibration比例为8.13%--10.11%，Full/Person
  索引完全一致且没有图像组跨分区；真实两样本score dump为stored probability schema v2。
- GPU0真实ViT-B/16 layer1/b32/Adapter-ASL AMP/TF32 smoke通过，trainable739,130、初始残差误差
  `3.8147e-05`、无冻结塔梯度；compact恢复覆盖41个method tensors/1,089,002参数并排除87,281,642
  总状态中的冻结视觉塔。8-task顺序门控合成smoke也通过。
- 验证记录提交`65d3ca4`已推送，服务器实验worktree ff-only同步后启动唯一batch
  `learned_reliability_gate_seed012_20260904_021636`，tmux为
  `multilane_learned_gate_20260904_021636`。GPU0--5连续两次至少21GB空闲后同时启动6个source run。
- 6份config全部记录clean`65d3ca4`，seed0/1/2×full/person_crop无缺失；图像组calibration0.10、
  validation_search、compact state开启、完整checkpoint关闭及冠军模型参数均正确。task0 cycle1全部
  76 steps、skipped0，显存约2.0--2.5GB，错误扫描为空。当前继续运行，不启动重复batch。

## 2026-09-04：同步learned gate基础结果并诊断中断

- 六个source run全部complete，240 epochs/12,660 updates/skipped0，val/calibration概率重算和两视图
  ID/target对齐通过。197个原始文件（sources180/control9/logs8）同步本地，逐文件SHA-256一致。
- 门控select入口只载入train geometry，首个候选h8_prior0p1在seed0/task0的val特征提取时报Missing
  geometry；只有一份task0门控状态，未形成完整候选或selection，未触发test。这是前次实现的集成缺陷。
- 本次只复算固定Full0.8/Person0.2，不搜索新权重。final validation均值Full42.3777±0.8938、
  Person37.6004±0.2742、固定融合43.1073±0.8320；配对平均提高0.7296，8个task平均增益均正。
- task6 calibration仅51样本，三类正例22/24/17；应保留强正则，补跑后再判断门控是否学到可靠性。
- 报告及原始产物位于`output/emotic_track_a_learned_reliability_gate/20260904_021636/`。
  本次未修业务代码或启动补跑；下一步建议分离train/val geometry、增加不同split ID的集成测试，
  复用全部source scores/compact states，仅补原定门控阶段，保留失败输出，不重训六组模型。

## 2026-09-04：修复learned gate train/val metadata路由

- 用户授权后创建`codex/fix-learned-gate-split-routing`，保留并一并纳入上轮结果分析文档。
- calibration/validation geometry分开传递；选择前验证train/val sample ID前缀与覆盖，不读取test几何。
- 新增回归：不同split ID的8-task选择、缺val元数据提前失败、只加载train/val、改val标签不改变任一
  task gate state、原参数网格不变。补充每task两split权重均值/std/分位数和current BCE/mAP对照诊断。
- 新增resume脚本只复用20260904_021636六组已完成source runs。hidden8/16、prior0.1/0.3/1/3、
  80 epochs/task、batch64、AdamW1e-3/WD1e-4、范围0.10--0.35及0.20先验、三seed选优规则均不变。
- 本地静态编译/shell语法/diff检查通过；下一步Git-only新worktree验证与补跑。
- 实现146145b已推送，新建独立worktree`/mnt/haoyuan/workspace/multi-lane-main-learned-gate-routing-fix`。
  首轮测试仅新合成夹具遗漏source config导致失败；补全提交2ee06fb后102项完整测试通过。
- 真实train16,001/val2,397 geometry与三seed全部scores ID覆盖通过，未读test；GPU smoke通过。
  对原保存的seed0 Full/Person task7 compact states恢复，val首批64样本的概率逐元素完全等于原dump。
- batch`learned_gate_split_fix_20260904_113753`在tmux`multilane_gate_resume_20260904_113753`完成。
  仅训练8种原定门控、每种3seed/8task，六个source不重训且180原文件哈希完全不变。
- 8候选均不合格，均值最高h16/prior1.0 final val mAP43.106145±0.840327，低固定对照
  43.107346±0.831960约0.001201；所有候选均值下降0.0012--0.0116。程序正常test_skipped，无test导出。
- task6共24候选/seed均改善calibration BCE，但15组validation BCE恶化、17组三类validation mAP下降；
  seed1/2所有候选的三类val mAP均下降。强prior3的6组task6里5组std<0.001，近似固定0.20。
- task7均值最高候选新3类AP平均提高0.0064、旧23类下降0.0022，提示共享gate当前类监督与旧类排序
  的耦合；只作机制诊断，不改选优规则，不宣称极小差值具有统计显著性。
- control196/logs2/preflight4共202文件同步本地并逐文件SHA匹配，192个gate states与selection记录
  哈希一致。完整报告/审计JSON位于`output/emotic_track_a_learned_reliability_gate/20260904_113753/`。
# 2026-09-09：实现固定R1三种子正式test

- 新建`codex/fixed-three-view-seed12-validation`，按用户最新要求跳过seed1/2 validation，直接准备
  seed0/1/2一次锁定正式test。
- Face manifest入口向后兼容地增加test split；provenance仅在审计文件声明test时强制验证并哈希
  test manifest。Face runner允许完整train的test score source，仍禁止test calibration。
- 新增固定融合器：验证既有validation selection的固定SHA，要求恰好seed0/1/2，逐seed复算专家
  metrics，可靠Face应用`0.64/0.16/0.20`，不可靠Face应用`0.80/0.20/0`，汇总mean/std与配对差。
- 新增test manifest缓存复用脚本、单Face正式test入口和三GPU安全launcher。launcher没有权重参数，
  三个Face source全部成功后才执行一次固定融合。
- 本地Python语法、shell语法和diff检查通过；macOS系统Python缺Pillow/numpy，完整测试将在服务器
  ddp环境执行。
- 服务器首轮150项完整单测有149项通过；唯一失败是新增测试对float32归一化和采用了过严的7位
  十进制断言，实际为`0.99999994`。生产融合逻辑无误，测试容差修正为6位后重新执行全量测试。
- 修正后150项全部通过。独立manifest复用train/val缓存并完成test 3,682图/5,368人物：有效Face
  4,677，最终可靠Face 3,261，partial=false、重复分配0；provenance含test manifest SHA。
- 真实test dataset对齐、有效/无效Face transform、4样本CLIP GPU、layer1/b32 Adapter-ASL GPU及
  task0两次optimizer update的test score dump均通过；score为正确的task0 `3142×5`，skipped0。
- 正式batch `fixed_three_view_seed012_test_20260909_213052`在GPU0/1/2启动。三份config均为clean
  `acf4634`、test/fixed_test_fusion、30 epochs×8 tasks、无checkpoint、test manifest SHA一致；
  task0前3轮每轮64 steps、skipped0，显存约2GB/卡，无OOM或非有限值。
- batch已完成：三组均240 epochs/10,980 updates/skipped0，退出码0；固定融合只执行一条锁定规则。
  三视图final mAP `33.0119 ± 0.3038`，相对Full+Person三个seed均提高，平均`+0.1856`；average
  mAP/cF1/oF1也分别提高`+0.3414/+0.0747/+0.1351`，forgetting增加`0.0144`。
- Face sources/control/logs共46文件约9.4MB已同步本地，三部分组合SHA与服务器一致；无checkpoint。
  task0--7的三种子平均增益全部为正。下一步优先用既有validation产物核查并修正taskwise Router
  只作用于自身类别lane的增量语义，再决定是否进入端到端三视图特征融合。
- 创建`codex/taskwise-three-view-router`开始上述修正。新增task-lane重组评估器、test冻结CLIP描述符
  导出器和单GPU安全launcher；旧task lane权重变化不能影响其他task类别，并新增对应回归测试。
- 本批不训练专家或Router。validation重新评估已有R2/R3各3个prior；按用户要求，validation选定的
  R1/R2/R3还会在已有seed0/1/2 test端点上同时做探索性评估，明确禁止test参与选择。
- `9ab41a7`已Git-only同步服务器新worktree；152项完整单测、真实Router state及task0等价性smoke
  通过。批次`taskwise_three_view_router_20260909_233435`完成，test描述符5,368样本/3,261可靠Face。
- validation R2由42.6804恢复到42.8372并略高R1 42.7974；R3由42.6322恢复到42.7639但仍低于
  两者，未通过既定条件。test三种子R1/R2/R3为33.0119/33.0029/32.9758，R1保持正式冠军。
- 9个结果文件约460KB已同步，result/control/logs组合SHA与服务器一致。结论是语义修正有效，但
  calibration小样本和低维全局描述仍限制泛化；结束当前Router路线，转端到端特征融合。

## 2026-09-10：开始端到端taskwise多视图融合

- 从`d439905`新建`codex/end-to-end-taskwise-three-view-fusion`。新增taskwise特征融合模块：J0固定
  三路、J1动态三路、J3动态Full+Person；soft Router每task独立hidden16并从锁定固定权重初始化。
- EMOTIC新增显式multi-view数据通路；Full/Person/Face共享flip，Person/Face完整letterbox，Face
  reliability同时控制输入占位、融合mask和辅助损失mask。
- runner新增特征融合参数组、BCE/ASL一致的0.1分支辅助监督、融合权重日志、协议配置和seed0
  validation汇总器；新增三GPU launcher与回归测试。当前仅完成本地语法/shell/diff检查，服务器
  完整测试和smoke尚待Git-only同步后执行。
- 首次服务器159项全测通过；真实三方案task0 smoke暴露不可靠Face占位图错误调用实例`image.new`，
  J0/J1在首批数据前安全停止，J3完成84 updates/skipped0。修复为`PIL.Image.new`并补不可靠Face
  变换回归后，需要重新完整全测与三方案smoke；未启动正式validation。
- 修复提交`a9c897e`后二次159项全测全通过。真实批次
  `smoke_end_to_end_view_fusion_20260910_002`中J0/J1/J3均完成task0一轮84 updates、skipped0，
  batch64无OOM；耗时35.3/34.2/28.3秒。三份config均clean且记录同一提交，错误扫描为空。
- smoke记录提交`9f17799`后服务器ff-only同步。GPU0/1/2空闲均超过24GB时启动唯一完整validation
  batch`end_to_end_view_fusion_seed0_20260910_005421`；三组config均为seed0/val-only、30epochs×8task、
  checkpoint关闭，且训练前2轮稳定、skipped0、无OOM/Traceback/非有限值。

## 2026-09-10：同步并分析端到端taskwise多视图融合结果

- J0/J1/J3均正常退出并完成240 epochs、13,950 optimizer updates、skipped0；26份result/control/log
  原始文件已同步本地并与服务器三部分SHA-256一致，无checkpoint、OOM、非有限loss或运行异常。
- final/average validation mAP：J0=`42.5330/49.7360`，J1=`39.8569/46.5638`，
  J3=`38.3318/44.8969`。J1相对J0为`-2.6762`，相对J3为`+1.5251`；按预设规则拒绝晋级，
  seed1/2和test均未运行。
- J1和J3从task0起所有task都低于J0。J1最后各task的Person权重达到0.537--0.966，J3达到
  0.793--0.991；无约束竞争式softmax把Person当作当前类训练捷径并压制Full上下文，是主要失败机制。
- 独立训练的固定三专家seed0 validation为`43.5812`，共享任务参数的J0低`1.0481`，进一步提示
  多视图共用selectors/prompts/head/Adapter存在梯度冲突和专门化损失。结果报告写入
  `output/emotic_track_a_end_to_end_view_fusion/20260910_005421/analysis.md`。
- 下一阶段建议改成Full-anchored complementary residual：Full系数固定1，Person/Face经各自taskwise
  零初始化轻量投影以小有界残差加入；先只做A0 Full、A1 Full+Person、A2 Full+Person+Face三组
  seed0完整validation。当前不修改代码、不启动新实验。

## 2026-09-10：实现Full-anchored complementary residual

- 从`6b7a850`新建`codex/full-anchored-complementary-residual`。扩展TaskwiseViewFusion加入
  `residual_full_person/residual_three_view`，每task/每辅助视图各有独立bottleneck16投影与标量gate。
- Full权重固定1；投影末层零初始化，残差向量经`x/(1+||x||)`限幅，系数为
  `0.1*sigmoid(gate)`。无效Face系数为0；新task激活时只有当前task残差参数可训练，旧task冻结。
- 模型先关闭Full Image-token Adapter并以no-grad生成Person/Face源特征，再恢复Adapter并正常生成Full，
  避免辅助视图更新或复用共享Full Adapter；
  A1/A2关闭branch auxiliary loss。新增初始化等价、残差限幅、Face mask、stop-gradient、task冻结与
  严格validation汇总规则测试。
- 新增A0/A1/A2单组runner、三GPU launcher、汇总器及协议文档。macOS系统Python缺torch，已完成
  Python静态编译和shell语法检查；完整测试、真实数据与GPU smoke待服务器ddp环境执行。
- 实现`93433e2`在服务器独立worktree的163项全测通过。首轮真实smoke中A0完成，A1/A2暴露AMP
  cast cache与no-grad辅助前向交互，Adapter-ASL在首步报告参数未参与图；两组安全停止且未启动full。
- 修复提交`f0c3a82`令Person/Face不再运行Full Image-token Adapter，之后恢复Adapter再计算Full；
  164项全测全通过。真实A0/A1/A2 task0 smoke各完成1 epoch/84 updates/skipped0，无OOM或错误，
  val mAP诊断值`40.7614/41.1825/41.4034`。该一轮值不参与正式选择，只证明三条训练路径有效。
- smoke记录提交`f98f74c`已推送并Git-only同步。GPU0/1/2连续空闲后启动唯一完整batch
  `full_anchored_residual_seed0_20260910_103042`；三组config的commit/tree、30 epochs、冠军Adapter、
  BCE+Adapter-ASL、residual scale0.1、aux loss0和val-only均核验一致。前三轮范围内skipped0、无错误。

## 2026-09-10：同步并分析Full-anchored residual结果

- A0/A1/A2及自动汇总均exit0；每组240 epochs、13,950 updates、skipped0，无OOM/非有限值/异常。
  result/control/logs共27个原始文件约1MB已同步，三部分本地/服务器组合SHA-256完全一致，无checkpoint。
- final/average validation mAP：A0=`42.3799/49.2652`、A1=`41.8802/47.9054`、
  A2=`42.1556/48.8605`。A2低A0`0.2243`，未通过预设晋级规则，seed1/2与test均不运行。
- A1/A2相对A0的8个累计task mAP全部为负。残差gate正常学习且远未饱和；A2虽把task7 current-only
  mAP提高`1.3924`，但全类final下降且forgetting增加`0.0916`，属于当前新类拟合与旧类排序的交换。
- A2改善Doubt/Confusion、Pleasure、Surprise、Sadness，但损失Anger、Aversion、Pain、Suffering等，
  净mAP为负。停止训练级残差调参，保留独立专家固定R1正式冠军。
- 若继续动态分路，下一项应先设计image-group K-fold OOF专家预测协议：独立专家不被Router反传，使用
  完整OOF训练样本训练跨task共享小Router，taskwise输出冻结。结果报告位于
  `output/emotic_track_a_full_anchored_residual/20260910_103042/analysis.md`。

## 2026-09-10：恢复服务器训练与Face环境

- 新容器`b234c89a7a5b`审计：8张RTX4090、driver580.173.02、系统CUDA12.4；挂载盘代码、
  数据、权重和结果完整，但Conda只剩base，旧`ddp`/`cocoer-preprocess`和SSH授权文件丢失。
- 根据历史config与Face audit恢复两套Python3.9.23环境。训练栈锁定torch2.0.1+cu118、
  torchvision0.15.2+cu118、numpy1.26.4和requirements固定版本；Face栈锁定InsightFace0.7.3、
  ONNXRuntime1.18.0、OpenCV4.10.0、Pillow11.3.0、SciPy1.13.1，并补同版本torch栈满足项目导入。
- 两套环境`pip check`通过；ddp识别8卡并完成CUDA矩阵运算。SCRFD权重大小16,923,827字节，
  SHA-256=`5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`，真实EMOTIC图检测成功。
- OOF实验worktree从`e490da0`安全ff-only到`e0e6d57`，未改主工作树或test-only；test-only仍停在
  `0b59138`，其原有修改和未跟踪文件均保留。169项完整单测全通过；冠军Image-token layer1/b32
  Adapter-ASL GPU smoke和真实Face→CLIP GPU smoke通过。
- 最小端到端smoke使用Full/seed0/OOF fold0/task0/1epoch/batch64/AMP+TF32，完成56步、
  skipped0，val mAP39.543071；禁止test、无checkpoint、无完整训练。
- 重新安装本机公钥到服务器，普通SSH免密通过；本地SSH config启用IdentityFile和agent forwarding，
  服务器GitHub SSH鉴权通过，未传输私钥。恢复详情见`docs/server_environment_restore_20260910.md`。

## 2026-09-10：实现分层class-aware OOF stacking

- 从环境恢复提交创建`codex/class-aware-oof-stacking`。新增C0固定R1、C1每类三视图bias、C2
  rank-2共享样本可靠性×类别view factor；全部直接读取已有OOF scores/descriptors和validation
  endpoints，不包含专家训练或test入口。
- C1只更新当前task引入类别；C2冻结最佳C1 bias，旧类interaction行冻结，但共享投影通过已见任务
  OOF rehearsal保持稳定，并为每个task保存快照。评估旧类严格读取其引入task的前缀特征与快照。
- 网格锁定为C1/C2正则各`{1,3,10}`，FP32、120 epochs/task、batch512、Adam1e-2、rank2。
  降低后的advance规则为C2相对C0/C1均至少+0.05 final mAP，并通过两类以上和80%最大贡献占比检查。
- 新增4项初始化、Face mask、类别隔离和rank2零初始化回归，以及单GPU安全launcher和协议文档。
  本地Python编译、shell语法和diff检查通过；完整依赖测试与真实数据运行待服务器执行。
-
## 2026-09-14：准备C2/J0/A2锁定test横向补测

- 从`exp/oof-advantage-ranking-distillation`创建`exp/three-architecture-locked-test`；保留既有4份
  未跟踪调参文档和`tmp/`，未纳入本次提交。
- 新增C2 test evaluator：校验validation selection SHA-256、逐task state SHA、test descriptor/
  Face manifest与三种子固定R1源；只评估唯一best C2，并输出逐seed、逐task、逐类权重和聚合指标。
- 新增J0/A2 seed0/1/2锁定test runner、7卡并行launcher和统一汇总器。6个训练进程分配GPU0--5，
  C2只读评估分配GPU6；只设显存余量门槛，不要求GPU空闲，以支持用户指定的显存叠加运行。
- 本地系统Python缺少numpy/torch，依赖型单测留待服务器`ddp`环境；Python源码静态编译、shell
  语法、diff检查需在提交前完成。尚未启动服务器实验。
- 初次smoke在训练前安全暴露两项schema适配：旧默认Face manifest不含test，C2的feature schema
  位于顶层protocol而非候选摘要。改用既有R1正式test manifest并从锁定protocol注入schema，未改变模型。
- 修复提交`42d0785`后服务器206项全测通过；J0/A2 task0一轮均完成、score dump齐全，C2对三种子
  test只读评估通过。C2 final mAP=`33.0153±0.3074`，配对R1增益`+0.0034±0.0062`。
- 验证记录提交后服务器ff-only至`b1cddff`，正式batch`locked_architecture_test_20260914_013250`
  在tmux `multilane_locked_arch_test_013250`启动。六个训练进程全部写出clean config并完成首轮
  84 updates/skipped0；J0/A2分别约3.6/2.1GiB显存，无错误。C2统一控制结果已完成。
- batch自动汇总正常结束；J0三seed和A2 seed0/2为240 epochs、13,950 updates、0 skipped。
  A2 seed1为240 epochs、13,949 updates，在task3 epoch27有一次AMP scale下降，loss仍有限。
- 三seedfinal mAP为R1/C2/J0/A2=`33.0119/33.0153/32.1639/32.4201`。C2配对增益
  `+0.0094/-0.0030/+0.0038`，不具方向一致性；J0/A2在全部8个累计task均低于R1。
- 仅同步JSON、日志和控制文件共50项约2.6MB，无checkpoint/score NPZ；远端与本地SHA-256逐项一致。
# 2026-09-14：实现Shared multi-view DGL阶段0/1

- 新建分支`exp/shared-multiview-dgl-audit`，没有触碰已有未跟踪用户文档或`tmp/`。
- `MultiLaneModel`支持仅对融合输入detach，并新增遵循task lane归属的fused/Full/Person/Face seen logits；
  主参数拆成共享表示参数和Head/Fusion预测参数，但普通训练的优化器参数集合保持不变。
- runner将逐视图BCE/ASL显式化；完整DGL通过`autograd.grad`把单视图BCE路由到selectors/prompts、
  单视图ASL路由到当前Image-token Adapter、融合BCE路由到Head/Fusion，三个参数组严格互斥。
- 新增每task首batch梯度范数/余弦与每task完整validation视图诊断，包括Face mask、权重分位数和相对
  Full的正负样本对纠正/损伤；新增G0/G1/G2 runner、三卡launcher、严格汇总器和回归测试。
- 静态编译、shell语法和`git diff --check`通过。本地运行单测因本机Python没有torch/pytest无法执行，
  不是代码测试失败；下一步提交推送后在服务器独立worktree运行完整测试和三种真实GPU smoke。
- clean提交`aff1eee`已推送并在服务器新建独立worktree；210项`unittest discover`全通过。首次三卡
  smoke中G0/G1正常完成，G2在模型加载前触发不存在的CLI Namespace字段；修复仅移除main层冗余字段
  检查，`train_task`内部DGL互斥校验保持不变。
- `ed17f45`同步后210项全测再次通过，G2真实task0 smoke完成84 updates/skipped0；G0/G1/G2均有完整
  梯度和评估诊断。G0审计显示融合梯度相对Person/Face单视图梯度明显更大，G1/G2的融合表示梯度为0，
  参数路由断言成立。三组task0一轮值不作实验选择，下一步在同一新提交启动fresh正式batch。
- `f83f648`在服务器独立worktree启动batch`shared_dgl_seed0_20260914_215006`，GPU0/1/2并行
  G0/G1/G2。三份config已核验同commit、seed0、8-task validation、无checkpoint/test，当前正常初始化
  并进入task0；只等待该唯一batch，不重复启动。
- 2026-09-15：G0/G1/G2均正常完成240 epochs与13,950 updates，0 skipped，无OOM/NaN；自动汇总退出码
  为0。通过SCP同步15个结果、6个控制、7个worktree日志和1个外部launcher日志，四棵文件集合均
  逐项SHA-256一致。
- 同批final validation mAP：G0=`43.0754`、G1=`41.4177`、G2=`42.4147`。G2相对G1
  `+0.9970`，说明只截断融合梯度而仍用0.1辅助监督会明显欠训练；但G2相对G0 final/average/Full
  分别为`-0.6607/-0.5943/-0.8841`，距独立R1的差距也扩大，全部推进检查失败。
- G0全task梯度审计没有负余弦，融合梯度也未持续大于单视图梯度；G2把Person和可靠Face各提高约
  `0.54`且forgetting改善`0.1440`，却降低Full并造成Sadness/Suffering的大幅交换。按预注册规则
  结束当前共享DGL路线：不跑seed1/2、test、alpha或动态Router。

## 2026-09-17：同步并分析视图专用 Selector validation

- 服务器 batch `selector_validation_seed0_20260916` 的 S0/S1/S2 自动完成，`validation_status.txt` 确认三组完成；每组240 epochs、13,950 updates、0 skipped。同步 15 个结果 JSON 与 5 个日志/控制文件到本地 `output/`、`logs/`，逐文件 SHA-256 完全一致，未传输 checkpoint。
- S1(view-specific×10/视图) final/average mAP `41.3856/47.7380`，低 S0(shared×10) `1.3669/1.9383`，也低同参数量 S2(shared×30) `0.3983/1.2442`。S1每个累计 task 均低 S0，并使 Full/Person/Face/reliable-Face final mAP 分别降低 `1.3876/3.0040/2.6409/3.2301`。
- 预注册晋级检查失败，结束本版“共享 encoder + 视图专用 Selector”路线；不运行 seed1/2、test 或后续动态 Router。完整分析：`output/emotic_track_a_view_specialized_selector/selector_validation_seed0_20260916/analysis.md`。

## 2026-09-17：视图专用 Selector 工程审计

- 完成严格路由回归：仅扰动 S1 Person Selector，Person 输出变化，Full/Face 输出最大绝对差为0；结合225/225全测、config差异审计和三组零skip完成记录，未发现工程接线错误。
- 原因分析写入 validation report：S0共享 Selector持续获得三视图梯度并形成正耦合；S1 bank隔离使representation path cosine接近0，失去跨视图正则化。S1 task0训练loss降至0.6390但val mAP仅56.2591，30 epochs后相对S0下降1.3669；S2共享30也下降0.9686，支持容量/过拟合与固定融合失配解释。

## 2026-09-17：视图专用下游组件实验启动修复

- 在`exp/view-private-components`上修复view-specific Prompt初始化的维度扩展错误；Prompt bank现为
  `[key/value, view, task, prompt, head, head_dim]`，三路仍从同一共享初始化复制。
- 修正新增回归测试对`current_all_logits_with_view_features`的输出契约断言：该接口保留全类别输出，
  当前task列由训练loss按索引选择，因此task0输出为8类而不是5类。
- 本地静态检查通过；服务器需在同一独立worktree重新运行完整单测和P1/P2/H1/H3代表性GPU smoke，
  通过后启动P0--P3/H0--H3 seed0 validation。
- AMP smoke还发现三视图未启用selector conditioning时`selector_condition_max_initial_difference`未初始化；已在smoke入口统一初始化为0，避免训练前打印阶段误退。
- 36de565同步后服务器228项全测通过；P1/P2/H1/H3四组AMP/TF32 task0 smoke通过。正式batch `view_private_components_seed0_20260917_102434` 已在 tmux `multilane_view_private_102434` 用GPU0--7并行启动，首轮均84 updates、0 skipped、显存约3.5GiB/卡。

## 2026-09-17：视图专用下游组件 seed-0 validation 结果

- batch `view_private_components_seed0_20260917_102434` 已结束并同步；171份输出/日志逐文件SHA-256与服务器一致，未访问test、未保存完整checkpoint。P0/P1/P2/H0/H1完整完成240 epochs、13,950 updates且zero skip；H3有166 AMP skips并少166 updates，P3/H2在task3出现non-finite logits后安全停止，后三者不用于正式指标比较。
- P0精确复现S1 (`41.3856`)；P1/P2/H0/H1 final mAP为`41.8436/42.2652/41.4274/41.5362`。P2相对P0 `+0.8796` final、`+0.9336` average，Full/Person/Face/reliable-Face均提高，八个累计task全正；1705原图组2000次配对bootstrap区间为`[+0.0403,+1.6639]`。
- P2通过P0局部筛选，但距S0 shared Selector `42.7525`仍低`0.4873`，只恢复S1损失的64.3%，结论为部分缓解而非路线翻转。P1未过+0.50阈值；H0说明fusion placement影响极小，H1私有head降低Full；P3/H2/H3共同的task3 AMP instability说明不能以all-private组合继续选择。完整报告：`output/emotic_track_a_view_private_components/view_private_components_seed0_20260917_102434/analysis.md`。

## 2026-09-17：P3稳定性修正版

- 新分支`exp/view-private-components-p3-stable`针对P3 task3的AMP溢出增加可选全局梯度裁剪；
  `--gradient-clip-norm 1.0`在`GradScaler.unscale_`后、optimizer step前作用于全部参数组，默认0保持旧行为。
- P3的Prompt、独立Image-token Adapter、Selector、学习率、ASL、joint routing、AMP/TF32和固定融合均不变，
  只验证裁剪后P1/P2收益是否可叠加。实验协议见`docs/view_private_components_p3_stability_validation.md`。
- 首次gradient-clip=1.0试跑仍在task3出现同样的134+32 skipped并于epoch18产生non-finite logits，
  证明overflow发生在裁剪前；稳定版改为仅`--no-amp`的FP32控制，clip恢复0。

## 2026-09-17：同步并分析P3 FP32稳定性验证

- P3 FP32 batch `view_private_components_p3_stable_fp32_seed0_20260917_152148`已结束；本地同步output/log后，服务器与本地逐文件SHA-256一致。完成240 epochs、13,950 updates、zero skipped，`MULTI_LANE_TRACK_A_COMPLETE`存在，未访问test、未保存full checkpoint，日志无OOM、non-finite、traceback或error。`amp=false`、`tf32=true`、`gradient_clip_norm=0`与协议一致。
- P3 final/average mAP=`40.9340/47.5726`，较P0为`-0.4516/-0.1654`、较P1为`-0.9096/-1.0226`、较P2为`-1.3312/-1.0989`。P3在每个累计task都低于P1/P2；相对P0仅task0 `+0.4619`和task4 `+0.0300`为正，其余为负。
- final Full/Person/Face/reliable-Face为`39.9004/35.5129/31.8530/34.4468`；相对P0 Person/reliable-Face下降`0.9828/0.6897`，不通过预注册保护。P3−P2的1705原图组2000次bootstrap为`-1.3160`、95%`[-2.0876,-0.5579]`；P3−P0为`-0.4548`、95%`[-1.2159,+0.2937]`。
- FP32证明数值稳定性问题可消除，不能证明Prompt+Adapter有效叠加。P3相对P1只多约5%可训练参数，故纯容量解释较弱；更合理的待验证机制是Prompt和Adapter同时私有化移除了两者中仅存的跨视图耦合，而固定融合与共享post-fusion head需要三路表征保持兼容。由于P3关闭AMP，不能把它与旧AMP P0/P1/P2作无混杂因果归因；不跑P3 seed1/2或test。完整报告位于`output/emotic_track_a_view_private_components_p3_stable/view_private_components_p3_stable_fp32_seed0_20260917_152148/analysis.md`。

## 2026-09-17：FP32四组交互与选择性 Prompt 修改

- 从 `exp/view-private-components-p3-stable` 的 `c6ef90b` 创建 `exp/view-private-components-fp32-selective-prompt`。
- 新增 Prompt 模式 `view_specific_full_face`，实现 Person 共享 bank、Full/Face 私有 bank；初始化和增量 task 复制保持确定性。同步扩展 runner/smoke 参数、实验元数据、回归测试。
- 新增 `scripts/emotic/launch_multilane_track_a_fp32_interaction_selective_val.sh`，固定 `--no-amp`、TF32、seed0、8 tasks、30 epochs/task、batch64、既有 P2 Adapter/Selector/fusion 协议，运行 F0/F1/F2/F3/S1；新增 `summarize_fp32_interaction_selective.py` 计算 `F3-F1-F2+F0` 与 `S1-F2`。
- 本地 `python3 -m py_compile`、shell `bash -n`、`git diff --check` 通过；本机无 torch，依赖型单测待服务器 `ddp` 执行。

## 2026-09-17：FP32交互与选择性 Prompt validation 已启动

- 服务器 `ddp` 完整单测 229/229 通过；selective Prompt 真实 ViT task0 FP32 smoke 通过，初始化等价、路由和 zero-skip 检查成立。
- 正式 batch `fp32_interaction_selective_seed0_20260917_230307` 已在 tmux `multilane_fp32_interaction_230307` 启动，GPU0--4 并行 F0/F1/F2/F3/S1。由于 GPU 当时约 5.6GiB 空闲且单 run 历史约3.5GiB，launcher 显存门槛设为5GiB；配置仍为 FP32/TF32、8-task validation、无checkpoint/test。

## 2026-09-18：FP32交互与选择性 Prompt validation 结果

- 五组均完成 240 epochs、13,950 updates、zero skipped，无 OOM/NaN/traceback；output 94 个文件、logs 10 个文件已从服务器同步，本地与服务器逐项 SHA-256 一致。
- F0/F1/F2/F3/S1 final mAP 分别为 `41.7772/41.4559/41.6545/41.2212/41.8203`，average mAP 分别为 `49.1334/48.7443/48.0103/48.4650/48.0093`。同精度控制显示 F1/F2 单独臂相对 F0 都略低，FP32 本身改变了旧 AMP 基线的绝对数值。
- 预注册交互项 `F3-F1-F2+F0`：final `-0.1121`，average `+0.8438`；逐 task 为 `[+1.4001,+1.0433,+1.2592,+1.1022,+1.2414,+0.7942,+0.0222,-0.1121]`。因此 final 末任务存在轻微负交互，但累计平均并非负，证据不足以确认结构性负交互。
- 选择性 S1（Person Prompt 共享、Full/Face 私有）相对 F2：final `+0.1659`、average `-0.0010`；最终单视图 Full/Person/Face/reliable-Face 变化为 `+0.627/-0.962/+0.262/+0.621`。它改善 Full、Face 和可靠 Face，但 Person 仍是瓶颈，融合总收益有限。
- 完整汇总见 `output/emotic_track_a_fp32_interaction_selective/fp32_interaction_selective_seed0_20260917_230307/interaction_summary.md`。

## 2026-09-18：共享基座视图残差与晚层 Prompt 实验启动

- 新分支 `exp/shared-view-residuals-late-prompts` 从同精度实验分支创建并推送；核心提交链为 `949e58d`、`0925f5c`、`69a6d42`、`da607db`、`d9d8b2a`。保留既有用户文档和 `tmp/`，未纳入提交。
- 新增四臂协议：R0 共享 Selector/Adapter/Prompt；R1 共享 Selector+零初始化视图残差并加通用 L2；R2 共享 Adapter+零初始化视图残差；R3 前3层共享 Prompt、最后2层 Full/Face 私有、Person共享。
- 服务器完整单测 231/231 通过；四组真实 ViT FP32 task0 smoke 均通过，包含初始化 roundoff、视图路由和 optimizer 参数审计。
- 正式 batch `shared_residuals_late_prompt_seed0_20260918_103705` 已在 tmux `multilane_shared_residuals_103705` 启动，GPU0--3 并行；固定 FP32/TF32、seed0、8 tasks、30 epochs/task、batch64、joint gradient、auxiliary loss 0.1、validation-only、无 checkpoint/test。

## 2026-09-18：共享基座残差与晚层 Prompt 结果已同步

- 从服务器 `/mnt/haoyuan/workspace/multi-lane-main-shared-residuals` 同步 batch `shared_residuals_late_prompt_seed0_20260918_103705` 的 output/log 到本地；远端与本地共98个文件 SHA-256 全部一致。
- R0/R1/R2/R3均完整结束：240 epochs、13,950 updates、zero skipped，无 OOM/NaN/traceback。
- 指标（final/average mAP）：R0 `43.0394/50.5326`；R1 Selector residual `41.9762/48.7030`；R2 Adapter residual `42.4221/49.6954`；R3 晚层 Full/Face Prompt residual `42.4917/49.9806`。三种残差均低于 R0。
- task7 fused 与单路 mAP：R0 `43.036/41.640/40.152/34.107/37.849`；R1 `41.973/40.370/40.373/33.823/36.894`；R2 `42.417/41.070/38.766/33.955/37.291`；R3 `42.494/41.279/38.702/33.813/37.026`（顺序 fused/full/person/face/reliable-face）。
- 结论：共享基座加小视图残差在当前固定融合与增量训练协议下未超过共享基线；R2/R3后期 Person/Face退化，R1虽Person略升但Full/Face下降。下一步优先研究融合校准/样本级路由，并用多 seed 验证，而非继续放大私有残差。

## 2026-09-18：R0 与动态三视图融合 test 脚本

- 用户要求将当前 R0 的固定融合与动态融合版本做 held-out test 对照。确认代码已有 `soft_three_view` task-local hidden-16 masked-softmax Router；新增独立 R0/D1 test runner 与双 GPU launcher，避免修改历史 test 入口。
- test 协议锁定：R0=`fixed_three_view`，D1=`soft_three_view`；二者共享 Selector/Prompt/Adapter，FP32（`--no-amp`）+ TF32，seed0，8 tasks×30 epochs，train/eval batch64，Adam reset/task、cosine、main LR0.0125、Adapter LR4e-4、view Router LR4e-4、auxiliary0.1、无checkpoint、score dump 开启、无test-side调参。
- 动态版本沿用固定先验初始化和可靠性 mask；本批结果只作 test 诊断，不能反过来改变 validation 阶段的候选选择。

## 2026-09-18：R0 与动态三视图融合 test 已完成

- batch `r0_dynamic_fusion_test_20260918_152435` 已同步本地；R0/D1均240 epochs、13,950 updates、zero skipped，日志无 OOM/NaN/traceback，34个文件 SHA-256 与服务器一致。
- test 指标：R0 fixed `31.9883/38.8315/32.7963/48.9977/4.8298`，D1 dynamic `29.5224/36.1115/32.1142/47.2332/4.7045`（依次 final mAP/average mAP/cF1/oF1/forgetting）。D1 相对 R0 final mAP `-2.4658`、average mAP `-2.7200`。
- D1 从 task0 到 task7 全部低于 R0；最终 task7 fused/full/person/face/reliable-face 为 D1 `29.524/29.223/28.242/23.356/26.499`，R0 `31.987/30.713/30.163/25.140/28.023`。
- 动态 Router 最终明显偏向 Person，task7 lane0/lane1/lane2/lane7 平均权重约为 `0.038/0.909/0.053`、`0.019/0.896/0.085`、`0.091/0.838/0.072`、`0.046/0.771/0.183`（Full/Person/Face），解释了性能下降。动态路由没有利用三路互补，出现 Person collapse。
- 完整分析见 `output/emotic_track_a_r0_dynamic_test/r0_dynamic_fusion_test_20260918_152435/analysis.md`。

## 2026-09-23：ParaX level routing implementation

- On `exp/parax-level-routing`, implemented a shared expert-center token-only ParaX image adapter for frozen CLIP block boundaries, with optional level conditioning, static uniform control, official/small initialization, and separate optimizer wiring.
- Added gate/residual diagnostics to per-epoch history and extended smoke/model CLI configuration. Added `tests/test_track_a_parax.py`; local dependency tests remain unavailable because this macOS environment lacks torch/numpy.
- Local `py_compile` and `git diff --check` pass. Next: commit only ParaX code/test/context files, push, create a clean server worktree through Git, then run ddp tests and real ViT task0 smoke for B0/P-post/P-10/P-8:10/P-8:10-level/Static-control.
- Server Git-only worktree `/mnt/haoyuan/workspace/multi-lane-main-parax` is clean at commit `5163e27`; existing `/mnt/haoyuan/workspace/multi-lane-main-test-only` was not modified. `python -m unittest discover -s tests -p 'test*.py'` passed 234/234.
- Real ViT six-arm smoke is queued by resource state only: all 8 server GPUs were occupied (1.6--2.2 GiB free, >93% utilization), so no GPU job was started or preempted.
- Once GPUs became free, all six task0 real ViT smoke arms ran. Five passed; Static-control initially failed because the generic smoke demanded gradients for intentionally frozen static routers. Updated static control to exclude routers from trainable parameters; full 234-test suite passed again, then all six smoke arms passed (including Static-control).
- Added formal validation run/launcher scripts for the locked seed0, 8-task, 30-epoch, batch64 fixed-three-view plan. They are being validated before the run is launched.
- Six-arm task0 smoke batch `parax_task0_smoke_20260923_retry` completed on GPUs 0--5; all six reported `MULTI_LANE_TRACK_A_SMOKE_OK`. Static-control was separately rerun after the static-router trainability fix and passed.
- Full seed0 validation batch `parax_shared_level_seed0_20260923_20260923_2045` started on server GPUs 0--5 with commit `f769d3f`; six runs are active, each entered task0 and completed its first 84-update epoch with zero skipped updates. Current first-epoch losses: B0 0.67692, P-post 0.67800, P-10 0.67707, P-8:10 0.67603, P-8:10-level 0.67728, Static-control 0.67607. These are training losses only, not validation evidence. Outputs: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_shared_level_v0.1/parax_shared_level_seed0_20260923_20260923_2045`; logs/status: `logs/emotic_track_a_parax/parax_shared_level_seed0_20260923_20260923_2045` and matching `output/` manifest/status.

## 2026-09-25：后置特征残差结果与logit校准实现

- 同步批次`post_calibration_control_20260925_134822`；B0/V-post-residual均完成3 tasks×30 epochs、5,010 updates、zero skipped，日志无OOM/NaN/traceback并以`MULTI_LANE_TRACK_A_COMPLETE`结束。10份JSON与4份日志的本地/服务器SHA-256逐项一致；未同步NPZ。
- 同批B0 final/average mAP=`43.4145/52.3126`，V-post-residual=`43.3882/52.0427`，差`-0.0263/-0.2699`；forgetting只改善`0.0222`。task0差`-0.6981`，final Full/Person/Face/reliable-Face差`-1.0351/-0.2047/-0.9411/-1.2377`。
- V-post-residual task2相对Full纠正`312,681`个错误pair但损坏`257,918`个正确pair，净收益`54,763`，低于B0的`73,414`；说明 residual 有信号但选择性不足，不能稳定保留原排序。
- 配置审计发现本批唯一相对strict paired B0的差异为view auxiliary loss `0.1→0`，解释了本批B0从`45.2395`降至`43.4145`。停止现有feature residual，不做seed1/2或test。
- 新建`exp/fixed-fusion-residual-calibration`，实现`logit_residual_three_view`：固定融合产生B0 logits后，用task-local逐类别Person/Face差值做bounded residual；输入detach、系数零初始化、幅度`±0.1`、每task52参数。训练将原B0 loss仅路由到Selector/Prompt/head，calibrated BCE仅路由到residual参数。
- 新增配对runner/launcher、初始化/Face mask/task隔离/梯度隔离测试与实验协议。macOS静态编译和shell语法检查待完成；本机无torch，完整单测和真实GPU smoke必须在服务器`ddp`环境执行。
- 服务器clean worktree完整242项单测通过。首次真实candidate smoke在训练前暴露CLI校验误读不存在的Namespace字段；B0正常运行。该接线错误已本地修复，需提交后ff-only同步并重跑candidate smoke。
- CLI修复后目标23项与完整242项单测再次通过。真实task0一轮smoke：B0和L-post-logit均84 updates、zero skipped；两组base loss同为`0.67691673`。候选每task52参数，Person/Face系数平均绝对值`1.49e-4/7.87e-5`、最大`0.00103`，有限非零且未饱和；smoke val mAP相对B0仅`+0.00093`。正式两臂validation可启动。
- 正式batch`logit_residual_control_20260925_163436`已从clean server HEAD `865aec1`启动，GPU0/1分别运行B0-paired与L-post-logit。固定seed0、task0--2、30 epochs/task、batch64、auxiliary0.1、AMP/TF32、validation-only、无checkpoint/test。两组首轮84 updates、zero skipped，base loss同为`0.67691673`，无训练错误；后续不持续轮询。

## 2026-09-25：logit residual正式结果与下一实验实现

- 通过SSH tar同步`logit_residual_control_20260925_163436`两组完整结果、六个NPZ、四份日志和控制文件；本地/远端24个SHA-256逐项一致。
- 两组均`status=complete`，90 epochs、5,010 updates、zero skipped。B0/L-post-logit final=`45.2395/45.3251`、average=`54.9341/55.0368`、forgetting=`2.7760/2.7458`。
- L-post-logit在三个task均小幅提高mAP，但final只`+0.0857`，未过预注册`+0.10`。Full/Person/Face/reliable-Face和base BCE完全一致；排序净纠正增加但正确pair损坏也增加。按规则停止该候选，不运行seed1/2或test。
- 创建分支`exp/shared-selector-independent-adapter`。实现新的三组task0--2配对入口：A0共享b32、A-cap共享b97、A-view三路独立b32；后两者每task参数只差1。
- 修正独立view bank初始化：bottleneck相同时从同task共享bank复制到Full/Person/Face，再冻结未使用共享bank，使A0/A-view初始输出严格对齐。
- 增加只读per-view residual/token ratio和shared/Full/Person/Face Adapter梯度范数，诊断不参与loss。新增初始化、参数量和诊断单测；本地`py_compile`、`bash -n`和`git diff --check`通过。
- 提交并推送`284a34d`；服务器审计全部worktree后创建独立clean worktree，保留有历史修改的主工作树和`multi-lane-main-test-only`不变。
- 服务器`ddp`完整单测245/245通过。三组真实task0一轮smoke均84 updates、zero skipped、无训练错误：A0/A-cap/A-view mAP为`39.8053/40.7147/39.8119`，仅作短程诊断。
- A-view的Full/Person/Face梯度范数约`9.96e-5/5.61e-5/8.13e-5`，residual ratio约`0.00168/0.00328/0.00677`；三路均有效更新。共享b97 residual更大，正式配对容量控制不可省略。
- 提交smoke记录`323ba28`并ff-only同步服务器clean worktree。首次正式启动在执行launcher前因外层nohup日志父目录不存在而退出，没有产生训练或结果；创建目录后以同一batch ID安全重启。
- 唯一正式batch`shared_selector_independent_adapter_20260925_1803`已在GPU0--2运行。A0/A-cap/A-view首轮loss为`0.67687206/0.67701335/0.67696667`，各84 updates、zero skipped；参数量和日志路径正确。训练期间不持续监控，结束后统一同步。

## 2026-09-25：启动共享 Adapter 容量完整 validation 准备

分析 `shared_selector_independent_adapter_20260925_1803`：A-cap b97 相对 A0 b32 的 task0/task1/task2 fused mAP 提升 `+2.5218/+1.6715/+1.0746`，最终 `+1.0746`、average `+1.7560`，且 Full/Person/Face/reliable-Face 全部提升；A-view 独立三路虽较 A0 提升 `+0.6245`，但相对容量匹配 A-cap 低 `0.4501`，Face/reliable-Face 相对 A0 下降 `-1.0424/-2.2674`，按预注册规则停止。

旧类负梯度不是主要原因：`legacy_full_zero` 把非当前类别 logit 用 `index_fill` 常数化为0，旧类分支不产生梯度；因此未实施无依据的旧分类行冻结。新分支 `exp/shared-capacity-lane-freeze` 新增 A0/A-cap 完整 8-task validation 脚本和协议文档，先验证 b97 收益能否跨后期任务保持。

## 2026-09-25：共享 Adapter 容量完整 validation 启动

分支 `exp/shared-capacity-lane-freeze` 提交 `5ef84d2` 已推送。服务器创建独立 clean worktree，未修改主工作树或 `multi-lane-main-test-only`；ddp 全测245/245通过。A0/A-cap真实task0 smoke各84 updates、zero skipped，trainable parameters 739130/839035，总 Adapter bank 399616/1198856，smoke loss分别0.67687206/0.67701335。正式 batch `shared_capacity_full_validation_20260925_2015` 已在GPU0/1启动，结果根目录 `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_capacity_full_v0.1/`，等待完成后同步分析。
- 2026-09-25：共享 Adapter 容量完整 validation 结果与 residual-scale follow-up

  A0 shared b32 完成 8 tasks、240 epochs、13,950 updates、zero skipped，final/average mAP 为 `42.7525/49.6763`，forgetting `0.8915`。A-cap shared b97 在 task0--2 分别高 A0 `+2.5218/+1.6715/+1.0746`，task3 cycle1--17完成后累计140次 AMP skip，在cycle18因`ASL received non-finite training logits`中止。b97已保存task0/1 residual-energy/token-energy ratio明显高于b32，但task3失败batch没有保留ratio快照，不能把早期ratio当成故障瞬间读数。详见`docs/shared_capacity_full_validation_results_20260925.md`。

  新建`exp/shared-capacity-residual-control`，新增scale=0.03的b32/b97配对完整validation入口。服务器完整245项单测与两组真实ViT smoke通过；正式batch`shared_capacity_residual_scale_validation_20260925_222440`已在GPU0/1启动，首轮各84 updates、zero skipped，validation-only、无checkpoint/test。
# 2026-09-26：共享 Adapter residual-scale validation 结果与 test 启动

- 服务器 validation batch `shared_capacity_residual_scale_validation_20260925_222440` 的 b32/b97 均完成 8 tasks × 30 epochs，无 AMP skipped。A0/b32 final/mean mAP=`42.3199/48.9646`，A-cap/b97=`42.5853/50.2722`；b97 差 `+0.2654/+1.3077`。它通过数值稳定性检查，但 final 增益未达到预注册 `+0.5` 晋级条件。
- 用户要求执行 test 后，按 paired capacity comparison 启动两臂；未将 b97 单独挑作胜者。配置锁定 seed0、EMOTIC Track A、8 tasks、30 epochs/task、batch64、冻结 CLIP ViT-B/16、layer1 shared Image-token Adapter、b32/b97、residual scale0.03、fixed reliable-Face fusion、BCE+Adapter ASL、auxiliary loss0.1、AMP/TF32。test 不参与训练、调阈值或调融合权重。
- 无 validation checkpoint；test 两臂均从头用 train split 训练，逐 task 在 test split 报告。服务器 batch `shared_capacity_residual_scale_test_20260925_160954` 已在 GPU2/3、tmux `ml_captest_160954` 进入 task0。test manifest train/test 文件存在。
- Face 可靠性操作定义为 manifest `valid_face && !ambiguous_match && face_short_side>=24 && face_detection_score>=0.6`。mask 由 `continual_datasets.py` 生成，fixed prior 在 `view_fusion.py` 应用：可靠 `[0.64,0.16,0.20]`，不可靠 `[0.80,0.20,0]`。这是预先设定的数据质量过滤标准，不等于检测器概率经过校准。
- test manifest 共5,368个person样本；4,677个检测有效（87.13%），4,179个匹配无歧义（77.85%），3,261个满足完整可靠规则（60.75%）。这只是可靠Face覆盖率，是否提供情绪识别增益要看Face单路与固定融合的held-out指标。
- 计划和完整路径：`docs/shared_capacity_residual_scale_test_plan_20260926.md`。test 完成后同步 logs/output 并分析覆盖率、Face/reliable-Face 单路和融合指标及逐 task 结果。

# 2026-09-26：shared capacity residual-scale validation 分析

- 同步批次`shared_capacity_residual_scale_validation_20260925_222440`两个完整结果目录（含validation score dumps）及日志；远程/本地30个文件SHA-256全部一致。
- A0 b32与A-cap b97均完成240 epochs、13,950更新、zero skipped、无异常。Final/mean mAP分别为`42.3199/48.9646`与`42.5853/50.2722`；b97差`+0.2654/+1.3077`。Final cF1/oF1差`+0.0609/+0.0130`，forgetting差`+0.0655`。
- b97 fused mAP逐task均高于b32，但差值从task0的`+3.2866`递减到task7的`+0.2654`。task7 Full/Person/Face/reliable-Face差为`+0.255/+0.346/+0.054/-0.125`；task0--3 reliable-Face均下降，收益主要在Full/Person。
- per-view Adapter residual诊断为训练batch均值`sum(delta²)/sum(frozen_token²)`能量比。b97 task7 Full/Person/Face=`7.076/8.277/10.702`，task3=`115.259/142.915/187.276`；高于b32且无AMP跳步。scale0.03不是有效能量上界，但该诊断不足以归因AP变化。
- 审查`TaskImageTokenAdapterBank`/`_lane_block`确认当前是共享Selector证据Adapter，视觉CLIP冻结、无ParaX、三视图固定融合；该实验没有证明encoder层级动态路由。
- 由于只有seed0且final增益未达预注册`+0.5`，下一步用seed1/2复现b32/b97 8-task validation，GPU0/1、validation-only、无checkpoint/test；seed0 test继续在GPU2/3。完整结果见`docs/shared_capacity_residual_scale_validation_results_20260926.md`。
- 提交`1df3398`并推送`exp/shared-capacity-multiseed-validation`；服务器创建clean worktree并通过245项unittest、bash -n、两组真实ViT smoke。正式 batch `shared_capacity_residual_scale_multiseed_seed12_20260926` 已在tmux `ml_caps12_20260926`启动，seed1 b32/b97进入GPU0/1；完成后自动执行seed2，test仍在GPU2/3。


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
# 2026-10-15 Task-Forward MoE 首版启动

用户批准开始实施/实验并指定 GPU0、8任务 validation/test 并行。已核查
本地独立 `codex/moe-adapters-three-view-20261015` worktree 与远程 97 个
worktrees；远程主工作树和 test-only 分别存在未提交内容，保持原样。
确认 GPU0 RTX 4090 24 GiB、当前约 24 GiB 空闲；EMOTIC、CLIP、
train/val/test Face manifest 和 ddp Python 存在。锁定首版独立任务池设计
与执行参数：`docs/moe_task_ffn_v1_experiment_20261015.md`。已添加 MoE bank、
model/runner 接口、8 checkpoint val/test 并行评估器、运行脚本和单测。
下一步在本分支检查、提交、推送，服务器新 worktree 测试与 GPU0 smoke，
通过后启动完整运行。当前尚无实验结果。


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

### 预检进展（2026-10-17）

锁定实现`2ec1b58`已在远程独立worktree通过305项Track-A单测。
真实val256人实际专家输入探针通过：Full/Person/Face归一化相邻距离
0.37475/0.58036/0.56503，768维均变化；初始残差严格零，关闭TF32
开关MoE的最大logit差2.7474e-7（门槛1e-6）。CPU严格identity通过。
GPU0四组seed0两更新smoke已并行启动，tmux
`ml_moe_ffn_v4_smoke_20261017`；四组初始主参数SHA同为35af16da90b3...。
GPU1–7未启动本次作业。正式8任务尚待smoke、同人群重放与配对审计通过。

四组GPU0 smoke已全部退出0，各2成功更新/0跳步，训练人群checkpoint重放
mAP差均0。三个Adapter组梯度非零/有限，dynamic Person CLS残差/输入比
约0.00103；这只验证路径。配对审计脚本初次把tasks字典误当列表，已按
现有schema修正，待真实产物审计通过后启动正式批次；模型/训练代码未变。

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

## 2026-10-17：执行v4既有checkpoint只读梯度归因

用户授权开始task0/2/7、四组固定训练batch梯度诊断，不训练v5，继续只用GPU0。
新分支`codex/moe-v4-gradient-audit-20261017`从`bc5f572`创建，保留v4未提交分析。
新增独立gradient_attribution模块、launcher、机制单测及完整协议，模型/训练入口不改。
每task4个原图组不跨batch的至多64人batch，缓存像素跨四组/两数值配置复用；
共96 snapshot，FP32/Expert dropout off与训练AMP/dropout敏感性对照。源checkpoint
位于原v4服务器worktree，均只读；不创建optimizer，无参数更新，验证state/checkpoint
与像素SHA不变。详见`docs/moe_v4_gradient_attribution_plan_20261017.md`。
下一步提交/Git-only远程新worktree、全测、8-snapshot smoke，通过后运行正式诊断。


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


## 2026-10-20：新建Frozen Forward路由归因分支
已检查两端worktree及状态；远程GPU0空闲，primary153/test-only5个dirty条目保留。创建codex/moe-frozen-route-control-20261020，受控代码只经Git同步。修改late_route_control支持Frozen[1]/Top3/冻3，保留原Late[11]/Top2；加载器恢复moe_freeze_count以匹配checkpoint。新增GPU0入口及针对白名单、Top3 schema恢复、实际Frozen路径固定/关闭干预的测试。声明完整回归及9条件小样本smoke，全部通过才启动正式。无模型训练变更，不叠加独立Image Token Adapter。

2026-10-20预检：首次完整367项测试中366通过，新增schema恢复测试的合成训练前向误用no_grad，按设计不累计训练激活，finish_task因此拒绝空计数。已修正测试准备步骤，不改变模型/真实诊断逻辑；未启动真实smoke或正式验证，失败日志保留。

离线分析补记：首次汇总JSON输出遇到numpy.int64序列化错误，已只修正导出标量转换；analysis02完成，原分析日志保留，训练和预测数据均未改变。绘图使用服务器已有matplotlib3.9.0，未安装或修改训练依赖。
