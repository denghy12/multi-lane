# 下一步任务

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
