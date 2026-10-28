# 项目上下文

2026-10-28预检补充：2297dbe远程独立worktree已Git-only建立，5项新增机制测试、434项Track-A与52项相关回归全部通过。增加跨臂原参数/关键loss/增强batch摘要与完整OFF/单路NPZ逐值审计，评估仍共享同次三视图编码；清理新入口未用import，不改旧前向。下一步六组两任务×4更新真实smoke，尚未正式训练，无效果结论。

## 2026-10-28：执行受保护联合摘要Attention六组8任务（实现预检中）

用户授权按方案修改并完整跑8task Val/Test，明确本轮不使用旧task0效果晋级门槛。新分支codex/protected-joint-attention-20261028，起点09a6729；保留此前dirty分析及其他工作树。新增protected_joint_attention/experiment/evaluation/run入口、GPU0队列与机制测试；原model.py不改，runner仅默认关闭的私有优化接口，历史build增加可选模型类型。

C0原B0、C1 CLS-MLP、C2联合摘要、C3目标区域、C4内容门控、FULL_ONLY新增分支仅Full；各seed1/8task/30epoch/B64，原主LR.0125/Adapter .0004/b32/index1/scale.03/aux.1/BCE-ASL，72线程/历史报告RNG，私有Adam-AMP/RNG隔离。第6层摘要，宽64/4头，固定门控.1/上界.1；新门控初值.1。真实标签自训、无A/B参数/答案、同一学习分类头。全部正式臂每任务要求原参数/loss/RNG精确匹配历史边界，否则报机制错误；新增分支不以效果决定是否继续8task。

GPU0空闲约24GB，GPU1–7忙且不使用；最多4作业、最多2评估、最低余量6GB/共享内存1GB自动排队。一次图像前向同时得到ON/OFF/移除新增Face和Person证据，评估2个单线程spawn workers，原CPU每batch sigmoid。epoch恢复包含两个优化器/调度器/scaler与完整RNG，验证前保存。输出output/protected_joint_attention、日志logs同前缀。当前编译/bash/diff检查通过，尚未远程测试/smoke/正式训练；不声称有效。

## 2026-10-28：规划受保护的三视图交互摘要Attention（未实现/未训练）

用户要求沿Attention寻找事先定义、自行训练的单模型方案，不使用A/B参数或答案拼装。重新核对完整均值33.6248/40.5129、旧OFF控制、独立Selector/只训Attention、近期epoch与2×2结果及早期B0保真logit校准；完整均值证明函数互补，不能独归Attention或Face。保护原B0与后置残差已有先例，本身不包装成新贡献。

候选保留32.6800历史配方的原三路Task Forward与原BCE/Adapter-ASL，从初始即共同30轮学习；另设私有交互读出，读取三视图Summarized Patches，目标框限定的并行Full Summarize、Person/任务条件Query、可靠Face softmax前mask、内容门控及有界末端特征残差，同一个原头分类。新增真实标签目标仅训练私有模块；原θ/head依原固定融合目标更新，独立RNG/optimizer/GradScaler，需逐步审计原函数重放。不是冻结随机头、额外5轮、外部教师或两模型；也不保证最终融合泛化或原Face AP再提升。

方案docs/protected_joint_attention_plan_20261028.md列C0强B0/C1 CLS-MLP/C2联合摘要/C3人物区域/C4门控及Full-only source控制；先task0再0–2后8任务，GPU后续仅0。本轮只研究/新增方案和更新三上下文，当前分支仍codex/teacher-free-epoch-head-attribution-20261028，HEAD09a6729；不修改模型/训练入口、不启动GPU、不提交推送，保留原dirty分析和其他工作树。

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

## 2026-10-27：无A/B教师的三视图共同学习规划（未实现/未训练）

用户明确最终方法必须由同一模型在事先规定流程中自己学会保留Full并利用Person/Face，不以A/B提供融合答案或训练后组装。当前新规划分支`codex/teacher-free-shared-view-plan-20261027`，HEAD仍28a1cbb，保留此前未提交分析文件；不改业务源码、不提交/推送、不运行GPU。

建议首版保持强B0原三路Task Forward、共享Adapter/Selectors/Prompts、固定特征融合和共享头；标准CLIP+fresh任务参数，每任务原30轮全程head可训。新增真实标签的Full条件互补排序，仅额外loss从Person/可靠Face反传；原BCE/ASL及融合梯度保留。对新增共享更新做分组限幅及基于实际Adam候选步的Full/融合一阶保护，α=0回退本模型当前状态的原B0步，不是调用旧教师。不是概率蒸馏或固定头30+5；一阶训练保护不保证Val/Test AP、Full预测不变或历史全轨迹相同。

规划包含U0原B0/U1普通排序/U2Full条件排序/U3Full保护/U4Full+融合保护及必要等剂量控制，先task0，再0–2，最后seed1完整8任务Val/Test；目标Test32.6800/39.5292同时超越。旧DGL/梯度诊断未证明持续冲突、旧OOF排序失败，均作为设计约束，不包装重复方案为已验证改进。详细协议`docs/teacher_free_shared_view_learning_plan_20261027.md`。

完成TRAIN-only元数据模拟：task0/2/7的B64每批可靠Face类别正负对均值612.67/1186.07/1227.54，零Face对batch为0；同原图人物对仅.095/.5/.292。仅独立shuffle可行性统计，非历史sampler重放；无图像读取/模型前向/参数更新，不能当方法成效。JSON已回传`output/teacher_free_shared_plan/train_pair_feasibility_20261027.json`，SHA256 `ceea9e58fd7c54b25e74360f498f121dfd7a3b6a10561812d67ecdf319232071`。所有三个上下文已更新，主树/test-only保留。


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


## 2026-10-26：强分路复用的只读组合与研究方案

补充500次原图配对：只换Person的Test最终CI[.0446,.2994]、平均[.0369,.3115]均正；双换最终[.0088,.3810]、平均[-.0345,.4124]。优先真实验证Person替换而非只追最终最高的双换；仍无seed方差/Test选择校正。正式训练成本需含B0头/Person/Dual三份30epoch来源，共90epoch/任务，不能当一次30epoch。实际模型尚未运行。

用户希望解决Face共训变强但融合不升，目标为Test最终与阶段平均同时提高。本地新分析分支codex/face-protected-fusion-plan-20261026，保留既有未提交文档；仅新增CPU预测组合与方案，无训练/模型前向/GPU/远程源码变更。A=强Person+B0头32.8864/39.8687，B=双Attention+B0头32.7573/39.7004，48源NPZ IDs/标签/类别/mask/SHA核对。

固定原融合权重、lambda=1主比较：A为底只换B Face得到Val43.1784/50.4692、Test32.9419/39.9239；只换Person43.4224/50.5673、33.0549/40.0482；换Person+Face43.4681/50.7165、33.0608/40.0413；只换Full43.4784/50.5185、33.0472/39.9369。Face替换Val8/8、Test6/8阶段升，Person替换及Person+Face均Val/Test8/8升；双换比单Person最终微升但平均微降，不能宣称稳定叠加。完整Face lambda0/.25/.5/.75/1都报告，不只挑Test最好点。

这是保存logit离线组合，不是完整模型复现或已训练新方法。AMP特征融合与head舍入造成全换B重构Test32.7609而原B32.7573，基线sigmoid AP最大差.000368；不可靠Face概率在Face-only组合中精确保持A。500原图配对阶段耦合Face替换Test最终CI[-.0676,.1918]、平均[-.0809,.2360]含0，未估seed方差/校正Test选择。单路AP与融合贡献不等价：B Full虽单路更低，单独换Full也改善融合，不能将先前ASP代数分解唯一归因Full。

建议下一步完整模型无训练R0(A重放)/R1(只换Face)/R2(只换Person)/R3(双换)/R4(Full控制)8任务Val/Test，尚未执行。保留完整来源Adapter/Selectors/Prompts/Attention/B0头及B辅助Attention依赖的B Full摘要，不能仅把权重或Full源换错；R3辅助两路仍共享B库。先验证函数复用及实际成本，再考虑双教师软功能约束或受限门控；不同于失败的随机独立S和从零只训Face M。计划docs/face_protected_fusion_research_plan_20261026.md，分析output/face_protected_fusion/readonly_20261026_03，脚本scripts/emotic/analyze_protected_face_swap.py，logs同名；源checkpoint未改。

## 2026-10-26：独立Selectors与冻结强Person Face-only完整结果已核验

formal_seed1_20261026_01的12个作业全部exit0，GPU0/seed1/历史seen-positive标准，8任务Val/Test及OFF均完成。远程f6ef130保持clean；24原checkpoint和16新checkpoint SHA复核，主树/test-only HEAD及dirty状态SHA均未变。576个非权重产物已回传并逐SHA校验，归档cadada4716c97a241e298326d6e1f01eb611e7690d2769c60899ad852022fb12。CPU独立重算208阶段融合/832视图指标，最大误差7.11e-14，80旧参考逐值重放、32冻结预测保护、16独立Selector Face mask检查通过。本地分析分支codex/selector-face-analysis-20261026，未改远程源码/启动新训练。

独立Selectors各视图10个，共30个/任务：Val最终38.6066/平均46.9717，Test29.2649/37.7092（B0头）；原共训头40.9621/48.1452、30.7724/38.2750。原共享双Attention+B0头43.3394/50.5522、32.7573/39.7004；原共训头43.1264/50.3256、31.9459/39.1951。即使匹配原共训头仍退步，B0头不匹配放大而非解释全部损失。独立组全Attention OFF Test29.2269/37.5399，Face OFF29.2223/37.5757；大部分损失留在训练后表征/同B0读出。三路都退步，训练BCE/ASL更低，task0已有损失，不直接称旧任务遗忘；独立S参数确实分化，但A/P及Full K/V梯度仍共享，非完整隔离。

冻结强Person只训新增Face：Val43.0787/50.2760，Test32.8792/39.8493，原强Person43.0741/50.2828、32.8864/39.8687。Full/Person全部预测、不可靠Face人群输出逐值不变；Face OFF精确恢复强候选。额外40epoch/2325更新/0跳步，Face可靠人群logit平均变化.1410，非死分支。可靠Face单路Test28.7366→28.7627，可靠人群融合33.8408→33.8537，但全体32.8864→32.8792；AP非子集加权平均，跨可靠/不可靠组排序变化。新增Q仍任务Selector、K/V为Full，未显式按当前Face及人物归属挑选互补证据。保护机制成功，模块净收益未达标。

500次原图配对、阶段耦合bootstrap：独立S+B0相对共享组Test最终CI[-4.1489,-2.9404]；冻结Face相对强Person最终[-.0333,.0185]、平均[-.0487,.0052]。不含训练seed方差/不校正Test筛选。保留原Person+B0与正式B0；不扩展独立Selectors或本版Full→Face模块，不自动重训。下一步如继续，先检查现有Face错误互补及人物对应，避免重跑已有失败的后置残差。详细报告docs/selector_face_controls_results_20261026.md，CPU脚本scripts/emotic/analyze_selector_face_controls.py，分析output/selector_face_controls/analysis_20261026_01。下文启动状态为历史记录。


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

## 2026-10-23 14:28 CST：三模块八组完整分析完成（覆盖较早等待状态）

formal_seed0_20261023_01已完整结束：八组各240epoch/12660更新/0跳步，16路完整validation/test及F111 task0/2/7四路关闭归因全部exit0，completion为complete、无failure。远程执行树clean、HEAD0fb85f1，原primary153行/test-only5行及SHA保持。598份非checkpoint产物同步本地并SHA全等；128份checkpoint仅记录SHA未下载。独立CPU复算652项逐类AP最大误差3.98e-13，八组task7 val存档逐值重放；本轮无新模型前向/训练。

按基线/A/B/C/AB/AC/BC/ABC，最终val为42.5650/42.4311/42.4323/42.6478/42.9530/41.8702/42.4916/42.3712；最终test为31.9391/31.8358/32.2804/32.1179/32.4286/31.8614/32.2555/32.1765。AB最佳最终val/test，较本轮基线+.3880/+.4896，阶段平均48.5646/35.7581（基线48.5026/35.4273）；C单独val平均48.8647最高，ABC test平均35.8087最高。不能称AB四指标全面最优，也不支持三模块普遍叠加。

ABC−AB的task6三类对26类mAP贡献val−.6332/test−.3875，其余23类分别+.0515/+.1354；两split都集中小样本task6。F111 task7关A/B/C/全关val为42.3373/42.2222/42.3755/42.2100（全开42.3712），关C未恢复AB42.9530或task6三类。主要差异留在共同训练的任务参数，不能断言即时位置bias过强或旧任务漂移。AB的Full两split变强，Person端点未同步增强，融合−Full只小增；不能宣称人物对应/互补已解决。

1000次原图配对bootstrap：AB−基线val95%[−.3395,+1.0274]、test[+.0793,+.8921]；AB−Attention的test区间跨0，ABC−AB与全部F111开关区间均跨0。仅固定seed0的样本不确定性、无多重校正，不代表训练seed稳定或机制已证实。不要用历史全TRAIN B0替代本轮90%fit匹配基线；历史test阶段平均36.2781仍高于本轮所有组。

完整报告docs/three_module_stacking_complete_results_20261023.md，复算/分层/逐类/CI及同步清单在output/three_module_stacking/analysis_20261023_complete_01。下一候选是现有AB checkpoint0/2/7在validation做开关A/B/全关，尚未执行；当前不训练新三模块、不扩seed、不做test调参。本次只改本地分析/上下文，无业务代码变化、提交/推送/合并。

## 2026-10-23 12:02 CST：八组训练结束，完整评估仍在运行；阶段性分析

12:07补记：F010完整validation也已结束，最终42.4323、阶段平均48.3613；新增17份正式val产物同步本地并SHA全等（合计309份），task7与训练归档逐值一致。当前3/16完整评估作业结束，test及关闭归因仍未齐全。

本条覆盖此前“首批刚启动”的状态。formal_seed0_20261023_01八组均240epoch/12660更新/0跳步，预算、原图holdout隔离、旧参数冻结及compact/完整恢复一致核验通过。远程执行分支exp/three-module-stacking-20261022、HEAD0fb85f1、clean；primary153行/test-only5行及原SHA保持，不改远程锁定代码或其他作业。

正式16路val/test及F111 task0/2/7关闭归因仍在GPU0自动队列，截至12:02只有F000/F100完整val结束，无completion/failure。八组train存档task7均完整val2397人/1705图，sample_ids/targets一致，CPU复算指标吻合；F000/F100正式task7预测与训练存档逐值相同。不要把早期不同人群的训练8阶段均值当正式平均，也不要填造test。

最终val按F000/100/010/001/110/101/011/111为42.5650/42.4311/42.4323/42.6478/42.9530/41.8702/42.4916/42.3712。A+B相对本轮基线+.3880，1000次原图bootstrap95%[−.3395,+1.0274]；ABC对AB−.5817，[−1.3295,+.1306]；全部单/组合对基线区间跨0，不能宣布成功或稳定负交互。ABC-AB的task6三类对26类均值贡献−.6332，其余23类合计+.0515；优先检查小样本任务，而非泛化为全局融合损坏。

A改变Summarize K/V但不写Frozen流，C改变Full读取权重，B以任务Selectors Q读Full摘要再写Person，属于相依证据链。B训练残差非零且较大，C注意力KL较小/未饱和，A作用任务差异大；推理因果仍须已排开关。292份完成非checkpoint产物同步本地并SHA全等；CPU分析脚本/JSON在output/three_module_stacking/analysis_20261023_partial_01。报告docs/three_module_stacking_partial_results_20261023.md。只更新本地分析/上下文，无新增训练/提交/推送/合并。

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

## 2026-10-22：现有三模块叠加消融详细方案，尚未实施

用户最新目标是验证Image-token Adapter、摘要Attention、目标人物位置摘要能否叠加，不实施此前新Query设计。方案docs/three_module_stacking_ablation_plan_20261022.md：完整2^3八组，顺序同批基线及三个单模块→三个双模块→全开。建议锁定A=image_token shared、索引1/b32/ReLU/scale0.1；B=task_full Selector Query、索引5/4头/总宽64/scale0.1；C=target原6维几何、12层共享60参数/任务。ABC新增246620参数/任务，共1972960。

统一seed0、EMOTIC8任务、原图90/10 fit/holdout、30epoch/task、B64、Adam/cosine、主LR.0125/模块LR.0004、joint BCE＋aux.1、无ASL；所有格子同批重新训练，历史全TRAIN/aux0 Attention或Adapter ASL不能直接填入。每组预期12660更新，八组101280。保留原三Task Forward、固定特征融合及任务冻结；A只改Selector读取支路，不能写回Frozen Forward。

代码支持缺口：model/runner有互斥和B aux0限制；_lane_block的A分支当前没有消费空间bias；组合需统一单次读取softmax、正确Full摘要缓存、参数组及评估/恢复配置，不是简单增加三个CLI开关。后续先测八种开关/配对初始化/冻结/恢复及最重配置显存，再按GPU约束调度。

主表八组8阶段完整val/test；F111 task0/2/7 val做全开/关A/关B/关C/全关。区分重训增量、条件推理贡献和交互，配对bootstrap不能替代训练seed。两两阴性不自动删格或跳过三模块。首轮不额外设计Query、绑定loss或调组合专属超参。本轮仅制定方案，无模型修改、提交推送或实验启动；本条为最新研究方向。

## 2026-10-22：对比摘要Attention与目标位置摘要，讨论结合方向（未实施）

核对两轮报告：摘要Attention使用全部TRAIN、13950更新、辅助视图BCE=0；目标位置摘要使用原图90% fit、12660更新、辅助BCE=0.1。两轮只可分别对匹配基线归因，不能按绝对mAP认定优劣或把增量相加。任务Selector Query开启/关闭最终val42.8976/42.9132、test32.4993/32.5461；人物Query关闭后改善不等于该残差有效。目标位置开启/关闭最终val42.9717/42.9723，未做目标关闭test。共同现象支持进一步研究共同训练参数/优化路径，但尚未证明训练辅助或正则化机制有效。

结构上可组合：旧Attention在摘要后向Person写入Full生成特征；位置方案在Full Summarize前调整原patch读取分数。二者分别研究跨视图证据传递与人物条件证据选择。更合理的融合假设是让任务语义、Person内容和人物几何共同决定Full原patch的摘要权重，保留原Full Value及Person/Face独立Task Forward；不是直接叠加已无推理净收益的旧残差，也不是平均两套checkpoint。此为未验证新设计，当前几何与Query正确身份优势均未建立。

保留下一步只读整图位置task0/2/7关闭归因建议；如用户后续授权新训练，先统一split/辅助监督/初始化/预算，做无条件项、仅几何、仅人物内容、二者联合的匹配控制，并在TRAIN原图holdout与validation检查开关贡献及同图正确/交换条件；不据test选方案。仅更新分析上下文，无代码修改、提交推送或实验启动。

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

## 2026-10-22：目标人物空间摘要完整实验已启动

正式批次formal_seed0_20261022_01已通过tmux spatial_summary_formal_seed0_20261022启动，源码锁定b0d826342a55165f805ff25eec846df5688296ea，两端同分支codex/target-anchored-summary-20261022、upstream一致。GPU4/5/6分别baseline/absolute/target，训练PID776820/776823/776826均存活并进入task0，初始底座SHA相同；尚未产生最终性能结论。每组完整8任务×30epoch、B64、seed0，保持TRAIN原图90/10隔离、12660成功更新预算和aux.1。

训练后自动进行跨组初始化/采样/恢复/旧参数审计，再于GPU1–6并行完整validation/test，最后target task0/2/7做完整validation空间项关闭与同图框交换。预检及两轮短跑155份静态日志/结果已同步本地并逐文件SHA256一致；checkpoint保留服务器。原主树153行/test-only5行状态SHA与启动前相同。当前代码不再修改；本条及执行文档的启动状态仅本地更新，不改变已锁定的远程运行版本。

2026-10-22正式启动前检查全部通过：415项完整回归；真实CLIP零参数/历史task0/2/7逐值重放；smoke_seed0_20261022_02三组各4更新/0跳步、原图隔离/初始化和采样配对/旧参数与恢复核验；六评估进程及总batch exit0，正常预测与训练时存档逐值重放，框交换/关闭只改Full而Person/Face固定。33可交换者的Full在task0/1分别31/33、33/33人变化。smoke01与02的六份方法状态逐值相同，确认修复未改训练轨迹。实测训练峰值baseline2575.41、两空间组2576.23MiB，保留约2710–2714MiB。
准备formal_seed0_20261022_01，GPU4/5/6三组并行完整8任务，每组12660更新，训练后六split评估GPU1–6，每卡单进程/内存上限16%/余量门槛5600MiB；原图90/10留出及aux.1保持，无绑定loss。用户8任务要求覆盖先导性能晋级门槛，不因中途指标变更条件。当前正式待调度，尚无最终结果。


2026-10-22短跑_01：三组各2task×2updates均完成、0跳步；空间参数梯度有限非零。验证器用compact的weights_only读取完整恢复文件，因TorchVersion/RNG元数据被严格拒绝，批次exit1且未开始评估；已修复为仅读取本实验可信完整快照的显式加载并补回归，失败产物保留。顺便将无效/不可见框的目标覆盖诊断归零，预测及优化公式不变。拟smoke_02重新跑全链路，通过才正式启动。


2026-10-22真实预检通过：CLIP/真实val首batch64上absolute/target零参数与原方法逐值相同；默认路径对历史task0/2/7预测逐值重放，源SHA与冻结视觉不变；非零空间参数交换框只改变Full，Person/Face不变。真实检查峰值分配760.5MiB/保留1044MiB。补充配对Dataset仅换框且保留本人可见性/像素/标签回归，准备最新414项全测与三组2任务短跑；正式尚未启动。


2026-10-22实现验证更新：新增10项空间机制测试及远程413项完整回归通过；源码9f9f6de已通过Git-only同步新远程worktree。原主树和test-only保持原样。新增三组批次控制器、恢复/旧参数/原图holdout核验、只读空间关闭/框交换评估，以及真实CLIP零初始化与历史task0/2/7重放检查；正式尚未启动，下一步真实预检和2任务×2更新短跑。训练GPU4/5/6、评估GPU1–6拟每卡1进程，每进程Torch分配器上限16%（约3.9GiB），启动要求至少5600MiB余量；不足等待，不干预外部进程。


## 2026-10-22：开始实现目标人物空间摘要，用户要求完整8任务双split

新隔离分支codex/target-anchored-summary-20261022，起点e81f2f2；旧分析/计划及原主/test-only工作树保留。用户要求8task覆盖原三任务先导，保留原图90/10 fit/holdout；三组disabled/absolute/target，seed0、8×30ep、B64、12660更新，主LR.0125/空间LR.0004、aux.1，禁用MoE/ParaX/adapter/旧cross-view。未来训练后并行完整val2397/test5368，不启用绑定损失。
已实现未验证：未截断真实crop/flip框元数据、Full原汇聚logits几何偏置、60参数/task跨层共享、零初始化/无效回退、优化器/保存恢复/旧任务保护；新增机制测试。当前尚未正式运行，先Git-only远程完整单测/真实smoke。完整配置docs/target_anchored_summary_execution_20261022.md。


## 2026-10-22研究计划：目标人物锚定的Patch Summarization，尚未实现

用户要求针对当前失败机制仔细研究下一阶段，并明确先不改代码。本轮只读论文/当前实现/已有validation预测，新增docs/target_anchored_patch_summarization_plan_20261022.md及ignored只读统计JSON，未修改业务代码、配置、训练脚本、源checkpoint；未新建实现分支、提交推送、启动GPU或访问新的test。

新核对：aux0匹配基线完整val2397人/1705图，同图多人527图/1219人，其中519图/1203人有标签集合不同；692次同图与第一人比较，Full logits逐值完全相同，Person平均绝对差2.4388。说明Full提供共用场景判断，不自动为指定人物选择不同证据；不是Full无用或提分已被保证。来源SHA及统计在output/target_anchored_summarization/planning_20261022。

建议首个新假设：保留三路Task Forward，在Full Summarize的softmax前加入“人物框–patch相对几何”标量偏置；原语义Selectors和值tokens不改，不再向Person写生成残差。每任务10×6=60空间参数，12层共享，八任务480；全零初始退化B0；按真实Full crop/flip映射未截断框、可见性连续门控，完全不可见回退。保持原MSA/Prefix-Tuning/D&R/分类头/末端权重，所有旧依赖完整封存。

先导拟P0原方法/P1无目标位置但相同visibility门控/P2真实目标位置三组，seed0 task0–2、TRAIN原图90/10 holdout、30ep/B64、主LR.0125/空间LR.0004、融合BCE+aux.1统一监督；先不test。P1/P2同参数初始化；全部需新匹配训练，不能复用fullTRAIN点值冒充控制。正确框/关闭空间项/只换框且保留本人的visibility和Person/Face验证净贡献。门槛及风险见计划，不自动启动。

DSCT/SCG仅借鉴空间与语义关系原则，不照搬检测器；本方案首先是融合前的目标相关证据提取，尚不是已验证三路交互创新。只有机制先导通过、TRAIN当前类配对量足够，才单独比较同图人物当前类绑定监督；不将全26类或同情绪人物当负样本。历史val/test已反复观察，不能称独立未见验证。

---


## 2026-10-22最新结论：跨视图Attention完整8任务归因结束

formal_seed0_20261021_01四进程及batch均exit0，全部8边界、val2397/test5368固定全人群结束；batch verified。100份正式结果/日志同步本地，逐文件SHA一致。训练/新模型前向均未在本次分析中执行，源checkpoint及参数前后不变；正式归因评估本身为只读前向。

任务Selector Query on/off：val最终42.8976/42.9132、平均48.8781/48.8744；test最终32.4993/32.5461、平均35.9290/35.9438，未显示推理残差净收益。人物摘要Query on/off：val最终42.1003/43.1086、平均47.8098/48.9567；test最终31.7499/32.3252、平均35.1945/35.7301；完整人群全部8阶段关闭更好。人物端点关闭恢复val5.9295/test4.4624，Full/Face逐值不变。

同图可见多人Query交换：373人/175val图、811人/372test图；最终可交换子集本人/另一人Query mAP为39.3885/39.6085、27.9853/28.2418。所有可交换者预测均改变，未显示正确人物Query优势；几何不合格者完全不变。标签不同者描述性分层仍无原Query总体优势，不能因此声称反向优势或所有人物条件方法无效。

validation原图配对bootstrap500次：人物Query关闭−开启+1.0083，95%[.6946,1.3213]；任务Query+.0157，[-.1622,.1920]；交换者swap−原+.2200，[-.6086,1.7794]。固定seed0人群抽样区间，不替代训练seed重复。关闭后相对aux0基线点估计优势仍在，但区间跨0，不能宣称训练策略成功。

主要解释：有内容依赖不等于学到人物对应；实际残差/底座.262–1.407，可能改写人物证据；Full重复注入与仅融合BCE缺少互补性约束。摘要压缩/D&R是共同限制而非已证实唯一原因。task0已见损害，当前不是新增遗忘证据。存在局部类别及早期test可交换子集受益，不能说每个样本都受损。

结论：停止扩展本版人物摘要写回方案；任务Query不作为有效推理融合模块；保留B0。后续若研究训练后移除分支，须视为独立新假设，不能直接当新基线；本轮不启动新训练/不改超参/不合并main。
报告docs/cross_view_attention_attribution_results_20261022.md；可复核JSON/脚本output/cross_view_attribution/analysis_20261022。当前分支codex/cross-view-attention-attribution-20261021，HEAD e81f2f2；分析仅本地文档，未提交推送，业务源码未改。原主树/test-only/source状态与启动一致。

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

机制证据：Person Query/attention随图像变化且已训练；实际残差/摘要范数比.262–1.407（系数.1已计入），人物端点val39.0175→35.0551。离线移除Person项的条件收益：基线+0.5501，人物Query−0.3619。不是分支未动；完整冻结保护也未解决当前任务习得/泛化。任务Query test Full本身+.4895，融合总+.4501，不能直接宣称互补性增强。摘要压缩/D&R为两组共同限制，不足以独立解释差距。

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

## 2026-10-18：WAF/AF正式结束与完整分析

GPU1 WAF（ceded63）和GPU0 AF（07fd35c）均8task×30epochs、13950更新、0跳步，
整体/val/test全部退出0，checkpoint来源、累计冻结、CLIP/旧冻结权重不变审计通过。
固定全人群val2397/test5368，与fresh B0/v1的ID/targets/class及配对初始化/样本SHA
逐项一致。四组157份JSON/NPZ同步并与新服务器SHA核对，16个AF/WAF checkpoint
当前SHA与verification/val/test记录交叉一致（checkpoint留服务器）。
B0最终val/test=42.1912/32.0610，AF=42.4007/31.7904，WAF=42.2952/31.6815；
阶段均值分别48.4382/36.2781、48.0956/35.1105、47.9071/35.0797。WAF−AF四项均
负，但500次原图配对bootstrap的两组差区间跨0；两版test全部8阶段低B0，阶段均值
相对B0的test区间均为负。只对应seed0，不能替代训练seed复验。
机制：两版完整val均23/24任务视图使用固定Top2；新只读64人probe确认raw CLS去均值
能量约0.07%–0.11%、gate多饱和。task0两版logits完全一致，test已低B0 2.8205。
AF/WAF最终冻结15/16名，WAF首次冻结[6,13]保护Person/Face，但Full依赖[4,9]均
未冻结。冻结旧Router不保护它仍访问的可变专家；test forgetting AF/WAF .2485/.3918，
高于B0 .0071/v1 .0090；固定旧类logit漂移及同边界train probe也明显增加。
GPU1只读probe4个checkpoint已完成，无更新且模型state/源checkpoint SHA不变，
3份probe产物同步SHA通过；首次manifest字段类型失败保留，无前向/更新。
决策保留B0，不扩seed/温度/容量、不新训练/合并main；若继续，先预锁边界做旧访问
可变Expert的内存回滚因果对照，再讨论完整依赖封存/快照及容量成本，不能只改分数。
完整报告：docs/moe_weighted_freeze_results_analysis_20261018.md；机器结果output/moe_weighted_freeze/analysis_20261018/。
本次仅分析/文档，业务代码及两个服务器tracked checkout保持不变。

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


## 2026-10-16：开始 MoE v3，用户授权8任务与val/test并行

从v2分析提交`2d6346f`创建`codex/moe-task-ffn-v3-20261016`。实现1共享+
3专门专家、无路由噪声和ST的soft Top-2，shared-only系数修正为0.5；
固定视图组同样激活两名专门专家。增加输出范数/CLS比值/夹角/软硬负载
日志和specialist-only validation消融。用户本次明确要求8-task val/test
并行，覆盖此前先3task/不访问test的建议；通过单测和三组smoke后GPU0
三组顺序训练，每组val/test两个进程同时评估。配置见
`docs/moe_task_ffn_v3_experiment_20261016.md`。
v3业务提交`6a4d585`远程296项Track-A单测通过，三组各2-update smoke
均0跳步、CLIP不变、checkpoint同人群重放差0；dynamic专门/共享比约
0.67–0.70，路径与日志均生效。新增并行启动/失败处理测试两项本地通过；
下一步同步新测试后启动正式批次。
最终298项远程全测通过。正式批次已在GPU0、tmux
`ml_moe_ffn_v3_20261016`启动，远程固定clean `7eb1f64`，三组run_id为
`formal_<mode>_seed0_20261016_01`。每组三组8任务训练后val/test并行评估；
两专门组随后val specialist-off。文档后续提交不在运行中更新服务器checkout。

## 2026-10-16：MoE v2 三组 validation 完成并分析

三组各完成8任务×30epochs、13950更新、0跳步、冻结CLIP不变，test
未加载。固定2397人 validation：dynamic 最终/阶段平均
42.2877/48.2518，shared_fixed 41.7671/48.9959，view_fixed
41.9757/47.7554；B0为42.1912/48.4382。dynamic 最终仅+B0
0.0965，原图分组bootstrap区间[-0.748,+1.007]，阶段平均反低0.1864。
Router 已按图片使用多个专家，但task7共享输出范数是专门输出的
11.6–13.4倍；同权重关闭dynamic残差最终只降0.0058，说明专门路由
没有形成实质预测贡献。下一版优先缩为1共享+3专门、训练推理一致的
soft Top-2、匹配控制组共享系数和活跃容量；先task0–2 validation，
不做test/跨任务复用/main合并。详见
`docs/moe_task_ffn_v2_results_analysis_20261016.md`。

## 2026-10-16：MoE v2 正式 validation 中途状态

GPU0 顺序批次的 dynamic 与 shared_fixed 两组均已完成8任务训练和固定
2397人全人群 validation，各13950成功更新、0跳步。dynamic 的最终/
阶段平均 mAP 为42.2877/48.2518，shared_fixed 为41.7671/48.9959。
view_fixed 巡检时在 task6 第7/30 epoch，已完成 task0–5，尚无正式
全人群验证结果；三组结论需等待其完成后统一分析。test 未启动。

## 2026-10-16：Task-Forward MoE v2 独立实验分支

从首版分析提交 `b8fe8ea` 创建
`codex/moe-task-ffn-v2-20261016`，保持索引 1 的 Task Forward
MSA/Drop & Replace 后 FFN 并联位置与冻结图像流。v2 Router 使用本块
Selector 汇总的归一化图片证据；每任务 7 专家中一名三视图必经共享专家，
另按内容选一名专门专家。提供固定共享及固定视图路由控制，训练期门控
噪声退火、温度与轻量负载约束。先做真实 val 图片证据探针与 smoke，
再以同预算三组固定人群 validation 判定，暂不读取新方案 test 或合并
main。协议见 `docs/moe_task_ffn_v2_experiment_20261016.md`。
远程初轮 8 项相关单测仅有一处测试期望类别数笔误，已修正并复测通过。
`3b5612d` 已在服务器独立 v2 worktree 复测：292 项 Track-A 单测全过。
未训练模型固定 val 前256人的图片证据相邻距离分别为
Full0.3747/Person0.5803/Face0.5650，而 raw CLS 仅
0.0125/0.0173/0.0233；三视图证据均超过预定门槛。初始路由已有多名
专门专家被访问，但尚不能据此判定泛化收益。
dynamic/shared_fixed/view_fixed 三组 task0 两更新 smoke 均成功、0跳步，
固定全人群 val mAP 分别25.1634/25.1626/25.1681；CLIP SHA 未变，
主参数初始 SHA 完全相同。动态组 adapter 梯度有限且非零，筛选人群
checkpoint 重放与训练末尾26.687860逐值相同。可进入三组同预算正式
validation 训练，test 暂缓。
正式三组 seed0 validation 批次已在服务器 v2 独立 worktree 的 tmux
`ml_moe_ffn_v2_20261016` 顺序启动，训练代码固定 `eb90978`；
当前先运行 dynamic，随后 shared_fixed 和 view_fixed。正式输出在
`./output/moe_task_ffn_v2/formal_<mode>_seed0_20261016/`，日志在
对应 `./logs/`。不读取 test。

## 2026-10-15：MoE 首版最终诊断结论

全人群重评估、task7 val Router FP32 探针和同权重 MoE-off
验证集消融已完成。路由几乎是每视图固定单专家，6/8任务三视图
完全无专家重叠；关闭 MoE 后 task7 val fused mAP 从42.0342
升至42.5185（配对 bootstrap 95%区间[+0.0948,+0.9536]），
Full/Person 分支分别提高0.8786/2.1521。首版尚无收益，下一版
优先换成图片证据驱动的小专家池、显式共享专家和训练/推理对齐的
门控，保留 Task Forward 插入点和旧任务整池冻结。详见
`docs/moe_task_ffn_v1_results_analysis_20261015.md`。

## 2026-10-15：MoE 首版完整人群复核及残差消融

正式训练固定在 `17a9b84`、8任务×30 epochs、13950更新、0跳步。
原评估按已见正例筛选早期任务人群；用冻结 checkpoint 对固定
2397 val / 5368 test 人群重评估，两路 exit 0。相对配对 B0，
val 最终42.0342 vs42.1912、阶段均值47.9082 vs48.4382；
test 最终31.7789 vs32.0610、阶段均值35.1268 vs36.2781。
8个 test 阶段均低于 B0，差距从 task0 的−2.8205 收窄到 task7
的−0.2822。所有任务/视图推理均固定选择2名专家，22专家池
三视图合计只使用5–6名。新增只读 validation MoE-off 重放开关，
用于诊断残差即时作用，详见实验报告。

## 2026-10-15：Task-Forward MoE 正式 8 任务完成，进行评估口径复核

GPU0 批次 `moe_task_ffn_v1_seed0_20261015_01` 在独立远程 worktree、
clean commit `17a9b84` 完成 8 tasks×30 epochs、13950 更新、0 跳步；
CLIP 权重未变，峰值保留显存 3022 MiB。原 val/test 阶段评估按已见类别
正例筛选人群，只有 task7 可与旧 B0 完整人群直接比较：val 42.0342
vs 42.1912，test 31.7789 vs 32.0610。全部 task/view 的评估 Router
均仅使用固定两名专家，没有跨图片专家切换。正在补完整人群重评估，之后
才能比较阶段均值和遗忘。详见 `docs/moe_task_ffn_v1_experiment_20261015.md`。


## 2026-10-15：Task Forward FFN 并联 MoE 首版实施中

用户明确要求在独立于 ParaX 的新分支实现第一个版本，并在 GPU0 做
8 任务顺序训练和随后 validation/test 并行评估。实施设计和完整锁定配置见
`docs/moe_task_ffn_v1_experiment_20261015.md`。新模块
`multi_lane/track_a/moe_task_ffn.py`：索引1的 Drop & Replace 后 raw task lane
进入共享任务内专家池，每任务/视图独立 noisy Top-2 CLS Router，整任务冻结。
新增训练参数、compact checkpoint 评估脚本和复现 shell 脚本。当前尚未运行
smoke/full，也尚无新指标；与下文上一轮“仅分析”状态按日期顺序区分。

## 2026-10-15：追问后改按独立 MoE-Adapters 核心整体迁移分析

用户要求以MULTI-LANE论文术语解释位置，并参照MoE-Adapters官方代码分析尽量全部迁移。补充报告 `docs/moe_adapters_full_transfer_and_multilane_location_20261015.md`。上轮“Selector图像证据路径”准确对应Figure2 Frozen Forward→Summarize的输入连线（索引1为第2块读取前），不是原法FFN并联位置。旧方案每任务独立池也不包含原法跨任务复用，已标为历史保守变体。

本方向当前主要候选：Task Forward在MSA/Drop & Replace残差后、FFN旁并联完整输出MoE；每层一个跨任务/视图共享池，task/view gate与noise，分析原频率activate-freeze及独立保护版。文本塔和DDAS不能原封搬到当前多标签/all-lane协议。固定官方commit源码审计发现down硬编码64、任务/层/AE数量硬编码、全局task路由、noise训练开关、未并入训练的CV项及compact漏存风险；22×b64×12层视觉专家与8×3路由/noise约35.904M新增参数。此为独立MoE-Adapters方法，不以ParaX结论替代其效果。无源码/配置修改、无运行实验、无commit/push/服务器操作。

## 2026-10-15：MoE-Adapters 三视图迁移分析，新独立分支

用户要求新建分支并先分析，不启动实验。新分支 `codex/moe-adapters-three-view-20261015`，基点/HEAD `25fc8617b20f60e96ee57a7b4a4827e048014501`，无 upstream、未提交推送；独立工作树 `/Users/denghaoyuan/.codex/worktrees/moe-adapters-three-view-20261015/multi-lane-main`。主工作区未提交文档和现有九个 worktree 保留；未操作服务器。

分析见 `docs/moe_adapters_three_view_transfer_feasibility_20261015.md`。工程上可行：首选沿用索引1的Selector图像证据路径，每任务一个完整输出专家池、Full/Person/Face三个内容相关Router，固定最终融合；先检验任务内跨视图共享，再讨论读取冻结旧专家。论文高频Top-2冻结不是严格旧函数保护，后续必须保存所有旧依赖和访问mask。EMOTIC多标签/all-lane推理不直接引入DDAS单任务选择。

已重新阅读CVPR原论文和固定commit官方MTIL源码，并参考主工作区未提交的最新结果：静态重叠v1 val/test42.7858/31.9202、v2 42.0629/32.0727，对应B0 42.1912/32.0610，尚无稳定净收益；后置旧函数三seed val均值42.6780低于B0 42.6979。动态视图路由仍是待验证新假设。旧基点下文为历史记录，不代表当前队列；本次只维护分析和上下文，无业务代码/配置/模型修改，无训练、测试或新增实验结果。

## 2026-10-10：ParaX 新版验证全部通过，准备正式四组

业务代码c440321；本地/服务器287项单测通过。新worktree `multi-lane-main-parax-audit-paired-staged` clean。四组真实ViT三task smoke各12updates/zero skipped，配对初始化/采样、identity、旧参数保护和新增残差关闭比较通过，peak reserved2712/2754MiB。原30epochs/task task1 checkpoint重建task2，新审计三组均通过且源SHA不变，0训练更新。分阶段真实smoke完成B0 12updates＋train缓存校准167updates，当前新专家实际改变/梯度非零、B0和旧参数hash不变，与基线配对一致。79个静态产物同步SHA一致；正式分阶段未启动。接下来从头运行batch `protected_parax_audit_v2_seed0_val_20261010_01`，GPU0四组、seed0 task0–2、30epochs/task、OMP1，validation-only，fixed fusion/.02界，不持续监督。配置及产物详见 `docs/protected_parax_audit_paired_staged_plan_20261010.md`。

## 2026-10-10：执行 ParaX 审计修正、残差消融与分阶段实现

新分支 `exp/parax-audit-paired-staged` 从de9b0f8开始，保留全部既有未提交修改。受保护后置ParaX采用精确零残差identity且梯度有效的归一化表达；审计分离零投影/raw delta/缓存特征与独立完整前向重复误差，旧参数/mask逐值检查不变。新增每task关闭全部后置残差的同模型推理分数、固定旧人物/新旧类AP汇总。分阶段能力已实现：B0阶段不含ParaX优化参数，train缓存后冻结B0，只学当前新专家/Router/投影；旧专家保护，非冻结随机center旧方案。正式先统一新代码四组task0–2 seed0 validation、30epochs、GPU0；不加载test、不扩大8-task、不自动启动正式分阶段。完整配置 `docs/protected_parax_audit_paired_staged_plan_20261010.md`。本地287项单测已通过，新增五项机制检查；准备远程全测及真实四组/分阶段smoke，检查通过后正式启动，不持续监督。

## 2026-09-25：P-post-zero 闭环与后置校准转向

`parax_post_zero_control_20260925_020000` 已完成。B0 与 P-post-zero 的 task0--2 mAP、average mAP、forgetting、F1、所有视图指标、class AP 和 pairwise 统计完全相同，差异为0；ParaX residual ratio/gradient 也为0，初始 logits diff `2.33e-08`。这证明完整训练可比性没有隐藏问题，P-post-tiny 的 `-0.5319` mAP 确实来自非零残差。详细报告见 `docs/parax_post_zero_results_20260925.md`。

ParaX image-stream 路线停止：small/static/tiny 都没有超过 B0，scale 降低只会逼近 identity，不能形成有效收益。阶段四蒸馏与阶段五 level routing 不再执行。下一步转向独立后置特征校准：关闭 ParaX，保持 Full anchor，在冻结 CLIP 输出后的融合模块中只训练 task-local、zero-initialized Person/Face residual，验证是否能利用辅助视图而不改写共享 image stream。

## 2026-09-25：P-post-tiny 结果与 strict identity control

批次 `parax_post_tiny_control_20260925_010500` 已同步。P-post-tiny final/average mAP `44.7075/54.1956`、forgetting `2.3772`，比 P-post-small 提升 `0.4115/0.4686`，但仍低于 B0 final `0.5319`。task0 mAP `59.5844`，低 B0 `0.9381`。最终 residual/token ratio 仅约 `1.7e-7`，gate entropy 约 `1.094`、view gate L1 `0.009`，说明 scale `0.0001` 已几乎 identity，同时路由也几乎没有有效区分。task2 Face 单路比 B0 高 `0.7680`，但 Full/Person/reliable-Face 下降，固定融合仍未受益。详细报告见 `docs/parax_post_tiny_results_20260925.md`。

下一步加入 `P-post-zero`：保留 ParaX 模块和 optimizer group，但 fixed output scale 为 `0`，验证完整训练过程能否严格复现 B0。若 zero 仍偏离 B0，说明 paired 可比性还有隐藏数值/优化轨迹差异；若 zero 对齐，则确认 tiny 的剩余损失来自真实残差并停止 ParaX image-stream 扩展。阶段四蒸馏与阶段五 level routing 继续暂停。

`P-post-zero` 真实 ViT smoke 已通过，初始 logits diff `2.33e-08`。批次 `parax_post_zero_control_20260925_020000` 已在 GPU0/1 启动 B0-paired/P-post-zero；seed0、task0--2、30 epochs/task、validation-only、无 checkpoint、禁止 test。

## 2026-09-25：P-post-static 结果与 P-post-tiny 下一步

静态对照批次 `parax_post_static_control_20260925_003631` 已同步。B0 final/average mAP `45.2395/54.9341`、forgetting `2.7760`；P-post-static final/average `43.9191/52.8834`、forgetting `2.2440`。静态 uniform gate 比 dynamic P-post-small (`44.2960/53.7270`, forgetting `2.5475`) 更稳定但精度更低，说明动态路由提供了部分 task/view 适应，却没有解决共同的后置特征坐标失配。task0 静态 mAP 已低 B0 `3.0228`，阶段四蒸馏不能作为首要修复。详细结果见 `docs/parax_post_static_control_results_20260925.md`。

已加入 `P-post-tiny` 入口，保持 dynamic P-post 全部配置不变，仅将 fixed output scale 降至 `0.0001`，用于验证减小后置残差是否能恢复 task0。阶段五 level routing 继续暂停；若 tiny 仍低于 B0，将停止扩展 ParaX image stream，转向 view-specific post-feature calibration 或 logit-level residual。

## 2026-09-25：ParaX P-post control 结果与静态对照

批次 `parax_post_control_20260925_230104` 已同步。B0 与 P-post-small 均完成 task0--2、90 epochs、5010 updates、validation-only，无 OOM/NaN/traceback，未访问 test。P-post-small final/average mAP 为 `44.2960/53.7270`，B0 为 `45.2395/54.9341`；forgetting 为 `2.5475`，略优于 B0 的 `2.7760`，但 task0 mAP 已低 `1.3981`，说明后置残差本身在 task0 就改写了有效特征坐标。task2 Full/Person/Face/reliable-Face mAP 为 `42.3548/42.2803/33.2613/38.0635`，均低于 B0 的 `43.7058/42.5883/33.9794/39.4689`。最终 residual/token ratio 约为 `0.035/0.045/0.050`，gate 主要塌缩到 expert2，未形成可验证的 level-specific 收益。详细报告见 `docs/parax_post_control_results_20260925.md`。

结论：后置位置比 block10 插入更稳定，但仍不能提升精度；阶段四蒸馏暂不执行，因为蒸馏无法修复 task0 表征损失。当前分支 `exp/parax-post-static-control` 已加入 `P-post-static` 入口，下一步执行参数量匹配的 static post-encoder control，保持 rank32/3 experts/scale0.001/task0 后冻结 center，仅将动态 router 替换为 uniform gate。若 static 也低于 B0，则停止继续扩展 ParaX image stream，转向 strict identity 或 view-specific 受限后置专家。

静态入口已修正为独立的 `post_static` 模式，避免误用原有 block-10 `static` 模式。smoke 初始 logits diff 为 `1.68e-08`。批次 `parax_post_static_control_20260925_003631` 已在 GPU0/1 启动 B0-paired 与 P-post-static，仍为 task0--2 validation-only、无 checkpoint、禁止 test。

## 2026-09-25：启动 ParaX post-encoder control

Frozen-center controlled residual 已结束：small final mAP `44.3814`（B0 `45.2395`），penalty final `43.5579`；penalty 已将 residual ratio 压至约 `0.006--0.009`，仍无恢复，说明继续调 penalty 信息价值低。新分支 `exp/parax-post-adapter-control` 提交 `e12a54b` 修正 P-post 的 forward 时序，改为保持冻结 CLIP/lane encoder 顺序后在最终 lane feature 追加 ParaX；239 项单测、P-post smoke 通过，初始 logits diff `1.86e-08`。批次 `parax_post_control_20260925_230104` 已启动 B0-paired/P-post-small validation。

## 2026-09-24：启动 Frozen-center controlled residual 阶段

分支 `exp/parax-frozen-center-controlled-residual`（提交 `c71da1d`）已完成服务器 239 项单测和 strict zero-output smoke，初始 logits diff 约 `1.9e-08`。唯一 batch `parax_frozen_controlled_20260924_220444` 已在 GPU0/1/2 启动 B0-paired、Frozen-center-small、Frozen-center-penalty 三组；仅 validation、无 checkpoint、禁止 test。

## 2026-09-24：ParaX shared-center stability 阶段三结果

batch `parax_center_stability_20260924_173712` 已同步。Shared-live/Frozen-center/Task-local-delta final mAP 为 `39.2991/43.0695/42.6452`，相对 B0 `45.2395` 分别为 `-5.9403/-2.1700/-2.5943`；forgetting 为 `8.1971/3.3804/6.7104`，B0 为 `2.7760`。Shared-live 与 Frozen-center 的 task0 完全相同、task1 起分化，说明冻结 shared expert center 能显著控制跨 task 漂移；但 Frozen-center 仍低于 B0，Task-local gate 也重新引入较大残差。阶段3完成定位但未达到收益门槛。详细报告见 `docs/parax_center_stability_results_20260924.md`。下一步仅保留 Frozen-center，测试受控小残差和必要时 Full-only CPU-cache 蒸馏；阶段5 level routing 继续暂停。

## 2026-09-24：启动 ParaX shared-center stability 阶段三

分支 `exp/parax-shared-center-stability`（提交 `2282b2a`）已完成 239 项服务器单测和真实 ViT smoke；strict zero-output initial logits diff 为 `1.49e-08`。服务器独立 worktree 为 `/mnt/haoyuan/workspace/multi-lane-main-parax-center-stability`。唯一 batch `parax_center_stability_20260924_173712` 已在 GPU0/1/2 启动 `Shared-live`、`Frozen-center`、`Task-local-delta` 三组，配置为 seed0、task0--2、30 epochs/task、P-10/rank32/3 experts、无 level embedding、validation-only、无 checkpoint/test。

## 2026-09-24：ParaX paired stability validation 结果

批次 `parax_paired_stability_20260924_161643` 已完成并同步到 `output/emotic_track_a_parax_paired/` 和 `logs/emotic_track_a_parax_paired/`。分支 `exp/parax-paired-stability`、提交 `6f4853f`。B0、P10-identity、P10-small、P10-zeroB 均为 task0--2 validation、90 epochs、5010 updates、0 skipped，无 OOM/NaN，未访问 test。

final/average mAP 分别为 B0 `45.2395/54.9341`、P10-small `44.3357/53.5132`、P10-zeroB `38.9207/49.9796`、P10-identity `38.6066/50.8907`；forgetting 分别为 `2.7760/2.6773/9.6834/10.9558`。P10-small 仅说明固定极小残差可控制遗忘，未带来收益；identity 与 zeroB 的实际残差均快速变大并显著退化。详细结果见 `docs/parax_paired_stability_results_20260924.md`。

五阶段状态：阶段1已完成 RNG fork、初始化模式和 ParaX 梯度诊断，但完整模型级 initial token/logit/hash/DataLoader 对齐仍未完成；阶段2 paired task0--2 已完成且所有 ParaX 组未超过 B0；阶段3 shared-live/frozen-center/task-local delta、阶段4低显存蒸馏和阶段5 level routing 尚未执行。下一步只补阶段1诊断并做阶段3短 validation，暂不恢复 level embedding、连续层、rank64、seed1/2 或 test。

## 2026-09-24：ParaX 根因定位与下一步方案

结合 shared-level validation 和 P-10 stability test，当前不能把 ParaX 的退化简单归因于动态路由容量不足。首要混淆因素是 ParaX bank 初始化未使用 `torch.random.fork_rng()`，导致 B0/P10 的 Selector、Prompt、classifier 和后续随机轨迹未严格配对；其次 `small` 仍是随机非零残差，ratio cap 会改变梯度；更核心的问题是同一份 ParaX expert/router 在增量 task 间持续更新，旧 task 的图像表征随之漂移。P10-router 的残差几乎为零，P10-experts 的 view gate L1 仅约 0.0132，说明路由与专家变换没有形成有效联动；Static-control 也失败，说明问题不只是动态 gate。

新增详细方案见 `docs/parax_root_cause_and_next_plan_20260924.md`。下一步只做阶段 0 可比性修复、identity/zero-init paired smoke 和 task0-2 短 validation；暂不恢复连续层、rank64、level embedding、seed1/2 或 test。当前本轮只更新分析文档，未修改业务代码、未启动实验。

## 2026-09-24：ParaX P-10 stability test 完成

test-only batch `parax_p10_stability_test_20260924_111500` 已同步。P10-small/P10-router/P10-experts 完整结束，final mAP 为 `31.2761/31.9207/31.7548`，冻结三视图 test 锚点为 `32.5365`；三组均未超过锚点。P10-router forgetting 最低 `4.9013`，但仍高于锚点 `4.7308`。ratio cap 将 task7 residual ratio 控制在约0.09；experts-only 的 view gate L1仅0.0132，router-only 无有效残差。P10-small-distill 在 task1因完整 teacher/student 三视图前向 OOM，结果不完整。详细报告见 `docs/parax_p10_stability_test_results_20260924.md`。下一步停止直接扩大 ParaX；如继续，先实现低显存蒸馏并只做短程 validation。

## 2026-09-24：开始 ParaX P-10 稳定性 test-only 实验

用户要求直接运行 held-out test，不跑 validation。已从 `exp/parax-level-routing` 创建 `exp/parax-p10-stability-test`。实现新增 ParaX trainable component 选择（all/router/experts）、固定输出 scale、residual/token ratio cap，以及按 task 快照的旧 task logit distillation；默认旧配置兼容。四组 test arm 为 P10-small、P10-router、P10-experts、P10-small-distill，均只用 P-10、rank32、3 experts、small init、fixed scale0.001、ratio cap0.10，蒸馏权重0.2。已完成本地 py_compile、bash -n、diff check；本机无 torch，服务器 ddp 全测与真实 smoke 尚未执行。实验协议见 `docs/parax_p10_stability_test_plan_20260924.md`。

## 2026-09-24：ParaX shared level routing seed0 结果

ParaX 六组 validation 已同步并完成分析。B0 final/average mAP 为 `42.9847/50.0234`；P-post、P-10、P-8:10、P-8:10-level、Static-control 的 final mAP 分别为 `36.3403/37.5118/31.3970/33.0375/33.9727`，全部低于 B0。所有组均完成 13,950 updates、0 skipped，无 OOM/NaN，未访问 test。P-10 虽然三视图 gate 有明显差异，但 residual/token ratio 已达 0.46–0.58；P-8:10-level layer10 的 Person/Face 残差比达 1.016/0.934。Static-control 同样退化，说明首要问题是残差幅度和跨 task 表征漂移，不是动态路由容量。forgetting 从 B0 1.0785 增至 P-10 6.5905、P-8:10 12.0309。当前停止连续层、rank64、seed1/2 和 test；下一步只设计 P-10 identity/小 scale、残差约束、冻结或蒸馏稳定性小实验，稳定后再恢复 level routing。详细结果见 `docs/parax_shared_level_routing_analysis_20260923.md`。

## 2026-09-23：停止视图专用 Selector，评估 ParaX 动态参数路由

用户确认停止个性化 Selector 路线，转而评估 ParaX（Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing）能否改善 Full/Person/Face 共享参数下的表征与融合。本轮只读检查当前代码和本地论文实现 `/Users/denghaoyuan/workspace/MyCode/ParaX-main`，未修改业务代码、配置或训练脚本，也未运行测试/实验。ParaX 原生 ViT 方案在每个 Transformer block 后对 patch tokens 施加输入条件化低秩残差，各层 Adapter 共享 expert-center 参数池，但各自 router 根据当前输入生成混合系数；CLS 不经过 ParaX。

首选评估位置是冻结 CLIP 图像 token 流内部、目标 ViT block 输出之后及下一 block 之前，优先覆盖能被后续 Selector 读取的高层 blocks；而不是 `task_lane` Adapter（它修改任务 token）或当前 `image_token` Adapter（只改变 Selector 匹配/patch 汇聚输入，不把更新结果传入后续 CLIP block）。当前图像流的 `no_grad()` 以及 Selector 所读 image tokens 的 `detach()` 会切断 ParaX 梯度，若实现内部插入必须显式调整 autograd 路径，但冻结的 CLIP 权重仍可保持 `requires_grad=False`。ParaX 的类增量旧任务稳定性、level 与 task 的路由混淆、初始化输出偏离原基线均为待验证风险。

研究建议先做 seed0 validation 分阶段对照：复现固定三视图融合基线；以冻结 backbone 上最终 tokens 的 ParaX adapter 作低风险位置对照；再测试 block 间 late ParaX（zero-based 8/9/10 等可被后续 Selector 使用的层），遵照论文共享 expert center、按输入动态路由，比较 token-only 与显式 level-conditioned routing。固定其余训练/数据/融合协议，观察分 view/task gate、梯度、残差幅度、参数量、旧 task 遗忘及每路/融合 mAP；出现稳定收益后再补参数量对照和 seed1/2。详细协议与选择逻辑见 `docs/parax_shared_level_routing_analysis_20260923.md`。当前尚未批准或启动实现/实验。

## 2026-09-16：开始实现视图专用 Selector 实验

用户确认按 Full/Person/Face 视图专用 Selector 方案开始修改。新分支为
`exp/view-specialized-selector`。已在 `MultiLaneModel` 增加 `selector_mode=shared/view_specific`，视图专用模式使用
`[task, view, selector, width]` 参数 bank，从同一共享初始化复制到三路，按视图选择并保持增量 task 隔离；默认
`shared` 行为保持兼容。runner、compact 状态构建和 GPU smoke 已接入 `selector_mode` 及可配置
`num_selectors`，新增视图专用 Selector 单测与预注册方案 `docs/view_specialized_selector_validation.md`。
已完成本地提交并推送；尚未运行真实 GPU smoke 或正式实验。

已提交并推送 `45a4bff` 后，服务器独立 worktree 的完整测试首次为 224/225；唯一失败是新增测试使用
`copy.deepcopy` 复制带 autograd 的参数切片。该测试已改为 `detach().clone()`，待重新推送并复跑完整测试。

## 2026-09-16：新增导师讨论版实验总结

新增`docs/three_view_shared_fusion_experiment_summary_for_advisor_20260916.md`，用通俗语言串联独立专家基线、
DGL式梯度隔离、视图专用小Adapter、融合公式对照、三路预测来源替换和Full独立分类头实验。文档明确区分
已定位的Full预测/Full-Body互补性问题与尚未证明的共享表示同质化，并给出Full完整私有Adapter的下一步建议。
本次只整理既有结果，未修改业务代码、未运行新实验。

## 2026-09-16：Full分类头独立诊断完成，直接拆head失败

用户确认按第一批结果推进最小Full机制实验。新分支`exp/full-private-head-diagnostic`，方案见
`docs/full_private_head_diagnostic_plan_20260915.md`。H0为共享head逐路logit融合，HF仅Full使用从共享head
复制初始化的独立head，Body/Face仍共享原head；其余P0协议完全固定。先做seed0两组配对，明确主效应后才补
seed1/2。已实现模型、runner参数、分数/compact导出、2000次bootstrap汇总及双GPUlauncher。服务器完整测试
221/221通过；H0/HF真实task0/1epoch配对smoke均完成84 updates、0 skipped，参数统计和compact状态符合预期。
正式batch`full_private_head_seed0_20260915_231942`已完成，实验HEAD`87b406f`。两组均240 epochs、13,950
updates、0 skipped，H0/HF/summary退出码均为0，无checkpoint/test。H0/HF logit final为
`42.6627/41.1145`，HF低`1.5482`；average低`1.9876`，Full单路低`1.3932`，Person/Face/可靠Face也全部
下降。1705个原图组2000次bootstrap的HF-H0区间为`[-2.6199,-0.5365]`。HF head已明显分化
（cosine 0.5853），故不能解释为未学开。直接Full独立分类head不受支持，不补seed1/2、不访问test。
H0比历史P0低0.4127，未过0.10复现门槛；逐路head再融合与历史先融合特征再过head虽实数等价，但AMP训练
轨迹发生漂移。因此H0/HF内部结论有效，不能将HF与历史P0作精确比较。完整报告见
`docs/full_private_head_results_20260916.md`。结果与日志已同步，本地/服务器summary SHA-256一致。

## P0差距第一批已完成并同步

批次`p0_r1_gap_seed0_20260915_1640`完成240 epochs/13950 updates/0 skipped，精确复现P0 final 43.075407。
报告见`docs/p0_r1_gap_results_20260915.md`。A：I-logit 43.445371、I-prob 43.581193、S-logit 43.075407、
S-prob 42.823759，原差距拆为0.757435−0.251648。B仅替换Full得到ISS 43.721708；Full独立来源四种
背景都更好，共享Face四种背景都更好，Body依赖Full背景。尚未确定参数模块层面的原因。
自动分析原因AMP代数误差阈值失败，已修复为单独报告舍入误差并成功完成2000次bootstrap；分数本身一致。
原差距95%图像组bootstrap区间跨0；仍须seed1/2验证。全部分数、compact状态和日志已同步本地，未启动第二批。

## 当前优先：解释P0相对R1的0.5058差距

用户要求先定位原因，方案见`docs/p0_r1_gap_diagnostic_plan.md`。已在`exp/p0-r1-gap-diagnostic`实现第一批，
尚未启动正式训练。
核对历史来源：P0相对独立专家Full=-0.6982、Person=+1.4158、Face全量=+1.3868、可靠Face=+1.9182；
不能用P2相对P0的全面退化解释此差距。R1 average mAP为50.7509。
代码确认：共享线性head且融合后不再归一化，P0特征融合等价于固定logit融合；R1为概率融合。
先复现一次P0保存compact状态与三路scores，再同预测比较融合运算和八种来源替换。剩余差距才进入head、
Adapter、Selector/Prompt解绑及目标对照。独立Face训练valid非ambiguous与P0 strict reliable差异须独立审计。
实现复用原P0既有诊断验证前向导出三路FP32 logits/概率、Face mask与IDs，并保存8个task compact状态；
离线分析固定权重计算A四格、B八组合、边际/Shapley、可靠性子集、排序纠错/破坏和2000次原图组bootstrap。
不搜索融合权重，不访问test，不包含第二批解绑。
服务器完整测试218/218通过；真实task0/1epoch smoke的普通/诊断fused分数逐元素一致，三路logit重建概率
最大误差2.5326e-4，compact状态成功读取。正式8-task P0已于2026-09-15启动：batch
`p0_r1_gap_seed0_20260915_1640`、GPU0、实验HEAD`79b86c6`；完成后自动运行A/B与2000次bootstrap。

## 2026-09-15：融合文献与机制解释修正

文献笔记见`docs/shared_view_fusion_literature_20260915.md`。候选为AdapterFusion两阶段组合、VLMo模块级共享、
MBT瓶颈token交互，MISA作为共享/私有表示备选；均为待验证方向。此前“正余弦排除DGL”和“弱监督是主因”的
解释过强：DGL也处理梯度幅度减弱，当前证据仅支持已测G2/P2未胜出；共享导致同质化尚未被证明。
P2的Adapter余弦混合共享/专用参数，需分开分析。独立R1概率融合与P0特征融合还存在运算差异。
本次只更新文献与上下文，未启动新训练。

## 2026-09-15：开始共享主体与视图专用Image-token Adapter阶段

用户确认执行DGL诊断后的下一阶段。新建`exp/view-specialized-shared-adapter`，先验证少量
Full/Person/Face专用参数能否在共享CLIP、selectors、prompts和head内保留视图互补性，本阶段不引入
动态Router或DGL。锁定seed0完整8-task validation三组：P0共享b32；P1参数量对照共享b45；P2共享
b32加三路各b4。每task Adapter参数分别为49,952/69,933/70,700，P1/P2仅差1.10%。

三组均使用固定可靠Face特征权重、joint gradient、0.1逐视图辅助监督及冠军BCE/Adapter-ASL协议。
新增对每task epoch0/14/29各前三batch的分视图VJP梯度审计，分别报告Selector/Prompt BCE与Adapter
ASL的融合路径/单视图范数、比例、余弦和三路路径间余弦，补足旧DGL只看task首batch合计共享梯度的
不足。只有P2 final分别超过P0/P1至少0.10、average与Standalone Full不低P0且缩小到独立R1的差距，
才进入Selector-aware动态Router。协议见`docs/view_specialized_shared_adapter_validation.md`。

实现提交`e5c52ca`及AMP诊断修复`32f8169`已推送。服务器新独立clean worktree完成213项全测；
P0/P1/P2真实EMOTIC task0一轮smoke均完成84 updates、0 skipped，三组参数计数严格符合协议。
修复后的3-batch审计中，共享表示的融合/单视图梯度范数比例约为Full/Person/Face
`0.685/0.174/0.130`，Adapter约`0.682/0.168/0.123`，两组参数的全部view梯度非零有限。
该结果只覆盖task0 epoch0，用于证明完整早中晚诊断值得执行，尚不作为DGL机制最终结论。
用户确认每卡约8GiB空闲足够叠加运行后，正式batch
`view_specialized_adapter_seed0_20260915_104759`在服务器clean `10ef477`的tmux
`ml_view_adapter_104759`中放宽为至少7,000MiB空闲、允许当前利用率并启动GPU0/1/2。三份config均已
落盘，commit、seed0、8-task validation、P0/P1/P2结构、无checkpoint/test均符合协议；首个epoch各完成
84 updates、0 skipped，耗时38--39秒，loss有限。每卡训练后约用19.5GiB、仍余约4.5GiB，无OOM或错误。

正式batch已完整结束：三组均240 epochs、13,950 updates、0 skipped，P0/P1/P2及汇总器exit0，未保存
checkpoint或访问test。final/average validation mAP分别为P0 `43.0754/49.9807`、P1
`42.7028/50.1309`、P2 `42.7961/49.7551`；P2比P0 final低`0.2793`，五项预注册门槛全部失败。
P2相对P0的Full/Person/Face/可靠Face单路分别下降`0.1549/0.1337/0.3033/0.7226`，且task1--7
累计mAP全部下降，没有建立更强的视图专家。固定融合仍比各自Full高`1.27--1.48`，说明互补性存在。
早中晚梯度审计显示不均衡幅度主要跟随固定融合系数，同视图梯度保持正向；P2跨视图负余弦少于P0却
性能更低，不支持DGL式冲突是主因。按协议停止本P2，不进Router、seed1/2或test。29个结果/控制/日志
文件已同步且三组组合SHA-256与服务器一致；完整报告见本地batch目录的`analysis.md`。

## 2026-09-13：OOF优势限定排序蒸馏失败，OOF蒸馏路线结束

从`83cceb2`新建`exp/oof-advantage-ranking-distillation`，仅实现一组E1 seed0 validation。
hard BCE恢复1.0，Adapter hard-label ASL不变；每个hard batch64附加一个确定性Full视图
pair batch16，pairwise logistic loss权重0.05。仅启用13个R1-Full OOF AP三折一致提升类别，
且pair必须满足Full OOF排错、R1 OOF纠正。task2/6无合格类，严格回退原训练。
详见`docs/oof_advantage_ranking_distillation.md`。

实现提交`0c0b753`已推送。服务器独立clean worktree
`/mnt/haoyuan/workspace/multi-lane-main-oof-advantage-ranking`完成203项全测；真实EMOTIC/OOF
GPU smoke确认13类及pair bank符合预注册审计，hard BCE、ranking loss和Adapter ASL均有限，
1 update/0 skipped，CLIP冻结，峰值显存约1587MiB。唯一batch
`oof_advantage_ranking_seed0_20260913_203305`完成240 epochs/13,950 updates/0 skipped，无OOM、
NaN、checkpoint或test访问。Full final/average/固定R1 final相对D0为
`-2.6785/-3.2720/-2.0639`，全部门槛失败。13个受ranking监督类final AP平均下降`3.8765`，
且ranking loss明显收敛，说明发生pair-bank过拟合而不是目标未优化。

自动比较器的专家训练目标一致性检查已修正，只允许E1的Full目标来源标签差异；数据、seed、任务、
预处理和评估约束仍严格检查，服务器204项全测通过。结果已同步本地并逐文件SHA-256一致。按预注册
规则不补seed1/2、不test、不搜索ranking/mix，正式结束OOF蒸馏路线。详见
`docs/oof_advantage_ranking_results_20260913.md`。

## 2026-09-13：D3 OOF R1教师直接BCE蒸馏失败

D3 batch `oof_r1_teacher_seed0_20260913_192759`完成240 epochs/13,950 updates/0 skipped，
无OOM/NaN/test访问。Full final/average/固定R1 final相对D0为
`-0.1689/-0.4322/-0.3271`，三项门槛均失败，不补seed1/2、不test。

强教师仍失败说明瓶颈在raw-probability BCE与mAP排序/随机裁剪不匹配，不再搜索mix。
若最后保留一次OOF教师尝试，应仅对13个三折优势一致类别、仅对R1纠正Full排序的
样本对增加小型排序损失，hard BCE保持1.0不稀释。详见
`docs/oof_r1_teacher_distillation_results_20260913.md`。

## 2026-09-13：开始D3 OOF固定R1集成教师

从`1640c50`新建`exp/oof-r1-teacher-distillation`，仅增加D3 seed0 validation。
可靠Face的OOF教师固定为`[0.64,0.16,0.20]`，无效Face为`[0.80,0.20,0]`；
mix仍为0.20，其他Full/Adapter/训练设置与D0/D2完全一致。仅对比已有D0，
禁止test，详见`docs/oof_r1_teacher_distillation.md`。
实现提交`b57f900`已推送，服务器独立worktree
`/mnt/haoyuan/workspace/multi-lane-main-oof-r1-teacher`的199项全测通过。真实OOF/GPU smoke确认
8-task覆盖一致、R1重建误差为0、三项loss有限。batch
`oof_r1_teacher_seed0_20260913_192759`已在GPU3启动并进入task0训练。

## 2026-09-13：OOF跨视图蒸馏D1/D2未通过门槛

batch `oof_distillation_seed0_20260913_230500`的D0/D1/D2均正常完成，未访问test。
D1 Person OOF的Full final/average/固定R1 final相对D0为`-0.2908/-1.1380/-0.4773`；
D2 Person+Face OOF为`+0.1928/-0.4150/-0.0754`，因此不补seed1/2、不test。

D2改善Full的Sadness/Sensitivity/Suffering `+4.5477/+0.5620/+3.7853`，证明跨视图
信息可蒸馏；但auxiliary-only OOF教师在8个task上均弱于Full OOF且BCE更高，导致
早期类别广泛损失。下一个最小实验应仅为D3：用在8 task上均强于Full OOF的无泄漏
固定R1 OOF集成教师，保持mix0.20和其他设置不变。详见
`docs/oof_cross_view_distillation_results_20260913.md`。

## 2026-09-13：实现OOF跨视图教师蒸馏Full lane

创建`exp/oof-cross-view-distillation`，复用已有三折image-group OOF Person/Face预测，不重训
教师。新增训练batch软目标和严格OOF样本ID/标签/类别/覆盖审计；主模型损失固定为
`0.8 * hard BCE + 0.2 * OOF soft BCE`，两项都使用legacy full-zero视图，Adapter仍只使用
真实标签ASL 9.8/0/0.05。

首轮seed0完整8-task validation比较D0 fresh Full、D1 Person OOF teacher、D2 Person+Face OOF
teacher。D2在可靠Face上按锁定R1辅助比例`4/9:5/9`合成，无效Face回退Person。候选只有
同时满足Full final `+0.05`、Full average不降、锁定R1 final `+0.05`才补seed1/2；本阶段
禁止test。详见`docs/oof_cross_view_distillation.md`。

服务器独立worktree `/mnt/haoyuan/workspace/multi-lane-main-oof-distillation`已创建，实现提交
`a3f7ee7`与测试修正`f3f978c`已同步。选择门槛回归测试提交`9fe1a58`后197项全测通过。真实OOF审计确认8个task覆盖
`[5353,4394,861,9931,4352,2536,627,1526]`与Full训练池一致；真实4样本GPU smoke的
硬BCE/软BCE/Adapter ASL及主参数/Adapter梯度均非零有限，CLIP冻结，峰值显存约572MiB。
批次`oof_distillation_seed0_20260913_230500`已在GPU0/1/2并行启动D0/D1/D2，三组均进入
task0训练，每epoch约13–14秒，启动后各卡仍余约4.3–4.5GiB，无OOM/非有限loss。

## 2026-09-13：CLIP Face + AffectNet补充残差失败，Face专用表征路线结束

seed0完整8-task validation batch `face_expression_residual_seed0_20260913_143719_retry1`完成：
240 epochs、10,980 updates、0 skipped，无OOM/NaN、checkpoint或test访问。候选的Face all/可靠Face/
固定R1 final mAP为`31.7562/35.3436/43.3047`，相对旧CLIP Face锚点变化
`-1.5959/-0.9660/-0.2765`。R1的8个task全部下降，forgetting从`0.9039`增至`0.9701`。

两项`+0.05`门槛均失败，不补seed1/2、不test，按协议结束Face专用表征路线；保留旧
CLIP Face与固定R1。下一个首选ViT-B/16方向是复用现有OOF预测做当前新类的跨视图教师蒸馏，
辅助视图仅提供训练监督，不再直接写入推理特征。结果详见
`docs/face_expression_residual_results_20260913.md`。

## 2026-09-13：实现CLIP Face + AffectNet补充残差

从`376885a`创建`codex/face-expression-residual-validation`。上一轮AffectNet完全替换CLIP Face失败后，
本轮保留旧CLIP Face输入、MULTI-LANE Selector、Image-token Adapter和head；同一目标脸另做五点对齐
AffectNet输入。冻结1280维embedding和8类logits分别LayerNorm后拼接，经每task独立、末层零初始化的
rank32投影，以固定scale0.1写入CLIP Face最终表征。残差用BCE/LR4e-4，CLIP Adapter继续ASL/LR4e-4。

只运行一个seed0完整8-task validation，固定R1 beta0.20、不存checkpoint、不访问test。可靠Face与
固定R1 final mAP必须均相对旧CLIP Face至少+0.05才继续；详见
`docs/face_expression_residual_validation.md`。

服务器独立worktree已固定`7840aac`，191项完整单测通过。真实64样本双encoder GPU前后向通过：
零残差与纯CLIP lane最大绝对误差`1.55e-7`；FP32/AMP下Image-token Adapter ASL梯度均非零且有限，
AffectNet与CLIP主干无梯度。当前8卡仅约6.4GiB空闲，launcher改为等待任一卡达到8GiB后自动启动，
避免与现有任务叠加导致OOM。唯一batch为`face_expression_residual_seed0_20260913_143719`，
tmux为`ml_face_expr_residual_143719`，结果根目录为
`/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_expression_residual_v0.1/face_expression_residual_seed0_20260913_143719`。
用户确认6.3--6.5GiB可叠加后改用GPU5/6GiB门槛立即启动；首batch在优化前暴露通用训练循环
将所有dict输入误当成Selector conditioning。已改为仅在存在`condition_valid`时统计该路径，
并增加双Face dict batch完整训练回归测试；失败运行未执行优化、未读test。
修复提交`fe054d5`在服务器192项全测通过。干净重试batch
`face_expression_residual_seed0_20260913_143719_retry1`已在GPU5启动；训练后剩余3.8GiB，task0前3个
epoch均64 updates、0 skipped，loss与Adapter loss均有限。tmux为`ml_face_expr_residual_retry1`。

## 2026-09-12：实现Face表情预训练专用表征

创建`codex/face-expression-pretrained-validation`，停止Router与基础Face输入超参搜索。新增manifest
五点landmark完整性/相似变换/逐task有效正例审计；EMOTIC新增默认关闭的五点相似变换对齐，锁定
margin0.15和无效点letterbox回退。首轮锁定官方EmotiEffLib AffectNet EfficientNet-B0并冻结，比较
task-specific Linear投影与task-specific b32零初始化残差Adapter；后者保持主BCE/Adapter
ASL9.8/0/0.05。Full、Person、旧CLIP Face和可靠性mask均复用，R1 beta严格固定0.20。
官方权重SHA-256锁定为`47c1423f...c652b17`。

本轮仅seed0完整8-task validation，两组并行，30 epochs/task、batch64、Adam、head LR0.0125、
Adapter LR4e-4、cosine min0、无warmup、AMP/TF32，不存checkpoint、不访问test。只有可靠Face子集
与固定R1 final mAP均相对旧锚点至少提高0.05，才补seed1/2。协议见
`docs/face_expression_pretrained_validation.md`。

服务器独立worktree为`/mnt/haoyuan/workspace/multi-lane-main-face-expression-pretrained`。187项完整
单测通过；官方权重在锁定`timm0.6.7`兼容映射后加载成功，冻结特征维度1280。五点审计train/val
覆盖为99.968%/100%，仅4个train样本回退，变换失败0；真实4样本projection/Adapter GPU训练步均
有限，峰值显存分别190/194MiB。

唯一正式batch `face_expression_seed0_20260912_1405`完整结束。projection/Adapter均完成240 epochs、
10,980 updates、0 skipped，无OOM/NaN/checkpoint/test。旧CLIP Face、projection、Adapter的可靠Face
final val mAP为`36.3096/35.7820/35.1329`，固定R1为`43.5812/43.3234/43.2659`；最佳projection
仍相对锚点下降`-0.5276/-0.2578`，双`+0.05`门槛失败，不补seed1/2、不test。完整结果见
`docs/face_expression_pretrained_results_20260913.md`。

## 2026-09-12：reliable_m15等更新量实验完成，基础Face输入路线停止

唯一batch `face_reliable_equal_updates_seed0_20260911_094039`完整结束：逐task实际updates严格等于
`[1920,1620,360,3570,1710,960,240,600]`，总计10,980，0 skipped、各scheduler终点LR0，未读test、
无checkpoint。可靠Face final val mAP由旧锚点36.3096降至36.1713（-0.1384），固定R1由43.5812
降至43.4675（-0.1137），双`+0.05`门槛失败，不补seed1/2、不test。

等更新量版本相对同过滤30-epoch版本的可靠Face又下降1.2229，R1下降0.0257，说明之前失败不是少
21.6%更新造成的欠拟合；更像较小、偏置可靠子集上的排序过拟合。按预案停止margin、过滤阈值和基础
增强搜索。下一阶段改用五点landmark对齐的表情预训练Face encoder与task-specific轻量投影/Adapter，
固定Full、Person与R1验证其是否带来真正新增信息。详见
`docs/face_reliable_equal_updates_results_20260912.md`。

## 2026-09-11：实现reliable_m15等更新量公平实验

创建`codex/face-reliable-equal-updates`，只补一个seed0完整8-task validation控制。新增按task指定成功
optimizer updates的runner协议，预算锁定为`[1920,1620,360,3570,1710,960,240,600]`，每task
CosineAnnealingLR以对应updates为T_max并在零LR结束。实验保持reliable_m15的过滤、margin0.15、冠军
Image-token Adapter和主BCE/Adapter ASL不变；Full、Person及固定R1权重均复用。

比较器新增显式等更新量审计，核验配置、逐task实际完成量、总计10,980、零skipped及scheduler终点。
候选仍须相对旧Face锚点同时提高可靠Face和R1 final validation mAP各至少0.05；失败即停止margin、过滤
和基础增强，转向表情预训练Face encoder。协议见`docs/face_reliable_equal_updates_validation.md`。

实现提交`b0c4ba2`已推送，服务器独立worktree
`/mnt/haoyuan/workspace/multi-lane-main-face-reliable-equal-updates`固定该提交且clean。179项完整单测通过；
真实Face/ViT-B/16/Adapter task0两步GPU smoke完成2 updates、0 skipped、scheduler终点LR0，写出完整val
scores且无checkpoint。唯一batch `face_reliable_equal_updates_seed0_20260911_094039`已在GPU0启动，tmux
为`multilane_face_equal_updates_20260911_094039`；task0首轮48 updates、0 skipped，无OOM/非有限值。

## 2026-09-11：独立Face expert基础质量诊断完成

batch `face_quality_seed0_20260911_001`三组均完成240 epochs/8,610 updates、0 skipped，退出码全0，
无test/checkpoint。旧Face锚点、reliable_m15、reliable_m05、reliable_m05+jitter的可靠Face final
mAP为`36.3096/37.3942/36.2327/36.4197`，固定R1 final mAP为
`43.5812/43.4933/43.3646/43.3363`。三组均未同时提高Face和R1，advance=false，不补seed1/2、
不运行test。

可靠过滤使Face自身提高1.0846却使R1下降0.0879；m05与jitter进一步降低融合，说明更紧crop丢失
头姿/周边语境，颜色增强主要改变阈值校准而没有增加互补排序信息。过滤还将旧锚点10,980 updates
降至8,610，存在21.6%训练量混杂。下一步只值得做一次reliable_m15等更新量seed0 validation公平
控制；若仍失败，停止crop/filter/基础增强，转向landmark对齐的面部表情预训练encoder。详见
`docs/face_expert_quality_results_20260911.md`。

## 2026-09-11：启动独立Face expert质量诊断

在动态Router/stacking收口后创建`codex/face-expert-quality-validation`。Full、Person及R1固定不变：
可靠Face使用`[0.64,0.16,0.20]`，其余严格`[0.80,0.20,0]`，Face beta锁死0.20且不搜索。
Runner新增从manifest原始检测框按配置margin重算crop、Face专用ColorJitter，以及Face loss训练样本
短边/检测分阈值；融合可靠性mask仍固定短边24px/检测分0.6，不随训练过滤漂移。

既有margin0.15、仅valid非歧义Face source作为冻结锚点；三组seed0完整8-task validation为：
`reliable_m15`（训练与融合可靠性对齐）、`reliable_m05`（缩小margin提高脸部有效分辨率）、
`reliable_m05_jitter`（再加0.1@0.5轻量颜色增强）。三组共享原模型/优化协议，GPU并行，不读test、
不存checkpoint。候选必须相对锚点同时提高可靠Face final mAP和锁定R1 final mAP各至少0.05，才补
seed1/2；详见`docs/face_expert_quality_validation.md`。

服务器独立worktree为`/mnt/haoyuan/workspace/multi-lane-main-face-expert-quality`，177项完整单测、
真实数据crop/filter审计和最严格候选task0一轮GPU smoke均通过。正式batch
`face_quality_seed0_20260911_001`已在clean `0972d1a`启动，tmux为
`multilane_face_quality_20260911`；GPU0/1/2分别运行m15、m05、m05+jitter，初始约2GB/卡、0 skipped，
无OOM/非有限值。三组完成后launcher自动执行固定R1比较。

## 2026-09-11：class-aware OOF stacking完成，动态Router路线收口

修复历史数值环境与CUDA mask兼容后，唯一有效run `class_aware_oof_seed0_20260911_002`退出码0，
明确`test_accessed=false`。C0固定R1、最佳C1 prior1、最佳C2 rank2/prior10的final validation
mAP为`43.581193/43.586666/43.600490`；C2相对C0/C1仅`+0.019297/+0.013824`，低于已放宽
的双重`+0.05`门槛，因此不补seed1/2、不运行test。

C2相对C0在8个task mAP均小幅提高，并有8类AP增益超过0.01，最大单类占正增益42.53%；收益
不是单类偶然。但主要提升Sensitivity/Fear/Sympathy `+0.2357/+0.1602/+0.0600`，同时
Suffering/Pain下降`-0.0314/-0.0253`。最强prior10胜过prior1/3，表明动态交互越接近固定先验越
稳定，可学习融合上限很低。按规则停止搜索Router prior/rank/hidden/描述符；后续若追求mAP，应
先提高独立Face expert，再用锁定R1验证，而非继续重排三个高度相关专家。详见
`docs/class_aware_oof_stacking_results_20260911.md`。

## 2026-09-11：class-aware OOF 首次启动前审计修正

首次正式启动在训练前被严格 endpoint 审计安全中止，未产生任何 C1/C2 训练结果。根因不是 GPU
或 score 损坏：历史 schema-v1 dump 只保存 logits，NumPy 1.26.4 与历史环境对并列分数的
`argsort`顺序不同，使 task0 mAP 漂移约0.0013。临时使用 NumPy 1.23.5 后，Full、Person、Face
三组共24个 validation task 的 mAP/cF1/oF1 及逐类AP全部按1e-10容差复现，因此 Track-A `ddp`
环境和`requirements.txt`改回1.23.5；Face预处理环境继续保持1.26.4。

同时在正式结果汇总前发现并修复字段名错误：序列化指标使用`per_class_ap`列表，而非不存在的
`class_ap`字典；逐类增益现按`CLASS_ORDER`显式映射，并增加回归测试。完成服务器全测与真实OOF
GPU smoke后重新启动时，又在首个C1反向捕获PyTorch2.0.1 CUDA高级索引写入的内部断言；梯度行
mask改用兼容的`index_fill_`。失败run均在产生候选前停止并保留，下一新run仍禁止test，门槛与
C0/C1/C2协议不变。

## 2026-09-10：三视图OOF/cross-fitting完成，当前Router停止

唯一batch`three_view_oof_seed0_20260910_160250`完整结束：9个fold/view专家、描述符和Router退出码
全0；每个专家8 tasks×30 epochs、0 skipped，无OOM/非有限值，明确`test_accessed=false`。完整
结果264文件约22MB已同步本地，压缩包SHA-256为`c3f67534...529d203`。固定R1、最佳OOF-R2/
prior1、最佳OOF-R3/prior1的final validation mAP为`43.581193/43.587281/43.586777`；R3高R1
仅`0.005584`但低R2 `0.000504`，不满足R3双重胜出规则。R3虽将average/cF1/oF1分别改善
`0.061706/0.271860/0.083202`并将forgetting降低`0.003036`，但幅度不足以支持seed1/2或test。

OOF把task6监督从51条扩大到627条仍未产生实质收益；无prior的R2/R3反降至`42.9327/42.6948`，
只有prior1接近固定R1。权重均值/std几乎可由可靠Face mask下的固定R1混合完全解释，说明Router近似
常数。根本结构限制是同task全部类别共享一组三路权重：R3对task6 Sadness/Sensitivity/Suffering
分别`-0.9809/+0.1050/+0.0187`，类别需求相反而互相抵消。停止扩大当前Router。若继续，优先复用
现有OOF产物做CPU-only分层class-aware stacking，而非重训专家；详见本批`analysis.md`。

## 2026-09-10：启动三视图image-group OOF/cross-fitting路线

在固定R1已取得正式三种子`33.0119 ± 0.3038`、而10% calibration Router和两种训练级融合均
失败后，创建`codex/three-view-oof-crossfit`，执行原动态路由计划中尚未完成的3-fold备用方案。
新增Runner的稳定image-group三折排除/留出接口：每个fold的Full、Person、Face专家只在另外两折
训练，留出折只用于导出OOF概率；三个fold合并后覆盖完整train池，每条Router训练预测均来自未见
该图的专家。专家固定seed0、8 tasks、30 epochs/task、batch64、cosine min0、主LR0.0125、
Image-token Adapter layer1/b32/LR4e-4/scale0.1/ReLU/independent、主BCE+Adapter ASL9.8/0/0.05、
CLIP normalization及AMP/TF32。服务器独立worktree当前HEAD `0e357e3`已通过166项完整单测；
真实EMOTIC三视图16,001条train样本ID/target完全对齐，task0--7三折留出分别完整覆盖
`5353/4394/861/9931/4352/2536/627/1526`条eligible，互不重复。task0的Full/Person/Face
并行1-epoch OOF smoke均0 skipped，三路留出1,793条ID、target和概率形状逐项一致；Face训练折在
loss前正确剔除848条无效/歧义样本，但留出score仍覆盖全部样本。完整train/val冻结CLIP描述符
导出smoke也通过，分别覆盖16,001/2,397条，可靠Face为9,696/1,460，明确未访问test。

Router改为跨task共享hidden16可靠性主干加每task仅3维偏置，按task递进训练并保存task快照；旧类别
始终使用其原task快照，无效Face权重严格为0。R2使用质量/预测统计，R3再加入覆盖完整train的冻结
CLIP三路余弦；prior仍只比较`0/0.1/1`。验证端复用100% train的seed0 Full/Person/Face endpoints，
只在完整8-task validation比较固定R1、OOF-R2、OOF-R3。仅当R3 final validation mAP同时超过R1和
R2才进入seed1/2；本批禁止test。实现当前已通过Python与shell语法检查；macOS系统Python缺NumPy，
以上验收通过后允许启动唯一seed0 OOF validation批次。
唯一批次`three_view_oof_seed0_20260910_160250`已在服务器clean HEAD `e490da0`启动，tmux为
`multilane_three_view_oof_20260910`。GPU0--7已分别启动前8个fold/view专家；初始显存约2.0--2.1GB/
卡，task0每轮Full/Person约56--57 updates、Face约43 updates，当前均0 skipped且无OOM/非有限值。
第9个`fold2_face`将在任一GPU释放后自动接续，随后才执行完整描述符和CPU Router选择。

## 2026-09-09：R1中心Router公平性修正完成，动态Router停止

提交`5675a3b`只将可靠Face初值/正则中心从`[0.72,0.18,0.10]`校正为固定R1赢家
`[0.64,0.16,0.20]`，无效Face仍严格`[0.8,0.2,0]`；服务器147项完整单测通过。复用原三专家
scores和描述符只重跑CPU Router，未重训、未使用GPU或test。修正后R1/最佳R2/最佳R3 final val
mAP为`42.7974/42.6804/42.6322`；R3同时低于R1 `0.1652`和R2 `0.0482`，不进入动态Router
seed1/2或test。可靠Face平均权重已对齐约0.20但仍失败，排除prior中心偏低作为主因；弱prior继续
显著过拟合，确认主要限制是per-task calibration样本量和当前类监督对旧类权重的冲突。本路线正式
停止，不扩大hidden或类别权重。校正结果52个文件已同步并通过组合SHA-256核验，详见本批
`analysis.md`。

## 2026-09-09：三视图Router seed0 validation完成

batch`three_view_router_seed0_20260909_163655`的90/10 Face source完成240 epochs、9,930
updates、0 skipped；描述符和Router均完成，明确未访问test。首次Router因无脸manifest的nullable
检测分读取失败；修复`16c0b87`增加null fail-closed回归，服务器147项完整单测通过后复用既有
scores/descriptors仅重跑CPU Router成功。85个非compact结果/状态/日志已同步本地，三组组合
SHA-256与服务器一致。

R0固定FP、R1固定Face0.20、最佳R2、最佳R3的final val mAP依次为
`42.4101/42.7974/42.6674/42.6707`；R3比R2仅高0.0033但比R1低0.1268，8个task均未超过
R1，因此不补seed1/2且不运行test。R2/R3均由prior1胜出，无prior时final降至41.55/41.39并
显著恶化forgetting，证实小calibration过拟合。当前Router prior中心Face权重为0.10而固定赢家为
0.20；最强prior使可靠Face平均权重仅约0.095--0.151，构成不公平起点。下一步仅建议复用现有产物
做一次R1中心`[0.64,0.16,0.20]`的CPU公平性修正；仍失败则停止per-task calibration Router。
详细见本批`analysis.md`。

## 2026-09-09：三视图样本级Router实现与验收完成

已从Face endpoint结果提交创建`codex/three-view-router-validation`。新增90/10 Face source支持，
复用已有seed0 Full/Person source；新增R0固定FP、R1固定Face beta0.20、R2质量/预测统计Router、
R3再加三个冻结CLIP跨视图余弦的validation-only选择。Router逐task独立、hidden16、masked softmax，
无效Face权重严格为0；R2/R3各只比较prior 0/0.1/1。视觉描述只覆盖train calibration与val，
不触及test。实现提交`906ac10`、测试断言修正`ada5bf1`均已推送并Git-only同步至服务器独立
worktree。服务器146项完整单测通过；真实train/val三视图ID、target与稳定分桶对齐通过。冻结CLIP
GPU smoke同时覆盖可靠/不可靠Face，三项余弦有限且不可靠Face相关值精确为0。Face source task0
两步GPU smoke完成且0 skipped；5,353条eligible中499条只用于calibration，fit侧另剔除1,152条
无效/歧义Face，3,702条实际参与loss，val/calibration scores及compact checkpoint均完整。
现已具备启动seed0完整validation条件。详细协议见`docs/three_view_router_validation.md`。
正式batch`three_view_router_seed0_20260909_163655`已在服务器clean HEAD `9b5d08c`启动，tmux为
`multilane_three_view_router_seed0_20260909`。GPU0训练90/10 Face source，GPU1并行导出三视图
描述符；启动后task0前三轮均58 updates、0 skipped，显存有余且错误扫描为空。两路成功后launcher
才会在CPU运行R0/R1/R2/R3选择，并写明`test_accessed=false`。

## 2026-09-09：Face endpoint validation完成

批次`face_endpoint_val_seed0_20260909_1320`在clean commit `285f3e1`完成240 epochs、10,980
updates、0 skipped，无checkpoint且未读取test。18个小产物已同步本地并逐文件SHA-256一致。
固定FP=`0.8F+0.2P`的final/average val mAP为`43.3035/50.4283`；可靠Face beta
`0.05/0.10/0.20`的final分别为`43.4777/43.5468/43.5812`，预声明赢家beta0.20提高
`+0.2777`，8个task mAP均提高。可靠子集FP到赢家为`44.7192→45.2443`，不可靠子集指标完全
相等，排除占位Face污染。Face-only全量/可靠子集仅`33.3521/36.3096`，说明收益来自弱专家错误
互补而非Face替代主路。代价是final cF1 `-0.7834`、oF1 `-0.0796`、forgetting `+0.0048`。
阶段1满足进入样本级三路Router的条件，但不直接test、不继续扩静态beta。详见本批`analysis.md`。

## 2026-09-09：Face endpoint validation实现与验收

已从阶段0提交创建`codex/face-endpoint-validation`。新增严格manifest驱动的`face_crop`输入：Face
expert训练和内部validation只包含`valid_face && !ambiguous_match`，过滤在DataLoader之前完成，
无脸/歧义样本不会进入BCE或ASL loss。最终validation score仍覆盖完整原始样本池。
融合固定既有`0.8 Full + 0.2 Person`，只对短边>=24px且检测分>=0.6的可靠Face尝试
beta `{0,0.05,0.10,0.20}`；其余样本逐元素精确回退锚点。新增来源/hash/ID/target/指标重算审计、
分组诊断、单GPU安全等待入口与回归测试。本阶段seed0、完整8-task、validation-only、无checkpoint，
不读取test。服务器142项完整单测、真实manifest数据检查和task0 1-epoch GPU/score-dump smoke均通过：
64 updates、0 skipped；2,285个完整task0 val样本中1,382个可靠Face可参与融合，903个不可靠样本
已验证逐元素精确回退FP。协议详见`docs/face_endpoint_validation.md`。
完整seed0 validation随后在服务器独立worktree以clean HEAD `285f3e1`运行，tmux为
`multilane_face_endpoint_val_seed0_20260909`，批次`face_endpoint_val_seed0_20260909_1320`，GPU0。
启动检查显示task0每epoch 64 updates且skipped0；launcher在完整训练后已自动审计既有
Full/Person端点并执行固定beta融合，结果见上节，全程未读取test。

## 2026-09-08：Full–Person–Face动态路由路线（计划）

根据组会方向，停止共享Person CLS引导Selector及当前query-only扩展，暂缓此前四组Person residual
计划。新主线为Face离线检测/匹配审计→Face互补性→同协议三路软/稀疏路由→任务独立统一特征融合
→三seed validation与锁定test。固定legacy Full、letterbox Person及冠军Image-token Adapter。
详细工作包、矩阵、预算、缺失脸规则和停止条件见
`docs/full_person_face_dynamic_routing_plan_20260908.md`。
阶段0已在`codex/face-manifest`实现离线Face manifest与审计入口，尚未启动完整训练。任务独立router按lane输出三路权重，不宣称全26类共用一组权重；
90%fit/calibration诊断不能混用100%train锚点。新路线优先解决Face质量及旧router持续更新风险。
阶段0固定InsightFace SCRFD `det_10g.onnx`、640输入、阈值0.5、CPU provider，只读取train/val。
同图检测复用缓存；目标人体与脸做一对一全局分配，并记录候选歧义而不把它误称为真实错配率。
产物包含逐样本稳定ID/脸框/原始尺寸/有效mask、每task当前类有效正例和200个train目标可视化抽检。
全量阶段0已完成：train/val有效率`84.98%/86.32%`，歧义风险占全部目标`7.22%/8.68%`，
有效脸短边中位数约36.5px且约40%小于32px。所有task/当前类均有有效正例；task6三类train有效
`229/274/161`、val有效`106/99/91`。稳定ID、合法框和一对一无复用检查通过，19个同步产物哈希一致。
结论是允许进入Face endpoint的seed0 validation，但必须mask歧义，并将短边/检测分作为质量条件；
不允许直接正式test或让低质量Face无条件主导。详细见阶段输出`analysis.md`。
排队的真实Face crop→letterbox/CLIP normalization→冻结ViT-B/16 GPU smoke随后在GPU0通过；
4个样本覆盖短边4.74--994.21px，输入`[4,3,224,224]`与输出`[4,512]`均有限。阶段0现已完整结束。

## 2026-09-07：Full + 初始Person crop固定融合对照（已完成）

用户要求补齐此前缺失的`Full + initial Person-only`对照。新分支
`exp/emotic-initial-person-fusion-control`只重训seed0初始Person分支并保存task0--7 test scores；
Full直接复用已核验的seed0 `32.5365`逐样本scores，不重复训练。Person严格复现初版：bbox
margin0.15、legacy RandomResizedCrop scale0.70--1.0、评估Resize256+CenterCrop224、无ColorJitter；
模型仍为Image-token layer1/b32/LR4e-4/scale0.1/ReLU/independent、主BCE+Adapter ASL9.8/0/0.05、
30 epochs/task、AMP/TF32。融合固定转移原letterbox validation锁定规则：probability Full0.80/
Person0.20、threshold0.5，test不搜索。新增严格来源审计、单一规则融合入口、单GPU训练/融合脚本和
回归测试；不保存checkpoint。commit `44ad8ed`已在服务器独立worktree通过129项完整单测和
真实EMOTIC train/test样本检查。8张GPU当前均高负载占用，等待器将在单卡连续两次满足
used<=2GB、util<=10%时，先做Adapter GPU smoke，通过后自动启动正式test。

实验已在GPU0完成：240 epochs、13,950 updates、skipped0，无OOM/非有限值异常，8份test
scores完整。初始Person-only精确复现final mAP `28.7815`；固定Full0.80/Person0.20融合为
`33.1429`，比Full `32.5365`高`0.6064`，但8个task均低于letterbox融合，后者仍以`33.2672`
保持seed0冠军。小结果和日志已同步本地且逐文件SHA-256一致，详细分析见
`output/emotic_track_a_initial_person_fusion/20260907_0145/analysis.md`。

## 2026-09-06：selector-specific Person patches 2×2完成

四组seed0完整8-task test均在`f1ada22`完成，各240 epochs/13,950 updates/skipped0，结果、
task0--7 test scores及日志52文件已同步并逐SHA匹配，manifest aggregate为
`0cc78d2b1f78ada2d59b3403daf630cc128fce82668f4215b00dbfc9286b4670`。
legacy-disabled / legacy-patches / target-disabled / target-patches final mAP依次为
`32.5365/31.6826/30.5018/30.2505`。legacy下patch下降`0.8540`，只在task0略升，task1--7
均下降；target-aware无论是否加patch都显著更差。target-aware虽将test bbox可见率从0.8032提高到
0.9778，训练loss也更低，但test mAP下降1.43--2.03，指向上下文/中心构图或裁剪分布泛化问题。
真实train无空Person mask，因此已知空mask bug不是本轮退化主因。
当前query-only与target-aware路线停止，不补seed1/2或调其参数。下一轮固定legacy crop，比较
每Selector Person value residual、frozen deep Person CLS residual、task-adapted deep Person lane
residual与fresh disabled；先seed0完整validation，赢家再做aux/等算力对照、多seed与锁定test。
完整报告见`output/emotic_track_a_person_conditioned_selector/selector_patch_2x2_test_20260906_analysis/analysis.md`。

## 2026-09-06：下一阶段分析与历史解释修正

本轮仅审计与制定方案，未改业务代码或启动新实验。01:52远端快照四组2×2约在task0 epoch14--15。
代码发现Person只改query，不直接注入value；padding未在Person前序self-attention屏蔽；空patch
mask会被safe_mask放开且输出未归零（合成用例已复现，真实频率未知）。下一步优先比较Person
summary残差与深层特征残差，暂不重复Adapter大网格。完整方案见
`docs/person_fusion_next_experiments_20260906.md`。
纠正旧说明：非当前logits置零切断旧head梯度、Adam逐task重建且head WD0，不能仅凭共享head
认定旧类持续被改坏。历史disabled task0五类AP为52.7473，task7相同ID为52.7182，扩展样本池为
36.3269；至少此部分下降主要来自评估池扩张。原forgetting仍报告，但需新增固定ID漂移诊断。

## 当前开发：target-aware crop × selector-specific Person patches（2026-09-06）

用户要求在第一版shared Person CLS条件失败后执行干净2×2。已从`455617a`创建
`codex/selector-specific-person-patches`：Full新增`legacy/target_aware`裁剪；target-aware
训练随机采样包含完整目标bbox的场景范围，评估在Resize256后选择尽量包含完整bbox的224裁剪。
Person继续bbox+margin0.15、letterbox224及共享flip，并新增14×14有效patch mask，softmax前屏蔽
letterbox padding。新`person_patches`模式在zero-based layer1使用与Full相同深度的冻结Person
patch tokens；每个Selector以自身query分别汇总Person patches，再经过当前task独立hidden32/ReLU/
scale0.1的零输出MLP产生不同查询增量。原Full Selector、Drop & Replace、Image-token Adapter及
BCE/Adapter-ASL路由不变。新增单组参数与4卡2×2启动入口；因val/test完整训练成本相同，按用户
要求本轮直接seed0完整8-task test，明确标记为结构探索，不在test搜索结构内参数。
本地新增4项针对性回归，完整127项CPU单测、Python编译、Shell语法、CLI解析和diff检查通过。
服务器独立worktree已通过真实数据与127项全测；用户随后确认允许在现有GPU任务上叠加。
disabled/person_patches真实CLIP Adapter-ASL GPU smoke均通过，AMP零初始化最大差分别为
4.58e-5/9.92e-5。四组seed0 test已于2026-09-06 01:43在GPU0--3启动，tmux为
`multilane_selector_patch_2x2_test_20260906`，源码固定`f1ada22`。四组均已生成config并完成task0
首个epoch，skipped0、无OOM/非有限错误；叠加后GPU0--3总显存14.7--15.2GB，仍余约9GB。

## 当前开发：Person-conditioned Full Selector（2026-09-05）

用户要求开始方向一代码修改并创建新分支。从`3f8327a`创建
`codex/person-conditioned-selector`，保留先前3份上下文文档的未提交结果记录。
实现disabled/bbox/person/bbox_person对照，默认只在zero-based layer1把条件增量加到
Selector查询；Drop & Replace仍恢复原Selector，不写回frozen image tokens。
Person内容来自共用冻结CLIP的最终ln_post CLS（投影前），不加载Person分类器或Adapter。
每task/layer独立hidden32/ReLU/scale0.1 MLP，末层零初始化，初始化RNG隔离；
新增参数走主模型BCE，独立LR4e-4，Adapter保留ASL9.8/0/0.05；旧task条件模块冻结。
数据返回同标注full/person/框/有效性；Full增强像素与RNG保持原路径，Person采用margin0.15
letterbox并共享flip，框映射到实际Full裁剪，完全不可见/无效目标关闭条件增量。
paired训练、val/test评估与score dump及compact模型重建已接入。提供val-only单组脚本，
支持四模式，固定原100%train/seed0完整8-task/30ep/batch64/main LR0.0125/AMP+TF32。
用户随后授权Git-only服务器同步及四组seed0运行；因val/test训练耗时相同，明确要求直接test。
实现提交`51b94a8`及GPU smoke容差修复`a8a759f`已推送。服务器主工作树两轮Automatic
Upload文件逐项SHA一致、分别备份后恢复clean；备份为
`git-sync-backup-person-conditioned-selector-20260905-preworktree`与
`git-sync-backup-person-conditioned-selector-20260905-smoke-fix`。独立worktree为
`/mnt/haoyuan/workspace/multi-lane-main-person-conditioned-selector`，test-only未改。
本地123项CPU回归（其中新增13项）、Python3.9编译、Shell语法及命令参数解析已通过。
服务器完整123项与真实EMOTIC paired数据smoke通过；三条件GPU smoke修复后通过，AMP
最大初始差bbox/person/bbox_person为3.05e-5/6.10e-5/4.58e-5，均低于1e-4。
四组seed0完整8-task test已在GPU0--3启动，tmux
`multilane_person_conditioned_test_20260905`，批次`person_conditioned_selector_test_20260905`。
disabled/bbox/person/bbox_person均进入task0，初始显存约2.0--2.1GB，无OOM；源码固定`a8a759f`。
输出写独立worktree的`output/emotic_track_a_person_conditioned_selector/`，日志写`logs/`。
四组现均完整结束：各240 epochs/13,950 successful updates/0 skipped，8份test scores齐全。
52个结果/日志文件已同步，本地与服务器聚合SHA均为`edb5458b783f1ae319d4910885e8ea4516c0a98973697d4778b47361bead6b1b`。
disabled/bbox/person/bbox_person final mAP分别为32.5365/32.0791/31.1695/31.3557；
average mAP为39.1445/38.8454/37.8552/38.0583。三条件均未超过disabled，forgetting也均恶化。
disabled与历史Full seed0所有task logits逐元素相同，证明paired数据路径保持基线。
固定0.8/0.2融合历史Person后final mAP为33.2672/32.6246/31.9731/32.0676，仍无条件化收益。
Person两组训练BCE更低而test更差；训练目标条件有效率88.31%、平均人物可见面积54.89%，
支持共享条件过拟合以及Full随机crop/Person完整crop失配的解释。详细分析见本批次`analysis.md`。
下一步建议停止本版shared-delta参数搜索，先做target-aware Full crop同批disabled锚点，再将
Person有效patch summaries以selector-specific查询交互；回到validation，失败则结束该路线。
方案与边界详见`docs/person_conditioned_selector.md`；实验四模式disabled/bbox/person/bbox_person，
GPU0--3并行，test只作为结构探索结果；保留独立Full及固定0.8/0.2融合两种报告。

## 最新结果：严格同seed Full+Full正式test重复对照已完成（2026-09-04）

用户要求对标同seed Full+Person的32.8263±0.4025。新分支`codex/same-seed-full-full-test`，
分别真实重训第二个Full_seed0/1/2，与原同seed Full按0.8/0.2融合，不将同一预测复制作为新实验。
保持完全相同的全局seed、训练参数及source提交：seed0使用clean c9fa74c，seed1/2使用clean d2442ee。
这属于严格重复性对照，两次Full可能相同/极接近；不是独立随机初始化集成，不改变seed制造多样性。
新增3卡调度器、独立同seed汇总器和3项回归。GPU0/1/2各一组，完整8-task/30epochs、batch64、
main LR0.0125、cosine/min0/no warmup、Image-token layer1/b32/LR4e-4/scale0.1/ReLU/independent、
BCE+Adapter ASL9.8/0/0.05、AMP/TF32 on、Full crop0.05、无checkpoint。先完整单测及两版source GPU smoke。
调度结果写入`output/emotic_track_a_same_seed_full_full/`、日志写入`logs/`；完成后自动比较原同seed
Full+Person、记录两次Full逐task logits/probability差异及训练轨迹。不运行新Person或其他候选。

实现`20dae80`已推送，独立调度工作树`/mnt/haoyuan/workspace/multi-lane-main-same-seed-full-full`
通过110项完整单测、六组历史score审计及c9fa74c/d2442ee两版Adapter GPU smoke。批次
`same_seed_full_full_test_20260904_122941`、tmux `multilane_same_seed_full_full_122941`已启动，
GPU0/1/2分别为seed0/1/2。三份新config与对应原Full逐字段（含Git元数据）完全相同，
task0首轮均84 steps/skipped0、loss有限；以下为本轮完成后核验结果。
Automatic Upload匹配副本已校验备份至`/mnt/haoyuan/workspace/git-sync-backup-same-seed-full-full-uD5CDy`，
primary恢复clean；运行中的原source及调度工作树不更新提交，test-only未改动。

三组现均exit0、完整240epochs/13950updates/skipped0、24份test scores齐全；训练/汇总成功。
三seed全部task的重复Full logits/probabilities逐元素相同，训练历史除耗时外相同；同seed Full+Full
全部指标与单Full相同，final test为32.1562±0.3701。原同seed Full+Person为32.8263±0.4025，
逐task/summary精确复现，配对增益0.6701±0.0670。这是重复性对照，不是独立随机集成无效的证据。
已有循环seed对照仍显示Full+Full32.6551、匹配Full+Person32.7957，额外人物贡献0.1406。
新输出及日志已同步，本地/远端99项source哈希一致；结果JSON SHA为
f7847c4ebfa158d307c05a650fbdd83f435ed57ab32e44a303accdbc5c76590c。
详见`output/emotic_track_a_same_seed_full_full/same_seed_full_full_test_20260904_122941/analysis.md`。
不再重复相同Full或搜索输出权重；建议下一阶段为最小Person条件Selector/bbox条件对照，先seed0
完整8-task val，再候选seed1/2、锁定后test。保持原100%训练数据协议，不混入门控90%fit协议。
本轮未实现或启动新结构；仅本地更新完成记录，不提交推送或改变实验source。

## 最新状态：2026-09-04 Full+Full固定正式test对照

用户要求补齐此前仅validation的Full+Full正式test。新分支`codex/full-full-locked-test`，
复用原seed0/1/2已保存的Full和Person test预测，不重训或新增推理。固定probability 0.8/0.2、
threshold0.5、循环辅助seed映射0→1/1→2/2→0；同步报告matched Full+Person及同seed旧融合锚点。
新增独立test入口及回归，validation入口仍拒绝test。不搜索权重、配对、阈值或挑选test赢家。
先完整单测，再服务器独立worktree离线评估；source逐task指标/类别AP和ID审计、记录SHA-256。
下方原顶部及后续条目保留历史时间线；本节为当前执行状态。

已完成：clean实现`c54e330`通过107项完整单测，服务器独立`multi-lane-main-full-full-test`仅CPU离线
计算。Full+Full test为`32.6551 ± 0.2622`（32.9567/32.5271/32.4815）；匹配辅助seed的Full+Person
为`32.7957 ± 0.2378`，配对净增益`+0.1110/+0.1656/+0.1453`，平均`+0.1406`，24个task配对均正。
同seed旧融合`32.8263 ± 0.4025`逐字段精确复现；不能把其与Full+Full的差0.1712当作匹配辅助归因。
Full+Full相对单Full平均增加0.4989，人物视图仍有额外收益但小于validation的0.3281；forgetting
相对Full+Full平均恶化0.0421。结果/日志已同步，JSON SHA一致，source66个文件哈希记录完整。
详见`output/emotic_track_a_ensemble_control_test/full_full_locked_test_20260904/analysis.md`。
无新训练/推理/test搜索。Automatic Upload匹配副本备份到`git-sync-backup-full-full-test-S0yRpG`；
primary恢复clean，test-only保持不变。Person条件Selector仍未启动。

最后一次更新：2026-09-04；当前本地分支为`exp/emotic-constrained-gated-fusion`。
用户已要求在三种子validation上搜索Full/Person全局概率权重、完成bbox可靠性分层，并尝试跨三个seed
方向一致的强约束类别/几何门控；validation锁定后只允许一次正式test评估。当前从`343ddca`创建
独立分支，复用已有六组validation和六组test score，不重新训练模型、不生成checkpoint。
全局person权重搜索固定`0.00--1.00`步长0.01，以三种子final mAP均值优先、average mAP破平；
门控以0.20为base，类别与质量方向需三seed一致且幅度至少0.05 AP/mAP，只搜索
`class_delta,quality_delta ∈ {0,0.025,0.05}`。门控必须逐seed不低于全局权重赢家才可进入一次test。
几何分层固定面积占比`<0.10/0.10--0.30/>=0.30`、绝对长宽比`<=2/>2`和单人/多人，不从test拟合。
实现`69cff55`及数据根目录修复`baf2263`已提交推送；服务器新独立worktree通过91项完整单测。
首次路径失败发生在validation selection前且未加载test；修复后新batch先锁定validation规则，再对
test只评估一个候选。全局validation最佳为Full0.75/Person0.25，final mAP`43.3060 ± 0.3732`；
受限门控以base0.20、class delta0.05、quality delta0.025达到`43.3420 ± 0.3906`并逐seed超过
全局赢家，因此锁定进入test。唯一test结果为`32.8380 ± 0.4002`，比固定0.20提高`0.0117`；
三个seed均为正但效应极小。当前静态score-level门控视为基本饱和，下一步优先从train/calibration
学习共享样本级可靠性门控，仍无效再转Person token辅助selector结构。
用户已要求执行统计流程修复和固定权重的等计算量validation对照。本分支在已有三种子结果基础上，
新增schema2 score dump（实际probabilities、batch长度及数值来源），兼容schema1按原batch重建；
离线anchor逐字段/逐类AP核验，不放宽容差。runner只增加预测保存元数据，不改变优化或前向算法。
新增full/full与full/person配对validation分析：固定0.8/0.2、threshold0.5，主seed0/1/2对应
辅助seed1/2/0；另报告同seed双视图。复用seed0两视图val，计划GPU0--3补seed1/2四个run。
不搜索融合参数，不启动新test。提交同步后先完整单测、历史数值回归、真实score dump和GPU smoke。
旧seed0的schema1 validation缺少后增的score-purpose字段，仅在val/ID/指标全部审计通过后兼容，
不改写旧文件、不把此例外应用到schema2或test，也不放宽训练配置一致性要求。
实现`5274d14`及受限兼容`ebf49a2`已提交推送；独立服务器工作树为
`/mnt/haoyuan/workspace/multi-lane-main-ensemble-control`，训练源码固定clean`ebf49a2`。
完整87项单测通过；三种子历史固定test结果与上轮恢复结果完全一致，seed0 validation固定融合
精确复现43.30347536439419；真实EMOTIC两视图各67样本、batch64+3的schema2保存/读回和ID对齐、
真实Adapter GPU smoke均通过。batch`fixed_ensemble_control_val_seed12_20260903_191515`已启动，
GPU0/1为seed1 full/person，GPU2/3为seed2 full/person；只运行validation，没有新test。
四份config与对应seed0除seed/Git及规范化旧用途字段外一致；task0前几轮均84 steps/skipped0，
无OOM/非有限值，GPU0--3仍余约21.6--22.1GB显存。
全部完成后自动输出`output/emotic_track_a_ensemble_control/<batch>/ensemble_comparison.json`。
四组训练已完整exit0/240 epochs/13950 updates/skipped0且新schema2 scores齐全；初次汇总因新runner
写入`val_scores/`而汇总器沿用旧`validation_scores/`路径失败，原launcher exit1保留。当前修复
显式兼容两种目录且要求只能存在一种，并增加回归测试；将直接使用现有scores汇总，不重跑训练。
目录兼容修复`343ddca`已推送；服务器独立实验工作树ff-only同步后88项完整单测通过。恢复汇总
使用原有六组validation scores完成，未重训、未读test、未搜索融合参数。等计算量循环配对中，
Full+Person相对Full+Full的final mAP三组均提高`+0.2939/+0.3183/+0.3722`，平均
`+0.3281 ± 0.0401`；average mAP平均提高`+0.4616 ± 0.0820`，24个seed×task mAP差值全部为正。
同seed Full+Person相对单Full的final mAP平均提高`+0.8931 ± 0.1050`。因此人物视图具有超出普通
双Full集成的稳定互补信息，但本轮只是validation归因，不产生新test成绩。
服务器结果与日志已同步到本地`output/emotic_track_a_ensemble_control/20260903_191515/`并完成
整体SHA-256一致性校验；详细报告为该目录`analysis.md`。
Automatic Upload的逐文件一致副本已备份到`/mnt/haoyuan/workspace/git-sync-backup-ensemble-control-1CWO07Me`，
primary已恢复clean，test-only保持不变；后续运行记录只提交本地文档，不更新活动训练工作树。
服务器seed0独立正式工作树使用clean实现提交`c9fa74c`。validation预先锁定的full0.80/person0.20
probability融合已完成seed0完整8-task正式test：final mAP`33.2672`，超过同批单视图冠军
`32.5365`共`0.7307`，8个task的mAP/cF1/oF1均提高。结果已同步本地并逐组核对哈希；
未在test搜索融合参数。用户现已要求保持全部参数不变，补seed1/2正式配对确认。
已审计服务器171份config：旧full seed1/2只有指标、没有scores/checkpoint，person seed1/2不存在；
因此需补四个训练进程：seed1 full/person用GPU0/1，seed2 full/person用GPU2/3。
本分支仅增加seed参数化、四卡调度和三种子严格配对汇总，不改变模型、数据或优化算法。
实现`d2442ee`已提交推送；服务器独立`multi-lane-main-dual-view-seed12`工作树clean，完整75项
单测、Adapter GPU smoke、原seed0训练预算审计均通过。batch
`dual_view_locked_formal_seed12_20260903_174938`已在GPU0--3并行启动；四份config逐字段核对，
与对应seed0除seed/Git外完全一致。固定probability full0.80/person0.20/threshold0.5。
seed1/2现已完整结束，四组训练均exit0/240 epochs/13950 updates/skipped0；自动融合因全表与原
batch64 sigmoid的细微浮点差异导致anchor指标校验失败，原launcher exit1记录保留。已按原batch64
严格重建六个run所有task指标，并保持原固定融合计算不变，在独立analysis_recovery产物中完成汇总。
融合final mAP为`32.8263 ± 0.4025`，full为`32.1562 ± 0.3701`；配对增益
`+0.7307/+0.5981/+0.6815`，均值`+0.6701 ± 0.0670`。24个seed×task mAP差值均正。
task6 Sadness/Suffering平均提高`1.6915/3.5961`，Sensitivity下降`0.0333`；forgetting平均恶化
`0.0519`。70个结果/score/log/状态文件已同步并通过分组SHA校验，详见
`output/emotic_track_a_dual_view_seed_confirmation/20260903_174938/analysis.md`。
上述结果分析轮没有新训练；当前开发轮开始通用修复及validation对照。Automatic Upload副本已逐文件哈希校验、
备份到`/mnt/haoyuan/workspace/git-sync-backup-dual-view-seed12-DbO8J9xs`并恢复primary clean。
test-only工作树保持原样，未被本轮同步、分析或实验修改。以下旧实验条目作为历史记录保留。

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
- PyCharm Automatic Upload 按规则必须关闭；此前实际仍多次上传文档，不能仅凭默认配置认定已关闭。
  `.git/` 永远不得通过 SFTP 同步。
- 当前Track-A实验使用服务器`ddp`环境；历史实验曾使用`multilane`。
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
- 正式batch`image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816`已在等待GPU释放
  后自动完成。两组均240 epochs/13,950 updates/skipped0，退出码全0，无checkpoint/错误；两组非计时
  训练记录与对应validation run逐字段完全相同。固定融合重新计算与已保存结果完全一致。
- full/person/fusion final mAP分别为`32.5365/28.7474/33.2672`；融合average mAP`40.0545`，
  cF1`32.9486`，oF1`49.5948`，forgetting`4.7478`。相对full依次提高final/average/cF1/oF1
  `0.7307/0.9100/0.3729/0.2121`，forgetting轻微恶化`0.0169`。
- 8个task的mAP/cF1/oF1全部提高，final有21/26类AP提高。task6 Sadness/Sensitivity/Suffering
  提高`0.1103/0.1351/4.4223`。新成绩是seed0双分支最佳，不是三种子均值，也不能将收益全归因于Adapter。
- 34份原始结果/NPZ/状态/日志已同步到`output/emotic_track_a_dual_view_formal_test/20260903_145816/`；
  同目录`analysis.md`记录完整审计、曲线与限制。本轮不重跑已完成实验，不使用test搜索新融合参数。

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

## 2026-09-04：样本级人物可靠性门控

- 静态融合、bbox阈值和类别独立权重搜索已经停止。当前分支开始实现跨task共享的小型MLP，每个
  样本只输出一个Person概率融合权重，结构性限制为`[0.10,0.35]`并正则回固定锚点`0.20`。
- 门控固定使用bbox面积/长宽比/人物数量/多人标记，以及Full/Person两视图的置信度、二元熵和
  概率差异共10维特征。门控不读取类别ID，也不输出类别独立权重。
- 为防止validation标签泄漏，基础Full/Person模型将train按稳定图像组哈希拆成90% fit和10%
  calibration；同图全部人物、两个视图和三个seed使用一致分桶。门控每个task只在当前task的
  calibration标签上顺序训练，不回放旧task样本，validation只负责选择正则强度。
- 计划网格为hidden`{8,16}`×归一化prior strength`{0.1,0.3,1,3}`，固定AdamW LR`1e-3`、
  WD`1e-4`、80 epochs/task、batch64。候选必须在三个seed各自的final validation mAP上不低于
  同批固定0.20锚点；合格后按三种子mean final mAP、mean average mAP排序。
- runner新增calibration score和不含冻结CLIP视觉塔的per-task compact state；只有validation明确
  锁定赢家后，独立导出器才从compact state对test推理，并且最终只评估这一种门控。若无候选通过，
  launcher必须结束且不访问test。
- 实现已提交为`2e352e0`并在服务器独立worktree ff-only同步；97项完整单测、真实EMOTIC同图分组
  划分、真实calibration score dump、真实ViT-B/16 Adapter-ASL AMP/TF32、compact state恢复和
  8-task顺序门控合成smoke全部通过。完整validation可安全启动。
- 唯一batch`learned_reliability_gate_seed012_20260904_021636`已在tmux
  `multilane_learned_gate_20260904_021636`启动，GPU0--5分别运行3 seeds×Full/Person。6份config
  均为clean`65d3ca4`且协议审计通过；task0首轮均76 steps、skipped0，显存安全、无错误。launcher
  将在validation结束后自动执行门控选择，只有通过三种子硬门槛才导出一次锁定test。
- 结果同步后确认：六组基础训练已完成（各240 epochs/12,660 updates/skipped0），但门控入口仅加载
  train geometry，第一次val评估因缺少val sample ID中断。只有h8/prior0.1/seed0/task0状态，没有
  selection或test；不能判断学习型门控有效性。197个原始产物（含48份compact states）已同步并逐文件
  SHA-256匹配，详见`output/emotic_track_a_learned_reliability_gate/20260904_021636/analysis.md`。
- 同批固定0.20融合final validation mAP为42.4101/44.0283/42.8836，均值43.1073±0.8320；相对Full
  配对提高0.7971/0.6679/0.7239。下一步只修train/val元数据路由与集成测试，复用已完成基础模型补跑
  门控；本次分析未启动新训练/test，也未改业务代码。
- 用户随后授权修复并补跑，创建`codex/fix-learned-gate-split-routing`。选择入口显式分离train/val
  geometry并在训练前检查所有ID覆盖；新增不同split的8-task回归测试及validation标签改变不影响
  门控参数的测试。额外记录权重分布和分split的current BCE/mAP诊断，但不改变任何参数或选优门槛。
- 补跑脚本只读取既有六组source scores和compact states，使用新输出目录；不重训基础模型、不覆盖
  失败目录，三种子validation通过后才有条件执行一次锁定test。
- 修复后102项完整测试、真实train/val全部ID覆盖、GPU smoke及保存的Full/Person task7首批64样本
  compact-state恢复概率精确一致检查通过。补跑`learned_gate_split_fix_20260904_113753`在clean
  `2ee06fb`完成原定8候选×3seed×8task，192份gate state与selection哈希通过，原180个source文件不变。
- 无候选合格且所有final val mAP均值均低固定0.20的43.107346；最高h16/prior1.0为43.106145，
  差-0.001201。程序正常跳过test。task6强prior3权重均值近0.20（std0.00041--0.00131）；弱prior0.1
  学到变化但三类AP未稳定迁移。24组task6 calibration BCE均降低，却有17组validation三类mAP下降。
- 新结果202文件同步并逐文件SHA-256一致，报告位于
  `output/emotic_track_a_learned_reliability_gate/20260904_113753/analysis.md`。本轮结束，保留固定融合，
  不扩MLP/新比例、不放宽门槛、不运行test。
# 2026-09-09 固定三视图三种子正式test

- 用户将上一阶段seed1/2 validation计划改为直接运行seed0/1/2 held-out test。规则严格锁定为：
  可靠Face使用Full/Person/Face=`0.64/0.16/0.20`，否则精确回退`0.80/0.20/0`，threshold0.5。
- 不训练Router，不在test搜索权重、阈值或Face质量门槛。复用已有同seed Full/Person正式test
  probability dumps，只补三个seed的100% train Face expert。
- Face test源固定冠军协议：8 tasks×30 epochs、batch64、main LR0.0125、layer1/b32/Adapter LR4e-4/
  scale0.1/ReLU/independent、main BCE+Adapter ASL9.8/0/0.05、AMP/TF32、无checkpoint。
- 本分支新增test Face manifest、固定三视图审计融合器及三GPU launcher。完整说明见
  `docs/fixed_three_view_seed012_test.md`。
- 服务器150项完整单测、test Face数据对齐、真实test crop→CLIP、Image-token Adapter和两次更新的
  test score-dump smoke均通过。正式batch `fixed_three_view_seed012_test_20260909_213052`已在GPU0/1/2
  并行启动，三个seed均记录clean commit `acf4634`及同一test manifest哈希。
- 正式test已完成并同步：固定三视图final mAP为`33.0119 ± 0.3038`，相对同seed Full+Person
  `32.8263 ± 0.4025`提高`0.1856 ± 0.1000`，seed0/1/2分别提高
  `0.0719/0.2247/0.2602`。task0--7平均mAP变化全部为正，当前升级为正式三种子冠军。
- Face新增收益主要体现在Anger、Aversion和Suffering；Fear、Confidence和Sensitivity下降。
  forgetting仅恶化`0.0144`。完整分析位于
  `output/emotic_track_a_fixed_three_view_test/20260909_213052/analysis.md`。

## 2026-09-09 taskwise三视图Router语义修正

- 用户要求开始修正Router增量语义，并同时计算R1/R2/R3的validation与探索性test。
- 新实现令Router k只融合task k引入的类别；Router特征也只读取其训练时已见的类别前缀。评估task t
  时分别运行冻结Router0...t并拼接类别lane，避免新Router覆盖旧类。
- 复用已有90/10专家scores、6组Router states和validation descriptors；不重训专家或Router。
  validation在原prior `{0,0.1,1}`中重新选择修正后的R2/R3，R3仍须同时超过R1/R2。
- 按用户明确要求，对validation定义的R1/R2/R3同时计算已有三seed test；test标记exploratory且不参与
  候选选择。只需GPU导出一次无标签的test冻结CLIP描述符，随后CPU完成全部重组。
- 实验协议见`docs/taskwise_three_view_router_evaluation.md`。
- 实现`9ab41a7`已推送；服务器独立worktree通过152项全测和真实保存state重组smoke。唯一批次
  `taskwise_three_view_router_20260909_233435`已完成并同步，描述符与CPU评估退出码均为0。
- 修正后validation：R1/R2/R3=`42.7974/42.8372/42.7639`。R2比旧语义提高`0.1568`并略超R1，
  证明lane覆盖确有损失；R3仍未超过R1/R2，既定advance条件失败。
- 三种子探索性test：R1/R2/R3=`33.0119/33.0029/32.9758`。R2比R1低`0.0090`，R3三个seed
  均下降。停止当前离线小样本Router路线，固定R1继续保持正式冠军；下一步进入端到端taskwise
  三视图特征融合validation。

## 2026-09-10 端到端taskwise多视图特征融合

- 用户确认开始下一阶段，新建`codex/end-to-end-taskwise-three-view-fusion`。固定冠军Image-token
  Adapter（zero-based layer1/b32/LR4e-4/scale0.1/ReLU/independent）、主模型BCE+Adapter
  ASL9.8/0/0.05、30 epochs/task、seed0完整8-task validation，禁止test。
- J0在归一化task-lane CLS特征层固定融合Full/Person/Face；J1使用每task独立hidden16 MLP输出
  样本级三路masked-softmax；J3是相同机制的Full+Person控制。可靠Face先验为0.64/0.16/0.20，
  不可靠Face严格为0并回退0.80/0.20；soft Router从对应先验初始化。
- Router k只融合task k lane，task结束后冻结。三个视图共享冻结CLIP、selectors/prompts/head和当前
  task Adapter；主融合目标外增加0.1倍有效分支平均辅助监督，Face辅助损失仅使用可靠样本。
- 详细协议见`docs/end_to_end_taskwise_three_view_fusion.md`。下一步完整单测、真实数据和GPU smoke；
  全部通过后并行J0/J1/J3。只有J1 final validation mAP同时超过J0/J3才补seed1/2。
- 服务器修复后159项全测通过；真实task0 smoke在GPU0/1/2分别完成J0/J1/J3，每组84 updates、
  skipped0、batch64无OOM/非有限值。J1学得平均权重约0.699/0.184/0.116，与J0有效/无效Face混合
  后的0.709/0.177/0.114不同，确认Router参与更新；可启动唯一完整validation批次。
- clean提交`9f17799`上的唯一批次`end_to_end_view_fusion_seed0_20260910_005421`已在服务器独立
  worktree启动；tmux为`multilane_e2e_fusion_20260910_005421`，GPU0/1/2并行J0/J1/J3。
  三组均已完成前2个task0 epoch，每轮84 updates/skipped0，显存峰值观察约3.6GB，无错误。
- 本批已完整结束并同步：J0/J1/J3均完成240 epochs、13,950 updates、skipped0，无OOM、非有限值或
  运行错误。final validation mAP分别为`42.5330/39.8569/38.3318`，J1比J0低`2.6762`，未通过
  同时超过J0/J3的预设条件，因此不补seed1/2且不运行test。
- J1各task最终Person平均权重为`0.966/0.941/0.761/0.799/0.616/0.806/0.537/0.794`；J3更接近
  Person-only。下降从task0开始，说明主要是竞争式softmax Router把Full场景通路关闭，而非task6
  小样本或增量lane覆盖。J1仍比J3高`1.5251`，Face存在补充信息但不能抵消Person塌缩。
- 既有独立专家固定三视图seed0 validation为`43.5812`，比共享参数J0高`1.0481`；训练机制不同，
  不能作为严格同批消融，但提示共享selectors/prompts/head/Adapter削弱视图专门化。下一步停止当前
  simplex Router，改做Full系数固定为1、Person/Face为零初始化有界taskwise残差的最小A0/A1/A2
  validation。完整报告见`output/emotic_track_a_end_to_end_view_fusion/20260910_005421/analysis.md`。

## 2026-09-10 Full-anchored complementary residual

- 用户确认停止softmax Router并开始A0/A1/A2。新建`codex/full-anchored-complementary-residual`，
  Full系数恒为1；Person/Face分别通过task-specific bottleneck16残差Adapter补充，输出投影零初始化，
  残差先约束到单位范数内，再乘`0.1*sigmoid(task gate)`，有效强度严格不超过0.1。
- Person/Face源特征关闭Full Image-token Adapter并在残差模块前stop-gradient，使共享task参数与
  Image-token Adapter只接受Full主路径梯度；本阶段分支辅助loss为0，直接消除上一轮共享多视图监督的梯度冲突。各task
  残差模块训练后冻结，无效Face权重与残差严格为0。
- A0为fresh Full冠军锚点，A1为Full+Person residual，A2为Full+Person+Face residual。固定seed0、
  8 tasks×30 epochs、冠军Adapter和BCE/ASL协议，validation-only、no-checkpoint、test forbidden。
  只有A2 final mAP同时超过A0/A1才补seed1/2。协议见
  `docs/full_anchored_complementary_residual.md`。
- 首轮163项测试通过，但真实A1/A2 smoke发现AMP cast cache会复用先前no-grad辅助视图中的Adapter
  权重，使Full ASL梯度断开；两组在首个反传安全停止，未启动完整实验。修复为辅助视图完全关闭
  Full Image-token Adapter，再恢复Adapter执行Full前向，明确落实视图专门化并消除cache歧义。
- 修复后164项完整测试通过；A0/A1/A2真实task0 smoke均完成84 updates、skipped0，无OOM/非有限值。
  一轮诊断val mAP为`40.7614/41.1825/41.4034`，仅证明通路有效，不用于选优。下一步在同一clean
  提交上启动唯一三GPU完整validation批次。
- clean提交`f98f74c`上的唯一批次`full_anchored_residual_seed0_20260910_103042`已在服务器独立
  worktree启动，tmux为`multilane_full_anchored_residual_20260910_103042`，GPU0/1/2运行A0/A1/A2。
  三份config均为seed0、30 epochs×8 tasks、validation-only、no-checkpoint和同一clean tree；A0已完成
  task0前4轮，A1/A2前2轮，全部84 updates/轮、skipped0且错误扫描为空。
- 本批已完成并同步：三组均240 epochs/13,950 updates/skipped0，无运行错误。final validation mAP
  A0/A1/A2=`42.3799/41.8802/42.1556`；A2比A1高`0.2754`但比A0低`0.2243`，8个累计task均
  未超过A0，因此`advance=false`，不补seed1/2且不运行test。
- 残差并未失活或饱和：Person最终系数约0.052--0.064，Face含无效样本的batch均值约0.033--0.042。
  A2提高task6/7 current-only mAP，却恶化全部已见类排序与forgetting，说明当前类目标让小残差学习
  新类捷径并间接拉偏共享head/Full路径。停止调残差容量、scale、gate和辅助loss。
- 现有证据支持保留独立专家固定R1正式冠军`33.0119±0.3038`。若继续个性化分路，应使用按image-group
  cross-fitting生成完整训练集无泄漏OOF专家预测，再训练跨task共享的小Router；不再用几十条
  calibration或端到端反传破坏专家。完整报告见
  `output/emotic_track_a_full_anchored_residual/20260910_103042/analysis.md`。

## 2026-09-10 服务器重置与环境恢复

- 审计确认服务器重置只清除了容器层：`/mnt/haoyuan/workspace` 下的所有Git工作树、EMOTIC、
  CLIP/SCRFD权重、历史日志和实验结果均保留；丢失的是`/opt/conda/envs/ddp`、
  `cocoer-preprocess`和`/root/.ssh`授权状态。
- 已恢复`ddp`：Python3.9.23、torch2.0.1+cu118、torchvision0.15.2+cu118、numpy1.26.4及
  项目固定依赖；恢复`cocoer-preprocess`：InsightFace0.7.3、ONNXRuntime1.18.0、
  OpenCV4.10.0、Pillow11.3.0、SciPy1.13.1及相同torch栈。重置后的base Python3.11/
  torch2.6不得用于复现实验。
- 两环境`pip check`、8卡CUDA计算、SCRFD真实图检测、4样本Face crop到frozen CLIP、
  Image-token layer1/b32 Adapter-ASL forward/backward均通过；项目169项完整单测全通过。
- 额外完成Full/seed0/OOF fold0/task0/1 epoch端到端恢复smoke：56 updates、skipped0、
  val mAP39.543071，未访问test且未保存checkpoint。没有启动任何新完整实验。
- 已恢复本地到服务器的公钥登录与SSH agent forwarding，服务器通过转发的本地密钥访问GitHub；
  私钥未复制到服务器。服务器OOF worktree经Git-only fast-forward至`e0e6d57`；test-only仍为
  `0b59138`且其原有未提交改动完整保留，本次没有触碰。
- 完整版本、命令、资产哈希和日志路径见`docs/server_environment_restore_20260910.md`。

## 2026-09-10 class-aware OOF stacking

- 用户确认执行OOF结果后的C0/C1/C2，并允许恢复环境后正常使用GPU。新分支
  `codex/class-aware-oof-stacking`复用已有9个专家的无泄漏OOF scores、三视图描述符和100% train
  validation endpoints；不重训专家、不读取test。
- C0为固定R1；C1为每类独立的中心化三视图logit bias，只用该类引入task的OOF样本训练并以
  `{1,3,10}`强正则回R1；C2冻结最佳C1 bias，增加共享线性样本可靠性投影与每类view factor的
  rank-2交互，不增加类别MLP，交互正则仅比较`{1,3,10}`。
- C1/C2固定FP32、Adam LR1e-2、batch512、120 epochs/task；无效Face权重严格0。旧类别参数冻结，
  每类使用其引入task结束时的投影/参数快照，保持增量语义。训练支持CUDA并计划在空闲GPU执行。
- 用户允许稍降门槛，预注册为C2同时超过C0/C1至少`0.05` final validation mAP；至少两个类别AP
  提高超过0.01，且单一类别不超过全部正增益的80%。全部满足才进入seed1/2，当前阶段禁止test。
- 实现、测试、单GPU launcher和协议见`docs/class_aware_oof_stacking.md`。下一步Git-only同步服务器
  独立worktree，运行完整单测和真实OOF短smoke，通过后执行唯一seed0完整validation选择。
-
## 2026-09-14 validation锁定结构的横向test补测

- 用户为组会横向比较明确要求补齐仅做过validation的最佳结构。固定R1已有seed0/1/2正式test
  `33.0119±0.3038`，不重复训练；本轮新增C2、J0、A2的锁定test，禁止在test搜索权重或参数。
- C2复用`class_aware_oof_seed0_20260911_002`由seed0三折image-group OOF选出的
  `C2_rank2_prior10p0`及其8个task快照，并把同一锁定stacker应用到三个seed的独立专家test预测。
  该项是探索性跨seed泛化评估，不重新训练C2，也不把test结果用于回选。
- J0/A2各从头运行seed0/1/2完整8-task test：30 epochs/task、batch64、main LR0.0125、per-task
  cosine min0无warmup、CLIP normalization、legacy crop、Image-token layer1/b32/LR4e-4/scale0.1/
  ReLU/independent、主BCE+Adapter ASL9.8/0/0.05、AMP/TF32开启。J0为固定三路特征融合+0.1
  辅助监督；A2为Full恒1的Person/Face taskwise零初始化残差、aux loss0。
- 六组训练不保存checkpoint，保存`evaluation_scores`用于复核；结果写入服务器
  `emotic_benchmark_runs/multi_lane_locked_architecture_test_v0.1/`，日志、状态和综合汇总写入当前
  worktree的`logs/output/emotic_track_a_locked_architecture_test/`。启动前须完整单测及真实GPU smoke。
- 服务器独立worktree已Git-only同步至`42d0785`；206项完整单测通过。J0/A2真实EMOTIC task0、
  1 epoch、test score-dump smoke均完成，C2锁定状态、descriptor、Face mask及三种子R1复现通过。
  C2只读test预结果为`33.0153±0.3074`，相对R1均值仅`+0.0034`；该值不用于回选参数。
- 正式batch `locked_architecture_test_20260914_013250`已在clean提交`b1cddff`启动：GPU0--2运行
  J0 seed0/1/2，GPU3--5运行A2 seed0/1/2，GPU6完成C2统一结果写入。六组首轮均84 updates、
  skipped0；峰值显存约3.6GiB(J0)/2.1GiB(A2)，每卡仍余20GiB以上，无OOM或错误。
- 本批已结束并同步50个JSON/日志/状态文件，本地与服务器SHA-256全一致。final test mAP：固定R1
  `33.0119±0.3038`、C2 `33.0153±0.3074`、J0 `32.1639±0.6258`、A2
  `32.4201±0.1352`。C2相对R1仅`+0.0034`且三seed方向不一致，视为持平；J0/A2三seed均负。
- A2 seed1在task3 epoch27发生一次AMP GradScaler skip，最终13,949而非13,950 updates；loss有限且
  无OOM/NaN。仅占总预算0.0072%，不会解释A2相对R1的`-0.5918`，但正式归档应脚注或单独复跑。
  完整报告见`output/emotic_track_a_locked_architecture_test/20260914_013250/analysis.md`。
# 2026-09-14：开始共享多视图DGL梯度审计与G0/G1/G2

- 从`exp/three-architecture-locked-test`的clean HEAD `cadb40b`创建
  `exp/shared-multiview-dgl-audit`；保留四份用户未跟踪Adapter调参文档和`tmp/`，未纳入本实验。
- 目标是在Full/Person/Face继续共享selectors/prompts/head/Image-token Adapter的前提下，区分普通
  J0联合反传、仅截断融合到表示的梯度G1、以及单视图目标只更新表示且融合BCE只更新Head的完整G2。
- runner新增`joint/fusion_detach/dgl`梯度模式、显式参数分组、可靠Face逐视图BCE/ASL、首batch逐task
  梯度范数/余弦审计；评估新增三分支独立mAP、可靠Face、逐lane权重分布，以及融合相对Full纠正/破坏
  的正负排序对统计。前向数值、固定R1特征权重和增量task冻结语义不变。
- 新增三GPU seed0完整8-task validation launcher与严格汇总器。G0为fresh J0(aux0.1)，G1保持aux0.1
  但detach融合输入，G2使用alpha1完整双向梯度隔离且关闭旧aux项。固定30 epochs/task、batch64、冠军
  Adapter、主BCE+Adapter ASL、AMP/TF32、val-only、无checkpoint、禁止test。
- 协议见`docs/shared_multiview_dgl_validation.md`。当前已完成本地py_compile、shell语法与diff检查；本地
  无torch/pytest，完整测试和真实GPU smoke须在服务器`ddp`环境执行。
- 首次服务器210项`unittest`全测通过；G0/G1真实task0 smoke通过，G2在任何训练前因main入口错误
  引用了仅存在于内部`train_task`的蒸馏CLI字段而安全停止。已删除该无效入口检查；DGL内部仍严格
  拒绝蒸馏/ranking组合，需提交后重跑全测与G2 smoke。
- 修复提交`ed17f45`已ff-only同步独立worktree，210项全测再次通过。G2真实task0一轮完成84 updates、
  skipped0；三组均生成view diagnostics。G0融合梯度范数`0.000662`，对Full/Person/Face单视图范数比
  `0.734/7.487/4.078`，余弦`0.432/0.091/0.165`；G1/G2因detach而融合到共享表示梯度精确为0。
  task0 smoke G0/G1/G2 fused mAP约`39.7243/39.9594/41.5154`，仅证明路径可训练，不参与选择。
- smoke记录提交`f83f648`已推送并ff-only同步。GPU0/1/2各约24GiB空闲时启动唯一完整batch
  `shared_dgl_seed0_20260914_215006`，tmux `multilane_shared_dgl_215006`。三组config均为clean commit
  `f83f648`、seed0、8 tasks、val-only、checkpoint off；routing/aux分别为
  `joint/0.1`、`fusion_detach/0.1`、`dgl/0`，当前均已进入task0。
- batch已完成并同步；三组均为240 epochs、13,950 updates、0 skipped，无OOM/NaN且未访问test。
  results/control/logs共29项逐文件SHA-256与服务器一致。G0/G1/G2 final validation mAP分别为
  `43.0754/41.4177/42.4147`，average mAP为`49.9807/48.1986/49.3864`，forgetting为
  `0.9672/0.9526/0.8232`。G2比G1恢复`+0.9970`，但仍比同批G0低`0.6607`，Standalone Full低
  `0.8841`，四项预注册门槛全部失败，不进入动态Router、seed1/2或test。
- G0八个task共24个fused-to-view余弦均为正，Full/Person/Face均值为`0.7835/0.6645/0.6497`；
  task0对Person/Face的范数放大未持续到后续task。G2提高Person/可靠Face约`+0.55`并减少forgetting，
  但牺牲主导Full并出现Suffering`+9.7569`与Sadness`-10.1794`的类别交换，当前证据不支持融合梯度
  冲突是共享三视图性能差距的主因。完整结果写入`docs/shared_multiview_dgl_validation.md`。

## 2026-09-17：视图专用 Selector seed0 validation 已结束

- 实验分支 `exp/view-specialized-selector`（`ce0dc19`）的 S0/S1/S2 在服务器 clean worktree 完成：均为 8 task × 30 epochs、13,950 updates、0 skipped，val-only、无 checkpoint、无 OOM/NaN/traceback。
- 对照 S0(shared×10)、S1(view-specific×10/视图)、S2(shared×30) final/average mAP 为 `42.7525/49.6763`、`41.3856/47.7380`、`41.7839/48.9822`。S1 比 S0 final `-1.3669`，也比同容量 S2 `-0.3983`；S1 八个任务及 Full/Person/Face/reliable-Face 单路均低。
- 按预注册规则停止该实现版本：不补 seed1/2、不进 test、不启动态 Router/ViT 解冻。15 份结果 JSON 和 5 份日志/控制文件已同步，逐文件 SHA-256 与服务器一致；详见 `output/emotic_track_a_view_specialized_selector/selector_validation_seed0_20260916/analysis.md`。

## 2026-09-17：视图专用 Selector 工程审计结论

- 对 S1 的实现审计已完成：225/225 单测通过；Person bank 单独扰动只改变 Person 分支，Full/Face 逐元素不变；三份 config 除 Selector 预期字段外一致；三组均 13,950 updates、0 skipped、无 OOM/NaN/traceback。因此没有发现 Selector 索引、初始化、任务复制或优化器接线错误。
- S1 的下降更符合机制原因：S0 一套 Selector 同时接收三视图梯度，形成跨视图平均/正则化；S1 三个 bank 各自只接收单路梯度，与仍采用固定融合的结构组合后造成独立 bank 过拟合和特征失配。S1 task0 训练 loss 更低但 validation 更差，且一轮 smoke 的短期优势在30 epochs后反转。
- S2 共享30 Selector也比S0差，说明增加容量本身无益；S1仍比同容量S2低，说明视图拆分有额外代价。详细证据见 `output/emotic_track_a_view_specialized_selector/selector_validation_seed0_20260916/analysis.md`。

## 2026-09-17：视图专用下游组件实验

- 分支`exp/view-private-components`扩展了Prompt、Image-token Adapter和分类head的视图独立模式，
  用P0--P3/H0--H3验证S1下降是否来自下游共享；固定CLIP、Selector和三视图融合协议不变。
- 修复Prompt view-specific bank初始化扩展维度，并将回归测试改为符合全类别logit接口的8类断言；
  修复后须在服务器`ddp`环境完成完整单测和代表性GPU smoke再启动正式validation。
- 代表性AMP smoke首次运行暴露smoke入口变量未初始化（模型未开始训练）；已修复并需重新执行四组smoke。
- 服务器正式batch `view_private_components_seed0_20260917_102434` 已启动（commit `36de565`，tmux `multilane_view_private_102434`，GPU0--7）；首轮8组均84 updates、0 skipped，当前等待8-task validation完成。

## 2026-09-17：视图专用下游组件 validation 结论

- P0复现S1；P2（仅Image-token Adapter改为Full/Person/Face独立bank）是唯一完整通过局部筛选的arm：final/average mAP `42.2652/48.6716`，相对P0 `+0.8796/+0.9336`，所有单视图最终mAP均提升，且1705原图组bootstrap 95%区间为`[+0.0403,+1.6639]`。P2仅恢复S1相对S0损失64.3%，仍距S0 `0.4873`，不能称为shared-Selector问题已恢复。
- Prompt独立P1只有`+0.4580` final，私有分类head没有弥补特征层瓶颈。Prompt+独立Adapter组合在task3出现AMP数值不稳定：P3和H2非有限logit退出，H3虽完成仍有166 skipped updates；不以这些结果选择后续方向。
- 本批结果、失败日志和完整分析已本地同步且哈希一致，报告位于`output/emotic_track_a_view_private_components/view_private_components_seed0_20260917_102434/analysis.md`。

## 2026-09-17：P3稳定性修正版

- 从`exp/view-private-components`创建`exp/view-private-components-p3-stable`，只在runner增加可选的
  post-unscale global gradient clipping；P3验证固定阈值1.0，其他训练配置保持不变。
- 目标是判断Prompt与独立Adapter的收益是否叠加。必须先完成服务器全测、P3 task0 smoke和4-task稳定性检查，
  再启动唯一8-task seed0 validation；不访问test。
- gradient-clip=1.0试跑未能阻止AMP overflow；后续P3稳定验证只关闭AMP并保持其余配置不变，以隔离数值精度因素。

## 2026-09-17：P3 FP32稳定性结果

- FP32 P3 batch `view_private_components_p3_stable_fp32_seed0_20260917_152148`（commit `c6ef90b`）已完整结束并同步；240 epochs、13,950 updates、zero skipped，日志无non-finite/OOM/traceback，服务器与本地逐文件SHA-256一致，未访问test、未保存full checkpoint。AMP数值不稳定已被`--no-amp`消除。
- 但P1私有Prompt与P2独立Adapter的收益没有叠加：P3 final/average mAP为`40.9340/47.5726`，相对P0为`-0.4516/-0.1654`，相对P2为`-1.3312/-1.0989`；P3低于P1/P2的每一个累计task。Person/reliable-Face相对P0分别`-0.9828/-0.6897`，预注册分支保护失败。
- 1705原图组、2000次配对bootstrap的P3−P0为`-0.4548`（95%`[-1.2159,+0.2937]`），P3−P2为`-1.3160`（95%`[-2.0876,-0.5579]`）。前者对固定模型的样本不确定性跨0，不能据此声称总体退化；但不支持叠加且明显低于P2。
- P3与旧臂唯一训练差异为AMP关闭（TF32仍开），因此FP32/AMP是严格因果对照的混杂因素。当前证据只保留P2为局部候选；不运行P3 seed1/2或test，也不继续all-private下游组合。完整报告见`output/emotic_track_a_view_private_components_p3_stable/view_private_components_p3_stable_fp32_seed0_20260917_152148/analysis.md`。

## 2026-09-17：FP32交互项与选择性 Prompt 实验准备

- 新分支 `exp/view-private-components-fp32-selective-prompt` 基于 `c6ef90b` 创建，保留所有既有未提交用户文档和 `tmp/`。
- `MultiLaneModel` 新增 `prompt_mode=view_specific_full_face`：bank 0 为 Person 使用的共享 Prompt，bank 1/2 分别为 Full/Face 私有 Prompt；三者从同一初始化复制，任务激活时逐 bank 复制上一任务。
- runner/smoke CLI、Prompt 元数据、回归测试和 validation launcher 已更新。新增 FP32/TF32 固定精度 F0/F1/F2/F3 与选择性 S1 五臂入口及交互项汇总器。
- 待本地静态检查后提交并同步服务器，在 `ddp` 环境运行完整测试、task0 smoke，再启动唯一 validation batch；仅 validation、禁止 test、无 full checkpoint。
- 正式 batch `fp32_interaction_selective_seed0_20260917_230307` 已在服务器 tmux `multilane_fp32_interaction_230307` 启动，使用 GPU0--4（当时每卡约5.6GiB空闲，launcher 门槛按5GiB执行）。

## 2026-09-18：FP32交互与选择性 Prompt validation 结果

- F0/F1/F2/F3/S1 均完成 240 epochs、13,950 updates、zero skipped；日志均以 `MULTI_LANE_TRACK_A_COMPLETE` 结束。服务器与本地 output/log 共132个文件逐项 SHA-256 一致；仅 validation，未访问 test、未保存 full checkpoint。
- 同精度 final/average mAP：F0 `41.7772/49.1334`，F1 `41.4559/48.7443`，F2 `41.6545/48.0103`，F3 `41.2212/48.4650`，S1 `41.8203/48.0093`。
- 交互项 `F3-F1-F2+F0` 为 final mAP `-0.1121`、average mAP `+0.8438`；final 交互项略负但 average 为正，不能宣称稳定的结构负交互。S1 相对 F2 final `+0.1659`、average `-0.0010`，Full/Face/reliable-Face 改善而 Person 下降。
- 汇总报告：`output/emotic_track_a_fp32_interaction_selective/fp32_interaction_selective_seed0_20260917_230307/interaction_summary.md`。

## 2026-09-18：共享基座视图残差与晚层 Prompt 实验启动

- 新分支 `exp/shared-view-residuals-late-prompts` 已推送，最新提交 `d9d8b2a`。实现 `shared_residual` Selector、共享 Adapter 加视图残差，以及逐层布局的 `late_view_residual_full_face` Prompt。
- 服务器 `ddp` 完整单测 231/231 通过；R0/R1/R2/R3 四组真实 ViT FP32 task0 smoke 全部通过。R1 Selector residual scale=0.1、通用 L2 系数=0.1；R3 前3层共享、最后2层 Full/Face 私有、Person全层共享。
- 正式 batch `shared_residuals_late_prompt_seed0_20260918_103705` 已在 tmux `multilane_shared_residuals_103705` 启动，GPU0--3 并行，validation-only、无 full checkpoint、禁止 test。

## 2026-09-18：共享基座视图残差与晚层 Prompt validation 结果

- batch `shared_residuals_late_prompt_seed0_20260918_103705` 的 R0/R1/R2/R3 均完成 240 epochs、13,950 updates、zero skipped；日志无 OOM、NaN、traceback，服务器与本地 98 个 output/log 文件逐项 SHA-256 一致。仅 validation，未访问 test、未保存 full checkpoint。
- Final/average mAP：R0 43.0394/50.5326，R1 Selector residual 41.9762/48.7030，R2 Adapter residual 42.4221/49.6954，R3 晚层 Full/Face Prompt residual 42.4917/49.9806。相对 R0，R1 -1.0632/-1.8296，R2 -0.6173/-0.8372，R3 -0.5477/-0.5520；三种残差在 seed0 上都没有超过共享基线。
- 最终 task7 累计 mAP：R0 43.039、R1 41.976、R2 42.422、R3 42.492。逐 task mAP 也显示 R1 从 task0 开始全面落后；R2/R3只有早期局部接近或超过R0，后期仍落后。
- task7 视图诊断（fused/full/person/face/reliable-face mAP）：R0 43.036/41.640/40.152/34.107/37.849；R1 41.973/40.370/40.373/33.823/36.894；R2 42.417/41.070/38.766/33.955/37.291；R3 42.494/41.279/38.702/33.813/37.026。R1 对 Person 相对R0略升但 Full/Face/reliable-Face下降；R2/R3的 Person 和 Face 均下降，未形成可叠加的单路收益。
- 解释：通用小残差并没有带来可见收益，说明当前瓶颈可能不是参数自由度不足，而是三路表示在固定融合权重和共享后续 head 下的校准/兼容性，以及增量任务中的累计漂移。R1 的 Selector residual 还可能破坏原共享 Selector 的跨视图正则；R2/R3保留共享基座但残差仍改变了各路坐标，无法自动改善融合。
- 本结果不支持继续盲目增加残差容量。若继续，应先做多 seed 复现或冻结训练策略下的融合/校准实验，明确验证动态融合是否能利用三路互补；不要直接启动 test。

## 2026-09-18：R0 与动态三视图融合 test 准备

- 用户要求比较当前共享基座 R0 与动态融合版本。当前实现已有 `soft_three_view`：每个增量 task 使用 hidden-16 masked-softmax Router，输入对应 Full/Person/Face task-lane 特征，初始化为固定可靠 Face 先验 `[0.64,0.16,0.20]`，不可靠 Face 时强制 Face 权重为0并回退到 Full/Person。
- 新增专用脚本 `scripts/emotic/run_multilane_track_a_r0_dynamic_test.sh` 与 `scripts/emotic/launch_multilane_track_a_r0_dynamic_test.sh`。R0 与 D1 仅改变 `fixed_three_view`/`soft_three_view`，其余保持当前 R0：shared Selector/Prompt/Adapter、Image-token layer1 bottleneck32、FP32+TF32、seed0、8 tasks、30 epochs/task、batch64、joint gradient、auxiliary view loss0.1、无checkpoint、一次性test无test-side搜索。
- 历史同类动态 J1 validation final mAP 为`39.8569`，固定 J0 为`42.5330`，因此本批动态 test 属于用户明确要求的诊断性 held-out comparison，不把 test 结果用于回溯选择模型或调权重。

## 2026-09-18：R0 与动态三视图融合 test 结果

- batch `r0_dynamic_fusion_test_20260918_152435` 的 R0/D1 均完成 240 epochs、13,950 updates、zero skipped，无 OOM、NaN、非有限值或 traceback；服务器与本地 34 个 output/log 文件 SHA-256 全部一致。仅按锁定规则评估 test，未在 test 搜索权重。
- R0 fixed final/average mAP=`31.9883/38.8315`，D1 soft Router=`29.5224/36.1115`；D1 相对 R0 final/average mAP `-2.4658/-2.7200`，cF1 `-0.6820`，oF1 `-1.7644`，forgetting 略低 `-0.1254`但不足以抵消性能损失。
- D1 在全部 8 个累计 task 上都低于 R0，task0--7 mAP 差值为 `-4.0987/-3.2065/-2.4326/-2.1804/-2.6097/-2.2856/-2.4806/-2.4658`。
- task7 D1 Router 权重明显偏向 Person：lane0/lane1/lane2/lane7 的平均 Full/Person/Face 分别约为 `0.038/0.909/0.053`、`0.019/0.896/0.085`、`0.091/0.838/0.072`、`0.046/0.771/0.183`；而 R0 实际平均约 `0.703/0.176/0.122`。动态路由出现 Person collapse，未学到稳定互补组合。
- 结论：动态比例在当前联合训练下不能直接替代固定融合；问题更像是路由主损失驱动、缺乏先验/校准约束，而非工程或数值故障。若继续，只应在 validation 上预注册受限 prior/KL、OOF image-group calibration 或小幅 residual Router，并做多 seed 复现；不基于本次 test 继续搜索。

## 2026-09-23：ParaX shared level routing 实验实现

- 当前分支为 `exp/parax-level-routing`；新增 token-only `ParaXImageAdapterBank`，采用 CLS 旁路、共享 expert center、逐层 router，可选 Full/Person/Face level embedding 与 static uniform routing 对照。
- ParaX 主路径位于冻结 CLIP block 之间：图像 block 参数保持冻结，但允许梯度穿过 block 输入返回 ParaX；现有 image-token Adapter 默认关闭。`post` 是最后 block 前的 terminal pre-consumer proxy。
- runner/smoke 已接入 ParaX mode、rank、expert 数、层索引、router hidden、残差尺度和 official/small 初始化；training history 记录逐视图/逐层 gate 均值、熵、top-expert 频率、跨视图 gate L1 距离和 residual/token norm ratio。
- 已完成本地 `py_compile` 与 `git diff --check`；本机缺少 torch/numpy，依赖型单测与真实 ViT smoke 待服务器 `ddp` 环境完成。

## 2026-09-25：ParaX 后置与融合残差结论

- ParaX strict zero-output 后置路径已经与 B0 完全对齐，证明此前非零 ParaX 下降来自真实 residual，不是 RNG、DataLoader 或 forward 实现错误；image-stream ParaX 与后置 ParaX 均停止扩展。
- `post_calibration_control_20260925_134822` 两组均完成90 epochs、5,010 updates、zero skipped；本地同步的10份结果JSON与4份日志和服务器SHA-256一致。`V-post-residual` 相对同批 B0 final/average mAP 为 `-0.0263/-0.2699`，task0 为 `-0.6981`，final Full/Person/Face 为 `-1.0351/-0.2047/-0.9411`。
- 本批 B0 final mAP 只有`43.4145`，低于 strict paired B0 `45.2395`；配置审计确认唯一差异是 auxiliary view loss 从`0.1`改为`0`。因此本批只支持“特征 residual 没有优于其同批控制”，不能替代既有B0。
- 下一步转为固定B0训练梯度不变的task-local逐类别logit residual：Person/Face相对B0 logit差值使用detached输入，系数零初始化并由tanh限制到`±0.1`，每task仅52参数；另立`exp/fixed-fusion-residual-calibration`。
- 服务器clean worktree在`865aec1`通过242项全测与真实task0 smoke；正式batch`logit_residual_control_20260925_163436`已在GPU0/1启动B0-paired和L-post-logit。两组首轮均84 updates、zero skipped，base loss精确同为`0.67691673`；等待自然完成后统一同步分析。

## 2026-09-25：logit residual结果与独立视图Adapter下一阶段

- 批次`logit_residual_control_20260925_163436`已结束并完整同步：两组均90 epochs、5,010 updates、zero skipped；24个本地/服务器文件SHA-256逐项一致，无OOM/NaN/traceback。
- B0/L-post-logit final mAP=`45.2395/45.3251`，差`+0.0857`；average差`+0.1027`，task0/1/2差`+0.0977/+0.1248/+0.0857`，forgetting改善`0.0302`。单路指标和base BCE完全相同，说明增益仅来自校准。
- task0/1/2排序净纠正分别增加`1,389/4,139/4,186`，但task2额外损坏12,546个正确pair；系数有限非零且未饱和。final未达到预注册`+0.10`，因此不补seed1/2或test，关闭当前ParaX/residual/calibration序列。
- 新建`exp/shared-selector-independent-adapter`。下一阶段保持冻结CLIP、共享Selector/Prompt/head和固定融合，比较A0共享b32、A-cap共享b97（149,857参数/task）和A-view三路独立b32（149,856参数/task）。三路bank从配对共享b32初始化复制，确保初始forward严格一致。
- 新增只读per-view Adapter residual ratio和梯度范数诊断，不进入loss；新增task0--2 validation入口与预注册协议`docs/shared_selector_independent_adapter_plan_20260925.md`。待提交、服务器全测和三组真实smoke通过后启动正式实验。
- 分支提交`284a34d`已推送，服务器新建clean worktree `/mnt/haoyuan/workspace/multi-lane-main-shared-selector-independent-adapter`；完整245项单测通过，未修改主工作树或`multi-lane-main-test-only`。
- A0/A-cap/A-view真实task0一轮smoke均84 updates、zero skipped，参数量严格为`49,952/149,857/149,856`。A-view三路梯度有限非零，Full/Person/Face residual ratio约`0.00168/0.00328/0.00677`；可以启动正式task0--2 validation。
- 正式batch`shared_selector_independent_adapter_20260925_1803`已从clean commit `323ba28`启动，GPU0/1/2分别运行A0/A-cap/A-view。固定seed0、task0--2、30 epochs/task、batch64、BCE/ASL、auxiliary0.1、AMP/TF32、validation-only、无checkpoint/test。三组首轮均84 updates、zero skipped，参数计数正确，无立即错误；后续等待自然完成，不持续监控。

## 2026-09-25：共享 Adapter 容量完整 validation 已声明

三任务配对筛选显示共享 b97 是当前唯一同时提升融合与三路单视图的候选，三路独立 b32 未通过门槛。新分支 `exp/shared-capacity-lane-freeze` 新增 A0/A-cap 两组完整 8-task validation 入口；配置保持冻结 CLIP、共享 Selector/Prompt/head、固定 reliable-Face 融合、layer1 Image-token Adapter、BCE+Adapter ASL、auxiliary0.1、seed0、30 epochs/task、batch64、AMP/TF32，仅比较共享 bottleneck 32 与 97。validation-only，禁止 Router、level embedding、private view、蒸馏、checkpoint 和 test。

## 2026-09-25：共享 Adapter 容量完整 validation 已启动

服务器 clean worktree `/mnt/haoyuan/workspace/multi-lane-main-shared-capacity-full` 使用 commit `5ef84d2`，完整单测 `245/245` 通过；A0 shared b32 与 A-cap shared b97 的真实 task0 一轮 smoke 均完成84 updates、zero skipped，参数量分别为49,952与149,857/task，无数值错误。正式 batch `shared_capacity_full_validation_20260925_2015` 已在 GPU0/1 启动，8 tasks、30 epochs/task、validation-only、无 checkpoint/test；训练期间不持续轮询。
- 2026-09-25 shared-capacity full validation: A0 shared b32 completed all 8 tasks (240 epochs, 13,950 updates), final/average mAP `42.7525/49.6763`, forgetting `0.8915`. A-cap shared b97 improved task0--2 by `+2.5218/+1.6715/+1.0746`, then completed task3 cycles 1--17 with 140 skipped AMP updates before non-finite logits stopped cycle18. Saved task0/1 residual-energy ratios were much larger than b32; no task3 failure-batch ratio was persisted. See `docs/shared_capacity_full_validation_results_20260925.md`.
- Follow-up branch `exp/shared-capacity-residual-control`, commit `a6b73a9`: b32/b97 paired 8-task validation at Adapter residual scale `0.03`. Both task0 smokes passed; batch `shared_capacity_residual_scale_validation_20260925_222440` entered task0 on GPUs 0/1, validation only, no test/checkpoint.

## 2026-09-26：共享 Adapter residual-scale test 已启动

- residual-scale validation 已结束：A0/b32 与 A-cap/b97 均完成 8 tasks × 30 epochs，所有 task 0 AMP skipped。A0 final/mean mAP=`42.3199/48.9646`；A-cap=`42.5853/50.2722`，差值 `+0.2654/+1.3077`。b97 的 final 增益低于预注册 `+0.5` 晋级线；按用户明确请求，test 将两组作为 paired capacity comparison，不作 test-side 搜索。
- test batch `shared_capacity_residual_scale_test_20260925_160954` 已在服务器 GPU2/3 启动，seed0、8 tasks、30 epochs/task、batch64，b32/b97 两臂 residual scale 0.03，其余训练和三视图固定融合设置保持一致。无 checkpoint，因此两臂从头在 train split 训练并在 held-out test 评估；不是复用 validation 权重。
- Face reliable mask 固定为 `valid_face && !ambiguous_match && face_short_side >= 24 && face_detection_score >= 0.6`；可靠时融合 `[0.64,0.16,0.20]`，否则 `[0.80,0.20,0]`。该规则是 manifest 的数据质量代理，不是学习得到的 confidence。协议和路径见 `docs/shared_capacity_residual_scale_test_plan_20260926.md`。
- test Face manifest 共5,368个person样本，其中3,261个（60.75%）通过固定可靠性筛选；该比例只表示可用覆盖率，不表示Face情绪预测有效。
- 日志、结果、control 分别在服务器 `logs/emotic_track_a_shared_capacity_residual_scale/shared_capacity_residual_scale_test_20260925_160954/`、外部 benchmark 结果根目录和 `output/emotic_track_a_shared_capacity_residual_scale/shared_capacity_residual_scale_test_20260925_160954/`；不要重复启动，待完成后同步 test 输出和日志。

## 2026-09-26：shared capacity residual-scale validation 分析与多 seed 复现

- validation JSON、NPZ score dumps、日志已同步到本地并核验30个文件SHA-256全部一致。两臂均240 epochs、13,950 updates、zero skipped。
- b97相对b32的final/mean cumulative-task mAP为`+0.2654/+1.3077`，final cF1/oF1为`+0.0609/+0.0130`，forgetting高`0.0655`。fused mAP八个task均正向，但优势从task0 `+3.287`降至task7 `+0.265`；主要提升来自Full/Person。task7 Face `+0.054`、reliable-Face `-0.125`，可靠Face在task0--3均下降。
- task7 b97三路Adapter residual energy/token energy ratios为`7.076/8.277/10.702`，task3为`115.259/142.915/187.276`，显著高于b32且无数值跳步。这是每epoch跨训练batch的能量比，不是有界norm比例；scale0.03不保证实际残差很小，但不能单凭该诊断归因AP变化。
- 架构审计：b97没有解冻CLIP，也没有ParaX。共享Image-token Adapter以同一组权重处理不同视图tokens并为selector提供选择证据；CLIP视觉image-token residual stream仍冻结。三路使用相同固定可靠Face融合先验。
- 下一步用seed1、2复现b32/b97 scale0.03的8-task validation，检验容量增益是否跨seed稳定；配置只更换seed，其余同seed0，GPU0/1运行，held-out seed0 test继续使用GPU2/3，validation不访问test。详细分析见`docs/shared_capacity_residual_scale_validation_results_20260926.md`。
- 已在服务器 clean worktree `/mnt/haoyuan/workspace/multi-lane-main-shared-capacity-multiseed-validation`（HEAD `1df3398`）启动 batch `shared_capacity_residual_scale_multiseed_seed12_20260926`，tmux `ml_caps12_20260926`。seed1 的 b32/b97 已分别占用GPU0/1进入训练；seed1完成后脚本自动在同卡运行seed2。服务器245项 unittest、bash语法检查和两组真实ViT smoke均通过；seed0 test仍在GPU2/3，未被新validation占用。


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


## 2026-10-20 Frozen Forward路由只读对照
用户授权执行task0/2/7动态Top3、TRAIN锁定固定Top3及关闭残差的validation。新独立分支codex/moe-frozen-route-control-20261020基于b63616b，保留原分支未提交分析文档。源Frozen完整训练已结束，val42.0412/avg48.3758、test31.6928/avg35.4285。仅GPU0，4 TRAIN批/阶段（人物上限64、原图分组）后完整val2397人、workers0、bootstrap500、0训练更新、无test。独立image_token Adapter未启用；MoE本身已适配图像tokens，两模式当前互斥。源main/test-only保持原dirty状态，不修改。协议docs/moe_frozen_route_control_20261020.md。

2026-10-20预检：首次完整367项测试中366通过，新增schema恢复测试的合成训练前向误用no_grad，按设计不累计训练激活，finish_task因此拒绝空计数。已修正测试准备步骤，不改变模型/真实诊断逻辑；未启动真实smoke或正式验证，失败日志保留。
