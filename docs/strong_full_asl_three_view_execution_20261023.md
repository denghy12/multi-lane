# 强Full ASL三视图执行记录

状态：全部实验及完整validation／test／关闭归因已完成，结果已同步核验，详见strong_full_asl_three_view_results_20261023.md。

用户授权所有显卡按空闲容量并行。七组新配置＋R历史Full复现；seed0，EMOTIC，全TRAIN，8任务×30epoch/B64，预期13950更新/组；主Adam LR.0125，Adapter/Attention/位置LR.0004，WD0/cosine，aux0。BCE→Selectors/Prompt/head/Attention/位置，ASL9.8/0/.05→Image-token Adapter。新七组AMP1024/growth1e9；R保留f1ada22旧数值与全局采样并单列，不要求新S-ASL精确32.5365。

输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、既有face_manifest_train_val_test_v1_20260909。入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，日志logs/strong_full_asl_three_view/<run_id>，结果output/strong_full_asl_three_view/<run_id>。训练保存边界恢复/compact，之后完整固定人群validation/test及预声明task0/2/7关闭归因；无按test选模型或改配置。

来源远程primary153行SHA ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf；test-only5行SHA e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152。保护这些及其他既有工作树。旧f1ada22工作树clean，不改其源码，包装器通过Git受控新代码加载旧模块，保存不消耗RNG的任务边界及明确历史导出元数据。

## 实现与第一轮预检

432项回归通过，真实CLIP B64初始化7组公共参数/RNG/同视图输出逐值一致；冻结视觉不变。首次smoke因单Full误传Face参数被入口拦截（0更新），修正后smoke_seed0_20261023_02八组训练均complete/0跳步。最重2818.77MiB/保留3034，单Full ASL1357.87MiB。验证器补区分短跑updates和正式epoch预算；新调度按训练→逐组完整性核验→该组val/test及预声明归因的依赖流水执行，评估只用本批已释放卡，每卡最多2路，不与本批活动训练抢同卡。正式未启动。

2026-10-23预检补记：3868f61完整433项回归通过；新流水smoke03八组训练均0跳步，16份同checkpoint validation重放逐值一致，历史Full导出/单视图评估也通过。恢复probe从cd1489d task0接续至task1，model/requires_grad/optimizer-scheduler-scaler/RNG/sampler及两阶段val logits全部逐值一致。当前最后6路关闭归因短跑仍在执行，正式未启动。补充首批各参数组梯度范数只读日志，并实现仅新三视图协议单前向导出融合及各视图score，待数值等价核验；没有更改训练损失或模型公式。


2026-10-23预检全部完成补记：3868f61 smoke03完整30作业（8训练/16validation与重放/6关闭归因）全部exit0，0跳步，16份scores重放逐值一致；恢复probe所有状态与logits逐值一致。历史原始config进一步核对workers=2，严格R保留2/旧随机流；此前包装器smoke用0仅作接口验证，新七组仍固定workers0。5d56449增加单前向融合+视图导出及首batch参数组梯度只读日志，待最后回归/真实评估等价核验后锁定正式，尚未启动正式。

## 2026-10-23：全部预检通过，准备正式八组多卡运行

新协议7组＋旧Full严格复现R，正式全TRAIN/seed0/8task×30epoch/B64/13950updates每组/aux0/主LR.0125/模块LR4e-4/WD0/cosine。BCE→原任务及Attention/位置，ASL9.8/0/.05→Adapter；新7组scaler1024/growth1e9/workers0，R原f1ada22/全局随机流/workers2。R最终另记是否数值匹配32.5365/39.1445，不把预算verified冒充历史得分已复现。

最终434项回归通过，真实CLIP B64公共初始化/RNG/同视图输出逐值一致；完整smoke03共30作业全部成功、0跳步，16份重复预测逐值相同。最后19项真实检查全true：5个三视图模型×2边界单前向导出与原评估逐值相同，历史Full当前/旧代码2边界重放相同，新增梯度日志后恢复model/requires_grad/optimizer-scheduler-scaler/RNG/sampler全等，六参数组梯度范数已记录，R workers2短跑成功。无新超参搜索。

多卡调度每卡1训练，完成核验后释放卡接续每卡最多2路val/test，6路预声明validation关闭归因自动执行；GPU0–7按实际显存准入/不足等待，不改B64或其他作业。正式run_id拟formal_seed0_20261023_01，入口scripts/emotic/run_strong_full_asl_three_view_ablation.sh，输出output/strong_full_asl_three_view/formal_seed0_20261023_01、同名logs；本条为启动前锁定记录，不能作已启动证据。远程primary/test-only和历史代码树继续保持。

## 2026-10-23 15:42 CST：正式八组已同时启动，首轮健康检查通过

覆盖此前“尚未启动/准备启动”状态。运行分支exp/strong-full-asl-three-view-20261023，Git-only同步并锁定HEAD 86a8c1c36a6cdbbe7fac5a584aaba1b7f00aadf4；远程独立worktree multi-lane-main-strong-full-asl-20261023 clean。正式run_id=formal_seed0_20261023_01，tmux=strong_asl_formal_seed0_20261023_01，8组训练均已运行，无待启动组。

GPU0三视图ASL＋Attention＋目标位置；GPU1严格历史Full；GPU2单Full BCE；GPU3单Full Adapter-ASL；GPU4三视图BCE；GPU5三视图Adapter-ASL；GPU6三视图ASL＋Attention；GPU7三视图ASL＋目标位置。各组均seed0/8task/30epoch/B64，内部task0→7顺序继承增量状态，不能把阶段并行当同一实验。GPU1既有外部作业保留，本批小型历史Full使用剩余容量。

正式首轮八组全部84steps/0skips，并已完成后续多个epoch，未发现Traceback/OOM/failure.json。每组训练完成且通过预算、冻结及恢复核验后，自动接续完整固定人群validation/test；释放卡每卡最多并行2路评估，6项预声明validation关闭归因及最终汇总自动执行。尚无正式最终结果，不以启动健康当性能提升。

434项回归、30作业短跑、19项最终真实检查与断点恢复/预测等价预检均通过。已同步416份预检与启动配置文件并逐SHA核验，另存44个checkpoint SHA而未下载权重；归档SHA 0a299b321264e96e52fa69d7473cfb19b9045bf853a5ccd8fa5c5819f6e40870。审计output/strong_full_asl_three_view/startup_audit_20261023_01/{remote_manifest,local_sync_verification,formal_health}.json。primary/test-only原153/5行状态SHA及历史源clean保持，未改用户树。启动后仅补本地执行与上下文记录，不再变动远程运行代码。

| GPU | 运行组 | 健康检查时已完成task0 epoch | 最近记录 |
| --- | --- | --- | --- |
| 0 | T_ASL_BOTH | 3 | 84steps，0skips |
| 1 | R_LEGACY | 12 | 84steps，0skips |
| 2 | S_BCE | 10 | 84steps，0skips |
| 3 | S_ASL | 9 | 84steps，0skips |
| 4 | T_BCE | 4 | 84steps，0skips |
| 5 | T_ASL | 4 | 84steps，0skips |
| 6 | T_ASL_ATTN | 4 | 84steps，0skips |
| 7 | T_ASL_POS | 4 | 84steps，0skips |

上述为启动后快照，实时进度见服务器同run_id的train.log；完整val/test尚未完成。此后无业务代码修改，434项验证无需重复执行。

2026-10-23 18:28 CST最新进度：历史R及S_BCE/S_ASL均训练与完整val/test exit0；仍有五组训练，其中用户所指四个ASL核心组task5进度分别13/15/20/5epoch（T_ASL/ATTN/BOTH/POS），T_BCE为task6第16epoch。总预算13950updates，各组剩1700/1620/1420/2020/860；最近每update约.75–.83秒，四ASL纯训练还19–28分钟，计任务边界报告/保存预留20–35分钟。后接完整val/test与关闭归因，整批粗估还50–80分钟、北京时间约19:20–19:50（保守20:00左右）；此为早期耗时更新，非保证。所有已记录epoch均0skips，无failure；未改配置或进程。快照output/strong_full_asl_three_view/startup_audit_20261023_01/progress_eta_1828.json。

2026-10-23用户要求GPU1–7当前训练跑完即释放，后续完整validation/test与关闭归因仅GPU0。已仅SIGSTOP旧派发器1108543，五个训练PID1108598/1108624/1108650/1108670/1108697继续不重启；新建exp/strong-full-asl-gpu0-eval-20261023，实现只接管原训练PID的独立调度脚本scripts/emotic/resume_strong_full_asl_gpu0.py。GPU0本身训练完成前不派发评估，之后GPU0最多2路，固定B64/workers0/原模型HEAD86a8c1c与全部数据监督不变。新脚本从独立Git工作树运行，原实验代码树不切换；当前正验证接管，不能视为已完成迁移。

2026-10-23 18:42 CST纠正前次接管状态并记录故障：旧派发器SIGTERM/CONT退出导致旧tmux会话关闭，五个原训练子进程收到会话退出影响而终止，原“训练不重启”目标未达到，已向用户明确说明并致歉。原始日志/全部任务边界恢复/模型权重保留，不删除旧failure。最新用户要求剩余工作全部GPU0并行，覆盖GPU1–7续完原进程安排。恢复边界T_BCE/task6、T_ASL/task5、ATTN/task5、POS/task4、BOTH/task6；只需重算各自未完成任务（原日志已做26/10/15/24/1epoch），不能声称epoch无损恢复。

新增scripts/emotic/recover_strong_full_asl_gpu0.py，原模型HEAD86a8c1c clean环境下从全状态边界续到8task，保存至新recovered_gpu0_seed0_20261023_01输出树；3组已完成Full训练及val/test只读链接复用，原输出不覆盖。GPU0至多5训练/2评估且总5进程、显存准入6500MiB，所有子进程新建独立session，防止调度tmux退出连带终止。修改旧接管脚本为保留暂停旧派发器到全部子进程结束后才退出；当前按最新要求不再使用该接管方案。正在预检/部署，尚未宣称恢复已启动。

2026-10-23 18:48 CST：五组剩余训练已全部GPU0并行恢复，PID T_BCE1121327/T_ASL1121349/ATTN1121367/POS1121389/BOTH1121407，均独立session且已进入新的训练epoch，首轮0跳步。当前运行模型HEAD86a8c1c clean，调度脚本独立Git工作树HEAD6eb1e3e；新run_id=recovered_gpu0_seed0_20261023_01，日志/输出均在对应strong_full_asl_three_view目录；原三组已完成Full产物只读链接复用。五组边界validation的全部npz字段（含logits/概率/人群/标签）与原输出逐值相同，已重算的前2/5/4/1/1epoch训练loss/LR/步数均与原轨迹日志一致（耗时除外）。这支持当前恢复正确，不等于未完成部分已验证。

恢复前完整状态校验5/5、调度测试4/4、真实tmux父会话关闭后独立子进程存活探针均通过。原源恢复checkpoint SHA保持，primary/test-only状态仍153/5行且SHA不变。19份故障/预检/恢复启动元数据已同步本地逐SHA核验，归档1001319ab44360837744f1631cb58be0b9886dc480822284881c9438619abf6a。全部后续训练/val/test/6项关闭归因只GPU0，至多5总进程/2评估，显存不足自动等位；旧多卡ETA不再适用。此条覆盖前次“尚未恢复”状态，不掩盖旧tmux接管造成的中断与重算。

## 2026-10-23：强Full ASL三视图与叠加实验全部结束，完整分析完成

recovered_gpu0_seed0_20261023_01 completion=complete，21个恢复批次作业全部exit0；合并原3组Full已完成结果，八组训练/16份完整val-test/6项关闭归因齐全。7个新模型均13950updates/0skips/旧任务依赖冻结/恢复与compact模型一致。479份产物已同步并逐SHA核验，独立CPU复算128份常规阶段score与18份运行时score，人群/标签/类别逐值相同，最大mAP误差4.98e−14，模型源码86a8c1c clean。保留前次tmux故障与任务边界重算事实，不将恢复说成逐epoch无损。

四列val最终/val阶段均/test最终/test阶段均：单Full BCE42.3467/48.1833/31.7109/35.1143，单Full ASL42.1754/47.7626/32.2519/35.6855，三视图BCE41.9839/48.1421/31.1035/34.7656，三视图ASL42.5578/49.1965/32.1638/35.6521，+Attention42.5144/48.8822/31.8968/35.5416，+位置42.8656/49.2204/31.8404/35.2431，+两者42.4552/49.2248/31.8472/35.2948。ASL改善三视图BCE但迁移到三视图未超过同批Full test；新增两模块和组合没有验证出叠加收益。位置与组合test全部8阶段低于ASL底座，Attention val全部8阶段低；val最终bootstrap关键差值区间均跨0，只有seed0。

关闭条件贡献（开启−关闭）task7：ASL底座Adapter−.4199、Attention单独−.0412、位置+.0107、组合关Attention+.2081/关位置+.0043/关两者+.2177。不能把关闭版本当独立重训，也没有test关闭数据。Attention实际残差范数足够大，未形成稳定净收益；位置bias小、softmax KL约1e−5–1e−4，val提升集中Sadness等类别，不支持已建立可靠人物对应。三视图ASL test Full端点30.5910→融合32.1638，仍低独立Full32.2519，互补补偿未超过共享训练代价；未做跨视图梯度夹角，不能宣称已证明冲突。

必须纠正R“严格复现”称呼：旧源码Full重跑固定人群41.8703/48.0657/31.9535/35.2819，旧人群test均38.5130，未达历史32.5365/39.1445。原config paired_full_person=true/margin.15/jitter.1/.2，R为false/0/0，匹配不完整，差值因果未分解；保留旧32.5365外部强参照，不以低R替代。详细报告docs/strong_full_asl_three_view_results_20261023.md，CPU复算脚本scripts/emotic/analyze_strong_full_asl_results.py，分析表/审计output/strong_full_asl_three_view/analysis_20261023_01。

当前只完成结果同步/CPU分析与文档，不再运行GPU前向、训练或test搜索。暂停扩展Attention/位置叠加，不合并main。先修正历史Full复现、审计独立专家补救Full的净空间；同时回查既有Full-anchored残差A0/A1/A2 42.3799/41.8802/42.1556以及后置logit+.0857的失败记录，不能将旧路线换名后再训。
