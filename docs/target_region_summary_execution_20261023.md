# 人物区域与区域外上下文分别归一化：方案与执行记录

2026-10-23。用户已授权独立改进目标人物位置摘要路线，seed0、完整8任务validation/test，在GPU0排队。本轮从已验证空间摘要版本1e28bfb创建exp/target-region-summary-20261023，独立工作树；不改另一分支的三模块叠加实验。

## 为什么修改

上一版Full Summarization在语义logits上加小幅几何bias，最终val42.9717，关闭位置项42.9723；同图换框也没有可靠的正确人物优势。它获得了训练后模型的性能候选，但未证明推理时位置读取有效。单纯放大bias不能解决区域竞争问题，人物内容Query/Attention组合则由另一分支研究。

本轮假设：全图归一化可能使小人物的证据被更多场景patch稀释；将人体区域和区域外上下文分别归一化，再控制它们在摘要中的份额，可能让任务Selectors读到更平衡的证据。这是待验证的结构假设，不是已确认的唯一失败原因。区域外仍包含其他人，深层patch仍有全局混合，归一化可能放大小区域噪声。

## 数据流与精确定义

原Frozen Forward不改。对每层Full的原归一化图像tokens F，任务Selectors S产生原语义logits e。用实际Full裁剪/翻转后未截断人体框，计算每个patch的人体覆盖率m；CLS的区域权重为0。

- 原全图注意力A0=softmax(e)，保持原dtype与收缩方式。
- 人体注意力Ab=softmax(e+log(m))，m=0严格屏蔽。
- 区域外注意力Ac=softmax(e+log(1-m))，CLS也屏蔽；边界patch按覆盖率分配，并非硬二值分割。
- 每任务每Selector一个标量u，q=0.25+0.5 sigmoid(u)，初值0.5、范围[0.25,0.75]；三视图不新增Router。
- r=0.5×真实可见性。只要人体或区域外小于0.25个patch等价面积，或框无效/不可见，r=0。
- A=(1-r)A0+r[q Ab+(1-q)Ac]。摘要仍为A F，不新增Value投影或生成残差。

因此至少50%原全图读取被保留，区域分配在训练和推理一致。它不会增加Summarized Patches数量（仍10个）或改变Task Forward的MSA、Drop & Replace、FFN及分类头。Person/Face保留原路径；同任务共享Selector/Prompt/head，末端原固定特征融合。每任务10个参数、12层共享，8任务80个；旧任务参数冻结，无跨任务可变共享池。新模式关闭时逐值恢复该checkpoint的原softmax，不回滚共同训练参数。

与旧零bias初始化不同，新区域结构在u=0时已经生效；不宣称开启时初始预测等于基线。要求三组共同原参数和RNG一致，新增模式关闭/无效时与基线逐值相同，Person/Face初始端点相同。

## 锁定实验

三组同批重训：baseline=disabled、old_bias=target、regional=regional。EMOTIC26类，任务5/3/3/3/3/3/3/3；seed0、30epoch/task、B64，原图固定90/10 fit/holdout，每组12660成功更新，workers/eval-workers0。Adam每任务重置、主LR0.0125、空间参数LR0.0004、WD0、cosine到0；AMP1024/growth1e9、TF32。融合joint BCE＋0.1倍可用视图辅助BCE，legacy_full_zero，无ASL、绑定loss或蒸馏。

CLIP ViT-B/16从原预训练权重独立初始化，不接着旧训练checkpoint训练；Full legacy crop0.05–1；Person margin0.15/letterbox、无color jitter；Face既有manifest。可靠Face的Full/Person/Face特征权重0.64/0.16/0.20，否则0.80/0.20/0。MoE、ParaX、image-token Adapter、cross-view Attention、joint-summary及旧Selector conditioner均关闭。

沿用已确认的相对资源：

- 数据：../multi-lane-main/datasets/EMOTIC
- 预训练：../CODE_DDP/pretrained/clip/ViT-B-16.pt
- Face：../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
- 只读历史重放：../multi-lane-main-target-summary-gpu0-resume-20261022/output/target_anchored_summarization/formal_gpu0_resume_seed0_20261022_01/baseline/train

入口bash scripts/emotic/run_target_region_summary.sh，正式run-id formal_seed0_20261023_01。输出output/target_region_summary/<run-id>/，日志logs/target_region_summary/<run-id>/。先CPU回归，通过后提交push，服务器只通过Git fetch创建独立worktree。

## GPU0排队和预检

明确等待另一分支output/three_module_stacking/formal_seed0_20261023_01/completion.json，必须status=complete、smoke=false、tasks=8；不存在则等待，失败则本批报错并保留状态，不启动训练。叠加实验的smoke完成或训练完成不足以放行。本批所有GPU子进程CUDA_VISIBLE_DEVICES=0，分配器上限16%，最多3并发、启动余量至少5600MiB；不足自动等、不降B64、不停止其他作业。

依赖结束后自动执行：真实CLIP B64初始化/历史0、2、7重放→三组2任务×2更新短跑→regional从task0完整快照恢复并对齐模型/优化器/RNG/采样/预测→六路短跑validation及位置干预→正式完整8任务→三组完整validation2397人/test5368人→old_bias与regional的task0/2/7 validation正常/关闭/同图换框。

每任务先保存完整可恢复状态，再报告；保留compact、原图划分、采样/预算/冻结审计、全部预测、summary.csv及完成/失败状态。不依据中间val/test删组或改超参；test用于用户要求的最终报告，不参与选择。判定需同时看相对匹配基线/旧bias的性能、开启相对关闭净收益，以及正确框相对同图换框的优势；目标区域质量变大本身不是成功。只有seed0不能证明跨seed稳定。

当前：实现及本地静态检查阶段，未声称GPU预检或正式实验已运行。排队后另记录代码版本、远程分支及真实状态。

## 回归与Git-only部署

代码d9757f1已提交push，服务器通过git fetch及worktree add建立独立分支。427项Track-A回归通过（1项仅CUDA检查跳过），新增9项全部通过。原主工作树及test-only状态SHA与开始完全一致。真实GPU初始化、短跑和完整恢复仍待前序叠加批次完成后执行，尚无正式性能结果。下一步启动tmux持久队列并核对status.json，不占用前序批次GPU资源。
