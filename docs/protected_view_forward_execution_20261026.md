# 完整模型的视图来源组合：执行配置


前向完成与导出修复：780732e预检、5组隔离测速及Val/Test完整8任务均exit0，A/B每batch来源及R0/R1保护通过。后验发现将CPU logits拼接后整体sigmoid与历史逐batch sigmoid在Test R0最终仅3个概率位置相差ULP，logits全部相同、mAP约差1e-6；已确认按原batch_lengths分批sigmoid可精确重放。现修复仅CPU导出，新增第8项专项测试，不重训、不重复GPU前向，不改来源checkpoint。将保留formal_01原始产物，生成formal_seed1_20261026_02修正导出及概率逐值审计；bench资源记录仍来自原真实前向源码780732e，export Git单独记录。



执行更新：780732e已Git-only部署remote独立worktree multi-lane-main-protected-view-forward-20261026，branch/HEAD/upstream一致且clean；主树153行及test-only5行dirty状态SHA、HEAD均保持。完整502测试通过（50.496s）。formal_seed1_20261026_01已在tmux protected_view_forward_seed1_20261026运行，真实CLIP task0/2/7各Val首64人预检22.77s成功，A/B全部来源预测、所选分路、R0融合和R1无可靠Face融合均逐值一致，零训练更新。隔离R0–R4测速正在自动执行，随后Val/Test两进程并发完成8阶段。来源24checkpoint SHA预检保持；本地仅追加执行记录，不改变remote 780732e运行源码。首64人mAP不代表完整指标，不用于方案选择。


用户2026-10-26要求落实R0–R3完整模型无训练验证，并保留Full替换控制。实验分支`codex/protected-view-forward-20261026`。仅GPU0，seed1，EMOTIC旧`samples_intersect_seen_classes`标准，完整8阶段Val/Test。无训练、优化器、学习率或参数更新；不生成新的训练checkpoint。

## 目标及结构

A=原Person-only共训表征+B0头；B=原Person+可靠Face共训表征+B0头。R0=(A,A,A)，R1=(A,A,B)，R2=(A,B,A)，R3=(A,B,B)，R4=(B,A,A)，依次为Full/Person/Face来源。原特征融合在B0分类头之前，不用离线logit加权代替。

两个来源分别执行完整三视图encode，保留各自Full摘要K/V及内部Attention；选最终任务特征时才组合。共享同一冻结ViT对象和对应阶段B0头，A/B任务参数仍独立且全部冻结；中间Frozen Forward tokens暂不缓存。R0单独测速只保留A，其他组参考实现执行两套来源、各三视图；不将参数共享误称为图像前向次数减半。

只读来源先完整核验24份checkpoint SHA。task0/2/7的Val首批64人作为前向预检，两个来源所有分路和融合logits/probabilities逐值重放既有NPZ；各组合分路逐值重放所选来源，R0融合精确回到A，R1无可靠Face人群融合精确保持。所有任务参数及冻结ViT哈希不变。

预检通过后，GPU0隔离测量R0–R4的task7/Val首64人：每组独立进程，3次warmup、10次计时，记录CUDA事件/墙钟、峰值allocated/reserved、参数量、冻结图像前向次数和torch.profiler可覆盖FLOPs（不称全算子精确FLOPs）。测速不与评估并发。然后Val和Test两个进程并发，每批只计算一次A/B来源并形成全部5组输出，避免每组重跑两套骨干。

完整评估每阶段、每批都对齐旧NPZ的ID/targets/class/mask及来源输出；保存所有分路/融合预测、分路mAP和可靠Face mAP、fused最终及阶段平均mAP/cF1/oF1、逐阶段审计。分类阈值仍概率>0.5，可靠Face融合0.64/0.16/0.20，不可靠0.80/0.20/0；不调整权重或挑选任务路径。Test继续按用户授权用于探索。

## 配置及路径

- 模型：冻结CLIP ViT-B/16；来源Image-token Adapter b32、第2层、scale0.03/ReLU，来源Attention第6层/inner64/4heads/scale0.1/cap0.1；沿用ASL训练所得权重，评估不计算训练目标。
- seed1，eval batch64，workers2（测速workers0），OMP/MKL72线程，AMP/TF32；不改变历史增强或Face可靠性规则。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
- Face清单：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- B0：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- A：同上来源根`ATTENTION_BOUND/train`。
- B：`../multi-lane-main-head-anchored-eight-face-20261025/output/head_anchored_eight_face/formal_seed1_20261025_01/FACE_JOINT/train`。
- 逐值参考：`../multi-lane-main-selector-face-controls-20261026/output/selector_face_controls/formal_seed1_20261026_01`的REFERENCE_PERSON和REFERENCE_DUAL。
- 输出`output/protected_view_forward/<run_id>`，日志`logs/protected_view_forward/<run_id>`；正式run_id=`formal_seed1_20261026_01`。
- 主入口：`python -m multi_lane.track_a.run_protected_view_forward`，带上述source-run/person-run/dual-run/reference-root/data-root/clip-checkpoint/face-manifest-root和run-id参数。评估/测速入口`multi_lane.track_a.protected_view_forward`。

## 代码与验证

新增protected_view_forward.py、run_protected_view_forward.py及tests/test_protected_view_forward.py；不改原训练器/来源模型实现。专项测试覆盖完整来源调用、融合在头之前、任务类别归属、R0/R1预测保护、共享权重及冻结、来源不匹配拒绝和预测对齐审计。先语法/差异检查、远程专项及完整Track-A测试，通过后同源码预检/测速/正式评估。所有结果留存，不依据离线最好值省略其他组。

远程独立worktree拟为`/mnt/haoyuan/workspace/multi-lane-main-protected-view-forward-20261026`，Git-only同步；主树、test-only和既有实验工作树不修改。此前本地只读分析与报告随该实验分支保留。只有同源码预检成功才接续正式评估。
