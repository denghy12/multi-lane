# 三视图静态 MoE 第一版：seed0 八任务配对实验

2026-10-12最新修订：用户要求validation和test并行，覆盖本文原先val-before-test顺序。训练配置及固定checkpoint均不变；接管/并行来源锁和新评价分支详见 `docs/static_moe_parallel_evaluation_20261012.md`，本文其余顺序描述保留为最初启动历史。

授权日期：2026-10-12。用户要求开始修改第一版并启动实验；依据 AGENTS.md 的实验执行持续授权，可提交/push、Git-only 同步独立服务器 worktree、完成单元测试和 smoke 后启动本批非破坏性实验。不合并 main，不修改主/test-only/其他工作树。

## 目标和四组

检验不同静态视图组合是否比同预算完全共享更适合三路增量学习。不是新的 ParaX，也不训练逐图片 Router。

| 方法 | 可训练 Adapter |
| --- | --- |
| BASELINE | 关闭 Adapter/ParaX |
| SHARED | 每任务一个宽度48的普通共享瓶颈 Adapter |
| MOE_UNIFORM | 每任务3个宽度16的完整输出专家，三路权重均1/3 |
| MOE_OVERLAP | 同一专家池；Full=.5/.5/0，Person=.5/0/.5，Face=0/.5/.5 |

同一任务的三路共享专家池；任务结束后旧池冻结。无跨任务专家复用。每个专家有 down/bias/ReLU/up，up 不含私有输出 bias，池中一个共享输出 bias；这样 SHARED/两种MoE 都精确74544参数/task，八任务总596352。共享输出 bias 本身是三路全共享的少量可训练参数。MoE 首版前向均计算全部3专家，0系数仅令对应梯度为0；uniform/overlap计算图和参数预算一致，不宣称稀疏加速。

位置：代码索引1，即 Task Forward 在第2个 block 读取图像证据时，对 block.ln_1 的图像 tokens 做任务专属残差，供 Selector 匹配/聚合使用。第1个 Frozen Forward block 的输出及后续原始图像残差流不被覆盖。不是原内部 ParaX 索引10的替换，也不是最终特征后置校准。

共享冻结 CLIP ViT-B/16；Selector/Prompt/head 仍按任务分配、在三路间共享。原固定三视图归一化特征融合：可靠 Face=.64/.16/.20，否则=.8/.2/0；保留原 Face 有效性掩码和辅助损失规则。

## 完整训练配置

- EMOTIC：输入 `../multi-lane-main/datasets/EMOTIC`，标注 CVPR17_Annotations.mat。
- CLIP pretrained：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；SHA256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`；沿用有效、无歧义、短边>=24、检测分>=.6；记录train/val/test及detector hash。
- 四组均从同seed0 CLIP pretrained开始训练；不加载历史增量checkpoint，不使用全量26类模型。
- task0–7，类别5/3/3/3/3/3/3/3，30epochs/task；预期240epochs、13950成功updates/组，按实际train人数复核。
- train/eval batch64，workers2，OMP1。Adam按任务重置；任务参数LR=.0125，Adapter LR=.0004，weight decay0。cosine每task重置，minLR0、warmup0。
- joint BCE，legacy_full_zero，temperature1，supervised scale1；可用视图辅助BCE权重.1。无ASL、蒸馏、动态融合、额外正则或路由搜索。
- Adapter ReLU、总b48；MoE每专家b16；固定残差scale=.03；up及输出bias严格零初始化，down独立随机；每task独立池。
- AMP初始1024，growth interval1e9，TF32沿用；旧函数/identity审计FP32且TF32-off。
- Full裁剪legacy、crop_scale .05–1、CLIP归一化；Person margin.15/letterbox/no color jitter；Face沿用manifest处理。
- 四组共享原始Selector/Prompt/head初始化及独立每task采样种子seed+1009*task。MoE uniform/overlap专家初始化hash完全相同，唯一差异为静态路由表。

## Git、入口和资源

- 分支：`exp/three-view-static-moe-v1`，基于815d949。
- 本地独立managed worktree：`/Users/denghaoyuan/.codex/worktrees/three-view-static-moe/multi-lane-main`。
- 服务器新worktree：`../multi-lane-main-three-view-static-moe-v1`；Git fetch/create，核对同branch、HEAD、upstream；只fast-forward更新。
- 服务器已只读检查86个worktree；主目录和test-only有用户改动，保留。其余工作树不修改。
- GPU0；预检时空闲约24GiB，其余卡忙。smoke单组顺序、正式最多2组并行，启动前分别至少8/14GiB空闲；不足时在后台等资源，最长12h，不停止其他进程。
- 单组入口：`bash scripts/emotic/run_static_moe_v1.sh`。
- 总入口：`python -u -m multi_lane.track_a.static_moe_pipeline`。
- smoke batch：`static_moe_v1_smoke_seed0_20261012_01`，全部四组×八任务×2updates/共64updates；每阶段仅64个validation人物，随后32个checkpoint恢复/来源val重放/64人完整cohort预检。不加载test，不作精度判断。
- 正式batch：`static_moe_v1_seed0_eight_task_val_test_20261012_01`。
- 输出：`./output/emotic_static_moe/<batch>/`。日志：`./logs/emotic_static_moe/<batch>/`。

```bash
/opt/conda/envs/ddp/bin/python -u -m multi_lane.track_a.static_moe_pipeline \
  --batch-id static_moe_v1_smoke_seed0_20261012_01 --gpu 0 --parallel 1 --smoke
# 上述成功后启动正式配置
/opt/conda/envs/ddp/bin/python -u -m multi_lane.track_a.static_moe_pipeline \
  --batch-id static_moe_v1_seed0_eight_task_val_test_20261012_01 --gpu 0 --parallel 2
```

## 验证和预期产物

源码语法/Shell/diff检查、全部Track-A单元测试；新增回归覆盖完整专家输出组合、路由梯度支持、精确预算、初始化与采样RNG隔离、旧专家在AdamW下不变、未来task禁用、Face mask、checkpoint恢复、test前置条件。

每个任务审计原旧Selector/Prompt/head逐值不变、旧专家和静态路由表逐值不变、当前专家确实学习、零初始化raw token严格identity；固定16个validation人物只作无优化诊断，旧logit漂移阈值1e-4、三路mean cosine distance绝对值阈值1e-5。异常时保留日志/检查点并停止流水线，不自动调参。

四组全部训练和配对审计完成后，先恢复各组8个固定第30轮checkpoint，在完整2397人validation评价所有已见类，再完成整个四组full_val/comparison.json后才允许创建test数据集。随后对全部5368人做同样八阶段只读test。不选择组别、epoch或checkpoint；所有组都报告，不以validation胜负作为test开关。

来源checkpoint先以训练时同样的过滤validation批次重放，阈值.01；不把完整cohort重新分批后的每张logit硬当成相同batch的数值重放。源模型/CLIP/Face manifest hash前后必须一致。

预期保存：32个compact checkpoint、训练history/逐任务val和view scores、配对采样/旧函数/专家审计、四组完整val/test NPZ、最终mAP及各视图/可靠Face/新旧类/固定cohort遗忘曲线、pipeline manifest/state和退出码。checkpoint留服务器，结束后提醒同步日志和非checkpoint结果再分析。

本轮首版seed0属于结构筛查，不能声称多seed稳定；最终研究结论仍需后续配对多seed复核。正式实验已于2026-10-12 17:40:31（北京时间）启动，运行HEAD固定957832632a372936b5cb752f788a617f77a42443；下面保留预检修复历史及启动证据。

## 预检进展及评价schema修复

实现4f89854已push并Git-only同步新服务器worktree；307项Track-A测试通过。smoke四组八任务各2updates共64updates全部训练exit0。统一评价校验器把更新预算smoke的epochs_per_task错误要求为30，而实际runner写null；配对损失路由字段实际为parameter_group_loss_routing。保留初次failed状态/日志和32个checkpoint，修复校验schema、补正式/更新预算及配对负例回归；直接补恢复评价，不重训，不改模型/超参数/数值阈值。正式调度先BASELINE和MOE_OVERLAP，让主候选从开始就与基线并行，再SHARED/MOE_UNIFORM。

## 最终预检与正式启动证据

9578326通过309项服务器Track-A测试。现有四组smoke共64成功updates/0 skipped，32个checkpoint全部恢复/同batch validation重放成功，最大logit差异0；raw token identity差异0、独立完整前向identity最大6.26e-7、旧logit漂移最大9.24e-7、mean cosine distance绝对值最大1.12e-7，均低于声明阈值。初始化/采样配对、74544参数/task、旧参数/专家/路由表保护、当前专家学习和冻结CLIP检查通过。未加载test、未报告smoke精度。

初始failed状态备份为smoke batch的pipeline_initial_failure.json，最终pipeline_state.json=smoke_passed，并在smoke_acceptance.json记录复用检查点补评价；preflight_val/comparison.json记录所有32次只读重放。非checkpoint产物147文件已同步本地主工作树，归档logs/emotic_static_moe/verification/static_moe_v1_preflight_20261012.tar.gz的两端SHA256均为d51e9e0e3647aece520dace14c577adfcab92a7975c1132e655d5ed430774bc8；checkpoint/anchor留服务器。

正式tmux为ml_static_moe_v1_20261012，pipeline_state=training、smoke=false，GPU0最多2组并行。17:42启动确认时BASELINE/MOE_OVERLAP均完成task0第1轮，各84updates/0 skipped，loss=.67747971/.67737051，分别36.5/45.5秒。GPU0占用约8146MiB，未见OOM/Traceback。完整配置均为8task/30epochs/batch64，预期13950updates/组，两组原始Selector/Prompt/head初始化hash相同。startup_receipt.json保存完整启动配置及首轮日志。

正式运行固定clean HEAD957832632a372936b5cb752f788a617f77a42443；后续仅文档提交不pull进运行中的服务器worktree。四组按固定来源继续完成；自然结束后同步logs/output，再核对完成状态、source/hash、完整val2397/test5368人群、总更新数/0 skipped和各项审计。当前未取得正式mAP结果。

实现文件：static_moe.py提供任务专属完整输出专家池，model.py/runner.py接入静态视图路由及采样/预算元数据；static_moe_audit.py检查identity和旧任务保护；evaluate_static_moe.py实现四组固定checkpoint只读评价及val-before-test条件；static_moe_pipeline.py与scripts/emotic/run_static_moe_v1.sh提供锁定配置和两组调度；tests/test_track_a_static_moe.py提供新增回归。以上Python模块均在multi_lane/track_a/下。

