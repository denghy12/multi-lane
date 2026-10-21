# 人物条件跨视图摘要注意力：seed0完整8任务执行方案

用户于2026-10-21明确要求开始实现并运行8task validation和test，覆盖此前仅3任务先导的建议。不是先导自动晋级，所有设置先于本轮test固定。

## 方法和对照

保留三视图Frozen Forward及三条Task Forward，在第6层（索引5）的Summarize之后、原任务MSA之前只增加一次摘要注意力。人物摘要提供Q，Full摘要提供K/V，输出以0.1残差写回Person摘要。Full和Face路径、Drop & Replace、FFN、固定末端融合及共享head沿用原结构。此处借鉴MulT的定向crossmodal attention思想，并非原论文完整模型的复制。

| 组别 | Q | K/V | 回写位置 | 每任务新增参数 |
| --- | --- | --- | --- | --- |
| 监督匹配三路基线 | 无 | 无 | 无 | 0 |
| person_self：额外容量对照 | Person摘要 | Person摘要 | Person摘要 | 196608 |
| task_full：任务查询对照 | 原任务Selector | Full摘要 | Person摘要 | 196608 |
| person_full：人物内容条件交互 | Person摘要 | Full摘要 | Person摘要 | 196608 |

三组注意力均4头、总内部宽64、无bias/新FFN/dropout/新门控，输入parameter-free LayerNorm，输出矩阵零初始化。Q/K/V按任务独立隔离RNG初始化，八任务新增1572864参数。每任务结束完整冻结该任务模块，后续不复用。Full摘要不detach，当前Full Selector可通过交互分支接收梯度；不能宣称Full训练结果不变。两个Full来源组的Person端点是Full-conditioned Person，不是独立Person。

Full摘要缓存只在单次三视图前向内有效，按Full→Person→Face顺序运行。原图三次编码仍保留；不是单Task Forward或骨干前融合。Face仅参与原可靠性屏蔽末端融合，本版不向交互模块引入Face。

## 固定训练和评估设置

- EMOTIC，seed0，8任务类别数5/3/3/3/3/3/3/3，全TRAIN，calibration_fraction=0。
- 每任务30 epoch，batch64，主LR=0.05×64/256=0.0125；新增注意力LR=0.0004。
- Adam每任务重置，weight_decay=0，cosine降到0，无warmup，temperature=1，threshold=.5。
- AMP初始scale1024，growth_interval=1000000000，TF32开启，gradient_clip_norm=0。
- legacy_full_zero、joint_bce、supervised_loss_scale=1、aux=0；MoE/ParaX/image-token adapter/Selector conditioner关闭。
- 10 Selectors，原Prompt数量及层数沿用runner（保存在config），共享Selector/Prompt/head，pre-head归一化。
- Full224及CLIP归一化，legacy RandomResizedCrop(.05,1)；Person letterbox、margin.15、无颜色扰动；Face沿用现有manifest。
- 可靠Face末端权重Full/Person/Face=.64/.16/.20，不可靠时=.80/.20/0。
- 训练workers2，评估workers0；避免4GiB共享内存预取故障。
- 任务训练人数5353/4394/861/9931/4352/2536/627/1526，预期13950成功更新、240epoch、0跳步。
- 每任务先保存含optimizer/scheduler/scaler/RNG的恢复状态和compact checkpoint，再运行边界报告。每组训练完成并核验后，同GPU并行完整validation2397人、test5368人，对全部8个checkpoint计算最终mAP和阶段均值。

## 机器、路径和入口

GPU5/6本次已被外部任务占用，不停止外部作业；采用空闲GPU0，三组新增模块并行训练，须先通过同配置并行短跑显存验证。

输入相对远程新worktree：

- 数据：`../multi-lane-main/datasets/EMOTIC`
- 预训练：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`

产物：`output/person_conditioned_cross_view/<run>/<arm>/{train,full_val,full_test}`；日志：`logs/person_conditioned_cross_view/<run>/`。

入口：`bash scripts/emotic/run_person_conditioned_cross_view_batch.sh`。短跑设置独立`RUN_ID`和`SMOKE_UPDATES=2`，每组2任务、每任务2更新，只访问validation。正式默认run为`formal_seed0_20261021_01`，正式完整8任务。

匹配基线优先只读复用20261020 fullTRAIN aux0 baseline_noaux，来源f057e712。启动前审计配置、初始化、采样记录、13950更新、完整全人群评估与源checkpoint SHA；新组逐任务匹配初始化和样本顺序。新默认关闭路径还需重放既有task0/2/7验证预测。若不能匹配则补独立基线，不能静默混用。

## 必要验证与解释限制

测试涵盖零初始化输出一致、RNG隔离、优化器成员、两步Q/K/V/O梯度、Full证据梯度、输入因果对照、Face屏蔽、三路/单层位置、缓存异常清理、旧任务参数和预测保护、compact重载、混合精度。真实短跑检查训练跳步、恢复状态、checkpoint重放、并行显存及query/attention随图片变化。

模块额外保存摘要/残差范数及其比例、夹角、注意力熵、跨图片Q/注意力变化量。注意力不同不等于性能提高；最终要比较人物条件组与同容量两组对照及匹配基线的validation最终/阶段平均，test仅按用户要求固定报告，不回头据此选参数。

Full摘要已经压缩，可能缺少局部线索；没有框坐标条件，不保证空间上锁定目标人物；Full本身含人物，不是纯背景。共享监督仍可能让Person增强路径学习重复信息。上述均为待检验假设，不能预先认定是历史失败原因。
