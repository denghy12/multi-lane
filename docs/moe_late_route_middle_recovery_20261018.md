# Late路由只读归因与Middle完整重跑

2026-10-18。用户明确要求执行上一轮建议，授权Late只读对照及Middle独立完整重跑。
新分支exp/moe-late-route-middle-recovery-20261018，从f17a256创建；源实验worktree及
主树/test-only未提交内容保留。仅Git同步受控源码，不覆盖任何旧输出或checkpoint。

## 两项工作与固定配置

### GPU0：Middle重跑

- EMOTIC，seed0，8task类别5/3/3/3/3/3/3/3，30epochs/task，train/eval batch64。
- 原Task Forward MSA及Drop & Replace后与FFN并联；单层索引[5]、22名共享Expert，
  bottleneck64、Top-2、dropout0.1、scale0.1、零up初始化；noisy训练、无噪声推理。
- 全程训练gate mass float64，Full+Person+可靠Face总和选Top-2累计冻结，可重选不补足。
  所有历史Router可访问冻结专家，训练未冻结专家，冻结Expert保持输入/新Router梯度。
- 冻结CLIP ViT-B/16，ParaX关闭，shared Selector/Prompt/head，10selector/10prompt、
  5 prompt层，pre-head归一化；三视图预处理与融合沿用原WAF/六臂协议。
- Adam每任务重置；源LR0.05×64/256=0.0125，MoE LR0.0004，weight decay0，cosine、
  warmup0、min ratio0。legacy_full_zero、joint BCE、view auxiliary0.1，无新损失。
- AMP initial scale1024、growth interval1e9、TF32开。训练workers2；内部/外部评估workers0。
- 每任务训练和冻结结束后、reporting验证前保存原子恢复快照，包括compact模型及冻结CLIP
  源SHA/config、optimizer/scheduler/scaler、optimizer参数名称映射、Python/NumPy/CPU/
  可见CUDA RNG、train sampler generator、已完成训练历史/报告/冻结和配对审计。
  原子fsync+rename，新文件已存在即拒绝。仅在任务边界恢复，不声称支持任意batch中途恢复。
  完整模型compact仍在正常验证后保存；pre-reporting快照可保护验证失败时刚训练完的权重。
- 预期13950成功更新/0跳步、8个compact与8个恢复快照；每个恢复模型与验证后compact
  逐tensor一致；配对B0核验后同GPU0双进程full_val/full_test，均workers0。
- 入口scripts/emotic/run_moe_middle_recovery.sh。正式run_id=formal_seed0_20261018_02，
  output/moe_middle_recovery/<id>/train、full_val、full_test；logs同名前缀。
- smoke两任务各2更新，run_id=smoke_seed0_20261018_01；核验来源、冻结、checkpoint
  重放、恢复状态、与旧Middle同预算smoke模型逐tensor兼容。正式不重复另外5臂。

### GPU1：Late只读动态/固定/关闭残差

- 源：../multi-lane-main-moe-layer-routing-ablation-20261018/output/moe_layer_routing_ablation/
  formal_seed0_20261018_01/L11_K2_F2，3410546；仅task0、task2、task7。
- 不训练、无optimizer/backward；模型eval/no noise/no dropout，AMP/TF32沿用原评估。
- 每checkpoint先取已见类别有正例的TRAIN人物（稀疏类别ID判断，排除未来类独有图片）。
  按原图分组，seed20261018+task，固定4批、每批≤64人且不拆同图人物组，实际约256人。
  在该checkpoint所有已见task lanes上统计每层/任务/视图门控mass，Face只纳入可靠样本。
- 用CPU float64 gate mass从22专家中选两名，同分ID升序；两名的mass归一化为固定权重。
  各旧lane独立校准，保存TRAIN IDs与像素SHA。所有三checkpoint表锁定后，才构造val数据。
  不用历史训练的带噪累计频率替代当前无噪声依赖，也不读取validation/test选择专家。
- A dynamic：原图片相关Top-2；B fixed：锁定上述IDs和权重；C off：同checkpoint关闭MoE。
  A/B同活跃数、原Expert和scale。C的Prompt/Selector/head曾与MoE共同训练，C不等于B0。
- 完整val2397人，batch64，workers0，分别输出fused/Full/Person/Face/可靠Face指标和scores。
  dynamic与原validation逐元素核对logits/概率/IDs；每条件compact model SHA不变、无grad；
  CLIP SHA前后相同、三个源checkpoint SHA不变。原源数据/模型不写入。
- 同原图整体重采样500次，三个条件配对比较dynamic−fixed/off的融合mAP与95%区间。
  仅这3个阶段，不能称为完整8阶段平均。test不访问，不用于筛选。
- 预锁继续研究证据：dynamic−fixed三个阶段点估计均>0且至少一个区间下界>0；否则不自动
  训练下一版。未通过只表示证据不足，不证明路由真实效应严格为0。对off单列差和CI，
  CI跨0不当作证明等效。无论结果如何，本次不自动扩seed或训练新MoE结构。
- 入口scripts/emotic/run_moe_late_route_control.sh；正式run_id=formal_seed0_20261018_01。
  日志logs/moe_late_route_control/<id>/；输出output/moe_late_route_control/<id>/。
- smoke同3个checkpoint、校准2批、val前64人、bootstrap20，核验三条件与动态逐值重放；
  smoke结果不用于选配置。run_id=smoke_seed0_20261018_01。

## 输入与执行

两项相对输入均与原六臂实验相同：
- data ../multi-lane-main/datasets/EMOTIC
- CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt
- Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
- paired B0 ../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_baseline_seed0_20261017_01/train
Python /opt/conda/envs/ddp/bin/python。服务器独立worktree计划
/mnt/haoyuan/workspace/multi-lane-main-moe-late-route-middle-recovery-20261018。

新增单元检查覆盖可靠Face校准/质量守恒、TRAIN分组无未来类独有图片、确定同分规则、
固定Top2/scale混合、异常退出恢复原Router、配对bootstrap、原子写失败/禁止覆盖、保存
不推进RNG、保存后模型/Adam/scheduler/RNG/sampler继续一步逐值恢复。
全Track-A检查、两个真实smoke通过后，两GPU并行启动正式。当前仅本地实现，未开正式。
数据/日志同步本地后核验SHA。原主树/test-only保留，不合并main。
