# Selector key/value 分离：固定 checkpoint validation 诊断

用户授权开始执行“先定位，证据支持后才训练value-only”。本轮先实现/运行诊断，不从test选择前向或ρ；不提前训练下一版。

## 配置

- 分支 `exp/static-moe-key-value-diagnostic-20261013`；独立远程worktree `../multi-lane-main-static-moe-kv-20261013`，Git-only同步，旧运行/primary/test-only均不改。
- EMOTIC完整validation2397人，seed0，全部8任务，每任务对应固定第30轮compact checkpoint。v1读取 `../multi-lane-main-three-view-static-moe-v1/output/emotic_static_moe/static_moe_v1_seed0_eight_task_val_test_20261012_01` 的SHARED/UNIFORM/OVERLAP；v2读取 `../multi-lane-main-static-moe-bounded-v2-20261013/output/emotic_static_moe_v2/static_moe_v2_rho01_seed0_eight_task_val_test_20261013_01` 的SHARED/OVERLAP。
- 五来源×八阶段×四前向=160次正式任务级validation前向。每来源均先跑完整八任务、64人的同四前向smoke；五来源smoke全部成功才进入正式评价。
- 四前向：both原始key+value；value_only固定未适配key、只适配value；key_only只适配key、固定未适配value；none两者均用未适配token。干预在原Image-token Adapter插入层1、三个视图统一施加，不改残差scale、v2ρ、专家路由、Selector/Prompt/head、固定融合。
- none仍保留联合训练出的其他参数，不等于无Adapter原基线。key/value是Selector证据匹配/聚合所用token，不是修改CLIP主干attention的K/V投影。
- batch/eval64、workers2、OMP1，AMP/TF32沿用源配置；没有optimizer、LR或epochs更新，model.eval/no_grad/零参数更新。冻结CLIP ViT-B/16来自 `../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；数据 `../multi-lane-main/datasets/EMOTIC`；Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。只构造val数据集，不打开test数据或test score。
- 资源核查时仅GPU0空闲，其余忙；GPU0最多两进程，五来源排队。启动至少16384MiB空余，不停止其他会话进程。
- 正式batch `static_moe_kv_validation_seed0_20261013_01`；日志 `logs/emotic_static_moe_kv/<batch>/`，结果 `output/emotic_static_moe_kv/<batch>/`。smoke/full_val分目录，已有输出拒绝覆盖。

## 诊断和审计

- 每阶段全量已见类的融合/Full/Person/Face/可靠Face AP、当前新类块及任务曲线；原始both与此前full_val保存的同人群logits逐项重放，允许原协议0.01数值容差，并记录实际误差。
- 当前任务lane、layer1、每人物/视图的KL(current||frozen)、KL(frozen||current)、top10 token重合度、熵/最大权重。参考注意力使用相同当前query与原冻结token；不是跨模型基线。保留原始逐样本NPZ及均值/p50/p95/max。
- 完整候选残差d=adapted−frozen（含scale、bias及v2边界），按token均值分解d=centered+constant；记录每人物sum(d²)/sum(h²)及中心化/常量分量。三者使用相同分母，可检查能量分解。记录的是候选残差，即使该模式把它断开也不将残差统计改为0。
- attention/top10参考计算在同一个前向中，不改变随机数或参数。value_only/none的key保持原样，其本层KL必须为0；key_only与both在layer1应具有相同注意力。
- 来源config/checkpoint/训练审计/validation参考SHA，Face val manifest及CLIP hash；推理前后compact模型状态/冻结CLIP/来源文件不变，无梯度。模式是运行时属性，不写入checkpoint。

## 预声明推进门槛

在v1与v2两个OVERLAP来源都要求：最终value_only−both >=0.1点、value_only−none >=0.1点、key_only−none <=0；阶段平均前三个差值分别>=0、>=0、<=0。以两个来源一致性避免从五组/某任务中事后挑有利一行。这是保守工程门槛，不是显著性检验，也不是value-only重训必定提高的证明。

通过后，先分析逐任务/视图与attention诊断，再实现单变量value-only候选，保留专家数量、原路由、损失、融合和完整八任务预算，仍有B0/普通Adapter/重叠MoE对照。具体训练配置需在训练启动前另行完整声明；未通过则停在诊断阶段。流水线写decision.json并明确training_started=false，避免仅凭推理开关分数自动训练或部署。

## 入口

```bash
/opt/conda/envs/ddp/bin/python -u -m multi_lane.track_a.static_moe_kv_pipeline \
  --batch-id static_moe_kv_validation_seed0_20261013_01 --gpus 0 \
  --v1-batch ../multi-lane-main-three-view-static-moe-v1/output/emotic_static_moe/static_moe_v1_seed0_eight_task_val_test_20261012_01 \
  --v2-batch ../multi-lane-main-static-moe-bounded-v2-20261013/output/emotic_static_moe_v2/static_moe_v2_rho01_seed0_eight_task_val_test_20261013_01 \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
```
