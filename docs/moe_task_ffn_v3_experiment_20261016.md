# Task-Forward MoE v3：四专家与 soft Top-2

## 用户授权与范围

2026-10-16 用户要求按v3计划修改，并明确要求8-task validation/test同时
并行评估。因此本次直接运行三组8任务seed0，覆盖此前“先task0–2筛选再
决定8任务/test”的建议。三组训练依次使用GPU0；每组完成训练后，用两个
进程在GPU0并行重放同一套8个checkpoint，分别评估固定全人群val/test。
参数在test运行前锁定，不按test结果选择模型、系数或checkpoint。

## 模型

- 保持视觉Transformer索引1的Task Forward MSA、Drop & Replace后
  FFN并联位置；Frozen Forward图像流不变，Router读取本块Selector
  `selected.mean(dim=2)`图片证据，LayerNorm后单位向量归一化。
- 每任务1共享+3专门专家，bottleneck64；每任务403456参数，总计3227648。
  三视图共享专家池，但各有独立Router；整池随任务结束冻结，不跨任务复用。
- dynamic：余弦logits scale2、temperature1，在3专门专家中取Top-2，
  对被选logits做softmax，两权重和为1，普通可微梯度；无直通估计/路由噪声。
- shared_fixed（shared-only）：`0.5*shared`。
- view_fixed：`0.5*shared+0.5*specialist`，Full配对专门1/2、Person2/3、
  Face3/1，专门权重各0.5。
- dynamic同样为`0.5*shared+0.5*specialist`；两种专门控制激活相同数量
  专家，三组相同共享系数与residual scale0.1，专家dropout0.1。
- 动态组轻量负载loss权重0.001，对实际稀疏soft分配平均值施加均匀约束。
  所有模式保留相同模型形状，未激活的参数不产生梯度。

## 日志与机制评价

每batch记录三视图shared/specialist/combined输出平均范数、加权专门/共享
范数比、两输出余弦、CLS单独范数和比值、soft权重熵、每专家硬使用率与
soft权重。epoch日志按batch平均；正式评估按样本累计，路由统计不进入
checkpoint。范数在最终residual scale之前计算，shared/spec权重同为0.5，
故加权后比值与未加权比值相同。

正式val/test并行完成后，dynamic/view_fixed自动增加全8任务validation
`--disable-specialist-runtime`消融，只关闭专门分量，保留0.5共享。关键
判断为专门/共享比值是否达到0.25、关闭specialist是否降低val、dynamic
是否超过匹配控制以及Person是否退化。输出比值是机制检查，不替代mAP。

## 锁定数据与训练配置

- EMOTIC train，8任务大小5/3/3/3/3/3/3/3；seed0，各30epochs，batch64。
- 冻结OpenAI CLIP ViT-B/16；Full/Person/Face，CLIP归一化，Full legacy
  crop scale0.05–1，Person letterbox/margin0.15，无color jitter；既有Face
  manifest与可靠性mask，固定三视图融合，shared Selector/Prompt/head。
- Adam，每任务重建优化器；主source LR0.05按64/256缩放为0.0125，
  MoE LR0.0004，weight decay0；cosine，warmup0，minLR ratio0。
- BCE `legacy_full_zero`、`joint_bce`，监督scale1，视图辅助loss0.1；
  AMP初始scale1024、growth interval1e9，TF32；workers2，OMP threads1。
- 训练期间不加载test；每任务末保存compact checkpoint。评估batch64，
  AMP/TF32，阈值0.5，每阶段固定val2397人/test5368人，仅评价已见类。

输入路径相对服务器新v3 worktree：
`../multi-lane-main/datasets/EMOTIC`、
`../CODE_DDP/pretrained/clip/ViT-B-16.pt`、
`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
输出`./output/moe_task_ffn_v3/<run_id>/`，日志`./logs/moe_task_ffn_v3/<run_id>/`。

## 执行入口

先远程Track-A单测，再各模式task0两次成功更新smoke（`SMOKE_UPDATES=2`），
检查零输出identity、两专家/Router梯度、旧池冻结、AMP稀疏累加、
403456参数预算、RNG配对、checkpoint同人群重放；smoke只访问val。

```
ROUTE_MODE=dynamic SMOKE_UPDATES=2 RUN_ID=smoke_dynamic_seed0_20261016_01 bash scripts/emotic/run_moe_task_ffn_v3.sh
BATCH_ID=seed0_20261016_01 bash scripts/emotic/run_moe_task_ffn_v3_batch.sh
```

正式run_id为`formal_<dynamic|shared_fixed|view_fixed>_seed0_20261016_01`。
预期每组13950成功更新、240epochs，8个checkpoint，full_val/full_test
各8阶段分数、逐类/逐视图/路由与输出诊断，split退出状态；两专门组另有
full_val_specialist_off。任何训练/评估失败会使批次停止，不覆盖失败目录。
不合并main，不删除数据或checkpoint，不修改远程test-only工作树。

## 预检记录

业务提交`6a4d585`已在服务器独立v3 worktree通过296项Track-A单测。
三组`smoke_<mode>_seed0_20261016_01`均完成2次更新、0跳步，CLIP
SHA未变；初始主参数SHA同为`35af16da90b3...`，三组checkpoint同人群
重放mAP差均0。峰值reserved为2818/2798/2804MiB。
dynamic smoke的MoE梯度0.00036335、finite=1，三视图专门/共享输出比
约0.692/0.672/0.701。两次更新只用于验证路径，不视为正式效果。
新增两项无需真实数据的脚本测试已本地通过，检查val/test时间重叠、
两者结束后才做validation消融，以及test失败仍等待val并返回失败。
