# Task-Forward MoE v4：图片证据专家与CLS残差

## 问题与修改假设

v3解决了专门输出偏弱，但seed0最终val/test为42.2628/31.6540，
未超过B0与共享控制。至少99.9821%的专门残差能量位于恢复后的Selector，
它们的专家输入跨图片不变。归一化图片证据只进入Router，未进入专家。
本次假设：让Adapter实际变换图片证据、直接更新任务CLS，更有机会产生
图像相关的有效适配。它是待验证假设，不承诺改善。

用户2026-10-17明确要求执行修改；后续最新指令锁定**仅GPU0、seed0、
完整8任务，validation/test并行**，覆盖初提task0–2/两seed/全部空闲GPU计划。新分支
`codex/moe-task-ffn-v4-20261017`从`44c40fe`创建；既有本地/远程用户改动
及test-only工作树保持原样，受控代码只经Git同步。v3完整诊断作为文档
归档到本分支。

## 结构与四组控制

仍在视觉Transformer索引1、Task Forward MSA/Drop & Replace后，
与FFN并联。Frozen Forward不变，旧任务整池冻结，不跨任务复用、
不引入ParaX或DDAS。每任务池仍为4个768→64→768 Adapter，
共403456参数/任务（含三视图Router），模型预分配8任务3227648参数。
shared_fixed控制保持同形状，但只有共享专家参与，不激活的参数无梯度。

```
e = layer_norm(mean_selectors(selected_image_evidence))
g = softmax(top2(view_router(e)))
delta_cls = 0.1 * (0.5 * shared(e) + 0.5 * sum_i g_i * specialist_i(e))
lane = lane + FFN(LN2(lane)) + concat(delta_cls, zeros_like(selectors))
```

专家和Router读取同一图片证据；专家输入为参数无关LN，Router进一步
单位归一化、余弦logits scale2、temperature1。LN保留对当前Selector
的梯度；CLIP图像流detach。MoE不直接写Selector；任务训练仍正常更新
当前Selector。取消v3全token范数比≥0.25晋级门槛；记录CLS输出范数、
残差/输入CLS比、夹角、图片证据batch内变化、路由软权重/硬负载。

- baseline/B0：adapter disabled，重新训练，匹配seed与采样协议。
- shared_fixed：普通共享Adapter，`0.5*shared(e)`，同位置/相同接口。
- view_fixed：`0.5*shared+0.5*fixed Top-2`；Full专家1/2、Person2/3、Face3/1；专门权重各0.5。
- dynamic：`0.5*shared+0.5*dynamic Top-2`；普通可微selected softmax，无噪声或ST。

固定与动态活跃专家数量和融合系数相同；共享控制判断新增图片Adapter
是否有价值，固定控制判断动态Router是否有独立收益。所有专家dropout0.1，
零输出初始化，residual scale0.1。动态负载loss0.001沿用v3，不搜索新LR/scale。

## 锁定正式配置（8任务validation/test）

- 数据EMOTIC；8任务定义5/3/3/3/3/3/3/3，完整训练task0–7。
- 四组×seed0，共4个GPU0并行作业，每组30epochs/task，batch64；每作业
  预期13950成功更新，训练期间不加载test。
- 冻结OpenAI CLIP ViT-B/16，Full/Person/Face固定融合，shared Selector/
  Prompt/head；Selector10、Prompt10、Prompt layers5、pre-head normalization。
- Adam；source LR0.05按batch64/256缩放为0.0125，Adapter LR0.0004，
  weight decay0，cosine、warmup0、minLR ratio0；每任务重建optimizer。
- BCE legacy_full_zero、joint_bce，监督scale1，视图辅助loss0.1。
- AMP初始scale1024、growth interval1e9，TF32，workers2/作业，OMP threads1。
- Full legacy crop0.05–1、CLIP归一化；Person letterbox/margin0.15，无color
  jitter；沿用train/val Face manifest与可靠性mask。
- 训练采样generator固定seed+1009*task；同seed四组初始主参数、训练样本
  SHA、首批ID、CLIP SHA核对。新增专家初始化使用fork_rng，不改变主参数/RNG。
- 每组训练完成后两个进程并行遍历同一套8个checkpoint，评估固定2397人
  validation与5368人test，仅已见类，
  batch64/AMP/TF32/threshold0.5；训练期筛选人群分数不作正式晋级指标。
- dynamic/view_fixed追加specialist-off validation，三Adapter组追加whole-MoE-off。
  关闭分支消融保留联合训练权重，不等同于重新训练B0/shared-only。

相对远程独立worktree的输入：

- `../multi-lane-main/datasets/EMOTIC`
- `../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`

输出`./output/moe_task_ffn_v4/<run_id>/`，日志`./logs/moe_task_ffn_v4/<run_id>/`。
仅GPU0：启动前检查无compute进程、free≥20000MiB、util≤5%，然后四组
并行训练。每组结束后val/test两个进程也在GPU0并行；峰值并发最多8个
评估进程，预计显存低于24GiB。GPU1–7不使用；任何已有作业不被中断。

## 执行与产物

1. 远程Track-A单测，覆盖新接口、零输出数值恒等性、专家/Router/图片证据梯度、
   Selector零写入、AMP、旧池冻结和RNG配对。
2. 真实val前256人探针：捕获实际共享专家down输入，验证随图片变化，
   初始残差严格为零，开关MoE的FP32 logits最大差≤1e-6（关闭TF32审计）。
   真实GPU重复前向本身有数值波动：TF32约1e-4，禁用TF32约3e-7；
   因此不把GPU逐值相同作为恒等门槛。CPU单测仍要求严格逐值相同。
3. 四组seed0各task0两成功更新smoke，完整validation与训练人群checkpoint
   重放差<1e-7；同seed配对审计、CLIP不变、梯度有限/非零、0跳步。
4. 检查通过后GPU0并行启动四组8任务正式训练；每作业8个compact
   checkpoint、full_val/full_test逐类/逐视图/逐样本分数、路由/CLS诊断
   和validation消融产物。test仅作锁定报告，不用于调参。

入口：

```
PHASE=smoke BATCH_ID=seed0_20261017_01 bash scripts/emotic/run_moe_task_ffn_v4_batch.sh
PHASE=formal BATCH_ID=seed0_20261017_01 bash scripts/emotic/run_moe_task_ffn_v4_batch.sh
```

单组入口`GPU=0 ROUTE_MODE=dynamic SEED=0 bash scripts/emotic/run_moe_task_ffn_v4.sh`。
正式run ID `formal_<baseline|shared_fixed|view_fixed|dynamic>_seed0_20261017_01`。

## 预声明判断与范围

先核对完整性/配对/同人群，比较seed0各阶段及阶段平均fused mAP、Person
和来源/可靠Face子群，并评估specialist-off差值。动态候选应在最终及阶段平均
超过B0、普通共享、固定路由，Person不出现持续退化，专门分量关闭应造成
validation下降；单seed不能证明训练稳定性。所有参数在test前锁定，不以
test搜索系数、关闭分支或选择checkpoint。
如果共享控制已解释收益，优先普通Adapter；如果新接口仍无收益，停止
当前Task-Forward MoE方向，保留B0。本轮按用户指令运行8任务/test，不自动扩seed、不合并main。
运行结束后同步日志/分数回本地并核对SHA；checkpoint保留服务器。

### 预检进展（2026-10-17）

锁定实现`2ec1b58`已在远程独立worktree通过305项Track-A单测。
真实val256人实际专家输入探针通过：Full/Person/Face归一化相邻距离
0.37475/0.58036/0.56503，768维均变化；初始残差严格零，关闭TF32
开关MoE的最大logit差2.7474e-7（门槛1e-6）。CPU严格identity通过。
GPU0四组seed0两更新smoke已并行启动，tmux
`ml_moe_ffn_v4_smoke_20261017`；四组初始主参数SHA同为35af16da90b3...。
GPU1–7未启动本次作业。正式8任务尚待smoke、同人群重放与配对审计通过。

四组GPU0 smoke已全部退出0，各2成功更新/0跳步，训练人群checkpoint重放
mAP差均0。三个Adapter组梯度非零/有限，dynamic Person CLS残差/输入比
约0.00103；这只验证路径。配对审计脚本初次把tasks字典误当列表，已按
现有schema修正，待真实产物审计通过后启动正式批次；模型/训练代码未变。

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
