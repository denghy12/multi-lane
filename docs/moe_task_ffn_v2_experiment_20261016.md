# Task-Forward MoE v2：三视图共享专家与图片证据路由

## 问题与方法

首版 22 专家 noisy Top-2 在推理中坍缩为每任务/视图固定的两名专家，
实际 Top-1 权重约 0.998–0.9995，8 任务固定人群 test 均值
35.1268，低于配对 B0 的 36.2781。本版保持视觉 Transformer 索引 1 的
Task Forward 路径：该块 MSA 与 Drop & Replace 完成后，专家残差与冻结
FFN 并联相加，不写回 Frozen Forward 的图像 token。

Router 输入为本块 Selector 对当前图像 patch 聚合的 `selected` 在 selector
维度求均值。门控内部先做 LayerNorm、单位向量归一化，然后与单位化门控
列计算余弦分数。不同视图有独立门控参数，专家权重在同任务的 Full、Person、
Face 间共享。每任务 7 专家：编号 0 永远执行，其余 6 名中按图片选 1 名；
输出按 0.5/0.5 混合，残差系数 0.1。新任务启用时整体冻结旧任务的专家、
Router、bias 等参数，所有旧任务推理继续使用本任务池。ParaX 关闭。

动态路由使用训练期 Gaussian 门控噪声（初始标准差 0.5，随 30 个 epoch
线性衰减至 0）、温度 1、余弦 logit scale 2、hard Top-1 与直通梯度。
轻量负载损失对每视图每块的 clean softmax 平均分配施加均匀约束，权重
0.001。`shared_fixed` 只执行共享专家；`view_fixed` 执行共享专家及按
Full/Person/Face 固定分配的 1/2/3 专家。三组均使用相同任务顺序、
数据、epoch/update 上限、主训练参数和验证口径。控制组保留同一 7 专家
模型形状，但未使用专家不产生梯度；因此比较的是有效路径，不是相同活跃
专家数量。

## 训练前门槛

`python -m multi_lane.track_a.probe_moe_task_ffn_v2` 在未训练模型上读取
固定 validation 前 256 人，分别汇总三视图中归一化图片证据与 raw task
CLS 的跨图片变化。每视图证据相邻样本平均距离必须大于 0.01，变化坐标
至少 8 个。门槛失败则不启动训练。探针无参数更新，也不读取 test。

## 锁定配置与执行

- 数据：EMOTIC；训练 split 为 train，选择 split 为固定全人群 val；
  Face 使用已有 manifest。测试仅在 validation 选定方案后执行。
- 预训练：OpenAI CLIP ViT-B/16，视觉塔冻结；Full/Person/Face 输入，
  CLIP 归一化，Full legacy 裁剪，Person letterbox，Face 可靠性 mask。
- 顺序：8 tasks，seed 0，各 30 epochs，batch 64，评估 batch 64，
  训练样本/首批 ID/初始化哈希与既有配对审计核对。
- 优化：原始 MULTI-LANE 参数 source LR 0.05，按 64/256 缩放为
  0.0125；MoE LR 0.0004；weight decay 0；Adam 与 cosine schedule，
  warmup 0、min LR 比 0；AMP、TF32。训练 loss 为
  `legacy_full_zero` + `joint_bce`，三视图辅助权重 0.1。
- MoE：单层索引 1，bottleneck 64，7 专家，专家 dropout 0.1，
  残差 scale 0.1，shared fraction 0.5，动态门控参数如上。
- 运行入口：`scripts/emotic/run_moe_task_ffn_v2.sh`。`ROUTE_MODE` 为
  `dynamic`、`shared_fixed` 或 `view_fixed`；`SMOKE_UPDATES=2`
  可限制到 task0 两次更新作 smoke。正式三组顺序运行，均用 GPU0。
- 输入路径（相对远程 v2 worktree）：`../multi-lane-main/datasets/EMOTIC`、
  `../CODE_DDP/pretrained/clip/ViT-B-16.pt`、
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 输出：`./output/moe_task_ffn_v2/<run_id>/`；日志：
  `./logs/moe_task_ffn_v2/<run_id>/`；每任务 compact checkpoint、
  train config/history、固定全人群 val 分数、逐视图分数和路由统计。

## 判定

先比较三组固定 2397 人群 validation 的最终和 8 阶段均值 mAP、
Full/Person/Face 与可靠 Face、遗忘、MoE-on/off 同权重残差收益、
每任务/视图专家使用率。动态路由不同本身不算成功。只有验证集显示
稳定且有实际收益，才冻结选择并单次报告 test。暂不做跨任务旧专家
复用，不合并 main。
