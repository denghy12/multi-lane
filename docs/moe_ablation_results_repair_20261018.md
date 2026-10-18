# 六臂消融结果核验与只读validation补评估

2026-10-18用户要求仔细分析已结束的实验。原运行3410546的batch状态实际failed：
L1_K2_F2、L1_K3_F3完成8任务训练与checkpoint审计，但validation因DataLoader Bus error
中断；L5_K2_F2完成task7第30个epoch及冻结后，在训练内validation崩溃，未写task7
checkpoint、最终训练history/summary与完整配对审计。另三臂全部完成，五臂test完整。
日志均指向共享内存耗尽，无GPU OOM证据。

原并行启动器漏算了共享内存预算：三视图batch64 float32图像约110.25MiB，workers2、
默认prefetch_factor2约441MiB/loader；正式12个评估进程峰值约5.17GiB，超过容器
/dev/shm=4GiB（未计其他对象）。smoke每臂两类validation串行，仅6个评估进程，
未覆盖正式val/test双进程峰值。该执行故障不能解释成模型精度失败。

## 本次只读补评估配置（不训练）

- 目标：补L1_K2_F2、L1_K3_F3完整8阶段validation。
- 方法/权重：原3410546对应各8个compact checkpoint，完整CLIP预训练权重不变。
- EMOTIC val2397完整人群、8task、seed0、batch64、AMP/TF32沿用原评估。
- 唯一运行变化：evaluation workers=0，消除DataLoader共享内存队列；来源训练config
  workers=2保持不动，结果额外记录evaluation_workers与evaluation_git。
- GPU0、GPU1各1个进程，启动前各≥8000MiB空闲，不停止任何已有作业。
- 无optimizer/LR/epoch或backward，不重复训练，不读取test调整方案。
- 输入：相对新repair worktree的
  ../multi-lane-main-moe-layer-routing-ablation-20261018/output/moe_layer_routing_ablation/formal_seed0_20261018_01。
- data/CLIP/Face输入均沿用原协议相对路径。
- 入口：`bash scripts/emotic/repair_moe_ablation_validation.sh`。
- run_id=val_workers0_20261018_01；logs/moe_ablation_repair/<run_id>/；
  output/moe_ablation_repair/<run_id>/<arm>/full_val/。
- 预期：2个result.json、16份fused scores与16份view scores，源checkpoint SHA前后不变；
  对原失败目录已完成阶段逐值核对logits/概率/targets/IDs。

新分支fix/moe-ablation-eval-workers-20261018，独立工作树；原运行源码、checkpoint与
失败目录全部保留。新加--workers仅影响评估DataLoader，不改训练模型或完整性门槛。
Middle缺task7无法伪造恢复，不用7任务结果充当完整消融。先分析5个完整模型与B0，
明确层位置比较仅Early/Late有效；完整Middle补跑需单独安排，当前不启动新训练。

当前：本地修复，待测试与Git同步后执行只读补评估。分析包含最终/阶段平均mAP、
分支、Face可靠性子集、旧类漂移/遗忘、冻结覆盖和推理权重；仅seed0不当作跨seed结论。
