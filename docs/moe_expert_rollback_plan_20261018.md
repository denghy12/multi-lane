# AF/WAF旧Expert只读回滚归因：预锁协议

用户2026-10-18授权执行。独立分支codex/moe-expert-rollback-20261018，从WAF运行
提交ceded63创建；旧分支未提交文档复制到本分支，原文件保留。方法训练代码不改。

## 目标与五个条件

AF（07fd35c）/WAF（ceded63）、seed0、task0→1与task6→7两个边界。
比较训练前、训练后、训练后仅回滚TRAIN访问的可变Expert、训练后仅回滚此前冻结
Expert（阴性控制）、训练后回滚全部此前未冻结Expert（覆盖上界）。所有回滚只在
内存中复制完整down/up weight+bias，不改Router/Selector/Prompt/head/Norm/mask。
已冻结的阴性控制权重必须与训练前逐值相同，预测也必须逐值相同。

两个端点统一只运行旧task数的lanes：boundary1为1条/5类，boundary7为7条/23类。
因此不把lane数变化引起的GEMM误差当作功能漂移。新任务类不纳入本次恢复主指标，
不能把全局回滚池的模型作为可部署新模型或公平的新任务训练结果。

## 数据与数值设置

- EMOTIC TRAIN确定性三视图变换（相同val式预处理），从旧类有正例的人物按原图
  分组抽4批，上限64人/批，组内eligible人物不拆散；选择seed20261018+boundary。
  两方法、各条件、两数值设置使用同一IDs和像素；实际样本数/像素SHA记录。
- 选专家集合仅使用这4批TRAIN，取前后端点及FP32/AMP两设置的访问并集，剔除
  无效Face访问，再与边界前未冻结集合相交。不得用validation扩充集合。
- validation固定全2397人，所有条件相同旧类targets、ID、batch64、workers2；
  不访问test。记录融合及Full/Person/Face/可靠Face旧类mAP、逐类AP与原始logits。
- 主归因FP32、TF32 off、eval（无dropout/noise），TRAIN额外AMP/TF32 on复核。
  无optimizer/backward，LR/scheduler/epochs不适用，更新数0。
- Frozen CLIP ViT-B/16、22专家、宽64、单层index1、scale0.1，模型从原config重建。
  源checkpoint哈希核对原validation读取记录；前后旧非Expert坐标必须相同。
  每次推理前后state哈希不变；patch作用域外完整恢复，源文件SHA前后不变。

## 路径与运行

- 入口`bash scripts/emotic/run_moe_expert_rollback.sh`，仅空闲GPU1（至少12GiB），
  不停止其他作业。`CUDA_VISIBLE_DEVICES=1`，OMP1，seed0。
- AF源`../multi-lane-main-moe-activate-freeze-20261018/output/moe_activate_freeze/formal_seed0_20261018_01/train`。
- WAF源`../multi-lane-main-moe-weighted-freeze-20261018/output/moe_weighted_freeze/formal_seed0_20261018_01/train`。
- 数据`../multi-lane-main/datasets/EMOTIC`；CLIP
  `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 输出`output/moe_expert_rollback/<run_id>/`，日志`logs/moe_expert_rollback/<run_id>/`。
  正式run_id=formal_seed0_20261018_01；产物run.json、4个边界result、TRAIN特征/
  logit NPZ、20个validation NPZ、SHA与汇总报告。checkpoint不复制/改写。
- 先有意义的patch/异常恢复/阴性控制/可靠Face/旧坐标测试，再全Track-A测试，
  Git-only远程独立worktree，真实smoke（每边界1 TRAIN批/val64）通过后正式执行。

## 预锁判断

首要机制条件：阴性控制逐值不变，回滚全部可变Expert应将旧函数恢复到FP32底噪；
若未恢复，先排查其他旧依赖/数值设置，不直接宣称共享Expert是唯一原因。

进入完整保护方案设计的门槛，以WAF为主、AF为机制对照：两个预定边界的validation
旧类mAP恢复均为正，至少一个边界的500次原图配对bootstrap恢复区间严格大于0；
同时两边界、两数值设置各至少3/4 TRAIN批的fused特征相对漂移降低≥90%。
这些是本次决策门槛，不是普适统计定理，两个边界也不是两个训练seed。

若只恢复特征而无可重复validation收益，不训练完整保护版，保留B0，结束当前位置
的MoE扩展。若达到门槛，仅设计并报告快照/封存访问/容量预算控制，本轮不自动训练。
首任务已低于B0的问题仍独立存在，不能以消除遗忘代替泛化收益。

## 检查记录

ca82541全Track-A 337项通过。首轮smoke在Git元数据root参数缺失处退出，
没有模型前向/更新；修复后以smoke_seed0_20261018_02重跑，原失败产物保留。
