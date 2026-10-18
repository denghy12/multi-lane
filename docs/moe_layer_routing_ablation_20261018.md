# MoE插入层与Router／冻结数量消融（2026-10-18）

## 授权与目的

用户明确要求修改并在GPU0、GPU1并行完成两组8task、seed0、validation和test实验。
此前只读回滚未证明稳定validation收益，原停止建议仍保留为历史证据；本轮是用户
新授权的预设消融，不把修改机制当作精度收益，不自动扩seed、专家池或合并main。

分别检验：更深的Task Forward插入位置是否改变图像相关路由和泛化；在同层独立改变
Router Top-k与每任务冻结数量，区分激活数、冻结覆盖与组合效应。采用上一版WAF
加权激活统计，不同时切回硬计数，不引入v2–v4证据输入、额外共享专家或新损失。

## 六个唯一配置与并行安排

| arm | GPU | 层索引（从0开始） | 实际层 | Router Top-k | 冻结Top-f |
|---|---:|---:|---:|---:|---:|
| L1_K2_F2 | 0 | 1 | 2 | 2 | 2 |
| L5_K2_F2 | 0 | 5 | 6 | 2 | 2 |
| L11_K2_F2 | 0 | 11 | 12 | 2 | 2 |
| L1_K2_F3 | 1 | 1 | 2 | 2 | 3 |
| L1_K3_F2 | 1 | 1 | 2 | 3 | 2 |
| L1_K3_F3 | 1 | 1 | 2 | 3 | 3 |

层位置组前三项；路由／冻结组为第一项与后三项。L1_K2_F2在同一新提交重新训练
一次，两组引用同一份checkpoint；不把历史WAF结果直接充当新代码对照。各组内3臂
并行、两GPU同时使用，启动前检查GPU0/1空闲且各≥22000MiB，无进程会被停止。
其他GPU不使用。共6次训练、48个任务checkpoint、6组val及6组test完整阶段曲线。
每臂先顺序训练8任务、独立审计checkpoint，再在该GPU上并行val/test两个进程；
训练任务本身不并行。任何失败保留目录，等待所有已启动子进程后记录失败。

## 模型与冻结口径

- CLIP ViT-B/16视觉主干冻结，Frozen Forward不变；独立于ParaX。
- 仅一层Task Forward，在MSA与Drop & Replace后，与该层FFN并联。
  不同时插入[1,5,11]。Router保持该块raw task CLS输入，每任务×视图独立gate/noise。
- 该层22名跨任务、跨视图共享Expert，bottleneck64、dropout0.1、零up初始化，
  residual scale0.1；训练noisy Top-k、推理无噪声，选中权重softmax归一化。
  Router k改变活跃计算量，冻结f改变后续训练容量，六臂分配参数量相同，预计2,992,000。
- 全训练期实际门控权重detach后float64累计，Full+Person+可靠Face合并排名，
  按expert ID升序打破同分；从全部22名选Top-f加入累计冻结集合。
  可以再次选到已冻结Expert，不补足新的f名；不是三视图各冻f名。
- 冻结完整Expert down/up weight+bias与该任务Router；新任务仍可访问全部Expert，
  训练未冻结Expert及新Router。冻结权重保留输入／Router梯度。
  此规则仍不能保证历史函数不变，因为旧Router可访问可变共享Expert。
- 所有训练硬激活计数仍记录，新增非持久化推理expert_gate_mass及排序位置平均权重，
  用于判断Top-3第三名是否获得实质权重；新增统计不参与梯度、不改变随机数。
  推理路由统计沿用现有接口，Face包括占位无效Face；训练冻结统计仅可靠Face。
- 默认f2保留原freeze_top2 checkpoint键及形状，f3使用freeze_topk[...,3]；
  旧checkpoint未记录freeze_count时恢复为2。新训练配置显式记录k/f和冻结策略。

## 固定训练与评估配置

- EMOTIC，8tasks类别数5/3/3/3/3/3/3/3，seed0，各30epochs，train/eval batch64，workers2。
- Adam，每任务重置；源LR0.05按64/256缩放为0.0125，MoE LR0.0004，weight decay0；
  cosine调度，warmup0、min ratio0，无额外梯度裁剪。预期13950次成功更新、0跳步。
- AMP initial scale1024、growth interval1000000000，TF32开启；沿用原WAF训练数值设置，
  不引入只读归因专用的FP32确定性运行设置。
- legacy_full_zero BCE、joint_bce、辅助视图loss0.1、joint视图梯度，无新增负载损失。
- shared Selector/Prompt/classifier，10 selectors、10 prompts、5 prompt层、pre-head归一化。
- Full legacy随机裁剪scale[0.05,1]、Person letterbox margin0.15、无Person色彩扰动，
  CLIP输入归一化；Face使用现有manifest可靠性；固定三视图融合沿用原配置。
- 训练不逐阶段读取test；保存8个compact checkpoint后，统一完整人群val2397与test5368。
  记录最终及阶段均值mAP、Full/Person/可靠Face分支、旧类遗忘、冻结集合／剩余容量，
  结合训练与推理路由权重解释结果，不以路由多样性代替精度收益。
  六配置预先固定，test仅报告，不用于继续调整配置或选超参数。

## 输入、输出与入口

服务器新worktree：`/mnt/haoyuan/workspace/multi-lane-main-moe-layer-routing-ablation-20261018`。
新分支：`codex/moe-layer-routing-ablation-20261018`，从41567eb创建；受控源码仅Git同步。
本地主树、旧实验工作树与服务器test-only均保持原状。

相对新worktree输入：

- data：`../multi-lane-main/datasets/EMOTIC`
- CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`
- B0配对审计：`../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_baseline_seed0_20261017_01/train`

```bash
# 全部Track-A测试通过后，先执行全部六臂两任务各2更新smoke（不读取test）。
SMOKE_UPDATES=2 RUN_ID=smoke_seed0_20261018_01 \
  bash scripts/emotic/run_moe_layer_routing_ablation.sh
# smoke机制、来源、重放均通过后执行正式六臂；入口锁定seed0、GPU0/1。
RUN_ID=formal_seed0_20261018_01 \
  bash scripts/emotic/run_moe_layer_routing_ablation.sh
```

Python默认`/opt/conda/envs/ddp/bin/python`。
日志`logs/moe_layer_routing_ablation/<run_id>/<arm>/`（train/verification/full_val/full_test）。
结果`output/moe_layer_routing_ablation/<run_id>/<arm>/`（train/full_val/full_test）。
批次根保存batch_manifest.json，记录两组映射、共同对照、Git提交及各子进程退出状态。
日志根arm_status.tsv与exit_status.txt记录完整批次状态。

## 验证门槛与当前状态

新增机制测试覆盖k/f独立、三位置接入、f3严格checkpoint恢复、重选不补足、全池冻结
时Router／输入梯度、只读权重日志与篡改审计。启动器测试覆盖六臂不重复、GPU分配、
先训练再审计再双split评估、失败后等待全部子进程。旧Track-A测试必须全部通过。
真实六臂smoke检查B0初始化／采样配对、冻结排名重算、权重守恒、已冻结Expert和旧
Router不变、CLIP不变、0跳步、checkpoint验证集重放一致。共同对照另与旧WAF同预算
smoke逐参数对比，确认默认k2/f2训练计算兼容。

当前：本地实现及启动器三项测试通过，尚未启动服务器smoke或正式实验。
完成后将日志与结果同步本地并核对SHA256，checkpoint留服务器；不自动合并main。
