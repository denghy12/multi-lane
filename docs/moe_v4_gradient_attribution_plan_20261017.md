# v4现有checkpoint只读梯度归因

## 授权与边界

用户2026-10-17要求执行既有checkpoint梯度诊断，不训练v5。仅GPU0；
新分支`codex/moe-v4-gradient-audit-20261017`从v4文档HEAD`bc5f572`建立，
保留v4原工作树的未提交分析，并将其上下文/报告及epoch字段审计修复复制
到本诊断分支。训练/模型结构不变，12个源checkpoint与CLIP仅被读取。
受控代码经Git提交/push/fetch，在服务器新worktree运行，源v4/main/test-only
工作树不修改。

## 数据、模型和输入

- EMOTIC train only。任务task0/2/7，四组baseline/shared_fixed/view_fixed/dynamic。
- 源运行目录相对新服务器worktree：
  `../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_<mode>_seed0_20261017_01/train/`。
- 源checkpoint：`compact_checkpoints/task0.pth`、`task2.pth`、`task7.pth`。
  来源训练clean `bfd8ef99f5a13d2c223b069522cf7fe2a099b6e7`，seed0，各30epochs。
- 数据`../multi-lane-main/datasets/EMOTIC`；CLIP
  `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face manifest
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 任务训练eligible人物集合逐项hash对齐四组原始paired audit。按原图聚组，
  RNG seed `0+1009*task`置换组，组内所有当前任务eligible人物一起装入batch，
  每批最多64人（组边界可能导致不足64），每任务4批。原图不跨本任务batch。
- 保持原训练augmentation：Full legacy random crop0.05–1/224、Person
  letterbox/margin0.15、无jitter、Face原manifest处理和可靠性mask、CLIP归一化。
  workers0，生成CPU像素缓存一次，四组与两种数值配置复用同一批tensor/labels；
  记录完整stable IDs、原图数、Face有效人数、输入tensor/targets SHA。
  不声称这些组抽样等于原训练epoch0第一批；它是同分布、固定且可复现的诊断抽样。

## 目标与精度

共4模型×3tasks×4batches×2数值配置=96个batch-gradient snapshot。
不创建optimizer/scheduler，学习率不适用，epochs=0，optimizer updates=0。
只使用autograd.grad，不调用.backward/step，不写参数.grad。

- deterministic_fp32：模型保留training标志使Router balance loss生效，
  仅Expert dropout关闭；前向FP32/TF32 off，梯度scale1。
- training_amp：Expert dropout0.1、AMP FP16/TF32 on；每项loss先乘原训练
  scale1024再求导，随后除1024，记录未裁剪/未预条件梯度。
- dropout seed `0+1009*task+37*batch`固定。两精度配置同时改变dropout和
  算术精度，只作敏感性对照，不能分离两者因果作用；不是两个训练seed。
- 参数范围：当前任务Selector、Prompt、当前类classifier行，当前任务共享
  Expert、3名专门Expert、Router。CLIP、旧/未来专家冻结。主参数只提取当前
  task坐标；独立视图Router不是共享参数，跨视图正交不解释为“没有冲突”。

复用原训练loss函数，legacy_full_zero实际将非当前类logits置零、对26维求均值，
非当前classifier行梯度为零。实际监督为：

```
L = BCE(fused) + 0.1 * mean(BCE(Full), BCE(Person), BCE(Face_reliable))
    + 0.001 * Router_balance    # 仅动态组有非零balance
```

Face无可靠样本时mean只含Full/Person。分别记录：fused、三个单路原始BCE、
实际加权auxiliary、supervised、balance、total；Full/Person与Face在相同可靠
人群的梯度，以及Full/Person/fused在不可靠人群的条件mean梯度。
Face_unreliable_shadow只是假设性loss，不参与实际训练；其实际aux系数为0，
并断言fused loss对不可靠Face endpoint的梯度严格零。

## 梯度归因方法与保护

1. 求各loss对参数组的L2范数、Gram/cosine矩阵。缺失/零梯度标为null，
   不计为“非冲突”；单独报告specialist1/2/3，以区分参数重叠与不重叠。
2. 用原始fused loss对每视图endpoint的VJP，再链式求共享表示/专家梯度，
   精确区分fused中的Full/Person/Face路径贡献。三路之和须重构fused表示梯度。
   classifier直接梯度不属于endpoint路径，不在该分解中解释。
3. 记录实际auxiliary相对fused的范数比、cosine及沿fused方向分量，避免只凭
   两个单路loss夹角为负就推断总体更新被破坏。
4. 对三MoE组做同一参数点的MoE-off额外前向：对照原B0形式supervised梯度。
   另与fresh B0 checkpoint的同坐标梯度比较，明确后者是在不同已学参数点，
   不能当作同参数点因果对照。
5. 每checkpoint诊断前后model.state_dict SHA一致，源checkpoint SHA一致，
   所有参数.grad保持None；缓存像素SHA不变。无validation/test样本读取。

## 预声明解释规则

以描述性门槛cosine<−0.1记“明显冲突”；每任务至少3/4 batch、至少2/3
任务、两数值配置同向才称可重复信号。它不是显著性检验或训练晋级条件。
必须区分：B0本身已有的视图冲突、MoE新增/增强的冲突，以及实际加权aux
是否抵消至少10%的fused方向分量。比较group时限定相同参数坐标与相同样本人群。

负cosine本身不证明泛化损害。只读梯度是endpoint局部、未使用Adam moment
的几何，不能还原整个训练过程，也不能替代梯度隔离重训的因果对照。
若发现稳定额外干扰，才提出最小梯度平衡/隔离控制；若只是共同存在的冲突、
实际aux较弱或total仍与fused基本同向，不据此启动v5。当前仍保留B0。

## 入口、验证与输出

- 模块：`python -m multi_lane.track_a.gradient_attribution`。
- Launcher：`bash scripts/emotic/run_moe_v4_gradient_audit.sh`。
- 先运行Track-A全测，再`SMOKE=1 RUN_ID=smoke_20261017_01`：四组task0、
  每组1批最多16人、两精度，共8 snapshot。通过后正式96 snapshot。
- 输出：`./output/moe_v4_gradient_audit/<run_id>/`；日志
  `./logs/moe_v4_gradient_audit/<run_id>/run.log`。
- 产物：config/输入审计、每模型/task/精度/batch的梯度范数/夹角JSON、源权重
  不变审计、汇总报告；不保存新模型或原始图像，不覆盖已有run目录。
- 完成后同步日志/JSON并做SHA校验，更新PROJECT_CONTEXT/WORK_LOG/TODO_NEXT。
