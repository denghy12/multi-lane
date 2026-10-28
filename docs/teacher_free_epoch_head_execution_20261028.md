# 无教师共同学习：epoch轨迹与表征×分类头只读归因

2026-10-28用户授权执行。分支codex/teacher-free-epoch-head-attribution-20261028，基于701e8df。仅新增独立归因/队列/测试，不改变模型、原训练/评估、checkpoint、增强或超参数；不新建optimizer，不做任何训练更新。

## 目标与范围

使用此前task0三组U0原配方、U2互补排序、U4排序＋Full/融合保护的已有epoch1/5/10/20/30。固定同一历史seen-positive人物：Val2285、Test3142，当前5类。报告完整轨迹，不按Test选择epoch，不当成8任务结果。

每个epoch的7种读出：
- U0表征＋U0头（两矩阵共用基准）
- U0表征＋U2头、U2表征＋U0头、U2表征＋U2头
- U0表征＋U4头、U4表征＋U0头、U4表征＋U4头

表征指共享Adapter/Selectors/Prompts共同产生的三路最终任务CLS及原固定特征融合；头指同epoch对应模型完整共享线性分类头（包含bias），保留原task类别归属和前归一化；不再缩放、归一化或训练cross-head。交叉读出只做归因，不部署拼装模型。

每份表征每batch仅编码一次，复用Full/Person/Face/fused特征用于对应head读出，共3份表征编码覆盖7条件。CPU数据仅加载一次供三来源复用。一个host模型在独立autocast context之外加载source state，CLIP权重共享且不变；不在原模型中插入新模块。Val/Test两进程GPU0并行。

## 固定条件

- 原source训练：seed1、task0×30轮、B64、Adam/cosine、主LR.0125/Adapter.0004、b32/index1/scale.03、BCE/Adapter-ASL、aux.1、Person颜色扰动0。
- 本次：0训练epoch/0优化更新；epoch数是读取的既有边界。
- GPU0；OMP/MKL模型72线程；AMP/TF32；batch64；2个单线程spawnworker、prefetch1；原CPU分batch sigmoid；Face原可靠性mask。
- Val/Test样本、类别、顺序、batch长度、固定融合权重、head位置、阈值.5均保留。
- BCE独立按当前5类logits计算均值及每类值，不含legacy其余21类常数，明确与训练总loss定义不同。
- 全部四路及可靠Face保存mAP/逐类AP/cF1/oF1/BCE；附可靠/不可靠Face人物上的Full/融合结果。
- 两个2×2都计算固定原头下的表征差、固定原表征下的head差及交互项；mAP非线性，这些是条件干预差，不是唯一训练因果贡献。
- Test已参与开发，完整轨迹不作为未参与选择的独立验证。

## 来源与产物

原source代码701e8dfa71ae769128c34c7bd34d2c222d83fdae。
输入：`../multi-lane-main-teacher-free-shared-learning-20261027/output/teacher_free_shared_learning/seed1_20261027_01`：
三组各`train/epoch_recovery/task0/epoch01/05/10/20/30.pth`（实际文件名epoch01.pth等），以及finalcompact及已有Val/Test预测做重放。15个epoch文件均存在，每份约10MB。文件按来源完整恢复payload只读加载，不恢复训练随机状态，不写source。

其余输入沿用：
- 数据`../multi-lane-main/datasets/EMOTIC`
- CLIP`../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- Face`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`

输出`output/teacher_free_epoch_head/seed1_20261028_01`；日志`logs/teacher_free_epoch_head/seed1_20261028_01`。
入口`bash scripts/emotic/run_teacher_free_epoch_head_gpu0.sh seed1_20261028_01`，默认ddpPython。

预期：smoke前64人5epochs×7条件；完整两个split各5epochs×7条件的预测NPZ/逐类指标/BCE、2×2交互和参数漂移、输入人群恒定证明、源SHA及无更新记录、耗时显存和completion。epoch-completion marker支持自身已完成产物SHA验证后复用，不覆盖source。无best-epoch挑选、无跨任务推断。

## 验收和部署

先机制测试＋既有Track-A/相关回归，再Validation首64执行全部条件。以下都必须通过：
1. 原模型/变换/指标等推理依赖与训练701e8df逐字节一致。
2. epoch1三source全部模型Tensor相同，全部7条件预测逐值相同。
3. 直接读取缓存特征＋对应头与原seen_logits_with_views逐值一致。
4. epoch30权重与finalcompact相同；原生四路logits/CPU概率对已有完整预测逐值重放（smoke核前64，正式核完整）。
5. 同一split各epoch与各条件人物/标签/mask/batch恒定；无梯度/参数变化；CLIP与15epoch+3final及原预测SHA前后保持。
6. parent先smoke，通过后在GPU0启动Val/Test，显存至少4.5GB、共享内存至少768MB；不使用其他GPU。
7. full结果回传SHA核验，主树/test-only保持。

当前实现已完成、本地编译及bash检查通过；远程测试/smoke/正式待启动，不预称已有归因结论。
