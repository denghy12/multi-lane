# Person-conditioned Full Selector：第一版与patch交互版

## 第二版：selector-specific Person patches

第一版实验表明共享Person CLS增量降低test mAP，因此第二版不再让所有Selector共享同一个
Person向量。新增`person_patches`模式，在指定Full block（当前固定zero-based layer1）的输入深度，
取同一冻结CLIP Person支路的patch tokens。每个normalized Selector分别作为query，对有效Person
patches做scaled dot-product attention，得到各自不同的Person summary；summary再经过当前task独立的
`Linear(768,32) -> ReLU -> Linear(32,768)`并乘scale0.1，作为该Selector的query residual。
末层weight/bias为零，因此初始输出严格退化为原Full Selector通路；Person视觉支路全冻结，旧task
条件模块继续冻结。条件只影响Full token汇总query，Drop & Replace仍恢复原Selector。

Person依旧采用bbox+margin0.15后letterbox到224。转换器同时生成14×14 patch mask，仅patch中心落在
原Person内容区的token可参与softmax，padding颜色不会成为伪人物证据。水平翻转与Person像素同步应用
到mask。Person只推进到条件层需要的深度，不运行无用的后续block或Person分类头。

Full新增`--full-crop-mode legacy|target_aware`。target-aware训练仍从0.05--1.0面积和3/4--4/3
长宽比随机采样，但只接受完整包含目标bbox的裁剪；采样约束不可行时回退完整场景。评估先按短边256
resize，再把中心裁剪平移到能完整包含目标的位置；如果目标本身大于224，则以目标中心作最大保留。
无效bbox沿用legacy随机/中心裁剪并关闭条件。这样可独立比较裁剪主效应与Person patch交互。

2×2入口为`launch_multilane_track_a_selector_patch_2x2.sh`，组合是legacy/target-aware ×
disabled/person_patches。其余训练配置保持当前Full冠军不变。val与test需要相同完整训练预算；本轮依
用户要求直接seed0 test，但须作为结构探索报告，不能再用其反复搜索内部超参数。

## 结构与边界

在 `MultiLaneModel._lane_block` 的 Summarize 前，增加一个由当前样本产生的
Selector 查询增量。基础 Full visual、Image-token Adapter、task CLS、prompts、
Drop & Replace 和分类头保持原有计算结构。

`query = normalized_selector + 0.1 * condition_mlp(features)`。
增量只影响读取 Full tokens 的相似度；Drop & Replace 恢复的仍是原 normalized
selectors，增量不会写回 frozen image tokens 或 selector 参数本身。
第一版每个样本产生一个 768 维增量，广播到所有 selectors；不是每个 selector
独立产生查询，也不是完整 cross-attention。后者留给本版验证后的结构消融。

| mode | 条件输入 | 人物视觉编码 |
| --- | --- | --- |
| disabled | 无，可接收 paired 字典作为对照 | 不执行 |
| bbox | Full 实际裁剪中的归一化 xyxy、可见面积比例、框有效标记 | 不执行 |
| person | 冻结 CLIP 最后 block 后、ln_post 后、投影前 CLS | 额外一次冻结前向 |
| bbox_person | 上述两类特征拼接 | 额外一次冻结前向 |

人物编码和 Full 共用同一份冻结 CLIP 权重，不使用 Person 分类器或 Person Adapter，
也不加载经过全 EMOTIC 类别监督的教师。两个视图有两次视觉编码，不能宣称一次前向成本。
Person-only 内容模式仍使用 bbox 生成 crop，并使用可见性掩码，但 bbox 数值不输入 MLP。

每个任务、每个指定层分配独立 `Linear -> ReLU -> Linear`，hidden32，末层权重与偏置
初始化为零，前层随机初始化。整个 bank 初始化隔离 PyTorch RNG。新任务保持独立初始化，
旧任务模块冻结、清理梯度并从新优化器中排除；restore_task 恢复相同路由。关闭模式
不增加 state_dict 键，旧 checkpoint 保持兼容。新状态保存所有任务模块。

默认只在 zero-based layer1 加条件模块。参数量每任务每层：bbox 25,568、person
49,952、bbox_person 50,144。它们属于主模型目标（本实验 BCE），不是 Adapter ASL
参数组；独立 LR4e-4，跟随同一个 per-task cosine scheduler。
旧分类头仍训练，因此冻结旧 conditioner 不能保证整个模型零遗忘。

## 成对输入与随机性

EMOTIC 按一条人物标注同时提供 full/person/bbox/condition_valid，沿用稳定 sample ID。
两个人共享一张图时，Full 可以相同，Person crop、框和标签仍各自对应。

- Full train 精确复用原 RandomResizedCrop 的取样顺序与双线性 resize，随后一次水平翻转。
- Full eval 保持 Resize256/bicubic + CenterCrop224，包括奇数尺寸的舍入规则。
- bbox 随实际 Full crop/resize/flip 映射，返回裁剪后的 xyxy 与可见面积比例。
- Person 使用原图目标 bbox+margin0.15，再 letterbox/Resize224；不做二次随机方形裁剪。
- Person 与 Full 使用同一次 flip。轻量 ColorJitter 在 forked RNG 内执行，保持外部
  Full augmentation/DataLoader 的 PyTorch 随机轨迹。
- 无效框、完全落在 Full crop 外的框将 condition_valid 置零，所有模式均回退原查询。
  部分可见目标仍使用完整 Person crop。这是第一版预先固定的规则，没有搜索可见性阈值。
- letterbox 的补边不是实际人体内容，但第一版只使用 CLS，不做 patch 级空间交互；
  如果后续读取 Person patches，需要补边 token mask 与更细的坐标映射。

## 已实现的入口

`python -m multi_lane.track_a.runner` 新参数：

```text
--selector-conditioning disabled|bbox|person|bbox_person
--selector-condition-layers 1
--selector-condition-hidden-dim 32
--selector-condition-scale 0.1
--selector-condition-learning-rate 0.0004
--paired-full-person
--input-mode full
--person-transform-mode letterbox
```

训练、current validation、all-seen evaluation、score dump 均接受 paired 字典。
config 保存条件模式、来源、层、参数量、LR、输入几何规则。GPU smoke 入口也接受
`--selector-conditioning` 和 `--selector-condition-layers`，检查初始预测等价与条件梯度。
training_history 逐 epoch 记录条件LR、有效目标比例和平均可见面积比例，便于解释
Full crop0.05造成多少样本的条件路径被关闭。

Linux validation 脚本：
`bash scripts/emotic/run_multilane_track_a_person_conditioned_val.sh`。
需要设置 GPU、MODE、RUN_ID、CLIP_CHECKPOINT；可覆盖 SEED、DATA_ROOT、OUTPUT_BASE、LOG_DIR、
REPORTING_SPLIT（val或test，默认val）。脚本为两种split设置对应的score purpose。
脚本要求 clean worktree，拒绝覆盖同名结果/日志；完整配置在启动前打印。
默认结果写 `output/emotic_track_a_person_conditioned_selector/`，日志写
`logs/emotic_track_a_person_conditioned_selector/`。

固定 EMOTIC/seed0/完整8-task/100%训练数据/30ep/batch64/Adam/main LR0.0125，
Image-token layer1/b32/LR4e-4/scale0.1/ReLU/independent，主模型BCE+Adapter
ASL9.8/0/0.05，AMP/TF32 on，CLIP normalization，Full crop0.05–1.0。
保存实际概率、logits、IDs 和标签，不保存 checkpoints。
该脚本只运行一组，不自动搜索融合权重。

后续四组：disabled、bbox、person、bbox_person。分别分析条件化 Full 的独立指标；
需要评价原固定融合时，与同 seed 的既有 Person 分支按 0.8/0.2 probability 配对。
冻结 CLS 条件支路本身不产生那个 Person 分支预测；若同时保留外部 Person 分类器，
应如实计入额外计算，不能将本实现称为已经完成统一单头融合。

选择优先级：final validation mAP、average mAP、task5–7 mAP；逐类/F1/forgetting
为解释指标。seed0有效后再补seed1/2、锁定一次test。实际GPU实验尚未启动。

## 验证

新增测试覆盖：所有模式的零初始化等价/RNG、可学习条件影响、无效目标回退、BCE/ASL
梯度路由、冻结视觉、优化器参数覆盖、独立新任务与旧任务冻结、compact state重建、
训练/val/score dump 成对输入、Full像素及RNG等价、框裁剪翻转、人物crop完整性、
同图多人稳定ID和标签配对。真实CLIP/CUDA及真实EMOTIC数据 smoke 仍需Linux服务器。
GPU AMP初检发现额外冻结Person前向会使重复Full计算不再逐bit相同；smoke沿用项目既有
AMP 1e-4/FP32 1e-6容差并报告最大差值，超过容差仍直接失败。
