# 固定真实基线，先检验旧功能，再学习新专家

日期2026-10-12。用户已授权修改并执行。只研究增量冻结与复用；seed0、task0–7完整Validation/Test，不纳入全量类别联合学习。

## 本轮问题和共同设置

上一轮完整输出组合仍低于基线；旧功能不变，但旧残差约为新残差的1.5%–1.8%，task0已表现为训练升、外部降。因此本轮同时建立新增模块的TRAIN留出，并用阶段化拟合检验旧功能。分组与学习顺序均属于明确变化，跨轮不能归因于单独一个因素。

- 固定同一批8个真实原基线checkpoint，原CLIP、Selector、Prompt、Task Forward和head均不训练。
- 位置保持Task Forward最终归一化CLS之后、固定三视图特征融合之前；Image-token Adapter和旧patch Projector关闭。
- 每个完整专家是`B_i GELU(P_birth(i) A_i LN(x))`，两个同生专家共享出生任务P。旧功能保留原P/norm/scale。
- 三个level共享专家和任务固定系数。没有图像相关Router、level条件、动态融合、蒸馏或新的损失。
- 每任务两个新专家，rank32；总残差经过同一2%平滑范数界。新P零初始化，其余A/B非零；旧功能每项有独立的`tanh(c_i)`系数，c_i从0开始，零处直接获得梯度，不再先乘一个共享零门后才训练组内Router。
- 旧／新输出允许带符号相加。记录实际系数和残差，不将它们解释为概率gate；model诊断对这类系数不再输出概率熵和top频率。

## 四个对照和共同旧专家来源

| 对照 | task0 | task1–7 | 每任务本组优化预算 |
| --- | --- | --- | --- |
| 固定真实基线 | 恢复原权重 | 恢复原权重 | 0 |
| 只学新完整专家 | 学两个新专家 | 只学当前两个新专家 | 两阶段各5轮，共10轮 |
| 只组合旧完整功能 | 没有旧功能，等于基线 | 冻结全部函数，只学旧输出的任务固定系数 | 5轮；task0为0轮 |
| 先组合旧功能，再学新专家 | 与只新组相同的10轮 | 先学旧系数5轮，冻结系数，再学两个新函数5轮 | 共10轮 |

两种旧功能组读取**同一套供体池**：供体来自本轮“只新专家”组在拟合部分学到的完整函数，学完该任务才封存加入池，下一任务起才能使用。分阶段组新学出的函数用于本组当前/旧任务识别，但本轮不把它们转成另一套独立供体，避免两个组读取不同的旧知识。

这是受控的旧功能价值诊断，不是各组独立自举专家池的完整最终方法。供体训练成本必须计入来源说明；不能把只旧组写成完全不需训练专家。共同供体及每组自身新专家在实现中分开保存；checkpoint包含供体来源SHA、访问范围和封存状态。

只新组与分阶段组总更新次数匹配，但可训练参数量与分配不同：新阶段更新两对A/B、P和两个固定新组合logits（视觉维度512时66594参数）；旧阶段仅2t个系数（2–14参数）。只旧组是较低成本的机制诊断，不能声称四组严格等容量。旧MLP Router保留兼容结构但全程冻结且不参与计算。

## TRAIN原图分组留出

- 从完整TRAIN的canonical `image#person=...` ID提取原图key。
- 使用`SHA256('module-holdout-seed0-v1:'+image_key)`前8字节模5，等于0为留出，其余拟合。目标80%/20%，实际比例按原图计数记录。
- 分配与任务、标签、样本顺序无关，task0起保持一致；同图所有人物、所有level、所有任务不能跨组。
- 所有供体、新专家和组合系数只拟合fit部分。缓存来自冻结原基线，可以复用；上一轮看过全部TRAIN的专家权重不复用。
- 原基线曾使用全部TRAIN；这里是新增模块的留出，不是原模型也从未见过的独立数据。正式Validation/Test仍作外部评价。
- 留出只记录baseline/adapted mAP、BCE、残差，不用于选轮次、开关或系数。每阶段固定5轮。记录所有任务两侧ID及类别正例数；不按稀有类别重新抽分组。
- 仍沿用当前任务缓存人群，不在本轮同时加入旧图片、全负样本或排序损失。

## 完整实验配置与入口

- 数据EMOTIC train/val/test，CLIP ViT-B/16，seed0，任务大小[5,3,3,3,3,3,3,3]。
- 每阶段5epochs，batch64，FP32 Adam lr0.0004、WD0、clip1；cosine每5轮重置。只新组两阶段新5+新5，分阶段组旧5+新5；task0两者新5+新5且严格配对。
- 每阶段独立CPU排列生成器：seed+1009*task+100000*phase_index。同一旧阶段的只旧／分阶段组初值、数据顺序和供体相同，系数须逐值一致。
- 目标为固定融合BCE＋0.1可靠单路辅助BCE，乘当前类别数/26；consistency0。
- 可靠Face的固定特征融合Full/Person/Face=0.64/0.16/0.20，否则0.80/0.20/0。可靠条件：有效、无匹配歧义、短边≥24、检测分≥0.6。
- 正式所有拟合/留出/Validation完成后才创建Test数据集；恢复24份新模块checkpoint只读Test，原基线Test沿用同源已有结果。禁止根据Test选择配置。
- 原基线：`../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed0/baseline`。
- TRAIN缓存：`../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed0/train_cache/task{t}.pth`，逐task验证来源checkpoint、样本、标签和缓存SHA。
- 数据路径：`../multi-lane-main/datasets/EMOTIC`；预训练：`../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`；Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- Python入口：`python -u -m multi_lane.track_a.fixed_baseline_staged_functions`。
- Shell：`scripts/emotic/run_fixed_baseline_staged_functions.sh`。
- 预检batch：`staged_old_functions_smoke_20261012_01`；正式：`staged_old_functions_seed0_val_test_20261012_01`。
- 输出：`./output/emotic_staged_functional_experts/<batch>/`；日志：`./logs/emotic_staged_functional_experts/<batch>/`。
- 产物：原图划分清单、每任务fit/holdout ID和正例数、两阶段损失/梯度/残差、TRAIN与留出诊断、逐任务/单路Validation/Test分数、checkpoint、旧参数/完整函数/FP32预测审计、共同供体来源SHA。
- 分支`codex/staged-old-functions-holdout-20261012`，独立远程worktree `../multi-lane-main-staged-old-functions`，Git-only同步；原primary和test-only工作树不改。
- GPU0，初查空闲约24GiB；优先GPU_POLICY=idle，其他GPU任务不触碰。若启动时资源变化，沿用此前明确授权的GPU0共享方式且至少8GiB空余；不足则不启动，不停止他人任务。

```bash
GPU=0 GPU_POLICY=idle BATCH=staged_old_functions_seed0_val_test_20261012_01 \
 bash scripts/emotic/run_fixed_baseline_staged_functions.sh
```

## 验证与停止规则

1. 完整Track-A单测与Shell语法；专门覆盖原图跨人/跨任务隔离、holdout无法进入optimizer、八任务旧路径不变、共同供体安装不影响旧预测、阶段参数冻结、零起点梯度、2%界、checkpoint来源/owner/mask/未来供体拒绝。
2. 真实ViT smoke同样跑8任务、3个模块组，每任务fit/holdout各最多64样本、val/test最多64样本，阶段轮数不变。smoke不作正式精度。
3. task0只新与分阶段预测应一致，只旧应与基线一致；task1–7旧阶段两组系数逐值一致；所有旧函数及旧参数不变，新任务合法访问，readonly Test状态不变。
4. 通过后启动正式，确认训练正常即停止实时监督，结束后统一分析。
5. 旧功能单独在留出和Validation无一致收益，停止当前后置专家扩展。只保住基线不能算新增收益；旧功能单独有用、加入新专家失效，才继续研究竞争；简单固定组合有效后再研究图像Router。最终必须报告全部八任务Test，不按Test挑任务启用。
