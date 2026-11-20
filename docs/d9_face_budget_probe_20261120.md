# Face 更新预算与同配方学习融合的缓存验证

EMOTIC，seed1，全部八任务，90% 分支原图与 10% 融合专用 TRAIN 原图均沿用历史划分。候选在三个清晰度条件中分别取等步数模型 Full、Person 和原预算模型 Face 的对应 logits。用两来源同一批人、相同类别和 Face 标记重估配对 A。每个任务仅使用当时的留出 TRAIN，将可用视图等权初始化，用 Adam 0.01、300 步、每人物等权的多版本 BCE 拟合分组比例。当前任务的 A 封存用于该任务类别。固定比例仍为可靠 64/16/20、不可用 80/20/0。最终输出为加权 logits 的 sigmoid。

按自然 Validation 八阶段和原 AP 实现比较全体、可靠、不可靠人群；先完成 Validation 封存，再同批评估 Test。是否推进单轨迹训练在 Validation 阶段确定：候选学习比例两项均达到候选固定比例，且至少一项达到当前正式等步数 A。严格使用未舍入数值。另列与历史全 TRAIN 固定的距离。Test 不参与拟合或推进决策。该混合预算缓存由两条历史轨迹生成，仅用于验证训练假设。

运行入口为 `bash scripts/emotic/run_d9_face_budget_probe.sh`。只用 CPU，复用现有预测和原配对 A 实现。结果存放在 `output/d9_face_budget_probe/seed1_20261120`，运行日志在 `logs/d9_face_budget_probe/seed1_20261120`。
