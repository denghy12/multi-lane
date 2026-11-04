# 最终任务 CLS 的 QK 尺度最小对照

上一轮 QK 在固定 Val 探针上接近均匀。完整 Val 均匀替换后 mAP 43.9833／51.1484，学得权重为43.9734／51.1407。原 R1 为44.0900／51.2467。本轮重训均匀读取和 QK 归一化 Attention，检验来源选择能否带来额外排序收益。

| 项目 | 固定配置 |
|---|---|
| 数据与模型阶段 | EMOTIC 原 train/val/test，seed1、8任务；共同原生阶段复用 r1_final_cls/seed1_20261104 的本轮自行训练 checkpoint |
| 原生配方 | CLIP ViT-B/16，任务独立Adapter b32/index1/scale0.1，原Selectors与Prompts，BCE及Adapter-ASL；三路各30轮/任务，batch64，主LR0.0125、Adapter LR0.0004 |
| 交互输入 | 三路512维最终任务CLS，逐阶段加载当期checkpoint；确定性TRAIN变换，当前任务类别人群 |
| 交互输出 | Person及可靠Face修正，Full原输出；特征L2范数上限0.05，原分类头权重投影后加到logits |
| 均匀组 | 对有效外部来源均匀加权，重新训练输入投影、视图嵌入、V和输出投影 |
| 归一化组 | 与均匀组相同初始状态；Q／K线性投影后分别无仿射LayerNorm，再按sqrt64计算masked softmax；V尺度沿用原值 |
| 预算 | 每任务5轮、B64、Adam4e-4/WD0、cosine；同一shuffle generator及初始化；总存储参数78016/任务，Uniform有效训练参数69824，QK组78016 |
| 最终读出 | 可靠Face 0.64/0.16/0.20，不可靠0.8/0.2/0；先融合logits再sigmoid |

固定TRAIN probe取当前任务内最多32个可靠Face与32个不可靠Face，记录ID。零输出初始化的首次反传只有out获得非零梯度，随后在probe副本更新输出再检查Q/K/V梯度。Uniform QK梯度须为零，归一化组在可靠Face样本上Q/K须有非零有限梯度。probe副本不进入正式训练，两组正式初始化SHA相同。

每轮保存训练BCE、Q/K/V/out梯度、来源权重、熵、打分差、残差范数与cap命中率。恢复状态包括模型、Adam、scheduler、已完成轮次、shuffle及CPU/CUDA/Python/NumPy随机状态。任务结束后模块冻结，逐阶段检查旧模块和分类头块。TRAIN/Val/Test特征缓存按来源SHA与Git协议绑定。

额外记录归一化checkpoint强制均匀的条件前向。它与重新训练的Uniform分别回答推理权重作用和训练路线差异。Val/Test完整八阶段均报告全人群、可靠Face、不可靠Face的最终和平均mAP以及逐类AP。原R1与冻结patch Face Pool为同logits口径参照。

推进方向轮次的规则为归一化组Val最终和阶段平均同时超过原R1及重训Uniform。条件成立后仍需判断正确人物来源是否优于错配，才能把收益解释为人物内容选择。QK随图变化、训练BCE降低或cap命中本身均不能替代这一判断。

入口 `bash scripts/emotic/run_r1_cls_qk_control_gpu0.sh`。GPU0空余至少6000MiB时启动；两任务TRAIN/Val smoke先通过，再启动正式8任务。输入原生阶段 `../multi-lane-main-r1-final-cls/output/r1_final_cls/seed1_20261104`，参照 `../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02`。输出 `output/r1_cls_qk_control/seed1_20261104`，日志 `logs/r1_cls_qk_control/seed1_20261104`。

本轮首先检验投影尺度假设。如果QK活跃而仍弱于Uniform，应检查接收方语义与来源对应，再研究最终CLS是否保留足够的互补证据。接收方向、Full写回以及额外容量的变化留待本轮结果判断。
