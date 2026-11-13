# TRAIN临时分工与同轨迹回填：执行记录

日期：2026-11-13。用户授权开始执行；状态：预检通过，准备启动正式训练。

## 协议

GPU0、seed1、8任务、batch64。EMOTIC历史划分，原Frozen9 patch Person+Face交互。Full原路径、各路Task Selectors/Summarize/Task Forward/MLP Head保留。独立CLIP预训练起点，不导入历史分支或融合权重。

原强参照复用Val44.0329/51.1335、Test33.1770/40.2083。新训练两条轨迹：full_extend全TRAIN30轮+全TRAIN5轮，固定融合；holdout_return按原图80%训练30轮，20%在同一模型上BCE反传融合100步后冻结，全部TRAIN回填5轮。同一holdout轨迹报告固定与学习融合，原固定可靠0.64/0.16/0.20、不可靠0.80/0.20/0。可学习两标量等权初始化，不用原比例锚点。

初阶段主LR0.0125、Adapter/交互0.0004，Adam/cosine、WD0、b32/index1/scale0.1、原BCE/Adapter-ASL。恢复阶段保持Adam动量、同一参数、同一scaler，峰值LR降到原0.1倍，5轮cosine；分类头正常更新。融合FP32 Adam LR0.02 WD0，仅TRAIN当前类别；不向分支回传。

Face原筛样保持。按原图固定hash划分及前任务曝光审计，明确后续H可能已在历史任务训练，非全轨迹严格未见。A/H曝光35/5轮，不称全TRAIN每图30轮。同一run内完整checkpoint保存branches/optimizer/scaler/RNG/fusion，任务结束后封存。

## 路径与运行

代码分支codex/r1-fusion-return-train-20261113，基于aa8dbd3。入口python -m scripts.emotic.run_r1_fusion_return_train。

输入沿已有相对路径：../multi-lane-main/datasets/EMOTIC；CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face manifest ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。配置记录来自../emotic_benchmark_runs但不加载历史训练参数。

正式输出output/r1_fusion_return_train/seed1_20261113，日志logs/r1_fusion_return_train/seed1_20261113。两条轨迹依次，每条Full/Person/Face并行，仅GPU0。正式结果自动完整8阶段Val/Test；task0/2/7回填前后同checkpoint固定/学习读出诊断，不按Test选择轮次。

## 预检

需要：数据划分/Face筛样/全量回填、同Adam动量跨阶段、学习率边界、融合TRAIN限制/旧任务冻结、完整checkpoint两任务恢复状态与两split前向。smoke为2任务、2轮初阶段+1轮回填，每轮2batch、融合2step；不能把smoke效果当正式结果。

本次暂不增加SAM、门控、排序损失或额外Attention。通过预检后启动已授权完整实验；记录源码HEAD、预训练SHA与输出来源。

预检记录：首次单测发现浮点LR严格相等断言过严（.0012500000000000002与.00125），改为数值精度比较；模型/Adam状态逐值断言保留。原runner回填后calibration_scores明确标为已见诊断，融合只读回填前observations。

门禁结果：21项检查通过。smoke01两条件共72次有效分支更新、零跳步、16组双split完整checkpoint前向，exit0。源码798fcd9。正式启动将记录独立tmux和运行状态。
