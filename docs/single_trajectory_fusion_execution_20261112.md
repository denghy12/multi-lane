# 单一训练轨迹可学习融合执行记录

状态：实现与预检通过，正式训练即将启动。分支codex/r1-single-trajectory-fusion-20261112。

## 配方与数据

EMOTIC历史划分、seed1、8任务，每任务30轮、batch64、Adam/cosine/WD0。主LR0.0125、Adapter和交互0.0004；b32/index1/scale0.1、原BCE及Adapter-ASL。Full原路径，Person/Face第9层patch交互，原始冻结发送tokens、QK归一化、5%残差上限、P/F固定AMP4096。Full数值策略保留。全部合法TRAIN用于分支，Face原筛样；不永久留出TRAIN，不读旧实验的模型/融合参数。

输入沿用../multi-lane-main/datasets/EMOTIC；CLIP ../CODE_DDP/pretrained/clip/ViT-B-16.pt；Face manifest沿用input_paths()中历史train/val/test版本。reference_root只读取历史配置JSON，不装入历史权重。融合FP32 Adam .02/WD0、当前任务BCE、等权初始化。每轮同步后、当前分支eval/no-grad、标准评估预处理完整当前TRAIN对齐输出，C1/C2各自更新；图像tokens/Selectors/Head不接受新增融合梯度。

C0原固定；C1第6–30轮每轮4次；C2第6–15轮每轮10次、16–30轮冻结。均100次/任务。三种读出同时依附于唯一原分支训练轨迹，避免重复训练相同的三路；每个任务完整checkpoint含全部三路及两种融合/优化恢复状态。C2冻结是本模型自己的预定训练日程。最终从同一checkpoint直接执行三种读出。

## 入口和产物

`CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_single_trajectory_fusion run`

输出output/r1_single_trajectory_fusion/seed1_20261112；日志logs/r1_single_trajectory_fusion/seed1_20261112；完整任务checkpoint、latest恢复状态、每轮融合轨迹、TRAIN观察、两split真实图像预测与分群/逐类/阶段mAP。训练72线程，观察2线程恢复全部RNG和模式，观察独立DataLoader generator和worker1线程。GPU0并行三路；融合仅轻量参数。

## 门禁

单测检查100次预算、旧任务冻结、不可靠Face零权重、停止梯度、状态恢复、拒绝Val训练、异常时RNG/模式恢复。真实smoke使用2任务×3轮×2batch，融合C1轮2/3各1次、C2轮2共2次；之后额外从头运行无观察控制，比较6条任务路径参数/RNG及18份epoch优化器/scheduler/scaler状态。通过后正式运行；任何跳步立即保留状态并停止，不隐瞒预算变化。

训练结束直接由完整checkpoint做8阶段Val/Test真实前向，不在Val/Test上反传/拟合/混装。原固定四指标44.0329/51.1335、33.1770/40.2083作为历史参照，正式新控制先重放；未承诺学习必胜。

## 预检实际结果

20项单元/回归测试通过。smoke03训练2任务×3轮×每轮2batch，三路总36次有效更新、零跳步；任务checkpoint完整模型真实图像前向覆盖Val/Test。对6条任务路径及18个epoch状态核对，参数/RNG及Adam/scheduler/scaler均与无额外观察控制逐值一致；smoke_replay.json通过。

预检修复：原smoke要求显式更新预算，改为声明6次并以3×2执行；原atomic_save禁止覆盖，完整恢复改为recovery/task*/epoch*.pth并由checkpoints/latest.json索引；输入搬运把可靠性标记转float，导出时显式bool，补评估成功。失败日志保留，最后只补评估和对照，smoke训练checkpoint未修改。

正式代码c69961d，正式运行将从头初始化；检查通过不代表正式效果已提升。原主树与test-only未改。无跨实验训练参数导入。
