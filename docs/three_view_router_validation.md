# Three-view sample-wise router validation

## 实现验收（2026-09-09）

分支`codex/three-view-router-validation`已完成Git-only服务器同步。服务器ddp环境146项完整单测、
真实train/val三视图ID与target对齐、稳定90/10分桶、冻结CLIP描述符GPU smoke以及Face source
task0两步GPU smoke均通过。Face smoke确认calibration在专家fit之外，且fit侧无效/歧义Face在
DataLoader前过滤；概率文件和compact task state成功写出。当前允许启动seed0完整validation，
仍禁止访问或运行test。

## 问题与实验边界

阶段1已证明可靠Face能将seed0 Full+Person完整validation final mAP从43.3035提高到43.5812，
但固定权重无法按样本判断应由Full、Person还是Face主导。本阶段只在validation验证样本级Router；
不访问test，不训练类别独立权重，不扩大网络或搜索静态Face比例。

## 数据与三个固定专家

- 数据集：EMOTIC；seed0；8个增量task。
- 分桶：稳定image-group SHA-256，90% fit训练专家、10% calibration训练Router。
- Full/Person：复用2026-09-04已完成且带calibration/validation scores和compact checkpoint的端点。
- Face：重新按同一分桶训练；loss view严格为`valid_face && !ambiguous_match`，calibration score
  保留完整分桶样本以与Full/Person按stable ID对齐。
- 三个专家保持30 epochs/task、batch64、main LR0.0125、per-task cosine、Image-token layer1/
  b32/LR4e-4/scale0.1/ReLU/independent、main BCE+Adapter ASL9.8/0/0.05、AMP/TF32。
- 专家训练和Router训练均不读取validation标签以外的选择信息；calibration图像组不进入专家fit。

## 四类比较

- R0：`0.8 Full + 0.2 Person`。
- R1：可靠Face固定beta0.20，即`0.64 Full + 0.16 Person + 0.20 Face`；无效Face回退R0。
- R2：样本级软路由，输入body/Face质量、三路confidence、entropy及两两预测差异。
- R3：R2再增加三个冻结CLIP最终图像embedding余弦：Full–Person、Full–Face、Person–Face。

R3只增加3维真实视觉内容描述，不输入512维特征。描述符只导出全局10% train calibration和完整val；
Face无效时相关余弦归零并由独立mask强制Face权重为0。冻结CLIP描述不依赖某个task checkpoint，
不会用task7特征冒充早期task特征。

## Router

- 每个task独立参数与标准化统计；共享的只有架构。
- 单隐藏层GELU，hidden16，输出每样本一个三路权重，task内类别共享。
- masked softmax；Face不可靠时其权重严格为0，至少Full/Person始终可用。
- 有效Face初值`[0.72,0.18,0.10]`；无效Face初值`[0.80,0.20,0]`。
- 当前task calibration类别BCE；AdamW LR1e-3、WD1e-4、batch64、80 epochs/task。
- R2/R3各只比较prior强度`{0,0.1,1}`；prior为输出权重到相应初值的平方距离。
- 特征均值/标准差只在当前task calibration拟合，随后冻结并应用于validation。

task6 calibration只有51条，且三类正例有限。因此禁止增加hidden、类别独立权重、更多prior或按val
早停。必须输出calibration/validation BCE gap、权重均值/std/分位数、Face可靠数和完整逐task指标。

## 选择和停止规则

R2与R3分别按final mAP、average mAP、较强prior选出内部最佳。只有最佳R3的完整validation final
mAP同时严格高于R1和最佳R2，才支持“视觉内容驱动的动态路由”，并进入seed1/2 validation。
否则停止扩大Router：R1胜出则保留固定三路；R2胜出则说明质量/预测统计足够；所有动态方案失败则
保留阶段1固定融合。无论结果如何，本批`run_test=false`。

## 入口与产物

- Face source：`scripts/emotic/run_multilane_track_a_reliability_source_val.sh`，`VIEW=face`。
- 冻结视觉描述：`scripts/emotic/run_multilane_track_a_three_view_descriptors.sh`。
- 两GPU安全排队与完整批次：`scripts/emotic/launch_multilane_track_a_three_view_router_val.sh`。
- 外部结果：`/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_router_v0.1/`。
- 控制结果：`output/emotic_track_a_three_view_router/`。
- 日志：`logs/emotic_track_a_three_view_router/`。
