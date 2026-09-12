# Face 表情预训练专用表征：seed0 validation 协议

## 目标与边界

本阶段不再调整 Router hidden、rank、prior、描述符、类别独立网络、Face margin、可靠性阈值或
ColorJitter。唯一变量是 Face 表征本身，Full、Person、可靠性 mask 与固定 R1 融合均保持不变。

旧锚点为 CLIP ViT-B/16 Face expert。新候选共享以下输入和训练池：目标 Face 匹配结果、manifest
已有五点 landmark、五点相似变换、margin0.15 头部语境、224×224、ImageNet normalization，以及
`valid_face && !ambiguous_match` 训练 mask。若关键点无效则数据读取回退原 margin0.15 + letterbox；
正式训练前审计要求 train/val 有效非歧义 Face 的有效五点覆盖率均至少 99%。

## 冻结 encoder

首轮锁定官方 EmotiEffLib `enet_b0_8_best_afew.pt`，即 AffectNet 表情训练的 EfficientNet-B0。
项目地址与模型许可为 Apache-2.0：

- <https://github.com/sb-ai-lab/EmotiEffLib>
- <https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/enet_b0_8_best_afew.pt>

权重放置在服务器
`/mnt/haoyuan/workspace/pretrained/emotiefflib/enet_b0_8_best_afew.pt`。运行脚本要求显式传入
SHA-256 `47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17`并逐次校验，
encoder始终冻结且保持eval，BatchNorm统计不得跨task漂移。FERPlus保留为后续
独立方向，本轮不混合预训练来源。

## 两个候选

1. `projection`：冻结表情encoder输出直接进入每task独立Linear投影；新task只训练自己的投影，旧
   task投影冻结。损失为BCE。
2. `bottleneck_adapter`：在同一冻结特征后增加每task独立bottleneck=32的零初始化残差Adapter，
   scale0.1、ReLU、independent；Linear投影用BCE，Adapter参数用ASL 9.8/0/0.05。task结束后投影和
   Adapter均冻结。

两组均为seed0、EMOTIC完整8-task validation-only，30 epochs/task、batch64、Adam、head LR
0.0125、Adapter LR4e-4、per-task cosine min LR=0、无warmup、AMP/TF32开启；保存逐样本实际概率，
不保存checkpoint，不访问test。

## 固定比较与决策

复用相同seed0 Full、Person和旧CLIP Face validation scores。可靠Face固定使用
`0.64 Full + 0.16 Person + 0.20 Face`；无效Face精确回退`0.80 Full + 0.20 Person`。不搜索beta。

候选必须同时满足：

- 可靠Face子集 final validation mAP相对旧CLIP Face至少`+0.05`；
- 固定R1 final validation mAP相对旧锚点至少`+0.05`。

只有通过双门槛的候选才补seed1/2 validation；本阶段绝不运行test。
