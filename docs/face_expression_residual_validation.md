# CLIP Face + 表情预训练残差：seed0 validation 协议

## 唯一研究变量

上一轮用AffectNet特征完全替换CLIP Face后，最佳projection使可靠Face和固定R1 final validation mAP
分别下降0.5276和0.2578。本轮不再单独使用AffectNet，也不搜索Router、融合权重、Face过滤、margin、
增强、rank、scale或学习率；只验证“保留旧CLIP Face，表情特征作为补充残差”。

## 结构

同一目标Face生成两个同步水平翻转的输入：

- CLIP输入严格保留旧Face路径：原检测框margin0.15、pad/letterbox、224、CLIP normalization；
- 表情输入使用五点相似变换、margin0.15头部语境、224、ImageNet normalization。

旧CLIP ViT-B/16 MULTI-LANE Face主路径保持Image-token Adapter layer1/b32/LR4e-4/scale0.1/ReLU/
independent。冻结EmotiEffLib AffectNet EfficientNet-B0输出1280维embedding及其冻结8类logits；二者
分别LayerNorm后拼接。每个task拥有独立`1288→32→512`的ReLU低秩投影，末层零初始化，输出乘固定
0.1后加到对应CLIP Face lane最终表征。新task训练时旧task残差冻结；AffectNet encoder/classifier始终
冻结并保持eval。

优化目标保持主路径BCE、Image-token Adapter ASL9.8/0/0.05；表情残差归入BCE参数组，LR4e-4。
零初始化保证训练开始时输出等于旧CLIP Face，不允许表情分支关闭或替换CLIP语义。

## 实验与门槛

唯一候选为seed0完整8-task validation：30 epochs/task、batch64、Adam、main LR0.0125、两种轻量模块
LR4e-4、per-task cosine min0、无warmup、AMP/TF32开启；训练池仍为`valid && !ambiguous`，保存实际
validation概率，不保存checkpoint、不访问test。

复用旧CLIP Face、Full和Person scores；可靠Face固定R1为`[0.64,0.16,0.20]`，无效Face严格回退
`[0.80,0.20,0]`，不搜索beta。候选必须相对旧CLIP Face同时提高可靠Face与固定R1 final mAP各至少
0.05，才补seed1/2；否则结束Face专用表征路线。
