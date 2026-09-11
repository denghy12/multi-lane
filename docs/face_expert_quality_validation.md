# 独立Face expert质量提升：seed0 validation协议

## 目标

锁定现有Full、Person专家与R1融合，不再搜索融合器或Face权重。只改变独立Face expert的训练样本
质量、Face在224输入中的有效分辨率和轻量增强，要求Face自身与固定R1融合同时改善。

## 固定部分

- EMOTIC、seed0、完整8-task validation-only，禁止test。
- Full、Person直接复用既有scores，不重训。
- R1固定：可靠Face样本权重`[Full, Person, Face]=[0.64,0.16,0.20]`；其他样本严格
  `[0.80,0.20,0]`。
- Face模型保持Image-token layer1、b32、Adapter LR4e-4、scale0.1、ReLU、independent，主模型
  BCE、Adapter ASL9.8/0/0.05，30 epochs/task、batch64、main LR0.0125、per-task cosine min0、
  AMP/TF32 on、无checkpoint。
- Face crop均使用原检测框重新计算margin，完整pad/letterbox后Resize224，无随机裁剪。

## 三个候选

既有`baseline_m15`作为冻结锚点：`valid&&!ambiguous`训练、margin0.15、flip。

1. `reliable_m15`：训练集再要求原始Face短边≥24px、检测分≥0.6，使训练分布与R1启用Face的
   可靠性条件完全一致；margin0.15、flip。
2. `reliable_m05`：保持可靠性过滤，将margin从0.15缩到0.05，使脸部在224输入中占据更多像素，
   用来检验有效Face分辨率是否是瓶颈。
3. `reliable_m05_jitter`：在候选2上增加强度0.1、概率0.5的轻量ColorJitter，检验小样本Face
   expert是否受外观过拟合限制。

三组在GPU0/1/2并行。最终只读取每组实际validation概率，以固定beta0.20重建R1，不搜索静态比例。

## 选择规则

候选必须相对冻结锚点同时满足：可靠Face子集final mAP至少`+0.05`，固定R1全量final validation
mAP至少`+0.05`。通过者按固定R1 final mAP、可靠Face final mAP排序。只有赢家通过两项门槛，才补
seed1/2 validation；本阶段永不运行test。未通过则说明问题不在这三项基础Face输入质量设置，停止
继续密集调margin/增强。
