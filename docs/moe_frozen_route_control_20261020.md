# Frozen Forward MoE：动态／TRAIN固定Top-3／关闭残差只读验证

2026-10-20用户授权执行。只使用GPU0，不训练新模型，不读取test。

## 前提及目的

源分支codex/moe-frozen-forward-20261019，运行b63616b，正式seed0已完成8task×30epoch、13950更新/0跳步及完整val/test。最终val42.0412/阶段均值48.3758，test31.6928/阶段均值35.4285，未超过B0。新位置已改善路由分配，但尚无内容选择的预测净收益证据。

本次分别回答：动态选择是否优于匹配的固定组合；同一checkpoint的MoE残差是否对预测有净贡献。关闭残差后Selector/Prompt/head仍是与MoE共同训练的参数，不能视作重新训练的B0。

## 固定配置

- EMOTIC确定性三视图、CLIP ViT-B/16原权重；seed0现有task0/2/7 compact，各自评估全部已见任务路径/类别，不使用测试任务标签挑路径。
- 源MoE第2视觉块[1]，22共享专家、Top3、768→64→768、residual0.1、累计训练权重冻3；本次不改任何参数或训练设置。模型eval、no_grad、无dropout/噪声、原AMP/TF32。
- A dynamic：原图片相关Top3及其权重。B fixed：每task/view仍调用3名专家，使用TRAIN校准锁定的固定IDs和归一化权重。C off：只关闭MoE残差，保留同模型任务路径、Selector/Prompt/head。
- 每个源checkpoint用4个固定TRAIN batch校准，每批人物上限64；按原图分组、不拆同图人物，实际人数及像素SHA记录。仅从已有类有正例的人物中取样。seed=20261020+checkpoint task。对每条已见lane/每视图累计实际无噪gate的CPU float64权重，Face只计可靠样本；排名相同时小Expert ID优先，取Top3再归一化。
- 三个checkpoint的全部固定表在加载validation图像前锁定；不按validation/test选择IDs、权重或路由配置。
- 完整val2397人/1705原图，batch64、workers0；分别报告融合、Full、Person、可靠Face及相对Full的融合收益，保存ID/targets/logits/probabilities。
- 融合mAP差做500次原图聚类配对bootstrap，seed=20261020+task；反映样本不确定性，不含训练seed差异，不以test判定。
- 动态条件必须与源full_val对应保存分数逐值重放一致；每条件前后模型state SHA不变、所有grad=None、CLIP SHA及源checkpoint SHA不变。
- 源冻结预算等元数据照常恢复，不调用optimizer/backward/finish_task，不改写源结果。

## Image Token Adapter启用情况

源adapter_mode=moe_frozen_ffn_activate_freeze，独立image_token模式未启用。MoE本身处理图像CLS和patch tokens；不能因此误称“图像token没有适配”。目前模型用单个adapter_bank按adapter_mode择一创建，独立Image Token Adapter与MoE不是两个可同时打开的开关。叠加需要另作实现和配对消融；优先在MoE单独显示净收益后再研究。本次三组均不叠加，ParaX关闭。

## 入口、路径和执行顺序

独立分支codex/moe-frozen-route-control-20261020，从b63616b创建。扩展既有late_route_control的源配置白名单并保留旧Late Top2行为；读取构造器恢复moe_freeze_count，避免将Top3 checkpoint错误重建成旧Top2冻结schema。核心前向/训练实现不变。

入口：`bash scripts/emotic/run_moe_frozen_route_control.sh`。

源run：`../multi-lane-main-moe-frozen-forward-20261019/output/moe_frozen_forward/formal_seed0_20261019_01`。
数据：`../multi-lane-main/datasets/EMOTIC`。
CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`。
Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。

正式RUN_ID=formal_seed0_20261020_01；结果`output/moe_frozen_route_control/<RUN_ID>/`，日志`logs/moe_frozen_route_control/<RUN_ID>/`。

先完整Track-A回归及新增关键测试，再真实GPU0 smoke：task0/2/7、每checkpoint两TRAIN批、每条件val前64人、bootstrap20。smoke覆盖9条件、固定表恢复及动态重放；只验证执行正确性，不选择方案。通过后GPU0串行跑正式9条件完整val，bootstrap500。所有源工作树及checkpoint保留。

## 结果解释及继续研究门槛

沿用先前路由诊断的保守门槛：dynamic−fixed三个阶段点估计均>0，且至少一个原图配对95%区间下界>0，才有继续研究内容路由的证据；未通过不等于证明真实差异为0。对dynamic−off另列差/区间，区间跨0不能当作证明等效。

有路由收益仍需整体超过B0后才谈采用；没有净收益不继续只追求路由多样性、增专家数或调冻结数。本次不自动追加新训练、不扩seed、不合并main。预期产物包括protocol、TRAIN校准IDs/像素SHA/锁表、9组分数和视图诊断、配对区间及前后完整性检查。
