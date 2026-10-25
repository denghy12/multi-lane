# 在32.6800历史b32底座上分别加入摘要模块

2026-10-25，用户授权GPU0并行。只用seed1，完整8任务validation/test；允许Test参与探索，完整报告全部配置，不按最好Test换seed。不同时叠加Attention和位置。

## 为什么重做

历史seed1三视图b32-ASL已精确复现32.6800218743/39.5291640411。本轮保持该底座，判断两个原模块在此配方上的边际作用，并各补一个针对适配强度/AMP数值的受限版本。不是沿用此前scale.1、aux0、独立采样器的弱底座。

## 五组

| 组别 | 模块 | 适配 |
|---|---|---|
| BASE | 无新增摘要模块 | 重跑并精确核验历史八阶段预测 |
| ATTENTION | 第6层任务Selector Query读取Full摘要，写入Person摘要 | 原版，inner64/4heads，scale.1，196608参数/任务 |
| ATTENTION_BOUND | 同位置、同参数Attention | 平滑限制增量RMS不超过Person摘要RMS的.1（浮点舍入容差内） |
| POSITION | Full各层Summarize logits加入目标人物相对位置偏置 | 原6维几何，60参数/任务，跨层共用 |
| POSITION_CALIBRATED | 同位置、同参数位置摘要 | 几何偏置乘当前patch内容logits标准差，detach后截断[.25,4]；FP32相加及softmax后转回原dtype |

Attention受限版本delta'=delta/sqrt(1+mean(delta²)/(.1²×mean(stopgrad(Person)²)))；零增量处导数为1，不截断启用新模块的首步梯度。位置校准只缩放原几何偏置，零参数不更改内容温度，CLS偏置仍0，无效人物位置仍屏蔽。两者不增加参数，不写回Frozen Forward，不预设必然有效。

## 锁定的底座

EMOTIC全TRAIN，seed1，8任务5/3/3/3/3/3/3/3，每任务30epoch，B64，13950次更新；CLIP ViT-B/16冻结，Image-token Adapter index[1]、b32/ReLU/scale.03。主参数LR.0125、Adapter和新增模块LR.0004，Adam每任务重置、WD0、cosine。Person margin.15/letterbox，颜色扰动0/0，Full legacy crop .05–1，paired输入。workers2，OMP/MKL72，旧全局RNG采样器，AMP65536/growth2000、TF32。

主参数和新增模块使用BCE(fused)+.1 meanBCE(views)；仅b32 Adapter使用ASL(fused)+.1 meanASL(views)，ASL9.8/0/.05，Face辅助只用可靠Face。三路Task Forward及固定归一化特征融合保留：可靠Face .64/.16/.20，否则.8/.2/0。MoE/ParaX/联合摘要/Selector Conditioner关闭。新profile不会启用旧strong_full_asl/module_stacking训练标志，以免改动采样器和AMP；模型内部共享原摘要执行路径。

新增模块每任务结束冻结，检查旧任务参数不变。训练前完整恢复状态与compact保存，在每任务评估之前写出，保留优化器/scheduler/scaler/全RNG。正式BASE必须四指标及8阶段score精确匹配32.6800存档，否则停止继续派发新作业查因，不把近似值当复现。

## 验证与调度

先CPU测试：五组配置、初始化RNG/零输出、真实train_task的BCE/ASL辅助梯度路由、RMS界和零点梯度、位置校准有效梯度。运行完整回归。GPU smoke先单独跑真实CLIP/AMP初始化审计，所有组零输出与原旧路径相同；再五组2task×2update，旧36b7753源码独立native BASE短跑，核对全部方法张量/loss/预测等价。完成两次val重放、旧参数与新增模块冻结、恢复等于compact、同CPU采样RNG检查后才正式启动。

正式最多5个GPU0作业，评估最多4个，显存空闲6000MiB/共享内存1024MiB才放行，启动间隔12秒，不足自动排队。每组训练完成自动完整8阶段val/test；四组新增模块另做task0/2/7 validation关闭B或C归因，仍保留共同训练的Selectors/Prompt/head，不当作独立重训基线。

## 入口、输入与产物

python -m multi_lane.track_a.run_historical_b32_modules。输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。旧native源码../multi-lane-main-b32-parax-p10-combined，历史score在../multi-lane-main/output/emotic_track_a_b32_parax_p10_test/b32_parax_p10_seed1_test_20260929_01/B32。

输出output/historical_b32_modules/<run-id>，日志logs/historical_b32_modules/<run-id>；包含compact与恢复状态、三路/融合预测、Attention原始及受限残差比、几何偏置与内容标准差/KL、冻结和SHA审计、summary/contrasts/completion/failure。统一旧samples_intersect_seen_classes评估，完整Val/Test，不改固定全人群。

当前新分支codex/historical-b32-summary-modules-seed1-20261025，本地实现完成、语法检查通过；尚未远程测试、smoke或正式启动。原主树153行/test-only5行及SHA保持，历史00fbf90和a30120c工作树clean，旧产物不覆盖。
