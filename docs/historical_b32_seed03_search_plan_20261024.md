# 历史b32三视图配方：seed0/1/3复核与参数探索

2026-10-24。用户已授权执行，并明确允许参考Test调参。只用GPU0，最多4进程，显存不足自动排队；其他卡不使用。分支codex/historical-b32-seed03-search-20261024。

## 问题与参照

先恢复36b77535cb2bc7f866a7950395ef14d26c1b4058下seed1三视图B32的Test32.68002187429384／39.52916404114066。原Full/Person/Face端点31.2372928150／30.4315303579／25.5216973766。原无validation和checkpoint，本轮保存完整边界并补齐两集合。再回答seed0和seed3是否也相对同条件独立Full获益，并做单因素参数探索。独立强Full seed0 32.5365/39.1445仍为外部强参照，不用本轮弱模型替代。

## 共同配方

EMOTIC全TRAIN，8任务类别5/3/3/3/3/3/3/3，每任务30epoch，B64，共240epoch/13950更新；Adam每任务重置，主LR.0125、Adapter LR.0004、WD0、cosine，AMP初始65536/growth2000、TF32、原全局RNG采样器、workers2、OMP/MKL72。Person margin.15/letterbox，paired输入；Person颜色扰动0/0，Full legacy crop .05–1。

CLIP ViT-B/16冻结；Image-token Adapter瓶颈32/ReLU、索引[1]、scale.03，仅Selector图像证据副本、不写回Frozen Forward。任务Selectors/Prefix-Tuning/Adapter/head三视图共享；每任务完成冻结。关闭MoE/ParaX/Attention/人物位置/联合摘要。固定归一化任务特征融合：可靠Face .64/.16/.20，不可靠.8/.2/0。主参数BCE(fused)+aux×meanBCE(views)，Adapter ASL(fused)+aux×meanASL(views)，基准aux.1，ASL9.8/0/.05，Face辅助仅可靠。单Full同scale和优化器、Adapter-ASL，但aux0、无视图融合。

## 18组完整实验

| 配方 | scale | Adapter LR | aux | seeds |
|---|---:|---:|---:|---|
| THREE_BASE |.03|.0004|.1|0,1,3|
| FULL_BASE |.03|.0004|0|0,1,3|
| THREE_SCALE_001 |.01|.0004|.1|0,3|
| THREE_SCALE_005 |.05|.0004|.1|0,3|
| THREE_LR_0002 |.03|.0002|.1|0,3|
| THREE_LR_0008 |.03|.0008|.1|0,3|
| THREE_AUX_0 |.03|.0004|0|0,3|
| THREE_AUX_02 |.03|.0004|.2|0,3|

每行除声明因素之外保持不变。全18组均完整8任务validation/test，不用短跑mAP筛选；先不自动组合多个最佳超参数。按seed完整报告，不用挑seed替代参照。Test参与参数选择的结果标为探索结果，保存按val/test最终与阶段平均分别排名；不假称独立未触碰的Test验证。

## 短跑和复现守卫

当前入口跑seed1 THREE_BASE/FULL_BASE和seed3 THREE_AUX_0，各2task×2update；不可修改的历史36b7753源码独立跑seed1两基准。逐任务比较所有方法张量、loss/LR/步数、完整预测逐值等价。检查seed3、ASL/BCE梯度与Face屏蔽、边界恢复等于compact、冻结骨干/旧任务不变。每个新模型两次完整旧标准validation重评估，逐值一致才通过。单元测试覆盖全部18组配方与错误配置拒绝。短跑不用于精度筛选。

正式seed1 THREE_BASE结束时核对历史四指标和8阶段原生Test score全字段；不一致则记录mismatch并停止继续发出新作业，保留已完成与进行中的任务记录，先查原因。不能把近似数值称精确复现。

## 入口与路径

入口python -m multi_lane.track_a.run_historical_b32_search；短跑加--smoke，正式加--smoke-root。输入沿../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。旧源码../multi-lane-main-b32-parax-p10-combined，旧产物../multi-lane-main/output/emotic_track_a_b32_parax_p10_test/b32_parax_p10_seed1_test_20260929_01/B32。

输出output/historical_b32_search/<run-id>/seedN/PROFILE，日志logs/historical_b32_search/<run-id>；保存compact、评估前优化器/scheduler/scaler/全RNG恢复状态、完整三路/融合预测、checkpoint SHA与冻结审计、summary.csv、探索排名及completion/failure。评估沿历史samples_intersect_seen_classes，非固定全人群；旧人数与标签对齐，重评估Test必须等于训练阶段Test。原历史权重与产物不覆盖。

## 当前状态

已建立新分支，完成本地实现，待远程单测与短跑；尚未启动正式，尚无新性能结论。远程原主树153行/test-only5行及SHA已复核保持；旧36b7753和上一轮0073a2f工作树clean。代码仅Git同步，实验产物以后另行同步。
