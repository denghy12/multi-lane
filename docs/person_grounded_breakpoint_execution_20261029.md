# 人物条件Summarize三个断点：只读执行协议

日期2026-10-29。用户明确授权开始；新分支codex/person-grounded-breakpoint-audit-20261029从daaacef建立，保留此前全部dirty分析。只对已有seed1模型做无梯度前向，不重新训练、不换头、不安装其他模型参数、不改变原模型源码。

## 目标及范围

检查“Person条件→主体Query→Summarize分布→摘要→读出hidden→有界delta→分类头W·delta”的变化在哪一步消失；验证自己人物框相对他人框的取证；评估修正是否改善同类别正负排序，重点task6。六组现有模型CONST/CONTENT/GROUND ×POOL/ATTN全部保留；task0/2/6/7的最终30轮compact checkpoint；Validation/Test均完整历史人群。

先标准完整batch顺序前向，逐值匹配原ON/OFF和三视图logits，保存解析FP64分类头投影与历史原始概率。随后原同图分组batch前向，逐值匹配已有native／early_swap／late_query_swap的logits及CPU float32 sigmoid概率。增加记录不改变任何推理函数；诊断重算alpha／subject value和原实现必须逐值相等，trace delta也必须精确重放原delta。

## 固定配置

- EMOTIC、seed1，全部源模型仍为历史b32／scale0.03、Person颜色扰动off、原辅助BCE与Adapter-ASL的30epoch训练终点；本轮epoch、LR、optimizer、scheduler均不适用，更新次数0。
- 原CLIP ViT-B/16权重与训练配置SHA核对；原模型build初始化72线程，加载全部自己的compact参数；推理4线程、每进程1个spawn DataLoader worker／prefetch1／batch64。CPU线程调整仅作资源控制，必须由历史逐值重放证明预测不变；不放宽匹配容差。
- 仅GPU0，六个arm诊断作业最多同时6个；GPU启动余量>=3500MiB、/dev/shm余量>=512MiB、12秒错开，自动排队；其他卡不动。
- 新诊断入口multi_lane/track_a/person_grounded_breakpoints.py；调度scripts/emotic/run_person_grounded_breakpoints.py；相关测试tests/test_person_grounded_breakpoints.py。
- 使用新独立远程worktree multi-lane-main-person-grounded-breakpoint-audit-20261029，经Git-only同步。运行前源码clean，训练source commit4349449与新HEAD的全部原multi_lane文件必须无差异；只新增观察脚本，不修改原方法。
- 原runtime/source/test-only worktree及所有pth保留。每case前后method参数digest及源checkpoint SHA检查；每arm冻结视觉权重digest前后检查。所有前向torch.no_grad，模型requires_grad_(False)。

## 输入与输出

相对新Linux实验worktree：

- 源训练：../multi-lane-main-person-grounded-summary-20261029/output/person_grounded_summary/seed1_20261029_01/<arm>/train。
- 源完整评分与配对评分：同arm的val／test，仅作逐值核验，不能提供预测给模型。
- 数据与CLIP及Face manifest均由源config读取，仍分别../multi-lane-main/datasets/EMOTIC、../CODE_DDP/pretrained/clip/ViT-B-16.pt、../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
- 正式结果output/person_grounded_breakpoints/seed1_20261029_01/<arm>/<split>/task{0,2,6,7}/。
- 正式日志logs/person_grounded_breakpoints/seed1_20261029_01/。
- 预检输出/日志output/person_grounded_breakpoints/preflight_20261029与logs/person_grounded_breakpoints/preflight_20261029；只CONST_ATTN和GROUND_ATTN的task0 Val首个原始full和paired完整batch，不截断batch，不依据精度筛组。

## 保存的证据

full.npz：完整历史人物、标签、原ON/OFF logits与原概率、FP64 base及W·delta、Face可靠性、实际变换框/可见性、修正范数。FP64只解析线性头，不替换历史正式评分。原分类头在h0之后是线性的；预head归一化已发生在原三路特征生成中，此处不能误加一次归一化。

pairs.npz：全部同图人物对的native／early_swap／zero_condition／no_subject／no_face／late_query_swap输出；每lane每人的Person/LN条件、preactivation、tanh条件、低秩shift、Query、alpha、摘要、LN摘要、Value、lateQuery、hidden、raw和delta的RMS/变化RMS/相对差/余弦；alpha总变差、tanh导数与|pre|>2／>3比例。常量组除了原Person输入差异，所有早期/最终张量都应逐值不变。

空间记录：双方Full crop像素必须逐值相同才可共用坐标；保存自己/供体框及各自面积、IoU、独占区域面积、取证质量与面积归一化密度，分别记录native和早期换人。binding gain=(native自己−他人取证)−(换人自己−他人取证)，不把大框自然有更多质量当人物对应。分层锁定可见性、框大小/面积比、重叠、当前已见标签是否有差异；不使用标签挑donor。

tensor_examples.npz：首个原配对batch中按既定ID顺序取前16名有效recipient，保存各lane上述原始中间张量及native／换人／零条件，供复核；不按预测结果挑样本。其他人物保存逐样本统计，避免数十GB重复tokens。

每case result.json保存checkpoint、人数、时间与不变性；每arm result.json保存完整8个诊断case完成情况；队列最后completion.json或failure.json。完成后回传非pth产物并独立CPU复算，三上下文更新。不把任何诊断干预当作新训练方法的正式结果。

## 验证及入口

先7项诊断机制测试与既有人物Summarize相关测试，真实CUDA AMP不得跳过；再两组真实task0 Val单batch smoke。通过后：

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 \
      /opt/conda/envs/ddp/bin/python -u scripts/emotic/run_person_grounded_breakpoints.py \
      --source ../multi-lane-main-person-grounded-summary-20261029/output/person_grounded_summary/seed1_20261029_01 \
      --output output/person_grounded_breakpoints/seed1_20261029_01 \
      --logs logs/person_grounded_breakpoints/seed1_20261029_01 --maximum-jobs 6

创建协议时仅本地静态编译/diff检查通过，远程测试/smoke/正式尚未执行，不能提前报告断点。后续真实执行记录追加。

## 实际预检通过与自动CPU分析

观察入口2622b73，22项相关测试通过且无skip；CONST_ATTN/GROUND_ATTN各task0 Val原full首64、paired首batch62名有效供体通过，native/early/late logits与CPU sigmoid逐值一致，traced alpha/summary/delta与原实现精确，所有参数/源checkpoint未变。每case约10.4/10.7秒（仅单batch，不是完整时间预估）。新增纯NumPy排序与区域控制4项本地测试通过；只补协议class_order/task_sizes和独立分析后ff同步，原观察计算不变。

正式队列48case完成后自动调用scripts/emotic/analyze_person_grounded_breakpoints.py生成analysis目录，覆盖full mAP/逐类AP/AUC、解析Wdelta及相邻正负排序、paired各断点/区域分层、500次原图聚类均值区间。FP64为解释工具，历史CPU32概率仍为正式评分。队列分别保存forward completion与analysis completion，失败写独立failure而不冒充成功。正式启动后追加实际PID和健康状态。

## 严格回放中止与小尾batch覆盖修正

seed1_20261029_01六组均已退出、无活进程，完整task0 Val原ON/OFF/三路2285人已精确；配对最后4人batch的原native（在trace前）未精确。独立同input/同模型72→4→72重复前向也变化，max差.00390625；不放宽匹配阈值，不将其算作模型机制。01失败日志/文件保留。

02固定只对原batch至少32人做新张量前向，不更改任何batch组成或排序。官方full NPZ仍包含全部源人物/标签/历史概率，尾人projection_available=False、Wdelta NaN，保存excluded ID；投影与相邻负例统计只在实际trace人群上，正式AP/AUC仍来自全历史人群。paired NPZ只含严格重放的原大batch，人群/条件完全配对，另报原配对总数与小尾名单。此处不是更换历史评估标准，而是明确只读张量证据覆盖限制。原先“全部完整真实前向”的表述由本节修正。正式新ID seed1_20261029_02，须23相关+4纯NumPy与两个Attention task0 Val全量大batch预检后再启动。
