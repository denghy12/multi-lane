# AF-v1：跨任务共享专家与累计频率冻结

用户2026-10-18明确要求实现论文式任务后Top-2频率冻结，并运行seed0完整8task
validation/test，仅GPU0。该授权开启独立AF实验，不是自动延续此前已停止的v4结构搜索。
分支`codex/moe-activate-freeze-20261018`从`d81b7e2`创建；将前一分析分支未提交的
本会话报告/上下文复制归档，原工作树不动。无ParaX，无文本塔/DDAS迁移，不合并main。

## 机制与论文对应

固定参考官方CVPR版本commit `1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885`：
[Router/硬激活计数](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885/mtil/clip/model.py)、
[任务后统计及冻结](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb99e56b9d7a1d41f0b475a0ac626e6f3885/mtil/src/models/finetune.py)。

以本项目v1接口为对照：视觉块索引1的Task Forward MSA、Drop & Replace及残差相加
之后，令状态为u，`u + FFN(LN2(u)) + 0.1*sum gate*Expert(u)`。Router读raw task
CLS，专家逐token处理整个task-lane状态。不是v4的证据到CLS接口。

- **每层一个跨任务、跨视图共享的22专家池**，预分配8任务×3视图gate/noise矩阵。
  768→64→768 ReLU Adapter、dropout0.1、up零初始化；Top-2选中logits做softmax。
- 训练noisy Top-2，noise std为`softplus(CLS @ noise_matrix)+0.01`；推理关闭noise。
- 每任务累计训练前向实际派发的非零gate硬计数（不是soft权重，不是末尾val重评估），
  Full+Person+可靠Face三路次数相加，**每层总计选2名**，不是每视图各冻结2名。
  排名范围包括已冻结专家，允许重复入选，不补足两名尚未冻结者；同频按expert ID升序。
- Face可靠性排除是多视图扩展：invalid placeholder不应决定共享池的冻结名单。
  记录三视图原始次数及分母，不作各视图频率等权再平均。
- 任务后冻结入选专家的全部down/up权重及bias，冻结名单累积；冻结旧Router/noise。
  后续Router仍访问全部22专家，可选冻结/未冻结的任意组合；仅未冻结专家及当前Router
  进入新任务Adam。冻结专家不放在no_grad里，保留对其输入及Router的梯度。
- 冻结仅针对参数，训练期所有专家dropout仍随model.train启用，与官方weight-freeze
  语义一致；eval关闭dropout。不存在v2/v3强制必经共享专家或额外负载loss。
- shared pool全部专家参数2,180,992，8任务三视图Router/noise811,008，合计
  **2,992,000**。allocation counts将共享池存储只计入task0一次，其他任务仅计Router，
  不把预分配存储错误描述为每任务恒定可训练参数；边界另记实际可训练参数数量。
- checkpoint保存共享权重、全部Router、训练计数、每task Top-2、累计冻结mask和历史。
  eval读取各阶段自己的快照，不拿最终专家池去重放早期阶段。

冻结高频专家不保证旧任务函数完全不变：旧Router也可能调用未冻结专家，它们会被新
任务更新。每任务审计先前冻结专家及旧Router SHA，必须不变；另用16个固定train
样本测量旧lane输出漂移，只记录、不声称严格保护旧预测。该探针隔离RNG，不参与
频率统计，不使用val/test决定冻结。保留MULTI-LANE全部已见lane多标签推理。

## 完整运行配置（运行前已告知用户）

- 数据EMOTIC；train训练，固定全人群val2397人/test5368人，8task大小5/3/3/3/3/3/3/3。
- seed0，8任务顺序训练，各30epochs，train/eval batch64、workers2；预计13950更新。
- 冻结OpenAI CLIP ViT-B/16；CLIP归一化；Full legacy随机裁剪0.05–1，Person
  letterbox/margin0.15，无jitter；Face沿用既有manifest及可靠性mask。
- shared Selector/Prompt/head，Selector10、Prompt10、Prompt层5，pre-head normalization。
  可靠Face固定融合0.64/0.16/0.20；不可靠Face0.80/0.20/0。
- Adam每任务重建；source LR0.05按64/256缩放为0.0125；专家/Router LR0.0004；
  weight decay0；CosineAnnealingLR每任务重建，无warmup、min LR0。
- legacy_full_zero BCE、joint_bce，auxiliary view BCE权重0.1，supervised scale1；
  无蒸馏/专家正则/负载loss。AMP scale1024、growth interval1e9，TF32，无梯度裁剪。
- 输入相对新服务器worktree：数据`../multi-lane-main/datasets/EMOTIC`；预训练
  `../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face
  `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 入口`bash scripts/emotic/run_moe_activate_freeze.sh`；正式run_id
  `formal_seed0_20261018_01`。训练后固定8个checkpoint，val/test两个GPU0进程并行。
- 日志`./logs/moe_activate_freeze/<run_id>/`；输出`./output/moe_activate_freeze/<run_id>/`。
  产物train config/history/paired audit/freeze audit、8个compact checkpoint、两split逐阶段
  mAP/阶段平均、单视图分数、路由统计和退出状态。test不用于修改冻结名单或选择模型。

## 验证与启动门槛

1. 本地Python/shell语法、diff检查和launcher并发/失败处理测试。
2. Git-only在服务器新worktree同步；保留main、test-only及其他历史工作树。
3. 全部Track-A测试，重点覆盖重复冻结不补足、可靠Face计数、冻结/可训练专家共同派发、
   两冻结专家下Router梯度、AdamW不动冻结权重、无未来Router、checkpoint完整重放。
4. `SMOKE_UPDATES=2 RUN_ID=smoke_seed0_20261018_01`：task0/1各2次成功更新，
   真正经过两次冻结边界并复用共享池；源CLIP、旧冻结专家及旧Router SHA不变。
   独立val重放与原阶段mAP逐值一致，检查保存mask与累计计数、配对初始化/采样SHA。
5. 检查GPU0空闲且显存>=12GiB后启动正式作业；不终止其他用户进程。

目前本地实现及launcher测试通过，准备Git同步、服务器全测及真实smoke。未启动正式训练。
