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

## 预检进展

clean1070854服务器全部322项测试通过。GPU0两任务smoke（smoke_seed0_20261018_01）
完成各2成功更新、0 skipped；task0 Top-2为9/19、task1为4/20，累计4名冻结；旧冻结
专家与Router SHA未变。train原阶段与独立重放mAP26.760875/27.958062逐值相同。
新增verify_moe_activate_freeze离线验证器，在正式split推理前核对真实checkpoint权重、
计数/mask/Top-2及与既有fresh B0的初始化/CLIP/训练样本/首批ID/主要超参数一致。

## 正式启动

两任务smoke、全部322项测试和checkpoint直接核验/B0配对校验通过。smoke peak reserved
2826MiB；22份日志/非checkpoint结果已同步并逐项SHA复核，源checkpoint保留服务器。
正式run_id为formal_seed0_20261018_01，tmux ml_moe_af_seed0_20261018，clean运行提交
07fd35cf97a2efc416d56348a1e82a2f00572291；已进入GPU0 task0训练，8任务各30epochs。
运行期间服务器checkout保持该提交；新增本地文档不拉入。正式指标待运行结束，不用smoke
判断效果。训练后先核验全部阶段冻结状态，再GPU0并行full_val/full_test。

## AF-v1与论文/官方MTIL实现对照（2026-10-18）

对齐的是共享专家池、CLS条件Top-2 gated summation及任务后按频率累计冻结，新Router
可复用冻结专家并训练未冻结专家。原v1是任务独立池，新AF-v1已恢复跨任务专家复用。
以官方CVPR固定commit为参照，不混用单Router的CIL设置或MoE-Adapters++。

| 维度 | 论文/官方MTIL | AF-v1 |
|---|---|---|
| Router条件 | task-specific，gate依赖当前CLS | task-and-view-specific，gate依赖各视图task CLS |
| 插入范围 | 图像/文本两编码器各Transformer block | 仅视觉Task Forward索引1块 |
| 专家读取 | 编码器MSA后的特征tokens | MSA、Drop & Replace及残差后的task CLS与恢复Selectors |
| 图像流 | Adapter参与编码器表示更新 | Frozen Forward图像流不写回，残差仅沿Task Forward |
| 冻结统计 | 同层对应模态池的激活次数，选Top-2 | 唯一插入层共享池，Full+Person+可靠Face实际硬激活次数相加，再总选2 |
| 预测 | 图文相似度分类；DDAS选任务/原CLIP分支 | 原线性分类头/多标签BCE，遍历全部已见lane，三视图固定融合 |
| 可训练模块 | Adapter/Router/noise为主要适配参数 | 除未冻结专家/当前Router外，沿用Selector、Prompt、classifier训练 |
| 优化预算 | 官方MTIL的AdamW/迭代预算及交叉熵 | 配对MULTI-LANE Adam、30epochs/task、BCE及0.1辅助视图loss |

专家结构参照官方adapter.py的down→ReLU→dropout→up及up零初始化，不能仅因论文
使用LoRA一词就描述为对Q/K/V权重的直接低秩更新。AF-v1保留v1 raw-CLS和全task-token
专家接口，没有叠加v4的图片证据输入/CLS-only输出。这是冻结迁移，不能自动解决旧版
Router内容变化弱或Selector残差占优的问题；是否发生需新运行结果验证。

冻结名单是旧集合与当前Top-2的并集，允许重复入选、不凑够两名新冻结专家。三视图
原始计数汇总，Face样本少时票数较少，不是三路归一化频率等权；这是明确的迁移选择。
两名选中专家均冻结时，MoE内部仍可训练Router，但Selector/Prompt/head也仍更新，
不能称整个模型只训练Router；冻结专家保持输入梯度和训练期dropout语义。

AF-v1相对旧v1同时改变了专家池归属和冻结策略；若后续要分离冻结本身的因果收益，
需要同一跨任务共享池、不冻结的匹配控制，现有差值不能全部归于单一冻结操作。
本次仅文档对照，不改服务器运行配置/代码，也不新增控制组训练。
