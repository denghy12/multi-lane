# 三视图联合摘要与单Task Forward实验（2026-10-20）

## 为什么做
此前MoE逐视图适配，Expert/Router并未共同观察另外两路信息；只读对照尚未显示内容路由净收益。
本轮检验不同假设：同一任务的Selectors逐层从三视图提取摘要，先联合证据，再在唯一任务路径内
形成任务表示。不是把旧MoE换个插入层，也不再沿用跨任务可变专家池。

## 预先确定的五组比较

|组|融合位置/方式|每任务注意力路径|辅助视图BCE|本次训练|
|---|---|---:|---:|---|
|原B0|三路最终特征固定融合|3|0.1|复用历史seed0结果|
|baseline_noaux|同B0，只取消辅助监督|3|0|是|
|fixed|每层三份Selector摘要，按原可靠性权重固定混合|1|0|是|
|dense|固定联合摘要 + 普通联合输入MLP残差|1|0|是|
|moe|固定联合摘要 + 联合输入稀疏融合专家残差|1|0|是|

先比较baseline_noaux/B0排除监督改变，fixed/baseline_noaux识别路径合并的作用，dense/fixed识别
学习交互的作用，moe/dense识别专家选择的作用。所有组事前锁定，不使用test筛选或调参。

## 数据流与保护
每个视图使用同一个冻结CLIP ViT-B/16的权重，各有独立图像token流，跨任务复用；不写入Adapter。
每层沿用MULTI-LANE从该层输入图像tokens读取的时序。归一化的同一组任务Selectors读取三视图，
分别得到10份摘要；相同Selector索引代表同一任务查询，不等于裁剪后patch几何位置对齐。
三路摘要融合成10份联合摘要，与单个任务CLS组成11tokens，执行原prefix attention、Drop & Replace、FFN。
下一层沿用同一任务状态，再读取三路图像证据。CLS最后投影、pre-head归一化，进入原位置分类头。
推理运行全部已见任务路径并按类归属拼接logits，不挑“效果最好”的Router，不用DDAS。
无独立三路最终CLS，因此没有原分支mAP/辅助BCE，评估不伪造分支指标。

Face可靠性取已有manifest的face_reliable；在专家/Router输入归一化前屏蔽无效Face，不把condition_valid
（Full裁剪条件有效性）误当Person有效性。可靠时先验(.64,.16,.20)，不可靠(.8,.2,0)。
额外joint_view_valid仅供机制测试/未来只读输入干预，正式训练不启用。
所有融合器、Router、Expert按任务和层独立保存，只激活当前任务，旧任务整套冻结；没有跨任务专家复用。
Selectors/Prompts沿用原任务复制规则，分类头沿用当前类梯度屏蔽及每任务重置Adam。
独立Image-token Adapter、旧Task/Frozen MoE、ParaX、Selector conditioner均关闭。

## 可学习模块（12层均启用）
输入每份摘要的三视图非仿射LayerNorm拼接+3位有效性，共2307维。
输出均为固定联合摘要 + 0.1×可学习残差，up投影初始0，因此初始dense/moe等于fixed。
dense为2307→32→768 GELU MLP；moe为4名2307→16→768 GELU专家，soft Top-2权重归一化，
训练/推理一致，无噪声、无直通估计、无dropout或额外负载损失。Router为2307→16→4，
输入是三路归一化摘要在10个Selector上的均值及有效性；同图片每层一个任务条件gate。
专家均看到三路联合证据，不预定义“表情专家”等语义。实际仅分派被Top-2选中的图片给专家。
本轮先比较MoE能否优于普通融合，尚不能把总参数容量与稀疏选择完全分开：

|组|每层参数|每任务12层参数|8任务新增参数|
|---|---:|---:|---:|
|fixed|0|0|0|
|dense（宽32）|99,200|1,190,400|9,523,200|
|moe（4×宽16+Router）|236,932|2,843,184|22,745,472|

Dense与MoE只匹配活跃瓶颈总宽32；MoE还有Router计算和额外up bias，不声称参数或FLOPs精确一致。
日志保存逐任务逐层base/residual范数、soft权重累计、gate跨图片标准差/熵、可靠Face分组权重、hard负载及旧融合参数SHA保护。
硬负载按实际稀疏dispatch统计，即便选中权重下溢为0也计数，和soft gate mass明确分开。

## 完整运行配置（smoke与正式）
数据EMOTIC，task sizes=5/3/3/3/3/3/3/3，seed0；训练只用TRAIN。
正式8任务，每任务30epoch，batch64，Adam每任务重置，主LR=.05×64/256=.0125，融合LR=.0004，
weight_decay=0，cosine降到0，无warmup；legacy_full_zero BCE，temperature1，supervised scale1，aux0。
10 Selectors、10 Prompts、前5层prefix prompt、pre-head normalize，AMP initial1024/growth1e9，TF32，clip0。
输入224，CLIP归一化；Full legacy random crop(.05,1)，Person letterbox margin.15，无颜色扰动；Face沿用manifest。
训练workers2，所有评估workers0；避免4GiB共享内存并发预取故障。
用户已确认GPU5/6：GPU5依次baseline_noaux、dense；GPU6依次fixed、moe。
每组训练完成、验证8份checkpoint及恢复状态后，在该GPU并行完整validation与test。
固定全人群validation2397/test5368，每一阶段checkpoint在该split的全人群上评估当前已见类，保存8阶段分数、
最终mAP和阶段平均，test不参与选择。原B0复用val42.1912/avg48.4382、test32.0610/avg36.2781。

训练入口bash scripts/emotic/run_joint_view_summary_batch.sh；单组ARM=... GPU=... bash scripts/emotic/run_joint_view_summary.sh。
smoke使用SMOKE_UPDATES=2，四组各2任务×2成功更新，验证集checkpoint重放+完整val，不接触test。
正式RUN_ID=formal_seed0_20261020_01，smoke另用smoke_seed0_20261020_01；每次必须新目录。
训练前CLIP来自../CODE_DDP/pretrained/clip/ViT-B-16.pt，不从旧MoE checkpoint续训。
数据../multi-lane-main/datasets/EMOTIC；Face ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
历史B0路径../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_baseline_seed0_20261017_01/train。
日志./logs/joint_view_summary/<run>/<arm>/；产物./output/joint_view_summary/<run>/<arm>/。
预期每组：config/history/paired审计，8个compact checkpoint，8个在评估前原子写入的完整恢复状态
（模型/优化器/调度器/GradScaler/随机数/sampler），8阶段完整val/test分数与summary、机制与恢复核验。
旧工作树所有改动保留；仅新分支codex/joint-view-summary-20261020，Git-only同步独立服务器worktree。

## 成功判断与限制
以同监督baseline_noaux为结构对照，并报告相对原B0差异；最终与阶段平均共同观察，不能只挑某一阶段。
单seed结果不能宣称稳定收益。MoE不优于dense时，不能把提升归因动态专家选择；也可能只需普通联合融合。
Face可靠性是已有二值信号，未引入新的质量模型；参数随任务线性增长，三路冻结骨干和摘要读取仍保留，
不能声称整网提速3倍。完整8task是用户本次明确要求，不以历史只读对照的停止建议否定此次新假设。

## 执行状态
373项全测通过（16.580s），其中新增6项机制测试。7b20c70四组真实smoke已在GPU5/6启动，前两组训练完成，机制审计通过、val重放中。
本地追加门控统计/极端权重计数测试及并发validation检查，待后续测试；正式尚未启动。

smoke01：fixed来源/机制/重放通过；MoE在首批AMP累加处发现Float/Half错误，修复dtype转换并加混合精度测试。失败产物保留，正式未启动。新版需smoke02全部通过。
