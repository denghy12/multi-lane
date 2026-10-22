# 保留人物摘要的独立上下文Attention：方案与GPU0排队协议

日期：2026-10-23。新分支`exp/summary-context-attention-20261023`，起点`e81f2f2`。用户已要求修改并运行seed0完整8任务validation/test，GPU0排队；不修改另一个分支正在进行的位置/Attention/Adapter叠加实验。

## 为什么这样改

上一轮Full→Person摘要残差可达到原摘要的0.262–1.407倍；同checkpoint关闭人物Query残差后，Person及融合恢复，说明直接写入没有保护人物证据。任务Selector Query的模型间收益也在关闭残差后保留，尚不能归功于推理交互。此次假设是：让原Person Summarized Patches保持完整，把Full上下文作为独立K/V组供原任务MSA读取，比直接生成特征并加到Person摘要更容易保留其互补作用。它仍是待验证方案，不能保证提升。

## 数据流

```text
Full Frozen Forward → 任务Patch Selectors → Full Summarized Patches ──→ source
                                                                            ↓
任务Patch Selectors（控制）／Person Summarized Patches（人物内容） → Q/K读取权重
                                                                            ↓
                                                      原Full Value的加权汇聚
                                                                            ↓
Person Frozen Forward → 同任务Selectors → 原Person摘要 ──→ 原任务MSA ← 独立上下文K/V
                                                               ↓
                                             原Drop & Replace → FFN → Person CLS
Full/Person/Face CLS → 原固定特征融合 → 原分类头
```

上下文读取采用4个Q/K注意力头，平均权重后汇聚原Full摘要；不学习V或输出投影。原任务MSA保留原CLS/摘要/Prefix K/V组，另对上下文K/V归一化读取。两组概率以`g=0.1*sigmoid(a)`混合，`a`按任务学习、初始化0，因此上下文总注意力份额初始5%、最多10%，原组至少90%。这等价于具有显式组概率的联合读取；不是给零向量后仍让其无约束占据softmax。此上限约束概率份额，不承诺最终CLS或logits范数变化最多10%。

原Person摘要不会被替换或相加。Drop & Replace仍传播原CLS更新与原Selectors，不延续上下文tokens。无几何位置、MoE、ParaX、image-token Adapter、联合摘要单路径或额外分类头。Frozen Forward不改写。每任务独立Q/K与gate，任务结束后封存；测试运行全部已见任务路径并拼接类别logits。

新增参数：每任务`2*768*64+1=98,305`，8任务786,440；旧残差对照每任务196,608、8任务1,572,864。两者容量不同，需要报告；本轮研究的是改变写入接口，不把结果单独归因为一个gate或参数量。新版初始为近似原路径，**并非严格零初始化等价**；运行时关闭上下文则严格走原公式。

## 四组完整实验

| 组别 | 读取Query | 如何使用Full信息 | 目的 |
| --- | --- | --- | --- |
| baseline | 原Selectors | 原三视图方法 | 同监督同预算控制 |
| legacy_task_residual | 任务Selectors | 旧Q/K/V/输出投影残差加到Person摘要 | 与新接口比较，全部重新训练 |
| task_context | 任务Selectors | 独立上下文K/V，原Person摘要保留 | 判断接口变化是否有益 |
| person_context | Person摘要 | 独立上下文K/V，原Person摘要保留 | 判断人物内容Query是否额外有益 |

所有组均使用辅助视图BCE0.1，不能与上一轮aux0绝对结果作为纯结构因果比较。四组不包含位置叠加。新版同时改变Value形式与接收接口，四组不能完全拆分这两个因素；若有正向证据，再考虑必要消融。

## 完整配置与输入输出

- EMOTIC，seed0；任务5/3/3/3/3/3/3/3，累计26类。全部TRAIN，无90/10留出。本轮锁定配置完整跑8任务，不按test或中途validation调参。
- CLIP ViT-B/16，10个Selectors、10个Prompts、原5层Prefix。三视图共享当前任务Selectors/Prompt/head，任务间整体保护。第6层（索引5）Summarize之后、任务MSA中读取上下文。
- 每任务30epoch，B64，Adam每任务重置、WD0；主LR0.0125，模块LR0.0004，余弦至0，无warmup；预期13950成功更新/组，0跳步。AMP初始1024/growth1e9、TF32；train/eval workers0，避免共享内存预取故障。
- 融合BCE +0.1倍可用视图辅助BCE，legacy_full_zero/joint路由。Full legacy裁剪0.05–1；Person margin0.15/letterbox、无color jitter；可靠Face原manifest；可靠时融合0.64/0.16/0.20，否则0.8/0.2/0。
- 数据`../multi-lane-main/datasets/EMOTIC`；预训练`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；Face `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 入口`bash scripts/emotic/run_summary_context_attention.sh`；所有GPU子进程固定`CUDA_VISIBLE_DEVICES=0`，同阶段最多2进程，每进程allocator上限30%，新进程要求空闲≥8500MiB。
- 正式ID `formal_seed0_20261023_01`；`output/summary_context_attention/<ID>/`存协议、checkpoint、恢复快照、8阶段概率/分支结果；对应日志`logs/summary_context_attention/<ID>/`。队列状态位于`output/summary_context_attention/<ID>_queue/status.json`。

## 排队与验证

先等待另一个工作树`../multi-lane-main-three-module-stacking-20261022/output/three_module_stacking/formal_seed0_20261023_01/completion.json`：必须正式8任务complete，不能把smoke结束当完整结束；前序failure则停止本队列并保留原因。之后GPU0连续3次20秒检查无计算进程、利用率≤5%、空闲≥22000MiB才启动，不暂停其他进程。

CPU全测记录必须与排队HEAD一致。轮到后先自动四组各2任务/每任务2更新GPU smoke：核验AMP、梯度、任务保护、compact与完整恢复模型相同、validation重放；只用validation，不访问test。全部通过才启动四组正式8任务。训练完成先审计，再并行validation/test；保存每任务报告之前的完整恢复快照。任一作业失败停止发新作业，已启动作业完成后退出，不覆盖结果。

正式validation额外对三种Attention模型task0/2/7进行同checkpoint开关归因；person_context同时做固定规则同图Query交换，核验Full/Face逐值不变、不可交换者不变、本人Query重放一致。仅test正常推理，不以test选择开关。

成功判断首先看匹配基线和旧残差的validation最终/阶段平均，以及开启对关闭的融合净收益；Person端点只作证据，不以单路分数替代融合指标。若人物Query优于任务Query，还需本人条件相对交换条件有可重复优势。若仍只有关闭后保留的模型间收益，结论转为训练过程影响，不能宣称上下文推理有效。所有结论仍限seed0。
