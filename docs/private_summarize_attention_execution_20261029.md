# 并行交互专用Summarize：四组完整8任务执行协议

2026-10-29用户授权按详细方案实现并执行；新分支codex/private-summarize-attention-20261029，起点0e366d9。保留既有dirty分析，Git-only同步独立服务器worktree，旧main/test-only不改。

## 四组与共同条件

| 内部名 | 查询 | 读出 |
|---|---|---|
| SUMMARY_FIXED_POOL | 原第6层查询sg副本，U固定0 | 同摘要池化MLP |
| SUMMARY_FIXED_ATTN | 同上 | 联合摘要Attention |
| SUMMARY_LEARN_POOL | sg查询+可学习U，模长匹配 | 同摘要池化MLP |
| SUMMARY_LEARN_ATTN | 同上 | 联合摘要Attention |

原三路Task Forward、Selectors/Prompts、Image-token Adapter、head完整保留。新增分支从原index5 block.ln_1图像tokens和原归一化Selectors的detach缓存取证，四组统一FP32 Summarize重算。U每任务10×768、三视图共享，仅learn两组为参数；固定两组是零buffer，configure_current不会意外开启。新摘要对U保留梯度。U零初始化，从原查询方向开始，保持原模长，零范数安全回退。

共同Query(h0,Person CLS)1024→64、Value768→64、Output64→512和视图类型编码先构造、按929100+task独立种子初始化，随后构造Key或池化MLP，避免不同算子打乱共同参数。Attention64维4头/单层/零Null K-V；池化每视图Value均值，与共同Query及可靠Face标记拼接257→152→64 GELU。两者输出层零初始化。新增参数/任务固定Pool198768、固定Attention198912、learn Pool206448、learn Attention206592；U每任务7680、8任务61440。

四组最终`delta=Bound_.01(raw;h0)`，无学习gain/区域mask/新loss/专家。Face在softmax前mask，Pool等价置零并给标记。原监督继续更新原参数/分类头；新增真实标签BCE只更新U/读出，h0和head在新目标中detach，独立optimizer/scaler/RNG；没有外部模型参数或教师答案。任务结束封存旧私有查询/读出。

## 配方

EMOTIC，seed1，8任务[5,3,3,3,3,3,3,3]，每任务30epoch/B64；Adam/WD0/cosine，主LR.0125、Adapter/U/读出LR.0004，原BCE/Adapter-ASL分工，aux.1，ASL负gamma9.8/正0/clip.05。CLIP ViT-B/16冻结，原b32/index1/scale.03/ReLU、10共享Selectors/原Prompts、Person颜色扰动关闭；原可靠Face .64/.16/.20融合及无Face fallback不变。原分类头fresh、完整30轮自行学习，不安装B0、不换头、不额外5轮。

72线程初始化/训练、OMP/MKL72、AMP初始65536/TF32、train workers2/prefetch1；历史报告两次CPU draw保持。评估workers2单线程spawn/prefetch1，CPU float32按原batch sigmoid。历史samples_intersect_seen_classes，每阶段仅累计已见类至少一正标签人群，8阶段mAP算术平均；不Test挑epoch、类别或任务。

各组240epoch、13950主更新，主和私有均须零skip；新query第一步可能无梯度，但输出层更新后必须出现query梯度和实际偏移。各任务原参数/30轮关键loss/Python-NumPy-CPU-CUDA RNG与历史强B0精确比对。只读历史checkpoint用于核验，不加载到学生。

## 路径和入口

相对独立服务器实验worktree：

- 配方/历史核验：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- 数据`../multi-lane-main/datasets/EMOTIC`。
- 预训练`../CODE_DDP/pretrained/clip/ViT-B-16.pt`，核验历史SHA。
- Face manifest `../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 输出`output/private_summarize_attention/<run_id>/<arm>/{train,val,test}`，日志`logs/private_summarize_attention/<run_id>`。
- 每epoch/任务验证前保存双优化器/双scheduler/scaler/完整RNG和私有查询恢复状态；最终紧凑checkpoint、跨臂和历史审计、mAP/F1/逐类/可靠Face分数。

```bash
bash scripts/emotic/run_private_summarize_attention_gpu0.sh smoke_seed1_20261029_01 --smoke --smoke-updates 4
bash scripts/emotic/run_private_summarize_attention_gpu0.sh seed1_20261029_01
```

只有GPU0（初始free约24GB），最多4作业、2个非训练GPU作业；启动余量GPU6GB/shm1GB、间隔12秒，不中断其他作业，不使用GPU1–7。四组先新机制测试/完整Track-A和相关回归，再两任务×4更新真实smoke（只首64Val，无Test），验证原参数/loss/RNG/OFF/三路一致、U可导可更新、参数量、初始恒等、无效Face/1%cap、双优化器恢复后启动完整八任务；不按早期精度截断。

## 自动TRAIN标定与完整归因

每组训练结束先排TRAIN-only calibration，再允许该组Val/Test读取。每任务从当前类正样本TRAIN人群，以原图分组/固定seed929650+t选4个至多64人batch（smoke1），使用确定性eval增强。仅对本任务checkpoint的本任务delta取FP64均值方向；保存所有sample IDs、源checkpoint/config SHA及train-only声明。均值与比例不从Val/Test选择，无优化器、不写权重。标定文件`train/summary_calibration/result.json`，两split使用同一SHA。

完整Val/Test全8阶段同时计算ON/OFF、U归0、仅Face memory屏蔽、仅Person memory屏蔽；两种memory屏蔽均严格保持自己原h0/Person Query/head。原三路原生分数及所有人物/标签/mask/batch边界完整保存。固定U组的zero_query应逐值等于ON，无可靠Face人物的Face屏蔽应逐值不变。

task0/2/6/7额外比较固定TRAIN均值方向和可靠性/Full可见性分组内错配方向，都保持接收者native delta范数；方向为零时回退接收者原方向，避免额外幅度变化。错配是诊断，不是部署利用其他Test人物的方案。保存实际RMS、原始logits/概率和query norm/cosine/offset/summary variance记录。

task0/2/6/7另用同原图多人分组batch做真实前向，图像组不跨batch。分别只换Person Query、只换新增Face memory、只换新增Person memory；保留接收者h0/head，其余memory/Query。Person供体要求双方Full可见，Face供体还需双方可靠；同图循环选择，无候选者不换，保存eligible mask/donor IDs。配对重新前向的native作为各自同人群基准，不混用不同batch的正式分数。只读干预不计为重训组或组装模型。

重点报告task6 Sadness/Sensitivity/Suffering排序与逐类AP/BCE；原Full/Person及旧路径必须保持。新查询有梯度、热图改变或TRAIN BCE下降不等于净收益。只有learn Attention同时优于原摘要Attention、同输入learn Pool及B0/CLS-MLP两项Test才推进；只Pool获益则围绕取证而非Attention归因。

## 代码与审计边界

新增private_summarize_attention/private_summary_evaluation/run_private_summarize_attention、GPU0shell及新机制测试。复用protected训练/恢复；原model.py、runner.py不改。通用queue增加训练后calibration依赖，旧实验默认无此步骤；训练build/config/parser/eval支持新模型，私有backward额外记录query梯度。原main/test-only状态SHA应保持ea2e261e…/e9d1479f…；保留现有worktree和结果，不合并main、不删checkpoint。

本协议创建时尚未通过远程预检/启动，后续实际记录追加于此，不把计划命令当作已运行。
