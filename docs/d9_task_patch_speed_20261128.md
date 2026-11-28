# 2026-11-28 融合实验执行加速

按用户最新要求，正式接续仅使用GPU0，GPU1留空。GPU1此前只用于已完成的测速和预检。本次只优化执行，不改变上一批实验的结构、数据或监督。

## 原因与修改

当前融合已经是全批训练：每一步使用当前拟合集的全部原图及完整人物列表。配置中的batch size 64用于三路图像特征提取，调大它不会加快当前融合拟合。

1. CLS和四分区patch的描述不含可学习读取参数。原程序在300步中反复计算；现在每个拟合集只计算一次。可学习patch读取器的输出仍每步重算。
2. 冻结patch的float32转换、缺脸遮蔽和有限值检查提前一次完成；三路同图人物汇总合并传输。仍使用原CPU逐列prefix sum，保留计算顺序和反向梯度。PyTorch 2.0.1的CUDA浮点cumsum在严格确定性模式下不可用，本轮不改变该协议。
3. 一个任务、一个来源的拟合相互独立。比较同GPU的1/2/4个进程后，采用GPU0最多4个任务，剩余工作多的任务先调度。每个任务仍完成原图三折和最后全拟合集拟合。
4. Validation/Test在GPU0依次处理两个来源；同一Task Forward的特征只搬入GPU一次。无可学习读取器的描述按方法及实际patch交换/位置打乱分别缓存。可学习patch输出仍由各自封存参数计算。
5. 复制原运行中已完成的候选拟合JSON，核对旧协议、每份SHA、拟合配置和特征身份。独立输出保留完整导入清单；不重写原结果。两个来源全部Validation完成、封存后才开始Test。

## 真实TRAIN测速

GPU1，RTX4090，FP32，TF32关闭，严格确定性，CPU单线程。使用现有EMOTIC TRAIN特征，不读取Val/Test指标选择优化。每组预热2步，再完整拟合300步；Adam LR0.001、参数均方L2 0.001、seed1，与正式实验相同。下表为单次实测，不代表端到端倍率。

| 来源／任务 | 特征读取 | 优化前秒 | 优化后秒 | 单拟合倍率 |
|---|---|---:|---:|---:|
| 90%来源 task6，47图／68人物 | CLS | 1.859 | 1.399 | 1.33 |
| 同上 | CLS＋四分区patch | 2.202 | 1.421 | 1.55 |
| 同上 | CLS＋可学习patch | 4.286 | 3.786 | 1.13 |
| 全TRAIN task3，8557图／11175人物 | CLS | 5.361 | 1.752 | 3.06 |
| 同上 | CLS＋四分区patch | 36.862 | 1.579 | 23.35 |
| 同上 | CLS＋可学习patch | 21.745 | 22.353 | 0.97 |

六组完整状态（包括全部参数、TRAIN中心、损失轨迹）及预测逐值相同。最大任务的可学习patch单拟合没有变快，不能把短测速的改善当成长跑收益。

在同一个GPU1上，用全TRAIN task3的可学习patch方法各跑100步：

| 同时进行的拟合数 | 整批墙钟秒 | 完成拟合数／秒 | 相对吞吐 |
|---:|---:|---:|---:|
| 1 | 7.029 | 0.1423 | 1.00 |
| 2 | 7.636 | 0.2619 | 1.84 |
| 4 | 9.046 | 0.4422 | 3.11 |

七次拟合全部状态SHA相同。采用已测选项中吞吐最高的4进程配置；这不是两卡全流程加速3.11倍的承诺。数据加载、提取、压缩和指标统计仍有CPU工作，GPU利用率不必持续100%。

## 配置与复现

独立分支`codex/d9-task-patch-speed`。配置`scripts/emotic/configs/d9_task_patch_speed_20261128.json`与原配置只有输出及日志目录不同。来源、checkpoint及缓存路径全部沿用原配置；三路冻结，最终CLS和196patch固定投影32维，读取隐藏宽度8；Full分数1、辅助0初始化，原14个拟合候选及20条评估行全部保留。两来源合计40行，task0–7 Validation/Test；每项300步、原图三折、100次分数交换、10次patch交换、一次位置打乱。仍对照Test33.1770／40.2083及同来源固定比例。

```bash
# 从独立服务器worktree根目录执行，环境使用项目原ddp环境。
CUDA_VISIBLE_DEVICES=1 python -m scripts.emotic.d9_task_patch_benchmark --steps 300
CUDA_VISIBLE_DEVICES=1 python -m scripts.emotic.d9_task_patch_concurrency
CUDA_VISIBLE_DEVICES=1 python -m scripts.emotic.d9_task_patch_parallel --phase preflight --gpus 1
CUDA_VISIBLE_DEVICES=0 python -m scripts.emotic.d9_task_patch_parallel --phase formal --gpus 0 \
  --resume-from ../multi-lane-main-d9-task-patch-20261128/output/d9_task_patch/seed1_20261128/formal
```

输出`output/d9_task_patch_speed/seed1_20261128/`；日志`logs/d9_task_patch_speed/seed1_20261128/`。正式恢复需要匹配当前代码的完整预检；旧来源协议固定为`e3afb150ab6f5bdb59c3a8aa34a11a4d49c697a5930912c6c0df4c8de6db90b6`。

26项Linux测试已通过：新增缓存/梯度/预测/换图一致性3项，原任务路径10项，原patch路径13项。实际预检及正式切换的最终状态见本目录运行记录及三个根上下文文档。
