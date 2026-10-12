# 静态 MoE v1：validation/test 并行评价

用户于2026-10-12明确要求“validation和test并行”，本文件覆盖原v1计划中的先完成全部validation再加载test规则。四组训练、seed0、8任务×30轮、超参数、数据及固定checkpoint选择均保持原计划；只修改评价调度。

## 配置和输出

- 训练batch：static_moe_v1_seed0_eight_task_val_test_20261012_01；BASELINE、SHARED、MOE_UNIFORM、MOE_OVERLAP，运行HEAD957832632a372936b5cb752f788a617f77a42443。原训练worktree：/mnt/haoyuan/workspace/multi-lane-main-three-view-static-moe-v1。
- 新评价分支：exp/three-view-static-moe-parallel-eval。新服务器评价worktree：/mnt/haoyuan/workspace/multi-lane-main-static-moe-parallel-eval。通过Git-only部署，原训练checkout不pull任何新代码。
- 所有四组训练完成并通过原配对/旧参数/专家审计后，在GPU0启动两个独立进程：一个对全部四组×8checkpoint做validation，另一个做test。各batch64、workers2、OMP1、AMP/TF32沿用源config；两个进程之间没有模型/optimizer状态共享。启动前空闲显存至少14000MiB，不干预其他任务，等待上限12小时。
- 输入：源config记录的EMOTIC、CLIP ViT-B/16（SHA5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f）、Face manifest。完整val2397人，test5368人，各阶段全部已见类，不筛选组别/epoch，不用val或test重选权重。
- 输出仍在原batch的full_val/和full_test/；日志仍在原batch的val_evaluation.log和test_evaluation.log。源训练日志、32个compact checkpoint及旧审计保留。
- 共享parallel_evaluation_source_lock.json锁定四组固定checkpoint、config、seed_summary、paired/static审计、history和val score的SHA，且记录评价Git版本；两个进程各自检查同清单及前后不可变，完成后再次交叉验证。
- 最新调度状态：parallel_evaluation_state.json；旧控制器结束接管后pipeline_state.json转为并行评价状态。原training manifest与startup_receipt不改写。

## 正在运行批次的无中断接管

旧Python控制器已经载入顺序评价代码，不能通过pull让它热更新；pull还会造成后续两组训练来源版本不一致。因此保留原训练控制器和全部子进程，新增独立接管任务。

新任务先核对旧控制器PID、/proc启动时刻、命令行和cwd，只为尚不存在的full_val创建PARALLEL_EVALUATION_RESERVED.json。旧控制器继续完成全部四组，保存exit.json后进入评价，其原有“输出必须不存在”保护会在推理前退出。新任务等待这个可识别的预期退出，核对所有训练exit0，保存pipeline_legacy_handoff.json，归档原val日志为legacy_val_reservation_exit.log、预留目录为legacy_evaluation_reservation，然后同时启动两个新评价进程。该旧评价退出是明确接管步骤，不是训练失败。

如果任一训练失败、旧控制器提前退出、输出已有真实数据或退出原因不符合预留保护，新任务停止并保留所有产物；不启动部分评价，不删除数据。等待训练上限24小时。新tmux：ml_static_moe_parallel_eval_20261012。

入口（由助手在核对PID和新worktree clean后实际启动）：

```bash
/opt/conda/envs/ddp/bin/python -u -m multi_lane.track_a.static_moe_parallel_evaluation \
  --batch ../multi-lane-main-three-view-static-moe-v1/output/emotic_static_moe/static_moe_v1_seed0_eight_task_val_test_20261012_01 \
  --controller-pid 3138735 --gpu 0
```

## 修改及验证

evaluate_static_moe.py保留兼容的顺序模式，新增显式parallel-splits/source-lock模式；并行test解除先完成val限制但仍禁止smoke/test、检查全部训练完成、同来源锁和只读推理。static_moe_pipeline.py的新完整实验默认使用并行评价；static_moe_parallel_evaluation.py提供并行调度与本批接管。训练模型、loss、runner和训练Shell入口未修改。

新增测试验证两个真实工作线程均启动后才完成、任一split失败保留两者退出码、源checkpoint变化拒绝通过、无锁并行test拒绝、错误控制器拒绝、接管归档及意外已有数据保护。计划在新Linux评价worktree运行全套Track-A测试，再复用原smoke的32checkpoint进行带来源锁的64人val预检，不重训、不加载test；通过后实际挂接本批。当前为实施配置，尚未宣称新验证或接管完成。
