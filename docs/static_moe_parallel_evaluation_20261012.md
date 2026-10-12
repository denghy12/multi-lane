# 静态 MoE v1：validation/test 并行评价

用户于2026-10-12明确要求“validation和test并行”，本文件覆盖原v1计划中的先完成全部validation再加载test规则。四组训练、seed0、8任务×30轮、超参数、数据及固定checkpoint选择均保持原计划；只修改评价调度。

## 配置和输出

- 训练batch：static_moe_v1_seed0_eight_task_val_test_20261012_01；BASELINE、SHARED、MOE_UNIFORM、MOE_OVERLAP，运行HEAD957832632a372936b5cb752f788a617f77a42443。原训练worktree：/mnt/haoyuan/workspace/multi-lane-main-three-view-static-moe-v1。
- 新评价分支：exp/three-view-static-moe-parallel-eval。新服务器评价worktree：/mnt/haoyuan/workspace/multi-lane-main-static-moe-parallel-eval。通过Git-only部署，原训练checkout不pull任何新代码。
- 所有四组训练完成并通过原配对/旧参数/专家审计后，在GPU0启动两个独立进程：一个对全部四组×8checkpoint做validation，另一个做test。各batch64、workers2、OMP1、AMP/TF32沿用源config；两个进程之间没有模型/optimizer状态共享。启动前空闲显存至少14000MiB，不干预其他任务，等待上限12小时。
- 输入：源config记录的EMOTIC、CLIP ViT-B/16（SHA5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f）、Face manifest。完整val2397人，test5368人，各阶段全部已见类，不筛选组别/epoch，不用val或test重选权重。
- 输出仍在原batch的full_val/和full_test/；日志仍在原batch的val_evaluation.log和test_evaluation.log。源训练日志、32个compact checkpoint及旧审计保留。
- 共享parallel_evaluation_source_lock.json锁定四组固定checkpoint、config、seed_summary、paired/static审计、history和val score的SHA，且记录评价Git版本；两个进程各自检查同清单及前后不可变，完成后再次交叉验证。
- 接管等待/最终状态：parallel_evaluation_state.json；旧控制器结束接管后，以pipeline_state.json查看实时并行评价状态。原training manifest与startup_receipt不改写。

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

新增测试验证两个真实工作线程均启动后才完成、任一split失败保留两者退出码、源checkpoint变化拒绝通过、无锁并行test拒绝、错误控制器拒绝、接管归档及意外已有数据保护。新Linux评价worktree已通过318项Track-A测试；复用原smoke全部32checkpoint完成带来源锁的64人val预检，全部重放差异0、来源/CLIP不变，未重训、未加载test。

## 实际部署与启动确认

实现已提交/push并Git-only部署，评价固定clean HEAD ea722ac625060f303d519bbcc6a36b2e4445da52；原训练仍固定clean HEAD957832632a372936b5cb752f788a617f77a42443。两者model/runner/static_moe/adapter及训练Shell逐文件无差异，四组训练来源不混用新评价版本。

2026-10-12 18:01:10（北京时间）确认：tmux ml_static_moe_parallel_eval_20261012已运行，日志输出STATIC_MOE_PARALLEL_HANDOFF_ARMED waiting_for_training；原控制器PID3138735仍运行，原pipeline_state仍training，新的parallel_evaluation_state为waiting_for_training，training_interrupted=false。接管任务已实际挂接，尚未宣称正式val/test开始或取得精度结果。

源batch的parallel_handoff_receipt.json记录上述验证、两个运行Git版本、318项测试及32个重放结果；原smoke batch的preflight_parallel_val/comparison.json记录带锁恢复评价。39个新非checkpoint诊断文件已同步主本地工作树，归档logs/emotic_static_moe/verification/static_moe_parallel_eval_preflight_20261012.tar.gz两端SHA256均为a3826a9e3a2ebd6bf7b89c81b38befec5d70ba3af8bd26ed7727a6b74e0d0750。

后续文档提交不pull进正在运行的训练/评价worktree。所有训练结束后接管任务自动启动val/test两个进程；结束后同步full_val/full_test和评价日志，检查同一来源锁hash、四组各8checkpoint、完整人群及只读/来源不变结果，再统一分析。
