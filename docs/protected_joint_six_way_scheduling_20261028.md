# GPU0六组并行接管：保留现有训练进度

2026-10-28，用户要求C4内容门控、FULL_ONLY新增分支仅Full不再排队，提前并行。

## 不改变实验配置

模型源码、训练入口继续使用原独立worktree `../multi-lane-main-protected-joint-attention-20261028` 的3950b35。六组seed1、8任务、30epoch/任务、batch64、原BCE/Adapter-ASL/aux.1、所有学习率/随机流/历史评估标准不变，沿用原配置及checkpoint。完整配置见protected_joint_attention_execution_20261028.md。

只新增独立队列控制器，使用另一个Git-only worktree部署，避免在训练进行中更改原模型HEAD，造成后续评估cfg/git核验失败。控制器接管原四个detached训练进程，只结束旧controller PID；不向训练进程/worker发送信号，不重启或恢复其训练。

## 显存与并发

检查时GPU0总24564MiB、已用14245MiB、空闲9837MiB；GPU1–7不使用。单组smoke allocated约2821MiB，原四组实际含保留缓存/上下文平均约3560MiB，预计增加两组后约21GB，仍有约3GB余量。

新上限6作业，其中评估最多2作业；启动前要求GPU空闲>=6000MiB、/dev/shm空闲>=1024MiB，启动间隔12秒。若实际资源不足，保留自动等待。原先protocol中的4并发保留为原始记录，新的runtime_scheduling_policy.json及concurrency_handoff.json记录此次6并发变更。

原输出和日志继续位于原模型worktree：

- `output/protected_joint_attention/seed1_20261028_01/`
- `logs/protected_joint_attention/seed1_20261028_01/`

## 预检与接管

新增5项测试、复用8项调度测试共13项本地通过：不重复启动原4组、6训练槽与2评估槽、显存门槛、活进程不能因完成文件被误判退出、Linux退出码/PID复用检查。

远程执行前dry-run要求：原模型worktree clean/HEAD与protocol一致；old controller命令和工作目录精确匹配；原四训练是各自session/process-group leader且无tty；原队列无失败。

```bash
/opt/conda/envs/ddp/bin/python scripts/emotic/expand_protected_joint_queue.py \
  --source-worktree ../multi-lane-main-protected-joint-attention-20261028 \
  --run-id seed1_20261028_01 --old-controller 3339902 --dry-run
```

实际接管使用同一命令移除--dry-run，在新的tmux会话运行。接管后保存原队列快照、PID/start_ticks和新控制器版本，复核原四PID未变、epoch推进、两组新PID开始更新、无重复进程及OOM。

## 后续自动工作

原四个被接管训练用/proc start_ticks和进程状态监控，必须退出且完整8任务产物成立才算完成；新训练用Popen返回码监控。完成后自动继续各自Val/Test，保留ON/OFF/移除新增Person或Face证据的共享前向评估，最后执行原完整跨臂参数/损失/增强batch/预测逐值核验并生成summary.csv。

只改变调度，无新增训练配置、无checkpoint覆盖或main合并。旧controller和模型HEAD、完整接管记录均保留于产物中。
