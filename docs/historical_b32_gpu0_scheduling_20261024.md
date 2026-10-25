# GPU0并发扩容：保留历史b32训练进度

> 2026-10-25完成：18组训练及36份完整validation/test均已结束，seed1精确复现历史，seed0最终目标未达，阶段平均三个seed均正向。完整数据与机制分析见[结果报告](historical_b32_seed03_results_20261025.md)。下文保留方案及执行历史，不再代表运行中状态。

> 当前已成功：a30120c控制器接管，6组GPU0训练同时运行；原4PID/start ticks完整保留，新增seed1/3 Full前5轮0跳步，所有batch64与模型00fbf90不变。显存约17056MiB、GPU计算100%；8项本地测试及22项远程相关检查通过。成功控制日志driver02.log，首次失败尝试driver.log保留。

2026-10-24。用户提出利用空闲显存，允许增大batch或增加并行组数。实测原4组共用约12.8GB/24.6GB，GPU计算利用率100%，共享内存185MiB/4GiB。此次采用增加并行，不改batch64及任何训练参数，不新增实验组，不重启现有训练。增加显存占用不保证线性提速，需观察稳定吞吐。

模型源码继续锁定原独立worktree的00fbf90，18组/8任务/30epoch/旧标准/seed0,1,3/原CPU72与采样RNG全部保持。调度控制器通过Git-only部署到另一个独立工作树，不能在运行中的模型工作树更新受控源码。

接管先核实4个训练PID都是独立session/process-group leader，且无控制终端（TTY=0），再只向旧调度器PID发送SIGTERM；不给训练PID或进程组发送信号。新控制器接管原queue.json、记录进程start ticks、完成后核对真实Linux退出状态或明确标记的产物/日志证据，并沿用原冻结/恢复/seed1历史精确复现/预测重放检查。新训练仍从原00fbf90工作树、原CLI启动，B64不变。

运行策略：训练最多6进程、同时评估最多4进程；新作业要求GPU空闲至少6000MiB、/dev/shm空闲至少1024MiB，启动间隔12秒。策略写入原run的runtime_scheduling_policy.json，可在允许范围动态降并发，不需再次迁移。记录策略变更历史、接管前队列、接管来源和原PID，不覆盖checkpoint，不触碰GPU1–7。

入口scripts/emotic/expand_historical_b32_queue.py；先--dry-run，核对源码HEAD/原调度器身份/队列，再接管。输出与日志继续使用原formal_seed013_20261024_01；控制器日志另外存logs/historical_b32_search/concurrency_20261024_01。新增7项单元测试覆盖显存/共享内存准入、评估限流、动态降并发、PID与退出码判断。当前本地8项已通过，尚未完成正式接管。

首次暂停式接管未通过：7fa561d控制器发SIGSTOP后没有观测到持久T状态，因此在接管前中止，原4训练继续；已SIGCONT回滚旧调度器。原PID/start ticks、源00fbf90均保持，日志保留driver.log。新增严格独立会话/TTY检查后改为只退出旧调度器；源Popen本就start_new_session=True，已实测四PID各自SID=PGID=PID且TTY=0。该策略额外通过回归；不猜测暂停状态未保持的外部原因。

成功接管审计已同步本地：26份文件逐SHA验证，归档ea80ee565ebf0e2ec33cab00bd610b02fdbb3ce7b93611a5cb87792ddd62d394；原4训练PID/start ticks完整保留，新2组前5轮0skip，seed1 task0预测精确匹配历史。
