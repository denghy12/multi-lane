# 未启动作业分流GPU1，保留GPU0训练进度

用户于2026-10-23要求检查进度并把排队任务放到GPU1并行。实查GPU1仅使用11MiB、空闲24071MiB；GPU0已较拥挤。叠加7组完成8任务，F111位于task6末尾；区域baseline/old_bias在task3，上下文baseline完成task2。待启动为区域regional、上下文legacy_task_residual/task_context/person_context共4组。两类实验全部原GPU smoke已经complete。

新分支exp/gpu1-pending-split-20261023只调整外部调度，不修改区域6bea373、上下文7c25e13或叠加0fb85f1科学源码。不重启现有GPU0训练；仍按各自原数据划分、seed0、8任务、30epoch、B64、原Adam/LR/cosine、aux0.1、workers0、AMP/TF32运行。区域与上下文仍分别90% fit和全TRAIN，不能混作同一匹配对照。

接管时持有旧共享放行锁，SIGSTOP两个旧调度父进程，核验并记录它们已经启动的所有子进程、PID启动标识、命令、数据路径、原排队参数及checkpoint SHA。训练子进程继续运行。旧父进程保持停止仅用于保留子进程完成后的/proc zombie状态，使新控制器可读取真实exit_status；全部原子进程完成后，向旧父进程发送TERM再CONT收尾。禁止对训练子进程发送停止或重启信号。

新控制器只把未启动命令放在CUDA_VISIBLE_DEVICES=1，并严格比较参数与原protocol/launch.json。GPU1启动前通过共享锁预留：训练4500MiB、评估2500MiB，保留2000MiB余量，并检查实际free。最多4训练或8评估进程，混合按总预算限制；allocator仍16%，OMP1。原GPU0进程不占GPU1预算。每任务内部仍顺序训练，未改变增量学习任务顺序。

沿用原验证顺序：本组全部训练完成→原旧参数/compact/恢复核验→完整8阶段validation/test→原validation开关/Query或位置归因→真实completion。输出、日志沿用原run目录；实际GPU路由另写resource_override_gpu01.json和training_gpu01/evaluation_gpu1/attribution_gpu1.json，不伪造已完成任务，不覆写原协议。

新增CPU测试覆盖GPU1混合预算、4训练/8评估上限、真实空闲保护、继承进程不重启、PID复用与丢失拒绝、真实zombie退出码和信号退出码。当前为实施计划，实际接管与GPU1运行状态随后补充。叠加控制器不迁移，其后续评估仍可在GPU0执行，与GPU1新训练并行。
