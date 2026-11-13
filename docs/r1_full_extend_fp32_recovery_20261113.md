# 全TRAIN控制：Person FP32数值恢复复验

日期2026-11-13；用户已授权补齐控制。原AMP4096和降低到AMP32的真实task3第14轮均出现7次Person跳步，Full与Face零跳步。AMP32实际生效，末scale为0.25；不能将失败当作零跳步控制或从失败e14继续。

静态无更新156batch在safe3e13参数上AMP4096/32/FP32全有限，证明静态检查未重现更新后的故障，不证明动态FP32一定成功。接下来仅Person改为完整FP32反向，其余Full/Face保留AMP，仍使用同一safe task3 epoch13模型、Adam、scheduler、RNG、数据、loss和LR，不重置优化器、不调LR或新增裁剪。

独立分支codex/r1-full-extend-fp32-recovery-20261113基于c957d66，独立worktree。正在运行的人群对照保留原分支c957d66不变，禁止更新其源码。GPU0继续承担人群对照；先在内存许可时单独验证Person的真实第14轮FP32（156有效、零skip），再将完整三路控制放到人群对照之后自动运行。若单Personprobe有效，其同轨迹合法e14可由完整控制保存为三路共同e14；Full/Face仍从e13计算e14。失败probe不得作为成功状态续跑。

数据/原配方继续EMOTIC、seed1、8任务、B64、主LR.0125、Adapter/交互LR.0004、Adam/cosine/WD0、Frozen9 Person＋Face、b32/index1/scale.1、原BCE和Adapter-ASL、总30＋5轮。目标补齐完整Val/Test固定融合控制，结果必须标明数值策略边界，不与原AMP轨迹声称逐值同一。

输入相对路径沿既有EMOTIC/CLIP/Face manifest。源恢复state仍旧worktree `output/r1_fusion_return_train/seed1_20261113/full_extend/recovery/task3/epoch13.pth`，SHA15d570f6d928854312d473ccdf0a41de3110ce111fc964581feeadcbcceedfd8。

新输出位于output/r1_full_extend_fp32_recovery/seed1_20261113，日志logs/r1_full_extend_fp32_recovery/seed1_20261113。源码和测试完成后先验安全状态/配置/FP32与probe边界，所有源文件SHA需保持。不是改融合人群实验，也不从其他实验导入分类头或比例。

## 实施门禁

09d6a1f代码已通过远程18项恢复/FP32测试，prepare复制977份安全文件并校验原source及copy SHA一致。运行前锁定最终提交；先仅Person真实epoch14 probe，再由独立队列等待人群对照结束后启动完整三路。probe及后续控制使用同一固定源码，不改变训练配置中的git来源。

FP32安全检查涵盖loss/梯度/参数/Adam，不将关闭GradScaler造成的“无跳步”误当成有限更新证明。原AMP32失败记录保留；FP32数值策略边界需在结果报告中明确。

## 自动队列

入口`python -m scripts.emotic.launch_r1_fp32_recovery`。要求真实Person FP32 e14 probe通过；等待正在运行的人群实验训练、8阶段Validation/Test完成且进程退出。随后GPU0连续三次空闲显存至少20480 MiB才启动本恢复的Full/Person/Face三路并行。队列锁定clean HEAD，失败明确停止，不自动接续含跳步状态。

状态`output/r1_fp32_recovery_queue/seed1_20261113/status.json`，训练日志`logs/r1_full_extend_fp32_recovery/seed1_20261113`。同一源码先probe后queue，避免config.git来源不匹配；运行后只在规范本地仓库更新文档，不修改运行工作树。
