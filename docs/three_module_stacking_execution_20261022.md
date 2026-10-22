# 三模块叠加消融执行记录

2026-10-22，用户已要求实现并在GPU0自动并行/分轮执行完整8任务validation和test。当前状态：实现完成第一版，等待远程单元测试、真实初始化和八组smoke；**正式实验尚未启动**。

本地实验分支`exp/three-module-stacking-20261022`，从`1e28bfb`创建，保留此前未提交分析文档。远程拟使用独立工作树`/mnt/haoyuan/workspace/multi-lane-main-three-module-stacking-20261022`，不改原主树/test-only/已完成来源实验。

## 锁定配置

EMOTIC、seed0、八任务5/3/3/3/3/3/3/3类、30epoch/task、B64、原图90/10 fit/holdout，预期12660更新/组。主Adam LR0.0125、模块LR0.0004、WD0、余弦到0；joint BCE＋aux0.1，无ASL；AMP/TF32、scale1024/growth1e9、workers0。原三视图输入/可靠Face规则、固定归一化特征融合和共享分类头不变。

A=shared Image-token Adapter，索引1/b32/ReLU/scale0.1；B=task Selector Query读取Full摘要并写回Person，索引5/总宽64/4头/scale0.1；C=target原六维几何、12层共享60参数/任务。新增读取统一收缩路径；A不写回Frozen Forward；B源缓存包含当前A/C影响。八组F000/100/010/001/110/101/011/111，不据中途成绩删除格子。

输入沿用已确认资源，通过环境/命令参数传入：

- `DATA_ROOT=../multi-lane-main/datasets/EMOTIC`
- `CLIP_CHECKPOINT=../CODE_DDP/pretrained/clip/ViT-B-16.pt`
- `FACE_MANIFEST_ROOT=../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`
- 初始化只读参考：`../multi-lane-main-target-summary-gpu0-resume-20261022/output/target_anchored_summarization/formal_gpu0_resume_seed0_20261022_01/baseline/train`；不复用其训练权重。

输出：`output/three_module_stacking/<run_id>/`；日志：`logs/three_module_stacking/<run_id>/`。入口`bash scripts/emotic/run_three_module_stacking_ablation.sh`；controller自动启动全部训练、训练完整性核验、16路完整val/test作业队列和F111 task0/2/7的四组validation关闭归因（全开重放正式val）。最后输出summary.csv、interactions.json及1000次原图bootstrap的validation最终差值。

GPU0初始24GiB空闲。拟训练/评估最多4进程、每进程Torch分配器上限18%、启动需5600MiB余量，最终并发值以真实最重配置smoke为准。队列容量不足自动等待、多轮启动；不降低batch，不终止外部进程。失败记录未运行作业及错误，不能标记完成或静默删实验。

## 保护与预检

服务器原主树状态153行，SHA256 `ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf`；test-only状态5行，SHA256 `e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152`。不修改这些工作树。

本地语法预检及diff检查通过；尚未声称Torch测试或真实GPU短跑通过。测试将覆盖：八组零初始化等价、优化器与梯度、A+C偏置生效、C经Full缓存影响B、冻结图像流不变、旧任务冻结、compact/完整恢复、严格协议与交互公式。smoke仅validation，不读取test。
