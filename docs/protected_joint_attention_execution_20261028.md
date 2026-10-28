# 受保护联合摘要Attention：六组完整八任务执行协议

2026-10-28用户授权执行；以用户本轮完整8任务请求覆盖规划中的task0效果筛选。本轮只有实现/保真预检门槛，不按任务0或Test表现截断训练。

## 配方与对照

- EMOTIC，seed1，任务[5,3,3,3,3,3,3,3]；历史samples_intersect_seen_classes，8阶段算术平均。
- C0原强三视图B0；C1三路CLS→MLP；C2三视图联合摘要Attention；C3增加目标框Full摘要；C4增加图片内容门控；FULL_ONLY与C4相同但新增分支只读Full、不读Person Query。原三路主路径各组均不变。
- 每组每任务30epoch、batch64，原Adam/WD0/cosine/每任务重置，主LR.0125、Image-token Adapter LR.0004；CLIP ViT-B/16冻结，b32/index1/scale.03/ReLU；10共享Selectors/原Prompts，Person颜色扰动关闭。原融合及辅助BCE/Adapter-ASL，aux .1、gamma负9.8/正0/clip.05。分类头来自本模型，全30轮正常学习。
- 新分支第6层证据、64维/4头/1层，学习率.0004、Adam/WD0/cosine；固定门控.1；C4/FULL_ONLY门控初值.1，残差RMS上界.1再乘门控。只有输出投影置零，防止双零初始化。C1隐藏96，与C2新增参数近似匹配，报告实际参数。
- 独立私有optimizer/GradScaler，无新增目标向原参数反传；原损失根据未加新残差的h0计算。没有教师或训练好的任务权重加载，没有换头、额外5轮或后验组合。
- OMP/MKL/model线程72、AMP初始65536/TF32，train workers2/prefetch1；历史两个CPU报告seed draw每任务一次。新增模块使用隔离CPU随机流、无dropout。

## 路径与入口

相对服务器实验worktree：

- 配方及只读历史保真参照：`../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train`。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- 标准CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`；与历史config记录SHA核验。
- Face manifest：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- 训练/紧凑checkpoint/完整恢复/每epoch恢复：`./output/protected_joint_attention/<run_id>/<arm>/train/`。
- Val/Test和ON/OFF/证据移除NPZ：同run下`<arm>/val/`、`<arm>/test/`。
- 日志/队列：`./logs/protected_joint_attention/<run_id>/`。

```bash
bash scripts/emotic/run_protected_joint_attention_gpu0.sh smoke_seed1_20261028_01 --smoke --smoke-updates 4
bash scripts/emotic/run_protected_joint_attention_gpu0.sh seed1_20261028_01
```

尚未执行的命令不能当作启动完成。正式执行前必须记录实际commit和smoke结论。

## 预检和资源

- Git-only新远程worktree；原服务器主树/test-only状态SHA分别ea2e261e581efb0c9c7a56bfcf7078611374fcd25f9348512e102de0ab888ccf、e9d1479fb6a0354fee4017234173b00a8cc237895f556f80c7129e4555d16152，必须保留。
- 专项：真实新增更新时原模型/原loss/RNG精确一致、梯度及优化器隔离、初始恒等、无效Face像素不影响输出、目标mask排CLS/不可见安全、两个优化器epoch恢复。随后相关完整回归。
- 六组两任务短smoke并检查各组原参数/采样流一致、零AMP跳步、新梯度有限非零、只读Val首64。smoke不访问Test、不作效果依据。
- 正式各组每任务检查原B0参数、30轮关键loss、Python/NumPy/CPU/CUDA RNG与历史边界逐值相同；历史checkpoint仅作比较，不安装入本模型。检查失败必须修复实现，不带漂移继续跑。
- GPU仅0；上限4进程/2评估，GPU空闲低6GB或/dev/shm低1GB排队，启动间隔12秒，禁止中断其他作业。初始GPU0约24GB空闲；正式前再核对smoke峰值与调度。
- 每epoch原子保存包含主/私有Adam、scheduler、GradScaler、完整RNG、历史/采样摘要的可恢复状态；最终报告前再保存边界。保留已有文件，不覆盖checkpoint。
- 每个评估batch共享三路编码和原Task Forward，再计算ON/OFF/新增Face证据移除/新增Person证据移除；保留原CPU FP32分batch sigmoid及评估人群。最多2评估进程、每进程2个单线程spawn worker。

## 产物与解释

输出六组8task Val/Test最终与阶段平均、逐阶段/逐视图/可靠Face指标、ON/OFF与移除证据预测、原函数保真审计、恢复状态和资源信息。最终OFF应回到配对C0，并与历史B0相同。先确认原模型和旧任务保护，再判断Attention是否胜过MLP/Full-only和Face是否有条件贡献。

目标Test最终32.6800、阶段平均39.5292均提升；33.6248/40.5129的完整模型均值仅为有成本标注的经验参照，不是监督答案。Test参与开发，保留所有组，不按Test选择早停epoch冒充新结果。
