# Class-aware OOF stacking结果（2026-09-11）

## 实验与完整性

- run：`class_aware_oof_seed0_20260911_002`
- split：seed0完整8-task validation；`test_accessed=false`
- C0：固定R1。可靠Face使用Full/Person/Face=`0.64/0.16/0.20`，无效Face严格回退
  `0.80/0.20/0`。
- C1：每类中心化三视图bias，prior=`{1,3,10}`。
- C2：冻结最佳C1，增加rank-2样本可靠性×类别/view交互，prior=`{1,3,10}`。
- C1/C2：FP32、TF32 off、Adam LR `1e-2`、batch 512、120 epochs/task；训练只使用无泄漏
  image-group OOF scores，不重训专家，不访问test。
- 服务器退出码0；174项完整单测、24个历史endpoint严格复算和C1/C2真实GPU反向均通过。

## 主要结果

| 方法 | final val mAP | average mAP | cF1 | oF1 | forgetting |
|---|---:|---:|---:|---:|---:|
| C0 fixed R1 | 43.581193 | 50.750848 | 37.097821 | 58.520786 | 0.903941 |
| C1 prior1 | 43.586666 | 50.755218 | 37.102380 | 58.519047 | 0.904108 |
| C2 prior10 | **43.600490** | **50.761652** | 37.097298 | 58.520346 | **0.903925** |

C1相对C0仅提高`0.005472` final mAP；C2相对C0/C1分别提高`0.019297/0.013824`。
C2在8个task上的mAP相对C0均为正，task0到task7依次约为
`+0.0097/+0.0026/+0.0068/+0.0050/+0.0110/+0.0122/+0.0199/+0.0193`，但始终是小效应。

C2三档prior的final mAP为：prior1 `43.584517`、prior3 `43.589479`、prior10
`43.600490`。最强收缩反而最好，说明动态交互只有在接近固定先验时才稳定，并不支持继续放大
自由度。

## 逐类别结论

C2相对C0有8类AP提升超过0.01，最大单类占全部正增益42.53%，通过“收益不由单类独占”检查。
主要正增益为Sensitivity `+0.2357`、Fear `+0.1602`、Sympathy `+0.0600`、Aversion
`+0.0304`、Fatigue `+0.0265`和Disconnection `+0.0200`。主要负增益为Suffering
`-0.0314`、Pain `-0.0253`和Yearning `-0.0136`。

相对C1，C2仍将Sensitivity/Fear/Sympathy提高`+0.2344/+0.1581/+0.0571`，但Suffering
下降`-0.0962`。这说明class-aware机制缓解了原task级共享权重的类别冲突，却没有消除OOF训练
目标与held-out validation排序之间的冲突。

## 决策与原因

预注册门槛已从0.10降到0.05 final mAP，但C2相对C0/C1只有`+0.0193/+0.0138`，因此
`advance_to_seed1_seed2_validation=false`，不补seed1/2、不运行test。不能在看到结果后再次降低
门槛，否则属于对同一validation结果追随式调参。

失败的主要原因不是训练异常，而是可学习融合的上限较低：

1. 固定R1已经处于强局部最优，C1学到的类别平均权重仍非常接近固定先验。
2. C2最优落在最强prior边界，说明样本级动态信号在OOF到validation迁移时不够稳定；弱prior使
   final mAP更差。
3. 三个专家共享相同数据和CLIP主体，错误高度相关；Router只能重新分配现有信息，不能产生新的
   可分辨特征。
4. Face endpoint本身明显弱于Full与Person，且无效Face必须回退，因此动态Face权重可作用的样本
   和收益上限都有限。

## 下一步

按预注册规则结束当前动态Router/stacking路线：不继续搜索prior、rank、hidden、描述符或类别独立
大网络。保留C2作为“OOF class-aware仅带来小幅、未过门槛收益”的完整消融。

若目标是提高最终mAP，下一阶段应先提升独立Face expert的信息质量，再用锁定R1验证增量：只在
validation比较少量、结构清晰的Face输入/训练改进（更稳的目标人脸匹配、保留高分辨率的
margin+letterbox、面部局部增强或面部专用表征），Full与Person专家及融合权重保持锁定。只有新的
Face expert在自身AP和固定R1融合mAP上均有稳定提升，才值得补seed1/2和一次正式test。

## 产物

- 本地结果：`output/emotic_track_a_class_aware_oof/class_aware_oof_seed0_20260911_002/`
- 本地/服务器压缩包：`class_aware_oof_seed0_20260911_002.tar.gz`
- SHA-256：`978e6c72b9eede62518959d8f5f8957148c5509c4b0ad1d1751b554b19e841d8`

