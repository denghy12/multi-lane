# B0 保真 logit residual 校准结果

正式批次：`logit_residual_control_20260925_163436`。代码来自 clean server
commit `865aec1`，只运行 seed0、task0--2 validation，未访问 test。

## 完整性

`B0-paired` 与 `L-post-logit` 均完成 90 epochs、5,010 optimizer updates、
zero skipped；状态为 `complete`，日志以 `MULTI_LANE_TRACK_A_COMPLETE` 结束，
没有 OOM、traceback、NaN 或 non-finite。同步到本地的 24 个输出、NPZ、日志和
控制文件逐项 SHA-256 与服务器一致。

## 结果

| 指标 | B0-paired | L-post-logit | 差值 |
|---|---:|---:|---:|
| final mAP | 45.2395 | 45.3251 | +0.0857 |
| average mAP | 54.9341 | 55.0368 | +0.1027 |
| final cF1 | 36.8870 | 36.8898 | +0.0028 |
| final oF1 | 61.6314 | 61.7974 | +0.1660 |
| forgetting | 2.7760 | 2.7458 | -0.0302 |
| task0 mAP | 60.5225 | 60.6202 | +0.0977 |
| task1 mAP | 59.0402 | 59.1650 | +0.1248 |
| task2 mAP | 45.2395 | 45.3251 | +0.0857 |

两组的 base BCE history 以及 Full、Person、Face、reliable-Face 单路指标完全
相同，证明小增益只来自后置校准，没有改变 Selector、Prompt、classifier 或单路
表征。相对 B0，task0/1/2 的 Full-relative ranking 净纠正 pair 分别增加
`1,389/4,139/4,186`。但 task2 在多纠正 16,732 个错误 pair 的同时也损坏
12,546 个正确 pair，选择性仍然不足。

校准系数有限、非零且远未饱和。task0/1/2 结束时最大绝对值为
`0.0412/0.0410/0.0093`，低于 `0.10` 上界。task2 的 Person/Face 平均绝对
系数只有 `0.00096/0.00047`；这与 task2 仅 420 次更新、三路对新类的互补信号
较弱一致。

## 决策

候选满足 task0、average、forgetting 和系数检查，但 final mAP 仅提高
`+0.0857`，低于预注册的 `+0.10`。因此不补 seed1/2、不运行 test，也不继续
搜索 residual scale、学习率、rank、expert 或 level embedding。该结果应表述为
“稳定但不足以晋级的小正效应”。

这也结束当前 ParaX image-stream、post-feature residual 和 logit residual 校准
序列。下一步转向冻结 CLIP 后已有的 Image-token Adapter 接口，比较共享 b32、
近乎等参数量的共享 b97，以及 Full/Person/Face 三路独立 b32；先验证是否能形成
更强的视图专家，再讨论 Router。
