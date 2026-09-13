# OOF优势限定的排序蒸馏

## E1锁定方案

- seed0 EMOTIC完整8-task validation，30 epochs/task，hard batch64。
- Full hard BCE权重1.0，随机legacy crop；Adapter hard-label ASL 9.8/0/0.05。
- 每个hard update增加16个确定性Full视图的正负纠错pair，pairwise logistic loss权重0.05。
- 只启用R1-Full OOF AP在三折均为正的13类；只使用Full OOF排错但R1 OOF纠正的pair。
- task2/task6无合格类，严格回退原hard BCE+Adapter ASL。
- 排序损失只更新Full的selectors/prompts/head，不更新Adapter；旧task lane冻结。
- 不存checkpoint，不访问test，不搜索batch/weight。

相对已有D0必须同时满足Full final mAP `>=+0.05`、Full average mAP不降、
固定R1 final mAP `>=+0.05`。失败即结束OOF蒸馏路线。
