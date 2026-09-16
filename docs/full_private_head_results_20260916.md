# Full分类头独立诊断结果

批次：`full_private_head_seed0_20260915_231942`。训练代码HEAD为`87b406f`。所有结果均为
seed0完整8-task validation；未访问test，也未运行seed1/2。完整结果位于
`output/emotic_track_a_full_private_head/full_private_head_seed0_20260915_231942/`，日志位于对应的
`logs/`目录。

## 完成与同步审计

- H0、HF和自动汇总器退出码均为0；两组各完成240 epochs、13,950 optimizer updates、0 skipped。
- 未保存checkpoint，未访问test；没有OOM、NaN或残留训练进程。
- 本地与服务器`validation_summary.json`的SHA-256均为
  `73017155294c4ecd71238dd7cd39b2ee1dc9c5e1db920ff9c39cf3096aaf0913`。
- 服务器正式训练前已通过221/221单元测试及H0/HF真实task0一轮配对smoke。

## 实验回答什么问题

两组共享冻结CLIP ViT-B/16、Selector、Prompt和Image-token Adapter b32，固定三视图权重、Face可靠
mask、训练目标、数据顺序、优化器和学习率日程完全相同：

- **H0**：同一个分类head分别作用于Full/Person/Face，再固定加权融合三路logits。
- **HF**：仅给Full复制一个独立分类head；Person/Face继续共用原head。复制初始化不消耗额外RNG。

因此H0/HF比较直接检验：P0的Full预测较弱，是否主要因为Full被迫与Body/Face共享分类决策边界。
训练设置为seed0、8 tasks、每task 30 epochs、batch 64、Adam每task重置、初始学习率0.0125、
CosineAnnealingLR、AMP、固定融合及0.1逐视图辅助损失。

## 主要结果

| 指标 | H0共享head | HF Full独立head | HF-H0 |
| --- | ---: | ---: | ---: |
| final mAP，训练logit融合 | **42.6627** | 41.1145 | **-1.5482** |
| average mAP，训练logit融合 | **50.3853** | 48.3977 | **-1.9876** |
| final mAP，离线概率融合 | **42.4076** | 41.2419 | **-1.1656** |
| Full单路mAP | **41.0116** | 39.6184 | **-1.3932** |
| Person单路mAP | **40.0137** | 37.4710 | -2.5427 |
| Face单路mAP | **35.4149** | 33.8664 | -1.5485 |
| 可靠Face单路mAP | **38.9194** | 37.0066 | -1.9128 |
| final cF1 | **38.1126** | 37.0213 | -1.0913 |
| final oF1 | **58.2650** | 57.5312 | -0.7338 |

HF在task0--7的累计mAP全部低于H0，差值依次为
`-3.5886/-2.5708/-1.7947/-1.6150/-1.5111/-1.5106/-1.7617/-1.5482`，不是只由最后一个task
或遗忘造成。Full私有head也确实学成了不同决策边界：最终与共享head的weight cosine为0.5853，
相对权重L2差异为0.8201；因此失败不是因为两个head没有分化。

按1705个原图组做2000次配对bootstrap：

- HF logit减H0 logit：均值`-1.5553`，95%区间`[-2.6199, -0.5365]`；
- HF probability减H0 probability：均值`-1.1788`，95%区间`[-2.2723, -0.1508]`。

这两个固定训练实例上的图像抽样区间均低于0，支持“本次直接拆Full head有害”。它不能替代训练seed方差，
但效果方向和幅度已经不值得补seed1/2。

## 基线漂移及结论边界

H0 final为42.6627，比历史P0 43.0754低0.4127，未满足预注册的0.10复现门槛。原因是H0把历史P0的
“先融合特征、再过共享线性head”改写为“各视图先过共享head、再融合logit”。二者在实数代数中等价，
但AMP运算次序不同；固定模型离线误差很小，长期非凸训练轨迹仍可能被浮点差异和Adam状态放大。

因此不能把HF直接与历史P0作精确因果比较，也不能用本实验断言任何一种更广义的私有head都必然失败。
不过H0/HF使用相同逐路logit训练路径、相同随机协议，内部对照仍然有效：**简单复制并放开Full分类head没有
改善Full，反而扰动共享上游表示，使三路一起退化。分类head共享不是当前主要瓶颈。**

## 与此前实验合并后的定位

1. P0复现+A排除了融合运算：共享P0从logit改为概率融合反而由43.0754降到42.8238；独立R1的概率/Logit
   差只有+0.1358。历史0.5058差距不是“P0选错概率还是logit”造成的。
2. B的八种预测来源替换把差异定位到Full输出：共享概率端点差0.7574的Shapley分配为
   Full `+0.7884`、Body `+0.0910`、Face `-0.1220`；共享Face实际有益，Body效果依赖Full背景。
3. 本轮继续排除Full分类head：Full独立head使融合和所有单路都下降，且head已充分分化。
4. 旧P2“共享b32+每路b4小Adapter增量”也未解决：final 42.7961，比P0低0.2793；这只否定该小增量结构，
   不能替代“Full使用完整私有Adapter”的受控检验。
5. 旧G2 DGL式双向梯度隔离为42.4147，比同批G0低0.6607；G0全task融合/单路梯度余弦均为正，未观察到
   持续破坏性冲突。因此已测DGL训练法也不是答案，但这不等于排除所有梯度幅度或共享干扰问题。

当前最窄、最可靠的表述是：**共享模型相对独立R1的损失主要表现在Full来源的预测质量及其与Body的配合，
不是Face、不是概率/logit融合算子，也不是可由简单Full独立分类head修复的问题。真正的训练机制尚未定位到
Adapter或Selector/Prompt。** 历史0.5058的原图组bootstrap区间跨0，跨训练seed稳定性仍未确认。

## 下一步

不补HF seed1/2，不访问test。若继续定位，优先做严格保留历史P0前向的Full完整私有Adapter对照：
H0必须继续使用原“特征融合后过head”的代码路径；实验组只让Full使用完整b32私有Adapter，Body/Face仍共享
b32，Selector/Prompt/head和全部训练协议不变。若Full和融合同时恢复，再补等参数量共享容量对照；若仍失败，
再转向Full Selector/Prompt解绑。这样可避免本轮等价改写带来的AMP基线漂移。
