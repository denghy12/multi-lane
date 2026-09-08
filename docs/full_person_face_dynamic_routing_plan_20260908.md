# Full–Person–Face 样本级动态路由实施计划

日期：2026-09-08。状态：设计完成，业务实现与实验尚未启动。

## 1. 研究目标与当前决定

目标是根据每个样本的视觉内容、视图质量与可用性，选择 Full、Person、Face 分支并学习融合权重，
在 EMOTIC 类增量学习中提高完整8-task final mAP，同时保持旧任务路由稳定。

停止共享 Person CLS 引导 Selector，以及当前 selector-specific query-only 路线。此前安排的四组
Person value/deep residual 实验不再自动启动；其深层任务分支思想纳入本计划最后的统一融合阶段。
保留 legacy Full crop、letterbox Person 和已验证的 Image-token Adapter 作为固定基础。

Full/Person/Face来自同一RGB图像，论文宜称多视图或多尺度，不应描述为三个独立传感模态。
Face提供表情视觉特征，不进行身份识别，也不把表情直接当作人物真实心理状态。

### 已有证据

| 已完成方案 | 指标/结论 | 对下一步的约束 |
|---|---|---|
| 单Full冠军seed0 | test final mAP 32.5365 | 保留原训练和输入路径 |
| 同seed Full+letterbox Person | seed0 33.2672；三seed 32.8263±0.4025 | 是新增Face的直接成本参照 |
| Full+初始Person crop | seed0 33.1429 | Person固定letterbox |
| legacy+Person patch query | 31.6826，低Full 0.8540 | 不再只用人物改变query |
| target-aware disabled/patch | 30.5018/30.2505 | 不替换legacy Full crop |
| 旧两路学习门控 | 八候选三seed validation均未胜固定0.20 | 不重复仅靠低维统计量的小网格 |
| 旧门控task6 calibration | 51条、三类正例22/24/17 | 小样本下的负结果不能直接否定全部路由 |

以上test和validation分开报告；历史test已多次用于探索，后续锁定test属于同一基准上的确认，
不宣称恢复成从未观察过的独立测试集。三种子也不能消除历史选参偏差。

## 2. 全阶段不变的基础协议

- EMOTIC注册的8-task类别顺序、sample ID和官方split，seed0筛选，seed1/2确认。
- 每task 30 epochs；batch64；main Adam LR0.0125、WD0；per-task cosine至0、无warmup。
- Image-token Adapter：zero-based layer1、b32、LR4e-4、scale0.1、ReLU、independent。
- 主模型BCE，Adapter ASL gamma_neg9.8/gamma_pos0/clip0.05；AMP/TF32开启。
- Full：legacy crop0.05–1.0和CLIP normalization；Person：bbox margin0.15+letterbox224。
- Face首先沿用同样CLIP和Adapter配置作为对照，不声称这些参数已对Face最优。
- loss标签只使用当前task允许的类别，保留原legacy_full_zero主路径；新分支不得偷看未来标签。
- final validation mAP为第一排序指标，average mAP第二，task5–7均值第三。
- 类别AP、cF1/oF1、forgetting用于诊断，不恢复逐类不下降的硬门槛。
- 数据源、代码提交、预训练权重、manifest及结果文件都记录SHA-256。
- 新阶段开始前给出实际入口、全部参数、输入/输出/日志路径、checkpoint来源与预期产物。

计算预算以成功optimizer updates、GPU时、峰值显存和每样本推理成本报告。Face有效样本可能少，
若只在有效脸上训练，其updates应按实际样本计算，不能要求等于Full的13,950或声称等计算量。

## 3. 阶段0：数据与接口准备

建议实现分支：`codex/face-manifest`。

### 3.1 离线人脸检测与目标匹配

1. 每张原始图像只检测一次，使用冻结的检测模型；固定版本、权重SHA、输入尺寸和检测阈值。
2. 仅使用人脸框/关键点/置信度。CocoER demo使用InsightFace buffalo_l；具体运行依赖、权重来源及
   使用许可在实现时核对，不需要启用身份embedding模块。
3. 使用已有EMOTIC目标人体框，不重跑人体检测改变已有标注。
4. 候选匹配依据：中心落在目标框、脸框面积包含比例、相对头部位置和检测置信度。
   所有判断均为二维；多人情况比较候选得分，并对近似同分标记ambiguous，不沿用仅横坐标匹配。
5. 不强制脸永远位于人体框最上方；坐姿、弯腰、卧姿用软位置先验，避免几何硬规则大量漏检。
6. 第一轮只在train按固定抽样检查裁剪质量后锁定检测/匹配规则；val用于覆盖率统计，test留到最终导出。
7. 无脸、低分、极小脸、歧义分别记录原因。初版不自动用人物上1/3代替真实脸，也不生成补全脸。

建议manifest字段：schema_version、split、sample_id、image_id、person_bbox、face_bbox、score、
landmarks（若有）、original_face_size、match_score、ambiguous、face_valid、invalid_reason、
detector_revision、checkpoint_sha256、matching_revision。

建议路径：`output/emotic_face_manifest/<version>/{train,val}_faces.jsonl`，
`audit.json`、`audit.md`与抽样可视图存同目录；日志放`logs/emotic_face_manifest/`。
图片裁剪可按需读取，优先缓存元数据，避免重复存完整图片。

### 3.2 Face transform与缺失处理

- 初始工程默认：face bbox四边各外扩原框宽/高的15%，裁至图像边界，pad-to-square后Resize224。
  明确margin定义，不把总宽增加15%与每边15%混用；第一轮不搜索margin。
- CLIP normalization；共享水平flip；轻量增强沿用Person已实现策略，隔离RNG。
- 不做随机方形二次裁剪；缓存原图坐标，再统一转换，不从已裁掉目标的Full tensor中截Face。
- 单独维护person_valid、face_valid与Full crop可见率：Full里看不到目标，不等于原图Face不可用。
- 无脸可用占位tensor组batch，但在loss、attention和最终融合处都必须被mask；屏蔽投影bias。
- 支持全batch无脸、单样本无脸、padding全空，避免全负无穷softmax和空batch归约NaN。

### 3.3 验收

- stable ID逐split全覆盖，重复/错位为零，所有合法框非空且位于图内。
- train固定抽样至少200个目标人工/图像审阅，覆盖多人、小脸、侧脸、遮挡、检测失败。
- 输出按task的有效率、原始尺寸分位数、歧义率及当前类有效脸正例数。
- 不设拍脑袋的统一覆盖率成功线；若某task没有有效脸或某类无有效正例，必须报告并关闭相应监督。
- 测试Face RGB通道、坐标/flip、缺失mask、Full像素及RNG不变性。修复复用路径里的空patch mask bug。
- 先做真实数据smoke和GPU smoke，尚不训练完整模型。

## 4. 阶段1：Face是否提供增量信息

建议分支：`codex/face-endpoint-validation`。先做seed0完整8-task validation。

### 4.1 端点与公平对照

- F：Full冠军端点；P：letterbox Person端点；H：新Face端点。
- F/P已有100%train、同协议且有实际概率/稳定ID的完整val结果可复用；逐字段审计后记录来源。
- H用当前task有效Face训练，不删除其他视图的无脸样本。最终validation必须覆盖完整原始样本池。
- Face有效子集单独指标只作诊断；不得和全量Full mAP直接比较。
- 增加H后初始比较固定规则：FP=0.8F+0.2P；FPH=(1−beta)FP+beta H。
  无脸时精确回退FP。beta只在validation选取{0,0.05,0.10,0.20}，0即FP锚点。
- 此处小型静态对照只用于测Face增量价值，不恢复之前无止境的静态比例搜索。

### 4.2 输出与判断

- config、task_metrics、training_history、seed_summary和每task实际概率、logits、targets、IDs、face_valid。
- 全量final/average mAP、有效/缺失Face分组、逐类增益、错误相关性、低分辨率与多人子集表现。
- 只有完整样本池的FPH final validation mAP超过FP，才优先推进Face动态融合。
- 若未超过：先区分匹配质量、有效率、Face训练不足和真实互补性不足。最多做一次针对明确问题的修正；
  无明确原因不展开Face层数/LR网格。Face弱但在可信子集互补时，可保留后续软门控诊断，标明证据较弱。
- 此阶段不运行test。所有beta选择记录写入selection JSON，禁止在test再调整。

## 5. 阶段2：固定三路端点，验证样本级路由

建议分支：`codex/three-view-router-validation`。
目的：把“路由是否有用”与“联合训练改变了专家”分开。

### 5.1 数据协议与复用边界

- 初版沿用稳定image-group哈希的90%fit/10%calibration，三个视图、所有seed使用同一分桶。
- 基础专家只用fit训练，router只用当前task calibration训练，validation只选配置。
- 历史90%fit F/P端点可在来源一致时复用；100%train端点不能充当本阶段配对锚点。
- 若要输入深层特征，必须由对应task compact state重新导出；只有概率文件不能恢复特征。
  缺compact时如实计入重训成本，不混用task7特征冒充早期任务特征。
- 保存小型、预定义维度的特征描述，避免在51条calibration上训练大容量视觉router。
- 此阶段低样本限制仍存在。若明显过拟合，备用方案为按图像组3-fold out-of-fold路由训练：
  每条训练预测来自未见该图的模型；所有对照遵守相同协议。额外成本较大，仅诊断支持时启动，
  不把扩大MLP当作解决办法。

### 5.2 路由输入、输出与增量约束

- 输入：三路低维特征描述、可用性、人物框面积/长宽比、脸原始尺寸/score/匹配质量、
  当前task三路预测的置信度/熵/差异。质量特征缩放参数只在当前task calibration拟合并冻结。
- 不输入未来类别logits/标签、test统计量或身份特征。不同task的3/5类输入用固定维统计避免宽度变化。
- 每个task一个小router，建议hidden16、单隐藏层；共享的是架构，参数/标准化统计逐task冻结。
- 每个sample、每个task输出三路权重；同task类别共享权重。明确这不是全26类统一一组三路权重。
- 初版权重用masked softmax，Face无效权重严格为0；至少Full可用，避免分母0。
- 当前task训练当前router；推理执行全部已见task路由并拼接各task类别，无需真实task ID。

### 5.3 先软后稀疏的实验矩阵

第一轮固定同一批专家：

| 组别 | 融合 | 用途 |
|---|---|---|
| R0 | FP固定0.8/0.2 | 两视图锚点 |
| R1 | FPH全局固定权重 | 三路新增信息对照 |
| R2 | 样本级软路由，仅质量/预测统计 | 对应旧门控思路的三路版本 |
| R3 | 样本级软路由，加入低维视觉描述 | 验证语义输入价值 |

R1 beta仍只用{0,0.05,0.10,0.20}。R2/R3权重可覆盖整个可用单纯形，不再限定Person0.10–0.35。
工程初始值：AdamW LR1e-3、WD1e-4、batch64、80 epochs/task、hidden16；固定轮数不按test早停。
可各测试prior强度{0,0.1,1}共6个router候选，prior为与固定起始权重的平方距离；其余参数不搜索。
起始有效Face权重(0.72,0.18,0.10)，无脸(0.8,0.2,0)；起始prior不是必须保持的最终比例。
先评估R3是否超过R1及R2，再对胜出软路由追加一组masked sparsemax路由，保持其它条件相同。
固定top-2不能表示三路都选，因此不作为默认。

### 5.4 验收与失败判据

- 必须报告全8-task曲线、不同质量分组增益、权重均值/std/分位数、稀疏支持集频率。
- R3优于R1才能称动态融合收益；R3优于R2才支持视觉描述的价值。
- 同时看calibration/validation损失差；训练损失降低不保证mAP提高。
- 权重几乎恒定且mAP不提升：停止扩大router，保留固定融合。
- 稀疏不如软路由：保留软权重，不能为了展示“选两路”牺牲主要指标。
- sample-wise oracle仅可在validation事后用于上界诊断；真实标签不能用于推理选择或当部署算法。

## 6. 阶段3：统一的特征融合模型

建议分支：`codex/taskwise-three-view-feature-fusion`。优先条件：阶段1有Face互补，阶段2有路由收益。
若阶段2因calibration过小失败但互补证据明确，允许做本阶段最小试验，以train联合监督检验，
须明确是假设检验，不能称路由已被证明有效。

### 6.1 模型与训练

- 共享冻结CLIP权重，各视图建立task独立Image-token Adapter、selectors/prompts和lane。
- 三路task CLS经过各自投影和归一化后，router根据内容和质量输出权重，融合后进入分类头。
- 新投影不得全部零初始化，否则全部融合特征为零；使用明确的Full主通路初始化，
  Person/Face以零输出残差接入，或先训练固定特征融合锚点。零初始化等价性写入测试。
- 若希望权重可解释为贡献，应固定/记录各路尺度。保留Full残差主通路时权重仅是修正系数，
  不能把Full权重0描述为实际移除了Full。最终支持真正关闭Full的版本采用归一化加权和并显式消融。
- 新融合和分类模块走当前task BCE；两路/三路Image-token Adapter继续ASL路由，加入梯度单测。
- Face无效时其辅助loss与融合贡献都为零。有效样本loss按有效样本数归约并固定系数，记录预算。
- 先5 epochs/task训练各路辅助分类、固定融合；后25 epochs打开router训练。两段合计30 epochs，
  cosine按完整30 epochs连续，不偷偷增加训练预算。阶段切换规则对全部配对组一致。
- 零输出末层会暂时阻断上游梯度，要在真实smoke检查后续更新中Person/Face分支梯度恢复。

### 6.2 最小对照

| 组别 | 特征融合 | 分支辅助监督 |
|---|---|---|
| J0 | 三路固定 | 有 |
| J1 | 三路软路由 | 有 |
| J2 | 三路软路由 | router开启后关闭辅助监督 |
| J3 | 两路Full+Person软路由 | 有 |

辅助loss初始总权重0.1，三路按可用分支平均；J1−J0测路由，J1−J2测辅助监督，J1−J3测Face。
J0/J1使用同一个预声明固定权重初始化。J3作为少一路成本对照，不能声称与三路等算力。
额外报告阶段2概率融合最佳作为参考，不能将不同训练流程的差值单独归因于特征融合位置。
只在软路由胜出后补稀疏版本、再考虑分支dropout；避免第一轮同时加所有机制。

## 7. 阶段4：稳定性、归因与一次锁定test

1. seed0选出一个结构/配置；补seed1/2完整validation，并为每个seed补同协议固定对照。
2. 主要标准为三种子平均final mAP超过匹配对照；完整报告逐seed差值及标准差，不要求每个类都提高。
   若只一个seed明显有效，先报告不稳定，不直接宣布最佳方法。
3. 固定sample ID集合测旧类概率漂移、固定集合AP；同时保留原动态评估池forgetting。
4. 用移除/替换某分支的干预检查路由：高Face权重组移除Face后是否更受损。
   权重大并不自动证明因果贡献；分支高度相关时尤其如此。
5. 补不同seed Full专家等预算对照时明确其随机性；同seed同数据完全重复Full不能用于否定集成收益。
6. 锁定模型、权重规则、阈值0.5、Face检测/匹配版本和所有超参数后，三seed各一次test评估。
   主指标面向全样本池；Face有效子集只作补充。缺失Face精确回退已经锁定的两路规则。
7. 对照匹配协议的Full+Person三seed；历史32.8263±0.4025仅在协议一致时直接比较。

## 8. 代码工作包、验收与产物

以下文件是拟新增项，不代表已经实现。实际路径可随代码审查调整并记录。

| 工作包 | 拟改动/新增 | 必要验证 |
|---|---|---|
| W0 人脸数据 | face_manifest.py、audit_face_manifest.py；paired_transforms.py | ID/匹配/缺失/RGB/Full RNG |
| W1 Face端点 | model.py、runner.py、Face validation launcher | loss掩码、ASL路由、全量评估 |
| W2 特征导出 | evaluation_scores.py、compact恢复入口 | 原概率精确恢复、task来源一致 |
| W3 离线路由 | three_view_router.py、router selection入口 | split隔离、权重和1、缺脸0、旧router冻结 |
| W4 统一模型 | feature_fusion.py、model.py、runner.py | 初始化基线、梯度、旧状态不变、全无脸batch |
| W5 汇总锁定 | summarize_three_view.py、locked test入口 | 拒绝test搜索、哈希和完整预算 |

代码参考：multi_lane/track_a/model.py、selector_conditioning.py、paired_transforms.py、
learned_reliability_gate.py、evaluation_scores.py、export_compact_test_scores.py。
复用旧统计/数据接口，不复用旧共享router的持续更新假设。

各阶段结果统一写`output/emotic_three_view/<stage>/<batch>/<run>/`，日志写
`logs/emotic_three_view/<stage>/<batch>/`。保存JSON、实际概率NPZ、必要router/compact state、
selection、manifest与analysis。Full冻结权重不重复放入compact state。
需要后续test推理的阶段保留逐task compact state；不能一边禁止保存所有状态，一边计划无需重训的锁定test。
小结果包排除完整checkpoint，compact state单独打包并给清单/SHA。分析数据可能含多人图像，按image_id
分组做bootstrap，不把同图人物当作完全独立样本；区间仅作为不确定性辅助，三seed统计也有局限。

## 9. GPU调度、Git与执行顺序

- 文档阶段不启动训练。每个工作包开始前建立相应`codex/`分支，保留现有用户改动。
- 本地提交/push，服务器审计所有worktree状态，新独立worktree fetch/ff-only；test-only保持不动。
- 完整单测、真实数据smoke和GPU smoke通过后才能启动已声明阶段。
- 阶段0检测先单卡小批测吞吐；阶段1端点一组一卡；阶段2端点按缺失组安排，离线路由CPU或单卡。
- 统一三分支的显存不能照搬旧2GB单分支估计。先测包含全部已见lane的task7真实batch峰值，
  并检查外部进程和安全余量，再决定一张4090能否放两组。
- 不为了填满8张卡扩无证据网格。独立配置/seed可并行；数据阶段和验收依赖顺序执行。
- 运行前打印完整配置，失败写明确退出码；完成后核验8-task和update预算再同步，逐文件SHA验证。
- Automatic Upload应关闭；若发现它同步受控代码，先辨明并备份，不覆盖未知用户改动。

## 10. 下一次具体执行清单

- [ ] 建立`codex/face-manifest`分支，核对当前可用数据与检测权重。
- [ ] 实现train/val离线检测、二维人物匹配、版本化manifest与质量审计。
- [ ] 增加Face letterbox/mask与稳定ID接口，检验Full/Person路径不变。
- [ ] 导出至少200个train抽样匹配可视图，修正明确的裁剪/匹配缺陷并锁定规则。
- [ ] 汇报覆盖率、每task有效正例和原始脸分辨率，决定阶段1端点训练是否具备条件。
- [ ] 阶段1通过后再启动阶段2，阶段2不自动触发test。

## 11. 研究依据与表达边界

- CocoER本地inference.py的FaceAnalysis/buffalo_l、人体/人脸匹配和raw_head crop提供实现参考；
  dataset.py直接读取head_arr，未公开的训练裁剪生成细节不能自行补成论文事实。
- CocoER models_sw.py三路特征与内容交互支持多尺度建模的动机；其全类别和推理时更新协议
  不直接移植到本项目的类增量设定。
- SimMLM（ICCV2025）：https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.pdf
  支持动态专家融合与缺失输入的设计参考，不证明本方案在EMOTIC必然提升。
- Confidence-Guided Gate：https://arxiv.org/abs/2505.19525，提供专家塌缩和可靠性路由的研究参考。
- 最终论文贡献需由实验支持：Face新增信息、样本路由增益、缺失视图鲁棒性、旧任务稳定性。
  仅增加Face和MLP不足以声称新颖性；严格文献对比在得到机制结果后另做，不承诺必然提升或全局最优。
