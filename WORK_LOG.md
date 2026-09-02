# 工作日志

最后一次更新：2026-09-01。

## 2026-09-01：阶段一完成并启动阶段二ASL局部8组正式test

- 阶段一8组均完成240 epochs、13950 updates、skipped0，无运行异常。同批
  LR4e-4/scale0.1精确复现single1历史冠军final/average mAP`32.5365/39.1445`并排名第一。
- 次优LR5e-4/scale0.075为`32.3851/39.2462`，average高`0.1017`但final低`0.1514`；
  task0--4略高、task5--7下降，Sadness提高约`4.18`而Suffering下降约`4.50` AP。
  LR6e-4两组至少低`0.4977`，不扩展7e-4；阶段二固定LR4e-4/scale0.1。
- 阶段一40个小文件已同步，无checkpoint。180KB archive两端SHA-256均为
  `403004fba2f2244dc9c1f630c1f1490f89cc256dfe734d808a9cda64a941bc89`；分析文件为
  `output/emotic_track_a_image_token_asl_layer1_lr_scale_formal/20260901_171236/analysis.md`。
- 阶段二按用户要求继续直接report test。批次
  `image_token_asl_layer1_asl_local_formal_seed0_20260901_182542`包含锚点9.8/0/0.05、
  gamma-neg 8/12、clip0.0375/0.075及gamma-pos0.25/0.5/1.0共8组；固定layer1/b32/
  LR4e-4/scale0.1/ReLU/independent、AMP/TF32 on、无checkpoint，GPU0--7各一组。

## 2026-09-01：启动layer1 Adapter LR×scale阶段一8组正式test

- 用户明确validation与test耗时相近，要求跳过validation并直接并行全部8组held-out test。
  固定zero-based layer1、b32、ReLU、independent、主模型BCE + Adapter ASL 9.8/0/0.05、
  seed0、8 tasks×30 epochs、batch64、CLIP normalization、crop0.05、AMP/TF32 on、无checkpoint。
- 组合为LR4e-4×scale`0.075/0.1/0.125`、LR5e-4×三种scale、LR6e-4×scale`0.075/0.1`；
  `4e-4/0.1`作为同批冠军锚点，高风险`6e-4/0.125`未运行。
- 批次ID为`image_token_asl_layer1_lr_scale_formal_seed0_20260901_171236`。两次SSH握手被服务器
  reset，端口与密钥复核成功后才执行状态检查，未产生半启动目录。
- GPU0--7启动前均空闲、无runner；clean实验worktree为`94f3327`。8组各占一张GPU，均已生成
  config并开始初始化；每卡约1.8--2.5GB、最低仍余21.6GB，无OOM或启动异常。

## 2026-09-01：同步5组每层b32多层结果，多层路线收口

- 5组均完成240 epochs、13950 updates、skipped0，日志无OOM、非有限值、Traceback或写盘
  错误。最佳`[1,4]` b32/s0.05的final/average mAP为`32.2213/38.9746`，相对single1冠军
  低`0.3152/0.1699`；task1--7均下降，task6/7低`0.2804/0.3152`。
- `[0,1]` b32/s0.05、三层`[0,1,2]` b32/s1/30、`[1,8]` b32/s0.05、`[1,2]`
  b32/s0.1的final mAP依次为`32.0605/31.9493/31.7306/31.6235`，均未超过冠军或首批
  多层最佳`[1,2]` b16/s0.05的`32.3233`。
- b32相对小容量对应组的final变化为`[0,1] +0.1816`、`[1,4] -0.0277`、
  `[1,8] -0.3211`、三层`+0.1629`、`[1,2]` s0.1 `-0.1307`。容量效应没有一致方向，
  无法形成可推广的多层收益。
- 5组25个小文件已单独同步，不含checkpoint；112KB archive两端SHA-256均为
  `72995ac436e75d0f9d51eff3da2c0c1c0823cbc8051b47c4b13d3b7e8eb37698`。详细分析见
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_b32_extension5/analysis.md`。
- 服务器已无本批runner。20组统一结论为固定single1 b32/LR4e-4/scale0.1；停止多层、
  bottleneck和Adapter LR扩展。当前未启动新实验。

## 2026-09-01：首批15组完成并追加5组每层b32多层test

- `image_token_asl_layer1_local_multilayer_formal_seed0_20260901_100337`首批15组均完成240
  epochs、13950 updates、skipped0。layer1局部网格最佳b32/LR3e-4 final mAP`32.1281`，
  其余b24/b28/b40组合为`31.5292--32.0788`，均未超过已完成的single1 b32/LR4e-4
  冠军`32.5365`。single7 FP32补测为`31.7931`，也未取胜。
- 多层归一化最佳为`[1,2]` b16/scale0.05的`32.3233`，其次`[1,4]` b16/scale0.05
  `32.2490`；`[1,8]`为`32.0516`、`[0,1]`为`31.8789`、三层b12为`31.7864`。
  `[1,2]` b32/scale0.05仅`31.7764`，而b16/scale0.1为`31.7542`，说明容量或总残差强度
  放大都没有改善该相邻双层。
- 用户要求多层再补每层b32版本。新增5组：`[0,1]/[1,4]/[1,8]` b32/scale0.05、
  `[0,1,2]` b32/scale1/30、`[1,2]` b32/scale0.1；统一FP32/TF32 on、LR4e-4、正式test、
  无checkpoint。它们用于补齐跨深度组合和容量/强度上界，不改变已完成的归一化结论。
- 三层b32 FP32/TF32-off严格smoke通过，可训练参数839034、零初始化最大差`1.49e-8`。
  5组已在GPU0--4启动，20个总配置目录均已生成，初始核验无OOM、非有限值或Traceback。
- 首批15组已独立同步本地：75个小文件、无checkpoint，压缩包328KB，两端SHA-256均为
  `b6eb43e335ec0031f36d022ac701ff80aba0f78c29be27af393e906221169cbb`。新增5组仍在服务器运行，
  没有混入该包。
- 详细逐task与类别分析确认：最佳多层`[1,2]` b16/s0.05相对single1在8个task全部下降，
  final/average mAP低`0.2132/0.2309`，仅forgetting改善`0.1205`；single7 FP32在task6/7低
  `0.7912/0.7434`。分析文件为
  `output/emotic_track_a_image_token_asl_layer1_multilayer_formal/20260901_100337_initial15/analysis.md`。

## 2026-09-01：并行启动single7 FP32、layer1局部网格和归一化多层test

- 用户要求直接并行三条held-out test路线。批次
  `image_token_asl_layer1_local_multilayer_formal_seed0_20260901_100337`共15组，统一为seed0、
  8 tasks×30 epochs、batch64、主模型BCE + Adapter ASL 9.8/0/0.05、ReLU、independent、
  CLIP normalization、crop0.05、TF32 on、无checkpoint。
- single7使用FP32、layer7/b32/LR4e-4/scale0.1，补跑此前AMP非有限失败的高validation候选。
  layer1局部网格使用AMP，运行bottleneck`24/28/32/40` × LR`3e-4/4e-4`中除已完成
  b32/LR4e-4冠军外的7组。
- 多层使用FP32并围绕layer1设计7组：`[0,1]/[1,2]/[1,4]/[1,8]`采用每层
  b16/scale0.05，使总参数量和总残差强度近似single1 b32/scale0.1；增加`[1,2]`
  b32/scale0.05与b16/scale0.1两个解耦控制，以及`[0,1,2]` b12/scale1/30三层归一化。
- 初始FP32/TF32-on smoke被纯FP32容差拦截，单层/双层/三层最大差约`1.1e-5--2.2e-5`；
  判定为TF32重复前向数值差而非多层特有错误。FP32/TF32-off严格复测中single7、四种双层及
  三层全部通过，最大差约`6.52e-8`；正式训练按用户要求保持TF32 on。
- 服务器主实验worktree为clean`94f3327`，8张GPU启动前均空闲。GPU0--6各两组、GPU7一组；
  15组全部生成config并进入task0。各卡总显存约2.6--5.5GB，最低仍余18.6GB；启动核验没有
  OOM、非有限值或Traceback。当前不持续盯跑，完成后同步JSON与日志分析。

## 2026-09-01：同步层位置正式test，layer1成为探索性新冠军

- 批次`image_token_asl_layer_formal_seed0_20260831_154234`已结束：single1/2/3/4/5/11均完整
  完成240 epochs、13950 updates、skipped0；single7在task3 epoch16后因非有限ASL logits
  终止；pair`[2,3]`完成240 epochs，但task3 epoch17--29共跳过91次AMP optimizer update。
- zero-based single1的final/average mAP为`32.5365/39.1445`，相对layer8旧冠军
  `32.4193/38.8789`提高`0.1172/0.2656`；cF1/oF1提高`0.4167/0.2811`，forgetting降低
  `0.0934`。除层索引外配置逐字段相同，8个task mAP均高于layer8，因此它是当前正式指标的
  探索性新冠军。
- single1的收益集中于task4/5（`+0.5355/+0.6315`），task6/7仅`+0.0329/+0.1172`；最终
  Fear/Fatigue/Aversion等改善，但Suffering/Sadness下降`7.0468/4.9255` AP，仍有类别交换。
- 完整候选其余排名为single4`32.0577`、single5`31.9281`、single2`31.9014`、single3
  `31.3633`、single11`31.0015`、pair`[2,3]``30.8747`。多层既无总mAP收益又有AMP跳步，
  不再增加层数。
- 旧FP32 validation与新AMP test在6个完整单层上的final mAP Pearson相关仅`-0.273`；
  layer3的validation第一没有外推到test，数值模式与split共同造成排序反转。single7的失败
  也表明AMP对ASL层位置并非无关扰动。
- 结果已同步至本地，39个小文件、不含checkpoint。192KB archive在服务器与本地SHA-256均为
  `d8e22c7c8cab09af697ce9c705530af5446c7de51b168d2b3049037cf46380c4`。分析见
  `output/emotic_track_a_image_token_asl_layer_formal/image_token_asl_layer_formal_seed0_20260831_154234/analysis.md`。
- `+0.1172`是从多个held-out test候选中筛出的探索性小效应，存在多重比较乐观偏差；不能作为
  稳定或显著提升。当前未启动新实验。

## 2026-08-31：审计层位置结果并排队8组正式test

- 重新核对历史产物后确认：Image-token Adapter单层`0--11`搜索、15组多层screen以及
  single8/single9/pair8_9 BCE确认均为seed0完整8-task validation-only，此前没有对
  layer3/layer7/layer11或多层结构运行held-out test。
- validation与test使用相同训练协议但不同评估split，并非互不相关；此前不同批次validation
  波动约`0.4--0.6` mAP，而且层搜索使用AMP off/TF32 on、当前正式冠军使用AMP/TF32 on，
  因此validation排序不能保证test排序。
- 按用户当前“只追最终test mAP、个别类别可下降”的目标，取消旧的类别/F1/forgetting否决
  门槛。单层ASL同批超过disabled的候选为zero-based layer`1/2/3/4/5/7/11`；其中layer3、
  layer7相对同批layer8 final validation mAP分别高约`1.1007/0.8227`。多层仅保留各自同批
  超过disabled且最高的`[2,3]`，其余多层ASL不进入test。
- 8组共同配置为EMOTIC seed0、8 tasks×30 epochs、batch64、主模型BCE + Image-token
  Adapter ASL 9.8/0/0.05、b32、Adapter LR4e-4、scale0.1、ReLU、independent、CLIP
  normalization、crop0.05、AMP/TF32 on、reporting test、无checkpoint。
- 批次ID为`image_token_asl_layer_formal_seed0_20260831_154234`，输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_asl_layer_formal_v0.1/`。
  启动时8张卡均被另一批进程占用、仅余约6--10GB；已为GPU0--7各挂一个等待器，要求空闲
  显存至少20GB且连续两次60秒检查通过后才启动。8个等待器均已核验存活，当前未抢占显存。

## 2026-08-31：直接启动 b32 附近容量/LR 8组正式test

- 用户明确validation与test耗时接近时可跳过validation，直接追求最高正式test mAP。因此固定
  当前最优b32/LR4e-4/scale0.1/ReLU协议，将bottleneck`24/28/32/40` × LR`3e-4/4e-4`
  全8组直接运行seed0 held-out test，不先运行validation。
- 批次ID为`image_token_asl_local_capacity_lr_amp_formal_seed0_20260831_124751`，GPU0--7各一组。
  共同配置为8 tasks、30 epochs/task、batch64、layer8、scale0.1、ReLU、independent、主模型BCE +
  Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、AMP/TF32 on、无checkpoint。
- 启动前8卡均有其他高利用率任务，但仍余8.6--11.8GB显存；按用户要求不等待空闲，不停止
  其他进程，每卡追加一组。启动后最低仍余约6.2GB，未接近OOM。
- 8组task0已到epoch3--4，均84 steps、skipped0，单epoch约15--17秒，无OOM、NaN、Inf、
  Traceback、RuntimeError或写盘错误。服务器实验worktree为clean`94f3327`。
- 8组随后均完成240 epochs、13950 updates、skipped0，完成标记各唯一，日志无OOM、NaN、Inf、
  Traceback、RuntimeError或写盘错误。同批b32/LR4e-4精确复现历史冠军全部主指标和曲线，
  final mAP为`32.4193`，仍是第一。
- 排名2--8为b24/LR3e-4`32.2492`、b28/LR4e-4`32.1666`、b32/LR3e-4`32.1172`、
  b40/LR4e-4`31.9602`、b28/LR3e-4`31.8790`、b24/LR4e-4`31.2931`、b40/LR3e-4
  `31.1764`；没有新冠军。
- b24/LR3e-4虽final mAP低`0.1702`，但average mAP高`0.2668`、task5--7平均高`0.0104`。
  组别分解显示它改善多个早期任务，但task6组均低`3.5030`，Suffering低`8.2892`，因而最终落后。
- 41个小文件已同步本地，archive 180KB，SHA-256为
  `acc4e0682ecc6ff170dc92e1663dafb4d94e97cc5aa7d56c03696b440dc1a978`，不含checkpoint。统一容量/LR
  搜索基本收口；下一步应转ASL局部微调或task-dependent Adapter超参。

## 2026-08-31：并行启动 b16 AMP 正式test与 scale×activation 搜索

- 用户要求不等b16正式结果，直接利用8卡同时完成一组b16 AMP/TF32-on held-out test和
  8组b32 scale×activation完整validation。为对齐历史冠军，validation也使用AMP/TF32 on。
- 共同配置为seed0、8 tasks×30 epochs、batch64、layer8、LR4e-4、independent、主模型BCE +
  Image-token Adapter ASL 9.8/0/0.05、CLIP normalization、crop0.05、无checkpoint。formal使用b16、
  scale0.1/ReLU/reporting test；validation固定b32，搜索scale`0.025/0.05/0.1/0.2`和ReLU/GELU。
- 启动前GPU0--7全空闲、无残留runner，服务器实验worktree为clean`94f3327`。初次SSH握手
  被重置两次，端口和密钥复核后恢复，未产生半启动任务。
- formal batch为`image_token_asl_b16_amp_formal_seed0_20260831_114209`，validation batch为
  `image_token_asl_scale_activation_amp_seed0_20260831_114209`。GPU0同时运行formal和一组validation，
  GPU1--7各一组；共9个tmux会话。
- 9组task0 epoch2均完成84 steps、skipped0，GPU0双进程约5.2GB，其他卡约2.4GB，无
  OOM、NaN、Inf、Traceback、RuntimeError或写盘错误。预计约45--55分钟完成。
- b16 AMP/TF32-on formal完整完成240 epochs、13950 updates、skipped0，final/average mAP为
  `31.3796/38.8129`，未超过历史b32冠军`32.4193/38.8789`。它在task0--4略高，但task6/7
  低`1.0856/1.0397`；因而排除“只对齐AMP/TF32就能让b16取胜”的假设。
- scale×activation中`0.1/GELU`、`0.2/ReLU`、`0.2/GELU`在task3因非有限ASL logits终止，
  其余5组完成240 epochs、13950 updates、skipped0。原`0.1/ReLU`在final/average/task5--7平均mAP
  三项均第一，为`42.6673/49.2378/44.8851`；其他有效组final mAP低`0.5031--0.7962`。
- 结论是保留b32/LR4e-4/scale0.1/ReLU/AMP+TF32，不运行新正式test，不继续扩scale或修复
  GELU。下一局部搜索候选为bottleneck`24/28/32/40` × LR`3e-4/4e-4`。
- 43个小文件已同步本地，archive为160KB、SHA-256
  `29c87d99ddcc868bd0c1989ec921d5c22c20a36c9a6a6c77c9d8220254241461`，不含checkpoint。

## 2026-08-31：启动 b16/b64 LR4e-4 正式test

- 用户明确目标改为尽快最大化正式test final mAP，不再以Sadness、Sensitivity、
  Suffering、F1或forgetting作为候选否决门槛。阶段一validation最高的b16/LR4e-4
  与次高b64/LR4e-4进入一次性seed0 held-out test。
- 服务器GPU0--7启动前均空闲；clean实验worktree为
  `/mnt/haoyuan/workspace/multi-lane-main-asl-routing@94f3327`。b64真实CLIP FP32/TF32-off
  Adapter-ASL smoke通过，可训练参数788314。
- 两组于00:35在GPU0/1并行启动，batch ID为
  `image_token_asl_capacity_formal_seed0_20260831_003546`。共同配置为8 tasks、30 epochs/task、
  batch64、layer8、Adapter LR4e-4、scale0.1、ReLU、independent、主模型BCE + Adapter ASL
  9.8/0/0.05、CLIP normalization、crop0.05、AMP/TF32 off、test reporting；仅bottleneck为16/64。
- 服务器磁盘显示100%；为避免历史正式运行每组约2.7GB checkpoint导致中途写入失败，
  本轮显式关闭checkpoint，预计每组仅约150KB JSON外加日志。根目录在root保留块上的
  16MB写入测试通过。
- b16/b64 task0 epoch1均完成84 steps、skipped0，loss为`0.614753/0.614830`，Adapter loss为
  `0.028270/0.028179`；GPU显存约3.4/3.0GB，无OOM、NaN、Inf、Traceback或RuntimeError。
- 两组后续均完成240 epochs、13950 updates、skipped0，完成标记各唯一且日志无数值、显存或
  写盘错误。b16的final/average mAP为`32.0269/38.6674`，b64为`31.3193/38.4295`；
  两者均未超过历史b32冠军`32.4193/38.8789`。
- b16相对冠军final mAP仅低`0.3924`，cF1/oF1反而高`0.1317/0.2205`；b64的final mAP
  低`1.1000`。b16比b64高`0.7076` final mAP，容量64不再保留。
- 历史b32配置为AMP/TF32 on，本轮b16/b64为AMP/TF32 off；核心代码差异只增加非有限值
  检查和checkpoint开关，正常loss路径未改变。因此下一个最小对照应只b16对齐冠军精度模式。
- 11个小文件与两份日志已同步本地；压缩包48KB，SHA-256为
  `66ef7139375d2913830bc5683f26c378506c3e2430fc7f18b47d0b9bd74d1484`，不含checkpoint。
- 按用户批准对服务器checkpoint做了限定清理：只删除CocoER、旧Task Adapter Bank、
  EMOT-Net+CCIM和DSCT的8个明确失败/中止实验`checkpoints/`目录，保留日志、
  配置与指标。实际释放`6796956 KiB`（约6.48 GiB），8个目标均已复核不存在。
  删除未触及任何成功实验或当前b16/b64运行；两组正式test仍正常。ext4预留块使
  `df`仍显示100%，root可用的实际空间约139.3 GiB。

## 2026-08-26：准备 Image-token Adapter-ASL 容量/LR阶段一

- 用户明确固定主模型BCE、Adapter ASL，不再把Adapter-BCE作为主优化方向。从
  `exp/emotic-image-token-pair89-confirmation@4213310`创建
  `exp/emotic-image-token-asl-capacity-lr`。
- 阶段一预注册12个Adapter-ASL组合：zero-based layer8、scale0.1、ReLU、independent、
  ASL 9.8/0/0.05固定，联合搜索bottleneck`8/16/32/64`和Adapter LR
  `1e-4/2e-4/4e-4`；加入fresh Adapter-disabled joint-BCE锚点，共13组。
- 8卡launcher先并行8组，GPU0--4完成首组后自动接续剩余5组，一次启动即可完成阶段一。
  每组均为seed0完整8-task val-only、30 epochs/task、batch64、legacy、CLIP normalization、
  crop0.05、AMP off、TF32 off、无checkpoint；不运行正式test。
- 汇总器逐组核验同一clean commit/tree、240 epochs、13950 updates、skipped0、无checkpoint、
  精确Adapter参数量和ASL路由。候选需相对disabled与b32/LR4e-4锚点通过全部类别/F1/forgetting
  门槛，且final mAP至少提高0.5，才允许进入scale×activation阶段。
- worker已增加`NO_TF32`开关，smoke增加`--no-tf32`并输出实际TF32状态；新增b64最大容量
  Adapter-ASL smoke、13组自动队列、严格汇总器和3项测试。本地Shell语法、Python3.9编译及
  `git diff --check`通过；本机系统Python缺少PyTorch，完整测试待服务器`ddp`环境执行。
- 实现提交`94f3327`已推送并由服务器独立实验worktree安全切换；服务器完整Track-A测试42/42
  通过。GPU1 b64真实CLIP严格FP32 Adapter-ASL smoke通过，日志确认TF32 off、可训练参数
  788314、zero-init最大差`1.8626451e-08`。
- 13组队列在tmux`mla_image_token_asl_caplr_s0_20260826_164945`启动，batch ID为
  `image_token_asl_capacity_lr_seed0_20260826_164945`。首轮GPU0--7依次运行disabled、冠军
  b32/LR4e-4、b8三种LR及b16三种LR；8组均已完成至少task0 epoch4，84 steps、skipped0，
  ASL loss有限且无异常。
- 启动观察时每卡总显存约2.8--3.0GB、剩余21GB以上。GPU0--4首组完成后会自动接续b32低LR
  和b64三种LR；当前不运行正式test，实验worktree固定clean`94f3327`直至13组完成并汇总。
- 用户确认可以单卡并行多组时，首轮8组已到task4，因此不停止重跑。新增临时detached clean
  worktree`/mnt/haoyuan/workspace/multi-lane-main-asl-capacity-lr-parallel@94f3327`，把原计划
  第二轮5组立即分别加入GPU0--4；13组现已全部同时运行。
- 新增5组task0 epoch1均完成84 steps、skipped0，约24--25秒；GPU0--4双进程实际总占用
  5.9--6.1GB、剩余约18GB，无OOM或非有限ASL loss。launcher本地实现同步改为默认13进程
  同启，便于未来一键复现；当前批次完成后需手动运行严格汇总，避免原串行父launcher的重复
  目录检测影响结果判定。

- 13/13组随后全部完成并已同步到本地结果包；每组均为240 epochs、13950 updates、skipped0，
  无OOM、NaN、Inf、Traceback或RuntimeError。fresh disabled、b16/LR4e-4、b64/LR2e-4的
  final mAP分别为`42.0628/42.5718/42.3616`，但最高mAP候选牺牲Sensitivity并恶化
  forgetting；b32/LR1e-4虽通过锚点门槛，却相对fresh disabled压低task6的Sadness和Suffering。
  严格汇总无合格winner，不进入scale×activation、不扩大容量/LR、不运行正式test。详细结果与
  白名单归档见本地`output/emotic_image_token_tuning/asl_capacity_lr/...results_package_v2/`。
- 2026-08-30再次从服务器核验并同步同一batch，严格汇总内容与原分析完全一致。
  本地新归档为`..._results_20260830.tar.gz`（344 KB），SHA-256校验通过，含68个结果文件和
  19个日志，不含checkpoint。服务器实验worktree仍为clean`94f3327`。

## 2026-08-26：同步并分析 pair8/9 BCE 同批确认

- batch`image_token_pair89_confirmation_seed0_20260825_132139`四组全部完成；每组8 tasks、
  240 epochs、13950 updates、skipped0、exit code0，同一clean`e50b4a3`，无OOM、NaN、Inf、
  异常或checkpoint。
- disabled/single8/single9/pair8_9的final mAP分别为
  `41.863114/42.422405/41.615440/42.350681`，average mAP分别为
  `48.406695/49.309657/47.820751/49.008467`。single8是本批final/average mAP最高且
  forgetting最低的结构。
- pair相对fresh single9虽提高final/average/task6 mAP`0.735241/1.187716/1.020656`，但
  Sensitivity下降`1.210430`、forgetting恶化`0.216919`；相对disabled也使forgetting恶化
  `0.193886`。严格结果为`confirmed_for_formal_test=false`，不运行正式test。
- single8相对disabled在task0--7各阶段mAP均提高，final/average/task6提高
  `0.559291/0.902963/0.652256`且forgetting改善`0.093862`，但Sadness下降`5.071147`，仍未
  通过全部硬门槛。single9整体退化；停止single9、pair、多层、扩容与ASL。
- task6训练视图只有627个样本，三类正例为Sadness260、Sensitivity316、Suffering188；
  Sensitivity并非最稀有却只有约6--7 AP。pair将每task Adapter参数从49952加倍到99904后主要
  产生类别收益交换和更高forgetting，现象更符合小数据下的方差/过拟合而非容量不足。
- 28文件小结果包已同步本地并逐项SHA-256通过，不含checkpoint；archive SHA为
  `3e449853a5520b30d7c413a41a3283ddf332757065f1ac5bd33fbcb386982ad2`，详细分析见结果目录
  `analysis.md`。
- 配置复核发现此前日志中的FP32表示AMP关闭，但`config.json`实际为`tf32=true`；四组相对比较
  仍公平，下一轮若筛选约0.1--0.5点小效应，应先在launcher显式传`--no-tf32`并同批重建
  disabled/single8控制，避免继续混用“AMP off”和“严格FP32”概念。

## 2026-08-25：准备 pair8/9 BCE 最小同批确认

- 用户确认执行fresh disabled、single8、single9、pair`[8,9]`四组最小BCE validation；从
  `exp/emotic-image-token-multilayer-screen@1014831`创建
  `exp/emotic-image-token-pair89-confirmation`。四组使用GPU0--3同批一轮，GPU4--7不占用。
- 固定seed0、完整8-task val-only、FP32、30 epochs/task、batch64、legacy、CLIP normalization、
  crop0.05、b32、Adapter LR4e-4、scale0.1、ReLU、independent、无checkpoint；不运行ASL或
  held-out test。
- 新增4-GPU资源门控启动器和严格确认汇总器。pair除相对disabled的全部硬门槛外，还必须相对
  fresh single9保护final/average/task6 mAP、task6三个新类与forgetting，F1下降不超过0.5；
  只有全部通过才标记`confirmed_for_formal_test=true`。
- 本地Shell语法、Python3.9编译、3项隔离汇总器测试和`git diff --check`通过；完整Track-A测试
  与pair8/9真实CLIP smoke待Git同步服务器后执行。
- 实现提交`e50b4a3`已推送并安全同步到服务器独立实验worktree；服务器完整Track-A测试
  39/39通过。pair`[8,9]`真实CLIP FP32 BCE smoke通过，可训练参数789082，zero-init最大差
  `1.4901161e-08`，未发现冻结、梯度路由或数值异常。
- 四组已于2026-08-25 13:21在tmux`mla_image_token_pair89_s0_20260825_132139`启动，batch ID为
  `image_token_pair89_confirmation_seed0_20260825_132139`。GPU0--3分别运行disabled、single8、
  single9和pair8_9；四组task0 epoch1均完成84 steps、skipped0，耗时12.5--14.6秒，训练loss
  有限且日志中无OOM、NaN、Inf、Traceback或RuntimeError。
- 启动观察时每卡总显存占用约2.4--2.8GB、剩余21.2GB以上；GPU4--7保持空闲。实验worktree
  固定clean`e50b4a3`直至四组完成并自动生成严格确认汇总；当前没有ASL、其他多层或正式test。

## 2026-08-25：准备 Image-token Adapter 多层受控诊断

- 用户提出直接对layer3/layer11运行正式test并探索多层。根据上一阶段预注册规则，两者都不能
  视为预期稳健：layer3 ASL虽final mAP提高`0.976536`但Suffering下降`2.695190`；layer11
  ASL虽改善Sadness/Suffering但Sensitivity下降`1.126123`、forgetting恶化`0.300807`。因此
  暂不消耗held-out test，先做同批validation复核。
- 从`exp/emotic-image-token-structure-tuning@5ba055b`创建
  `exp/emotic-image-token-multilayer-screen`。设计15组8卡两轮：disabled，single3/single11
  的BCE/ASL，以及`[2,3]`、`[3,7]`、`[3,11]`、`[8,9]`、`[7,8,9,10,11]`五种多层的
  BCE/ASL。最后一组表示物理第8--12个block；所有代码参数继续使用zero-based索引。
- 新增多层8卡启动器和严格汇总器。多层必须先通过相对disabled的全部硬门槛，ASL还必须
  相对同结构BCE通过；此外final/task6 mAP不能低于同批single3/single11同loss最佳值。只有
  通过者才允许讨论正式test，不能只按多层final mAP排序。
- 当前实现的多层Image-token Adapter在各指定block独立处理冻结`LN1(CLS+patch)`，仅影响该层
  selector匹配，不把适配token写回CLIP残差流，也不是前一层Adapter输出喂给后一层的级联结构。
  5层配置每task Adapter参数为249760，8卡24GB 4090预计有充足空间，仍先以真实CLIP FP32
  `[3,11]` smoke验证forward/backward、梯度路由和冻结行为。
- 本地Shell语法、Python3.9编译、3项隔离多层汇总测试与`git diff --check`已通过；完整Track-A
  测试和GPU smoke待代码经Git同步到服务器独立实验worktree后执行。
- 实现提交`976f2a1`已推送并安全同步；服务器完整Track-A单测36/36通过。GPU0真实CLIP
  layers3+11 FP32 Adapter-ASL smoke通过，可训练参数789082，zero-init最大差`1.7695e-08`，
  无冻结、梯度或数值异常。
- 15组队列在tmux`mla_image_token_multilayer_s0_20260825_111638`启动，batch ID为
  `image_token_multilayer_screen_seed0_20260825_111638`。第一轮8组均完成task0 epoch1，84
  steps/skipped0、16.0--18.0秒；GPU总占用约2.4--3.3GB且每卡仍余20.8GB以上，无OOM、NaN、
  Traceback或RuntimeError。剩余7组由各lane在第一轮结束后自动接续，不启动正式test。
- 15组最终全部完成，每组240 epochs/13950 updates/skipped0/exit code0，同一clean`976f2a1`，
  无异常且无checkpoint。唯一严格合格多层为`[8,9]` BCE：相对disabled final/average/task6
  mAP提高`0.939906/0.844164/1.007650`，task6三类提高
  `3.473313/2.236642/3.889481`，cF1提高`1.052756`，forgetting改善`0.048449`。
- 5个多层ASL相对各自BCE的final mAP全部下降`0.252--1.369`，没有ASL赢家；五层dense后段
  同样压低Suffering并恶化forgetting，停止ASL和继续堆层。
- `[8,9]`相对fresh single11 BCE的final/task6优势仅`0.185944/0.189651`；本批缺少fresh
  single9对照，而跨批pair相对旧single9绝对final只高`0.113899`，低于disabled跨批
  `0.423396`波动。因此当前只保留候选，不启动正式test，下一步最小验证是BCE-only fresh
  single8/single9/pair8_9。
- 93文件小结果包已同步本地并逐项SHA-256通过，不含checkpoint；archive SHA为
  `ddd5c6c0a07702a70094f4b89cec2cf0a2ef47db5be5a2006d60630688ce2f44`，详细分析已写入本地
  结果目录`analysis.md`。

## 2026-08-25：同步并分析 Image-token Adapter 单层位置搜索

- batch`image_token_layer_search_seed0_20260822_235242`的25/25组全部完成：同一clean
  `d63da6b`、每组8 tasks/240 epochs/13950 updates/skipped0/exit code0，总计6000 epochs与
  348750 updates，无OOM、NaN、Inf或异常，无checkpoint；tmux已退出且无残留runner。
- disabled锚点final/average mAP为`41.955741/48.656124`。BCE的aggregate最佳为layer9，
  final/task6 mAP提高`0.402611/0.469971`，但Sadness下降`1.836305`、forgetting恶化
  `0.216600`；全部12个BCE层都使forgetting恶化，没有合格层。
- ASL aggregate最佳为layer3，final/average/task6 mAP提高`0.976536/0.956794/1.048385`，
  但Suffering下降`2.695190`、forgetting恶化`0.043960`。layer7牺牲Sadness换Suffering，
  layer11同时提高Sadness/Suffering但牺牲Sensitivity并显著恶化forgetting；没有ASL层同时
  通过相对同层BCE和disabled的全部门槛。
- 自动汇总最终为`eligible_bce_layers=[]`、`eligible_asl_layers=[]`、两者winner均null，按预先
  规则不进入bottleneck/LR，不扩大容量或多层，不启动test。位置效应非单调且主要表现为
  task6类别收益交换，不能只按final mAP选择layer3。
- 本批layer8 BCE与上一批同训练数学/seed的FP32 layer8 BCE仍有final/average mAP
  `-0.307112/-0.719150`重复运行差异，提示小于约0.3--0.7点的收益可能落在GPU数值轨迹波动
  范围内；本轮更不能把layer9 BCE的`+0.4026`视作稳健改进。
- 153文件白名单结果包已同步本地，无checkpoint；archive和本地SHA均为
  `8f6d4dd4f6fa0cb9aa0f917dd6f7f971c0acde77b5742e01608142ab356e7c05`，包内153/153逐项
  校验通过。详细分析位于结果目录`analysis.md`。

## 2026-08-22：准备 Image-token Adapter 结构调参第一阶段

- 从`exp/emotic-image-token-asl-stable-refine@fb8037f`创建
  `exp/emotic-image-token-structure-tuning`。本阶段系统搜索Image-token Adapter本身的层位置，
  不使用此前Task-lane layer5/8/11结论，也不同时改变层、容量、LR、scale或activation。
- 第一阶段预注册25组：1个FP32 Adapter-disabled joint-BCE锚点，以及zero-based layer0--11
  各一组Image-token joint-BCE和一组主模型BCE + Adapter ASL。结构固定b32、Adapter LR4e-4、
  scale0.1、ReLU、independent；ASL固定9.8/0/0.05。全部为seed0完整8-task validation-only、
  30 epochs/task、batch64、legacy监督、CLIP normalization、crop0.05，不保存checkpoint。
- validation worker支持显式`disabled/image_token`和独立日志目录。新增8-GPU启动器，把25组轮转
  分配到GPU0--7；全局smoke使用最先空闲卡，随后每张卡在每个任务前独立等待连续两次
  `free>=8000MiB`且`util<=10%`，默认间隔60秒，不抢占当前GPU任务。
- 新增层搜索严格汇总器和3项合成结果测试。汇总器拒绝配置/精度/Git漂移、不完整epoch、更新
  数不符、skipped step与checkpoint；位置选择必须同时保护task6 Sadness和Suffering，并以
  同层BCE为Adapter-ASL的直接对照；ASL还必须相对disabled通过同一组门槛。后续
  bottleneck/LR阶段依赖本阶段赢家，本轮不提前排队。
- 本地Shell语法、Python 3.9编译、`git diff --check`和3项隔离汇总器测试通过；完整Track-A
  unittest需服务器`ddp`环境，因本地Python没有torch而不能按正常包入口运行完整套件。
- 实现提交`d63da6b`已推送；Automatic Upload漂移备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-layer-search-upload-20260822`后，服务器
  ASL独立worktree安全切到该clean提交，完整Track-A测试33/33通过，test-only未触碰。
- 8卡队列进入tmux`mla_image_token_layer_s0_20260822_235242`等待，batch为
  `image_token_layer_search_seed0_20260822_235242`。首轮门控快照8卡仅余1906--4604MiB，
  均未达到8000MiB；输出只有manifest，无config、GPU smoke或训练runner。实验worktree固定
  `d63da6b`直到全部运行结束，避免各轮结果Git元数据不一致。

## 2026-08-22：同步并分析 ASL FP32 稳定版局部搜索

- batch`image_token_asl_stable_refine_seed0_20260821_231444`的13组全部完成：12个ASL
  局部组合与1个joint-BCE均为8 tasks、240 epochs、13950 updates、skipped0、exit code0，
  同一clean`3564e12`且`amp=false`。无OOM、非有限数值或异常，FP32成功消除上一轮AMP失败。
- 严格`stable_refine_fp32`汇总没有合格赢家。12个ASL的final cF1全部低于BCE，forgetting
  全部更差；虽然部分提高final mAP、oF1和Suffering，但没有配置同时保护Sadness与宏观指标。
- FP32 BCE五项为`42.164859/48.795463/37.501026/58.083137/0.832476`。final mAP最高的
  `gn9.8/clip0.05`五项变化为`+0.425109/+0.660861/-0.635873/+0.443694/+0.058057`，
  task6 Sadness/Sensitivity/Suffering变化`-2.088062/-1.408021/+6.074236`。
- `gn16/clip0.075`只失败Sadness硬门槛，average mAP/oF1/Suffering分别提高
  `0.850003/0.841133/6.925002`，但Sadness下降`2.326867`、forgetting恶化`0.213311`，
  不按单项收益放宽预注册规则。
- 同参数9.8/0.05由AMP切到FP32后，cF1与Sadness增益从`+0.6864/+1.4049`变为
  `-0.6359/-2.0881`；Suffering仍提高但幅度从`8.6751`降至`6.0742`。说明小幅收益对精度
  与优化轨迹敏感，单seed不能视为稳健结论。
- 81个JSON、日志、清单文件已打包同步，无checkpoint；服务器/本地archive SHA-256均为
  `5e917f92033491bf6d281a6e9f281167434f40194dd6188cd624ac6a1f302e22`，归档内逐文件校验
  全部通过。详细分析写入本地结果目录`analysis.md`，未启动gamma-pos或新test。

## 2026-08-21：同步并分析 Image-token Adapter-only ASL 第一阶段结果

- 第一阶段batch `image_token_asl_loss_seed0_20260820_191924`已结束。21组预注册配置中18组
  完整完成，均为8 tasks、240 epochs、13950 updates、skipped0；3组在task3因非有限ASL
  loss失败：`gn4/clip0`、`gn6/clip0.025`、`gn9.8/clip0.025`。失败不是OOM，按预注册
  完整性规则判为无效且不重跑。
- 缺失的`gn9.8/clip0`已在原实验提交`b99480e`补齐并正常完成。failure-aware汇总器生成
  `loss_search_summary.json`，服务器完整单元测试29/29通过，当前无本轮训练进程且GPU0--7
  均空闲。
- 相对joint-BCE validation，严格赢家`gn9.8/clip0.05`的final/average mAP为
  `42.667273/49.237821`，提高`0.709193/0.922159`；final cF1/oF1提高
  `0.686412/0.430169`，8个task mAP均提高。forgetting提高到`1.017189`，即恶化
  `0.076460`。
- 赢家task6 mAP提高`0.695724`；Sadness/Sensitivity/Suffering分别变化
  `+1.404916/-0.530427/+8.675063`，三类均值提高`3.183184`。它通过所有硬门槛，但只证明
  Suffering和整体排序获益，不能证明Sensitivity或旧类保持已解决。
- 只有`gn9.8/clip0.05`和`gn9.8/clip0`通过全部门槛；后者final mAP提高`0.526062`，但
  final cF1下降`0.176099`、forgetting恶化`0.108917`，综合列第二。其余若只看final mAP
  会忽略Sadness/Suffering或F1门槛退化。
- 107个结果/日志/清单文件已在服务器按白名单打包，无checkpoint，压缩包512KB；服务器与
  本地archive SHA-256均为
  `90e0ea0817ec0ea266e72f3961a58e4f2abe89603c31f4755ef91c8ad0ab7684`，逐文件SHA校验通过。
  本地已解压至`output/emotic_track_a_image_token_asl_hparam/`并新增`analysis.md`。
- 本轮未启动下一阶段。赢家gamma-neg仍处搜索上边界，建议先局部扩展到
  `gamma_neg={8,9.8,12,16}`、`clip={0.0375,0.05,0.075}`，复用已有赢家后需11个新运行；
  只有确认内部最优后才搜索gamma-pos，并继续只用validation。
- 用户确认使用8张4090并行继续。进一步审计发现三个失败配置都在task3约2000次连续更新后
  开始出现AMP GradScaler跳步，随后跳步增多并产生非有限logits/loss；这与默认GradScaler
  增大loss scale的时间点吻合，不是数据缺失或OOM。
- 从当前分支创建`exp/emotic-image-token-asl-stable-refine`。为避免把不同数值协议混排，
  稳定版统一关闭AMP，以FP32重跑12个局部ASL组合及1个joint-BCE；因此不复用旧9.8/0.05。
  新增8-GPU两轮队列、FP32 GPU smoke入口、稳定版严格汇总profile和有限logit/target诊断。
  实现提交`3564e12`已推送并由服务器ASL独立worktree安全切换；服务器30/30完整单测通过。
- GPU0真实CLIP FP32 Adapter-ASL smoke通过：trainable parameters为739130，zero-up与disabled
  logits最大差`1.70e-08`，无梯度、冻结或数值异常。8张GPU启动前均有23.6GB以上空闲。
- 13组队列在tmux`mla_asl_stable_refine_s0_20260821_231444`启动，batch ID为
  `image_token_asl_stable_refine_seed0_20260821_231444`。第一轮8组全部进入task0 epoch2；
  epoch1均84 steps/skipped0、约16--17秒，单卡占用约2.8--3.0GB，无错误信号。第二轮5组由
  每卡lane在第一组完成后自动接续；不保存checkpoint，不读取held-out test。

## 2026-08-20：Image-token Adapter-only ASL 参数搜索框架

- 从`exp/emotic-image-token-asl-routing@129c73d`创建
  `exp/emotic-image-token-asl-hparam-search`，并保留此前尚未提交的四组结果分析与三份上下文
  文档改动。
- 第一阶段固定seed0、完整8-task validation、30 epochs/task、batch64、legacy监督、CLIP
  normalization、crop0.05、Image-token layer8/b32/LR4e-4/scale0.1/ReLU/independent；主模型
  继续BCE，只联合搜索Adapter ASL的5个gamma-neg与4个clip，并加入joint-BCE对照，共21组。
- 新增通用validation worker与8-GPU队列启动器；每张GPU同一时刻只运行一组，21组按轮转
  队列分配，全部完成后自动生成严格汇总。新增`--no-save-checkpoints`，只影响显式开启该参数
  的调参运行，历史/正式入口继续默认保存checkpoint。
- 队列新增可复现资源门控，默认要求每卡连续两次达到`free>=5000MiB`且`utilization<=10%`，
  间隔60秒；队列启动前在指定卡运行真实CLIP Image-token Adapter-ASL smoke。每条GPU lane
  在开始下一组前都会重新过门槛，防止其他任务临时占卡时继续叠加。
- 新增专用汇总器和3项单测：强制完整21组网格、相同clean commit/tree、240 epochs、13950
  updates、skipped0、无checkpoint，并按预注册task6/final/Sadness/Suffering/F1硬门槛排名。
  本地Python编译、Shell语法、3项独立汇总器单测与`git diff --check`通过；完整torch单测待
  服务器`ddp`环境执行。
- 服务器预检发现GPU0--7均被现有任务占用约20.4--22.2GB，当前不具备安全启动余量，因此未
  启动调参、未终止现有进程。Automatic Upload漂移已备份至
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-hparam-upload-20260820`并恢复主worktree
  clean；ASL独立worktree保持clean，test-only worktree未触碰。
- 首个实现提交`397d74d`已推送并由服务器ASL独立worktree通过Git切换到同一clean HEAD；
  服务器完整Track-A单元测试28/28通过。GPU smoke没有挤入繁忙卡，改由资源门控在队列真正
  启动前自动执行。
- 资源门控补充提交`a0e0aee`已推送并由服务器fast-forward。第一阶段已在tmux
  `mla_image_token_asl_hparam_s0_20260820_191924`排队，batch ID为
  `image_token_asl_loss_seed0_20260820_191924`；首个GPU5快照为free3520MiB/util99%，门控
  正确等待。当前只是CPU侧每60秒轮询，没有GPU smoke或训练进程，也未增加显存占用。

## 2026-08-13：Image-token Adapter 的参数组级 ASL 三组实验

- 审计本地`CODE_DDP`实现与已有结果，沿用其Adapter ASL默认：gamma-neg9.8、gamma-pos0、
  probability clip0.05、eps1e-8、detach focal weight。MULTI-LANE保留legacy 26维监督视图和
  mean reduction，避免同时改变loss形状与历史梯度尺度。
- 从Image-token分支创建`exp/emotic-image-token-asl-routing`，新增参数组loss路由：
  `model_asl`、`adapter_asl`、`both_asl`。混合模式通过`autograd.grad`把model目标只送入
  selectors/prompts/head，把Adapter目标只送入当前task Adapter，消除联合图上的梯度串流。
- 新增ASL数值/梯度路由测试、GPU smoke路由参数、三组seed0正式worker和3-GPU启动器。三组
  固定full image、CLIP normalization、crop0.05、legacy监督、Image-token layer8/b32、
  30 epochs/task、batch64、held-out test。
- 实现提交`6d5430a`已推送。服务器为ASL分支创建独立worktree
  `/mnt/haoyuan/workspace/multi-lane-main-asl-routing`，不切换GPU2仍在运行原Image-token BCE的
  主worktree；Automatic Upload文件已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-image-token-asl-upload-20260813`并恢复主worktree clean。
- 服务器25/25完整单测通过；GPU3/4/7依次完成`model_asl/adapter_asl/both_asl`真实CLIP
  smoke，三组可训练参数均为739130，初始等价最大差均在`4.2e-05`内，参数组梯度、冻结visual
  和concat inference正常。三组正式训练尚未启动。
- 验证记录提交`8840e96`已推送并由服务器ASL worktree fast-forward。三组seed0 held-out test
  正式实验在tmux `mla_image_token_asl_seed0_011602`启动，GPU3/4/7分别对应
  `model_asl/adapter_asl/both_asl`，run ID共同后缀为`20260813_011602`。
- 三组task0 epoch1均为84 optimizer steps、skipped0、约16.5秒；GPU占用2.0--2.4GB且均有
  21GB以上余量。model-ASL的model/Adapter loss为`0.01572653/0.76665431`，adapter-ASL为
  `0.61468159/0.02817775`，both-ASL为`0.01577781/0.01577781`；路由行为可观测且无异常。
- 原Image-token BCE与三组ASL均完成240 epochs/13950 updates、skipped0、exit code0，运行
  config为clean Git。按白名单打包四组JSON与正式/launcher日志，共24个小文件，不含checkpoint；
  各源文件已通过服务器/本地SHA-256一致性校验，本地统一104KB压缩包SHA-256为
  `60cb61ebe7f64d639f351821834c95539bad149ad0abe6e1d3f4796a2d8df884`。
- 四组final/average mAP依次为：BCE `31.676786/38.564709`、model-ASL
  `30.462064/38.238173`、Adapter-ASL `32.419329/38.878898`、both-ASL
  `32.199410/39.148237`。Adapter-ASL相对BCE的task6/final mAP提高
  `0.735797/0.742543`，task6 Sadness/Sensitivity/Suffering分别提高
  `7.524682/0.652493/8.239046`，三类均值提高`5.472074`。
- model-ASL/both-ASL的final oRecall接近100%、oPrecision约17%，oF1相对BCE下降
  `20.004362/19.356518`。确认是gamma-neg9.8强烈削弱负监督引起的固定阈值校准坍塌，不是
  训练未完成、OOM或梯度路由错误；Adapter-only ASL保留主模型BCE，因此F1基本稳定。
- 完整对照报告写入本地结果目录`comparison_analysis.md`。本轮结果只有seed0且直接使用
  held-out test，记录为探索性证据，不据此继续调gamma、阈值或启动新实验。

## 2026-08-13：实现 Image-token Adapter 并准备 seed0 正式实验

- 将上一阶段task-lane Adapter位置诊断与路线收口文档以提交`ea5f204`推送到
  `origin/exp/emotic-adapter-position-diagnostics`，随后新建
  `exp/emotic-image-token-adapter`，避免两种架构混入同一提交。
- 新增`image_token` Adapter模式：每个task独立路由Adapter，输入为目标CLIP block经`ln_1`
  归一化后的冻结`CLS + patch tokens`；输出仅修改selector读取的图像token视图，不修改冻结
  CLIP主干的后续残差流。zero-up初始化保证初始输出与disabled路径一致，初始化RNG继续隔离。
- runner、GPU smoke、配置元数据和单元测试已扩展到新模式；新增seed0专用正式启动器
  `scripts/emotic/run_multilane_track_a_image_token_adapter_seed0.sh`。
- 用户明确只跑seed0并允许跳过validation。正式配置：Split-EMOTIC B5-C3、full image、held-out
  test、8 tasks、30 epochs/task、batch64、seed0、legacy loss、CLIP normalization、crop0.05、
  image-token Adapter layer8/bottleneck32/LR4e-4/scale0.1/ReLU/independent。
- 本地Python编译与Shell语法通过；本地系统Python缺少torch/numpy，完整单测和GPU smoke改在
  服务器`ddp`环境执行。
- 实现提交`e10324f`已推送到远端并由服务器通过Git安全切换；额外test-only worktree未修改。
  服务器完整22项单元测试全部通过。
- GPU2真实ViT-B/16 Image-token Adapter smoke通过：trainable parameters为`739130`，初始
  等价最大差`3.4332275e-05`，Adapter梯度有限、visual encoder无梯度、无OOM。正式训练尚未
  启动，下一步按确认配置只启动seed0。
- 记录验证结果的`a322419`已推送并由服务器fast-forward到clean同HEAD。seed0正式实验已在
  GPU2/tmux `mla_image_token_seed0_004500`启动，run ID为
  `image_token_adapter_b32_layer8_seed0_20260813_004500`。task0 epoch1正常完成：loss
  `0.61481267`、84 optimizer steps、skipped0、17.7秒；GPU显存约2.0GB，无错误信号。

## 2026-08-12：实现 loss、预处理与 Adapter 层位置分阶段诊断

- 按预定停止规则正式结束当前task-lane Adapter路线：不再扩展layer5/8/11、不做多层、不扩大
  bottleneck、不增加Adapter深度、不使用正式test调参。当前保留的validation基线为Adapter
  disabled、legacy loss、CLIP normalization、crop `(0.05,1.0)`；其seed0五项为
  `42.175807/48.421923/37.198819/58.301308/0.994957`。
- 已确认服务器不存在loss/preprocessing/layer诊断runner，GPU2/3/4已释放；已关闭三个完成后
  停留在shell的tmux `mla_loss_diag_s0_173322`、`mla_prep_diag_s0_182014`、
  `mla_layer_diag_s0_192723`。所有服务器日志、JSON与本地同步副本均保留，未删除实验数据，
  未启动新实验。
- 第三阶段单层位置诊断已在tmux `mla_layer_diag_s0_192723`启动，batch ID为
  `adapter_layer_position_seed0_20260812_192723`。固定legacy loss、CLIP normalization、
  crop `(0.05,1.0)`、seed0、完整8-task val-only、30 epochs/task、batch64、LR0.0125；
  independent Adapter为b64、LR4e-4、ReLU、scale0.1，GPU2/3/4分别运行zero-based layer
  `5/8/11`。基线为已完成的clip+crop0.05 disabled运行。
- 三组config均记录clean `001d8a0`，仅layer index不同；首两轮均84 steps、skipped0，每卡约
  1.94GB且仍空闲约22.1GB，无OOM、CUDA error、Traceback或NaN/Inf。不读取test、不保存
  checkpoint。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_adapter_layer_position_v0.3/adapter_layer_position_seed0_20260812_192723/`。
- 最终判定在原三项task6 mAP、final mAP、task6新类均值改善之外，增加Suffering AP不得低于
  disabled基线的硬性条件；若无单层同时满足，则结束task-lane Adapter路线，不做多层扩容。
- layer5/8/11均以240 epochs、13950 updates、launcher exit code0完成，日志无OOM、NaN或异常。
  相对clip+crop0.05 disabled基线，五项变化分别为：layer5
  `final mAP -0.093808 / average mAP +0.391947 / final cF1 +0.701436 / final oF1 +0.061258 /
  forgetting improvement +0.006156`；layer8为
  `+0.380562/+1.013492/+0.975789/+0.132606/+0.178909`；layer11为
  `-0.439175/+1.389888/+0.425703/-0.124769/+0.195441`。
- 四项硬条件没有任何层通过。layer5的task6新类均值/Suffering为`-0.525884/-0.528539`；
  layer8为`-1.296133/-2.105999`；layer11为`-8.770622/-14.799475`。layer8虽然aggregate
  最优，但task6总mAP`+0.540776`来自旧类均值`+0.816313`掩盖新类下降；layer11同样表现为
  旧类`+0.924071`掩盖新类`-8.770622`，证明层位置不能修复后期可塑性问题。
- 三组12个运行JSON、3个paired report及3份日志已同步到本地
  `./output/emotic_track_a_adapter_layer_diagnostics/adapter_layer_position_seed0_20260812_192723/`；
  服务器与本地18个文件SHA-256逐项一致，没有checkpoint。按预定规则停止task-lane Adapter
  多层与容量扩张，当前未启动新实验。
- 第二阶段2×2预处理诊断已在tmux `mla_prep_diag_s0_182014`启动，batch ID为
  `preprocessing_seed0_20260812_182014`。固定`legacy_full_zero`、seed0、EMOTIC train→val、
  完整8 tasks、30 epochs/task、batch64、LR0.0125、Adapter disabled、full image、AMP/TF32；
  GPU2/3/4/7分别对应`none+crop0.05`、`clip+crop0.05`、`none+crop0.50`、
  `clip+crop0.50`。不读取test、不保存checkpoint、不启动Adapter层实验。
- 四组启动config均记录clean `001d8a0`。首个epoch均84 steps、skipped0；每张卡新增约1.94GB，
  仍空闲约22.1GB，未发现OOM、CUDA error、Traceback或NaN/Inf。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_preprocessing_diagnostics_v0.3/preprocessing_seed0_20260812_182014/`，
  日志位于`./logs/emotic_track_a_adapter_diagnostics/`。
- 四组均以240 epochs、13950 updates、launcher exit code0完成。五项
  `final_mAP/average_mAP/final_cF1/final_oF1/forgetting`分别为：none+0.05
  `41.964051/48.323627/37.289303/58.586581/0.890804`；clip+0.05
  `42.175807/48.421923/37.198819/58.301308/0.994957`；none+0.50
  `40.645741/46.624393/37.102879/57.290124/0.971548`；clip+0.50
  `41.172109/47.777290/37.403789/57.218872/0.800238`。
- crop0.50在两种normalization下均使8个task mAP全部下降；末轮训练loss反而更低，说明训练
  视图更简单但泛化更差，因此淘汰。clip+crop0.05相对旧默认使task6/final mAP提升
  `0.134184/0.211756`、task6新类均值提升`0.484693`，按预先规则选为层位置基线；但其
  Sadness/Sensitivity/Suffering变化为`+7.019647/+0.362923/-5.928491`，后续必须单列
  Suffering，不能只凭aggregate mAP选层。
- 结果与四份日志已同步到本地
  `./output/emotic_track_a_preprocessing_diagnostics/preprocessing_seed0_20260812_182014/`；服务器与
  本地21个文件SHA-256逐项一致，没有checkpoint。当前未启动Adapter层位置实验。
- 第一阶段loss二选一已在tmux `mla_loss_diag_s0_173322`启动，batch ID为
  `training_loss_seed0_20260812_173322`。GPU2为`legacy_full_zero`，GPU3为`current_only`；
  两组均为clean `001d8a0`、seed0、EMOTIC train→val、完整8 tasks、30 epochs/task、batch64、
  LR0.0125、Adapter disabled、full image、none normalization、crop `(0.05,1.0)`、AMP/TF32。
- 启动前GPU2/3已有进程约占7.8/8.3GB，空闲15.8/16.1GB；本轮进程各占约1.9GB，进入稳定
  训练后仍空闲约14.1GB。两组task0前5个epoch均84 steps、skipped0，无OOM、CUDA error、
  Traceback或NaN/Inf。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_loss_diagnostics_v0.3/training_loss_seed0_20260812_173322/`，
  日志位于`./logs/emotic_track_a_adapter_diagnostics/`。
- 两组均完成240 epochs与13950 updates，launcher exit code 0，并自动生成
  `loss_diagnostics_summary.json`。legacy五项为
  `41.964051/48.323627/37.289303/58.586581/0.890804`；current-only为
  `41.318942/48.227313/36.413423/58.508179/0.795174`。current-only虽在task0–3 mAP分别
  微升`0.237138/0.392014/0.198054/0.079564`，但task4开始转负，task6/7下降
  `0.757291/0.645109`。
- task6新三类引入AP：legacy为Sadness `41.064792`、Sensitivity `5.960459`、Suffering
  `51.880423`；current-only为`40.507695/6.710906/38.328382`。均值从`32.968558`降至
  `28.515661`，主要失败点是Suffering `-13.552041`，因此下一阶段固定legacy loss。
- 新legacy运行与上一轮clean disabled运行的8-task mAP及五项summary差值全部为0；结果已连同
  两份日志同步到本地`./output/emotic_track_a_loss_diagnostics/training_loss_seed0_20260812_173322/`，
  11个文件经服务器/本地SHA-256逐项校验完全一致，没有checkpoint。
- 从clean `5b2f791` 创建本地分支 `exp/emotic-adapter-position-diagnostics`；服务器仍停留在
  `exp/emotic-multilane-transformer-adapter`，本轮代码尚未提交推送或运行实验。
- runner新增 `--training-loss-mode legacy_full_zero|current_only`。默认legacy保持历史复现；
  current-only直接在当前task logits/targets上计算BCE。测试将验证有效logit梯度相对legacy
  放大 `26/current_classes`，同时config记录Adam会弱化常数梯度缩放，避免预设结果。
- runner新增 `--input-normalization none|clip` 和 `--train-crop-scale MIN MAX`；默认仍为none和
  `(0.05,1.0)`。CLIP选项使用OpenAI mean/std，normalization与crop可独立组合并写入config。
- 新增统一seed0/8-task/val-only诊断worker，支持disabled或task-lane；所有入口要求clean Git，
  不构造test。新增loss两组launcher、预处理2×2 launcher、以及independent Adapter零基索引
  layer `5/8/11` launcher；后两阶段强制显式传入上一阶段选中的配置，防止混杂。
- paired validation分析器扩展为严格核对loss、input mode、normalization与crop scale，支持
  显式 `independent/copy_previous` candidate，并把输出候选字段改为对应初始化模式。
- 新增通用多运行validation diagnostics汇总器及测试，用于disabled loss/预处理阶段；输出每组
  8-task mAP/cF1/oF1、五项summary和task6三类引入/final AP，不擅自按单一指标自动选优。
- GPU smoke新增 `--adapter-layer-indices`，使同一个真实CLIP forward/backward入口可验证零基
  layer `5/8/11` 的Adapter零初始化、梯度、冻结参数、参数量与路由，不必启动训练。
- 计划顺序固定：disabled legacy/current-only完整validation选loss；选中loss下跑none/clip ×
  crop0.05/0.50；选中预处理下再跑layer5/8/11 independent Adapter。只有中层通过task6、
  task7 final与task6新类平均AP三项条件，才考虑小范围多层；当前未启动任何组。
- 诊断实现已以 `5766860` 提交并推送；服务器主工作树安全切到同一分支和提交。Automatic
  Upload遗留的4个同内容文件已先备份到
  `/mnt/haoyuan/workspace/git-sync-backup-adapter-position-diagnostics-final-20260812`，再恢复旧分支
  clean后通过Git切换；test-only worktree原有改动保持不动。
- 服务器完整Track-A测试20/20通过。GPU2/3/4并行真实CLIP batch2 smoke分别验证zero-based
  layer `5/8/11`，三组均为 `788314` 个可训练参数，初始logits最大差均为
  `6.103515625e-05`，forward/backward、Adapter梯度、视觉塔冻结和concat推理均通过。本轮没有
  读取EMOTIC、没有写checkpoint，也没有启动loss、预处理或层位置实验。

## 2026-08-11：启动 seed0 完整8-task warm-start validation

- 在服务器主工作树clean `58886f1` 上启动唯一候选：seed0、EMOTIC train→val、完整8 tasks、
  30 epochs/task、batch64、base LR0.0125、Adapter LR4e-4、b64、layer11、ReLU、scale0.1、
  `copy_previous`、隔离初始化RNG、AMP/TF32；不构造或读取test split。
- run ID为
  `task_lane_adapter_full_val_seed0_b64_lr0.0004_copy_previous_20260811_174404`，tmux为
  `mla_warm_val_s0_174404`，使用GPU1。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_warmstart_val_v0.2/`，
  日志位于 `./logs/emotic_track_a_adapter_warmstart_val/`。
- 启动核验已进入task0并完成至少3个epoch；每epoch 84 updates、skipped0，暂未发现OOM、NaN、
  inf或traceback。等待8 tasks完成后只分析validation曲线，重点比较task6的Sadness、
  Sensitivity、Suffering；不根据正式test调参。
- 为形成同代码、同seed、同随机轨迹的配对对照，在GPU2启动 Adapter disabled 的完整8-task
  val-only运行 `multi_lane_disabled_full_val_seed0_paired_clean_20260811_174847`，tmux为
  `mla_disabled_val_s0_174847`。首次对照因Automatic Upload使config记录dirty而在task0早期
  主动终止并保留原目录；clean重启版本config记录 `dirty=false`，已完成至少3个epoch且无错误。
- warm-start与clean disabled均完成240 epochs/8 tasks并输出completion marker。逐task val mAP
  差值为 `+2.041904/+2.367560/+1.808337/+1.307971/+0.890832/+0.539336/-0.334822/-0.448594`，
  明确在task6/7转负；final cF1虽提高 `+0.482067`，但final oF1下降 `-0.770218`。
- warm-start相对disabled的average mAP提高 `+1.021565`、forgetting改善 `0.079821`，但final
  mAP下降 `0.448594`。task6新三类引入时平均AP下降 `6.165031`：Sadness `-8.788894`、
  Sensitivity `+0.997110`、Suffering `-10.703310`；final三类平均差进一步为 `-6.248597`。
- 新增 `multi_lane/track_a/compare_validation.py`，严格检查同commit/seed/protocol/val-only与
  clean状态，自动生成8-task曲线、task6逐类AP及停止判定；结果写入本地
  `./output/emotic_track_a_adapter_warmstart_val/paired_validation_comparison.json`。按预先规则
  三项继续条件全部失败，结论为停止task-lane Adapter容量扩张，不再增加bottleneck或层数。
- 配对分析脚本、3项新测试和上下文以 `3d7c5aa` 提交推送并在服务器fast-forward同步；原10项
  Track-A回归测试加3项分析测试共13/13通过，未启动新实验。
- 进一步诊断显示warm-start的当前任务训练BCE在8个task均低于disabled，但新类平均AP差值从
  task3开始转负：task0–7依次为 `+2.041904/+2.073189/+0.601783/-0.703644` 和
  `-0.832849/-1.669440/-6.165031/-1.103683`。总mAP前期的正增益主要由已学旧lane保留，
  而非新task获取。
- task6 train view仅627样本、10 updates/epoch、总计300 updates；Sadness/Sensitivity/
  Suffering训练正例为260/316/188，seen-scope val正例仅113/110/102。最终checkpoint中Adapter
  参数范数从task0 `12.3918`累积到task6 `17.8840`；task6与task5余弦相似度 `0.9841`、参数
  距离 `3.1801`，是最强的相邻继承之一，说明低更新预算无法消除task5的非零残差偏置。
- 原MULTI-LANE已复制selectors/prompts，Adapter再次完整warm-start造成重复继承；同时较低BCE
  未转化为AP排序提升，说明优化更贴合训练概率而非稀疏新类排序。现状属于结构偏置与可塑性
  不足，不是随机轨迹、容量太小或训练崩溃，因此继续堆容量缺乏依据。

## 2026-08-11：实现 Adapter RNG 隔离与跨任务 warm-start

- 当前结构明确为“每任务独立 Adapter”，不是所有任务共享一个：8个 Adapter 预分配，训练
  task t 时只解冻第t个，推理按lane id路由。上一版本每个task独立随机初始化且不复制。
- 按用户确认的计划增加 `adapter_task_initialization=independent|copy_previous`；默认
  `independent` 保留上一轮复现。`copy_previous` 在task>0激活时把上一task Adapter完整复制
  到当前task，然后只训练当前副本，旧task仍冻结。
- Adapter Bank 初始化被包在 `torch.random.fork_rng(devices=[])` 中；Adapter仍获得确定性的
  seed相关初始化，但构造结束后全局torch RNG恢复到与disabled模型相同的状态，避免改变
  DataLoader shuffle和随机增强序列。
- runner/config/smoke新增 `--adapter-task-init`；task0脚本默认保持independent。新增
  `scripts/emotic/run_multilane_track_a_adapter_full_val.sh`，用于seed0完整8-task、val-only、
  b64/lr4e-4 warm-start验证，不构造test dataset。
- 单元测试新增disabled/adapter构造后RNG state完全一致，以及warm-start参数逐tensor复制、
  旧task冻结和当前task解冻检查。本地py_compile、bash -n和git diff --check均已通过；
  实现以 `313618c` 提交并推送，服务器主工作树fast-forward到同一提交，test-only未触碰。
- 服务器完整单元测试10/10通过；GPU1真实CLIP `copy_previous` Adapter smoke通过，可训练参数
  `788314`，零初始化Adapter与disabled logits最大差 `9.1552734375e-05`，在既定AMP容差
  `1e-4` 内。smoke仅执行batch2 forward/backward，未加载EMOTIC、未生成checkpoint、未启动
  validation或正式训练。

## 2026-08-11：启动 Task-lane Adapter task0 validation screen

- 用户确认 Automatic Upload 已关闭，并确认可以启动 task0 validation screen。启动前服务器
  主工作树干净，local/origin/server 均为
  `exp/emotic-multilane-transformer-adapter@6e8b15d`；test-only worktree 未触碰。
- 完整配置：EMOTIC full image、seed0、task0/5 classes、train→val、30 epochs、batch64、
  eval batch64、Adam base LR0.0125、weight decay0、temperature1、每任务 cosine、AMP/TF32、
  threshold0.5、layer11、ReLU、scale0.1；不构造 test dataset。
- batch ID：`task_lane_adapter_task0_seed0_20260811_153033`。b32/lr1e-4、b32/lr4e-4、
  b64/lr1e-4、b64/lr4e-4 分别运行在 GPU1/2/3/4，对应 tmux：
  `mla_t0_b32_lr1e4_153033`、`mla_t0_b32_lr4e4_153033`、
  `mla_t0_b64_lr1e4_153033`、`mla_t0_b64_lr4e4_153033`。
- 初始检查四组均约进入 epoch3/30，每 epoch 84 optimizer updates、skipped 0，无 OOM、NaN
  或 traceback。输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_screen_v0.1/`，日志位于
  `./logs/emotic_track_a_adapter/`。本地上下文改动待实验完成、补齐结果后再确认提交。
- 四组均以 exit code0 完成 30 epochs、2520 optimizer updates、skipped 0。b32/lr1e-4、
  b32/lr4e-4、b64/lr1e-4、b64/lr4e-4 的 seed0 task0 val mAP 分别为
  `56.792729/58.974777/57.720757/59.697275`；相对严格复现 seed0 `57.339154` 的增益分别为
  `-0.546425/+1.635623/+0.381603/+2.358121`，只晋级 b64/lr4e-4。
- b64/lr4e-4 的 task0 seed1/2 确认于 15:53 启动，run ID 为
  `task_lane_adapter_task0_confirm_b64_lr4e4_20260811_155343_seed1/seed2`，tmux 为
  `mla_t0_confirm_s1_155343` / `mla_t0_confirm_s2_155343`，分别使用 GPU1/2。复用已有 seed0，
  晋级规则为三种子平均增益≥+0.5、至少2/3 seeds提升、任一seed降幅不超过1.0；仍不读取test。
- 老师随后明确不需要额外多 seed 确认。停止信号到达前 seed1/2 已正常完成，val mAP 为
  `59.329355/56.178599`，相对各自严格复现基线 `56.932921/54.973777` 仍分别提升
  `+2.396434/+1.204822`；产物保留，但不再作为必须前置步骤。
- 按用户要求新增 b128/lr4e-4 seed0 task0 val screen：run ID
  `task_lane_adapter_task0_seed0_b128_lr4e4_20260811_155954`，tmux
  `mla_t0_b128_lr4e4_155954`，GPU3。每任务 Adapter 参数 `197504`、当前总可训练参数
  `886682`。最终 val mAP `56.824802`，比 b64/lr4e-4 的 `59.697275` 低 `2.872474`，因此
  淘汰 b128，正式配置确定为 b64/lr4e-4。
- 正式 run ID：`multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927`。seed0/1/2
  分别在 GPU1/2/3、tmux `mla_adapter_formal_s0_160927`、
  `mla_adapter_formal_s1_160927`、`mla_adapter_formal_s2_160927` 启动。完整配置为 8 tasks、
  30 epochs/task、batch64、Adam base LR0.0125、Adapter LR4e-4、b64/layer11/ReLU/scale0.1、
  cosine per task、AMP/TF32、test reporting、threshold0.5；运行提交 clean `6e8b15d`。
- 正式输出位于服务器
  `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_formal_v0.1/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`，日志位于
  `./logs/emotic_track_a_adapter_formal/multi_lane_task_lane_adapter_b64_lr4e4_seed012_20260811_160927/`。
  启动检查三组均完成模型构建并进入 task0，GPU 显存约1.9GB，Git仍为clean。
- 正式三组均以 completion marker 和 tmux exit code0 完成；每 seed 240 epochs、8 tasks、
  skipped0，config 均为 test reporting、b64/lr4e-4、Git clean `6e8b15d`。
- 聚合五项 mean±sample std：final mAP `31.128461±0.082307`、final cF1
  `31.994054±0.322333`、final oF1 `48.878881±0.080051`、average mAP
  `38.599579±0.360451`、forgetting `4.869595±0.112483`。相对无 Adapter 基线均值变化为
  `-0.171025/+0.182944/-0.230322/+0.601003/+0.081138`。
- 使用显式白名单打包 formal summary、每 seed 四个 JSON、三份日志和 SHA manifest，共17个
  普通文件，确认无 checkpoint。服务器压缩包大小61813 bytes，SHA-256
  `16e0a9fcf9dc7d66dbf8ac5ba0a681539a400fdcf0f53916bd122071818946e1`；已通过 SSH key 下载到
  本地 `./output/emotic_track_a_adapter/<run_id>/`，校验一致并解压到 `synced_files/`。
- 逐 task 对齐发现 Adapter 的 seen-class test mAP 在 task0–5 分别提升
  `+1.665288/+1.280988/+0.912148/+0.433352/+0.510573/+0.410443`，但 task6/7 变为
  `-0.233741/-0.171025`。task6 新类平均 AP 下降 `4.514579`，Sadness/Suffering 分别下降
  `6.289815/7.842116`；到 final 仍保持类似差值，说明是 task6 获取不足而非后续遗忘。
- task6 train view 仅627样本、10 updates/epoch、总计300 updates；其 Adapter up-weight norm
  `2.494252` 为8个 task最低。全部 task 的最终 BCE 均低于基线，但 task3/5/6/7 val mAP
  更差，支持低数据任务欠适应/目标错配诊断。参数全部 finite，无爆炸或未训练迹象。
- 发现严格控制缺陷：Adapter Bank 的 down projection 随机初始化会在 DataLoader 前消耗
  全局 torch RNG，使相同 seed 的数据 shuffle/augmentation 不再与 disabled 基线相同，
  不同 bottleneck 也不完全配对。下一轮必须先隔离初始化 RNG，并优先测试上一 task Adapter
  warm-start；不再根据已观察的正式 test 反复调参。

## 2026-08-11：开始 Task-lane Transformer Adapter 阶段 0

- 用户确认按分析方案开始修改；从严格复现分支 `exp/emotic-multilane-track-a-repro` 的
  `ce7d9a0` 创建本地分支 `exp/emotic-multilane-transformer-adapter`。服务器尚未切换，
  当前改动尚未 commit/push。
- 新增 `multi_lane/track_a/adapter.py`：实现 task-specific、layer-specific bottleneck
  Adapter bank；down 使用 Xavier 初始化，up 权重/偏置全零，按 lane id 路由，激活新任务
  时只解冻当前 task Adapter，不复制或覆盖旧任务参数。
- `multi_lane/track_a/model.py` 在 lane block 冻结 MLP 旁路接入 Adapter；默认
  `adapter_mode=disabled`，因此原模型不会创建 Adapter 或额外消耗初始化随机数。新增运行时
  开关及基础/Adapter 参数分组接口。
- `multi_lane/track_a/runner.py` 增加独立 Adapter LR、bottleneck/layer/scale/activation、
  `max_tasks` 和 `reporting_split`；默认仍是 disabled、8 tasks、test。val screen 模式不创建
  test dataset；单 task summary 的 forgetting 明确定义为 0。
- 新增 `scripts/emotic/run_multilane_track_a_adapter_task0.sh`，固定 task0、val-only、
  layer11、ReLU、scale0.1，并允许通过环境变量筛选 bottleneck 与 Adapter LR；脚本尚未运行。
- 单元测试扩展为覆盖零初始化基线等价、运行时旁路、lane 路由、任务冻结、无 Adapter 复制、
  参数计数和单任务 forgetting。
- GPU smoke 新增显式 `--adapter-mode task_lane` 路径，检查真实 CLIP 上零初始化 logits
  等价、Adapter 梯度、冻结视觉塔和当前任务参数量；默认 disabled 路径仍执行原基线检查。
- 本机系统 Python 3.9 已通过 `py_compile`，新 shell 入口通过 `bash -n`，`git diff --check`
  通过；本机缺少 PyTorch，张量级单元测试与 GPU smoke 等待提交推送后在服务器执行。
- 阶段 0 主实现以 `531f3f3` 提交并推送；服务器主工作树检查为空后通过 `fetch` 和新建跟踪
  分支同步到同一提交。test-only worktree 的历史修改和未跟踪文件保持原样。
- 服务器 CPU 单元测试 8/8 通过；原基线 GPU1 smoke 通过，可训练参数仍严格为 `689178`。
- 首次 Adapter GPU smoke 的逐 bit logits 检查失败。诊断确认 Adapter delta、up weight 和
  up bias 的非零元素均为 0；重复 forward 的 AMP 最大差为 `6.1035e-5`，FP32 最大差为
  `1.4901e-8`，属于 GPU 重复计算数值差，而非 Adapter 非零输出。GPU smoke 因此改为直接
  检查 Adapter residual 严格为零，并对两次 AMP logits 使用 `atol/rtol=1e-4`；CPU 单测的
  逐 bit 等价检查保持不变。
- AMP smoke 修正以 `fc32af7` 提交推送并同步。最终提交上服务器 8/8 单元测试再次通过；
  GPU1 Adapter smoke 通过，可训练参数 `788314`，其中新增当前任务 Adapter `99136`，
  零初始化 AMP logits 最大差 `3.0518e-5`。全程未读取 EMOTIC、未生成 checkpoint、未启动
  task0 或正式训练。

## 2026-08-11：开始在当前仓库移植 MULTI-LANE Track-A 严格复现

- 用户确认在当前 `multi-lane-main` 中复现注册的 EMOTIC B5-C3 三种子结果，随后从该
  结果继续修改，而不是切换到 DDP 仓库开发。
- 从 `fix/independent-git-worktrees@1b1dcd2` 创建本地分支
  `exp/emotic-multilane-track-a-repro`；服务器分支尚未切换。
- 对照确认旧 `clip_vit_b16_patch` 与目标协议存在权重来源、768/512 维分类头、
  val+test/test、0.8/0.5 阈值、batch 256/64、AMP 和三种子口径等差异。
- 新增 `multi_lane/track_a/`：固定 OpenAI CLIP checkpoint 加载、目标 MULTI-LANE
  模型、严格训练/评估 runner、GPU smoke 和三种子汇总器。
- 新增 `scripts/emotic/` worker、直接 launcher 和 tmux launcher；旧入口保持不变。
- 新增单元测试，覆盖 task-slice 复制、concat inference、冻结视觉塔、固定阈值指标和
  逐类 forgetting。本地系统 Python 缺少 PyTorch/NumPy，已完成 AST/compile 与 shell
  语法检查，完整测试待服务器执行。
- 首次服务器单元测试中 3 个模型/forgetting 测试通过；AP 测试因旧公式的 `1e-8`
  epsilon 返回 `99.99999925` 而非数学上的 100，已将断言调整为 5 位小数容差。
- 容差修正后服务器 4/4 单元测试通过。
- 当前仓库 checkpoint loader 与目标 benchmark loader 的 152 个 CLIP visual state
  tensor 完全相同，最大权重误差和同输入输出误差均为 `0.0`，输出为 512 维。
- GPU0 forward/backward smoke 通过，可训练参数为 `689178`，冻结视觉塔没有梯度。
- 本地、远端和服务器同步后，以 `8cff911` 启动正式三种子训练：run ID
  `multi_lane_main_track_a_seed012_20260811_132448`，tmux
  `emotic_multilane_track_a_seed012`，seed 0/1/2 对应 GPU 0/1/2。
- task 0 三个 seed 的 mAP/cF1/oF1 与注册实验逐项完全相同；停止持续轮询，让 tmux 后台
  独立完成。最后一次快照位于 task 1 epoch 14–17/30，错误、NaN、OOM、skip 计数均为 0。
- 三种子正式实验已完成并输出 completion marker。最终 final mAP/cF1/oF1/average
  mAP/forgetting 为 `31.2994855/31.8111103/49.1092038/37.9985757/4.7884572`，所有
  mean/std 与原注册实验完全一致。
- 在服务器生成 `multi_lane_main_track_a_seed012_20260811_132448_download.tar.gz`，包含
  13 个 JSON 和 4 个日志、共 17 个文件，明确无 `checkpoints/`；大小约 80K，SHA-256
  `d3a2f23080da5ff6bd5d0fde99c6ebb9eae1916db31b1e09e9ebb6ed5fb15dcc`。
- 压缩包已通过 SSH 下载到本地项目 `./output/emotic_track_a/<run_id>/`，本地校验一致并
  解压到 `synced_files/`，可直接进行后续分析。

## 当前 Git 状态

权威 Git 仓库位置：

```text
/mnt/haoyuan/workspace/multi-lane-main
```

当前工作分支：

```text
feature/clip-taskCLS-posneg-text-head
```

当前服务器本地提交：

```text
Add CLIP DDP prompt head
```

状态说明：

- 该提交只存在于服务器本地分支，尚未 push 到远端。
- `train_c100.sh` 仍有一个未提交的末尾换行变化，未纳入 DDP 提交。

干净起点：

```text
52cf6b1 Add CLIP task CLS text head
```

该提交同时对应：

```text
origin/feature/clip-taskCLS-text-head
origin/feature/clip-taskCLS-text-siglip-scale
feature/clip-taskCLS-text-head
feature/clip-taskCLS-text-siglip-scale
```

当前 DDP 半成品历史：

```text
stash@{0}: On clip-taskCLS-posneg-text-head: wip clip_ddp draft before clean restart
stash@{1}: On clip-taskCLS-text-siglip-scale: wip siglip scale calibration experiment
stash@{2}: On clip-task-text-head: wip clip_patch_text experiment
```

## 本轮已完成修改

从干净分支重新实现 DDP v1 / PCD 结构，未直接应用 `stash@{0}`。

2026-06-22 已按用户要求在服务器上创建本地提交 `Add CLIP DDP prompt head`，未同步到远端。

2026-06-22 开始下一阶段修改：

- 新增 DDP eval diagnostics：评估时打印并保存 predicted positives、micro precision/recall/F1、prob/logit 分布和阈值 sweep。
- 新增 eval-only checkpoint 复评估入口：`--eval --eval_checkpoint <path>`，也兼容 `--eval_dir` 或 `output_dir/checkpoints.pth`。
- 新增 CLI 参数：`--ddp_diagnostics`、`--ddp_diagnostic_thresholds`、`--eval_checkpoint`。
- 当前这批下一阶段修改尚未提交。

业务代码改动：

- `multi_lane/clip_vit.py`
  - 新增 `clip_ddp` 独立 forward 分支。
  - 新增 class-specific positive/negative text prompts。
  - 新增 class-specific positive/negative visual prompts。
  - 新增 late-layer DDP prompt attention 注入。
  - 新增 token-wise similarity aggregation，支持 `mean`、`max`、`cls`。
  - 新增 PCD temperature 计算。
  - DDP final logit 使用 `CLIP logit_scale * (s_pos - s_neg) / tau`，避免纯 cosine margin 在固定阈值 F1 下不可达。
  - DDP 模式跳过 MULTI-LANE selectors/task CLS 分类路径。
- `multi_lane/engine.py`
  - DDP 训练 loss 只在当前 task classes 上计算。
  - DDP 评估 loss 与 mAP 使用已见类别 clean logits/targets。
  - DDP loss 不再额外使用项目通用 `args.temperature`，温度只由 DDP 内部训练 `tau=1` / 评估 PCD 控制。
- `multi_lane/utils.py`
  - DDP 模式只训练 `ddp_text_prompts` 和 `ddp_visual_prompts`。
  - DDP optimizer 参数组关闭 weight decay，降低旧类 prompt 漂移风险。
- `multi_lane/configs/*.py`
  - 增加 `head_mode=clip_ddp`。
  - 增加 DDP/PCD CLI 参数。

上下文文档：

- 重建 `AGENTS.md`。
- 重建 `PROJECT_CONTEXT.md`、`WORK_LOG.md`、`TODO_NEXT.md`。

## 已运行验证

本地已运行：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器 `multilane` 环境已运行：

```text
git diff --check
python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

服务器配置解析检查：

```text
REMOTE_PARSER clip_ddp True 3.0 0.7
```

服务器参数冻结检查：

```text
TRAINABLE ['ddp_text_prompts', 'ddp_visual_prompts']
PROMPTS (8, 2, 16, 512) (8, 2, 16, 768)
MASK_T0 [0, 1, 2]
OPT_GROUPS 1 0.0 327680
MASK_T1 1 [3, 4]
```

服务器 dummy forward/backward smoke：

```text
TRAIN_LOGITS (1, 4)
TRAIN_DEBUG ... 'class_ids': (2,), 'tau': 1.0, 'final_logits': (1, 4)
TRAIN_ACTIVE_GRAD [0, 1]
EVAL_LOGITS (1, 4)
EVAL_DEBUG ... 'class_ids': (4,), 'tau': 3.0, 'final_logits': (1, 4)
MASK_T1 [2, 3]
```

## 风险点

- `clip_ddp` 已完成 dummy forward/backward smoke，但新增 logit scale 修正后需要重新做一次服务器静态检查和 VOC smoke。
- 旧类 prompt 冻结逻辑已通过 dummy 梯度检查确认，正式训练前仍建议在首个真实 batch 再打印一次 learnable/gradient 状态。
- DDP binary softmax 当前用 `CLIP logit_scale * (s_pos - s_neg) / tau` 形式交给 `BCEWithLogitsLoss`；该尺度修正用于让 PCD 后的 score 与项目固定阈值 F1 兼容。
- EMOTIC 的 `tau_max/gamma` 没有论文指定值，当前只是采用 COCO 风格默认。

## VOC DDP smoke 分析

2026-06-22 的 `voc_clip_ddp_smoke` 结果：

```text
task0 mAP 96.1067, oF1 0.0, cF1 0.0
task1 mAP 71.9124, oF1 0.0, cF1 0.0
task2 mAP 64.1040, oF1 0.0, cF1 0.0
task3 mAP 51.0589, oF1 0.0, cF1 0.0
task4 mAP 50.4372, amAP 66.7238, oF1 0.0, cF1 0.0
```

明细报告显示每个 task 的 `predicted_positive` 总数均为 0。原因是 F1 使用项目固定规则 `sigmoid(logits) > 0.8`，而原实现 DDP logit 为纯 cosine margin `(s_pos - s_neg) / tau`；随着 PCD `tau` 增大，阈值所需 margin 不可达。已修正为带 CLIP `logit_scale` 的 DDP logit，需要重新运行 VOC smoke 验证。

2026-06-22 重新运行 `voc_clip_ddp_smoke_scale`，使用 `CLIP logit_scale * (s_pos - s_neg) / tau`：

```text
task0 mAP 98.7263, amAP 98.7263, oF1 93.9921, cF1 93.7758, loss 0.0862
task1 mAP 90.7199, amAP 94.7231, oF1 80.1021, cF1 78.4673, loss 0.2587
task2 mAP 80.4492, amAP 89.9651, oF1 69.1805, cF1 69.8794, loss 0.3413
task3 mAP 74.8414, amAP 86.1842, oF1 67.4487, cF1 65.8933, loss 0.3784
task4 mAP 71.5174, amAP 83.2508, oF1 64.0643, cF1 62.1179, loss 0.3756
```

该结果确认 scale 修正有效：最终 task 的 predicted positives 总数为 7175，20 个 seen classes 全部有正预测。当前只是 1 epoch VOC B4-C4 smoke，不能直接对齐论文 VOC B4-C2/B5-C3/B0-C4/B10-C2 协议；但相对旧 smoke，final mAP 从 50.44 提升到 71.52，OF1/CF1 从 0 恢复到 64.06/62.12。

## VOC 论文协议实验

2026-06-22 已为严格对齐 VOC B0-C4 修正 `multi_lane/datasets.py` 的 split 逻辑：

- `base_classes=0,num_tasks=5` 现在切分为 `[0-3],[4-7],[8-11],[12-15],[16-19]`。
- `base_classes=10,num_tasks=6` 保持 B10-C2：`[0-9],[10-11],[12-13],[14-15],[16-17],[18-19]`。

已启动 tmux：

```text
voc_b0c4_ddp_gpu1
  GPU: 1
  config: VOC B0-C4, epochs=20, batch_size=16, tau_max=7.0, gamma=0.2
  log: ./logs/voc_ddp_b0c4_20ep_tau7_g02.log
  output: ./output/voc_ddp_b0c4_20ep_tau7_g02

voc_b10c2_ddp_gpu0
  GPU: 0
  config: VOC B10-C2, epochs=20, tau_max=7.0, gamma=0.2
  initial batch_size=16/chunk=2 and batch_size=16/chunk=1 both OOM on GPU0 because other processes already occupied memory
  current retry: batch_size=8, ddp_class_chunk_size=1
  log: ./logs/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1.log
  output: ./output/voc_ddp_b10c2_20ep_tau7_g02_bs8_chunk1
```

用户明确要求：如果显存 OOM 退出，就等待，不要动别人服务器上正在跑的实验。

### 2026-06-22 结果分析

本地已同步 `voc_ddp_b0c4_20ep_tau7_g02` 完整结果：

```text
task0 mAP 99.2469, amAP 99.2469, oF1 95.3933, cF1 95.2475, loss 0.0536
task1 mAP 90.0728, amAP 94.6598, oF1 52.7571, cF1 49.3263, loss 0.3966
task2 mAP 82.8824, amAP 90.7340, oF1 33.8562, cF1 31.9064, loss 0.4603
task3 mAP 79.5050, amAP 87.9267, oF1 23.5792, cF1 23.1313, loss 0.4751
task4 mAP 76.9800, amAP 85.7374, oF1 21.5977, cF1 22.8244, loss 0.4872
```

关键症状：

- final mAP 仍有 76.98，说明排序信号没有完全坏掉。
- final F1 崩塌来自 recall 极低：20 类总 support 为 7632，只预测 943 个正样本，TP 926、FP 17、FN 6706。
- final precision 约 0.982，recall 约 0.121，说明 PCD/阈值过强，模型过度保守。
- 对比 1 epoch `voc_clip_ddp_smoke_scale`：final precision/recall 约 0.661/0.621；本次 VOC B0-C4 使用 `tau_max=7,gamma=0.2` 后 precision/recall 变为约 0.982/0.121。

当前判断：

- `tau_max=7,gamma=0.2` 来自论文附录 VOC B4-C2 的超参分析，不应直接假定适合 VOC B0-C4/B10-C2。
- 当前实现中的 PCD 对 VOC B0-C4 明显抑制过强，是 final CF1/OF1 远低于论文值的第一嫌疑。
- B10-C2 本地同步日志尚未出现最终 `Average performances` 或 detail JSON；当前只能确认其训练阶段 mAP/oF1 较高但 cF1 偏低，不能作为完整最终结果分析。

### 2026-06-22 DDP 诊断实验

本地已同步三组 VOC B0-C4 诊断结果：

```text
voc_ddp_b0c4_diag_tau3_g07_3ep
voc_ddp_b0c4_diag_eval_tau3_g07
voc_ddp_b0c4_diag_eval_pcd_off
```

关键结果：

```text
3ep tau3/gamma0.7 train final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval tau3/gamma0.7 final:     mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
eval PCD off final:           mAP 76.1609, amAP 85.3464, oF1 55.9951, cF1 59.0601, loss 0.4391
```

诊断结论：

- `train_tau3` 与 `eval_tau3` 每个 task 指标一致，说明 `--eval --eval_checkpoint` 路径可复现训练末尾评估。
- `tau_max=3,gamma=0.7` 不会出现 tau7/gamma0.2 的 recall 崩塌；final predicted positives/support 为 6544/7632，micro precision/recall/F1 为 0.695/0.596/0.642。
- `pcd_off` 的 final predicted positives/support 为 15170/7632，micro precision/recall/F1 为 0.421/0.837/0.560；关闭 PCD recall 更高，但误报太多，F1 反而低。
- mAP 在 tau3 与 PCD off 下基本一致，因为 PCD 是正尺度变换，主要影响固定阈值下的 F1，而不是排序。
- 当前 `tau_max=3,gamma=0.7` 是比 `pcd_off` 和 `tau_max=7,gamma=0.2` 更好的默认候选。

### 2026-06-23 PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
tau2/gamma0.7 final: mAP 76.1613, amAP 85.3464, oF1 62.9751, cF1 63.1456, loss 0.3584
tau5/gamma0.5 final: mAP 76.1613, amAP 85.3464, oF1 46.4039, cF1 45.9509, loss 0.4374
tau7/gamma0.2 final: mAP 76.1613, amAP 85.3464, oF1 20.3382, cF1 23.7697, loss 0.4850
```

与已有结果合并比较：

```text
pcd_off:       OF1 55.9951, CF1 59.0601, pred/support 15170/7632, P/R/F1 0.421/0.837/0.560
tau2/gamma0.7: OF1 62.9751, CF1 63.1456, pred/support 10102/7632, P/R/F1 0.553/0.732/0.630
tau3/gamma0.7: OF1 64.2071, CF1 62.1797, pred/support 6544/7632,  P/R/F1 0.695/0.596/0.642
tau5/gamma0.5: OF1 46.4039, CF1 45.9509, pred/support 2643/7632,  P/R/F1 0.902/0.312/0.464
tau7/gamma0.2: OF1 20.3382, CF1 23.7697, pred/support 884/7632,   P/R/F1 0.980/0.113/0.203
```

结论：

- tau7/gamma0.2 明确过强，复现 recall collapse；该配置不适合作为 VOC B0-C4 默认。
- tau5/gamma0.5 也偏保守，precision 高但 recall 明显不足。
- tau2/gamma0.7 与 tau3/gamma0.7 接近：tau2 提高 recall 和 CF1，tau3 提高 precision、micro F1 和 OF1。
- 若按论文同时看 OF1/CF1，tau3/gamma0.7 当前更均衡；tau2/gamma0.7 可作为 macro-F1 候选保留。
- 下一步不应继续调更强 PCD，应进入 aggregation 对照，优先用 tau3/gamma0.7，同时可保留 tau2/gamma0.7 做备选复核。

### 2026-06-23 aggregation 对照

本地已同步 VOC B0-C4 3 epoch aggregation 对照，固定 `tau_max=3.0,gamma=0.7`：

```text
mean final: mAP 76.1613, amAP 85.3464, oF1 64.2071, cF1 62.1797, loss 0.3767
max final:  mAP 71.6560, amAP 81.9845, oF1 59.6344, cF1 59.1139, loss 0.3544
cls final:  mAP 77.7732, amAP 86.6709, oF1 64.0544, cF1 63.8777, loss 0.3535
```

诊断对比：

```text
mean: pred/support 6544/7632, P/R/F1 0.695/0.596/0.642
max:  pred/support 6920/7632, P/R/F1 0.627/0.569/0.596
cls:  pred/support 9569/7632, P/R/F1 0.576/0.722/0.641
```

类别层面：

- `cls` 明显改善了部分 mean 下偏保守的局部/物体类：diningtable F1 0.270 -> 0.529，pottedplant 0.329 -> 0.552，motorbike 0.645 -> 0.853，bus 0.583 -> 0.775，chair 0.372 -> 0.441。
- `cls` 也带来一些误报：sheep 0.521 -> 0.291，bicycle 0.615 -> 0.445，bird 0.671 -> 0.466，train 0.690 -> 0.590。
- `max` 没有达到“局部 token 最大值更好”的预期，final mAP、OF1、CF1 均低于 mean/cls，暂停作为主路线。

当前判断：

- `cls` 是当前 aggregation 的最好候选，mAP 和 CF1 均优于 mean，OF1 与 mean 基本持平。
- `cls` 的分数分布更偏召回，后续建议在 `cls` checkpoint 上复评估 tau2/tau3/tau5，确认是否能通过 PCD 强度把 precision/recall 再平衡。
- 如果 `cls + tau3` 或 `cls + tau5` 复评估稳定优于 mean，再用该配置跑 20 epoch B0-C4。

### 2026-06-23 cls PCD sweep 复评估

基于同一个 checkpoint：

```text
./output/voc_ddp_b0c4_diag_tau3_g07_cls_3ep/checkpoints.pth
```

已完成三组新增复评估：

```text
cls tau2/gamma0.7 final: mAP 77.7733, amAP 86.6713, oF1 60.3633, cF1 62.0513, loss 0.3796
cls tau5/gamma0.5 final: mAP 77.7733, amAP 86.6708, oF1 55.3263, cF1 55.4358, loss 0.3792
cls tau7/gamma0.2 final: mAP 77.7732, amAP 86.6708, oF1 36.0121, cF1 39.6079, loss 0.4183
```

与已有 `cls tau3/gamma0.7` 合并比较：

```text
cls tau2/gamma0.7: pred/support 13013/7632, P/R/F1 0.479/0.816/0.604
cls tau3/gamma0.7: pred/support  9569/7632, P/R/F1 0.576/0.722/0.641
cls tau5/gamma0.5: pred/support  4825/7632, P/R/F1 0.714/0.452/0.553
cls tau7/gamma0.2: pred/support  2298/7632, P/R/F1 0.778/0.234/0.360
```

结论：

- 固定项目/论文主阈值 `sigmoid(logits)>0.8` 下，`cls + tau3/gamma0.7` 最均衡，OF1/CF1 最高。
- `cls + tau2/gamma0.7` recall 过强，误报变多，OF1/CF1 均低于 tau3。
- `cls + tau5/gamma0.5` 与 `cls + tau7/gamma0.2` 均偏保守，尤其 task3/task4 的 recall 明显下降。
- `cls + tau5` 在诊断阈值 0.7 下 micro-F1 接近 tau3，但主实验不能改阈值，因此不作为当前主配置。
- 已按 `head_mode=clip_ddp, ddp_similarity_aggregation=cls, ddp_tau_max=3.0, ddp_gamma=0.7` 跑完 VOC B0-C4 20 epoch，并保留 checkpoint。

### 2026-06-23 VOC B0-C4 cls 20 epoch

本地已同步并分析：

```text
log:    ./logs/voc_ddp_b0c4_cls_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_cls_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_cls_tau3_g07_20ep/checkpoints.pth
```

实验配置：

```text
dataset Split-VOC, num_tasks=5, base_classes=0
backbone clip_vit_b16_patch
head_mode clip_ddp
epochs=20, batch_size=16, optimizer=adam, lr=0.05 scaled to 0.003125
ddp_prompt_length=16, ddp_prompt_layers=5
ddp_similarity_aggregation=cls
ddp_pcd=true, ddp_tau_max=3.0, ddp_gamma=0.7
ddp_class_chunk_size=2, store_model=true
```

逐 task 结果：

```text
task0 mAP 99.5071, amAP 99.5071, oF1 97.0199, cF1 96.9398, loss 0.0418
task1 mAP 91.8224, amAP 95.6648, oF1 76.2983, cF1 79.7620, loss 0.3046
task2 mAP 84.4233, amAP 91.9176, oF1 66.9186, cF1 71.0141, loss 0.3542
task3 mAP 81.4017, amAP 89.2886, oF1 68.3450, cF1 69.1338, loss 0.3587
task4 mAP 79.7877, amAP 87.3884, oF1 65.7037, cF1 64.7468, loss 0.3690
```

最终诊断：

```text
pred/support 8568/7632
TP/FP/FN 5322/3246/2310
micro precision/recall/F1 0.621/0.697/0.657
threshold sweep F1: th0.5 0.465, th0.7 0.613, th0.8 0.657, th0.9 0.589
```

对比：

```text
vs cls 3ep tau3:   +2.01 mAP, +0.72 amAP, +1.65 OF1, +0.87 CF1
vs mean 3ep tau3:  +3.63 mAP, +2.04 amAP, +1.50 OF1, +2.57 CF1
vs mean 20ep tau7: +2.81 mAP, +1.65 amAP, +44.11 OF1, +41.92 CF1
```

类别层面：

- 最差 F1 类别：chair 0.327，bicycle 0.368，sheep 0.415，diningtable 0.477，sofa 0.499，pottedplant 0.508，bottle 0.520，bird 0.547。
- 20 epoch 相比 3 epoch 改善较大：sheep、dog、tvmonitor、bird、bottle、cat、train、cow。
- 20 epoch 相比 3 epoch 下降较大：chair、sofa、motorbike、bicycle、diningtable、pottedplant、bus。

结论：

- `cls + tau3/gamma0.7` 20 epoch 是当前 VOC B0-C4 最好结果，且已经解决 tau7/gamma0.2 的 recall collapse。
- 20 epoch 让整体 mAP 和 micro F1 上升，precision 从 0.576 提到 0.621，recall 从 0.722 降到 0.697，说明训练更稳但更保守。
- 当前 gap 不再主要是 PCD 强度一眼错配，而是 per-class calibration 和 prompt 表达不足；chair/diningtable/sofa/pottedplant 这类小目标/上下文依赖类偏低，bicycle/sheep/bird 有明显误报。
- 下一步优先使用该 checkpoint 做 eval-only PCD 微调，不先重训：复评估 `pcd_off`、`tau2/gamma0.7`、`tau2.5/gamma0.7`，必要时再看 `tau3.5/gamma0.7`。

### 2026-06-23 DDP v1/PCD 结构核对

已核对 DDP v1 论文源码 `arxiv:2509.23335v1` 和当前 `clip_ddp` 实现。

确认匹配：

- 一类一套 positive/negative text prompts 与 positive/negative visual prompts。
- 训练时只训练当前 task classes prompts，推理时对 seen classes 计算。
- 冻结 CLIP encoders，不使用 MULTI-LANE selectors/task CLS 作为 DDP 分类路径。
- PCD v1 公式与当前实现一致：训练 `tau=1`，推理按 `tau_max/gamma` 随 seen classes 增长。
- visual prompt 插入最后 5 个 CLIP visual layers，并在 MSA 输出后切掉 prompt token；interlayer shared prompts 是论文写法，不需要每层一套 prompt。
- optimizer 每 task 重建，只接收 DDP prompts 且 weight decay 为 0，配合 gradient mask 能避免旧类 prompt 被 Adam 动量或 weight decay 继续更新。

高风险差异/疑点：

- 论文公式是 `softmax(s+ / tau, s- / tau)`，`s` 为 cosine similarity；当前实现输出 `CLIP logit_scale * (s+ - s-) / tau`。该改动解决了项目固定 `sigmoid(logits)>0.8` 下全 0 的问题，但会改变 PCD 温度含义。需要把严格 DDP probability/logit 路径和项目阈值兼容路径显式分开。
- visual prompt 注入时，当前代码对原始 visual tokens 做 `block.norm1(x)`，但 prompts 本身没有经过同一个 LayerNorm 后再进入 Q/K/V。更严格的 pre-norm ViT 实现应考虑先拼接 `[P; x]` 再对整段序列做 `norm1` 和 attention，最后切掉 prompt tokens。
- 论文 v1 正文明确描述了 interlayer prompting 内部的 token 维度：`P^c` 为 `L_P x d`，image hidden states 为 `L x d`，prompt 参与 MSA 后切掉 prompt tokens 并保持 `L x d`。但公式中的 `E_V(P_V^c, x)` 被抽象为可直接与 text feature 做 cosine 的 visual feature，正文没有明确要求 token-wise similarity 再聚合。因此，当前 `mean/max/cls` 以及后续 `patch_mean/patch_max/topk` 应表述为实现级 visual evidence aggregation 诊断，而不是论文明确规定。
- 论文 v1 implementation details 写 prompt length 为 16；当前匹配。论文后续 DeCLIP/AST 稿改成 prompt length 4 和 `lambda=16`，这属于版本差异，不应混到当前 DDP v1/PCD 目标中。
- 论文附录提到 VOC DDP learnable params 约 `0.49M`，当前 VOC 为 `0.819M = text prompts + visual prompts`。这说明论文参数统计或实际实现可能没有把 text prompts 作为同等可训练参数计入，需优先核对；也应做 text-only/visual-only/pos-only/neg-only ablation。
- text side 当前是 `[SOT] + learnable prompt + class name + '.' + [EOT]`，positive/negative 只靠不同 learnable embeddings 区分；论文未给自然语言正负模板，所以这不是明确 bug，但需要通过 prompt 初始化/模板 ablation 验证。

结构排查结论：

- 当前实现已经是 DDP 大框架，但还不能说严格复现论文实现。
- 最值得先改/查的是 logit scale 与 PCD 的关系、visual prompt LayerNorm 注入位置、CLIP 常规 CLS/pooled visual feature 路径，以及 learnable parameter 统计/文本 prompt 是否该完全可训练；token-wise aggregation 只作为补充实验。

### 2026-06-23 DDP 结构开关实现

本地已完成第一批结构修正开关，默认值保持旧实验可复现：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_logit_scale_mode clip|none`。
  - 新增 `--ddp_prompt_norm_mode legacy|prompted`。
  - 扩展 `--ddp_similarity_aggregation`：`pooled_cls`、`patch_mean`、`patch_max`、`cls_plus_patch_max`、`topk_mean`。
  - 新增 `--ddp_similarity_topk`。
  - 新增 `--ddp_train_text_prompts`、`--ddp_train_visual_prompts`、`--ddp_prompt_polarity`。
- `multi_lane/clip_vit.py`
  - `ddp_logit_scale_mode=none` 使用严格 raw cosine margin `(s+ - s-) / tau`；`clip` 保留当前 `CLIP logit_scale * (s+ - s-) / tau`。
  - `ddp_prompt_norm_mode=prompted` 将 `[prompt; image tokens]` 一起送入 `block.norm1` 再做 MSA，之后切掉 prompt tokens；`legacy` 保留旧路径。
  - `pooled_cls` 与 `cls` 都使用 class-conditioned CLS/pooled feature 和 text feature 直接 cosine，作为最贴近论文公式抽象 `cos(E_T(...), E_V(...))` 的路径命名。
  - `patch_mean/patch_max/cls_plus_patch_max/topk_mean` 作为实现级 visual evidence aggregation 诊断项。
  - gradient mask 支持 text/visual prompt 分支和 positive/negative 分支训练开关。
- `multi_lane/utils.py`
  - DDP optimizer 参数选择遵守 `ddp_train_text_prompts` 和 `ddp_train_visual_prompts`，避免被冻结分支进入 optimizer。

论文对照：

- 对应公式 1/2：新增 `ddp_logit_scale_mode=none`，可回到论文写法中的 cosine similarity binary softmax；`clip` 是项目固定阈值兼容分支。
- 对应 interlayer prompting 公式 4：新增 `prompted` pre-norm 路径，让 prompt 和 image tokens 在 MSA 前共同归一化，再切片回 `L x d`。
- 对应 `E_V(P_V^c,x)` 抽象：新增 `pooled_cls` 命名，明确表示单个 class-conditioned visual feature，而不是声称论文要求 token-wise 聚合。

已验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。本地环境缺少 `open_clip`，forward smoke 需要在服务器 `multilane` 环境执行。

### 2026-06-23 VOC B0-C4 结构对照 3 epoch

已同步并分析三组 VOC B0-C4 结构对照实验，配置均为 `num_tasks=5, base_classes=0, epochs=3, prompt_length=16, prompt_layers=5, PCD tau3/gamma0.7, batch_size=8, chunk_size=1, store_model=true`。

结果：

```text
pooled_cls + prompted + raw cosine:
  final mAP/amAP/OF1/CF1 = 81.7107 / 89.1806 / 0.0000 / 0.0000
  pred/support = 0 / 7632 at threshold 0.8
  threshold 0.5 micro P/R/F1 = 0.361 / 0.910 / 0.517

pooled_cls + prompted + CLIP logit_scale:
  final mAP/amAP/OF1/CF1 = 79.8608 / 87.8689 / 61.5493 / 62.9698
  pred/support = 10906 / 7632
  micro P/R/F1 = 0.523 / 0.748 / 0.615
  threshold 0.9 micro P/R/F1 = 0.635 / 0.562 / 0.596

cls + legacy + CLIP logit_scale control:
  final mAP/amAP/OF1/CF1 = 77.1030 / 85.8240 / 61.5543 / 64.1565
  pred/support = 10846 / 7632
  micro P/R/F1 = 0.524 / 0.745 / 0.616
  threshold 0.9 micro P/R/F1 = 0.661 / 0.549 / 0.600
```

结论：

- `raw cosine` 更接近 DDP 论文公式，但在当前项目主指标 `sigmoid(logits)>0.8` 下完全不可用；它不是排序崩了，因为 mAP 达到 `81.71`，而是 logit 尺度太小，固定阈值不兼容。
- `pooled_cls + prompted + clip` 相比旧 control 提高 `+2.76 mAP`、`+2.04 amAP`，OF1 基本持平，CF1 低 `1.19`。这说明更贴近论文的 visual prompt pre-norm 和 pooled visual feature 路径是正向结构改动。
- F1 没明显变好，主要因为两组 `clip` 配置在固定阈值下的 precision/recall 形态几乎一样：predicted positives 都约 `1.09e4`，micro F1 都约 `0.615`。
- 下一步应把 `pooled_cls + prompted + clip + tau3/gamma0.7` 作为新的 20 epoch 候选，而不是继续围绕旧 `cls + legacy` 做长实验。
- `raw cosine` 保留为论文公式对照/排序诊断，不作为当前项目主表配置，除非后续把评估指标从固定 sigmoid 阈值改成 DDP binary probability 或按验证集校准阈值。

### 2026-06-23 DDP gap 分析补充

用户提供了 GPT5.5Thinking 对当前 DDP gap 的独立分析。其判断与当前实验结论基本一致：当前 `clip_ddp` 已接近 DDP 主框架，但论文差距更可能来自 score/probability 语义、阈值协议、CLIP preprocessing、text prompt 初始化、visual pooling 和 VOC protocol，而不是缺少正负 prompt 或 PCD 这类大模块。

更新后的高优先级排查顺序：

- 增加严格 DDP probability 评估路径：保留 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`p_pos=sigmoid(margin/tau)`，并支持 probability mode 下不再二次 sigmoid。
- 增加更密集 threshold sweep 和 oracle threshold 诊断，用来区分结构排序能力与固定阈值 calibration 问题。
- 保存每个 task/class/sample 的 `y_true/s_pos/s_neg/margin/p_pos/scaled_logit/class_id`，定位哪些类是过度预测、哪些类过度保守。
- 增加 CLIP mean/std 输入归一化选项，和当前 no-normalize/ImageNet normalize 对照。
- 打印 VOC task class order/seen class order，确认是否严格符合论文 lexicographic protocol。
- 在上述诊断后，再跑 `pooled_cls + prompted + clip` 20 epoch；如果 strict probability/oracle F1 仍差，再做 text/visual/polarity prompt ablation 与 semantic text initialization。

当前最强证据仍是 raw cosine 3 epoch `mAP=81.71` 但主阈值 F1 为 0：排序能力和固定阈值 calibration 已经明显解耦。

### 2026-06-23 DDP probability / calibration 诊断实现

已实现 GPT5.5Thinking 建议的第一批诊断改动：

- `multi_lane/clip_vit.py`
  - DDP forward 现在除主 logits 外，会缓存完整 `s_pos`、`s_neg`、`margin=s_pos-s_neg`、`ddp_prob=sigmoid(margin/tau)`、`scaled_logit=CLIP logit_scale*margin/tau`。
  - 主训练/默认评估仍返回 scaled logits，不破坏旧实验复现。
- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_eval_score_mode logits|probability`。
  - 新增 `--ddp_eval_threshold`，默认 `0.8`。
  - 新增 `--ddp_score_dump true|false`。
  - 默认 DDP diagnostic thresholds 扩展为 `0.1 ... 0.9`。
  - 新增 `--clip_normalize_input true|false`。
- `multi_lane/engine.py`
  - DDP eval metrics 可选择 `logits` 模式：`sigmoid(scaled_logit)>threshold`，或 `probability` 模式：`ddp_prob>threshold`，后者不再二次 sigmoid。
  - DDP diagnostics 增加 dense threshold sweep、global oracle threshold、per-class oracle F1 诊断。
  - detail report 增加 `s_pos/s_neg/margin/ddp_prob/scaled_logit` 分布字段。
  - `--ddp_score_dump true` 时保存 `detail/{run_name}_task{t}_ddp_scores.npz`，包含 `y_true/eval_score/s_pos/s_neg/margin/ddp_prob/scaled_logit/class_ids`。
- `multi_lane/datasets.py`
  - 新增 CLIP mean/std normalization：`mean=(0.48145466,0.4578275,0.40821073)`，`std=(0.26862954,0.26130258,0.27577711)`。
  - `--clip_normalize_input true` 优先于旧 `--normalize_input`；默认仍保持旧行为。
- `main.py`
  - 每次运行打印并保存 `class_order.json`，用于核对 VOC lexicographic task protocol。

验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

下一步实验目的：

- 用同一 checkpoint 做 eval-only 对照，比较 `logits` 与 `probability` 评估语义。
- 开启 `ddp_score_dump`，检查 strict DDP probability/oracle threshold 下 F1 是否接近论文。
- 如果 strict probability + oracle 仍差，再继续查 text prompt 初始化、CLIP normalize 和 visual pooling。

### 2026-06-23 DDP calibration 诊断实验启动

已按用户要求在服务器 `/mnt/haoyuan/workspace/multi-lane-main` 启动三个 tmux 实验：

```text
tmux: voc_ddp_eval_logits_diag
gpu: 0
log: ./logs/voc_ddp_eval_logits_diag.log
output: ./output/voc_ddp_eval_logits_diag
purpose: 同一 pooled_cls+prompted checkpoint，使用当前 scaled-logit 语义复评估，并保存 dense threshold/oracle/score dump。

tmux: voc_ddp_eval_probability_diag
gpu: 1
log: ./logs/voc_ddp_eval_probability_diag.log
output: ./output/voc_ddp_eval_probability_diag
purpose: 同一 checkpoint，使用严格 DDP probability 语义复评估，检查 raw DDP probability 下阈值/校准问题。

tmux: voc_ddp_3ep_clipnorm
gpu: 4
log: ./logs/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm.log
output: ./output/voc_ddp_3ep_pooledcls_prompted_clip_clipnorm
purpose: 训练 3 epoch CLIP mean/std normalized 输入，对照 no-normalize 下的 pooled_cls+prompted+clip。
```

启动后确认：

- 三个 tmux session 均存在。
- `voc_ddp_eval_logits_diag` 已输出 task2 diagnostics。
- `voc_ddp_eval_probability_diag` 已输出 task1 diagnostics。
- `voc_ddp_3ep_clipnorm` 已进入训练并打印配置。

### 2026-06-23 DDP calibration 诊断结果

三组实验已同步并分析。

1. `voc_ddp_eval_logits_diag`

使用同一 `pooled_cls + prompted + clip` 3 epoch checkpoint，评估语义为当前工程 scaled logits：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8689 / 61.5493 / 62.9698
pred/support = 10906 / 7632
micro P/R/F1 = 0.523 / 0.748 / 0.615
global oracle threshold = 0.8, oracle micro-F1 = 0.615
per-class oracle F1 = 0.696
```

结论：dense threshold sweep 证明 final task 的全局最佳阈值仍是 `0.8`，不是简单调全局阈值就能接近论文。per-class oracle 从 `62.97` 提到约 `69.6`，说明存在明显 class-wise calibration 问题，但 oracle 后仍低于论文主表。

2. `voc_ddp_eval_probability_diag`

同一 checkpoint，评估语义为严格 DDP probability：`ddp_prob=sigmoid((s_pos-s_neg)/tau)`，主阈值 `0.5`：

```text
final mAP/amAP/OF1/CF1 = 79.8607 / 87.8687 / 46.2331 / 51.5324
pred/support = 22459 / 7632
micro P/R/F1 = 0.310 / 0.911 / 0.462
global oracle threshold = 0.5, oracle micro-F1 = 0.462
per-class oracle F1 = 0.515
```

结论：严格 DDP probability 没有改善 F1，反而明显更差。`ddp_prob` 分布集中在 0.5 附近，低于 0.5 基本全预测，高于 0.6 基本无预测，说明 raw margin 尺度太小；当前论文公式路径排序仍可用，但 probability calibration 不适合项目固定阈值。

3. `voc_ddp_3ep_pooledcls_prompted_clip_clipnorm`

在 `pooled_cls + prompted + clip` 基础上启用 CLIP mean/std 输入归一化训练 3 epoch：

```text
final mAP/amAP/OF1/CF1 = 81.5703 / 88.7328 / 65.0564 / 66.6065
pred/support = 10792 / 7632
micro P/R/F1 = 0.555 / 0.785 / 0.651
global oracle threshold = 0.9, oracle micro-F1 = 0.669
per-class oracle F1 = 0.713
```

对比 no-normalize 的同结构 3 epoch：

```text
no-normalize: mAP/OF1/CF1 = 79.8607 / 61.5493 / 62.9698
CLIP normalize: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
gain: +1.71 mAP, +3.51 OF1, +3.64 CF1
```

对比旧 `cls + legacy + clip` 20 epoch：

```text
old 20ep: mAP/OF1/CF1 = 79.7877 / 65.7037 / 64.7468
clipnorm 3ep: mAP/OF1/CF1 = 81.5703 / 65.0564 / 66.6065
```

结论：CLIP preprocessing 是重要 gap。仅 3 epoch 的 `pooled_cls + prompted + clip + CLIP normalize` 已经超过旧 20 epoch 的 mAP/CF1，OF1 基本接近。下一步应优先跑该配置 20 epoch。

当前 updated 判断：

- 严格 probability 评估说明 gap 不是“把评估改回 DDP probability 就解决”，raw margin 尺度本身过小。
- CLIP mean/std normalize 明确有效，应作为后续 VOC 主配置。
- 当前剩余问题更偏 class-wise calibration 和 prompt/类别语义，尤其 sheep/bicycle/bottle 等误报仍多，diningtable/chair 等仍偏难。

### 2026-06-23 VOC B0-C4 pooled_cls+prompted+clipnorm 20 epoch

本地已同步主候选 20 epoch 结果：

```text
log:    ./logs/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep.log
output: ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep
ckpt:   ./output/voc_ddp_b0c4_pooledcls_prompted_clipnorm_tau3_g07_20ep/checkpoints.pth
```

配置：

```text
dataset: VOC B0-C4
head: clip_ddp
aggregation: pooled_cls
prompt_norm_mode: prompted
logit_scale_mode: clip
clip_normalize_input: true
PCD: tau_max=3.0, gamma=0.7
epochs: 20
batch_size: 8
chunk_size: 1
```

最终结果：

```text
task1: mAP/OF1/CF1 = 99.6331 / 97.8536 / 97.8136
task2: mAP/OF1/CF1 = 94.1302 / 76.4492 / 81.0003
task3: mAP/OF1/CF1 = 86.0975 / 67.8669 / 72.8785
task4: mAP/OF1/CF1 = 83.5890 / 69.1588 / 72.1785
task5: mAP/OF1/CF1 = 82.9480 / 67.3165 / 68.9025
```

final diagnostics:

```text
threshold 0.8: pred/support=10527/7632, P/R/F1=0.581/0.801/0.673
threshold 0.9: pred=6730, P/R/F1=0.736/0.649/0.690
global oracle threshold=0.9, oracle micro-F1=0.690
per-class oracle F1=0.748
```

对比：

```text
vs old cls+legacy+clip 20ep:
  +3.16 mAP, +1.61 OF1, +4.16 CF1

vs clipnorm 3ep:
  +1.38 mAP, +2.26 OF1, +2.30 CF1

vs no-normalize pooled_cls+prompted 3ep:
  +3.09 mAP, +5.77 OF1, +5.93 CF1
```

结论：

- 这是当前 VOC B0-C4 最佳结果，证明 `pooled_cls + prompted + CLIP normalize` 是正确主线。
- 20 epoch 没有过拟合崩塌，较 3 epoch 继续提升。
- 但当前 final task 全局最佳阈值已经偏向 `0.9`，说明模型仍偏误报；下一步优先做 eval-only PCD/threshold 复评估，而不是马上改结构。
- 相对 DDP 论文 VOC B0-C4 主表，当前约低 `7.3 mAP / 13.5 OF1 / 8.0 CF1`，gap 明显缩小但还没到论文水平。
- 最差类别仍集中在 `bicycle/sheep/bottle/chair/diningtable/car`。其中 bicycle/sheep/bottle 是高 recall 低 precision 的误报问题； diningtable 是高 precision 低 recall 的漏检问题；chair AP/F1 都偏低。

下一步建议：

- 用该 checkpoint 做 eval-only：`tau2.5/tau3/tau3.5`，并比较 `ddp_eval_threshold=0.8/0.9`。
- 若 threshold 0.9 + 合理 PCD 明显提升 OF1/CF1，再跑 B10-C2 或 20 epoch 正式论文协议。
- 若仍明显低于论文，再进入 text prompt semantic initialization 和 class-wise calibration。

### 2026-06-23 下一阶段 margin-first 计划

用户提供 GPT5.5Thinking 新判断：当前核心问题应从 PCD/threshold 转向 positive/negative prompt margin separation。结合最新 `pooled_cls+prompted+clipnorm` 20 epoch 结果，当前判断如下：

- CLIP normalize 和 prompted pooled feature 已确认有效，主结构路线正确。
- strict DDP probability final OF1/CF1 只有 `46.23/51.53`，且 `ddp_prob` 大量挤在 0.5 附近，说明 raw margin 本身不够大。
- scaled-logit 分支能得到 `82.95/67.32/68.90`，说明排序信息存在，但依赖 CLIP scale/threshold 做工程校准。
- final threshold 0.9 比 0.8 的 micro-F1 更高，说明误报仍偏多，尤其 `bicycle/sheep/bottle`；而 `diningtable/chair` 仍有漏检/表达弱问题。

下一阶段计划：

1. 先做 checkpoint eval-only calibration sweep，但目标是读 margin 而不是继续盲调 PCD：
   - `tau2.5/tau3/tau3.5` × `threshold0.8/0.9`
   - 重点看 `s_pos/s_neg/margin/ddp_prob` 分布、per-class margin gap、oracle threshold。
2. 增加 margin summary report：
   - 每类正样本 margin mean/p10/p50/p90。
   - 每类负样本 margin mean/p10/p50/p90。
   - `pos_margin_mean - neg_margin_mean`。
   - 正样本 raw prob 大于 0.5/0.6 的比例、负样本 raw prob 大于 0.5/0.6 的比例。
3. 做 prompt branch ablation：
   - full text+visual pos+neg。
   - visual-only。
   - text-only。
   - text frozen vs text trainable。
   - visual frozen vs visual trainable。
   - positive-only / negative ablation。
4. 做 semantic text initialization：
   - positive templates: `a photo containing a {class}.`, `a photo of a {class}.`, `a photo with a {class}.`
   - negative templates: `a photo without a {class}.`, `a photo not containing a {class}.`, `a photo with no {class}.`
   - 观察是否提升 positive sample 上的 `s_pos`、negative sample 上的 `s_neg`，并扩大 margin gap。
5. 如果 ablation 证明 DDP loss 本身不足以拉开 margin，再考虑 margin regularizer；该步骤属于超出 DDP v1 的改进，需要单独标记为 extension，而不是论文复现配置。

### 2026-06-23 margin-first 代码改动

本轮按用户要求进入 positive/negative prompt margin separation 排查，已完成以下未提交改动：

- `multi_lane/configs/custom_types.py`
  - 新增 `--ddp_text_init random|same|semantic`。
  - 新增 `--ddp_positive_text_template`、`--ddp_negative_text_template`。
- `multi_lane/clip_vit.py`
  - `random` 保持旧 DDP text prompt 随机初始化，保证已有实验可复现。
  - `same` 让 positive/negative text prompt 从相同随机值开始，用于诊断正负分支是否真的学出分离。
  - `semantic` 用正/负模板前缀分别初始化 learnable text prompt，例如 positive 默认来自 `a photo containing a`，negative 默认来自 `a photo without a`。
  - semantic 初始化只在 `set_class_names()` 后执行一次；eval-only 加载 checkpoint 时，checkpoint 参数会覆盖初始化值，不污染旧模型复评估。
- `multi_lane/engine.py`
  - detail report 新增 per-class positive/negative margin 分布。
  - overall diagnostics 新增 `diag_margin_gap_mean`、正/负样本 margin 大于 0 的比例、正/负样本 raw `ddp_prob` 大于 0.5/0.6 的比例。
  - 评估日志新增 `[DDP margin task ...]` 行，便于直接观察 raw margin 是否拉开。
- `PROJECT_CONTEXT.md`、`TODO_NEXT.md`、`WORK_LOG.md`
  - 已同步记录本轮改动、目的和下一轮实验方向。

本地验证：

```text
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

本地 `python3 main.py voc --help` 因本机 Python 缺少 `numpy` 无法运行；已用直接 parser 检查确认新增参数可见。服务器 `multilane` 环境仍需运行完整 `main.py voc --help` 和 smoke。

### 2026-06-23 VOC B0-C4 margin init 对照结果

用户已同步两组 3 epoch 对照：

```text
random:
  log: ./logs/voc_ddp_margin_random_3ep.log
  output: ./output/voc_ddp_margin_random_3ep
semantic:
  log: ./logs/voc_ddp_margin_semantic_3ep.log
  output: ./output/voc_ddp_margin_semantic_3ep
```

共同配置：

```text
VOC B0-C4, num_tasks=5, base_classes=0, epochs=3
head_mode=clip_ddp
aggregation=pooled_cls
prompt_norm_mode=prompted
logit_scale_mode=clip
clip_normalize_input=true
tau_max=3.0, gamma=0.7
ddp_score_dump=true
```

final 结果：

```text
random:   mAP 82.3370, amAP 89.1434, OF1 66.7329, CF1 67.8947
semantic: mAP 83.8680, amAP 90.1785, OF1 70.2598, CF1 70.6409
delta:   +1.5310 mAP, +1.0351 amAP, +3.5269 OF1, +2.7462 CF1
```

final threshold 0.8 诊断：

```text
random:   pred/support=10491/7632, P=0.5764, R=0.7923, F1=0.6673
semantic: pred/support= 8417/7632, P=0.6698, R=0.7387, F1=0.7026
```

margin 诊断：

```text
random:   pos_margin_mean=0.0940, neg_margin_mean=-0.0747, gap=0.1688
semantic: pos_margin_mean=0.0711, neg_margin_mean=-0.0645, gap=0.1356
```

结论：

- `semantic` 明显提升主指标，主要原因是减少过度预测，precision 从 `0.5764` 提升到 `0.6698`。
- `semantic` 没有直接扩大 raw margin gap；gap 反而从 `0.1688` 降到 `0.1356`。
- 两组 final `pos_ddp_prob>0.6` 和 `neg_ddp_prob>0.6` 都是 0，说明 raw DDP probability 仍然挤在 0.5 附近，raw margin 绝对尺度仍不足。
- 因此 semantic init 可以作为下一轮主配置候选，但它解决的是 scaled-logit 校准/误报问题，不是严格意义上直接把 raw DDP margin 拉大。
- 下一步应在 semantic 配置上继续做 branch-level 排查：冻结 text prompts / 冻结 visual prompts / positive-only / negative-only，以定位 margin gap 主要受哪一支限制。

### 2026-06-23 semantic 20ep 与 branch ablation 结果

用户已同步五组结果：

```text
semantic 20ep:
  log: ./logs/voc_ddp_margin_semantic_20ep.log
  output: ./output/voc_ddp_margin_semantic_20ep

branch ablation 3ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_visual_frozen_3ep.log
  ./logs/voc_ddp_ablate_sem_posonly_3ep.log
  ./logs/voc_ddp_ablate_sem_negonly_3ep.log
```

final summary：

```text
random 3ep:        mAP 82.3370, OF1 66.7329, CF1 67.8947, gap 0.1688, pred 10491
full semantic 3ep: mAP 83.8680, OF1 70.2598, CF1 70.6409, gap 0.1356, pred  8417
semantic 20ep:     mAP 84.2737, OF1 70.5397, CF1 72.5721, gap 0.1252, pred  9323
text frozen 3ep:   mAP 82.3217, OF1 71.1562, CF1 72.7656, gap 0.1298, pred  9044
visual frozen 3ep: mAP 82.4743, OF1 68.7695, CF1 69.7780, gap 0.1270, pred  8695
positive-only 3ep: mAP 81.2522, OF1 70.9906, CF1 71.5840, gap 0.1186, pred  8156
negative-only 3ep: mAP 81.3779, OF1 68.4138, CF1 72.4098, gap 0.1355, pred  9768
old best 20ep:     mAP 82.9480, OF1 67.3165, CF1 68.9025
```

关键结论：

- `semantic 20ep` 达到当前最高 mAP：`84.27`，相对 old best `82.95` 提升 `+1.33 mAP`，相对 `semantic 3ep` 只提升 `+0.41 mAP`。
- 固定阈值 F1 的当前最好不是 full semantic 20ep，而是 `text_frozen 3ep`：`OF1 71.16 / CF1 72.77`。
- `text_frozen` 表示固定 semantic text prompts，只训练 visual prompts。它比 `visual_frozen` 明显好，说明当前 F1 主要依赖 visual prompt 学到 class-conditioned evidence。
- full semantic 从 3ep 到 20ep 主要增加 recall：`0.739 -> 0.784`，但 precision 从 `0.670 -> 0.641` 降低，predicted positives 从 `8417 -> 9323` 增加，因此 OF1 只小幅提升。
- raw margin 绝对尺度仍未解决：所有组 `raw_prob_pos_mean` 约 `0.505-0.508`，`raw_prob_neg_mean` 约 `0.494-0.496`，`raw_prob>0.55` 几乎为 0。当前 F1 仍依赖 CLIP logit scale 放大 margin。
- `random` 的 raw margin gap 最大，但 F1 更差，说明“gap 大”本身不够；还需要正负样本分布位置和阈值校准合理。
- `positive-only` 更保守，precision 较高、recall 较低；`negative-only` 更容易过预测，OF1 较低但 CF1 接近。这说明 presence branch 对抑制 false positive/固定阈值 precision 很重要，absence branch 单独训练不能充分压住误报。

下一步建议：

- 优先跑 `semantic text frozen 20ep`：固定语义 text anchor，只训练 visual prompts，验证 3ep 最佳 F1 是否能随训练拉长继续保持/提升。
- 同时可跑 `semantic positive-only 20ep` 作为 precision-oriented 对照。
- 若目标是严格 DDP raw probability，需要考虑 margin regularizer 或 raw binary-softmax loss 加强；这属于 DDP extension，不能混入论文复现主配置。

### 2026-06-24 semantic 长训消融与 seed1 稳定性复验

用户同步四组完整结果：

```text
text frozen 20ep:
  ./logs/voc_ddp_ablate_sem_text_frozen_20ep.log
  ./output/voc_ddp_ablate_sem_text_frozen_20ep

positive-only 20ep:
  ./logs/voc_ddp_ablate_sem_posonly_20ep.log
  ./output/voc_ddp_ablate_sem_posonly_20ep

text frozen 3ep seed1:
  ./logs/voc_ddp_ablate_sem_text_frozen_3ep_seed1.log
  ./output/voc_ddp_ablate_sem_text_frozen_3ep_seed1

full semantic 3ep seed1:
  ./logs/voc_ddp_margin_semantic_3ep_seed1.log
  ./output/voc_ddp_margin_semantic_3ep_seed1
```

所有实验均为 VOC B0-C4、`pooled_cls + prompted + clip + CLIP normalize`、
PCD `tau_max=3.0,gamma=0.7`、batch size 16、seed0/seed1，并保存了 detail/score dump；
两个 20 epoch 实验保存了 checkpoint。

最终结果：

```text
full semantic 3ep seed0:  mAP/amAP/OF1/CF1 = 83.8680/90.1785/70.2598/70.6409
full semantic 3ep seed1:  mAP/amAP/OF1/CF1 = 83.6778/90.1794/70.5977/70.8508
text frozen 3ep seed0:    mAP/amAP/OF1/CF1 = 82.3217/89.1572/71.1562/72.7656
text frozen 3ep seed1:    mAP/amAP/OF1/CF1 = 81.5404/88.5455/69.2907/71.3656
text frozen 20ep seed0:   mAP/amAP/OF1/CF1 = 80.8415/87.9108/68.2167/71.2956
positive-only 3ep seed0:  mAP/amAP/OF1/CF1 = 81.2522/88.3632/70.9906/71.5840
positive-only 20ep seed0: mAP/amAP/OF1/CF1 = 83.0647/89.3826/70.0879/72.6094
full semantic 20ep seed0: mAP/amAP/OF1/CF1 = 84.2737/90.3961/70.5397/72.5721
```

结论：

- full semantic 3ep 两个 seed 几乎重合，mAP 波动仅 `0.19`，是当前最稳定的结构。
- text-frozen 3ep 的 seed0 高 F1 未稳定复现；seed1 回落 `0.78 mAP/1.87 OF1/1.40 CF1`。
- text-frozen 拉长到 20 epoch 后全面退化。pred/support 从 3ep seed0 的 `9044/7632`
  增到 `10475/7632`，precision `0.656 -> 0.590`、recall `0.777 -> 0.809`，
  说明 visual prompts 长训后过预测，而不是 margin 完全没有学习。
- text-frozen 20ep 的 global oracle threshold 为 `0.9`，oracle micro-F1 `0.698`，
  仍低于 3ep seed0 主阈值 F1 `0.712`；问题不能只靠换阈值补回。
- positive-only 20ep 相对 3ep 提升 `+1.81 mAP/+1.03 CF1`，但 OF1 下降 `0.90`；
  相对 full semantic 20ep 仍低 `1.21 mAP/0.45 OF1`，没有证明应删除 negative branch。
- raw DDP probability 仍集中在 0.5：四组 final positive mean 约 `0.506`，
  negative mean 约 `0.495`，`>0.6` 比例仍为 0。
- 主要困难类别继续是 `chair/bicycle/bottle/car/diningtable`。text-frozen 20ep 的
  bicycle 出现 `P=0.202,R=0.940`，是典型严重误报；chair 的 margin gap 仅 `0.043`，
  表达分离仍弱。
- text-frozen learnable 参数 `491,520` 与论文约 `0.49M` 很接近，是重要结构线索；
  但其效果和长训趋势表明还需核对论文参数统计口径、训练调度及 prompt 更新方式，
  不能把参数量相等直接当作结构已严格复现。

下一步：

- 主线恢复为 full semantic text+visual prompts；text-frozen/positive-only 作为消融保留。
- 优先做 full semantic 的学习率/训练时长对照，避免当前 20 epoch 出现 precision 下降。
- 基于已有 20ep checkpoints 做 threshold/PCD eval-only，不重训地确认 calibration 上限。
- 在继续增加非论文 loss 前，先核对论文官方实现的 optimizer、LR、scheduler、warmup、
  visual prompt 初始化和 `0.49M` 参数统计口径。

### 2026-06-24 Git 提交与远端推送

- 服务器分支：`feature/clip-taskCLS-posneg-text-head`。
- 功能提交：`19417f1 Add DDP semantic prompt diagnostics`。
- 提交包含 DDP semantic text initialization、prompt branch ablation 开关、
  CLIP normalization、eval-only/checkpoint、score/margin diagnostics 及上下文文档。
- `python -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py
  multi_lane/continual_datasets/*.py` 已在服务器 `multilane` 环境通过。
- 已首次推送并设置上游：
  `origin/feature/clip-taskCLS-posneg-text-head`。
- 推送后本地与远端完整哈希一致：
  `19417f1500dd3e1046e79afe18711628f835eb84`。
- 无关的 `train_c100.sh` 文件末尾换行改动未纳入功能提交，仍保留在服务器工作区。

### 2026-06-24 margin 与 mAP 阶段结论

- full semantic 20ep final margin：
  `pos_mean=0.0731`、`neg_mean=-0.0521`、gap `0.1252`。
- `93.4%` 正样本的 margin 大于 0，说明多数正样本方向正确；
  `15.0%` 负样本的 margin 仍大于 0，说明 false-positive 排序仍明显存在。
- task5 PCD `tau=3` 后 strict DDP probability 仍为
  positive `0.5061`、negative `0.4957`，绝对尺度没有充分拉开。
- random 3ep gap `0.1688` 大于 semantic 20ep 的 `0.1252`，但 mAP 只有 `82.34`
  而非 `84.27`。因此 mAP 需要改善每类正负样本排序，不能只优化 overall mean gap。
- full semantic 20ep 最低 AP 类别为：
  `chair 38.51`、`bottle 51.73`、`tvmonitor 68.62`、`pottedplant 73.80`、
  `diningtable 75.33`、`sofa 76.15`、`car 78.05`。
- 代码核对发现配置中的 warmup/min-lr 参数未被实际 scheduler 使用；
  当前每 task 使用裸 `CosineAnnealingLR(T_max=epochs)`，并无条件重建 optimizer。
  这是下一阶段论文协议核对和 mAP 优化的高优先级风险点。

### 2026-06-24 官方 supplementary 对齐改造

从 OpenReview supplementary material 取得官方代码包 `CODE_DDP`，核对后确认：

```text
visual token similarity scale: 20
token attention: softmax(positive token similarities)
same token attention weights applied to positive and negative branches
weighted pair-logit scale: 5
effective final margin scale: 20 * 5 = 100
loss: binary pair softmax BCE, reduction=sum, loss weight=0.03
optimizer: Adam, lr=5.9e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0
scheduler: MultiStepLR, milestones=[0,20], gamma=0.1
optimizer lifetime: one optimizer across all tasks
epochs: 20
batch size: 8
train transform: Resize(224), CutoutPIL(0.2), RandAugment, CLIP normalize
val transform: Resize(224), CLIP normalize
visual prompt layers: [7,8,9,10,11]
text/visual prompt length: 16
```

因此此前计划中的“raw DDP train 是论文主路径”不成立。论文公式省略了实现级 scale，
但 supplementary code 明确使用固定 100 的强尺度。raw margin train 只保留为诊断
ablation，不再标记为严格论文配置。

本轮修改：

- `multi_lane/configs/custom_types.py`
  - 新增 `paper_attention` aggregation。
  - 新增 train/eval logit scale 分离和固定 paper scale。
  - 新增 `paper_sum` loss、持续 optimizer、paper scheduler 和 paper transform 参数。
- `multi_lane/clip_vit.py`
  - 实现官方 positive-token attention：positive similarities 经 `softmax(20*s)`，
    同一权重用于正负分支 token 汇总。
  - train/eval 可分别选择 `clip/none/paper` scale。
- `multi_lane/engine.py`
  - 实现官方 sum binary-softmax 等价 loss：margin BCE sum × `0.03`。
  - 支持 optimizer 跨 task 持续存在和全局 MultiStepLR。
- `multi_lane/utils.py`
  - 清理非当前类别 Adam state，确保旧类 prompt 在持续 optimizer 下不受历史动量漂移。
- `multi_lane/datasets.py`
  - 新增官方 Resize + CutoutPIL + RandAugment + CLIP normalize 路径。
- `main.py`
  - 支持 DDP unscaled optimizer LR override，并打印实际 LR。
- `requirements.txt`
  - 新增 `randaugment==1.0.2`。

服务器验证：

```text
py_compile: passed
paper attention formula equality: passed
pair-softmax BCE == margin BCE(sum): passed
full CPU forward/backward: passed
current task prompts have gradients, future prompts zero: passed
continual Adam frozen-state test: passed
paper scheduler: 5.9e-4 at init, 5.9e-5 after global epoch 20
paper transform with isolated randaugment 1.0.2: passed
```

注意：

- supplementary 的公开 `DDP.py` 硬编码示例是 VOC B4-C2，论文主表还包含 B0-C4；
  方法级参数可以对齐，但 B0-C4 的 PCD 超参仍需用新 attention 路径重新验证。
- 官方代码没有显式清理旧类 Adam 动量；本实现的 state 清理是为了严格满足论文
  “past prompts are frozen as knowledge anchors”的方法描述。

### 2026-06-24 official random 首次启动失败修复

- `voc_ddp_paperattn_semantic_3ep` 已在 GPU0 正常运行。
- `voc_ddp_official_random_3ep` 首次启动后 tmux 立即消失。
- 日志确认不是 OOM；GPU1 为空闲。失败发生在 DataLoader worker：
  `randaugment==1.0.2` 使用了 NumPy 1.24 后删除的 `np.int`。
- 已在 `_paper_randaugment()` 中对该官方旧依赖增加 `np.int = int` 兼容映射，
  不改变 RandAugment 策略。
- 服务器 NumPy 1.26.4 下已验证：
  `Resize -> CutoutPIL -> RandAugment -> ToTensor -> CLIP Normalize`
  可正常输出 `[3,224,224]` tensor。
- 当前 GPU1 空闲，可重启 official random；GPU2-7 均被其他任务占用，不启动 semantic
  official 对照，也不触碰其他用户进程。

### 2026-06-24 official-path 三组 3ep 结果

三组实验均完整结束并同步日志、detail JSON 和 score dumps：

```text
A semantic + paper attention:
  mAP/amAP/OF1/CF1 = 84.4317/90.5850/74.5444/70.9225
  P/R/F1 = 0.849/0.665/0.745
  margin gap = 0.1039

B official random:
  mAP/amAP/OF1/CF1 = 86.0550/91.9571/59.0481/51.9978
  P/R/F1 = 0.954/0.428/0.590
  margin gap = 0.1482
  oracle global threshold 0.7, F1 = 0.745

C official semantic:
  mAP/amAP/OF1/CF1 = 88.0259/93.3177/37.9391/28.7707
  P/R/F1 = 0.985/0.235/0.379
  margin gap = 0.1250
  oracle global threshold 0.6, F1 = 0.796
```

对照结论：

- A 对旧 semantic 3ep：`+0.56 mAP/+0.41 amAP/+4.28 OF1/+0.28 CF1`。
  token attention 显著提高 precision，并确认官方聚合不是无效细节。
- B 对 A：`+1.62 mAP/+1.37 amAP`，说明官方 loss、augmentation、optimizer/scheduler
  组合继续改善排序和 margin。
- C 对 B：`+1.97 mAP/+1.36 amAP`。semantic 初始化在官方路径上仍然有效，
  尽管 overall mean margin gap 更小，再次证明 AP 不能只由 mean gap 判断。
- C 对旧 semantic 3ep：`+4.16 mAP/+3.14 amAP`；对旧 semantic 20ep：
  `+3.75 mAP/+2.92 amAP`。仅 3 epoch 已明显超过此前所有 mAP。
- C 距 DDP 论文 VOC B0-C4：last mAP `-2.17`，average mAP `-1.48`。

official semantic 对旧 semantic 3ep 的关键 per-class AP 变化：

```text
chair       34.95 -> 61.61 (+26.66)
car         80.11 -> 92.69 (+12.57)
tvmonitor   75.01 -> 86.01 (+11.00)
pottedplant 68.26 -> 78.13 (+9.87)
diningtable 77.75 -> 84.77 (+7.02)
sofa        71.77 -> 77.44 (+5.67)
person      89.94 -> 95.74 (+5.79)
```

固定阈值解释：

- B/C 使用 official VOC `tau_max=7,gamma=0.2`。在 B0-C4 final task，tau=7 将
  confidence 大幅压向中间，threshold 0.8 导致 recall collapse。
- C 在 threshold 0.6 时已有 `P=0.763,R=0.832,F1=0.796`，说明排序和可校准上限明显
  高于当前主表 F1。
- 对固定 0.8 阈值，`tau≈2.0` 与 tau7/threshold0.6 的 margin 边界近似等价；
  因此下一步优先 eval-only `tau2/gamma0.7` 和 `tau3/gamma0.7`。

### 2026-06-24 official semantic tau2 优化运行脚本

按用户要求，已基于当前最高 mAP 的 `official semantic` 配置新增两个可直接运行的
VOC B0-C4 脚本：

```text
run_voc_ddp_official_semantic_tau2_3ep.sh
run_voc_ddp_official_semantic_tau2_20ep.sh
```

配置策略：

- 保持 official semantic recipe 不变：`paper_attention`、`paper` logit scale、
  `paper_sum` loss、continual optimizer、paper MultiStepLR、paper transform、
  semantic text initialization。
- 只把 PCD 从官方 VOC 风格的 `tau_max=7.0,gamma=0.2` 改为
  `tau_max=2.0,gamma=0.7`。
- 目的不是改排序，而是先修复 fixed-threshold 0.8 下的 confidence 过度压缩问题。

脚本行为：

- 日志写入：
  - `./logs/voc_ddp_official_semantic_tau2_g07_3ep.log`
  - `./logs/voc_ddp_official_semantic_tau2_g07_20ep.log`
- 输出写入：
  - `./output/voc_ddp_official_semantic_tau2_g07_3ep`
  - `./output/voc_ddp_official_semantic_tau2_g07_20ep`
- 均开启 `--store_model` 和 `--ddp_score_dump true`。
- 可通过 `GPU=<id> ./script.sh` 指定显卡。

已完成本地检查：

```text
bash -n run_voc_ddp_official_semantic_tau2_3ep.sh run_voc_ddp_official_semantic_tau2_20ep.sh
python3 -m py_compile main.py multi_lane/*.py multi_lane/configs/*.py multi_lane/continual_datasets/*.py
```

结果：通过。

2026-06-24 追加修复：

- 远程服务器用 `bash run_voc_ddp_official_semantic_tau2_3ep.sh` 启动时，
  `/root/.bashrc` 在 `set -u` 下引用未定义 `PS1`，导致脚本提前退出。
- 已在两个 tau2 脚本中对 `source ~/.bashrc` 前后临时切换 `set +u` / `set -u`，
  仅规避 shell 初始化兼容问题，不改变任何实验配置。
- 已重新运行 `bash -n`，结果通过。

注意：当前本地目录不是 Git 仓库，无法在本地创建实验分支或提交；Git 状态仍以服务器
`/mnt/haoyuan/workspace/multi-lane-main` 为准。

### 2026-06-24 official semantic tau2 3ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_3ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_3ep
ckpt:   ./output/voc_ddp_official_semantic_tau2_g07_3ep/checkpoints.pth
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.0259/93.3177/79.4418/78.6858
pred/support = 8420/7632
micro precision/recall/F1 = 0.7572/0.8354/0.7944
oracle global threshold = 0.80, F1 = 0.794
per-class oracle F1 = 0.824, threshold mean = 0.790
```

对比：

```text
official semantic tau7 3ep: 88.0259/93.3177/37.9391/28.7707
official semantic tau2 3ep: 88.0259/93.3177/79.4418/78.6858
delta:                         +0.00/+0.00/+41.50/+49.92

official random tau7 3ep:      86.0550/91.9571/59.0481/51.9978
delta tau2 semantic:            +1.97/+1.36/+20.39/+26.69

paper-attention semantic 3ep:   84.4317/90.5850/74.5444/70.9225
delta tau2 semantic:            +3.59/+2.73/+4.90/+7.76
```

结论：

- `tau2/gamma0.7` 没改变排序，mAP 与 tau7 official semantic 完全一致；
  它只改变 fixed-threshold calibration。
- tau2 主阈值 0.8 的结果几乎复现了 tau7 score dump 中 threshold 0.6 的 oracle 行，
  验证此前“tau≈2 对齐主阈值 0.8”的判断成立。
- 当前已接近 DDP 论文 VOC B0-C4 OF1 `80.8`，但 CF1 仍低约 `5` 点，剩余问题主要在
  类别级 calibration，特别是 `bottle/chair/bicycle/diningtable/sheep/sofa`。

### 2026-06-24 official semantic tau2 20ep 结果分析

已同步并分析：

```text
log:    ./logs/voc_ddp_official_semantic_tau2_g07_20ep.log
output: ./output/voc_ddp_official_semantic_tau2_g07_20ep
```

最终结果：

```text
task5 mAP/amAP/OF1/CF1 = 88.5226/93.4123/72.7760/76.3618
pred/support = 10286/7632
micro precision/recall/F1 = 0.6339/0.8543/0.7278
margin: pos=0.0813, neg=-0.0598, gap=0.1411
global oracle threshold = 0.90, F1 = 0.7584
per-class oracle F1 = 0.805, threshold mean = 0.795
```

关键对比：

```text
official semantic tau2 3ep:  88.0259/93.3177/79.4418/78.6858
official semantic tau2 20ep: 88.5226/93.4123/72.7760/76.3618
delta:                        +0.4967/+0.0946/-6.6658/-2.3240

paper B0-C4:                  90.2/94.8/80.8/76.9
20ep gap:                     -1.68/-1.39/-8.02/-0.54
```

结论：

- 20ep 让 ranking 和 margin 继续改善，但收益不大：mAP 只增加 `0.50`，
  margin gap 从 `0.1250` 增加到 `0.1411`。
- tau2 在 3ep 时校准正好，在 20ep 时偏激进。预测正类从 `8420` 增到 `10286`，
  precision 从 `0.757` 降到 `0.634`，而 recall 仅从 `0.835` 升到 `0.854`。
- 20ep 最佳扫描阈值已经移到 `0.9`，因此下一步先使用现有 checkpoint 做
  eval-only tau `2.5/3/4` 对照，不能把 OF1 下降误判成结构退化。
- mAP 剩余问题集中在困难类和跨任务退化。最终弱类：
  `bottle 61.96`、`chair 61.45`、`sofa 77.13`、`pottedplant 79.00`、
  `diningtable 82.03`。
- 从类别刚学完到 task5，主要 AP 下降：
  `bottle -29.97`、`chair -14.96`、`bicycle -10.53`、`cow -9.03`、
  `diningtable -5.92`。这不是旧 prompt 参数漂移的直接证据，因为 prompts 已冻结；
  更可能是后续 seen-class 评估中类别间分数竞争、任务数据分布和负样本覆盖不足。

### 2026-06-24 最佳 20ep tau2 版本提交与推送

- 分支：`feature/clip-taskCLS-posneg-text-head`
- 提交：`5ac25a8 Align CLIP DDP with official training recipe`
- 已推送：`origin/feature/clip-taskCLS-posneg-text-head`
- 提交包含 paper-aligned DDP 实现、official semantic tau2 3ep/20ep 运行脚本及实验记录。
- `logs/`、`output/`、`checkpoints.pth` 均未进入 Git。
- 服务器工作区仍有无关的 `train_c100.sh` 末尾换行变化，未纳入提交。

### 2026-06-24 创建 EMOTIC DDP tau2 分支

- 从 `feature/clip-taskCLS-posneg-text-head` 的 `01c10fc` 创建
  `feature/emotic-ddp-semantic-tau2`。
- 数据集确认存在于服务器 `./datasets/EMOTIC`，加载参数使用 `--data_path ./datasets`。
- 采用 Split-EMOTIC 默认 alphabetical B5-C3 协议：
  `base_classes=5,num_tasks=8`，共 26 个类别。
- `multi_lane/configs/emotic.py` 的 DDP semantic prompt 默认值改为情感专用前缀：
  positive `a photo of a person clearly feeling`，negative
  `a photo of a person not feeling`。
- 新增 `run_emotic_ddp_official_semantic_tau2_20ep.sh`，完整迁移 VOC 当前最佳
  paper-aligned DDP 20ep tau2 配置，输出独立 log/output/checkpoint/score dump，
  detail HTML/JSON 继续复用同一评估引擎。
- 服务器验证结果：
  - Python 静态编译和脚本 `bash -n` 通过；
  - parser 默认提示与脚本提示一致；
  - train/eval 人物样本数为 `16001/7765`，26 类任务切分为
    `[5,3,3,3,3,3,3,3]`；
  - DDP text/visual prompt shapes 为 `(26,2,16,512)` / `(26,2,16,768)`；
  - positive/negative semantic seeds 不相同；
  - 报告 smoke 成功生成 `.html` 和 `.json` 双文件。

### 2026-06-24 EMOTIC B5-C3 real-data smoke

- 用户确认在 GPU0 运行 1ep smoke。
- 完整配置保持正式 20ep recipe 不变，仅使用 `epochs=1`；B5-C3、batch8、full image、
  paper attention/loss/optimizer/scheduler/transform、semantic emotion prompts、
  PCD tau2/gamma0.7，并保留 checkpoint、score dump、HTML/JSON。
- 独立 tmux：`ddp_emotic_tau2_smoke_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_smoke`
- 用户确认 GPU0 剩余显存足够后，取消等待并立即并行启动；未触碰
  `/mnt/haoyuan/workspace/CODE_DDP` 的现有实验。
- 启动后 GPU0 总占用约 `9.4/24.6GB`，本 smoke 进程约 `5.2GB`；
  task0 首个训练 batch 正常完成，无 OOM。

### 2026-06-24 EMOTIC B5-C3 30ep 正式配置

- 参考对照：
  `emotic_b5c3_alphabetical_valtest_clip_taskCLS_text_bias_lr001563_v1`
- 对齐项：Split-EMOTIC、26 类字母序、B5-C3、full image、val+test、seed0、
  224 输入、8 tasks、每 task 30 epochs、Adam、weight decay 0、相同 detail
  HTML/JSON 指标口径。
- batch 对齐采用 physical batch8、gradient accumulation32、effective batch256；
  避免 class-wise positive/negative visual branches 在 physical batch256 下 OOM。
- DDP 方法专属项保留当前最佳配置：drop_last、paper attention、
  paper summed BCE `*0.03`、continual Adam 配置 LR `0.0059`、paper multistep、
  semantic emotion prompts、PCD tau2/gamma0.7。
- 旧 taskCLS 对照的 batch256、cosine scheduler 和 LR `0.001563` 不直接迁移，
  其中 batch256 通过梯度累积等效实现；cosine scheduler 和 LR `0.001563` 不迁移，
  因为当前最佳结果来自 official DDP 优化协议。
- 修复 `train_one_epoch` 的 accumulation step 边界，并区分 mean loss 与
  official `paper_sum` 的梯度缩放；accumulation1 行为保持不变。
- 正式脚本：`run_emotic_ddp_official_semantic_tau2_30ep.sh`
- tmux：`ddp_emotic_tau2_30ep_gpu0`
- log：`./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep.log`
- output：`./output/emotic_ddp_official_semantic_tau2_g07_b5c3_30ep`
- 首次 tmux 启动在 Python 前退出，launcher log 确认非交互 shell 中
  `/root/.bashrc` 提前 return，`conda` 未定义。EMOTIC 20ep/30ep 脚本已改为直接
  source `/opt/conda/etc/profile.d/conda.sh`，不影响训练参数。
- 第二次启动发现 tmux 继承的 `csc` 环境 deactivate hook 与 `set -u` 不兼容；
  已仅在 `conda activate multilane` 前后临时切换 `set +u/set -u`。
- 修复 shell 初始化后正式实验成功进入 task0 epoch1：
  - accumulation32、pin_mem false、effective optimizer LR `0.0059` 配置均已解析；
  - scheduler 初始化后的实际 LR 为 `0.000590`；
  - 正式实验自身显存峰值约 `4.3GB`；
  - GPU0 三个并行实验总占用约 `15.6/24.6GB`；
  - 未发现 OOM、Traceback、RuntimeError 或 loss NaN。

### 2026-06-25 EMOTIC DDP smoke/30ep 结果分析

两组均完整跑完 8 tasks，HTML/JSON、8 个 score dump 和服务器 checkpoint 均生成。

```text
1ep smoke: mAP/amAP/OF1/CF1 = 29.8272/36.6230/21.8038/9.5456
30ep run:  mAP/amAP/OF1/CF1 = 31.3195/37.8293/25.9253/10.1560
delta:                              +1.4923/+1.2063/+4.1215/+0.6104
```

正式 30ep 与历史对照：

```text
DDP semantic tau2: 31.3195/37.8293/25.9253/10.1560
taskCLS text bias: 31.1848/37.2170/57.2128/22.7483
CLIP patch concat: 32.8635/39.8831/47.0667/20.2515
```

关键判断：

- 30ep effective batch256 与 1ep batch8 的 optimizer update 数接近：
  每 task 约 `30*ceil(num_batches/32)` 对比 smoke 的 `num_batches`，因此两者结果接近
  是合理现象；30ep 的梯度统计更接近历史 batch256 对照。
- DDP 对 taskCLS-text-bias 的 final mAP 仅提高 `0.13`，amAP 提高 `0.61`；
  对普通 CLIP patch baseline 则低 `1.54/2.05`。排序收益不足。
- final threshold0.8 precision/recall/F1 为 `0.754/0.157/0.259`；
  threshold0.5 为 `0.432/0.646/0.518`。tau2 在 EMOTIC 上明显过保守。
- 即使使用 global oracle threshold，OF1 仍比 taskCLS-text-bias 低约 `5.45` 点，
  因此不能把差距全部归因于阈值。
- final raw margin：positive `0.0075`、negative `-0.0210`、gap `0.0285`；
  positive margin >0 rate `64.6%`，negative margin >0 rate `20.3%`。
  26 类中 17 类 positive margin mean <=0，说明 macro/class-wise 分离不足。
- 最弱 AP 包括：Embarrassment `2.12`、Sensitivity `4.47`、Fear `8.49`、
  Surprise `8.57`、Aversion `8.95`、Yearning `11.21`。
- 相对 taskCLS-text-bias，DDP 大幅改善 Affection `+19.84 AP`，并改善
  Confidence `+4.51`、Pleasure `+2.98`；主要下降为 Suffering `-9.08`、
  Disquietment `-6.15`、Doubt/Confusion `-4.89`、Sadness `-3.57`。
- 从类别刚学完到最终评估，早期类 AP 下降主要来自评估样本/负样本分布扩展，
  不是旧 prompt 参数更新：Anticipation `-26.20`、Affection `-13.38`、
  Annoyance `-8.98`。模型参数冻结不保证在新增负样本上的 AP 不变。
### 2026-07-01 EMOTIC test-only 评估入口

- 发现历史 `emotic_b5c3_alphabetical_valtest_clip_vit_b16_patch_v1` 报告使用
  `val+test`。该实验没有保存 checkpoint，因此无法只重跑测试，必须重训才能得到
  对应模型的纯 test 结果。
- 在 EMOTIC CLI 增加 `--emotic_eval_splits`，默认 `val test`，可显式设为 `test`。
- 修复数据构建入口未传递 `emotic_input_mode` 的问题，并把选择的 eval splits 传入
  EMOTIC dataset。
- 新增历史 CLIP patch concat baseline 的纯 test 30ep 脚本，保持 B5-C3、26 类字母序、
  batch256、seed0、full image 等训练配置不变，新增 checkpoint 保存。
- 预期产物为独立 log、checkpoint、class order JSON、逐类逐任务 HTML/JSON。
- 用户确认历史代码位于 `feature/clip-vit-b16`。已从 `0b59138` 建立独立 worktree
  `/mnt/haoyuan/workspace/multi-lane-main-test-only` 和分支
  `fix/emotic-test-only-clip-vit-b16`，没有切换或污染当前 DDP 工作区。
- 实测 eval 样本数为 val `2397`、test `5368`、val+test `7765`；test-only 仍为
  26 类字母序及 B5-C3 八任务。
- 纯 test 30ep 正式实验已在 GPU1、tmux `emotic_clip_b16_testonly_30ep` 启动。
  task0 前两轮训练指标与历史 val+test 日志逐项一致，验证训练协议未改变；GPU1
  峰值约 8.2GB，无 OOM。
- B5-C3 纯 test 已完成，final `29.9174/36.6938/45.5877/19.5187`，耗时 `36:01`。
- 旧 upper bound 配置核对为 joint training：`dataset=EMOTIC`、1 task、26 base classes、
  30 epochs、batch256、CLIP patch concat；旧实验 `store_model=False`，因此纯 test 必须
  重训。
- 新增 upper-bound test-only 脚本，在 GPU0/tmux `emotic_upper_testonly_gpu0` 启动。
  Namespace 已确认 `emotic_eval_splits=['test']`，首轮训练数值与旧 val+test upper
  bound 一致，预计约 14–16 分钟完成。

### 2026-08-11 本地/服务器独立 Git 工作树迁移

- 只读审计确认服务器主工作树原分支为 `fix/emotic-test-only-eval@b984688`；该分支没有
  upstream，但 HEAD 已存在于 `origin/feature/emotic-ddp-semantic-tau2`，没有未推送的
  普通提交。服务器原有三个 stash 均为仅存于服务器的 WIP。
- 审计确认额外 worktree `/mnt/haoyuan/workspace/multi-lane-main-test-only` 位于
  `fix/emotic-test-only-clip-vit-b16@0b59138`，包含两处代码修改、两个未跟踪运行脚本和
  指向主数据集目录的符号链接；迁移期间未修改该 worktree。
- 迁移前完整备份位于
  `/mnt/haoyuan/workspace/git-migration-backups/multi-lane-main-20260811-1145/`：包含经
  `git bundle verify` 验证的全引用 bundle、两个 worktree 的 binary patch、状态清单、
  stash 清单、未跟踪 PNG/脚本副本及 SHA-256 校验清单。大型数据、日志、output 和权重
  未复制或纳入 Git。
- 创建并推送安全分支 `backup/pre-local-git-migration-20260811-1145@b984688`，随后将
  服务器主工作树迁移前改动保存到命名 stash `7d0366b`，并正常切换到
  `feature/clip-vit-b16@0b59138`。
- 本地原目录已执行 `git init`，添加相同 origin 并 fetch；使用 mixed 基线识别出的修改
  与服务器备份 patch 及 PNG SHA-256 完全一致。本地迁移前改动保存到命名 stash
  `a9a4ef5`，`.idea/` 和 `.DS_Store` 写入本地 `.git/info/exclude`。
- 从 `feature/clip-vit-b16@0b59138` 创建并推送
  `fix/independent-git-worktrees`。本地和服务器主工作树均跟踪该远端分支，Git 元数据
  完全独立；PyCharm Automatic Upload 已关闭。
- 本次迁移未运行训练或模型测试；验证范围为 Git 引用、远端提交、stash/patch 备份、
  两端 HEAD/upstream、工作树状态和文件校验和。
- 用户已确认将 `.gitignore` 和四份迁移上下文文档提交并推送到
  `fix/independent-git-worktrees`，再由服务器使用 `git pull --ff-only` 同步。

## 2026-09-01：同步并分析layer1 ASL局部正式test

- 确认服务器已无本批runner，8个目录均包含config、task metrics、training history和
  complete seed summary；服务器实验worktree保持clean`94f3327`。
- 结果和日志已打包同步到本地，不含checkpoint；archive两端SHA-256均为
  `c8258d3215747c4dfd7abc30c35c004e6b9196ea230d394e1a8c2ae71f665691`。
- 锚点ASL 9.8/0/0.05 final/average mAP为`32.5365/39.1445`，且8个task mAP逐项第一。
  其余7组final mAP为`31.6761--32.1476`，没有新冠军。
- gamma-pos1在task3共跳过11次AMP optimizer update；其余7组skipped0。由于该组final
  mAP仍低锚点`0.3889`，不补FP32，也不扩大gamma-pos。
- 阶段二ASL局部搜索正式收口。阶段三/四尚未启动；它们需要新增按task bottleneck和
  Adapter专属weight decay入口，并在用户确认提交推送后才能按Git流程部署服务器。

## 2026-09-01：实现阶段三/四并行实验入口

- 用户确认运行阶段三和阶段四，创建`exp/emotic-image-token-stage34`实验分支。
- `adapter.py/model.py/runner.py/smoke.py`新增每task bottleneck、Adapter-only weight decay、
  逐task参数统计及GPU smoke参数；优化器仍保持selectors/prompts使用全局weight decay、head为0，
  Adapter单独使用新值。
- 修复潜在对照混杂：非统一task0容量构造后，显式推进到uniform b32参考RNG位置，使后续task
  Adapter初始化不受task0形状影响；增加对应相等性测试。
- 新增单组正式入口和8卡launcher。11组同时启动，GPU0--2各两组、GPU3--7各一组；launcher
  启动前要求8卡连续两次至少12GB空闲，全部无checkpoint，结果写外部benchmark目录，日志写`logs/`。
- shell语法、Python静态编译和`git diff --check`通过。本地系统Python缺少NumPy，完整单元测试
  留待服务器`ddp`环境在部署后执行。

## 2026-09-01：部署并启动阶段三/四共11组正式test

- 提交`8e56b8a Add taskwise Adapter stage 3 and 4 search`已推送到
  `origin/exp/emotic-image-token-stage34`。服务器先审计全部worktree，再创建独立clean worktree
  `/mnt/haoyuan/workspace/multi-lane-main-stage34`，未切换ASL、primary或test-only工作树。
- Automatic Upload把本次代码和脚本也写入服务器primary；已把3份文档和7个代码/脚本逐文件
  归档到`/mnt/haoyuan/workspace/git-sync-backup-stage34-deploy-docs-20260901-ooz0O8`，然后恢复primary
  clean。上传代码归档SHA-256为
  `0a843bfc3a5be08372c9006c6c8d72e3d155fef37ce95af6b7694a2033b38522`。
- 服务器`ddp`环境21项完整单元测试通过；uniform b32和task0-b24混合容量的真实CLIP
  AMP/TF32-on Adapter-ASL GPU smoke均通过，trainable参数分别为739130/726834。
- batch`image_token_asl_layer1_stage34_formal_seed0_20260901_203100`已在tmux
  `image_token_stage34_20260901_203100`启动。11组config全部生成且逐字段核验正确，均记录
  clean commit`8e56b8a`。GPU0--2双进程约4.2--4.6GB，GPU3--7单进程约2.1GB。
- 11组已到task0 epoch4--5，每epoch84 steps、skipped0，日志无OOM、non-finite、Traceback或
  RuntimeError。当前让实验继续运行，不持续盯跑；结束后只同步JSON与日志，不下载checkpoint。

## 2026-09-01：同步并分析阶段三/四结果

- 11/11组完成，launcher退出码均为0；每组240 epochs、13950 optimizer updates、skipped0，
  无OOM、non-finite、Traceback、RuntimeError或checkpoint目录。
- 小结果、manifest、状态与日志已同步本地；2.5MB解压目录中的archive两端SHA-256均为
  `bde0e9746985cb74751cc946332b8ada3e823401a7ea5bb1db732def30fcecfd`。
- 阶段三b24/b28在task0有正收益，但从task1开始整体转负，final mAP为`31.9938/32.0368`。
  因后续Adapter RNG已严格对齐锚点，结论指向selectors/prompts复制造成的持续路径差异。
- 阶段四冠军仍为LR0.0125/WD0的`32.5365`。LR0.015/WD1e-5达到`32.5023`，只低`0.0343`，
  同时获得本批最高average mAP`39.6697`和cF1`33.1460`；task0--6均高、task7仅低`0.0343`。
- 下一方向收缩到LR0.015/WD1e-5附近的窄网格；不继续task0小容量或全局weight decay扩展。

## 2026-09-01：准备统一LR×Adapter WD局部直接test

- 用户将下一阶段从validation-only改为直接held-out test，并接受其仅为exploratory结论。
- 创建`exp/emotic-image-token-global-lr-wd`，新增9组8卡launcher；GPU0承载锚点和一组候选，
  GPU1--7各一组。
- 候选实际主模型LR为`0.014/0.0145/0.015/0.0155`，对应source LR
  `0.056/0.058/0.060/0.062`；Adapter WD为`3e-6/1e-5`。所有task使用同一组合，保持未知未来
  增量数据设定，不使用按task手调参数。
- 固定seed0、8 tasks×30 epochs、batch64、Image-token layer1/b32、Adapter LR4e-4、
  scale0.1/ReLU、主模型BCE + Adapter ASL9.8/0/0.05、CLIP normalization、crop0.05、
  AMP/TF32 on、无checkpoint。
- launcher与上下文已提交推送为`4b45dd9`；服务器创建clean独立worktree并通过21项单元测试。
- Automatic Upload副本备份位于
  `/mnt/haoyuan/workspace/git-sync-backup-global-lr-wd-deploy-20260901-RsLHVP`，primary恢复clean。
- 8卡当前各被其他任务占用约19GB，未发现本项目runner。已在tmux
  `image_token_global_lr_wd_20260901_224909`挂起自动等待；batch ID为
  `image_token_asl_layer1_global_lr_wd_formal_seed0_20260901_224909`。每30秒检查一次，要求
  全部GPU连续两次至少12GB空闲后再启动，不停止或抢占现有进程。

## 2026-09-02：同步全局LR×Adapter WD首批8组并补跑资源失败组

- 9组launcher已经运行；其中8组完整产出`seed_summary.json`，每组240 epochs、13950 updates、
  skipped0。LR0.0155/WD1e-5原运行在task6开始时OOM，日志显示GPU0仅剩16.62MiB，属于外部
  GPU进程临时占用引发的资源碰撞，并非loss非有限或模型容量超限。
- GPU释放后，在GPU1以完全相同配置从task0重新补跑，run ID为
  `image_token_asl_layer1_global_lr_wd_formal_seed0_20260901_224909_lr0155_wd1e5_retry1`，tmux为
  `image_token_global_lr_wd_retry1_20260902`。启动后约占2.1GB，尚余约22GB，首4轮正常。
- 已同步8组完整JSON、原launcher日志和失败诊断日志到本地，不含checkpoint。压缩包两端
  SHA-256均为`4be9b24748df8dc32f3a4759b31157b1e44b2384b1e3c690f43b6372234356b4`。
- 暂时排名仍由LR0.0125/WD0锚点`32.5365`领先；LR0.015/WD1e-5为`32.5023`，差`0.0343`，
  但average mAP从`39.1445`升到`39.6697`，cF1从`32.5756`升到`33.1460`，oF1从
  `49.3827`升到`49.5592`。其余6个候选final mAP均不超过`32.3674`。
- 在三个已成对完成的LR点上，WD1e-5相对WD3e-6的final mAP分别提高`0.2662/0.0725/0.1348`，
  average mAP分别提高`0.2588/0.3571/0.5826`，说明较强的候选WD方向一致优于3e-6；但主模型
  LR收益在0.015附近才出现，非简单单调关系。

## 2026-09-02：完成全局LR×Adapter WD最终9组分析

- GPU1 clean retry完整结束，状态complete，240 epochs、13950 optimizer updates、skipped0；tmux与
  runner均已退出。run记录clean`4b45dd9`，配置与原失败组一致。
- LR0.0155/WD1e-5补跑final/average mAP为`31.8789/39.2960`，相对冠军final下降`0.6576`；
  task5/6/7分别下降`0.0223/0.5944/0.6576`，说明LR0.0155已越过有效区域上界。
- 最终冠军仍是LR0.0125/WD0的`32.5365`。LR0.015/WD1e-5以`32.5023`第二，虽提高average
  mAP和F1，但没有满足用户以final mAP为主的替换条件。
- WD1e-5相对3e-6在LR0.014/0.0145/0.015分别提高final mAP`0.2662/0.0725/0.1348`，到
  LR0.0155反而下降`0.1775`，确认LR与WD存在窄区间交互，停止继续微调二者。
- 最终小结果包已同步本地，archive两端SHA-256均为
  `724611f09c4df5501257b92238790cb4309074571828c692eec09862767a2fba`，不含checkpoint。
  最终分析写入`output/emotic_track_a_image_token_global_lr_wd_formal/20260901_224909/analysis.md`。

## 2026-09-02：实现固定预算、Adapter输出正则与可学习gate

- 从`4b45dd9`创建`exp/emotic-image-token-training-mechanisms`，保留已更新但尚未提交的三份上下文
  文档。固定gate/无正则/epoch预算保持旧训练数学路径，作为同批control复现`32.5365`。
- runner新增`--optimizer-updates-per-task`：每task按成功GradScaler/optimizer step精确停止，跳步不计
  入预算；DataLoader按需重新shuffle循环，cosine scheduler逐成功step推进。config记录预算模式、
  每task目标和scheduler语义，summary记录实际data cycles与总成功updates。
- Image-token Adapter新增两种可微输出度量：scaled residual/frozen token二范数比、adapted/frozen
  token cosine distance。正则只加入Adapter ASL分支，主模型selectors/prompts/head仍只接收BCE梯度。
  每task用最初30个成功updates进行无正则量级校准，之后固定权重，避免zero-init早期除以极小量造成
  梯度爆炸。
- 新增按task独立的sigmoid residual gate。gate作为对应task Adapter optimizer参数，激活新task时
  仅当前gate可训练；旧gate及旧Adapter冻结。固定模式不新增参数，learnable模式每task仅增加1参数。
- 新增固定预算精确停止、校准权重、辅助度量、gate初始化/冻结/参数统计单元测试；GPU smoke增加正则
  和gate入口。新增单组正式脚本与18组8卡launcher，外部结果目录不保存checkpoint。
- 本地`py_compile`、两个shell的`bash -n`、18组数组长度和`git diff --check`通过。本地完整测试因系统
  Python缺少Torch/NumPy无法执行，这是环境缺失而非测试失败；部署后必须在服务器`ddp`环境跑完整套件。
- 提交`2cafb04 Add Image-token Adapter training mechanism search`已推送。Automatic Upload写入primary的
  8个受控文件和2个未跟踪脚本已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-training-mechanisms-deploy-20260902-jsBiUR`并逐路径恢复；primary
  clean，test-only原有改动未触碰。
- 服务器从远端创建clean独立worktree`/mnt/haoyuan/workspace/multi-lane-main-training-mechanisms`，HEAD、
  upstream均为`exp/emotic-image-token-training-mechanisms@2cafb04`。
- 服务器`ddp`环境完整50项单元测试全部通过。真实ViT-B/16 AMP/TF32-on GPU smoke覆盖control、
  residual_ratio10%、feature_cosine10%和learnable gate0.1，均确认主模型/Adapter梯度隔离、冻结视觉塔、
  concat inference与有限梯度；固定模式trainable739130，gate模式739131。
- 正式18组batch ID预定为
  `image_token_asl_layer1_training_mechanisms_formal_seed0_20260902_150731`。启动前仍需确认8卡连续两次
  至少18GB空闲；launcher随后一次并行启动全部配置，不保存checkpoint。
- 验证记录follow-up提交`5505a36`已推送并在服务器实验worktree`git pull --ff-only`。Automatic
  Upload再次写入primary的三份文档已备份到
  `/mnt/haoyuan/workspace/git-sync-backup-training-mechanisms-validation-docs-20260902-FMyt8a`并恢复clean。
- 8卡启动前均余23.6GB以上，launcher连续两次通过18GB门槛后在tmux
  `image_token_training_mechanisms_20260902_150731`原子启动18组。run目录与config均为18/18；GPU0/1
  各3组、其余每卡2组，峰值初始占用约6.7GB，无资源冲突。
- 18份config均记录clean`5505a36`、seed0/test/no-checkpoint、main LR0.0125、Adapter LR4e-4/WD0、
  layer1/b32、BCE+Adapter-ASL、AMP/TF32。预算分布为epochs11组/updates7组，updates值精确为
  `900--2700`；正则6组与gate4组字段、每task参数49952/49953均正确。
- control task0 cycle1--5与历史冠军loss、LR和Adapter loss逐项一致。六个正则组在30个成功updates
  后均生成非零固定权重；task0首cycle的1%/3%/10% residual权重为
  `15.7632/47.2897/157.6324`，cosine为`31.5044/94.5132/315.0440`，验证校准比例正确。
- 当前全部日志无OOM、non-finite、FloatingPointError、RuntimeError或Traceback，首轮skipped0。
  让实验继续运行，结束后只同步JSON、manifest、launcher status和日志，不下载checkpoint。

## 2026-09-02：同步并分析训练机制三阶段18组结果

- tmux与18个runner均已退出，batch status complete，18个launcher exit code全部为0；18份config和
  complete seed summary齐全。所有运行skipped0、无OOM/non-finite/Traceback/RuntimeError，且没有
  checkpoint目录。
- 7个固定预算组总成功updates分别精确为`7200/9600/12000/14400/16800/19200/21600`；control、
  6个正则组和4个gate组均为240 cycles、13950 updates，协议完整性通过。
- 阶段一control继续以`32.5365/39.1445`领先。固定updates最佳1800为
  `30.8757/38.2894`，final下降`1.6608`；其他固定预算下降`1.8004--2.1743`。
- 阶段二最佳final为residual1%的`32.0751`，下降`0.4614`；cosine最佳为10%的`32.0490`，
  下降`0.4875`。所有正则都压低task5--7和task6，没有稳定性换取final收益。
- 阶段三gate init0.05最接近，final/average为`32.3397/39.0774`，低control`0.1969/0.0671`；
  cF1提高`0.0271`、forgetting改善`0.0868`，但不符合用户final mAP主目标。最终8个gate为
  `0.0696/0.0720/0.0570/0.0860/0.0688/0.0689/0.0550/0.0633`。
- 三个机制各自最佳均为负收益，因此不补组合实验。保留30 epochs、fixed scale0.1、无输出正则、
  LR0.0125/Adapter WD0冠军；关闭固定updates、输出正则与learnable gate路线。
- 结果JSON、manifest、launcher status和日志已同步本地。860KB压缩包两端SHA-256均为
  `fc051087a60a293ae06824396c7981c8cbeaeb9f04a24d0cd1226981a81c003a`，不含checkpoint。

## 2026-09-02：实现8组统一epoch正式test搜索

- 从`exp/emotic-image-token-training-mechanisms@5505a36`创建
  `exp/emotic-image-token-epoch-search`，保留上一批已更新但未提交的三份上下文文档。
- 新增`run_multilane_track_a_image_token_epoch_search_formal.sh`，只把`--epochs`暴露为严格正整数，
  其余训练参数精确固定为当前single1 b32冠军；每组使用独立输出根和日志，不保存checkpoint。
- 新增`launch_multilane_track_a_image_token_epoch_search_formal_8gpu.sh`，8卡分别运行
  `18/22/26/30/34/38/42/48` epochs，启动前要求8卡连续两次至少18GB空闲，生成manifest与逐组
  exit code，并在全部成功后自动调用严格汇总器。
- 新增`multi_lane.track_a.summarize_image_token_epoch_search`，核验完整网格、clean Git、共同
  commit/tree、固定协议、task/epoch/update/skipped/checkpoint完整性；输出final/average/late-task/
  task6/F1/forgetting排名及相对30-epoch锚点变化，并按预声明规则给出refinement。
- 新增3项单元测试，覆盖内部34-epoch赢家应建议`31/32/33/35/36/37`、30-epoch赢家应停止搜索、
  以及batch size协议漂移必须拒绝。下一步先完成本地静态检查，再提交推送并在服务器ddp环境运行
  完整单元测试；通过后才启动8组正式实验。
- 实现提交`7449b30 Add Image-token Adapter epoch search`已推送。Automatic Upload写入primary的
  7个文件已逐项备份至`/mnt/haoyuan/workspace/git-sync-backup-epoch-search-deploy-20260902-BmmkzK`
  并恢复primary clean；test-only原有修改未触碰。
- 服务器新建独立clean worktree`/mnt/haoyuan/workspace/multi-lane-main-epoch-search`，HEAD/upstream均为
  `exp/emotic-image-token-epoch-search@7449b30`。ddp环境53项完整单元测试通过；GPU0真实CLIP
  Image-token layer1/b32、Adapter-ASL、AMP/TF32 smoke通过，trainable739130、冻结视觉塔无梯度、
  初始零残差max logit差`7.63e-05`。8卡当前均余23.6GB以上，可进入正式启动。
- 验证记录已提交推送为`7091da6`，服务器实验worktree使用`git pull --ff-only`同步并保持clean。
  Automatic Upload写入primary的三份文档已备份至
  `/mnt/haoyuan/workspace/git-sync-backup-epoch-search-validation-docs-20260902-afQHL2`并恢复；test-only未触碰。
- batch`image_token_asl_layer1_epoch_search_formal_seed0_20260902_171919`已在tmux
  `image_token_epoch_search_20260902_171919`启动。8卡连续两次至少23.6GB空闲后，一卡一组运行
  epoch18/22/26/30/34/38/42/48；8份config均为clean`7091da6`、seed0/test/batch64、冠军结构、
  AMP/TF32 on、no-checkpoint，除epochs外字段一致。
- 八组task0 cycle1均完成84 optimizer steps、skipped0，loss`0.61478711`、Adapter ASL
  `0.02839343`完全相同；每组GPU约占1.99--2.07GB，仍余约22GB。日志扫描无OOM/non-finite/
  FloatingPointError/RuntimeError/Traceback，当前继续运行且不启动其他batch。

## 2026-09-02：同步并分析统一epoch搜索

- tmux与runner均已结束；batch status complete、8份seed summary和8个exit code齐全，exit code全0。
  严格汇总确认所有run同一clean`7091da6`、task0--7完整、总epochs和updates与声明epoch精确对应、
  skipped0、无checkpoint；日志无OOM/non-finite/Traceback/RuntimeError。
- 30 epochs精确复现历史冠军final/average mAP`32.5365/39.1445`并排名第一。26 epochs第二，
  final`32.3681`低`0.1684`，average`39.1940`高`0.0495`；其提升集中于task0--4，task5--7
  分别低`0.0619/0.1539/0.1684`，不满足用户final mAP目标。
- 34/38/42/48 epochs final分别为`31.9376/31.8459/31.9648/31.5501`。34与48相对30在8个
  task全部退化；48的task6/final低`1.0287/0.9864`，而多数末轮训练loss更低，属于延长训练后的
  过拟合/泛化下降，不是优化不足。
- 按预声明规则，30保持第一即停止epoch搜索，不运行31--33或27--29。下一步固定30 epochs，进入
  对所有task统一的scheduler模式搜索；若仍无冠军，再转full image + person crop双视图。
- JSON、严格汇总、manifest、launcher status与原始日志已同步本地，压缩包两端SHA-256均为
  `460890c5b44eb2788f2254b333a3d3376e3df6a310cbc18431065537d2632f30`，不含checkpoint。

## 2026-09-02：实现8组统一scheduler搜索

- 从`exp/emotic-image-token-epoch-search@4e8616d`创建
  `exp/emotic-image-token-scheduler-search`，保留已同步epoch结果与三份待提交上下文更新。
- `runner.py`新增`cosine/linear/constant/multistep`模式、相对min LR、warmup比例、multistep比例/
  gamma入口。历史无warmup/min cosine继续走原生PyTorch路径；其他模式用共同relative multiplier
  同步缩放主模型LR0.0125和Adapter LR4e-4。
- warmup定义为30 epochs下前2/3 epochs从`1/W`线性达到基础LR，再进入余下周期的cosine-to-zero；
  multistep里程碑为ceil(30×0.6/0.85)=18/26，gamma0.1。config记录模式、比例、实际warmup epochs、
  milestone比例/epoch、gamma与step unit。
- 新增4项scheduler数学测试，覆盖relative min对所有parameter group、2/3 epoch warmup、linear/
  constant/multistep端点和非法组合拒绝；新增完整scheduler搜索汇总器3项测试，覆盖赢家选择、锚点
  收口以及逐epoch LR漂移拒绝。
- 新增正式单组runner与一组一卡8-GPU launcher；所有run固定30 epochs和single1 b32冠军参数，结束后
  自动严格核验240 epochs、13950 updates、skipped0、clean共同commit、no-checkpoint和四条LR轨迹。
- 本地Python编译、shell语法和diff检查已通过；完整Torch单测需提交推送后在服务器ddp环境执行，
  通过并完成8种代表性scheduler轨迹smoke后才启动正式实验。
