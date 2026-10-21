# 跨视图Attention条件贡献与人物Query归因：seed0完整8任务

用户要求执行前轮提出的两项归因，并扩展到8task validation和test。本轮复用已有checkpoint，训练更新为0；没有新模型训练或超参数搜索。原先task0/2/7、validation的建议由用户的完整8任务双split要求覆盖。

## 科学问题与操作

- 对task_full及person_full：同一checkpoint正常开启残差，对照关闭新增Attention残差。保留共同训练后的Selector/Prompt/head；关闭结果不是重新训练的B0。
- 对person_full：只将同图另一人的Person摘要用作Query。当前人的Person原摘要残差底座、Full Key/Value、Face、分类头不换。替换发生在Query归一化后、投影前，等价于替换归一化的Query输入。
- Query来自额外的只读Person路径：在第6层交互前捕获donor摘要，关闭这次捕获路径的交互，丢弃其最终表示。正常三路仍按原顺序/原batch计算，不改变评估组批。由此产生额外推理成本，不宣称提速。

配对标准先验锁定为：真实legacy Resize256/CenterCrop224中，原人体框至少90%面积保留、框中心可见；同原图至少两名符合条件且身体框不同的目标。按sample ID字典序循环交换，一人一位donor，不使用类别或预测选择。重复同框只取一位代表。其余样本保持本人Query。

报告固定全人群结果、可交换子集、不可交换子集、可靠/不可靠Face子集；不同子集mAP不可直接比高低，主要比较同一子集的配对变化。子集仍对全部已见类别求macro AP，无正样本类别AP为0。

## 输入和固定设置

- 数据EMOTIC，seed0，全部8个任务边界（5/3/3/3/3/3/3/3类）；validation2397人、test5368人。
- 来源：`../multi-lane-main-person-conditioned-cross-view-20261021/output/person_conditioned_cross_view/formal_seed0_20261021_01/{task_full,person_full}/train/compact_checkpoints/task0.pth` 至 `task7.pth`。来源训练代码8fef770，源工作树74ece12。
- 新评估模型配置与来源一致：三路共享Frozen Forward权重，三条Task Forward；第6层、4头总宽64、scale.1、aux0；无MoE/ParaX/image-token adapter。
- 训练配置仅作来源记录：每任务30epoch/B64，Adam/cosine，主LR.0125/模块LR.0004。当前无optimizer/scheduler、新epoch或梯度更新。
- eval batch64、workers0、AMP/TF32开启、threshold.5、固定全人群，维持历史样本及batch顺序。
- 数据：`../multi-lane-main/datasets/EMOTIC`。
- CLIP：`../CODE_DDP/pretrained/clip/ViT-B-16.pt`，核对来源SHA。
- Face：`../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`。
- GPU0空闲，使用GPU0；正式四进程并行（两来源×validation/test），不干预其他GPU作业。

## 入口、日志、结果

入口：`bash scripts/emotic/run_cross_view_attribution.sh`。
正式run：`formal_seed0_20261021_01`，路径类别为`cross_view_attribution`，与来源实验隔离。
结果：`output/cross_view_attribution/<run>/{task_full,person_full}_{val,test}/`。
日志：`logs/cross_view_attribution/<run>/`。
各进程保存on/residual_off/query_swap（后者仅person_full）的每边界预测NPZ、Query配对manifest、分支及子集mAP、最终/阶段平均、SHA与源模型不变核验。

短跑：`SMOKE=1 RUN_ID=smoke_seed0_20261021_02 bash scripts/emotic/run_cross_view_attribution.sh`，仅validation，task0/2/7各原始前2batch，须实际包含可交换样本，不访问test图片。正式评估没有沿用短跑子集。

## 验证及保护

1. 正常开启的预测/分支结果与历史完整预测逐值核对。
2. 同人Query控制须逐值重放原结果；正式每个person_full checkpoint首batch也执行。
3. 两项干预均检查Full、Face端点逐值不变；Query交换的不可交换样本及其Person端点逐值不变。
4. CPU机制测试检查只改Q，不改Person底座、K/V、权重，缓存异常清理、训练时拒绝干预、几何与真实transform一致、确定性非自身同图配对及重复框/不可见排除。
5. checkpoint/源配置/Face manifest前后SHA一致，模型状态/冻结CLIP状态不变；不保存或覆盖源checkpoint。
6. 原工作树及test-only保留；代码通过新分支/Git-only新远程工作树同步。

结果只说明固定已训练模型的条件贡献。Query交换造成下降本身不能单独证明学会了正确人物关系，还可能包含输入扰动影响；没有下降也不能证明所有人物条件方法无效。test仅按用户要求固定报告，不选择配对阈值或参数。

初次smoke `_01`发现历史输入搬运会把新布尔mask转float，交换条件被严格检查中止，批次exit1；保留失败产物。已在归因入口单独保留mask类型，加入回归测试，修正后使用新run `_02`重跑，不覆盖旧结果。原模型正常开启/残差关闭及本人Query首batch重放已通过，validation几何筛选373人/175原图，不据此改阈值。

## 启动前核验

修正后403项全量测试通过（CPU运行跳过1项CUDA RNG，随后原cross-view 10项含CUDA RNG补测通过）。smoke_seed0_20261021_02两个来源的task0/2/7各2batch均完成，batch exit0。正常开启结果及三分支与历史逐值一致；同人Query重放逐值一致；Full/Face和不可交换样本不变；原checkpoint/配置/Face manifest SHA及模型/视觉参数前后不变。短跑每边界实际33名可交换者均产生Person logits变化，确认真实替换生效；这不是效果结论。

validation固定90%规则共373人/175原图可交换。所有筛选只用几何，不修改阈值。正式拟formal_seed0_20261021_01，4进程并行（任务Query/人物Query×val/test），全部8边界和完整人群。

## 正式运行已启动

完整8任务双split归因已由clean提交ae56624启动，tmux=`cross_view_attribution_full8_seed0_20261021`，run ID保持`formal_seed0_20261021_01`。GPU0四进程：task_full_val、task_full_test、person_full_val、person_full_test。启动健康快照确认四进程正常，task0正常/关闭两条件已完成、原预测逐值重放；各进程继续后续边界。GPU约6275MiB/100%。

validation可交换373人/175图；test可交换811人/372图。16份源checkpoint SHA仍与历史核验记录一致，原主树/test-only/来源树状态未变。`output/cross_view_attribution/implementation_20261021/formal_start_health.json`保存启动快照。后续仅文档同步，不改变运行源码；完整结果尚未结束。
