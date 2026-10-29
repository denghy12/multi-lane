# 人物条件Summarize六组：执行协议（seed1，8任务，GPU0）

用户2026-10-29明确授权执行完整Validation/Test，沿用已讨论六组方案；不是仅任务0–2筛选。分支codex/person-grounded-summary-20261029，起点b4015bc；本地已有dirty分析保持，不stage无关文件。通过Git-only在服务器创建独立multi-lane-main-person-grounded-summary-20261029，不更新旧实验源码或main/test-only工作树。

## 固定实验

六组PERSON_CONST_POOL/ATTN、PERSON_CONTENT_POOL/ATTN、PERSON_GROUND_POOL/ATTN。Full8全局+2主体，Person/Face各10，单层64维4头、一个读出Query；Pool322→127→64；最终1%RMS限幅；查询模长匹配，低秩16、主体U2×768。常量/内容/定位三档分别控制压缩前Person内容、TRAIN定位监督，所有组名义参数/公共投影初始化配对。估算Pool221789、Attention221728/任务，落地统计核对。

原三路Task Forward、共享Image-token Adapter/Selectors/Prompts/head与原监督完整保留。新增BCE只更新私有分支，head与原特征detach；定位loss仅更新新主体U和条件器，不更新原Full。模型fresh按自身流程训练，不安装A/B/教师，不换头、不额外5epoch。当前任务结束封存所有旧依赖，推理拼接全部已见任务类别。

EMOTIC/seed1/8任务[5,3,3,3,3,3,3,3]/每任务30epoch/B64，主Adam LR.0125、Adapter/私有LR.0004，WD0、任务内cosine；b32/index1/scale.03，原aux BCE.1、Adapter-ASL neg gamma9.8/pos0/clip.05，Person颜色扰动off。原72线程/AMP-TF32/scaler65536和每任务两次报告CPU随机draw保持。新Summarize/私有读出显式FP32，原head的AMP与历史按batch CPU float32 sigmoid保持。原所有类别loss归一化legacy_full_zero保持。

定位权重.01，epoch1–5按.002/.004/.006/.008/.01升温，恢复由独立scheduler状态确定。仅合法框、>=90%Full可见且有patch相交的人物监督；完全不可见仅屏蔽主体memory，不删全局/Person/Face。变换框随真实crop/flip/resize，CLS只从主体候选移除。第一步可能无分类query梯度，但后续必须可学；定位组可直接从定位loss学查询。

各组240epoch/13950主更新，均0主私有跳步。6组合计83700主更新；不改变batch/epoch或Full裁剪补监督。历史TRAIN人数5353/4394/861/9931/4352/2536/627/1526，task6只有300主更新；先用专用TRAIN几何统计核对有效监督（独立RNG，不接触Val/Test、不读模型）。

## 路径与入口

相对独立服务器实验worktree：

- 配方/只读核验：../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train。
- 数据：../multi-lane-main/datasets/EMOTIC。
- 预训练：../CODE_DDP/pretrained/clip/ViT-B-16.pt（核验历史SHA）。
- Face manifest：../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909。
- TRAIN统计：output/person_grounded_summary/train_geometry_census_20261029/result.json。
- 输出：output/person_grounded_summary/<run_id>/<arm>/{train,val,test}。
- 日志：logs/person_grounded_summary/<run_id>/；正式run拟seed1_20261029_01，smoke拟smoke_seed1_20261029_01。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 /opt/conda/envs/ddp/bin/python -m multi_lane.track_a.grounded_geometry_census \
 --source-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train \
 --data-root ../multi-lane-main/datasets/EMOTIC \
 --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
 --output output/person_grounded_summary/train_geometry_census_20261029
bash scripts/emotic/run_person_grounded_summary_gpu0.sh smoke_seed1_20261029_01 --smoke --smoke-updates 4
bash scripts/emotic/run_person_grounded_summary_gpu0.sh seed1_20261029_01
```

上述是预定命令，不等于已运行。先相关新机制和完整Track-A回归，再真实smoke，才能正式启动。只GPU0，最多4作业/2非训练，启动前free>=6000MiB、shm>=1024MiB、12秒间隔，内存不足自动排队，保留其他GPU任务。

## 自动产物与审计

每epoch/任务验证前保存完整双optimizer/scaler/scheduler/RNG恢复，任务compact、训练loss/新query/定位有效数、框内质量、摘要/修正范数。原参数/原30轮loss/RNG与历史逐任务精确核对，旧TRAIN探针和跨六组增强流/OFF/三路预测核对。复现失败不得扩大容差冒充匹配。

每组完成8任务先TRAIN每任务固定4batch锁定自己的delta均值方向，带checkpoint/config SHA，后允许Val/Test；首64 smoke锁1batch。两split均全8阶段ON/OFF、零早期条件、屏蔽主体/全局/Person/Face入口；0/2/6/7再做同范数TRAIN均值和质量组内错配方向，保存原logits/按batch概率、人物/类别/Face/batch边界和query profiles。

0/2/6/7同图分组完整配对前向：native、只换早期Person条件、早期交换后匹配native修正范数、只换晚期Query、只换Face/Person memory。早期交换保持自己h0/head/晚期Query及其他memory。供体按ID、双方可见，不按标签/预测选人；报告全部>0可见和双方>=90%子群，以及当前可评价标签差异后的分层/有符号margin；不使用未来标签。paired native单独保存，禁止混用不同batch人群。

固定B0及CLS-MLP32.6800/39.5292、32.7139/39.5923作参照。定位、主体条件贡献、两项Test超过强参照、同取证Attention超过Pool分别验收；没有结果前不宣称有效。无需用户手动同步代码或启动；完成后回传日志/结果并核验SHA。

## 实际进度

创建协议时仅本地编译、shell语法及diff检查通过；尚无远程测试/统计/smoke/正式精度结论。后续实际记录追加，保留旧main/test-only状态SHA ea2e261e…/e9d1479f…。

## 预检通过与首次正式启动

- 模型运行commit434944950677935c26d9484e4a05b319bd866f2a，remote独立worktree clean。本地/upstream/remote HEAD一致。
- 40项专项/相关（包含11新机制、CUDA AMP实际执行）与434项Track-A均通过。TRAIN统计task6有效定位127/627、10batch均非空，其余任务也全部非空。
- 六组两任务各4更新smoke/标定/Val首64/全部归因通过；各8更新0主私有skip，12阶段原参数/loss/batch/OFF/三路逐值匹配。每stage61名Person、8名Face供体实际可交换。参数/任务Pool221789、Attention221728。
- 475份预检产物已同步本地逐SHA，归档SHA3e52bae439ada02ebc1ad66b118844e66b6b23596ebb7ab19c11d4546f565a82。
- 正式seed1_20261029_01已在原model worktree启动，初始四训练/两排队。用户随后要求六组同时启动，故仅调度上限改为6，所有模型/数据/超参数和原四进程保持。需要独立调度worktree接管，不能在运行中pull或改模型源码。
