# Task Forward Face Pool 与第9层 Frozen patch

组合一沿用原 R1 Full、Task Forward Pool Face，使用第9层 patch Person 的完整保存预测。关闭对照加载 Person 原阶段 checkpoint，在相同 Val／Test 人群统一补足八阶段。OFF 路径只读取原 Person tensor，task7 与历史 OFF logits 逐值核验。旧worktree输出与日志 `r1_pool_patch_person_off/seed1_20261106`，完整汇总为 `pool_reference.json`、`person_patch_on.json`、`person_patch_off.json`。

组合二在同一条新 Face 路径使用两个原接口。第6个 Task block 后，Pool 对 Full 与 Person 的第6层冻结 patch 各自聚合，以 MLP 修正 Face Task CLS。第9个 Frozen block 后，Face 自身 patch 作 Q，读取自身图像 CLS 和另外两视图的原第9层空间 patch，修正空间流。第10层起由同一任务路径读出，最终只用一个 Face 头。发送端一条冻结前向取两处特征，两个新增模块均为任务私有，零输出初始化。Pool 保留原无显式限幅，记录相对 Task CLS 的修正范数；patch 上限为图像 CLS 范数的5%。Full、Person 使用原 R1 预测。可靠 Face 的固定 logits 权重0.64/0.16/0.20，否则0.80/0.20/0，再过 sigmoid。

训练 EMOTIC 历史划分，seed1、8任务、30轮/任务、B64。Adam、cosine、WD0，主LR0.0125、Adapter 和新增0.0004。Adapter b32/index1/scale0.1、原 BCE＋Adapter-ASL（9.8/0/0.05）、AMP固定4096与TF32。预训练 CLIP ViT-B/16 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`，数据 `../multi-lane-main/datasets/EMOTIC` 和原 Face manifest。新 Face 不加载历史任务权重。Pool 参照使用历史动态AMP训练，合并版固定scale4096，因此组间不是严格只改结构的精度匹配。开关只用同一共训checkpoint。

GPU0入口 `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/conda/envs/ddp/bin/python -u -m scripts.emotic.run_r1_face_pool_patch`。先2任务×2更新真实预检，再按实测显存启动8任务。逐任务Test、完整Validation，自动汇总R1／Pool／patch／combined。task0/2/7两split记录全开、只开Pool、只开patch、全关。日志 `logs/r1_face_pool_patch/seed1_20261106`，输出 `output/r1_face_pool_patch/seed1_20261106`，含训练恢复状态、分支预测、融合和开关结果。
