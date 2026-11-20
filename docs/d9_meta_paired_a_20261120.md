# 人数与人脸分层模型，在本模型的留出 TRAIN 上学习融合比例

本次仅更换三路预测的来源及与其配套的留出名单，沿用原随机等步数模型的配对融合方案。EMOTIC 官方划分，seed1，冻结人数／可靠 Face 分层已完成的八份 Frozen9 checkpoint。每个阶段使用该阶段 checkpoint，当期新增类与该划分自己的留出人物，不读取未来任务标签。名单全局为 11,623 张分支图、1,292 张融合留出图。每人自然版本和 Person 降至 112 后恢复的版本，可靠 Face 人物另有 Face 降至 112 后恢复的版本。各人有效版本等权平均 BCE，每个任务按 Face 可用性分两组，从可用视图等权开始，全批 Adam 学习率 0.01、300 步。旧任务比例封存，八阶段评估按原模型各任务头使用对应比例。属于历史 checkpoint 逐阶段重放，视觉分支训练在本实验开始前已经完成。

GPU0 提取留出 TRAIN 预测，batch64、两个加载进程，CPU 进行融合拟合与缓存评估。验证用全部自然 Val 和 Test，依次封存中心、Val、Test。与同分层固定、原随机自己的配对比例以及全 TRAIN 固定对照；固定比例对人脸可靠人物为 64/16/20，其余 80/20/0。判断分别比较 Val 最终和阶段平均，再查看锁定后的 Test。比较整个方法是否可迁移需关注不同名单同时改变了训练分支及融合监督样本。

运行 `bash scripts/emotic/run_d9_meta_paired_a.sh 0`。分层输入为 `../multi-lane-main-d9-split90/output/d9_split90/seed1_20261120`。原随机参照在 `../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal`。数据来自 `../multi-lane-main/datasets/EMOTIC` 和项目既有人脸清单，预训练权重复用 `../CODE_DDP/pretrained/clip/ViT-B-16.pt`。封存留出预测、每任务两套比例、逐类与分组 Val/Test、汇总见 `output/d9_meta_paired_a/seed1_20261120`，日志见 `logs/d9_meta_paired_a/seed1_20261120/run.log`。
