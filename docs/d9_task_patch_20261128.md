## 2026-11-28 启动各Task Forward独立融合与有界调整实验

用户明确“开始执行”上轮下一步方案。新分支codex/d9-task-patch-fusion，本地tmp/d9_task_patch_20261128，基于5ceea9b；带入上一轮未提交分析文档，保留旧worktree原改动。服务器计划独立multi-lane-main-d9-task-patch-20261128；主工作树154项、test-only5项未提交状态保持。

本轮两来源各20结果行，共40：固定/初始2项，原共用/CLS/四分区/可学习patch四方法×当前广播与各头独立2方式共8项，三特征读法加有界共用项3项，下界-.5的共用与三特征4项，三特征增加逐图u均方惩罚0.01共3项。各任务拟合14模型，其中4复用父实验封存结果、10新增。所有输入仍为32维最终CLS与196patch，三路冻结、无原logits比例输入、Full1/辅助0起步、上界+.25；负向有界函数B*tanh(.25*u/B)与正向.25*tanh(u)在0处斜率均.25，B=.25或.50，不导入历史比例。新共用b在总有界映射内部。逐图惩罚只作用u，不惩罚b。

EMOTIC seed1；90%分支用另10% TRAIN当期合法人物拟合，全TRAIN用全部当期合法人物，训练曝光限制不变。原图三折、Adam全批300步LR.001、参数均方L2.001、weight_decay0无scheduler；100比例交换/10patch换图/一次分区位置打乱。所有Val先完成并封存，之后所有Test；各头自己的方法中按Val平均/最终/声明顺序选定，仍报告全部候选。最终task7全部26类及8阶段全已见类平均，对照Test33.1770/40.2083与同源固定。预检拟合全部8头（小人群、5步），评价task0/7（2交换/2patch换图），不能作效果结论。

新增可选task_final_tokens接口只读取每条Task Forward的真实最终CLS及其Frozen9后空间流；捕获开关原logits逐值重放，正式当前头特征必须与父实验全部存档逐值相同。新输出明确区分prediction__/control__命名，修复旧CLS均值对照与CLS+均值方法数组重名问题，旧产物不覆盖。新模块task_patch_fusion.py、提取/拟合评估/执行脚本、JSON配置和10项测试。已有四个模型文件仅新增可选捕获；旧训练推理默认路径保留。

原输入来源沿用d9_equal_updates/seed1_20261119/formal、d9_sample_fusion/seed1_20261118/sample_full，原logits/名单缓存d9_summary_eight/seed1_20261127/formal，父patch缓存d9_final_patch/seed1_20261127/formal。入口CUDA_VISIBLE_DEVICES=0 bash scripts/emotic/run_d9_task_patch.sh execute；输出output/d9_task_patch/seed1_20261128，日志logs/d9_task_patch/seed1_20261128。预计正式2–4小时，须预检后校准。分支层面分折重训、更高投影维度、扩网络及新损失均等90%证据成立后再推进，本批不启动。当前静态AST/JSON/bash检查通过，Linux测试/真实预检/正式启动待完成；无main合并。


