**第6层图像CLS Person：FP32数值恢复**

固定scale4096的续训仍在原epoch19出现3skip，因此该尝试被保留为失败记录，不继续加载其epoch19。第9层六组在独立9d84f55源继续，本文件不改变它们的训练配置。

从原epoch18做内存Adam重放，在batch140首次获得非有限梯度。保存更新前完整方法状态及增强输入后，AMP4096和AMP1都出现Selectors、Prompts0/1及Adapter非有限，FP32所有梯度有限；新CA参数梯度没有非有限。这是动态训练状态下的直接诊断，区别于此前固定epoch18权重的24批均有限检查。仍不能仅凭这些参数名断言唯一出错算子。

恢复继续使用原epoch18模型、优化器、scheduler、完整RNG；任务0–2和任务3前18轮按原有效状态继承。之后整个Person接收分支FP32训练（AMP/TF32关闭），非有限梯度在更新前立即拦截，完整每任务30轮/B64/seed1/8任务预算不变；原BCE/Adapter-ASL、LR.0125/.0004、QKV64维4头、index5、5%总输入修正和R1固定logits读出不变。不是原AMP数值轨迹的完全重放，结果须标为数值稳定恢复版本。

入口：PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/ddp/bin/python -u scripts/emotic/run_r1_layer6_fp32_recovery.py。输出output/r1_layer6_fp32_recovery/seed1_20261105，日志logs/r1_layer6_fp32_recovery/seed1_20261105，不覆盖两份AMP失败目录或原checkpoint。先测试、真实epoch18两更新FP32 smoke，按实测峰值显存准入，与第9层GPU0作业共享空间，显存不足自动等待。

继承的任务0–2 Test缓存仅用于训练历史记录，不作为统一正式评估。训练完成后将Person的所有8个checkpoint重新做完整FP32 Validation/Test，保存到fp32_readout/person，再与原Full/Face固定logits融合，输出completed_layer6_results_fp32.json。不会换入其他分类头或从失败epoch19续跑。新数据供数值恢复的Person-only和Person+Face两配置使用；其他四个第6层配置为只读旧来源。

新版本只修复数值精度，不能证明方法会提高mAP；后移是否有效仍由独立第9层完整Val/Test回答。若需要纯位置因果，须另行匹配精度控制，不把FP32和AMP差值全部归于深度。
