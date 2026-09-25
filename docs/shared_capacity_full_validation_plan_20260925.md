# Shared Adapter capacity full validation

The three-task paired screen found that shared Image-token Adapter bottleneck 97 improved every measured view, while fully independent view banks damaged Face. This validation asks whether the shared capacity gain survives the complete continual-learning sequence.

Locked protocol: EMOTIC Track A, seed 0, tasks 0--7, validation only; frozen OpenAI CLIP ViT-B/16; shared Selector, Prompt, classifier and fixed reliable-Face fusion; compare shared bottleneck 32 (49,952 parameters/task) against shared bottleneck 97 (149,857 parameters/task). Both use layer 1, ReLU, residual scale 0.1, independent task initialization, BCE for shared parameters, ASL 9.8/0/0.05 for Adapter parameters, auxiliary view loss 0.1, joint gradients, Adam reset per task, main LR 0.0125, Adapter LR 4e-4, cosine to zero, batch 64, 30 epochs/task, AMP and TF32.

No Router, level embedding, private view bank, distillation, checkpoint or test is used. Report the complete trajectory, final/average mAP, cF1/oF1, forgetting, all single-view scores, residual ratios, gradient audits and frozen-CLIP checks. Retain b97 only if its gain survives late tasks without systematic single-view damage or material forgetting increase.
