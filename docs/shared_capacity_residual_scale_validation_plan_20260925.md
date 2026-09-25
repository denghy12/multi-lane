# Shared Adapter residual-scale validation

The complete capacity run showed that shared bottleneck 97 improved the first three tasks but became numerically unstable in task 3. Its measured residual ratios were already much larger than the b32 control before the failure. This paired validation holds the architecture and optimizer fixed while reducing the Adapter residual scale for both capacities.

Protocol: EMOTIC Track A, seed 0, tasks 0--7, validation only; frozen OpenAI CLIP ViT-B/16; shared Selector, Prompt, classifier and fixed reliable-Face fusion; shared Image-token Adapter at layer 1; compare b32 and b97 at `adapter_residual_scale=0.03`. Both use 30 epochs/task, batch 64, Adam reset per task, main LR 0.0125, Adapter LR 4e-4, cosine to zero, BCE for shared parameters, ASL 9.8/0/0.05 for Adapter parameters, auxiliary view loss 0.1, joint gradients, AMP and TF32.

The experiment does not add Router, level embedding, private view banks, ParaX, distillation, checkpoint or test. It is intended to answer whether b97's early accuracy gain survives when the residual path is kept in a numerically controlled range. Retain b97 only if both arms complete with zero skipped updates and it improves the late-task trajectory without systematic Face damage or a material forgetting increase.
