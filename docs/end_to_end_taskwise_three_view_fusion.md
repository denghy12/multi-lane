# End-to-end taskwise multi-view fusion protocol

## Goal

Replace the stopped, calibration-only probability Router with a train-time
feature-fusion mechanism. The stage asks whether per-sample content routing is
useful when it is supervised by the complete current-task training stream and
preserves MULTI-LANE's incremental task boundaries.

## Shared protocol

- Dataset: EMOTIC, frozen B5/C3 eight-task order, seed 0.
- Reporting: complete eight-task validation only. Test is forbidden.
- Training: 30 epochs/task, batch 64, Adam reset per task, main LR 0.0125,
  per-task cosine annealing to zero, no warmup, weight decay 0.
- Backbone: frozen OpenAI CLIP ViT-B/16.
- MULTI-LANE: 10 selectors, 10 prompts, 5 prompt layers, concat inference.
- Champion Adapter: Image-token, zero-based layer 1, bottleneck 32, LR 4e-4,
  scale 0.1, ReLU, independent initialization.
- Objectives: shared model parameters use BCE; Adapter parameters use ASL
  `gamma_neg=9.8`, `gamma_pos=0`, `clip=0.05`.
- Inputs: legacy Full crop; Person bbox plus 0.15 margin, CLIP-mean letterbox;
  Face manifest crop, CLIP-mean letterbox. All views share the horizontal flip.
- Auxiliary supervision: fused objective plus 0.1 times the mean objective of
  available view logits. Face contributes only when `valid && !ambiguous`,
  short side is at least 24 pixels, and detector score is at least 0.6.
- No checkpoints and no validation score dumps are required for this screen.

## Methods

- J0 `fixed_three_view`: fuse normalized task-lane CLS features with
  Full/Person/Face weights 0.64/0.16/0.20 for reliable Face, otherwise exact
  0.80/0.20/0 fallback. This is the learned-joint-training control.
- J1 `soft_three_view`: one hidden-16 MLP per task reads the three matching lane
  features and emits one sample-level masked-softmax vector. It initializes
  exactly at J0's priors. Unreliable Face features are zeroed before the Router
  and their output weight is exactly zero.
- J3 `soft_full_person`: one hidden-16 MLP per task reads matching Full/Person
  lane features, initializes at 0.80/0.20, and does not access Face.

Router `k` is trained only while task `k` is active. When task `k+1` begins,
Router `k` is frozen; inference uses Router `k` only for lane/classes introduced
by task `k`. The Router LR is 4e-4 and is reset with the task optimizer.

## Decision rule

Run J0/J1/J3 together from one clean commit. J1 advances to seed1/2 validation
only if its final validation mAP exceeds both J0 and J3. Test results and the
existing test-class breakdown cannot change this rule. If J1 fails, stop this
exact task-local end-to-end Router before adding capacity or new descriptors.
