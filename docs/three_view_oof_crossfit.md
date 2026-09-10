# Three-view image-group OOF/cross-fitting protocol

## Purpose

Replace the 10% per-task calibration Router with leakage-free predictions over the complete
EMOTIC train pool. This tests whether the previous negative result was caused by calibration
scarcity rather than by the dynamic-routing hypothesis itself.

## Expert cross-fitting

- Seed: 0; selection split: validation; test access: forbidden.
- Partition: stable SHA-256 image-group 3-fold assignment. All annotated people from the same
  source image remain in one fold.
- For every held-out fold, train separate Full, Person, and Face experts on the other two folds.
- Export held-out scores after every incremental task. The union of the three held-out folds must
  cover each eligible train sample exactly once.
- Total source runs: 3 folds × 3 views = 9; up to eight run concurrently, at most one per GPU.
- Full uses the legacy crop. Person uses bbox margin 0.15, square letterbox and the existing light
  jitter. Face trains only on valid, non-ambiguous crops; all samples remain present in held-out
  score dumps so the three views align.

The fixed expert protocol remains: 30 epochs/task, batch 64, cosine annealing to zero without
warmup, main LR 0.0125, Image-token Adapter layer 1, bottleneck 32, Adapter LR 4e-4, residual
scale 0.1, ReLU, independent initialization, main BCE, Adapter ASL 9.8/0/0.05, CLIP
normalization, AMP and TF32 enabled.

## Shared Router

- R1: reliable Face uses `[0.64, 0.16, 0.20]`; invalid Face uses `[0.80, 0.20, 0]`.
- R2: geometry, quality, confidence, entropy and prediction-disagreement features.
- R3: R2 plus three frozen-CLIP cross-view cosine descriptors.
- Architecture: one hidden-16 shared reliability trunk and a three-value bias per task.
- Training: AdamW, LR 1e-3, weight decay 1e-4, batch 64, 80 epochs/task, prior strength in
  `{0, 0.1, 1}`.
- Incremental semantics: training progresses task by task. Each task stores its Router snapshot;
  at evaluation Router k can fuse only task-k classes and later tasks cannot change that snapshot.
- Invalid Face always receives exactly zero weight.

## Decision

Evaluate on the same complete-fit seed0 Full/Person/Face validation endpoints. Rank within R2/R3
by final mAP and then average mAP. Continue to seed1/2 validation only if the best R3 exceeds both
fixed R1 and the best R2 in final validation mAP. Do not access test during this batch.
