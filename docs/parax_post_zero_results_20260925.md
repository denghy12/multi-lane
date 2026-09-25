# ParaX P-post-zero results — 2026-09-25

`parax_post_zero_control_20260925_020000` completed paired B0 and P-post-zero validation on task0--2. Both runs completed 90 epochs and 5010 optimizer updates, with no skipped steps, OOM, NaN, checkpoint, or test evaluation.

| metric | B0-paired | P-post-zero | difference |
|---|---:|---:|---:|
| final mAP | 45.239467305 | 45.239467305 | 0 |
| average mAP | 54.934054559 | 54.934054559 | 0 |
| forgetting | 2.775999154 | 2.775999154 | 0 |
| final cF1 | 36.886985349 | 36.886985349 | 0 |
| final oF1 | 61.631419940 | 61.631419940 | 0 |

Task0, task1, and task2 mAP, all Full/Person/Face/reliable-Face metrics, pairwise ranking counts, and class AP values were identical. P-post-zero residual/token ratio and ParaX gradient norms were exactly zero. The initial logits smoke difference was `2.33e-08`.

This closes the comparability question: the ParaX module, registration, extra optimizer group, and post forward path do not change the training trajectory when output scale is zero. The P-post-tiny loss (`-0.5319` final mAP) therefore comes from a real nonzero residual, even though its measured ratio was only about `1.7e-7`.

The ParaX image-stream route should stop here. P-post-small, P-post-static, and P-post-tiny all remain below B0; lower scale approaches B0 but removes useful routing signal. Stage-4 distillation cannot fix the task0 loss, and stage-5 level routing would add complexity to a residual interface that has already failed the identity-preservation test.

The next controlled direction is an independent post-feature calibration: keep CLIP and ParaX disabled, preserve Full as the anchor, and train a zero-initialized task-local residual calibration for Person/Face in the fusion module. This tests whether the useful Face signal can be aligned after frozen encoding without changing the shared image stream.
