# Shared Image-token Adapter capacity: complete validation result

## Protocol and run integrity

The paired validation used EMOTIC Track A, seed 0, tasks 0--7, 30 epochs per task, batch size 64, frozen OpenAI CLIP ViT-B/16, shared Selector/Prompt/classifier, fixed reliable-Face fusion, Image-token Adapter after block 1, BCE for shared parameters and ASL (9.8/0/0.05) for Adapter parameters, auxiliary view loss 0.1, Adam reset per task, main LR 0.0125, Adapter LR 4e-4, cosine schedule, AMP and TF32. The only intended difference was shared Adapter bottleneck 32 versus 97. Test and checkpoints were disabled.

The b32 control completed all eight tasks: 240 epochs and 13,950 successful optimizer updates, with zero skipped updates. The b97 arm completed tasks 0--2 and 17 epochs of task 3, then stopped with `FloatingPointError: ASL received non-finite training logits`. Its status code is 1; this is a failed/incomplete run, not an eight-task result. It had 140 skipped updates during task 3 before the exception. The task3 skip sequence began at epoch 14 and rose across epochs 14--17 to 5, 29, 52 and 54; the next cycle failed. The b32 arm completed the same task3 with 30/30 epochs and zero skipped updates.

The output, four logs and three control/status files were synchronized locally and their SHA-256 values matched the server. No test split was accessed.

## Accuracy evidence available before the failure

| Cumulative task | b32 mAP | b97 mAP | b97 − b32 | b32 cF1 / oF1 | b97 cF1 / oF1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 59.2605 | 61.7823 | +2.5218 | 40.6627 / 78.4922 | 42.2654 / 79.4426 |
| 1 | 58.5271 | 60.1986 | +1.6715 | 45.0690 / 72.9274 | 46.3620 / 74.1698 |
| 2 | 44.8063 | 45.8809 | +1.0746 | 36.1886 / 61.2452 | 37.2968 / 62.5110 |

Across these three comparable checkpoints, b97's average mAP was 55.9540 versus 54.1980 for b32, a gain of 1.7560 points. The gain shrank at each cumulative task; this is promising early evidence but does not establish a final eight-task benefit.

At task 2, b97 improved Full, Person, Face and reliable-Face mAP by `+1.095`, `+0.915`, `+0.557` and `+0.158` over b32. At task 0, Face and reliable-Face were lower by `-0.841` and `-1.001`, while Full and Person were higher by `+2.219` and `+2.442`. Thus the larger shared Adapter did not specialize one view, and its gains were strongest in Full/Person; the Face effect varied over tasks.

## What the complete b32 control says

The b32 arm reached final/average mAP `42.7525/49.6763`, final cF1/oF1 `38.2968/58.0892`, and mean forgetting `0.8915`. Its cumulative task mAP was `59.2605, 58.5271, 44.8063, 49.7810, 49.2285, 47.5841, 45.4701, 42.7525`. On task 7, fused/Full/Person/Face/reliable-Face mAP was `42.753/41.305/39.500/34.886/38.367`.

The aggregate forgetting value is modest, but early classes still show visible losses: Annoyance `5.45`, Aversion `4.92`, Affection `2.43`, Anger `2.34` and Anticipation `2.13` points. This is a separate continual-learning issue; it does not explain the b97 numerical failure by itself.

## Diagnosis

1. **The b97 early accuracy gain is real within tasks 0--2, but the formal arm is incomplete.** Do not compare its three-task average with the b32 eight-task average or claim that b97 is the better full continual learner.
2. **The immediate failure is numerical instability in the larger Adapter path.** A-cap's loss began turning upward in task3 late epochs while AMP skipped an increasing fraction of updates; the runner then detected non-finite logits before ASL. Task3 has 156 batches per epoch, compared with 14 in task2, so it sharply increases update exposure. The b32 arm handled that same task and schedule without skipped updates.
3. **Saved residual diagnostics show that b97 changes the token stream more strongly.** At the end of task0, the recorded mean per-token residual-energy/token-energy ratios for Full/Person/Face were `40.84/41.09/58.47` for b97 versus `15.29/15.22/21.35` for b32. At the end of task1, b97 reached `81.35/74.33/96.44`, versus b32 `21.62/20.81/30.42`. These are energy ratios, not raw norm ratios. There is no saved task3 failure-batch ratio, so these values support a risk mechanism but must not be reported as the exact ratio at the crash.
4. **The result does not point to old-class negative gradients.** Training uses `legacy_full_zero`, which zeroes hidden-class logits as constants; the more direct signal is that the larger Adapter's active path becomes numerically unstable under the same shared objective.
5. **The wider bottleneck is not disproven.** Its early gains warrant one controlled stability test. But adding experts, private views, routing, more layers, distillation or lower-precision tweaks now would make the cause harder to isolate.

## Next experiment and decision rule

The follow-up is running as batch `shared_capacity_residual_scale_validation_20260925_222440`, on branch `exp/shared-capacity-residual-control`, commit `a6b73a9`. It compares the same shared b32 and b97 banks while reducing the fixed Adapter residual scale from 0.1 to 0.03 for both arms. It retains seed 0, tasks 0--7, 30 epochs/task, batch 64, all optimizer/loss/fusion settings, AMP/TF32 and validation-only evaluation. No test, checkpoint, Router, level embedding, view-private parameters, ParaX or distillation is included.

The change tests whether lowering the Adapter's effect on the frozen CLIP token stream can preserve b97's capacity benefit without the task3 overflow. Adam may partly compensate for a smaller fixed scale through its parameter updates, so stability and validation metrics must be measured rather than inferred from the scale setting. Both task0 smokes passed, and both formal arms entered task0 with 84 updates and zero skips.

Retain b97 only if the arm completes all eight tasks without non-finite logits or sustained AMP skips, has final mAP at least 0.5 points above the b32 scale-0.03 control, has average mAP no lower than that control, and has no task7 Full/Person/Face/reliable-Face regression worse than 0.5 points or material increase in forgetting. If the reduced-scale b97 arm still overflows, stop expanding its width and test only one further stabilization change (lower Adapter LR) in a separate paired run. Do not resume the failed run from a partial state because checkpoints were disabled.

## Artifacts

- Synchronized outputs: `output/emotic_track_a_shared_capacity_full_validation/shared_capacity_full_validation_20260925_192017/results/`
- Synchronized logs: `logs/emotic_track_a_shared_capacity_full_validation/shared_capacity_full_validation_20260925_192017/`
- Synchronized status and launch manifest: `output/emotic_track_a_shared_capacity_full_validation/shared_capacity_full_validation_20260925_192017/control/`
- Follow-up protocol: `docs/shared_capacity_residual_scale_validation_plan_20260925.md`
