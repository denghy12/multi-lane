# Shared Adapter residual-scale validation results

## Run integrity

Validation batch `shared_capacity_residual_scale_validation_20260925_222440` used EMOTIC Track A, seed 0, tasks 0–7, 30 epochs per task, batch 64, frozen CLIP ViT-B/16, fixed three-view fusion, shared Image-token Adapter after block 1, residual scale 0.03, shared BCE plus Adapter ASL, auxiliary view loss 0.1, AMP/TF32. The only intended difference was shared Adapter bottleneck 32 versus 97. Both runs completed 240 epochs and 13,950 optimizer updates with zero skipped updates, non-finite values, or errors. Thirty synchronized result/log files matched remote SHA-256 values.

## Accuracy

| Metric | A0 shared b32 | A-cap shared b97 | b97 − b32 |
|---|---:|---:|---:|
| Final mAP | 42.3199 | 42.5853 | +0.2654 |
| Mean of 8 cumulative-task mAPs | 48.9646 | 50.2722 | +1.3077 |
| Final cF1 | 38.0010 | 38.0619 | +0.0609 |
| Final oF1 | 58.4350 | 58.4480 | +0.0130 |
| Mean forgetting | 0.9534 | 1.0190 | +0.0655 |

Per-task fused mAP (b32 → b97; difference):

| Task | b32 | b97 | Difference |
|---:|---:|---:|---:|
| 0 | 58.1677 | 61.4543 | +3.2866 |
| 1 | 57.4367 | 59.3343 | +1.8977 |
| 2 | 44.2557 | 45.7307 | +1.4750 |
| 3 | 49.0979 | 50.4293 | +1.3314 |
| 4 | 48.5295 | 49.7520 | +1.2225 |
| 5 | 46.9539 | 47.6882 | +0.7343 |
| 6 | 44.9552 | 45.2039 | +0.2487 |
| 7 | 42.3199 | 42.5853 | +0.2654 |

The wider adapter improves fused validation mAP on all eight checkpoints, but most of its average gain comes from earlier tasks. Its final gain does not pass the prior promotion threshold of +0.5 mAP.

## Which views improved

At final task 7, the mAP values for fused / Full / Person / Face / reliable-Face were:

- b32: `42.320 / 40.760 / 39.696 / 34.599 / 38.042`.
- b97: `42.585 / 41.014 / 40.042 / 34.653 / 37.917`.
- Difference: `+0.265 / +0.255 / +0.346 / +0.054 / -0.125`.

b97's advantage comes mainly from Full and Person. Its all-Face score is nearly unchanged, while reliable-Face is slightly lower. Across tasks 0–3, b97's Face mAP is lower than b32 by `0.570, 0.478, 0.199, 0.018`; reliable-Face is lower by `1.423, 1.428, 0.706, 0.437`. The fixed fusion weights and reliability mask are identical between arms, so these differences are not caused by a learned router.

The test manifest has 5,368 person samples; 3,261 (60.75%) pass the fixed Face reliability rule. The rule is `valid_face && !ambiguous_match && face_short_side >= 24 && face_detection_score >= 0.6`. Reliable rows use `[0.64,0.16,0.20]`; other rows use `[0.80,0.20,0]`. This is a data-quality filter, not calibrated prediction confidence.

## Stability and residual diagnostics

Both arms trained stably: 240/240 epochs, 13,950/13,950 updates, zero AMP skips, finite Adapter gradients, and no NaN/OOM/traceback. b97's mean forgetting is 0.066 points higher, so it did not improve continual retention.

The logged per-view residual diagnostic is the average per-token energy ratio `sum(delta²) / sum(frozen_token²)` over training batches in an epoch; it is not a norm ratio or a bounded percentage. At task 7 it was Full/Person/Face `1.413/2.327/3.194` for b32 and `7.076/8.277/10.702` for b97. At task 3 it was `37.612/45.187/60.038` for b32 and `115.259/142.915/187.276` for b97. A fixed multiplier of 0.03 therefore does not bound the learned residual energy, particularly for b97. These diagnostics are not proof that residual magnitude caused any specific AP change.

The Adapter reads Full/Person/Face token inputs with the same shared parameters to generate selector evidence; it does not unfreeze CLIP or replace the frozen image-token residual stream. Three views are then combined by the same fixed, reliability-masked prior. This validation tests shared selector-evidence capacity, not ParaX or encoder-level dynamic routing.

## Decision and next experiment

Do not add view experts, learned routing, level embeddings, or another residual penalty based on one seed. Earlier Image-token Adapter searches found that generic residual regularization and learnable gates can suppress useful residual signal, so repeating those knobs before verifying the capacity effect would be poorly controlled.

The next validation is a paired seed 1 and seed 2 replication of b32 and b97 at scale 0.03. It changes only the seed, uses GPUs 0/1 while the already-requested seed-0 test stays on GPUs 2/3, and does not access test. Promote wider capacity only if it remains numerically stable and the three-seed validation aggregate clears the registered final-mAP margin without meaningful per-view or forgetting regressions. Otherwise retain b32 and end the width expansion.

## Synchronized artifacts

- Results: `output/emotic_benchmark_runs/multi_lane_shared_capacity_residual_scale_v0.1/`.
- Logs: `logs/emotic_track_a_shared_capacity_residual_scale/shared_capacity_residual_scale_validation_20260925_222440/`.
- Original protocol: `docs/shared_capacity_residual_scale_validation_plan_20260925.md`.
- Held-out test configuration: `docs/shared_capacity_residual_scale_test_plan_20260926.md`.
