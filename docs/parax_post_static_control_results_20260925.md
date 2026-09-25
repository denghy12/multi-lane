# ParaX P-post static control results — 2026-09-25

## Protocol

Batch `parax_post_static_control_20260925_003631` repeated the paired B0 protocol on EMOTIC Track A, seed0, task0--2, 30 epochs per task, batch size 64, Adam reset per task, main learning rate `0.0125`, validation only, no checkpoint, and no test evaluation.

`P-post-static` used the same final lane-feature placement as P-post-small, rank32, three experts, fixed output scale `0.001`, strict `zero_output` initialization, and froze the expert center after task0. Its only routing change was uniform expert weights; no input-dependent router was trained.

## Results

| method | final mAP | average mAP | forgetting | final cF1 | final oF1 |
|---|---:|---:|---:|---:|---:|
| B0-paired | 45.2395 | 54.9341 | 2.7760 | 36.8870 | 61.6314 |
| P-post-small, dynamic | 44.2960 | 53.7270 | 2.5475 | 35.4220 | 61.1974 |
| P-post-static, uniform | 43.9191 | 52.8834 | 2.2440 | 36.6518 | 61.9663 |

The static control is the most stable ParaX variant so far, but it has the lowest mAP. Relative to B0, its task0/task1/task2 mAP values are `57.4997/57.2315/43.9191`, compared with B0 `60.5225/59.0402/45.2395`. The task0 loss is already `-3.0228`, so later-task distillation cannot be the primary fix.

At task2, static Full/Person/Face/reliable-Face mAP is `42.2100/41.3794/33.6396/38.1449`, versus B0 `43.7058/42.5883/33.9794/39.4689`. Static improves final oF1 (`61.9663` versus `61.6314`) and reduces forgetting, but the ranking and calibration losses lower mAP.

## What the dynamic/static comparison proves

Dynamic P-post is `+0.3769` mAP above static at the final task and `+1.6248` mAP above static at task0. Therefore input-dependent routing is not useless: it recovers some task-specific/view-specific adaptation. However, both variants remain below B0, and neither changes the fixed fusion weights. The common failure is the post-residual feature coordinate shift; dynamic routing only partially compensates for it.

Static gates remain exactly uniform (`entropy=1.0986`, view gate L1 distance `0`). Its residual/token ratios fall to roughly `0.013--0.021` by task2, yet mAP remains below B0. This rules out router collapse and large residual magnitude as the sole explanation. The branch must be coupled to a calibration or prediction interface that preserves the B0 feature geometry.

## Five-stage status

1. **Stage 1, comparability and initialization:** operationally complete for the current paired runs. Strict zero-output smoke is below `2e-8`, and repeated B0 results match. Selector/head hash, first DataLoader index, and anchor feature drift are still useful bookkeeping but are no longer the leading explanation.
2. **Stage 2, paired task0--2 validation:** complete and failed the accuracy gate. P10 and P-post variants do not exceed B0.
3. **Stage 3, shared center stability:** complete. Shared-live caused the largest forgetting; Frozen-center reduced it but did not recover accuracy.
4. **Stage 4, CPU-cache distillation:** not executed. It is deprioritized because P-post-small and P-post-static already have forgetting below B0; distillation cannot repair the task0 gap.
5. **Stage 5, level routing:** remains paused. Existing gate differences have not yielded a single-view or fused mAP gain, so adding level embeddings would confound an unresolved post-feature calibration problem.

## Next experiment

Run `P-post-tiny` with the dynamic router and fixed output scale `0.0001`, keeping rank32, three experts, task0 center training, task1--2 center freezing, and all B0 settings unchanged. This is the smallest informative test of whether reducing the common feature-coordinate shift can close the task0 gap while retaining the dynamic gain. If it remains below B0, stop expanding ParaX in the image stream and move to a view-specific post-feature calibration or logit-level residual.
