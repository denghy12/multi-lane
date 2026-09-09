# Taskwise three-view Router reassembly

## Goal

Correct the incremental semantics of the saved three-view Router without retraining any expert or
Router. Router `k` is trained from task-`k` current classes and may therefore alter only the class
lane introduced by task `k`. Evaluation concatenates the independently fused lanes for all seen
tasks.

For evaluation task `t`:

```text
Router0(x) -> fuse Full/Person/Face probabilities for C0
Router1(x) -> fuse Full/Person/Face probabilities for C1
...
Routert(x) -> fuse Full/Person/Face probabilities for Ct
concatenate C0...Ct -> compute metrics
```

Each Router state and its calibration normalization remain frozen after their task. Features for
Router `k` use only the class prefix available when Router `k` was trained, so later class
probabilities cannot change an old Router weight.

## Reused inputs

- seed0 90/10 Full, Person and Face validation/calibration scores;
- all six saved R2/R3 Router candidates (`prior={0,0.1,1}`);
- saved validation CLIP descriptors;
- existing seed0/1/2 100%-train Full, Person and Face formal-test probability dumps;
- locked R1 validation selection SHA-256
  `683e611f883a515fd39b4915f80375a72aa98c91f9c0a9b44e38e5fca39501ea`.

No expert or Router is retrained. Validation re-evaluates all three existing priors in each family
under the corrected lane semantics and selects the best R2 and best R3 by final mAP, then average
mAP, then stronger prior. R1 remains fixed at reliable-Face weights `0.64/0.16/0.20` and invalid-Face
fallback `0.80/0.20/0`.

## Validation and diagnostic test

Validation compares R1, the corrected best R2, and the corrected best R3. R3 advances only if its
final validation mAP exceeds both R1 and R2.

At the user's explicit request, the same three validation-defined methods are also evaluated on
the existing seed0/1/2 test score dumps. This test section is exploratory: test metrics do not
select a candidate or change any Router, weight, threshold, or Face reliability rule. R3 test
requires one deterministic export of frozen-CLIP Full/Person/Face descriptors for the complete test
pool; descriptor extraction does not use labels.

## Execution

- Entry: `scripts/emotic/launch_multilane_track_a_taskwise_three_view_router.sh`.
- GPU work: one frozen CLIP descriptor export, batch 128; no optimization.
- CPU work: validation reassembly and three-seed test reassembly.
- External descriptors: `emotic_benchmark_runs/multi_lane_taskwise_three_view_router_v0.1/`.
- Result: `output/emotic_track_a_taskwise_three_view_router/<batch>/`.
- Logs: `logs/emotic_track_a_taskwise_three_view_router/<batch>/`.
