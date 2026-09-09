# Fixed three-view seed0/1/2 formal test

## Locked decision

This stage performs one held-out test evaluation of the validation-selected R1 rule. It does not
train a Router and does not search weights, thresholds, or Face-quality cutoffs on test.

- reliable Face (`valid && !ambiguous`, short side >= 24 px, detection score >= 0.6):
  `[Full, Person, Face] = [0.64, 0.16, 0.20]`;
- invalid or unreliable Face: exact fallback `[0.80, 0.20, 0]`;
- threshold: `0.5`;
- seeds: `0, 1, 2`;
- validation lock SHA-256:
  `683e611f883a515fd39b4915f80375a72aa98c91f9c0a9b44e38e5fca39501ea`.

The user explicitly chose direct three-seed test on 2026-09-09, replacing the earlier seed1/2
validation gate. This is therefore a locked confirmatory evaluation, not a new tuning round.

## Face source protocol

Each seed trains a Face expert on the complete train split for 8 tasks and evaluates held-out test:

- 30 epochs/task, batch 64, main LR 0.0125;
- per-task cosine annealing to zero, no warmup;
- Image-token Adapter layer 1, bottleneck 32, Adapter LR 4e-4;
- residual scale 0.1, ReLU, independent initialization;
- main BCE plus Adapter ASL `(gamma_neg=9.8, gamma_pos=0, clip=0.05)`;
- AMP and TF32 enabled;
- test probability dumps enabled, full checkpoints disabled.

The test Face manifest uses exactly the audited SCRFD detector and matching configuration from the
train/validation manifest. Test detection and matching do not use test labels. Existing Full and
Person formal-test probability dumps are reused for their matching seeds.

## Outputs

- Face sources: external `emotic_benchmark_runs/multi_lane_fixed_three_view_test_v0.1/`.
- Control summary: `output/emotic_track_a_fixed_three_view_test/`.
- Logs: `logs/emotic_track_a_fixed_three_view_test/`.
- Final artifact: `fixed_three_view_seed012_test_summary.json`, containing per-seed results,
  mean/sample standard deviation, and paired R1 minus Full+Person differences.
