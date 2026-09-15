# Shared backbone with view-specialized Image-token Adapters

## Question

Test whether a small Full/Person/Face-specific parameter allocation can retain
view specialization inside a largely shared MULTI-LANE model. This is the
representation stage before any new dynamic Router is introduced.

The current independent fixed-R1 reference is `43.5812` seed0 final validation
mAP. The same-commit joint shared control is trained again because this stage
adds denser gradient diagnostics and a parameter-matched capacity control.

## Shared protocol

- EMOTIC B5-C3, seed0, all eight tasks, validation only.
- Frozen OpenAI CLIP ViT-B/16; shared task Selectors, prompts and linear head.
- Full legacy crop, Person margin0.15 letterbox, audited Face letterbox and the
  locked reliable-Face mask.
- Fixed feature weights `[0.64,0.16,0.20]` for reliable Face and
  `[0.80,0.20,0]` otherwise.
- Fused BCE plus `0.1` times the mean valid-view BCE trains shared main
  parameters. Fused ASL plus `0.1` times the valid-view ASL trains Image-token
  Adapter parameters; ASL is `gamma_neg=9.8`, `gamma_pos=0`, `clip=0.05`.
- 30 epochs/task, batch64, Adam reset per task, main LR0.0125, Adapter LR4e-4,
  per-task cosine to zero, no warmup or weight decay, AMP/TF32.
- Image-token Adapters operate at zero-based CLIP block1 with fixed residual
  scale0.1, ReLU and independent task initialization.
- No checkpoint, test access, score search or dynamic Router.

## Locked methods and parameter control

| Method | Adapter structure per task | Parameters/task |
| --- | --- | ---: |
| P0 | one shared b32 Adapter | 49,952 |
| P1 | one shared b45 Adapter | 69,933 |
| P2 | shared b32 + Full/Person/Face b4 Adapters | 70,700 |

For P2 and view `v`, the selected image tokens are produced from

```text
frozen_tokens + 0.1 * (shared_b32(frozen_tokens) + view_b4_v(frozen_tokens))
```

All four residual outputs are zero-initialized. P1 differs from P2 by only
767 trainable Adapter parameters per task (`1.10%` of P2), so P1 controls for
the additional capacity without allocating view-specific functions.

## Gradient diagnostics

For every task, epochs `0`, `14` and `29` sample the first three training
batches. For each Full/Person/Face endpoint, the fused-loss gradient is first
computed with respect to that endpoint and then vector-Jacobian-producted back
through only that view graph. This separates paths that otherwise sum on the
shared parameters.

The report records, separately for shared Selector/Prompt parameters under BCE
and active Adapter parameters under ASL:

- fused-path and corresponding unimodal gradient norms;
- fused-to-unimodal norm ratios and cosine similarities;
- pairwise cosines among the three fused gradient paths.

Face objectives and diagnostics use the same reliable-Face mask as fusion.
Each completed task also records standalone view metrics, reliable-Face metrics,
fusion weights and Full-relative corrected/damaged ranking pairs.

## Selection

P2 advances to the separately implemented Selector-aware dynamic Router stage
only if all checks pass:

1. final validation mAP is at least P0 + 0.10;
2. final validation mAP is at least parameter-matched P1 + 0.10;
3. average validation mAP does not fall below P0;
4. standalone Full final mAP does not fall below P0;
5. the gap to independent fixed R1 `43.5812` becomes smaller than P0's gap.

Failure ends this exact b32+b4 specialization design. The diagnostics then
decide whether a different specialization location or gradient method has a
supported mechanism; they do not authorize post-hoc bottleneck or LR search.

## Entrypoints and outputs

- Single run: `scripts/emotic/run_multilane_track_a_view_specialized_adapter_val.sh`
- Three-GPU batch: `scripts/emotic/launch_multilane_track_a_view_specialized_adapter_val.sh`
- Summary: `python -m multi_lane.track_a.summarize_view_specialized_adapter`
- Server results: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_view_specialized_adapter_v0.1/`
- Repository control: `./output/emotic_track_a_view_specialized_adapter/`
- Repository logs: `./logs/emotic_track_a_view_specialized_adapter/`

The batch launcher waits until all three selected GPUs have at least 12,000 MiB
free and at most 10% utilization for two consecutive checks, 30 seconds apart.

## Formal run status

After the user confirmed that the observed per-GPU free memory was sufficient,
the formal batch `view_specialized_adapter_seed0_20260915_104759` was started on
GPUs 0/1/2 from clean commit `10ef477`. Only this invocation overrides the
launcher gate to 7,000 MiB free, 100% maximum utilization and one readiness
check; the registered model and training configuration is unchanged. P0/P1/P2
all completed epoch 0 with 84 optimizer updates, zero skipped updates, finite
losses and no OOM. The batch remains validation-only and checkpoint-free.

## Result

The batch completed normally. P0/P1/P2 final validation mAP is
`43.0754/42.7028/42.7961`, and average mAP is `49.9807/50.1309/49.7551`.
P2 is `0.2793` below P0 final and fails all five registered checks. Relative to
P0, its final Full/Person/Face/reliable-Face standalone mAP changes by
`-0.1549/-0.1337/-0.3033/-0.7226`; only task 0 improves, while tasks 1--7 are
all lower. Fixed fusion still improves over each method's Full view by
`1.27--1.48`, so complementary evidence remains but P2 does not create stronger
experts.

Late cross-view gradient cosines sometimes weaken, but P2 records fewer
negative cross-view cosines than P0 and still performs worse. The observed
fused-to-unimodal norm ratios mainly track the fixed weights, while same-view
gradient directions remain positive. This does not support DGL-style gradient
conflict as the primary bottleneck. P2 therefore does not advance to dynamic
routing, additional seeds or test. The synchronized local batch directory
contains the full `analysis.md` report.
