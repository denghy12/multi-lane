# Shared Selector with independent view Adapter control

## Goal

Test the narrowest remaining shared-encoder hypothesis: frozen CLIP features may
be reusable across Full, Person, and Face, while the small task-local token
adaptation immediately before the shared Selector must differ by view.

This experiment is not dynamic routing. It asks whether explicit view experts
can first be formed without changing the frozen encoder, Selector, Prompt,
classifier, or fusion rule. Dynamic routing is considered only after the view
experts themselves improve the paired control.

## Locked methods

| arm | Image-token Adapter/task | parameters/task | purpose |
|---|---|---:|---|
| A0-shared-b32 | one shared b32 | 49,952 | strict baseline |
| A-cap-shared-b97 | one shared b97 | 149,857 | matched capacity control |
| A-view-independent-b32 | Full/Person/Face each use an independent b32 | 149,856 | view specialization |

All three operate at zero-based CLIP block 1, use residual scale `0.1`, ReLU,
and independent task initialization. In `A-view`, the three banks start from
identical weights; specialization arises only from their view-specific training
gradients. Each old task bank freezes when the next task starts.

This differs from prior negative experiments:

- the shared-Selector b32+b4 experiment kept a shared b32 center and added only
  very small per-view deltas;
- the full independent-b32 result used view-specific Selectors and therefore
  could not isolate Adapter specialization under the successful shared Selector;
- ParaX modified frozen CLIP patch/lane streams and added routing, whereas this
  experiment reuses the established Image-token Adapter interface.

## Shared protocol

- Dataset: EMOTIC Track A, aligned Full/Person/Face views.
- Seed: 0; tasks 0--2 only for the first screen.
- Backbone: frozen OpenAI CLIP ViT-B/16.
- Shared trainable components: Selector, Prompt, classifier.
- Fusion: fixed reliable-Face weights `0.64/0.16/0.20`, fallback
  `0.80/0.20/0`.
- Loss: fused BCE for shared parameters; Image-token Adapter ASL
  `gamma_neg=9.8`, `gamma_pos=0`, `clip=0.05`; valid-view auxiliary weight
  `0.1`; joint view gradients.
- Training: 30 epochs/task, batch 64, Adam reset/task, main LR `0.0125`,
  Adapter LR `4e-4`, cosine to zero, no warmup, no weight decay.
- Inputs: legacy Full crop, Person margin-0.15 letterbox, audited reliable Face
  letterbox; CLIP normalization.
- Precision: AMP and TF32.
- Reporting: validation only; no checkpoint and no test.

## Required diagnostics

- initial shared/independent Adapter equality and parameter counts;
- 90 epochs, 5,010 optimizer updates, zero skipped;
- final/average/task0--2 mAP, cF1/oF1, and forgetting;
- Full/Person/Face/reliable-Face mAP;
- Full-relative corrected/damaged/net ranking pairs;
- per-view Adapter residual ratio or feature cosine;
- current-task Adapter gradient finite and nonzero;
- frozen CLIP gradients and weight changes remain zero.

## Decision rule

`A-view` advances to a full eight-task validation only if all conditions hold:

1. task0 mAP is at least `A0`;
2. task2 final mAP is at least `A0 + 0.10`;
3. task2 final mAP is at least `A-cap + 0.10`;
4. average mAP is not below `A0`;
5. at least two of Full/Person/Face final mAP do not fall, and no view falls by
   more than `0.25`;
6. forgetting does not increase by more than `0.10`.

Failure stops this exact independent-b32 shared-Selector route. Do not add a
Router, level embedding, larger bottleneck, more Adapter layers, seed1/2, or
test after failure.

## Entrypoints and outputs

- single run: `scripts/emotic/run_multilane_track_a_shared_selector_independent_adapter.sh`
- three-GPU launcher: `scripts/emotic/launch_multilane_track_a_shared_selector_independent_adapter.sh`
- server results: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_selector_independent_adapter_v0.1/`
- logs: `./logs/emotic_track_a_shared_selector_independent_adapter/`
- control/output: `./output/emotic_track_a_shared_selector_independent_adapter/`

## Smoke status

Server clean commit `284a34d` passed 245/245 unit tests. Three real EMOTIC
task-0, one-epoch smoke runs completed with 84 optimizer updates and zero
skipped steps:

| arm | parameters/task | validation mAP | residual ratio Full/Person/Face |
|---|---:|---:|---:|
| A0-shared-b32 | 49,952 | 39.8053 | 0.00326 / 0.00517 / 0.00797 |
| A-cap-shared-b97 | 149,857 | 40.7147 | 0.02404 / 0.03883 / 0.05902 |
| A-view-independent-b32 | 149,856 | 39.8119 | 0.00168 / 0.00328 / 0.00677 |

All Adapter gradients were finite and nonzero. A-view Full/Person/Face gradient
norms were approximately `9.96e-5/5.61e-5/8.13e-5`. These short-run metrics are
engineering diagnostics only and do not select an arm.
