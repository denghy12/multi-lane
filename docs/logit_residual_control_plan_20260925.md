# B0-preserving logit residual calibration control

## Motivation

The completed `post_calibration_control_20260925_134822` batch does not
support the feature residual candidate.  Relative to its paired control,
`V-post-residual` changed final/average validation mAP by
`-0.0263/-0.2699`, task-0 mAP by `-0.6981`, and final Full/Person/Face mAP
by `-1.0351/-0.2047/-0.9411`.  Its lower forgetting (`-0.0222`) is too small
to explain or offset the immediate task-0 loss.  The residual also corrected
more Full ranking errors while damaging still more previously correct pairs,
reducing net corrected pairs from `73,414` to `54,763` at task 2.

The batch also exposed a protocol drift: auxiliary view loss was `0`, whereas
the strict ParaX paired B0 used `0.1`.  This was the only material config
difference and moved B0 final mAP from `45.2395` to `43.4145`.  The completed
batch is therefore useful as an internal paired comparison, but it cannot be
used to replace the established B0.

## Mechanism

The next candidate retains ordinary fixed three-view B0 training and adds a
task-local classwise residual at the logit endpoint:

`z = z_B0 + a_person * (z_person - z_B0) + mask_face * a_face * (z_face - z_B0)`

where `a_person` and `a_face` are classwise values bounded to `[-0.1, 0.1]`
through `0.1 * tanh(raw)`.  Raw parameters start at zero.  Each task therefore
adds only `2 * 26 = 52` parameters and starts exactly at B0.

The shared representation and classifier receive the original B0 fused loss
plus auxiliary view loss `0.1`.  The residual coefficients receive a separate
calibrated BCE computed from detached B0 and per-view logits.  This prevents
the candidate from redirecting Selector, Prompt, or classifier gradients.
Old task residual coefficients freeze after their task.

## Locked experiment

- Dataset: EMOTIC Track A, aligned Full/Person/Face inputs.
- Backbone: frozen OpenAI CLIP ViT-B/16.
- Shared trainable path: Selector, Prompt, classifier; Adapter and ParaX off.
- Arms: `B0-paired` fixed three-view and `L-post-logit`.
- Seed: 0.
- Scope: tasks 0--2, 30 epochs/task, batch size 64.
- Optimizer: Adam reset per task; main LR `0.0125`; residual LR `4e-4`; cosine.
- Loss: legacy joint BCE, view auxiliary weight `0.1`.
- Precision: AMP and TF32.
- Reporting: validation only, no checkpoint, no test.

## Checks before launch

1. Full server unit suite passes in the `ddp` environment.
2. Zero-initialized candidate logits equal fixed B0 logits.
3. Base and calibration gradient groups are disjoint and finite.
4. Real EMOTIC task-0 smoke completes with zero skipped updates.
5. The server uses a clean Git-only worktree and leaves
   `multi-lane-main-test-only` unchanged.

## Decision rule

Advance only if `L-post-logit` is at least tied with paired B0 on task-0 and
average mAP, improves final mAP by at least `0.10`, does not increase
forgetting by more than `0.10`, and coefficients do not all remain effectively
zero or saturate at the bound.  Otherwise stop the current residual/calibration
route; do not add level embeddings, ParaX experts, distillation, seed 1/2, or
test evaluation.
