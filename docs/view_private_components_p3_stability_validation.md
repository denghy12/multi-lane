# P3 Prompt plus Adapter stability validation

This follow-up tests whether the P1 Prompt gain and P2 independent
Image-token Adapter gain add when both are enabled. It starts from the locked
P3 configuration on `exp/view-private-components-p3-stable`.

The first trial added `--gradient-clip-norm 1.0`, but the failure occurs during
AMP backpropagation before clipping can run. The actual stability control is
therefore FP32 (`--no-amp`) with gradient clipping set to zero, isolating the
precision change. Main LR 0.0125, Adapter LR 4e-4, ASL 9.8/0/0.05, joint
routing, auxiliary view loss 0.1, 30 epochs/task, batch 64, frozen CLIP,
view-specific Selector x10, and fixed reliable-Face fusion remain unchanged.
A zero clip value preserves the old optimizer behavior.

The run is validation-only on EMOTIC seed 0, all eight tasks, without test
access or full checkpoints. It uses the existing
`run_multilane_track_a_view_private_components_p3_stable_val.sh` entry point.
The run is considered numerically valid only if all 240 epochs complete with
13,950 successful optimizer updates, zero skipped updates, finite losses and
finite saved scores.

The additive-effect screen compares the stabilized P3 against P0, P1 and P2:
final mAP must exceed P0 by 0.50, average mAP may not fall by 0.25, and no
final Full/Person/Face/reliable-Face branch may lose 0.50. Reaching S0's
42.7525 final mAP within 0.10 remains the separate recovery criterion.
