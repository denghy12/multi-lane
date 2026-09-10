# Full-anchored complementary residual validation protocol

## Objective

Test whether Person and Face can improve EMOTIC incremental recognition as bounded
feature residuals without allowing an unconstrained softmax Router to suppress the
Full-image context path. This stage is seed0 complete eight-task validation only.
Test is forbidden.

## Architecture

For task lane `k`, the fused representation is:

```text
z_fused,k = z_full,k
          + c_person,k * P_person,k(stopgrad(z_person,k))
          + face_reliable * c_face,k * P_face,k(stopgrad(z_face,k))
```

Each `P_view,k` is a task-specific `LayerNorm(no affine) -> Linear(D,16) ->
GELU -> Linear(16,D)` residual Adapter. The final linear layer is initialized
to zero. Its output is divided by `1 + L2_norm`, so its norm remains below one.
The coefficient is `0.1 * sigmoid(gate_view,k)` and is therefore strictly in
`[0, 0.1]`. Full always has coefficient one. No branch competes through a
sum-to-one constraint.

Person and Face source features are generated without the Full Image-token Adapter
and detached before their view-specific residual Adapters. Consequently the shared
Selector/Prompt/Image-token Adapter pathway is updated only through Full; auxiliary
views cannot distort the champion Full path or reuse its view-specific Adapter.
Residual modules are trained only for the current task and frozen after that task.
Unreliable Face samples have an exact zero Face coefficient and residual.

No auxiliary branch classification loss is used in this stage. This is deliberate:
the previous J0/J1/J3 run showed that shared branch supervision and fusion gradients
reduced view specialization. The new experiment isolates whether small complementary
residuals themselves transfer useful information.

## Locked experiment matrix

| Method | Input and fusion |
|---|---|
| A0 | Fresh Full champion anchor; no multi-view module |
| A1 | Full + detached Person taskwise residual |
| A2 | Full + detached Person + reliable-Face taskwise residuals |

All methods use seed0, EMOTIC, 8 tasks, 30 epochs/task, batch64, main LR 0.0125,
per-task cosine annealing to zero without warmup, CLIP normalization, legacy Full
crop, Image-token Adapter at zero-based layer1 with bottleneck32/LR4e-4/scale0.1/
ReLU/independent initialization, main BCE plus Adapter ASL 9.8/0/0.05, AMP and TF32.
No checkpoints are saved.

Residual settings are fixed: bottleneck16, LR4e-4, maximum coefficient0.1 and
zero-initialized output. There is no residual-scale or gate grid.

## Decision rule

A2 advances to seed1/2 validation only when its final validation mAP exceeds both
A0 and A1. Average mAP is only a ranking tie-breaker. Test cannot affect this
decision and is not accessed in this stage. If A1 and A2 both fail to beat A0,
stop training-level fusion and retain the independently trained fixed R1 ensemble.

## Entrypoints and outputs

- Single run: `scripts/emotic/run_multilane_track_a_full_anchored_residual_val.sh`
- Three-GPU launcher: `scripts/emotic/launch_multilane_track_a_full_anchored_residual_val.sh`
- Summary: `python -m multi_lane.track_a.summarize_full_anchored_residual`
- Logs: `./logs/emotic_track_a_full_anchored_residual/`
- Control: `./output/emotic_track_a_full_anchored_residual/`
- Server results: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_full_anchored_residual_v0.1/`
