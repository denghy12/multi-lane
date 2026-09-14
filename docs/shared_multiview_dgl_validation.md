# Shared multi-view DGL gradient audit and seed0 validation

## Question

Test whether the gap between independently trained endpoint fusion and shared
Full/Person/Face feature fusion is caused in part by fusion gradients suppressing
the shared MULTI-LANE representation.  This stage adapts ICCV 2025 Disentangled
Gradient Learning to correlated views from one RGB image; it does not claim that
Full/Person/Face are independent sensor modalities.

The existing references are independent fixed R1 seed0 validation `43.5812`,
joint J0 `42.5330`, soft J1 `39.8569`, and soft Full/Person J3 `38.3318`.

## Locked shared architecture and training protocol

- Dataset: EMOTIC, frozen B5/C3 order, seed0, all eight tasks, validation only.
- Frozen OpenAI CLIP ViT-B/16; Full/Person/Face share selectors, prompts, head,
  and the active Image-token Adapter.
- Full uses legacy crop; Person uses bbox margin0.15 letterbox; Face uses the
  audited manifest letterbox and the existing reliable mask.
- Fixed feature weights are `[0.64,0.16,0.20]` for reliable Face and
  `[0.80,0.20,0]` otherwise.
- 30 epochs/task, batch64, Adam reset per task, main LR0.0125, cosine to zero,
  no warmup or weight decay, AMP/TF32 enabled.
- Image-token Adapter: zero-based layer1, bottleneck32, LR4e-4, scale0.1,
  ReLU, independent task initialization.
- Shared main representation uses hard-label BCE; Adapter uses hard-label ASL
  `gamma_neg=9.8`, `gamma_pos=0`, `clip=0.05`.
- No checkpoints, no test access, and no dynamic Router in this stage.

## Methods

### G0: joint-gradient fresh J0

The fused hard-label objective plus `0.1` times the mean valid-view objective
uses normal joint backpropagation.  This is a same-commit fresh control.

### G1: multimodal/fusion-gradient truncation only

The Full/Person/Face features passed to fixed fusion are detached.  The fused
objective therefore cannot update the shared representation, while the legacy
`0.1` view auxiliary objectives retain their ordinary gradient path through the
shared head.  This isolates the first side of DGL.

### G2: full DGL-inspired bidirectional truncation

The fused inputs are detached.  Parameter-targeted autograd then enforces:

```text
mean(Full/Person/reliable-Face BCE) * alpha -> selectors/prompts only
mean(Full/Person/reliable-Face ASL) * alpha -> active Image-token Adapter only
fused hard-label BCE                         -> shared head/fusion only
```

`alpha=1.0`.  Unimodal objectives pass through the shared head to form useful
representation gradients but cannot update the head; fused BCE cannot update
the representation or Adapter.  Face loss is normalized only over reliable
samples, and invalid Face has exact zero fusion weight.

## Diagnostics

The first training batch of every task records a conceptual concatenation of
the BCE gradient on selectors/prompts and ASL gradient on the active Adapter:

- fused and Full/Person/Face gradient norms;
- fused-to-view cosine similarity;
- fused-to-view norm ratio.

Every completed task also records:

- fused and standalone Full/Person/Face validation metrics;
- reliable-Face metrics;
- per-lane weight mean/std/p05/p50/p95;
- positive-negative ranking pairs that fusion corrects or damages relative to
  standalone Full, including per-class counts.

G0 provides the causal conflict audit.  G1/G2 are also audited so zero fused
representation gradient is verified rather than assumed.

## Selection

G2 advances to a separately implemented constrained dynamic-Router stage only
if all conditions hold:

1. final validation mAP is at least G0 + 0.10;
2. average validation mAP does not decrease;
3. standalone Full final mAP does not decrease;
4. the absolute gap to independent fixed R1 `43.5812` shrinks.

Individual classes do not veto advancement.  Historical J0 is context only;
selection uses the fresh same-commit G0.  Test remains forbidden regardless of
the result.

## Entrypoints and outputs

- Single run: `scripts/emotic/run_multilane_track_a_shared_dgl_val.sh`
- Three-GPU batch: `scripts/emotic/launch_multilane_track_a_shared_dgl_val.sh`
- Summary: `python -m multi_lane.track_a.summarize_shared_dgl`
- Server results: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_dgl_v0.1/`
- Repository control/logs: `output/emotic_track_a_shared_dgl/` and
  `logs/emotic_track_a_shared_dgl/`
