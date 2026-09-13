# OOF cross-view teacher distillation of the Full lane

## Goal

Test whether leakage-free Person/Face predictions can improve the dominant Full expert when they
are used only as training supervision, rather than being injected into inference features or fitted
by a small validation Router.

## Fixed expert protocol

- Dataset: EMOTIC; seed0; complete eight-task validation only; test forbidden.
- Full input: legacy random crop, CLIP normalization, train scale `(0.05, 1.0)`.
- 30 epochs/task, batch64, Adam reset/task, main LR `0.0125`, cosine to zero, no warmup.
- Image-token Adapter: zero-based layer1, bottleneck32, LR `4e-4`, scale0.1, ReLU,
  independent task initialization.
- Main supervised objective: legacy full-zero BCE. Adapter objective: ASL `9.8/0/0.05`.
- AMP/TF32 enabled; no checkpoints; save all validation score dumps.

## Leakage-free teacher

Reuse `three_view_oof_seed0_20260910_160250`: three deterministic SHA-256 image-group folds,
where every teacher prediction comes from an expert trained on the other two folds. All people
from one source image remain in the same fold. The pooled predictions must cover every eligible
Full training sample exactly once for every task.

Only current-task new-class probabilities are used. Before training each candidate, sample IDs,
binary labels, task/class layout, source provenance and complete pool coverage are checked exactly.
Old task selectors, prompts, heads and Adapters retain their existing freeze semantics.

The Full base-parameter loss is

`0.80 * hard-label BCE + 0.20 * OOF soft-label BCE`.

Both terms use the same legacy full-zero view, so their weights sum to one and do not change the
overall objective scale. The Image-token Adapter continues to receive only hard-label ASL.

## Three candidates

- `D0`: fresh Full champion anchor, no teacher.
- `D1`: Person OOF probabilities are the teacher.
- `D2`: for reliable Face samples, teacher probabilities are
  `4/9 * Person + 5/9 * Face`, derived from the locked R1 auxiliary ratio `0.16:0.20`;
  invalid/unreliable Face samples fall back exactly to Person. No weight is searched.

All three start from seed0 in independent processes on the same Git commit and run in parallel.

## Selection rule

A distilled candidate advances only if, relative to the same-batch D0 anchor:

- Full final validation mAP improves by at least `0.05`;
- Full average validation mAP does not decrease;
- locked R1 final validation mAP improves by at least `0.05`.

Locked R1 remains `[0.64, 0.16, 0.20]` for reliable Face and `[0.80, 0.20, 0]` otherwise.
No test is launched by this stage. If neither D1 nor D2 passes, do not search a distillation-weight
grid without first diagnosing teacher calibration and per-class agreement.
