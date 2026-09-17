# View-private Prompt, Adapter and classifier validation

## Objective

Test whether the performance loss of view-specific Selectors is caused by keeping
the downstream Prompt, Image-token Adapter and classifier shared across Full,
Person and Face.

## Fixed protocol

- EMOTIC validation, seed 0, all 8 incremental tasks.
- 30 epochs per task, batch size 64, Adam reset per task, main LR 0.0125.
- Image-token Adapter layer 1, bottleneck 32, LR 4e-4, residual scale 0.1,
  ReLU, independent task initialization, Adapter ASL 9.8/0/0.05.
- Frozen CLIP ViT, view-specific Selector bank with 10 Selectors per view,
  fixed reliable-Face three-view weights 0.64/0.16/0.20 and 0.80/0.20/0.
- Joint gradient routing, view auxiliary loss 0.1, AMP/TF32 enabled.
- Validation only; per-view score dumps and diagnostics enabled; no test access
  and no full checkpoints.

## Arms

P0--P3 preserve feature-level fusion and the current post-fusion classifier.
H0--H3 use fixed per-view logit fusion so classifier sharing is tested with a
matched fusion-placement control.

| Arm | Prompt | Image-token Adapter | Classifier | Purpose |
| --- | --- | --- | --- | --- |
| P0 | shared | shared | shared post-fusion | S1 reproduction |
| P1 | view-specific | shared | shared post-fusion | Prompt effect |
| P2 | shared | independent per view | shared post-fusion | Adapter effect |
| P3 | view-specific | independent per view | shared post-fusion | Prompt + Adapter interaction |
| H0 | shared | shared | one shared head per view | Head/fusion placement control |
| H1 | shared | shared | independent Full/Person/Face heads | Head sharing effect |
| H2 | view-specific | independent per view | one shared head per view | All upstream private, shared head |
| H3 | view-specific | independent per view | independent Full/Person/Face heads | All-private candidate |

All private modules copy the corresponding shared initialization. The new
`adapter_view_mode=independent` uses only the view bank; the existing shared
bank is excluded from optimization and reporting in that mode. The new
`view_classifier_mode=private_per_view` keeps the Full head in the historical
`head.*` state names and adds exact Person/Face copies.

## Selection rules

- Screen each arm against P0 using fused final/average mAP, cF1/oF1,
  forgetting, Full/Person/Face/reliable-Face mAP, train/validation curves and
  image-group bootstrap from saved validation scores.
- An arm advances only if final mAP improves P0 by at least 0.50, average mAP
  does not fall by more than 0.25, and no view branch loses more than 0.50.
- To claim recovery of the shared-selector problem, H3 or a component winner
  must also reach the S0 shared-selector reference (42.7525 final mAP) within
  the pre-registered 0.10 margin. A private arm that only beats P0 is reported
  as partial mitigation.
- This seed-0 validation screen does not select test parameters. If a clear
  winner passes, repeat the locked configuration at seeds 1 and 2 before any
  test evaluation.

## Outcome

P0 reproduced S1 at 41.3856 final mAP. P2, which only replaces the shared
Image-token Adapter with independent Full/Person/Face banks, completed cleanly
at 42.2652 final and 48.6716 average mAP: `+0.8796/+0.9336` versus P0, with
all four final view diagnostics improved. Its paired original-image bootstrap
95% interval is `[+0.0403, +1.6639]`. It passes the P0 component screen but
remains 0.4873 below S0's 42.7525, so it is partial mitigation rather than
recovery.

P1 gains only 0.4580 final mAP. H0 and H1 show that per-view logit fusion and
private classification heads do not explain P2; H1 also lowers Full. P3 and
H2 terminate in task 3 with non-finite ASL logits, while H3 completes with 166
AMP-skipped updates. The all-private combination is numerically unstable under
this locked joint-gradient protocol and is excluded from selection. The full
report is in `output/emotic_track_a_view_private_components/view_private_components_seed0_20260917_102434/analysis.md`.
