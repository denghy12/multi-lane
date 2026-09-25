# ParaX P-post-tiny results — 2026-09-25

## Result

Batch `parax_post_tiny_control_20260925_010500` completed seed0 task0--2 validation for paired B0 and P-post-tiny. Both runs completed 90 epochs and 5010 optimizer updates without skipped steps, OOM, NaN, checkpoint saving, or test evaluation.

| method | scale | final mAP | average mAP | forgetting | final cF1 | final oF1 |
|---|---:|---:|---:|---:|---:|---:|
| B0-paired | 0 | 45.2395 | 54.9341 | 2.7760 | 36.8870 | 61.6314 |
| P-post-small | 0.001 | 44.2960 | 53.7270 | 2.5475 | 35.4220 | 61.1974 |
| P-post-static | 0.001 | 43.9191 | 52.8834 | 2.2440 | 36.6518 | 61.9663 |
| P-post-tiny | 0.0001 | 44.7075 | 54.1956 | 2.3772 | 35.3581 | 60.9149 |

P-post-tiny is the strongest ParaX result so far: it recovers `+0.4115` final mAP over P-post-small and `+0.7884` over P-post-static. It is still `-0.5319` below B0. Task0/task1/task2 mAP is `59.5844/58.2948/44.7075`, compared with B0 `60.5225/59.0402/45.2395`.

## Diagnostics

The final measured residual/token ratios are only about `1.7e-7` for all three views. Gate entropy remains near uniform (`1.094`) and Full/Person/Face gate L1 separation is only `0.009`. Thus scale `0.0001` makes the effective ParaX transformation nearly identity and also makes the router almost uninformative.

At task2, P-post-tiny Full/Person/Face/reliable-Face mAP is `43.0715/42.3408/34.7474/39.2184`. Face improves over B0 by `+0.7680`, but Full, Person, and reliable Face remain below B0 by `-0.6343/-0.2475/-0.2505`; the fixed fusion therefore remains `-0.5319`.

The remaining gap cannot be attributed simply to a large residual. Because ParaX is active but produces an almost zero output, the next required experiment is a full-training strict identity control: keep the ParaX module and optimizer group present while fixing output scale to exactly zero. This distinguishes a real tiny-residual effect from CUDA/optimizer/random-trajectory differences caused by adding the module.

## Decision

Run paired B0 versus `P-post-zero`, identical to P-post-tiny except fixed output scale `0`. If `P-post-zero` is numerically aligned with B0, the tiny residual itself causes the remaining loss and the ParaX image-stream route should stop. If it remains separated from B0, comparability is still incomplete and must be repaired before interpreting further routing experiments. Stage-4 distillation and stage-5 level routing remain paused.
