# ParaX P-post control results — 2026-09-25

## Protocol

Batch `parax_post_control_20260925_230104` used the same seed, data, task order, optimizer, scheduler, and three-view fusion for `B0-paired` and `P-post-small`. Both runs completed task 0–2 validation with 30 epochs per task, batch size 64, 5010 optimizer updates, no skipped steps, no checkpoint, and no test evaluation.

`P-post-small` kept the frozen CLIP and lane path unchanged and applied a strict zero-output ParaX branch to the final lane feature. It used rank 32, three experts, router hidden size 16, fixed output scale `0.001`, and froze the expert center after task 0.

## Main results

| method | final mAP | average mAP | forgetting | final cF1 | final oF1 |
|---|---:|---:|---:|---:|---:|
| B0-paired | 45.2395 | 54.9341 | 2.7760 | 36.8870 | 61.6314 |
| P-post-small | 44.2960 | 53.7270 | 2.5475 | 35.4220 | 61.1974 |
| P-post − B0 | -0.9434 | -1.2071 | -0.2285 | -1.4650 | -0.4340 |

The post-encoder placement is safer than the previous block-10 insertion for forgetting, but it does not recover accuracy. The loss is already present at task 0 (`59.1245` versus `60.5225` mAP), so old-task drift alone cannot explain the result.

| task | B0 mAP | P-post mAP | difference |
|---:|---:|---:|---:|
| 0 | 60.5225 | 59.1245 | -1.3981 |
| 1 | 59.0402 | 57.7604 | -1.2798 |
| 2 | 45.2395 | 44.2960 | -0.9434 |

At task 2, the fused and single-view mAP values were:

| view | B0 | P-post | difference |
|---|---:|---:|---:|
| fused | 45.2395 | 44.2960 | -0.9434 |
| Full | 43.7058 | 42.3548 | -1.3511 |
| Person | 42.5883 | 42.2803 | -0.3080 |
| Face | 33.9794 | 33.2613 | -0.7181 |
| reliable Face | 39.4689 | 38.0635 | -1.4054 |

The fixed fusion weights are identical between the two groups. The drop therefore comes from the transformed view features, especially Full and reliable Face, rather than from a changed fusion prior.

## Routing diagnostics

The final task-2 P-post residual/token ratios were approximately `0.035` (Full), `0.045` (Person), and `0.050` (Face). Thus the branch was small, but a 3–5% feature residual still changed the classifier-relevant coordinate system. The final gate entropy fell to `0.524`, `0.451`, and `0.439`; top-expert frequencies were dominated by expert 2 (`0.624`, `0.641`, `0.756`). Full/Person/Face gates were therefore not balanced level-specific routes. Their L1 separation was `0.177`, but it did not produce a single-view gain.

The strict zero-output smoke passed with an initial logit difference below `2e-8`, so the implementation starts aligned with B0. The degradation is learned during task-0 optimization, not an initialization or RNG mismatch. Because the branch is after the frozen encoder, this result also separates the location problem from the earlier middle-block problem: moving ParaX later reduces interference with CLIP blocks, but a free residual still changes the feature geometry in a way the existing shared classifier and fixed fusion do not exploit.

## Decision

Do not start CPU-cache distillation yet. Distillation can control later-task forgetting, but it cannot repair the task-0 loss. The next experiment is a parameter-matched static post-encoder control (`P-post-static`) with the same rank, experts, scale, center-freezing schedule, and placement, but uniform expert weights and no input-dependent router. If static and dynamic are similarly below B0, the issue is the post-feature residual/optimization interface rather than routing capacity. If static is better, the dynamic router is the source of harmful feature movement.

The static control remains validation-only on task 0–2. If both controls fail, stop expanding ParaX inside the image stream and compare a strict identity post-feature adapter and a view-specific post-feature expert with a constrained residual. Only after a post-feature variant reaches B0 should Full-only CPU-cache distillation be tested.
