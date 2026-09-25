# Shared Adapter residual-scale held-out test

## Face reliability

The fixed fusion protocol treats a Face crop as reliable only when its manifest record has a valid face and an unambiguous body/person match, the detected box has a short side of at least 24 pixels, and the detector score is at least 0.6. Reliable rows use fixed Full/Person/Face weights `[0.64, 0.16, 0.20]`; other rows set the Face weight to zero and use `[0.80, 0.20, 0]`. This is a predeclared data-quality proxy, not a model confidence estimate. Report coverage and reliable-Face-only metrics beside all-Face metrics; do not tune the cutoff against test labels.

Implementation references: `multi_lane/continual_datasets/continual_datasets.py` computes the sample mask, `multi_lane/track_a/three_view_router.py` loads manifest provenance, and `multi_lane/track_a/view_fusion.py` applies the fixed priors.

## Validation gate and interpretation

Batch `shared_capacity_residual_scale_validation_20260925_222440` completed on branch `exp/shared-capacity-residual-control`, commit `ce21fa4`. Both seed-0 arms completed tasks 0–7, 30 epochs per task, with 0 skipped AMP updates:

| Arm | Final mAP | Mean task mAP | Task mAP, 0–7 |
|---|---:|---:|---|
| A0 shared b32, scale 0.03 | 42.3199 | 48.9646 | 58.1677, 57.4367, 44.2557, 49.0979, 48.5295, 46.9539, 44.9552, 42.3199 |
| A-cap shared b97, scale 0.03 | 42.5853 | 50.2722 | 61.4543, 59.3343, 45.7307, 50.4293, 49.7520, 47.6882, 45.2039, 42.5853 |

b97 gains `+0.2654` final mAP and `+1.3077` mean task mAP. It clears the numerical-stability concern at scale 0.03, but it does not meet the previously registered final-mAP promotion threshold of `+0.5`. The user requested a test, so the test is a paired comparison of both already-declared capacity arms; test results must not be used for further tuning or to claim that the original promotion threshold passed.

## Locked test run

- Batch: `shared_capacity_residual_scale_test_20260925_160954`.
- Code: server worktree `/mnt/haoyuan/workspace/multi-lane-main-shared-capacity-residual-control`, commit `ce21fa4`.
- Dataset: EMOTIC Track A. Train on `train`; report on held-out `test` after each incremental task.
- Arms: A0 shared Image-token Adapter b32 and A-cap shared b97; both inserted after block 1 with fixed residual scale 0.03. CLIP ViT-B/16 frozen; Selector, Prompt, classifier, and fixed three-view fusion shared.
- Seed 0; tasks 0–7; 30 epochs/task; batch 64; Adam reset per task; main LR 0.0125; Adapter LR 0.0004; cosine to zero, no warmup; shared BCE and Adapter ASL `(9.8, 0, 0.05)`; auxiliary view loss 0.1; AMP/TF32; threshold 0.5.
- Face manifest: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909`.
- Test manifest coverage (5,368 person samples): 4,677 valid detections (87.13%), 4,179 valid and unambiguous matches (77.85%), and 3,261 records pass all four reliability checks (60.75%). This is coverage of the predeclared rule, not proof that Face improves emotion prediction.
- Logs: `/mnt/haoyuan/workspace/multi-lane-main-shared-capacity-residual-control/logs/emotic_track_a_shared_capacity_residual_scale/shared_capacity_residual_scale_test_20260925_160954/`.
- Results: `/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_capacity_residual_scale_test_v0.1/shared_capacity_residual_scale_test_20260925_160954/`.
- No checkpoint was saved by validation. Therefore each test arm trains from scratch on train and then evaluates test; it does not reuse validation weights. Test labels are evaluation-only, with no threshold or fusion search.
- Launched in tmux session `ml_captest_160954` on GPUs 2 and 3. Do not launch a duplicate. The paired job was confirmed entering task 0; wait for the completion marker before analyzing test output.
