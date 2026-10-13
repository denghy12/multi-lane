#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_PREFIX="${BATCH_PREFIX:?Set a unique BATCH_PREFIX}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
FACE_PYTHON="${FACE_PYTHON:-/opt/conda/envs/cocoer-preprocess/bin/python}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC_PAMI}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-../baseline_sources/cocoer_insightface/models/buffalo_l/det_10g.onnx}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_pami_face_manifest/${BATCH_PREFIX}}"
RESULT_ROOT="${RESULT_ROOT:-./output/emotic_pami_joint26}"
LOG_ROOT="${LOG_ROOT:-./logs/emotic_pami_joint26}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-30}"
SMOKE_UPDATES="${SMOKE_UPDATES:-}"
METHODS=(FULL_ONLY THREE_VIEW THREE_VIEW_PARAX)
RUN_REPORTING_SPLIT=test
if [[ -n "${SMOKE_UPDATES}" ]]; then RUN_REPORTING_SPLIT=val; fi

[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the training worktree" >&2; exit 2;
}
for path in "${DATA_ROOT}/Annotations/Annotations.mat" "${CLIP_CHECKPOINT}" "${DETECTOR_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "Invalid EPOCHS=${EPOCHS}" >&2; exit 2; }
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"
MANIFEST_PATH="${RESULT_ROOT}/${BATCH_PREFIX}_experiment_manifest.txt"
[[ ! -e "${MANIFEST_PATH}" ]] || { echo "Batch manifest already exists: ${MANIFEST_PATH}" >&2; exit 2; }
cat > "${MANIFEST_PATH}" <<EOF
batch_prefix=${BATCH_PREFIX}
branch=$(git branch --show-current)
commit=$(git rev-parse HEAD)
dataset_release=PAMI; official Annotations.mat train/val/test; final-epoch test only in formal runs, no validation selection
annotation_sha256=$(sha256sum "${DATA_ROOT}/Annotations/Annotations.mat" | cut -d' ' -f1)
methods=${METHODS[*]}; seeds=${SEEDS}; GPU=${GPU}; reporting_split=${RUN_REPORTING_SPLIT}; smoke_updates=${SMOKE_UPDATES:-none}
model=frozen OpenAI CLIP ViT-B/16; one shared encoder, Selector, Prompt, classifier; image-token Adapter disabled
ParaX=only THREE_VIEW_PARAX; Frozen Forward after block index 10; rank32; 3 shared experts; router hidden16; official init; learnable output scale0.1
fusion=Full/Person/Face fixed reliable [0.64,0.16,0.20] else [0.80,0.20,0]
training=all26 joint, one Task Forward, final epoch; epochs=${EPOCHS}; batch64; Adam main_lr0.0125 ParaX_lr0.0004; weight_decay0; cosine no warmup; AMP initial1024 growth1000000000
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${DETECTOR_CHECKPOINT}
face_manifest=${FACE_MANIFEST_ROOT}; detector=SCRFD det_10g CPU 640 threshold0.5 margin0.15 ambiguity_gap0.08
outputs=${RESULT_ROOT}; logs=${LOG_ROOT}
EOF

if [[ ! -f "${FACE_MANIFEST_ROOT}/audit_summary.json" ]]; then
  mkdir -p "${FACE_MANIFEST_ROOT}"
  "${FACE_PYTHON}" -m multi_lane.track_a.face_manifest \
    --data-root "${DATA_ROOT}" --dataset-release pami \
    --detector-checkpoint "${DETECTOR_CHECKPOINT}" \
    --output-root "${FACE_MANIFEST_ROOT}" --splits train val test \
    --det-size 640 --det-threshold 0.5 --face-margin 0.15 --ambiguity-gap 0.08 \
    --visual-samples 200 --resume \
    2>&1 | tee "${LOG_ROOT}/${BATCH_PREFIX}_face_manifest.log"
fi
for split in train val test; do
  [[ -s "${FACE_MANIFEST_ROOT}/manifests/${split}.jsonl" ]] || {
    echo "Missing PAMI Face manifest: ${split}" >&2; exit 2;
  }
done

export AMP_INITIAL_SCALE=1024 AMP_GROWTH_INTERVAL=1000000000
for seed in ${SEEDS}; do
  [[ "${seed}" =~ ^[012]$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
  BATCH_ID="${BATCH_PREFIX}_seed${seed}"
  for method in "${METHODS[@]}"; do
    [[ ! -e "${RESULT_ROOT}/${BATCH_ID}/${method}" ]] || {
      echo "Training output already exists: ${BATCH_ID}/${method}" >&2; exit 2;
    }
  done
  while (( $(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') < 20000 )); do
    echo "Waiting for GPU${GPU} to have 20000 MiB free before seed${seed}"; sleep 60
  done
  pids=()
  for method in "${METHODS[@]}"; do
    (
      METHOD="${method}" GPU="${GPU}" RUN_ID="${BATCH_ID}" PYTHON="${PYTHON}" \
        DATA_ROOT="${DATA_ROOT}" DATASET_RELEASE=pami CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
        FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" RESULT_ROOT="${RESULT_ROOT}" \
        LOG_ROOT="${LOG_ROOT}" SEED="${seed}" EPOCHS="${EPOCHS}" REPORTING_SPLIT="${RUN_REPORTING_SPLIT}" \
        SMOKE_UPDATES="${SMOKE_UPDATES}" \
        bash scripts/emotic/run_joint26_parax_view_ablation.sh \
        > "${LOG_ROOT}/${BATCH_ID}_${method}.launcher.log" 2>&1
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "Training failed for ${BATCH_ID}" >&2; exit 1; }
  if [[ -n "${SMOKE_UPDATES}" ]]; then
    "${PYTHON}" - "${RESULT_ROOT}/${BATCH_ID}" "${SMOKE_UPDATES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
updates = int(sys.argv[2])
reference = None
for method in ("FULL_ONLY", "THREE_VIEW", "THREE_VIEW_PARAX"):
    run = root / method
    summary = json.loads((run / "seed_summary.json").read_text())
    audit = json.loads((run / "joint_protocol_audit.json").read_text())
    history = json.loads((run / "training_history.json").read_text())["0"]
    assert summary["status"] == "complete" and summary["completed_optimizer_updates"] == updates, method
    assert sum(row["skipped_optimizer_steps"] for row in history) == 0, method
    assert summary["config"]["dataset_release"] == "pami", method
    assert summary["config"]["reporting_split"] == "val" and not audit["test_loaded"], method
    assert audit["frozen_visual_unchanged"], method
    pair = (audit["train_sample_ids_sha256"], audit["first_train_batch_ids_sha256"],
            audit["initial_selector_prompt_classifier_sha256"])
    if reference is None:
        reference = pair
    assert pair == reference, method
print("PAMI_JOINT26_SMOKE_AUDIT_PASSED", flush=True)
PY
    echo "Smoke completed for ${BATCH_ID}; skipping comparison and test"
    continue
  fi
  "${PYTHON}" -m multi_lane.track_a.compare_joint26 \
    --batch-root "${RESULT_ROOT}/${BATCH_ID}" --methods "${METHODS[@]}" --split test \
    2>&1 | tee "${LOG_ROOT}/${BATCH_ID}_comparison.log"
done
echo "PAMI_JOINT26_BATCH_COMPLETE batch=${BATCH_PREFIX}"
