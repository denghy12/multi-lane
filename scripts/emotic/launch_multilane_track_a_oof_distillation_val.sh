#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-2500}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
OOF_ROOT="${OOF_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_oof_v0.1/three_view_oof_seed0_20260910_160250/oof_sources}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
FACE_RUN="${FACE_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
BATCH_ID="${BATCH_ID:-oof_distillation_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_oof_distillation_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_oof_distillation/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_oof_distillation/${BATCH_ID}"

[[ ${#GPU_LIST[@]} -ge 3 ]] || { echo "Three GPUs are required for D0/D1/D2" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
for source in "${OOF_ROOT}" "${FACE_MANIFEST_ROOT}" "${PERSON_RUN}" "${FACE_RUN}"; do
  [[ -e "${source}" ]] || { echo "Missing source: ${source}" >&2; exit 2; }
done
mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
printf 'variant\tgpu\tseed\trun\ttest_access\n' > "${CONTROL_DIR}/manifest.tsv"

variants=(D0 D1 D2)
pids=()
for index in 0 1 2; do
  variant="${variants[${index}]}"
  gpu="${GPU_LIST[${index}]}"
  while true; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib >= MIN_FREE_MIB )); then break; fi
    echo "Waiting for GPU${gpu}: free=${free_mib}MiB required=${MIN_FREE_MIB}MiB"
    sleep "${WAIT_SECONDS}"
  done
  run_id="${BATCH_ID}_${variant}"
  printf '%s\t%s\t0\t%s\tforbidden\n' "${variant}" "${gpu}" "${RESULT_BASE}/${run_id}" >> "${CONTROL_DIR}/manifest.tsv"
  GPU="${gpu}" VARIANT="${variant}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
    DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    OOF_ROOT="${OOF_ROOT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
    bash scripts/emotic/run_multilane_track_a_oof_distillation_val.sh \
    > "${LOG_DIR}/${run_id}.launcher.log" 2>&1 &
  pids+=("$!")
  echo "Started ${variant} on GPU${gpu} pid=${pids[${index}]}"
done

failed=0
for index in 0 1 2; do
  if wait "${pids[${index}]}"; then rc=0; else rc=$?; failed=1; fi
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/${variants[${index}]}.exit_code"
done
(( failed == 0 )) || { echo "At least one OOF distillation run failed" >&2; exit 1; }

"${PYTHON}" -m multi_lane.track_a.compare_oof_distillation_validation \
  --anchor "D0=${RESULT_BASE}/${BATCH_ID}_D0" \
  --candidate "D1=${RESULT_BASE}/${BATCH_ID}_D1" \
  --candidate "D2=${RESULT_BASE}/${BATCH_ID}_D2" \
  --person-run "${PERSON_RUN}" --face-run "${FACE_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/comparison.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "OOF_DISTILLATION_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
