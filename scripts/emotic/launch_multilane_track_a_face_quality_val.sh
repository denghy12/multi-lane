#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
BATCH_ID="${BATCH_ID:-face_quality_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ')" -eq 3 ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }

RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_quality_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_face_quality/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_face_quality/${BATCH_ID}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
ANCHOR_FACE="${ANCHOR_FACE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch output exists" >&2; exit 2; }
for path in "${FULL_RUN}" "${PERSON_RUN}" "${ANCHOR_FACE}"; do
  [[ -f "${path}/seed_summary.json" ]] || { echo "Missing source run: ${path}" >&2; exit 2; }
done

mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
printf 'variant\tgpu\tmargin\tmin_short_side\tmin_score\tjitter\trun\n' > "${CONTROL_DIR}/manifest.tsv"
variants=(reliable_m15 reliable_m05 reliable_m05_jitter)
margins=(0.15 0.05 0.05)
jitters=(none none strength0.1_probability0.5)
for index in 0 1 2; do
  run_id="${BATCH_ID}_${variants[$index]}"
  printf '%s\t%s\t%s\t24\t0.6\t%s\t%s\n' "${variants[$index]}" "${gpus[$index]}" "${margins[$index]}" "${jitters[$index]}" "${RESULT_BASE}/${run_id}" >> "${CONTROL_DIR}/manifest.tsv"
done

for gpu in "${gpus[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  [[ "${free_mib}" -ge 8000 ]] || { echo "GPU ${gpu} has only ${free_mib} MiB free" >&2; exit 2; }
done

pids=()
for index in 0 1 2; do
  variant="${variants[$index]}"
  run_id="${BATCH_ID}_${variant}"
  (
    GPU="${gpus[$index]}" VARIANT="${variant}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
      bash scripts/emotic/run_multilane_track_a_face_quality_candidate_val.sh
    printf '0\n' > "${CONTROL_DIR}/status/${variant}.exit_code"
  ) > "${LOG_DIR}/${variant}.launcher.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${variants[$index]}.exit_code"
    failed=1
  fi
done
[[ ${failed} -eq 0 ]] || { echo "FACE_QUALITY_BATCH_FAILED" >&2; exit 1; }

"${PYTHON}" -m multi_lane.track_a.compare_face_expert_quality \
  --full-run "${FULL_RUN}" --person-run "${PERSON_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --anchor "baseline_m15=${ANCHOR_FACE}" \
  --candidate "reliable_m15=${RESULT_BASE}/${BATCH_ID}_reliable_m15" \
  --candidate "reliable_m05=${RESULT_BASE}/${BATCH_ID}_reliable_m05" \
  --candidate "reliable_m05_jitter=${RESULT_BASE}/${BATCH_ID}_reliable_m05_jitter" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "FACE_QUALITY_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
