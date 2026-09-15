#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-p0_r1_gap_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
READY_CHECKS="${READY_CHECKS:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_p0_r1_gap/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_p0_r1_gap/${BATCH_ID}"
RESULT_BASE="${CONTROL_DIR}/runs"
RUN_ID="${BATCH_ID}_P0_seed0_val"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
FACE_RUN="${FACE_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"

[[ -z "$(git status --porcelain)" ]] || { echo "Diagnostic requires a clean Git worktree" >&2; exit 2; }
for run in "${FULL_RUN}" "${PERSON_RUN}" "${FACE_RUN}"; do
  [[ -f "${run}/seed_summary.json" ]] || { echo "Missing independent source: ${run}" >&2; exit 2; }
done
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
scope=one P0 reproduction plus offline Experiment A and Experiment B only
selection=seed0 complete 8-task validation; test forbidden; no hyperparameter search
P0=shared Selector/Prompt/head/Image-token Adapter b32; exact historical training protocol
A=same endpoint predictions, fixed reliable weights, logit versus probability fusion
B=fixed probability fusion, all 8 I/S source combinations in Full/Person/Face order
uncertainty=2000 paired original-image-group bootstrap replicates, seed 20260915
outputs=fused/view FP32 logits and probabilities, labels, IDs, Face mask, compact task states, complete JSON analysis
EOF

consecutive=0
while (( consecutive < READY_CHECKS )); do
  snapshot="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r free_mib utilization <<< "${snapshot}"
  free_mib="${free_mib// /}"
  utilization="${utilization// /}"
  echo "GPU ${GPU} free_mib=${free_mib} utilization=${utilization}"
  if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  if (( consecutive < READY_CHECKS )); then sleep "${WAIT_SECONDS}"; fi
done

GPU="${GPU}" RUN_ID="${RUN_ID}" OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
  FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
  bash scripts/emotic/run_multilane_track_a_p0_gap_reproduction_val.sh \
  > "${LOG_DIR}/P0.launcher.log" 2>&1
printf '0\n' > "${CONTROL_DIR}/status/P0.exit_code"

"${PYTHON}" -m multi_lane.track_a.p0_r1_gap_diagnostic \
  --shared-run "${RESULT_BASE}/${RUN_ID}" \
  --independent-full-run "${FULL_RUN}" \
  --independent-person-run "${PERSON_RUN}" \
  --independent-face-run "${FACE_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --bootstrap-replicates 2000 \
  --output "${CONTROL_DIR}/p0_r1_gap_diagnostic.json" \
  2>&1 | tee "${LOG_DIR}/diagnostic.log"
printf '0\n' > "${CONTROL_DIR}/status/diagnostic.exit_code"
echo "P0_R1_GAP_BATCH_COMPLETE batch=${BATCH_ID} output=${CONTROL_DIR}/p0_r1_gap_diagnostic.json"
