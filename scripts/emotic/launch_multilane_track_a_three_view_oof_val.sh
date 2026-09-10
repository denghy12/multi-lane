#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
FACE_RUN="${FACE_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
BATCH_ID="${BATCH_ID:-three_view_oof_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_oof_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_three_view_oof/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_three_view_oof/${BATCH_ID}"
OOF_ROOT="${RESULT_BASE}/oof_sources"
DESCRIPTOR_ROOT="${RESULT_BASE}/descriptors"
SELECTION_DIR="${CONTROL_DIR}/router_selection"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

[[ ${#GPU_LIST[@]} -ge 1 ]] || { echo "At least one GPU is required" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
for run in "${FULL_RUN}" "${PERSON_RUN}" "${FACE_RUN}"; do
  [[ -f "${run}/seed_summary.json" ]] || { echo "Missing full-train validation endpoint: ${run}" >&2; exit 2; }
done
mkdir -p "${OOF_ROOT}" "${CONTROL_DIR}/status" "${LOG_DIR}"

cat > "${CONTROL_DIR}/protocol.txt" <<EOF
Selection only: seed0 EMOTIC complete 8-task validation; test access forbidden.
Three deterministic image-group folds. Each of nine Full/Person/Face sources trains on two folds
and exports predictions only for the unseen fold; pooled OOF predictions cover the train pool.
Expert protocol: 30 epochs/task, batch64, cosine min0 no warmup, main LR0.0125,
Image-token Adapter layer1/b32/LR4e-4/scale0.1/ReLU/independent,
main BCE + Adapter ASL 9.8/0/0.05, CLIP normalization, AMP/TF32 on.
Router: shared hidden16 reliability trunk plus three-value task bias, task-progressive snapshots,
masked Face, AdamW1e-3/wd1e-4, 80 epochs/task, priors 0/0.1/1.
Validation compares fixed R1 against OOF-R2 and OOF-R3 on the same complete-fit endpoints.
Advance only if R3 exceeds both R1 and R2 in final validation mAP; no test is launched here.
EOF
printf 'fold\tview\trun\tgit_head\ttest_access\n' > "${CONTROL_DIR}/manifest.tsv"
for fold in 0 1 2; do
  for view in full person face; do
    printf '%s\t%s\t%s\t%s\tforbidden\n' "${fold}" "${view}" "${OOF_ROOT}/fold${fold}_${view}" "$(git rev-parse HEAD)" >> "${CONTROL_DIR}/manifest.tsv"
  done
done

tasks=()
for fold in 0 1 2; do for view in full person face; do tasks+=("${fold}:${view}"); done; done
declare -A gpu_pid=()
declare -A gpu_task=()
next=0
failed=0
while (( next < ${#tasks[@]} || ${#gpu_pid[@]} > 0 )); do
  for gpu in "${!gpu_pid[@]}"; do
    pid="${gpu_pid[${gpu}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then rc=0; else rc=$?; failed=1; fi
      printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/${gpu_task[${gpu}]//:/_}.exit_code"
      unset 'gpu_pid['"${gpu}"']' 'gpu_task['"${gpu}"']'
    fi
  done
  if (( failed )); then
    for gpu in "${!gpu_pid[@]}"; do kill "${gpu_pid[${gpu}]}" 2>/dev/null || true; done
    echo "An OOF expert failed" >&2
    exit 1
  fi
  for gpu in "${GPU_LIST[@]}"; do
    (( next < ${#tasks[@]} )) || break
    [[ -z "${gpu_pid[${gpu}]:-}" ]] || continue
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB || utilization > MAX_UTILIZATION )); then continue; fi
    IFS=: read -r fold view <<< "${tasks[${next}]}"
    run_id="fold${fold}_${view}"
    GPU="${gpu}" VIEW="${view}" FOLD="${fold}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${OOF_ROOT}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
      DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
      bash scripts/emotic/run_multilane_track_a_oof_source_val.sh \
      > "${LOG_DIR}/${run_id}.launcher.log" 2>&1 &
    gpu_pid[${gpu}]=$!
    gpu_task[${gpu}]="${fold}:${view}"
    echo "Started fold=${fold} view=${view} gpu=${gpu} pid=${gpu_pid[${gpu}]}"
    next=$((next + 1))
  done
  if (( next < ${#tasks[@]} || ${#gpu_pid[@]} > 0 )); then sleep 15; fi
done

descriptor_gpu=""
while [[ -z "${descriptor_gpu}" ]]; do
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then descriptor_gpu="${gpu}"; break; fi
  done
  [[ -n "${descriptor_gpu}" ]] || sleep 15
done
CUDA_VISIBLE_DEVICES="${descriptor_gpu}" "${PYTHON}" -m multi_lane.track_a.export_three_view_descriptors \
  --data-root "${DATA_ROOT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" --output-dir "${DESCRIPTOR_ROOT}" \
  --device cuda --batch-size 128 --workers 2 --train-scope all \
  2>&1 | tee "${LOG_DIR}/descriptor_export.log"
printf '0\n' > "${CONTROL_DIR}/status/descriptors.exit_code"

"${PYTHON}" -m multi_lane.track_a.three_view_oof_router \
  --oof-root "${OOF_ROOT}" --full-run "${FULL_RUN}" --person-run "${PERSON_RUN}" \
  --face-run "${FACE_RUN}" --data-root "${DATA_ROOT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --descriptor-root "${DESCRIPTOR_ROOT}" \
  --output-dir "${SELECTION_DIR}" 2>&1 | tee "${LOG_DIR}/router_selection.log"
printf '0\n' > "${CONTROL_DIR}/status/router.exit_code"
printf 'validation_complete_test_not_accessed\n' > "${CONTROL_DIR}/batch_status.txt"
echo "THREE_VIEW_OOF_BATCH_COMPLETE batch=${BATCH_ID} selection=${SELECTION_DIR}/validation_selection.json"
