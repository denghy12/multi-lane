#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
BATCH_ID="${BATCH_ID:-constrained_gated_fusion_seed012_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="./output/emotic_track_a_constrained_gated_fusion/${BATCH_ID}"
LOG_DIR="./logs/emotic_track_a_constrained_gated_fusion"
mkdir -p "${LOG_DIR}"

VAL0_BASE="/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720"
VAL12_BASE="/mnt/haoyuan/workspace/multi-lane-main-ensemble-control/output/emotic_track_a_ensemble_control/fixed_ensemble_control_val_seed12_20260903_191515/runs"
TEST0_BASE="/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_formal_v0.1/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816"
TEST12_BASE="/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938"

"${PYTHON_BIN}" -m multi_lane.track_a.search_constrained_gated_fusion \
  --validation-full-runs \
    "${VAL0_BASE}/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor" \
    "${VAL12_BASE}/fixed_ensemble_control_val_seed12_20260903_191515_seed1_full" \
    "${VAL12_BASE}/fixed_ensemble_control_val_seed12_20260903_191515_seed2_full" \
  --validation-person-runs \
    "${VAL0_BASE}/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox" \
    "${VAL12_BASE}/fixed_ensemble_control_val_seed12_20260903_191515_seed1_person" \
    "${VAL12_BASE}/fixed_ensemble_control_val_seed12_20260903_191515_seed2_person" \
  --test-full-runs \
    "${TEST0_BASE}/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816_full_anchor" \
    "${TEST12_BASE}/seed1/dual_view_locked_formal_seed12_20260903_174938_seed1_full_anchor" \
    "${TEST12_BASE}/seed2/dual_view_locked_formal_seed12_20260903_174938_seed2_full_anchor" \
  --test-person-runs \
    "${TEST0_BASE}/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816_person_letterbox" \
    "${TEST12_BASE}/seed1/dual_view_locked_formal_seed12_20260903_174938_seed1_person_letterbox" \
    "${TEST12_BASE}/seed2/dual_view_locked_formal_seed12_20260903_174938_seed2_person_letterbox" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/${BATCH_ID}.log"

printf 'CONSTRAINED_GATED_FUSION_COMPLETE output=%s log=%s\n' "${OUTPUT_DIR}" "${LOG_DIR}/${BATCH_ID}.log"
