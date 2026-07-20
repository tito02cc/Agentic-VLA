#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${AGENTIC_VLA_OPENPI_ROOT:-/home/admin1/ct/openpi-official}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${1:-${PROJECT_ROOT}/results/ablation_A2_transition_libero10_20260513}"
RESULT_JSON="${RESULT_DIR}/summary.json"
VIDEO_DIR="${RESULT_DIR}/videos"

echo "[INFO] Resuming A2 Transition on libero_10"
echo "[INFO] Results: ${RESULT_JSON}"
echo "[INFO] Videos:  ${VIDEO_DIR}"

AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero" \
conda run -n openpi python "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
  --task-suite libero_10 \
  --transition \
  --ablation-tag A2-Transition \
  --results-json "${RESULT_JSON}" \
  --video-dir "${VIDEO_DIR}"
