#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${AGENTIC_VLA_OPENPI_ROOT:-/home/admin1/ct/openpi-official}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_QWEN_DIR="${PROJECT_ROOT}/weights/Qwen3-VL-8B-Instruct"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-ablation_FULL_tuned_libero10_${STAMP}}"
ABLATION_TAG="${ABLATION_TAG:-Full-Agentic-VLA-Tuned}"
RESULT_DIR="${PROJECT_ROOT}/results/${RUN_NAME}"
RESULT_JSON="${RESULT_DIR}/summary.json"
VIDEO_DIR="${RESULT_DIR}/videos"
DEFAULT_QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
if [[ -d "${LOCAL_QWEN_DIR}" ]]; then
  DEFAULT_QWEN_MODEL="${LOCAL_QWEN_DIR}"
fi
QWEN_MODEL="${QWEN_MODEL:-${DEFAULT_QWEN_MODEL}}"
QWEN_QUANT="${QWEN_QUANT:-4bit}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${VIDEO_DIR}"

echo "[INFO] Launching Full Agentic-VLA on libero_10"
echo "[INFO] Ablation: ${ABLATION_TAG}"
echo "[INFO] Results: ${RESULT_JSON}"
echo "[INFO] Videos:  ${VIDEO_DIR}"
echo "[INFO] Qwen:    ${QWEN_MODEL} (${QWEN_QUANT})"

AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero" \
conda run -n openpi python "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
  --task-suite libero_10 \
  --transition \
  --graph-rag \
  --critic \
  --ablation-tag "${ABLATION_TAG}" \
  --qwen-model "${QWEN_MODEL}" \
  --qwen-quant "${QWEN_QUANT}" \
  --results-json "${RESULT_JSON}" \
  --video-dir "${VIDEO_DIR}" \
  ${EXTRA_ARGS}
