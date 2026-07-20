#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${AGENTIC_VLA_OPENPI_ROOT:-/home/admin1/ct/openpi-official}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_QWEN_DIR="${PROJECT_ROOT}/weights/Qwen3-VL-8B-Instruct"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK_IDS="${TASK_IDS:-8 9 6}"
TRIALS="${TRIALS:-10}"
VARIANTS="${VARIANTS:-full transition graph critic}"
PORT="${PORT:-8000}"
DEFAULT_QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
if [[ -d "${LOCAL_QWEN_DIR}" ]]; then
  DEFAULT_QWEN_MODEL="${LOCAL_QWEN_DIR}"
fi
QWEN_MODEL="${QWEN_MODEL:-${DEFAULT_QWEN_MODEL}}"
QWEN_QUANT="${QWEN_QUANT:-4bit}"

run_variant() {
  local variant="$1"
  local task_id="$2"
  local run_name ablation_tag flags

  case "${variant}" in
    full)
      ablation_tag="Full-Agentic-VLA-Tuned-v2-Task${task_id}"
      run_name="ablation_FULL_tuned_v2_task${task_id}_${STAMP}"
      flags=(--transition --graph-rag --critic)
      ;;
    transition)
      ablation_tag="A2-Transition-Tuned-v2-Task${task_id}"
      run_name="ablation_A2_transition_tuned_v2_task${task_id}_${STAMP}"
      flags=(--transition)
      ;;
    graph)
      ablation_tag="A3-GraphRAG-Tuned-v2-Task${task_id}"
      run_name="ablation_A3_graphrag_tuned_v2_task${task_id}_${STAMP}"
      flags=(--graph-rag)
      ;;
    critic)
      ablation_tag="A4-Critic-Tuned-v2-Task${task_id}"
      run_name="ablation_A4_critic_tuned_v2_task${task_id}_${STAMP}"
      flags=(--critic)
      ;;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  local result_dir="${PROJECT_ROOT}/results/${run_name}"
  local result_json="${result_dir}/summary.json"
  local video_dir="${result_dir}/videos"
  mkdir -p "${video_dir}"

  echo
  echo "============================================================"
  echo "[INFO] Variant: ${variant} | Task: ${task_id} | Trials: ${TRIALS}"
  echo "[INFO] Tag: ${ablation_tag}"
  echo "[INFO] Results: ${result_json}"
  echo "============================================================"

  AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
  PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero" \
  conda run -n openpi python "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
    --task-suite libero_10 \
    --task-id "${task_id}" \
    --trials "${TRIALS}" \
    "${flags[@]}" \
    --ablation-tag "${ablation_tag}" \
    --qwen-model "${QWEN_MODEL}" \
    --qwen-quant "${QWEN_QUANT}" \
    --port "${PORT}" \
    --results-json "${result_json}" \
    --video-dir "${video_dir}"
}

echo "[INFO] Starting low-success ablation sequence"
echo "[INFO] Tasks:    ${TASK_IDS}"
echo "[INFO] Variants: ${VARIANTS}"
echo "[INFO] Trials:   ${TRIALS}"
echo "[INFO] Port:     ${PORT}"
echo "[INFO] Qwen:     ${QWEN_MODEL} (${QWEN_QUANT})"

for task_id in ${TASK_IDS}; do
  for variant in ${VARIANTS}; do
    run_variant "${variant}" "${task_id}"
  done
done

echo
echo "[INFO] Low-success ablation sequence finished."
