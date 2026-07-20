#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_STAMP="${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRIALS="${TRIALS:-10}"

detect_task8_pid() {
  pgrep -f "run_agentic_vla_libero.py .*--ablation-tag Full-Agentic-VLA-Tuned-v2-Task8" | tail -n 1 || true
}

CURRENT_TASK8_PID="${CURRENT_TASK8_PID:-$(detect_task8_pid)}"

echo "[INFO] Pending low-success pipeline stamp: ${PIPELINE_STAMP}"
echo "[INFO] Trials per run: ${TRIALS}"

if [[ -n "${CURRENT_TASK8_PID}" ]]; then
  echo "[INFO] Waiting for current Task 8 Full v2 run to finish: PID=${CURRENT_TASK8_PID}"
  while kill -0 "${CURRENT_TASK8_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[INFO] Current Task 8 Full v2 run has finished."
else
  echo "[INFO] No active Task 8 Full v2 PID detected. Continuing immediately."
fi

echo "[INFO] Running remaining Task 8 ablations (transition / graph / critic)"
STAMP="${PIPELINE_STAMP}" \
TRIALS="${TRIALS}" \
TASK_IDS="8" \
VARIANTS="transition graph critic" \
bash "${PROJECT_ROOT}/scripts/run_low_success_ablations.sh"

echo "[INFO] Running full low-success matrix for Task 9 and Task 6"
STAMP="${PIPELINE_STAMP}" \
TRIALS="${TRIALS}" \
TASK_IDS="9 6" \
VARIANTS="full transition graph critic" \
bash "${PROJECT_ROOT}/scripts/run_low_success_ablations.sh"

echo "[INFO] Pending low-success pipeline finished."
