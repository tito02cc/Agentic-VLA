#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-12000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/carve_pi05_phase_adaptive_smoke}"

free_vram_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( free_vram_mib < MIN_FREE_VRAM_MIB )); then
  echo "CARVE smoke deferred: ${free_vram_mib} MiB free, ${MIN_FREE_VRAM_MIB} MiB required." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src" \
  conda run -n openpi python scripts/eval_robosuite_stack_pi05_policy_trials.py \
  --output-dir "${OUTPUT_DIR}" \
  --method-tag carve_phase_adaptive_retry \
  --trials 1 \
  --seed 20260715 \
  --compute-policy phase_adaptive \
  --adaptive-low-steps 1 \
  --adaptive-high-steps 2 \
  --adaptive-high-risk-phases grasp_or_lift,transport \
  --adaptive-escalate-after-stall-steps 15 \
  --inference-deadline-ms 200 \
  --agentic \
  --retry-skill geometric_stack \
  --no-save-first-success-video
