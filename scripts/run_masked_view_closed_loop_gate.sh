#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/admin1/openpi/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
FALLBACK_MANIFEST="${FALLBACK_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
SNAPSHOT="${SNAPSHOT:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots/task06_episode000_step0019.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/carve_optimize/masked_view_t89_5states_20260717}"
PORT="${PORT:-18082}"

mkdir -p "${OUTPUT_DIR}"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src" \
OPENPI_DISABLE_TORCH_COMPILE=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${OPENPI_PYTHON}" "${PROJECT_ROOT}/scripts/serve_openpi_policy_no_compile.py" \
    --port "${PORT}" \
    --policy-config pi05_libero \
    --policy-dir "${CHECKPOINT}" \
    --profile-manifest "${MANIFEST}" \
    --fallback-profile-manifest "${FALLBACK_MANIFEST}" \
    --warmup-snapshot "${SNAPSHOT}" \
    --warmup-calls 3 \
    >"${OUTPUT_DIR}/policy_server.log" 2>&1 &
server_pid="$!"

for _ in $(seq 1 360); do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited during startup. See ${OUTPUT_DIR}/policy_server.log." >&2
    exit 1
  fi
  sleep 1
done
if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Policy server did not become ready on port ${PORT}." >&2
  exit 1
fi

AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "${OPENPI_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --task-suite libero_10 \
    --task-ids 8,9 \
    --trials 5 \
    --seed 7 \
    --replan-steps 10 \
    --fixed-policy-noise \
    --carve-runtime \
    --carve-joint-controller \
    --carve-physical-recovery \
    --carve-fast-inference-steps 2 \
    --carve-accurate-inference-steps 2 \
    --carve-commit-steps 8 \
    --control-deadline-ms 80 \
    --results-json "${OUTPUT_DIR}/results.json" \
    --video-dir "${OUTPUT_DIR}/videos" \
    --episode-trace-jsonl "${OUTPUT_DIR}/episode_traces.jsonl" \
    --carve-trace-jsonl "${OUTPUT_DIR}/policy_calls.jsonl" \
    --ablation-tag CARVE-masked-view-t89-5states

echo "Masked-view closed-loop gate completed: ${OUTPUT_DIR}"
