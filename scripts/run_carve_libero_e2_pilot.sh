#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
PORT="${PORT:-8000}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-14000}"
TASK_ID="${TASK_ID:-8}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/carve_e2_pi05_pilot}"
SERVER_LOG="${OUTPUT_ROOT}/policy_server.log"

free_vram_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( free_vram_mib < MIN_FREE_VRAM_MIB )); then
  echo "E2 pilot deferred: ${free_vram_mib} MiB free, ${MIN_FREE_VRAM_MIB} MiB required." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src" \
  OPENPI_DISABLE_TORCH_COMPILE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${OPENPI_PYTHON}" "${PROJECT_ROOT}/scripts/serve_openpi_policy_no_compile.py" \
      --port "${PORT}" \
      --policy-config pi05_libero \
      --policy-dir "${CHECKPOINT}" \
      >"${SERVER_LOG}" 2>&1 &
  server_pid="$!"
  for _ in $(seq 1 180); do
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited during startup. See ${SERVER_LOG}." >&2
      exit 1
    fi
    sleep 1
  done
fi

if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Policy server did not become ready on port ${PORT}." >&2
  exit 1
fi

run_episode() {
  local name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${output_dir}"
  AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${OPENPI_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --task-suite libero_10 \
      --task-id "${TASK_ID}" \
      --trials 1 \
      --seed "${SEED}" \
      --replan-steps 10 \
      --fixed-policy-noise \
      --results-json "${output_dir}/results.json" \
      --video-dir "${output_dir}/videos" \
      --episode-trace-jsonl "${output_dir}/episode_traces.jsonl" \
      --ablation-tag "${name}" \
      "$@"
}

run_episode "e2_legacy"
run_episode "e2_adapter" \
  --carve-runtime \
  --carve-trace-jsonl "${OUTPUT_ROOT}/e2_adapter/policy_calls.jsonl"
run_episode "e2_carve_joint" \
  --carve-runtime \
  --carve-joint-controller \
  --carve-branch-dir "${OUTPUT_ROOT}/e2_carve_joint/failure_snapshots" \
  --carve-trace-jsonl "${OUTPUT_ROOT}/e2_carve_joint/policy_calls.jsonl"

echo "E2 pilot completed: ${OUTPUT_ROOT}"
