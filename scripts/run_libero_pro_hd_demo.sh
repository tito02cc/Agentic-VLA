#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
POLICY_PYTHON="${POLICY_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/home/admin1/ct/benchmark-sources/LIBERO-PRO}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
PRIMARY_MANIFEST="${PRIMARY_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
FALLBACK_MANIFEST="${FALLBACK_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots/task06_episode001_step0013.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/libero_pro_hd_demo_t8_seed7}"
PORT="${PORT:-18099}"
TRIALS="${TRIALS:-3}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-720}"

mkdir -p "${OUTPUT_DIR}/videos"
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
  "${POLICY_PYTHON}" "${PROJECT_ROOT}/scripts/serve_openpi_policy_no_compile.py" \
    --port "${PORT}" \
    --policy-config pi05_libero \
    --policy-dir "${CHECKPOINT}" \
    --profile-manifest "${PRIMARY_MANIFEST}" \
    --fallback-profile-manifest "${FALLBACK_MANIFEST}" \
    --warmup-snapshot "${WARMUP_SNAPSHOT}" \
    --warmup-calls 2 \
    >"${OUTPUT_DIR}/policy_server.log" 2>&1 &
server_pid="$!"

for _ in $(seq 1 360); do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "PI0.5 server exited during startup; see ${OUTPUT_DIR}/policy_server.log" >&2
    exit 1
  fi
  sleep 1
done
timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null || {
  echo "PI0.5 server did not become ready on port ${PORT}." >&2
  exit 1
}

env \
  AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
  AGENTIC_VLA_LIBERO_ROOT="${LIBERO_PRO_ROOT}" \
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_PRO_ROOT}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "${LIBERO_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --task-suite libero_10_object \
    --task-ids 8 \
    --trials "${TRIALS}" \
    --seed 7 \
    --fixed-policy-noise \
    --perturbation clean \
    --video-dir "${OUTPUT_DIR}/videos" \
    --video-render-size "${VIDEO_RENDER_SIZE}" \
    --video-camera agentview \
    --results-json "${OUTPUT_DIR}/summary.json" \
    --episode-trace-jsonl "${OUTPUT_DIR}/episode_traces.jsonl" \
    --carve-trace-jsonl "${OUTPUT_DIR}/policy_calls.jsonl" \
    --carve-runtime \
    --carve-joint-controller \
    --carve-physical-recovery \
    --carve-max-recovery-attempts 1 \
    --carve-monitor-warmup-steps 60 \
    --carve-fast-inference-steps 2 \
    --carve-accurate-inference-steps 2 \
    --ablation-tag LIBERO-PRO-HD-Demo-T8-Fixed-Recovery \
    >"${OUTPUT_DIR}/run.log" 2>&1

echo "LIBERO-PRO HD demo completed: ${OUTPUT_DIR}"
