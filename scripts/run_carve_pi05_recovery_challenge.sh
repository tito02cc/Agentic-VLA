#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
POLICY_PYTHON="${POLICY_PYTHON:-/home/admin1/openpi/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
PRIMARY_MANIFEST="${PRIMARY_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
FALLBACK_MANIFEST="${FALLBACK_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${SNAPSHOT_DIR}/task06_episode001_step0013.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/carve_pi05_recovery_challenge_20260719}"
PORT="${PORT:-18094}"
NOISE_SEED="${NOISE_SEED:-7}"

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
    echo "Policy server exited during startup." >&2
    exit 1
  fi
  sleep 1
done
if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Policy server did not become ready on port ${PORT}." >&2
  exit 1
fi

common_env=(
  "AGENTIC_VLA_OPENPI_ROOT=${OPENPI_ROOT}"
  "PYTHONPATH=${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO"
  "MUJOCO_GL=egl"
  "PYOPENGL_PLATFORM=egl"
  "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
)

env "${common_env[@]}" "${LIBERO_PYTHON}" \
  "${PROJECT_ROOT}/scripts/run_carve_libero_failure_branches.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --snapshot-names \
    task06_episode001_step0013 \
    task09_episode001_step0013 \
    task08_episode003_step0099 \
  --branches continue,accurate,recovery,physical_recovery \
  --branch-horizon 440 \
  --replan-steps 8 \
  --recovery-chunks 2 \
  --fixed-inference-steps 2 \
  --deadline-ms 80 \
  --noise-seed "${NOISE_SEED}" \
  --save-videos \
  --output "${OUTPUT_DIR}/paired_branches.json"

mkdir -p "${OUTPUT_DIR}/online_controller"
env "${common_env[@]}" "${LIBERO_PYTHON}" \
  "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --task-suite libero_10 \
  --task-id 6 \
  --trials 1 \
  --seed 7 \
  --replan-steps 10 \
  --fixed-policy-noise \
  --carve-runtime \
  --carve-joint-controller \
  --carve-physical-recovery \
  --carve-fast-inference-steps 2 \
  --carve-accurate-inference-steps 2 \
  --carve-commit-steps 10 \
  --control-deadline-ms 80 \
  --results-json "${OUTPUT_DIR}/online_controller/results.json" \
  --video-dir "${OUTPUT_DIR}/online_controller/videos" \
  --episode-trace-jsonl "${OUTPUT_DIR}/online_controller/episode_traces.jsonl" \
  --carve-trace-jsonl "${OUTPUT_DIR}/online_controller/policy_calls.jsonl" \
  --ablation-tag CARVE-PI05-online-recovery-sentinel

PYTHONPATH="${PROJECT_ROOT}" "${POLICY_PYTHON}" \
  "${PROJECT_ROOT}/scripts/summarize_carve_recovery_challenge.py" \
  --branches "${OUTPUT_DIR}/paired_branches.json" \
  --online "${OUTPUT_DIR}/online_controller/results.json" \
  --output-json "${OUTPUT_DIR}/summary.json" \
  --output-md "${OUTPUT_DIR}/REPORT.md"

PYTHONPATH="${PROJECT_ROOT}" "${POLICY_PYTHON}" \
  "${PROJECT_ROOT}/scripts/plot_carve_recovery_challenge.py" \
  --summary "${OUTPUT_DIR}/summary.json" \
  --output "${OUTPUT_DIR}/recovery_challenge.png"

echo "CARVE PI0.5 recovery challenge completed: ${OUTPUT_DIR}"
