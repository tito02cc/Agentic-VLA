#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
POLICY_PYTHON="${POLICY_PYTHON:-/home/admin1/openpi/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
FALLBACK_MANIFEST="${FALLBACK_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots/task06_episode000_step0019.npz}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/carve_libero10_repro_20260717}"
TASK_IDS="${TASK_IDS:-}"
TRIALS="${TRIALS:-20}"
SEED="${SEED:-7}"
VARIANTS="${VARIANTS:-baseline agentic}"
PORT="${PORT:-18083}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-14000}"
WARMUP_CALLS="${WARMUP_CALLS:-3}"

for required in \
  "${CHECKPOINT}/model.safetensors" \
  "${MANIFEST}" \
  "${FALLBACK_MANIFEST}" \
  "${WARMUP_SNAPSHOT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
for python_path in "${POLICY_PYTHON}" "${LIBERO_PYTHON}"; do
  if [[ ! -x "${python_path}" ]]; then
    echo "Python environment is not executable: ${python_path}" >&2
    exit 2
  fi
done
if ! [[ "${TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer, got ${TRIALS}" >&2
  exit 2
fi

PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO" \
  "${LIBERO_PYTHON}" -c \
  "from libero.libero import benchmark; from openpi_client import websocket_client_policy" \
  >/dev/null

free_vram_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( free_vram_mib < MIN_FREE_VRAM_MIB )); then
  echo "CARVE gate deferred: ${free_vram_mib} MiB free, ${MIN_FREE_VRAM_MIB} MiB required." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"
server_log="${RUN_ROOT}/policy_server.log"
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
    --profile-manifest "${MANIFEST}" \
    --fallback-profile-manifest "${FALLBACK_MANIFEST}" \
    --warmup-snapshot "${WARMUP_SNAPSHOT}" \
    --warmup-calls "${WARMUP_CALLS}" \
    >"${server_log}" 2>&1 &
server_pid="$!"

for _ in $(seq 1 360); do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited during startup. See ${server_log}." >&2
    exit 1
  fi
  sleep 1
done
if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Policy server did not become ready on port ${PORT}." >&2
  exit 1
fi

task_args=()
if [[ -n "${TASK_IDS}" ]]; then
  task_args=(--task-ids "${TASK_IDS}")
fi

run_variant() {
  local variant="$1"
  local output_dir="${RUN_ROOT}/${variant}"
  local variant_args=()
  local tag=""
  case "${variant}" in
    baseline)
      tag="B0-VLA-SMVE-2step"
      variant_args=(
        --carve-runtime
        --carve-fixed-inference-steps 2
      )
      ;;
    agentic)
      tag="CARVE-joint-physical-SMVE-2step"
      variant_args=(
        --carve-runtime
        --carve-joint-controller
        --carve-physical-recovery
        --carve-fast-inference-steps 2
        --carve-accurate-inference-steps 2
        --carve-commit-steps 10
      )
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 2
      ;;
  esac

  mkdir -p "${output_dir}"
  echo "Running ${variant}: trials=${TRIALS}, tasks=${TASK_IDS:-all}, output=${output_dir}"
  AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${LIBERO_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --task-suite libero_10 \
      "${task_args[@]}" \
      --trials "${TRIALS}" \
      --seed "${SEED}" \
      --replan-steps 10 \
      --fixed-policy-noise \
      --control-deadline-ms 80 \
      --results-json "${output_dir}/results.json" \
      --video-dir "${output_dir}/videos" \
      --episode-trace-jsonl "${output_dir}/episode_traces.jsonl" \
      --carve-trace-jsonl "${output_dir}/policy_calls.jsonl" \
      --ablation-tag "${tag}" \
      "${variant_args[@]}"
}

for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

echo "CARVE reproducible LIBERO-10 gate completed: ${RUN_ROOT}"
