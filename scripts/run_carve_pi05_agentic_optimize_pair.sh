#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
POLICY_PYTHON="${POLICY_PYTHON:-/home/admin1/openpi/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${SNAPSHOT_DIR}/task06_episode001_step0013.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/carve_pi05_agentic_optimize_pair_20260719}"
PORT="${PORT:-18095}"

EAGER_MANIFEST="${EAGER_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/pi05_eager_bf16_2step_h10_deadline80_rep1.json}"
COMPILED_MANIFEST="${COMPILED_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
SMVE_MANIFEST="${SMVE_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
SMVE_FALLBACK="${SMVE_FALLBACK:-${COMPILED_MANIFEST}}"

mkdir -p "${OUTPUT_DIR}"
server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

common_env=(
  "AGENTIC_VLA_OPENPI_ROOT=${OPENPI_ROOT}"
  "PYTHONPATH=${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO"
  "MUJOCO_GL=egl"
  "PYOPENGL_PLATFORM=egl"
  "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
)

run_profile() {
  local profile_name="$1"
  local manifest="$2"
  local fallback_manifest="$3"
  local require_promoted="$4"
  local profile_dir="${OUTPUT_DIR}/${profile_name}"
  mkdir -p "${profile_dir}/online_controller"

  local server_args=(
    --port "${PORT}"
    --policy-config pi05_libero
    --policy-dir "${CHECKPOINT}"
    --profile-manifest "${manifest}"
    --warmup-snapshot "${WARMUP_SNAPSHOT}"
    --warmup-calls 2
  )
  if [[ -n "${fallback_manifest}" ]]; then
    server_args+=(--fallback-profile-manifest "${fallback_manifest}")
  fi
  if [[ "${require_promoted}" != "true" ]]; then
    server_args+=(--no-require-promoted-profile)
  fi

  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src" \
  OPENPI_DISABLE_TORCH_COMPILE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${POLICY_PYTHON}" "${PROJECT_ROOT}/scripts/serve_openpi_policy_no_compile.py" \
      "${server_args[@]}" >"${profile_dir}/policy_server.log" 2>&1 &
  server_pid="$!"

  for _ in $(seq 1 360); do
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited during ${profile_name} startup." >&2
      return 1
    fi
    sleep 1
  done
  if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    echo "Policy server did not become ready for ${profile_name}." >&2
    return 1
  fi

  env "${common_env[@]}" "${LIBERO_PYTHON}" \
    "${PROJECT_ROOT}/scripts/run_carve_libero_failure_branches.py" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --snapshot-names task06_episode001_step0013 task09_episode001_step0013 \
    --branches physical_recovery \
    --branch-horizon 440 \
    --replan-steps 8 \
    --recovery-chunks 2 \
    --fixed-inference-steps 2 \
    --deadline-ms 80 \
    --save-videos \
    --output "${profile_dir}/paired_branches.json"

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
    --results-json "${profile_dir}/online_controller/results.json" \
    --video-dir "${profile_dir}/online_controller/videos" \
    --episode-trace-jsonl "${profile_dir}/online_controller/episode_traces.jsonl" \
    --carve-trace-jsonl "${profile_dir}/online_controller/policy_calls.jsonl" \
    --ablation-tag "CARVE-PI05-${profile_name}-online-recovery"

  stop_server
}

run_profile eager_bf16 "${EAGER_MANIFEST}" "" false
run_profile compiled_bf16 "${COMPILED_MANIFEST}" "" true
run_profile compiled_smve "${SMVE_MANIFEST}" "${SMVE_FALLBACK}" true

PYTHONPATH="${PROJECT_ROOT}" "${POLICY_PYTHON}" \
  "${PROJECT_ROOT}/scripts/summarize_carve_agentic_optimize_pair.py" \
  --root "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json" \
  --output-md "${OUTPUT_DIR}/REPORT.md" \
  --output-plot "${OUTPUT_DIR}/agentic_optimize_pair.png"

echo "CARVE PI0.5 Agentic-Optimize pair completed: ${OUTPUT_DIR}"
