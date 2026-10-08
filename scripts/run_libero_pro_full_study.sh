#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${PROJECT_ROOT}/third_party/openpi_official}"
POLICY_PYTHON="${POLICY_PYTHON:-${PROJECT_ROOT}/openpi/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-/home/admin1/miniconda3/envs/g2agent/bin/python}"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/home/admin1/ct/benchmark-sources/LIBERO-PRO}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
VLM_MODEL="${VLM_MODEL:-/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B}"
PRIMARY_MANIFEST="${PRIMARY_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_smve_promoted.json}"
FALLBACK_MANIFEST="${FALLBACK_MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots/task06_episode001_step0013.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/libero_pro_full_study_20260825}"
VLA_PORT="${VLA_PORT:-18081}"
VLM_PORT="${VLM_PORT:-18070}"
TRIALS="${TRIALS:-10}"
VIDEO_SIZE="${VIDEO_SIZE:-256}"
SUITES="${SUITES:-libero_10 libero_10_object libero_10_swap libero_10_task}"
METHODS="${METHODS:-frozen_vla fixed_recovery agentic}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
STALL_RECOVERY_STREAK="${STALL_RECOVERY_STREAK:-2}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

mkdir -p "${OUTPUT_ROOT}/_services"
vla_pid=""
vlm_pid=""

port_ready() {
  timeout 1 bash -c "</dev/tcp/127.0.0.1/$1" 2>/dev/null
}

wait_for_port() {
  local port="$1"
  local pid="$2"
  local limit="$3"
  for _ in $(seq 1 "${limit}"); do
    if port_ready "${port}"; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

stop_process() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_process "${vlm_pid}"
  stop_process "${vla_pid}"
}
trap cleanup EXIT INT TERM

start_vla() {
  stop_process "${vla_pid}"
  if port_ready "${VLA_PORT}"; then
    echo "VLA port ${VLA_PORT} is already occupied" >&2
    return 1
  fi
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src" \
  OPENPI_DISABLE_TORCH_COMPILE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${POLICY_PYTHON}" "${PROJECT_ROOT}/scripts/serve_openpi_policy_no_compile.py" \
      --port "${VLA_PORT}" \
      --policy-config pi05_libero \
      --policy-dir "${CHECKPOINT}" \
      --profile-manifest "${PRIMARY_MANIFEST}" \
      --fallback-profile-manifest "${FALLBACK_MANIFEST}" \
      --warmup-snapshot "${WARMUP_SNAPSHOT}" \
      --warmup-calls 2 \
      --warmup-fixed-noise \
      >>"${OUTPUT_ROOT}/_services/vla_server.log" 2>&1 &
  vla_pid="$!"
  wait_for_port "${VLA_PORT}" "${vla_pid}" 420 || {
    echo "PI0.5 service failed to start" >&2
    return 1
  }
}

start_vlm() {
  stop_process "${vlm_pid}"
  if port_ready "${VLM_PORT}"; then
    echo "VLM port ${VLM_PORT} is already occupied" >&2
    return 1
  fi
  "${VLM_PYTHON}" "${PROJECT_ROOT}/scripts/serve_carve_vlm_profile.py" \
    --model "${VLM_MODEL}" \
    --profile bf16 \
    --host 127.0.0.1 \
    --port "${VLM_PORT}" \
    --receipt "${OUTPUT_ROOT}/_services/qwen35_4b_bf16_receipt.json" \
    --log-level warning \
    >>"${OUTPUT_ROOT}/_services/vlm_server.log" 2>&1 &
  vlm_pid="$!"
  wait_for_port "${VLM_PORT}" "${vlm_pid}" 240 || {
    echo "Qwen Planner service failed to start" >&2
    return 1
  }
}

cell_complete() {
  local summary="$1"
  [[ -f "${summary}" ]] && jq -e \
    --argjson trials "${TRIALS}" \
    '.trials == $trials and (.episodes | length) == $trials' \
    "${summary}" >/dev/null
}

restart_services_for_method() {
  local method="$1"
  if [[ "${method}" == "agentic" ]]; then
    stop_process "${vlm_pid}"
    vlm_pid=""
  fi
  start_vla
  if [[ "${method}" == "agentic" ]]; then
    start_vlm
  fi
}

run_cell() {
  local suite="$1"
  local method="$2"
  local task_id="$3"
  local cell_root="${OUTPUT_ROOT}/${suite}/${method}/task_$(printf '%02d' "${task_id}")"
  local summary="${cell_root}/summary.json"
  mkdir -p "${cell_root}"
  if cell_complete "${summary}"; then
    echo "[skip] ${suite} ${method} task=${task_id}"
    return 0
  fi
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "[run] ${suite} ${method} task=${task_id} attempt=${attempt}"
    if env \
      AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
      AGENTIC_VLA_LIBERO_ROOT="${LIBERO_PRO_ROOT}" \
      PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_PRO_ROOT}" \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      "${LIBERO_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero_pro_canonical.py" \
        --suite "${suite}" \
        --task-id "${task_id}" \
        --trials "${TRIALS}" \
        --seed 7 \
        --method "${method}" \
        --vla-port "${VLA_PORT}" \
        --deadline-ms 80 \
        --max-steps 520 \
        --monitor-warmup-steps 60 \
        --stall-recovery-streak "${STALL_RECOVERY_STREAK}" \
        --video-size "${VIDEO_SIZE}" \
        --results-root "${cell_root}" \
        --run-id-prefix carve-full-study \
        --resume \
        >"${cell_root}/run_attempt_${attempt}.log" 2>&1; then
      if cell_complete "${summary}"; then
        echo "[done] ${suite} ${method} task=${task_id}"
        return 0
      fi
    fi
    echo "[retry] ${suite} ${method} task=${task_id}" >&2
    restart_services_for_method "${method}"
  done
  echo "[failed] ${suite} ${method} task=${task_id}" >&2
  return 1
}

available_kib=$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')
if (( available_kib < 20 * 1024 * 1024 )); then
  echo "At least 20 GiB free disk is required; available KiB=${available_kib}" >&2
  exit 1
fi

start_vla
for method in ${METHODS}; do
  if [[ "${method}" == "agentic" ]] && ! port_ready "${VLM_PORT}"; then
    start_vlm
  fi
  for suite in ${SUITES}; do
    for task_id in ${TASK_IDS}; do
      run_cell "${suite}" "${method}" "${task_id}"
    done
  done
done

if [[ "${SKIP_AGGREGATE}" != "1" ]]; then
  PYTHONPATH="${PROJECT_ROOT}" "${POLICY_PYTHON}" \
    "${PROJECT_ROOT}/scripts/summarize_libero_pro_full_study.py" \
      --study-root "${OUTPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}/aggregate"
fi

echo "LIBERO-Pro full study completed: ${OUTPUT_ROOT}"
