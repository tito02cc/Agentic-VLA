#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/home/admin1/openpi}"
POLICY_PYTHON="${POLICY_PYTHON:-/home/admin1/openpi/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/admin1/miniconda3/envs/openpi/bin/python}"
CHECKPOINT="${CHECKPOINT:-${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/results/carve_optimize/pi05_masked_view_compile_bf16_2step_h10_fidelity45_rep1.json}"
WARMUP_SNAPSHOT="${WARMUP_SNAPSHOT:-${PROJECT_ROOT}/results/carve_t689_paired_5states_v6/joint/failure_snapshots/task06_episode000_step0019.npz}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/carve_semantic_shadow/paired_t89_3trials}"
TASK_IDS="${TASK_IDS:-8,9}"
TRIALS="${TRIALS:-3}"
PORT="${PORT:-18083}"
VLM_ENDPOINT="${VLM_ENDPOINT:-http://127.0.0.1:18070/v1/chat/completions}"
VLM_MODEL="${VLM_MODEL:-/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B}"

mkdir -p "${RUN_ROOT}"
systemctl --user reset-failed g2-agent-vlm.service || true
systemctl --user start g2-agent-vlm.service
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:18070/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  systemctl --user stop g2-agent-vlm.service || true
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
    --warmup-snapshot "${WARMUP_SNAPSHOT}" \
    --warmup-calls 3 \
    >"${RUN_ROOT}/policy_server.log" 2>&1 &
server_pid="$!"

for _ in $(seq 1 360); do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited during startup" >&2
    exit 1
  fi
  sleep 1
done

run_variant() {
  local variant="$1"
  shift
  local output_dir="${RUN_ROOT}/${variant}"
  mkdir -p "${output_dir}"
  AGENTIC_VLA_OPENPI_ROOT="${OPENPI_ROOT}" \
  PYTHONPATH="${PROJECT_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/LIBERO" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${LIBERO_PYTHON}" "${PROJECT_ROOT}/scripts/run_agentic_vla_libero.py" \
      --host 127.0.0.1 --port "${PORT}" \
      --task-suite libero_10 --task-ids "${TASK_IDS}" --trials "${TRIALS}" \
      --seed 7 --replan-steps 10 --fixed-policy-noise \
      --perturbation mid_episode_nudge --mid-nudge-step 80 --mid-nudge-xy 0.03 \
      --carve-runtime --carve-joint-controller --carve-physical-recovery \
      --carve-fast-inference-steps 2 --carve-accurate-inference-steps 2 \
      --carve-commit-steps 10 --control-deadline-ms 80 \
      --results-json "${output_dir}/results.json" \
      --video-dir "${output_dir}/videos" \
      --episode-trace-jsonl "${output_dir}/episode_traces.jsonl" \
      --carve-trace-jsonl "${output_dir}/policy_calls.jsonl" \
      --ablation-tag "CARVE-${variant}" "$@"
}

run_variant no_semantic
run_variant semantic12_labels \
  --carve-semantic-shadow \
  --carve-semantic-endpoint "${VLM_ENDPOINT}" \
  --carve-semantic-model "${VLM_MODEL}" \
  --carve-semantic-cooldown-steps 100 \
  --carve-semantic-min-slack-ms 20 \
  --carve-semantic-timeout-sec 30 \
  --carve-semantic-max-tokens 12

"${POLICY_PYTHON}" "${PROJECT_ROOT}/scripts/summarize_semantic_shadow_gate.py" \
  --run-root "${RUN_ROOT}"
