#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
SOURCE_ROOT="${PROJECT_ROOT}/artifacts/robomme/unmaskswap_fixed_five_20260923"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 2
fi
for episode in 15 16 17 18 19; do
  test -s "${SOURCE_ROOT}/memory/VideoUnmaskSwap_ep${episode}_memory.json" || {
    echo "missing fixed memory for episode ${episode}" >&2
    exit 2
  }
done
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "policy or planner port already in use; refusing to alter an existing service" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/rollouts"
policy_pid=""
planner_pid=""
gpu_pid=""
cleanup() {
  for pid in "${gpu_pid}" "${planner_pid}" "${policy_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/agentic_vla/runtime/semantic.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_unmaskswap_runtime_bc.sh" \
  > "${OUTPUT_ROOT}/source_sha256.txt"
CHECKPOINT_ROOT="${ROBOMME_CHECKPOINT:-${HOME}/.cache/carve-vla/checkpoints/robomme/mme_vla_suite/symbolic-grounded-subgoal/79999}"
MODEL_ROOT="${ROBOMME_PLANNER_MODEL:-${HOME}/.cache/carve-vla/checkpoints/Qwen3-VL-4B-Instruct}"
ADAPTER_ROOT="${ROBOMME_PLANNER_ADAPTER:-${HOME}/.cache/carve-vla/checkpoints/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200}"
sha256sum "${MODEL_ROOT}"/model-*.safetensors \
  "${ADAPTER_ROOT}/adapter_model.safetensors" \
  "${CHECKPOINT_ROOT}/assets/robomme/norm_stats.json" \
  "$(dirname "${CHECKPOINT_ROOT}")/history_config.txt" \
  > "${OUTPUT_ROOT}/model_sha256.txt"
find "${CHECKPOINT_ROOT}/params" -type f -print0 | sort -z | \
  xargs -0 sha256sum >> "${OUTPUT_ROOT}/model_sha256.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' > "${OUTPUT_ROOT}/docker_image_id.txt"

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" \
  > "${OUTPUT_ROOT}/policy_service.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" \
  > "${OUTPUT_ROOT}/planner_service.log" 2>&1 &
planner_pid="$!"

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "${policy_pid}" 2>/dev/null || ! kill -0 "${planner_pid}" 2>/dev/null; then
    echo "a model service exited during startup" >&2
    exit 2
  fi
  if ss -ltn | rg -q ':8011 ' && \
     curl -fsS --max-time 2 http://127.0.0.1:18070/v1/models >/dev/null; then
    ready=1
    break
  fi
  sleep 3
done
if [[ "${ready}" != 1 ]]; then
  echo "model services did not become ready within 360 seconds" >&2
  exit 2
fi

run_one() {
  local episode="$1" arm="$2" max_steps="$3" name="$4"
  local output="${OUTPUT_ROOT}/rollouts/${name}"
  local memory="/workspace/artifacts/robomme/unmaskswap_fixed_five_20260923/memory/VideoUnmaskSwap_ep${episode}_memory.json"
  local schedule=(--planner-schedule every_chunk)
  if [[ "${arm}" == "C" ]]; then
    schedule=(--planner-schedule selective --verified-point-reuse)
  fi
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite ${output}" >&2
    exit 2
  fi
  echo "starting ${name} $(date -u +%FT%TZ)" | tee -a "${OUTPUT_ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" \
    /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task VideoUnmaskSwap --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      "${schedule[@]}" --task-memory-file "${memory}" \
      --memory-hint-style precise --memory-use-policy always \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/unmaskswap_runtime_bc_20260923/rollouts/${name}" \
    > "${OUTPUT_ROOT}/rollouts/${name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${OUTPUT_ROOT}/run_order.log"
  if [[ ! -s "${output}/summary.json" ]]; then
    echo "missing summary for ${name}; stopping after infrastructure failure" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["task"],d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True)' \
    "${output}/summary.json"
}

# Unscored compile warmup: one known public-demo episode, same servers, no A/B result.
run_one 15 B 16 "VideoUnmaskSwap_ep15_warmup"
nvidia-smi --query-gpu=timestamp,memory.used,power.draw \
  --format=csv,noheader,nounits --loop-ms=1000 \
  > "${OUTPUT_ROOT}/gpu_usage.csv" 2> "${OUTPUT_ROOT}/gpu_monitor.log" &
gpu_pid="$!"

for entry in 15:B 15:C 16:C 16:B 17:B 17:C 18:C 18:B 19:B 19:C; do
  run_one "${entry%:*}" "${entry#*:}" 1300 "VideoUnmaskSwap_ep${entry%:*}_${entry#*:}"
done

python3 "${PROJECT_ROOT}/scripts/summarize_robomme_unmaskswap_runtime_bc.py" \
  --root "${OUTPUT_ROOT}"
