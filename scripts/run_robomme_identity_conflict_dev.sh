#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT}/artifacts/robomme/identity_conflict_dev_20260923"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"

if [[ -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite development rollouts" >&2
  exit 2
fi
for episode in 15 16 20 21; do
  if (( episode < 20 )); then
    memory_root="${PROJECT_ROOT}/artifacts/robomme/unmaskswap_fixed_five_20260923"
  else
    memory_root="${PROJECT_ROOT}/artifacts/robomme/service_block_pilot_20260923"
  fi
  test -s "${memory_root}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
done
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "required model port is occupied" >&2
  exit 2
fi
mkdir -p "${ROOT}/rollouts"
sha256sum -c "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923/model_sha256.txt" \
  > "${ROOT}/model_integrity.log"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_identity_conflict_dev.sh" \
  > "${ROOT}/source_sha256.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' > "${ROOT}/docker_image_id.txt"

policy_pid=""
planner_pid=""
gpu_pid=""
cleanup() {
  for pid in "${planner_pid}" "${policy_pid}" "${gpu_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" \
  > "${ROOT}/policy_service.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" \
  > "${ROOT}/planner_service.log" 2>&1 &
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
  echo "model services did not become ready" >&2
  exit 2
fi
nvidia-smi --query-gpu=timestamp,memory.used,power.draw \
  --format=csv,noheader,nounits --loop-ms=1000 \
  > "${ROOT}/gpu_usage.csv" 2> "${ROOT}/gpu_monitor.log" &
gpu_pid="$!"

run_case() {
  local episode="$1" arm="$2" max_steps="$3" output_name="$4"
  local memory_source="unmaskswap_fixed_five_20260923"
  if (( episode >= 20 )); then
    memory_source="service_block_pilot_20260923"
  fi
  local memory="/workspace/artifacts/robomme/${memory_source}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
  local flag=()
  if [[ "${arm}" == "D" ]]; then
    flag=(--verified-identity-conflict)
  fi
  local output="${ROOT}/rollouts/${output_name}"
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite ${output}" >&2
    exit 2
  fi
  echo "starting ${output_name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task VideoUnmaskSwap --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      --planner-schedule every_chunk --task-memory-file "${memory}" \
      --memory-hint-style precise --memory-use-policy always \
      "${flag[@]}" \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/identity_conflict_dev_20260923/rollouts/${output_name}" \
    > "${ROOT}/rollouts/${output_name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${output_name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  if [[ ! -s "${output}/summary.json" ]]; then
    echo "missing summary after ${output_name}; stopping on infrastructure failure" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True)' \
    "${output}/summary.json"
}

run_case 15 B 16 warmup_ep15_B
run_case 16 B 1300 VideoUnmaskSwap_ep16_B
run_case 16 D 1300 VideoUnmaskSwap_ep16_D
run_case 20 D 1300 VideoUnmaskSwap_ep20_D
run_case 20 B 1300 VideoUnmaskSwap_ep20_B
run_case 21 B 1300 VideoUnmaskSwap_ep21_B
run_case 21 D 1300 VideoUnmaskSwap_ep21_D
