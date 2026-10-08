#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
ROOT="${PROJECT_ROOT}/artifacts/robomme/service_block_pilot_20260923"
OLD_MEMORY="/workspace/artifacts/robomme/unmaskswap_fixed_five_20260923/memory/VideoUnmaskSwap_ep15_memory.json"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
RESUME_BLOCK2="${1:-}"

if [[ -n "${RESUME_BLOCK2}" && "${RESUME_BLOCK2}" != "--resume-block2" ]]; then
  echo "usage: $0 [--resume-block2]" >&2
  exit 2
fi

if [[ "${RESUME_BLOCK2}" == "--resume-block2" ]]; then
  for name in \
    VideoUnmaskSwap_ep15_B \
    VideoUnmaskSwap_ep20_A VideoUnmaskSwap_ep20_B VideoUnmaskSwap_ep20_C \
    VideoUnmaskSwap_ep21_A VideoUnmaskSwap_ep21_B VideoUnmaskSwap_ep21_C \
    VideoUnmask_ep20_A VideoUnmask_ep20_B \
    VideoUnmask_ep21_A VideoUnmask_ep21_B; do
    test -s "${ROOT}/rollouts/block1/${name}/summary.json" || {
      echo "block1 is incomplete: ${name}" >&2
      exit 2
    }
  done
  if [[ -e "${ROOT}/rollouts/block2/VideoUnmaskSwap_ep15_B" ]]; then
    echo "refusing to overwrite block2 rollouts" >&2
    exit 2
  fi
elif [[ -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite service-block rollouts" >&2
  exit 2
fi
for task in VideoUnmaskSwap VideoUnmask; do
  for episode in 20 21; do
    test -s "${ROOT}/preflight/${task}_ep${episode}/summary.json"
    test -s "${ROOT}/memory/${task}_ep${episode}_memory.json"
  done
done
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "required model port is already occupied" >&2
  exit 2
fi
mkdir -p "${ROOT}/rollouts/block1" "${ROOT}/rollouts/block2"
sha256sum -c "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923/model_sha256.txt" \
  > "${ROOT}/model_integrity.log"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/agentic_vla/runtime/semantic.py" \
  "${PROJECT_ROOT}/agentic_vla/benchmarks/robomme_memory.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_service_block_pilot.sh" \
  > "${ROOT}/source_sha256.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' > "${ROOT}/docker_image_id.txt"

policy_pid=""
planner_pid=""
gpu_pid=""
stop_services() {
  for pid in "${planner_pid}" "${policy_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  policy_pid=""
  planner_pid=""
}
cleanup() {
  stop_services
  if [[ -n "${gpu_pid}" ]]; then
    kill "${gpu_pid}" 2>/dev/null || true
    wait "${gpu_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_services() {
  local block="$1" ready=0
  "${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" \
    > "${ROOT}/rollouts/${block}/policy_service.log" 2>&1 &
  policy_pid="$!"
  "${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" \
    > "${ROOT}/rollouts/${block}/planner_service.log" 2>&1 &
  planner_pid="$!"
  for _ in $(seq 1 120); do
    if ! kill -0 "${policy_pid}" 2>/dev/null || ! kill -0 "${planner_pid}" 2>/dev/null; then
      echo "a model service exited during ${block} startup" >&2
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
    echo "model services not ready in ${block}" >&2
    exit 2
  fi
}

run_one() {
  local block="$1" task="$2" episode="$3" arm="$4" max_steps="$5"
  local name="${task}_ep${episode}_${arm}"
  local output="${ROOT}/rollouts/${block}/${name}"
  local schedule=(--planner-schedule every_chunk)
  local memory_args=()
  if [[ "${arm}" == "C" ]]; then
    schedule=(--planner-schedule selective --verified-point-reuse)
  fi
  if [[ "${arm}" != "A" ]]; then
    local memory="/workspace/artifacts/robomme/service_block_pilot_20260923/memory/${task}_ep${episode}_memory.json"
    if [[ "${episode}" == 15 ]]; then
      memory="${OLD_MEMORY}"
    fi
    local policy="always"
    if [[ "${task}" == "VideoUnmask" ]]; then
      policy="motion_gate"
    fi
    memory_args=(--task-memory-file "${memory}" --memory-hint-style precise --memory-use-policy "${policy}")
  fi
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite ${output}" >&2
    exit 2
  fi
  echo "starting ${block} ${name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task "${task}" --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      "${schedule[@]}" "${memory_args[@]}" \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/service_block_pilot_20260923/rollouts/${block}/${name}" \
    > "${ROOT}/rollouts/${block}/${name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${block} ${name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  if [[ ! -s "${output}/summary.json" ]]; then
    echo "missing summary for ${block} ${name}; stopping after infrastructure failure" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["task"],d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True)' \
    "${output}/summary.json"
}

nvidia-smi --query-gpu=timestamp,memory.used,power.draw \
  --format=csv,noheader,nounits --loop-ms=1000 \
  > "${ROOT}/gpu_usage.csv" 2> "${ROOT}/gpu_monitor.log" &
gpu_pid="$!"

blocks=(block1 block2)
if [[ "${RESUME_BLOCK2}" == "--resume-block2" ]]; then
  blocks=(block2)
fi
for block in "${blocks[@]}"; do
  start_services "${block}"
  # Prewarm a previously seen state; it is excluded from all method denominators.
  run_one "${block}" VideoUnmaskSwap 15 B 16
  if [[ "${block}" == "block1" ]]; then
    order=(
      VideoUnmaskSwap:20:A VideoUnmaskSwap:20:B VideoUnmaskSwap:20:C
      VideoUnmaskSwap:21:C VideoUnmaskSwap:21:B VideoUnmaskSwap:21:A
      VideoUnmask:20:A VideoUnmask:20:B
      VideoUnmask:21:B VideoUnmask:21:A
    )
  else
    order=(
      VideoUnmaskSwap:20:C VideoUnmaskSwap:20:B VideoUnmaskSwap:20:A
      VideoUnmaskSwap:21:A VideoUnmaskSwap:21:B VideoUnmaskSwap:21:C
      VideoUnmask:20:B VideoUnmask:20:A
      VideoUnmask:21:A VideoUnmask:21:B
    )
  fi
  for entry in "${order[@]}"; do
    IFS=: read -r task episode arm <<< "${entry}"
    run_one "${block}" "${task}" "${episode}" "${arm}" 1300
  done
  stop_services
  if [[ "${block}" == "block1" ]]; then
    for _ in $(seq 1 30); do
      if ! ss -ltn | rg -q ':8011 |:18070 '; then
        break
      fi
      sleep 2
    done
    if ss -ltn | rg -q ':8011 |:18070 '; then
      echo "model port remained occupied after block1 shutdown" >&2
      exit 2
    fi
  fi
done
