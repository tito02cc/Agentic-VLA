#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/robomme/unmaskswap_fixed_five_20260923"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"

if [[ ! -d "${OUTPUT_ROOT}/memory" ]]; then
  echo "missing precompiled identity memory" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/rollouts"

for entry in 15:A 15:B 16:B 16:A 17:A 17:B 18:B 18:A 19:A 19:B; do
  episode="${entry%:*}"
  arm="${entry#*:}"
  name="VideoUnmaskSwap_ep${episode}_${arm}"
  output="${OUTPUT_ROOT}/rollouts/${name}"
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite ${output}" >&2
    exit 2
  fi
  memory_arg=""
  if [[ "${arm}" == "B" ]]; then
    memory="${OUTPUT_ROOT}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
    if [[ ! -s "${memory}" ]]; then
      echo "missing memory for episode ${episode}" >&2
      exit 2
    fi
    memory_arg="--task-memory-file /workspace/artifacts/robomme/unmaskswap_fixed_five_20260923/memory/VideoUnmaskSwap_ep${episode}_memory.json --memory-hint-style precise --memory-use-policy always"
  fi

  echo "starting ${name}"
  docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" \
    bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_vlm_groundsg.py \
      --task VideoUnmaskSwap --episode ${episode} --max-steps 1300 --action-horizon 16 \
      --planner-profile carve --planner-context native --grounding-authority observe_only \
      --procedure-authority observe_only --execution-feedback execution_chunks \
      --planner-demo-mode always --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-schedule every_chunk --planner-gripper-change-threshold 0.004 \
      --planner-max-reusable-points 0 ${memory_arg} \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output /workspace/artifacts/robomme/unmaskswap_fixed_five_20260923/rollouts/${name}" \
    > "${OUTPUT_ROOT}/rollouts/${name}.stdout.log" 2>&1
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("finished",d["task"],d["episode"],"success",d["success"],"status",d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],flush=True)' "${output}/summary.json"
done
