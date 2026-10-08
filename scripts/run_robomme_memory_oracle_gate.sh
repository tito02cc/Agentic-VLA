#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_memory_oracle_gate_20260829}"

mkdir -p "${OUTPUT_ROOT}"
for task in StopCube VideoUnmaskSwap VideoRepick RouteStick; do
  output="${OUTPUT_ROOT}/${task}_ep0"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip completed ${task} episode 0"
    continue
  fi
  docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" \
    bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_oracle_upper_bound.py \
      --task '${task}' --episode 0 --max-steps 1300 --action-horizon 16 \
      --policy-label 'Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999' \
      --output '/workspace/results/$(basename "${OUTPUT_ROOT}")/${task}_ep0'"
done

docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target
