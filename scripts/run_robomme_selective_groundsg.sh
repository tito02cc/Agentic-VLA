#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_vlm_groundsg_adaptive_selective2_bf16_20260829}"
MAX_REUSE_CHUNKS="${MAX_REUSE_CHUNKS:-2}"
VISUAL_CHANGE_THRESHOLD="${VISUAL_CHANGE_THRESHOLD:-0.09}"
GRIPPER_CHANGE_THRESHOLD="${GRIPPER_CHANGE_THRESHOLD:-0.15}"
MAX_REUSABLE_POINTS="${MAX_REUSABLE_POINTS:-1}"

mkdir -p "${OUTPUT_ROOT}"

for episode in 0 1 2 3 4; do
  episode_output="${OUTPUT_ROOT}/MoveCube_ep${episode}"
  if [[ -s "${episode_output}/summary.json" ]]; then
    echo "skip completed MoveCube episode ${episode}"
    continue
  fi
  docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" \
    bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_vlm_groundsg.py \
      --task MoveCube --episode '${episode}' --max-steps 1300 --action-horizon 16 \
      --planner-schedule selective --planner-min-reuse-chunks 1 \
      --planner-max-reuse-chunks '${MAX_REUSE_CHUNKS}' \
      --planner-visual-change-threshold '${VISUAL_CHANGE_THRESHOLD}' \
      --planner-gripper-change-threshold '${GRIPPER_CHANGE_THRESHOLD}' \
      --planner-max-reusable-points '${MAX_REUSABLE_POINTS}' \
      --policy-label 'Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999' \
      --planner-label 'Qwen3-VL-4B-GroundSG-BF16+CARVE-adaptive-selective${MAX_REUSE_CHUNKS}' \
      --output '/workspace/results/$(basename "${OUTPUT_ROOT}")/MoveCube_ep${episode}'"
done

docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target
