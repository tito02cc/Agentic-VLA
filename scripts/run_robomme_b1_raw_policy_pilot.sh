#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_b1_raw_policy_20260831}"
TASKS="${TASKS:-StopCube VideoRepick RouteStick VideoUnmaskSwap}"
EPISODES="${EPISODES:-0 1 2 3 4}"

mkdir -p "${OUTPUT_ROOT}"

for task in ${TASKS}; do
  for episode in ${EPISODES}; do
    episode_output="${OUTPUT_ROOT}/${task}_ep${episode}"
    if [[ -s "${episode_output}/summary.json" ]]; then
      echo "skip completed ${task} episode ${episode}"
      continue
    fi

    if ! docker run --rm --gpus all --network host \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
      -e SAPIEN_RENDER_DEVICE=cuda \
      -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
      -v "${PROJECT_ROOT}:/workspace" \
      -v "${POLICY_REPO}:/policy:ro" \
      "${IMAGE}" \
      bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_vlm_groundsg.py \
        --task '${task}' --episode '${episode}' --max-steps 1300 --action-horizon 16 \
        --planner-profile raw --planner-schedule every_chunk \
        --planner-repair-attempts 0 \
        --policy-label 'Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999' \
        --planner-label 'Qwen3-VL-4B-GroundSG-BF16-raw' \
        --output '/workspace/results/$(basename "${OUTPUT_ROOT}")/${task}_ep${episode}'"; then
      if [[ -s "${episode_output}/summary.json" ]]; then
        echo "recorded failed task outcome for ${task} episode ${episode}; continuing"
      else
        echo "infrastructure failure without summary for ${task} episode ${episode}" >&2
        exit 3
      fi
    fi
  done
done

docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target
