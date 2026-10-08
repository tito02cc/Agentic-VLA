#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_memory_pilot_adaptive_bf16_20260829}"
TASKS="${TASKS:-StopCube VideoUnmaskSwap VideoRepick RouteStick}"
EPISODES="${EPISODES:-0}"
PLANNER_SCHEDULE="${PLANNER_SCHEDULE:-selective}"
MEMORY_ROOT="${MEMORY_ROOT:-}"
RELATIONAL_COLOR_GROUNDING="${RELATIONAL_COLOR_GROUNDING:-0}"
BUTTON_GROUNDING="${BUTTON_GROUNDING:-0}"
SEMANTIC_TRANSITION_CONFIRMATIONS="${SEMANTIC_TRANSITION_CONFIRMATIONS:-1}"
PLANNER_MIN_REUSE_CHUNKS="${PLANNER_MIN_REUSE_CHUNKS:-1}"
PLANNER_MAX_REUSE_CHUNKS="${PLANNER_MAX_REUSE_CHUNKS:-2}"
PLANNER_VISUAL_CHANGE_THRESHOLD="${PLANNER_VISUAL_CHANGE_THRESHOLD:-0.09}"
PLANNER_GRIPPER_CHANGE_THRESHOLD="${PLANNER_GRIPPER_CHANGE_THRESHOLD:-0.15}"

mkdir -p "${OUTPUT_ROOT}"

for task in ${TASKS}; do
  for episode in ${EPISODES}; do
    episode_output="${OUTPUT_ROOT}/${task}_ep${episode}"
    if [[ -s "${episode_output}/summary.json" ]]; then
      echo "skip completed ${task} episode ${episode}"
      continue
    fi
    memory_args=()
    grounding_args=()
    button_grounding_args=()
    if [[ -n "${MEMORY_ROOT}" ]]; then
      memory_file="${MEMORY_ROOT}/${task}_ep${episode}_memory.json"
      if [[ ! -s "${memory_file}" ]]; then
        echo "missing structured memory: ${memory_file}" >&2
        exit 2
      fi
      memory_args=(--task-memory-file "/workspace/${memory_file#${PROJECT_ROOT}/}")
    fi
    if [[ "${RELATIONAL_COLOR_GROUNDING}" == "1" ]]; then
      grounding_args=(--relational-color-grounding)
    fi
    if [[ "${BUTTON_GROUNDING}" == "1" ]]; then
      button_grounding_args=(--button-grounding)
    fi
    docker run --rm --gpus all --network host \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
      -e SAPIEN_RENDER_DEVICE=cuda \
      -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
      -v "${PROJECT_ROOT}:/workspace" \
      -v "${POLICY_REPO}:/policy:ro" \
      "${IMAGE}" \
      bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_vlm_groundsg.py \
        --task '${task}' --episode '${episode}' --max-steps 1300 --action-horizon 16 \
        --planner-schedule '${PLANNER_SCHEDULE}' \
        --planner-min-reuse-chunks '${PLANNER_MIN_REUSE_CHUNKS}' \
        --planner-max-reuse-chunks '${PLANNER_MAX_REUSE_CHUNKS}' \
        --planner-max-reusable-points 1 \
        --semantic-transition-confirmations '${SEMANTIC_TRANSITION_CONFIRMATIONS}' \
        --planner-visual-change-threshold '${PLANNER_VISUAL_CHANGE_THRESHOLD}' \
        --planner-gripper-change-threshold '${PLANNER_GRIPPER_CHANGE_THRESHOLD}' \
        ${memory_args[*]} \
        ${grounding_args[*]} \
        ${button_grounding_args[*]} \
        --policy-label 'Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999' \
        --planner-label 'Qwen3-VL-4B-GroundSG-BF16+CARVE-${PLANNER_SCHEDULE}' \
        --output '/workspace/results/$(basename "${OUTPUT_ROOT}")/${task}_ep${episode}'"
  done
done

docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target
