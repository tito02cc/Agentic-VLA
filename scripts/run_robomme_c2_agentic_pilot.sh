#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_c2_agentic_every_chunk_20260831}"
TASKS="${TASKS:-StopCube VideoRepick RouteStick VideoUnmaskSwap}"
EPISODES="${EPISODES:-0 1 2 3 4}"
PLANNER_SCHEDULE="${PLANNER_SCHEDULE:-every_chunk}"
STOPCUBE_MEMORY_ROOT="${STOPCUBE_MEMORY_ROOT:-}"
VIDEOREPICK_MEMORY_ROOT="${VIDEOREPICK_MEMORY_ROOT:-}"
UNMASK_MEMORY_ROOT="${UNMASK_MEMORY_ROOT:-}"

memory_file_for() {
  local task="$1"
  local episode="$2"
  case "${task}" in
    StopCube)
      if [[ -n "${STOPCUBE_MEMORY_ROOT}" ]]; then
        printf '%s\n' "${STOPCUBE_MEMORY_ROOT}/${task}_ep${episode}_memory.json"
      else
        printf '%s\n' "${PROJECT_ROOT}/results/robomme_pilot_memory_hints_admitted_20260831/${task}_ep${episode}_memory.json"
      fi
      ;;
    VideoRepick)
      if [[ -n "${VIDEOREPICK_MEMORY_ROOT}" ]]; then
        printf '%s\n' "${VIDEOREPICK_MEMORY_ROOT}/${task}_ep${episode}_memory.json"
        return
      fi
      case "${episode}" in
        0|1|2)
          printf '%s\n' "${PROJECT_ROOT}/results/robomme_video_instance_memory_sam2_admitted_20260829/${task}_ep${episode}_memory.json"
          ;;
        3)
          printf '%s\n' "${PROJECT_ROOT}/results/robomme_pilot_memory_hints_v3_20260831/${task}_ep${episode}_memory.json"
          ;;
        4)
          printf '%s\n' "${PROJECT_ROOT}/results/robomme_video_instance_memory_motion_sam2_20260831/${task}_ep${episode}_memory.json"
          ;;
      esac
      ;;
    VideoUnmaskSwap)
      if [[ -n "${UNMASK_MEMORY_ROOT}" ]]; then
        printf '%s\n' "${UNMASK_MEMORY_ROOT}/${task}_ep${episode}_memory.json"
      else
        printf '%s\n' "${PROJECT_ROOT}/results/robomme_unmask_swap_sam2_memory_20260831/${task}_ep${episode}_memory.json"
      fi
      ;;
    RouteStick)
      ;;
    BinFill|PickXtimes|ButtonUnmask|PickHighlight|MoveCube)
      ;;
    *)
      echo "unsupported C2 pilot task: ${task}" >&2
      return 2
      ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}"

for task in ${TASKS}; do
  for episode in ${EPISODES}; do
    episode_output="${OUTPUT_ROOT}/${task}_ep${episode}"
    if [[ -s "${episode_output}/summary.json" ]]; then
      echo "skip completed ${task} episode ${episode}"
      continue
    fi

    memory_args=()
    memory_file="$(memory_file_for "${task}" "${episode}")"
    if [[ -n "${memory_file}" ]]; then
      if [[ ! -s "${memory_file}" ]]; then
        echo "missing task memory: ${memory_file}" >&2
        exit 2
      fi
      memory_args=(--task-memory-file "/workspace/${memory_file#${PROJECT_ROOT}/}")
    fi

    grounding_args=()
    button_args=()
    gripper_threshold="0.15"
    if [[ "${task}" == "StopCube" ]]; then
      button_args=(--button-grounding)
    elif [[ "${task}" == "VideoRepick" ]]; then
      grounding_args=(--relational-color-grounding)
      button_args=(--button-grounding)
      gripper_threshold="0.004"
    elif [[ "${task}" == "VideoUnmaskSwap" ]]; then
      gripper_threshold="0.004"
    elif [[ "${task}" == "PickXtimes" ]]; then
      grounding_args=(--relational-color-grounding)
      gripper_threshold="0.004"
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
        --planner-schedule '${PLANNER_SCHEDULE}' \
        --planner-min-reuse-chunks 1 --planner-max-reuse-chunks 2 \
        --planner-max-reusable-points 1 \
        --planner-visual-change-threshold 0.09 \
        --planner-gripper-change-threshold '${gripper_threshold}' \
        ${memory_args[*]} ${grounding_args[*]} ${button_args[*]} \
        --policy-label 'Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999' \
        --planner-label 'Qwen3-VL-4B-GroundSG-BF16+CARVE-${PLANNER_SCHEDULE}' \
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
