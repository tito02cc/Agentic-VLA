#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_oracle_survey_20260831}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8011}"
POLICY_LABEL="${POLICY_LABEL:-Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999}"
MAX_STEPS="${MAX_STEPS:-1300}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
EPISODES="${EPISODES:-0}"

# The four tasks in the paired CARVE study are intentionally excluded. This
# survey gates the remaining official RoboMME tasks before broader evaluation.
DEFAULT_TASKS=(
  BinFill PickXtimes SwingXtimes
  ButtonUnmask VideoUnmask ButtonUnmaskSwap
  PickHighlight VideoPlaceButton VideoPlaceOrder
  MoveCube InsertPeg PatternLock
)

if [[ -n "${TASKS:-}" ]]; then
  read -r -a task_list <<< "${TASKS}"
else
  task_list=("${DEFAULT_TASKS[@]}")
fi
read -r -a episode_list <<< "${EPISODES}"

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' "${task_list[@]}" > "${OUTPUT_ROOT}/expected_tasks.txt"

for task in "${task_list[@]}"; do
  for episode in "${episode_list[@]}"; do
    output="${OUTPUT_ROOT}/${task}_ep${episode}"
    if [[ -s "${output}/summary.json" ]]; then
      echo "skip completed ${task} episode ${episode}"
      continue
    fi

    echo "run oracle gate: ${task} episode ${episode}"
    docker run --rm --gpus all --network host \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
      -e SAPIEN_RENDER_DEVICE=cuda \
      -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
      -v "${PROJECT_ROOT}:/workspace" \
      -v "${POLICY_REPO}:/policy:ro" \
      "${IMAGE}" \
      bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_oracle_upper_bound.py \
        --host '${POLICY_HOST}' --port '${POLICY_PORT}' \
        --task '${task}' --episode '${episode}' \
        --max-steps '${MAX_STEPS}' --action-horizon '${ACTION_HORIZON}' \
        --policy-label '${POLICY_LABEL}' \
        --output '/workspace/results/$(basename "${OUTPUT_ROOT}")/${task}_ep${episode}'"
  done
done

docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target

python "${PROJECT_ROOT}/scripts/summarize_robomme_oracle_survey.py" \
  --input-root "${OUTPUT_ROOT}"
