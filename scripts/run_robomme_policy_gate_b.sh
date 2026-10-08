#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8011}"
POLICY_LABEL="${POLICY_LABEL:-Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/robomme_policy_gate_b_20260828}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"

# These fixed strings are official episode-0 initial subgoals used only to
# establish the GroundSG input format and policy protocol. They are never a
# CARVE method condition or a benchmark success result.
declare -A SUBGOALS=(
  [StopCube]='move to the top of the button at <58, 87> to prepare'
  [VideoUnmaskSwap]='pick up the container at <105, 87> that hides the green cube'
  [VideoRepick]='pick up the correct cube at <82, 140> for the first time'
  [RouteStick]='move to the nearest right target by circling around the stick clockwise'
)

mkdir -p "${OUTPUT_ROOT}"

for task in StopCube VideoUnmaskSwap VideoRepick RouteStick; do
  docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${POLICY_REPO}:/policy:ro" \
    -v "${OUTPUT_ROOT}:/output" \
    "${IMAGE}" \
    bash -lc "cd /workspace && /app/.venv/bin/python scripts/run_robomme_policy_admission.py \
      --host '${POLICY_HOST}' --port '${POLICY_PORT}' \
      --task '${task}' --episode 0 --max-steps 16 --action-horizon 16 \
      --policy-label '${POLICY_LABEL}' \
      --subgoal '${SUBGOALS[${task}]}' \
      --output '/output/${task}_ep0'"
done

# The official container runs as root. Restore ownership of retained evidence
# so subsequent summaries can be written without elevated privileges.
docker run --rm -v "${OUTPUT_ROOT}:/target" "${IMAGE}" \
  chown -R "$(id -u):$(id -g)" /target
