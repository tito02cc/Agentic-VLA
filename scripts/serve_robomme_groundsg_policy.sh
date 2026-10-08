#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="${ROBOMME_POLICY_REPO:-${PROJECT_ROOT}/third_party/robomme_policy_learning}"
POLICY_PYTHON="${POLICY_PYTHON:-${PROJECT_ROOT}/openpi/.venv/bin/python}"
CHECKPOINT="${ROBOMME_CHECKPOINT:-${HOME}/.cache/carve-vla/checkpoints/robomme/mme_vla_suite/symbolic-grounded-subgoal/79999}"
PORT="${ROBOMME_POLICY_PORT:-8011}"
SEED="${ROBOMME_POLICY_SEED:-7}"

for required in \
  "${CHECKPOINT}/params" \
  "${CHECKPOINT}/assets/robomme/norm_stats.json" \
  "$(dirname "${CHECKPOINT}")/history_config.txt"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing RoboMME checkpoint component: ${required}" >&2
    exit 2
  fi
done

cd "${POLICY_REPO}"
export PYTHONPATH="${POLICY_REPO}/src:${POLICY_REPO}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

exec "${POLICY_PYTHON}" scripts/serve_policy.py \
  --seed "${SEED}" \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.dir "${CHECKPOINT}" \
  --policy.config mme_vla_suite
