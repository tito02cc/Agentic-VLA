#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <shadow|fixed|adaptive|planner|planner_replay> <seed> [artifact_dir]" >&2
  exit 2
fi

condition=$1
seed=$2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date +%Y%m%d)"
artifact_dir=${3:-"${repo_root}/artifacts/robodojo/starvla_pi_v3_build_tower_c3_fault640_${condition}_seed${seed}_${stamp}"}
artifact_dir="$(realpath -m "${artifact_dir}")"
planner_replay_path=""

case "${condition}" in
  shadow)
    carve_mode=shadow
    semantic_role=critic
    recovery_ddim=6
    recovery_horizon=32
    recovery_calls=0
    ;;
  fixed)
    carve_mode=full
    semantic_role=critic
    recovery_ddim=6
    recovery_horizon=32
    recovery_calls=0
    ;;
  adaptive)
    carve_mode=full
    semantic_role=critic
    recovery_ddim=10
    recovery_horizon=16
    recovery_calls=4
    ;;
  planner)
    carve_mode=full
    semantic_role=planner
    recovery_ddim=6
    recovery_horizon=32
    recovery_calls=0
    ;;
  planner_replay)
    carve_mode=full
    semantic_role=planner
    recovery_ddim=6
    recovery_horizon=32
    recovery_calls=0
    planner_replay_path=${CARVE_PLANNER_REPLAY_PATH:-"${repo_root}/artifacts/robodojo/build_tower_planner_recovery_ticket_20260901.json"}
    ;;
  *)
    echo "Unknown condition: ${condition}; choose shadow, fixed, adaptive, planner or planner_replay." >&2
    exit 2
    ;;
esac

if [[ -e "${artifact_dir}/run.log" ]]; then
  echo "Refusing to overwrite existing artifact: ${artifact_dir}" >&2
  exit 1
fi
mkdir -p "${artifact_dir}"

policy_env=${STARVLA_POLICY_ENV:-/home/admin1/miniconda3/envs/StarVLA}
sim_env=${ROBODOJO_SIM_ENV:-/home/admin1/miniconda3/envs/RoboDojo}
base_vlm=${STARVLA_BASE_VLM:-/home/admin1/models/Qwen3-VL-4B-Instruct}
critic_model=${CARVE_CRITIC_MODEL:-/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B}
critic_endpoint=${CARVE_CRITIC_ENDPOINT:-http://127.0.0.1:18071/v1/chat/completions}

cd "${repo_root}"
env \
  STARVLA_HF_LOCAL_FILES_ONLY=1 \
  STARVLA_HF_SKIP_WEIGHT_HASH=1 \
  STARVLA_BASE_VLM="${base_vlm}" \
  STARVLA_CARVE_MODE="${carve_mode}" \
  STARVLA_CARVE_NUM_DDIM_STEPS=6 \
  STARVLA_CARVE_EXECUTE_HORIZON=32 \
  STARVLA_CARVE_RECOVERY_NUM_DDIM_STEPS="${recovery_ddim}" \
  STARVLA_CARVE_RECOVERY_EXECUTE_HORIZON="${recovery_horizon}" \
  STARVLA_CARVE_RECOVERY_COMPUTE_CALLS="${recovery_calls}" \
  STARVLA_CARVE_TRACE_PATH="${artifact_dir}/runtime_trace.jsonl" \
  STARVLA_CARVE_MONITOR_TRACE_PATH="${artifact_dir}/monitor_trace.jsonl" \
  STARVLA_CARVE_PLANNER_ENDPOINT="${critic_endpoint}" \
  STARVLA_CARVE_PLANNER_MODEL="${critic_model}" \
  STARVLA_CARVE_PLANNER_REPLAY_PATH="${planner_replay_path}" \
  STARVLA_CARVE_SEMANTIC_ROLE="${semantic_role}" \
  STARVLA_CARVE_PLANNER_TRACE_PATH="${artifact_dir}/planner_trace.jsonl" \
  STARVLA_CARVE_PLANNER_TIMEOUT_S=90 \
  STARVLA_CARVE_PLANNER_MAX_TOKENS=192 \
  STARVLA_CARVE_PLANNER_MAX_CALLS_PER_EPISODE=3 \
  STARVLA_CARVE_PLANNER_COOLDOWN_STEPS=128 \
  STARVLA_CARVE_FAULT_HOLD_STEP=640 \
  bash third_party/robodojo_official/XPolicyLab/policy/starVLA/scripts/eval_hf_robodojo.sh \
    pi_v3 build_tower "${seed}" 0 0 "${policy_env}" "${sim_env}" 1 \
  2>&1 | tee "${artifact_dir}/run.log"
