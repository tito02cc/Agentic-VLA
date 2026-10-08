#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  cat >&2 <<'USAGE'
Usage: run_robodojo_starvla_nominal.sh <B0|C1|C2|C3|shadow> <task> <layout_set> [artifact_dir]

  layout_set  Official RoboDojo layout-set index (0, 1 or 2). RoboDojo's own CLI
              calls this a "seed", but it selects a packaged layout directory
              under Assets/Eval_Layout/RoboDojo/arx_x5/<layout_set>/ rather than
              an RNG seed. It also becomes the default request-level action seed
              base; override that independently with CARVE_ACTION_SEED_BASE.

Episode count:
  CARVE_EVAL_NUM   Episodes (official layouts) to evaluate, taken in ascending
                   layout_id order starting at 0. Default 1, which reproduces
                   the historical single-episode mechanism runs. Use "native"
                   for the official per-task count from RoboDojo's _task.yml
                   (stack_bowls 25, build_tower 50, put_bottles 50). RoboDojo
                   clamps any numeric value to that official count.

Reset audit:
  ROBODOJO_REQUIRE_AUDITED_RESET defaults to 1. Explicit 0 is accepted only for
                                   legacy diagnostics and is recorded as false.
USAGE
  exit 2
fi

condition=${1^^}
task=$2
seed=$3
eval_num=${CARVE_EVAL_NUM:-1}
if [[ "${eval_num}" != "native" && ! "${eval_num}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CARVE_EVAL_NUM must be a positive integer or 'native'; got: ${eval_num}" >&2
  exit 2
fi
require_audited_reset="${ROBODOJO_REQUIRE_AUDITED_RESET-1}"
case "${require_audited_reset}" in
  0|1)
    ;;
  *)
    echo "ROBODOJO_REQUIRE_AUDITED_RESET must be 0 or 1; got: ${require_audited_reset}" >&2
    exit 2
    ;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date +%Y%m%d)"
artifact_dir=${4:-"${repo_root}/artifacts/robodojo/nominal_${task}_${condition,,}_seed${seed}_${stamp}"}
artifact_dir="$(realpath -m "${artifact_dir}")"
layout_set_dir="${repo_root}/third_party/robodojo_official/Assets/Eval_Layout/RoboDojo/arx_x5/${seed}"

# RoboDojo calls this value a seed, but it selects a packaged layout set rather
# than an unrestricted RNG seed. Reject missing sets before loading the models.
if [[ ! -d "${layout_set_dir}" ]]; then
  available_sets=$(find "$(dirname "${layout_set_dir}")" -mindepth 1 -maxdepth 1 -type d -printf '%f ' | sort -V)
  echo "Missing official RoboDojo layout set ${seed}; available sets: ${available_sets}" >&2
  exit 2
fi

case "${condition}" in
  B0)
    carve_mode=baseline
    nominal_ddim_steps=4
    nominal_execute_horizon=16
    ;;
  C1)
    carve_mode=runtime
    nominal_ddim_steps=${CARVE_INFERENCE_STEPS:-${CARVE_DDIM_STEPS:-2}}
    # Cross-task qualification rejected h32 as a universal default. Longer
    # open-loop horizons must now be opted into after task-family admission.
    nominal_execute_horizon=${CARVE_EXECUTE_HORIZON:-16}
    ;;
  C2)
    carve_mode=agentic
    nominal_ddim_steps=4
    nominal_execute_horizon=16
    ;;
  C3)
    carve_mode=full
    nominal_ddim_steps=${CARVE_INFERENCE_STEPS:-${CARVE_DDIM_STEPS:-2}}
    nominal_execute_horizon=${CARVE_EXECUTE_HORIZON:-16}
    ;;
  SHADOW)
    carve_mode=shadow
    nominal_ddim_steps=4
    nominal_execute_horizon=16
    ;;
  *)
    echo "Unknown condition: ${condition}; choose B0, C1, C2, C3 or shadow." >&2
    exit 2
    ;;
esac

recovery_ddim_steps=${CARVE_RECOVERY_INFERENCE_STEPS:-${CARVE_RECOVERY_DDIM_STEPS:-${nominal_ddim_steps}}}
recovery_execute_horizon=${CARVE_RECOVERY_EXECUTE_HORIZON:-${nominal_execute_horizon}}
recovery_compute_calls=${CARVE_RECOVERY_COMPUTE_CALLS:-0}
if [[ "${condition}" == "C2" ]]; then
  recovery_ddim_steps=${CARVE_RECOVERY_INFERENCE_STEPS:-${CARVE_RECOVERY_DDIM_STEPS:-4}}
  recovery_execute_horizon=${CARVE_RECOVERY_EXECUTE_HORIZON:-16}
  recovery_compute_calls=${CARVE_RECOVERY_COMPUTE_CALLS:-2}
elif [[ "${condition}" == "C3" ]]; then
  recovery_ddim_steps=${CARVE_RECOVERY_INFERENCE_STEPS:-${CARVE_RECOVERY_DDIM_STEPS:-4}}
  recovery_execute_horizon=${CARVE_RECOVERY_EXECUTE_HORIZON:-16}
  recovery_compute_calls=${CARVE_RECOVERY_COMPUTE_CALLS:-2}
fi

if [[ -e "${artifact_dir}" ]]; then
  echo "Refusing to overwrite existing artifact: ${artifact_dir}" >&2
  exit 1
fi

available_layouts=$(find "${layout_set_dir}" -maxdepth 1 -type f \
  -regextype posix-extended -regex ".*/${task}_[0-9]+\.json" -printf '.' | wc -c)
if [[ "${available_layouts}" -eq 0 ]]; then
  echo "Layout set ${seed} contains no '${task}_<n>.json' layouts." >&2
  exit 2
fi

policy_env=${STARVLA_POLICY_ENV:-/home/admin1/miniconda3/envs/StarVLA}
sim_env=${ROBODOJO_SIM_ENV:-/home/admin1/miniconda3/envs/RoboDojo}
base_vlm=${STARVLA_BASE_VLM:-/home/admin1/models/Qwen3-VL-4B-Instruct}
planner_model=${CARVE_PLANNER_MODEL:-}
planner_endpoint=${CARVE_PLANNER_ENDPOINT:-}
semantic_role=${CARVE_SEMANTIC_ROLE:-planner}
semantic_recovery_mode=${CARVE_SEMANTIC_RECOVERY_MODE:-task_preserving}
planner_backend=${CARVE_PLANNER_BACKEND:-task_preserving}
staged_residency=${STARVLA_STAGED_RESIDENCY:-transfer}
staged_warmup=${CARVE_STAGED_WARMUP:-0}
case "${staged_warmup}" in
  0|1) ;;
  *) echo "CARVE_STAGED_WARMUP must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${staged_warmup}" == "1" && "${planner_backend}" != "staged_vlm" ]]; then
  echo "Startup preparation requires staged_vlm" >&2
  exit 2
fi
if [[ "${staged_warmup}" == "1" ]]; then
  export STARVLA_STAGED_WARMUP_RECEIPT="${artifact_dir}/startup_preparation.json"
else
  unset STARVLA_STAGED_WARMUP_RECEIPT
fi
case "${staged_residency}" in
  transfer|cpu_mirror) ;;
  *) echo "STARVLA_STAGED_RESIDENCY must be transfer or cpu_mirror" >&2; exit 2 ;;
esac
if [[ "${staged_residency}" == "cpu_mirror" && "${planner_backend}" != "staged_vlm" ]]; then
  echo "cpu_mirror requires the staged_vlm backend" >&2
  exit 2
fi
if [[ "${planner_backend}" == "staged_vlm" ]]; then
  if [[ "${condition}" != "C2" && "${condition}" != "C3" ]] &&
     [[ "${condition}" != "SHADOW" || -z "${CARVE_REOBSERVATION_SHADOW_CONFIG:-}" ]]; then
    echo "Staged VLM requires C2/C3 or explicit reobservation SHADOW configuration." >&2
    exit 2
  fi
  if [[ -z "${planner_model}" || ! -d "${planner_model}" ]]; then
    echo "Staged VLM requires CARVE_PLANNER_MODEL pointing to a local model directory." >&2
    exit 2
  fi
  export STARVLA_STAGED_VLM_PATH="${planner_model}"
else
  unset STARVLA_STAGED_VLM_PATH
fi
planner_replay_path=${CARVE_PLANNER_REPLAY_PATH:-}
semantic_predicate_id=${CARVE_SEMANTIC_PREDICATE_ID:-}
semantic_predicate_question=${CARVE_SEMANTIC_PREDICATE_QUESTION:-}
semantic_predicate_comparison=${CARVE_SEMANTIC_PREDICATE_COMPARISON:-eq}
semantic_predicate_target=${CARVE_SEMANTIC_PREDICATE_TARGET:-1}
semantic_predicate_unit=${CARVE_SEMANTIC_PREDICATE_UNIT:-count}
semantic_predicate_tolerance=${CARVE_SEMANTIC_PREDICATE_TOLERANCE:-0}
semantic_expected_object_count=${CARVE_SEMANTIC_EXPECTED_OBJECT_COUNT:-}
semantic_extraction_mode=${CARVE_SEMANTIC_EXTRACTION_MODE:-scalar}
semantic_checkpoint_steps=${CARVE_SEMANTIC_CHECKPOINT_STEPS:-0}
semantic_deadline_step=${CARVE_SEMANTIC_DEADLINE_STEP:-0}
semantic_stale_checks=${CARVE_SEMANTIC_STALE_CHECKS:-2}
semantic_progress_only=${CARVE_SEMANTIC_PROGRESS_ONLY:-false}
semantic_max_calls_per_episode=${CARVE_SEMANTIC_MAX_CALLS_PER_EPISODE:-3}
semantic_event_reserve_calls=${CARVE_SEMANTIC_EVENT_RESERVE_CALLS:-0}
semantic_max_recoveries=${CARVE_SEMANTIC_MAX_RECOVERIES:-1}
semantic_guidance_calls=${CARVE_SEMANTIC_GUIDANCE_CALLS:-2}
task_plan_mode=${CARVE_TASK_PLAN_MODE:-disabled}
# Local ports can be reused by unrelated services. Never silently choose a
# Planner endpoint/model for an online Agent condition.
if [[ ( "${condition}" == "C2" || "${condition}" == "C3" ) &&
      "${planner_backend}" == "openai" && -z "${planner_replay_path}" &&
      ( -z "${planner_endpoint}" || -z "${planner_model}" ) ]]; then
  echo "Online Planner requires explicit CARVE_PLANNER_ENDPOINT and CARVE_PLANNER_MODEL." >&2
  exit 2
fi
procedure_memory_manifest=${CARVE_PROCEDURE_MEMORY_MANIFEST:-}
procedure_memory_sha256=${CARVE_PROCEDURE_MEMORY_SHA256:-}
if [[ -n "${procedure_memory_manifest}" ]]; then
  if [[ "${condition}" != "C2" && "${condition}" != "C3" ]]; then
    echo "Procedure memory requires an Agent condition (C2/C3)." >&2
    exit 2
  fi
  if [[ "${planner_backend}" == "task_preserving" || -n "${planner_replay_path}" ]]; then
    echo "Procedure memory requires a live semantic planner, not deterministic recovery/replay." >&2
    exit 2
  fi
  if [[ ! "${procedure_memory_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "CARVE_PROCEDURE_MEMORY_SHA256 must pin the memory manifest." >&2
    exit 2
  fi
fi
task_plan_verifier_backend=${CARVE_TASK_PLAN_VERIFIER_BACKEND:-auto}
semantic_image_profile=${CARVE_SEMANTIC_IMAGE_PROFILE:-vla_224}
inspection_shadow=${CARVE_INSPECTION_SHADOW:-false}
case "${inspection_shadow}" in
  true|false) ;;
  *) echo "CARVE_INSPECTION_SHADOW must be true or false" >&2; exit 2 ;;
esac
case "${semantic_image_profile}" in
  vla_224|head_native) ;;
  multiview_native)
    if [[ "${carve_mode}" != "shadow" ]]; then
      echo "multiview_native is observation-only; use the shadow condition until admission" >&2
      exit 2
    fi
    ;;
  *) echo "CARVE_SEMANTIC_IMAGE_PROFILE must be vla_224, head_native or multiview_native" >&2; exit 2 ;;
esac
vla_instruction_capability=${CARVE_VLA_INSTRUCTION_CAPABILITY:-task_only}
task_plan_checkpoint_steps=${CARVE_TASK_PLAN_CHECKPOINT_STEPS:-128}
task_plan_max_checks=${CARVE_TASK_PLAN_MAX_CHECKS:-4}
task_plan_check_schedule=${CARVE_TASK_PLAN_CHECK_SCHEDULE:-}
if [[ ! "$task_plan_check_schedule" =~ ^[0-9:]*$ ]]; then
  echo "CARVE_TASK_PLAN_CHECK_SCHEDULE must contain only digits and colons" >&2
  exit 2
fi
task_plan_max_attempts_per_stage=${CARVE_TASK_PLAN_MAX_ATTEMPTS_PER_STAGE:-2}
task_plan_stale_checks=${CARVE_TASK_PLAN_STALE_CHECKS:-2}
task_plan_max_protocol_failures=${CARVE_TASK_PLAN_MAX_PROTOCOL_FAILURES:-1}
planner_max_tokens=${CARVE_PLANNER_MAX_TOKENS:-192}
semantic_safe_stop_enabled=${CARVE_SEMANTIC_SAFE_STOP_ENABLED:-false}
action_seed_base=${CARVE_ACTION_SEED_BASE:-${seed}}
fault_hold_step=${CARVE_FAULT_HOLD_STEP:--1}

# Per-request evidence for the native (no-CarveRuntime) policy path. Opt-in, so
# baseline runs recorded before this switch existed keep the same measurement
# scope; when enabled it appends to the same runtime_trace.jsonl the wrapped
# arms use, which is what makes the arms directly comparable.
native_request_trace=${CARVE_NATIVE_REQUEST_TRACE:-false}
case "${native_request_trace}" in
  true|false) ;;
  *) echo "CARVE_NATIVE_REQUEST_TRACE must be true or false" >&2; exit 2 ;;
esac
if [[ "${native_request_trace}" == "true" ]]; then
  native_trace_path="${artifact_dir}/runtime_trace.jsonl"
else
  native_trace_path=""
fi

# Bounded real-payload capture, for replaying the recorded observations through
# both the native and the wrapped path inside one process. Requires the trace,
# because each captured payload is keyed by the digests the trace computes.
native_payload_capture=${CARVE_NATIVE_PAYLOAD_CAPTURE:-false}
case "${native_payload_capture}" in
  true|false) ;;
  *) echo "CARVE_NATIVE_PAYLOAD_CAPTURE must be true or false" >&2; exit 2 ;;
esac
native_payload_capture_max=${CARVE_NATIVE_PAYLOAD_CAPTURE_MAX:-16}
if [[ "${native_payload_capture}" == "true" ]]; then
  if [[ "${native_request_trace}" != "true" ]]; then
    echo "CARVE_NATIVE_PAYLOAD_CAPTURE requires CARVE_NATIVE_REQUEST_TRACE=true" >&2
    exit 2
  fi
  native_payload_capture_dir="${artifact_dir}/captured_payloads"
else
  native_payload_capture_dir=""
fi

reset_receipt_schema=xpolicylab.episode-reset.v1
state_inventory_schema=starvla.episode-state.v1
model_reset_schema=carve.policy-reset.v1
expected_model_module_id=XPolicyLab.policy.starVLA.model
policy_reset_ledger_filename=policy_reset_ledger.json
reset_receipts_filename=reset_receipts.jsonl
starvla_model_path="${repo_root}/third_party/robodojo_official/XPolicyLab/policy/starVLA/model.py"
if [[ ! -f "${starvla_model_path}" ]]; then
  echo "Missing StarVLA model source for reset audit: ${starvla_model_path}" >&2
  exit 1
fi
expected_model_code_sha256=$(sha256sum -- "${starvla_model_path}" | cut -d' ' -f1)
if [[ ! "${expected_model_code_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Failed to compute StarVLA model SHA256: ${starvla_model_path}" >&2
  exit 1
fi
# The reset inventory lives here, outside the model.py digest, and the
# finalizer imports this same file to build its expectations. Pin it too.
reset_contract_path="${repo_root}/third_party/robodojo_official/XPolicyLab/policy/starVLA/reset_contract.py"
if [[ ! -f "${reset_contract_path}" ]]; then
  echo "Missing StarVLA reset contract for reset audit: ${reset_contract_path}" >&2
  exit 1
fi
expected_reset_contract_sha256=$(sha256sum -- "${reset_contract_path}" | cut -d' ' -f1)
if [[ ! "${expected_reset_contract_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Failed to compute reset contract SHA256: ${reset_contract_path}" >&2
  exit 1
fi
# Fix the run id here rather than letting the client generate one, so the
# artifact records which official run directory its evidence must come from.
# The official output tree is shared across conditions, so without this a
# condition that exits early can be finalized against another condition's run.
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
if [[ "${require_audited_reset}" == "1" ]]; then
  require_audited_reset_json=true
else
  require_audited_reset_json=false
  echo "WARNING: ROBODOJO_REQUIRE_AUDITED_RESET=0 is for legacy diagnostics only." >&2
fi

mkdir -p "${artifact_dir}"

# Record the resolved launch parameters next to the traces. Condition labels are
# only trustworthy if the values that actually reached the runtime are auditable.
cat > "${artifact_dir}/run_config.json" <<CONFIG
{
  "schema_version": "carve.robodojo.run-config.v1",
  "condition": "${condition}",
  "task": "${task}",
  "layout_set": ${seed},
  "requested_eval_num": "${eval_num}",
  "available_layouts_in_set": ${available_layouts},
  "carve_mode": "${carve_mode}",
  "nominal_inference_steps": ${nominal_ddim_steps},
  "nominal_execute_horizon": ${nominal_execute_horizon},
  "recovery_inference_steps": ${recovery_ddim_steps},
  "recovery_execute_horizon": ${recovery_execute_horizon},
  "recovery_compute_calls": ${recovery_compute_calls},
  "action_seed_base": ${action_seed_base},
  "fault_hold_step": ${fault_hold_step},
  "planner_backend": "${planner_backend}",
  "staged_residency": "${staged_residency}",
  "staged_warmup": ${staged_warmup},
  "planner_endpoint": "${planner_endpoint}",
  "planner_model": "${planner_model}",
  "planner_max_tokens": ${planner_max_tokens},
  "task_plan_verifier_backend": "${task_plan_verifier_backend}",
  "semantic_image_profile": "${semantic_image_profile}",
  "inspection_shadow": ${inspection_shadow},
  "native_request_trace": ${native_request_trace},
  "native_trace_path": "${native_trace_path}",
  "native_payload_capture": ${native_payload_capture},
  "native_payload_capture_dir": "${native_payload_capture_dir}",
  "native_payload_capture_max": ${native_payload_capture_max},
  "planner_max_images": "${CARVE_PLANNER_MAX_IMAGES:-3}",
  "task_plan_checkpoint_steps": ${task_plan_checkpoint_steps},
  "task_plan_max_checks": ${task_plan_max_checks},
  "task_plan_check_schedule": "${task_plan_check_schedule}",
  "semantic_role": "${semantic_role}",
  "semantic_safe_stop_enabled": ${semantic_safe_stop_enabled},
  "task_plan_mode": "${task_plan_mode}",
  "procedure_memory_manifest": "${procedure_memory_manifest}",
  "procedure_memory_sha256": "${procedure_memory_sha256}",
  "vla_instruction_capability": "${vla_instruction_capability}",
  "require_audited_reset": ${require_audited_reset_json},
  "reset_receipt_schema": "${reset_receipt_schema}",
  "expected_reset_receipt_schema_version": "${reset_receipt_schema}",
  "state_inventory_schema": "${state_inventory_schema}",
  "expected_state_inventory_version": "${state_inventory_schema}",
  "model_reset_schema": "${model_reset_schema}",
  "expected_model_reset_schema_version": "${model_reset_schema}",
  "expected_model_module_id": "${expected_model_module_id}",
  "expected_model_code_sha256": "${expected_model_code_sha256}",
  "expected_reset_contract_sha256": "${expected_reset_contract_sha256}",
  "robodojo_run_id": "${ROBODOJO_RUN_ID}",
  "policy_reset_ledger_filename": "${policy_reset_ledger_filename}",
  "reset_receipts_filename": "${reset_receipts_filename}",
  "layout_set_dir": "${layout_set_dir}"
}
CONFIG

cd "${repo_root}"
env \
  PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID}" \
  ROBODOJO_REQUIRE_AUDITED_RESET="${require_audited_reset}" \
  STARVLA_HF_LOCAL_FILES_ONLY=1 \
  STARVLA_HF_SKIP_WEIGHT_HASH=1 \
  STARVLA_BASE_VLM="${base_vlm}" \
  STARVLA_STAGED_RESIDENCY="${staged_residency}" \
  STARVLA_EXECUTE_HORIZON="${nominal_execute_horizon}" \
  STARVLA_ACTION_SEED_BASE="${action_seed_base}" \
  STARVLA_CARVE_MODE="${carve_mode}" \
  STARVLA_CARVE_NUM_DDIM_STEPS="${nominal_ddim_steps}" \
  STARVLA_CARVE_EXECUTE_HORIZON="${nominal_execute_horizon}" \
  STARVLA_CARVE_RECOVERY_NUM_DDIM_STEPS="${recovery_ddim_steps}" \
  STARVLA_CARVE_RECOVERY_EXECUTE_HORIZON="${recovery_execute_horizon}" \
  STARVLA_CARVE_RECOVERY_COMPUTE_CALLS="${recovery_compute_calls}" \
  STARVLA_CARVE_TRACE_PATH="${artifact_dir}/runtime_trace.jsonl" \
  STARVLA_CARVE_MONITOR_TRACE_PATH="${artifact_dir}/monitor_trace.jsonl" \
  STARVLA_CARVE_NATIVE_TRACE_PATH="${native_trace_path}" \
  STARVLA_CARVE_NATIVE_PAYLOAD_CAPTURE_DIR="${native_payload_capture_dir}" \
  STARVLA_CARVE_NATIVE_PAYLOAD_CAPTURE_MAX="${native_payload_capture_max}" \
  STARVLA_CARVE_PLANNER_ENDPOINT="${planner_endpoint}" \
  STARVLA_CARVE_PLANNER_MODEL="${planner_model}" \
  STARVLA_CARVE_PLANNER_BACKEND="${planner_backend}" \
  STARVLA_CARVE_PLANNER_REPLAY_PATH="${planner_replay_path}" \
  STARVLA_CARVE_SEMANTIC_ROLE="${semantic_role}" \
  STARVLA_CARVE_SEMANTIC_RECOVERY_MODE="${semantic_recovery_mode}" \
  STARVLA_CARVE_PLANNER_TRACE_PATH="${artifact_dir}/planner_trace.jsonl" \
  STARVLA_CARVE_PLANNER_TIMEOUT_S=90 \
  STARVLA_CARVE_PLANNER_MAX_TOKENS="${planner_max_tokens}" \
  STARVLA_CARVE_PLANNER_MAX_IMAGES="${CARVE_PLANNER_MAX_IMAGES:-3}" \
  STARVLA_CARVE_PLANNER_MAX_CALLS_PER_EPISODE=3 \
  STARVLA_CARVE_PLANNER_COOLDOWN_STEPS=128 \
  STARVLA_CARVE_SEMANTIC_PREDICATE_ID="${semantic_predicate_id}" \
  STARVLA_CARVE_SEMANTIC_PREDICATE_QUESTION="${semantic_predicate_question}" \
  STARVLA_CARVE_SEMANTIC_PREDICATE_COMPARISON="${semantic_predicate_comparison}" \
  STARVLA_CARVE_SEMANTIC_PREDICATE_TARGET="${semantic_predicate_target}" \
  STARVLA_CARVE_SEMANTIC_PREDICATE_UNIT="${semantic_predicate_unit}" \
  STARVLA_CARVE_SEMANTIC_PREDICATE_TOLERANCE="${semantic_predicate_tolerance}" \
  STARVLA_CARVE_SEMANTIC_EXPECTED_OBJECT_COUNT="${semantic_expected_object_count}" \
  STARVLA_CARVE_SEMANTIC_EXTRACTION_MODE="${semantic_extraction_mode}" \
  STARVLA_CARVE_SEMANTIC_CHECKPOINT_STEPS="${semantic_checkpoint_steps}" \
  STARVLA_CARVE_SEMANTIC_DEADLINE_STEP="${semantic_deadline_step}" \
  STARVLA_CARVE_SEMANTIC_STALE_CHECKS="${semantic_stale_checks}" \
  STARVLA_CARVE_SEMANTIC_MIN_STALL_INTERVAL_STEPS="${CARVE_SEMANTIC_MIN_STALL_INTERVAL_STEPS:-}" \
  STARVLA_CARVE_SEMANTIC_OBSERVATION_VALIDITY_STEPS="${CARVE_SEMANTIC_OBSERVATION_VALIDITY_STEPS:-}" \
  STARVLA_CARVE_SEMANTIC_MIN_UNCHANGED_STEPS="${CARVE_SEMANTIC_MIN_UNCHANGED_STEPS:-}" \
  STARVLA_CARVE_SEMANTIC_PROGRESS_ONLY="${semantic_progress_only}" \
  STARVLA_CARVE_SEMANTIC_MAX_CALLS_PER_EPISODE="${semantic_max_calls_per_episode}" \
  STARVLA_CARVE_SEMANTIC_EVENT_RESERVE_CALLS="${semantic_event_reserve_calls}" \
  STARVLA_CARVE_SEMANTIC_MAX_RECOVERIES="${semantic_max_recoveries}" \
  STARVLA_CARVE_SEMANTIC_GUIDANCE_CALLS="${semantic_guidance_calls}" \
  STARVLA_CARVE_SEMANTIC_SAFE_STOP_ENABLED="${semantic_safe_stop_enabled}" \
  STARVLA_CARVE_TASK_PLAN_MODE="${task_plan_mode}" \
  STARVLA_CARVE_PROCEDURE_MEMORY_MANIFEST="${procedure_memory_manifest}" \
  STARVLA_CARVE_PROCEDURE_MEMORY_SHA256="${procedure_memory_sha256}" \
  STARVLA_CARVE_TASK_PLAN_VERIFIER_BACKEND="${task_plan_verifier_backend}" \
  STARVLA_CARVE_SEMANTIC_IMAGE_PROFILE="${semantic_image_profile}" \
  STARVLA_CARVE_INSPECTION_SHADOW="${inspection_shadow}" \
  STARVLA_CARVE_VLA_INSTRUCTION_CAPABILITY="${vla_instruction_capability}" \
  STARVLA_CARVE_TASK_PLAN_CHECKPOINT_STEPS="${task_plan_checkpoint_steps}" \
  STARVLA_CARVE_TASK_PLAN_MAX_CHECKS="${task_plan_max_checks}" \
  STARVLA_CARVE_TASK_PLAN_CHECK_SCHEDULE="${task_plan_check_schedule}" \
  STARVLA_CARVE_TASK_PLAN_MAX_ATTEMPTS_PER_STAGE="${task_plan_max_attempts_per_stage}" \
  STARVLA_CARVE_TASK_PLAN_STALE_CHECKS="${task_plan_stale_checks}" \
  STARVLA_CARVE_TASK_PLAN_MAX_PROTOCOL_FAILURES="${task_plan_max_protocol_failures}" \
  STARVLA_CARVE_FAULT_HOLD_STEP="${fault_hold_step}" \
  bash third_party/robodojo_official/XPolicyLab/policy/starVLA/scripts/eval_hf_robodojo.sh \
    pi_v3 "${task}" "${seed}" 0 0 "${policy_env}" "${sim_env}" "${eval_num}" \
  2>&1 | tee "${artifact_dir}/run.log"
