#!/usr/bin/env bash
# Drive the pre-registered RoboDojo B0/C1/C2/C3 matrix to completion.
#
# Protocol: docs/plans/RAL_ROBODOJO_PREREGISTRATION_20260907.md
# Defaults reproduce the preregistered stack_bowls set-0 protocol; callers may
# override LAYOUT_SET, EPISODES and STAMP. Condition order remains fixed:
# B0 -> C1 -> C2 -> C3.
#
# Conditions already finished are skipped, so this is safe to re-run after an
# interruption. It stops at the first failure rather than continuing, because a
# partially OOM'd condition must be re-run in full per section 9.
set -uo pipefail

REPO=/home/admin1/ct/CARVE-VLA
TASK=stack_bowls
LAYOUT_SET=${LAYOUT_SET:-0}
EPISODES=${EPISODES:-25}
STAMP=${STAMP:-20260907}
if ! [[ "${LAYOUT_SET}" =~ ^[0-9]+$ ]]; then
  echo "LAYOUT_SET must be a non-negative integer; got: ${LAYOUT_SET}" >&2
  exit 2
fi
if ! [[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES must be a positive integer; got: ${EPISODES}" >&2
  exit 2
fi
export ROBODOJO_REQUIRE_AUDITED_RESET=1

PREREG="${REPO}/docs/plans/RAL_ROBODOJO_PREREGISTRATION_${STAMP}.md"
RESULTS="${REPO}/results/robodojo_condition_matrix_${STAMP}"
LOGDIR="${REPO}/artifacts/system/ral_matrix_logs_${STAMP}"
EVAL_ROOT="${REPO}/third_party/robodojo_official/eval_result/RoboDojo/${TASK}/starVLA/arx_x5"

mkdir -p "${LOGDIR}" "${RESULTS}"
cd "${REPO}"

artifact_dir() { echo "${REPO}/artifacts/robodojo/ral_matrix_${STAMP}_${TASK}_$(echo "$1" | tr 'A-Z' 'a-z')_set${LAYOUT_SET}"; }

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOGDIR}/driver.log"; }

newest_official_dir() {
  find "${EVAL_ROOT}" -maxdepth 2 -mindepth 2 -type d -name '2026-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

record_gpu() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.free \
    --format=csv,noheader > "$1"
}

finalize_condition() {
  local cond=$1 adir=$2 official=$3
  if [[ ! -d "${official}" ]]; then
    log "  FAIL ${cond}: cannot locate official run dir"
    return 1
  fi
  log "  finalizing ${cond} from $(basename "${official}") with strict reset audit"
  PYTHONPATH=. "${REPO}/openpi/.venv/bin/python" scripts/finalize_robodojo_nominal_run.py \
    --condition "${cond}" --task "${TASK}" --seed "${LAYOUT_SET}" \
    --artifact "${adir}" --official-run "${official}" \
    --require-valid-reset-audit \
    --expected-episodes "${EPISODES}" \
    >> "${LOGDIR}/${cond}_finalize.log" 2>&1
}

validate_existing_summary() {
  local summary_path=$1
  "${REPO}/openpi/.venv/bin/python" -c '
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
try:
    with open(path, encoding="utf-8") as fp:
        summary = json.load(fp)
except Exception as exc:
    print(f"cannot read summary: {exc}")
    raise SystemExit(1)

audit = summary.get("policy_reset_audit")
completeness = summary.get("run_completeness")
aggregate = summary.get("aggregate")
if not isinstance(audit, dict) or audit.get("valid") is not True:
    print("summary.policy_reset_audit.valid is not true")
    raise SystemExit(1)
if not isinstance(completeness, dict) or completeness.get("valid") is not True:
    print("summary.run_completeness.valid is not true")
    raise SystemExit(1)
declared = completeness.get("expected_episodes")
if declared != expected:
    # Only double quotes inside this snippet: it is passed through a
    # single-quoted shell argument, so a single quote here would terminate it.
    print(f"summary.run_completeness.expected_episodes={declared!r}, expected {expected}")
    raise SystemExit(1)
if not isinstance(aggregate, dict):
    print("summary.aggregate is missing")
    raise SystemExit(1)
episodes = aggregate.get("episodes")
if type(episodes) is not int or episodes != expected:
    print(f"summary.aggregate.episodes={episodes!r}, expected {expected}")
    raise SystemExit(1)
print(f"valid reset audit and {episodes}/{expected} episodes")
' "${summary_path}" "${EPISODES}"
}

run_condition() {
  local cond=$1
  local adir; adir=$(artifact_dir "${cond}")

  if [[ -f "${adir}/summary.json" ]]; then
    local summary_status
    if summary_status=$(validate_existing_summary "${adir}/summary.json" 2>&1); then
      log "SKIP ${cond}: ${summary_status}"
      return 0
    fi
    log "STOP ${cond}: existing summary failed strict completion gate: ${summary_status}"
    return 1
  fi

  # A run.log without a summary.json means the episodes ran but were never
  # finalized, which is the state a run launched outside this driver leaves
  # behind. Finalize it instead of re-running, but only if it did not OOM and it
  # actually produced the pre-registered episode count. The shared finalizer
  # invocation above applies the strict reset-audit gate to this adoption path.
  if [[ -e "${adir}/run.log" ]]; then
    if grep -qE 'out of memory|OUT_OF_DEVICE_MEMORY' "${adir}/run.log"; then
      log "STOP ${cond}: existing run.log shows an OOM. Remove ${adir} and re-run the condition in full."
      return 1
    fi
    local official; official=$(newest_official_dir)
    local produced; produced=$(ls "${official}" 2>/dev/null | grep -c 'cam_head.*\.mp4$')
    if [[ "${produced}" -ne "${EPISODES}" ]]; then
      log "STOP ${cond}: existing run produced ${produced}/${EPISODES} episodes. Remove ${adir} and re-run in full."
      return 1
    fi
    log "ADOPT ${cond}: episodes already ran (${produced}/${EPISODES}), strict-finalizing only"
    finalize_condition "${cond}" "${adir}" "${official}" || return 1
    local s; s=$("${REPO}/openpi/.venv/bin/python" -c "
import json;d=json.load(open('${adir}/summary.json'))
a=d.get('aggregate',{});print('%s/%s'%(a.get('successes'),a.get('episodes')))" 2>/dev/null)
    log "  DONE ${cond}: ${s} (not interpreted until the full matrix finishes)"
    return 0
  fi

  log "START ${cond}  (${EPISODES} episodes, layout set ${LAYOUT_SET}, audited reset required)"
  record_gpu "${LOGDIR}/${cond}_gpu_before.csv"
  local t0; t0=$(date +%s)

  ROBODOJO_REQUIRE_AUDITED_RESET=1 CARVE_EVAL_NUM="${EPISODES}" \
    bash scripts/run_robodojo_starvla_nominal.sh \
    "${cond}" "${TASK}" "${LAYOUT_SET}" "${adir}" \
    > "${LOGDIR}/${cond}_run.out" 2>&1
  local rc=$?

  local t1; t1=$(date +%s)
  record_gpu "${LOGDIR}/${cond}_gpu_after.csv"
  log "  wall=$(( (t1-t0)/60 ))m rc=${rc}"

  if grep -qE 'out of memory|OUT_OF_DEVICE_MEMORY' "${LOGDIR}/${cond}_run.out"; then
    log "  STOP ${cond}: CUDA/Vulkan OOM detected. Per section 9 this condition must be re-run in full."
    return 1
  fi

  # A non-zero exit means the run did not finish the block. rc=99 in particular
  # is "PhysX exhausted its restart budget", which leaves a truncated but
  # internally consistent condition; the reset audit tolerates the abandoned
  # episode, so this is the gate that has to catch it.
  if [[ "${rc}" -ne 0 ]]; then
    log "  STOP ${cond}: run exited rc=${rc}. Remove ${adir} and re-run the condition in full."
    return 1
  fi

  local official; official=$(newest_official_dir)
  local produced; produced=$(ls "${official}" 2>/dev/null | grep -c 'cam_head.*\.mp4$')
  log "  official episodes produced: ${produced}/${EPISODES}"
  if [[ "${produced}" -ne "${EPISODES}" ]]; then
    log "  STOP ${cond}: produced ${produced}/${EPISODES} episodes. Remove ${adir} and re-run the condition in full."
    return 1
  fi

  finalize_condition "${cond}" "${adir}" "${official}" || return 1

  local succ; succ=$("${REPO}/openpi/.venv/bin/python" -c "
import json;d=json.load(open('${adir}/summary.json'))
a=d.get('aggregate',{});print('%s/%s'%(a.get('successes'),a.get('episodes')))" 2>/dev/null)
  log "  DONE ${cond}: ${succ} (not interpreted until the full matrix finishes)"
}

log "=== RAL RoboDojo matrix driver ==="
log "settings task=${TASK} layout_set=${LAYOUT_SET} episodes=${EPISODES} stamp=${STAMP} require_audited_reset=${ROBODOJO_REQUIRE_AUDITED_RESET}"
log "prereg sha256 $(sha256sum "${PREREG}" | cut -d' ' -f1)"

# Wait out a matrix run that is already in flight.
while pgrep -f 'run_robodojo_starvla_nominal.sh' > /dev/null 2>&1; do
  log "waiting for the in-flight run to finish..."
  sleep 60
done

for cond in B0 C1 C2 C3; do
  run_condition "${cond}" || { log "ABORT at ${cond}"; exit 1; }
  sleep 20
done

log "=== all four conditions complete, aggregating ==="
runs=()
for cond in B0 C1 C2 C3; do runs+=(--run "$(artifact_dir "${cond}")"); done

PYTHONPATH=. "${REPO}/openpi/.venv/bin/python" scripts/combine_robodojo_condition_matrix.py \
  "${runs[@]}" \
  --preregistration "${PREREG}" \
  --expected-episodes-per-condition "${EPISODES}" \
  --output "${RESULTS}/summary.json" \
  --episodes-csv "${RESULTS}/episodes.csv" \
  2>&1 | tee -a "${LOGDIR}/aggregate.log"

log "=== matrix summary written to ${RESULTS}/summary.json ==="
