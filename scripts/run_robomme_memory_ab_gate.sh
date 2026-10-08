#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_NAME="${ROOT_NAME:-memory_ab_gate_20260923}"
EP_START="${EP_START:-30}"
EP_END="${EP_END:-37}"
PAIR_MODE="${PAIR_MODE:-AB}"
TASK="${TASK:-VideoUnmaskSwap}"
RUN_SINGLE_ARM="${RUN_SINGLE_ARM:-}"
MEMORY_SOURCE_NAME="${MEMORY_SOURCE_NAME:-${ROOT_NAME}}"
ROOT="${PROJECT_ROOT}/artifacts/robomme/${ROOT_NAME}"
MEMORY_ROOT="${PROJECT_ROOT}/artifacts/robomme/${MEMORY_SOURCE_NAME}"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"

if [[ "${PAIR_MODE}" != AB && "${PAIR_MODE}" != BC ]]; then
  echo "PAIR_MODE must be AB or BC" >&2
  exit 2
fi
if [[ "${TASK}" != VideoUnmaskSwap && "${TASK}" != VideoUnmask ]]; then
  echo "TASK must be VideoUnmaskSwap or VideoUnmask" >&2
  exit 2
fi
if [[ -n "${RUN_SINGLE_ARM}" && "${RUN_SINGLE_ARM}" != B ]]; then
  echo "RUN_SINGLE_ARM is only supported for the B-only ep49 completion" >&2
  exit 2
fi
if [[ -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite ${PAIR_MODE} rollouts" >&2
  exit 2
fi
for episode in $(seq "${EP_START}" "${EP_END}"); do
  test -s "${MEMORY_ROOT}/preflight/${TASK}_ep${episode}/summary.json"
  test -s "${MEMORY_ROOT}/memory/${TASK}_ep${episode}_memory.json"
done
python3 - "${MEMORY_ROOT}" "${EP_START}" "${EP_END}" "${TASK}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task = sys.argv[4]
seen = {}
for episode in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
    summary = json.loads((root / "preflight" / f"{task}_ep{episode}" / "summary.json").read_text())
    digest = summary["initial_observation_sha256"]
    if digest in seen:
        raise SystemExit(f"duplicate initial state: ep{seen[digest]} and ep{episode}")
    seen[digest] = episode
PY
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "required model port is occupied" >&2
  exit 2
fi
mkdir -p "${ROOT}/rollouts"
sha256sum -c "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923/model_sha256.txt" \
  > "${ROOT}/model_integrity.log"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_memory_ab_gate.sh" \
  "${PROJECT_ROOT}/agentic_vla/benchmarks/robomme_memory.py" \
  > "${ROOT}/source_sha256.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' > "${ROOT}/docker_image_id.txt"

policy_pid=""
planner_pid=""
gpu_pid=""
cleanup() {
  for pid in "${planner_pid}" "${policy_pid}" "${gpu_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" > "${ROOT}/rollouts/policy_service.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" > "${ROOT}/rollouts/planner_service.log" 2>&1 &
planner_pid="$!"
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "${policy_pid}" 2>/dev/null || ! kill -0 "${planner_pid}" 2>/dev/null; then
    echo "a model service exited during startup" >&2
    exit 2
  fi
  if ss -ltn | rg -q ':8011 ' && \
     curl -fsS --max-time 2 http://127.0.0.1:18070/v1/models >/dev/null; then
    ready=1
    break
  fi
  sleep 3
done
if [[ "${ready}" != 1 ]]; then
  echo "model services did not become ready" >&2
  exit 2
fi
nvidia-smi --query-gpu=timestamp,memory.used,power.draw \
  --format=csv,noheader,nounits --loop-ms=1000 \
  > "${ROOT}/gpu_usage.csv" 2> "${ROOT}/gpu_monitor.log" &
gpu_pid="$!"

run_case() {
  local episode="$1" arm="$2" output_name="$3" max_steps="$4"
  local memory_args=()
  local schedule=(--planner-schedule every_chunk)
  if [[ "${arm}" == B || "${arm}" == C ]]; then
    local source="${MEMORY_SOURCE_NAME}"
    if [[ "${TASK}" == VideoUnmaskSwap && "${episode}" == 15 ]]; then
      source="unmaskswap_fixed_five_20260923"
    fi
    memory_args=(--task-memory-file "/workspace/artifacts/robomme/${source}/memory/${TASK}_ep${episode}_memory.json" \
                 --memory-hint-style precise --memory-use-policy always)
  fi
  if [[ "${arm}" == C ]]; then
    schedule=(--planner-schedule selective --verified-point-reuse)
  fi
  echo "starting ${output_name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task "${TASK}" --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      "${schedule[@]}" "${memory_args[@]}" \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/${ROOT_NAME}/rollouts/${output_name}" \
    > "${ROOT}/rollouts/${output_name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${output_name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local summary="${ROOT}/rollouts/${output_name}/summary.json"
  if [[ ! -s "${summary}" ]]; then
    echo "missing summary after ${output_name}; infrastructure failure" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True); budget_end=d["status"]=="ongoing" and d["executed_steps"]>=d["max_steps"]; valid=d["failure"] is None and (sys.argv[2].startswith("warmup_") or d["status"] in ("success","fail") or budget_end); sys.exit(0 if valid else 2)' "${summary}" "${output_name}"
}

if [[ "${PAIR_MODE}" == AB ]]; then
  run_case 15 A warmup_ep15_A 16
else
  if [[ "${TASK}" != VideoUnmaskSwap ]]; then
    echo "BC warmup is only configured for VideoUnmaskSwap" >&2
    exit 2
  fi
  run_case 15 B warmup_ep15_B 16
fi
if [[ -n "${RUN_SINGLE_ARM}" ]]; then
  for episode in $(seq "${EP_START}" "${EP_END}"); do
    run_case "${episode}" "${RUN_SINGLE_ARM}" "${TASK}_ep${episode}_${RUN_SINGLE_ARM}" 1300
  done
  exit 0
fi
for episode in $(seq "${EP_START}" "${EP_END}"); do
  memory="${MEMORY_ROOT}/memory/${TASK}_ep${episode}_memory.json"
  if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["admission"]["admitted"])' "${memory}")" != True ]]; then
    echo "excluded ep${episode}: memory_not_admitted $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
    continue
  fi
  if (( episode % 2 == 0 )); then
    first="${PAIR_MODE:0:1}"
    second="${PAIR_MODE:1:1}"
  else
    first="${PAIR_MODE:1:1}"
    second="${PAIR_MODE:0:1}"
  fi
  run_case "${episode}" "${first}" "${TASK}_ep${episode}_${first}" 1300
  run_case "${episode}" "${second}" "${TASK}_ep${episode}_${second}" 1300
  python3 -c '
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); e=int(sys.argv[2]); arms=sys.argv[3]; task=sys.argv[4]; name=f"{task}_ep{e}"
left=json.loads((root/f"{name}_{arms[0]}"/"summary.json").read_text())
right=json.loads((root/f"{name}_{arms[1]}"/"summary.json").read_text())
if left["initial_observation_sha256"]!=right["initial_observation_sha256"]:
    raise SystemExit("initial observations differ")
print(json.dumps({"episode":e,arms[0]:left["status"],arms[1]:right["status"],"rescue":not left["success"] and right["success"],"harm":left["success"] and not right["success"]}),flush=True)
' "${ROOT}/rollouts" "${episode}" "${PAIR_MODE}" "${TASK}"
done
