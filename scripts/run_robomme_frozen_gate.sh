#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT}/artifacts/robomme/frozen_gate_20260923"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
RESUME="${1:-}"

if [[ -n "${RESUME}" && "${RESUME}" != "--resume-after-ep27-F" ]]; then
  echo "usage: $0 [--resume-after-ep27-F]" >&2
  exit 2
fi
if [[ -z "${RESUME}" && -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite frozen-gate rollouts" >&2
  exit 2
fi
if [[ "${RESUME}" == "--resume-after-ep27-F" ]]; then
  test -s "${ROOT}/rollouts/VideoUnmaskSwap_ep27_F/summary.json"
  test ! -e "${ROOT}/rollouts/VideoUnmaskSwap_ep27_B"
fi
for episode in {22..29}; do
  test -s "${ROOT}/preflight/VideoUnmaskSwap_ep${episode}/summary.json"
  test -s "${ROOT}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
done
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "required model port is occupied" >&2
  exit 2
fi
mkdir -p "${ROOT}/rollouts"
sha256sum -c "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923/model_sha256.txt" \
  > "${ROOT}/model_integrity${RESUME:+_resume}.log"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_frozen_gate.sh" \
  "${PROJECT_ROOT}/agentic_vla/benchmarks/robomme_memory.py" \
  > "${ROOT}/source_sha256${RESUME:+_resume}.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' > "${ROOT}/docker_image_id${RESUME:+_resume}.txt"

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

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" > "${ROOT}/rollouts/policy_service${RESUME:+_resume}.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" > "${ROOT}/rollouts/planner_service${RESUME:+_resume}.log" 2>&1 &
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
  > "${ROOT}/gpu_usage${RESUME:+_resume}.csv" 2> "${ROOT}/gpu_monitor${RESUME:+_resume}.log" &
gpu_pid="$!"

run_case() {
  local episode="$1" arm="$2" output_name="$3" max_steps="$4"
  if [[ "${RESUME}" == "--resume-after-ep27-F" && "${output_name}" == "VideoUnmaskSwap_ep27_F" ]]; then
    echo "retaining original ep27 F budget-exhausted rollout from previous service" | tee -a "${ROOT}/run_order.log"
    return
  fi
  local memory_root="frozen_gate_20260923"
  if [[ "${episode}" == 15 ]]; then
    memory_root="unmaskswap_fixed_five_20260923"
  fi
  local memory="/workspace/artifacts/robomme/${memory_root}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
  local flag=()
  if [[ "${arm}" == F ]]; then
    flag=(--verified-identity-conflict)
    local count
    count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["memory"]["required_color_order"]))' \
      "${ROOT}/memory/VideoUnmaskSwap_ep${episode}_memory.json")"
    if (( count >= 2 )); then
      flag+=(--stage-receipt-hint)
    fi
  fi
  echo "starting ${output_name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task VideoUnmaskSwap --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      --planner-schedule every_chunk \
      --task-memory-file "${memory}" \
      --memory-hint-style precise --memory-use-policy always \
      "${flag[@]}" \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/frozen_gate_20260923/rollouts/${output_name}" \
    > "${ROOT}/rollouts/${output_name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${output_name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local summary="${ROOT}/rollouts/${output_name}/summary.json"
  if [[ ! -s "${summary}" ]]; then
    echo "missing summary after ${output_name}; infrastructure failure" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True); budget_end=d["status"]=="ongoing" and d["executed_steps"]>=d["max_steps"]; valid=d["failure"] is None and (sys.argv[2].startswith("warmup_") or d["status"] in ("success","fail") or budget_end); sys.exit(0 if valid else 2)' "${summary}" "${output_name}"
}

if [[ "${RESUME}" == "--resume-after-ep27-F" ]]; then
  run_case 15 B warmup_resume_ep15_B 16
  episodes=(27 28 29)
else
  run_case 15 B warmup_ep15_B 16
  episodes=(22 23 24 25 26 27 28 29)
fi
for episode in "${episodes[@]}"; do
  memory="${ROOT}/memory/VideoUnmaskSwap_ep${episode}_memory.json"
  if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["admission"]["admitted"])' "${memory}")" != True ]]; then
    echo "excluded ep${episode}: memory_not_admitted $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
    continue
  fi
  if (( episode % 2 == 0 )); then
    first=B
    second=F
  else
    first=F
    second=B
  fi
  run_case "${episode}" "${first}" "VideoUnmaskSwap_ep${episode}_${first}" 1300
  run_case "${episode}" "${second}" "VideoUnmaskSwap_ep${episode}_${second}" 1300
  python3 -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
episodes = [e for e in range(22, 30) if all((root / f"VideoUnmaskSwap_ep{e}_{a}" / "summary.json").is_file() for a in ("B", "F"))]
pairs = []
for episode in episodes:
    rows = {arm: json.loads((root / f"VideoUnmaskSwap_ep{episode}_{arm}" / "summary.json").read_text()) for arm in ("B", "F")}
    if rows["B"]["initial_observation_sha256"] != rows["F"]["initial_observation_sha256"]:
        raise SystemExit("initial observations differ")
    if rows["B"]["memory_provenance"]["memory_sha256"] != rows["F"]["memory_provenance"]["memory_sha256"]:
        raise SystemExit("memory differs")
    pairs.append({"episode": episode, "B": rows["B"]["success"], "F": rows["F"]["success"]})
rescues = sum(not row["B"] and row["F"] for row in pairs)
harms = sum(row["B"] and not row["F"] for row in pairs)
f_success = sum(row["F"] for row in pairs)
print(json.dumps({"completed_pairs": len(pairs), "pairs": pairs, "rescues": rescues, "harms": harms, "F_success": f_success}), flush=True)
if harms >= 2 or (len(pairs) >= 4 and f_success == 0):
    raise SystemExit(3)
' "${ROOT}/rollouts" || {
    echo "quality stop rule triggered; ending frozen gate" | tee -a "${ROOT}/run_order.log"
    exit 3
  }
done
