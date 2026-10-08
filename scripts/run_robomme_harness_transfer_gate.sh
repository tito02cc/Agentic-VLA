#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT}/artifacts/robomme/harness_transfer_gate_20260924"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
TASKS=(RouteStick PickHighlight)
EP_START=22
EP_END=37

if [[ -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite rollouts" >&2
  exit 2
fi
for task in "${TASKS[@]}"; do
  for episode in $(seq "${EP_START}" "${EP_END}"); do
    test -s "${ROOT}/preflight/${task}_ep${episode}/summary.json"
  done
done
python3 - "${ROOT}/preflight" "${EP_START}" "${EP_END}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
start, end = int(sys.argv[2]), int(sys.argv[3])
for task in sys.argv[4:]:
    digests = []
    for episode in range(start, end + 1):
        summary = json.loads((root / f"{task}_ep{episode}" / "summary.json").read_text())
        if summary["task"] != task or summary["episode"] != episode:
            raise SystemExit(f"wrong preflight identity: {task} ep{episode}")
        digests.append(summary["initial_observation_sha256"])
    if len(set(digests)) != len(digests):
        raise SystemExit(f"duplicate initial state in {task}")
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
  "${PROJECT_ROOT}/scripts/run_robomme_harness_transfer_gate.sh" \
  "${PROJECT_ROOT}/agentic_vla/runtime/semantic.py" \
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
    echo "model service exited during startup" >&2
    exit 2
  fi
  if ss -ltn | rg -q ':8011 ' && curl -fsS --max-time 2 http://127.0.0.1:18070/v1/models >/dev/null; then
    ready=1
    break
  fi
  sleep 3
done
if [[ "${ready}" != 1 ]]; then
  echo "model services did not become ready" >&2
  exit 2
fi
nvidia-smi --query-gpu=timestamp,memory.used,power.draw --format=csv,noheader,nounits --loop-ms=1000 \
  > "${ROOT}/gpu_usage.csv" 2> "${ROOT}/gpu_monitor.log" &
gpu_pid="$!"

run_case() {
  local task="$1" episode="$2" arm="$3" max_steps="$4"
  local name="${task}_ep${episode}_${arm}"
  local profile=raw
  if [[ "${arm}" == H ]]; then
    profile=carve
  fi
  local exit_code=0
  echo "starting ${name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task "${task}" --episode "${episode}" --max-steps "${max_steps}" \
      --action-horizon 16 --planner-profile "${profile}" --planner-schedule every_chunk \
      --planner-repair-attempts 0 --demo-history-mode official \
      --planner-demo-mode always --planner-image-format png \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/harness_transfer_gate_20260924/rollouts/${name}" \
    > "${ROOT}/rollouts/${name}.stdout.log" 2>&1 || exit_code="$?"
  echo "finished ${name} exit=${exit_code} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  local summary="${ROOT}/rollouts/${name}/summary.json"
  if [[ ! -s "${summary}" ]]; then
    echo "missing summary; infrastructure failure: ${name}" >&2
    exit 2
  fi
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("result",d["task"],d["episode"],d["status"],"steps",d["executed_steps"],"planner",d["planner_calls"],"vla",d["policy_calls"],"failure",d["failure"],flush=True); valid=d["failure"] is None and (sys.argv[2]=="warmup" or d["status"] in ("success","fail") or (d["status"]=="ongoing" and d["executed_steps"]>=d["max_steps"])); sys.exit(0 if valid else 2)' "${summary}" "$([[ "${max_steps}" == 16 ]] && echo warmup || echo counted)"
}

run_case RouteStick 15 R 16
for task in "${TASKS[@]}"; do
  for episode in $(seq "${EP_START}" "${EP_END}"); do
    if (( episode % 2 == 0 )); then
      first=R; second=H
    else
      first=H; second=R
    fi
    run_case "${task}" "${episode}" "${first}" 1300
    run_case "${task}" "${episode}" "${second}" 1300
    python3 - "${ROOT}/rollouts" "${task}" "${episode}" <<'PY'
import json
import sys
from pathlib import Path
root, task, episode = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
name = f"{task}_ep{episode}"
raw = json.loads((root / f"{name}_R" / "summary.json").read_text())
harness = json.loads((root / f"{name}_H" / "summary.json").read_text())
if raw["initial_observation_sha256"] != harness["initial_observation_sha256"]:
    raise SystemExit(f"initial observations differ: {name}")
print(json.dumps({"task": task, "episode": episode, "raw": raw["status"],
                  "harness": harness["status"], "rescue": not raw["success"] and harness["success"],
                  "harm": raw["success"] and not harness["success"]}), flush=True)
PY
  done
done
