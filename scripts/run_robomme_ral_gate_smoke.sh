#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT}/artifacts/robomme/ral_gate_20260924"
RUN_LABEL="${CARVE_SMOKE_RUN_LABEL:-smoke_rollouts_v6}"
RUN_ROOT="${ROOT}/${RUN_LABEL}"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
read -r -a CASES <<< "${CARVE_SMOKE_CASES:-VideoRepick:41 VideoUnmask:38}"
SELECTED_ARMS="${CARVE_SMOKE_ARMS:-ABC}"
PREFLIGHT_MANIFEST="${CARVE_SMOKE_PREFLIGHT_MANIFEST:-${ROOT}/preflight_manifest.json}"
if [[ "${SELECTED_ARMS}" != ABC && "${SELECTED_ARMS}" != C ]]; then
  echo "CARVE_SMOKE_ARMS must be ABC or C" >&2
  exit 2
fi

test -f "${PREFLIGHT_MANIFEST}"
python3 - "${PREFLIGHT_MANIFEST}" "${CASES[@]}" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
if not manifest["admitted_for_paired_gate"]:
    raise SystemExit("preflight did not pass")
for case in sys.argv[2:]:
    task, episode_text = case.split(":", 1)
    episode = int(episode_text)
    if not any(r["task"] == task and r["episode"] == episode and not r["prior_summary_paths"]
               for r in manifest["records"]):
        raise SystemExit(f"case not present in frozen preflight: {task} ep{episode}")
PY
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite existing smoke rollouts" >&2
  exit 2
fi
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "model ports 8011/18070 are occupied" >&2
  exit 2
fi
docker image inspect "${IMAGE}" >/dev/null
test -s "${ROOT}/memory_bound/VideoRepick_ep41_memory.json"
test -s "${ROOT}/memory/VideoUnmask_ep38_memory.json"
mkdir -p "${RUN_ROOT}"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_ral_gate_smoke.sh" \
  "${PROJECT_ROOT}/agentic_vla/benchmarks/robomme_memory.py" \
  > "${RUN_ROOT}/source_sha256.txt"
docker image inspect "${IMAGE}" --format '{{.Id}}' \
  > "${RUN_ROOT}/docker_image_id.txt"

policy_pid=""
planner_pid=""
cleanup() {
  for pid in "${planner_pid}" "${policy_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" \
  > "${RUN_ROOT}/policy_service.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" \
  > "${RUN_ROOT}/planner_service.log" 2>&1 &
planner_pid="$!"
ready=0
planner_port=""
for _ in $(seq 1 120); do
  if ! kill -0 "${policy_pid}" 2>/dev/null || ! kill -0 "${planner_pid}" 2>/dev/null; then
    echo "model service exited during startup" >&2
    exit 2
  fi
  if ss -ltn | rg -q ':8011 '; then
    for port in $(seq 18070 18079); do
      if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
        | rg -q 'robomme-groundsg-qwen3vl4b'; then
        planner_port="${port}"
        ready=1
        break
      fi
    done
  fi
  [[ "${ready}" == 1 ]] && break
  sleep 3
done
if [[ "${ready}" != 1 ]]; then
  echo "model services did not become ready" >&2
  exit 2
fi
printf '%s\n' "${planner_port}" > "${RUN_ROOT}/planner_port.txt"

run_case() {
  local task="$1" episode="$2" arm="$3"
  local name="${task}_ep${episode}_${arm}"
  local memory=""
  if [[ "${task}" == "VideoRepick" ]]; then
    memory="/workspace/artifacts/robomme/ral_gate_20260924/memory_bound/${task}_ep${episode}_memory.json"
  else
    memory="/workspace/artifacts/robomme/ral_gate_20260924/memory/${task}_ep${episode}_memory.json"
  fi
  local options=(--planner-profile raw)
  if [[ "${arm}" != A ]]; then
    options=(--planner-profile carve --execution-feedback execution_chunks
      --procedure-authority observe_only --planner-context native
      --grounding-authority observe_only)
    if [[ "${arm}" == C ]]; then
      options+=(--task-memory-file "${memory}")
      if [[ "${CARVE_SMOKE_STAGE_HINT:-0}" == 1 ]]; then
        options+=(--stage-receipt-hint)
      fi
    fi
  fi
  echo "starting ${name} $(date -u +%FT%TZ)" | tee -a "${RUN_ROOT}/run_order.log"
  local exit_code=0
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task "${task}" --episode "${episode}" --max-steps 1300 \
      --action-horizon 16 --planner-schedule every_chunk \
      --planner-repair-attempts 1 --demo-history-mode official \
      --planner-endpoint "http://127.0.0.1:${planner_port}/v1/chat/completions" \
      --planner-demo-mode always --planner-image-format png \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      "${options[@]}" \
      --output "/workspace/artifacts/robomme/ral_gate_20260924/${RUN_LABEL}/${name}" \
    > "${RUN_ROOT}/${name}.stdout.log" 2>&1 || exit_code="$?"
  python3 - "${RUN_ROOT}/${name}/summary.json" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
exit_code = int(sys.argv[2])
failure = data["failure"]
semantic_failure = (isinstance(failure, str)
                    and failure.startswith("grounded Planner repair failed without procedural override:"))
if (failure is not None and not semantic_failure) or (
        data["status"] == "ongoing" and data["executed_steps"] < data["max_steps"]
        and not semantic_failure) or (exit_code != 0 and not semantic_failure):
    raise SystemExit("incomplete rollout: " + str(sys.argv[1]))
print(json.dumps({"task": data["task"], "episode": data["episode"],
                  "success": data["success"], "steps": data["executed_steps"],
                  "planner_calls": data["planner_calls"], "policy_calls": data["policy_calls"],
                  "semantic_failure": semantic_failure}))
PY
}

for item in "${CASES[@]}"; do
  task="${item%%:*}"
  episode="${item##*:}"
  if (( episode % 2 == 0 )); then
    arms=(A B C)
  else
    arms=(C B A)
  fi
  for arm in "${arms[@]}"; do
    if [[ "${SELECTED_ARMS}" != *"${arm}"* ]]; then
      continue
    fi
    run_case "${task}" "${episode}" "${arm}"
  done
  [[ "${SELECTED_ARMS}" == ABC ]] || continue
  python3 - "${RUN_ROOT}" "${task}" "${episode}" <<'PY'
import json
import sys
from pathlib import Path
root, task, episode = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = {arm: json.loads((root / f"{task}_ep{episode}_{arm}" / "summary.json").read_text())
        for arm in "ABC"}
hashes = {arm: row["initial_observation_sha256"] for arm, row in rows.items()}
if len(set(hashes.values())) != 1:
    raise SystemExit(f"initial states differ: {task} ep{episode}: {hashes}")
print(json.dumps({"task": task, "episode": episode,
                  "success": {arm: row["success"] for arm, row in rows.items()},
                  "initial_observation_sha256": next(iter(hashes.values()))}))
PY
done
