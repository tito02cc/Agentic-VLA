#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT}/artifacts/robomme/stage_receipt_dev_20260923"
IMAGE="${ROBOMME_IMAGE:-carve-robomme:cuda12.8}"
POLICY_REPO="${PROJECT_ROOT}/third_party/robomme_policy_learning"

if [[ -e "${ROOT}/rollouts" ]]; then
  echo "refusing to overwrite development rollouts" >&2
  exit 2
fi
for episode in 15 19; do
  test -s "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_fixed_five_20260923/memory/VideoUnmaskSwap_ep${episode}_memory.json"
done
docker image inspect "${IMAGE}" >/dev/null
if ss -ltn | rg -q ':8011 |:18070 '; then
  echo "required model port is occupied" >&2
  exit 2
fi
mkdir -p "${ROOT}/rollouts"
sha256sum -c "${PROJECT_ROOT}/artifacts/robomme/unmaskswap_runtime_bc_20260923/model_sha256.txt" \
  > "${ROOT}/model_integrity.log"
sha256sum "${PROJECT_ROOT}/scripts/run_robomme_vlm_groundsg.py" \
  "${PROJECT_ROOT}/scripts/run_robomme_stage_receipt_dev.sh" \
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

"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_policy.sh" > "${ROOT}/policy_service.log" 2>&1 &
policy_pid="$!"
"${PROJECT_ROOT}/scripts/serve_robomme_groundsg_planner.sh" > "${ROOT}/planner_service.log" 2>&1 &
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

for episode in 15 19; do
  name="VideoUnmaskSwap_ep${episode}_E"
  output="${ROOT}/rollouts/${name}"
  echo "starting ${name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  timeout 900s docker run --rm --gpus all --network host --workdir /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    -e SAPIEN_RENDER_DEVICE=cuda \
    -e PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src \
    -v "${PROJECT_ROOT}:/workspace" -v "${POLICY_REPO}:/policy:ro" \
    "${IMAGE}" /app/.venv/bin/python /workspace/scripts/run_robomme_vlm_groundsg.py \
      --task VideoUnmaskSwap --episode "${episode}" --max-steps 1300 \
      --action-horizon 16 --planner-profile carve --planner-context native \
      --grounding-authority observe_only --procedure-authority observe_only \
      --execution-feedback execution_chunks --planner-demo-mode always \
      --demo-history-mode official --planner-image-format png \
      --memory-on-uncertain without_memory --planner-repair-attempts 0 \
      --planner-gripper-change-threshold 0.004 --planner-max-reusable-points 0 \
      --planner-schedule every_chunk --stage-receipt-hint \
      --task-memory-file "/workspace/artifacts/robomme/unmaskswap_fixed_five_20260923/memory/VideoUnmaskSwap_ep${episode}_memory.json" \
      --memory-hint-style precise --memory-use-policy always \
      --policy-label Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999 \
      --planner-label Qwen3-VL-4B-GroundSG-BF16-SDPA \
      --output "/workspace/artifacts/robomme/stage_receipt_dev_20260923/rollouts/${name}" \
    > "${ROOT}/rollouts/${name}.stdout.log" 2>&1 || {
      echo "rollout process failed; stopping before next episode" >&2
      exit 2
    }
  echo "finished ${name} $(date -u +%FT%TZ)" | tee -a "${ROOT}/run_order.log"
  python3 -c '
import json, re, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
trace = summary["planner_trace"]
put_index = next((i for i, row in enumerate(trace) if "put down" in row["grounded_subgoal"].lower()), None)
post_put_green = [] if put_index is None else [
    row["step"] for row in trace[put_index + 1:]
    if re.search(r"pick up.*green cube", row["grounded_subgoal"], re.I)
]
hint_calls = sum(bool(row.get("stage_receipt_hint")) for row in trace)
print(json.dumps({"episode": summary["episode"], "status": summary["status"],
                  "steps": summary["executed_steps"], "planner_calls": summary["planner_calls"],
                  "first_put_step": None if put_index is None else trace[put_index]["step"],
                  "post_put_green_steps": post_put_green, "hint_calls": hint_calls}))
if summary["status"] != "success" or put_index is None or post_put_green or hint_calls == 0:
    sys.exit(3)
' "${output}/summary.json" || {
    echo "development gate failed; stopping before next episode" >&2
    exit 3
  }
done
