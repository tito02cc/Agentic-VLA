#!/usr/bin/env python3
"""Record one real VLM asynchronous-planner lifecycle without a robot rollout.

This is the first frozen integration-smoke stage. It verifies the actual image
transport, guarded JSON contract, non-blocking ticket poll, bounded boundary
wait, and admitted PI0.5 profile receipt. It is deliberately not a task-success
experiment and never sends actions to a robot or simulator.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import imageio.v3 as iio

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import (  # noqa: E402
    AsyncGuardedHighLevelAgent,
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    OpenAICompatibleVisionPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="JSON receipt; parent directories are created when necessary.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument(
        "--profile-manifest",
        default=(
            "results/carve_optimize/deployment/pi05_smve_promoted.json"
        ),
    )
    parser.add_argument("--boundary-timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def _result_payload(completed: Any | None) -> dict[str, Any] | None:
    if completed is None:
        return None
    result = completed.result
    return {
        "ticket_id": completed.ticket.ticket_id,
        "submitted_timestep": completed.ticket.context.timestep,
        "trigger": completed.ticket.context.trigger,
        "accepted": bool(result.accepted),
        "elapsed_ms": float(result.elapsed_ms),
        "error": result.error,
        "decision": result.decision.to_dict(),
    }


def main() -> int:
    args = parse_args()
    if args.boundary_timeout_sec <= 0 or args.max_tokens <= 0:
        raise ValueError("boundary timeout and max tokens must be positive")
    image_path = pathlib.Path(args.image).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    profile_path = (PROJECT_ROOT / args.profile_manifest).resolve()
    frame = iio.imread(image_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    guarded = GuardedHighLevelAgent(
        OpenAICompatibleVisionPlanner(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=float(args.boundary_timeout_sec),
            max_tokens=int(args.max_tokens),
        ),
        HighLevelAgentConfig(max_calls_per_episode=1),
    )
    planner = AsyncGuardedHighLevelAgent(guarded)
    started = time.perf_counter()
    try:
        ticket = planner.submit(
            HighLevelAgentContext(
                task_instruction=args.task,
                trigger="task_start",
                episode_id="frozen-integration-smoke:0",
                timestep=0,
                frames={"agentview": frame},
                robot_state=(0.0,) * 9,
                risk={"event": None, "bucket": "low", "score": 0.0},
                current_subgoal="execute",
                available_skills=(),
                remaining_retries=0,
                remaining_recoveries=0,
                deadline_slack_ms=80.0,
            )
        )
        immediate = planner.take(wait=False)
        completed = planner.take(
            wait=True,
            timeout_s=float(args.boundary_timeout_sec),
        )
        receipt = {
            "schema_version": 1,
            "kind": "carve_async_planner_integration_smoke",
            "claim_boundary": (
                "Real VLM lifecycle and profile receipt only; no robot action, "
                "closed-loop, or task-success claim."
            ),
            "image": str(image_path),
            "planner": {
                "ticket_id": ticket.ticket_id,
                "nonblocking_poll_returned": immediate is not None,
                "boundary_timeout_sec": float(args.boundary_timeout_sec),
                "completed": _result_payload(completed),
                "timed_out_fail_closed": completed is None,
                "wall_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
            "profile_receipt": {
                "path": str(profile_path),
                "profile_id": profile.get("profile", {}).get("profile_id"),
                "backend": profile.get("profile", {}).get("backend"),
                "fidelity_passed": bool(profile.get("fidelity", {}).get("passed")),
                "admission_status": profile.get("admission", {}).get("status"),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if completed is not None else 2
    finally:
        planner.close(wait=False)


if __name__ == "__main__":
    raise SystemExit(main())
