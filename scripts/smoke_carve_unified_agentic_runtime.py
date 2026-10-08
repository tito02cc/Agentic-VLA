#!/usr/bin/env python3
"""Exercise the CARVE state machine with real VLM and VLA services.

This verifies orchestration, episode isolation, safe hold, typed VLM decisions,
the model-agnostic VLA adapter, and deployment-profile tracing. It does not send
actions to a simulator or robot and is not task-success evidence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import (  # noqa: E402
    AsyncAgenticHarnessController,
    AsyncGuardedHighLevelAgent,
    CarveRuntime,
    GuardedHighLevelAgent,
    HarnessState,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    InferenceControls,
    InferenceRequest,
    JointRecoveryComputeController,
    OpenAICompatibleVisionPlanner,
    PausedExecutionSafeHoldAdapter,
    TaskStartPolicy,
)
from agentic_vla.runtime.adapters import Pi05Adapter  # noqa: E402
from scripts.benchmark_carve_pi05_profile import (  # noqa: E402
    monitor_proprio_to_policy_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--vlm-endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--vlm-model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=18081)
    parser.add_argument("--boundary-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--vlm-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for the structured VLM decision.",
    )
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument(
        "--require-promoted-vla-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require CARVE deployment-profile admission in the VLA trace. Disable only "
            "for a framework-integration smoke against an explicitly unprofiled service."
        ),
    )
    return parser.parse_args()


def main() -> int:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    args = parse_args()
    image_path = pathlib.Path(args.image).expanduser().resolve()
    snapshot_path = pathlib.Path(args.snapshot).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    frame = iio.imread(image_path)
    with np.load(snapshot_path, allow_pickle=False) as arrays:
        observation = {
            "observation/image": np.asarray(arrays["observation_agentview_image"]),
            "observation/wrist_image": np.asarray(arrays["observation_wrist_image"]),
            "observation/state": monitor_proprio_to_policy_state(
                arrays["observation_proprio"]
            ),
        }

    planner_client = OpenAICompatibleVisionPlanner(
        endpoint=str(args.vlm_endpoint),
        model=str(args.vlm_model),
        timeout_s=float(args.boundary_timeout_sec),
        max_tokens=int(args.vlm_max_tokens),
    )

    def planner_factory(_episode_id):
        return AsyncGuardedHighLevelAgent(
            GuardedHighLevelAgent(
                planner_client,
                HighLevelAgentConfig(max_calls_per_episode=1),
            )
        )

    hold_adapter = PausedExecutionSafeHoldAdapter()
    harness = AsyncAgenticHarnessController(
        JointRecoveryComputeController(),
        planner_factory=planner_factory,
        task_start_policy=TaskStartPolicy.STARTUP_WAIT,
        safe_hold_adapter=hold_adapter,
        safe_hold_timeout_s=float(args.boundary_timeout_sec),
    )
    episode_id = "unified-real-service-smoke:0"
    context = HighLevelAgentContext(
        task_instruction=str(args.task),
        trigger="task_start",
        episode_id=episode_id,
        timestep=0,
        frames={"agentview": frame},
        robot_state=(0.0,) * 9,
        risk={
            "event": None,
            "bucket": "low",
            "score": 0.0,
            "components": {},
            "evidence": {"action_age_steps": 0},
        },
        current_subgoal="execute",
        available_skills=(),
        remaining_retries=0,
        remaining_recoveries=0,
        deadline_slack_ms=float(args.deadline_ms),
    )

    started_s = time.perf_counter()
    submitted = harness.start_episode(
        episode_id,
        task_start_context=context,
    )
    immediate = harness.poll_planner()
    planner_transition = (
        immediate
        if immediate is not None
        else harness.await_planner(timeout_s=float(args.boundary_timeout_sec))
    )
    planner_elapsed_ms = (time.perf_counter() - started_s) * 1000.0

    vla_trace = None
    action_shape = None
    instruction = str(args.task)
    if planner_transition.planner is not None:
        decision = planner_transition.planner.result.decision
        if decision.vla_instruction:
            instruction = decision.vla_instruction

    if (
        planner_transition.state is HarnessState.EXECUTE_FAST
        and planner_transition.decision_applied
    ):
        client = WebsocketClientPolicy(host=args.vla_host, port=int(args.vla_port))
        runtime = CarveRuntime(Pi05Adapter(client), fallback_mode="strict")
        chunk = runtime.infer(
            InferenceRequest(
                observation=observation,
                instruction=instruction,
                controls=InferenceControls(
                    inference_steps=2,
                    max_actions=10,
                    deadline_ms=float(args.deadline_ms),
                ),
                episode_id=episode_id,
                timestep=0,
                agentic={
                    "trigger": "task_start",
                    "planner_ticket": (
                        planner_transition.planner.ticket.ticket_id
                        if planner_transition.planner is not None
                        else None
                    ),
                    "decision": (
                        planner_transition.planner.result.decision.to_dict()
                        if planner_transition.planner is not None
                        else None
                    ),
                },
                metadata={
                    "noise": np.random.default_rng(20260724).standard_normal(
                        (10, 32), dtype=np.float32
                    ),
                    "action_age_steps": 0,
                    "trace_context": {
                        "harness_state": planner_transition.state.value,
                        "task_start_policy": TaskStartPolicy.STARTUP_WAIT.value,
                    },
                },
            )
        )
        action_shape = list(np.asarray(chunk.actions).shape)
        if runtime.last_trace is not None:
            vla_trace = runtime.last_trace.to_dict()

    profile_admitted = (
        isinstance(vla_trace, dict)
        and vla_trace.get("metadata", {})
        .get("optimization_profile", {})
        .get("admission", {})
        .get("status")
        == "promoted"
    )
    passed = bool(
        submitted.ticket is not None
        and planner_transition.planner is not None
        and planner_transition.planner.result.accepted
        and planner_transition.decision_applied
        and planner_transition.state is HarnessState.EXECUTE_FAST
        and [event["event"] for event in hold_adapter.events] == ["enter", "release"]
        and action_shape == [10, 7]
        and (profile_admitted or not args.require_promoted_vla_profile)
    )
    receipt = {
        "schema_version": 1,
        "kind": "carve_unified_real_service_smoke",
        "claim_boundary": (
            "Real VLM, real admitted VLA, and Agentic state-machine integration; "
            "no simulator/robot action and no task-success claim."
            if args.require_promoted_vla_profile
            else "Real VLM, real eager PI0.5 VLA, and Agentic state-machine integration; "
            "this is not deployment-profile or task-success evidence."
        ),
        "passed": passed,
        "vla_profile_admitted": profile_admitted,
        "episode_id": episode_id,
        "task_start_policy": TaskStartPolicy.STARTUP_WAIT.value,
        "submitted_state": submitted.state.value,
        "planner_transition": {
            "state": planner_transition.state.value,
            "decision_applied": planner_transition.decision_applied,
            "timed_out": planner_transition.timed_out,
            "stale": planner_transition.stale,
            "reason": planner_transition.reason,
            "elapsed_ms": planner_elapsed_ms,
            "ticket_id": (
                planner_transition.planner.ticket.ticket_id
                if planner_transition.planner is not None
                else None
            ),
            "result": (
                {
                    "accepted": planner_transition.planner.result.accepted,
                    "elapsed_ms": planner_transition.planner.result.elapsed_ms,
                    "error": planner_transition.planner.result.error,
                    "decision": (
                        planner_transition.planner.result.decision.to_dict()
                    ),
                }
                if planner_transition.planner is not None
                else None
            ),
        },
        "safe_hold_events": hold_adapter.events,
        "vla_instruction": instruction,
        "vla_action_shape": action_shape,
        "vla_runtime_trace": vla_trace,
        "harness_counters": harness.counters.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "planner_transition": receipt["planner_transition"],
                "safe_hold_events": receipt["safe_hold_events"],
                "vla_action_shape": action_shape,
                "vla_profile": (
                    vla_trace.get("metadata", {}).get("optimization_profile")
                    if isinstance(vla_trace, dict)
                    else None
                ),
            },
            indent=2,
        )
    )
    harness.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
