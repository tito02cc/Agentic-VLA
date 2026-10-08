#!/usr/bin/env python3
"""Run the canonical CARVE session against local Qwen-VL and PI0.5.

The generated action chunk is inspected behind the VLA primitive boundary and
is deliberately withheld from any simulator or robot. This is framework and
model-integration evidence, not task-success evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.configuration import ArtifactRunConfig, CarveRunConfig  # noqa: E402
from agentic_vla.assembly import build_profile_admitted_tool_bindings  # noqa: E402
from agentic_vla.runtime import (  # noqa: E402
    AgentIntent,
    CarveRuntime,
    HighLevelAgentContext,
    HarnessState,
)
from agentic_vla.runtime.adapters import Pi05Adapter  # noqa: E402
from agentic_vla.session import CarveAgentSession  # noqa: E402
from agentic_vla.toolchain import (  # noqa: E402
    PrimitiveExecutionReport,
    PrimitiveStatus,
    ToolExecutionContext,
    VerificationReport,
)
from agentic_vla.toolchain.vla import (  # noqa: E402
    ActionExecutionReport,
)
from scripts.benchmark_carve_pi05_profile import (  # noqa: E402
    monitor_proprio_to_policy_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/carve_pi05_qwen_local.json"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--results-root", default="results/carve_framework_freeze_20260824"
    )
    parser.add_argument("--run-id", default="canonical-qwen-pi05-smoke")
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=18081)
    return parser.parse_args()


def main() -> int:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    args = parse_args()
    config = CarveRunConfig.load(args.config)
    config = dataclasses.replace(
        config,
        run_id=str(args.run_id),
        artifacts=ArtifactRunConfig(
            results_root=str(args.results_root), persist_frames=False
        ),
    )
    frame = iio.imread(pathlib.Path(args.image).expanduser().resolve())
    with np.load(
        pathlib.Path(args.snapshot).expanduser().resolve(), allow_pickle=False
    ) as arrays:
        policy_observation = {
            "observation/image": np.asarray(arrays["observation_agentview_image"]),
            "observation/wrist_image": np.asarray(
                arrays["observation_wrist_image"]
            ),
            "observation/state": monitor_proprio_to_policy_state(
                arrays["observation_proprio"]
            ),
        }

    client = WebsocketClientPolicy(host=args.vla_host, port=int(args.vla_port))
    runtime = CarveRuntime(
        Pi05Adapter(client, adapter_id=config.vla.adapter_id),
        fallback_mode="strict",
    )
    private_action_receipt: dict[str, object] = {}

    def inspect_without_execution(actions, context):
        shape = list(np.asarray(actions).shape)
        private_action_receipt.update(
            {
                "shape": shape,
                "withheld_from_environment": True,
                "episode_id": context.episode_id,
            }
        )
        return ActionExecutionReport(
            status=PrimitiveStatus.INTERRUPTED,
            ended_timestep=context.timestep,
            observed_outcome="dry-run boundary; action chunk withheld from environment",
            requires_semantic_check=False,
            metadata={"dry_run": True, "action_count": shape[0]},
        )

    def disabled_skill(skill_id, _arguments, context):
        return PrimitiveExecutionReport(
            status=PrimitiveStatus.INTERRUPTED,
            started_timestep=context.timestep,
            ended_timestep=context.timestep,
            observed_outcome=f"skill unavailable in model-integration smoke: {skill_id}",
        )

    bindings = build_profile_admitted_tool_bindings(
        config,
        runtime=runtime,
        observation_source=lambda _context: policy_observation,
        action_executor=inspect_without_execution,
        observe=lambda _context: {
            "available_views": ["agentview"],
            "robot_state_dim": int(len(policy_observation["observation/state"])),
        },
        retrieve_memory=lambda query, limit, _context: {
            "query": query,
            "limit": limit,
            "records": [],
        },
        run_skill=disabled_skill,
        verify=lambda expected, _context: VerificationReport(
            status="inconclusive",
            observed_outcome=f"not executed: {expected}",
            confidence=0.0,
            metadata={"reason": "no environment execution in this smoke"},
        ),
        safe_hold=lambda reason, _context: {"holding": True, "reason": reason},
    )
    session = CarveAgentSession(config, bindings=bindings)
    episode_id = f"{config.run_id}:0"
    planner_context = HighLevelAgentContext(
        task_instruction=str(args.task),
        trigger="task_start",
        episode_id=episode_id,
        timestep=0,
        frames={"agentview": frame},
        robot_state=tuple(float(value) for value in policy_observation["observation/state"]),
        risk={
            "event": None,
            "bucket": "low",
            "score": 0.0,
            "components": {},
            "evidence": {"action_age_steps": 0},
        },
        current_subgoal="select the first bounded VLA primitive",
        deployment_profile_id=config.vla.deployment_profile_id,
        allowed_intents=(AgentIntent.VLA_ACT, AgentIntent.SAFE_STOP),
        remaining_retries=config.harness.retry_budget,
        remaining_recoveries=config.harness.recovery_budget,
        deadline_slack_ms=config.optimize.deadline_ms,
    )
    tool_context = ToolExecutionContext(
        episode_id=episode_id,
        timestep=0,
        at_safe_boundary=True,
        allowed_tools=config.harness.allowed_tools,
        deployment_profile_id=config.vla.deployment_profile_id,
    )

    submitted = session.start(planner_context)
    transition = session.await_planner()
    tool_result = session.apply_planner_transition(
        transition, context=tool_context
    )
    primitive = (
        None
        if tool_result is None
        else session.tools.primitive_outcome(tool_result, episode_id=episode_id)
    )
    session.close(reason="model-integration-smoke-complete")

    passed = bool(
        submitted.state is HarnessState.PLAN_AT_SAFE_BOUNDARY
        and transition.state is HarnessState.EXECUTE_FAST
        and transition.decision_applied
        and transition.planner is not None
        and transition.planner.result.accepted
        and tool_result is not None
        and tool_result.accepted
        and primitive is not None
        and primitive.status is PrimitiveStatus.INTERRUPTED
        and primitive.metadata.get("profile_id")
        == config.vla.deployment_profile_id
        and private_action_receipt.get("shape") == [10, 7]
        and private_action_receipt.get("withheld_from_environment") is True
    )
    receipt = {
        "schema_version": 1,
        "kind": "carve_canonical_local_models_smoke",
        "claim_boundary": (
            "Canonical session with real local Qwen-VL and profile-admitted PI0.5; "
            "the action chunk was withheld from the environment, so this is not "
            "task-success evidence."
        ),
        "passed": passed,
        "config_fingerprint": config.fingerprint,
        "workspace": str(config.workspace_path),
        "planner": {
            "mode": config.planner.mode.value,
            "accepted": (
                transition.planner is not None
                and transition.planner.result.accepted
            ),
            "state": transition.state.value,
            "decision": (
                None
                if transition.planner is None
                else transition.planner.result.decision.to_dict()
            ),
        },
        "tool": None if tool_result is None else tool_result.to_dict(),
        "private_action_receipt": private_action_receipt,
    }
    receipt_path = config.workspace_path / "model_integration_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
