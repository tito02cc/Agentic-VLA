"""Tests for the Codex-facing CARVE tool gateway."""

from __future__ import annotations

import json

import pytest

from agentic_vla.external_agent import ExternalToolAgentGateway
from agentic_vla.configuration import CarveRunConfig
from agentic_vla.runtime import HighLevelAgentContext
from agentic_vla.session import CarveAgentSession
from agentic_vla.toolchain import (
    EmbodiedToolBindings,
    PrimitiveExecutionReport,
    RunWorkspace,
    ToolExecutionContext,
    VerificationReport,
)


def bindings():
    return EmbodiedToolBindings(
        observe=lambda _context: {"risk_bucket": "low"},
        retrieve_memory=lambda query, limit, _context: {
            "query": query,
            "records": [],
            "limit": limit,
        },
        vla_act=lambda instruction, context: PrimitiveExecutionReport(
            status="interrupted",
            started_timestep=context.timestep,
            ended_timestep=context.timestep,
            expected_outcome=instruction,
            observed_outcome="test boundary",
        ),
        run_skill=lambda skill_id, _args, context: PrimitiveExecutionReport(
            status="interrupted",
            started_timestep=context.timestep,
            ended_timestep=context.timestep,
            observed_outcome=f"{skill_id} withheld",
        ),
        verify=lambda expected, _context: VerificationReport(
            status="inconclusive",
            observed_outcome=f"not checked: {expected}",
            confidence=0.0,
        ),
        safe_hold=lambda reason, _context: {"holding": True, "reason": reason},
    )


def planner_context():
    return HighLevelAgentContext(
        task_instruction="put the mug on the plate",
        trigger="task_start",
        episode_id="episode-1",
        timestep=8,
        risk={"score": 0.0, "bucket": "low", "components": {}, "evidence": {}},
    )


def test_codex_gateway_discovers_tools_and_uses_runtime_owned_context(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(config, bindings=bindings(), workspace=workspace)
    context = ToolExecutionContext(
        episode_id="episode-1",
        timestep=8,
        at_safe_boundary=True,
        allowed_tools=config.harness.allowed_tools,
        deployment_profile_id=config.vla.deployment_profile_id,
    )
    session.start(planner_context())
    gateway = ExternalToolAgentGateway(session, context_source=lambda: context)

    assert [tool["name"] for tool in gateway.catalog()["tools"]] == [
        "finish",
        "observe",
        "retrieve_memory",
        "run_skill",
        "safe_hold",
        "verify",
        "vla_act",
    ]
    response = gateway.invoke(
        {
            "name": "observe",
            "arguments": {},
            "expected_episode_id": "episode-1",
            "expected_timestep": 8,
        }
    )
    assert response["accepted"]
    transcript = json.loads(workspace.transcript_path.read_text().splitlines()[0])
    assert transcript["role"] == "tool"


def test_codex_gateway_rejects_stale_and_direct_action_requests(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    context = ToolExecutionContext(
        episode_id="episode-1",
        timestep=9,
        at_safe_boundary=True,
        allowed_tools=config.harness.allowed_tools,
    )
    gateway = ExternalToolAgentGateway(session, context_source=lambda: context)

    stale = gateway.invoke(
        {
            "name": "observe",
            "arguments": {},
            "expected_episode_id": "episode-1",
            "expected_timestep": 8,
        }
    )
    assert not stale["accepted"] and "stale" in stale["error"]
    with pytest.raises(ValueError, match="actions"):
        gateway.invoke(
            {
                "name": "vla_act",
                "arguments": {"instruction": "move", "actions": [[0.0] * 7]},
                "expected_episode_id": "episode-1",
                "expected_timestep": 9,
            }
        )
