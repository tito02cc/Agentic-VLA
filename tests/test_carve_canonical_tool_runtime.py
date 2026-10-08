"""End-to-end CPU tests for the canonical seven-tool runtime."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest

from agentic_vla.runtime import AgentIntent, HighLevelAgentDecision
from agentic_vla.toolchain import (
    CanonicalToolRuntime,
    EmbodiedToolBindings,
    PrimitiveExecutionReport,
    RunManifest,
    RunWorkspace,
    ToolExecutionContext,
    VerificationReport,
    core_tool_specs,
)


def make_runtime(tmp_path, *, observe=None):
    workspace = RunWorkspace(
        tmp_path / "run",
        RunManifest(
            run_id="run-1",
            environment_id="fixture",
            task_id="suite:task",
            seed=7,
            planner_id="scripted",
            policy_id="pi05",
            deployment_profile_id="pi05-compiled",
        ),
    )
    bindings = EmbodiedToolBindings(
        observe=observe or (lambda _ctx: {"frame_id": 3, "risk": "normal"}),
        retrieve_memory=lambda query, limit, _ctx: {
            "query": query,
            "records": ["verified"][:limit],
        },
        vla_act=lambda instruction, ctx: PrimitiveExecutionReport(
            status="succeeded",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 10,
            expected_outcome=instruction,
            observed_outcome="subgoal completed",
            requires_semantic_check=True,
            metadata={"deadline_miss": False},
        ),
        run_skill=lambda skill_id, _args, ctx: PrimitiveExecutionReport(
            status="succeeded",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 2,
            observed_outcome=f"{skill_id} completed",
        ),
        verify=lambda expected, _ctx: VerificationReport(
            status="confirmed",
            observed_outcome=expected,
            confidence=1.0,
        ),
        safe_hold=lambda reason, _ctx: {"holding": True, "reason": reason},
    )
    return CanonicalToolRuntime(bindings=bindings, workspace=workspace)


def context(*, timestep=4):
    return ToolExecutionContext(
        episode_id="episode-1",
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="pi05-compiled",
    )


def test_vla_decision_becomes_bounded_primitive_and_workspace_trace(tmp_path) -> None:
    runtime = make_runtime(tmp_path)
    decision = HighLevelAgentDecision(
        intent=AgentIntent.VLA_ACT,
        rationale="target is visible",
        confidence=0.9,
        vla_instruction="grasp the mug",
        expected_outcome="mug is visibly held",
    )

    result = runtime.invoke_decision(decision, context=context())

    assert result is not None and result.accepted
    outcome = result.output["primitive_outcome"]
    assert outcome["status"] == "succeeded"
    assert outcome["primitive_name"] == "vla_act"
    assert outcome["expected_outcome"] == "mug is visibly held"
    assert outcome["requires_semantic_check"]
    assert "actions" not in json.dumps(outcome)
    assert runtime.workspace.recipe_path.exists()
    event = json.loads(runtime.workspace.event_path.read_text().splitlines()[0])
    assert event["event_type"] == "tool_result"


def test_expected_outcome_does_not_force_chunk_level_semantic_verification(
    tmp_path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.bindings = EmbodiedToolBindings(
        **{
            **runtime.bindings.__dict__,
            "vla_act": lambda _instruction, ctx: PrimitiveExecutionReport(
                status="succeeded",
                started_timestep=ctx.timestep,
                ended_timestep=ctx.timestep + 10,
                requires_semantic_check=False,
            ),
        }
    )

    result = runtime.invoke_decision(
        HighLevelAgentDecision(
            intent=AgentIntent.VLA_ACT,
            rationale="execute task stage",
            confidence=0.9,
            vla_instruction="place the first moka pot on the stove",
            expected_outcome="first moka pot is on stove",
        ),
        context=context(),
    )

    assert result is not None and result.accepted
    outcome = result.output["primitive_outcome"]
    assert outcome["expected_outcome"] == "first moka pot is on stove"
    assert not outcome["requires_semantic_check"]


def test_coding_agent_and_vlm_share_identical_tool_boundary(tmp_path) -> None:
    runtime = make_runtime(tmp_path)
    direct = runtime.invoke(
        "run_skill",
        {"skill_id": "retract", "skill_args": {}},
        context=context(),
        call_id="codex-call",
    )
    model = runtime.invoke_decision(
        HighLevelAgentDecision(
            intent=AgentIntent.RUN_SKILL,
            rationale="failed grasp needs clearance",
            confidence=0.9,
            skill_id="retract",
        ),
        context=context(timestep=8),
    )

    assert direct.accepted and model is not None and model.accepted
    assert direct.output["primitive_outcome"]["primitive_name"] == "retract"
    assert model.output["primitive_outcome"]["primitive_name"] == "retract"


def test_raw_action_leak_is_rejected_and_finish_is_persisted(tmp_path) -> None:
    runtime = make_runtime(tmp_path, observe=lambda _ctx: {"actions": [[0.0] * 7]})
    rejected = runtime.invoke("observe", {}, context=context())
    assert not rejected.accepted
    assert "forbidden field" in str(rejected.error)

    finished = runtime.invoke(
        "finish",
        {"status": "safe_stop", "summary": "planner output rejected"},
        context=context(),
    )
    assert finished.accepted
    summary = json.loads(
        (runtime.workspace.root / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "safe_stop"


@pytest.mark.parametrize("name,arguments", [
    ("vla_act", {"instruction": "change target"}),
    ("run_skill", {"skill_id": "retract", "skill_args": {}}),
    ("verify", {"expected_outcome": "object placed"}),
    ("finish", {"status": "completed", "summary": "done"}),
    ("safe_hold", {"reason": "planner requested hold"}),
])
def test_active_vla_call_cannot_be_preempted_by_another_tool(tmp_path, name, arguments):
    runtime = make_runtime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    results = []

    def execute(instruction, ctx):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test synchronization timed out")
        return PrimitiveExecutionReport(
            status="succeeded", started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 10, observed_outcome="chunk returned",
            requires_semantic_check=True,
        )

    runtime.bindings = dataclasses.replace(runtime.bindings, vla_act=execute)
    worker = threading.Thread(target=lambda: results.append(
        runtime.invoke("vla_act", {"instruction": "keep active goal"}, context=context())
    ))
    worker.start()
    try:
        assert entered.wait(timeout=2)
        rejected = runtime.invoke(name, arguments, context=context())
        assert not rejected.accepted
        assert "another embodied tool is active" in rejected.error
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].accepted
    assert results[0].output["primitive_outcome"]["requires_semantic_check"]
    # Tool safe_hold shares the mutex; it is not a preemptive hardware stop.
    assert runtime.invoke("observe", {}, context=context(timestep=14)).accepted


def test_tool_hold_is_allowed_outside_chunk_boundary_when_executor_is_idle(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.invoke(
        "safe_hold", {"reason": "adapter hold requested"},
        context=dataclasses.replace(context(), at_safe_boundary=False),
    )
    assert result.accepted and result.output["holding"]
