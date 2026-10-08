"""Tests for the canonical CARVE agent session."""

from __future__ import annotations

import dataclasses
import json

import pytest

from agentic_vla.configuration import CarveRunConfig, PlannerRunConfig
from agentic_vla.runtime import (
    HighLevelAgentContext,
    HarnessState,
    ProceduralStep,
    RiskAssessment,
)
from agentic_vla.session import CarveAgentSession
from agentic_vla.toolchain import (
    EmbodiedToolBindings,
    PrimitiveExecutionReport,
    PrimitiveStatus,
    RunWorkspace,
    ToolExecutionContext,
    VerificationReport,
    VerificationStatus,
    core_tool_specs,
)


def bindings(verification_status="confirmed"):
    return EmbodiedToolBindings(
        observe=lambda _ctx: {"frame_id": 1},
        retrieve_memory=lambda query, limit, _ctx: {
            "query": query,
            "records": [] if limit else [],
        },
        vla_act=lambda instruction, ctx: PrimitiveExecutionReport(
            status="succeeded",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 4,
            expected_outcome=instruction,
            observed_outcome="done",
            requires_semantic_check=True,
        ),
        run_skill=lambda skill, _args, ctx: PrimitiveExecutionReport(
            status="succeeded",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 1,
            observed_outcome=f"{skill} done",
        ),
        verify=lambda expected, _ctx: VerificationReport(
            status=verification_status,
            observed_outcome=(
                expected
                if verification_status == "confirmed"
                else f"expected outcome not observed: {expected}"
            ),
            confidence=1.0 if verification_status != "inconclusive" else 0.0,
        ),
        safe_hold=lambda reason, _ctx: {"holding": True, "reason": reason},
    )


def planner_context(*, episode="episode-1", timestep=0, trigger="control_boundary"):
    return HighLevelAgentContext(
        task_instruction="put the mug on the plate",
        trigger=trigger,
        episode_id=episode,
        timestep=timestep,
        frames={},
        robot_state=(0.0,) * 8,
        risk={"score": 0.0, "bucket": "low", "components": {}, "evidence": {}},
        available_skills=("retract",),
        remaining_retries=2,
        remaining_recoveries=2,
        deployment_profile_id="pi05-compiled-smve",
    )


def tool_context(*, timestep=0):
    return ToolExecutionContext(
        episode_id="episode-1",
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="pi05-compiled-smve",
    )


def test_embedded_scripted_planner_runs_through_tool_and_primitive_boundary(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-scripted",
        planner=PlannerRunConfig(
            mode="scripted",
            planner_id="scripted",
            model="fixture",
            max_calls_per_episode=3,
        ),
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "vla_act",
                "rationale": "target is visible",
                "confidence": 0.9,
                "subgoal": "move mug",
                "vla_instruction": "put the mug on the plate",
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        scripted_infer=infer,
        workspace=workspace,
    )
    submitted = session.start(planner_context())
    assert submitted.state is HarnessState.PLAN_AT_SAFE_BOUNDARY
    transition = session.await_planner(timeout_s=1.0)
    result = session.apply_planner_transition(transition, context=tool_context())
    assert result is not None and result.accepted
    outcome = session.tools.primitive_outcome(result, episode_id="episode-1")
    boundary_context = planner_context(timestep=outcome.ended_timestep)
    _outcome, boundary = session.record_primitive_result(
        result, planner_context=boundary_context
    )
    assert boundary.ticket is not None
    assert workspace.transcript_path.exists()
    session.close()


def test_codex_session_has_explicit_tool_authority_only(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(config, bindings=bindings(), workspace=workspace)

    started = session.start(planner_context())
    assert started.ticket is None
    result = session.invoke_external_tool(
        "observe", {}, context=tool_context()
    )
    assert result.accepted and result.output["frame_id"] == 1
    session.close()


def test_session_records_scheduled_semantic_checkpoint(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-semantic-checkpoint",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
        harness=dataclasses.replace(base.harness, task_start_policy="event_only"),
    )

    def infer(_request):
        return {
            "intent": "vla_act",
            "rationale": "remaining target is visible",
            "confidence": 0.9,
            "subgoal": "move remaining mug",
            "vla_instruction": "put the remaining mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "failure_type": "",
            "scene_graph_update": {},
        }

    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        scripted_infer=infer,
        workspace=workspace,
    )
    session.start(planner_context())
    transition = session.request_semantic_checkpoint(
        dataclasses.replace(
            planner_context(timestep=250),
            trigger="scheduled_semantic_checkpoint",
        ),
        reason="scheduled semantic checkpoint at step 250",
    )

    assert transition.ticket is not None
    resolved = session.await_planner(timeout_s=1.0)
    assert resolved.decision_applied
    events = [
        json.loads(line)
        for line in (workspace.root / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["event_type"] == "semantic_checkpoint_requested" for event in events)
    session.close()


def test_session_supersedes_late_planner_after_plan_completion(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-late-planner",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
        harness=dataclasses.replace(base.harness, task_start_policy="event_only"),
    )

    def infer(_request):
        return {
            "intent": "vla_act",
            "rationale": "retry the active stage",
            "confidence": 0.9,
            "subgoal": "place mug on plate",
            "vla_instruction": "put the mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is on plate",
            "proposed_plan": [],
            "failure_type": "",
            "scene_graph_update": {},
        }

    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        scripted_infer=infer,
        workspace=workspace,
    )
    session.start(planner_context())
    session.task_plan.install(
        (
            ProceduralStep(
                stage="place",
                intent="vla_act",
                subgoal="place mug on plate",
                expected_outcome="mug is on plate",
            ),
        )
    )
    submitted = session.request_semantic_checkpoint(
        dataclasses.replace(
            planner_context(timestep=250),
            trigger="scheduled_semantic_checkpoint",
            task_plan=session.task_plan.to_dict(),
        ),
        reason="scheduled semantic checkpoint at step 250",
    )
    assert submitted.ticket is not None
    resolved = session.await_planner(timeout_s=1.0)

    session.task_plan.apply_verification(
        VerificationReport(
            status="confirmed",
            observed_outcome="mug is on plate",
            confidence=1.0,
        )
    )
    result = session.apply_planner_transition(resolved, context=tool_context())

    assert result is None
    assert session.task_plan.completed
    events = [
        json.loads(line) for line in workspace.event_path.read_text().splitlines()
    ]
    superseded = [
        event for event in events
        if event["event_type"] == "planner_decision_superseded"
    ]
    assert len(superseded) == 1
    assert superseded[0]["payload"]["intent"] == "vla_act"
    session.close()


def test_finalize_primitive_skips_vlm_after_confirmed_verification(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-verified",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "vla_act",
                "rationale": "target is visible",
                "confidence": 0.9,
                "subgoal": "move mug",
                "vla_instruction": "put the mug on the plate",
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    session = CarveAgentSession(
        config,
        bindings=bindings("confirmed"),
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    transition = session.await_planner(timeout_s=1.0)
    result = session.apply_planner_transition(transition, context=tool_context())
    completion = session.finalize_primitive(
        result, planner_context=planner_context(timestep=4)
    )

    assert completion.verification.status is VerificationStatus.CONFIRMED
    assert completion.outcome.status is PrimitiveStatus.SUCCEEDED
    assert not completion.outcome.requires_semantic_check
    assert completion.transition.ticket is None
    assert not completion.memory_recorded


def test_session_installs_and_advances_harness_owned_task_plan(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-task-plan",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return {
            "intent": "vla_act",
            "rationale": "execute first planned stage",
            "confidence": 0.9,
            "subgoal": "place mug",
            "vla_instruction": "put the mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is on plate",
            "proposed_plan": [
                {
                    "stage": "place",
                    "intent": "vla_act",
                    "subgoal": "place mug",
                    "expected_outcome": "mug is on plate",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "retract",
                    "intent": "run_skill",
                    "subgoal": "clear gripper",
                    "expected_outcome": "gripper is clear",
                    "skill_id": "retract",
                    "constraints": [],
                },
            ],
        }

    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(
        config,
        bindings=bindings("confirmed"),
        scripted_infer=infer,
        workspace=workspace,
    )
    session.start(planner_context(trigger="task_start"))
    transition = session.await_planner(timeout_s=1.0)
    before = session.task_plan.to_dict()
    blocked = session.apply_planner_transition(
        transition,
        context=dataclasses.replace(tool_context(), at_safe_boundary=False),
    )
    assert blocked is None
    assert session.task_plan.to_dict() == before
    assert not workspace.recipe_path.exists()
    result = session.apply_planner_transition(transition, context=tool_context())

    assert session.task_plan.active is not None
    assert session.task_plan.active.step.stage == "place"
    assert session.task_plan.active.attempts == 1

    completion = session.finalize_primitive(
        result,
        planner_context=dataclasses.replace(
            planner_context(timestep=4), current_subgoal="place mug"
        ),
    )

    assert completion.verification.status is VerificationStatus.CONFIRMED
    assert session.task_plan.active is not None
    assert session.task_plan.active.step.stage == "retract"
    session.select_active_task_plan_step(
        timestep=4,
        source="verified_plan_progression",
    )
    assert session.task_plan.active.attempts == 1
    events = [
        json.loads(line)
        for line in workspace.event_path.read_text().splitlines()
    ]
    assert any(event["event_type"] == "task_plan_installed" for event in events)
    assert any(event["event_type"] == "task_plan_step_verified" for event in events)
    assert any(
        event["event_type"] == "task_plan_step_selected"
        and event["source"] == "verified_plan_progression"
        for event in events
    )


def test_recovery_verification_cannot_complete_active_vla_stage(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-recovery-plan-gate",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return {
            "intent": "vla_act",
            "rationale": "execute planned placement",
            "confidence": 0.9,
            "subgoal": "place mug",
            "vla_instruction": "put the mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is on plate",
            "proposed_plan": [
                {
                    "stage": "place",
                    "intent": "vla_act",
                    "subgoal": "place mug",
                    "expected_outcome": "mug is on plate",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

    base_bindings = bindings("confirmed")
    recovery_bindings = dataclasses.replace(
        base_bindings,
        run_skill=lambda skill, _args, ctx: PrimitiveExecutionReport(
            status="succeeded",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep + 1,
            expected_outcome="tool cleared contact",
            observed_outcome=f"{skill} moved",
            requires_semantic_check=True,
        ),
    )
    session = CarveAgentSession(
        config,
        bindings=recovery_bindings,
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context(trigger="task_start"))
    transition = session.await_planner(timeout_s=1.0)
    vla_result = session.apply_planner_transition(transition, context=tool_context())
    assert vla_result is not None
    recovery_result = session.tools.invoke(
        "run_skill",
        {"skill_id": "retract", "skill_args": {}},
        context=tool_context(timestep=4),
    )

    completion = session.finalize_primitive(
        recovery_result,
        planner_context=dataclasses.replace(
            planner_context(timestep=5), current_subgoal="place mug"
        ),
    )

    assert completion.verification.status is VerificationStatus.CONFIRMED
    assert session.task_plan.active is not None
    assert session.task_plan.active.step.stage == "place"
    assert session.task_plan.active.status.value == "active"


def test_finalize_primitive_records_only_verified_contradiction(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-contradicted",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "vla_act",
                "rationale": "target is visible",
                "confidence": 0.9,
                "subgoal": "move mug",
                "vla_instruction": "put the mug on the plate",
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    session = CarveAgentSession(
        config,
        bindings=bindings("contradicted"),
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    transition = session.await_planner(timeout_s=1.0)
    result = session.apply_planner_transition(transition, context=tool_context())
    completion = session.finalize_primitive(
        result, planner_context=planner_context(timestep=4)
    )

    assert completion.verification.status is VerificationStatus.CONTRADICTED
    assert completion.outcome.status is PrimitiveStatus.FAILED
    assert completion.memory_recorded
    assert len(session.harness.failure_memory) == 1
    assert completion.transition.ticket is not None


def test_finalize_primitive_never_persists_inconclusive_evidence(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-inconclusive",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "vla_act",
                "rationale": "target is visible",
                "confidence": 0.9,
                "subgoal": "move mug",
                "vla_instruction": "put the mug on the plate",
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    session = CarveAgentSession(
        config,
        bindings=bindings("inconclusive"),
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    transition = session.await_planner(timeout_s=1.0)
    result = session.apply_planner_transition(transition, context=tool_context())
    completion = session.finalize_primitive(
        result, planner_context=planner_context(timestep=4)
    )

    assert completion.verification.status is VerificationStatus.INCONCLUSIVE
    assert not completion.memory_recorded
    assert len(session.harness.failure_memory) == 0
    assert completion.transition.ticket is not None


def test_session_finish_persists_summary_and_closes_episode(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    workspace = RunWorkspace(tmp_path / "run", config.to_run_manifest())
    session = CarveAgentSession(config, bindings=bindings(), workspace=workspace)
    session.start(planner_context())

    result = session.finish(
        status="success",
        summary="task completed through the canonical tool boundary",
        timestep=8,
    )
    summary = json.loads(
        (workspace.root / "run_summary.json").read_text(encoding="utf-8")
    )

    assert result.accepted
    assert summary["status"] == "success"
    assert summary["summary"] == "task completed through the canonical tool boundary"
    assert session.harness.episode_id is None
    session.close()


def test_session_enforces_configured_recovery_budget_for_external_agent(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    config = dataclasses.replace(
        base,
        run_id="session-recovery-budget",
        harness=dataclasses.replace(base.harness, recovery_budget=2),
    )
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())

    first = session.invoke_external_tool(
        "run_skill",
        {"skill_id": "retract", "skill_args": {}},
        context=tool_context(),
    )
    second = session.invoke_external_tool(
        "run_skill",
        {"skill_id": "retract", "skill_args": {}},
        context=tool_context(),
    )
    exhausted = session.invoke_external_tool(
        "run_skill",
        {"skill_id": "retract", "skill_args": {}},
        context=tool_context(),
    )

    assert first.accepted and second.accepted
    assert not exhausted.accepted
    assert "budget exhausted" in str(exhausted.error)
    assert session.harness.counters.recovery_attempts == 2


def test_session_enters_safe_hold_after_semantic_retry_budget(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-semantic-budget",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
        harness=dataclasses.replace(base.harness, retry_budget=1),
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "vla_act",
                "rationale": "retry the bounded VLA primitive",
                "confidence": 0.9,
                "subgoal": "move mug",
                "vla_instruction": "put the mug on the plate",
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    session = CarveAgentSession(
        config,
        bindings=bindings("contradicted"),
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    initial = session.await_planner(timeout_s=1.0)
    first_result = session.apply_planner_transition(initial, context=tool_context())
    first_completion = session.finalize_primitive(
        first_result, planner_context=planner_context(timestep=4)
    )
    assert first_completion.transition.ticket is not None

    retry = session.await_planner(timeout_s=1.0)
    second_result = session.apply_planner_transition(
        retry, context=tool_context(timestep=4)
    )
    second_completion = session.finalize_primitive(
        second_result, planner_context=planner_context(timestep=8)
    )

    assert second_completion.transition.state is HarnessState.SAFE_HOLD
    assert second_completion.transition.ticket is None
    assert second_completion.transition.reason == "semantic retry budget exhausted"
    assert session.harness.counters.semantic_retries == 1


def test_session_routes_monitor_decision_and_forwards_optimize_controls(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    seen_controls = []
    routed_bindings = dataclasses.replace(
        bindings(),
        vla_act=lambda instruction, ctx: (
            seen_controls.append((instruction, dict(ctx.inference_controls)))
            or PrimitiveExecutionReport(
                status="succeeded",
                started_timestep=ctx.timestep,
                ended_timestep=ctx.timestep + 4,
                observed_outcome="chunk executed",
            )
        ),
    )
    session = CarveAgentSession(
        config,
        bindings=routed_bindings,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    transition = session.route_execution(
        RiskAssessment(
            score=0.0,
            bucket="low",
            event=None,
            components={},
            evidence={},
        ),
        planner_context=planner_context(),
        deadline_ms=80.0,
        deadline_slack_ms=80.0,
    )
    result = session.apply_execution_transition(
        transition,
        context=tool_context(),
        vla_instruction="put the mug on the plate",
    )

    assert result is not None and result.accepted
    assert seen_controls[0][0] == "put the mug on the plate"
    assert seen_controls[0][1]["inference_steps"] == 2
    assert seen_controls[0][1]["max_actions"] == 10
    assert seen_controls[0][1]["deadline_ms"] == 80.0


def test_session_promotes_only_verified_primitive_trace(tmp_path) -> None:
    base = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    config = dataclasses.replace(
        base,
        run_id="session-procedure-promotion",
        planner=PlannerRunConfig(
            mode="scripted", planner_id="scripted", model="fixture"
        ),
    )

    def infer(_request):
        return {
            "intent": "vla_act",
            "rationale": "target is visible",
            "confidence": 0.9,
            "subgoal": "place mug on plate",
            "vla_instruction": "put the mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "failure_type": "",
            "scene_graph_update": {},
        }

    session = CarveAgentSession(
        config,
        bindings=bindings("confirmed"),
        scripted_infer=infer,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    transition = session.await_planner(timeout_s=1.0)
    result = session.apply_planner_transition(transition, context=tool_context())
    session.finalize_primitive(
        result,
        planner_context=dataclasses.replace(
            planner_context(timestep=4),
            current_subgoal="place mug on plate",
        ),
    )
    record = session.promote_verified_procedure(
        task_family="mug plate placement",
        object_categories=("mug", "plate"),
        task_verification=VerificationReport(
            status="confirmed",
            observed_outcome="private evaluator confirmed task completion",
            confidence=1.0,
        ),
    )

    assert len(session.procedure_trace) == 1
    assert record.steps[0].intent == "vla_act"
    retrieved = session.knowledge_provider.procedural_memory.retrieve(
        task_instruction="put the mug on the plate",
        limit=1,
    )
    assert retrieved[0]["procedure_id"] == record.procedure_id
    events = [
        json.loads(line)
        for line in session.workspace.event_path.read_text().splitlines()
    ]
    assert events[-1]["event_type"] == "procedure_promoted"


def test_session_warm_starts_from_verified_symbolic_procedure(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    steps = (
        ProceduralStep(
            stage="place_first",
            intent="vla_act",
            subgoal="place first object",
            expected_outcome="first object is placed",
        ),
        ProceduralStep(
            stage="place_second",
            intent="vla_act",
            subgoal="place second object",
            expected_outcome="both objects are placed",
        ),
    )

    session.install_retrieved_task_plan(
        steps,
        available_skills=("retract",),
        timestep=0,
        procedure_id="verified-procedure-1",
    )
    session.select_active_task_plan_step(timestep=0, source="verified_procedure_memory")

    assert session.task_plan.installed
    assert session.task_plan.active is not None
    assert session.task_plan.active.step.stage == "place_first"
    assert session.task_plan.active.attempts == 1
    events = [json.loads(line) for line in session.workspace.event_path.read_text().splitlines()]
    installed = next(event for event in events if event["event_type"] == "task_plan_installed")
    assert installed["source"] == "verified_procedure_memory"
    assert installed["payload"]["procedure_id"] == "verified-procedure-1"


def test_session_binds_cumulative_outcome_to_warm_started_final_stage(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    session.install_retrieved_task_plan(
        (
            ProceduralStep(
                stage="first",
                intent="vla_act",
                subgoal="place first moka pot on stove",
                expected_outcome="first moka pot is on stove",
            ),
            ProceduralStep(
                stage="second",
                intent="vla_act",
                subgoal="place second moka pot on stove",
                expected_outcome="second moka pot is on stove",
            ),
        ),
        available_skills=(),
        timestep=0,
        procedure_id="moka-procedure",
        task_instruction="put both moka pots on the stove",
    )

    assert session.task_plan.receipts[-1].step.expected_outcome == (
        "full task visibly satisfied: put both moka pots on the stove"
    )


def test_session_promotes_only_a_completed_installed_task_plan(tmp_path) -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    session = CarveAgentSession(
        config,
        bindings=bindings(),
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(planner_context())
    steps = (
        ProceduralStep(
            stage="place_left",
            intent="vla_act",
            subgoal="place white mug on left plate",
            expected_outcome="white mug is on left plate",
        ),
        ProceduralStep(
            stage="place_right",
            intent="vla_act",
            subgoal="place yellow and white mug on right plate",
            expected_outcome="yellow and white mug is on right plate",
        ),
    )
    session.install_retrieved_task_plan(
        steps,
        available_skills=("retract",),
        timestep=0,
        procedure_id="candidate-plan",
    )
    task_verification = VerificationReport(
        status="confirmed",
        observed_outcome="trusted evaluator confirmed the complete task",
        confidence=1.0,
    )

    with pytest.raises(RuntimeError, match="task plan to be complete"):
        session.promote_verified_procedure(
            task_family="dual mug placement",
            object_categories=("mug", "plate"),
            task_verification=task_verification,
        )

    session.verify_active_plan_step(
        task_verification,
        timestep=10,
        source="trusted_task_verifier",
    )
    session.verify_active_plan_step(
        task_verification,
        timestep=10,
        source="trusted_task_verifier",
    )
    record = session.promote_verified_procedure(
        task_family="dual mug placement",
        object_categories=("mug", "plate"),
        task_verification=task_verification,
    )

    assert session.task_plan.completed
    assert record.steps == steps
