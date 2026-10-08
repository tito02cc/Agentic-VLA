"""CPU tests for CARVE's agent-facing embodied tool protocol."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest

from agentic_vla.toolchain import (
    EmbodiedToolRegistry,
    PrimitiveOutcome,
    PrimitiveStatus,
    RunManifest,
    RunWorkspace,
    ToolCall,
    ToolExecutionContext,
    core_tool_specs,
)


def context(*, safe: bool = True) -> ToolExecutionContext:
    return ToolExecutionContext(
        episode_id="episode-1",
        timestep=12,
        at_safe_boundary=safe,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="pi05-compiled-smve",
    )


def call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(
        call_id=f"call-{name}",
        name=name,
        arguments=arguments,
        episode_id="episode-1",
        timestep=12,
    )


def registry() -> EmbodiedToolRegistry:
    tools = EmbodiedToolRegistry()
    for spec in core_tool_specs():
        tools.register(
            spec,
            lambda arguments, ctx, name=spec.name: {
                "tool": name,
                "arguments": dict(arguments),
                "profile": ctx.deployment_profile_id,
            },
        )
    return tools


def test_catalog_exposes_stable_embodied_tools() -> None:
    assert tuple(spec.name for spec in core_tool_specs()) == (
        "observe",
        "retrieve_memory",
        "vla_act",
        "run_skill",
        "verify",
        "safe_hold",
        "finish",
    )


def test_primitive_outcome_is_bounded_and_rejects_raw_actions() -> None:
    outcome = PrimitiveOutcome(
        call_id="vla-1",
        primitive_name="vla_act",
        status="succeeded",
        episode_id="episode-1",
        started_timestep=12,
        ended_timestep=32,
        expected_outcome="object is grasped",
        requires_semantic_check=True,
    )

    assert outcome.status is PrimitiveStatus.SUCCEEDED
    assert not outcome.abnormal
    assert outcome.to_planner_dict()["primitive_name"] == "vla_act"

    with pytest.raises(ValueError, match="actions"):
        PrimitiveOutcome(
            call_id="unsafe",
            primitive_name="run_skill",
            status="failed",
            episode_id="episode-1",
            started_timestep=12,
            ended_timestep=13,
            metadata={"actions": [[0.0] * 7]},
        )

    with pytest.raises(ValueError, match="object_pose"):
        PrimitiveOutcome(
            call_id="privileged",
            primitive_name="vla_act",
            status="succeeded",
            episode_id="episode-1",
            started_timestep=12,
            ended_timestep=13,
            metadata={"object_pose": [0.0, 0.0, 0.0]},
        )


def test_dispatches_typed_vla_call_and_emits_trace() -> None:
    traces = []
    tools = EmbodiedToolRegistry(trace_sink=traces.append)
    spec = next(item for item in core_tool_specs() if item.name == "vla_act")
    tools.register(spec, lambda arguments, _: {"instruction": arguments["instruction"]})

    result = tools.execute(call("vla_act", {"instruction": "close microwave door"}), context())

    assert result.accepted
    assert result.output["instruction"] == "close microwave door"
    assert traces == [result]


def test_rejects_stateful_tool_outside_safe_boundary() -> None:
    result = registry().execute(
        call("run_skill", {"skill_id": "retract", "skill_args": {}}),
        context(safe=False),
    )

    assert not result.accepted
    assert "safe execution boundary" in str(result.error)


def test_rejects_direct_action_and_stale_episode_inputs() -> None:
    with pytest.raises(ValueError, match="actions"):
        call("vla_act", {"instruction": "move", "nested": {"actions": [[0.0] * 7]}})

    stale = ToolCall(
        call_id="stale",
        name="observe",
        arguments={},
        episode_id="episode-2",
        timestep=12,
    )
    result = registry().execute(stale, context())
    assert not result.accepted
    assert "another episode" in str(result.error)


def test_enforces_schema_allowlist_and_per_episode_budget() -> None:
    tools = registry()
    unknown = tools.execute(
        call("vla_act", {"instruction": "move", "extra": True}),
        context(),
    )
    assert not unknown.accepted
    assert "unknown fields" in str(unknown.error)

    restricted = ToolExecutionContext(
        episode_id="episode-1",
        timestep=12,
        at_safe_boundary=True,
        allowed_tools=("observe",),
    )
    denied = tools.execute(
        call("run_skill", {"skill_id": "retract", "skill_args": {}}),
        restricted,
    )
    assert not denied.accepted
    assert "not allowed" in str(denied.error)

    for index in range(3):
        result = tools.execute(
            ToolCall(
                call_id=f"skill-{index}",
                name="run_skill",
                arguments={"skill_id": "retract", "skill_args": {}},
                episode_id="episode-1",
                timestep=12,
            ),
            context(),
        )
        assert result.accepted
    exhausted = tools.execute(
        ToolCall(
            call_id="skill-4",
            name="run_skill",
            arguments={"skill_id": "retract", "skill_args": {}},
            episode_id="episode-1",
            timestep=12,
        ),
        context(),
    )
    assert not exhausted.accepted
    assert "budget exhausted" in str(exhausted.error)


def test_single_active_operation_rejects_concurrent_tool() -> None:
    entered = threading.Event()
    release = threading.Event()
    tools = EmbodiedToolRegistry()
    observe = next(item for item in core_tool_specs() if item.name == "observe")

    def blocking_handler(_arguments, _context):
        entered.set()
        release.wait(timeout=1.0)
        return {"ok": True}

    tools.register(observe, blocking_handler)
    first_result = []
    worker = threading.Thread(
        target=lambda: first_result.append(tools.execute(call("observe", {}), context()))
    )
    worker.start()
    assert entered.wait(timeout=1.0)
    second = tools.execute(
        ToolCall(
            call_id="observe-2",
            name="observe",
            arguments={},
            episode_id="episode-1",
            timestep=12,
        ),
        context(),
    )
    release.set()
    worker.join(timeout=1.0)

    assert not second.accepted
    assert "another embodied tool" in str(second.error)
    assert first_result[0].accepted


def test_run_workspace_persists_manifest_recipe_trace_and_summary(tmp_path) -> None:
    workspace = RunWorkspace(
        tmp_path / "run",
        RunManifest(
            run_id="libero-pro-t9-s7",
            environment_id="libero-pro",
            task_id="libero_10_task:9",
            seed=7,
            planner_id="carve-guarded-vlm",
            policy_id="pi05-libero",
            deployment_profile_id="pi05-compiled-smve",
        ),
    )
    tool_call = call("vla_act", {"instruction": "close microwave door"})
    tool_result = registry().execute(tool_call, context())

    workspace.append_event("monitor", {"risk": "stall"}, source="harness")
    workspace.append_recipe(tool_call, tool_result)
    workspace.append_transcript(role="planner", content="close the microwave door")
    video = workspace.root / "episode.mp4"
    video.write_bytes(b"video-placeholder")
    workspace.register_artifact("episode.mp4", path=video, kind="episode_video")
    summary = workspace.finish(status="success", summary="task completed")

    manifest = json.loads((workspace.root / "run_manifest.json").read_text())
    recipe = json.loads(workspace.recipe_path.read_text().splitlines()[0])
    final = json.loads(summary.read_text())
    assert manifest["deployment_profile_id"] == "pi05-compiled-smve"
    assert recipe["tool"] == "vla_act"
    assert "output" not in recipe
    assert final["status"] == "success"
    assert final["artifacts"] == ["episode.mp4"]


def test_run_workspace_supports_evaluator_blind_completed_status(tmp_path) -> None:
    workspace = RunWorkspace(
        tmp_path / "blind-complete",
        RunManifest(
            run_id="blind-complete",
            environment_id="sim",
            task_id="task",
            seed=0,
            planner_id="planner",
            policy_id="policy",
            deployment_profile_id="profile",
        ),
    )

    summary = json.loads(
        workspace.finish(
            status="completed",
            summary="Control rollout ended; evaluator result is private.",
        ).read_text()
    )

    assert summary["status"] == "completed"


def test_run_workspace_rejects_artifact_escape_and_raw_action_recipe(tmp_path) -> None:
    workspace = RunWorkspace(
        tmp_path / "run",
        RunManifest(
            run_id="run",
            environment_id="libero-pro",
            task_id="task-9",
            seed=7,
            planner_id="planner",
            policy_id="pi05",
            deployment_profile_id="profile",
        ),
    )
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="inside"):
        workspace.register_artifact("outside.mp4", path=outside, kind="video")

    with pytest.raises(ValueError, match="trajectory"):
        ToolCall(
            call_id="unsafe",
            name="run_skill",
            arguments={"skill_id": "unsafe", "trajectory": [[0.0] * 7]},
            episode_id="episode-1",
            timestep=12,
        )


def test_run_workspace_resume_preserves_sequence_and_artifacts(tmp_path) -> None:
    manifest = RunManifest(
        run_id="resume-run",
        environment_id="libero-pro",
        task_id="task-9",
        seed=7,
        planner_id="planner",
        policy_id="pi05",
        deployment_profile_id="profile",
    )
    root = tmp_path / "run"
    first = RunWorkspace(root, manifest)
    first.append_event("episode_start", {"episode_id": "episode-1"}, source="test")
    video = root / "episode.mp4"
    video.write_bytes(b"video-placeholder")
    first.register_artifact("episode.mp4", path=video, kind="episode_video")

    resumed = RunWorkspace(root, manifest)
    resumed_event = resumed.append_event(
        "episode_resume", {"episode_id": "episode-1"}, source="test"
    )
    summary = json.loads(
        resumed.finish(status="success", summary="resumed run completed").read_text()
    )

    assert resumed_event["sequence"] == 1
    assert summary["event_count"] == 2
    assert summary["artifacts"] == ["episode.mp4"]


def test_run_workspace_rejects_manifest_mismatch_on_resume(tmp_path) -> None:
    manifest = RunManifest(
        run_id="resume-run",
        environment_id="libero-pro",
        task_id="task-9",
        seed=7,
        planner_id="planner",
        policy_id="pi05",
        deployment_profile_id="profile",
    )
    root = tmp_path / "run"
    RunWorkspace(root, manifest)

    with pytest.raises(ValueError, match="different manifest"):
        RunWorkspace(root, dataclasses.replace(manifest, seed=8))
