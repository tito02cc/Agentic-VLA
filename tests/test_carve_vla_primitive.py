"""Tests for profile-admitted VLA primitive execution."""

from __future__ import annotations

import dataclasses

from agentic_vla.assembly import build_profile_admitted_tool_bindings
from agentic_vla.configuration import CarveRunConfig
from agentic_vla.runtime import CarveRuntime
from agentic_vla.runtime.adapters import Pi05Adapter
from agentic_vla.toolchain import PrimitiveStatus, ToolExecutionContext, core_tool_specs
from agentic_vla.toolchain.vla import (
    ActionExecutionReport,
    ProfileAdmittedVlaPrimitive,
)


class FakePromotedPi05:
    def __init__(self, profile_id: str) -> None:
        self.executed = 0
        self.last_payload = None
        self._server_metadata = {
            "carve_capabilities": {"configurable_inference_steps": True},
            "carve_deployment_profile": {
                "backend_id": "torch_compile_masked_views",
                "profile": {"profile_id": profile_id},
            },
        }

    def infer(self, payload):
        self.executed += 1
        self.last_payload = payload
        assert payload["runtime_controls"]["inference_steps"] == 2
        return {
            "actions": [[0.0] * 7 for _ in range(10)],
            "policy_timing": {"infer_ms": 40.0},
        }


def context(*, inference_controls=None):
    return ToolExecutionContext(
        episode_id="episode-1",
        timestep=12,
        at_safe_boundary=True,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="pi05-torch_compile_masked_views-bf16-2step-h10",
        inference_controls=inference_controls or {},
    )


def test_profile_admitted_vla_keeps_actions_private_and_returns_receipt() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    policy = FakePromotedPi05(config.vla.deployment_profile_id)
    runtime = CarveRuntime(Pi05Adapter(policy, adapter_id="pi05"), fallback_mode="strict")
    action_chunks = []

    def execute(actions, ctx):
        action_chunks.append(actions)
        return ActionExecutionReport(
            status="succeeded",
            ended_timestep=ctx.timestep + len(actions),
            observed_outcome="chunk executed",
            requires_semantic_check=True,
            metadata={"protected_contact": False},
        )

    primitive = ProfileAdmittedVlaPrimitive(
        config=config,
        runtime=runtime,
        observation_source=lambda _ctx: {"observation/state": [0.0] * 8},
        action_executor=execute,
    )
    report = primitive("close the drawer", context())

    assert report.status.value == "succeeded"
    assert len(action_chunks[0]) == 10
    assert report.metadata["profile_id"] == config.vla.deployment_profile_id
    assert report.metadata["action_count"] == 10
    assert "actions" not in report.to_dict()


def test_unadmitted_runtime_receipt_fails_before_action_execution() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    policy = FakePromotedPi05("unknown-profile")
    runtime = CarveRuntime(Pi05Adapter(policy, adapter_id="pi05"), fallback_mode="strict")
    executed = []
    primitive = ProfileAdmittedVlaPrimitive(
        config=config,
        runtime=runtime,
        observation_source=lambda _ctx: {},
        action_executor=lambda actions, _ctx: executed.append(actions),
    )

    report = primitive("move", context())

    assert report.status.value == "failed"
    assert report.metadata["error_type"] == "ValueError"
    assert executed == []


def test_static_manifest_identity_mismatch_is_rejected() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    bad = dataclasses.replace(config.vla, model_id="different-model")
    config = dataclasses.replace(config, vla=bad)
    runtime = CarveRuntime(
        Pi05Adapter(
            FakePromotedPi05(config.vla.deployment_profile_id), adapter_id="pi05"
        )
    )

    try:
        ProfileAdmittedVlaPrimitive(
            config=config,
            runtime=runtime,
            observation_source=lambda _ctx: {},
            action_executor=lambda _actions, _ctx: None,
        )
    except ValueError as exc:
        assert "model_id" in str(exc)
    else:
        raise AssertionError("mismatched manifest should be rejected")


def test_harness_commit_horizon_is_applied_inside_admitted_profile() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    policy = FakePromotedPi05(config.vla.deployment_profile_id)
    runtime = CarveRuntime(Pi05Adapter(policy, adapter_id="pi05"), fallback_mode="strict")
    executed = []
    primitive = ProfileAdmittedVlaPrimitive(
        config=config,
        runtime=runtime,
        observation_source=lambda _ctx: {"observation/state": [0.0] * 8},
        action_executor=lambda actions, ctx: (
            executed.append(actions)
            or ActionExecutionReport(
                status="succeeded",
                ended_timestep=ctx.timestep + len(actions),
                observed_outcome="bounded chunk executed",
            )
        ),
    )

    report = primitive(
        "close the drawer",
        context(
            inference_controls={
                "inference_steps": 2,
                "max_actions": 4,
                "precision": "bf16",
                "reuse_context": False,
                "deadline_ms": 80.0,
            }
        ),
    )

    assert report.status is PrimitiveStatus.SUCCEEDED
    assert len(executed[0]) == 4
    assert report.metadata["action_count"] == 4


def test_harness_cannot_request_controls_outside_admitted_profile() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    policy = FakePromotedPi05(config.vla.deployment_profile_id)
    runtime = CarveRuntime(Pi05Adapter(policy, adapter_id="pi05"), fallback_mode="strict")
    executed = []
    primitive = ProfileAdmittedVlaPrimitive(
        config=config,
        runtime=runtime,
        observation_source=lambda _ctx: {},
        action_executor=lambda actions, _ctx: executed.append(actions),
    )

    report = primitive(
        "move",
        context(inference_controls={"inference_steps": 3}),
    )

    assert report.status is PrimitiveStatus.FAILED
    assert "outside the admitted profile" in report.metadata["error_message"]
    assert policy.executed == 0
    assert executed == []


def test_canonical_factory_builds_profile_admitted_seven_tool_bindings() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    policy = FakePromotedPi05(config.vla.deployment_profile_id)
    runtime = CarveRuntime(
        Pi05Adapter(
            policy, adapter_id="pi05"
        ),
        fallback_mode="strict",
    )
    bindings = build_profile_admitted_tool_bindings(
        config,
        runtime=runtime,
        observation_source=lambda _ctx: {"observation/state": [0.0] * 8},
        action_executor=lambda actions, ctx: ActionExecutionReport(
            status="succeeded",
            ended_timestep=ctx.timestep + len(actions),
            observed_outcome="chunk executed",
        ),
        observe=lambda _ctx: {"frame_id": 1},
        retrieve_memory=lambda query, limit, _ctx: {
            "query": query,
            "limit": limit,
            "records": [],
        },
        run_skill=lambda _skill, _args, _ctx: None,
        verify=lambda _expected, _ctx: None,
        safe_hold=lambda reason, _ctx: {"holding": True, "reason": reason},
        inference_metadata_source=lambda _ctx: {
            "noise": [[0.0] * 32 for _ in range(10)]
        },
    )

    report = bindings.vla_act("close the drawer", context())

    assert report.status is PrimitiveStatus.SUCCEEDED
    assert report.metadata["profile_id"] == config.vla.deployment_profile_id
    assert bindings.observe(context())["frame_id"] == 1
    assert len(policy.last_payload["runtime_noise"]) == 10
