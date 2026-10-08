"""Tests for the canonical CARVE framework configuration."""

from __future__ import annotations

import json

import pytest

from agentic_vla.configuration import (
    CarveRunConfig,
    PlannerExecutionMode,
)
from agentic_vla.runtime import PlannerProvider, PrimitiveBoundaryPolicy


def test_local_qwen_config_builds_provider_and_manifest() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")

    assert config.planner.mode is PlannerExecutionMode.EMBEDDED_VLM
    assert config.planner.provider is PlannerProvider.OPENAI_COMPATIBLE
    assert config.harness.primitive_boundary_policy is PrimitiveBoundaryPolicy.SELECTIVE
    provider = config.planner.provider_config()
    assert provider is not None
    assert provider.endpoint.endswith("/v1/chat/completions")

    manifest = config.to_run_manifest()
    assert manifest.planner_id == "qwen-vl-local"
    assert manifest.deployment_profile_id == config.vla.deployment_profile_id
    assert manifest.metadata["config_fingerprint"] == config.fingerprint
    assert len(config.fingerprint) == 16


def test_codex_is_external_tool_agent_not_fake_vlm_endpoint() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")

    assert config.planner.mode is PlannerExecutionMode.EXTERNAL_CODING_AGENT
    assert config.planner.provider_config() is None
    assert config.planner.endpoint == ""
    assert config.to_run_manifest().metadata["planner_mode"] == (
        "external_coding_agent"
    )


def test_round_trip_is_stable_and_secret_free(tmp_path) -> None:
    original = CarveRunConfig.load("configs/carve_pi05_qwen_local.json")
    saved = original.save(tmp_path / "run.json")
    restored = CarveRunConfig.load(saved)

    assert restored == original
    assert restored.fingerprint == original.fingerprint
    payload = saved.read_text(encoding="utf-8")
    assert "api_key\"" not in payload


def test_rejects_inline_secrets_unknown_fields_and_unsafe_tool() -> None:
    payload = json.loads(
        open("configs/carve_pi05_qwen_local.json", encoding="utf-8").read()
    )
    payload["planner"]["api_key"] = "do-not-store-this"
    with pytest.raises(ValueError, match="environment variable"):
        CarveRunConfig.from_dict(payload)

    payload = json.loads(
        open("configs/carve_pi05_qwen_local.json", encoding="utf-8").read()
    )
    payload["benchmark"]["simulator_state"] = [0.0]
    with pytest.raises(ValueError, match="unknown fields"):
        CarveRunConfig.from_dict(payload)

    payload = json.loads(
        open("configs/carve_pi05_qwen_local.json", encoding="utf-8").read()
    )
    payload["harness"]["allowed_tools"].append("raw_joint_action")
    with pytest.raises(ValueError, match="unknown tools"):
        CarveRunConfig.from_dict(payload)


def test_external_agent_cannot_configure_embedded_endpoint() -> None:
    payload = json.loads(
        open("configs/carve_pi05_codex_tool_agent.json", encoding="utf-8").read()
    )
    payload["planner"]["endpoint"] = "http://127.0.0.1:9999"
    with pytest.raises(ValueError, match="tool boundary"):
        CarveRunConfig.from_dict(payload)
