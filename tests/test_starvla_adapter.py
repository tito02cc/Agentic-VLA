from __future__ import annotations

import numpy as np

from agentic_vla.optimization import (
    OptimizationProfile,
    StarVlaModelPlugin,
    create_default_registry,
)
from agentic_vla.runtime import ActionSpec, CarveRuntime, InferenceControls, InferenceRequest
from agentic_vla.runtime.adapters import (
    StarVlaAdapter,
    hash_starvla_request_input,
    normalize_starvla_action_chunk,
    summarize_starvla_request_input,
)


class FakeStarVlaClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def predict_action(self, payload: dict):
        self.payloads.append(payload)
        actions = np.arange(50 * 14, dtype=np.float32).reshape(1, 50, 14)
        return {"ok": True, "data": {"actions": actions}}


def _action_spec() -> ActionSpec:
    return ActionSpec(
        action_dim=14,
        representation="dual_arm_joint_position",
        coordinate_frame="robot_joint",
        gripper_convention="xpolicy_joint_gripper",
        control_frequency_hz=25.0,
        normalization_id="starvla/arx_x5",
    )


def test_starvla_adapter_preserves_official_payload_and_action_order() -> None:
    client = FakeStarVlaClient()
    adapter = StarVlaAdapter(client, action_spec=_action_spec())
    request = InferenceRequest(
        observation={
            "lang": "build a tower",
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)] * 3,
            "state": np.zeros((1, 14), dtype=np.float32),
        },
        instruction="build a tower",
        controls=InferenceControls(inference_steps=6, max_actions=16),
    )

    chunk = CarveRuntime(adapter).infer(request)

    payload = client.payloads[0]
    assert payload["examples"][0]["lang"] == "build a tower"
    assert payload["do_sample"] is False
    assert payload["use_ddim"] is True
    assert payload["num_ddim_steps"] == 6
    assert payload["unnorm_key"] == "arx_x5"
    assert chunk.actions.shape == (16, 14)
    np.testing.assert_array_equal(chunk.actions[0], np.arange(14, dtype=np.float32))
    assert adapter.capabilities.configurable_inference_steps is True


def test_starvla_pi_v3_uses_explicit_flow_matching_step_parameter() -> None:
    client = FakeStarVlaClient()
    adapter = StarVlaAdapter(
        client,
        default_num_ddim_steps=4,
        inference_steps_key="num_inference_steps",
    )
    request = InferenceRequest(
        observation={"lang": "build a tower", "image": []},
        instruction="build a tower",
        controls=InferenceControls(inference_steps=2),
    )

    chunk = adapter.infer(request)

    assert client.payloads[0]["num_inference_steps"] == 2
    assert chunk.metadata["inference_steps"] == 2
    assert chunk.metadata["inference_steps_parameter"] == "num_inference_steps"


def test_starvla_raw_payload_is_not_reencoded() -> None:
    client = FakeStarVlaClient()
    adapter = StarVlaAdapter(client, default_num_ddim_steps=10)
    request = InferenceRequest(
        observation={},
        instruction="ignored",
        controls=InferenceControls(),
        metadata={
            "raw_payload": {
                "examples": [{"lang": "official", "image": []}],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
                "unnorm_key": "arx_x5",
            }
        },
    )

    adapter.infer(request)

    assert client.payloads[0]["examples"][0]["lang"] == "official"
    assert client.payloads[0]["num_ddim_steps"] == 10


def test_starvla_adapter_preserves_request_level_action_seed() -> None:
    client = FakeStarVlaClient()
    adapter = StarVlaAdapter(client, default_num_ddim_steps=10)
    request = InferenceRequest(
        observation={},
        instruction="ignored",
        metadata={
            "raw_payload": {
                "examples": [{"lang": "official", "image": []}],
                "action_seed": 41,
            }
        },
    )

    chunk = adapter.infer(request)

    assert client.payloads[0]["action_seed"] == 41
    assert chunk.metadata["action_seed"] == 41


def test_starvla_input_hash_tracks_instruction_state_and_pixels() -> None:
    payload = {
        "examples": [
            {
                "lang": "stack bowls",
                "image": [np.zeros((2, 3, 3), dtype=np.uint8)],
                "state": np.zeros((1, 14), dtype=np.float32),
            }
        ]
    }
    same = {
        "examples": [
            {
                "lang": "stack bowls",
                "image": [np.zeros((2, 3, 3), dtype=np.uint8)],
                "state": np.zeros((1, 14), dtype=np.float32),
            }
        ]
    }
    changed = {
        "examples": [
            {
                "lang": "stack bowls",
                "image": [np.ones((2, 3, 3), dtype=np.uint8)],
                "state": np.zeros((1, 14), dtype=np.float32),
            }
        ]
    }

    assert hash_starvla_request_input(payload) == hash_starvla_request_input(same)
    assert hash_starvla_request_input(payload) != hash_starvla_request_input(changed)


def test_starvla_input_summary_separates_modalities() -> None:
    payload = {
        "examples": [
            {
                "lang": "stack bowls",
                "image": [np.zeros((2, 3, 3), dtype=np.uint8)],
                "state": np.zeros((1, 14), dtype=np.float32),
            }
        ]
    }

    summary = summarize_starvla_request_input(payload)["examples"][0]

    assert len(summary["language_sha256"]) == 64
    assert len(summary["image_sha256"]) == 1
    assert len(summary["image_sha256"][0]) == 64
    assert len(summary["state_sha256"]) == 64


def test_normalize_starvla_rejects_multi_environment_response() -> None:
    output = {"ok": True, "data": {"actions": np.zeros((2, 50, 14))}}
    try:
        normalize_starvla_action_chunk(output)
    except ValueError as exc:
        assert "one environment" in str(exc)
    else:
        raise AssertionError("multi-environment response should be rejected")


def test_starvla_server_error_is_propagated() -> None:
    try:
        normalize_starvla_action_chunk({"ok": False, "error": "bad checkpoint"})
    except RuntimeError as exc:
        assert "bad checkpoint" in str(exc)
    else:
        raise AssertionError("server errors must not be converted into actions")


def test_default_registry_resolves_starvla_plugin_and_validates_horizon() -> None:
    adapter = StarVlaAdapter(FakeStarVlaClient(), max_action_horizon=50)
    model = create_default_registry().resolve_model(adapter)

    assert isinstance(model, StarVlaModelPlugin)
    model.validate_profile(
        adapter,
        OptimizationProfile(
            profile_id="starvla-ddim6-h16",
            backend="eager",
            deployment_precision="bf16",
            inference_steps=6,
            action_horizon=16,
            options={"server_action_chunk_size": 50},
        ),
    )
    try:
        model.validate_profile(
            adapter,
            OptimizationProfile(profile_id="too-long", action_horizon=51),
        )
    except ValueError as exc:
        assert "server maximum" in str(exc)
    else:
        raise AssertionError("oversized action horizons must be rejected")
