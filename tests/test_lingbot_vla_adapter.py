from __future__ import annotations

import numpy as np

from agentic_vla.runtime import ActionSpec, InferenceControls, InferenceRequest
from agentic_vla.runtime.adapters import (
    LingBotVlaAdapter,
    normalize_lingbot_action_chunk,
)


class FakeXPolicyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def call(self, func_name: str, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            return [
                {
                    "left_arm_joint_state": np.arange(6),
                    "left_ee_joint_state": np.array([6]),
                    "right_arm_joint_state": np.arange(7, 13),
                    "right_ee_joint_state": np.array([13]),
                },
                {
                    "left_arm_joint_state": np.arange(14, 20),
                    "left_ee_joint_state": np.array([20]),
                    "right_arm_joint_state": np.arange(21, 27),
                    "right_ee_joint_state": np.array([27]),
                },
            ]
        return None


class FakeNativeLingBotClient:
    def __init__(self) -> None:
        self.reset_names: list[str] = []
        self.payloads: list[dict] = []

    def reset(self, robot_name: str) -> None:
        self.reset_names.append(robot_name)

    def infer(self, payload: dict):
        self.payloads.append(payload)
        return {
            "action.arm.position": np.zeros((3, 12), dtype=np.float32),
            "action.effector.position": np.zeros((3, 2), dtype=np.float32),
        }


def _action_spec() -> ActionSpec:
    return ActionSpec(
        action_dim=14,
        representation="dual_arm_joint_position",
        coordinate_frame="robot_joint",
        gripper_convention="robotwin_joint_gripper",
        control_frequency_hz=30.0,
        normalization_id="lingbot-vla/robotwin_50",
    )


def test_lingbot_adapter_uses_xpolicylab_lifecycle_and_action_order() -> None:
    client = FakeXPolicyClient()
    adapter = LingBotVlaAdapter(client, action_spec=_action_spec())
    request = InferenceRequest(
        observation={"vision": {"cam_head": {"color": np.zeros((8, 8, 3))}}},
        instruction="open the microwave",
        controls=InferenceControls(max_actions=2),
        episode_id="episode-1",
    )

    adapter.reset(request.episode_id)
    chunk = adapter.infer(request)

    assert [name for name, _ in client.calls] == ["reset", "update_obs", "get_action"]
    update_payload = client.calls[1][1]
    assert update_payload["instruction"] == "open the microwave"
    assert chunk.actions.shape == (2, 14)
    np.testing.assert_array_equal(chunk.actions[0], np.arange(14))
    np.testing.assert_array_equal(chunk.actions[1], np.arange(14, 28))
    assert chunk.metadata["backend"] == "xpolicylab_ws"
    assert adapter.capabilities.action_chunking is True
    assert adapter.capabilities.max_action_horizon == 50
    assert adapter.capabilities.configurable_inference_steps is False


def test_normalize_lingbot_split_feature_output_interleaves_grippers() -> None:
    output = {
        "action.arm.position": np.array(
            [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], dtype=np.float32
        ),
        "action.effector.position": np.array([[6, 13]], dtype=np.float32),
    }

    actions = normalize_lingbot_action_chunk(output)

    np.testing.assert_array_equal(actions, np.arange(14, dtype=np.float32)[None, :])


def test_lingbot_adapter_uses_native_server_reset_contract() -> None:
    client = FakeNativeLingBotClient()
    adapter = LingBotVlaAdapter(client, action_spec=_action_spec())
    adapter.reset("episode-2")
    chunk = adapter.infer(
        InferenceRequest(
            observation={},
            instruction="open the microwave",
            metadata={
                "raw_payload": {
                    "observation.state": np.zeros(14, dtype=np.float32),
                    "task": "open the microwave",
                }
            },
        )
    )

    assert client.reset_names == ["robotwin"]
    assert client.payloads[0]["task"] == "open the microwave"
    assert chunk.actions.shape == (3, 14)
    assert chunk.metadata["backend"] == "lingbot_ws"


def test_lingbot_adapter_attaches_auditable_deployment_profile() -> None:
    client = FakeNativeLingBotClient()
    receipt = {
        "profile": {"profile_id": "lingbot-compile-bf16-d10-c50"},
        "backend_id": "torch_compile",
    }
    adapter = LingBotVlaAdapter(
        client,
        action_spec=_action_spec(),
        deployment_metadata=receipt,
    )

    chunk = adapter.infer(
        InferenceRequest(
            observation={},
            instruction="open the microwave",
            metadata={
                "raw_payload": {
                    "observation.state": np.zeros(14, dtype=np.float32),
                    "task": "open the microwave",
                }
            },
        )
    )

    assert chunk.metadata["optimization_profile"] == receipt
    assert chunk.metadata["optimization_profile"] is not receipt


def test_lingbot_adapter_rejects_malformed_dual_arm_mapping() -> None:
    try:
        normalize_lingbot_action_chunk({"left_arm_joint_state": np.zeros(6)})
    except ValueError as exc:
        assert "four dual-arm" in str(exc)
    else:
        raise AssertionError("malformed action mapping should be rejected")
