"""LingBot-VLA adapter for the RoboTwin/XPolicyLab deployment contract."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..adapter import PolicyAdapter
from ..contracts import (
    ActionChunk,
    ActionSpec,
    InferenceRequest,
    ModelCapabilities,
    PolicyFamily,
)


_ROBOTWIN_ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)


def encode_lingbot_robotwin_request(request: InferenceRequest) -> dict[str, Any]:
    """Build the observation accepted by the XPolicyLab LingBot-VLA wrapper."""

    raw_payload = request.metadata.get("raw_payload")
    if isinstance(raw_payload, Mapping):
        return dict(raw_payload)
    payload = dict(request.observation)
    payload.setdefault("instruction", request.instruction)
    payload.setdefault("instructions", [request.instruction])
    return payload


def _flatten_robotwin_action(action: Mapping[str, Any]) -> np.ndarray:
    if all(key in action for key in _ROBOTWIN_ACTION_KEYS):
        return np.concatenate(
            [np.asarray(action[key], dtype=np.float32).reshape(-1) for key in _ROBOTWIN_ACTION_KEYS]
        )
    raise ValueError(
        "LingBot-VLA RoboTwin mapping must contain the four dual-arm joint-action fields"
    )


def normalize_lingbot_action_chunk(output: Any) -> np.ndarray:
    """Normalize upstream and XPolicyLab responses to a dense ``[T, D]`` chunk."""

    if isinstance(output, Mapping):
        if "actions" in output:
            return normalize_lingbot_action_chunk(output["actions"])
        if "action" in output:
            return normalize_lingbot_action_chunk(output["action"])
        if "action.arm.position" in output and "action.effector.position" in output:
            arm = np.asarray(output["action.arm.position"], dtype=np.float32)
            effector = np.asarray(output["action.effector.position"], dtype=np.float32)
            if arm.ndim == 1:
                arm = arm[None, :]
            if effector.ndim == 1:
                effector = effector[None, :]
            if arm.ndim != 2 or effector.ndim != 2 or arm.shape[0] != effector.shape[0]:
                raise ValueError("LingBot-VLA split action features must share a [T, D] shape")
            if arm.shape[1] != 12 or effector.shape[1] != 2:
                raise ValueError("RoboTwin split actions require 12 arm and 2 effector values")
            return np.concatenate(
                (
                    arm[:, :6],
                    effector[:, :1],
                    arm[:, 6:],
                    effector[:, 1:],
                ),
                axis=1,
            )
        return _flatten_robotwin_action(output)[None, :]

    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        if output and isinstance(output[0], Mapping):
            return np.stack([_flatten_robotwin_action(item) for item in output]).astype(
                np.float32
            )

    actions = np.asarray(output, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"LingBot-VLA actions must have shape [T, D], got {actions.shape}")
    return actions


class LingBotVlaAdapter(PolicyAdapter):
    """Expose a frozen LingBot-VLA action-chunk service through CARVE."""

    def __init__(
        self,
        client: Any,
        *,
        adapter_id: str = "lingbot-vla-4b-robotwin",
        precision: str = "bf16",
        max_action_horizon: int = 50,
        action_spec: ActionSpec | None = None,
        reset_robot_name: str | None = "robotwin",
        deployment_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not hasattr(client, "call") and not hasattr(client, "infer"):
            raise TypeError("client must expose call(...) or infer(payload)")
        if max_action_horizon <= 0:
            raise ValueError("max_action_horizon must be positive")
        self._client = client
        self._adapter_id = str(adapter_id)
        self._precision = str(precision).lower()
        self._action_spec = action_spec
        self._reset_robot_name = reset_robot_name
        self._deployment_metadata = dict(deployment_metadata or {})
        self._capabilities = ModelCapabilities(
            policy_family=PolicyFamily.VLA,
            action_chunking=True,
            configurable_inference_steps=False,
            autoregressive_decoding=False,
            kv_cache=False,
            predictive_context=False,
            uncertainty=False,
            async_inference=False,
            supported_precisions=frozenset({self._precision}),
            max_action_horizon=int(max_action_horizon),
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def action_spec(self) -> ActionSpec | None:
        return self._action_spec

    @property
    def client(self) -> Any:
        return self._client

    def reset(self, episode_id: str | int | None = None) -> None:
        del episode_id
        if hasattr(self._client, "call"):
            self._client.call(func_name="reset")
            return
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            if self._reset_robot_name is None:
                reset()
            else:
                reset(self._reset_robot_name)

    def infer(self, request: InferenceRequest) -> ActionChunk:
        payload = encode_lingbot_robotwin_request(request)
        started_s = time.perf_counter()
        if hasattr(self._client, "call"):
            self._client.call(func_name="update_obs", obs=payload)
            output = self._client.call(func_name="get_action")
            backend = "xpolicylab_ws"
        else:
            output = self._client.infer(payload)
            backend = "lingbot_ws"
        wall_ms = (time.perf_counter() - started_s) * 1000.0
        actions = normalize_lingbot_action_chunk(output)
        if self.action_spec is not None:
            self.action_spec.validate(actions)
        metadata = {
            "backend": backend,
            "decoding": "flow_matching_action_chunk",
            "precision": self._precision,
            "action_spec": (
                None if self.action_spec is None else self.action_spec.to_dict()
            ),
        }
        if self._deployment_metadata:
            metadata["optimization_profile"] = dict(self._deployment_metadata)
        return ActionChunk(
            actions=actions,
            model_latency_ms=max(0.0, wall_ms),
            raw_output={"actions": actions},
            metadata=metadata,
        )
