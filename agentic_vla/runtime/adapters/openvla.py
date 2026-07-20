"""OpenVLA adapter without importing the OpenVLA or Transformers packages."""

from __future__ import annotations

import time
from collections.abc import Mapping
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


def encode_openvla_request(request: InferenceRequest) -> dict[str, Any]:
    raw_payload = request.metadata.get("raw_payload")
    if isinstance(raw_payload, Mapping):
        return dict(raw_payload)
    image = request.observation.get("image")
    if image is None:
        image = request.observation.get("agentview_image")
    if image is None:
        raise ValueError("OpenVLA request requires image or agentview_image")
    payload = {
        "image": image,
        "instruction": request.instruction,
        "do_sample": False,
    }
    if unnorm_key := request.metadata.get("unnorm_key"):
        payload["unnorm_key"] = str(unnorm_key)
    return payload


class OpenVlaAdapter(PolicyAdapter):
    """Wrap a deployment object exposing ``infer(payload)`` for OpenVLA."""

    def __init__(
        self,
        policy: Any,
        *,
        adapter_id: str = "openvla-7b",
        precision: str = "bf16",
        action_spec: ActionSpec | None = None,
    ) -> None:
        if not hasattr(policy, "infer"):
            raise TypeError("policy must expose infer(payload)")
        self._policy = policy
        self._adapter_id = str(adapter_id)
        self._precision = str(precision).lower()
        self._action_spec = action_spec
        self._capabilities = ModelCapabilities(
            policy_family=PolicyFamily.VLA,
            action_chunking=False,
            configurable_inference_steps=False,
            autoregressive_decoding=True,
            kv_cache=False,
            predictive_context=False,
            uncertainty=False,
            async_inference=False,
            supported_precisions=frozenset({self._precision}),
            max_action_horizon=1,
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
    def policy(self) -> Any:
        return self._policy

    def infer(self, request: InferenceRequest) -> ActionChunk:
        payload = encode_openvla_request(request)
        started_s = time.perf_counter()
        output = self._policy.infer(payload)
        wall_ms = (time.perf_counter() - started_s) * 1000.0
        if isinstance(output, Mapping):
            if "actions" not in output:
                raise ValueError("OpenVLA response does not contain 'actions'")
            raw_output = dict(output)
            actions = output["actions"]
            timing = output.get("policy_timing")
            model_ms = (
                float(timing.get("infer_ms", wall_ms))
                if isinstance(timing, Mapping)
                else wall_ms
            )
        else:
            raw_output = {"actions": output}
            actions = output
            model_ms = wall_ms
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        if action_array.ndim != 2 or action_array.shape[0] != 1:
            raise ValueError("OpenVLA must return exactly one action vector per call")
        if self.action_spec is not None:
            self.action_spec.validate(action_array)
        return ActionChunk(
            actions=action_array,
            model_latency_ms=max(0.0, model_ms),
            raw_output=raw_output,
            metadata={
                "backend": "openvla",
                "decoding": "autoregressive_action_tokens",
                "action_spec": (
                    None if self.action_spec is None else self.action_spec.to_dict()
                ),
            },
        )
