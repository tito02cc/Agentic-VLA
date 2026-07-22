"""pi0.5 adapter for local OpenPI policies and legacy policy clients.

This module intentionally imports no OpenPI package. A concrete policy/client is
injected so CARVE's core remains backend independent.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from ..adapter import PolicyAdapter
from ..contracts import (
    ActionChunk,
    ActionSpec,
    InferenceControls,
    InferenceRequest,
    ModelCapabilities,
    PolicyFamily,
)


RequestEncoder = Callable[[InferenceRequest], dict[str, Any]]
ResponseDecoder = Callable[[Mapping[str, Any], float], ActionChunk]


def encode_pi05_request(request: InferenceRequest) -> dict[str, Any]:
    raw_payload = request.metadata.get("raw_payload")
    if isinstance(raw_payload, Mapping):
        return dict(raw_payload)

    payload = dict(request.observation)
    payload["prompt"] = request.instruction
    if request.episode_id is not None:
        payload["episode_id"] = request.episode_id
    if request.timestep is not None:
        payload["timestep"] = request.timestep
    if request.agentic is not None:
        payload["agentic"] = dict(request.agentic)
    return payload


def decode_pi05_response(output: Mapping[str, Any], wall_ms: float) -> ActionChunk:
    if "actions" not in output:
        raise ValueError("pi0.5 response does not contain 'actions'")
    timing = output.get("policy_timing")
    model_ms = wall_ms
    if isinstance(timing, Mapping):
        try:
            model_ms = float(timing.get("infer_ms", wall_ms))
        except (TypeError, ValueError):
            model_ms = wall_ms
    return ActionChunk(
        actions=output["actions"],
        model_latency_ms=max(0.0, model_ms),
        raw_output=dict(output),
        uncertainty=output.get("uncertainty"),
        predictive_context=output.get("predictive_context"),
    )


class Pi05Adapter(PolicyAdapter):
    """Wrap a local OpenPI policy or a remote infer(payload) client."""

    def __init__(
        self,
        policy: Any,
        *,
        adapter_id: str = "pi05",
        precision: str = "bf16",
        max_action_horizon: int | None = None,
        action_spec: ActionSpec | None = None,
        request_encoder: RequestEncoder = encode_pi05_request,
        response_decoder: ResponseDecoder = decode_pi05_response,
        lock: threading.RLock | None = None,
    ) -> None:
        if not hasattr(policy, "infer"):
            raise TypeError("policy must expose infer(payload)")
        self._policy = policy
        self._adapter_id = adapter_id
        self._precision = precision.lower()
        self._request_encoder = request_encoder
        self._response_decoder = response_decoder
        self._action_spec = action_spec
        self._lock = lock or threading.RLock()
        self._sample_kwargs = getattr(policy, "_sample_kwargs", None)
        local_dynamic_steps = isinstance(self._sample_kwargs, MutableMapping)
        server_metadata = getattr(policy, "_server_metadata", None)
        remote_capabilities = (
            server_metadata.get("carve_capabilities", {})
            if isinstance(server_metadata, Mapping)
            else {}
        )
        remote_deployment = (
            server_metadata.get("carve_deployment_profile", {})
            if isinstance(server_metadata, Mapping)
            else {}
        )
        self._deployment_metadata = (
            dict(remote_deployment) if isinstance(remote_deployment, Mapping) else {}
        )
        self._remote_dynamic_steps = bool(
            isinstance(remote_capabilities, Mapping)
            and remote_capabilities.get("configurable_inference_steps", False)
        )
        dynamic_steps = local_dynamic_steps or self._remote_dynamic_steps
        self._capabilities = ModelCapabilities(
            policy_family=PolicyFamily.VLA,
            action_chunking=True,
            configurable_inference_steps=dynamic_steps,
            autoregressive_decoding=False,
            kv_cache=False,
            predictive_context=False,
            uncertainty=False,
            async_inference=False,
            supported_precisions=frozenset({self._precision}),
            max_action_horizon=max_action_horizon,
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

    def reset(self, episode_id: str | int | None = None) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, request: InferenceRequest) -> ActionChunk:
        payload = self._request_encoder(request)
        call_kwargs: dict[str, Any] = {}
        if "noise" in request.metadata:
            call_kwargs["noise"] = request.metadata["noise"]

        with self._lock:
            previous_steps: Any = None
            had_steps = False
            if (
                request.controls.inference_steps is not None
                and isinstance(self._sample_kwargs, MutableMapping)
            ):
                had_steps = "num_steps" in self._sample_kwargs
                previous_steps = self._sample_kwargs.get("num_steps")
                self._sample_kwargs["num_steps"] = int(request.controls.inference_steps)
            elif request.controls.inference_steps is not None and self._remote_dynamic_steps:
                payload["runtime_controls"] = {
                    "inference_steps": int(request.controls.inference_steps)
                }
            elif request.controls.inference_steps is not None:
                raise RuntimeError("inference_steps reached a non-configurable pi0.5 client")
            try:
                started_s = time.perf_counter()
                output = self._policy.infer(payload, **call_kwargs)
                wall_ms = (time.perf_counter() - started_s) * 1000.0
            finally:
                if (
                    request.controls.inference_steps is not None
                    and isinstance(self._sample_kwargs, MutableMapping)
                ):
                    if had_steps:
                        self._sample_kwargs["num_steps"] = previous_steps
                    else:
                        self._sample_kwargs.pop("num_steps", None)

        if not isinstance(output, Mapping):
            raise TypeError("pi0.5 policy output must be a mapping")
        chunk = self._response_decoder(output, wall_ms)
        if self.action_spec is not None:
            self.action_spec.validate(chunk.actions)
        chunk.metadata.update(
            {
                "backend": "pi05",
                "transport": (
                    "local"
                    if isinstance(self._sample_kwargs, MutableMapping)
                    else "controlled_client" if self._remote_dynamic_steps else "client"
                ),
                "action_spec": None if self.action_spec is None else self.action_spec.to_dict(),
            }
        )
        if self._deployment_metadata:
            chunk.metadata["optimization_profile"] = dict(self._deployment_metadata)
        return chunk


class LegacyPolicyClientBridge:
    """Expose CARVE through the existing client.infer(payload) protocol."""

    def __init__(self, runtime: Any) -> None:
        if not hasattr(runtime, "infer"):
            raise TypeError("runtime must expose infer(request)")
        self.runtime = runtime

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("legacy payload must be a mapping")
        raw_controls = payload.get("runtime_controls")
        if isinstance(raw_controls, InferenceControls):
            controls = raw_controls
        elif isinstance(raw_controls, Mapping):
            allowed = {
                key: value
                for key, value in raw_controls.items()
                if key in {"inference_steps", "max_actions", "precision", "reuse_context", "deadline_ms"}
            }
            controls = InferenceControls(**allowed)
        else:
            controls = InferenceControls()

        trace_context = payload.get("runtime_trace_context")
        clean_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"runtime_controls", "runtime_trace_context"}
        }
        request = InferenceRequest(
            observation={
                key: value
                for key, value in clean_payload.items()
                if key not in {"prompt", "episode_id", "timestep", "agentic"}
            },
            instruction=str(clean_payload.get("prompt", "")),
            controls=controls,
            episode_id=clean_payload.get("episode_id"),
            timestep=clean_payload.get("timestep"),
            agentic=(
                clean_payload.get("agentic")
                if isinstance(clean_payload.get("agentic"), Mapping)
                else None
            ),
            metadata={
                "raw_payload": clean_payload,
                "trace_context": trace_context if isinstance(trace_context, Mapping) else {},
            },
        )
        return self.runtime.infer(request).as_legacy_output()
