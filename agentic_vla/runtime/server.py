"""Server-side wrappers for capability-aware per-request inference controls."""

from __future__ import annotations

import threading
from collections.abc import Mapping, MutableMapping
from typing import Any


class RuntimeControllablePolicy:
    """Expose bounded per-call flow steps around an OpenPI-like policy."""

    def __init__(
        self,
        policy: Any,
        *,
        minimum_inference_steps: int = 1,
        maximum_inference_steps: int = 16,
        deployment_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not hasattr(policy, "infer"):
            raise TypeError("policy must expose infer(request)")
        sample_kwargs = getattr(policy, "_sample_kwargs", None)
        if not isinstance(sample_kwargs, MutableMapping):
            raise TypeError("policy must expose mutable _sample_kwargs")
        if minimum_inference_steps <= 0 or maximum_inference_steps < minimum_inference_steps:
            raise ValueError("invalid inference-step bounds")
        self._policy = policy
        self._sample_kwargs = sample_kwargs
        self._minimum_steps = int(minimum_inference_steps)
        self._maximum_steps = int(maximum_inference_steps)
        self._lock = threading.RLock()
        metadata = dict(getattr(policy, "metadata", {}) or {})
        metadata["carve_capabilities"] = {
            "configurable_inference_steps": True,
            "deterministic_noise": True,
            "minimum_inference_steps": self._minimum_steps,
            "maximum_inference_steps": self._maximum_steps,
        }
        if deployment_metadata is not None:
            metadata["carve_deployment_profile"] = dict(deployment_metadata)
        self._metadata = metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("request must be a mapping")
        payload = dict(request)
        raw_controls = payload.pop("runtime_controls", None)
        runtime_noise = payload.pop("runtime_noise", None)
        requested_steps = None
        if isinstance(raw_controls, Mapping) and raw_controls.get("inference_steps") is not None:
            requested_steps = int(raw_controls["inference_steps"])
            if not self._minimum_steps <= requested_steps <= self._maximum_steps:
                raise ValueError(
                    f"inference_steps must be in [{self._minimum_steps}, {self._maximum_steps}]"
                )

        with self._lock:
            previous_steps = self._sample_kwargs.get("num_steps")
            had_steps = "num_steps" in self._sample_kwargs
            if requested_steps is not None:
                self._sample_kwargs["num_steps"] = requested_steps
            try:
                if runtime_noise is None:
                    output = self._policy.infer(payload)
                else:
                    output = self._policy.infer(payload, noise=runtime_noise)
            finally:
                if requested_steps is not None:
                    if had_steps:
                        self._sample_kwargs["num_steps"] = previous_steps
                    else:
                        self._sample_kwargs.pop("num_steps", None)
        if not isinstance(output, Mapping):
            raise TypeError("policy output must be a mapping")
        response = dict(output)
        response["runtime_controls"] = {
            "inference_steps": requested_steps,
            "applied": requested_steps is not None,
            "deterministic_noise": runtime_noise is not None,
        }
        return response
