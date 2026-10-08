"""StarVLA adapter for the RoboDojo/XPolicyLab deployment contract."""

from __future__ import annotations

import hashlib
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


def encode_starvla_request(
    request: InferenceRequest,
    *,
    default_num_ddim_steps: int = 10,
    inference_steps_key: str = "num_ddim_steps",
    default_use_ddim: bool = True,
    default_unnorm_key: str | None = "arx_x5",
) -> dict[str, Any]:
    """Build the request accepted by StarVLA's websocket policy server."""

    raw_payload = request.metadata.get("raw_payload")
    if isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    else:
        observation = dict(request.observation)
        if "examples" in observation:
            payload = dict(observation)
        else:
            example = dict(observation)
            example.setdefault("lang", request.instruction)
            payload = {"examples": [example]}

    payload.setdefault("do_sample", False)
    payload.setdefault("use_ddim", default_use_ddim)
    requested_steps = request.controls.inference_steps
    if inference_steps_key not in {"num_ddim_steps", "num_inference_steps"}:
        raise ValueError(f"unsupported StarVLA inference-steps key: {inference_steps_key}")
    payload[inference_steps_key] = int(
        default_num_ddim_steps if requested_steps is None else requested_steps
    )
    if default_unnorm_key is not None:
        payload.setdefault("unnorm_key", default_unnorm_key)
    return payload


def normalize_starvla_action_chunk(output: Any) -> np.ndarray:
    """Normalize a StarVLA response to one dense ``[T, D]`` action chunk."""

    if isinstance(output, Mapping):
        if output.get("ok") is False:
            raise RuntimeError(
                f"StarVLA inference failed: {output.get('error', dict(output))}"
            )
        if "data" in output:
            return normalize_starvla_action_chunk(output["data"])
        if "actions" in output:
            actions = np.asarray(output["actions"], dtype=np.float32)
            if actions.ndim == 3:
                if actions.shape[0] != 1:
                    raise ValueError(
                        "StarVLA adapter expects one environment per inference request"
                    )
                actions = actions[0]
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2:
                raise ValueError(
                    f"StarVLA actions must have shape [T, D], got {actions.shape}"
                )
            return actions

    actions = np.asarray(output, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"StarVLA actions must have shape [T, D], got {actions.shape}")
    return actions


def hash_starvla_request_input(payload: Mapping[str, Any]) -> str:
    """Hash the policy-relevant observation without serializing image payloads."""

    digest = hashlib.sha256()
    examples = payload.get("examples", ())
    for example in examples if isinstance(examples, (list, tuple)) else ():
        if not isinstance(example, Mapping):
            continue
        digest.update(str(example.get("lang", "")).encode("utf-8"))
        for key in ("image", "state"):
            value = example.get(key)
            values = value if key == "image" and isinstance(value, (list, tuple)) else (value,)
            for item in values:
                if item is None:
                    digest.update(b"<none>")
                    continue
                array = np.ascontiguousarray(np.asarray(item))
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
    return digest.hexdigest()


def summarize_starvla_request_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return per-modality hashes for reproducibility diagnostics."""

    examples = payload.get("examples", ())
    summaries: list[dict[str, Any]] = []
    for example in examples if isinstance(examples, (list, tuple)) else ():
        if not isinstance(example, Mapping):
            continue
        language = str(example.get("lang", ""))
        images = example.get("image")
        image_values = images if isinstance(images, (list, tuple)) else (images,)
        image_hashes: list[str | None] = []
        for image in image_values:
            if image is None:
                image_hashes.append(None)
                continue
            array = np.ascontiguousarray(np.asarray(image))
            image_hashes.append(hashlib.sha256(array.tobytes()).hexdigest())

        state = example.get("state")
        state_hash = None
        if state is not None:
            state_array = np.ascontiguousarray(np.asarray(state))
            state_hash = hashlib.sha256(state_array.tobytes()).hexdigest()

        summaries.append(
            {
                "language_sha256": hashlib.sha256(language.encode("utf-8")).hexdigest(),
                "image_sha256": image_hashes,
                "state_sha256": state_hash,
            }
        )
    return {"examples": summaries}


class StarVlaAdapter(PolicyAdapter):
    """Expose a frozen StarVLA action-chunk server through CARVE."""

    def __init__(
        self,
        client: Any,
        *,
        adapter_id: str = "starvla-pi-v3-robodojo",
        precision: str = "bf16",
        max_action_horizon: int = 50,
        default_num_ddim_steps: int = 10,
        inference_steps_key: str = "num_ddim_steps",
        use_ddim: bool = True,
        unnorm_key: str | None = "arx_x5",
        action_spec: ActionSpec | None = None,
        deployment_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not hasattr(client, "predict_action"):
            raise TypeError("client must expose predict_action(payload)")
        if max_action_horizon <= 0:
            raise ValueError("max_action_horizon must be positive")
        if default_num_ddim_steps <= 0:
            raise ValueError("default_num_ddim_steps must be positive")
        if inference_steps_key not in {"num_ddim_steps", "num_inference_steps"}:
            raise ValueError("unsupported StarVLA inference-steps key")
        self._client = client
        self._adapter_id = str(adapter_id)
        self._precision = str(precision).lower()
        self._default_num_ddim_steps = int(default_num_ddim_steps)
        self._inference_steps_key = inference_steps_key
        self._use_ddim = bool(use_ddim)
        self._unnorm_key = unnorm_key
        self._action_spec = action_spec
        self._deployment_metadata = dict(deployment_metadata or {})
        self._capabilities = ModelCapabilities(
            policy_family=PolicyFamily.VLA,
            action_chunking=True,
            configurable_inference_steps=True,
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

    def infer(self, request: InferenceRequest) -> ActionChunk:
        payload = encode_starvla_request(
            request,
            default_num_ddim_steps=self._default_num_ddim_steps,
            inference_steps_key=self._inference_steps_key,
            default_use_ddim=self._use_ddim,
            default_unnorm_key=self._unnorm_key,
        )
        started_s = time.perf_counter()
        output = self._client.predict_action(payload)
        wall_ms = (time.perf_counter() - started_s) * 1000.0
        actions = normalize_starvla_action_chunk(output)
        if self.action_spec is not None:
            self.action_spec.validate(actions)
        metadata: dict[str, Any] = {
            "backend": "starvla_websocket",
            "decoding": "ddim_action_chunk" if payload["use_ddim"] else "action_chunk",
            "precision": self._precision,
            "inference_steps": int(payload[self._inference_steps_key]),
            "inference_steps_parameter": self._inference_steps_key,
            "action_seed": payload.get("action_seed"),
            "input_sha256": hash_starvla_request_input(payload),
            "input_components": summarize_starvla_request_input(payload),
            "action_sha256": hashlib.sha256(
                np.ascontiguousarray(actions).tobytes()
            ).hexdigest(),
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
