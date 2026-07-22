"""Adapter abstraction and capability negotiation for embodied policies."""

from __future__ import annotations

import abc
import dataclasses
from typing import Literal

from .contracts import ActionChunk, ActionSpec, InferenceControls, InferenceRequest, ModelCapabilities


FallbackMode = Literal["graceful", "strict"]


class UnsupportedControlError(ValueError):
    """Raised when strict negotiation requests an unsupported native control."""


@dataclasses.dataclass(frozen=True)
class NegotiatedControls:
    controls: InferenceControls
    dropped: tuple[str, ...] = ()


class PolicyAdapter(abc.ABC):
    """Stable boundary between CARVE and a concrete action policy."""

    @property
    @abc.abstractmethod
    def adapter_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    @property
    def action_spec(self) -> ActionSpec | None:
        """Declared action semantics, or None for an unvalidated prototype."""

        return None

    def reset(self, episode_id: str | int | None = None) -> None:
        """Reset backend state when a new episode starts."""

    @abc.abstractmethod
    def infer(self, request: InferenceRequest) -> ActionChunk:
        raise NotImplementedError

    def negotiate(
        self,
        controls: InferenceControls,
        *,
        fallback_mode: FallbackMode = "graceful",
    ) -> NegotiatedControls:
        """Resolve controls without pretending unsupported features exist."""

        dropped: list[str] = []
        inference_steps = controls.inference_steps
        precision = controls.precision
        reuse_context = controls.reuse_context

        if inference_steps is not None and not self.capabilities.configurable_inference_steps:
            dropped.append("inference_steps")
            inference_steps = None
        if precision is not None and precision not in self.capabilities.supported_precisions:
            dropped.append("precision")
            precision = None
        if reuse_context and not (self.capabilities.kv_cache or self.capabilities.predictive_context):
            dropped.append("reuse_context")
            reuse_context = False

        if dropped and fallback_mode == "strict":
            joined = ", ".join(dropped)
            raise UnsupportedControlError(f"{self.adapter_id} does not support: {joined}")

        negotiated = dataclasses.replace(
            controls,
            inference_steps=inference_steps,
            precision=precision,
            reuse_context=reuse_context,
        )
        return NegotiatedControls(controls=negotiated, dropped=tuple(dropped))
