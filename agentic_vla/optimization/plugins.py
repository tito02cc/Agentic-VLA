"""Plugin interfaces for model discovery and inference backend preparation."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from typing import Any

from agentic_vla.runtime import InferenceControls, PolicyAdapter

from .contracts import OptimizationProfile


class ModelPlugin(abc.ABC):
    """Describe model-specific capabilities without leaking them into CARVE core."""

    priority: int = 0

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def matches(self, adapter: PolicyAdapter) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def component_groups(self, adapter: PolicyAdapter) -> Mapping[str, tuple[str, ...]]:
        raise NotImplementedError

    def validate_profile(self, adapter: PolicyAdapter, profile: OptimizationProfile) -> None:
        """Reject profiles that the model cannot represent."""


@dataclasses.dataclass(frozen=True)
class PreparedPolicy:
    """A policy adapter bound to one optimization profile and backend."""

    adapter: PolicyAdapter
    model_id: str
    backend_id: str
    profile: OptimizationProfile
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def apply_profile(self, controls: InferenceControls) -> InferenceControls:
        """Overlay deployment controls while preserving request-specific deadlines."""

        return dataclasses.replace(
            controls,
            inference_steps=(
                self.profile.inference_steps
                if self.profile.inference_steps is not None
                else controls.inference_steps
            ),
            max_actions=(
                self.profile.action_horizon
                if self.profile.action_horizon is not None
                else controls.max_actions
            ),
            precision=(
                self.profile.deployment_precision
                if self.profile.deployment_precision is not None
                else controls.precision
            ),
        )


class BackendPlugin(abc.ABC):
    """Build or validate an executable backend for one policy adapter."""

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(
        self,
        adapter: PolicyAdapter,
        model: ModelPlugin,
        profile: OptimizationProfile,
    ) -> PreparedPolicy:
        raise NotImplementedError
