"""Zero-transform eager execution for reference or predeployed policies."""

from __future__ import annotations

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile
from ..plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .base import validate_deployment_precision


class EagerBackend(BackendPlugin):
    @property
    def backend_id(self) -> str:
        return "eager"

    def prepare(
        self,
        adapter: PolicyAdapter,
        model: ModelPlugin,
        profile: OptimizationProfile,
    ) -> PreparedPolicy:
        if profile.backend != self.backend_id:
            raise ValueError(
                f"profile backend {profile.backend!r} cannot be prepared by {self.backend_id!r}"
            )
        if profile.async_execution:
            raise ValueError("eager backend does not implement asynchronous execution")
        if profile.module_precisions:
            raise ValueError("eager backend cannot apply module-specific precision transforms")
        validate_deployment_precision(adapter, profile)
        model.validate_profile(adapter, profile)
        precision = profile.deployment_precision
        if precision is None and len(adapter.capabilities.supported_precisions) == 1:
            precision = next(iter(adapter.capabilities.supported_precisions))
        behavioral_reference = bool(
            profile.options.get("behavioral_reference", precision in {None, "bf16"})
        )
        return PreparedPolicy(
            adapter=adapter,
            model_id=model.model_id,
            backend_id=self.backend_id,
            profile=profile,
            metadata={
                "transformed": False,
                "backend_execution": "eager",
                "predeployed_precision": precision,
                "behavioral_reference": behavioral_reference,
            },
        )
