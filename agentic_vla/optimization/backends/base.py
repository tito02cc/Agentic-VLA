"""Shared backend validation helpers."""

from __future__ import annotations

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile


def validate_deployment_precision(adapter: PolicyAdapter, profile: OptimizationProfile) -> None:
    precision = profile.deployment_precision
    if precision is not None and precision not in adapter.capabilities.supported_precisions:
        supported = ", ".join(sorted(adapter.capabilities.supported_precisions))
        raise ValueError(
            f"adapter {adapter.adapter_id!r} is deployed as {supported}; "
            f"profile requested {precision!r}"
        )
