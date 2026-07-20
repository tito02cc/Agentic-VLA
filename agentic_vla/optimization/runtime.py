"""Orchestration facade for preparing CARVE optimization profiles."""

from __future__ import annotations

from agentic_vla.runtime import PolicyAdapter

from .contracts import OptimizationProfile
from .plugins import PreparedPolicy
from .registry import PluginRegistry


class CarveOptimizeRuntime:
    """Resolve model/backend plugins and prepare a validated deployment profile."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def prepare(
        self,
        adapter: PolicyAdapter,
        profile: OptimizationProfile,
    ) -> PreparedPolicy:
        model = self.registry.resolve_model(adapter)
        backend = self.registry.get_backend(profile.backend)
        prepared = backend.prepare(adapter, model, profile)
        if prepared.adapter is not adapter and prepared.adapter.action_spec != adapter.action_spec:
            raise ValueError("optimization backend changed the declared action contract")
        return prepared
