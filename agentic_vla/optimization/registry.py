"""Explicit plugin registry for CARVE Optimize Runtime."""

from __future__ import annotations

from agentic_vla.runtime import PolicyAdapter

from .plugins import BackendPlugin, ModelPlugin


class PluginRegistry:
    """Register model and backend plugins without import-time global mutation."""

    def __init__(self) -> None:
        self._models: dict[str, ModelPlugin] = {}
        self._backends: dict[str, BackendPlugin] = {}

    def register_model(self, plugin: ModelPlugin) -> None:
        key = plugin.model_id.strip().lower()
        if not key:
            raise ValueError("model plugin id must not be empty")
        if key in self._models:
            raise ValueError(f"model plugin {key!r} is already registered")
        self._models[key] = plugin

    def register_backend(self, plugin: BackendPlugin) -> None:
        key = plugin.backend_id.strip().lower()
        if not key:
            raise ValueError("backend plugin id must not be empty")
        if key in self._backends:
            raise ValueError(f"backend plugin {key!r} is already registered")
        self._backends[key] = plugin

    def resolve_model(self, adapter: PolicyAdapter) -> ModelPlugin:
        matches = [plugin for plugin in self._models.values() if plugin.matches(adapter)]
        if not matches:
            raise LookupError(f"no model plugin matches adapter {adapter.adapter_id!r}")
        matches.sort(key=lambda plugin: plugin.priority, reverse=True)
        if len(matches) > 1 and matches[0].priority == matches[1].priority:
            ids = ", ".join(plugin.model_id for plugin in matches if plugin.priority == matches[0].priority)
            raise LookupError(f"ambiguous model plugins for {adapter.adapter_id!r}: {ids}")
        return matches[0]

    def get_backend(self, backend_id: str) -> BackendPlugin:
        key = str(backend_id).strip().lower()
        try:
            return self._backends[key]
        except KeyError as exc:
            raise LookupError(f"backend plugin {key!r} is not registered") from exc

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))
