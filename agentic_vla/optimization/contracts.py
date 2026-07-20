"""Stable contracts for CARVE optimization profiles and benchmark reports."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


def _normalized_mapping(values: Mapping[str, Any], *, lower_values: bool = False) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("mapping keys must not be empty")
        value = raw_value
        if lower_values and isinstance(value, str):
            value = value.strip().lower()
            if not value:
                raise ValueError(f"value for {key!r} must not be empty")
        normalized[key] = value
    return normalized


def _normalized_names(values: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    names = tuple(str(value).strip() for value in values)
    if not names or any(not value for value in names):
        raise ValueError(f"{field_name} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicates")
    return names


@dataclasses.dataclass(frozen=True)
class StaticMaskedViewContract:
    """Canonical input-view order and views guaranteed to be padding.

    The contract is independent of a specific VLA implementation. A backend
    remains responsible for deciding whether it can apply the contract to its
    model and for checking the declared mask on every inference call.
    """

    view_order: tuple[str, ...]
    masked_views: tuple[str, ...]

    def __post_init__(self) -> None:
        view_order = _normalized_names(self.view_order, field_name="view_order")
        masked_views = _normalized_names(self.masked_views, field_name="masked_views")
        unknown = tuple(name for name in masked_views if name not in view_order)
        if unknown:
            raise ValueError(f"masked_views are absent from view_order: {', '.join(unknown)}")
        if len(masked_views) == len(view_order):
            raise ValueError("masked-view contract cannot remove every input view")
        object.__setattr__(self, "view_order", view_order)
        object.__setattr__(self, "masked_views", masked_views)

    @property
    def masked_indices(self) -> tuple[int, ...]:
        indices = {self.view_order.index(name) for name in self.masked_views}
        return tuple(sorted(indices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_order": list(self.view_order),
            "masked_views": list(self.masked_views),
            "masked_indices": list(self.masked_indices),
            "runtime_mask_requirement": "all_batch_entries_false",
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "StaticMaskedViewContract":
        return cls(
            view_order=tuple(values["view_order"]),
            masked_views=tuple(values["masked_views"]),
        )


@dataclasses.dataclass(frozen=True)
class OptimizationProfile:
    """One prevalidated model/backend/hardware deployment configuration."""

    profile_id: str
    backend: str = "eager"
    deployment_precision: str | None = None
    module_precisions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    inference_steps: int | None = None
    action_horizon: int | None = None
    async_execution: bool = False
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        backend = str(self.backend).strip().lower()
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        if not backend:
            raise ValueError("backend must not be empty")
        if self.inference_steps is not None and self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        precision = self.deployment_precision
        if precision is not None:
            precision = str(precision).strip().lower()
            if not precision:
                raise ValueError("deployment_precision must not be empty")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "deployment_precision", precision)
        object.__setattr__(
            self,
            "module_precisions",
            _normalized_mapping(self.module_precisions, lower_values=True),
        )
        object.__setattr__(self, "options", _normalized_mapping(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "deployment_precision": self.deployment_precision,
            "module_precisions": dict(self.module_precisions),
            "inference_steps": self.inference_steps,
            "action_horizon": self.action_horizon,
            "async_execution": self.async_execution,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OptimizationProfile":
        return cls(**dict(values))


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    """Hardware identity used to prevent accidental profile reuse."""

    accelerator: str
    device_name: str
    total_memory_gb: float | None = None
    software: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        accelerator = str(self.accelerator).strip().lower()
        device_name = str(self.device_name).strip()
        if not accelerator or not device_name:
            raise ValueError("accelerator and device_name must not be empty")
        if self.total_memory_gb is not None and self.total_memory_gb <= 0:
            raise ValueError("total_memory_gb must be positive")
        object.__setattr__(self, "accelerator", accelerator)
        object.__setattr__(self, "device_name", device_name)
        object.__setattr__(self, "software", _normalized_mapping(self.software))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator": self.accelerator,
            "device_name": self.device_name,
            "total_memory_gb": self.total_memory_gb,
            "software": dict(self.software),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HardwareSpec":
        return cls(**dict(values))


@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """Warm inference latency and resource summary for one prepared profile."""

    samples: int
    runtime_p50_ms: float
    runtime_p95_ms: float
    runtime_p99_ms: float
    model_p50_ms: float
    model_p95_ms: float
    mean_action_count: float
    deadline_miss_rate: float
    peak_vram_gb: float | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if min(
            self.runtime_p50_ms,
            self.runtime_p95_ms,
            self.runtime_p99_ms,
            self.model_p50_ms,
            self.model_p95_ms,
            self.mean_action_count,
            self.deadline_miss_rate,
        ) < 0:
            raise ValueError("benchmark metrics must be non-negative")
        if not 0.0 <= self.deadline_miss_rate <= 1.0:
            raise ValueError("deadline_miss_rate must be in [0, 1]")
        object.__setattr__(self, "metadata", _normalized_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "metadata": dict(self.metadata),
        }
