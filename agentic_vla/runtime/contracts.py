"""Model-agnostic contracts for the CARVE embodied-policy runtime."""

from __future__ import annotations

import dataclasses
import enum
import time
import uuid
from collections.abc import Mapping
from typing import Any


class PolicyFamily(str, enum.Enum):
    """Broad policy families understood by the runtime."""

    VLA = "vla"
    WAM = "wam"
    OTHER = "other"


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    """Robot action semantics required by a policy deployment."""

    action_dim: int
    representation: str
    coordinate_frame: str
    gripper_convention: str
    control_frequency_hz: float
    normalization_id: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        for name in ("representation", "coordinate_frame", "gripper_convention"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")

    def validate(self, actions: Any) -> None:
        """Validate shape, finite values, and optional normalized bounds."""

        try:
            rows = list(actions)
        except TypeError as exc:
            raise TypeError("actions must be an iterable of action vectors") from exc
        if not rows:
            raise ValueError("actions must not be empty")
        tolerance = 1e-5
        for index, row in enumerate(rows):
            try:
                values = list(row)
            except TypeError as exc:
                raise TypeError(f"action {index} is not a vector") from exc
            if len(values) != self.action_dim:
                raise ValueError(
                    f"action {index} has dimension {len(values)}, expected {self.action_dim}"
                )
            for value in values:
                scalar = float(value)
                if scalar != scalar or scalar in (float("inf"), float("-inf")):
                    raise ValueError(f"action {index} contains a non-finite value")
                if self.minimum is not None and scalar < self.minimum - tolerance:
                    raise ValueError(f"action {index} is below minimum {self.minimum}")
                if self.maximum is not None and scalar > self.maximum + tolerance:
                    raise ValueError(f"action {index} is above maximum {self.maximum}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ModelCapabilities:
    """Native controls and outputs exposed by one policy adapter."""

    policy_family: PolicyFamily = PolicyFamily.VLA
    action_chunking: bool = True
    configurable_inference_steps: bool = False
    autoregressive_decoding: bool = False
    kv_cache: bool = False
    predictive_context: bool = False
    uncertainty: bool = False
    async_inference: bool = False
    supported_precisions: frozenset[str] = dataclasses.field(
        default_factory=lambda: frozenset({"bf16"})
    )
    max_action_horizon: int | None = None

    def __post_init__(self) -> None:
        precisions = frozenset(str(value).lower() for value in self.supported_precisions)
        if not precisions:
            raise ValueError("supported_precisions must not be empty")
        if self.max_action_horizon is not None and self.max_action_horizon <= 0:
            raise ValueError("max_action_horizon must be positive")
        object.__setattr__(self, "supported_precisions", precisions)


@dataclasses.dataclass(frozen=True)
class InferenceControls:
    """Optional per-call compute and execution controls.

    max_actions is a runtime output cap. Other fields request native model
    behavior and therefore require an adapter capability.
    """

    inference_steps: int | None = None
    max_actions: int | None = None
    precision: str | None = None
    reuse_context: bool = False
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.inference_steps is not None and self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.max_actions is not None and self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.precision is not None:
            object.__setattr__(self, "precision", self.precision.lower())


@dataclasses.dataclass(frozen=True)
class InferenceRequest:
    """Normalized observation, instruction, lifecycle, and compute request."""

    observation: Mapping[str, Any]
    instruction: str
    controls: InferenceControls = dataclasses.field(default_factory=InferenceControls)
    episode_id: str | int | None = None
    timestep: int | None = None
    agentic: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    request_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    created_at_s: float = dataclasses.field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise TypeError("observation must be a mapping")
        if not isinstance(self.instruction, str):
            raise TypeError("instruction must be a string")
        if self.timestep is not None and self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if self.agentic is not None and not isinstance(self.agentic, Mapping):
            raise TypeError("agentic must be a mapping")

    def with_controls(self, controls: InferenceControls) -> "InferenceRequest":
        return dataclasses.replace(self, controls=controls)


@dataclasses.dataclass
class ActionChunk:
    """Policy actions plus backend timing and auxiliary outputs."""

    actions: Any
    model_latency_ms: float
    raw_output: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    uncertainty: Any | None = None
    predictive_context: Any | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            action_count = len(self.actions)
        except TypeError as exc:
            raise TypeError("actions must be a sized action sequence") from exc
        if action_count <= 0:
            raise ValueError("actions must not be empty")
        if self.model_latency_ms < 0:
            raise ValueError("model_latency_ms must be non-negative")

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def limited(self, max_actions: int | None) -> "ActionChunk":
        if max_actions is None or self.action_count <= max_actions:
            return self
        return dataclasses.replace(self, actions=self.actions[:max_actions])

    def as_legacy_output(self) -> dict[str, Any]:
        output = dict(self.raw_output)
        output["actions"] = self.actions
        return output


@dataclasses.dataclass(frozen=True)
class RuntimeTrace:
    """One machine-readable CARVE policy-call trace."""

    request_id: str
    adapter_id: str
    policy_family: str
    episode_id: str | int | None
    timestep: int | None
    requested_controls: Mapping[str, Any]
    applied_controls: Mapping[str, Any]
    dropped_controls: tuple[str, ...]
    queue_age_ms: float
    runtime_latency_ms: float
    model_latency_ms: float
    action_count: int
    deadline_ms: float | None
    deadline_miss: bool
    success: bool
    error: str | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reaction_latency_ms: float | None = None
    task_cycle_latency_ms: float | None = None
    action_age_steps: int | None = None
    stage_latencies_ms: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        optional_latencies = (
            self.reaction_latency_ms,
            self.task_cycle_latency_ms,
        )
        if any(value is not None and value < 0 for value in optional_latencies):
            raise ValueError("optional runtime latencies must be non-negative")
        if self.action_age_steps is not None and self.action_age_steps < 0:
            raise ValueError("action_age_steps must be non-negative")
        if any(float(value) < 0 for value in self.stage_latencies_ms.values()):
            raise ValueError("stage latencies must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
