"""Model-neutral contracts for CARVE's agent-facing embodied tools."""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Mapping
from typing import Any


_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_FORBIDDEN_DIRECT_ACTION_FIELDS = frozenset(
    {
        "action",
        "actions",
        "joint_positions",
        "joint_targets",
        "object_pose",
        "object_poses",
        "reward",
        "sim_state",
        "simulator_state",
        "success",
        "torques",
        "trajectory",
    }
)


class ToolEffect(str, enum.Enum):
    """Authority class used by the harness permission gate."""

    OBSERVE = "observe"
    MEMORY = "memory"
    VLA = "vla"
    SKILL = "skill"
    LIFECYCLE = "lifecycle"


class PrimitiveStatus(str, enum.Enum):
    """Terminal status of one bounded robot primitive invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


class VerificationStatus(str, enum.Enum):
    """Deployable result of checking one expected primitive outcome."""

    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


@dataclasses.dataclass(frozen=True)
class VerificationReport:
    """Typed verification evidence that may gate persistent memory writes."""

    status: VerificationStatus | str
    observed_outcome: str
    confidence: float
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            status = (
                self.status
                if isinstance(self.status, VerificationStatus)
                else VerificationStatus(str(self.status).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("unsupported verification status") from exc
        if not self.observed_outcome.strip():
            raise ValueError("verification observed_outcome must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("verification confidence must be in [0, 1]")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("verification metadata must be a mapping")
        reject_direct_action_fields(self.metadata, path="verification.metadata")
        object.__setattr__(self, "status", status)

    @property
    def definitive(self) -> bool:
        return self.status is not VerificationStatus.INCONCLUSIVE

    @property
    def outcome_met(self) -> bool | None:
        if self.status is VerificationStatus.CONFIRMED:
            return True
        if self.status is VerificationStatus.CONTRADICTED:
            return False
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "observed_outcome": self.observed_outcome,
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "VerificationReport":
        if not isinstance(values, Mapping):
            raise TypeError("verification report must be a mapping")
        return cls(**dict(values))


@dataclasses.dataclass(frozen=True)
class PrimitiveOutcome:
    """Deployable evidence emitted when a primitive returns control to the harness.

    A primitive is a bounded callable robot operation such as one frozen-VLA
    contact attempt, a Cartesian move, or a registered recovery skill. The
    record deliberately excludes raw actions and privileged simulator state.
    """

    call_id: str
    primitive_name: str
    status: PrimitiveStatus | str
    episode_id: str | int
    started_timestep: int
    ended_timestep: int
    expected_outcome: str = ""
    observed_outcome: str = ""
    requires_semantic_check: bool = False
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.primitive_name.strip():
            raise ValueError("primitive call_id and name must not be empty")
        if self.started_timestep < 0 or self.ended_timestep < self.started_timestep:
            raise ValueError("primitive timesteps must be ordered and non-negative")
        try:
            status = (
                self.status
                if isinstance(self.status, PrimitiveStatus)
                else PrimitiveStatus(str(self.status).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("unsupported primitive status") from exc
        if not isinstance(self.metadata, Mapping):
            raise TypeError("primitive metadata must be a mapping")
        reject_direct_action_fields(self.metadata, path="primitive.metadata")
        object.__setattr__(self, "status", status)

    @property
    def abnormal(self) -> bool:
        return self.status is not PrimitiveStatus.SUCCEEDED

    def to_planner_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "primitive_name": self.primitive_name,
            "status": self.status.value,
            "started_timestep": self.started_timestep,
            "ended_timestep": self.ended_timestep,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "requires_semantic_check": self.requires_semantic_check,
            "metadata": dict(self.metadata),
        }


def reject_direct_action_fields(value: Any, *, path: str = "tool_input") -> None:
    """Keep coding agents behind typed VLA and skill interfaces."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_DIRECT_ACTION_FIELDS:
                raise ValueError(f"{path} contains forbidden field: {normalized}")
            reject_direct_action_fields(item, path=f"{path}.{normalized}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_direct_action_fields(item, path=f"{path}[{index}]")


def _validate_type(value: Any, expected: str, *, path: str) -> None:
    valid = {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, Mapping),
        "string": isinstance(value, str),
    }.get(expected)
    if valid is None:
        raise ValueError(f"{path} uses unsupported schema type: {expected}")
    if not valid:
        raise TypeError(f"{path} must be {expected}")


def validate_tool_input(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    path: str = "tool_input",
) -> None:
    """Validate the bounded JSON-schema subset used by CARVE tools."""

    if schema.get("type", "object") != "object":
        raise ValueError("CARVE tool input schema root must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    if not isinstance(properties, Mapping):
        raise TypeError("tool schema properties must be a mapping")
    if not isinstance(required, (list, tuple)):
        raise TypeError("tool schema required must be a sequence")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    if schema.get("additionalProperties", False) is False:
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ValueError(f"{path} contains unknown fields: {unknown}")
    for name, item in value.items():
        item_schema = properties.get(name)
        if item_schema is None:
            continue
        if not isinstance(item_schema, Mapping):
            raise TypeError(f"schema for {name} must be a mapping")
        expected = item_schema.get("type")
        if expected is not None:
            _validate_type(item, str(expected), path=f"{path}.{name}")
        if isinstance(item, str):
            if item_schema.get("minLength", 0) and not item.strip():
                raise ValueError(f"{path}.{name} must not be empty")
            choices = item_schema.get("enum")
            if choices is not None and item not in choices:
                raise ValueError(f"{path}.{name} must be one of {list(choices)}")
        if isinstance(item, list):
            maximum = item_schema.get("maxItems")
            if maximum is not None and len(item) > int(maximum):
                raise ValueError(f"{path}.{name} exceeds maxItems={maximum}")


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """One discoverable tool and its execution authority."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    effect: ToolEffect
    requires_safe_boundary: bool = False
    max_calls_per_episode: int | None = None

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        if not _TOOL_NAME.fullmatch(name):
            raise ValueError(f"invalid CARVE tool name: {self.name!r}")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if self.max_calls_per_episode is not None and self.max_calls_per_episode < 0:
            raise ValueError("tool max_calls_per_episode must be non-negative")
        if self.input_schema.get("type", "object") != "object":
            raise ValueError("tool input schema must describe an object")
        object.__setattr__(self, "name", name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "effect": self.effect.value,
            "requires_safe_boundary": self.requires_safe_boundary,
            "max_calls_per_episode": self.max_calls_per_episode,
        }


@dataclasses.dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    episode_id: str | int
    timestep: int

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.name.strip():
            raise ValueError("tool call_id and name must not be empty")
        if self.timestep < 0:
            raise ValueError("tool call timestep must be non-negative")
        reject_direct_action_fields(self.arguments)


@dataclasses.dataclass(frozen=True)
class ToolExecutionContext:
    """Episode-scoped authority supplied by the CARVE harness."""

    episode_id: str | int
    timestep: int
    at_safe_boundary: bool
    allowed_tools: tuple[str, ...]
    deployment_profile_id: str = ""
    inference_controls: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("tool context timestep must be non-negative")
        normalized = tuple(name.strip().lower() for name in self.allowed_tools)
        if any(not name for name in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("allowed_tools must contain unique non-empty names")
        if not isinstance(self.inference_controls, Mapping):
            raise TypeError("inference_controls must be a mapping")
        allowed_controls = {
            "inference_steps",
            "max_actions",
            "precision",
            "reuse_context",
            "deadline_ms",
        }
        unknown = sorted(set(self.inference_controls) - allowed_controls)
        if unknown:
            raise ValueError(f"inference_controls contains unknown fields: {unknown}")
        object.__setattr__(self, "allowed_tools", normalized)
        object.__setattr__(self, "inference_controls", dict(self.inference_controls))


@dataclasses.dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    accepted: bool
    output: Mapping[str, Any]
    elapsed_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
