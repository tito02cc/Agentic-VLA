"""Typed, action-free contracts for the Panda sorting demonstration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


@dataclass(frozen=True)
class AffordanceProfile:
    """Manipulation-relevant description used by HAA-RAG."""

    object_type: str
    material: str
    fragility: float
    grasp_region: str

    def __post_init__(self) -> None:
        if not self.object_type.strip() or not self.material.strip() or not self.grasp_region.strip():
            raise ValueError("affordance fields must not be empty")
        if not 0.0 <= self.fragility <= 1.0:
            raise ValueError("fragility must be in [0, 1]")


@dataclass(frozen=True)
class SortingTask:
    """Semantic task specification, independent from simulator object state."""

    task_id: str
    instruction: str
    ordered_objects: tuple[str, ...]
    destinations: Mapping[str, str]
    affordances: Mapping[str, AffordanceProfile]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.instruction.strip():
            raise ValueError("task_id and instruction must not be empty")
        if len(self.ordered_objects) < 2:
            raise ValueError("the demonstration must contain at least two sequential objects")
        if len(set(self.ordered_objects)) != len(self.ordered_objects):
            raise ValueError("ordered_objects must be unique")
        missing_destinations = set(self.ordered_objects) - set(self.destinations)
        missing_affordances = set(self.ordered_objects) - set(self.affordances)
        if missing_destinations or missing_affordances:
            raise ValueError("each ordered object requires a destination and affordance profile")


@dataclass(frozen=True)
class SceneObject:
    """An object estimate produced by a perception module, never simulator truth."""

    name: str
    pixel_center: tuple[float, float]
    table_position: tuple[float, float] | None
    confidence: float
    visible: bool = True
    world_position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scene object name must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.world_position is not None and len(self.world_position) != 3:
            raise ValueError("world_position must contain XYZ coordinates")


@dataclass(frozen=True)
class ExperienceCard:
    """Small structured RAG item. It stores evidence, not control trajectories."""

    card_id: str
    affordance: AffordanceProfile
    context_tag: str
    approach_height_m: float
    gripper_mode: str
    outcome: str
    confidence: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.card_id.strip() or not self.context_tag.strip() or not self.gripper_mode.strip():
            raise ValueError("experience card identifiers must not be empty")
        if self.outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.approach_height_m <= 0.0:
            raise ValueError("approach_height_m must be positive")


class RecoveryLevel(str, Enum):
    PARAMETER_RETRY = "L1_parameter_retry"
    SKILL_SWITCH = "L2_skill_switch"
    FULL_REPLAN = "L3_full_replan"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class RecoveryDecision:
    """Bounded recovery intent selected from monitor evidence."""

    level: RecoveryLevel
    failure_type: str
    rationale: str
    parameter_updates: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.failure_type.strip() or not self.rationale.strip():
            raise ValueError("failure_type and rationale must not be empty")
