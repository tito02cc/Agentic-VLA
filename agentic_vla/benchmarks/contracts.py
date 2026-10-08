"""Benchmark boundaries that keep evaluator truth outside CARVE decisions."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from agentic_vla.runtime.knowledge import reject_privileged_semantic_fields


@dataclasses.dataclass(frozen=True)
class DeployableBenchmarkObservation:
    """Observation bundle available to policy, planner, and Fast Guard."""

    policy_observation: Mapping[str, Any]
    planner_frames: Mapping[str, Any]
    robot_state: tuple[float, ...]
    timestep: int
    frame_id: str

    def __post_init__(self) -> None:
        if self.timestep < 0 or not self.frame_id.strip():
            raise ValueError("observation timestep and frame_id are invalid")
        if not isinstance(self.policy_observation, Mapping):
            raise TypeError("policy_observation must be a mapping")
        if not isinstance(self.planner_frames, Mapping) or not self.planner_frames:
            raise ValueError("planner_frames must contain at least one camera")
        reject_privileged_semantic_fields(
            self.policy_observation, path="policy_observation"
        )


@dataclasses.dataclass(frozen=True)
class PrivateBenchmarkResult:
    """Evaluator-only episode label; never accepted by planner constructors."""

    task_success: bool
    episode_steps: int
    terminated: bool
    evaluator_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episode_steps < 0:
            raise ValueError("episode_steps must be non-negative")
        if not isinstance(self.evaluator_metadata, Mapping):
            raise TypeError("evaluator_metadata must be a mapping")

    def to_private_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
