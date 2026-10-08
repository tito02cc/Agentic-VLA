"""Fail-closed reuse admission for CARVE's multi-rate inference runtime."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any


class ReuseLayer(str, enum.Enum):
    """Reusable computation exposed by an optimization backend."""

    ACTION_QUEUE = "action_queue"
    VISUAL_PREFIX = "visual_prefix"
    ACTION_WARM_START = "action_warm_start"


@dataclasses.dataclass(frozen=True)
class ReuseCandidate:
    """Identity and freshness contract attached to one cached artifact."""

    candidate_id: str
    layer: ReuseLayer
    profile_id: str
    instruction_fingerprint: str
    subgoal_fingerprint: str
    planner_epoch: int
    recovery_epoch: int
    age_steps: int
    similarity: float | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "profile_id",
            "instruction_fingerprint",
            "subgoal_fingerprint",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if self.planner_epoch < 0 or self.recovery_epoch < 0 or self.age_steps < 0:
            raise ValueError("reuse candidate epochs and age must be non-negative")
        if self.similarity is not None and not -1.0 <= self.similarity <= 1.0:
            raise ValueError("reuse similarity must be in [-1, 1]")


@dataclasses.dataclass(frozen=True)
class ReuseContext:
    """Current deployable context used to accept or invalidate reuse."""

    profile_id: str
    instruction_fingerprint: str
    subgoal_fingerprint: str
    planner_epoch: int
    recovery_epoch: int
    risk_bucket: str
    event: str | None = None
    visual_change: float | None = None
    proprio_change: float | None = None

    def __post_init__(self) -> None:
        for name in ("profile_id", "instruction_fingerprint", "subgoal_fingerprint"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if self.planner_epoch < 0 or self.recovery_epoch < 0:
            raise ValueError("reuse context epochs must be non-negative")
        bucket = str(self.risk_bucket).strip().lower()
        if bucket not in {"low", "medium", "high"}:
            raise ValueError("risk_bucket must be low, medium, or high")
        object.__setattr__(self, "risk_bucket", bucket)
        for name in ("visual_change", "proprio_change"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value:
                raise ValueError(f"{name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class ReuseGateConfig:
    """Conservative thresholds; deployment profiles may tighten them."""

    action_queue_max_age_steps: int = 10
    visual_prefix_max_age_steps: int = 1
    action_warm_start_max_age_steps: int = 1000
    max_visual_change: float = 0.01
    max_proprio_change: float = 0.05
    action_warm_start_min_similarity: float = 0.9

    def __post_init__(self) -> None:
        ages = (
            self.action_queue_max_age_steps,
            self.visual_prefix_max_age_steps,
            self.action_warm_start_max_age_steps,
        )
        if any(value < 0 for value in ages):
            raise ValueError("reuse age limits must be non-negative")
        if self.max_visual_change < 0 or self.max_proprio_change < 0:
            raise ValueError("reuse change limits must be non-negative")
        if not -1.0 <= self.action_warm_start_min_similarity <= 1.0:
            raise ValueError("warm-start similarity threshold must be in [-1, 1]")


@dataclasses.dataclass(frozen=True)
class ReuseDecision:
    """Auditable decision consumed by a backend or action-queue controller."""

    accepted: bool
    force_refresh: bool
    layer: ReuseLayer
    candidate_id: str
    reason: str
    invalidations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["layer"] = self.layer.value
        return payload


class EventCoherentReuseGate:
    """Admit reuse only while Agentic and physical context remain coherent.

    The gate does not implement a cache. It provides the shared invalidation
    protocol used by action queues and future visual-prefix or action-warm-start
    backend plugins.
    """

    def __init__(self, config: ReuseGateConfig | None = None) -> None:
        self.config = config or ReuseGateConfig()

    def evaluate(self, candidate: ReuseCandidate, context: ReuseContext) -> ReuseDecision:
        invalidations: list[str] = []
        if candidate.profile_id != context.profile_id:
            invalidations.append("profile_changed")
        if candidate.instruction_fingerprint != context.instruction_fingerprint:
            invalidations.append("instruction_changed")
        if candidate.subgoal_fingerprint != context.subgoal_fingerprint:
            invalidations.append("subgoal_changed")
        if candidate.planner_epoch != context.planner_epoch:
            invalidations.append("planner_epoch_changed")
        if candidate.recovery_epoch != context.recovery_epoch:
            invalidations.append("recovery_epoch_changed")
        if context.event is not None:
            invalidations.append(f"execution_event:{context.event}")
        if context.risk_bucket != "low":
            invalidations.append(f"risk_bucket:{context.risk_bucket}")
        if (
            context.visual_change is not None
            and context.visual_change > self.config.max_visual_change
        ):
            invalidations.append("visual_context_changed")
        if (
            context.proprio_change is not None
            and context.proprio_change > self.config.max_proprio_change
        ):
            invalidations.append("proprio_context_changed")

        max_age = {
            ReuseLayer.ACTION_QUEUE: self.config.action_queue_max_age_steps,
            ReuseLayer.VISUAL_PREFIX: self.config.visual_prefix_max_age_steps,
            ReuseLayer.ACTION_WARM_START: self.config.action_warm_start_max_age_steps,
        }[candidate.layer]
        if candidate.age_steps > max_age:
            invalidations.append("candidate_stale")
        if candidate.layer is ReuseLayer.ACTION_WARM_START:
            if candidate.similarity is None:
                invalidations.append("similarity_missing")
            elif candidate.similarity < self.config.action_warm_start_min_similarity:
                invalidations.append("similarity_below_threshold")

        accepted = not invalidations
        reason = "event-coherent reuse admitted" if accepted else invalidations[0]
        return ReuseDecision(
            accepted=accepted,
            force_refresh=not accepted,
            layer=candidate.layer,
            candidate_id=candidate.candidate_id,
            reason=reason,
            invalidations=tuple(invalidations),
        )
