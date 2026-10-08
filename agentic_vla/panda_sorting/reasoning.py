"""HAA-RAG, scene constraints, and bounded reflection for Panda sorting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import (
    AffordanceProfile,
    ExperienceCard,
    RecoveryDecision,
    RecoveryLevel,
    SceneObject,
)


class AffordanceMemory:
    """Compact retrieval store keyed by functional affordance, not visual identity."""

    def __init__(self, cards: Iterable[ExperienceCard] = (), max_cards: int = 128) -> None:
        if max_cards <= 0:
            raise ValueError("max_cards must be positive")
        self._cards: deque[ExperienceCard] = deque(cards, maxlen=max_cards)

    def add(self, card: ExperienceCard) -> None:
        self._cards.append(card)

    def retrieve(
        self,
        affordance: AffordanceProfile,
        *,
        context_tag: str,
        limit: int = 3,
    ) -> tuple[ExperienceCard, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        def score(card: ExperienceCard) -> tuple[float, float]:
            profile = card.affordance
            affinity = 0.0
            affinity += 2.0 if profile.object_type == affordance.object_type else 0.0
            affinity += 1.0 if profile.material == affordance.material else 0.0
            affinity += 1.0 if profile.grasp_region == affordance.grasp_region else 0.0
            affinity -= abs(profile.fragility - affordance.fragility)
            affinity += 0.5 if card.context_tag == context_tag else 0.0
            affinity += 0.25 if card.outcome == "success" else -0.25
            return affinity, card.confidence

        ranked = sorted(self._cards, key=score, reverse=True)
        return tuple(ranked[:limit])

    def __len__(self) -> int:
        return len(self._cards)


@dataclass(frozen=True)
class SceneConstraints:
    relations: tuple[str, ...]
    approach_height_delta_m: float
    lateral_clearance_m: float
    placement_margin_m: float
    lateral_escape_direction: tuple[float, float] | None = None


class SceneGraphReasoner:
    """Derive conservative grasp constraints from estimated tabletop relations."""

    def __init__(self, near_threshold_m: float = 0.11) -> None:
        if near_threshold_m <= 0.0:
            raise ValueError("near_threshold_m must be positive")
        self.near_threshold_m = near_threshold_m

    def infer(
        self,
        target: SceneObject,
        context: Iterable[SceneObject],
        *,
        target_fragility: float,
    ) -> SceneConstraints:
        relations: list[str] = []
        approach_delta = 0.0
        clearance = 0.02
        margin = 0.025 + 0.015 * target_fragility
        escape_direction: tuple[float, float] | None = None
        nearest_distance = float("inf")
        if target.table_position is None:
            return SceneConstraints(("target_position_uncertain",), 0.04, 0.05, margin + 0.02)

        tx, ty = target.table_position
        for other in context:
            if other.name == target.name or not other.visible or other.table_position is None:
                continue
            ox, oy = other.table_position
            distance = ((tx - ox) ** 2 + (ty - oy) ** 2) ** 0.5
            if distance < self.near_threshold_m:
                relations.append(f"near:{other.name}")
                approach_delta = max(approach_delta, 0.03)
                clearance = max(clearance, 0.05)
                if distance > 1e-6 and distance < nearest_distance:
                    escape_direction = ((tx - ox) / distance, (ty - oy) / distance)
                    nearest_distance = distance

        return SceneConstraints(
            tuple(relations) or ("clear_tabletop",),
            approach_delta,
            clearance,
            margin,
            escape_direction,
        )


class ReflectionPolicy:
    """Map explicit monitor evidence to the paper's L1 / L2 / L3 recovery levels."""

    def decide(
        self,
        evidence: Mapping[str, float | bool | str],
        *,
        retries_used: int,
        replans_used: int,
    ) -> RecoveryDecision:
        if bool(evidence.get("unsafe", False)):
            return RecoveryDecision(RecoveryLevel.SAFE_STOP, "unsafe_motion", "monitor requested a safe stop")
        if bool(evidence.get("target_lost", False)):
            if replans_used == 0:
                return RecoveryDecision(RecoveryLevel.FULL_REPLAN, "target_displaced", "target estimate changed after planning")
            return RecoveryDecision(RecoveryLevel.SAFE_STOP, "replan_budget_exhausted", "target remains uncertain after replan")
        if bool(evidence.get("object_not_lifted", False)):
            if retries_used == 0:
                return RecoveryDecision(
                    RecoveryLevel.PARAMETER_RETRY,
                    "grasp_missed",
                    "gripper closure did not produce a lift",
                    {"approach_height_delta_m": -0.01, "closure_hold_s": 0.25},
                )
            return RecoveryDecision(RecoveryLevel.SKILL_SWITCH, "repeated_grasp_miss", "use a wider re-approach skill")
        if bool(evidence.get("placement_invalid", False)):
            return RecoveryDecision(
                RecoveryLevel.PARAMETER_RETRY,
                "placement_deviation",
                "object is not visually verified inside its destination",
                {"placement_margin_delta_m": 0.02},
            )
        return RecoveryDecision(RecoveryLevel.FULL_REPLAN, "unclassified_failure", "reobserve the scene before continuing")
