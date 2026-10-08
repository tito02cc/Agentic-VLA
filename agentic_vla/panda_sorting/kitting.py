"""Non-scripted task contracts and planner for dynamic tabletop kitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .contracts import AffordanceProfile, SceneObject


class KittingSkill(str, Enum):
    """Registered semantic skills available to the high-level planner."""

    PLACE_TARGET = "place_target"
    CLEAR_TO_STAGING = "clear_to_staging"
    REOBSERVE = "reobserve_scene"
    VERIFY_FINAL_STATE = "verify_final_state"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class KittingTask:
    """Desired final state without a pre-scripted object execution order."""

    task_id: str
    instruction: str
    requested_objects: tuple[str, ...]
    destinations: Mapping[str, str]
    affordances: Mapping[str, AffordanceProfile]
    protected_objects: tuple[str, ...] = ()
    staging_region: str = "staging_region"

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.instruction.strip():
            raise ValueError("task_id and instruction must not be empty")
        if len(self.requested_objects) < 2:
            raise ValueError("a representative kitting task requires at least two targets")
        if len(set(self.requested_objects)) != len(self.requested_objects):
            raise ValueError("requested_objects must be unique")
        targets = set(self.requested_objects)
        if targets - set(self.destinations) or targets - set(self.affordances):
            raise ValueError("each requested object requires a destination and affordance")
        if targets & set(self.protected_objects):
            raise ValueError("requested and protected objects must be disjoint")


@dataclass(frozen=True)
class KittingSceneState:
    """Observation-side semantic state; no simulator labels or body poses."""

    objects: Mapping[str, SceneObject]
    completed_objects: frozenset[str] = field(default_factory=frozenset)
    destination_occupants: Mapping[str, str] = field(default_factory=dict)
    blocked_by: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    nearby_risks: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    staged_objects: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown_completed = set(self.completed_objects) - set(self.objects)
        if unknown_completed:
            raise ValueError(f"completed objects are absent from scene memory: {sorted(unknown_completed)}")


@dataclass(frozen=True)
class KittingSubgoal:
    """Auditable semantic decision handed to a registered physical skill."""

    skill: KittingSkill
    object_name: str | None
    destination: str | None
    reason: str
    target_object: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("subgoal reason must not be empty")
        if self.skill in {KittingSkill.PLACE_TARGET, KittingSkill.CLEAR_TO_STAGING}:
            if not self.object_name or not self.destination:
                raise ValueError("physical subgoals require object_name and destination")


class KittingStateEstimator:
    """Build graph predicates from camera-derived object positions and fixtures."""

    def __init__(
        self,
        destination_centers: Mapping[str, tuple[float, float, float]],
        *,
        occupancy_radius_m: float = 0.085,
        blocking_radius_m: float = 0.055,
        nearby_risk_radius_m: float = 0.140,
    ) -> None:
        if (
            occupancy_radius_m <= 0.0
            or blocking_radius_m <= 0.0
            or nearby_risk_radius_m <= blocking_radius_m
        ):
            raise ValueError("semantic radii must be positive")
        self.destination_centers = dict(destination_centers)
        self.occupancy_radius_m = float(occupancy_radius_m)
        self.blocking_radius_m = float(blocking_radius_m)
        self.nearby_risk_radius_m = float(nearby_risk_radius_m)

    def estimate(
        self,
        task: KittingTask,
        objects: Mapping[str, SceneObject],
        *,
        completed_objects: frozenset[str] = frozenset(),
        staged_objects: frozenset[str] = frozenset(),
    ) -> KittingSceneState:
        visible = {
            name: item
            for name, item in objects.items()
            if item.visible and item.table_position is not None
        }
        destination_occupants: dict[str, str] = {}
        for destination in set(task.destinations.values()):
            if destination not in self.destination_centers:
                raise ValueError(f"unknown destination fixture: {destination}")
            center = self.destination_centers[destination]
            nearest: tuple[float, str] | None = None
            for name, item in visible.items():
                distance = _planar_distance(item.table_position, center[:2])
                if distance <= self.occupancy_radius_m and (
                    nearest is None or distance < nearest[0]
                ):
                    nearest = (distance, name)
            if nearest is not None:
                destination_occupants[destination] = nearest[1]

        blocked_by: dict[str, tuple[str, ...]] = {}
        nearby_risks: dict[str, tuple[str, ...]] = {}
        for target in task.requested_objects:
            target_item = visible.get(target)
            if target_item is None or target in completed_objects:
                continue
            neighbors = [
                (
                    _planar_distance(target_item.table_position, item.table_position),
                    name,
                )
                for name, item in visible.items()
                if name != target
                and name not in completed_objects
                and name not in staged_objects
            ]
            blockers = [
                name for distance, name in neighbors if distance <= self.blocking_radius_m
            ]
            risks = [
                name
                for distance, name in neighbors
                if self.blocking_radius_m < distance <= self.nearby_risk_radius_m
            ]
            if blockers:
                blocked_by[target] = tuple(sorted(blockers))
            if risks:
                nearby_risks[target] = tuple(sorted(risks))

        return KittingSceneState(
            objects=dict(objects),
            completed_objects=completed_objects,
            destination_occupants=destination_occupants,
            blocked_by=blocked_by,
            nearby_risks=nearby_risks,
            staged_objects=staged_objects,
        )


class DependencyAwareKittingPlanner:
    """Select the next valid subgoal from final-state constraints.

    The planner does not receive an episode action script. It first chooses any
    currently executable requested object. Only when no target is executable
    does it emit the prerequisite clearing action for the highest-priority
    unfinished target.
    """

    def select_next(self, task: KittingTask, state: KittingSceneState) -> KittingSubgoal:
        """Return the deterministic fallback from the valid candidate set."""
        return self.candidate_subgoals(task, state)[0]

    def candidate_subgoals(
        self,
        task: KittingTask,
        state: KittingSceneState,
    ) -> tuple[KittingSubgoal, ...]:
        """Enumerate only precondition-valid semantic actions for a VLM router."""
        unfinished = [
            name for name in task.requested_objects if name not in state.completed_objects
        ]
        if not unfinished:
            return (
                KittingSubgoal(
                    KittingSkill.VERIFY_FINAL_STATE,
                    None,
                    None,
                    "all requested objects have verified completion records",
                ),
            )

        uncertain = [
            name
            for name in unfinished
            if name not in state.objects
            or not state.objects[name].visible
            or state.objects[name].table_position is None
        ]
        if uncertain:
            return (
                KittingSubgoal(
                    KittingSkill.REOBSERVE,
                    None,
                    None,
                    f"requested objects require a fresh observation: {','.join(uncertain)}",
                ),
            )

        # If the first requested target is constrained by another unfinished
        # requested object, complete that object first when its own destination
        # is available. This is a relation-derived prerequisite, not a fixed
        # object sequence.
        primary = unfinished[0]
        for neighbor in (
            *state.blocked_by.get(primary, ()),
            *state.nearby_risks.get(primary, ()),
        ):
            if neighbor not in unfinished:
                continue
            neighbor_destination = task.destinations[neighbor]
            occupant = state.destination_occupants.get(neighbor_destination)
            if occupant in {None, neighbor}:
                return (
                    KittingSubgoal(
                        KittingSkill.PLACE_TARGET,
                        neighbor,
                        neighbor_destination,
                        f"{neighbor} is a prerequisite for clearing {primary}'s approach",
                        target_object=neighbor,
                    ),
                )

        executable: list[tuple[int, int, str]] = []
        for request_index, target in enumerate(unfinished):
            destination = task.destinations[target]
            occupant = state.destination_occupants.get(destination)
            blockers = state.blocked_by.get(target, ())
            destination_available = occupant in {None, target}
            if destination_available and not blockers:
                executable.append(
                    (len(state.nearby_risks.get(target, ())), request_index, target)
                )
        if executable:
            candidates: list[KittingSubgoal] = []
            for _, _, target in sorted(executable):
                risks = state.nearby_risks.get(target, ())
                reason = (
                    f"target is executable with a constraint-aware route around {','.join(risks)}"
                    if risks
                    else "target is visible, approach is clear, and destination is available"
                )
                candidates.append(
                    KittingSubgoal(
                        KittingSkill.PLACE_TARGET,
                        target,
                        task.destinations[target],
                        reason,
                        target_object=target,
                    )
                )
            return tuple(candidates)

        target = unfinished[0]
        destination = task.destinations[target]
        occupant = state.destination_occupants.get(destination)
        if occupant not in {None, target}:
            if occupant in unfinished:
                return (
                    KittingSubgoal(
                        KittingSkill.PLACE_TARGET,
                        occupant,
                        task.destinations[occupant],
                        f"{occupant} must leave {destination} before {target}",
                        target_object=occupant,
                    ),
                )
            return (
                KittingSubgoal(
                    KittingSkill.CLEAR_TO_STAGING,
                    occupant,
                    task.staging_region,
                    f"{destination} is occupied by {occupant}",
                    target_object=target,
                ),
            )
        blockers = state.blocked_by.get(target, ())
        if blockers:
            blocker = self._choose_blocker(task, blockers)
            return (
                KittingSubgoal(
                    KittingSkill.CLEAR_TO_STAGING,
                    blocker,
                    task.staging_region,
                    f"{target} approach is blocked by {blocker}",
                    target_object=target,
                ),
            )
        return (
            KittingSubgoal(
                KittingSkill.SAFE_STOP,
                None,
                None,
                "no valid registered subgoal can be derived from the observed state",
                target_object=target,
            ),
        )

    @staticmethod
    def _choose_blocker(task: KittingTask, blockers: tuple[str, ...]) -> str:
        protected = [name for name in blockers if name in task.protected_objects]
        return protected[0] if protected else blockers[0]


def _planar_distance(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> float:
    if first is None or second is None:
        return float("inf")
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5
