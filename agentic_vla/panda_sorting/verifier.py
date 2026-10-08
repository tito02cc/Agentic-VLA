"""Control-side visual checks used to trigger bounded recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import SceneObject


@dataclass(frozen=True)
class LiftVerification:
    """Result of checking whether an object left its observed source location."""

    held: bool
    reason: str
    lateral_displacement_m: float | None
    height_increase_m: float | None
    confidence: float


@dataclass(frozen=True)
class PlacementVerification:
    """Observation-side evidence that an object is inside its assigned fixture."""

    placed: bool
    reason: str
    distance_to_destination_m: float | None
    confidence: float


class VisualLiftVerifier:
    """Verify a grasp from a re-observed object's lateral displacement.

    A staged waypoint is selected before this verifier runs, so a grasped
    object should be visible at a new tabletop projection.  The verifier uses
    only RGB-D perception estimates; it intentionally does not consume contact
    flags, object pose, or any simulator reward.
    """

    def __init__(
        self,
        min_lateral_displacement_m: float = 0.035,
        min_height_increase_m: float = 0.035,
    ) -> None:
        if min_lateral_displacement_m <= 0.0 or min_height_increase_m <= 0.0:
            raise ValueError("visual lift thresholds must be positive")
        self.min_lateral_displacement_m = float(min_lateral_displacement_m)
        self.min_height_increase_m = float(min_height_increase_m)

    def verify(
        self,
        *,
        source_xy: tuple[float, float],
        source_height_m: float | None,
        reobserved: SceneObject | None,
    ) -> LiftVerification:
        if reobserved is None or not reobserved.visible or reobserved.table_position is None:
            return LiftVerification(False, "target_not_visually_localized", None, None, 0.0)
        displacement = float(np.linalg.norm(np.asarray(reobserved.table_position) - np.asarray(source_xy)))
        reobserved_height = None if reobserved.world_position is None else reobserved.world_position[2]
        height_increase = (
            None
            if source_height_m is None or reobserved_height is None
            else float(reobserved_height - source_height_m)
        )
        held = (
            displacement >= self.min_lateral_displacement_m
            and height_increase is not None
            and height_increase >= self.min_height_increase_m
        )
        if held:
            reason = "object_lifted_and_moved_with_gripper"
        elif displacement >= self.min_lateral_displacement_m:
            reason = "object_moved_without_visual_lift"
        else:
            reason = "object_remains_near_source"
        return LiftVerification(
            held=held,
            reason=reason,
            lateral_displacement_m=displacement,
            height_increase_m=height_increase,
            confidence=float(reobserved.confidence),
        )


class VisualPlacementVerifier:
    """Check destination membership from calibrated RGB-D and fixture geometry."""

    def __init__(
        self,
        destination_radius_m: float = 0.085,
        *,
        destination_half_extents_m: tuple[float, float] | None = None,
    ) -> None:
        if destination_radius_m <= 0.0:
            raise ValueError("destination_radius_m must be positive")
        if destination_half_extents_m is not None and any(
            value <= 0.0 for value in destination_half_extents_m
        ):
            raise ValueError("destination_half_extents_m must be positive")
        self.destination_radius_m = float(destination_radius_m)
        self.destination_half_extents_m = destination_half_extents_m

    def verify(
        self,
        *,
        destination_xyz: tuple[float, float, float],
        reobserved: SceneObject | None,
    ) -> PlacementVerification:
        if reobserved is None or not reobserved.visible or reobserved.table_position is None:
            return PlacementVerification(False, "target_not_visually_localized", None, 0.0)
        distance = float(
            np.linalg.norm(
                np.asarray(reobserved.table_position)
                - np.asarray(destination_xyz[:2])
            )
        )
        delta = np.abs(
            np.asarray(reobserved.table_position)
            - np.asarray(destination_xyz[:2])
        )
        placed = (
            bool(
                delta[0] <= self.destination_half_extents_m[0]
                and delta[1] <= self.destination_half_extents_m[1]
            )
            if self.destination_half_extents_m is not None
            else distance <= self.destination_radius_m
        )
        return PlacementVerification(
            placed=placed,
            reason="object_inside_destination" if placed else "object_outside_destination",
            distance_to_destination_m=distance,
            confidence=float(reobserved.confidence),
        )
