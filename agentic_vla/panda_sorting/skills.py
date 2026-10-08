"""Closed-loop Panda Cartesian primitives for the sorting demonstration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .environment import PandaSortingEnvironment, PandaSortingObservation


FrameSink = Callable[[np.ndarray], None]


@dataclass(frozen=True)
class SkillConfig:
    position_tolerance_m: float = 0.010
    proportional_gain: float = 10.0
    max_steps_per_move: int = 140
    close_hold_steps: int = 45
    open_hold_steps: int = 30
    approach_height_m: float = 0.16
    lift_height_m: float = 0.24
    grasp_offset_m: float = 0.0
    place_offset_m: float = 0.015


@dataclass
class SkillTrace:
    name: str
    phases: list[dict[str, float | int | str]] = field(default_factory=list)
    action_count: int = 0
    reached: bool = True

    def record(self, phase: str, steps: int, final_error_m: float) -> None:
        self.phases.append(
            {"phase": phase, "steps": int(steps), "final_error_m": float(final_error_m)}
        )
        self.action_count += int(steps)
        self.reached = self.reached and final_error_m >= 0.0


class PandaCartesianSkills:
    """Observation-driven OSC primitives with no object-state access.

    The executor uses estimated target XY and task-level geometry priors for
    object height / destination fixture. Its feedback is Panda EEF pose and
    gripper state from the public control observation.
    """

    OPEN = -1.0
    CLOSE = 1.0

    def __init__(self, env: PandaSortingEnvironment, config: SkillConfig | None = None) -> None:
        self.env = env
        self.config = config or SkillConfig()

    def pick_and_place(
        self,
        observation: PandaSortingObservation,
        *,
        target_xy: tuple[float, float],
        object_surface_z: float,
        destination_xyz: tuple[float, float, float],
        frame_sink: FrameSink | None = None,
        grasp_offset_m: float | None = None,
    ) -> tuple[PandaSortingObservation, SkillTrace]:
        """Run approach, grasp, lift, transport, place, and retract phases."""
        trace = SkillTrace(name="pick_and_place")
        offset = self.config.grasp_offset_m if grasp_offset_m is None else float(grasp_offset_m)
        target = np.asarray([target_xy[0], target_xy[1], object_surface_z + offset], dtype=np.float64)
        destination = np.asarray(destination_xyz, dtype=np.float64)

        observation = self._move(observation, target + [0.0, 0.0, self.config.approach_height_m], self.OPEN, "approach", trace, frame_sink)
        observation = self._move(observation, target, self.OPEN, "descend", trace, frame_sink)
        observation = self._hold(observation, target, self.CLOSE, self.config.close_hold_steps, "close", trace, frame_sink)
        observation = self._move(observation, target + [0.0, 0.0, self.config.lift_height_m], self.CLOSE, "lift", trace, frame_sink)
        observation = self._move(observation, destination + [0.0, 0.0, self.config.lift_height_m], self.CLOSE, "transport", trace, frame_sink)
        observation = self._move(observation, destination + [0.0, 0.0, self.config.place_offset_m], self.CLOSE, "place", trace, frame_sink)
        observation = self._hold(observation, destination + [0.0, 0.0, self.config.place_offset_m], self.OPEN, self.config.open_hold_steps, "release", trace, frame_sink)
        observation = self._move(observation, destination + [0.0, 0.0, self.config.lift_height_m], self.OPEN, "retract", trace, frame_sink)
        return observation, trace

    def grasp_and_stage(
        self,
        observation: PandaSortingObservation,
        *,
        target_xy: tuple[float, float],
        object_surface_z: float,
        staging_xy: tuple[float, float],
        frame_sink: FrameSink | None = None,
        grasp_offset_m: float | None = None,
    ) -> tuple[PandaSortingObservation, SkillTrace]:
        """Close on a perceived object and move to a visible verification pose.

        ``staging_xy`` is deliberately a lateral tabletop waypoint instead of
        the final destination.  A monitor can therefore determine whether the
        object moved with the gripper from a fresh external RGB-D frame, before
        any placement hides the evidence.  This method has no access to object
        pose, contact state, or task-success labels.
        """
        trace = SkillTrace(name="grasp_and_stage")
        offset = self.config.grasp_offset_m if grasp_offset_m is None else float(grasp_offset_m)
        target = np.asarray([target_xy[0], target_xy[1], object_surface_z + offset], dtype=np.float64)
        stage = np.asarray([staging_xy[0], staging_xy[1], object_surface_z], dtype=np.float64)

        observation = self._move(
            observation,
            target + [0.0, 0.0, self.config.approach_height_m],
            self.OPEN,
            "approach",
            trace,
            frame_sink,
        )
        observation = self._move(observation, target, self.OPEN, "descend", trace, frame_sink)
        observation = self._hold(
            observation,
            target,
            self.CLOSE,
            self.config.close_hold_steps,
            "close",
            trace,
            frame_sink,
        )
        observation = self._move(
            observation,
            target + [0.0, 0.0, self.config.lift_height_m],
            self.CLOSE,
            "lift",
            trace,
            frame_sink,
        )
        observation = self._move(
            observation,
            stage + [0.0, 0.0, self.config.lift_height_m],
            self.CLOSE,
            "stage_for_visual_verification",
            trace,
            frame_sink,
        )
        return observation, trace

    def place_from_stage(
        self,
        observation: PandaSortingObservation,
        *,
        destination_xyz: tuple[float, float, float],
        frame_sink: FrameSink | None = None,
    ) -> tuple[PandaSortingObservation, SkillTrace]:
        """Place an already verified grasp into a task-level fixture pose."""
        trace = SkillTrace(name="place_from_stage")
        destination = np.asarray(destination_xyz, dtype=np.float64)
        observation = self._move(
            observation,
            destination + [0.0, 0.0, self.config.lift_height_m],
            self.CLOSE,
            "transport",
            trace,
            frame_sink,
        )
        observation = self._move(
            observation,
            destination + [0.0, 0.0, self.config.place_offset_m],
            self.CLOSE,
            "place",
            trace,
            frame_sink,
        )
        observation = self._hold(
            observation,
            destination + [0.0, 0.0, self.config.place_offset_m],
            self.OPEN,
            self.config.open_hold_steps,
            "release",
            trace,
            frame_sink,
        )
        observation = self._move(
            observation,
            destination + [0.0, 0.0, self.config.lift_height_m],
            self.OPEN,
            "retract",
            trace,
            frame_sink,
        )
        return observation, trace

    def release_for_retry(
        self,
        observation: PandaSortingObservation,
        *,
        frame_sink: FrameSink | None = None,
    ) -> tuple[PandaSortingObservation, SkillTrace]:
        """Open the gripper at the current safe pose before a bounded retry."""
        trace = SkillTrace(name="release_for_retry")
        observation = self._hold(
            observation,
            observation.eef_position.copy(),
            self.OPEN,
            self.config.open_hold_steps,
            "open_before_retry",
            trace,
            frame_sink,
        )
        return observation, trace

    def _move(
        self,
        observation: PandaSortingObservation,
        target: np.ndarray,
        gripper: float,
        phase: str,
        trace: SkillTrace,
        frame_sink: FrameSink | None,
    ) -> PandaSortingObservation:
        steps = 0
        while steps < self.config.max_steps_per_move:
            error = target - observation.eef_position
            if float(np.linalg.norm(error)) <= self.config.position_tolerance_m:
                break
            observation = self._step(observation, error, gripper, frame_sink)
            steps += 1
        trace.record(phase, steps, float(np.linalg.norm(target - observation.eef_position)))
        return observation

    def _hold(
        self,
        observation: PandaSortingObservation,
        target: np.ndarray,
        gripper: float,
        steps: int,
        phase: str,
        trace: SkillTrace,
        frame_sink: FrameSink | None,
    ) -> PandaSortingObservation:
        for _ in range(steps):
            observation = self._step(observation, target - observation.eef_position, gripper, frame_sink)
        trace.record(phase, steps, float(np.linalg.norm(target - observation.eef_position)))
        return observation

    def _step(
        self,
        observation: PandaSortingObservation,
        error: np.ndarray,
        gripper: float,
        frame_sink: FrameSink | None,
    ) -> PandaSortingObservation:
        action = np.zeros(self.env.action_dim, dtype=np.float64)
        action[:3] = np.clip(error * self.config.proportional_gain, -1.0, 1.0)
        action[-1] = float(gripper)
        next_observation, _, _, _ = self.env.step(action)
        if frame_sink is not None:
            frame_sink(next_observation.external_rgb)
        return next_observation
