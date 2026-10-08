"""Deployable analytic skill library for the CARVE LIBERO adapter.

The high-level planner selects bounded symbolic skills. This module owns the
controller math and never exposes raw actions, simulator object poses, or task
evaluator state to the planner.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from agentic_vla.toolchain import (
    PrimitiveExecutionReport,
    PrimitiveStatus,
    ToolExecutionContext,
)

from .contracts import DeployableBenchmarkObservation
from .libero_runtime import ActionStepCallback, LiberoRuntimeAdapter


LastActionSource = Callable[[], Sequence[float]]


@dataclasses.dataclass(frozen=True)
class EmbodiedSkillSpec:
    skill_id: str
    description: str
    arguments: Mapping[str, Any]
    postcondition: str

    def __post_init__(self) -> None:
        if not self.skill_id.strip() or not self.description.strip():
            raise ValueError("embodied skill identity and description must not be empty")
        if not self.postcondition.strip() or not isinstance(self.arguments, Mapping):
            raise ValueError("embodied skill arguments and postcondition must be valid")

    def to_planner_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "arguments": dict(self.arguments),
            "postcondition": self.postcondition,
        }


def libero_embodied_skill_specs() -> tuple[EmbodiedSkillSpec, ...]:
    """Stable planner-facing vocabulary for deterministic LIBERO skills."""

    return (
        EmbodiedSkillSpec(
            "visual_servo_above",
            "Move the end effector above a visible point selected in agentview RGB.",
            {
                "u": "number [0,1], left to right",
                "v": "number [0,1], top to bottom",
                "approach_height_m": "number [0.03,0.20]",
                "gripper": "keep | open | close",
            },
            "end effector reaches the RGB-D grounded approach point",
        ),
        EmbodiedSkillSpec(
            "move_relative",
            "Move the end effector by a small robot-base-frame Cartesian offset.",
            {
                "delta_xyz_m": "three numbers, each in [-0.20,0.20]",
                "gripper": "keep | open | close",
            },
            "end effector reaches the bounded relative target",
        ),
        EmbodiedSkillSpec(
            "rotate_wrist",
            "Rotate the wrist about the tool z axis while holding position.",
            {"delta_rad": "number in [-1.57,1.57]", "gripper": "keep | open | close"},
            "requested bounded wrist rotation is executed",
        ),
        EmbodiedSkillSpec(
            "rotate_pitch",
            "Tilt the tool about the robot-base x axis while holding position.",
            {"delta_rad": "number in [-1.20,1.20]", "gripper": "keep | open | close"},
            "requested bounded tool tilt is executed",
        ),
        EmbodiedSkillSpec(
            "set_gripper",
            "Open or close the gripper without moving the arm.",
            {"state": "open | close", "steps": "integer [3,20]"},
            "gripper command is held for the requested bounded duration",
        ),
        EmbodiedSkillSpec(
            "release_and_lift",
            "Release the held object, then lift vertically to clear contact.",
            {"lift_m": "number [0.03,0.15]"},
            "gripper is open and the tool has cleared the release pose",
        ),
    )


class LiberoEmbodiedSkillLibrary:
    """Closed-loop OSC skills sharing CARVE's action and audit boundary."""

    def __init__(
        self,
        adapter: LiberoRuntimeAdapter,
        *,
        last_action_source: LastActionSource,
        on_step: ActionStepCallback | None = None,
        action_scale_m: float = 0.05,
    ) -> None:
        if action_scale_m <= 0:
            raise ValueError("action_scale_m must be positive")
        self.adapter = adapter
        self.last_action_source = last_action_source
        self.on_step = on_step
        self.action_scale_m = float(action_scale_m)
        self.specs = libero_embodied_skill_specs()
        self._spec_by_id = {spec.skill_id: spec for spec in self.specs}

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(spec.skill_id for spec in self.specs)

    @property
    def planner_specs(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(spec.to_planner_dict() for spec in self.specs)

    def execute(
        self,
        skill_id: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> PrimitiveExecutionReport:
        normalized = str(skill_id).strip()
        if normalized not in self._spec_by_id:
            raise ValueError(f"unregistered LIBERO embodied skill: {normalized}")
        if not isinstance(arguments, Mapping):
            raise TypeError("embodied skill arguments must be a mapping")
        started = self.adapter.timestep
        if context.timestep != started:
            raise ValueError("embodied skill context is stale")

        if normalized == "visual_servo_above":
            report = self._visual_servo_above(arguments, context)
        elif normalized == "move_relative":
            report = self._move_relative(arguments, context)
        elif normalized in {"rotate_wrist", "rotate_pitch"}:
            report = self._rotate(normalized, arguments, context)
        elif normalized == "set_gripper":
            report = self._set_gripper(arguments, context)
        else:
            report = self._release_and_lift(arguments, context)
        return PrimitiveExecutionReport(
            status=report["status"],
            started_timestep=started,
            ended_timestep=self.adapter.timestep,
            observed_outcome=str(report["observed_outcome"]),
            requires_semantic_check=True,
            metadata={
                "skill_id": normalized,
                "controller": "closed_loop_osc",
                **dict(report.get("metadata", {})),
            },
        )

    def _visual_servo_above(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        u = self._bounded_float(arguments, "u", 0.0, 1.0)
        v = self._bounded_float(arguments, "v", 0.0, 1.0)
        height = self._bounded_float(arguments, "approach_height_m", 0.03, 0.20)
        point = self.adapter.planner_pixel_to_world(u=u, v=v)
        target = point.copy()
        target[2] += height
        if not (-1.0 <= target[0] <= 1.0 and -1.0 <= target[1] <= 1.0 and 0.4 <= target[2] <= 1.8):
            raise ValueError("RGB-D target lies outside the conservative robot workspace")
        gripper = self._gripper(arguments.get("gripper", "keep"))
        current = self._eef_position()
        clearance = max(float(current[2]), float(target[2]) + 0.06)
        waypoints = (
            np.asarray([current[0], current[1], clearance], dtype=np.float64),
            np.asarray([target[0], target[1], clearance], dtype=np.float64),
            target,
        )
        final_error = math.inf
        for waypoint in waypoints:
            final_error = self._servo_xyz(waypoint, gripper=gripper, context=context)
            if self.adapter.terminated:
                break
        succeeded = final_error <= 0.018
        return {
            "status": PrimitiveStatus.SUCCEEDED if succeeded else PrimitiveStatus.FAILED,
            "observed_outcome": f"RGB-D visual servo final error {final_error:.4f} m",
            "metadata": {
                "grounding_source": "agentview_rgbd",
                "normalized_pixel": [u, v],
                "target_xyz": [float(value) for value in target],
                "final_error_m": final_error,
            },
        }

    def _move_relative(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        delta = self._xyz(arguments.get("delta_xyz_m"), limit=0.20)
        target = self._eef_position() + delta
        gripper = self._gripper(arguments.get("gripper", "keep"))
        error = self._servo_xyz(target, gripper=gripper, context=context)
        succeeded = error <= 0.018
        return {
            "status": PrimitiveStatus.SUCCEEDED if succeeded else PrimitiveStatus.FAILED,
            "observed_outcome": f"relative Cartesian servo final error {error:.4f} m",
            "metadata": {
                "delta_xyz_m": [float(value) for value in delta],
                "final_error_m": error,
            },
        }

    def _rotate(
        self,
        skill_id: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> Mapping[str, Any]:
        limit = 1.57 if skill_id == "rotate_wrist" else 1.20
        delta = self._bounded_float(arguments, "delta_rad", -limit, limit)
        gripper = self._gripper(arguments.get("gripper", "keep"))
        axis = 5 if skill_id == "rotate_wrist" else 3
        remaining = delta
        steps = 0
        while abs(remaining) > 1e-4 and steps < 20 and not self.adapter.terminated:
            increment = float(np.clip(remaining, -0.10, 0.10))
            action = np.zeros(7, dtype=np.float32)
            action[axis] = increment / 0.10
            action[6] = gripper
            self._step(action, context)
            remaining -= increment
            steps += 1
        return {
            "status": PrimitiveStatus.SUCCEEDED,
            "observed_outcome": f"executed {delta - remaining:.3f} rad bounded rotation",
            "metadata": {"requested_delta_rad": delta, "steps": steps},
        }

    def _set_gripper(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        state = str(arguments.get("state", "")).strip().lower()
        if state not in {"open", "close"}:
            raise ValueError("set_gripper state must be open or close")
        steps = int(arguments.get("steps", 8))
        if not 3 <= steps <= 20:
            raise ValueError("set_gripper steps must be in [3, 20]")
        gripper = 1.0 if state == "open" else -1.0
        executed = 0
        for _ in range(steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = gripper
            self._step(action, context)
            executed += 1
            if self.adapter.terminated:
                break
        return {
            "status": PrimitiveStatus.SUCCEEDED,
            "observed_outcome": f"held gripper {state} command for {executed} steps",
            "metadata": {"gripper_state": state, "steps": executed},
        }

    def _release_and_lift(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        lift = self._bounded_float(arguments, "lift_m", 0.03, 0.15)
        self._set_gripper({"state": "open", "steps": 8}, context)
        target = self._eef_position() + np.asarray([0.0, 0.0, lift])
        error = self._servo_xyz(target, gripper=1.0, context=context)
        succeeded = error <= 0.018
        return {
            "status": PrimitiveStatus.SUCCEEDED if succeeded else PrimitiveStatus.FAILED,
            "observed_outcome": f"released and lifted with final error {error:.4f} m",
            "metadata": {"lift_m": lift, "final_error_m": error},
        }

    def _servo_xyz(
        self,
        target: np.ndarray,
        *,
        gripper: float,
        context: ToolExecutionContext,
        max_steps: int = 80,
        tolerance_m: float = 0.012,
    ) -> float:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        error = math.inf
        for _ in range(max_steps):
            current = self._eef_position()
            delta = target - current
            error = float(np.linalg.norm(delta))
            if error <= tolerance_m or self.adapter.terminated:
                break
            action = np.zeros(7, dtype=np.float32)
            step_delta = np.clip(delta, -0.025, 0.025)
            action[:3] = np.clip(step_delta / self.action_scale_m, -1.0, 1.0)
            action[6] = gripper
            self._step(action, context)
        return float(np.linalg.norm(target - self._eef_position()))

    def _step(self, action: np.ndarray, context: ToolExecutionContext) -> None:
        current = dataclasses.replace(context, timestep=self.adapter.timestep)
        self.adapter.execute_action_chunk((action,), current, on_step=self.on_step)

    def _eef_position(self) -> np.ndarray:
        return np.asarray(self.adapter.observe().robot_state[:3], dtype=np.float64)

    def _gripper(self, value: Any) -> float:
        state = str(value).strip().lower()
        if state == "open":
            return 1.0
        if state == "close":
            return -1.0
        if state != "keep":
            raise ValueError("gripper must be keep, open, or close")
        action = np.asarray(tuple(self.last_action_source()), dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError("last action source must return one 7-D action")
        return float(np.clip(action[-1], -1.0, 1.0))

    @staticmethod
    def _bounded_float(
        values: Mapping[str, Any], name: str, minimum: float, maximum: float
    ) -> float:
        value = float(values.get(name))
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        return value

    @staticmethod
    def _xyz(value: Any, *, limit: float) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("delta_xyz_m must contain three finite values")
        if np.any(np.abs(array) > limit):
            raise ValueError(f"delta_xyz_m values must be in [-{limit}, {limit}]")
        return array

