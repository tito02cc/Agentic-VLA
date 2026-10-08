"""Bounded qpos14 recovery skills for RoboTwin Agentic experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from agentic_vla.toolchain import (
    PrimitiveExecutionReport,
    PrimitiveStatus,
    ToolExecutionContext,
)

from .robotwin_runtime import RoboTwinRuntimeAdapter


class RoboTwinRecoverySkillLibrary:
    """Generate collision-aware lift/reobserve recovery in the VLA action space."""

    _SKILLS = (
        "qpos_retract_lift_reobserve",
        "qpos_release_lift_reobserve",
    )

    def __init__(
        self,
        adapter: RoboTwinRuntimeAdapter,
        *,
        last_action_source: Callable[[], np.ndarray],
        on_step: Callable[[Any, np.ndarray], None] | None = None,
        lift_m: float = 0.08,
        max_waypoints: int = 12,
    ) -> None:
        if lift_m <= 0 or max_waypoints <= 0:
            raise ValueError("recovery lift and waypoint budget must be positive")
        self.adapter = adapter
        self.last_action_source = last_action_source
        self.on_step = on_step
        self.lift_m = float(lift_m)
        self.max_waypoints = int(max_waypoints)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return self._SKILLS

    @property
    def planner_specs(self) -> tuple[Mapping[str, Any], ...]:
        return (
            {
                "skill_id": self._SKILLS[0],
                "description": (
                    "Lift the arm that most recently moved and reobserve after a stall."
                ),
                "arguments": {},
                "postcondition": "active tool is lifted clear of the suspected contact",
            },
            {
                "skill_id": self._SKILLS[1],
                "description": (
                    "Release the active gripper, lift the arm, and reobserve after a failed grasp."
                ),
                "arguments": {},
                "postcondition": "failed grasp is released and the tool is lifted clear",
            },
        )

    def _active_arm(self, state: Mapping[str, Any]) -> str:
        current = np.asarray(state["qpos_target14"], dtype=np.float64).reshape(-1)
        previous = np.asarray(self.last_action_source(), dtype=np.float64).reshape(-1)
        if current.shape != (14,) or previous.shape != (14,):
            raise ValueError("RoboTwin recovery requires current and prior qpos14")
        left = float(np.linalg.norm(previous[:7] - current[:7]))
        right = float(np.linalg.norm(previous[7:] - current[7:]))
        return "left" if left >= right else "right"

    def execute(
        self,
        skill_id: str,
        _arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> PrimitiveExecutionReport:
        if skill_id not in self._SKILLS:
            raise KeyError(f"unknown RoboTwin recovery skill: {skill_id}")
        state = self.adapter.robot_state()
        arm = self._active_arm(state)
        pose_key = f"{arm}_eef_pose"
        before_pose = np.asarray(state.get(pose_key), dtype=np.float64).reshape(-1)
        if before_pose.shape != (7,) or not np.isfinite(before_pose).all():
            raise ValueError(f"RoboTwin {pose_key} must contain seven finite values")
        target = before_pose.copy()
        target[2] += self.lift_m
        planned = self.adapter.plan_arm_path(arm, target)
        status = str(planned.get("status", ""))
        positions = planned.get("position")
        if status != "Success" or positions is None:
            return PrimitiveExecutionReport(
                status=PrimitiveStatus.FAILED,
                started_timestep=context.timestep,
                ended_timestep=context.timestep,
                expected_outcome="active tool is lifted clear for replanning",
                observed_outcome=f"RoboTwin recovery path planning failed: {status or 'unknown'}",
                requires_semantic_check=True,
                metadata={"skill_id": skill_id, "arm": arm, "plan_status": status},
            )

        arm_path = np.asarray(positions, dtype=np.float64)
        if arm_path.ndim != 2 or arm_path.shape[1] != 6 or len(arm_path) < 1:
            raise ValueError("RoboTwin recovery plan must have shape [N,6]")
        if len(arm_path) > self.max_waypoints:
            indices = np.linspace(0, len(arm_path) - 1, self.max_waypoints).astype(int)
            arm_path = arm_path[indices]

        qpos = np.asarray(state["qpos_target14"], dtype=np.float64).reshape(14)
        offset = 0 if arm == "left" else 7
        rows: list[np.ndarray] = []
        if skill_id == "qpos_release_lift_reobserve":
            for ratio in np.linspace(0.25, 1.0, 4):
                release = qpos.copy()
                release[offset + 6] = qpos[offset + 6] + ratio * (
                    1.0 - qpos[offset + 6]
                )
                rows.append(release)
                qpos = release
        for waypoint in arm_path:
            action = qpos.copy()
            action[offset : offset + 6] = waypoint
            rows.append(action)
            qpos = action

        execution = self.adapter.execute_action_chunk(
            np.asarray(rows, dtype=np.float32),
            context,
            on_step=self.on_step,
        )
        after_pose = np.asarray(
            self.adapter.robot_state()[pose_key], dtype=np.float64
        ).reshape(7)
        response_m = float(np.linalg.norm(after_pose[:3] - before_pose[:3]))
        moved = execution.metadata.get("executed_steps", 0) > 0 and response_m >= 0.02
        return PrimitiveExecutionReport(
            status=(PrimitiveStatus.SUCCEEDED if moved else PrimitiveStatus.FAILED),
            started_timestep=context.timestep,
            ended_timestep=execution.ended_timestep,
            expected_outcome="active tool is lifted clear for replanning",
            observed_outcome=f"{arm} tool displacement after recovery: {response_m:.4f} m",
            requires_semantic_check=True,
            metadata={
                "skill_id": skill_id,
                "arm": arm,
                "plan_status": status,
                "waypoints": len(rows),
                "physical_response_m": response_m,
            },
        )
