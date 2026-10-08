"""Thin official-LIBERO adapter for the canonical CARVE runtime."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from agentic_vla.toolchain import ToolExecutionContext
from agentic_vla.toolchain.vla import ActionExecutionReport

from .contracts import DeployableBenchmarkObservation, PrivateBenchmarkResult


ImageTransform = Callable[[np.ndarray], np.ndarray]
ActionStepCallback = Callable[[DeployableBenchmarkObservation, np.ndarray], None]


def quaternion_to_axis_angle(quaternion: Any) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1).copy()
    if quat.shape != (4,):
        raise ValueError("LIBERO end-effector quaternion must contain four values")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


class LiberoRuntimeAdapter:
    """Own LIBERO reset/step/evaluation without leaking simulator truth."""

    def __init__(
        self,
        env: Any,
        *,
        task_instruction: str,
        initial_state: Any,
        image_transform: ImageTransform | None = None,
        max_steps: int = 600,
        settle_steps: int = 0,
        settle_action: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        restore_state: Any | None = None,
    ) -> None:
        if not task_instruction.strip() or max_steps <= 0 or settle_steps < 0:
            raise ValueError("task_instruction and max_steps must be valid")
        if not hasattr(env, "reset") or not hasattr(env, "set_init_state"):
            raise TypeError("LIBERO environment must expose reset and set_init_state")
        if not hasattr(env, "step"):
            raise TypeError("LIBERO environment must expose step")
        self.env = env
        self.task_instruction = task_instruction
        self.initial_state = initial_state
        self.image_transform = image_transform or (lambda image: image)
        self.max_steps = int(max_steps)
        self.settle_steps = int(settle_steps)
        self.settle_action = tuple(float(value) for value in settle_action)
        self.restore_state = (
            None
            if restore_state is None
            else np.asarray(restore_state, dtype=np.float64).reshape(-1).copy()
        )
        if not self.settle_action:
            raise ValueError("settle_action must not be empty")
        if self.restore_state is not None and not hasattr(
            env, "regenerate_obs_from_state"
        ):
            raise TypeError(
                "LIBERO snapshot restoration requires regenerate_obs_from_state"
            )
        self._observation: Mapping[str, Any] | None = None
        self._timestep = 0
        self._terminated = False

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def terminated(self) -> bool:
        return self._terminated

    def reset(self) -> DeployableBenchmarkObservation:
        self.env.reset()
        if self.restore_state is None:
            observation = self.env.set_init_state(self.initial_state)
        else:
            observation = self.env.regenerate_obs_from_state(
                self.restore_state.copy()
            )
        if not isinstance(observation, Mapping):
            raise TypeError("LIBERO set_init_state must return an observation mapping")
        for _ in range(self.settle_steps):
            observation, _reward, done, _info = self.env.step(self.settle_action)
            if not isinstance(observation, Mapping):
                raise TypeError("LIBERO settle step must return an observation mapping")
            if done:
                break
        self._observation = observation
        self._timestep = 0
        self._terminated = False
        return self.observe()

    def observe(self) -> DeployableBenchmarkObservation:
        if self._observation is None:
            raise RuntimeError("reset must be called before observe")
        observation = self._observation
        required = (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        missing = [name for name in required if name not in observation]
        if missing:
            raise KeyError(f"LIBERO observation is missing fields: {missing}")
        base = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1])
        wrist = np.ascontiguousarray(
            np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1]
        )
        base_policy = np.asarray(self.image_transform(base))
        wrist_policy = np.asarray(self.image_transform(wrist))
        robot_state = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                quaternion_to_axis_angle(observation["robot0_eef_quat"]).astype(
                    np.float32
                ),
                np.asarray(
                    observation["robot0_gripper_qpos"], dtype=np.float32
                ).reshape(-1),
            )
        )
        return DeployableBenchmarkObservation(
            policy_observation={
                "observation/image": base_policy,
                "observation/wrist_image": wrist_policy,
                "observation/state": robot_state,
            },
            planner_frames={"agentview": base, "wrist": wrist},
            robot_state=tuple(float(value) for value in robot_state),
            timestep=self._timestep,
            frame_id=f"libero:{self._timestep}",
        )

    @property
    def depth_available(self) -> bool:
        """Whether the current deployable observation contains calibrated RGB-D."""

        return bool(
            self._observation is not None
            and "agentview_depth" in self._observation
            and getattr(self.env, "sim", None) is not None
        )

    def planner_pixel_to_world(
        self,
        *,
        u: float,
        v: float,
        camera_name: str = "agentview",
        patch_radius: int = 3,
    ) -> np.ndarray:
        """Back-project a normalized planner-image pixel using deployable RGB-D.

        The planner sees the same 180-degree image normalization as the frozen VLA.
        This method reverses that normalization before sampling the simulator depth
        image. It exposes calibrated perception, never object poses or evaluator state.
        """

        if self._observation is None:
            raise RuntimeError("reset must be called before RGB-D back-projection")
        if not 0.0 <= float(u) <= 1.0 or not 0.0 <= float(v) <= 1.0:
            raise ValueError("normalized image coordinates must be in [0, 1]")
        if patch_radius < 0:
            raise ValueError("patch_radius must be non-negative")
        depth_key = (
            "agentview_depth"
            if camera_name == "agentview"
            else f"robot0_{camera_name}_depth"
        )
        if depth_key not in self._observation:
            raise RuntimeError(f"camera depth is unavailable: {depth_key}")
        sim = getattr(self.env, "sim", None)
        if sim is None:
            raise RuntimeError("LIBERO environment does not expose camera calibration")

        from robosuite.utils import camera_utils

        raw_depth = np.asarray(self._observation[depth_key], dtype=np.float64)
        if raw_depth.ndim == 2:
            raw_depth = raw_depth[..., None]
        if raw_depth.ndim != 3 or raw_depth.shape[-1] != 1:
            raise ValueError("LIBERO camera depth must have shape [H, W, 1]")
        metric_depth = camera_utils.get_real_depth_map(sim, raw_depth)
        height, width = metric_depth.shape[:2]
        planner_col = int(round(float(u) * (width - 1)))
        planner_row = int(round(float(v) * (height - 1)))
        row = height - 1 - planner_row
        col = width - 1 - planner_col
        radius = int(patch_radius)
        patch = metric_depth[
            max(0, row - radius) : min(height, row + radius + 1),
            max(0, col - radius) : min(width, col + radius + 1),
            0,
        ]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            raise RuntimeError("selected image pixel has no valid metric depth")
        z = float(np.median(valid))
        world_to_pixel = np.asarray(
            camera_utils.get_camera_transform_matrix(
                sim, camera_name, height, width
            ),
            dtype=np.float64,
        )
        pixel_to_world = np.linalg.inv(world_to_pixel)
        point = pixel_to_world @ np.asarray(
            [float(col) * z, float(row) * z, z, 1.0], dtype=np.float64
        )
        if not np.all(np.isfinite(point[:3])):
            raise RuntimeError("RGB-D back-projection produced a non-finite point")
        return point[:3].copy()

    def policy_observation(self, _context: ToolExecutionContext) -> Mapping[str, Any]:
        return self.observe().policy_observation

    def execute_action_chunk(
        self,
        actions: Any,
        context: ToolExecutionContext,
        *,
        on_step: ActionStepCallback | None = None,
    ) -> ActionExecutionReport:
        if self._observation is None:
            raise RuntimeError("reset must be called before action execution")
        if context.timestep != self._timestep:
            raise ValueError("tool context timestep is stale for LIBERO")
        try:
            action_rows: Sequence[Any] = list(actions)
        except TypeError as exc:
            raise TypeError("VLA action chunk must be iterable") from exc
        if not action_rows:
            raise ValueError("VLA action chunk must not be empty")
        executed = 0
        for raw_action in action_rows:
            if self._terminated or self._timestep >= self.max_steps:
                break
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(action)):
                raise ValueError("VLA action contains non-finite values")
            observation, _reward, done, _info = self.env.step(action.tolist())
            if not isinstance(observation, Mapping):
                raise TypeError("LIBERO step must return an observation mapping")
            self._observation = observation
            self._timestep += 1
            executed += 1
            self._terminated = bool(done)
            if on_step is not None:
                on_step(self.observe(), action.copy())
        if executed <= 0:
            return ActionExecutionReport(
                status="interrupted",
                ended_timestep=self._timestep,
                observed_outcome="no action was executed",
                requires_semantic_check=True,
                metadata={"executed_steps": 0},
            )
        return ActionExecutionReport(
            status="succeeded",
            ended_timestep=self._timestep,
            observed_outcome="action chunk executed",
            requires_semantic_check=False,
            metadata={
                "executed_steps": executed,
            },
        )

    def private_result(self) -> PrivateBenchmarkResult:
        """Return evaluator truth only after or outside planner decisions."""

        return PrivateBenchmarkResult(
            task_success=self._terminated,
            episode_steps=self._timestep,
            terminated=self._terminated,
            evaluator_metadata={"source": "libero_done"},
        )

    def render_video_frame(
        self,
        *,
        camera_name: str = "agentview",
        size: int = 512,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("video frame size must be positive")
        sim = getattr(self.env, "sim", None)
        if sim is None or not hasattr(sim, "render"):
            return np.asarray(self.observe().planner_frames["agentview"])
        frame = sim.render(camera_name=camera_name, height=size, width=size)
        return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[::-1])

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
