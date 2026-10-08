"""Observation-bounded Robosuite ToolHang environment adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .contracts import ToolHangEvaluation, ToolHangObservation


TransitionSink = Callable[
    [ToolHangObservation, np.ndarray, ToolHangObservation], None
]


class ToolHangEnvironment:
    """Multi-stage Panda task with separate control and evaluation APIs."""

    EXTERNAL_CAMERA = "agentview"
    FRONT_CAMERA = "frontview"
    WRIST_CAMERA = "robot0_eye_in_hand"

    def __init__(
        self,
        *,
        horizon: int = 1000,
        image_size: int = 256,
        control_freq: int = 20,
        seed: int | None = None,
    ) -> None:
        if horizon <= 0 or image_size <= 0 or control_freq <= 0:
            raise ValueError("horizon, image_size, and control_freq must be positive")

        import robosuite
        import robosuite as suite

        minor_version = int(robosuite.__version__.split(".")[1])
        if minor_version >= 5:
            from robosuite.controllers import load_composite_controller_config

            controller_configs = load_composite_controller_config(
                controller="BASIC"
            )
        else:
            from robosuite.controllers import load_controller_config

            controller_configs = load_controller_config(
                default_controller="OSC_POSE"
            )

        self._seed = seed
        self._env = suite.make(
            "ToolHang",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=[
                self.EXTERNAL_CAMERA,
                self.FRONT_CAMERA,
                self.WRIST_CAMERA,
            ],
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=True,
            render_camera=self.EXTERNAL_CAMERA,
            horizon=horizon,
            control_freq=control_freq,
            ignore_done=True,
            hard_reset=True,
        )
        self.image_size = image_size
        self.control_freq = control_freq
        self._transition_sink: TransitionSink | None = None
        self._last_observation: ToolHangObservation | None = None

    @property
    def action_dim(self) -> int:
        return int(self._env.action_dim)

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self._env.action_spec
        return np.asarray(low).copy(), np.asarray(high).copy()

    def reset(self, *, seed: int | None = None) -> ToolHangObservation:
        """Reset reproducibly without exposing the sampled object state."""
        reset_seed = self._seed if seed is None else seed
        if reset_seed is None:
            observation = self._env.reset()
        else:
            rng_state = np.random.get_state()
            try:
                np.random.seed(int(reset_seed))
                observation = self._env.reset()
            finally:
                np.random.set_state(rng_state)
        control_observation = self._control_observation(observation)
        self._last_observation = control_observation
        return control_observation

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[ToolHangObservation, float, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (self.action_dim,):
            raise ValueError(
                f"expected action shape {(self.action_dim,)}, got {command.shape}"
            )
        applied = np.clip(command, -1.0, 1.0)
        observation, reward, done, info = self._env.step(applied)
        control_observation = self._control_observation(observation)
        if self._transition_sink is not None:
            if self._last_observation is None:
                raise RuntimeError(
                    "transition recording requires an initial observation"
                )
            self._transition_sink(
                self._last_observation,
                applied.copy(),
                control_observation,
            )
        self._last_observation = control_observation
        return control_observation, float(reward), bool(done), dict(info)

    def set_transition_sink(self, sink: TransitionSink | None) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("transition sink must be callable")
        self._transition_sink = sink

    def evaluate(self) -> ToolHangEvaluation:
        """Return private task labels for scoring and artifact generation only."""
        frame_assembled = bool(self._env._check_frame_assembled())
        tool_hung = bool(self._env._check_tool_on_frame())
        return ToolHangEvaluation.from_checks(
            frame_assembled=frame_assembled,
            tool_hung=tool_hung,
        )

    def calibration(self) -> dict[str, np.ndarray]:
        """Return fixed RGB-D camera transforms for observation-side perception."""
        from robosuite.utils import camera_utils

        calibration: dict[str, np.ndarray] = {}
        for camera_name in (
            self.EXTERNAL_CAMERA,
            self.FRONT_CAMERA,
            self.WRIST_CAMERA,
        ):
            world_to_pixel = np.asarray(
                camera_utils.get_camera_transform_matrix(
                    self._env.sim,
                    camera_name,
                    self.image_size,
                    self.image_size,
                ),
                dtype=np.float64,
            )
            calibration[f"{camera_name}_world_to_pixel"] = world_to_pixel
            calibration[f"{camera_name}_pixel_to_world"] = np.linalg.inv(
                world_to_pixel
            )
        return calibration

    def render(self) -> np.ndarray:
        raw = np.asarray(
            self._env.sim.render(
                camera_name=self.EXTERNAL_CAMERA,
                height=self.image_size,
                width=self.image_size,
            ),
            dtype=np.uint8,
        )
        return np.ascontiguousarray(raw[::-1])

    def close(self) -> None:
        self._env.close()

    def _control_observation(
        self,
        observation: dict[str, Any],
    ) -> ToolHangObservation:
        required = (
            f"{self.EXTERNAL_CAMERA}_image",
            f"{self.EXTERNAL_CAMERA}_depth",
            f"{self.FRONT_CAMERA}_image",
            f"{self.FRONT_CAMERA}_depth",
            f"{self.WRIST_CAMERA}_image",
            f"{self.WRIST_CAMERA}_depth",
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"missing control observation fields: {missing}")

        from robosuite.utils import camera_utils

        def metric_depth(camera_name: str) -> np.ndarray:
            return camera_utils.get_real_depth_map(
                self._env.sim,
                np.asarray(
                    observation[f"{camera_name}_depth"],
                    dtype=np.float64,
                ),
            )

        eef_position = np.asarray(
            observation["robot0_eef_pos"],
            dtype=np.float64,
        )
        eef_quaternion = np.asarray(
            observation["robot0_eef_quat"],
            dtype=np.float64,
        )
        gripper = np.asarray(
            observation["robot0_gripper_qpos"],
            dtype=np.float64,
        )
        policy_state = np.concatenate(
            [eef_position, eef_quaternion, gripper]
        )
        return ToolHangObservation(
            external_rgb=np.asarray(
                observation[f"{self.EXTERNAL_CAMERA}_image"]
            ),
            external_depth=metric_depth(self.EXTERNAL_CAMERA),
            wrist_rgb=np.asarray(observation[f"{self.WRIST_CAMERA}_image"]),
            wrist_depth=metric_depth(self.WRIST_CAMERA),
            proprio=policy_state,
            gripper=gripper,
            eef_position=eef_position,
            eef_quaternion=eef_quaternion,
            front_rgb=np.asarray(observation[f"{self.FRONT_CAMERA}_image"]),
            front_depth=metric_depth(self.FRONT_CAMERA),
        )
