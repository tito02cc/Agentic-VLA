"""Concrete RoboDojo/XPolicyLab port; no privileged task state crosses it."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .robodojo_sorting import SortingObservation


class XPolicyLabPi05Port:
    def __init__(self, environment, model_client):
        from XPolicyLab.utils.process_data import get_robot_action_dim_info
        self.environment = environment
        self.client = model_client
        self.dimensions = get_robot_action_dim_info("arx_x5")
        self._observation = None
        self.observation_reads = 0

    def observe(self):
        from XPolicyLab.utils.process_data import decode_obs_images, pack_robot_state
        if self._observation is not None:
            return self._observation
        raw = self.environment.get_obs()
        self.observation_reads += 1
        # Decode only public RGB fields with the same helper as the policy server.
        raw = decode_obs_images({"vision": raw["vision"], "state": raw["state"],
                                 "instruction": raw["instruction"]})
        frames = {}
        aliases = {
            "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
            "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
            "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
        }
        for name, candidates in aliases.items():
            value = next((raw["vision"][key] for key in candidates if key in raw["vision"]), None)
            if isinstance(value, Mapping):
                value = value.get("color", value.get("rgb"))
            if value is None:
                raise ValueError(f"missing camera {name}")
            frames[name] = np.array(value, copy=True)
            frames[name].setflags(write=False)
        state = pack_robot_state(raw, "joint", self.dimensions, source_type="obs")
        state = np.array(state, dtype=np.float32, copy=True)
        state.setflags(write=False)
        self._observation = SortingObservation(raw["instruction"], state, frames)
        return self._observation

    def infer(self, instruction):
        from XPolicyLab.utils.process_data import pack_robot_state
        current = self.observe()
        # Use the native Pi_05 public images/state branch, not a StarVLA payload.
        payload = {"images": dict(current.frames), "state": current.state, "instruction": instruction}
        self.client.call(func_name="update_obs", obs=payload)
        actions = self.client.call(func_name="get_action")
        return np.stack([pack_robot_state({"state": action}, "joint", self.dimensions,
                                          source_type="obs") for action in actions])

    def execute(self, absolute_joint_target):
        from XPolicyLab.utils.process_data import unpack_robot_state
        target = np.asarray(absolute_joint_target)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError("invalid ARX X5 target")
        # This synchronous simulator advances only through take_action. Even a
        # partially failed action invalidates the shared observation snapshot.
        self._observation = None
        self.environment.take_action(unpack_robot_state(target, "joint", self.dimensions, source_type="obs"))

    def done(self):
        return self.environment.is_episode_end()

    def audit_robot_kinematics(self):
        # Diagnostic-only host calibration; never included in model inputs.
        from .robodojo_kinematics import audit_robot_kinematics
        return audit_robot_kinematics(self.environment, self.observe().state)
