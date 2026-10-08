"""Robosuite adapter that separates control observations from simulator evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PandaSortingObservation:
    """Only data exposed to the planner, monitor, and skill executor."""

    external_rgb: np.ndarray
    external_depth: np.ndarray
    wrist_rgb: np.ndarray
    proprio: np.ndarray
    gripper: np.ndarray
    eef_position: np.ndarray
    front_rgb: np.ndarray | None = None
    front_depth: np.ndarray | None = None


TransitionSink = Callable[
    [PandaSortingObservation, np.ndarray, PandaSortingObservation], None
]


class PandaSortingEnvironment:
    """Two-object Panda sorting environment with an observation-only control API.

    Private MuJoCo state is intentionally retained only for final evaluation and
    evaluator-controlled disturbance injection. It is not returned by ``reset``
    or ``step``.
    """

    EXTERNAL_CAMERA = "agentview"
    FRONT_CAMERA = "frontview"
    WRIST_CAMERA = "robot0_eye_in_hand"
    OBJECT_COLORS_RGB = {
        "Milk": (245, 245, 245),
        "Bread": (56, 170, 90),
        "Cereal": (228, 45, 210),
        "Can": (35, 215, 225),
    }
    # Fixed task-fixture coordinates, not object state. Cereal and Can correspond
    # to PickPlace's two right-bin compartments (object indices 2 and 3).
    DESTINATION_CENTERS = {
        "milk_bin": (0.0025, 0.1575, 0.84),
        "bread_bin": (0.1975, 0.1575, 0.84),
        "cereal_bin": (0.0025, 0.4025, 0.84),
        "can_bin": (0.1975, 0.4025, 0.84),
    }

    def __init__(
        self,
        *,
        horizon: int = 700,
        image_size: int = 256,
        control_freq: int = 20,
        seed: int | None = None,
    ) -> None:
        if horizon <= 0 or image_size <= 0 or control_freq <= 0:
            raise ValueError("horizon, image_size, and control_freq must be positive")
        import robosuite as suite
        from robosuite.controllers import load_controller_config

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._env = suite.make(
            "PickPlace",
            robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=[self.EXTERNAL_CAMERA, self.FRONT_CAMERA, self.WRIST_CAMERA],
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
        self.object_colors_rgb = dict(self.OBJECT_COLORS_RGB)
        self._evaluator_contact_events: list[dict[str, str]] = []
        self._transition_sink: TransitionSink | None = None
        self._last_control_observation: PandaSortingObservation | None = None

    @property
    def action_dim(self) -> int:
        return int(self._env.action_dim)

    def reset(self, *, seed: int | None = None) -> PandaSortingObservation:
        """Reset to a reproducible physical scene when a seed is configured.

        Robosuite's PickPlace placement sampler consumes NumPy's global random
        generator.  Preserve the caller's RNG state while seeding that sampler,
        so a frozen evaluation seed yields the same physical layout without
        coupling the controller to private object coordinates.
        """
        reset_seed = self._seed if seed is None else seed
        self._evaluator_contact_events = []
        if reset_seed is None:
            self._env.reset()
        else:
            rng_state = np.random.get_state()
            try:
                np.random.seed(int(reset_seed))
                self._env.reset()
            finally:
                np.random.set_state(rng_state)
        self._apply_visual_labels()
        # ``reset`` rendered the first observation before visual task labels were
        # applied, so request a fresh camera frame rather than returning stale RGB.
        observation = self._control_observation(
            self._env._get_observations(force_update=True)
        )
        self._last_control_observation = observation
        return observation

    def step(self, action: np.ndarray) -> tuple[PandaSortingObservation, float, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {command.shape}")
        applied = np.clip(command, -1.0, 1.0)
        observation, reward, done, info = self._env.step(applied)
        self._record_evaluator_contacts()
        control_observation = self._control_observation(observation)
        if self._transition_sink is not None:
            if self._last_control_observation is None:
                raise RuntimeError("transition recording requires an initial observation")
            self._transition_sink(
                self._last_control_observation,
                applied.copy(),
                control_observation,
            )
        self._last_control_observation = control_observation
        return control_observation, float(reward), bool(done), dict(info)

    def set_transition_sink(self, sink: TransitionSink | None) -> None:
        """Register an optional public-observation transition recorder."""
        if sink is not None and not callable(sink):
            raise TypeError("transition sink must be callable")
        self._transition_sink = sink

    def render(self) -> np.ndarray:
        """Return only the configured external RGB camera image for video recording."""
        return self.to_display_rgb(np.asarray(self._env.sim.render(
            camera_name=self.EXTERNAL_CAMERA,
            height=self.image_size,
            width=self.image_size,
        )))

    @staticmethod
    def to_display_rgb(raw_rgb: np.ndarray) -> np.ndarray:
        """Convert the vertically flipped OpenGL buffer into display-ready RGB."""
        image = np.asarray(raw_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("raw_rgb must have shape [H, W, 3]")
        return np.ascontiguousarray(image[::-1])

    def calibration(self) -> dict[str, np.ndarray]:
        """Return fixed external/front RGB-D calibration for control perception."""
        from robosuite.utils import camera_utils

        def matrices(camera_name: str) -> tuple[np.ndarray, np.ndarray]:
            world_to_pixel = np.asarray(
                camera_utils.get_camera_transform_matrix(
                    self._env.sim, camera_name, self.image_size, self.image_size
                ),
                dtype=np.float64,
            )
            return world_to_pixel, np.linalg.inv(world_to_pixel)

        external_world_to_pixel, external_pixel_to_world = matrices(self.EXTERNAL_CAMERA)
        front_world_to_pixel, front_pixel_to_world = matrices(self.FRONT_CAMERA)
        return {
            "world_to_pixel": external_world_to_pixel,
            "pixel_to_world": external_pixel_to_world,
            "front_world_to_pixel": front_world_to_pixel,
            "front_pixel_to_world": front_pixel_to_world,
        }

    def evaluate(self) -> dict[str, Any]:
        """Evaluator-only task label. This is never passed into planning code."""
        return {
            "success": bool(self._env._check_success()),
            "objects_in_correct_bins": [int(value) for value in self._env.objects_in_bins],
        }

    def evaluate_objects(self, object_names: tuple[str, ...]) -> dict[str, Any]:
        """Evaluator-only labels for a declared subset of the sorting task."""
        statuses: dict[str, bool] = {}
        for name in object_names:
            normalized = name.strip().title()
            if normalized not in self._env.obj_body_id:
                raise ValueError(f"unknown PickPlace object: {name}")
            index = self._env.obj_names.index(normalized)
            body_position = self._env.sim.data.body_xpos[self._env.obj_body_id[normalized]]
            statuses[normalized.lower()] = not bool(self._env.not_in_bin(body_position, index))
        return {"objects": statuses, "success": all(statuses.values())}

    def evaluator_contact_summary(self) -> dict[str, Any]:
        """Return object-object contacts recorded for scoring only.

        This history is deliberately excluded from observations and cannot be
        consumed by the planner, monitor, or low-level skill.
        """
        unique = sorted({(entry["first"], entry["second"]) for entry in self._evaluator_contact_events})
        return {
            "object_object_contact_count": len(self._evaluator_contact_events),
            "unique_object_object_contacts": [list(pair) for pair in unique],
        }

    def evaluate_kitting(
        self,
        requested_objects: tuple[str, ...],
        *,
        protected_objects: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Evaluator-only final-state and protected-contact labels."""
        destinations = self.evaluate_objects(requested_objects)
        protected = {name.strip().lower() for name in protected_objects}
        protected_contacts = [
            entry
            for entry in self._evaluator_contact_events
            if entry["first"] in protected or entry["second"] in protected
        ]
        return {
            "requested_destinations": destinations,
            "object_contact_summary": self.evaluator_contact_summary(),
            "protected_object_contact_count": len(protected_contacts),
            "protected_object_contacts": protected_contacts,
            "success": bool(destinations["success"] and not protected_contacts),
        }

    def inject_xy_displacement(self, object_name: str, dx: float, dy: float) -> dict[str, Any]:
        """Evaluator-controlled physics disturbance; excluded from control observations."""
        normalized = object_name.strip().title()
        if normalized not in {"Milk", "Bread", "Cereal", "Can"}:
            raise ValueError("object_name must be one of Milk, Bread, Cereal, or Can")
        joint_name = f"{normalized}_joint0"
        qpos = np.asarray(self._env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        if qpos.size < 3:
            raise RuntimeError(f"unexpected free-joint state for {joint_name}")
        qpos[0] += float(dx)
        qpos[1] += float(dy)
        self._env.sim.data.set_joint_qpos(joint_name, qpos)
        self._env.sim.forward()
        return {"event": "object_displacement", "object": normalized, "dx": float(dx), "dy": float(dy)}

    def configure_scene_layout(
        self,
        object_xy: dict[str, tuple[float, float]],
        *,
        object_yaw_rad: dict[str, float] | None = None,
    ) -> PandaSortingObservation:
        """Apply a declared physics-scene layout before a trial begins.

        This is experiment initialization, analogous to placing objects at
        marked table locations before a real-robot trial.  It returns a normal
        RGB-D/proprioception observation and does not expose the configured
        coordinates to any planner or skill.
        """
        yaw_by_object = object_yaw_rad or {}
        unknown_yaw = set(yaw_by_object) - set(object_xy)
        if unknown_yaw:
            raise ValueError(
                f"yaw specified for objects absent from layout: {sorted(unknown_yaw)}"
            )
        for object_name, xy in object_xy.items():
            normalized = object_name.strip().title()
            if normalized not in self._env.obj_body_id:
                raise ValueError(f"unknown PickPlace object: {object_name}")
            x, y = float(xy[0]), float(xy[1])
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("scene layout coordinates must be finite")
            joint_name = f"{normalized}_joint0"
            qpos = np.asarray(self._env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
            qpos[0], qpos[1] = x, y
            if object_name in yaw_by_object:
                yaw = float(yaw_by_object[object_name])
                if not np.isfinite(yaw):
                    raise ValueError("scene yaw must be finite")
                qpos[3:7] = np.asarray(
                    [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    dtype=np.float64,
                )
            self._env.sim.data.set_joint_qpos(joint_name, qpos)
        self._env.sim.forward()
        observation = self._control_observation(
            self._env._get_observations(force_update=True)
        )
        self._last_control_observation = observation
        return observation

    def set_visual_label_colors(self, overrides: dict[str, tuple[int, int, int]]) -> PandaSortingObservation:
        """Change only rendered task-label colors for a declared experiment."""
        for object_name, rgb in overrides.items():
            normalized = object_name.strip().title()
            if normalized not in self.OBJECT_COLORS_RGB:
                raise ValueError(f"unknown PickPlace object: {object_name}")
            if len(rgb) != 3 or any(not 0 <= int(channel) <= 255 for channel in rgb):
                raise ValueError("visual RGB values must be three integers in [0, 255]")
            self.object_colors_rgb[normalized] = tuple(int(channel) for channel in rgb)
        self._apply_visual_labels()
        observation = self._control_observation(
            self._env._get_observations(force_update=True)
        )
        self._last_control_observation = observation
        return observation

    def close(self) -> None:
        self._env.close()

    def _control_observation(self, observation: dict[str, Any]) -> PandaSortingObservation:
        required = (
            f"{self.EXTERNAL_CAMERA}_image",
            f"{self.EXTERNAL_CAMERA}_depth",
            f"{self.FRONT_CAMERA}_image",
            f"{self.FRONT_CAMERA}_depth",
            f"{self.WRIST_CAMERA}_image",
            "robot0_proprio-state",
            "robot0_gripper_qpos",
            "robot0_eef_pos",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"missing control observation fields: {missing}")
        from robosuite.utils import camera_utils

        def metric_depth(camera_name: str) -> np.ndarray:
            return camera_utils.get_real_depth_map(
                self._env.sim,
                np.asarray(observation[f"{camera_name}_depth"], dtype=np.float64),
            )
        return PandaSortingObservation(
            external_rgb=np.asarray(observation[f"{self.EXTERNAL_CAMERA}_image"]),
            external_depth=metric_depth(self.EXTERNAL_CAMERA),
            wrist_rgb=np.asarray(observation[f"{self.WRIST_CAMERA}_image"]),
            proprio=np.asarray(observation["robot0_proprio-state"], dtype=np.float64),
            gripper=np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
            eef_position=np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            front_rgb=np.asarray(observation[f"{self.FRONT_CAMERA}_image"]),
            front_depth=metric_depth(self.FRONT_CAMERA),
        )

    def _apply_visual_labels(self) -> None:
        """Assign fixed semantic colors to task objects without changing physics."""
        for name, rgb in self.object_colors_rgb.items():
            rgba = np.asarray([*rgb, 255], dtype=np.float64) / 255.0
            for suffix in ("_g0", "_g0_visual"):
                geom_name = f"{name}{suffix}"
                try:
                    geom_id = self._env.sim.model.geom_name2id(geom_name)
                except ValueError:
                    continue
                # PickPlace meshes carry texture materials. Detach only their
                # visual material so the semantic label remains visible under
                # renderer lighting; collision geometry is unchanged.
                self._env.sim.model.geom_matid[geom_id] = -1
                self._env.sim.model.geom_rgba[geom_id] = rgba
            # Robosuite overlays a translucent visual proxy for each PickPlace
            # item. Hide that proxy so the colored physical mesh is the only
            # visual identity seen by the RGB detector.
            try:
                visual_id = self._env.sim.model.geom_name2id(f"Visual{name}_g0")
            except ValueError:
                continue
            self._env.sim.model.geom_rgba[visual_id, 3] = 0.0
        self._env.sim.forward()

    def _record_evaluator_contacts(self) -> None:
        object_names = tuple(self.OBJECT_COLORS_RGB)
        for contact_index in range(int(self._env.sim.data.ncon)):
            contact = self._env.sim.data.contact[contact_index]
            first_geom = self._env.sim.model.geom_id2name(contact.geom1) or ""
            second_geom = self._env.sim.model.geom_id2name(contact.geom2) or ""
            first_object = next((name for name in object_names if name in first_geom), None)
            second_object = next((name for name in object_names if name in second_geom), None)
            if first_object is None or second_object is None or first_object == second_object:
                continue
            first, second = sorted((first_object.lower(), second_object.lower()))
            self._evaluator_contact_events.append({"first": first, "second": second})
