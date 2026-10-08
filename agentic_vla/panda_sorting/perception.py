"""RGB-D color-label perception for the physics sorting demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .contracts import SceneObject
from .environment import PandaSortingObservation


@dataclass(frozen=True)
class ColorLabel:
    name: str
    rgb: tuple[int, int, int]
    hue_tolerance: int = 12
    min_area_px: int = 55
    projection_height_m: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.min_area_px <= 0 or self.hue_tolerance <= 0:
            raise ValueError("invalid color label")
        if any(not 0 <= channel <= 255 for channel in self.rgb):
            raise ValueError("rgb channels must be in [0, 255]")
        if self.projection_height_m is not None and self.projection_height_m <= 0.0:
            raise ValueError("projection_height_m must be positive when provided")


class ColorRgbdPerception:
    """Detect task labels in RGB and backproject their visible RGB-D centroid.

    This component is intentionally simple and replaceable. It never accesses
    Robosuite object observations, body poses, segmentation masks, rewards, or
    success labels. Its calibration argument represents an installed external
    camera calibration, which has an equivalent real-robot counterpart.
    """

    def __init__(self, labels: Mapping[str, ColorLabel]) -> None:
        if not labels:
            raise ValueError("at least one color label is required")
        if set(labels) != {label.name for label in labels.values()}:
            raise ValueError("label mapping keys must equal label names")
        self.labels = dict(labels)

    def perceive(
        self,
        observation: PandaSortingObservation,
        *,
        pixel_to_world: np.ndarray,
    ) -> tuple[SceneObject, ...]:
        return self.perceive_arrays(
            observation.external_rgb,
            observation.external_depth,
            pixel_to_world=pixel_to_world,
        )

    def perceive_front(
        self,
        observation: PandaSortingObservation,
        *,
        pixel_to_world: np.ndarray,
    ) -> tuple[SceneObject, ...]:
        """Detect labels from the fixed front camera when the external view occludes one."""
        if observation.front_rgb is None or observation.front_depth is None:
            raise ValueError("front RGB-D is unavailable in this observation")
        return self.perceive_arrays(
            observation.front_rgb,
            observation.front_depth,
            pixel_to_world=pixel_to_world,
        )

    def perceive_multiview(
        self,
        observation: PandaSortingObservation,
        *,
        external_pixel_to_world: np.ndarray,
        front_pixel_to_world: np.ndarray,
    ) -> tuple[SceneObject, ...]:
        """Use the external view first and front RGB-D as an occlusion fallback."""
        external = {
            item.name: item
            for item in self.perceive(
                observation,
                pixel_to_world=external_pixel_to_world,
            )
        }
        front = {
            item.name: item
            for item in self.perceive_front(
                observation,
                pixel_to_world=front_pixel_to_world,
            )
        }
        return tuple(
            external[name]
            if external[name].visible and external[name].table_position is not None
            else front[name]
            for name in self.labels
        )

    def perceive_arrays(
        self,
        raw_rgb: np.ndarray,
        raw_depth: np.ndarray,
        *,
        pixel_to_world: np.ndarray,
    ) -> tuple[SceneObject, ...]:
        """Detect labels in a calibrated camera frame without simulator metadata."""
        rgb = np.asarray(raw_rgb, dtype=np.uint8)
        depth = np.asarray(raw_depth, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("external_rgb must have shape [H, W, 3]")
        if depth.shape != (*rgb.shape[:2], 1):
            raise ValueError("external_depth must have shape [H, W, 1]")
        transform = np.asarray(pixel_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("pixel_to_world must have shape [4, 4]")

        # The renderer RGB buffer is vertically flipped, whereas Robosuite's
        # metric depth buffer already follows the pixel convention used by
        # ``camera_utils``. Normalize RGB orientation for color detection but
        # retain depth coordinates for calibrated backprojection.
        rgb = rgb[::-1].copy()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        detections: list[SceneObject] = []
        for label in self.labels.values():
            component = self._largest_component(hsv, label)
            if component is None:
                detections.append(SceneObject(label.name, (0.0, 0.0), None, 0.0, visible=False))
                continue
            cx, cy, area = component
            point = self._backproject(cy, cx, depth, transform)
            planar_point = (
                None
                if label.projection_height_m is None
                else self._project_ray_to_height(cy, cx, transform, label.projection_height_m)
            )
            confidence = min(1.0, area / max(4.0 * label.min_area_px, 1.0))
            detections.append(
                SceneObject(
                    label.name,
                    (float(cx), float(cy)),
                    (
                        None
                        if point is None and planar_point is None
                        else tuple(
                            float(value)
                            for value in (point if planar_point is None else planar_point)[:2]
                        )
                    ),
                    float(confidence),
                    visible=point is not None,
                    world_position=None
                    if point is None
                    else (float(point[0]), float(point[1]), float(point[2])),
                )
            )
        return tuple(detections)

    @staticmethod
    def _largest_component(hsv: np.ndarray, label: ColorLabel) -> tuple[float, float, int] | None:
        label_hsv = cv2.cvtColor(np.asarray([[label.rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hue = int(label_hsv[0])
        saturation = max(100, int(label_hsv[1]) - 75)
        value = max(60, int(label_hsv[2]) - 115)
        low_hue = hue - label.hue_tolerance
        high_hue = hue + label.hue_tolerance
        if low_hue < 0:
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, saturation, value]), np.array([high_hue, 255, 255])),
                cv2.inRange(hsv, np.array([180 + low_hue, saturation, value]), np.array([179, 255, 255])),
            )
        elif high_hue > 179:
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([low_hue, saturation, value]), np.array([179, 255, 255])),
                cv2.inRange(hsv, np.array([0, saturation, value]), np.array([high_hue - 180, 255, 255])),
            )
        else:
            mask = cv2.inRange(hsv, np.array([low_hue, saturation, value]), np.array([high_hue, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = [(int(stats[i, cv2.CC_STAT_AREA]), centroids[i]) for i in range(1, count)]
        candidates = [item for item in candidates if item[0] >= label.min_area_px]
        if not candidates:
            return None
        area, centroid = max(candidates, key=lambda item: item[0])
        return float(centroid[0]), float(centroid[1]), area

    @staticmethod
    def _backproject(
        row: float,
        col: float,
        depth: np.ndarray,
        pixel_to_world: np.ndarray,
    ) -> np.ndarray | None:
        r, c = int(round(row)), int(round(col))
        radius = 3
        patch = depth[max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1, 0]
        values = patch[np.isfinite(patch) & (patch > 0.0)]
        if values.size == 0:
            return None
        z = float(np.median(values))
        pixel_h = np.asarray([col * z, row * z, z, 1.0], dtype=np.float64)
        point = pixel_to_world @ pixel_h
        if not np.all(np.isfinite(point)):
            return None
        return point[:3]

    @staticmethod
    def _project_ray_to_height(
        row: float,
        col: float,
        pixel_to_world: np.ndarray,
        world_height_m: float,
    ) -> np.ndarray | None:
        """Intersect a calibrated camera ray with a declared task-height plane.

        The depth at a color centroid can lie on a slanted object side and vary
        substantially after a small rotation. For planar XY control, intersect
        that centroid ray with the object's known nominal center-height instead.
        The raw depth point is still retained as ``world_position`` for visual
        lift verification.
        """
        origin = pixel_to_world @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        direction = pixel_to_world @ np.asarray([col, row, 1.0, 0.0], dtype=np.float64)
        if abs(float(direction[2])) < 1e-9:
            return None
        depth_scale = (float(world_height_m) - float(origin[2])) / float(direction[2])
        point = origin + depth_scale * direction
        return point[:3] if np.all(np.isfinite(point)) else None
