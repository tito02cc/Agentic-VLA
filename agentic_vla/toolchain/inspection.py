"""Bounded camera evidence and read-only region inspection for planner tools."""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import ToolEffect, ToolExecutionContext, ToolSpec, validate_tool_input


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rgb_hash(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class _CameraEvidence:
    frame_id: str
    episode_id: str | int
    camera: str
    timestep: int
    rgb: np.ndarray
    rgb_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "camera": self.camera,
            "timestep": self.timestep,
            "shape": list(self.rgb.shape),
            "rgb_sha256": self.rgb_sha256,
        }


class VisualEvidenceStore:
    """Episode-local image handles; never evidence of task success by themselves.

    Images travel through resolve_images(), not JSON logs or model-written paths.
    This class uses simulation/control-step age, not a wall-clock deadline.
    The host resets this store and the tool registry at each episode boundary.
    """

    def __init__(
        self,
        *,
        max_frames: int = 6,
        max_frame_pixels: int = 1_048_576,
        max_age_steps: int = 8,
        max_inspections: int = 12,
    ) -> None:
        self.max_frames = _integer(max_frames, "max_frames", 1)
        self.max_frame_pixels = _integer(max_frame_pixels, "max_frame_pixels", 256)
        self.max_age_steps = _integer(max_age_steps, "max_age_steps")
        self.max_inspections = _integer(max_inspections, "max_inspections", 1)
        self._frames: OrderedDict[str, _CameraEvidence] = OrderedDict()
        self._inspections: OrderedDict[str, tuple[str, tuple[int, ...]]] = OrderedDict()
        self._episode_id: str | int | None = None
        self._latest_timestep = -1
        self._inspection_count = 0
        self._lock = threading.RLock()

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="inspect_region",
            description=(
                "Inspect a region of a captured RGB frame by frame_id. Coordinates are "
                "integer pixels: left/top inclusive, right/bottom exclusive. The result "
                "links the unchanged full frame and its crop, NOT two independent views. "
                "No new sensor measurement, robot motion, or task confirmation occurs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string", "minLength": 1},
                    **{
                        key: {"type": "integer"}
                        for key in ("left", "top", "right", "bottom")
                    },
                },
                "required": ["frame_id", "left", "top", "right", "bottom"],
                "additionalProperties": False,
            },
            effect=ToolEffect.OBSERVE,
            max_calls_per_episode=self.max_inspections,
        )

    def reset(self, episode_id: str | int) -> None:
        if type(episode_id) not in (str, int) or not str(episode_id).strip():
            raise ValueError("episode_id must be a nonempty string or integer")
        with self._lock:
            self._episode_id = episode_id
            self._frames.clear()
            self._inspections.clear()
            self._latest_timestep = -1
            self._inspection_count = 0

    def _check_context(self, context: ToolExecutionContext) -> None:
        _integer(context.timestep, "timestep")
        if (
            self._episode_id is None
            or type(context.episode_id) is not type(self._episode_id)
            or context.episode_id != self._episode_id
        ):
            raise ValueError("visual evidence belongs to another episode; reset first")
        if context.timestep < self._latest_timestep:
            raise ValueError("visual evidence context moved backwards")

    def capture(
        self, camera: str, rgb: np.ndarray, context: ToolExecutionContext
    ) -> dict[str, Any]:
        """Host-only capture; the planner cannot upload or replace sensor pixels."""
        if not isinstance(camera, str) or not camera.strip() or len(camera) > 128:
            raise ValueError("camera must be a bounded nonempty name")
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[-1] != 3
            or min(rgb.shape[:2]) < 16
            or rgb.shape[0] * rgb.shape[1] > self.max_frame_pixels
        ):
            raise ValueError(
                "camera frame must be bounded HxWx3 uint8 RGB, at least 16x16"
            )
        with self._lock:
            self._check_context(context)
            # Bytes-backed storage cannot be mutated through the caller's source array.
            frozen = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)
            frame = _CameraEvidence(
                uuid.uuid4().hex,
                context.episode_id,
                camera,
                context.timestep,
                frozen,
                _rgb_hash(frozen),
            )
            self._frames[frame.frame_id] = frame
            self._latest_timestep = context.timestep
            while len(self._frames) > self.max_frames:
                evicted, _ = self._frames.popitem(last=False)
                for key, (source, _) in tuple(self._inspections.items()):
                    if source == evicted:
                        del self._inspections[key]
            return frame.metadata()

    def _frame(self, frame_id: str, context: ToolExecutionContext) -> _CameraEvidence:
        self._check_context(context)
        if not isinstance(frame_id, str) or frame_id not in self._frames:
            raise ValueError("unknown or evicted camera frame")
        frame = self._frames[frame_id]
        age = context.timestep - frame.timestep
        if not 0 <= age <= self.max_age_steps:
            raise ValueError("camera frame expired or is from the future")
        self._latest_timestep = context.timestep
        return frame

    def inspect_region(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> dict[str, Any]:
        validate_tool_input(arguments, self.tool_spec.input_schema)
        with self._lock:
            frame = self._frame(arguments["frame_id"], context)
            if self._inspection_count >= self.max_inspections:
                raise ValueError("visual inspection budget exhausted")
            rect = tuple(
                _integer(arguments[key], key)
                for key in ("left", "top", "right", "bottom")
            )
            left, top, right, bottom = rect
            height, width = frame.rgb.shape[:2]
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("region must lie inside the source frame")
            if min(right - left, bottom - top) < 16:
                raise ValueError("region must be at least 16x16 pixels")
            crop = frame.rgb[top:bottom, left:right]
            inspection_id = uuid.uuid4().hex
            self._inspections[inspection_id] = (frame.frame_id, rect)
            self._inspection_count += 1
            return {
                "inspection_id": inspection_id,
                "source": frame.metadata(),
                "region_ltrb": list(rect),
                "region_shape": list(crop.shape),
                "region_rgb_sha256": _rgb_hash(crop),
                "relationship": "crop_of_same_frame_not_an_independent_view",
                "authority": "observation_only",
            }

    def resolve_frame(
        self, frame_id: str, context: ToolExecutionContext
    ) -> np.ndarray:
        """Host-only full-frame delivery, with the same expiry checks as crops."""
        with self._lock:
            return self._frame(frame_id, context).rgb.copy()

    def resolve_images(
        self, inspection_id: str, context: ToolExecutionContext
    ) -> dict[str, np.ndarray]:
        """Host-only multimodal payload lookup, rechecking freshness at delivery."""
        with self._lock:
            self._check_context(context)
            if (
                not isinstance(inspection_id, str)
                or inspection_id not in self._inspections
            ):
                raise ValueError("unknown or evicted inspection")
            frame_id, rect = self._inspections[inspection_id]
            frame = self._frame(frame_id, context)
            left, top, right, bottom = rect
            return {
                "global": frame.rgb.copy(),
                "region": frame.rgb[top:bottom, left:right].copy(),
            }
