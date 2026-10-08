"""Public contracts for the multi-stage ToolHang experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ToolHangPhase(str, Enum):
    """Evaluator-side task phases used for reporting, never policy input."""

    FRAME_UNASSEMBLED = "frame_unassembled"
    FRAME_ASSEMBLED = "frame_assembled"
    COMPLETE = "complete"


class ToolHangSkill(str, Enum):
    """High-level capabilities that a planner may request."""

    OBSERVE_SCENE = "observe_scene"
    ASSEMBLE_HOOK_FRAME = "assemble_hook_frame"
    HANG_TOOL = "hang_tool"
    RECOVER_GRASP = "recover_grasp"
    RECOVER_ALIGNMENT = "recover_alignment"
    RETRACT_AND_REOBSERVE = "retract_and_reobserve"
    VERIFY_STAGE = "verify_stage"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class ToolHangTask:
    """Natural-language task and its physical stage dependencies."""

    task_id: str
    instruction: str
    required_stages: tuple[str, ...] = ("assemble_hook_frame", "hang_tool")

    @classmethod
    def canonical(cls) -> "ToolHangTask":
        return cls(
            task_id="robosuite-tool-hang",
            instruction=(
                "Assemble the hook frame onto the upright stand, then pick up "
                "the wrench and hang it securely from the assembled hook."
            ),
        )


@dataclass(frozen=True)
class ToolHangObservation:
    """Sensor data exposed to the planner, monitor, and physical policy.

    Object poses, simulator state, rewards, and task-success labels are
    deliberately absent.
    """

    external_rgb: np.ndarray
    external_depth: np.ndarray
    wrist_rgb: np.ndarray
    wrist_depth: np.ndarray
    proprio: np.ndarray
    gripper: np.ndarray
    eef_position: np.ndarray
    eef_quaternion: np.ndarray
    front_rgb: np.ndarray | None = None
    front_depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        image_fields = (
            ("external_rgb", self.external_rgb, 3),
            ("external_depth", self.external_depth, 1),
            ("wrist_rgb", self.wrist_rgb, 3),
            ("wrist_depth", self.wrist_depth, 1),
        )
        for name, value, channels in image_fields:
            array = np.asarray(value)
            if array.ndim != 3 or array.shape[-1] != channels:
                raise ValueError(
                    f"{name} must have shape [height, width, {channels}]"
                )
        if np.asarray(self.proprio).shape != (9,):
            raise ValueError(
                "proprio must contain eef position, eef quaternion, and "
                "gripper position in a version-stable 9D state"
            )
        if np.asarray(self.gripper).shape != (2,):
            raise ValueError("gripper must have shape [2]")
        if np.asarray(self.eef_position).shape != (3,):
            raise ValueError("eef_position must have shape [3]")
        if np.asarray(self.eef_quaternion).shape != (4,):
            raise ValueError("eef_quaternion must have shape [4]")


@dataclass(frozen=True)
class ToolHangEvaluation:
    """Private simulator labels written only to evaluator artifacts."""

    frame_assembled: bool
    tool_hung: bool
    success: bool
    phase: ToolHangPhase

    @classmethod
    def from_checks(
        cls,
        *,
        frame_assembled: bool,
        tool_hung: bool,
    ) -> "ToolHangEvaluation":
        frame = bool(frame_assembled)
        tool = bool(tool_hung)
        if frame and tool:
            phase = ToolHangPhase.COMPLETE
        elif frame:
            phase = ToolHangPhase.FRAME_ASSEMBLED
        else:
            phase = ToolHangPhase.FRAME_UNASSEMBLED
        return cls(
            frame_assembled=frame,
            tool_hung=tool,
            success=frame and tool,
            phase=phase,
        )
