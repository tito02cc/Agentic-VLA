"""Recorded-view diagnostics must not leak review labels into model requests."""

import json

import numpy as np
import pytest

from scripts.probe_robodojo_sorting_critic import fixed_cases, frame_timestep, select_frames
from agentic_vla.toolchain.verifier import VisualVerificationContext, build_visual_verification_request


def test_original_cases_and_unknown_transfer_references():
    assert len(fixed_cases()) == 5
    expanded = fixed_cases(True)
    assert expanded[:5] == fixed_cases()
    assert len(expanded) == 8
    assert all(case[3] is None for case in expanded[5:])


@pytest.mark.parametrize("task,phase,expected", [
    ("organize_native", "initial", 0), ("organize_native", "final", 1000),
    ("classify_native", "final", 1100), ("organize_native", "frame500", 500),
    ("classify_native", "frame550", 550),
])
def test_timestep_matches_recorded_frame(task, phase, expected):
    assert frame_timestep(task, phase) == expected


def test_camera_selection_does_not_mutate_source():
    frames = {k: np.zeros((8, 8, 3), dtype=np.uint8)
              for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    assert list(select_frames(frames, "head")) == ["cam_high"]
    assert len(frames) == len(select_frames(frames, "all")) == 3
    with pytest.raises(ValueError):
        select_frames(frames, "invented")


def test_manual_reference_is_not_in_request():
    for task, phase, expected, reference in fixed_cases(True):
        ctx = VisualVerificationContext(task_instruction="public instruction",
            expected_outcome=expected, frames={"current_cam_high": np.zeros((8, 8, 3), dtype=np.uint8)},
            timestep=frame_timestep(task, phase))
        payload = json.loads(build_visual_verification_request(ctx)["user_prompt"])
        assert set(payload) == {"task_instruction", "expected_outcome", "active_stage", "timestep"}
        assert "manual_reference" not in payload


@pytest.mark.parametrize("phase", ["frame-1", "frameoops", "unknown"])
def test_bad_phase_rejected(phase):
    with pytest.raises(ValueError):
        frame_timestep("organize_native", phase)
