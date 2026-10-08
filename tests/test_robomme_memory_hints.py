from __future__ import annotations

import numpy as np

from scripts.build_robomme_memory_hints import hint_for
from scripts.track_robomme_video_memory import initial_identity, select_motion_candidate


def test_stop_cube_count_is_read_from_string_constraint() -> None:
    memory = {
        "task_constraints": "press button at second cube arrival at target",
        "persistent_memory": "cube is left of target",
    }
    assert hint_for("StopCube", memory).endswith("occurrence 2.")


def test_stop_cube_count_falls_back_to_instruction() -> None:
    memory = {"task_constraints": ["press button", "at target"]}
    hint = hint_for(
        "StopCube",
        memory,
        "press the button as it reaches the target for the fourth time",
    )
    assert hint.endswith("occurrence 4.")


def test_route_stick_preserves_string_sequence() -> None:
    hint = hint_for(
        "RouteStick",
        {
            "demonstrated_sequence": ["move left", "circle clockwise", "return"],
            "persistent_memory": "red stick is leftmost",
        },
    )
    assert "move left; circle clockwise; return" in hint
    assert "red stick is leftmost" in hint


def test_video_unmask_uses_generic_memory_shape_without_empty_unknowns() -> None:
    hint = hint_for(
        "VideoUnmaskSwap",
        {
            "persistent_memory": "green cube ends below the left container",
            "demonstrated_sequence": ["swap left and right containers"],
        },
    )
    assert "green cube ends below the left container" in hint
    assert "swap left and right containers" in hint


def test_video_repick_keeps_persistent_identity_for_string_sequence() -> None:
    hint = hint_for(
        "VideoRepick",
        {
            "persistent_memory": "the selected blue block is now top-left",
            "demonstrated_sequence": ["pick selected block", "put it down"],
            "task_constraints": "pick the same block once, then press",
        },
    )
    assert "selected blue block is now top-left" in hint
    assert "pick the same block once" in hint


def test_video_repick_extracts_nested_grasp_identity_before_persistent_object_list() -> None:
    hint = hint_for(
        "VideoRepick",
        {
            "demonstrated_sequence": [
                {
                    "action": "robot grasps the blue block",
                    "object_interactions": [
                        {
                            "object": "blue block",
                            "location": "top-left",
                            "action": "grasped",
                        }
                    ],
                }
            ],
            "persistent_memory": [
                {"object": "blue block", "final_location": "top-left"},
                {"object": "green block", "final_location": "bottom-left"},
            ],
        },
    )
    assert "persistent target identity is blue block at top-left" in hint


def test_motion_selector_prefers_persistently_changed_instance_patch() -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
    for frame in frames[2:9]:
        frame[35:46, 35:46] = 255
    candidates = [
        {"point": [15.0, 15.0], "box": [10, 10, 20, 20], "area": 100},
        {"point": [40.0, 40.0], "box": [35, 35, 45, 45], "area": 100},
    ]

    selected, diagnostic = select_motion_candidate(
        frames,
        candidates,
        patch_radius=6,
        change_threshold=30.0,
    )

    assert selected["point"] == [40.0, 40.0]
    assert diagnostic["selected_candidate_index"] == 1
    assert diagnostic["changed_frame_margin"] > 0


def test_tracker_identity_accepts_string_demonstration_steps() -> None:
    memory = {
        "demonstrated_sequence": [
            "approach green block at top-left",
            "grasp green block at top-left",
        ]
    }
    assert initial_identity(memory) == "green block at top-left"
