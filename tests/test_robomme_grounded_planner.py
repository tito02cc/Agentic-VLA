from __future__ import annotations

import pytest
import cv2
import numpy as np

from scripts.run_robomme_vlm_groundsg import (
    GroundedPlannerClient,
    RawGroundedPlannerClient,
    RouteStickTrajectoryController,
    button_center_grounding,
    build_button_unmask_memory,
    hidden_container_grounding,
    verified_identity_conflict,
    revalidate_unmask_pick,
    parse_grounded_subgoal,
    relational_color_grounding,
    RepetitionProgressController,
    RepeatedProcedureController,
    OrderedProcedureController,
    SemanticTransitionGate,
    StopCubeTemporalMonitor,
    requires_visual_grounding,
    native_skill_rejection,
)


def test_raw_planner_uses_official_style_prompt_without_carve_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"pick up the cube"}}]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    planner = RawGroundedPlannerClient(
        endpoint="http://planner.test/v1/chat/completions",
        model="groundsg",
        timeout_s=3.0,
        demo_video=None,
        task_goal="pick the cube",
    )

    result = planner.infer(np.zeros((32, 32, 3), dtype=np.uint8))

    assert result["grounded_subgoal"] == "pick up the cube"
    assert result["tool_grounding"] is None
    assert result["procedure_update"] is None
    assert "CARVE" not in str(captured["payload"])
    assert "This is the initial turn for prediction" in str(captured["payload"])
    assert captured["timeout"] == 3.0


def test_route_stick_system_prompt_enforces_learned_primitive_contract() -> None:
    planner = GroundedPlannerClient(
        endpoint="http://127.0.0.1:1",
        model="test",
        timeout_s=1.0,
        demo_video=None,
        task_name="RouteStick",
        task_goal="follow the demonstrated route",
    )

    prompt = planner._system_prompt()

    assert "nearest [left/right] target" in prompt
    assert "[clockwise/counterclockwise]" in prompt
    assert "Do not name stick colors" in prompt
    assert "robot-left appears on image-right" in prompt


def test_route_stick_trajectory_rewrites_camera_side_in_robot_frame() -> None:
    controller = RouteStickTrajectoryController([116.0, 84.0, 44.0, 84.0])
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(frame, (116, 85), 6, (255, 0, 0), 2)

    rewritten, update = controller.apply(
        "move to the nearest left target by circling around the stick clockwise",
        frame,
    )

    assert "nearest right target" in rewritten
    assert update is not None
    assert update["source"] == "demonstration_rgb"


def test_hidden_container_grounding_refines_sam2_memory_to_current_rgb() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[80:105, 110:132] = 220
    memory = {"hidden_container_points": {"green": [93.0, 120.0]}}

    result = hidden_container_grounding(frame, "green", memory)

    assert result is not None
    assert result["tool"] == "hidden_container_memory_grounding"
    assert np.linalg.norm(np.asarray(result["point"]) - np.asarray([92, 120])) < 3
    assert result["current_rgb_match"] is True


def test_verified_identity_conflict_requires_two_current_objects_and_wrong_identity() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[120:140, 115:135] = 220  # Green-hidden container, row/col near 130/125.
    frame[85:105, 94:114] = 220  # Red-hidden container, row/col near 95/104.
    memory = {"hidden_container_points": {"green": [129.5, 124.5], "red": [94.5, 103.5]}}

    result = verified_identity_conflict(
        frame, "green", [95, 104], memory, expected_color="green", grasp_observed=False,
    )
    assert result["resolved"] is True
    assert result["other_color"] == "red"
    assert result["point"] == [130, 124]
    assert verified_identity_conflict(
        frame, "green", [130, 125], memory, expected_color="green", grasp_observed=False,
    )["resolved"] is False
    assert verified_identity_conflict(
        frame, "green", [95, 104], memory, expected_color="red", grasp_observed=False,
    )["resolved"] is False
    assert verified_identity_conflict(
        frame, "green", [95, 104], memory, expected_color="green", grasp_observed=True,
    )["resolved"] is False
    assert verified_identity_conflict(
        frame[:110], "green", [95, 104], memory,
        expected_color="green", grasp_observed=False,
    )["resolved"] is False


def test_grounded_pick_reuse_requires_current_rgb_and_clear_grasp_state() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[80:105, 110:132] = 220
    memory = {"hidden_container_points": {"green": [92.0, 120.0]}}
    cached = {
        "grounded_subgoal": "pick up the container at <92, 120> that hides the green cube",
        "points": [[92, 120]],
    }

    accepted, receipt = revalidate_unmask_pick(frame, cached, memory, {"grasp_observed": False})
    assert accepted is not None and receipt["reason"] == "current_rgb_revalidated"
    assert accepted["points"] == [[92, 120]]
    assert revalidate_unmask_pick(frame, cached, memory, {"grasp_observed": True})[0] is None
    assert revalidate_unmask_pick(frame, cached, memory, None)[0] is None

    missing, receipt = revalidate_unmask_pick(
        np.zeros_like(frame), cached, memory, {"grasp_observed": False}
    )
    assert missing is None and receipt["reason"] == "current_rgb_target_missing"

    shifted = np.zeros_like(frame)
    shifted[80:105, 120:142] = 220
    missing, receipt = revalidate_unmask_pick(
        shifted, cached, memory, {"grasp_observed": False}
    )
    assert missing is None and receipt["reason"] == "grounded_point_shifted"


def test_unmask_swap_memory_activates_two_cycle_execution_monitor() -> None:
    planner = GroundedPlannerClient(
        endpoint="http://127.0.0.1:1",
        model="test",
        timeout_s=1.0,
        demo_video=None,
        task_name="VideoUnmaskSwap",
        task_goal="pick green then blue",
        task_memory={
            "planner_hint": "tracked hidden containers",
            "required_color_order": ["green", "blue"],
        },
    )

    assert planner.repeated_procedure.active
    assert planner.repeated_procedure.target == 2
    assert planner.repeated_procedure.final_pick_terminal
    assert planner._required_hidden_color() == "green"

    planner.repeated_procedure.cycle = 2
    planner.repeated_procedure.grasp_observed = True
    assert planner.repeated_procedure.validate("pick up the blue container") is None
    assert planner.repeated_procedure.validate("put it down") is not None
    assert planner.repeated_procedure.stage_fallback() == "continue_pick"


def test_videounmask_native_contract_admits_same_skills_without_stage_override():
    planner = GroundedPlannerClient(
        endpoint="http://127.0.0.1:1", model="test", timeout_s=1.0,
        demo_video=None, task_name="VideoUnmask",
        task_goal="pick the blue container then the green container",
        task_memory={"planner_hint": "public demo identity", "required_color_order": ["blue", "green"]},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
    )
    assert planner.repeated_procedure.final_pick_terminal
    assert native_skill_rejection(
        "VideoUnmask", "pick up the container at <104, 85> that hides the blue cube", [[104, 85]]
    ) is None
    assert native_skill_rejection("VideoUnmask", "put down the container", []) is None
    assert native_skill_rejection("VideoUnmask", "press the button", []) is not None


def test_repeated_procedure_updates_cycle_specific_pick_budget() -> None:
    controller = RepeatedProcedureController(
        "pick up and put down the same block three times",
        pick_prediction_schedule=(5, 2, 2),
    )
    assert controller.minimum_pick_predictions == 5
    controller.stage = "put"
    controller.release_observed = True
    controller.commit("pick up the block")
    assert controller.cycle == 2
    assert controller.minimum_pick_predictions == 2
    assert controller.pick_predictions == 1


def _stop_cube_frame(cube_point: tuple[int, int]) -> np.ndarray:
    frame = np.full((256, 256, 3), (170, 95, 45), dtype=np.uint8)
    cv2.circle(frame, (108, 100), 10, (220, 20, 220), -1)
    cv2.circle(frame, (108, 100), 6, (170, 95, 45), -1)
    cv2.circle(frame, (108, 100), 3, (220, 20, 220), -1)
    row, col = cube_point
    frame[row - 5 : row + 5, col - 6 : col + 6] = (20, 220, 60)
    return frame


def _same_color_stop_cube_frame(cube_point: tuple[int, int]) -> np.ndarray:
    frame = np.full((256, 256, 3), (170, 95, 45), dtype=np.uint8)
    cv2.circle(frame, (108, 100), 10, (220, 20, 220), -1)
    cv2.circle(frame, (108, 100), 6, (170, 95, 45), -1)
    cv2.circle(frame, (108, 100), 3, (220, 20, 220), -1)
    row, col = cube_point
    frame[row - 5 : row + 5, col - 6 : col + 6] = (220, 20, 220)
    return frame


def _low_contrast_stop_cube_frame(cube_point: tuple[int, int]) -> np.ndarray:
    frame = np.full((256, 256, 3), (170, 95, 45), dtype=np.uint8)
    cv2.circle(frame, (108, 100), 10, (220, 20, 220), -1)
    cv2.circle(frame, (108, 100), 6, (170, 95, 45), -1)
    cv2.circle(frame, (108, 100), 3, (220, 20, 220), -1)
    row, col = cube_point
    frame[row - 5 : row + 5, col - 6 : col + 6] = (188, 140, 94)
    return frame


def test_stop_cube_temporal_monitor_counts_debounced_target_entries() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the second time",
        _stop_cube_frame((75, 43)),
    )
    assert monitor.active
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    assert monitor.observe(_stop_cube_frame((100, 108)), 35)["observed_occurrences"] == 1
    assert monitor.observe(_stop_cube_frame((100, 108)), 36) is None
    assert monitor.observe(_stop_cube_frame((75, 43)), 70) is None
    second = monitor.observe(_stop_cube_frame((100, 108)), 113)
    assert second is not None
    assert second["observed_occurrences"] == 2


def test_stop_cube_temporal_monitor_reacquires_same_color_cube_after_overlap() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the second time",
        _same_color_stop_cube_frame((75, 43)),
    )
    assert monitor.active
    first = monitor.observe(_same_color_stop_cube_frame((100, 108)), 10)
    assert first is not None
    assert monitor.observe(_same_color_stop_cube_frame((75, 43)), 20) is None
    second = monitor.observe(_same_color_stop_cube_frame((100, 108)), 30)
    assert second is not None
    assert second["observed_occurrences"] == 2


def test_stop_cube_temporal_monitor_initializes_low_contrast_cube() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the second time",
        _low_contrast_stop_cube_frame((75, 43)),
    )
    assert monitor.active
    assert monitor.cube_hue is not None


def test_stop_cube_temporal_monitor_guards_press_timing() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the second time",
        _stop_cube_frame((75, 43)),
    )
    assert monitor.validate("remain static") is not None
    assert monitor.validate("move to the top of the button at <60, 90>") is None
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    assert monitor.validate("press the button") is not None
    monitor.observe(_stop_cube_frame((100, 108)), 10)
    monitor.observe(_stop_cube_frame((75, 43)), 20)
    monitor.observe(_stop_cube_frame((100, 108)), 30)
    assert monitor.validate("remain static") is not None
    assert monitor.validate("press the button") is None


def test_stop_cube_target_grounding_contract_uses_detected_bullseye() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the first time",
        _stop_cube_frame((75, 43)),
    )
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.record_executed_subgoal("move to the top of the button to prepare")
    event = monitor.observe(_stop_cube_frame((100, 108)), 10)
    assert event is not None
    assert np.linalg.norm(monitor.target_point - np.asarray([100.0, 108.0])) < 2.0


def test_stop_cube_temporal_monitor_anticipates_final_arrival() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the fourth time",
        _stop_cube_frame((75, 43)),
        action_horizon=16,
    )
    for _ in range(3):
        monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.observe(_stop_cube_frame((100, 108)), 34)
    monitor.observe(_stop_cube_frame((75, 43)), 70)
    monitor.observe(_stop_cube_frame((100, 108)), 114)
    monitor.observe(_stop_cube_frame((75, 43)), 150)
    monitor.observe(_stop_cube_frame((100, 108)), 194)
    monitor.observe(_stop_cube_frame((75, 43)), 220)
    event = monitor.observe(_stop_cube_frame((75, 43)), 256)
    assert event is not None
    assert event["anticipated_target"]
    assert event["predicted_target_step"] == 274
    assert monitor.ready_to_press
    assert monitor.validate("press the button") is None


def test_stop_cube_temporal_monitor_anticipates_second_arrival_from_rgb_half_cycle() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the second time",
        _stop_cube_frame((75, 43)),
        action_horizon=16,
    )
    for _ in range(3):
        monitor.record_executed_subgoal("move to the top of the button to prepare")
    monitor.observe(_stop_cube_frame((100, 108)), 35)
    monitor.observe(_stop_cube_frame((75, 43)), 70)
    event = monitor.observe(_stop_cube_frame((75, 43)), 95)
    assert event is not None
    assert event["anticipated_target"]
    assert event["predicted_target_step"] == 113
    assert event["period_estimate_source"] == "rgb_initial_half_cycle"


def test_stop_cube_prepare_completion_requests_replan_once() -> None:
    monitor = StopCubeTemporalMonitor(
        "press the button as the cube reaches the target for the fourth time",
        _stop_cube_frame((75, 43)),
    )
    assert not monitor.record_executed_subgoal(
        "move to the top of the button to prepare"
    )
    assert not monitor.record_executed_subgoal(
        "move to the top of the button to prepare"
    )
    assert monitor.record_executed_subgoal(
        "move to the top of the button to prepare"
    )
    assert not monitor.record_executed_subgoal(
        "move to the top of the button to prepare"
    )


def test_repeated_procedure_forces_progress_after_physical_completion() -> None:
    controller = RepeatedProcedureController(
        "pick up and put down the same block three times"
    )
    controller.open_reference = 0.04
    controller.observe_execution(0.018)
    assert controller.grasp_observed
    assert controller.validate("pick up the block") is not None
    assert controller.validate("put it down") is None
    controller.commit("put it down")
    controller.observe_execution(0.04)
    assert controller.release_observed
    assert controller.validate("put it down") is not None
    assert controller.validate("pick up the block") is None


def test_repeated_procedure_fallback_only_continues_incomplete_stage() -> None:
    controller = RepeatedProcedureController(
        "pick up and put down the same block three times"
    )
    assert controller.stage_fallback() == "continue_pick"
    controller.open_reference = 0.04
    controller.observe_execution(0.018)
    assert controller.stage_fallback() == "advance_put"
    controller.commit("put it down")
    assert controller.stage_fallback() == "continue_put"
    controller.observe_execution(0.04)
    assert controller.stage_fallback() == "advance_pick"


def test_parse_grounded_subgoal_scales_qwen_points() -> None:
    raw = (
        "Hook the cube at <|box_start|>(402,500)<|box_end|> to the target at "
        "<|box_start|>(378,437)<|box_end|> with the peg"
    )
    grounded, history, points = parse_grounded_subgoal(raw)
    assert grounded == "Hook the cube at <102, 128> to the target at <96, 111> with the peg"
    assert history == "Hook the cube at <bbox> to the target at <bbox> with the peg"
    assert points == [[102, 128], [96, 111]]


def test_parse_grounded_subgoal_accepts_official_point_free_subgoals() -> None:
    grounded, history, points = parse_grounded_subgoal(
        "move to the nearest right target by circling around the stick clockwise"
    )
    assert grounded == history
    assert points == []


def test_parse_grounded_subgoal_rejects_malformed_point_tokens() -> None:
    with pytest.raises(ValueError, match="malformed point token"):
        parse_grounded_subgoal("pick at <|box_start|>(120,500)")


def test_parse_grounded_subgoal_rejects_out_of_range_points() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        parse_grounded_subgoal("pick at <|box_start|>(1200,500)<|box_end|>")


def test_visual_grounding_requirement_distinguishes_targeted_and_route_actions() -> None:
    assert requires_visual_grounding("pick up the middle green cube")
    assert requires_visual_grounding("pick bottom-left red block")
    assert requires_visual_grounding("press the button to finish")
    assert not requires_visual_grounding("remain static")
    assert not requires_visual_grounding(
        "move to the nearest right target by circling around the stick clockwise"
    )


def test_relational_color_grounding_selects_topmost_green_component() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[35:45, 170:180] = (0, 255, 0)
    frame[120:135, 50:65] = (0, 255, 0)

    result = relational_color_grounding(frame, "pick up the topmost green block")

    assert result is not None
    assert result["color"] == "green"
    assert result["relation"] == "topmost"
    assert result["point"] == [40, 174]
    assert result["candidate_count"] == 2


def test_relational_color_grounding_ignores_untyped_target() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    assert relational_color_grounding(frame, "pick up the correct cube") is None


def test_relational_color_grounding_accepts_one_visible_color_without_relation() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[90:105, 130:145] = (0, 255, 0)

    result = relational_color_grounding(frame, "pick up the green cube")

    assert result is not None
    assert result["tool"] == "unique_color_grounding"
    assert result["relation"] == "unique-visible"
    assert result["point"] == [97, 137]


def test_button_unmask_memory_reacquires_container_after_occlusion() -> None:
    initial = np.zeros((256, 256, 3), dtype=np.uint8)
    initial[88:104, 80:96] = (255, 0, 0)
    memory = build_button_unmask_memory(
        initial, "first press the button, then pick the container hiding the red cube"
    )

    assert memory is not None
    assert memory["required_color_order"] == ["red"]
    assert memory["event_memory"]["targets"] == [
        {"color": "red", "point_row_col": [96.0, 88.0]}
    ]
    current = np.zeros((256, 256, 3), dtype=np.uint8)
    current[87:107, 79:99] = (220, 220, 220)
    current[110:130, 130:150] = (220, 220, 220)
    result = hidden_container_grounding(current, "red", memory)

    assert result is not None
    assert result["tool"] == "hidden_container_memory_grounding"
    assert result["point"] == [96, 88]


def test_button_unmask_memory_preserves_multiple_target_order() -> None:
    initial = np.zeros((256, 256, 3), dtype=np.uint8)
    initial[60:76, 40:56] = (255, 0, 0)
    initial[120:136, 160:176] = (0, 255, 0)

    memory = build_button_unmask_memory(
        initial,
        "press the button, pick the container hiding the red cube, then the green cube",
    )

    assert memory is not None
    assert memory["required_color_order"] == ["red", "green"]
    assert memory["hidden_container_points"] == {
        "red": [68.0, 48.0],
        "green": [128.0, 168.0],
    }


def test_button_unmask_repeated_suffix_waits_for_press_prefix() -> None:
    memory = {
        "planner_hint": "red then green",
        "required_color_order": ["red", "green"],
    }
    planner = GroundedPlannerClient(
        endpoint="http://unused",
        model="unused",
        timeout_s=1.0,
        demo_video=None,
        task_name="ButtonUnmask",
        task_goal=(
            "first press the button, then pick the container hiding the red cube, "
            "finally pick another container hiding the green cube"
        ),
        task_memory=memory,
    )

    assert planner.repeated_procedure.active is True
    assert planner.repeated_procedure.final_pick_terminal is True
    assert planner._repeated_control_enabled() is False
    for _ in range(6):
        planner.ordered_procedure.commit("press the button")
    planner.ordered_procedure.commit("pick up the red container")
    assert planner._repeated_control_enabled() is True


def test_relational_color_grounding_accepts_bottom_left_description() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[30:40, 180:190] = (255, 0, 0)
    frame[170:185, 45:60] = (255, 0, 0)

    result = relational_color_grounding(frame, "pick up the red block initially at bottom-left")

    assert result is not None
    assert result["relation"] == "bottom-left"
    assert result["point"] == [177, 52]


def test_relational_color_grounding_prioritizes_target_identity_clause() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[30:42, 35:47] = (0, 0, 255)
    frame[170:184, 180:194] = (0, 255, 0)
    text = (
        "The persistent target identity is blue block at top-left. "
        "Persistent evidence: green block at bottom-right."
    )

    result = relational_color_grounding(frame, text)

    assert result is not None
    assert result["color"] == "blue"
    assert result["relation"] == "top-left"
    assert result["point"] == [36, 40]


def test_button_center_grounding_refines_nearby_coarse_point() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[58:73, 125:142] = (210, 210, 210)
    frame[60:69, 128:139] = (100, 100, 100)

    result = button_center_grounding(
        frame,
        "press the button to stop",
        [[56, 130]],
    )

    assert result is not None
    assert result["tool"] == "button_center_grounding"
    assert result["detector"] == "dark_inset"
    assert result["point"] == [64, 133]


def test_button_center_grounding_does_not_override_non_button_action() -> None:
    frame = np.full((256, 256, 3), 100, dtype=np.uint8)
    assert button_center_grounding(frame, "pick up the cube", [[56, 130]]) is None


def test_button_center_grounding_recovers_missing_planner_point() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[58:74, 125:144] = (210, 210, 210)
    frame[61:70, 129:140] = (100, 100, 100)

    result = button_center_grounding(
        frame,
        "move to the top of the button to prepare",
        [],
    )

    assert result is not None
    assert result["source"] == "semantic_fallback"
    assert result["model_point"] is None
    assert result["detector"] == "gray_base"
    assert result["point"] == [66, 134]


def test_repetition_progress_rewrites_repeated_model_ordinal() -> None:
    controller = RepetitionProgressController("Pick and put down the same block three times")

    first, _, first_update = controller.apply(
        "pick up the cube for the first time", "pick up the cube for the first time"
    )
    controller.apply("put down the cube", "put down the cube")
    second, _, second_update = controller.apply(
        "pick up the cube for the first time", "pick up the cube for the first time"
    )

    assert first == "pick up the cube for the first time"
    assert first_update["current_repetition"] == 1
    assert second == "pick up the cube for the second time"
    assert second_update["current_repetition"] == 2


def test_action_parser_accepts_groundsg_bare_pick_and_put_it_down() -> None:
    assert RepetitionProgressController._action("pick bottom-left red block") == "pick"
    assert RepetitionProgressController._action("put") == "put"
    assert RepetitionProgressController._action("put it down") == "put"
    assert RepetitionProgressController._action("press") == "press"


def test_repetition_progress_is_inactive_without_count_constraint() -> None:
    controller = RepetitionProgressController("Pick up the demonstrated block")
    grounded, history, update = controller.apply("pick up the cube", "pick up the cube")
    assert (grounded, history, update) == ("pick up the cube", "pick up the cube", None)


def _candidate(text: str) -> dict[str, object]:
    return {
        "grounded_subgoal": text,
        "history_text": text,
        "points": [],
        "model_points": [],
        "tool_grounding": None,
    }


def test_semantic_transition_gate_requires_stable_new_primitive() -> None:
    gate = SemanticTransitionGate(confirmations=2)
    first, _ = gate.apply(_candidate("pick up the cube"))
    held, held_update = gate.apply(_candidate("put down the cube"))
    accepted, accepted_update = gate.apply(_candidate("put down the cube"))

    assert first["grounded_subgoal"] == "pick up the cube"
    assert held["grounded_subgoal"] == "pick up the cube"
    assert held_update["transition_accepted"] is False
    assert accepted["grounded_subgoal"] == "put down the cube"
    assert accepted_update["transition_accepted"] is True


def test_semantic_transition_gate_drops_one_turn_flicker() -> None:
    gate = SemanticTransitionGate(confirmations=2)
    gate.apply(_candidate("pick up the cube"))
    gate.apply(_candidate("put down the cube"))
    selected, update = gate.apply(_candidate("pick up the cube"))

    assert selected["grounded_subgoal"] == "pick up the cube"
    assert update["pending_count"] == 0


def test_ordered_procedure_rejects_semantic_regression() -> None:
    controller = OrderedProcedureController(
        "pick up the same block again, finally put it down and press the button"
    )
    controller.observe_execution(0.04)
    assert controller.validate("pick up the correct cube") is None
    controller.commit("pick up the correct cube")
    controller.observe_execution(0.02)
    assert controller.validate("put it down") is None
    controller.commit("put it down")
    controller.observe_execution(0.04)

    reason = controller.validate("pick up the correct cube")

    assert reason is not None
    assert "cannot regress" in reason
    assert "press" in reason


def test_ordered_procedure_waits_for_physical_gripper_events() -> None:
    controller = OrderedProcedureController("pick up the cube, put it down, press the button")
    controller.observe_execution(0.04)
    assert "gripper closure" in str(controller.validate("put it down"))

    controller.observe_execution(0.02)
    assert controller.validate("put it down") is None
    controller.commit("put it down")
    assert "gripper reopening" in str(controller.validate("press the button"))

    controller.observe_execution(0.04)
    assert controller.validate("press the button") is None


def test_ordered_procedure_uses_chunk_budget_for_press_transition() -> None:
    controller = OrderedProcedureController(
        "first press the button, then pick up the highlighted cube"
    )
    controller.observe_execution(0.04)
    for _ in range(5):
        assert controller.validate("press the button") is None
        controller.commit("press the button")

    reason = controller.validate("pick up the highlighted cube")
    assert reason is not None
    assert "executed press chunks" in reason
    assert controller.stage_fallback() == "continue_ordered_stage"

    controller.commit("press the button")
    assert controller.validate("pick up the highlighted cube") is None
    assert controller.stage_fallback() is None


def test_ordered_procedure_defers_numeric_repetition_to_repetition_controller() -> None:
    controller = OrderedProcedureController(
        "pick up and put down the same block three times, then press the button"
    )
    assert controller.active is False
    assert controller.validate("pick up the cube") is None


def test_repeated_procedure_requires_physical_events_for_each_cycle() -> None:
    controller = RepeatedProcedureController(
        "pick up and put down the same block three times, then press the button"
    )
    controller.observe_execution(0.04)
    assert "gripper closure" in str(controller.validate("put"))

    controller.observe_execution(0.02)
    assert controller.validate("put") is None
    controller.commit("put")
    assert "gripper reopening" in str(controller.validate("pick"))

    controller.observe_execution(0.04)
    assert controller.validate("pick") is None
    update = controller.commit("pick")
    assert update["current_cycle"] == 2
    assert update["current_stage"] == "pick"


def test_repeated_procedure_allows_press_only_after_final_release() -> None:
    controller = RepeatedProcedureController("pick and put the block two times, then press")
    controller.observe_execution(0.04)
    for cycle in (1, 2):
        controller.observe_execution(0.02)
        controller.commit("put")
        controller.observe_execution(0.04)
        if cycle == 1:
            controller.commit("pick")

    assert controller.validate("pick") is not None
    assert controller.validate("press") is None
    update = controller.commit("press")
    assert update["current_stage"] == "press"
