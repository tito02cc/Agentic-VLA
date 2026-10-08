from __future__ import annotations

import json

import numpy as np
import pytest

from agentic_vla.runtime import GroundedSubgoalScheduleConfig, GroundedSubgoalScheduler
from scripts.run_robomme_vlm_groundsg import (
    GroundedPlannerClient, RawGroundedPlannerClient, unmask_swap_skill_rejection, repick_skill_rejection,
)


@pytest.mark.parametrize("memory_enabled", [False, True])
def test_repick_observation_path_never_forces_stage_or_repetition(monkeypatch, memory_enabled):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the correct cube at <|box_start|>(392,392)<|box_end|> for the third time"
            }}]}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    common = dict(endpoint="http://planner.test", model="test", timeout_s=1, demo_video=None,
                  task_goal="pick up and put down the same block for two times, then press the button")
    raw = RawGroundedPlannerClient(**common)
    candidate = GroundedPlannerClient(
        **common, task_name="VideoRepick", execution_feedback="execution_chunks",
        procedure_authority="observe_only", planner_context="native", grounding_authority="observe_only",
        task_memory={"planner_hint": "source identity memory"} if memory_enabled else None,
    )
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in (1, 2):
        expected = raw.infer(frame)
        result = candidate.infer(frame)
        assert result["grounded_subgoal"] == expected["grounded_subgoal"]
        assert "third time" in result["grounded_subgoal"]
        assert result["tool_grounding"] is None and result["monitor_fallback"] is None
        if not memory_enabled:
            assert requests[-1] == requests[-2]
        receipt = candidate.record_executed_chunk(result["grounded_subgoal"], [state(0.018)] * 16,
                                                 chunk_id=i, start_step=(i - 1) * 16)
        assert not receipt["semantic_success_verified"]
    assert candidate.repeated_procedure.pick_prediction_schedule == ()


@pytest.mark.parametrize("text,points,accepted", [
    ("pick up the correct cube at <100, 150> for the third time", [[100, 150]], True),
    ("put it down", [], True),
    ("press the button at <55, 130> to finish", [[55, 130]], True),
    ("press the button at <55, 130> to finish", [], False),
    ("put it down then press the button", [], False),
    ("What's the next grounded language subgoal based on current observation?", [], False),
])
def test_repick_skill_contract(text, points, accepted):
    assert (repick_skill_rejection(text, points) is None) == accepted


def test_repick_repair_names_typed_skills_without_accepting_paraphrase():
    reason = repick_skill_rejection("pick up the correct block at <328,353>", [[328, 353]])
    assert "pick up the correct cube" in reason
    assert "put it down" in reason
    assert "press the button" in reason
    assert "<row, column>" not in reason


@pytest.mark.parametrize("text,points,accepted", [
    ("put down the container", [], True),
    ("put down the container.", [], True),
    ("pick up the container at <100, 150> that hides the blue cube", [[100, 150]], True),
    ("pick up the container at <100, 150> that hides the green cube", [[100, 150]], True),
    ("pick up the container at <100, 150> that hides the red cube", [[100, 150]], True),
    ("pick up the container at <100, 150> that hides the blue cube", [[101, 150]], False),
    ("pick up the container at <256, 150> that hides the blue cube", [[256, 150]], False),
    ("put down the container", [[100, 150]], False),
    ("What's the next grounded language subgoal based on current observation?", [], False),
    ("task complete", [], False),
    ("press the button at <100, 150>", [[100, 150]], False),
    ("put down the container then pick it up", [], False),
])
def test_unmask_skill_contract_does_not_choose_color_or_progress(text, points, accepted):
    assert (unmask_swap_skill_rejection(text, points) is None) == accepted


def test_question_echo_cannot_reach_vla_or_become_history(monkeypatch):
    text = "What's the next grounded language subgoal based on current observation?"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1, demo_video=None,
        task_name="VideoUnmaskSwap", task_goal="pick green then blue", repair_attempts=0,
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
    )
    with pytest.raises(RuntimeError, match="unsupported VideoUnmask action skill"):
        agent.infer(np.zeros((256, 256, 3), dtype=np.uint8))
    assert agent.request_count == 1
    assert agent.last_response_text == text
    assert agent.history_points == []
    assert agent.executed_steps == 0


@pytest.mark.parametrize("propose_grounding", [False, True])
def test_native_context_preserves_raw_request_and_keeps_execution_audit(monkeypatch, propose_grounding):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "pick up the blue cube at <|box_start|>(392,392)<|box_end|> for the first time"
            }}]}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    common = dict(endpoint="http://planner.test", model="test", timeout_s=1.0,
                  demo_video=None, task_goal="pick up the blue cube then place it")
    raw = RawGroundedPlannerClient(**common)
    candidate = GroundedPlannerClient(
        **common, task_name="PickXtimes", execution_feedback="execution_chunks",
        procedure_authority="observe_only", planner_context="native",
        grounding_authority="observe_only" if propose_grounding else "override",
        relational_color_grounding_enabled=propose_grounding,
    )
    if propose_grounding:
        monkeypatch.setattr("scripts.run_robomme_vlm_groundsg.relational_color_grounding",
                            lambda *args, **kwargs: {"tool": "test_grounding", "point": [20, 30]})
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for chunk in range(1, 3):
        raw.infer(frame)
        result = candidate.infer(frame)
        assert candidate.request_count == chunk
        assert requests[-2] == requests[-1]
        assert result["points"] == [[100, 100]]
        assert result["tool_grounding"] is None
        if propose_grounding:
            assert result["tool_grounding_proposal"]["point"] == [20, 30]
        receipt = candidate.record_executed_chunk(
            result["grounded_subgoal"], [state()] * 16,
            chunk_id=chunk, start_step=(chunk - 1) * 16,
        )
        assert receipt["end_step"] == chunk * 16
        assert receipt["semantic_success_verified"] is False
    assert candidate.executed_steps == 32


def test_rejected_planner_response_is_counted_and_retained(press_reply):
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1, demo_video=None,
        task_name="VideoUnmaskSwap", task_goal="pick green",
        task_memory={"planner_hint": "identity memory"}, repair_attempts=0,
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
    )
    with pytest.raises(RuntimeError, match="no grounding point"):
        agent.infer(np.zeros((256, 256, 3), dtype=np.uint8))
    assert agent.request_count == 1
    assert agent.last_response_text == "press the button at <100, 100>"


def test_native_context_rejects_legacy_authority():
    with pytest.raises(ValueError, match="native context requires"):
        GroundedPlannerClient(endpoint="http://planner.test", model="test", timeout_s=1,
                             demo_video=None, task_goal="pick", task_name="PickXtimes",
                             planner_context="native")


def test_observation_only_grounding_rejects_augmented_context():
    with pytest.raises(ValueError, match="observation-only grounding"):
        GroundedPlannerClient(endpoint="http://planner.test", model="test", timeout_s=1,
                             demo_video=None, task_goal="pick", task_name="PickXtimes",
                             grounding_authority="observe_only")


def test_identity_memory_does_not_rewrite_planner_color_or_stage(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the container at <|box_start|>(392,392)<|box_end|> that hides the blue cube"
            }}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1.0, demo_video=None,
        task_name="VideoUnmaskSwap", task_goal="pick green then blue",
        task_memory={"planner_hint": "historical identities", "required_color_order": ["green", "blue"],
                     "hidden_container_points": {"green": [20, 30], "blue": [40, 50]}},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
    )
    result = agent.infer(np.zeros((256, 256, 3), dtype=np.uint8))
    assert "blue cube" in result["grounded_subgoal"]
    assert result["points"] == [[100, 100]]
    assert result["tool_grounding"] is None
    assert result["monitor_fallback"] is None
    receipt = agent.record_executed_chunk(result["grounded_subgoal"], [state(0.0)] * 16,
                                         chunk_id=1, start_step=0)
    assert not receipt["semantic_success_verified"]


def test_opt_in_identity_conflict_resolves_only_grounded_point(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the container at <|box_start|>(375,406)<|box_end|> that hides the green cube"
            }}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[120:140, 115:135] = 220
    frame[85:105, 94:114] = 220
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1.0, demo_video=None,
        task_name="VideoUnmaskSwap", task_goal="pick green then red",
        task_memory={"planner_hint": "bound public-demo identity", "required_color_order": ["green", "red"],
                     "hidden_container_points": {"green": [129.5, 124.5], "red": [94.5, 103.5]}},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
        verified_identity_conflict_enabled=True,
    )

    result = agent.infer(frame)

    assert result["raw"].endswith("green cube")
    assert result["model_points"] == [[96, 103]]
    assert result["points"] == [[130, 124]]
    assert result["grounded_subgoal"] == "pick up the container at <130, 124> that hides the green cube"
    assert result["tool_grounding"] is None
    assert result["identity_conflict"]["resolved"] is True


def test_identity_conflict_opt_in_rejects_other_tasks():
    with pytest.raises(ValueError, match="verified identity conflict"):
        GroundedPlannerClient(
            endpoint="http://planner.test", model="test", timeout_s=1.0, demo_video=None,
            task_name="VideoUnmask", task_goal="pick green", execution_feedback="execution_chunks",
            procedure_authority="observe_only", planner_context="native",
            grounding_authority="observe_only", verified_identity_conflict_enabled=True,
        )


@pytest.mark.parametrize("task_name", ["VideoUnmask", "VideoUnmaskSwap"])
def test_stage_receipt_hint_only_after_executed_put_and_never_rewrites(monkeypatch, task_name):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the container at <|box_start|>(392,392)<|box_end|> that hides the green cube"
            }}]}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1.0, demo_video=None,
        task_name=task_name, task_goal="pick green then red",
        task_memory={"planner_hint": "public demo identity", "required_color_order": ["green", "red"]},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
        stage_receipt_hint_enabled=True,
    )
    agent.observe_robot_state(state(0.04))
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    first = agent.infer(frame)
    assert first["stage_receipt_hint"] is None
    agent.record_executed_chunk(first["grounded_subgoal"], [state(0.0)] * 16,
                                chunk_id=1, start_step=0)
    assert agent._stage_receipt_hint() is None
    agent.record_executed_chunk("put down the container", [state(0.0)] * 16,
                                chunk_id=2, start_step=16)
    second = agent.infer(frame)
    assert second["grounded_subgoal"] == first["grounded_subgoal"]
    assert "Gripper reopening observed: no" in second["stage_receipt_hint"]
    assert "next distinct target in the requested order is the red cube" in second["stage_receipt_hint"]
    assert second["stage_receipt_hint"] in requests[-1]["messages"][1]["content"][0]["text"]
    agent.record_executed_chunk("put down the container", [state(0.04)] * 16,
                                chunk_id=3, start_step=32)
    third = agent.infer(frame)
    assert "Gripper reopening observed: yes" in third["stage_receipt_hint"]
    assert third["grounded_subgoal"] == first["grounded_subgoal"]


def test_stage_receipt_hint_rejects_single_target_memory():
    with pytest.raises(ValueError, match="multi-target Unmask memory"):
        GroundedPlannerClient(
            endpoint="http://planner.test", model="test", timeout_s=1,
            demo_video=None, task_name="VideoUnmaskSwap", task_goal="pick green",
            task_memory={"required_color_order": ["green"]},
            execution_feedback="execution_chunks", procedure_authority="observe_only",
            planner_context="native", grounding_authority="observe_only",
            stage_receipt_hint_enabled=True,
        )


def test_unmask_question_echo_repairs_to_executable_skill_at_chunk_boundary(monkeypatch):
    requests = []
    responses = iter([
        "What's the next grounded language subgoal based on current observation?",
        "put down the container",
    ])

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": self.value}}]}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", capture)
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1.0, demo_video=None,
        task_name="VideoUnmask", task_goal="pick green then blue",
        task_memory={"planner_hint": "public demo identity", "required_color_order": ["green", "blue"]},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
        stage_receipt_hint_enabled=True, repair_attempts=1,
    )
    agent.observe_robot_state(state(0.04))
    agent.record_executed_chunk(
        "pick up the container at <100, 100> that hides the green cube",
        [state(0.0)] * 16, chunk_id=1, start_step=0,
    )
    agent.record_executed_chunk(
        "put down the container", [state(0.0)] * 16, chunk_id=2, start_step=16,
    )
    result = agent.infer(np.zeros((256, 256, 3), dtype=np.uint8))
    assert result["grounded_subgoal"] == "put down the container"
    assert result["repair_attempts"] == 1
    assert result["stage_receipt_hint"] is not None
    assert agent.executed_steps == 32
    assert agent.history_text == ["put down the container"]
    repair_text = requests[-1]["messages"][1]["content"][-1]["text"]
    assert "Do not repeat the question" in repair_text
    assert "<row, column>" not in repair_text


def test_demo_once_keeps_current_image_memory_and_all_planner_calls(monkeypatch):
    from pathlib import Path
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the container at <|box_start|>(392,392)<|box_end|> that hides the green cube"
            }}], "usage": {"prompt_tokens": 42, "completion_tokens": 12}}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    agent = GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1,
        demo_video=Path("/public_demo.mp4"), task_name="VideoUnmaskSwap", task_goal="pick green",
        task_memory={"planner_hint": "source identity memory"},
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only",
        planner_demo_mode="once_with_memory",
    )
    for _ in range(2):
        agent.infer(np.zeros((256, 256, 3), dtype=np.uint8))
    contents = [r["messages"][1]["content"] for r in requests]
    assert [sum(item["type"] == "video_url" for item in c) for c in contents] == [1, 0]
    assert all(sum(item["type"] == "image_url" for item in c) == 1 for c in contents)
    assert all("source identity memory" in str(c) for c in contents)
    assert agent.request_count == 2
    assert [r["demo_video_sent"] for r in agent.request_audit] == [True, False]
    assert agent.request_audit[-1]["usage"]["prompt_tokens"] == 42


def test_demo_omission_requires_bound_memory():
    with pytest.raises(ValueError, match="once_with_memory requires"):
        GroundedPlannerClient(
            endpoint="http://planner.test", model="test", timeout_s=1, demo_video=None,
            task_name="VideoUnmaskSwap", task_goal="pick green",
            execution_feedback="execution_chunks", procedure_authority="observe_only",
            planner_context="native", grounding_authority="observe_only",
            planner_demo_mode="once_with_memory",
        )


def state(gripper: float = 0.04) -> np.ndarray:
    return np.asarray([0.0] * 7 + [gripper], dtype=np.float32)


def planner(goal: str, mode: str = "execution_chunks") -> GroundedPlannerClient:
    return GroundedPlannerClient(
        endpoint="http://planner.test",
        model="test",
        timeout_s=1.0,
        demo_video=None,
        task_name="PickXtimes",
        task_goal=goal,
        execution_feedback=mode,
    )


@pytest.fixture
def press_reply(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "press the button at <100, 100>"
            }}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())


def test_predictions_do_not_become_execution_evidence(press_reply):
    agent = planner("press the button then pick up the block")
    agent.observe_robot_state(state())
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for _ in range(3):
        prediction = agent.infer(frame, robot_state=state())
        assert prediction["procedure_update"] is None
    assert agent.ordered_procedure.stage_predictions == 0
    assert agent.executed_chunks == 0
    assert not agent.ordered_procedure.stage_complete()[0]


def test_legacy_prediction_accounting_is_preserved(press_reply):
    agent = planner("press the button then pick up the block", "legacy_predictions")
    for _ in range(3):
        agent.infer(np.zeros((256, 256, 3), dtype=np.uint8), robot_state=state())
    assert agent.ordered_procedure.stage_predictions == 3
    assert agent.executed_chunks == 0


def test_reused_chunks_count_even_without_planner_calls():
    agent = planner("press the button then pick up the block")
    agent.observe_robot_state(state())
    for chunk_id in range(1, 7):
        receipt = agent.record_executed_chunk(
            "press the button", [state()] * 16,
            chunk_id=chunk_id, start_step=(chunk_id - 1) * 16,
        )
    assert agent.executed_chunks == agent.ordered_procedure.stage_predictions == 6
    assert receipt["end_step"] == 96
    assert receipt["execution_event"] == "procedure_boundary"
    assert receipt["semantic_success_verified"] is False


def test_all_observations_are_consumed_not_only_last_gripper_value():
    agent = planner("pick up and put down the block two times")
    agent.observe_robot_state(state())
    receipt = agent.record_executed_chunk(
        "pick up the block", [state(0.0), state(0.04)], chunk_id=1, start_step=0
    )
    assert receipt["grasp_observed"]
    assert not receipt["semantic_success_verified"]


def test_repeated_stage_handoff_uses_post_action_feedback():
    agent = planner("pick up and put down the block two times then press the button")
    agent.observe_robot_state(state())
    picked = agent.record_executed_chunk(
        "pick up the block", [state(0.0)], chunk_id=1, start_step=0
    )
    assert picked["execution_event"] == "procedure_boundary"
    placed = agent.record_executed_chunk(
        "put down the block", [state()], chunk_id=2, start_step=1
    )
    assert placed["release_observed"]
    assert placed["repeated_cycle"] == 1
    next_pick = agent.record_executed_chunk(
        "pick up the block", [state(0.0)], chunk_id=3, start_step=2
    )
    assert next_pick["repeated_cycle"] == 2
    assert not next_pick["release_observed"]


@pytest.mark.parametrize("states", [[], [np.zeros(7)], [np.full(8, np.nan)]])
def test_invalid_receipt_cannot_advance_procedure(states):
    agent = planner("press the button then pick up the block")
    with pytest.raises(ValueError, match="non-empty finite"):
        agent.record_executed_chunk("press the button", states, chunk_id=1, start_step=0)
    assert agent.executed_chunks == agent.ordered_procedure.stage_predictions == 0


@pytest.mark.parametrize("chunk_id,start_step", [(1, 0), (2, 0), (3, 1), (True, 1)])
def test_duplicate_or_out_of_order_receipt_is_rejected(chunk_id, start_step):
    agent = planner("press the button then pick up the block")
    agent.record_executed_chunk("press the button", [state()], chunk_id=1, start_step=0)
    with pytest.raises(ValueError, match="duplicate, stale, or out of order"):
        agent.record_executed_chunk(
            "press the button", [state()], chunk_id=chunk_id, start_step=start_step
        )
    assert agent.executed_chunks == agent.ordered_procedure.stage_predictions == 1


def test_boundary_event_overrides_reuse_dwell():
    scheduler = GroundedSubgoalScheduler(GroundedSubgoalScheduleConfig(
        min_reuse_chunks=3, max_reuse_chunks=4
    ))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    scheduler.record_planner_result(frame=frame, state=state())
    scheduler.record_chunk_executed()
    assert not scheduler.decide(frame=frame, state=state()).invoke
    decision = scheduler.decide(
        frame=frame, state=state(), execution_event="procedure_boundary"
    )
    assert decision.invoke
    assert decision.reason == "execution_event:procedure_boundary"


def test_legacy_mode_cannot_double_count_execution_receipts():
    agent = planner("press the button then pick up the block", "legacy_predictions")
    with pytest.raises(ValueError, match="requires execution_chunks"):
        agent.record_executed_chunk("press the button", [state()], chunk_id=1, start_step=0)


def observing_planner():
    return GroundedPlannerClient(
        endpoint="http://planner.test", model="test", timeout_s=1.0,
        demo_video=None, task_name="PickXtimes",
        task_goal="pick up and put down the green cube three times then press the button",
        execution_feedback="execution_chunks", procedure_authority="observe_only",
        relational_color_grounding_enabled=True,
    )


def test_gripper_closure_cannot_force_put_or_freeze_grounding(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "pick up the green cube at <100, 100> for the first time"
            }}]}).encode()

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", respond)
    locations = iter(([90, 90], [80, 80]))
    monkeypatch.setattr(
        "scripts.run_robomme_vlm_groundsg.relational_color_grounding",
        lambda *args: {"tool": "test_rgb_location", "point": next(locations)},
    )
    agent = observing_planner()
    agent.observe_robot_state(state())
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    first = agent.infer(frame, robot_state=state())
    receipt = agent.record_executed_chunk(
        first["grounded_subgoal"], [state(0.018)], chunk_id=1, start_step=0
    )
    assert receipt["execution_event"] == "gripper_review"
    second = agent.infer(frame, robot_state=state(0.018))
    assert second["grounded_subgoal"] == "pick up the green cube for the first time at <80, 80>"
    assert second["monitor_fallback"] is None
    assert second["repair_attempts"] == 0
    assert second["progress_update"] is None
    assert len(requests) == 2
    prompt = requests[-1]["messages"][1]["content"][0]["text"]
    assert "Repeated procedure constraint" not in prompt
    assert "not completed task repetitions" in prompt
    assert not receipt["semantic_success_verified"]


def test_unchanged_gripper_evidence_does_not_repeatedly_force_planning():
    agent = observing_planner()
    agent.observe_robot_state(state())
    first = agent.record_executed_chunk(
        "pick up the green cube", [state(0.018)], chunk_id=1, start_step=0
    )
    second = agent.record_executed_chunk(
        "pick up the green cube", [state(0.018)], chunk_id=2, start_step=1
    )
    assert first["execution_event"] == "gripper_review"
    assert second["execution_event"] is None
    assert second["grasp_observed"] and not second["semantic_success_verified"]


@pytest.mark.parametrize("task,feedback", [
    ("MoveCube", "execution_chunks"), ("PickXtimes", "legacy_predictions"),
])
def test_observe_only_is_not_silently_enabled_on_unadmitted_paths(task, feedback):
    with pytest.raises(ValueError, match="admitted only"):
        GroundedPlannerClient(
            endpoint="http://planner.test", model="test", timeout_s=1.0,
            demo_video=None, task_name=task, task_goal="pick up the block",
            execution_feedback=feedback, procedure_authority="observe_only",
        )
