"""Numerical execution facts only; no task-success classification."""

import numpy as np
import pytest
import json

from scripts.audit_robodojo_execution_feedback import audit_run, array_hash

from agentic_vla.benchmarks.robodojo_sorting import joint_target_feedback, execution_only_receipt


def test_splits_arm_radians_from_normalized_grippers():
    command = np.zeros(14)
    observed = np.zeros(14)
    command[1], command[7], command[6], command[13] = .3, -.6, 2., -1.
    observed[6], observed[13] = .8, .1
    result = joint_target_feedback(command, observed)
    assert result["left"]["arm_mean_abs_error_rad"] == pytest.approx(.05)
    assert result["right"]["arm_max_abs_error_rad"] == pytest.approx(.6)
    assert result["left"]["gripper_command_raw"] == 2
    assert result["left"]["gripper_command_effective"] == 1
    assert result["left"]["gripper_abs_error_normalized"] == pytest.approx(.2)
    assert result["right"]["gripper_abs_error_normalized"] == pytest.approx(.1)
    assert np.array_equal(command[[6, 13]], [2, -1])
    assert result["semantic_outcome"] == "not_verified"


@pytest.mark.parametrize("invalid", [np.zeros(13), np.zeros((1, 14)),
                                     np.full(14, np.nan), np.full(14, np.inf)])
@pytest.mark.parametrize("side", ["command", "observed"])
def test_bad_feedback_input_is_rejected(invalid, side):
    values = {"command": np.zeros(14), "observed": np.zeros(14)}
    values[side] = invalid
    with pytest.raises(ValueError, match="finite 14-D"):
        joint_target_feedback(**values)


def test_feedback_survives_memory_quarantine_without_model_postcheck():
    feedback = joint_target_feedback(np.zeros(14), np.zeros(14))
    receipt = {"metadata": {"joint_target_feedback": feedback,
                             "postcheck": {"status": "confirmed"}}}
    projected = execution_only_receipt(receipt)
    assert projected["metadata"]["joint_target_feedback"] == feedback
    assert "postcheck" not in projected["metadata"]
    assert "postcheck" in receipt["metadata"]


@pytest.fixture
def recorded_run(tmp_path):
    session = tmp_path / "sessions/fixture"
    policy = tmp_path / "policy"
    session.mkdir(parents=True)
    policy.mkdir()
    observed = np.full(14, .12, dtype=np.float32)
    traces, events = [], []
    for i in range(2):
        actions = np.full((50, 14), .1, dtype=np.float32)
        name = f"actions_{i + 1}.npy"
        np.save(policy / name, actions, allow_pickle=False)
        traces.append({"call": i + 1, "generation": 1, "actions_file": name,
                       "actions_sha256": array_hash(actions),
                       "input_audit": {"state": {"shape": [14], "dtype": "float32",
                                                 "sha256": array_hash(observed)}}})
        events.append({"event_type": "pi05_chunk_executed", "sequence": i,
                       "payload": {"started_timestep": i * 10, "ended_timestep": (i + 1) * 10,
                                   "executed_actions": 10, "generated_actions": 50, "discarded_actions": 40}})
    (tmp_path / "summary.json").write_text(json.dumps({"run_id": "fixture", "execution_summary": {"vla_calls": 2}}))
    (policy / "policy_trace.jsonl").write_text("\n".join(json.dumps(t) for t in traces))
    (session / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (session / "transcript.jsonl").write_text(json.dumps({
        "role": "user", "metadata": {"episode_id": "fixture", "timestep": 10},
        "content": json.dumps({"timestep": 10, "robot_state": observed.tolist()})}))
    return tmp_path


def test_recorded_audit_binds_exact_endpoint_and_excludes_missing(recorded_run):
    result = audit_run(recorded_run)
    assert result["bound_endpoints"] == 1
    assert result["max_arm_abs_error_rad"] == pytest.approx(.02)
    assert result["skipped"][0]["ended_timestep"] == 20
    assert result["source_drift"] == []
    assert result["new_robot_actions"] == 0


def test_recorded_audit_rejects_changed_actions(recorded_run):
    np.save(recorded_run / "policy/actions_1.npy", np.zeros((50, 14), np.float32))
    with pytest.raises(ValueError, match="changed action"):
        audit_run(recorded_run)


def test_recorded_audit_rejects_wrong_state_binding(recorded_run):
    path = recorded_run / "policy/policy_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["input_audit"]["state"]["sha256"] = "0" * 64
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="does not match"):
        audit_run(recorded_run)


def test_intervening_tool_is_not_silently_treated_as_same_endpoint(recorded_run):
    path = recorded_run / "sessions/fixture/events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["payload"].update(started_timestep=11, ended_timestep=21)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = audit_run(recorded_run)
    assert result["bound_endpoints"] == 0
    assert result["skipped"][0]["reason"] == "intervening_tool_or_no_next_inference_boundary"
    assert result["max_arm_abs_error_rad"] is None
