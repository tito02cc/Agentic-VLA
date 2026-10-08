"""Numerical/fake-port tests, not physical recovery evidence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pinocchio")
from agentic_vla.benchmarks import robodojo_motion_probe as probe
from agentic_vla.benchmarks.robodojo_kinematics import RobotArmKinematics
from agentic_vla.benchmarks.robodojo_sorting import SortingObservation


URDF = Path(__file__).resolve().parents[1] / "third_party/robodojo_official/Assets/Robots/x5/X5A.urdf"
NAMES = tuple(f"joint{i}" for i in range(1, 7))


@pytest.fixture
def episode():
    if not URDF.is_file():
        pytest.skip("RoboDojo assets not installed")
    kin = RobotArmKinematics(URDF, NAMES)
    robots = [SimpleNamespace(type="target", arm_name=f"{name}_arm", urdf_path=URDF,
        arm_joints_name=NAMES, base_link="base_link", ee_link_name="link6") for name in ("left", "right")]
    class Port:
        def __init__(self):
            self.state = np.tile([.1, -.2, .3, -.4, .1, .2, .3], 2)
            self.steps = 0
            self.actions = []
            self.stalled = False
            self.drift = False
        def observe(self):
            frame = np.full((4, 4, 3), self.steps, np.uint8)
            return SortingObservation("original task", self.state.copy(),
                {name: frame for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")})
        def execute(self, target):
            self.actions.append(target.copy())
            if not self.stalled:
                self.state = target.copy()
            if self.drift:
                self.state[0] += .02
            self.steps += 1
        def done(self):
            return False
    port = Port()
    def q(robot):
        offset = 0 if robot.arm_name.startswith("left") else 7
        return port.state[offset:offset + 6]
    def tip(robot, **kwargs):
        mat = kin.matrix(q(robot))
        xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        return {0: np.r_[mat[:3, 3], xyzw[3], xyzw[:3]]}
    port.environment = SimpleNamespace(robot_manager=SimpleNamespace(robot_list=robots,
        get_joint=lambda r: {0: q(r)}, get_real_endpose=tip,
        get_link_pose=lambda *a, **k: {0: np.array([0, 0, 0, 1, 0, 0, 0])}))
    events = []
    obj = SimpleNamespace(port=port, step=0, stopped=False, execution=SimpleNamespace(agent_enabled=False),
        task=SimpleNamespace(max_steps=1000), session=SimpleNamespace(workspace=SimpleNamespace(
            append_event=lambda kind, data, **kw: events.append((kind, data)))), events=events)
    obj._capture = lambda: setattr(obj, "observation", port.observe())
    obj._capture()
    return obj


def test_probe_tracks_and_preserves_other_arm_and_grippers(episode):
    proposal = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    receipt = probe.execute_translation_probe(episode, proposal)
    assert receipt["tracking_passed"] and len(receipt["samples"]) == 10
    assert not receipt["collision_checked"] and not receipt["semantic_recovery_confirmed"]
    for command in episode.port.actions:
        np.testing.assert_array_equal(command[:7], proposal.initial_state[:7])
        assert command[13] == proposal.initial_state[13]
    assert episode.step == 10
    with pytest.raises(ValueError):
        probe.execute_translation_probe(episode, proposal)


def test_vla_motion_preview_is_robot_only_and_does_not_execute(episode):
    from agentic_vla.benchmarks.robodojo_action_proposal import VlaActionProposal, robot_motion_preview
    from agentic_vla.runtime.knowledge import reject_privileged_semantic_fields
    actions = np.tile(episode.port.state, (50, 1))
    actions[:, 8] += np.linspace(0, .01, 50)
    p = VlaActionProposal.create(actions, timestep=0, observation_sha256=episode.observation.fingerprint(),
        instruction=episode.observation.instruction, inference_wall_ms=1)
    view = robot_motion_preview(episode.port, p)
    reject_privileged_semantic_fields(view)
    assert not view["collision_checked"] and not view["semantic_outcome_verified"]
    assert view["selectable_prefixes"] == [10, 50]
    assert [s["prefix_steps"] for s in view["arms"]["right"]["samples"]] == [1, 10, 25, 50]
    assert view["arms"]["left"]["max_adjacent_tip_distance_m"] == 0
    assert view["arms"]["right"]["max_adjacent_tip_distance_m"] > 0
    assert not episode.port.actions
    episode.port.steps += 1
    with pytest.raises(ValueError, match="stale"):
        robot_motion_preview(episode.port, p)


def test_preview_distinguishes_raw_and_clipped_gripper_without_changing_proposal(episode):
    from agentic_vla.benchmarks.robodojo_action_proposal import VlaActionProposal, robot_motion_preview
    actions = np.tile(episode.port.state, (50, 1))
    actions[:, 6], actions[:, 13] = 1.003, -.002
    p = VlaActionProposal.create(actions, timestep=0, observation_sha256=episode.observation.fingerprint(),
        instruction=episode.observation.instruction, inference_wall_ms=1)
    original = p.data
    view = robot_motion_preview(episode.port, p)
    for arm, raw, clipped in (("left", 1.003, 1.), ("right", -.002, 0.)):
        for sample in view["arms"][arm]["samples"]:
            assert sample["gripper_prediction_raw"] == raw
            assert sample["gripper_command"] == clipped
    assert p.data == original
    np.testing.assert_array_equal(p.actions(), actions)
    assert not episode.port.actions


@pytest.mark.parametrize("delta", [[0, 0, .004], [np.nan, 0, 0], [0, 0]])
def test_reject_large_or_invalid_move_without_actions(episode, delta):
    with pytest.raises(ValueError):
        probe.prepare_translation(episode.port, "right", delta)
    assert not episode.port.actions


def test_stale_image_rejects_before_action(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    episode.port.steps = 1
    with pytest.raises(ValueError, match="stale"):
        probe.execute_translation_probe(episode, p)
    assert not episode.port.actions


@pytest.mark.parametrize("index", [0, 6, 13])
def test_other_arm_and_gripper_tampering_rejected(episode, index):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    target = list(p.target_state)
    target[index] += .001
    with pytest.raises(ValueError, match="held"):
        probe.execute_translation_probe(episode, replace(p, target_state=tuple(target)))
    assert not episode.port.actions


def test_goal_tampering_rejected(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    with pytest.raises(ValueError, match="inconsistent"):
        probe.execute_translation_probe(episode, replace(p, delta_world_m=(0, 0, 0)))


def test_drift_stops_after_first_bad_feedback(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    episode.port.drift = True
    r = probe.execute_translation_probe(episode, p)
    assert not r["tracking_passed"] and r["failure"] == "tracking_guard"
    assert episode.step == 1


def test_stalled_endpoint_not_claimed_as_pass(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    episode.port.stalled = True
    r = probe.execute_translation_probe(episode, p)
    assert not r["tracking_passed"] and r["failure"] == "endpoint_tracking"


def test_diagnostic_cannot_run_as_agent_tool(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    episode.execution.agent_enabled = True
    with pytest.raises(ValueError, match="diagnostic"):
        probe.execute_translation_probe(episode, p)
    assert not episode.port.actions


def test_budget_rejects_before_action(episode):
    p = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    episode.step = 995
    with pytest.raises(ValueError, match="budget"):
        probe.execute_translation_probe(episode, p)
    assert not episode.port.actions


def test_new_proposal_allowed_at_revisited_observation_but_old_token_cannot_replay(episode):
    first = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    probe.execute_translation_probe(episode, first)
    # An exact revisited state is legal; a consumed command is not.
    episode.port.state = np.asarray(first.initial_state)
    episode.port.steps = 0
    second = probe.prepare_translation(episode.port, "right", [0, 0, .002])
    assert second.observation_sha256 == first.observation_sha256
    assert probe.execute_translation_probe(episode, second)["tracking_passed"]
    episode.port.state = np.asarray(first.initial_state)
    episode.port.steps = 0
    before = len(episode.port.actions)
    with pytest.raises(ValueError, match="consumed"):
        probe.execute_translation_probe(episode, first)
    assert len(episode.port.actions) == before


@pytest.mark.parametrize("approval", [None, {"approved": True, "run_id": "wrong"}])
def test_visual_gate_requires_matching_review(episode, tmp_path, approval):
    import json
    episode.config = SimpleNamespace(workspace_path=tmp_path, run_id="fixture")
    if approval is not None:
        (tmp_path / "motion_visual_admission.json").write_text(json.dumps(approval))
    with pytest.raises((ValueError, TimeoutError)):
        probe.wait_for_visual_admission(episode, timeout_s=.01 if approval else 0)
    assert not episode.port.actions


def test_visual_gate_accepts_exact_snapshot_only(episode, tmp_path):
    import json
    episode.config = SimpleNamespace(workspace_path=tmp_path, run_id="fixture")
    receipt = {"run_id": "fixture", "observation_sha256": episode.observation.fingerprint(),
               "approved": True, "reviewer": "test-fixture"}
    (tmp_path / "motion_visual_admission.json").write_text(json.dumps(receipt))
    assert probe.wait_for_visual_admission(episode, timeout_s=1) == receipt
    assert not episode.port.actions


@pytest.mark.parametrize("tracking_failure", [False, True])
def test_runner_resumes_original_vla_only_after_three_passes(episode, monkeypatch, tracking_failure):
    episode.ran = False
    episode.execution.vla_call_budget = 1
    episode.execution.subgoal_conditioning_admitted = False
    episode.execution.feedback_refresh_enabled = False
    episode.session.close = lambda **kw: None
    episode.instruction = "original task"
    episode._tool_context = lambda: None
    episode._stop = lambda *args: setattr(episode, "stopped", True)
    episode._summary = lambda start: {"steps": episode.step}
    calls = []
    episode._vla_act = lambda instruction, context: calls.append((instruction, episode.step))
    monkeypatch.setattr(probe, "wait_for_visual_admission", lambda *a: None)
    episode.port.drift = tracking_failure
    summary = probe.run_motion_probe_smoke(episode)
    assert summary["motion_probe_passed"] is not tracking_failure
    assert calls == ([] if tracking_failure else [("original task", 30)])
    assert not summary["autonomous_planner_used"]
    with pytest.raises(RuntimeError, match="single-use"):
        probe.run_motion_probe_smoke(episode)
