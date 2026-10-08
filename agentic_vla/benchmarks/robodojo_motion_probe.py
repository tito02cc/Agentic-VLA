"""Opt-in simulation motion admission, not an autonomous recovery skill.

Only robot geometry and public proprioception are used. Visual clearance is a
separate, observation-bound review, not inferred from successful numerical IK.
"""

from dataclasses import dataclass, asdict, field
import json
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from .robodojo_kinematics import RobotArmKinematics, audit_robot_kinematics, pose_matrix


@dataclass(frozen=True)
class TranslationProposal:
    arm: str
    observation_sha256: str
    initial_state: tuple
    target_state: tuple
    goal_world_matrix: tuple
    delta_world_m: tuple
    urdf_sha256: str
    proposal_id: str = field(default_factory=lambda: uuid4().hex)


def robot_geometry(port, arm):
    robots = [r for r in port.environment.robot_manager.robot_list
              if r.type == "target" and r.arm_name.split("_")[0] == arm]
    if arm not in {"left", "right"} or len(robots) != 1:
        raise ValueError("one known robot arm required")
    robot = robots[0]
    kin = RobotArmKinematics(robot.urdf_path, robot.arm_joints_name,
                             base=robot.base_link, tip=robot.ee_link_name)
    pose = port.environment.robot_manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
    return kin, pose_matrix(pose)


def prepare_translation(port, arm, delta_world_m):
    observation = port.observe()
    audit = audit_robot_kinematics(port.environment, observation.state)
    if not audit["fk_passed"]:
        raise ValueError("robot calibration failed")
    delta = np.asarray(delta_world_m, dtype=float)
    # Development probe: 2 mm outward, <=3 mm measured return; not a general tool.
    if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .003 + 1e-12:
        raise ValueError("motion probe translation must be within 3 mm")
    kin, root = robot_geometry(port, arm)
    offset = 0 if arm == "left" else 7
    q = observation.state[offset:offset + 6]
    candidate = kin.translation_candidate(q, root[:3, :3].T @ delta)
    if not candidate["kinematic_candidate_accepted"]:
        raise ValueError("numerical IK rejected motion probe")
    target = np.asarray(observation.state, dtype=float).copy()
    target[offset:offset + 6] = candidate["joint_target"]
    goal = root @ kin.matrix(q)
    goal[:3, 3] += delta
    return TranslationProposal(arm, observation.fingerprint(), tuple(observation.state.tolist()),
        tuple(target.tolist()), tuple(tuple(row) for row in goal.tolist()), tuple(delta.tolist()), kin.urdf_sha256)


def validate_proposal(port, proposal):
    observation = port.observe()
    initial, target = np.asarray(proposal.initial_state), np.asarray(proposal.target_state)
    goal, delta = np.asarray(proposal.goal_world_matrix), np.asarray(proposal.delta_world_m)
    if (not isinstance(proposal.proposal_id, str) or not proposal.proposal_id
            or proposal.arm not in {"left", "right"} or initial.shape != (14,) or target.shape != (14,)
            or goal.shape != (4, 4) or delta.shape != (3,)
            or not all(np.isfinite(v).all() for v in (initial, target, goal, delta))):
        raise ValueError("invalid motion proposal")
    if observation.fingerprint() != proposal.observation_sha256 or not np.array_equal(observation.state, initial):
        raise ValueError("stale motion proposal")
    offset = 0 if proposal.arm == "left" else 7
    moving = np.arange(offset, offset + 6)
    held = np.setdiff1d(np.arange(14), moving)
    kin, root = robot_geometry(port, proposal.arm)
    if kin.urdf_sha256 != proposal.urdf_sha256:
        raise ValueError("robot geometry changed")
    if (not np.array_equal(initial[held], target[held])
            or np.max(np.abs(target[moving] - initial[moving])) > .05 + 1e-12
            or np.any(target[moving] < kin.lower) or np.any(target[moving] > kin.upper)
            or np.linalg.norm(delta) > .003 + 1e-12):
        raise ValueError("motion proposal exceeds bounds or changes held joints")
    expected = root @ kin.matrix(initial[moving])
    expected[:3, 3] += delta
    if not np.allclose(goal, expected, atol=1e-9, rtol=0):
        raise ValueError("inconsistent motion goal")
    predicted = root @ kin.matrix(target[moving])
    if (np.linalg.norm(predicted[:3, 3] - goal[:3, 3]) > .001
            or Rotation.from_matrix(predicted[:3, :3] @ goal[:3, :3].T).magnitude() > .005):
        raise ValueError("motion target does not realize goal")
    return kin, root, moving, held


def execute_translation_probe(episode, proposal):
    """Ten bounded commands with fresh tracking feedback, test driver only."""
    if episode.execution.agent_enabled or episode.stopped or episode.port.done():
        raise ValueError("motion probe requires active diagnostic-only episode")
    if episode.step + 10 > episode.task.max_steps:
        raise ValueError("insufficient control budget")
    consumed = getattr(episode, "_consumed_motion_proposals", set())
    if proposal.proposal_id in consumed:
        raise ValueError("motion proposal already consumed")
    kin, root, moving, held = validate_proposal(episode.port, proposal)
    consumed.add(proposal.proposal_id)
    episode._consumed_motion_proposals = consumed
    initial, target = np.asarray(proposal.initial_state), np.asarray(proposal.target_state)
    goal = np.asarray(proposal.goal_world_matrix)
    record = {"proposal": asdict(proposal), "started_timestep": episode.step, "samples": [],
              "tracking_passed": False, "collision_checked": False, "semantic_recovery_confirmed": False}
    episode.session.workspace.append_event("motion_probe_prepared", record["proposal"], source="test_driver")
    try:
        for _ in range(10):
            if episode.port.done():
                record["failure"] = "environment_terminated"
                break
            episode.port.execute(target)
            episode.step += 1
            episode._capture()
            observed = episode.observation.state
            tip = root @ kin.matrix(observed[moving])
            sample = {"timestep": episode.step, "observed_state": observed.tolist(),
                "observation_sha256": episode.observation.fingerprint(),
                "position_error_m": float(np.linalg.norm(tip[:3, 3] - goal[:3, 3])),
                "rotation_error_rad": float(Rotation.from_matrix(tip[:3, :3] @ goal[:3, :3].T).magnitude()),
                "held_joint_drift": float(np.max(np.abs(observed[held] - initial[held]))),
                "moving_joint_travel_rad": float(np.max(np.abs(observed[moving] - initial[moving]))),
                "joint_tracking_error_rad": float(np.max(np.abs(observed[moving] - target[moving])))}
            record["samples"].append(sample)
            # Tracking bounds cannot certify clearance or the absence of contact.
            if (not np.isfinite(observed).all() or sample["held_joint_drift"] > .01
                    or sample["moving_joint_travel_rad"] > .06 or sample["position_error_m"] > .005):
                record["failure"] = "tracking_guard"
                break
        if len(record["samples"]) == 10 and "failure" not in record:
            last = record["samples"][-1]
            record["tracking_passed"] = (last["position_error_m"] <= .001
                and last["rotation_error_rad"] <= .01 and last["joint_tracking_error_rad"] <= .01)
            if not record["tracking_passed"]:
                record["failure"] = "endpoint_tracking"
        return record
    finally:
        record["ended_timestep"] = episode.step
        episode.session.workspace.append_event("motion_probe_feedback", record, source="test_driver")


def wait_for_visual_admission(episode, *, timeout_s=180):
    """Pause synchronous simulation for explicit review of original RGB only."""
    from PIL import Image

    out = episode.config.workspace_path
    for name, frame in episode.observation.frames.items():
        Image.fromarray(np.asarray(frame)).save(out / f"motion_preflight_{name}.png")
    request = {"run_id": episode.config.run_id, "observation_sha256": episode.observation.fingerprint(),
        "scope": "simulation-only right arm 2 mm world-up and return; not automatic collision checking",
        "review": "Confirm empty grippers, free space around right arm, no visible contact; otherwise reject."}
    (out / "motion_preflight.json").write_text(json.dumps(request, indent=2) + "\n")
    approval = out / "motion_visual_admission.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if approval.exists():
            receipt = json.loads(approval.read_text())
            if (receipt.get("run_id") != request["run_id"]
                    or receipt.get("observation_sha256") != request["observation_sha256"]
                    or receipt.get("approved") is not True or not receipt.get("reviewer")):
                raise ValueError("visual motion admission rejected or stale")
            episode.session.workspace.append_event("motion_visual_admission", receipt, source="test_driver")
            return receipt
        time.sleep(1)
    raise TimeoutError("no observation-bound visual admission")


def run_motion_probe_smoke(episode):
    """Forced round-trip test followed by one fresh native VLA chunk, never an Agent result."""
    if episode.ran:
        raise RuntimeError("SortingEpisode is single-use")
    if (episode.execution.agent_enabled or episode.execution.vla_call_budget != 1
            or episode.execution.subgoal_conditioning_admitted or episode.execution.feedback_refresh_enabled):
        raise ValueError("motion smoke requires Agent off, original instruction and one VLA call")
    episode.ran = True
    start, receipts = time.perf_counter(), []
    passed = False
    try:
        episode._capture()
        wait_for_visual_admission(episode)
        # Hold is measured with the same path rather than assuming a stationary arm.
        for phase in ("hold", "up", "return"):
            kin, root = robot_geometry(episode.port, "right")
            current = root @ kin.matrix(episode.observation.state[7:13])
            if phase == "hold":
                delta = np.zeros(3)
            elif phase == "up":
                origin = current[:3, 3].copy()
                delta = np.array([0., 0., .002])
            else:
                delta = origin - current[:3, 3]
            proposal = prepare_translation(episode.port, "right", delta)
            receipt = execute_translation_probe(episode, proposal)
            receipts.append({"phase": phase, **receipt})
            if not receipt["tracking_passed"]:
                break
        passed = len(receipts) == 3 and all(r["tracking_passed"] for r in receipts)
        if passed:
            episode._vla_act(episode.instruction, episode._tool_context())
        episode._stop("motion probe complete" if passed else "motion tracking failed", episode._tool_context())
        return {**episode._summary(start), "motion_probe_passed": passed, "motion_probe_receipts": receipts,
                "motion_probe_steps": sum(len(r["samples"]) for r in receipts),
                "autonomous_planner_used": False, "semantic_recovery_confirmed": False,
                "scope": "visually admitted simulation kinematics probe; not autonomous recovery or benchmark"}
    finally:
        episode.session.close(reason="motion_probe_smoke_end")
