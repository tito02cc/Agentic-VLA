"""Immutable, observation-bound VLA proposals and robot-only motion previews."""

from dataclasses import dataclass
import hashlib
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class VlaActionProposal:
    proposal_id: str
    timestep: int
    observation_sha256: str
    instruction: str
    inference_wall_ms: float
    dtype: str
    data: bytes

    @classmethod
    def create(cls, actions, *, timestep, observation_sha256, instruction, inference_wall_ms):
        actions = np.asarray(actions)
        if actions.shape != (50, 14) or actions.dtype.kind != "f" or not np.isfinite(actions).all():
            raise ValueError("official reference policy must return finite [50, 14] joint targets")
        return cls(uuid4().hex, timestep, observation_sha256, instruction, inference_wall_ms,
                   actions.dtype.str, actions.tobytes(order="C"))

    def actions(self):
        # Backed by immutable bytes, not a writeable view owned by a model client.
        return np.frombuffer(self.data, dtype=self.dtype).reshape(50, 14)

    def identity(self):
        return {"proposal_id": self.proposal_id, "timestep": self.timestep,
                "observation_sha256": self.observation_sha256,
                "action_sha256": hashlib.sha256(self.data).hexdigest(),
                "action_dtype": self.dtype, "generated_steps": 50}


def robot_motion_preview(port, proposal):
    from scipy.spatial.transform import Rotation
    from .robodojo_kinematics import audit_robot_kinematics
    from .robodojo_motion_probe import robot_geometry

    observation = port.observe()
    if observation.fingerprint() != proposal.observation_sha256:
        raise ValueError("stale VLA motion preview")
    if not audit_robot_kinematics(port.environment, observation.state)["fk_passed"]:
        raise ValueError("robot coordinate check failed")
    result = {**proposal.identity(), "coordinate_frame": "environment frame, meters; not image pixels",
        "meaning": "FK of proposed robot joint targets, NOT measured future motion or object prediction",
        "collision_checked": False, "semantic_outcome_verified": False,
        "gripper_units": "raw prediction may exceed [0,1]; command is clipped to [0,1] by executor, not measured contact or finger gap",
        "selectable_prefixes": [10, 50], "arms": {}}
    actions = proposal.actions()
    for arm, offset in (("left", 0), ("right", 7)):
        kin, root = robot_geometry(port, arm)
        current = root @ kin.matrix(observation.state[offset:offset + 6])
        poses = [root @ kin.matrix(row[offset:offset + 6]) for row in actions]
        samples = []
        for prefix in (1, 10, 25, 50):
            pose = poses[prefix - 1]
            samples.append({"prefix_steps": prefix,
                "predicted_tip_xyz_m": pose[:3, 3].tolist(),
                "delta_xyz_m": (pose[:3, 3] - current[:3, 3]).tolist(),
                "rotation_delta_rad": float(Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude()),
                "gripper_prediction_raw": float(actions[prefix - 1, offset + 6]),
                "gripper_command": float(np.clip(actions[prefix - 1, offset + 6], 0, 1))})
        positions = np.array([current[:3, 3], *(pose[:3, 3] for pose in poses)])
        result["arms"][arm] = {"current_tip_xyz_m": current[:3, 3].tolist(), "samples": samples,
            "max_adjacent_tip_distance_m": float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
            "out_of_urdf_limit_rows": int(np.any((actions[:, offset:offset + 6] < kin.lower)
                | (actions[:, offset:offset + 6] > kin.upper), axis=1).sum()),
            "urdf_sha256": kin.urdf_sha256}
    return result


def run_proposal_review_smoke(episode):
    """Two real proposal reviews, no forced choice and no full-task claim."""
    import dataclasses
    import time

    if episode.ran:
        raise RuntimeError("SortingEpisode is single-use")
    if (not episode.execution.agent_enabled or not episode.execution.proposal_review_enabled
            or episode.execution.vla_call_budget != 2 or episode.execution.subgoal_conditioning_admitted
            or episode.execution.feedback_refresh_enabled):
        raise ValueError("proposal review smoke requires Agent on, original task and two-call budget")
    episode.ran = True
    start, reviews = time.perf_counter(), []
    try:
        episode._capture()
        for index in range(2):
            if episode.stopped or episode.port.done():
                break
            proposal = episode._prepare_vla_proposal(episode.instruction)
            preview_start = time.perf_counter()
            preview = robot_motion_preview(episode.port, proposal)
            preview_ms = (time.perf_counter() - preview_start) * 1000
            episode.session.workspace.append_event("vla_motion_preview", {
                "preview": preview, "preview_wall_ms": preview_ms}, source="host_robot_kinematics")
            specs = tuple({"skill_id": f"execute_prefix_{n}",
                "description": f"Execute first {n} targets of THIS paid VLA proposal, discard remaining {50-n}. No new inference or changed instruction.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "cost": {"additional_vla_calls": 0, "control_steps_max": n}}
                for n in (10, 50))
            context = dataclasses.replace(episode._context("proposal_review"), action_proposal=preview,
                available_skills=tuple(s["skill_id"] for s in specs), available_skill_specs=specs,
                allowed_intents=("run_skill", "safe_stop"), compact_task_only_review=True)
            transition = (episode.session.start(context) if index == 0 else
                episode.session.request_semantic_checkpoint(context, reason="fresh paid VLA proposal", count_as_retry=False))
            if transition.ticket is not None and transition.planner is None:
                transition = episode.session.await_planner()
            if (transition.planner is None or transition.timed_out or transition.stale
                    or not transition.planner.result.accepted or not transition.decision_applied):
                episode._stop("proposal reviewer rejected, unavailable or stale", episode._tool_context())
                reviews.append({"proposal_id": proposal.proposal_id, "accepted": False})
                break
            decision = transition.planner.result.decision
            result = episode.session.apply_planner_transition(transition, context=episode._tool_context())
            reviews.append({"proposal_id": proposal.proposal_id, "accepted": result is not None and result.accepted,
                            "decision": decision.to_dict(), "preview_wall_ms": preview_ms})
            if result is None or not result.accepted:
                episode._stop("proposal execution rejected", episode._tool_context())
                break
            outcome = episode.session.tools.primitive_outcome(result, episode_id=episode.config.run_id)
            if outcome is not None:
                episode.last_primitive = outcome.to_planner_dict()
                if episode.execution.memory_enabled:
                    episode.memory.remember_execution(outcome)
                episode.session.record_primitive_result(result, planner_context=episode._context("tool_return"))
        episode._stop(episode.stop_reason or "two proposal reviews completed", episode._tool_context())
        return {**episode._summary(start), "proposal_reviews": reviews,
            "planner_choice_forced": False, "eef_correction_enabled": False,
            "scope": "autonomous prefix choices in a bounded integration test; not recovery efficacy or full benchmark"}
    finally:
        episode._discard_vla_proposal("diagnostic_end")
        episode.session.close(reason="proposal_review_smoke_end")
