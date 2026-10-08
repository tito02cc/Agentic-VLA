"""Read-only robot kinematics admission, not an object planner or motion tool.

Pinocchio owns URDF FK; SciPy solves small bounded numerical IK probes. Neither
candidate generation nor a successful FK check authorizes physical execution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose_matrix(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("pose requires finite xyz and wxyz")
    if abs(np.linalg.norm(value[3:]) - 1) > 1e-4:
        raise ValueError("pose quaternion must be unit wxyz")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = value[:3]
    return result


class RobotArmKinematics:
    """Six measured arm joints to the configured link, excluding fingers."""

    def __init__(self, urdf, joint_names, *, base="base_link", tip="link6"):
        import pinocchio as pin

        self.pin = pin
        self.urdf = Path(urdf)
        self.model = pin.buildModelFromUrdf(str(self.urdf))
        self.data = self.model.createData()
        names = tuple(joint_names)
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("exactly six distinct arm joints required")
        if not self.model.existFrame(base) or not self.model.existFrame(tip):
            raise ValueError("unknown robot base or tip frame")
        self.base_id, self.tip_id = self.model.getFrameId(base), self.model.getFrameId(tip)
        ids = [self.model.getJointId(name) for name in names]
        if any(i == 0 or i >= self.model.njoints for i in ids):
            raise ValueError("unknown arm joint")
        if any(self.model.joints[i].nq != 1 or self.model.joints[i].nv != 1 for i in ids):
            raise ValueError("only scalar bounded arm joints supported")
        # Read topology without Boost vector converters shared by Isaac plugins.
        parents = {j.find("child").attrib["link"]: j
                   for j in ET.parse(self.urdf).getroot().findall("joint")}
        chain, visited, cursor = [], set(), tip
        while cursor != base:
            if cursor in visited or cursor not in parents:
                raise ValueError("arm joint list does not match base-to-tip chain")
            visited.add(cursor)
            joint = parents[cursor]
            if joint.attrib["type"] != "fixed":
                chain.append(joint.attrib["name"])
            cursor = joint.find("parent").attrib["link"]
        if set(chain) != set(names):
            raise ValueError("arm joint list does not match base-to-tip chain")
        self.indices = [self.model.joints[i].idx_q for i in ids]
        self.lower = self.model.lowerPositionLimit[self.indices].copy()
        self.upper = self.model.upperPositionLimit[self.indices].copy()
        self.urdf_sha256 = hashlib.sha256(self.urdf.read_bytes()).hexdigest()

    def matrix(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("six finite joint positions required")
        q = self.pin.neutral(self.model)
        q[self.indices] = joints
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.base_id].inverse() * self.data.oMf[self.tip_id]).homogeneous.copy()

    def translation_candidate(self, joints, delta_base_m):
        """Probe a <=2 cm base-frame translation with fixed orientation.

        No gripper change, scene geometry, collision test, or physical command.
        Rejection never returns a dispatchable joint target.
        """
        q = np.asarray(joints, dtype=float)
        current = self.matrix(q)
        delta = np.asarray(delta_base_m, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .02 + 1e-12:
            raise ValueError("translation must be finite and within 2 cm")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("current joints outside URDF limits")
        goal = current.copy()
        goal[:3, 3] += delta

        def error(target):
            predicted = self.matrix(target)
            return np.r_[predicted[:3, 3] - goal[:3, 3],
                         .15 * Rotation.from_matrix(predicted[:3, :3] @ goal[:3, :3].T).as_rotvec()]

        lower, upper = np.maximum(self.lower, q - .05), np.minimum(self.upper, q + .05)
        fit = least_squares(error, q, bounds=(lower, upper), max_nfev=80,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        residual = error(fit.x)
        position_error = float(np.linalg.norm(residual[:3]))
        rotation_error = float(np.linalg.norm(residual[3:]) / .15)
        accepted = bool(fit.success and position_error <= .001 and rotation_error <= .005)
        return {
            "kinematic_candidate_accepted": accepted,
            "joint_target": fit.x.tolist() if accepted else None,
            "requested_delta_base_m": delta.tolist(),
            "position_residual_m": position_error, "rotation_residual_rad": rotation_error,
            "max_joint_delta_rad": float(np.max(np.abs(fit.x - q))),
            "solver_evaluations": int(fit.nfev), "solver_status": int(fit.status),
            "physical_execution_authorized": False, "collision_checked": False,
            "semantic_recovery_confirmed": False,
        }


def audit_robot_kinematics(environment, public_state):
    """Read only robot geometry/proprioception; never scene objects or rewards.

    Measured robot link poses are used solely for host-side calibration. They
    are not added to VLA/Planner observations or admitted as a new sensor.
    """
    state = np.asarray(public_state, dtype=float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("public state must have 14 finite entries")
    manager = environment.robot_manager
    robots = [r for r in manager.robot_list if r.type == "target"]
    arms = {r.arm_name.split("_")[0]: r for r in robots}
    if len(robots) != 2 or set(arms) != {"left", "right"}:
        raise ValueError("expected exactly two ARX arm instances")
    reports = {}
    for name, offset in (("left", 0), ("right", 7)):
        robot = arms[name]
        kin = RobotArmKinematics(robot.urdf_path, robot.arm_joints_name,
                                 base=robot.base_link, tip=robot.ee_link_name)
        q = state[offset:offset + 6]
        measured_q = np.asarray(manager.get_joint(robot)[0])
        if not np.allclose(q, measured_q, atol=1e-5, rtol=0):
            raise ValueError("public state and measured arm state differ")
        root = pose_matrix(manager.get_link_pose(robot, robot.base_link, is_relative=True)[0])
        measured = pose_matrix(manager.get_real_endpose(robot, is_relative=True)[0])
        predicted = root @ kin.matrix(q)
        position_error = float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
        rotation_error = float(Rotation.from_matrix(predicted[:3, :3] @ measured[:3, :3].T).magnitude())
        passed = position_error <= .002 and rotation_error <= .01
        probes = []
        if passed:
            for delta in (np.zeros(3), *(.002 * np.eye(3)), *(-.002 * np.eye(3))):
                probes.append(kin.translation_candidate(q, delta))
        reports[name] = {
            "urdf_path": str(kin.urdf.resolve()), "urdf_sha256": kin.urdf_sha256,
            "pinocchio_version": kin.pin.__version__, "base_link": robot.base_link,
            "tip_link": robot.ee_link_name, "joint_names": list(robot.arm_joints_name),
            "position_error_m": position_error, "rotation_error_rad": rotation_error,
            "fk_passed": passed, "measured_tip_xyz": measured[:3, 3].tolist(),
            "predicted_tip_xyz": predicted[:3, 3].tolist(), "translation_probes": probes,
        }
    return {"arms": reports, "fk_passed": all(r["fk_passed"] for r in reports.values()),
            "physical_probe_steps": 0, "physical_execution_authorized": False,
            "scope": "robot-only FK calibration and numerical IK; no collision or task recovery validation"}
