"""Real URDF numerical tests; not physical tracking or robot success tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio", reason="run robot-geometry tests in the RoboDojo environment")
from agentic_vla.benchmarks.robodojo_kinematics import RobotArmKinematics, audit_robot_kinematics, pose_matrix


URDF = Path(__file__).resolve().parents[1] / "third_party/robodojo_official/third_party/curobo/curobo/content/assets/robot/arx_x5_description/urdf/X5A.urdf"
NAMES = tuple(f"joint{i}" for i in range(1, 7))


@pytest.fixture
def arm():
    if not URDF.is_file():
        pytest.skip("official robot URDF not installed")
    return RobotArmKinematics(URDF, NAMES)


def test_neutral_fk_is_a_finite_rigid_transform(arm):
    value = arm.matrix(np.zeros(6))
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value[3], [0, 0, 0, 1])
    np.testing.assert_allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-9)


def test_joint_order_is_name_bound(arm):
    q = np.arange(6) * .1
    reverse = RobotArmKinematics(URDF, NAMES[::-1])
    np.testing.assert_allclose(reverse.matrix(q[::-1]), arm.matrix(q))


@pytest.mark.parametrize("names,tip", [(NAMES[:-1], "link6"), (("no_joint", *NAMES[1:]), "link6"),
                                      (NAMES, "link5"), (NAMES, "no_link")])
def test_reject_wrong_chain(names, tip):
    with pytest.raises(ValueError):
        RobotArmKinematics(URDF, names, tip=tip)


@pytest.mark.parametrize("q", [np.zeros(5), np.full(6, np.nan)])
def test_invalid_joint_state(arm, q):
    with pytest.raises(ValueError):
        arm.matrix(q)


@pytest.mark.parametrize("delta", [[.021, 0, 0], [np.nan, 0, 0], [0, 0]])
def test_reject_invalid_delta(arm, delta):
    with pytest.raises(ValueError):
        arm.translation_candidate(np.zeros(6), delta)


def test_noop_solver_does_not_grant_execution_authority(arm):
    result = arm.translation_candidate(np.zeros(6), [0, 0, 0])
    assert result["kinematic_candidate_accepted"]
    assert not result["physical_execution_authorized"]
    assert not result["collision_checked"]
    assert result["max_joint_delta_rad"] < 1e-7


def test_local_candidates_respect_bounds_even_if_rejected(arm):
    q = np.array([.1, -.2, .3, -.4, .1, .2])
    for delta in (.002 * np.eye(3)):
        result = arm.translation_candidate(q, delta)
        assert result["max_joint_delta_rad"] <= .05 + 1e-10
        assert result["physical_execution_authorized"] is False
        assert not result["semantic_recovery_confirmed"]
        if result["kinematic_candidate_accepted"]:
            target = np.array(result["joint_target"])
            assert np.all(target >= arm.lower) and np.all(target <= arm.upper)
        else:
            assert result["joint_target"] is None


def test_solver_rejects_outside_current_limits(arm):
    with pytest.raises(ValueError, match="outside"):
        arm.translation_candidate(np.full(6, 10), [0, 0, 0])


def test_pose_rejects_nonunit_quaternion():
    with pytest.raises(ValueError, match="unit"):
        pose_matrix([0, 0, 0, 2, 0, 0, 0])


def test_host_audit_rejects_bad_robot_calibration_before_ik(arm, monkeypatch):
    robots = [SimpleNamespace(type="target", arm_name=f"{name}_arm", urdf_path=URDF,
        arm_joints_name=NAMES, base_link="base_link", ee_link_name="link6") for name in ("left", "right")]
    manager = SimpleNamespace(robot_list=robots, get_joint=lambda r: {0: np.zeros(6)},
        get_link_pose=lambda *a, **k: {0: np.array([0, 0, 0, 1, 0, 0, 0])},
        get_real_endpose=lambda *a, **k: {0: np.array([10, 10, 10, 1, 0, 0, 0])})
    def forbidden(*args, **kwargs):
        pytest.fail("IK must not run after failed FK")
    monkeypatch.setattr(RobotArmKinematics, "translation_candidate", forbidden)
    audit = audit_robot_kinematics(SimpleNamespace(robot_manager=manager), np.zeros(14))
    assert not audit["fk_passed"]
    assert audit["physical_probe_steps"] == 0
    assert all(r["translation_probes"] == [] for r in audit["arms"].values())


def test_host_audit_rejects_stale_public_state():
    robots = [SimpleNamespace(type="target", arm_name=f"{name}_arm", urdf_path=URDF,
        arm_joints_name=NAMES, base_link="base_link", ee_link_name="link6") for name in ("left", "right")]
    manager = SimpleNamespace(robot_list=robots, get_joint=lambda r: {0: np.ones(6)})
    with pytest.raises(ValueError, match="state differ"):
        audit_robot_kinematics(SimpleNamespace(robot_manager=manager), np.zeros(14))
