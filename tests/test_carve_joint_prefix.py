import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.runtime import (
    ActionPrefixVerifier,
    ActionSpec,
    JointPositionPrefixVerifier,
)
from agentic_vla.runtime.prefetch import ActionPrefixThresholds


def spec():
    return ActionSpec(
        14, "dual_arm_joint_position", "robot_joint", "xpolicy_joint_gripper", 25
    )


def verifier(**kwargs):
    config = {
        "gripper_indices": (6, 13),
        "max_joint_error": 0.05,
        "joint_rms_max": 0.03,
        "max_gripper_error": 0.02,
    }
    return JointPositionPrefixVerifier(spec(), **{**config, **kwargs})


def test_identical_dual_arm_prefix_passes_without_changing_inputs():
    left = np.full((4, 14), 0.2)
    right = left.copy()
    result = verifier().evaluate(left, right)
    assert result.accepted and result.prefix_actions == 4
    assert result.max_joint_error == result.joint_rms == result.max_gripper_error == 0
    np.testing.assert_array_equal(left, right)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("dimension", [0, 5, 7, 12])
def test_all_arm_joints_are_checked(dimension):
    old = np.zeros((2, 14))
    new = old.copy()
    new[:, dimension] = 0.1
    result = verifier().evaluate(old, new)
    assert not result.accepted and "joint_max" in result.reason


@pytest.mark.parametrize("dimension", [6, 13])
def test_grippers_use_magnitude_not_only_sign(dimension):
    old = np.full((2, 14), 0.1)
    new = old.copy()
    new[:, dimension] = 0.9
    result = verifier().evaluate(old, new)
    assert not result.accepted and result.reason == "gripper"


def test_opposing_joint_errors_cannot_cancel_as_cartesian_displacements():
    old = np.zeros((2, 14))
    new = old.copy()
    new[:, 10] = [0.2, -0.2]
    assert not verifier().evaluate(old, new).accepted


def test_joint_rms_checks_many_small_errors():
    old = np.zeros((2, 14))
    new = np.full((2, 14), 0.04)
    new[:, [6, 13]] = 0
    result = verifier().evaluate(old, new)
    assert not result.accepted and result.reason == "joint_rms"


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[0] * 7],
        [0] * 14,
        [[0] * 14, [0] * 13],
        [[float("nan")] * 14],
        [[float("inf")] * 14],
        [["bad"] * 14],
    ],
)
def test_bad_action_data_is_rejected(values):
    result = verifier().evaluate(np.zeros((1, 14)), values)
    assert not result.accepted and result.reason == "invalid_prefix_contract"
    json.dumps(result.to_dict(), allow_nan=False)


def test_native_bounds_and_overflow_are_rejected():
    bounded = dataclasses.replace(spec(), minimum=-1, maximum=1)
    check = JointPositionPrefixVerifier(
        bounded,
        gripper_indices=(6, 13),
        max_joint_error=0.05,
        joint_rms_max=0.03,
        max_gripper_error=0.02,
    )
    assert not check.evaluate(np.zeros((1, 14)), np.full((1, 14), 2)).accepted
    result = verifier().evaluate(np.full((1, 14), -1e308), np.full((1, 14), 1e308))
    assert not result.accepted and result.reason == "nonfinite_prefix_metrics"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("indices", [(6,), (6, 6), (6, 14), (True, 13), (-1, 13)])
def test_action_layout_requires_both_explicit_grippers(indices):
    with pytest.raises(ValueError):
        verifier(gripper_indices=indices)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_joint_thresholds_must_be_calibrated_finite_values(value):
    with pytest.raises(ValueError):
        verifier(max_joint_error=value)


def test_cartesian_contract_is_not_silently_interpreted_as_joint_positions():
    with pytest.raises(ValueError):
        JointPositionPrefixVerifier(
            dataclasses.replace(spec(), representation="delta_pose"),
            gripper_indices=(6, 13),
            max_joint_error=0.05,
            joint_rms_max=0.03,
            max_gripper_error=0.02,
        )


@pytest.mark.parametrize("width", [6, 8, 14])
def test_legacy_verifier_refuses_non_7d_actions(width):
    actions = np.zeros((2, width))
    assert not ActionPrefixVerifier().evaluate(actions, actions).accepted


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_legacy_thresholds_cannot_disable_comparison_with_nonfinite_values(value):
    with pytest.raises(ValueError):
        ActionPrefixThresholds(translation_endpoint_max=value)
