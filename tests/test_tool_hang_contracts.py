from __future__ import annotations

import unittest

import numpy as np

from agentic_vla.tool_hang import (
    ToolHangEvaluation,
    ToolHangObservation,
    ToolHangPhase,
    ToolHangSkill,
    ToolHangTask,
)


def observation() -> ToolHangObservation:
    return ToolHangObservation(
        external_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        external_depth=np.ones((16, 16, 1), dtype=np.float32),
        wrist_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        wrist_depth=np.ones((16, 16, 1), dtype=np.float32),
        proprio=np.zeros(9, dtype=np.float64),
        gripper=np.zeros(2, dtype=np.float64),
        eef_position=np.zeros(3, dtype=np.float64),
        eef_quaternion=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )


class ToolHangContractsTest(unittest.TestCase):
    def test_public_observation_excludes_simulator_labels(self) -> None:
        value = observation()
        self.assertFalse(hasattr(value, "object_state"))
        self.assertFalse(hasattr(value, "environment_state"))
        self.assertFalse(hasattr(value, "frame_assembled"))
        self.assertFalse(hasattr(value, "success"))

    def test_observation_rejects_non_robot_state_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "9D"):
            ToolHangObservation(
                external_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                external_depth=np.ones((16, 16, 1), dtype=np.float32),
                wrist_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                wrist_depth=np.ones((16, 16, 1), dtype=np.float32),
                proprio=np.zeros(32),
                gripper=np.zeros(2),
                eef_position=np.zeros(3),
                eef_quaternion=np.zeros(4),
            )

    def test_private_evaluator_has_ordered_phases(self) -> None:
        unassembled = ToolHangEvaluation.from_checks(
            frame_assembled=False,
            tool_hung=False,
        )
        assembled = ToolHangEvaluation.from_checks(
            frame_assembled=True,
            tool_hung=False,
        )
        complete = ToolHangEvaluation.from_checks(
            frame_assembled=True,
            tool_hung=True,
        )
        self.assertEqual(unassembled.phase, ToolHangPhase.FRAME_UNASSEMBLED)
        self.assertEqual(assembled.phase, ToolHangPhase.FRAME_ASSEMBLED)
        self.assertEqual(complete.phase, ToolHangPhase.COMPLETE)
        self.assertTrue(complete.success)

    def test_canonical_task_requires_assembly_before_hanging(self) -> None:
        task = ToolHangTask.canonical()
        self.assertEqual(
            task.required_stages,
            ("assemble_hook_frame", "hang_tool"),
        )
        self.assertIn("wrench", task.instruction)

    def test_registered_skills_include_distinct_recovery_modes(self) -> None:
        self.assertIn(ToolHangSkill.RECOVER_GRASP, ToolHangSkill)
        self.assertIn(ToolHangSkill.RECOVER_ALIGNMENT, ToolHangSkill)
        self.assertNotEqual(
            ToolHangSkill.RECOVER_GRASP,
            ToolHangSkill.RECOVER_ALIGNMENT,
        )


if __name__ == "__main__":
    unittest.main()
