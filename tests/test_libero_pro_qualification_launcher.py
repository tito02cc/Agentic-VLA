"""CPU-only command-construction tests for LIBERO-PRO qualification."""

from __future__ import annotations

import pathlib
import unittest

from scripts.run_libero_pro_qualification import build_qualification_run


ROOT = pathlib.Path("/home/admin1/ct/benchmark-sources/LIBERO-PRO")


class LiberoProQualificationLauncherTest(unittest.TestCase):
    def test_full_agentic_profile_uses_canonical_knowledge_path(self) -> None:
        run = build_qualification_run(
            suite="libero_10_swap",
            method="full_agentic",
            libero_root=ROOT,
            results_root=pathlib.Path("/tmp/results"),
            python="python",
            host="127.0.0.1",
            port=8000,
            trials=3,
            seed=7,
            vlm_endpoint="http://127.0.0.1:18070/v1/chat/completions",
            vlm_model="vlm",
        )

        self.assertEqual(run.task_ids, (3, 8, 9))
        self.assertIn("--carve-canonical-harness", run.command)
        self.assertIn("--carve-agentic-knowledge", run.command)
        self.assertIn("--carve-semantic-before-recovery", run.command)
        recovery_budget_index = run.command.index("--carve-max-recovery-attempts")
        self.assertEqual(run.command[recovery_budget_index + 1], "1")
        warmup_index = run.command.index("--carve-monitor-warmup-steps")
        self.assertEqual(run.command[warmup_index + 1], "60")
        token_index = run.command.index("--carve-high-level-max-tokens")
        self.assertEqual(run.command[token_index + 1], "256")
        timeout_index = run.command.index("--carve-high-level-boundary-timeout-sec")
        self.assertEqual(run.command[timeout_index + 1], "15")
        cooldown_index = run.command.index("--planner-cooldown-steps")
        self.assertEqual(run.command[cooldown_index + 1], "50")
        self.assertNotIn("--graph-rag", run.command)
        self.assertNotIn("--critic", run.command)

    def test_frozen_profile_has_no_monitor_or_recovery(self) -> None:
        run = build_qualification_run(
            suite="libero_10_object",
            method="frozen_vla",
            libero_root=ROOT,
            results_root=pathlib.Path("/tmp/results"),
            python="python",
            host="127.0.0.1",
            port=8000,
            trials=3,
            seed=7,
            vlm_endpoint="endpoint",
            vlm_model="vlm",
        )

        self.assertEqual(run.task_ids, (3, 9))
        self.assertNotIn("--carve-joint-controller", run.command)
        self.assertNotIn("--carve-physical-recovery", run.command)
        self.assertIn("--carve-fixed-inference-steps", run.command)


if __name__ == "__main__":
    unittest.main()
