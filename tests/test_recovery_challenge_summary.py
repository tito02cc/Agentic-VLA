"""Tests for the PI0.5 recovery-challenge evidence summarizer."""

from __future__ import annotations

import unittest

from scripts.summarize_carve_recovery_challenge import render_markdown, summarize


def _branch(name: str, *, success: bool, safe_stop: bool = False) -> dict:
    no_call = safe_stop
    return {
        "branch": name,
        "success_within_horizon": success,
        "executed_steps": 0 if safe_stop else 10,
        "vla_calls": 0 if no_call else 2,
        "branch_wall_ms": 0.1 if safe_stop else 100.0,
        "deadline_miss_count": 0,
        "deadline_call_count": 0 if no_call else 2,
        "optimization_profile": (
            None
            if no_call
            else {
                "profile": {"profile_id": "pi05-smve"},
                "admission": {"accepted": True},
            }
        ),
        "physical_recovery": (
            {"outcome": {"status": "succeeded"}}
            if name == "physical_recovery" and not safe_stop
            else None
        ),
        "physical_recovery_error": "unsupported event" if safe_stop else None,
        "safe_stop": safe_stop,
    }


class RecoveryChallengeSummaryTest(unittest.TestCase):
    def test_safe_stop_is_a_valid_fail_closed_branch(self) -> None:
        branches = {
            "records": [
                {
                    "snapshot": "task08-stale",
                    "task_id": 8,
                    "trigger": "stale_action",
                    "branches": [
                        _branch("continue", success=False),
                        _branch("accurate", success=False),
                        _branch("recovery", success=False),
                        _branch("physical_recovery", success=False, safe_stop=True),
                    ],
                }
            ]
        }
        online = {
            "total_successes": 1,
            "total_episodes": 1,
            "trace_aggregate": {
                "inference": {"vla_latency_ms_p95": 60.0},
                "realtime": {"deadline_miss_rate_mean": 0.1},
                "recovery": {
                    "physical_recoveries_triggered_total": 1,
                    "physical_recoveries_verified_total": 1,
                    "physical_recovery_actions_total": 12,
                },
            },
        }

        result = summarize(branches, online)

        self.assertTrue(result["gates"]["complete"])
        self.assertTrue(result["scenarios"][0]["safe_stops"]["physical_recovery"])
        self.assertEqual(result["branches"]["physical_recovery"]["safe_stops"], 1)
        self.assertEqual(
            result["branches"]["physical_recovery"]["executed_scenarios"], 0
        )
        self.assertIn("safe stop", render_markdown(result))

    def test_succeeded_recovery_counts_as_verified(self) -> None:
        branch = _branch("physical_recovery", success=True)
        branches = {
            "records": [
                {
                    "snapshot": "task06-stall",
                    "task_id": 6,
                    "trigger": "stall",
                    "branches": [branch],
                }
            ]
        }
        online = {
            "total_successes": 1,
            "total_episodes": 1,
            "trace_aggregate": {
                "inference": {"vla_latency_ms_p95": 60.0},
                "realtime": {"deadline_miss_rate_mean": 0.0},
                "recovery": {
                    "physical_recoveries_triggered_total": 1,
                    "physical_recoveries_verified_total": 1,
                    "physical_recovery_actions_total": 12,
                },
            },
        }

        result = summarize(branches, online)

        self.assertEqual(
            result["branches"]["physical_recovery"][
                "verified_physical_recoveries"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
