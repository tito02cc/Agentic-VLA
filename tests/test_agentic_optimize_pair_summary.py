"""Tests for the coupled Agentic Harness and Optimize Runtime summarizer."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.summarize_carve_agentic_optimize_pair import summarize


class AgenticOptimizePairSummaryTest(unittest.TestCase):
    def test_reference_and_promoted_profiles_are_separated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, accepted, latency in (
                ("eager_bf16", False, 150.0),
                ("compiled_bf16", True, 65.0),
                ("compiled_smve", True, 55.0),
            ):
                profile_dir = root / name
                online_dir = profile_dir / "online_controller"
                online_dir.mkdir(parents=True)
                profile = {
                    "profile": {"profile_id": name},
                    "admission": {
                        "accepted": accepted,
                        "status": "promoted" if accepted else "unreviewed",
                    },
                }
                branch = {
                    "success_within_horizon": True,
                    "safe_stop": False,
                    "optimization_profile": profile,
                    "physical_recovery": {"outcome": {"status": "succeeded"}},
                }
                (profile_dir / "paired_branches.json").write_text(
                    json.dumps(
                        {
                            "records": [
                                {"branches": [branch]},
                                {"branches": [branch]},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                traces = [
                    {
                        "runtime_latency_ms": latency,
                        "model_latency_ms": latency - 10.0,
                        "deadline_miss": latency > 80.0,
                    }
                    for _ in range(4)
                ]
                (profile_dir / "paired_branches_policy_calls.jsonl").write_text(
                    "".join(json.dumps(trace) + "\n" for trace in traces),
                    encoding="utf-8",
                )
                (online_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "total_successes": 1,
                            "total_episodes": 1,
                            "trace_aggregate": {
                                "inference": {"vla_latency_ms_p95": latency},
                                "realtime": {"deadline_miss_rate_mean": 0.0},
                                "recovery": {
                                    "physical_recoveries_triggered_total": 1,
                                    "physical_recoveries_verified_total": 1,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            result = summarize(root)

        self.assertTrue(result["gates"]["evidence_complete"])
        self.assertTrue(result["gates"]["outcome_parity"])
        self.assertTrue(result["gates"]["eager_reference_misses_deadline"])
        self.assertTrue(result["gates"]["compiled_deadline_compliant"])
        self.assertTrue(result["gates"]["smve_deadline_compliant"])


if __name__ == "__main__":
    unittest.main()
