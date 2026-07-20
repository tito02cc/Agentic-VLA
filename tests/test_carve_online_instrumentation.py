"""Regression tests for online CARVE recovery instrumentation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.run_agentic_vla_libero import (
    EpisodeInstrumentation,
    _aggregate_episode_traces,
    _build_semantic_vlm_payload,
    _parse_semantic_label,
)


class OnlineRecoveryInstrumentationTest(unittest.TestCase):
    def test_semantic_code_protocol_maps_to_runtime_labels(self) -> None:
        self.assertEqual(_parse_semantic_label("N\n"), ("NOMINAL", "NONE"))
        self.assertEqual(_parse_semantic_label("A"), ("REPLAN", "MISALIGN"))
        self.assertEqual(_parse_semantic_label("C."), ("STOP", "COLLISION"))

    def test_semantic_payload_uses_temporal_images_without_evaluator_detail(self) -> None:
        payload = _build_semantic_vlm_payload(
            {
                "model": "vlm",
                "task": "put the mug in the microwave",
                "event": "visual execution anomaly detected",
                "reference_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "image": np.ones((8, 8, 3), dtype=np.uint8),
                "max_tokens": 12,
            }
        )

        content = payload["messages"][0]["content"]
        text = " ".join(item["text"] for item in content if item["type"] == "text")
        images = [item for item in content if item["type"] == "image_url"]

        self.assertEqual(len(images), 2)
        self.assertIn("visual execution anomaly detected", text)
        self.assertNotIn("dx", text)
        self.assertNotIn("dy", text)
        self.assertEqual(payload["max_tokens"], 12)

    def test_control_latency_is_split_by_vla_and_cached_steps(self) -> None:
        trace = EpisodeInstrumentation(
            method_tag="test",
            task_suite="libero_10",
            task_id=6,
            episode_idx=0,
            task_description="test task",
            control_deadline_ms=80.0,
        )
        trace.add_control_stage("env_step", 0.002)
        trace.record_control_step(0.100, fast_path=True, vla_call=True)
        trace.record_control_step(0.010, fast_path=True, vla_call=False)

        realtime = trace.finish(
            success=False,
            episode_steps=2,
            peak_gpu_mem_gb=None,
        )["realtime"]

        self.assertEqual(realtime["vla_control_steps"], 1)
        self.assertEqual(realtime["vla_control_deadline_misses"], 1)
        self.assertEqual(realtime["cached_control_steps"], 1)
        self.assertEqual(realtime["cached_control_deadline_misses"], 0)
        self.assertEqual(realtime["stages"]["env_step"]["mean_ms"], 2.0)

    def test_physical_recovery_counts_are_aggregated(self) -> None:
        traces = [
            {
                "success": True,
                "recoveries_triggered": 1,
                "recoveries_successful": 1,
                "physical_recoveries_triggered": 1,
                "physical_recoveries_verified": 1,
                "physical_recovery_actions": 12,
                "realtime": {},
                "latency": {},
                "lightweight": {},
                "gpu": {},
            },
            {
                "success": False,
                "recoveries_triggered": 1,
                "recoveries_successful": 0,
                "physical_recoveries_triggered": 1,
                "physical_recoveries_verified": 0,
                "physical_recovery_actions": 12,
                "realtime": {},
                "latency": {},
                "lightweight": {},
                "gpu": {},
            },
        ]

        recovery = _aggregate_episode_traces(traces)["recovery"]

        self.assertEqual(recovery["physical_recoveries_triggered_total"], 2)
        self.assertEqual(recovery["physical_recoveries_verified_total"], 1)
        self.assertEqual(recovery["physical_recovery_actions_total"], 24)
        self.assertEqual(recovery["physical_verification_rate"], 0.5)

    def test_prefix_consistency_shadow_metrics_are_aggregated(self) -> None:
        trace = EpisodeInstrumentation(
            method_tag="test",
            task_suite="libero_10",
            task_id=8,
            episode_idx=0,
            task_description="test task",
            control_deadline_ms=80.0,
            async_prefix_mode="shadow",
        )
        trace.record_async_prefix_consistency(
            {
                "accepted": False,
                "reason": "translation_endpoint",
                "prefix_actions": 2,
                "continuous_rms": 0.4,
                "translation_endpoint_l2": 0.6,
                "rotation_endpoint_l2": 0.1,
                "gripper_agreement": 1.0,
                "continuous_cosine": 0.7,
            },
            ticket_id="ticket",
            submitted_timestep=10,
            enforced_rejection=False,
        )
        finished = trace.finish(success=False, episode_steps=1, peak_gpu_mem_gb=None)

        prefix = finished["realtime"]["async_prefetch"]["prefix_consistency"]
        aggregate = _aggregate_episode_traces([finished])["realtime"][
            "async_prefetch"
        ]["prefix_consistency"]

        self.assertEqual(prefix["mode"], "shadow")
        self.assertEqual(prefix["would_reject"], 1)
        self.assertEqual(prefix["enforced_rejections"], 0)
        self.assertEqual(aggregate["checks"], 1)
        self.assertEqual(aggregate["continuous_rms_p95"], 0.4)

    def test_semantic_shadow_metrics_are_aggregated(self) -> None:
        trace = EpisodeInstrumentation(
            method_tag="test",
            task_suite="libero_10",
            task_id=8,
            episode_idx=0,
            task_description="test task",
            control_deadline_ms=80.0,
            semantic_mode="shadow",
        )
        trace.record_semantic_schedule(
            timestep=80,
            event="perturbation",
            invoke=True,
            reason="event and deadline gate passed",
            deadline_slack_ms=50.0,
        )
        trace.record_semantic_result(
            SimpleNamespace(
                elapsed_s=1.25,
                error=None,
                valid=True,
                response={"content": "REPLAN|MISALIGN", "model": "vlm"},
                context=SimpleNamespace(
                    ticket_id="semantic-ticket",
                    event="perturbation",
                    submitted_timestep=80,
                ),
            ),
            collected_timestep=100,
        )

        finished = trace.finish(success=True, episode_steps=120, peak_gpu_mem_gb=None)
        semantic = finished["realtime"]["semantic_observer"]
        aggregate = _aggregate_episode_traces([finished])["realtime"][
            "semantic_observer"
        ]

        self.assertEqual(semantic["submitted"], 1)
        self.assertEqual(semantic["completed"], 1)
        self.assertEqual(semantic["errors"], 0)
        self.assertEqual(semantic["latency_ms_mean"], 1250.0)
        self.assertTrue(semantic["observations"][0]["protocol_valid"])
        self.assertEqual(semantic["observations"][0]["status"], "REPLAN")
        self.assertEqual(aggregate["completed"], 1)
        self.assertEqual(aggregate["latency_ms_p95"], 1250.0)
        aggregate_inference = _aggregate_episode_traces([finished])["inference"]
        self.assertEqual(aggregate_inference["semantic_calls_total"], 1)
        self.assertEqual(aggregate_inference["vlm_calls_total"], 1)


if __name__ == "__main__":
    unittest.main()
