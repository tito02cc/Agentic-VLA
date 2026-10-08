"""CPU-only tests for CARVE's event-coherent reuse boundary."""

from __future__ import annotations

import unittest

from agentic_vla.optimization import (
    EventCoherentReuseGate,
    ReuseCandidate,
    ReuseContext,
    ReuseLayer,
)
from agentic_vla.runtime import ExecutionMode, JointRecoveryComputeController, RiskAssessment


def candidate(**changes) -> ReuseCandidate:
    values = {
        "candidate_id": "cached-7",
        "layer": ReuseLayer.ACTION_QUEUE,
        "profile_id": "pi05-smve",
        "instruction_fingerprint": "task-a",
        "subgoal_fingerprint": "reach-cup",
        "planner_epoch": 1,
        "recovery_epoch": 0,
        "age_steps": 2,
    }
    values.update(changes)
    return ReuseCandidate(**values)


def context(**changes) -> ReuseContext:
    values = {
        "profile_id": "pi05-smve",
        "instruction_fingerprint": "task-a",
        "subgoal_fingerprint": "reach-cup",
        "planner_epoch": 1,
        "recovery_epoch": 0,
        "risk_bucket": "low",
        "visual_change": 0.002,
        "proprio_change": 0.01,
    }
    values.update(changes)
    return ReuseContext(**values)


class EventCoherentReuseGateTests(unittest.TestCase):
    def test_admits_coherent_low_risk_action_queue(self) -> None:
        decision = EventCoherentReuseGate().evaluate(candidate(), context())

        self.assertTrue(decision.accepted)
        self.assertFalse(decision.force_refresh)
        self.assertEqual(decision.invalidations, ())

    def test_agentic_epoch_changes_force_refresh(self) -> None:
        decision = EventCoherentReuseGate().evaluate(
            candidate(),
            context(planner_epoch=2, recovery_epoch=1),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("planner_epoch_changed", decision.invalidations)
        self.assertIn("recovery_epoch_changed", decision.invalidations)

    def test_physical_event_and_scene_change_force_refresh(self) -> None:
        decision = EventCoherentReuseGate().evaluate(
            candidate(),
            context(event="stall", risk_bucket="high", visual_change=0.2),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("execution_event:stall", decision.invalidations)
        self.assertIn("risk_bucket:high", decision.invalidations)
        self.assertIn("visual_context_changed", decision.invalidations)

    def test_action_warm_start_requires_similarity(self) -> None:
        missing = EventCoherentReuseGate().evaluate(
            candidate(layer=ReuseLayer.ACTION_WARM_START),
            context(),
        )
        accepted = EventCoherentReuseGate().evaluate(
            candidate(layer=ReuseLayer.ACTION_WARM_START, similarity=0.95),
            context(),
        )

        self.assertIn("similarity_missing", missing.invalidations)
        self.assertTrue(accepted.accepted)

    def test_controller_replans_when_gate_rejects_cached_actions(self) -> None:
        risk = RiskAssessment(
            score=0.0,
            bucket="low",
            event=None,
            components={},
            evidence={},
        )

        decision = JointRecoveryComputeController().decide(
            risk,
            deadline_ms=80.0,
            deadline_slack_ms=40.0,
            cached_actions=4,
            reuse_permitted=False,
        )

        self.assertEqual(decision.mode, ExecutionMode.FAST_VLA)
        self.assertIn("invalidated", decision.reason)


if __name__ == "__main__":
    unittest.main()
