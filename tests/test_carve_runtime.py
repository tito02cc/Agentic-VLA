"""CPU-only conformance tests for the CARVE runtime boundary."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import threading
import unittest

import numpy as np

from agentic_vla.runtime import (
    ActionPrefixThresholds,
    ActionPrefixVerifier,
    ActionSpec,
    AsyncInferencePrefetcher,
    AsyncSemanticObserver,
    CarveRuntime,
    ExecutionMode,
    DeadlineAwareSemanticScheduler,
    ExecutionRiskMonitor,
    FailureEpisodeRecord,
    FailureMemory,
    InferenceControls,
    InferenceRequest,
    JointControllerConfig,
    JointRecoveryComputeController,
    JsonlTraceSink,
    MonitorConfig,
    PrefetchContext,
    PrefetchDutyCycle,
    RecoveryContext,
    RecoveryMemory,
    RecoverySkillRegistry,
    RecoveryStatus,
    RiskAssessment,
    RiskDeadlineController,
    RuntimeControllablePolicy,
    SemanticObservationContext,
    SemanticScheduleConfig,
    StatefulRecoveryExecutor,
    UnsupportedControlError,
    build_cartesian_retreat_plan,
    build_release_retreat_plan,
)
from agentic_vla.runtime.adapters import LegacyPolicyClientBridge, Pi05Adapter
from agentic_vla.experiments import (
    FailureSnapshotWriter,
    capture_simulator_snapshot,
    restore_simulator_snapshot,
)


class FakeLocalPi05:
    def __init__(self) -> None:
        self._sample_kwargs = {"num_steps": 7}
        self.calls: list[tuple[dict, int, object]] = []

    def infer(self, payload: dict, **kwargs):
        self.calls.append(
            (dict(payload), int(self._sample_kwargs["num_steps"]), kwargs.get("noise"))
        )
        return {
            "actions": [[float(index)] * 7 for index in range(6)],
            "policy_timing": {"infer_ms": 12.5},
            "agentic": {"action": "continue"},
        }


class FakeNoisePolicy(FakeLocalPi05):
    def __init__(self) -> None:
        super().__init__()
        self.noise_seen = None

    def infer(self, payload: dict, **kwargs):
        self.noise_seen = kwargs.get("noise")
        return super().infer(payload, **kwargs)


class FakeRemotePi05:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def infer(self, payload: dict):
        self.payloads.append(dict(payload))
        return {"actions": [[0.0] * 7 for _ in range(4)]}


class FakeControlledRemotePi05(FakeRemotePi05):
    def __init__(self) -> None:
        super().__init__()
        self._server_metadata = {
            "carve_capabilities": {"configurable_inference_steps": True},
            "carve_deployment_profile": {
                "backend_id": "torch_compile",
                "profile": {"profile_id": "pi05-compiled"},
            },
        }


class MalformedPolicy:
    def infer(self, payload: dict):
        return {"not_actions": []}


class FakeBranchEnv:
    def __init__(self) -> None:
        self.state = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    def get_sim_state(self):
        return self.state.copy()

    def regenerate_obs_from_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).copy()
        return {"state": self.state.copy()}


class CarveRuntimeTest(unittest.TestCase):
    def test_semantic_scheduler_gates_event_slack_and_cooldown(self) -> None:
        scheduler = DeadlineAwareSemanticScheduler(
            SemanticScheduleConfig(
                cooldown_steps=10,
                min_deadline_slack_ms=20.0,
                allowed_events=frozenset({"perturbation"}),
            )
        )

        self.assertFalse(
            scheduler.decide(
                event=None,
                timestep=5,
                deadline_slack_ms=50.0,
            ).invoke
        )
        self.assertFalse(
            scheduler.decide(
                event="perturbation",
                timestep=5,
                deadline_slack_ms=10.0,
            ).invoke
        )
        decision = scheduler.decide(
            event="perturbation",
            timestep=5,
            deadline_slack_ms=50.0,
        )
        self.assertTrue(decision.invoke)
        scheduler.record_submission(5)
        self.assertFalse(
            scheduler.decide(
                event="perturbation",
                timestep=12,
                deadline_slack_ms=50.0,
            ).invoke
        )
        self.assertTrue(
            scheduler.decide(
                event="perturbation",
                timestep=15,
                deadline_slack_ms=50.0,
            ).invoke
        )

    def test_async_semantic_observer_is_single_flight(self) -> None:
        release = threading.Event()

        def observe(request):
            release.wait(timeout=2.0)
            return {"assessment": request["event"]}

        observer = AsyncSemanticObserver(observe)
        context = SemanticObservationContext(
            episode_id="8:0",
            submitted_timestep=80,
            event="perturbation",
        )
        observer.submit({"event": "object_nudge"}, context)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            observer.submit({}, context)
        release.set()
        result = observer.take()
        observer.close()

        self.assertTrue(result.valid)
        self.assertEqual(result.context.ticket_id, context.ticket_id)
        self.assertEqual(result.response["assessment"], "object_nudge")

    def test_async_semantic_observer_captures_error(self) -> None:
        def fail(_):
            raise RuntimeError("vlm unavailable")

        with AsyncSemanticObserver(fail) as observer:
            observer.submit(
                {},
                SemanticObservationContext(
                    episode_id="8:0",
                    submitted_timestep=80,
                    event="perturbation",
                ),
            )
            result = observer.take()

        self.assertFalse(result.valid)
        self.assertIn("vlm unavailable", str(result.error))

    def test_action_prefix_verifier_accepts_consistent_suffix_anchor(self) -> None:
        verifier = ActionPrefixVerifier()
        executed = np.asarray(
            [[0.1, 0.0, 0.0, 0.0, 0.1, 0.0, -1.0]] * 2,
            dtype=np.float32,
        )
        predicted = executed.copy()
        predicted[:, 0] += 0.02

        result = verifier.evaluate(executed, predicted)

        self.assertTrue(result.accepted)
        self.assertEqual(result.prefix_actions, 2)
        self.assertGreater(result.continuous_cosine, 0.9)

    def test_action_prefix_verifier_rejects_temporal_and_gripper_mismatch(self) -> None:
        verifier = ActionPrefixVerifier(
            ActionPrefixThresholds(
                continuous_rms_max=0.1,
                translation_endpoint_max=0.1,
                rotation_endpoint_max=0.1,
                gripper_agreement_min=1.0,
            )
        )
        executed = np.asarray([[0.0] * 6 + [-1.0]] * 2, dtype=np.float32)
        predicted = np.asarray([[0.4] + [0.0] * 5 + [1.0]] * 2, dtype=np.float32)

        result = verifier.evaluate(executed, predicted)

        self.assertFalse(result.accepted)
        self.assertIn("translation_endpoint", result.reason)
        self.assertIn("gripper", result.reason)

    def test_action_prefix_verifier_rejects_malformed_contract(self) -> None:
        result = ActionPrefixVerifier().evaluate([[0.0] * 7], [[0.0] * 6])

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "invalid_prefix_contract")

    def test_prefetch_duty_cycle_requires_fresh_chunks_between_prefetches(self) -> None:
        schedule = PrefetchDutyCycle(interval=2)

        self.assertTrue(schedule.can_submit)
        schedule.record_prefetch_consumed()
        self.assertFalse(schedule.can_submit)
        schedule.record_synchronous_chunk()
        self.assertTrue(schedule.can_submit)

    def test_prefetch_duty_cycle_interval_one_allows_continuous_prefetch(self) -> None:
        schedule = PrefetchDutyCycle(interval=1)

        schedule.record_prefetch_consumed()

        self.assertTrue(schedule.can_submit)

    def test_async_prefetch_is_single_flight_and_returns_context(self) -> None:
        release = threading.Event()

        def infer(request):
            release.wait(timeout=2.0)
            return {"actions": request["actions"]}

        prefetcher = AsyncInferencePrefetcher(infer)
        context = PrefetchContext(
            episode_id="6:0",
            submitted_timestep=20,
            lead_actions=2,
        )
        prefetcher.submit({"actions": [[1.0] * 7]}, context)
        self.assertTrue(prefetcher.pending)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            prefetcher.submit({"actions": []}, context)
        release.set()

        result = prefetcher.take()
        prefetcher.close()

        self.assertTrue(result.valid)
        self.assertEqual(result.context.ticket_id, context.ticket_id)
        self.assertEqual(result.response["actions"][0][0], 1.0)

    def test_async_prefetch_invalidation_is_explicit(self) -> None:
        with AsyncInferencePrefetcher(lambda _: {"actions": [[0.0] * 7]}) as prefetcher:
            prefetcher.submit(
                {},
                PrefetchContext(
                    episode_id="6:0",
                    submitted_timestep=30,
                    lead_actions=2,
                ),
            )
            prefetcher.invalidate("risk_escalation")
            result = prefetcher.take()

        self.assertFalse(result.valid)
        self.assertEqual(result.invalidation_reason, "risk_escalation")

    def test_async_prefetch_captures_worker_error(self) -> None:
        def fail(_):
            raise RuntimeError("inference failed")

        with AsyncInferencePrefetcher(fail) as prefetcher:
            prefetcher.submit(
                {},
                PrefetchContext(
                    episode_id="6:0",
                    submitted_timestep=40,
                    lead_actions=2,
                ),
            )
            result = prefetcher.take()

        self.assertFalse(result.valid)
        self.assertIsInstance(result.error, RuntimeError)

    def test_local_pi05_applies_and_restores_inference_steps(self) -> None:
        policy = FakeLocalPi05()
        runtime = CarveRuntime(Pi05Adapter(policy))
        request = InferenceRequest(
            observation={"observation/state": [0.0] * 8},
            instruction="pick up the mug",
            controls=InferenceControls(inference_steps=2, max_actions=3),
            episode_id="episode-1",
            timestep=4,
            metadata={"noise": "fixed-noise", "action_age_steps": 3},
        )

        chunk = runtime.infer(request)

        self.assertEqual(chunk.action_count, 3)
        self.assertEqual(policy.calls[0][1], 2)
        self.assertEqual(policy.calls[0][2], "fixed-noise")
        self.assertEqual(policy._sample_kwargs["num_steps"], 7)
        self.assertEqual(policy.calls[0][0]["prompt"], "pick up the mug")
        self.assertEqual(policy.calls[0][0]["episode_id"], "episode-1")
        self.assertEqual(runtime.last_trace.applied_controls["inference_steps"], 2)
        self.assertEqual(runtime.last_trace.dropped_controls, ())
        self.assertEqual(runtime.last_trace.model_latency_ms, 12.5)
        self.assertGreaterEqual(
            runtime.last_trace.reaction_latency_ms,
            runtime.last_trace.runtime_latency_ms,
        )
        self.assertEqual(runtime.last_trace.action_age_steps, 3)

    def test_remote_client_gracefully_drops_native_step_control(self) -> None:
        policy = FakeRemotePi05()
        runtime = CarveRuntime(Pi05Adapter(policy), fallback_mode="graceful")
        request = InferenceRequest(
            observation={"observation/state": [0.0]},
            instruction="move",
            controls=InferenceControls(inference_steps=2, max_actions=2),
        )

        chunk = runtime.infer(request)

        self.assertEqual(chunk.action_count, 2)
        self.assertEqual(runtime.last_trace.dropped_controls, ("inference_steps",))
        self.assertIsNone(runtime.last_trace.applied_controls["inference_steps"])

    def test_remote_pi05_sends_fixed_noise_in_payload(self) -> None:
        policy = FakeControlledRemotePi05()
        runtime = CarveRuntime(Pi05Adapter(policy), fallback_mode="strict")
        noise = np.zeros((10, 32), dtype=np.float32)

        runtime.infer(
            InferenceRequest(
                observation={},
                instruction="move",
                controls=InferenceControls(inference_steps=2),
                metadata={"noise": noise},
            )
        )

        np.testing.assert_array_equal(policy.payloads[0]["runtime_noise"], noise)

    def test_controlled_remote_applies_native_step_control(self) -> None:
        policy = FakeControlledRemotePi05()
        runtime = CarveRuntime(Pi05Adapter(policy), fallback_mode="strict")
        runtime.infer(
            InferenceRequest(
                observation={},
                instruction="move",
                controls=InferenceControls(inference_steps=2),
            )
        )
        self.assertEqual(policy.payloads[0]["runtime_controls"]["inference_steps"], 2)
        self.assertEqual(runtime.last_trace.applied_controls["inference_steps"], 2)
        self.assertEqual(runtime.last_trace.metadata["transport"], "controlled_client")
        self.assertEqual(
            runtime.last_trace.metadata["optimization_profile"]["backend_id"],
            "torch_compile",
        )

    def test_agentic_event_and_deployment_profile_share_runtime_trace(self) -> None:
        runtime = CarveRuntime(Pi05Adapter(FakeControlledRemotePi05()))
        runtime.infer(
            InferenceRequest(
                observation={},
                instruction="recover",
                agentic={"event": "stall", "action": "retry"},
            )
        )

        metadata = runtime.last_trace.metadata
        self.assertEqual(metadata["agentic_event"]["event"], "stall")
        self.assertEqual(
            metadata["optimization_profile"]["profile"]["profile_id"],
            "pi05-compiled",
        )

    def test_strict_mode_rejects_unsupported_control(self) -> None:
        runtime = CarveRuntime(Pi05Adapter(FakeRemotePi05()), fallback_mode="strict")
        request = InferenceRequest(
            observation={},
            instruction="move",
            controls=InferenceControls(inference_steps=2),
        )

        with self.assertRaises(UnsupportedControlError):
            runtime.infer(request)

    def test_legacy_bridge_preserves_payload_and_auxiliary_output(self) -> None:
        policy = FakeRemotePi05()
        bridge = LegacyPolicyClientBridge(CarveRuntime(Pi05Adapter(policy)))
        payload = {
            "observation/image": "image",
            "observation/state": [1.0, 2.0],
            "prompt": "open the drawer",
            "episode_id": "8:0",
            "timestep": 10,
            "agentic": {"request_plan": True},
            "runtime_controls": {"max_actions": 2},
        }

        output = bridge.infer(payload)

        self.assertEqual(len(output["actions"]), 2)
        self.assertNotIn("runtime_controls", policy.payloads[0])
        self.assertEqual(policy.payloads[0]["prompt"], "open the drawer")
        self.assertEqual(policy.payloads[0]["agentic"], {"request_plan": True})

    def test_malformed_output_is_recorded_before_error_is_raised(self) -> None:
        runtime = CarveRuntime(Pi05Adapter(MalformedPolicy()))
        with self.assertRaises(ValueError):
            runtime.infer(InferenceRequest(observation={}, instruction="move"))

        self.assertFalse(runtime.last_trace.success)
        self.assertIn("does not contain", runtime.last_trace.error)

    def test_controller_couples_risk_compute_and_commit_horizon(self) -> None:
        controller = RiskDeadlineController()

        fast = controller.decide(
            risk_score=0.2,
            deadline_ms=200.0,
            deadline_slack_ms=100.0,
        )
        accurate = controller.decide(
            risk_score=0.9,
            deadline_ms=200.0,
            deadline_slack_ms=100.0,
        )
        urgent = controller.decide(
            risk_score=0.9,
            deadline_ms=200.0,
            deadline_slack_ms=10.0,
        )

        self.assertEqual(fast.controls.inference_steps, 1)
        self.assertGreater(fast.controls.max_actions, accurate.controls.max_actions)
        self.assertEqual(accurate.controls.inference_steps, 2)
        self.assertEqual(urgent.controls.inference_steps, 1)
        self.assertIn("low deadline slack", urgent.reason)

    def test_action_contract_rejects_wrong_dimension(self) -> None:
        spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )
        runtime = CarveRuntime(Pi05Adapter(FakeRemotePi05(), action_spec=spec))
        runtime.infer(InferenceRequest(observation={}, instruction="move"))
        self.assertEqual(runtime.last_trace.metadata["action_spec"]["action_dim"], 7)

        invalid = FakeRemotePi05()
        invalid.infer = lambda payload: {"actions": [[0.0] * 6]}
        invalid_runtime = CarveRuntime(Pi05Adapter(invalid, action_spec=spec))
        with self.assertRaisesRegex(ValueError, "dimension 6"):
            invalid_runtime.infer(InferenceRequest(observation={}, instruction="move"))

    def test_deployable_monitor_detects_stall_and_joint_controller_recovers(self) -> None:
        monitor = ExecutionRiskMonitor(
            MonitorConfig(window_size=3, stall_weight=1.0, staleness_weight=0.0,
                          uncertainty_weight=0.0, deadline_weight=0.0)
        )
        risk = None
        first_stall = None
        for index in range(5):
            risk = monitor.update(
                proprio=np.zeros(8),
                commanded_action=np.asarray([0.2] * 6 + [0.0]),
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
                deadline_slack_ms=100.0,
            )
            if index == 2:
                first_stall = risk
        self.assertEqual(risk.event, "stall")
        self.assertEqual(risk.bucket, "high")

        first_decision = JointRecoveryComputeController().decide(
            first_stall,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            recovery_attempts=0,
        )
        self.assertEqual(first_decision.mode, ExecutionMode.ACCURATE_VLA)
        self.assertEqual(first_decision.controls.inference_steps, 2)
        self.assertEqual(first_decision.controls.max_actions, 10)
        self.assertIn("next accurate replan", first_decision.reason)

        decision = JointRecoveryComputeController().decide(
            risk,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            recovery_attempts=0,
        )
        self.assertEqual(decision.mode, ExecutionMode.RECOVERY)
        self.assertEqual(
            decision.recovery_skill_id,
            "cartesian_retract_lift_reobserve",
        )
        self.assertTrue(decision.request_verification)

        direct_failure = type(risk)(
            score=0.9,
            bucket="high",
            event="misgrasp",
            components={"direct_failure": 1.0},
            evidence={"event_streak": 1},
        )
        recovery = JointRecoveryComputeController().decide(
            direct_failure,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            recovery_attempts=0,
        )
        self.assertEqual(recovery.mode, ExecutionMode.RECOVERY)
        self.assertEqual(
            recovery.recovery_skill_id,
            "release_retract_lift_reobserve",
        )

        unsupported_collision = type(risk)(
            score=0.9,
            bucket="high",
            event="collision",
            components={"direct_failure": 1.0},
            evidence={},
        )
        collision_decision = JointRecoveryComputeController().decide(
            unsupported_collision,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
        )
        self.assertEqual(collision_decision.mode, ExecutionMode.ACCURATE_VLA)

    def test_deployable_monitor_suppresses_startup_stall_during_warmup(self) -> None:
        monitor = ExecutionRiskMonitor(
            MonitorConfig(
                window_size=3,
                warmup_steps=4,
                stall_weight=1.0,
                staleness_weight=0.0,
                uncertainty_weight=0.0,
                deadline_weight=0.0,
            )
        )

        risks = [
            monitor.update(
                proprio=np.zeros(8),
                commanded_action=np.asarray([0.2] * 6 + [0.0]),
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            for _ in range(5)
        ]

        self.assertTrue(all(risk.event is None for risk in risks[:4]))
        self.assertEqual(risks[3].evidence["warmup_steps_remaining"], 1)
        self.assertEqual(risks[4].event, "stall")

    def test_monitor_detects_sustained_no_progress_without_misclassifying_stall(self) -> None:
        monitor = ExecutionRiskMonitor(
            MonitorConfig(
                window_size=3,
                no_progress_steps=4,
                idle_command_threshold=0.01,
                idle_state_response_threshold=0.02,
                idle_visual_response_threshold=0.01,
            )
        )

        risks = [
            monitor.update(
                proprio=np.full(8, index * 0.001, dtype=np.float32),
                commanded_action=np.full(8, 0.001, dtype=np.float32),
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            for index in range(7)
        ]

        self.assertTrue(all(risk.event is None for risk in risks[:5]))
        self.assertEqual(risks[5].event, "no_progress")
        self.assertEqual(risks[5].evidence["idle_streak"], 4)
        self.assertEqual(risks[5].components["stall"], 0.0)
        self.assertEqual(risks[5].components["no_progress"], 1.0)

    def test_no_progress_escalates_to_semantic_planner_without_physical_recovery(self) -> None:
        controller = JointRecoveryComputeController()
        first = RiskAssessment(
            score=0.45,
            bucket="medium",
            event="no_progress",
            evidence={"event_streak": 1},
            components={"no_progress": 1.0},
        )
        repeated = dataclasses.replace(first, evidence={"event_streak": 2})

        replan = controller.decide(
            first,
            deadline_ms=350.0,
            deadline_slack_ms=100.0,
            planner_available=True,
        )
        escalate = controller.decide(
            repeated,
            deadline_ms=350.0,
            deadline_slack_ms=100.0,
            planner_available=True,
        )

        self.assertEqual(replan.mode, ExecutionMode.ACCURATE_VLA)
        self.assertEqual(escalate.mode, ExecutionMode.PLANNER)
        self.assertIsNone(escalate.recovery_skill_id)

    def test_joint_controller_can_disable_request_level_inference_steps(self) -> None:
        controller = JointRecoveryComputeController(
            JointControllerConfig(
                fast_inference_steps=None,
                accurate_inference_steps=None,
                low_risk_commit=25,
                high_risk_commit=5,
            )
        )
        risk = RiskAssessment(
            score=0.0,
            bucket="low",
            event=None,
            evidence={},
            components={},
        )

        decision = controller.decide(
            risk,
            deadline_ms=100.0,
            deadline_slack_ms=100.0,
        )

        self.assertEqual(decision.mode, ExecutionMode.FAST_VLA)
        self.assertIsNone(decision.controls.inference_steps)
        self.assertEqual(decision.controls.max_actions, 25)

    def test_confirmed_stall_exhaustion_escalates_or_stops(self) -> None:
        risk = RiskAssessment(
            score=0.9,
            bucket="high",
            event="stall",
            evidence={"event_streak": 3},
            components={"stall": 1.0},
        )
        controller = JointRecoveryComputeController()

        stop = controller.decide(
            risk,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            recovery_attempts=2,
            planner_available=False,
        )
        planner = controller.decide(
            risk,
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            recovery_attempts=2,
            planner_available=True,
        )

        self.assertEqual(stop.mode, ExecutionMode.SAFE_STOP)
        self.assertEqual(planner.mode, ExecutionMode.PLANNER)

    def test_stateful_cartesian_recovery_executes_and_records_outcome(self) -> None:
        spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )
        memory = RecoveryMemory(max_records=4)
        executor = StatefulRecoveryExecutor(memory)
        plan = build_cartesian_retreat_plan(
            spec,
            [0.2, -0.1, -0.2, 0.0, 0.0, 0.0, -1.0],
        )
        context = RecoveryContext(
            episode_id="episode-physical",
            trigger_event="stall",
            attempt=0,
            timestep=20,
        )

        executor.start(plan, context)
        commands = [executor.next_command() for _ in plan.phases]
        self.assertEqual([item.phase_name for item in commands], [
            "stabilize",
            "retract",
            "lift",
            "settle_reobserve",
        ])
        self.assertEqual(executor.status, RecoveryStatus.AWAITING_VERIFICATION)
        self.assertTrue(commands[-1].request_verification)
        self.assertTrue(all(action[-1] == -1.0 for item in commands for action in item.actions))
        self.assertGreater(commands[2].actions[0][2], 0.0)

        outcome = executor.resolve_verification(
            True,
            evidence={"state_response": 0.02},
        )
        self.assertEqual(outcome.status, RecoveryStatus.SUCCEEDED)
        self.assertEqual(outcome.actions_executed, 12)
        self.assertTrue(outcome.request_replan)
        self.assertEqual(memory.success_rate(trigger_event="stall", skill_id=plan.skill_id), 1.0)
        self.assertEqual(plan.verification_rule, "reobserve_and_confirm_progress")
        self.assertTrue(plan.safe_hold_on_failure)
        self.assertGreater(plan.timeout_s, 0.0)

    def test_recovery_registry_builds_release_skill_with_open_gripper(self) -> None:
        spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )
        registry = RecoverySkillRegistry.with_default_skills()

        plan = registry.build(
            "release_retract_lift_reobserve",
            spec,
            [0.2, -0.1, -0.2, 0.0, 0.0, 0.0, -1.0],
        )

        self.assertEqual(
            registry.skill_ids,
            (
                "cartesian_retract_lift_reobserve",
                "release_retract_lift_reobserve",
            ),
        )
        self.assertEqual(plan.skill_id, "release_retract_lift_reobserve")
        self.assertTrue(
            all(action[-1] == 1.0 for phase in plan.phases for action in phase.actions)
        )
        self.assertEqual(plan.action_count, 12)

    def test_release_recovery_builder_requires_open_gripper_contract(self) -> None:
        spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="0=open,1=close",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )

        with self.assertRaisesRegex(ValueError, "gripper convention"):
            build_release_retreat_plan(spec, [0.0] * 7)

    def test_failure_memory_filters_expired_structured_evidence(self) -> None:
        memory = FailureMemory(max_records=3)
        shared = {
            "episode_id": "episode-memory",
            "task": "place the mug on the plate",
            "subgoal": "grasp mug",
            "policy_id": "pi05-base",
            "deployment_profile_id": "bf16-2step-h10",
            "failure_type": "misgrasp",
            "monitor_evidence": {"progress": 0.0, "event_streak": 3},
            "action_age_steps": 1,
            "intervention": "cartesian_retract_lift_reobserve",
            "retry_budget_consumed": 1,
            "recovery_budget_consumed": 1,
            "verification_result": "object_not_lifted",
            "terminal_outcome": "failed",
            "confidence": 0.8,
        }
        expired = FailureEpisodeRecord(
            context_fingerprint="scene-a",
            created_at_s=100.0,
            expires_at_s=110.0,
            **shared,
        )
        active = FailureEpisodeRecord(
            context_fingerprint="scene-a",
            created_at_s=101.0,
            expires_at_s=120.0,
            **shared,
        )
        other = FailureEpisodeRecord(
            context_fingerprint="scene-b",
            created_at_s=102.0,
            expires_at_s=120.0,
            **shared,
        )
        memory.record(expired)
        memory.record(active)
        memory.record(other)

        records = memory.retrieve(
            context_fingerprint="scene-a",
            failure_type="misgrasp",
            now_s=115.0,
        )

        self.assertEqual(records, (active,))
        self.assertEqual(len(memory), 3)
        self.assertNotIn("actions", active.to_dict())

    def test_cartesian_recovery_projects_finite_seed_to_action_bounds(self) -> None:
        spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )

        plan = build_cartesian_retreat_plan(
            spec,
            [1.4, -1.2, -0.2, 0.0, 0.0, 0.0, -1.1],
        )
        actions = [action for phase in plan.phases for action in phase.actions]

        spec.validate(actions)
        self.assertEqual(plan.phases[1].actions[0][0], -0.5)
        self.assertTrue(all(action[-1] == -1.0 for action in actions))

    def test_joint_controller_uses_reuse_and_safe_stop_modes(self) -> None:
        monitor = ExecutionRiskMonitor()
        low = monitor.update(
            proprio=np.zeros(8),
            commanded_action=np.zeros(7),
            action_age_steps=0,
        )
        controller = JointRecoveryComputeController()
        reuse = controller.decide(
            low,
            deadline_ms=80.0,
            deadline_slack_ms=60.0,
            cached_actions=4,
        )
        self.assertEqual(reuse.mode, ExecutionMode.REUSE)
        self.assertEqual(reuse.controls.inference_steps, 2)

        high = type(low)(
            score=0.9,
            bucket="high",
            event=None,
            components={"stall": 0.0},
            evidence={},
        )
        stop = controller.decide(
            high,
            deadline_ms=80.0,
            deadline_slack_ms=2.0,
        )
        self.assertEqual(stop.mode, ExecutionMode.SAFE_STOP)

    def test_jsonl_sink_records_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "calls.jsonl"
            runtime = CarveRuntime(
                Pi05Adapter(FakeRemotePi05()),
                trace_sink=JsonlTraceSink(path),
            )
            runtime.infer(InferenceRequest(observation={}, instruction="move"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["success"])
            self.assertEqual(rows[0]["adapter_id"], "pi05")

    def test_trace_context_is_recorded_but_not_sent_to_policy(self) -> None:
        policy = FakeRemotePi05()
        runtime = CarveRuntime(Pi05Adapter(policy))
        bridge = LegacyPolicyClientBridge(runtime)
        bridge.infer(
            {
                "prompt": "move",
                "runtime_trace_context": {"mode": "fast_vla", "risk": 0.2},
            }
        )
        self.assertNotIn("runtime_trace_context", policy.payloads[0])
        self.assertEqual(runtime.last_trace.metadata["controller"]["mode"], "fast_vla")

    def test_server_wrapper_applies_and_restores_per_call_steps(self) -> None:
        policy = FakeLocalPi05()
        controlled = RuntimeControllablePolicy(
            policy,
            maximum_inference_steps=4,
            deployment_metadata={"backend_id": "eager"},
        )
        output = controlled.infer(
            {"prompt": "move", "runtime_controls": {"inference_steps": 2}}
        )
        self.assertEqual(policy.calls[0][1], 2)
        self.assertEqual(policy._sample_kwargs["num_steps"], 7)
        self.assertTrue(output["runtime_controls"]["applied"])
        self.assertTrue(
            controlled.metadata["carve_capabilities"]["configurable_inference_steps"]
        )
        self.assertEqual(
            controlled.metadata["carve_deployment_profile"]["backend_id"],
            "eager",
        )

    def test_server_wrapper_forwards_deterministic_noise(self) -> None:
        policy = FakeNoisePolicy()
        controlled = RuntimeControllablePolicy(policy)
        noise = np.ones((10, 32), dtype=np.float32)
        output = controlled.infer({"prompt": "move", "runtime_noise": noise})
        np.testing.assert_array_equal(policy.noise_seen, noise)
        self.assertTrue(output["runtime_controls"]["deterministic_noise"])

    def test_server_wrapper_rejects_steps_outside_fixed_profile(self) -> None:
        controlled = RuntimeControllablePolicy(
            FakeLocalPi05(),
            minimum_inference_steps=2,
            maximum_inference_steps=2,
        )

        with self.assertRaisesRegex(ValueError, r"\[2, 2\]"):
            controlled.infer(
                {"prompt": "move", "runtime_controls": {"inference_steps": 1}}
            )
        capabilities = controlled.metadata["carve_capabilities"]
        self.assertEqual(capabilities["minimum_inference_steps"], 2)
        self.assertEqual(capabilities["maximum_inference_steps"], 2)

    def test_simulator_snapshot_restores_exact_state(self) -> None:
        env = FakeBranchEnv()
        snapshot = capture_simulator_snapshot(
            env,
            task_id=8,
            episode_id=2,
            timestep=120,
        )
        env.state += 10.0
        observation = restore_simulator_snapshot(env, snapshot)
        np.testing.assert_array_equal(observation["state"], snapshot.state)
        self.assertEqual(len(snapshot.fingerprint), 64)

    def test_failure_snapshot_writer_applies_episode_budget_and_cooldown(self) -> None:
        env = FakeBranchEnv()
        with tempfile.TemporaryDirectory() as directory:
            writer = FailureSnapshotWriter(
                directory,
                max_per_episode=2,
                minimum_gap_steps=10,
            )
            first = writer.write(
                env,
                task_id=8,
                episode_id=0,
                timestep=20,
                trigger="stall",
                instruction="move",
                observation={"proprio": np.zeros(8)},
                cached_actions=[np.zeros(7)],
                controller={"mode": "recovery"},
            )
            blocked = writer.write(
                env,
                task_id=8,
                episode_id=0,
                timestep=25,
                trigger="stall",
                instruction="move",
                observation={"proprio": np.zeros(8)},
                cached_actions=[],
                controller={"mode": "recovery"},
            )
            second = writer.write(
                env,
                task_id=8,
                episode_id=0,
                timestep=30,
                trigger="stall",
                instruction="move",
                observation={"proprio": np.zeros(8)},
                cached_actions=[],
                controller={"mode": "recovery"},
            )
            self.assertIsNotNone(first)
            self.assertIsNone(blocked)
            self.assertIsNotNone(second)
            metadata = json.loads(pathlib.Path(first["metadata"]).read_text())
            self.assertFalse(metadata["controller_uses_privileged_state"])


if __name__ == "__main__":
    unittest.main()
