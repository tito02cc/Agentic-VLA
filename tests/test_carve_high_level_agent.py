"""CPU-only tests for the guarded CARVE high-level agent."""

from __future__ import annotations

import json
import threading
import unittest

from agentic_vla.runtime import (
    AgentIntent,
    AgenticHarnessController,
    AnthropicVisionPlanner,
    AsyncAgenticHarnessController,
    AsyncGuardedHighLevelAgent,
    ExecutionMode,
    FailureEpisodeRecord,
    FailureMemory,
    GuardedHighLevelAgent,
    HarnessState,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    JointRecoveryComputeController,
    PausedExecutionSafeHoldAdapter,
    PlannerProviderConfig,
    PrimitiveBoundaryPolicy,
    RiskAssessment,
    SharedBackboneVisionPlanner,
    TaskPreservingRecoveryPlanner,
    TaskStartPolicy,
    build_high_level_agent_request,
    build_vision_planner,
    compose_grounded_vla_instruction,
)
from agentic_vla.toolchain import PrimitiveOutcome, PrimitiveStatus


def _context(**overrides) -> HighLevelAgentContext:
    values = {
        "task_instruction": "put the mug on the plate",
        "trigger": "repeated_failure",
        "episode_id": "task6:0",
        "timestep": 120,
        "robot_state": (0.1, 0.2, 0.3),
        "risk": {"event": "stall", "score": 0.8},
        "current_subgoal": "grasp mug",
        "failure_history": ("stall", "recovery_failed"),
        "memory": ("previous lift did not restore motion",),
        "available_skills": ("cartesian_retract_lift_reobserve",),
        "remaining_retries": 1,
        "remaining_recoveries": 1,
        "deadline_slack_ms": 45.0,
    }
    values.update(overrides)
    return HighLevelAgentContext(**values)


def _risk() -> RiskAssessment:
    return RiskAssessment(
        score=0.8,
        bucket="high",
        event="stall",
        components={"stall": 1.0},
        evidence={"event_streak": 3},
    )


class HighLevelAgentTest(unittest.TestCase):
    def test_grounded_vla_instruction_preserves_long_horizon_task(self) -> None:
        prompt = compose_grounded_vla_instruction(
            "put the mug in the microwave and close it",
            subgoal="close microwave door",
            planner_instruction="close the microwave door",
        )

        self.assertTrue(prompt.startswith("put the mug in the microwave and close it"))
        self.assertIn("Current recovery subgoal: close the microwave door", prompt)

    def test_request_exposes_typed_context_without_raw_actions(self) -> None:
        request = build_high_level_agent_request(_context())
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["trigger"], "repeated_failure")
        self.assertEqual(
            payload["available_skills"],
            ["cartesian_retract_lift_reobserve"],
        )
        self.assertNotIn("actions", payload)

    def test_request_defines_stale_subgoal_correction_policy(self) -> None:
        request = build_high_level_agent_request(
            _context(trigger="stale_subgoal", current_subgoal="old task")
        )

        self.assertIn("trigger is stale_subgoal", request["system_prompt"])
        self.assertIn("replace the stale command using vla_act", request["system_prompt"])

    def test_no_progress_request_exposes_harness_owned_active_plan(self) -> None:
        task_plan = {
            "installed": True,
            "completed": False,
            "active_stage": "top",
            "steps": [
                {
                    "stage": "top",
                    "intent": "vla_act",
                    "subgoal": "complete the top tower structure",
                    "expected_outcome": "top structure is visibly complete",
                    "skill_id": None,
                    "constraints": [],
                    "status": "active",
                    "attempts": 0,
                    "verification": None,
                }
            ],
        }

        request = build_high_level_agent_request(
            _context(trigger="no_progress", task_plan=task_plan)
        )
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["task_plan"]["active_stage"], "top")
        self.assertIn("Harness-owned task plan is installed", request["system_prompt"])
        self.assertIn("do not skip, replace, or rewrite stages", request["system_prompt"])

    def test_no_progress_exposes_registered_skill_contract_and_budget(self) -> None:
        spec = {
            "skill_id": "visual_reobserve",
            "arguments": {"view": "head | wrist"},
            "postcondition": "fresh view of the workspace",
        }
        request = build_high_level_agent_request(
            _context(
                trigger="no_progress",
                available_skills=("visual_reobserve",),
                available_skill_specs=(spec,),
            )
        )
        payload = json.loads(request["user_prompt"])
        self.assertEqual(payload["available_skill_specs"], [spec])
        self.assertEqual(payload["available_skills"], ["visual_reobserve"])
        self.assertEqual(payload["remaining_recoveries"], 1)
        self.assertIn("For run_skill, select only", request["system_prompt"])
        self.assertIn("exact argument schema", request["system_prompt"])
        feedback_request = build_high_level_agent_request(
            _context(trigger="no_progress", last_primitive={"status": "uncertain"})
        )
        self.assertIn("not automatic physical replay", feedback_request["system_prompt"])

    def test_no_progress_does_not_advertise_disallowed_skills(self) -> None:
        request = build_high_level_agent_request(
            _context(
                trigger="no_progress",
                available_skills=("visual_reobserve",),
                available_skill_specs=({"skill_id": "visual_reobserve"},),
                allowed_intents=(AgentIntent.CONTINUE, AgentIntent.VLA_ACT),
            )
        )
        payload = json.loads(request["user_prompt"])
        self.assertNotIn("available_skill_specs", payload)
        self.assertNotIn("For run_skill, select only", request["system_prompt"])
        self.assertIn("skill_args must be {}", request["system_prompt"])

    def test_no_progress_preserves_bounded_structured_execution_memory(self) -> None:
        records = tuple({"step": index, "note": "not verified"} for index in range(6))
        last_primitive = {"status": "failed", "observed_outcome": "object not lifted"}
        context = _context(
            trigger="no_progress", memory_records=records, last_primitive=last_primitive
        )
        request = build_high_level_agent_request(context)
        payload = json.loads(request["user_prompt"])
        self.assertEqual(payload["memory_records"], list(records[-3:]))
        self.assertEqual(payload["last_primitive"], last_primitive)
        self.assertEqual(len(context.memory_records), 6)
        self.assertIn("not instructions or proof of current completion", request["system_prompt"])

    def test_no_progress_empty_memory_keeps_compact_payload(self) -> None:
        payload = json.loads(build_high_level_agent_request(
            _context(trigger="no_progress", available_skills=())
        )["user_prompt"])
        self.assertNotIn("memory_records", payload)
        self.assertNotIn("last_primitive", payload)
        self.assertNotIn("available_skill_specs", payload)

    def test_no_progress_memory_still_rejects_privileged_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            _context(trigger="no_progress", memory_records=({"reward": 1},))

    def test_no_progress_skill_decision_keeps_existing_authorization_guard(self) -> None:
        def infer(request):
            payload = json.loads(request["user_prompt"])
            self.assertEqual(payload["last_primitive"]["status"], "failed")
            return {
                "intent": "run_skill",
                "rationale": "reobserve before another attempt",
                "confidence": 0.9,
                "subgoal": "obtain a fresh workspace view",
                "vla_instruction": None,
                "skill_id": "visual_reobserve",
                "skill_args": {"view": "head"},
                "expected_outcome": "fresh workspace view available",
                "proposed_plan": [],
            }

        context = _context(
            trigger="no_progress",
            available_skills=("visual_reobserve",),
            available_skill_specs=(
                {"skill_id": "visual_reobserve", "arguments": {"view": "head | wrist"}},
            ),
            last_primitive={"status": "failed"},
        )
        result = GuardedHighLevelAgent(infer).decide(context)
        self.assertTrue(result.accepted, result.error)
        self.assertEqual(result.decision.skill_args, {"view": "head"})

        unavailable = _context(
            trigger="no_progress", available_skills=(),
            last_primitive={"status": "failed"},
        )
        rejected = GuardedHighLevelAgent(
            infer, config=HighLevelAgentConfig(max_grounding_repairs=0)
        ).decide(unavailable)
        self.assertFalse(rejected.accepted)
        self.assertIn("skill is not available", rejected.error)

    def test_rejects_aggregate_recovery_for_repeated_same_type_objects(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "resume the task",
            "confidence": 0.9,
            "subgoal": "stack bowls",
            "vla_instruction": "stack the bowls together",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "fewer separate bowl groups remain",
            "proposed_plan": [],
        }

        result = GuardedHighLevelAgent(
            lambda _request: payload,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="no_progress",
                task_instruction="stack the three bowls together",
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("ordinal or spatial qualifier", result.error)

    def test_accepts_grounded_recovery_for_repeated_same_type_objects(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "select one visible bowl pair",
            "confidence": 0.9,
            "subgoal": "stack left bowl with center bowl",
            "vla_instruction": "stack the left bowl with the center bowl",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two separate bowl groups remain",
            "proposed_plan": [],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="no_progress",
                task_instruction="stack the three bowls together",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.decision.vla_instruction,
            "stack the left bowl with the center bowl",
        )

    def test_rejects_recovery_spatial_label_absent_from_critic_groups(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "select one visible pair",
            "confidence": 0.9,
            "subgoal": "stack left bowl with center bowl",
            "vla_instruction": "stack the left bowl with the center bowl",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two separate bowl groups remain",
            "proposed_plan": [],
        }

        result = GuardedHighLevelAgent(
            lambda _request: payload,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="no_progress",
                task_instruction="stack the three bowls together",
                risk={
                    "event": "semantic_deadline_incomplete",
                    "observed_groups": ["black bowl at left", "black bowl at right"],
                },
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("absent from the visual critic groups", result.error)

    def test_request_exposes_compiled_distinct_object_plan_contract(self) -> None:
        request = build_high_level_agent_request(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put both the alphabet soup and the tomato sauce in the basket"
                ),
            )
        )
        payload = json.loads(request["user_prompt"])

        self.assertEqual(
            payload["task_plan_contract"],
            {
                "pattern": "distinct_objects_shared_destination",
                "exact_stage_count": 2,
                "ordered_requirements": [
                    "put the alphabet soup in the basket",
                    "put the tomato sauce in the basket",
                ],
                "final_expected_outcome": (
                    "full task visibly satisfied: put both the alphabet soup and "
                    "the tomato sauce in the basket"
                ),
            },
        )

    def test_task_start_request_is_compact_and_excludes_runtime_history(self) -> None:
        request = build_high_level_agent_request(
            _context(
                trigger="task_start",
                task_instruction="Stack the three bowls together.",
            )
        )
        payload = json.loads(request["user_prompt"])

        self.assertLess(len(request["system_prompt"]), 2200)
        self.assertEqual(
            set(payload),
            {
                "protocol",
                "task_instruction",
                "task_plan_contract",
                "available_skill_specs",
            },
        )
        self.assertEqual(payload["protocol"], "semantic_stages_v1")
        self.assertEqual(
            payload["task_plan_contract"]["pattern"],
            "three_instance_stack",
        )
        self.assertEqual(payload["task_plan_contract"]["exact_stage_count"], 2)
        self.assertNotIn("robot_state", payload)
        self.assertIn("One row describes one complete", request["system_prompt"])
        self.assertIn("only colors explicitly named by the task", request["system_prompt"])

    def test_compiles_minimal_task_plan_proposal_into_guarded_decision(self) -> None:
        payload = {
            "confidence": 0.9,
            "proposed_plan": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the left bowl with the center bowl",
                    "expected_outcome": "two bowls form one partial stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the right bowl with the partial stack",
                    "expected_outcome": "all three bowls form one stack",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="Stack the three bowls together.",
            )
        )

        self.assertTrue(result.accepted, result.error)
        self.assertEqual(result.decision.intent, AgentIntent.VLA_ACT)
        self.assertEqual(
            result.decision.vla_instruction,
            "stack the left bowl with the center bowl",
        )
        self.assertEqual(
            result.decision.rationale,
            "execute first guarded task-plan stage",
        )

    def test_three_instance_stack_requires_two_semantic_stages(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "form the partial stack",
            "confidence": 0.9,
            "subgoal": "stack the left bowl with the center bowl",
            "vla_instruction": "place a bowl on the partial stack",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two bowls form one partial stack",
            "memory_note": "",
            "failure_type": "",
            "scene_graph_update": {},
            "proposed_plan": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the left bowl with the center bowl",
                    "expected_outcome": "two bowls form one partial stack",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the remaining bowl with the partial stack",
                    "expected_outcome": "the remaining bowl completes the stack",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="Stack the three bowls together.",
            )
        )

        self.assertTrue(result.accepted, result.error)
        self.assertEqual(len(result.decision.proposed_plan), 2)
        self.assertEqual(
            result.decision.vla_instruction,
            "stack the left bowl with the center bowl",
        )

    def test_rejects_ambiguous_three_instance_stack_plan(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "form the partial stack",
            "confidence": 0.9,
            "subgoal": "stack two visible bowls",
            "vla_instruction": "stack two visible bowls",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two bowls form one partial stack",
            "proposed_plan": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack two visible bowls",
                    "expected_outcome": "two bowls form one partial stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the remaining bowl with the partial stack",
                    "expected_outcome": "the remaining bowl completes the stack",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(
            lambda _request: payload,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction="Stack the three bowls together.",
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("two distinct spatial labels", result.error)

    def test_accepts_third_spatial_position_in_final_stack_stage(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "form the partial stack",
            "confidence": 0.9,
            "subgoal": "stack the left bowl with the center bowl",
            "vla_instruction": "stack the left bowl with the center bowl",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two bowls form one partial stack",
            "proposed_plan": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the left bowl with the center bowl",
                    "expected_outcome": "two bowls form one partial stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the right bowl with the partial stack",
                    "expected_outcome": "the right bowl completes the stack",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="Stack the three bowls together.",
            )
        )

        self.assertTrue(result.accepted, result.error)

    def test_request_compiles_three_stage_tower_contract(self) -> None:
        request = build_high_level_agent_request(
            _context(
                trigger="task_start",
                task_instruction=(
                    "Build a tower using the wooden blocks and wooden boards."
                ),
            )
        )
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["task_plan_contract"]["pattern"], "tower_assembly")
        self.assertEqual(payload["task_plan_contract"]["exact_stage_count"], 3)
        self.assertIn(
            "middle",
            payload["task_plan_contract"]["ordered_requirements"][1],
        )

    def test_request_hides_plan_contract_after_task_start(self) -> None:
        request = build_high_level_agent_request(_context(trigger="repeated_failure"))
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["task_plan_contract"], {})

    def test_request_prevents_nominal_appearance_only_safe_stop(self) -> None:
        request = build_high_level_agent_request(
            _context(trigger="control_boundary", current_subgoal="execute")
        )

        self.assertIn("do not safe_stop only because object appearance", request["system_prompt"])
        self.assertIn("reserve safe_stop for a clear hazard", request["system_prompt"])
        self.assertIn("Never select an object that is absent", request["system_prompt"])
        self.assertIn("action-free proposed_plan", request["system_prompt"])
        self.assertIn("Select only its first step", request["system_prompt"])
        self.assertIn("skip mappings that are visibly complete", request["system_prompt"])
        self.assertIn("scene_graph_update must be an empty object", request["system_prompt"])
        self.assertIn("return exactly two physical stages", request["system_prompt"])

    def test_accepts_bounded_task_start_plan(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "execute first visible stage",
            "confidence": 0.92,
            "subgoal": "place first moka pot",
            "vla_instruction": "place the first moka pot on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first moka pot is on stove",
            "proposed_plan": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": ["keep the second pot undisturbed"],
                },
                {
                    "stage": "place_second",
                    "intent": "vla_act",
                    "subgoal": "place second moka pot",
                    "expected_outcome": "both moka pots are on stove",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="put both moka pots on the stove",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.decision.proposed_plan), 2)
        self.assertEqual(result.decision.proposed_plan[1].stage, "place_second")

    def test_normalizes_semantic_plan_intent_aliases_without_granting_tools(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "execute first placement",
            "confidence": 0.92,
            "subgoal": "place first moka pot",
            "vla_instruction": "place the first moka pot on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first moka pot is on stove",
            "proposed_plan": [
                {
                    "stage": 1,
                    "intent": "place",
                    "subgoal": "place first moka pot",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": 2,
                    "intent": "pick_and_place",
                    "subgoal": "place second moka pot",
                    "expected_outcome": "both moka pots are on stove",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="put both moka pots on the stove",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            [step.intent for step in result.decision.proposed_plan],
            ["vla_act", "vla_act"],
        )
        self.assertTrue(all(step.skill_id is None for step in result.decision.proposed_plan))

    def test_minimal_task_plan_normalizes_task_verb_to_vla_executor(self) -> None:
        payload = {
            "confidence": 0.8,
            "proposed_plan": [
                {
                    "stage": 1,
                    "intent": "stack",
                    "subgoal": "stack left bowl with center bowl",
                    "expected_outcome": "left bowl and center bowl form a partial stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": 2,
                    "intent": "stack",
                    "subgoal": "stack right bowl with the partial stack",
                    "expected_outcome": "full task visibly satisfied: stack the three bowls together",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="stack the three bowls together",
            )
        )

        self.assertTrue(result.accepted, result.error)
        self.assertEqual(result.decision.intent, AgentIntent.VLA_ACT)
        self.assertEqual(
            [step.intent for step in result.decision.proposed_plan],
            ["vla_act", "vla_act"],
        )
        self.assertTrue(all(step.skill_id is None for step in result.decision.proposed_plan))

    def test_normalizes_motion_plan_intent_for_vla_without_granting_tools(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "the cube and destination are visible",
            "confidence": 0.92,
            "subgoal": "move cube to target",
            "vla_instruction": "move the cube to the target",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "cube is at target",
            "proposed_plan": [
                {
                    "stage": "move_cube",
                    "intent": "move cube to target",
                    "subgoal": "move cube to target",
                    "expected_outcome": "cube is at target",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="move the cube to the target",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.proposed_plan[0].intent, "vla_act")
        self.assertIsNone(result.decision.proposed_plan[0].skill_id)

    def test_normalizes_real_planner_plan_without_lifecycle_pseudo_steps(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "execute first placement",
            "confidence": 0.9,
            "subgoal": "place first moka pot on stove",
            "vla_instruction": "place the first moka pot on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first moka pot is on stove",
            "scene_graph_update": {
                "objects": ["moka pot", "stove"],
                "source": "vision",
            },
            "proposed_plan": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot on stove",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": "pi05",
                    "skill_args": {},
                    "constraints": [],
                    "vla_instruction": "place the first moka pot on the stove",
                },
                {
                    "stage": "place_second",
                    "intent": "vla_act",
                    "subgoal": "place second moka pot on stove",
                    "expected_outcome": "both moka pots are on stove",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
                {
                    "stage": "verify",
                    "intent": "run_skill",
                    "subgoal": "verify task completion",
                    "expected_outcome": "task is complete",
                    "skill_id": "stove_placement_check",
                    "skill_args": {},
                    "constraints": [],
                },
                {
                    "stage": "finish",
                    "intent": "safe_stop",
                    "subgoal": "finish task",
                    "expected_outcome": "robot is stopped",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="put both moka pots on the stove",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            [step.stage for step in result.decision.proposed_plan],
            ["place_first", "place_second"],
        )
        self.assertIsNone(result.decision.proposed_plan[0].skill_id)
        self.assertEqual(result.decision.scene_graph_update["entities"], ())
        self.assertEqual(result.decision.scene_graph_update["relations"], ())

    def test_normalizes_container_plan_wording_and_placeholder_padding(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "bowl and drawer visible",
            "confidence": 0.95,
            "subgoal": "place black bowl in bottom drawer",
            "vla_instruction": "place the black bowl into the bottom drawer",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "bowl is inside drawer",
            "proposed_plan": [
                {
                    "stage": "place the black bowl in the bottom drawer of the cabinet",
                    "intent": "vla_act",
                    "subgoal": "place the black bowl in the bottom drawer of the cabinet",
                    "expected_outcome": "bowl is inside the drawer",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": ["verify containment before closing"],
                },
                {
                    "stage": "close the bottom drawer",
                    "intent": "vla_act",
                    "subgoal": "close the bottom drawer",
                    "expected_outcome": "drawer is closed",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
                {
                    "stage": "verify task completion",
                    "intent": "continue",
                    "subgoal": "verify task completion",
                    "expected_outcome": "task is complete",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
                {
                    "stage": "none",
                    "intent": "none",
                    "subgoal": "none",
                    "expected_outcome": "none",
                    "skill_id": None,
                    "skill_args": {},
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put the black bowl in the bottom drawer of the cabinet and close it"
                ),
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.decision.proposed_plan), 2)
        self.assertEqual(
            result.decision.subgoal,
            "place the black bowl in the bottom drawer of the cabinet",
        )
        self.assertEqual(result.decision.expected_outcome, "bowl is inside the drawer")

    def test_rejects_plan_rewrite_after_task_start(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "rewrite plan after failure",
            "confidence": 0.9,
            "subgoal": "place mug",
            "vla_instruction": "put the mug on the plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is on plate",
            "proposed_plan": [
                {
                    "stage": "replace_plan",
                    "intent": "vla_act",
                    "subgoal": "place mug",
                    "expected_outcome": "mug is on plate",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(_context())

        self.assertFalse(result.accepted)
        self.assertIn("only at task_start", result.error)

    def test_accepts_continue_echo_for_harness_owned_active_step(self) -> None:
        task_plan = {
            "installed": True,
            "completed": False,
            "active_stage": "2",
            "steps": [
                {
                    "stage": "2",
                    "intent": "vla_act",
                    "subgoal": "Place second moka pot on the stove burner",
                    "expected_outcome": "Second moka pot is on the stove burner",
                    "skill_id": None,
                    "constraints": [],
                    "status": "active",
                    "attempts": 0,
                    "verification": None,
                }
            ],
        }
        payload = {
            "intent": "continue",
            "rationale": "first pot placed; proceed to second",
            "confidence": 0.95,
            "subgoal": "Place second moka pot on the stove burner",
            "vla_instruction": None,
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "Second moka pot is on the stove burner",
            "proposed_plan": list(task_plan["steps"]),
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="scheduled_semantic_checkpoint",
                task_instruction="put both moka pots on the stove",
                task_plan=task_plan,
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.CONTINUE)
        self.assertEqual(result.decision.proposed_plan, ())

    def test_repairs_continue_that_incorrectly_selects_vla_executor(self) -> None:
        task_plan = {
            "installed": True,
            "completed": False,
            "active_stage": "2",
            "steps": [
                {
                    "stage": "2",
                    "intent": "vla_act",
                    "subgoal": "place tomato sauce in basket",
                    "expected_outcome": "sauce in basket",
                    "skill_id": None,
                    "constraints": [],
                    "status": "retry_required",
                    "attempts": 1,
                    "verification": None,
                }
            ],
        }
        malformed = {
            "intent": "continue",
            "rationale": "retry active stage",
            "confidence": 0.9,
            "subgoal": "place tomato sauce in basket",
            "vla_instruction": "place the tomato sauce in the basket",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "sauce in basket",
            "proposed_plan": [],
            "failure_type": "",
            "scene_graph_update": {},
        }
        repaired = {**malformed, "vla_instruction": None}
        calls = []

        def infer(request):
            calls.append(request)
            return malformed if len(calls) == 1 else repaired

        result = GuardedHighLevelAgent(infer).decide(
            _context(
                trigger="scheduled_semantic_checkpoint",
                task_instruction=(
                    "put both the alphabet soup and the tomato sauce in the basket"
                ),
                task_plan=task_plan,
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(result.decision.intent, AgentIntent.CONTINUE)
        self.assertIsNone(result.decision.vla_instruction)
        feedback = json.loads(calls[1]["user_prompt"])["validation_feedback"]
        self.assertIn("must not select an executor", feedback["reason"])

    def test_repairs_grasp_place_motor_phase_split(self) -> None:
        split = {
            "intent": "vla_act",
            "rationale": "start with grasp",
            "confidence": 0.9,
            "subgoal": "grasp first moka pot",
            "vla_instruction": "grasp the first moka pot",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first moka pot is held",
            "proposed_plan": [
                {
                    "stage": "grasp_first",
                    "intent": "vla_act",
                    "subgoal": "grasp first moka pot",
                    "expected_outcome": "first moka pot is held",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot on stove",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }
        repaired = {
            **split,
            "rationale": "execute first placement",
            "subgoal": "place first moka pot on stove",
            "vla_instruction": "place the first moka pot on the stove",
            "expected_outcome": "first moka pot is on stove",
            "proposed_plan": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot on stove",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place_second",
                    "intent": "vla_act",
                    "subgoal": "place second moka pot on stove",
                    "expected_outcome": "both moka pots are on stove",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }
        calls = []

        def infer(request):
            calls.append(request)
            return split if len(calls) == 1 else repaired

        result = GuardedHighLevelAgent(infer).decide(
            _context(
                trigger="task_start",
                task_instruction="put both moka pots on the stove",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(calls), 2)
        self.assertIn("combine them", calls[1]["user_prompt"])
        self.assertEqual(result.decision.subgoal, "place first moka pot on stove")

    def test_rejects_partial_single_stage_for_both_object_task(self) -> None:
        partial = {
            "intent": "vla_act",
            "rationale": "execute first placement",
            "confidence": 0.9,
            "subgoal": "place first moka pot on stove",
            "vla_instruction": "place the first moka pot on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first moka pot is on stove",
            "proposed_plan": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot on stove",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        result = GuardedHighLevelAgent(
            lambda _request: partial,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction="put both moka pots on the stove",
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("requires exactly two", result.error)

    def test_rejects_partial_plan_for_two_put_clause_task(self) -> None:
        partial = {
            "intent": "vla_act",
            "rationale": "execute the first placement",
            "confidence": 0.9,
            "subgoal": "place the white mug on the left plate",
            "vla_instruction": "place the white mug on the left plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "white mug is on the left plate",
            "proposed_plan": [
                {
                    "stage": "place_white_mug",
                    "intent": "vla_act",
                    "subgoal": "place the white mug on the left plate",
                    "expected_outcome": "white mug is on the left plate",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        result = GuardedHighLevelAgent(
            lambda _request: partial,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put the white mug on the left plate and put the yellow and "
                    "white mug on the right plate"
                ),
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("requires exactly two ordered", result.error)

    def test_accepts_ordered_plan_for_two_put_clause_task(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "preserve the left and right assignments",
            "confidence": 0.9,
            "subgoal": "place the white mug on the left plate",
            "vla_instruction": "place the white mug on the left plate",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "white mug is on the left plate",
            "proposed_plan": [
                {
                    "stage": "place_white_mug",
                    "intent": "vla_act",
                    "subgoal": "place the white mug on the left plate",
                    "expected_outcome": "white mug is on the left plate",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place_yellow_white_mug",
                    "intent": "vla_act",
                    "subgoal": "place the yellow and white mug on the right plate",
                    "expected_outcome": "yellow and white mug is on the right plate",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put the white mug on the left plate and put the yellow and "
                    "white mug on the right plate"
                ),
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.decision.proposed_plan), 2)

    def test_rejects_partial_plan_for_distinct_both_object_container_task(self) -> None:
        partial = {
            "intent": "vla_act",
            "rationale": "place the first object",
            "confidence": 0.9,
            "subgoal": "place alphabet soup in basket",
            "vla_instruction": "put the alphabet soup in the basket",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "alphabet soup is in basket",
            "proposed_plan": [
                {
                    "stage": "place_soup",
                    "intent": "vla_act",
                    "subgoal": "place alphabet soup in basket",
                    "expected_outcome": "alphabet soup is in basket",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }
        result = GuardedHighLevelAgent(
            lambda _request: partial,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put both the alphabet soup and the tomato sauce in the basket"
                ),
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("requires exactly two ordered", result.error)

    def test_accepts_ordered_plan_for_distinct_both_object_container_task(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "place the first object",
            "confidence": 0.9,
            "subgoal": "place alphabet soup in basket",
            "vla_instruction": "put the alphabet soup in the basket",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "alphabet soup is in basket",
            "proposed_plan": [
                {
                    "stage": "place_soup",
                    "intent": "vla_act",
                    "subgoal": "place alphabet soup in basket",
                    "expected_outcome": "alphabet soup is in basket",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place_sauce",
                    "intent": "vla_act",
                    "subgoal": "place tomato sauce in basket",
                    "expected_outcome": "tomato sauce is in basket",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }
        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put both the alphabet soup and the tomato sauce in the basket"
                ),
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.decision.proposed_plan), 2)

    def test_rejects_reversed_plan_for_turn_then_place_task(self) -> None:
        reversed_plan = {
            "intent": "vla_act",
            "rationale": "place object first",
            "confidence": 0.9,
            "subgoal": "place moka pot on stove",
            "vla_instruction": "put the moka pot on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "moka pot is on stove",
            "proposed_plan": [
                {
                    "stage": "place",
                    "intent": "vla_act",
                    "subgoal": "place moka pot on stove",
                    "expected_outcome": "moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "activate",
                    "intent": "vla_act",
                    "subgoal": "turn on stove",
                    "expected_outcome": "stove is on",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }
        result = GuardedHighLevelAgent(
            lambda _request: reversed_plan,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction="turn on the stove and put the moka pot on it",
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("activate the appliance in its first stage", result.error)

    def test_accepts_ordered_plan_for_turn_then_place_task(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "activate before placement",
            "confidence": 0.9,
            "subgoal": "turn on stove",
            "vla_instruction": "turn on the stove",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "stove is on",
            "proposed_plan": [
                {
                    "stage": "activate",
                    "intent": "vla_act",
                    "subgoal": "turn on stove",
                    "expected_outcome": "stove is on",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place",
                    "intent": "vla_act",
                    "subgoal": "place moka pot on stove",
                    "expected_outcome": "moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }
        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(
                trigger="task_start",
                task_instruction="turn on the stove and put the moka pot on it",
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.decision.proposed_plan), 2)

    def test_rejects_multiple_objects_for_singular_compound_color_container_task(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "start first placement",
            "confidence": 0.9,
            "subgoal": "place first mug in microwave",
            "vla_instruction": "place the yellow mug in the microwave",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "first mug is inside",
            "proposed_plan": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first mug in microwave",
                    "expected_outcome": "first mug is inside",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "place_second",
                    "intent": "vla_act",
                    "subgoal": "place second mug in microwave",
                    "expected_outcome": "second mug is inside",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "close",
                    "intent": "vla_act",
                    "subgoal": "close microwave",
                    "expected_outcome": "microwave is closed",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        }

        result = GuardedHighLevelAgent(
            lambda _request: payload,
            config=HighLevelAgentConfig(max_grounding_repairs=0),
        ).decide(
            _context(
                trigger="task_start",
                task_instruction=(
                    "put the yellow and white mug in the microwave and close it"
                ),
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("singular 'put X in Y and close it'", result.error)

    def test_rejects_decision_that_skips_active_task_plan_step(self) -> None:
        task_plan = {
            "installed": True,
            "completed": False,
            "active_stage": "place_first",
            "steps": [
                {
                    "stage": "place_first",
                    "intent": "vla_act",
                    "subgoal": "place first moka pot",
                    "expected_outcome": "first moka pot is on stove",
                    "skill_id": None,
                    "constraints": [],
                    "status": "active",
                    "attempts": 1,
                    "verification": None,
                }
            ],
        }
        result = GuardedHighLevelAgent(
            lambda _request: {
                "intent": "vla_act",
                "rationale": "skip to second object",
                "confidence": 0.9,
                "subgoal": "place second moka pot",
                "vla_instruction": "place the second moka pot on the stove",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "both moka pots are on stove",
                "proposed_plan": [],
            }
        ).decide(
            _context(
                trigger="scheduled_semantic_checkpoint",
                task_instruction="put both moka pots on the stove",
                task_plan=task_plan,
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("does not match active task_plan step", result.error)

    def test_harness_preserves_cumulative_outcome_for_matching_active_step(self) -> None:
        cumulative = "full task visibly satisfied: put both moka pots on the stove"
        task_plan = {
            "installed": True,
            "completed": False,
            "active_stage": "place_second",
            "steps": [
                {
                    "stage": "place_second",
                    "intent": "vla_act",
                    "subgoal": "place second moka pot on stove",
                    "expected_outcome": cumulative,
                    "skill_id": None,
                    "constraints": [],
                    "status": "retry_required",
                    "attempts": 1,
                    "verification": None,
                }
            ],
        }
        result = GuardedHighLevelAgent(
            lambda _request: {
                "intent": "vla_act",
                "rationale": "retry the second placement",
                "confidence": 0.9,
                "subgoal": "place second moka pot on stove",
                "vla_instruction": "place the second moka pot on the stove",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "second moka pot is on the stove",
                "proposed_plan": [],
                "failure_type": "contradicted",
                "scene_graph_update": {},
            }
        ).decide(
            _context(
                trigger="scheduled_semantic_checkpoint",
                task_instruction="put both moka pots on the stove",
                task_plan=task_plan,
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.expected_outcome, cumulative)

    def test_request_exposes_bounded_primitive_outcome(self) -> None:
        outcome = PrimitiveOutcome(
            call_id="vla-1",
            primitive_name="vla_act",
            status=PrimitiveStatus.SUCCEEDED,
            episode_id="task6:0",
            started_timestep=100,
            ended_timestep=120,
            expected_outcome="mug is held",
            requires_semantic_check=True,
        )

        request = build_high_level_agent_request(
            _context(last_primitive=outcome.to_planner_dict())
        )
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["last_primitive"]["primitive_name"], "vla_act")
        self.assertEqual(payload["last_primitive"]["status"], "succeeded")
        self.assertNotIn("actions", payload["last_primitive"])

    def test_provider_factory_keeps_model_vendor_outside_harness(self) -> None:
        openai_compatible = build_vision_planner(
            PlannerProviderConfig(
                provider="openai_compatible",
                model="local-qwen-vl",
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
            )
        )
        anthropic = build_vision_planner(
            PlannerProviderConfig(
                provider="anthropic",
                model="claude-test",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            environ={"ANTHROPIC_API_KEY": "secret"},
        )

        self.assertEqual(openai_compatible.model, "local-qwen-vl")
        self.assertIsInstance(anthropic, AnthropicVisionPlanner)

    def test_shared_backbone_planner_limits_images_and_unwraps_text(self) -> None:
        observed = {}

        def transport(payload):
            observed.update(payload)
            return {
                "ok": True,
                "data": {"text": '{"intent":"continue"}'},
            }

        planner = SharedBackboneVisionPlanner(
            transport,
            max_tokens=64,
            max_images=1,
        )
        result = planner(
            {
                "system_prompt": "Return JSON.",
                "user_prompt": "Inspect progress.",
                "frames": {"head": "head-frame", "wrist": "wrist-frame"},
            }
        )

        self.assertEqual(result, '{"intent":"continue"}')
        self.assertEqual(observed["frames"], {"head": "head-frame"})
        self.assertEqual(observed["max_new_tokens"], 64)

    def test_task_preserving_replan_is_restricted_to_no_progress(self) -> None:
        planner = TaskPreservingRecoveryPlanner()
        request = build_high_level_agent_request(
            _context(trigger="no_progress", risk={"event": "no_progress", "score": 0.8})
        )

        decision = planner(request)

        self.assertEqual(decision["intent"], "continue")
        self.assertEqual(decision["subgoal"], "put the mug on the plate")
        self.assertIsNone(decision["vla_instruction"])
        self.assertEqual(decision["failure_type"], "no_progress")
        self.assertNotIn("joint", json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "restricted"):
            planner(build_high_level_agent_request(_context(trigger="task_start")))

    def test_task_preserving_retry_passes_guard_without_retargeting(self) -> None:
        for instruction in (
            "Stack the three bowls together.",
            "Build a tower using the blocks and boards.",
            "Put the bottles into the dustbin.",
        ):
            with self.subTest(instruction=instruction):
                agent = GuardedHighLevelAgent(TaskPreservingRecoveryPlanner())
                result = agent.decide(
                    _context(
                        task_instruction=instruction,
                        trigger="no_progress",
                        risk={"event": "no_progress", "score": 0.8},
                    )
                )
                self.assertTrue(result.accepted, result.error)
                self.assertEqual(result.decision.intent, AgentIntent.CONTINUE)
                self.assertEqual(result.decision.subgoal, instruction)
                self.assertIsNone(result.decision.vla_instruction)
                self.assertEqual(result.attempt_count, 1)

    def test_anthropic_provider_requires_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "API key"):
            build_vision_planner(
                PlannerProviderConfig(
                    provider="anthropic",
                    model="claude-test",
                    api_key_env="ANTHROPIC_API_KEY",
                ),
                environ={},
            )

    def test_accepts_registered_skill_decision(self) -> None:
        def infer(_request):
            return {
                "intent": "run_skill",
                "rationale": "the contact attempt remains stalled",
                "confidence": 0.9,
                "subgoal": "restore a reachable pre-contact state",
                "vla_instruction": None,
                "skill_id": "cartesian_retract_lift_reobserve",
                "skill_args": {},
                "expected_outcome": "end effector responds and scene can be reobserved",
                "memory_note": "do not repeat the failed contact pose",
            }

        result = GuardedHighLevelAgent(infer).decide(_context())

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.RUN_SKILL)
        self.assertEqual(
            result.decision.skill_id,
            "cartesian_retract_lift_reobserve",
        )

    def test_accepts_one_pure_json_code_block(self) -> None:
        raw = """```json
{
  "intent": "vla_act",
  "rationale": "bind the visible mug as the next contact target",
  "confidence": 0.9,
  "subgoal": "grasp mug",
  "vla_instruction": "grasp the visible mug",
  "skill_id": null,
  "skill_args": {},
  "expected_outcome": "the mug is held",
  "memory_note": ""
}
```"""

        result = GuardedHighLevelAgent(lambda _request: raw).decide(_context())

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.VLA_ACT)

    def test_repairs_truncated_json_with_compact_structured_output(self) -> None:
        calls = []

        def infer(request):
            calls.append(request)
            if len(calls) == 1:
                return '{"intent":"vla_act","rationale":"truncated'
            return {
                "intent": "vla_act",
                "rationale": "visible target supports retry",
                "confidence": 0.9,
                "subgoal": "grasp mug",
                "vla_instruction": "grasp the visible mug",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "mug is held",
                "proposed_plan": [],
                "scene_graph_update": {},
            }

        result = GuardedHighLevelAgent(infer).decide(_context())

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempt_count, 2)
        self.assertIn("below 400 tokens", calls[1]["user_prompt"])
        self.assertIn("truncated or malformed", result.validation_errors[0])

    def test_repairs_truncated_fenced_json_with_remaining_budget(self) -> None:
        calls = []
        repaired = {
            "intent": "vla_act",
            "rationale": "visible target supports retry",
            "confidence": 0.9,
            "subgoal": "grasp mug",
            "vla_instruction": "grasp the visible mug",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is held",
            "proposed_plan": [],
            "scene_graph_update": {},
        }

        def infer(request):
            calls.append(request)
            return "```json\n{\"intent\":\"vla_act\"" if len(calls) == 1 else repaired

        result = GuardedHighLevelAgent(infer).decide(_context())

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempt_count, 2)
        self.assertIn("truncated or malformed", result.validation_errors[0])

    def test_malformed_optional_scene_graph_does_not_discard_valid_intent(self) -> None:
        result = GuardedHighLevelAgent(
            lambda _request: {
                "intent": "vla_act",
                "rationale": "retry the visible mug",
                "confidence": 0.9,
                "subgoal": "insert mug",
                "vla_instruction": "put the mug in the microwave",
                "skill_id": None,
                "skill_args": {},
                "scene_graph_update": {
                    "entities": ["mug", "microwave"],
                    "relations": [],
                },
            }
        ).decide(_context())

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.VLA_ACT)
        self.assertEqual(result.decision.scene_graph_update, {})

    def test_optional_scene_graph_still_rejects_privileged_state(self) -> None:
        result = GuardedHighLevelAgent(
            lambda _request: {
                "intent": "continue",
                "rationale": "continue current execution",
                "confidence": 0.9,
                "subgoal": "grasp mug",
                "vla_instruction": None,
                "skill_id": None,
                "skill_args": {},
                "scene_graph_update": {
                    "object_pose": [0.1, 0.2, 0.3],
                },
            }
        ).decide(_context())

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)
        self.assertIn("object_pose", result.error)

    def test_rejects_prose_around_json_code_block(self) -> None:
        raw = "Here is the decision:\n```json\n{\"intent\":\"safe_stop\"}\n```"

        result = GuardedHighLevelAgent(lambda _request: raw).decide(_context())

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)

    def test_rejects_unknown_skill_and_fails_closed(self) -> None:
        def infer(_request):
            return {
                "intent": "run_skill",
                "rationale": "try an unregistered controller",
                "confidence": 0.99,
                "subgoal": "recover",
                "vla_instruction": None,
                "skill_id": "teleport_object",
                "skill_args": {},
                "expected_outcome": "object moves",
                "memory_note": "",
            }

        result = GuardedHighLevelAgent(infer).decide(_context())

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)
        self.assertIn("not available", result.error)

    def test_rejects_raw_joint_action_output(self) -> None:
        result = GuardedHighLevelAgent(
            lambda _request: {
                "intent": "continue",
                "rationale": "move directly",
                "confidence": 0.9,
                "actions": [[1.0] * 7],
            }
        ).decide(_context())

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)
        self.assertIn("forbidden action fields", result.error)

    def test_rejects_low_confidence_intervention(self) -> None:
        def infer(_request):
            return {
                "intent": "vla_act",
                "rationale": "a new grasp might work",
                "confidence": 0.2,
                "subgoal": "regrasp mug",
                "vla_instruction": "regrasp the mug from the side",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "mug is held",
                "memory_note": "",
            }

        result = GuardedHighLevelAgent(infer).decide(_context())

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)
        self.assertIn("confidence", result.error)

    def test_rejects_novel_color_target_absent_from_task(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "select the visible distractor",
            "confidence": 0.9,
            "subgoal": "grasp red mug",
            "vla_instruction": "grasp the red mug and place it on the plate",
            "skill_id": None,
            "skill_args": {},
        }
        result = GuardedHighLevelAgent(
            lambda _request: payload,
            HighLevelAgentConfig(max_calls_per_episode=1),
        ).decide(_context(task_instruction="put the yellow and white mug on the plate"))

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision.intent, AgentIntent.SAFE_STOP)
        self.assertIn("color referents absent", result.error)
        self.assertEqual(result.validation_errors, (result.error,))

    def test_accepts_compound_task_color_target(self) -> None:
        payload = {
            "intent": "vla_act",
            "rationale": "task target is visible",
            "confidence": 0.9,
            "subgoal": "grasp yellow-white mug",
            "vla_instruction": "place the yellow-white mug on the left plate",
            "skill_id": None,
            "skill_args": {},
        }
        result = GuardedHighLevelAgent(lambda _request: payload).decide(
            _context(task_instruction="put the yellow and white mug on the left plate")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.validation_errors, ())

    def test_repairs_novel_color_target_once_with_validator_feedback(self) -> None:
        requests = []

        def infer(request):
            requests.append(request)
            if len(requests) == 1:
                target = "red mug"
            else:
                feedback = json.loads(request["user_prompt"])["validation_feedback"]
                self.assertIn("red", feedback["reason"])
                target = "yellow and white mug"
            return {
                "intent": "vla_act",
                "rationale": "select task target",
                "confidence": 0.9,
                "subgoal": f"grasp {target}",
                "vla_instruction": f"put the {target} on the left plate",
                "skill_id": None,
                "skill_args": {},
            }

        agent = GuardedHighLevelAgent(
            infer,
            HighLevelAgentConfig(
                max_calls_per_episode=2,
                max_grounding_repairs=1,
            ),
        )
        result = agent.decide(
            _context(task_instruction="put the yellow and white mug on the left plate")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(agent.calls_in_episode, 2)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.validation_errors), 1)
        self.assertEqual(len(result.raw_outputs), 2)
        self.assertIn("red mug", result.raw_outputs[0]["vla_instruction"])
        self.assertIn("yellow and white mug", result.decision.vla_instruction)

    def test_enforces_per_episode_call_budget(self) -> None:
        payload = {
            "intent": "safe_stop",
            "rationale": "no safe intervention remains",
            "confidence": 1.0,
            "subgoal": "",
            "vla_instruction": None,
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "robot remains stationary",
            "memory_note": "",
        }
        agent = GuardedHighLevelAgent(
            lambda _request: payload,
            HighLevelAgentConfig(max_calls_per_episode=1),
        )

        first = agent.decide(_context())
        second = agent.decide(_context(timestep=121))

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertIn("budget exhausted", second.error)

    def test_harness_invokes_agent_only_on_planner_escalation(self) -> None:
        calls = []

        def infer(_request):
            calls.append(True)
            return {
                "intent": "vla_act",
                "rationale": "rebind the local grasp after repeated recovery failure",
                "confidence": 0.9,
                "subgoal": "regrasp mug",
                "vla_instruction": "grasp the mug from its visible side",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "mug is secured",
                "memory_note": "avoid the previous approach direction",
            }

        harness = AgenticHarnessController(
            JointRecoveryComputeController(),
            GuardedHighLevelAgent(infer),
        )
        normal = harness.decide(
            RiskAssessment(
                score=0.0,
                bucket="low",
                event=None,
                components={},
                evidence={"event_streak": 0},
            ),
            context=_context(trigger="task_start"),
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
        )
        escalated = harness.decide(
            _risk(),
            context=_context(),
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            repeated_failures=2,
        )

        self.assertEqual(normal.joint.mode, ExecutionMode.FAST_VLA)
        self.assertIsNone(normal.high_level)
        self.assertEqual(escalated.joint.mode, ExecutionMode.PLANNER)
        self.assertTrue(escalated.high_level.accepted)
        self.assertEqual(len(calls), 1)

    def test_async_agent_is_single_flight_and_boundary_timeout_is_non_destructive(self) -> None:
        release = threading.Event()

        def infer(_request):
            release.wait(timeout=1.0)
            return {
                "intent": "continue",
                "rationale": "the existing local objective remains valid",
                "confidence": 0.9,
                "subgoal": "grasp mug",
                "vla_instruction": None,
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "execution can continue",
                "memory_note": "",
            }

        async_agent = AsyncGuardedHighLevelAgent(GuardedHighLevelAgent(infer))
        try:
            ticket = async_agent.submit(_context())
            self.assertTrue(async_agent.pending)
            self.assertIsNone(async_agent.take(wait=True, timeout_s=0.001))
            self.assertTrue(async_agent.pending)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                async_agent.submit(_context(timestep=121))

            release.set()
            completed = async_agent.take(wait=True, timeout_s=1.0)
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.ticket.ticket_id, ticket.ticket_id)
            self.assertTrue(completed.result.accepted)
            self.assertFalse(async_agent.pending)
        finally:
            release.set()
            async_agent.close()

    def test_harness_without_agent_fails_safe_on_repeated_failures(self) -> None:
        harness = AgenticHarnessController(JointRecoveryComputeController())

        result = harness.decide(
            _risk(),
            context=_context(),
            deadline_ms=80.0,
            deadline_slack_ms=50.0,
            repeated_failures=2,
        )

        self.assertEqual(result.joint.mode, ExecutionMode.SAFE_STOP)
        self.assertIsNone(result.high_level)

    def test_async_wrapper_is_single_flight_and_nonblocking(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def infer(_request):
            started.set()
            self.assertTrue(release.wait(timeout=2.0))
            return {
                "intent": "vla_act",
                "rationale": "visible target supports a fresh VLA attempt",
                "confidence": 0.9,
                "subgoal": "grasp mug",
                "vla_instruction": "grasp the visible mug",
                "skill_id": None,
                "skill_args": {},
                "expected_outcome": "mug is held",
                "memory_note": "",
            }

        with AsyncGuardedHighLevelAgent(GuardedHighLevelAgent(infer)) as planner:
            ticket = planner.submit(_context())
            self.assertTrue(started.wait(timeout=1.0))
            self.assertTrue(planner.pending)
            self.assertIsNone(planner.take(wait=False))
            with self.assertRaisesRegex(RuntimeError, "already active"):
                planner.submit(_context(timestep=121))
            release.set()
            completed = planner.take(wait=True)

        self.assertIsNotNone(completed)
        self.assertEqual(completed.ticket.ticket_id, ticket.ticket_id)
        self.assertTrue(completed.result.accepted)
        self.assertEqual(completed.result.decision.intent, AgentIntent.VLA_ACT)

    def test_multi_rate_harness_event_only_does_not_call_planner_at_start(self) -> None:
        planner_calls = []

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(lambda _request: planner_calls.append(True))
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.EVENT_ONLY,
        )
        try:
            transition = harness.start_episode("task6:0")
            self.assertEqual(transition.state, HarnessState.EXECUTE_FAST)
            self.assertFalse(harness.planner_pending)
            self.assertEqual(planner_calls, [])
            self.assertEqual(harness.counters.planner_calls, 0)
        finally:
            harness.close()

    def test_explicit_safe_boundary_requests_post_recovery_semantic_check(self) -> None:
        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(
                    lambda _request: {
                        "intent": "vla_act",
                        "rationale": "rebind the target after recovery",
                        "confidence": 0.9,
                        "subgoal": "regrasp mug",
                        "vla_instruction": "regrasp the visible mug by its handle",
                        "skill_id": None,
                        "skill_args": {},
                    }
                )
            )

        hold = PausedExecutionSafeHoldAdapter()
        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            safe_hold_adapter=hold,
        )
        try:
            harness.start_episode("task6:0")
            submitted = harness.request_planner(
                _context(episode_id="task6:0", trigger="post_recovery_verification"),
                purpose="post_recovery_verification",
                reason="verify semantic progress",
            )
            self.assertEqual(submitted.state, HarnessState.PLAN_AT_SAFE_BOUNDARY)
            self.assertIsNotNone(submitted.ticket)
            self.assertIsNotNone(hold.active)

            completed = harness.await_planner(timeout_s=1.0)
            self.assertTrue(completed.decision_applied)
            self.assertEqual(completed.state, HarnessState.EXECUTE_FAST)
            self.assertIsNone(hold.active)
        finally:
            harness.close()

    def test_selective_boundary_policy_uses_vlm_only_for_semantic_checkpoints(self) -> None:
        calls = []

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(
                    lambda request: calls.append(json.loads(request["user_prompt"]))
                    or {
                        "intent": "continue",
                        "rationale": "primitive outcome matches the expected subgoal",
                        "confidence": 0.9,
                        "subgoal": "place mug",
                        "vla_instruction": None,
                        "skill_id": None,
                        "skill_args": {},
                    }
                )
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            primitive_boundary_policy=PrimitiveBoundaryPolicy.SELECTIVE,
            planner_cooldown_steps=0,
        )
        try:
            harness.start_episode("task6:0")
            routine = harness.record_primitive_boundary(
                _context(timestep=140),
                PrimitiveOutcome(
                    call_id="move-1",
                    primitive_name="move_to",
                    status="succeeded",
                    episode_id="task6:0",
                    started_timestep=120,
                    ended_timestep=140,
                ),
            )
            self.assertIsNone(routine.ticket)
            self.assertEqual(calls, [])

            semantic = harness.record_primitive_boundary(
                _context(timestep=160),
                PrimitiveOutcome(
                    call_id="vla-1",
                    primitive_name="vla_act",
                    status="succeeded",
                    episode_id="task6:0",
                    started_timestep=140,
                    ended_timestep=160,
                    expected_outcome="mug is held",
                    requires_semantic_check=True,
                ),
            )
            self.assertIsNotNone(semantic.ticket)
            completed = harness.await_planner(timeout_s=1.0)
            self.assertTrue(completed.decision_applied)
            self.assertEqual(calls[0]["trigger"], "primitive_completed")
            self.assertEqual(calls[0]["last_primitive"]["call_id"], "vla-1")
        finally:
            harness.close()

    def test_abnormal_primitive_always_escalates_in_event_only_mode(self) -> None:
        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(
                    lambda _request: {
                        "intent": "run_skill",
                        "rationale": "retract after the failed contact primitive",
                        "confidence": 0.9,
                        "subgoal": "restore pre-contact state",
                        "vla_instruction": None,
                        "skill_id": "cartesian_retract_lift_reobserve",
                        "skill_args": {},
                    }
                )
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            primitive_boundary_policy=PrimitiveBoundaryPolicy.EVENT_ONLY,
            planner_cooldown_steps=0,
        )
        try:
            harness.start_episode("task6:0")
            submitted = harness.record_primitive_boundary(
                _context(timestep=150),
                PrimitiveOutcome(
                    call_id="vla-failed",
                    primitive_name="vla_act",
                    status="failed",
                    episode_id="task6:0",
                    started_timestep=120,
                    ended_timestep=150,
                    expected_outcome="mug is held",
                    observed_outcome="gripper closed without object motion",
                ),
            )
            self.assertIsNotNone(submitted.ticket)
            completed = harness.await_planner(timeout_s=1.0)
            self.assertEqual(completed.state, HarnessState.RECOVER)
            self.assertEqual(
                completed.planner.result.decision.skill_id,
                "cartesian_retract_lift_reobserve",
            )
        finally:
            harness.close()

    def test_canonical_planner_cooldown_suppresses_immediate_reentry(self) -> None:
        calls = []

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(
                    lambda _request: calls.append(True)
                    or {
                        "intent": "continue",
                        "rationale": "continue after semantic verification",
                        "confidence": 0.9,
                        "subgoal": "grasp mug",
                        "vla_instruction": None,
                        "skill_id": None,
                        "skill_args": {},
                    }
                )
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            planner_cooldown_steps=50,
        )
        try:
            harness.start_episode("task6:0")
            harness.request_planner(
                _context(
                    episode_id="task6:0",
                    timestep=100,
                    trigger="post_recovery_verification",
                ),
                purpose="post_recovery_verification",
                reason="verify recovery",
            )
            harness.await_planner(timeout_s=1.0)
            suppressed = harness.request_planner(
                _context(
                    episode_id="task6:0",
                    timestep=108,
                    trigger="stall",
                ),
                purpose="event_escalation",
                reason="retry semantic verification",
            )

            self.assertIsNone(suppressed.ticket)
            self.assertIn("cooldown", suppressed.reason)
            self.assertEqual(len(calls), 1)
        finally:
            harness.close()

    def test_multi_rate_harness_startup_shadow_never_applies_decision(self) -> None:
        result = {
            "intent": "vla_act",
            "rationale": "bind the visible mug",
            "confidence": 0.9,
            "subgoal": "grasp mug",
            "vla_instruction": "grasp the visible mug",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is held",
            "proposed_plan": [
                {
                    "stage": "grasp",
                    "intent": "vla_act",
                    "subgoal": "grasp mug",
                    "expected_outcome": "mug is held",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(lambda _request: result)
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.STARTUP_SHADOW,
        )
        try:
            submitted = harness.start_episode(
                "task6:0",
                task_start_context=_context(
                    trigger="task_start",
                    timestep=0,
                    failure_history=(),
                    memory=(),
                    available_skills=(),
                    remaining_recoveries=0,
                ),
            )
            self.assertIsNotNone(submitted.ticket)
            completed = harness.await_planner(timeout_s=1.0)
            self.assertEqual(completed.state, HarnessState.EXECUTE_FAST)
            self.assertFalse(completed.decision_applied)
            self.assertEqual(harness.counters.planner_calls, 1)
        finally:
            harness.close()

    def test_multi_rate_harness_startup_wait_releases_safe_hold(self) -> None:
        hold = PausedExecutionSafeHoldAdapter()
        result = {
            "intent": "vla_act",
            "rationale": "bind the visible mug",
            "confidence": 0.9,
            "subgoal": "grasp mug",
            "vla_instruction": "grasp the visible mug",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "mug is held",
            "proposed_plan": [
                {
                    "stage": "grasp",
                    "intent": "vla_act",
                    "subgoal": "grasp mug",
                    "expected_outcome": "mug is held",
                    "skill_id": None,
                    "constraints": [],
                }
            ],
        }

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(lambda _request: result)
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.STARTUP_WAIT,
            safe_hold_adapter=hold,
        )
        try:
            submitted = harness.start_episode(
                "task6:0",
                task_start_context=_context(
                    trigger="task_start",
                    timestep=0,
                    failure_history=(),
                    memory=(),
                    available_skills=(),
                    remaining_recoveries=0,
                ),
            )
            self.assertEqual(submitted.state, HarnessState.PLAN_AT_SAFE_BOUNDARY)
            self.assertIsNotNone(hold.active)
            completed = harness.await_planner(timeout_s=1.0)
            self.assertTrue(completed.decision_applied)
            self.assertEqual(completed.joint.mode, ExecutionMode.ACCURATE_VLA)
            self.assertEqual(completed.state, HarnessState.EXECUTE_FAST)
            self.assertIsNone(hold.active)
            self.assertEqual([event["event"] for event in hold.events], ["enter", "release"])
        finally:
            harness.close()

    def test_multi_rate_harness_timeout_remains_in_safe_hold(self) -> None:
        release = threading.Event()
        hold = PausedExecutionSafeHoldAdapter()

        def infer(_request):
            release.wait(timeout=2.0)
            return {
                "intent": "continue",
                "rationale": "continue after delayed verification",
                "confidence": 0.9,
                "subgoal": "",
                "vla_instruction": None,
                "skill_id": None,
                "skill_args": {},
            }

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(GuardedHighLevelAgent(infer))

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.STARTUP_WAIT,
            safe_hold_adapter=hold,
        )
        try:
            harness.start_episode(
                "task6:0",
                task_start_context=_context(
                    trigger="task_start",
                    timestep=0,
                    failure_history=(),
                    memory=(),
                    available_skills=(),
                    remaining_recoveries=0,
                ),
            )
            timed_out = harness.await_planner(timeout_s=0.001)
            self.assertTrue(timed_out.timed_out)
            self.assertEqual(timed_out.state, HarnessState.SAFE_HOLD)
            self.assertIsNotNone(hold.active)
        finally:
            release.set()
            harness.close()

    def test_multi_rate_harness_episode_factory_isolates_planner_budget(self) -> None:
        calls_by_episode = {}

        def planner_factory(episode_id):
            calls_by_episode[episode_id] = 0

            def infer(_request):
                calls_by_episode[episode_id] += 1
                return {
                    "intent": "continue",
                    "rationale": "episode-local decision",
                    "confidence": 0.9,
                    "subgoal": "",
                    "vla_instruction": None,
                    "skill_id": None,
                    "skill_args": {},
                }

            return AsyncGuardedHighLevelAgent(
                GuardedHighLevelAgent(
                    infer,
                    HighLevelAgentConfig(max_calls_per_episode=1),
                )
            )

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.STARTUP_SHADOW,
        )
        try:
            for episode_id in ("task6:0", "task6:1"):
                harness.start_episode(
                    episode_id,
                    task_start_context=_context(
                        episode_id=episode_id,
                        trigger="task_start",
                        timestep=0,
                        failure_history=(),
                        memory=(),
                        available_skills=(),
                        remaining_recoveries=0,
                    ),
                )
                result = harness.await_planner(timeout_s=1.0)
                self.assertTrue(result.planner.result.accepted)
            self.assertEqual(calls_by_episode, {"task6:0": 1, "task6:1": 1})
        finally:
            harness.close()

    def test_multi_rate_harness_counters_are_independent(self) -> None:
        harness = AsyncAgenticHarnessController(JointRecoveryComputeController())
        try:
            harness.start_episode("task6:0")
            harness.record_failure()
            harness.record_failure()
            harness.record_recovery_attempt()
            harness.record_semantic_retry()
            self.assertEqual(
                harness.counters.to_dict(),
                {
                    "failure_streak": 2,
                    "recovery_attempts": 1,
                    "planner_calls": 0,
                    "semantic_retries": 1,
                },
            )
        finally:
            harness.close()

    def test_multi_rate_harness_injects_typed_failure_memory(self) -> None:
        requests = []
        memory = FailureMemory()

        def planner_factory(_episode_id):
            def infer(request):
                requests.append(request)
                return {
                    "intent": "continue",
                    "rationale": "use prior verified evidence",
                    "confidence": 0.9,
                    "subgoal": "",
                    "vla_instruction": None,
                    "skill_id": None,
                    "skill_args": {},
                }

            return AsyncGuardedHighLevelAgent(GuardedHighLevelAgent(infer))

        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            task_start_policy=TaskStartPolicy.STARTUP_SHADOW,
            failure_memory=memory,
        )
        try:
            harness.start_episode(
                "task6:0",
                task_start_context=_context(
                    trigger="task_start",
                    timestep=0,
                    failure_history=(),
                    memory=(),
                    available_skills=(),
                    remaining_recoveries=0,
                ),
            )
            harness.await_planner(timeout_s=1.0)
            harness.record_failure_evidence(
                FailureEpisodeRecord(
                    context_fingerprint="scene-a",
                    episode_id="task6:0",
                    task="put the mug on the plate",
                    subgoal="grasp mug",
                    policy_id="pi05",
                    deployment_profile_id="bf16-smve",
                    failure_type="stall",
                    monitor_evidence={"event_streak": 3},
                    action_age_steps=1,
                    intervention="retract",
                    retry_budget_consumed=1,
                    recovery_budget_consumed=1,
                    verification_result="no_progress",
                    terminal_outcome="failed",
                    confidence=0.8,
                )
            )
            harness.close_episode(reason="memory_setup_complete")
            submitted = harness.start_episode(
                "task6:1",
                task_start_context=_context(
                    episode_id="task6:1",
                    trigger="stall",
                    timestep=0,
                    memory=(),
                    memory_context_fingerprint="scene-a",
                    deployment_profile_id="bf16-smve",
                ),
            )
            self.assertIsNotNone(submitted.ticket)
            harness.await_planner(timeout_s=1.0)
            payload = json.loads(requests[-1]["user_prompt"])
            self.assertEqual(len(payload["memory_records"]), 1)
            self.assertEqual(
                payload["memory_records"][0]["verification_result"],
                "no_progress",
            )
            self.assertNotIn("actions", payload["memory_records"][0])
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()
