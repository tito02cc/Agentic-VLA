from __future__ import annotations

import unittest
import json

from agentic_vla.panda_sorting.contracts import AffordanceProfile, ExperienceCard, RecoveryLevel, SceneObject, SortingTask
from agentic_vla.panda_sorting.reasoning import AffordanceMemory, ReflectionPolicy, SceneGraphReasoner
from agentic_vla.panda_sorting.environment import PandaSortingObservation
from agentic_vla.panda_sorting.perception import ColorLabel, ColorRgbdPerception
from agentic_vla.panda_sorting.harness import AgenticRagVlmHarness
from agentic_vla.panda_sorting.vlm import SORTING_SKILLS, build_sorting_context
from agentic_vla.panda_sorting.skills import SkillConfig
from agentic_vla.panda_sorting.verifier import VisualLiftVerifier, VisualPlacementVerifier
from agentic_vla.panda_sorting.orchestrator import ExecutionMemory, ExecutionMemoryEntry, EventTriggeredSemanticRouter
from agentic_vla.panda_sorting.kitting import (
    DependencyAwareKittingPlanner,
    KittingSkill,
    KittingStateEstimator,
    KittingTask,
)
from agentic_vla.panda_sorting.kitting_scenarios import (
    canonical_kitting_scenes,
    sample_kitting_scene,
)
from agentic_vla.panda_sorting.kitting_vlm import CapabilityGatedKittingVlm
from agentic_vla.runtime import AgentIntent, GuardedHighLevelAgent, HighLevelAgentConfig
import numpy as np


class PandaSortingReasoningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = AffordanceProfile("container", "metal", 0.2, "body_center")

    def test_affordance_retrieval_prefers_functional_match(self) -> None:
        memory = AffordanceMemory(
            [
                ExperienceCard("match", self.profile, "clear_tabletop", 0.12, "parallel_close", "success", 0.7),
                ExperienceCard(
                    "mismatch",
                    AffordanceProfile("flat_package", "cardboard", 0.1, "side_edge"),
                    "clear_tabletop",
                    0.12,
                    "pinch",
                    "success",
                    1.0,
                ),
            ]
        )
        self.assertEqual(memory.retrieve(self.profile, context_tag="clear_tabletop", limit=1)[0].card_id, "match")

    def test_scene_graph_raises_clearance_for_nearby_object(self) -> None:
        target = SceneObject("can", (100, 100), (0.0, 0.0), 0.9)
        other = SceneObject("bread", (104, 102), (0.04, 0.0), 0.9)
        constraints = SceneGraphReasoner().infer(target, [other], target_fragility=0.2)
        self.assertIn("near:bread", constraints.relations)
        self.assertGreater(constraints.approach_height_delta_m, 0.0)
        self.assertEqual(constraints.lateral_escape_direction, (-1.0, 0.0))

    def test_reflection_escalates_missed_grasp(self) -> None:
        policy = ReflectionPolicy()
        first = policy.decide({"object_not_lifted": True}, retries_used=0, replans_used=0)
        second = policy.decide({"object_not_lifted": True}, retries_used=1, replans_used=0)
        self.assertEqual(first.level, RecoveryLevel.PARAMETER_RETRY)
        self.assertEqual(second.level, RecoveryLevel.SKILL_SWITCH)

    def test_vlm_context_exposes_no_simulator_object_state(self) -> None:
        observation = PandaSortingObservation(
            external_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            external_depth=np.zeros((8, 8, 1), dtype=np.float32),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            proprio=np.zeros(32, dtype=np.float64),
            gripper=np.zeros(2, dtype=np.float64),
            eef_position=np.zeros(3, dtype=np.float64),
        )
        context = build_sorting_context(
            observation,
            task_instruction="sort objects",
            episode_id="test",
            timestep=0,
            trigger="task_start",
            current_subgoal="inspect",
        )
        self.assertEqual(context.available_skills, SORTING_SKILLS)
        self.assertEqual(set(context.frames), {"external_rgb", "wrist_rgb"})
        self.assertFalse(hasattr(context, "object_positions"))

    def test_harness_connects_retrieval_constraints_and_semantic_context(self) -> None:
        observation = PandaSortingObservation(
            external_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            external_depth=np.zeros((8, 8, 1), dtype=np.float32),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            proprio=np.zeros(32, dtype=np.float64),
            gripper=np.zeros(2, dtype=np.float64),
            eef_position=np.zeros(3, dtype=np.float64),
        )
        card = ExperienceCard("can-clear", self.profile, "clear_tabletop", 0.12, "parallel_close", "success", 0.9)
        task = SortingTask(
            "two-object-sort",
            "sort the two objects into assigned bins",
            ("can", "bread"),
            {"can": "right_bin", "bread": "left_bin"},
            {"can": self.profile, "bread": AffordanceProfile("package", "cardboard", 0.1, "side_edge")},
        )
        plan = AgenticRagVlmHarness(memory=AffordanceMemory([card])).plan_subgoal(
            task,
            target_object="can",
            scene_objects=(SceneObject("can", (100, 100), (0.0, 0.0), 0.9), SceneObject("bread", (102, 100), (0.04, 0.0), 0.9)),
            observation=observation,
            episode_id="episode",
            timestep=4,
            context_tag="clear_tabletop",
        )
        self.assertEqual(plan.destination, "right_bin")
        self.assertEqual(plan.retrieval_cards[0].card_id, "can-clear")
        self.assertIn("near:bread", plan.scene_constraints.relations)
        self.assertEqual(plan.planner_context["camera_keys"], ("external_rgb", "wrist_rgb"))

    def test_rgbd_perception_uses_color_and_calibration_only(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        # The Robosuite renderer-backed perception adapter consumes RGB images
        # with vertical orientation corrected internally while retaining
        # calibrated depth-map pixel coordinates.
        rgb[20:34, 18:36] = (231, 76, 60)
        depth = np.ones((64, 64, 1), dtype=np.float64)
        observation = PandaSortingObservation(
            external_rgb=rgb,
            external_depth=depth,
            wrist_rgb=rgb,
            proprio=np.zeros(32),
            gripper=np.zeros(2),
            eef_position=np.zeros(3),
        )
        result = ColorRgbdPerception({"can": ColorLabel("can", (231, 76, 60))}).perceive(
            observation,
            pixel_to_world=np.eye(4),
        )[0]
        self.assertTrue(result.visible)
        self.assertGreater(result.confidence, 0.5)
        self.assertIsNotNone(result.table_position)

    def test_skill_config_has_positive_bounded_motion_values(self) -> None:
        config = SkillConfig()
        self.assertGreater(config.position_tolerance_m, 0.0)
        self.assertGreater(config.max_steps_per_move, 0)
        self.assertGreater(config.close_hold_steps, 0)

    def test_visual_lift_verifier_requires_observed_lateral_motion(self) -> None:
        verifier = VisualLiftVerifier(min_lateral_displacement_m=0.035)
        still_at_source = verifier.verify(
            source_xy=(0.10, 0.20),
            source_height_m=0.90,
            reobserved=SceneObject("cereal", (100, 100), (0.11, 0.20), 0.9, world_position=(0.11, 0.20, 0.90)),
        )
        staged = verifier.verify(
            source_xy=(0.10, 0.20),
            source_height_m=0.90,
            reobserved=SceneObject("cereal", (140, 100), (0.16, 0.20), 0.9, world_position=(0.16, 0.20, 0.98)),
        )
        self.assertFalse(still_at_source.held)
        self.assertEqual(still_at_source.reason, "object_remains_near_source")
        self.assertTrue(staged.held)
        self.assertEqual(staged.reason, "object_lifted_and_moved_with_gripper")

    def test_visual_lift_verifier_rejects_tabletop_sliding(self) -> None:
        outcome = VisualLiftVerifier().verify(
            source_xy=(0.10, 0.20),
            source_height_m=0.90,
            reobserved=SceneObject("cereal", (140, 100), (0.16, 0.20), 0.9, world_position=(0.16, 0.20, 0.90)),
        )
        self.assertFalse(outcome.held)
        self.assertEqual(outcome.reason, "object_moved_without_visual_lift")

    def test_visual_placement_verifier_uses_camera_estimate_and_fixture_only(self) -> None:
        verifier = VisualPlacementVerifier(destination_radius_m=0.08)
        inside = verifier.verify(
            destination_xyz=(0.20, 0.40, 0.84),
            reobserved=SceneObject("can", (30, 30), (0.21, 0.39), 0.9),
        )
        outside = verifier.verify(
            destination_xyz=(0.20, 0.40, 0.84),
            reobserved=SceneObject("can", (30, 30), (0.05, -0.20), 0.9),
        )
        self.assertTrue(inside.placed)
        self.assertFalse(outside.placed)

    def test_visual_placement_verifier_supports_rectangular_fixture(self) -> None:
        verifier = VisualPlacementVerifier(
            destination_half_extents_m=(0.09, 0.12)
        )
        diagonal_inside = verifier.verify(
            destination_xyz=(0.20, 0.40, 0.84),
            reobserved=SceneObject("cereal", (30, 30), (0.28, 0.50), 0.9),
        )
        self.assertTrue(diagonal_inside.placed)

    def test_execution_memory_is_bounded_and_action_free(self) -> None:
        memory = ExecutionMemory(max_entries=2)
        for attempt in range(3):
            memory.append(ExecutionMemoryEntry("grasp cereal", "grasp_missed", "retry", attempt, {"height": 0.0}))
        self.assertEqual(len(memory.to_dict()), 2)
        self.assertNotIn("actions", memory.prompt_records()[0])

    def test_semantic_router_accepts_only_the_registered_skill(self) -> None:
        def planner(_: object) -> dict[str, object]:
            return {
                "intent": "run_skill",
                "rationale": "retry grasp safely",
                "confidence": 0.9,
                "subgoal": "regrasp cereal",
                "vla_instruction": None,
                "skill_id": "retract_and_regrasp",
                "skill_args": {},
            }

        observation = PandaSortingObservation(
            external_rgb=np.zeros((8, 8, 3), dtype=np.uint8), external_depth=np.ones((8, 8, 1)),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8), proprio=np.zeros(32),
            gripper=np.zeros(2), eef_position=np.zeros(3),
        )
        agent = GuardedHighLevelAgent(planner, HighLevelAgentConfig(max_calls_per_episode=1))
        auth = EventTriggeredSemanticRouter(agent).authorize(
            event="grasp_missed", observation=observation, task_instruction="sort cereal", episode_id="episode",
            timestep=4, current_subgoal="grasp cereal", memory=ExecutionMemory(),
            risk={"event": "grasp_missed", "bucket": "medium", "score": 0.8},
            remaining_retries=1, remaining_recoveries=1,
        )
        self.assertTrue(auth.authorized)
        self.assertEqual(auth.result.decision.intent, AgentIntent.RUN_SKILL)

    def test_semantic_router_rejects_vla_action_when_no_vla_is_registered(self) -> None:
        def planner(_: object) -> dict[str, object]:
            return {
                "intent": "vla_act", "rationale": "try policy", "confidence": 0.9,
                "subgoal": "regrasp cereal", "vla_instruction": "grasp cereal", "skill_id": None, "skill_args": {},
            }

        observation = PandaSortingObservation(
            external_rgb=np.zeros((8, 8, 3), dtype=np.uint8), external_depth=np.ones((8, 8, 1)),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8), proprio=np.zeros(32),
            gripper=np.zeros(2), eef_position=np.zeros(3),
        )
        result = EventTriggeredSemanticRouter(GuardedHighLevelAgent(planner)).authorize(
            event="grasp_missed", observation=observation, task_instruction="sort cereal", episode_id="episode",
            timestep=4, current_subgoal="grasp cereal", memory=ExecutionMemory(),
            risk={"event": "grasp_missed", "bucket": "medium", "score": 0.8},
            remaining_retries=1, remaining_recoveries=1,
        )
        self.assertFalse(result.authorized)
        self.assertFalse(result.result.accepted)

    def test_kitting_planner_changes_order_when_primary_target_is_blocked(self) -> None:
        task = self._kitting_task()
        estimator = KittingStateEstimator(
            {
                "cereal_bin": (0.0025, 0.4025, 0.84),
                "can_bin": (0.1975, 0.4025, 0.84),
            }
        )
        planner = DependencyAwareKittingPlanner()
        clear = estimator.estimate(
            task,
            self._scene_objects(
                cereal=(0.10, -0.20), can=(0.15, -0.39), milk=(0.25, -0.09)
            ),
        )
        blocked = estimator.estimate(
            task,
            self._scene_objects(
                cereal=(0.08, -0.20), can=(0.20, -0.20), milk=(-0.15, -0.05)
            ),
        )
        self.assertEqual(planner.select_next(task, clear).object_name, "cereal")
        first = planner.select_next(task, blocked)
        self.assertEqual(first.skill, KittingSkill.PLACE_TARGET)
        self.assertEqual(first.object_name, "can")

        after_can = estimator.estimate(
            task,
            self._scene_objects(
                cereal=(0.08, -0.20), can=(0.1975, 0.4025), milk=(-0.15, -0.05)
            ),
            completed_objects=frozenset({"can"}),
        )
        constrained = planner.select_next(task, after_can)
        self.assertEqual(constrained.skill, KittingSkill.PLACE_TARGET)
        self.assertEqual(constrained.object_name, "cereal")
        self.assertIn("approach is clear", constrained.reason)

    def test_kitting_planner_resolves_destination_occupancy_by_sorting_occupant(self) -> None:
        task = self._kitting_task()
        estimator = KittingStateEstimator(
            {
                "cereal_bin": (0.0025, 0.4025, 0.84),
                "can_bin": (0.1975, 0.4025, 0.84),
            }
        )
        objects = self._scene_objects(
            cereal=(0.10, -0.20),
            can=(0.0025, 0.4025),
            milk=(0.25, -0.09),
        )
        state = estimator.estimate(task, objects)
        subgoal = DependencyAwareKittingPlanner().select_next(task, state)
        self.assertEqual(state.destination_occupants["cereal_bin"], "can")
        self.assertEqual(subgoal.skill, KittingSkill.PLACE_TARGET)
        self.assertEqual(subgoal.object_name, "can")

    def test_canonical_kitting_scenes_are_complete_and_frozen_by_family(self) -> None:
        scenes = canonical_kitting_scenes(seed=23)
        self.assertEqual(set(scenes), {"K0", "K1", "K2", "K3"})
        self.assertEqual({scene.seed for scene in scenes.values()}, {23})
        self.assertEqual(
            set(scenes["K2"].object_xy),
            {"Cereal", "Can", "Milk", "Bread"},
        )
        self.assertEqual(scenes["K2"].visual_label_overrides["Milk"], (240, 60, 55))
        self.assertEqual(
            scenes["K3"].post_first_subgoal_displacement,
            ("Cereal", -0.08, -0.08),
        )

    def test_randomized_kitting_scene_is_reproducible_and_preserves_k1_relation(self) -> None:
        first = sample_kitting_scene("K1", 29)
        repeated = sample_kitting_scene("K1", 29)
        different = sample_kitting_scene("K1", 30)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.object_xy, different.object_xy)
        cereal = np.asarray(first.object_xy["Cereal"])
        can = np.asarray(first.object_xy["Can"])
        distance = float(np.linalg.norm(cereal - can))
        self.assertGreater(distance, 0.11)
        self.assertLessEqual(distance, 0.125)
        self.assertEqual(set(first.object_yaw_rad), {"Cereal", "Can", "Milk", "Bread"})
        self.assertEqual(set(first.object_yaw_rad.values()), {0.0})

    def test_kitting_vlm_can_select_only_precondition_valid_candidate(self) -> None:
        task = self._kitting_task()
        state = KittingStateEstimator(
            {
                "cereal_bin": (0.0025, 0.4025, 0.84),
                "can_bin": (0.1975, 0.4025, 0.84),
            }
        ).estimate(
            task,
            self._scene_objects(
                cereal=(0.08, -0.20),
                can=(0.20, -0.20),
                milk=(-0.15, -0.05),
            ),
        )

        def planner(request: object) -> dict[str, object]:
            skills = json.loads(request["user_prompt"])["available_skills"]
            self.assertEqual(skills, ["place_can_in_can_bin"])
            return {
                "intent": "run_skill",
                "rationale": "clear cereal approach",
                "confidence": 0.9,
                "subgoal": "place can first",
                "vla_instruction": None,
                "skill_id": skills[0],
                "skill_args": {},
            }

        observation = PandaSortingObservation(
            external_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            external_depth=np.ones((8, 8, 1)),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            proprio=np.zeros(32),
            gripper=np.zeros(2),
            eef_position=np.zeros(3),
        )
        decision = CapabilityGatedKittingVlm(
            GuardedHighLevelAgent(planner)
        ).select(
            task,
            state,
            observation,
            episode_id="k1",
            timestep=0,
            memory=ExecutionMemory(),
        )
        self.assertIsNotNone(decision.selected)
        self.assertEqual(decision.selected.object_name, "can")

    def _kitting_task(self) -> KittingTask:
        return KittingTask(
            task_id="dynamic-kitting",
            instruction="Place cereal and can in their assigned kit slots without contacting milk.",
            requested_objects=("cereal", "can"),
            destinations={"cereal": "cereal_bin", "can": "can_bin"},
            affordances={
                "cereal": AffordanceProfile("upright_box", "cardboard", 0.2, "body_center"),
                "can": self.profile,
            },
            protected_objects=("milk",),
        )

    @staticmethod
    def _scene_objects(
        *,
        cereal: tuple[float, float],
        can: tuple[float, float],
        milk: tuple[float, float],
    ) -> dict[str, SceneObject]:
        return {
            "cereal": SceneObject("cereal", (10, 10), cereal, 0.9),
            "can": SceneObject("can", (20, 20), can, 0.9),
            "milk": SceneObject("milk", (30, 30), milk, 0.9),
        }


if __name__ == "__main__":
    unittest.main()
