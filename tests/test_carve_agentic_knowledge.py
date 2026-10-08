"""CPU-only tests for deployable scene-graph and HAA-RAG contracts."""

from __future__ import annotations

import json
import unittest

from agentic_vla.runtime import (
    AsyncAgenticHarnessController,
    AsyncGuardedHighLevelAgent,
    AffordanceExperience,
    AgenticKnowledgeProvider,
    GuardedHighLevelAgent,
    HAAExperienceIndex,
    HighLevelAgentContext,
    JointRecoveryComputeController,
    ProceduralStep,
    ProceduralTaskMemory,
    ProceduralTaskRecord,
    RiskAssessment,
    build_high_level_agent_request,
    parse_semantic_scene_graph,
)


class AgenticKnowledgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene_payload = {
            "entities": [
                {
                    "entity_id": "mug_1",
                    "category": "mug",
                    "attributes": ["white", "handle visible"],
                    "confidence": 0.94,
                },
                {
                    "entity_id": "microwave_1",
                    "category": "microwave",
                    "attributes": ["door open"],
                    "confidence": 0.92,
                },
            ],
            "relations": [
                {
                    "subject": "mug_1",
                    "relation": "near",
                    "object": "microwave_1",
                    "confidence": 0.88,
                }
            ],
        }
        self.experience = AffordanceExperience(
            experience_id="mug-handle-side-grasp",
            object_categories=("mug",),
            affordances=("graspable", "container"),
            grasp_regions=("handle", "side wall"),
            material="ceramic",
            fragility="medium",
            applicable_failures=("misgrasp", "collision"),
            constraints=("keep the rim upright", "verify containment before closure"),
            outcome="stable grasp with visible handle",
            confidence=0.9,
        )
        self.procedure = ProceduralTaskRecord(
            procedure_id="microwave-mug-place-close",
            task_family="microwave mug placement",
            object_categories=("mug", "microwave"),
            steps=(
                ProceduralStep(
                    stage="place",
                    intent="vla_act",
                    subgoal="place mug inside microwave",
                    expected_outcome="mug is contained by microwave",
                    constraints=("keep microwave door open",),
                ),
                ProceduralStep(
                    stage="close",
                    intent="vla_act",
                    subgoal="close microwave door",
                    expected_outcome="microwave door is closed",
                ),
            ),
            verification_result="verified",
            source_episode_id="task9:seed7",
            confidence=0.9,
        )

    def test_builds_scene_graph_and_retrieves_matching_affordance(self) -> None:
        provider = AgenticKnowledgeProvider(HAAExperienceIndex((self.experience,)))
        bundle = provider.build(
            task_instruction="put the mug in the microwave and close it",
            scene_graph=self.scene_payload,
            failure_type="misgrasp",
            timestep=42,
        )

        self.assertIsNotNone(bundle.scene_graph)
        assert bundle.scene_graph is not None
        self.assertEqual(bundle.scene_graph.source, "vlm_observation")
        self.assertEqual(bundle.scene_graph.timestep, 42)
        self.assertEqual(
            bundle.affordance_retrievals[0]["experience_id"],
            "mug-handle-side-grasp",
        )
        self.assertEqual(
            provider.retrieval_metrics(),
            {"build_calls": 1, "affordance_hits": 1, "procedural_hits": 0},
        )

    def test_provider_preserves_injected_empty_memory_instances(self) -> None:
        index = HAAExperienceIndex()
        memory = ProceduralTaskMemory()

        provider = AgenticKnowledgeProvider(index, memory)

        self.assertIs(provider.index, index)
        self.assertIs(provider.procedural_memory, memory)
        self.assertEqual(
            provider.retrieval_metrics(),
            {"build_calls": 0, "affordance_hits": 0, "procedural_hits": 0},
        )

    def test_context_exposes_knowledge_without_metric_pose_or_actions(self) -> None:
        provider = AgenticKnowledgeProvider(
            HAAExperienceIndex((self.experience,)),
            ProceduralTaskMemory((self.procedure,)),
        )
        bundle = provider.build(
            task_instruction="put the mug in the microwave and close it",
            scene_graph=self.scene_payload,
            failure_type="misgrasp",
        )
        context = HighLevelAgentContext(
            task_instruction="put the mug in the microwave and close it",
            trigger="misgrasp",
            episode_id="task9:0",
            timestep=42,
            scene_graph=bundle.scene_graph.to_dict() if bundle.scene_graph else {},
            affordance_retrievals=bundle.affordance_retrievals,
            procedural_retrievals=bundle.procedural_retrievals,
        )

        request = build_high_level_agent_request(context)
        payload = json.loads(request["user_prompt"])

        self.assertEqual(payload["scene_graph"]["source"], "vlm_observation")
        self.assertEqual(
            payload["affordance_retrievals"][0]["object_categories"],
            ["mug"],
        )
        self.assertEqual(
            payload["procedural_retrievals"][0]["procedure_id"],
            "microwave-mug-place-close",
        )
        self.assertNotIn("object_pose", json.dumps(payload))
        self.assertNotIn('"actions"', json.dumps(payload))
        self.assertIn("verified symbolic recipes", request["system_prompt"])
        self.assertIn("skip stages already supported as complete", request["system_prompt"])

    def test_procedural_memory_retrieves_verified_symbolic_stages(self) -> None:
        memory = ProceduralTaskMemory((self.procedure,))
        graph = parse_semantic_scene_graph(self.scene_payload)

        records = memory.retrieve(
            task_instruction="put the mug in the microwave and close it",
            scene_graph=graph,
        )

        self.assertEqual(records[0]["procedure_id"], "microwave-mug-place-close")
        self.assertEqual(records[0]["steps"][1]["subgoal"], "close microwave door")
        self.assertNotIn("action", records[0]["steps"][0])

    def test_procedural_memory_normalizes_underscored_family_and_categories(self) -> None:
        record = ProceduralTaskRecord(
            procedure_id="bowl-drawer-close",
            task_family="bowl_drawer_closure",
            object_categories=("black_bowl", "bottom_drawer"),
            steps=(
                ProceduralStep(
                    stage="place",
                    intent="vla_act",
                    subgoal="place the black bowl in the bottom drawer",
                    expected_outcome="bowl is inside the drawer",
                ),
                ProceduralStep(
                    stage="close",
                    intent="vla_act",
                    subgoal="close the bottom drawer",
                    expected_outcome="drawer is closed",
                ),
            ),
            verification_result="verified",
            source_episode_id="task3:0",
        )

        records = ProceduralTaskMemory((record,)).retrieve(
            task_instruction=(
                "put the black bowl in the bottom drawer of the cabinet and close it"
            )
        )

        self.assertEqual(records[0]["procedure_id"], "bowl-drawer-close")
        self.assertGreater(records[0]["retrieval_score"], 0.0)

    def test_procedural_memory_requires_verified_outcome_and_typed_skill(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified"):
            ProceduralTaskRecord(
                procedure_id="unverified",
                task_family="microwave placement",
                object_categories=("mug",),
                steps=self.procedure.steps,
                verification_result="assumed",
                source_episode_id="task9:probe",
            )
        with self.assertRaisesRegex(ValueError, "requires skill_id"):
            ProceduralStep(
                stage="recover",
                intent="run_skill",
                subgoal="retreat and reobserve",
                expected_outcome="end effector responds",
            )

    def test_rejects_privileged_simulator_pose_recursively(self) -> None:
        payload = {
            **self.scene_payload,
            "metadata": {"detector": {"object_pose": [0.1, 0.2, 0.3]}},
        }

        with self.assertRaisesRegex(ValueError, "object_pose"):
            parse_semantic_scene_graph(payload)

    def test_rejects_unknown_relation_and_unknown_entity(self) -> None:
        bad_relation = {
            **self.scene_payload,
            "relations": [
                {
                    "subject": "mug_1",
                    "relation": "teleported_to",
                    "object": "microwave_1",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_semantic_scene_graph(bad_relation)

        unknown_entity = {
            **self.scene_payload,
            "relations": [
                {
                    "subject": "mug_1",
                    "relation": "near",
                    "object": "drawer_1",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown entity"):
            parse_semantic_scene_graph(unknown_entity)

    def test_rejects_non_object_scene_graph_entries(self) -> None:
        malformed = {
            "entities": ["mug"],
            "relations": [],
        }

        with self.assertRaisesRegex(TypeError, "entity 0 must be a JSON object"):
            parse_semantic_scene_graph(malformed)

    def test_retrieval_drops_failure_only_cards_when_task_categories_match(self) -> None:
        unrelated = AffordanceExperience(
            experience_id="drawer-stall",
            object_categories=("drawer",),
            affordances=("openable",),
            applicable_failures=("stall",),
        )
        index = HAAExperienceIndex((self.experience, unrelated))

        records = index.retrieve(
            task_instruction="put the mug in the microwave",
            failure_type="stall",
        )

        self.assertEqual(
            [record["experience_id"] for record in records],
            ["mug-handle-side-grasp"],
        )

    def test_context_rejects_hidden_raw_action_in_memory(self) -> None:
        with self.assertRaisesRegex(ValueError, "actions"):
            HighLevelAgentContext(
                task_instruction="put the mug in the microwave",
                trigger="repeated_failure",
                episode_id="task9:0",
                timestep=42,
                memory_records=({"nested": {"actions": [[0.0] * 7]}},),
            )

        with self.assertRaisesRegex(ValueError, "trajectory"):
            HighLevelAgentContext(
                task_instruction="put the mug in the microwave",
                trigger="repeated_failure",
                episode_id="task9:0",
                timestep=42,
                procedural_retrievals=({"nested": {"trajectory": [[0.0] * 7]}},),
            )

    def test_harness_persists_vlm_scene_graph_and_injects_haa_retrieval(self) -> None:
        requests = []

        def infer(request):
            requests.append(json.loads(request["user_prompt"]))
            return {
                "intent": "continue",
                "rationale": "the current subgoal remains valid",
                "confidence": 0.9,
                "subgoal": "grasp mug",
                "vla_instruction": None,
                "skill_id": None,
                "skill_args": {},
                "failure_type": "misgrasp",
                "scene_graph_update": self.scene_payload,
            }

        def planner_factory(_episode_id):
            return AsyncGuardedHighLevelAgent(GuardedHighLevelAgent(infer))

        provider = AgenticKnowledgeProvider(
            HAAExperienceIndex((self.experience,)),
            ProceduralTaskMemory((self.procedure,)),
        )
        harness = AsyncAgenticHarnessController(
            JointRecoveryComputeController(),
            planner_factory=planner_factory,
            knowledge_provider=provider,
        )
        context = HighLevelAgentContext(
            task_instruction="put the mug in the microwave and close it",
            trigger="repeated_failure",
            episode_id="task9:0",
            timestep=42,
            risk={"event": "misgrasp", "score": 0.9, "bucket": "high"},
        )
        risk = RiskAssessment(
            score=0.9,
            bucket="high",
            event="misgrasp",
            components={"direct_failure": 1.0},
            evidence={"event_streak": 3},
        )
        try:
            harness.start_episode("task9:0")
            harness.record_failure()
            harness.record_failure()
            transition = harness.route(
                risk,
                context=context,
                deadline_ms=80.0,
                deadline_slack_ms=40.0,
            )
            self.assertIsNotNone(transition.ticket)
            completed = harness.await_planner(timeout_s=1.0)
            self.assertTrue(completed.decision_applied)
            self.assertIsNotNone(harness.scene_graph)
            self.assertEqual(
                requests[0]["affordance_retrievals"][0]["experience_id"],
                "mug-handle-side-grasp",
            )
            self.assertEqual(
                requests[0]["procedural_retrievals"][0]["procedure_id"],
                "microwave-mug-place-close",
            )
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()
