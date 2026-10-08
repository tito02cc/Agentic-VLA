"""Agentic RAG-VLM orchestration without simulator-state or action leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import ExperienceCard, RecoveryDecision, SceneObject, SortingTask
from .environment import PandaSortingObservation
from .reasoning import AffordanceMemory, ReflectionPolicy, SceneConstraints, SceneGraphReasoner
from .vlm import build_sorting_context


@dataclass(frozen=True)
class SortingSubgoalPlan:
    """Auditable hand-off from semantic reasoning to a closed-loop skill executor."""

    target_object: str
    destination: str
    retrieval_cards: tuple[ExperienceCard, ...]
    scene_constraints: SceneConstraints
    planner_context: Mapping[str, object]


class AgenticRagVlmHarness:
    """Compose HAA-RAG, scene constraints, reflection, and episodic memory.

    The harness deliberately stops before continuous action generation. A VLA or
    an analytic skill executor consumes ``SortingSubgoalPlan`` through the same
    interface, which keeps the paper's agentic reasoning distinct from a chosen
    low-level controller.
    """

    def __init__(
        self,
        *,
        memory: AffordanceMemory | None = None,
        scene_reasoner: SceneGraphReasoner | None = None,
        reflection: ReflectionPolicy | None = None,
    ) -> None:
        self.memory = memory or AffordanceMemory()
        self.scene_reasoner = scene_reasoner or SceneGraphReasoner()
        self.reflection = reflection or ReflectionPolicy()

    def plan_subgoal(
        self,
        task: SortingTask,
        *,
        target_object: str,
        scene_objects: Iterable[SceneObject],
        observation: PandaSortingObservation,
        episode_id: str | int,
        timestep: int,
        context_tag: str,
    ) -> SortingSubgoalPlan:
        if target_object not in task.ordered_objects:
            raise ValueError(f"target_object {target_object!r} is not part of the task")
        scene = tuple(scene_objects)
        target = next((item for item in scene if item.name == target_object), None)
        if target is None:
            raise ValueError("scene perception did not contain the requested target")
        affordance = task.affordances[target_object]
        retrieved = self.memory.retrieve(affordance, context_tag=context_tag)
        constraints = self.scene_reasoner.infer(
            target,
            scene,
            target_fragility=affordance.fragility,
        )
        planner_context = build_sorting_context(
            observation,
            task_instruction=task.instruction,
            episode_id=episode_id,
            timestep=timestep,
            trigger="subgoal_boundary",
            current_subgoal=f"sort {target_object} into {task.destinations[target_object]}",
            memory=tuple(card.card_id for card in retrieved),
        )
        return SortingSubgoalPlan(
            target_object=target_object,
            destination=task.destinations[target_object],
            retrieval_cards=retrieved,
            scene_constraints=constraints,
            planner_context={
                "task_instruction": planner_context.task_instruction,
                "current_subgoal": planner_context.current_subgoal,
                "memory": planner_context.memory,
                "available_skills": planner_context.available_skills,
                "camera_keys": tuple(planner_context.frames),
            },
        )

    def reflect(
        self,
        evidence: Mapping[str, float | bool | str],
        *,
        retries_used: int,
        replans_used: int,
    ) -> RecoveryDecision:
        return self.reflection.decide(evidence, retries_used=retries_used, replans_used=replans_used)

    def record_outcome(
        self,
        *,
        card: ExperienceCard,
    ) -> None:
        """Persist a structured success or failure card after an externally verified outcome."""
        self.memory.add(card)
