"""Capability-gated VLM selection over valid dynamic-kitting subgoals."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_vla.runtime import AgentIntent, GuardedHighLevelAgent, HighLevelAgentResult

from .environment import PandaSortingObservation
from .kitting import (
    DependencyAwareKittingPlanner,
    KittingSceneState,
    KittingSkill,
    KittingSubgoal,
    KittingTask,
)
from .orchestrator import ExecutionMemory
from .vlm import build_sorting_context


@dataclass(frozen=True)
class KittingSemanticDecision:
    selected: KittingSubgoal | None
    candidates: tuple[KittingSubgoal, ...]
    result: HighLevelAgentResult
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": None
            if self.selected is None
            else {
                "skill": self.selected.skill.value,
                "object": self.selected.object_name,
                "destination": self.selected.destination,
                "reason": self.selected.reason,
            },
            "candidates": [
                {
                    "id": _candidate_id(candidate),
                    "skill": candidate.skill.value,
                    "object": candidate.object_name,
                    "destination": candidate.destination,
                    "reason": candidate.reason,
                }
                for candidate in self.candidates
            ],
            "planner": {
                "accepted": self.result.accepted,
                "elapsed_ms": self.result.elapsed_ms,
                "error": self.result.error,
                "decision": self.result.decision.to_dict(),
            },
            "rejection_reason": self.rejection_reason,
        }


class CapabilityGatedKittingVlm:
    """Let a VLM choose only among scene-precondition-valid semantic actions."""

    def __init__(
        self,
        planner: GuardedHighLevelAgent,
        *,
        dependency_planner: DependencyAwareKittingPlanner | None = None,
    ) -> None:
        self.planner = planner
        self.dependency_planner = dependency_planner or DependencyAwareKittingPlanner()

    def select(
        self,
        task: KittingTask,
        state: KittingSceneState,
        observation: PandaSortingObservation,
        *,
        episode_id: str | int,
        timestep: int,
        memory: ExecutionMemory,
    ) -> KittingSemanticDecision:
        candidates = self.dependency_planner.candidate_subgoals(task, state)
        candidate_map = {_candidate_id(item): item for item in candidates}
        candidate_description = "; ".join(
            f"{identifier}: {item.reason}"
            for identifier, item in candidate_map.items()
        )
        context = build_sorting_context(
            observation,
            task_instruction=(
                f"{task.instruction} Select exactly one registered candidate. "
                f"Candidates: {candidate_description}"
            ),
            episode_id=episode_id,
            timestep=timestep,
            trigger="subgoal_boundary",
            current_subgoal="select the next valid kitting subgoal",
            memory=tuple(entry["event"] for entry in memory.prompt_records()),
            memory_records=memory.prompt_records(),
            risk={"event": None, "bucket": "low", "score": 0.0},
            remaining_retries=1,
            remaining_recoveries=1,
            available_skills=tuple(candidate_map),
            allowed_intents=(AgentIntent.RUN_SKILL, AgentIntent.SAFE_STOP),
        )
        result = self.planner.decide(context)
        selected = (
            candidate_map.get(result.decision.skill_id or "")
            if result.accepted and result.decision.intent is AgentIntent.RUN_SKILL
            else None
        )
        reason = None if selected is not None else "VLM did not select a valid candidate skill"
        return KittingSemanticDecision(selected, candidates, result, reason)


def _candidate_id(subgoal: KittingSubgoal) -> str:
    if subgoal.skill is KittingSkill.PLACE_TARGET:
        return f"place_{subgoal.object_name}_in_{subgoal.destination}"
    if subgoal.skill is KittingSkill.CLEAR_TO_STAGING:
        return f"clear_{subgoal.object_name}_to_staging"
    return subgoal.skill.value
