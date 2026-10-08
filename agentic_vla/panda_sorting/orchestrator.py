"""Event-boundary semantic authorization and execution memory for Panda tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agentic_vla.runtime import AgentIntent, GuardedHighLevelAgent, HighLevelAgentResult

from .environment import PandaSortingObservation
from .vlm import build_sorting_context


@dataclass(frozen=True)
class ExecutionMemoryEntry:
    """One action-free memory item available to a future semantic decision."""

    subgoal: str
    event: str
    outcome: str
    attempt: int
    evidence: Mapping[str, float | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subgoal.strip() or not self.event.strip() or not self.outcome.strip():
            raise ValueError("execution-memory text fields must not be empty")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")

    def to_prompt_record(self) -> dict[str, object]:
        return {
            "subgoal": self.subgoal,
            "event": self.event,
            "outcome": self.outcome,
            "attempt": self.attempt,
            "evidence": dict(self.evidence),
        }


class ExecutionMemory:
    """Bounded subgoal/outcome ledger without trajectories or simulator truth."""

    def __init__(self, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = int(max_entries)
        self._entries: list[ExecutionMemoryEntry] = []

    def append(self, entry: ExecutionMemoryEntry) -> None:
        self._entries.append(entry)
        del self._entries[:-self._max_entries]

    def prompt_records(self, limit: int = 6) -> tuple[Mapping[str, object], ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return tuple(entry.to_prompt_record() for entry in self._entries[-limit:])

    def to_dict(self) -> list[dict[str, object]]:
        return [entry.to_prompt_record() for entry in self._entries]


@dataclass(frozen=True)
class SemanticRecoveryAuthorization:
    """A VLM result narrowed to one event-compatible registered skill."""

    event: str
    expected_skill: str
    authorized: bool
    result: HighLevelAgentResult
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event,
            "expected_skill": self.expected_skill,
            "authorized": self.authorized,
            "rejection_reason": self.rejection_reason,
            "planner": {
                "accepted": self.result.accepted,
                "elapsed_ms": self.result.elapsed_ms,
                "error": self.result.error,
                "decision": self.result.decision.to_dict(),
            },
        }


class EventTriggeredSemanticRouter:
    """Allow a VLM to authorize only the recovery skill valid for an event.

    The router is deliberately not a trajectory planner. It sends the VLM a
    fresh visual context only at a safe boundary, constrains its capability set
    to the event's registered skill, and returns an auditable authorization.
    """

    _EVENT_SKILLS = {
        "grasp_missed": "retract_and_regrasp",
        "obstacle_conflict": "wide_reapproach",
        "target_displaced": "reobserve_scene",
        "placement_invalid": "verify_destination",
    }

    def __init__(self, planner: GuardedHighLevelAgent) -> None:
        self.planner = planner

    def authorize(
        self,
        *,
        event: str,
        observation: PandaSortingObservation,
        task_instruction: str,
        episode_id: str | int,
        timestep: int,
        current_subgoal: str,
        memory: ExecutionMemory,
        risk: Mapping[str, Any],
        remaining_retries: int,
        remaining_recoveries: int,
    ) -> SemanticRecoveryAuthorization:
        expected_skill = self._EVENT_SKILLS.get(event)
        if expected_skill is None:
            raise ValueError(f"event {event!r} has no registered semantic recovery skill")
        context = build_sorting_context(
            observation,
            task_instruction=task_instruction,
            episode_id=episode_id,
            timestep=timestep,
            trigger=event,
            current_subgoal=current_subgoal,
            memory=tuple(entry["event"] for entry in memory.prompt_records()),
            memory_records=memory.prompt_records(),
            risk=risk,
            remaining_retries=remaining_retries,
            remaining_recoveries=remaining_recoveries,
            available_skills=(expected_skill,),
            allowed_intents=(AgentIntent.RUN_SKILL, AgentIntent.SAFE_STOP),
        )
        result = self.planner.decide(context)
        authorized = (
            result.accepted
            and result.decision.intent is AgentIntent.RUN_SKILL
            and result.decision.skill_id == expected_skill
        )
        reason = None if authorized else "planner did not authorize the event-compatible recovery skill"
        return SemanticRecoveryAuthorization(event, expected_skill, authorized, result, reason)
