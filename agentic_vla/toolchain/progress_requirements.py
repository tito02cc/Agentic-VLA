"""Action-free progress from individually verified, conjunctive visual facts."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from typing import Any

from agentic_vla.runtime.agent import PlannerCallable

from .contracts import VerificationReport, VerificationStatus
from .progress import (
    GuardedProgressVerifier,
    VisualProgressContext,
    VisualProgressResult,
    VisualStagePredicate,
)


@dataclasses.dataclass(frozen=True)
class VisualStageRequirements:
    stage_id: str
    required_fact_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("stage id must be a nonempty string")
        if not isinstance(self.required_fact_ids, tuple):
            raise TypeError("required fact ids must be an immutable tuple")
        if not 1 <= len(self.required_fact_ids) <= 8:
            raise ValueError("stage requires one to eight facts")
        if any(not isinstance(fact, str) or not fact.strip() for fact in self.required_fact_ids):
            raise ValueError("fact ids must be nonempty strings")
        if len(set(self.required_fact_ids)) != len(self.required_fact_ids):
            raise ValueError("stage has duplicate required facts")


@dataclasses.dataclass(frozen=True)
class ConjunctiveProgressContext:
    task_instruction: str
    facts: tuple[VisualStagePredicate, ...]
    stages: tuple[VisualStageRequirements, ...]
    frames: Mapping[str, Any]

    def __post_init__(self):
        VisualProgressContext(self.task_instruction, self.facts, self.frames)
        if not isinstance(self.facts, tuple) or not isinstance(self.stages, tuple):
            raise TypeError("facts and stage requirements must be immutable tuples")
        if not 1 <= len(self.stages) <= 8:
            raise ValueError("progress requires one to eight stages")
        if any(not isinstance(stage, VisualStageRequirements) for stage in self.stages):
            raise TypeError("invalid stage requirements")
        ids = [stage.stage_id for stage in self.stages]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate stage id")
        available = {fact.stage_id for fact in self.facts}
        previous = set()
        for stage in self.stages:
            required = set(stage.required_fact_ids)
            if not required <= available:
                raise ValueError("stage references an unknown fact")
            if not previous <= required:
                raise ValueError("ordered stage requirements must be cumulative")
            previous = required
        if previous != available:
            raise ValueError("every requested fact must be used by a stage")


@dataclasses.dataclass(frozen=True)
class ConjunctiveProgressResult(VisualProgressResult):
    fact_reports: Mapping[str, VerificationReport] = dataclasses.field(default_factory=dict)


class GuardedConjunctiveProgressVerifier:
    """One model call; the host, not the model, combines required facts.

    This does not establish whether a visual fact is actually true. It prevents
    partial satisfaction from being promoted by bypassing a required fact.
    No robot, ledger or memory-writing bindings are available here.
    """

    def __init__(self, infer: PlannerCallable, *, minimum_confidence: float = 0.55):
        GuardedProgressVerifier(infer, minimum_confidence=minimum_confidence)
        self.infer = infer
        self.minimum_confidence = minimum_confidence

    def verify(self, context: ConjunctiveProgressContext) -> ConjunctiveProgressResult:
        started = time.perf_counter()
        def infer_facts(request):
            return self.infer({
                **request,
                "system_prompt": request["system_prompt"] + (
                    " Each stage_id in this request names ONE observable fact, not a "
                    "whole task stage. Judge each fact separately. The presence of one "
                    "piece cannot substitute for a different required piece. The host "
                    "will combine facts; do not output overall completion or actions."
                ),
            })

        fact_result = GuardedProgressVerifier(
            infer_facts, minimum_confidence=self.minimum_confidence
        ).verify(VisualProgressContext(context.task_instruction, context.facts, context.frames))
        reports = {}
        for stage in context.stages:
            facts = [fact_result.reports[fact] for fact in stage.required_fact_ids]
            confirmed = fact_result.accepted and all(
                fact.status is VerificationStatus.CONFIRMED for fact in facts
            )
            unconfirmed = [
                key for key, fact in zip(stage.required_fact_ids, facts, strict=True)
                if fact.status is not VerificationStatus.CONFIRMED
            ]
            reports[stage.stage_id] = VerificationReport(
                status=VerificationStatus.CONFIRMED if confirmed else VerificationStatus.INCONCLUSIVE,
                observed_outcome=(
                    "All required facts confirmed: " + ", ".join(stage.required_fact_ids)
                    if confirmed else "Required facts not confirmed: " + ", ".join(unconfirmed)
                ),
                confidence=min(fact.confidence for fact in facts),
                metadata={
                    "aggregation": "host_all_required_facts",
                    "required_fact_ids": list(stage.required_fact_ids),
                    "unconfirmed_fact_ids": unconfirmed,
                },
            )
        prefix = []
        for stage in context.stages:
            if reports[stage.stage_id].status is not VerificationStatus.CONFIRMED:
                break
            prefix.append(stage.stage_id)
        return ConjunctiveProgressResult(
            reports=reports,
            confirmed_prefix=tuple(prefix),
            accepted=fact_result.accepted,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=fact_result.error,
            raw_output=fact_result.raw_output,
            fact_reports=fact_result.reports,
        )
