"""Compile verified embodied traces into bounded procedural memory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence

from agentic_vla.runtime.knowledge import (
    ProceduralStep,
    ProceduralTaskMemory,
    ProceduralTaskRecord,
)

from .contracts import (
    PrimitiveOutcome,
    PrimitiveStatus,
    VerificationReport,
    VerificationStatus,
    reject_direct_action_fields,
)


@dataclasses.dataclass(frozen=True)
class VerifiedPrimitiveTrace:
    """Planner-visible description of one completed physical primitive."""

    outcome: PrimitiveOutcome
    subgoal: str = ""
    constraints: tuple[str, ...] = ()
    verification: VerificationReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, PrimitiveOutcome):
            raise TypeError("trace outcome must be a PrimitiveOutcome")
        if self.verification is not None and not isinstance(
            self.verification, VerificationReport
        ):
            raise TypeError("trace verification must be a VerificationReport or null")
        constraints = tuple(str(value).strip() for value in self.constraints)
        if any(not value for value in constraints):
            raise ValueError("trace constraints must not contain empty values")
        reject_direct_action_fields(
            self.outcome.to_planner_dict(), path="verified_trace.outcome"
        )
        object.__setattr__(self, "subgoal", self.subgoal.strip())
        object.__setattr__(self, "constraints", constraints)

    @property
    def admissible_success(self) -> bool:
        if self.outcome.status is not PrimitiveStatus.SUCCEEDED:
            return False
        if self.verification is not None:
            return self.verification.status is VerificationStatus.CONFIRMED
        return not self.outcome.requires_semantic_check


class VerifiedProcedureCompiler:
    """Promote successful symbolic traces only after trusted task verification.

    The compiler is intentionally outside Planner authority. A benchmark
    evaluator, robot-side verifier, or human may supply the task-level report;
    free-form model text cannot directly write persistent procedure memory.
    """

    def __init__(
        self,
        memory: ProceduralTaskMemory,
        *,
        minimum_task_confidence: float = 0.55,
    ) -> None:
        if not isinstance(memory, ProceduralTaskMemory):
            raise TypeError("procedure compiler requires ProceduralTaskMemory")
        if not 0.0 <= float(minimum_task_confidence) <= 1.0:
            raise ValueError("minimum task confidence must be in [0, 1]")
        self.memory = memory
        self.minimum_task_confidence = float(minimum_task_confidence)

    def compile(
        self,
        *,
        task_instruction: str,
        task_family: str,
        object_categories: Sequence[str],
        source_episode_id: str | int,
        trace: Sequence[VerifiedPrimitiveTrace],
        task_verification: VerificationReport,
        notes: Sequence[str] = (),
        verified_steps: Sequence[ProceduralStep] = (),
    ) -> ProceduralTaskRecord:
        if not task_instruction.strip() or not task_family.strip():
            raise ValueError("task instruction and family must not be empty")
        if not isinstance(task_verification, VerificationReport):
            raise TypeError("task verification must be a VerificationReport")
        if task_verification.status is not VerificationStatus.CONFIRMED:
            raise ValueError("persistent procedure memory requires confirmed task success")
        if task_verification.confidence < self.minimum_task_confidence:
            raise ValueError("task verification confidence is below memory threshold")

        trusted_steps = tuple(verified_steps)
        if trusted_steps:
            if any(not isinstance(step, ProceduralStep) for step in trusted_steps):
                raise TypeError("verified procedure contains an unsupported step")
            if any(step.intent not in {"vla_act", "run_skill"} for step in trusted_steps):
                raise ValueError("verified procedure contains a non-physical step")
            steps = list(trusted_steps)
        else:
            entries = tuple(trace)
            if not entries:
                raise ValueError("procedure trace must not be empty")
            if any(not isinstance(entry, VerifiedPrimitiveTrace) for entry in entries):
                raise TypeError("procedure trace contains an unsupported entry")
            if any(entry.outcome.episode_id != source_episode_id for entry in entries):
                raise ValueError("procedure trace mixes episode identities")

            steps = []
            for entry in entries:
                if not entry.admissible_success:
                    continue
                outcome = entry.outcome
                subgoal = (
                    entry.subgoal
                    or outcome.expected_outcome.strip()
                    or outcome.observed_outcome.strip()
                )
                expected = outcome.expected_outcome.strip()
                if not expected and entry.verification is not None:
                    expected = entry.verification.observed_outcome.strip()
                expected = expected or outcome.observed_outcome.strip() or subgoal
                if not subgoal or not expected:
                    continue
                is_vla = outcome.primitive_name == "vla_act"
                candidate = ProceduralStep(
                    stage=f"step_{len(steps) + 1:02d}",
                    intent="vla_act" if is_vla else "run_skill",
                    subgoal=subgoal,
                    expected_outcome=expected,
                    skill_id=None if is_vla else outcome.primitive_name,
                    constraints=entry.constraints,
                )
                if steps:
                    previous = steps[-1]
                    same_symbolic_primitive = (
                        previous.intent == candidate.intent
                        and previous.subgoal == candidate.subgoal
                        and previous.skill_id == candidate.skill_id
                        and previous.constraints == candidate.constraints
                    )
                    if same_symbolic_primitive:
                        continue
                steps.append(candidate)

        if not steps:
            raise ValueError("verified episode contains no admissible successful primitives")

        digest_payload = {
            "task_instruction": task_instruction.strip().lower(),
            "task_family": task_family.strip().lower(),
            "objects": sorted(str(value).strip().lower() for value in object_categories),
            "steps": [step.to_dict() for step in steps],
        }
        procedure_id = "procedure-" + hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return ProceduralTaskRecord(
            procedure_id=procedure_id,
            task_family=task_family,
            object_categories=tuple(object_categories),
            steps=tuple(steps),
            verification_result="verified",
            source_episode_id=source_episode_id,
            confidence=task_verification.confidence,
            notes=tuple(notes),
        )

    def compile_and_record(
        self,
        *,
        task_instruction: str,
        task_family: str,
        object_categories: Sequence[str],
        source_episode_id: str | int,
        trace: Sequence[VerifiedPrimitiveTrace],
        task_verification: VerificationReport,
        notes: Sequence[str] = (),
        verified_steps: Sequence[ProceduralStep] = (),
    ) -> ProceduralTaskRecord:
        record = self.compile(
            task_instruction=task_instruction,
            task_family=task_family,
            object_categories=object_categories,
            source_episode_id=source_episode_id,
            trace=trace,
            task_verification=task_verification,
            notes=notes,
            verified_steps=verified_steps,
        )
        self.memory.record(record)
        return record
