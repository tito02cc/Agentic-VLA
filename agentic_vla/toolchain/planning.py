"""Harness-owned state for bounded, symbolic robot task plans."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence

from agentic_vla.runtime.knowledge import ProceduralStep

from .contracts import VerificationReport, VerificationStatus


class PlanStepStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    RETRY_REQUIRED = "retry_required"


@dataclasses.dataclass(frozen=True)
class PlanStepReceipt:
    step: ProceduralStep
    status: PlanStepStatus
    attempts: int = 0
    verification: VerificationReport | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.step.to_dict(),
            "status": self.status.value,
            "attempts": self.attempts,
            "verification": (
                None if self.verification is None else self.verification.to_dict()
            ),
        }


class EmbodiedTaskPlan:
    """Evidence-gated task ledger owned by the Harness, not by the Planner."""

    def __init__(self, *, max_steps: int = 8) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self._receipts: list[PlanStepReceipt] = []

    def reset(self) -> None:
        self._receipts.clear()

    @property
    def installed(self) -> bool:
        return bool(self._receipts)

    @property
    def completed(self) -> bool:
        return self.installed and all(
            receipt.status is PlanStepStatus.CONFIRMED
            for receipt in self._receipts
        )

    @property
    def receipts(self) -> tuple[PlanStepReceipt, ...]:
        return tuple(self._receipts)

    @property
    def active(self) -> PlanStepReceipt | None:
        return next(
            (
                receipt
                for receipt in self._receipts
                if receipt.status
                in {PlanStepStatus.ACTIVE, PlanStepStatus.RETRY_REQUIRED}
            ),
            None,
        )

    def install(
        self,
        steps: Sequence[ProceduralStep],
        *,
        available_skills: Sequence[str] = (),
    ) -> None:
        if self.installed:
            raise RuntimeError("task plan is already installed")
        values = tuple(steps)
        if not values:
            raise ValueError("task plan must contain at least one step")
        if len(values) > self.max_steps:
            raise ValueError(f"task plan exceeds the {self.max_steps}-step limit")
        if any(not isinstance(step, ProceduralStep) for step in values):
            raise TypeError("task plan contains an unsupported step")
        stages = tuple(step.stage for step in values)
        if len(stages) != len(set(stages)):
            raise ValueError("task plan stage identifiers must be unique")
        allowed = set(available_skills)
        for step in values:
            if step.intent not in {"vla_act", "run_skill"}:
                raise ValueError("task plans may contain only physical primitives")
            if step.intent == "run_skill" and step.skill_id not in allowed:
                raise ValueError(f"task plan selects unavailable skill: {step.skill_id}")
        self._receipts = [
            PlanStepReceipt(
                step=step,
                status=(
                    PlanStepStatus.ACTIVE if index == 0 else PlanStepStatus.PENDING
                ),
            )
            for index, step in enumerate(values)
        ]

    def select(
        self,
        *,
        subgoal: str,
        intent: str,
        skill_id: str | None = None,
    ) -> PlanStepReceipt:
        """Select a currently admissible step without skipping pending work."""

        active = self.active
        if active is None:
            raise RuntimeError("task plan has no active step")
        normalized_skill = None if skill_id is None else skill_id.strip()
        if (
            active.step.subgoal != subgoal.strip()
            or active.step.intent != intent.strip().lower()
            or active.step.skill_id != normalized_skill
        ):
            raise ValueError("planner decision does not match the active task-plan step")
        index = self._receipts.index(active)
        selected = dataclasses.replace(active, attempts=active.attempts + 1)
        self._receipts[index] = selected
        return selected

    def apply_verification(self, report: VerificationReport) -> PlanStepReceipt:
        """Advance only on confirmed evidence; contradictions require a retry."""

        if not isinstance(report, VerificationReport):
            raise TypeError("task-plan transition requires VerificationReport")
        active = self.active
        if active is None:
            raise RuntimeError("task plan has no active step")
        index = self._receipts.index(active)
        if report.status is VerificationStatus.CONFIRMED:
            updated = dataclasses.replace(
                active,
                status=PlanStepStatus.CONFIRMED,
                verification=report,
            )
            self._receipts[index] = updated
            if index + 1 < len(self._receipts):
                following = self._receipts[index + 1]
                self._receipts[index + 1] = dataclasses.replace(
                    following, status=PlanStepStatus.ACTIVE
                )
            return updated
        status = (
            PlanStepStatus.RETRY_REQUIRED
            if report.status is VerificationStatus.CONTRADICTED
            else active.status
        )
        updated = dataclasses.replace(
            active,
            status=status,
            verification=report,
        )
        self._receipts[index] = updated
        return updated

    def to_dict(self) -> dict[str, object]:
        active = self.active
        return {
            "installed": self.installed,
            "completed": self.completed,
            "active_stage": None if active is None else active.step.stage,
            "steps": [receipt.to_dict() for receipt in self._receipts],
        }

    def reopen_confirmed(self, stage: str, report: VerificationReport) -> None:
        """Invalidate dependent stages when a trusted new observation contradicts one.

        This is a host/Critic operation, not a Planner-writable completion flag.
        Attempts remain in the ledger; previous receipts belong in the audit log.
        """
        if not isinstance(report, VerificationReport) or report.status is not VerificationStatus.CONTRADICTED:
            raise ValueError("reopening a stage requires contradictory verification")
        index = next((i for i, item in enumerate(self._receipts) if item.step.stage == stage), None)
        if index is None or self._receipts[index].status is not PlanStepStatus.CONFIRMED:
            raise ValueError("only an existing confirmed stage can be reopened")
        for i in range(index, len(self._receipts)):
            self._receipts[i] = dataclasses.replace(
                self._receipts[i],
                status=PlanStepStatus.RETRY_REQUIRED if i == index else PlanStepStatus.PENDING,
                verification=report if i == index else None,
            )
