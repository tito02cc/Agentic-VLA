"""Tests for the Harness-owned symbolic task-plan ledger."""

from __future__ import annotations

import pytest

from agentic_vla.runtime import ProceduralStep
from agentic_vla.toolchain import EmbodiedTaskPlan, VerificationReport


def step(stage: str, subgoal: str) -> ProceduralStep:
    return ProceduralStep(
        stage=stage,
        intent="vla_act",
        subgoal=subgoal,
        expected_outcome=f"{subgoal} is visibly complete",
    )


def report(status: str) -> VerificationReport:
    return VerificationReport(
        status=status,
        observed_outcome=f"visual check is {status}",
        confidence=0.9,
    )


def test_plan_advances_only_after_confirmed_verification() -> None:
    plan = EmbodiedTaskPlan()
    plan.install((step("first", "place first pot"), step("second", "place second pot")))

    selected = plan.select(subgoal="place first pot", intent="vla_act")
    plan.apply_verification(report("inconclusive"))

    assert selected.attempts == 1
    assert plan.active is not None
    assert plan.active.step.stage == "first"

    plan.apply_verification(report("confirmed"))

    assert plan.active is not None
    assert plan.active.step.stage == "second"
    assert not plan.completed

    plan.select(subgoal="place second pot", intent="vla_act")
    plan.apply_verification(report("confirmed"))
    assert plan.completed
    assert plan.active is None


def test_contradiction_requires_retry_without_skipping_stage() -> None:
    plan = EmbodiedTaskPlan()
    plan.install((step("first", "place first pot"), step("second", "place second pot")))
    plan.select(subgoal="place first pot", intent="vla_act")

    receipt = plan.apply_verification(report("contradicted"))

    assert receipt.status.value == "retry_required"
    assert plan.active is not None and plan.active.step.stage == "first"
    with pytest.raises(ValueError, match="does not match"):
        plan.select(subgoal="place second pot", intent="vla_act")


def test_plan_rejects_unregistered_skills_and_duplicate_stages() -> None:
    skill = ProceduralStep(
        stage="recover",
        intent="run_skill",
        subgoal="retract",
        expected_outcome="arm is clear",
        skill_id="retract",
    )
    with pytest.raises(ValueError, match="unavailable skill"):
        EmbodiedTaskPlan().install((skill,), available_skills=())

    with pytest.raises(ValueError, match="identifiers must be unique"):
        EmbodiedTaskPlan().install((step("same", "first"), step("same", "second")))
