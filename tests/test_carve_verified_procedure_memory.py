"""Tests for trusted promotion of robot traces into procedural memory."""

from __future__ import annotations

import pytest

from agentic_vla.runtime import ProceduralTaskMemory
from agentic_vla.toolchain import (
    PrimitiveOutcome,
    VerificationReport,
    VerifiedPrimitiveTrace,
    VerifiedProcedureCompiler,
)


def outcome(
    call_id: str,
    primitive_name: str,
    *,
    status: str = "succeeded",
    expected: str = "",
    observed: str = "",
    semantic: bool = False,
) -> PrimitiveOutcome:
    return PrimitiveOutcome(
        call_id=call_id,
        primitive_name=primitive_name,
        status=status,
        episode_id="task9:seed0",
        started_timestep=0,
        ended_timestep=10,
        expected_outcome=expected,
        observed_outcome=observed,
        requires_semantic_check=semantic,
    )


def confirmed(text: str = "task predicate confirmed") -> VerificationReport:
    return VerificationReport(
        status="confirmed",
        observed_outcome=text,
        confidence=0.95,
    )


def test_compiles_only_admissible_successes_and_records_memory() -> None:
    memory = ProceduralTaskMemory()
    compiler = VerifiedProcedureCompiler(memory)
    trace = (
        VerifiedPrimitiveTrace(
            outcome(
                "vla-1",
                "vla_act",
                expected="mug is inside microwave",
            ),
            subgoal="place mug inside microwave",
            constraints=("keep the door open",),
        ),
        VerifiedPrimitiveTrace(
            outcome("retry-1", "retract", status="failed", observed="still stalled"),
            subgoal="retract from failed contact",
            verification=VerificationReport(
                status="contradicted",
                observed_outcome="motion was not restored",
                confidence=0.9,
            ),
        ),
        VerifiedPrimitiveTrace(
            outcome(
                "close-1",
                "close_fixture",
                expected="microwave door is closed",
                semantic=True,
            ),
            subgoal="close microwave door",
            verification=confirmed("microwave door is visibly closed"),
        ),
    )

    record = compiler.compile_and_record(
        task_instruction="put the mug in the microwave and close it",
        task_family="microwave mug placement",
        object_categories=("mug", "microwave"),
        source_episode_id="task9:seed0",
        trace=trace,
        task_verification=confirmed(),
    )

    assert len(memory) == 1
    assert record.procedure_id.startswith("procedure-")
    assert [step.intent for step in record.steps] == ["vla_act", "run_skill"]
    assert record.steps[1].skill_id == "close_fixture"
    retrieved = memory.retrieve(
        task_instruction="place a mug in the microwave",
        limit=1,
    )
    assert retrieved[0]["procedure_id"] == record.procedure_id


def test_rejects_unverified_or_low_confidence_task_completion() -> None:
    compiler = VerifiedProcedureCompiler(ProceduralTaskMemory())
    trace = (
        VerifiedPrimitiveTrace(
            outcome("vla-1", "vla_act", expected="mug is grasped"),
            subgoal="grasp mug",
        ),
    )
    common = {
        "task_instruction": "grasp the mug",
        "task_family": "mug grasp",
        "object_categories": ("mug",),
        "source_episode_id": "task9:seed0",
        "trace": trace,
    }

    with pytest.raises(ValueError, match="confirmed task success"):
        compiler.compile(
            **common,
            task_verification=VerificationReport(
                status="inconclusive",
                observed_outcome="mug is occluded",
                confidence=0.8,
            ),
        )

    with pytest.raises(ValueError, match="below memory threshold"):
        compiler.compile(
            **common,
            task_verification=VerificationReport(
                status="confirmed",
                observed_outcome="mug appears grasped",
                confidence=0.4,
            ),
        )


def test_semantic_primitive_requires_confirmed_verification() -> None:
    compiler = VerifiedProcedureCompiler(ProceduralTaskMemory())
    with pytest.raises(ValueError, match="no admissible successful primitives"):
        compiler.compile(
            task_instruction="close the microwave",
            task_family="microwave closure",
            object_categories=("microwave",),
            source_episode_id="task9:seed0",
            trace=(
                VerifiedPrimitiveTrace(
                    outcome(
                        "vla-1",
                        "vla_act",
                        expected="microwave is closed",
                        semantic=True,
                    ),
                    subgoal="close microwave",
                ),
            ),
            task_verification=confirmed(),
        )


def test_compiler_collapses_repeated_chunks_of_one_symbolic_primitive() -> None:
    compiler = VerifiedProcedureCompiler(ProceduralTaskMemory())
    trace = (
        VerifiedPrimitiveTrace(
            outcome("chunk-1", "vla_act", expected="approach first moka pot"),
            subgoal="place both moka pots on the stove",
        ),
        VerifiedPrimitiveTrace(
            outcome("chunk-2", "vla_act", expected="continue dual placement"),
            subgoal="place both moka pots on the stove",
        ),
    )

    record = compiler.compile(
        task_instruction="put both moka pots on the stove",
        task_family="dual moka pot placement",
        object_categories=("moka_pot", "stove"),
        source_episode_id="task9:seed0",
        trace=trace,
        task_verification=confirmed(),
    )

    assert len(record.steps) == 1
    assert record.steps[0].expected_outcome == "approach first moka pot"


def test_procedural_memory_persists_across_process_boundaries(tmp_path) -> None:
    memory = ProceduralTaskMemory()
    compiler = VerifiedProcedureCompiler(memory)
    record = compiler.compile_and_record(
        task_instruction="put the mug in the microwave",
        task_family="microwave mug placement",
        object_categories=("mug", "microwave"),
        source_episode_id="task9:seed0",
        trace=(
            VerifiedPrimitiveTrace(
                outcome(
                    "vla-1",
                    "vla_act",
                    expected="mug is inside microwave",
                ),
                subgoal="place mug inside microwave",
            ),
        ),
        task_verification=confirmed(),
    )

    destination = memory.save(tmp_path / "procedural_memory.json")
    restored = ProceduralTaskMemory.load(destination)

    assert len(restored) == 1
    assert restored.records[0].procedure_id == record.procedure_id
    assert restored.records[0].steps[0].subgoal == "place mug inside microwave"
