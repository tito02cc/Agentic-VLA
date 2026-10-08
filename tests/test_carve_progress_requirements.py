import dataclasses
import itertools
import json

import numpy as np
import pytest

from agentic_vla.runtime.agent import PolicyServiceVisionPlanner
from agentic_vla.toolchain import (
    ConjunctiveProgressContext,
    GuardedConjunctiveProgressVerifier,
    VisualStagePredicate,
    VisualStageRequirements,
)


def context():
    return ConjunctiveProgressContext(
        "assemble a structure",
        tuple(VisualStagePredicate(fact, fact + " visibly placed") for fact in ("support", "block", "roof")),
        (
            VisualStageRequirements("base", ("support",)),
            VisualStageRequirements("top", ("support", "block", "roof")),
        ),
        {"head": np.zeros((32, 32, 3), dtype=np.uint8)},
    )


def response(states=("present", "present", "present")):
    return {"stages": [
        {"stage_id": fact, "state": state, "confidence": 0.9, "evidence": "Object rests on support"}
        for fact, state in zip(("support", "block", "roof"), states, strict=True)
    ]}


@pytest.mark.parametrize("states", list(itertools.product(("present", "absent", "unknown"), repeat=3)))
def test_partial_or_unknown_facts_never_confirm_whole_stage(states):
    calls = []
    raw = response(states)

    def infer(request):
        calls.append(request)
        return raw

    result = GuardedConjunctiveProgressVerifier(infer).verify(context())
    assert result.accepted and len(calls) == 1
    expected = ("base",) if states[0] == "present" else ()
    if all(state == "present" for state in states):
        expected += ("top",)
    assert result.confirmed_prefix == expected
    assert result.raw_output == raw
    assert set(result.fact_reports) == {"support", "block", "roof"}
    assert all(report.status.value != "contradicted" for report in result.reports.values())
    payload = json.loads(calls[0]["user_prompt"])
    assert set(payload) == {"task_instruction", "predicates"}
    assert "ONE observable fact" in calls[0]["system_prompt"]


def test_low_confidence_fact_is_not_inherited_from_other_facts():
    raw = response()
    raw["stages"][-1]["confidence"] = 0.4
    raw["stages"].reverse()
    result = GuardedConjunctiveProgressVerifier(lambda _: raw).verify(context())
    assert result.accepted and result.confirmed_prefix == ("base",)
    assert result.reports["top"].metadata["unconfirmed_fact_ids"] == ["roof"]
    assert result.reports["top"].confidence == 0.4


@pytest.mark.parametrize("kind", ["missing", "duplicate", "action", "summary_only", "contradictory", "nan"])
def test_bad_model_protocol_fails_atomically(kind):
    raw = response()
    if kind == "missing":
        raw["stages"].pop()
    elif kind == "duplicate":
        raw["stages"][-1] = raw["stages"][0]
    elif kind == "action":
        raw["action"] = "retry"
    elif kind == "summary_only":
        raw = {"stages": [{"stage_id": "top", "state": "present", "confidence": 1, "evidence": "done"}]}
    elif kind == "contradictory":
        raw["stages"][-1]["evidence"] = "The roof is not supported"
    elif kind == "nan":
        raw["stages"][-1]["confidence"] = float("nan")
    result = GuardedConjunctiveProgressVerifier(lambda _: raw).verify(context())
    assert not result.accepted and result.confirmed_prefix == ()
    assert all(r.status.value == "inconclusive" for r in result.reports.values())


def test_transport_failure_and_fence_compatibility():
    def fail(_):
        raise TimeoutError("deadline")

    result = GuardedConjunctiveProgressVerifier(fail).verify(context())
    assert not result.accepted and result.error == "deadline"
    result = GuardedConjunctiveProgressVerifier(
        lambda _: "```json\n" + json.dumps(response()) + "\n```"
    ).verify(context())
    assert result.accepted and result.confirmed_prefix == ("base", "top")


@pytest.mark.parametrize(
    "stages",
    [
        (),
        (VisualStageRequirements("base", ("not_declared",)),),
        (VisualStageRequirements("base", ("support",)),),
        (VisualStageRequirements("base", ("support",)), VisualStageRequirements("top", ("roof", "block"))),
        (VisualStageRequirements("base", ("support",)), VisualStageRequirements("base", ("support", "block", "roof"))),
    ],
)
def test_requirement_graph_rejects_gaps_unknowns_and_unused_facts(stages):
    with pytest.raises(ValueError):
        dataclasses.replace(context(), stages=stages)


@pytest.mark.parametrize("ids", [(), ("roof", "roof"), ("",), tuple(str(i) for i in range(9))])
def test_stage_requirements_bounded_and_unique(ids):
    with pytest.raises(ValueError):
        VisualStageRequirements("top", ids)


def test_requirements_cannot_be_mutated_through_lists():
    with pytest.raises(TypeError):
        VisualStageRequirements("top", ["roof"])


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_multiview_requirements_cannot_silently_drop_camera(capacity):
    calls = []
    frames = {name: np.full((16, 24, 3), i, dtype=np.uint8)
              for i, name in enumerate(("head", "left_wrist", "right_wrist"))}

    def transport(request):
        calls.append(request)
        return {"ok": True, "data": {"text": json.dumps(response())}}

    infer = PolicyServiceVisionPlanner(transport, max_images=capacity, max_tokens=768)
    result = GuardedConjunctiveProgressVerifier(infer).verify(dataclasses.replace(context(), frames=frames))
    assert result.accepted == (capacity == 3)
    assert len(calls) == int(capacity == 3)
    if calls:
        assert list(calls[0]["frames"]) == list(frames)
        for name, frame in frames.items():
            assert calls[0]["frames"][name] is frame
    else:
        assert "missing or truncated" in result.error
