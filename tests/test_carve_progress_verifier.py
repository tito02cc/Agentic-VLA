import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.toolchain import (
    GuardedProgressVerifier,
    VisualProgressContext,
    VisualStagePredicate,
    build_visual_progress_request,
)


def context():
    return VisualProgressContext(
        "assemble shelves",
        (
            VisualStagePredicate("base", "first supported shelf"),
            VisualStagePredicate("middle", "two supported shelves"),
            VisualStagePredicate("top", "two shelves with a block on top"),
        ),
        {"current_head": np.zeros((10, 10, 3), dtype=np.uint8)},
    )


def response(states=("present", "present", "unknown")):
    return {
        "stages": [
            {
                "stage_id": stage,
                "state": state,
                "confidence": 0.9,
                "evidence": "Board rests on support blocks",
            }
            for stage, state in zip(("base", "middle", "top"), states, strict=True)
        ]
    }


def test_request_has_no_time_plan_position_or_labels():
    request = build_visual_progress_request(context())
    payload = json.loads(request["user_prompt"])
    assert set(payload) == {"task_instruction", "predicates"}
    assert len(payload["predicates"]) == 3
    assert list(request["frames"]) == ["current_head"]
    assert "NOT a failure" in request["system_prompt"]


def test_one_call_can_confirm_multiple_stages_without_executing_them():
    calls = []

    def infer(request):
        calls.append(request)
        return json.dumps(response())

    result = GuardedProgressVerifier(infer).verify(context())
    assert result.accepted and len(calls) == 1
    assert result.confirmed_prefix == ("base", "middle")
    assert result.reports["top"].status.value == "inconclusive"


def test_complete_json_fence_preserves_raw_output_and_semantic_gates():
    calls = []
    raw = "```json\n" + json.dumps(response(("unknown", "present", "present"))) + "\n```"

    def infer(request):
        calls.append(request)
        return raw

    result = GuardedProgressVerifier(infer).verify(context())
    assert result.accepted and result.confirmed_prefix == ()
    assert result.raw_output == raw and len(calls) == 1


def test_fence_cannot_hide_action_fields():
    value = response()
    value["action"] = "retry"
    raw = "```json\n" + json.dumps(value) + "\n```"
    result = GuardedProgressVerifier(lambda _: raw).verify(context())
    assert not result.accepted and result.confirmed_prefix == ()
    assert result.raw_output == raw


@pytest.mark.parametrize(
    "states,prefix",
    [
        (("present", "present", "present"), ("base", "middle", "top")),
        (("unknown", "present", "present"), ()),
        (("present", "absent", "present"), ("base",)),
        (("absent", "absent", "absent"), ()),
    ],
)
def test_contiguous_evidence_required_and_absence_never_triggers_retry(states, prefix):
    result = GuardedProgressVerifier(lambda _: response(states)).verify(context())
    assert result.accepted and result.confirmed_prefix == prefix
    assert all(
        report.status.value != "contradicted" for report in result.reports.values()
    )


def test_canonical_stage_order_and_low_confidence_do_not_skip_gaps():
    raw = response(("present", "present", "present"))
    raw["stages"][1]["confidence"] = 0.4
    raw["stages"].reverse()
    result = GuardedProgressVerifier(lambda _: raw).verify(context())
    assert result.accepted and result.confirmed_prefix == ("base",)
    assert list(result.reports) == ["base", "middle", "top"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "success"),
        ("state", {}),
        ("confidence", "0.9"),
        ("confidence", True),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("evidence", ""),
        ("evidence", 1),
        ("evidence", "x" * 1025),
        ("evidence", "The board is not supported"),
        ("stage_id", "other"),
        ("stage_id", 2),
        ("stage_id", "middle"),
        ("actions", [1]),
        ("reward", 1),
    ],
)
def test_malformed_or_action_bearing_output_fails_closed(field, value):
    raw = response()
    raw["stages"][0][field] = value
    result = GuardedProgressVerifier(lambda _: raw).verify(context())
    assert not result.accepted and result.confirmed_prefix == ()
    assert result.error and result.raw_output == raw
    assert all(
        report.status.value == "inconclusive" for report in result.reports.values()
    )


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"stages": []},
        {"stages": {}},
        {"stages": [], "action": "retry"},
        "not json",
        {"stages": [None] * 3},
    ],
)
def test_incomplete_snapshots_fail_atomically(raw):
    result = GuardedProgressVerifier(lambda _: raw).verify(context())
    assert not result.accepted and result.confirmed_prefix == ()


def test_transport_failure_cannot_advance_progress():
    def infer(_):
        raise TimeoutError("timeout")

    result = GuardedProgressVerifier(infer).verify(context())
    assert not result.accepted and result.error == "timeout"


def test_context_is_bounded():
    ctx = context()
    for predicates in (
        (),
        ctx.predicates * 2,
        tuple(VisualStagePredicate(str(i), "visible") for i in range(9)),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(ctx, predicates=predicates)
    with pytest.raises(ValueError):
        dataclasses.replace(ctx, frames={})
    with pytest.raises(ValueError):
        dataclasses.replace(ctx, frames={str(i): object() for i in range(4)})
