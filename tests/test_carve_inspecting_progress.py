import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.runtime.agent import PolicyServiceVisionPlanner
from agentic_vla.toolchain import (
    RunManifest,
    RunWorkspace,
    ToolExecutionContext,
    VisualProgressContext,
    VisualStagePredicate,
)
from agentic_vla.toolchain.inspection_verifier import InspectingProgressVerifier


def inputs(tmp_path):
    image = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    context = VisualProgressContext(
        "assemble a shelf",
        (VisualStagePredicate("base", "board supported by two blocks"),),
        {"head": image},
    )
    execution = ToolExecutionContext("episode-1", 32, True, ("inspect_region",))
    workspace = RunWorkspace(
        tmp_path, RunManifest("inspection", "test", "test", 0, "spy", "none", "shadow")
    )
    return context, execution, workspace


def region(request):
    frame = json.loads(request["user_prompt"])["frame"]
    return {
        "inspect": {
            "frame_id": frame["frame_id"],
            "left": 8,
            "top": 16,
            "right": 40,
            "bottom": 48,
        }
    }


def confirmation():
    return {
        "stages": [
            {
                "stage_id": "base",
                "state": "present",
                "confidence": 0.9,
                "evidence": "The board rests on two blocks",
            }
        ]
    }


def assert_inconclusive(result):
    assert not result.accepted and result.error and result.confirmed_prefix == ()
    assert all(
        report.status.value == "inconclusive" for report in result.reports.values()
    )


def test_crop_pixels_reach_actual_policy_service_adapter_and_receipts(tmp_path):
    context, execution, workspace = inputs(tmp_path)
    original = context.frames["head"].copy()
    calls = []

    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            assert not request["frames"]["head"].flags.writeable
            selected = region(request)
            context.frames["head"][:] = 0
            return {"ok": True, "data": {"text": json.dumps(selected)}}
        return {"ok": True, "data": {"text": json.dumps(confirmation())}}

    infer = PolicyServiceVisionPlanner(transport, max_images=2, max_tokens=256)
    result = InspectingProgressVerifier(infer, workspace).verify(
        context,
        execution_context=execution,
        current_context=lambda: execution,
    )
    assert result.accepted and result.confirmed_prefix == ("base",) and len(calls) == 2
    assert list(calls[0]["frames"]) == ["head"]
    assert list(calls[1]["frames"]) == ["global", "region"]
    np.testing.assert_array_equal(calls[1]["frames"]["global"], original)
    np.testing.assert_array_equal(calls[1]["frames"]["region"], original[16:48, 8:40])
    payload = json.loads(calls[1]["user_prompt"])
    assert set(payload) == {"task_instruction", "predicates", "inspection_receipt"}
    assert (
        payload["inspection_receipt"]["relationship"]
        == "crop_of_same_frame_not_an_independent_view"
    )
    recipes = [
        json.loads(row) for row in workspace.recipe_path.read_text().splitlines()
    ]
    assert len(recipes) == 1 and recipes[0]["tool"] == "inspect_region"
    assert recipes[0]["accepted"]
    events = [json.loads(row) for row in workspace.event_path.read_text().splitlines()]
    assert events[-1]["payload"]["control_applied"] is False
    assert [
        row["payload"]["phase"]
        for row in events
        if row["event_type"] == "inspection_model_call"
    ] == [
        "region_selection",
        "relation_verification",
    ]


def test_no_crop_still_verifies_full_frame_without_tool_call(tmp_path):
    context, execution, workspace = inputs(tmp_path)
    calls = []

    def infer(request):
        calls.append(request)
        return {"inspect": None} if len(calls) == 1 else confirmation()

    result = InspectingProgressVerifier(infer, workspace).verify(
        context,
        execution_context=execution,
        current_context=lambda: execution,
    )
    assert result.accepted and len(calls) == 2
    assert list(calls[1]["frames"]) == ["head"]
    assert json.loads(calls[1]["user_prompt"])["inspection_receipt"] is None
    assert not workspace.recipe_path.exists()


@pytest.mark.parametrize("outside", [False, True])
def test_json_fences_do_not_bypass_region_or_progress_validation(tmp_path, outside):
    context, execution, workspace = inputs(tmp_path)
    calls = []

    def infer(request):
        calls.append(request)
        selected = region(request) if len(calls) == 1 else confirmation()
        if outside and len(calls) == 1:
            selected["inspect"]["right"] = 1000
        return "```json\n" + json.dumps(selected) + "\n```"

    result = InspectingProgressVerifier(infer, workspace).verify(
        context, execution_context=execution, current_context=lambda: execution
    )
    if outside:
        assert_inconclusive(result)
        assert len(calls) == 1
    else:
        assert result.accepted and result.confirmed_prefix == ("base",)
        assert len(calls) == 2


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "json",
        "oversized",
        "action",
        "unknown_frame",
        "outside",
        "small",
        "float",
        "bool",
        "path",
        "missing",
        "extra",
        "list",
    ],
)
def test_invalid_selection_never_reaches_semantic_verification(tmp_path, kind):
    context, execution, workspace = inputs(tmp_path)
    calls = []

    def infer(request):
        calls.append(request)
        selection = region(request)
        if kind == "empty":
            return {}
        if kind == "json":
            return "not json"
        if kind == "oversized":
            return " " * 16385
        if kind == "action":
            selection["action"] = "retry"
        if kind == "unknown_frame":
            selection["inspect"]["frame_id"] = "old-episode-frame"
        if kind == "outside":
            selection["inspect"]["right"] = 100
        if kind == "small":
            selection["inspect"]["right"] = 9
        if kind == "float":
            selection["inspect"]["left"] = 8.0
        if kind == "bool":
            selection["inspect"]["left"] = True
        if kind == "path":
            selection["inspect"]["frame_id"] = "/tmp/image.png"
        if kind == "missing":
            del selection["inspect"]["left"]
        if kind == "extra":
            selection["inspect"]["task_success"] = True
        if kind == "list":
            selection["inspect"] = [selection["inspect"]]
        return selection

    result = InspectingProgressVerifier(infer, workspace).verify(
        context,
        execution_context=execution,
        current_context=lambda: execution,
    )
    assert_inconclusive(result)
    assert len(calls) == 1


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("change", ["episode", "step", "error"])
def test_stale_response_is_rejected_before_it_can_confirm_progress(
    tmp_path, phase, change
):
    context, execution, workspace = inputs(tmp_path)
    calls = []

    def current():
        if len(calls) < phase:
            return execution
        if change == "episode":
            return dataclasses.replace(execution, episode_id="episode-2")
        if change == "step":
            return dataclasses.replace(execution, timestep=33)
        raise RuntimeError("host context invalidated")

    def infer(request):
        calls.append(request)
        return region(request) if len(calls) == 1 else confirmation()

    result = InspectingProgressVerifier(infer, workspace).verify(
        context,
        execution_context=execution,
        current_context=current,
    )
    assert_inconclusive(result)
    assert len(calls) == phase


@pytest.mark.parametrize("kind", ["permission", "two_cameras", "invalid_pixels"])
def test_invalid_context_makes_no_model_calls(tmp_path, kind):
    context, execution, workspace = inputs(tmp_path)
    if kind == "permission":
        execution = dataclasses.replace(execution, allowed_tools=())
    if kind == "two_cameras":
        context = dataclasses.replace(
            context, frames={**context.frames, "wrist": context.frames["head"]}
        )
    if kind == "invalid_pixels":
        context = dataclasses.replace(context, frames={"head": np.zeros((64, 96, 3))})
    calls = []
    result = InspectingProgressVerifier(
        lambda req: calls.append(req), workspace
    ).verify(
        context,
        execution_context=execution,
        current_context=lambda: execution,
    )
    assert_inconclusive(result)
    assert calls == []


@pytest.mark.parametrize("phase", [1, 2])
def test_provider_failure_is_inconclusive_without_retry(tmp_path, phase):
    context, execution, workspace = inputs(tmp_path)
    calls = []

    def infer(request):
        calls.append(request)
        if len(calls) == phase:
            raise TimeoutError("deadline exceeded")
        return region(request)

    result = InspectingProgressVerifier(infer, workspace).verify(
        context,
        execution_context=execution,
        current_context=lambda: execution,
    )
    assert_inconclusive(result)
    assert len(calls) == phase


@pytest.mark.parametrize(
    "required",
    [["global", "region"], ["missing"], "global", ["global", "global"], [1], None],
)
def test_policy_service_cannot_silently_drop_required_images(required):
    calls = []
    infer = PolicyServiceVisionPlanner(lambda req: calls.append(req), max_images=1)
    with pytest.raises(ValueError, match="missing or truncated"):
        infer(
            {
                "system_prompt": "system",
                "user_prompt": "user",
                "frames": {"global": 1, "region": 2},
                "required_frame_names": required,
            }
        )
    assert not calls


def test_policy_service_legacy_optional_frames_still_respect_capacity():
    calls = []

    def transport(request):
        calls.append(request)
        return {"ok": True, "data": {"text": "{}"}}

    assert (
        PolicyServiceVisionPlanner(transport)(
            {
                "system_prompt": "s",
                "user_prompt": "u",
                "frames": {"a": 1, "b": 2},
            }
        )
        == "{}"
    )
    assert calls[0]["frames"] == {"a": 1}
