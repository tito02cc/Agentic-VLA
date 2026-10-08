import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.runtime.agent import PolicyServiceVisionPlanner
from agentic_vla.toolchain import (
    CanonicalToolRuntime,
    EmbodiedToolBindings,
    GuardedObjectObserver,
    ObjectEvidenceMemory,
    ObjectRole,
    RunManifest,
    RunWorkspace,
    ToolExecutionContext,
)

ROLES = (ObjectRole("base", "longer board"), ObjectRole("upper", "shorter board"))


def ctx(step=0, episode="e1"):
    return ToolExecutionContext(episode, step, False, ("read_object_evidence",))


def answer():
    return {"objects": [
        {"role_id": role.role_id, "state": "located", "box": [10, 10 + i * 20, 60, 25 + i * 20],
         "confidence": 0.9, "evidence": "Horizontal board visible"}
        for i, role in enumerate(ROLES)
    ]}


def setup_memory():
    memory = ObjectEvidenceMemory(history_limit=2)
    memory.reset("e1")
    image = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    frame = memory.capture("head", image, ctx())
    return memory, image, frame


def observe(memory, frame, output=None, context=None, current=None):
    context = context or ctx()
    return GuardedObjectObserver(lambda _: answer() if output is None else output).observe(
        memory, frame, ROLES, context=context, current_context=current or (lambda: context))


def test_real_provider_boundary_pixels_and_untrusted_location():
    memory, source, frame = setup_memory()
    original = source.copy()
    source[:] = 0
    calls = []

    def transport(request):
        np.testing.assert_array_equal(request["frames"]["head"], original)
        request["frames"]["head"][:] = 42
        calls.append(request)
        return {"ok": True, "data": {"text": json.dumps(answer())}}

    provider = PolicyServiceVisionPlanner(transport, max_images=1, max_tokens=384)
    result = GuardedObjectObserver(provider).observe(memory, frame, ROLES, context=ctx(), current_context=ctx)
    assert result["accepted"] and len(calls) == 1
    value = memory.read(ctx())
    assert value["authority"] == "observation_only_no_task_confirmation"
    assert value["current"]["objects"][0]["localization_available"]
    assert value["current"]["objects"][0]["observation_ref"].startswith(frame["frame_id"])
    assert value["current"]["frame"] == frame
    value["current"]["objects"][0]["box"][0] = 999
    assert memory.read(ctx())["current"]["objects"][0]["box"][0] == 10


@pytest.mark.parametrize("source,expected,fraction", [
    ([30, 30, 40, 40], "center_inside", 1.0),
    ([0, 0, 10, 10], "center_outside", 0.0),
    ([10, 25, 30, 35], "boundary", 0.5),
    ([10, 25, 25, 35], "center_outside", 1/3),
])
def test_image_relation_is_numeric_not_semantic(source, expected, fraction):
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0]["box"] = source
    raw["objects"][1]["box"] = [20, 20, 60, 50]
    assert observe(memory, frame, raw)["accepted"]
    before = memory.read(ctx())
    result = memory.measure_image_relation("base", "upper", ctx())
    assert result["image_relation"] == expected
    assert result["source_box_overlap_fraction"] == pytest.approx(fraction)
    assert result["semantic_outcome"] == "not_verified"
    assert result["frame"] == frame
    assert memory.read(ctx()) == before
    result["frame"]["timestep"] = 999
    assert memory.read(ctx())["current"]["frame"] == frame


@pytest.mark.parametrize("kind", ["stale", "recaptured", "missing", "low_confidence", "same_box", "unrequested"])
def test_image_relation_abstains_without_bound_pair(kind):
    memory, image, frame = setup_memory()
    raw = answer()
    if kind == "missing":
        raw["objects"][0].update(state="unknown", box=None)
    if kind == "low_confidence":
        raw["objects"][0]["confidence"] = .1
    if kind == "same_box":
        raw["objects"][1]["box"] = raw["objects"][0]["box"].copy()
    observe(memory, frame, raw)
    if kind == "recaptured":
        memory.capture("head", image, ctx())
    result = memory.measure_image_relation("other" if kind == "unrequested" else "base", "upper",
        ctx(1) if kind == "stale" else ctx())
    assert result["image_relation"] == "unknown"
    assert result["semantic_outcome"] == "not_verified"


@pytest.mark.parametrize("source,target", [("base", "base"), ("", "upper"), (None, "upper")])
def test_image_relation_rejects_invalid_roles(source, target):
    memory, _, _ = setup_memory()
    with pytest.raises(ValueError):
        memory.measure_image_relation(source, target, ctx())


def test_image_relation_rejects_cross_episode():
    memory, _, frame = setup_memory()
    observe(memory, frame)
    with pytest.raises(ValueError):
        memory.measure_image_relation("base", "upper", ctx(episode="another"))


@pytest.mark.parametrize("target", [[10, 10, 60, 39], [11, 9, 59, 41]])
def test_image_relation_abstains_on_rounding_equivalent_regions(target):
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0]["box"] = [10, 10, 60, 40]
    raw["objects"][1]["box"] = target
    assert observe(memory, frame, raw)["accepted"]
    result = memory.measure_image_relation("base", "upper", ctx())
    assert result["image_relation"] == "unknown"
    assert result["reason"] == "indistinguishable_role_boxes_at_pixel_resolution"
    assert result["semantic_outcome"] == "not_verified"


@pytest.mark.parametrize("change", [
    {"box": [-1, 0, 10, 10]}, {"box": [0, 0, 97, 32]}, {"box": [0, 10, 12, 10]},
    {"box": [True, 0, 10, 10]}, {"box": [0.0, 0, 10, 10]}, {"box": None},
    {"state": "absent"}, {"state": "unknown"}, {"state": "confirmed"},
    {"confidence": True}, {"confidence": float("nan")}, {"confidence": float("inf")},
    {"confidence": -0.1}, {"confidence": 1.1}, {"evidence": ""}, {"evidence": "x" * 513},
    {"actions": [0]}, {"reward": 1}, {"role_id": "not-requested"},
])
def test_malformed_grounding_fails_whole_snapshot(change):
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0].update(change)
    result = observe(memory, frame, raw)
    assert not result["accepted"]
    value = memory.read(ctx())["current"]
    assert not value["accepted"] and value["objects"] == []


@pytest.mark.parametrize("state,confidence", [("unknown", 0.9), ("absent", 0.9), ("located", 0.1)])
def test_unknown_absent_and_low_confidence_have_no_current_location(state, confidence):
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0].update(state=state, confidence=confidence)
    if state != "located":
        raw["objects"][0]["box"] = None
    assert observe(memory, frame, raw)["accepted"]
    obj = memory.read(ctx())["current"]["objects"][0]
    assert not obj["localization_available"] and obj["current_box"] is None


@pytest.mark.parametrize("box,expected", [([0, 0, 1000, 1000], [0, 0, 96, 64]),
                                        ([100, 200, 600, 700], [9, 12, 58, 45])])
def test_normalized_coordinates_are_explicit_and_converted_once(box, expected):
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0]["box"] = box
    result = GuardedObjectObserver(lambda _: raw, coordinate_system="normalized_1000").observe(
        memory, frame, ROLES, context=ctx(), current_context=ctx)
    assert result["accepted"]
    obj = memory.read(ctx())["current"]["objects"][0]
    assert obj["model_box"] == box and obj["box"] == expected and obj["current_box"] == expected
    assert obj["model_coordinate_system"] == "normalized_1000"
    assert obj["coordinate_system"] == "native_pixels"


def test_invalid_normalized_coordinates_are_not_clipped_or_reinterpreted():
    memory, _, frame = setup_memory()
    raw = answer()
    raw["objects"][0]["box"] = [0, 0, 1001, 500]
    result = GuardedObjectObserver(lambda _: raw, coordinate_system="normalized_1000").observe(
        memory, frame, ROLES, context=ctx(), current_context=ctx)
    assert not result["accepted"]
    with pytest.raises(ValueError, match="explicit"):
        GuardedObjectObserver(lambda _: raw, coordinate_system="guess")


@pytest.mark.parametrize("raw", [{"objects": []}, {"objects": answer()["objects"] * 2},
                                 {"objects": [answer()["objects"][0]] * 2},
                                 {**answer(), "completed": True}, "truncated {"])
def test_coverage_unknown_fields_and_truncation_rejected(raw):
    memory, _, frame = setup_memory()
    assert not observe(memory, frame, raw)["accepted"]


def test_new_capture_at_same_step_invalidates_positions_and_identity():
    memory, image, frame = setup_memory()
    assert observe(memory, frame)["accepted"]
    newer = memory.capture("head", image, ctx())
    before = memory.read(ctx())
    assert before["current"] is None and len(before["history"]) == 1
    prior = before["history"][0]["objects"][0]
    assert prior["last_observed_box"] == [10, 10, 60, 25]
    assert prior["current_box"] is None and not prior["localization_available"]
    assert observe(memory, newer)["accepted"]
    assert memory.read(ctx())["current"]["objects"][0]["observation_ref"] != prior["observation_ref"]


def test_step_advance_revokes_current_and_cannot_read_backwards():
    memory, _, frame = setup_memory()
    assert observe(memory, frame)["accepted"]
    assert memory.read(ctx(1))["current"] is None
    with pytest.raises(ValueError, match="backwards"):
        memory.read(ctx())


def test_failed_capture_and_context_change_cannot_revive_old_positions():
    memory, image, frame = setup_memory()
    assert observe(memory, frame)["accepted"]
    assert memory.read(dataclasses.replace(ctx(), deployment_profile_id="changed"))["current"] is None
    assert memory.read(ctx())["current"] is None
    frame = memory.capture("head", image, ctx())
    assert observe(memory, frame)["accepted"]
    with pytest.raises(ValueError, match="RGB"):
        memory.capture("head", None, ctx())
    assert memory.read(ctx())["current"] is None


@pytest.mark.parametrize("operation", ["capture", "reset", "advance", "context_error"])
def test_late_model_reply_cannot_overwrite_new_context(operation):
    memory, image, frame = setup_memory()
    live = [ctx()]

    def infer(_):
        if operation == "reset":
            memory.reset("e1")
        elif operation == "capture":
            memory.capture("head", image, ctx())
        elif operation == "advance":
            live[0] = ctx(1)
        else:
            live[0] = None
        return answer()

    def context():
        if live[0] is None:
            raise RuntimeError("context unavailable")
        return live[0]

    result = GuardedObjectObserver(infer).observe(memory, frame, ROLES, context=ctx(), current_context=context)
    assert not result["accepted"]
    assert memory.read(live[0] or ctx())["current"] is None


def test_reset_and_episode_types_reject_old_handles():
    memory, image, frame = setup_memory()
    assert observe(memory, frame)["accepted"]
    memory.reset("e1")
    assert memory.read(ctx())["history"] == []
    assert not observe(memory, frame)["accepted"]
    memory.reset(1)
    for alias in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="another episode"):
            memory.capture("head", image, ctx(0, alias))


def test_history_is_bounded_and_repeat_call_does_not_run_provider():
    memory, image, frame = setup_memory()
    for step in range(5):
        frame = memory.capture("head", image, ctx(step))
        assert observe(memory, frame, context=ctx(step))["accepted"]
    assert len(memory.read(ctx(4))["history"]) == 2
    calls = []
    result = GuardedObjectObserver(lambda req: calls.append(req)).observe(
        memory, frame, ROLES, context=ctx(4), current_context=lambda: ctx(4))
    assert not result["accepted"] and calls == []


def test_optional_tool_permissions_budget_and_read_only_output(tmp_path):
    memory, _, frame = setup_memory()
    observe(memory, frame)

    def unavailable(*args):
        raise AssertionError("no physical, success or persistent memory binding allowed")

    workspace = RunWorkspace(tmp_path, RunManifest("objects", "offline", "ground", 0, "fixture", "none", "none"))
    kwargs = {"bindings": EmbodiedToolBindings(*([unavailable] * 6)), "workspace": workspace}
    default = CanonicalToolRuntime(**kwargs)
    assert "read_object_evidence" not in default.registry.tool_names
    runtime = CanonicalToolRuntime(**kwargs, object_evidence=memory, tool_call_budgets={"read_object_evidence": 1})
    denied = runtime.invoke("read_object_evidence", {}, context=dataclasses.replace(ctx(), allowed_tools=()))
    assert not denied.accepted
    with pytest.raises(ValueError, match="forbidden"):
        runtime.invoke("read_object_evidence", {"actions": [0]}, context=ctx())
    accepted = runtime.invoke("read_object_evidence", {}, context=ctx())
    assert accepted.accepted and accepted.output == memory.read(ctx())
    assert not runtime.invoke("read_object_evidence", {}, context=ctx()).accepted
    assert workspace.event_path.exists()
