import dataclasses
import json
import threading

import numpy as np
import pytest

from agentic_vla.toolchain.contracts import ToolExecutionContext
from agentic_vla.toolchain.object_memory import ObjectEvidenceMemory, ObjectRole
from agentic_vla.toolchain.reobservation import (
    BoundedObjectReobserver,
    NativeObjectObserver,
)

ROLES = (ObjectRole("base", "long board"), ObjectRole("upper", "short board"))
LABELS = {"long": "base", "short": "upper"}
IMAGE = np.zeros((48, 64, 3), dtype=np.uint8)


def context(step=0, episode="e"):
    return ToolExecutionContext(episode, step, False, ("read_object_evidence",))


def answer():
    return json.dumps([{"label": "long", "bbox_2d": [0, 0, 500, 200]},
                       {"label": "short", "bbox_2d": [500, 500, 800, 800]}])


def setup(infer=None, **kwargs):
    memory = ObjectEvidenceMemory()
    observer = NativeObjectObserver(infer or (lambda _: answer()), label_to_role=LABELS, user_prompt="Find boards")
    gate = BoundedObjectReobserver(observer, memory, **kwargs)
    gate.reset("e")
    return gate, memory


def call(gate, step=0, *, need=True, current=None, episode="e"):
    ctx = context(step, episode)
    return gate.maybe_observe(needs_reobservation=need, camera="head", image=IMAGE,
                             roles=ROLES, context=ctx, current_context=current or (lambda: ctx))


def test_budget_cooldown_no_stale_box_and_reset():
    calls = []
    gate, memory = setup(lambda req: calls.append(req) or answer(), max_calls=2, min_interval_steps=3)
    assert call(gate, need=False)["status"] == "not_needed"
    assert call(gate)["status"] == "hypothesis_available"
    obj = memory.read(context())["current"]["objects"][0]
    assert obj["confidence"] is None and obj["box"] == [0, 0, 32, 10]
    assert call(gate, 1)["status"] == "cooldown"
    assert memory.read(context(1))["current"] is None
    assert call(gate, 3)["provider_called"]
    assert call(gate, 4)["status"] == "budget_exhausted"
    assert len(calls) == 2
    gate.reset("new")
    assert call(gate, episode="new")["calls_used"] == 1
    assert memory.read(context(0, "new"))["history"] == []


@pytest.mark.parametrize("raw", ["[]", '[{"label":"long","bbox_2d":[0,0,500,200]},{"label":"short","bbox_2d":[0,0,500,200]}]'])
def test_omitted_or_duplicate_boxes_are_unknown(raw):
    gate, memory = setup(lambda _: raw)
    assert call(gate)["status"] == "unknown"
    objects = memory.read(context())["current"]["objects"]
    assert all(o["state"] == "unknown" and o["current_box"] is None for o in objects)


@pytest.mark.parametrize("raw", ["{", '{}', '[{"label":"bad","bbox_2d":[0,0,10,10]}]',
    '[{"label":"long","bbox_2d":[0,0,1001,1000]}]', '[{"label":"long","bbox_2d":[true,0,10,10]}]',
    '[{"label":"long","bbox_2d":[0,0,10,10],"action":1}]',
    '[{"label":"long","label":"short","bbox_2d":[0,0,10,10]}]',
    '```json\n[]\n', '[{"label":"long","bbox_2d":[0,0,10,10]},{"label":"long","bbox_2d":[0,0,10,10]}]'])
def test_invalid_native_output_consumes_budget_without_location(raw):
    gate, memory = setup(lambda _: raw, max_calls=1)
    result = call(gate)
    assert result["status"] == "rejected" and result["provider_called"]
    assert memory.read(context())["current"]["objects"] == []
    assert call(gate, 100)["status"] == "budget_exhausted"


def test_deadline_discards_returned_result_not_fake_cancellation():
    now = [0.0]
    def infer(_):
        now[0] = 13
        return answer()
    gate, memory = setup(infer, clock=lambda: now[0], reply_deadline_s=12)
    result = call(gate)
    assert result["status"] == "rejected" and result["provider_called"]
    assert "deadline" in result["result"]["error"]
    assert memory.read(context())["current"] is None


@pytest.mark.parametrize("reset", [False, True])
def test_inflight_coalescing_and_stale_reply(reset):
    started, release = threading.Event(), threading.Event()
    calls = []
    def infer(_):
        calls.append(1)
        started.set()
        assert release.wait(3)
        return answer()
    gate, memory = setup(infer)
    results = []
    worker = threading.Thread(target=lambda: results.append(call(gate)))
    worker.start()
    try:
        assert started.wait(3)
        if reset:
            gate.reset("e")
        assert call(gate, 1)["status"] == "coalesced"
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and len(calls) == 1
    assert results[0]["status"] == "stale_after_call"
    assert not results[0]["result"]["accepted"]
    assert memory.read(context(1))["current"] is None


def test_provider_failure_is_bounded():
    def fail(_):
        raise RuntimeError("transport failed")
    gate, memory = setup(fail, max_calls=1)
    assert call(gate)["status"] == "rejected"
    assert call(gate, 100)["status"] == "budget_exhausted"
    assert memory.read(context(100))["current"] is None


def test_live_context_and_validation_before_call():
    calls = []
    gate, _ = setup(lambda _: calls.append(1) or answer())
    assert call(gate, current=lambda: context(1))["status"] == "stale_before_call"
    assert not calls
    with pytest.raises(ValueError):
        call(gate, episode="wrong")
    with pytest.raises(TypeError):
        call(gate, need=1)
    call(gate, 2, need=False)
    with pytest.raises(ValueError):
        call(gate, 1)


@pytest.mark.parametrize("kwargs", [{"max_calls": 0}, {"max_calls": True}, {"min_interval_steps": 0},
                                    {"reply_deadline_s": float("nan")}, {"reply_deadline_s": 0}])
def test_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        setup(**kwargs)


def test_role_mismatch_never_calls_provider():
    calls = []
    gate, _ = setup(lambda _: calls.append(1) or answer())
    result = gate.maybe_observe(needs_reobservation=True, camera="head", image=IMAGE,
        roles=(dataclasses.replace(ROLES[0], role_id="other"), ROLES[1]), context=context(), current_context=context)
    assert not result["provider_called"] and result["status"] == "rejected" and not calls
