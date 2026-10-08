import json

import numpy as np
import pytest

from agentic_vla.benchmarks.robodojo_reobservation import RoboDojoReobservationShadow
from agentic_vla.toolchain import ToolExecutionContext


def config():
    return {"checkpoints": [0, 16], "model": "/local/vlm", "max_calls": 2,
            "min_interval_steps": 16, "reply_deadline_s": 12,
            "label_to_role": {"board": "target"}, "user_prompt": "Locate board", "system_prompt": "Observe only"}


def context(step):
    return ToolExecutionContext("e", step, False, ("read_object_evidence",))


def transport(_):
    return {"ok": True, "data": {"text": '[{"label":"board","bbox_2d":[100,100,500,500]}]',
            "backend": "cpu_staged_vlm", "action_model_restored": True, "model_path": "/local/vlm"}}


def test_live_shadow_returns_no_actions_and_invalidates(tmp_path):
    shadow = RoboDojoReobservationShadow(transport, config(), episode_id="e", output_dir=tmp_path / "shadow")
    pixels = np.full((48, 64, 3), 42, dtype=np.uint8)
    before = pixels.copy()
    for step in (0, 8, 16, 32):
        assert shadow.observe(pixels, step=step, current_context=lambda step=step: context(step)) is None
    np.testing.assert_array_equal(pixels, before)
    rows = [json.loads(s) for s in (tmp_path / "shadow/trace.jsonl").read_text().splitlines()]
    assert [r["outcome"]["provider_called"] for r in rows] == [True, False, True, False]
    assert [r["current_memory"] is not None for r in rows] == [True, False, True, False]
    assert all(r["control_enabled"] is False for r in rows)
    assert len(list((tmp_path / "shadow").glob("*.png"))) == 2
    shadow.reset()
    assert json.loads((tmp_path / "shadow/reset.json").read_text())["history"] == []
    with pytest.raises(RuntimeError, match="closed"):
        shadow.observe(pixels, step=33, current_context=lambda: context(33))


@pytest.mark.parametrize("change", [{"backend": "unexpected"}, {"action_model_restored": False}, {"model_path": "/wrong/model"}])
def test_wrong_residency_receipt_not_accepted(tmp_path, change):
    def invalid(request):
        response = transport(request)
        response["data"].update(change)
        return response
    shadow = RoboDojoReobservationShadow(invalid, config(), episode_id="e", output_dir=tmp_path / "shadow")
    shadow.observe(np.zeros((48, 64, 3), dtype=np.uint8), step=0, current_context=lambda: context(0))
    row = json.loads((tmp_path / "shadow/trace.jsonl").read_text())
    assert row["outcome"]["status"] == "rejected"
    assert not row["current_memory"]["accepted"]


def test_duplicate_checkpoints_rejected(tmp_path):
    values = config()
    values["checkpoints"] = [0, 0]
    with pytest.raises(ValueError):
        RoboDojoReobservationShadow(transport, values, episode_id="e", output_dir=tmp_path / "shadow")


def test_opt_in_reference_reaches_transport_and_reset_revokes(tmp_path):
    requests = []

    def provider(request):
        requests.append(request)
        return transport(request)

    shadow = RoboDojoReobservationShadow(provider, {**config(), "use_visual_reference": True},
        episode_id="e", output_dir=tmp_path / "reference")
    image = np.full((48, 64, 3), 42, dtype=np.uint8)
    for step in (0, 8, 16):
        assert shadow.observe(image, step=step, current_context=lambda step=step: context(step)) is None
    assert list(requests[0]["frames"]) == ["head"]
    assert list(requests[1]["frames"]) == ["reference_global", "reference_crop_target", "current"]
    rows = [json.loads(line) for line in (shadow.path / "trace.jsonl").read_text().splitlines()]
    assert [row["reference_used"] for row in rows] == [False, False, True]
    old_reference = shadow.reference
    shadow.reset()
    assert shadow.reference is None
    with pytest.raises(ValueError, match="revoked"):
        _ = old_reference.provenance


def test_failed_reference_does_not_silently_use_single_frame_fallback(tmp_path):
    calls = []

    def provider(request):
        calls.append(request)
        response = transport(request)
        response["data"]["text"] = "[]"
        return response

    shadow = RoboDojoReobservationShadow(provider, {**config(), "use_visual_reference": True},
        episode_id="e", output_dir=tmp_path / "reference")
    for step in (0, 16):
        shadow.observe(np.zeros((48, 64, 3), dtype=np.uint8), step=step,
            current_context=lambda step=step: context(step))
    assert len(calls) == 1
    rows = [json.loads(line) for line in (shadow.path / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["outcome"]["status"] == "reference_unavailable"
    assert rows[-1]["current_memory"] is None
