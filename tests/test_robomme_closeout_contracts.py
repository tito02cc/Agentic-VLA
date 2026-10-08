"""CPU-only closeout checks; none of these are robot performance measurements."""

import base64
import hashlib
import io
import json
import urllib.error

import imageio.v3 as iio
import numpy as np
import pytest

from agentic_vla.benchmarks.robomme_memory import (
    MemoryUncertainError, load_verified_identity_memory, planner_demo_frames,
)
from scripts import build_robomme_unmask_swap_memory as compiler
from scripts import run_robomme_vlm_groundsg as runner


def rejection_document(source, **changes):
    return {"protocol": "carve.robomme.unmask_swap_memory.v1", "task": "VideoUnmaskSwap",
            "episode": 11, "instruction": "pick green then blue", "memory": {},
            "evaluator_or_oracle_fields_used": False, "source_demo": str(source),
            "source_demo_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "admission": {"admitted": False, "reason": "ambiguous RGB association"},
            "compile_time_s": 12.5, "shared_model_load_s": 3.0, **changes}


def load_rejection(tmp_path, document, source, **kwargs):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(document))
    return load_verified_identity_memory(path, current_demo=source, task="VideoUnmaskSwap",
        episode=11, instruction="pick green then blue", on_uncertain="without_memory", **kwargs)


def test_demo_boundary_and_png_are_explicit_and_lossless():
    frame = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    frames = [frame, frame + 1, frame + 2]
    assert len(planner_demo_frames(frames, "official")) == 2
    assert len(planner_demo_frames(frames, "legacy_full")) == 3
    assert not planner_demo_frames([frame], "official")
    with pytest.raises(ValueError):
        planner_demo_frames(frames, "invented")
    encoded = runner.encode_image(frame, "png")
    assert encoded.startswith("data:image/png;base64,")
    decoded = iio.imread(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
    np.testing.assert_array_equal(frame, decoded)


def clients(demo=None):
    common = dict(endpoint="http://planner.test", model="test", timeout_s=1,
                  demo_video=demo, task_goal="pick green then blue", task_name="VideoUnmaskSwap",
                  image_format="png")
    return (runner.RawGroundedPlannerClient(**common), runner.GroundedPlannerClient(
        **common, execution_feedback="execution_chunks", procedure_authority="observe_only",
        planner_context="native", grounding_authority="observe_only", repair_attempts=0,
        semantic_transition_confirmations=1, planner_demo_mode="always"))


def fake_http(monkeypatch, payload, requests=None):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(payload).encode()

    def respond(request, **kwargs):
        if requests is not None:
            requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", respond)


def test_raw_and_native_have_equal_png_demo_prompt_history_and_output(monkeypatch, tmp_path):
    requests = []
    raw, native = clients(tmp_path / "demo.mp4")
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for content in ("pick up the container at <|box_start|>(390,400)<|box_end|> that hides the green cube",
                    "put down the container", "put down the container"):
        fake_http(monkeypatch, {"choices": [{"message": {"content": content}}]}, requests)
        baseline, candidate = raw.infer(frame), native.infer(frame)
        assert requests[-1] == requests[-2]
        assert baseline["grounded_subgoal"] == candidate["grounded_subgoal"]
        assert baseline["points"] == candidate["points"]
        assert candidate["monitor_fallback"] is None
        assert candidate["tool_grounding"] is None
    for client in (raw, native):
        assert client.request_count == 3
        assert all(r["demo_video_sent"] and r["output_status"] == "accepted" for r in client.request_audit)


@pytest.mark.parametrize("content", ["complete", "put it down", "", None,
    "pick up the container at <|box_start|>(300,400)<|box_end|> that hides the green cube <|box_start|>bad"])
def test_raw_and_native_reject_same_illegal_output_and_count_cost(monkeypatch, content):
    fake_http(monkeypatch, {"choices": [{"message": {"content": content}}]})
    for client in clients():
        with pytest.raises((RuntimeError, ValueError)):
            client.infer(np.zeros((256, 256, 3), dtype=np.uint8))
        assert client.request_count == 1 and not client.history_text
        assert client.request_audit[0]["output_status"] == "rejected"
        assert client.request_audit[0]["request_latency_ms"] >= 0


@pytest.mark.parametrize("error", [TimeoutError("timed out"), urllib.error.URLError("connection refused")])
def test_failed_planner_requests_are_timed_and_retained(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("urllib.request.urlopen", fail)
    for client in clients():
        with pytest.raises(RuntimeError, match="request failed"):
            client.infer(np.zeros((256, 256, 3), dtype=np.uint8))
        assert client.request_audit[0]["output_status"] == "request_failed"
        assert client.request_audit[0]["request_latency_ms"] >= 0


def test_uncertain_memory_falls_back_with_cost_and_strict_mode_still_raises(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same demo")
    doc = rejection_document(source)
    memory, receipt = load_rejection(tmp_path, doc, source)
    assert memory is None and receipt["fallback"] == "without_identity_memory"
    assert not receipt["admitted"] and receipt["compile_time_s"] == 12.5
    assert receipt["rejection_reason"] == "ambiguous RGB association"
    with pytest.raises(MemoryUncertainError):
        load_verified_identity_memory(tmp_path / "memory.json", current_demo=source,
            task="VideoUnmaskSwap", episode=11, instruction="pick green then blue")


@pytest.mark.parametrize("changes", [
    {"episode": 12}, {"task": "VideoRepick"}, {"evaluator_or_oracle_fields_used": True},
    {"instruction": "pick blue then green"}, {"memory": {"partial_hint": "do it"}},
    {"admission": {"admitted": False, "reason": ""}}, {"admission": {"admitted": "false"}},
    {"source_demo_sha256": "wrong"}, {"compile_time_s": -1}, {"compile_time_s": float("nan")},
    {"shared_model_load_s": True}, {"source_demo_relative": "/wrong"},
])
def test_invalid_memory_never_masquerades_as_uncertainty(tmp_path, changes):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same demo")
    with pytest.raises((ValueError, KeyError)) as caught:
        load_rejection(tmp_path, rejection_document(source, **changes), source)
    assert not isinstance(caught.value, MemoryUncertainError)


def test_compiler_writes_explicit_rejection_with_cost_and_does_not_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "initial_demo_front.mp4"
    source.write_bytes(b"demo fixture")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"task": "VideoUnmaskSwap", "episode": 11,
                                  "instruction": "pick green then blue", "demo_history_mode": "official"}))
    def reject(**kwargs):
        raise MemoryUncertainError("ambiguous source")
    monkeypatch.setattr(compiler, "_compile_case", reject)
    kwargs = dict(predictor=None, summary_path=summary, output_root=tmp_path / "memory", checkpoint=tmp_path / "unused")
    doc = compiler.compile_case(**kwargs, shared_model_load_s=2.0)
    assert doc["admission"] == {"admitted": False, "reason": "ambiguous source"}
    assert doc["compile_time_s"] >= 0 and doc["shared_model_load_s"] == 2.0
    assert doc["demo_history_mode"] == "official"
    memory, receipt = load_rejection(tmp_path, doc, source)
    assert memory is None and receipt["rejection_reason"] == "ambiguous source"
    with pytest.raises(FileExistsError):
        compiler.compile_case(**kwargs)


def test_videounmask_compiler_rejection_has_distinct_protocol(tmp_path, monkeypatch):
    source = tmp_path / "initial_demo_front.mp4"
    source.write_bytes(b"demo fixture")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"task": "VideoUnmask", "episode": 11,
                                   "instruction": "pick blue then green", "demo_history_mode": "official"}))
    def reject(**kwargs):
        raise MemoryUncertainError("tracking uncertain")
    monkeypatch.setattr(compiler, "_compile_case", reject)
    doc = compiler.compile_case(predictor=None, summary_path=summary,
                                output_root=tmp_path / "memory", checkpoint=tmp_path / "unused")
    assert doc["protocol"] == "carve.robomme.video_unmask_memory.v1"
    assert doc["admission"] == {"admitted": False, "reason": "tracking uncertain"}


def test_compiler_does_not_convert_infrastructure_error_into_memory_rejection(tmp_path, monkeypatch):
    (tmp_path / "initial_demo_front.mp4").write_bytes(b"demo")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"task": "VideoUnmaskSwap", "episode": 11, "instruction": "pick green"}))
    def fail(**kwargs):
        raise OSError("disk error")
    monkeypatch.setattr(compiler, "_compile_case", fail)
    with pytest.raises(OSError):
        compiler.compile_case(predictor=None, summary_path=summary, output_root=tmp_path / "memory",
                              checkpoint=tmp_path / "unused")
    assert not (tmp_path / "memory").exists()


def test_latency_summary_does_not_call_first_request_warm():
    result = runner.request_latency_summary([100., 10., 20.])
    assert result["first_ms"] == 100. and result["warm_count"] == 2
    assert result["warm_p50_ms"] == 15. and result["total_ms"] == 130.
    assert runner.request_latency_summary([])["warm_p95_ms"] is None
