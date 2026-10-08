import copy
import json

import numpy as np

import pytest

from agentic_vla.benchmarks.robomme_memory import (
    MemoryUncertainError,
    identity_motion_evidence,
    load_verified_identity_memory,
    validate_identity_memory,
)


def document():
    return {
        "protocol": "carve.robomme.unmask_swap_memory.v1",
        "task": "VideoUnmaskSwap", "episode": 0,
        "evaluator_or_oracle_fields_used": False,
        "memory": {
            "planner_hint": "Untrusted free text must not propagate",
            "required_color_order": ["green", "blue"],
            "hidden_container_points": {c: [10., 20.] for c in ("red", "green", "blue")},
            "video_instance_tracking": {"objects": {
                c: {"admission": {"admitted": True}, "trace": [{"point": [10., 20.]}]}
                for c in ("red", "green", "blue")
            }},
        },
    }


def validate(value):
    return validate_identity_memory(value, task="VideoUnmaskSwap", episode=0,
                                    instruction="pick green then blue")


def repick_document():
    return {"protocol": "carve.robomme.video_instance_memory.v1", "task": "VideoRepick", "episode": 11,
            "instruction": "repeat twice", "evaluator_or_oracle_fields_used": False,
            "memory": {"planner_hint": "untrusted completed repetitions: 2",
                       "video_instance_tracking": {"target_color": "red", "resolved_identity": "leftmost red block",
                       "admission": {"admitted": True},
                       "initial_selection": {"method": "initial_patch_persistence", "changed_frame_margin": 5},
                       "trace": [{"point": [100., 150.]}]}}}


def test_repick_memory_uses_only_bound_structured_identity():
    result = validate_identity_memory(repick_document(), task="VideoRepick", episode=11, instruction="repeat twice")
    assert "leftmost red block" in result["planner_hint"]
    assert "<|box_start|>" not in result["planner_hint"]
    assert "untrusted" not in result["planner_hint"]
    assert "not a current position or a completed repetition" in result["planner_hint"]
    assert set(result) == {"planner_hint"}


@pytest.mark.parametrize("field,value", [("target_color", "pink"), ("admission", {"admitted": False}),
    ("initial_selection", {"method": "initial_patch_persistence", "changed_frame_margin": 0}),
    ("trace", [{"point": [float("nan"), 20.]}]),
    ("resolved_identity", "leftmost blue block")])
def test_repick_rejects_ambiguous_or_invalid_identity(field, value):
    doc = repick_document()
    doc["memory"]["video_instance_tracking"][field] = value
    with pytest.raises(ValueError):
        validate_identity_memory(doc, task="VideoRepick", episode=11, instruction="repeat twice")


@pytest.mark.parametrize("field,value", [("episode", 12), ("instruction", "repeat three times"),
                                      ("evaluator_or_oracle_fields_used", True)])
def test_repick_rejects_foreign_task_instructions_and_oracle_provenance(field, value):
    doc = repick_document()
    doc[field] = value
    with pytest.raises(ValueError):
        validate_identity_memory(doc, task="VideoRepick", episode=11, instruction="repeat twice")


def test_structured_memory_does_not_copy_free_text_or_completion_claims():
    result = validate(document())
    assert "Untrusted" not in result["planner_hint"]
    assert "not verified current positions" in result["planner_hint"]
    assert "video_instance_tracking" not in result
    assert "<|box_start|>(39,78)<|box_end|>" in result["planner_hint"]


def test_videounmask_memory_uses_public_demo_without_swap_claim():
    value = document()
    value["task"] = "VideoUnmask"
    value["protocol"] = "carve.robomme.video_unmask_memory.v1"
    result = validate_identity_memory(
        value, task="VideoUnmask", episode=0, instruction="pick green then blue"
    )
    assert "at the end of the public demonstration" in result["planner_hint"]
    assert "swap" not in result["planner_hint"]
    assert result["required_color_order"] == ["green", "blue"]
    value["protocol"] = "carve.robomme.unmask_swap_memory.v1"
    with pytest.raises(ValueError, match="unsupported identity memory protocol"):
        validate_identity_memory(
            value, task="VideoUnmask", episode=0, instruction="pick green then blue"
        )


@pytest.mark.parametrize("task,protocol", [
    ("VideoUnmask", "carve.robomme.video_unmask_memory.v1"),
    ("VideoUnmaskSwap", "carve.robomme.unmask_swap_memory.v1"),
])
def test_coarse_grid_hint_retains_identity_without_exact_points(task, protocol):
    value = document()
    value["task"] = task
    value["protocol"] = protocol
    value["memory"]["hidden_container_points"] = {
        "red": [90., 156.], "green": [74., 89.], "blue": [104., 85.],
    }
    for color, point in value["memory"]["hidden_container_points"].items():
        value["memory"]["video_instance_tracking"]["objects"][color]["trace"][-1]["point"] = point
    result = validate_identity_memory(
        value, task=task, episode=0, instruction="pick green then blue",
        hint_style="coarse_grid",
    )
    hint = result["planner_hint"]
    assert "green-hidden container in grid cell (row 2, column 2)" in hint
    assert "blue-hidden container in grid cell (row 3, column 2)" in hint
    assert "<|box_start|>" not in hint
    assert "fresh exact point" in hint
    assert result["hidden_container_points"]["green"] == [74., 89.]


def test_coarse_grid_rejects_ambiguous_target_cells():
    value = document()
    with pytest.raises(MemoryUncertainError, match="share one coarse grid cell"):
        validate_identity_memory(
            value, task="VideoUnmaskSwap", episode=0,
            instruction="pick green then blue", hint_style="coarse_grid",
        )


def test_motion_gate_distinguishes_identity_swap_from_common_camera_motion():
    value = document()
    value["memory"]["video_instance_tracking"]["objects"] = {
        "red": {"trace": [{"point": [134., 146.]}, {"point": [90., 159.]}]},
        "green": {"trace": [{"point": [89., 110.]}, {"point": [89., 110.]}]},
        "blue": {"trace": [{"point": [90., 159.]}, {"point": [134., 146.]}]},
    }
    moved = identity_motion_evidence(value)
    assert moved["use_memory"] is True
    assert moved["max_relative_motion_px"] > 45
    assert moved["threshold_px"] == 16.

    for track in value["memory"]["video_instance_tracking"]["objects"].values():
        first = track["trace"][0]["point"]
        track["trace"][-1]["point"] = [first[0] + 20., first[1] + 10.]
    common = identity_motion_evidence(value)
    assert common["use_memory"] is False
    assert common["max_relative_motion_px"] == 0.

    value["memory"]["video_instance_tracking"]["objects"]["blue"]["trace"][0]["point"] = None
    with pytest.raises(MemoryUncertainError, match="invalid blue track endpoints"):
        identity_motion_evidence(value)


def test_motion_gate_keeps_admission_and_cost_when_memory_is_not_used(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same public demo")
    value = document()
    value["source_demo"] = str(source)
    value["compile_time_s"] = 2.5
    memory_file = tmp_path / "memory.json"
    memory_file.write_text(json.dumps(value))
    memory, receipt = load_verified_identity_memory(
        memory_file, current_demo=source, task="VideoUnmaskSwap", episode=0,
        instruction="pick green then blue", memory_use_policy="motion_gate",
    )
    assert memory is None
    assert receipt["admitted"] is True and receipt["used"] is False
    assert receipt["fallback"] is None
    assert receipt["compile_time_s"] == 2.5
    assert receipt["motion_evidence"]["use_memory"] is False

    from scripts.run_robomme_vlm_groundsg import GroundedPlannerClient

    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content":
                "pick up the container at <|box_start|>(391,316)<|box_end|> that hides the green cube"
            }}]}).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    common = dict(
        endpoint="http://planner.test", model="test", timeout_s=1,
        demo_video=source, task_name="VideoUnmaskSwap", task_goal="pick green then blue",
        repair_attempts=0, execution_feedback="execution_chunks",
        procedure_authority="observe_only", planner_context="native",
        grounding_authority="observe_only", planner_demo_mode="always",
        image_format="png",
    )
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    gated = GroundedPlannerClient(**common, task_memory=memory)
    control = GroundedPlannerClient(**common)
    assert gated.infer(frame)["grounded_subgoal"] == control.infer(frame)["grounded_subgoal"]
    assert requests[0] == requests[1]


@pytest.mark.parametrize("mutation", ["episode", "oracle", "order", "point", "tracking", "trace"])
def test_invalid_or_mismatched_memory_is_rejected(mutation):
    value = copy.deepcopy(document())
    memory = value["memory"]
    if mutation == "episode":
        value["episode"] = 1
    elif mutation == "oracle":
        value["evaluator_or_oracle_fields_used"] = True
    elif mutation == "order":
        memory["required_color_order"] = ["blue", "green"]
    elif mutation == "point":
        memory["hidden_container_points"]["red"] = [float("nan"), 20.]
    elif mutation == "tracking":
        memory["video_instance_tracking"]["objects"]["red"]["admission"]["admitted"] = False
    else:
        memory["video_instance_tracking"]["objects"]["red"]["trace"][-1]["point"] = [30., 40.]
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("matching", [True, False])
def test_demonstration_binding_checks_decoded_pixels(tmp_path, monkeypatch, matching):
    source = tmp_path / "source.mp4"
    current = tmp_path / "current.mp4"
    source.write_bytes(b"original encoding")
    current.write_bytes(b"new encoding")
    value = document()
    value["source_demo"] = str(source)
    memory_file = tmp_path / "memory.json"
    memory_file.write_text(json.dumps(value))

    class Reader:
        def __init__(self, pixel):
            self.pixel = pixel

        def __iter__(self):
            yield np.full((2, 2, 3), self.pixel, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("imageio.v2.get_reader",
                        lambda path: Reader(0 if matching or path == source else 1))
    kwargs = dict(current_demo=current, task="VideoUnmaskSwap", episode=0,
                  instruction="pick green then blue")
    if matching:
        _, receipt = load_verified_identity_memory(memory_file, **kwargs)
        assert receipt["demonstration_match"] == "decoded_rgb_exact"
        assert not receipt["completion_evidence"]
    else:
        with pytest.raises(ValueError, match="does not match"):
            load_verified_identity_memory(memory_file, **kwargs)
