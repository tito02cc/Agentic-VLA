"""Detector contracts; synthetic boxes do not establish visual accuracy."""

import dataclasses
import json
import numpy as np
import pytest

from agentic_vla.toolchain.grounded_detection import (
    select_detection, detector_candidates, GroundedDetectionProvider, DetectorObjectObserver,
    build_identity_request, parse_identity_response,
    build_blind_identity_request, parse_blind_identity_response, apply_blind_identity_veto,
    build_context_inspection,
)
from agentic_vla.toolchain import ObjectEvidenceMemory, ObjectRole, ToolExecutionContext


def candidate(box, score=.8):
    return {"box": box, "score": score, "label": "mouse"}


def test_context_inspection_is_exact_source_pixels_with_context():
    image = np.arange(240*320*3, dtype=np.uint8).reshape(240, 320, 3)
    before = image.copy()
    meta, crop = build_context_inspection(image, [candidate([90, 90, 110, 110])])
    assert meta["available"] and meta["crop_box"] == [20, 20, 180, 180]
    np.testing.assert_array_equal(crop, image[20:180, 20:180])
    crop[:] = 0
    np.testing.assert_array_equal(image, before)
    assert "confirmation" in meta["authority"]


@pytest.mark.parametrize("boxes", [[], [candidate([10, 10, 20, 20]), candidate([50, 50, 70, 70])]])
def test_context_inspection_abstains_on_missing_or_ambiguous_anchor(boxes):
    meta, crop = build_context_inspection(np.zeros((100, 100, 3), np.uint8), boxes)
    assert not meta["available"] and crop is None


def test_context_inspection_preserves_extent_at_edge_and_binds_pixels():
    image = np.zeros((100, 200, 3), np.uint8)
    meta, crop = build_context_inspection(image, [candidate([180, 80, 199, 99])])
    assert meta["crop_box"] == [40, 0, 200, 100]
    assert crop.shape == (100, 160, 3)
    image[0, 0, 0] = 1
    other, _ = build_context_inspection(image, [candidate([180, 80, 199, 99])])
    assert meta["source_rgb_sha256"] != other["source_rgb_sha256"]


@pytest.mark.parametrize("scale,side", [(0, 160), (float("nan"), 160), (4, 0), (True, 160)])
def test_context_inspection_rejects_invalid_extent(scale, side):
    with pytest.raises(ValueError):
        build_context_inspection(np.zeros((100, 100, 3), np.uint8), [], context_scale=scale, minimum_side=side)


def test_no_detection_is_unknown_not_absent():
    row = select_detection("mouse", [], (100, 100, 3))
    assert row["state"] == "unknown" and row["box"] is None


@pytest.mark.parametrize("labels", [[], [""]])
def test_empty_detector_batch_decode_is_not_an_error(labels):
    import torch
    assert detector_candidates({"boxes": torch.empty((0, 4)), "scores": torch.empty(0),
                                "text_labels": labels}) == []


@pytest.mark.parametrize("boxes,scores,labels", [([], [], ["mouse"]),
    ([[1, 1, 2, 2]], [.8], []), ([], [.8], [""]), ([[1, 1, 2, 2]], [.8], [None])])
def test_nonempty_mismatches_are_not_silently_discarded(boxes, scores, labels):
    import torch
    with pytest.raises(ValueError):
        detector_candidates({"boxes": torch.tensor(boxes), "scores": torch.tensor(scores),
                             "text_labels": labels})


def test_duplicates_suppressed_but_two_instances_are_unknown():
    row = select_detection("mouse", [candidate([10, 10, 30, 30]), candidate([11, 10, 30, 30], .7)], (100, 100, 3))
    assert row["state"] == "located" and row["box"] == [10, 10, 30, 30]
    row = select_detection("mouse", [candidate([10, 10, 30, 30]), candidate([50, 50, 80, 80])], (100, 100, 3))
    assert row["state"] == "unknown" and "multiple" in row["evidence"]


@pytest.mark.parametrize("box,score", [([0, 0, 10, 10], True), ([0, 0, 10, 10], float("nan")),
    ([0, 0, 10, 10], 1.1), ([True, 0, 10, 10], .9), ([0, 0, float("inf"), 10], .9), ([5, 5, 4, 10], .9)])
def test_invalid_detector_output_rejected(box, score):
    with pytest.raises(ValueError):
        select_detection("mouse", [candidate(box, score)], (100, 100, 3))


def test_valid_boxes_clip_to_image_and_round_outward():
    row = select_detection("pad", [candidate([-1.5, 1.2, 105.5, 98.1])], (100, 100, 3))
    assert row["box"] == [0, 1, 100, 99]
    assert select_detection("pad", [candidate([110, 10, 120, 20])], (100, 100, 3))["state"] == "unknown"


@pytest.mark.parametrize("value", [True, 0, 1, -.5, float("nan")])
def test_invalid_configuration_fails_before_loading(tmp_path, value):
    with pytest.raises(ValueError):
        GroundedDetectionProvider(tmp_path, threshold=value)


def test_detector_uses_existing_frame_bound_read_only_contract(tmp_path):
    class Stub(GroundedDetectionProvider):
        def __call__(self, request):
            assert list(request["frames"]) == ["head"]
            return {"objects": [select_detection("mouse", [candidate([10, 10, 30, 30])], (100, 100, 3))]}
    observer = DetectorObjectObserver(Stub(tmp_path))
    memory = ObjectEvidenceMemory()
    memory.reset("episode")
    context = ToolExecutionContext("episode", 0, True, ("read_object_evidence",))
    frame = memory.capture("head", np.zeros((100, 100, 3), np.uint8), context)
    result = observer.observe(memory, frame, (ObjectRole("mouse", "computer mouse"),),
        context=context, current_context=lambda: context)
    assert result["accepted"]
    current = memory.read(context)["current"]["objects"][0]
    assert current["authority"] == "model_localization_hypothesis_only"
    assert "not_calibrated" in current["confidence_source"]
    later = memory.read(dataclasses.replace(context, timestep=1))
    assert later["current"] is None
    assert not later["history"][-1]["objects"][0]["localization_available"]


def identity_fixture():
    rgb = np.arange(100*100*3, dtype=np.uint8).reshape(100, 100, 3)
    proposals = [{"role_id": "mouse", "caption": "computer mouse", "candidates": [candidate([10, 10, 30, 30])]},
                 {"role_id": "pad", "caption": "mouse pad", "candidates": []}]
    return rgb, build_identity_request(rgb, proposals)


def test_identity_inputs_are_original_pixels_without_scores_or_task_state():
    rgb, (request, catalog) = identity_fixture()
    np.testing.assert_array_equal(request["frames"]["global_rgb"], rgb)
    np.testing.assert_array_equal(request["frames"]["role0_candidate0"], rgb[10:30, 10:30])
    assert "score" not in request["user_prompt"]
    assert "completion" not in request["user_prompt"]
    assert catalog[0]["candidates"][0]["score"] == .8


def test_identity_can_reject_single_wrong_proposal_and_cannot_invent_absence():
    _, (_, catalog) = identity_fixture()
    raw = {"objects": [{"role_id": r, "candidate_id": None, "evidence": "No identifiable matching candidate"}
                       for r in ("mouse", "pad")]}
    parsed = parse_identity_response(raw, catalog)
    assert all(r["state"] == "unknown" and r["box"] is None for r in parsed["objects"])


@pytest.mark.parametrize("cid", ["invented", "role1_candidate0", 0, True, [], {}])
def test_identity_rejects_invented_or_wrong_role_ids(cid):
    _, (_, catalog) = identity_fixture()
    raw = {"objects": [{"role_id": "mouse", "candidate_id": cid, "evidence": "visible"},
                       {"role_id": "pad", "candidate_id": None, "evidence": "unknown"}]}
    with pytest.raises(ValueError):
        parse_identity_response(raw, catalog)


def test_identity_uses_proposal_coordinates_and_score_not_generated_coordinates():
    _, (_, catalog) = identity_fixture()
    raw = {"objects": [{"role_id": "mouse", "candidate_id": "role0_candidate0", "evidence": "computer mouse"},
                       {"role_id": "pad", "candidate_id": None, "evidence": "unknown"}]}
    obj = parse_identity_response(json.dumps(raw), catalog)["objects"][0]
    assert obj["box"] == [10, 10, 30, 30] and obj["confidence"] == .8
    raw["objects"][0]["box"] = [1, 1, 90, 90]
    with pytest.raises(ValueError):
        parse_identity_response(raw, catalog)


def test_identity_rejects_excess_candidates_without_selecting_top_score():
    proposals = [{"role_id": "obj", "caption": "object", "candidates":
                  [candidate([i*15, 0, i*15+10, 10]) for i in range(5)]}]
    with pytest.raises(ValueError, match="too many candidates"):
        build_identity_request(np.zeros((100, 100, 3), np.uint8), proposals)


def blind_fixture(kind="standalone_object"):
    _, (request, catalog) = identity_fixture()
    blind_request, mapping = build_blind_identity_request(request, catalog)
    identity = {"objects": [{"role_id": "mouse", "candidate_id": "role0_candidate0", "evidence": "visible mouse"},
                             {"role_id": "pad", "candidate_id": None, "evidence": "unknown"}]}
    blind = {"regions": [{"region_id": "region_0", "entity_type": kind, "name": "visible entity",
                           "evidence": "visible outline"}]}
    allowed = {role: frozenset({"standalone_object", "scene_fixture"}) for role in ("mouse", "pad")}
    return request, catalog, blind_request, mapping, identity, blind, allowed


def test_blind_perception_does_not_receive_role_labels_or_detector_scores():
    request, catalog, blind, mapping, *_ = blind_fixture()
    import hashlib
    def digest(frames):
        return sorted(hashlib.sha256(x.tobytes()).hexdigest() for x in frames.values())
    assert digest(request["frames"]) == digest(blind["frames"])
    assert mapping == {"role0_candidate0": "region_0"}
    text = blind["system_prompt"] + blind["user_prompt"]
    for withheld in ("mouse", "pad", "role0", "score"):
        assert withheld not in text
    request["user_prompt"] = "a completely different target identity"
    catalog[0]["description"] = "another goal"
    rebuilt, _ = build_blind_identity_request(request, catalog)
    assert rebuilt["user_prompt"] == blind["user_prompt"]
    assert rebuilt["system_prompt"] == blind["system_prompt"]


@pytest.mark.parametrize("kind,expected", [("standalone_object", "located"), ("scene_fixture", "located"),
                                        ("robot_part", "unknown"), ("unknown", "unknown")])
def test_blind_check_only_vetoes_never_selects_or_promotes(kind, expected):
    _, catalog, _, mapping, identity, blind, allowed = blind_fixture(kind)
    result = apply_blind_identity_veto(identity, catalog, blind, mapping, allowed_types=allowed)
    assert result["objects"][0]["state"] == expected
    assert result["objects"][1]["state"] == "unknown"
    if expected == "located":
        assert result["objects"][0]["box"] == [10, 10, 30, 30]
        assert result["objects"][0]["confidence"] == .8
    else:
        assert result["objects"][0]["box"] is None
        assert result["objects"][0]["confidence"] == 0


@pytest.mark.parametrize("change", [{"entity_type": "mouse"}, {"entity_type": True},
    {"region_id": "invented"}, {"name": ""}, {"box": [1, 1, 9, 9]}])
def test_blind_schema_rejects_untyped_or_invented_results(change):
    *_, mapping, identity, blind, allowed = blind_fixture()
    blind["regions"][0].update(change)
    with pytest.raises(ValueError):
        parse_blind_identity_response(blind, mapping)


def test_blind_schema_requires_full_coverage_no_duplicate_regions():
    _, _, _, mapping, _, blind, _ = blind_fixture()
    blind["regions"].append(dict(blind["regions"][0]))
    with pytest.raises(ValueError):
        parse_blind_identity_response(blind, mapping)
    blind["regions"] = []
    with pytest.raises(ValueError):
        parse_blind_identity_response(blind, mapping)


def test_type_authority_is_host_declared_and_mapping_cannot_be_swapped():
    _, catalog, _, mapping, identity, blind, allowed = blind_fixture()
    with pytest.raises(ValueError):
        apply_blind_identity_veto(identity, catalog, blind, {"invented": "region_0"}, allowed_types=allowed)
    allowed["mouse"] = frozenset({"unknown"})
    with pytest.raises(ValueError):
        apply_blind_identity_veto(identity, catalog, blind, mapping, allowed_types=allowed)
