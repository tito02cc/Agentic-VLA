"""Tests for the low-frequency visual Critic contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agentic_vla.toolchain import (
    GuardedGroupedVisualVerifier,
    GuardedScalarVisualVerifier,
    GuardedVisualVerifier,
    ScalarVisualObservation,
    ScalarVisualPredicate,
    VisualVerificationContext,
    build_visual_verification_request,
    evaluate_scalar_visual_predicate,
    parse_scalar_visual_observation,
)


def context() -> VisualVerificationContext:
    return VisualVerificationContext(
        task_instruction="put both moka pots on the stove",
        expected_outcome="first moka pot is on the stove",
        frames={"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
        timestep=150,
        active_stage="place_first",
    )


def evidence_reply(**overrides):
    return dict(visibility="clear", relation="supported", evidence_views=["agentview"],
                observed_outcome="The pot rests on the stove", confidence=0.9) | overrides


@pytest.mark.parametrize("visibility,relation,status", [
    ("clear", "supported", "confirmed"), ("clear", "refuted", "contradicted"),
    ("clear", "unknown", "inconclusive"), ("occluded", "supported", "inconclusive"),
    ("out_of_view", "refuted", "inconclusive"), ("ambiguous", "unknown", "inconclusive"),
])
def test_evidence_protocol_derives_status_and_vetoes_missing_visibility(visibility, relation, status):
    verifier = GuardedVisualVerifier(lambda _: evidence_reply(visibility=visibility, relation=relation), protocol="evidence")
    result = verifier.verify(context())
    assert result.accepted
    assert result.report.status.value == status
    assert result.report.metadata["evidence_is_model_reported"] is True


@pytest.mark.parametrize("overrides", [
    {"evidence_views": ["imaginary_camera"]}, {"evidence_views": []},
    {"evidence_views": ["agentview", "agentview"]}, {"evidence_views": "agentview"},
    {"confidence": True}, {"confidence": "0.9"}, {"confidence": float("nan")},
    {"confidence": 2}, {"status": "confirmed"}, {"visibility": "yes"},
    {"relation": "confirmed"}, {"observed_outcome": ""},
])
def test_evidence_protocol_rejects_schema_or_camera_invention(overrides):
    result = GuardedVisualVerifier(lambda _: evidence_reply(**overrides), protocol="evidence").verify(context())
    assert not result.accepted
    assert result.report.status.value == "inconclusive"


def test_evidence_protocol_cannot_cite_only_historical_image():
    import dataclasses
    ctx = dataclasses.replace(context(), frames={"before_agentview": np.zeros((8, 8, 3), dtype=np.uint8)})
    result = GuardedVisualVerifier(lambda _: evidence_reply(evidence_views=["before_agentview"]), protocol="evidence").verify(ctx)
    assert not result.accepted
    assert "historical" in result.error


@pytest.mark.parametrize("status", ["confirmed", "contradicted"])
@pytest.mark.parametrize("evidence", ["The drawer is not visible", "The object is outside the camera view", "Orientation is unclear"])
def test_legacy_definitive_report_cannot_rely_on_unobservable_evidence(status, evidence):
    result = GuardedVisualVerifier(lambda _: {"status": status, "observed_outcome": evidence, "confidence": 0.99}).verify(context())
    assert not result.accepted
    assert result.report.status.value == "inconclusive"


def test_evidence_payload_keeps_predicate_and_task_without_manual_labels():
    requests = []
    def infer(request):
        requests.append(request)
        return evidence_reply()
    GuardedVisualVerifier(infer, protocol="evidence").verify(context())
    payload = json.loads(requests[0]["user_prompt"])
    assert payload["expected_outcome"] == context().expected_outcome
    assert payload["task_instruction"] == context().task_instruction
    assert "reference" not in payload
    assert "not seeing an object never proves absence" in requests[0]["system_prompt"]


@pytest.mark.parametrize("counts,relations,complete,expected", [
    ([2], ["stacked"], True, "inconclusive"),
    ([3], ["stacked"], True, "confirmed"),
    ([3], ["touching"], True, "inconclusive"),
    ([None], ["nested"], False, "inconclusive"),
    ([3], ["nested"], False, "inconclusive"),
    ([2, 1], ["stacked", "singleton"], True, "contradicted"),
    ([2, 2], ["stacked", "stacked"], True, "inconclusive"),
])
def test_group_coverage_is_required_for_usable_progress(counts, relations, complete, expected):
    raw = {"visible": True, "groups": ["target group" for _ in counts],
           "member_counts": counts, "group_relations": relations,
           "all_targets_visible": complete, "confidence": 0.9}
    result = GuardedGroupedVisualVerifier(lambda _: raw, expected_object_count=3).verify(
        context(), ScalarVisualPredicate("groups", "Count target groups", "eq", 1),
    )
    assert result.accepted
    assert result.report.status.value == expected
    assert result.report.metadata["task_completion_authorized"] is False
    if expected == "inconclusive":
        assert result.report.metadata["observed_value"] is None
        assert result.report.metadata["coverage_valid"] is False


@pytest.mark.parametrize("change", [
    {"visible": "false"}, {"all_targets_visible": "true"},
    {"member_counts": [True]}, {"member_counts": [3.0]},
    {"member_counts": []}, {"group_relations": ["singleton"]},
    {"group_relations": ["invented"]},
])
def test_coverage_schema_rejects_invalid_values(change):
    raw = {"visible": True, "groups": ["stack at center"], "member_counts": [3],
           "group_relations": ["stacked"], "all_targets_visible": True, "confidence": 0.9}
    raw.update(change)
    result = GuardedGroupedVisualVerifier(lambda _: raw, expected_object_count=3).verify(
        context(), ScalarVisualPredicate("groups", "Count target groups", "eq", 1),
    )
    assert not result.accepted
    assert result.report.status.value == "inconclusive"


def test_coverage_mode_rejects_legacy_reply_and_discloses_inventory():
    requests = []
    def infer(request):
        requests.append(request)
        return {"visible": True, "groups": ["two bowls stacked at center"], "confidence": 0.9}
    result = GuardedGroupedVisualVerifier(infer, expected_object_count=3).verify(
        context(), ScalarVisualPredicate("groups", "Count target groups", "eq", 1),
    )
    assert not result.accepted
    assert json.loads(requests[0]["user_prompt"])["expected_object_count"] == 3


def test_request_exposes_only_symbolic_predicate_and_images() -> None:
    request = build_visual_verification_request(context())
    payload = json.loads(request["user_prompt"])

    assert payload["active_stage"] == "place_first"
    assert payload["expected_outcome"] == "first moka pot is on the stove"
    assert "reward" not in payload
    assert "Never infer hidden state, reward" in request["system_prompt"]
    assert "at most 16 words" in request["system_prompt"]
    assert "without Markdown" in request["system_prompt"]
    assert set(request["frames"]) == {"agentview"}


def test_guarded_verifier_accepts_valid_three_way_report() -> None:
    verifier = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "one moka pot is visibly supported by the stove",
            "confidence": 0.91,
        }
    )

    result = verifier.verify(context())

    assert result.accepted
    assert result.report.status.value == "confirmed"
    assert verifier.metrics()["calls"] == 1


def test_guarded_verifier_rejects_confirmation_with_negative_evidence() -> None:
    verifier = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "The left and center bowls are not stacked.",
            "confidence": 0.95,
        }
    )

    result = verifier.verify(context())

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "conflicts with negative observed evidence" in str(result.error)


def test_grouped_verifier_rejects_physically_impossible_group_count() -> None:
    verifier = GuardedGroupedVisualVerifier(
        lambda _request: {
            "visible": True,
            "groups": ["left bowl", "center bowl", "right bowl", "stacked bowls"],
            "confidence": 0.9,
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="three_bowl_stack_groups",
        question="Enumerate groups formed by three bowls.",
        comparison="eq",
        target=2,
        unit="groups",
        minimum_valid_value=1,
        maximum_valid_value=3,
    )

    result = verifier.verify(context(), predicate)

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "outside the physically valid range" in str(result.error)


def test_guarded_verifier_fails_to_inconclusive_on_bad_output() -> None:
    verifier = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "trust simulator reward",
            "confidence": 0.2,
        }
    )

    result = verifier.verify(context())

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "confidence threshold" in str(result.error)


def test_no_change_vetoes_semantic_confirmation() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    verifier = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "moka pot is on stove",
            "confidence": 0.95,
        }
    )
    result = verifier.verify(
        VisualVerificationContext(
            task_instruction="put the moka pot on the stove",
            expected_outcome="moka pot is on stove",
            frames={"before_agentview": frame, "current_agentview": frame.copy()},
            timestep=100,
            require_visual_change=True,
        )
    )

    assert result.accepted
    assert result.report.status.value == "inconclusive"
    assert result.report.metadata["mean_pixel_change"] == 0.0


def test_scalar_predicate_derives_status_from_observed_value() -> None:
    predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many spatially separate target-object groups remain?",
        comparison="eq",
        target=1,
    )
    observation = parse_scalar_visual_observation(
        {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 1,
            "confidence": 0.9,
            "evidence": "All targets form one visible group.",
        }
    )

    report = evaluate_scalar_visual_predicate(predicate, observation)

    assert report.status.value == "confirmed"
    assert report.metadata["observed_value"] == 1.0


def test_scalar_predicate_fails_closed_on_occlusion() -> None:
    predicate = ScalarVisualPredicate(
        predicate_id="placed_tools",
        question="How many tools are visibly inside the toolbox?",
        comparison="ge",
        target=3,
    )
    observation = ScalarVisualObservation(
        predicate_id="placed_tools",
        visible=False,
        value=None,
        confidence=0.8,
        evidence="The toolbox interior is occluded.",
    )

    report = evaluate_scalar_visual_predicate(predicate, observation)

    assert report.status.value == "inconclusive"


def test_guarded_scalar_verifier_ignores_free_form_success_labels() -> None:
    verifier = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 3,
            "confidence": 0.92,
            "evidence": "Three separate white bowl groups are visible.",
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many spatially separate white bowl groups are visible?",
        comparison="eq",
        target=1,
    )

    result = verifier.verify(context(), predicate)

    assert result.accepted
    assert result.report.status.value == "contradicted"
    assert result.report.metadata["observed_value"] == 3.0
    assert verifier.metrics()["calls"] == 1


def test_guarded_scalar_verifier_fails_closed_on_schema_drift() -> None:
    verifier = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 1,
            "confidence": 0.95,
            "evidence": "One group is visible.",
            "status": "success",
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many target groups are visible?",
        comparison="eq",
        target=1,
    )

    result = verifier.verify(context(), predicate)

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "unknown fields" in str(result.error)


def test_grouped_verifier_counts_enumerated_groups_in_harness() -> None:
    verifier = GuardedGroupedVisualVerifier(
        lambda _request: {
            "visible": True,
            "groups": ["green bowl on left", "white bowl on right"],
            "confidence": 0.92,
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="target_bowl_groups",
        question="Enumerate spatially separate target bowl groups.",
        comparison="eq",
        target=1,
    )

    result = verifier.verify(context(), predicate)

    assert result.accepted
    assert result.report.status.value == "contradicted"
    assert result.report.metadata["observed_value"] == 2.0
    assert result.report.metadata["groups"] == [
        "green bowl on left",
        "white bowl on right",
    ]


def test_grouped_verifier_rejects_model_supplied_count() -> None:
    verifier = GuardedGroupedVisualVerifier(
        lambda _request: {
            "visible": True,
            "groups": ["green bowl on left", "white bowl on right"],
            "value": 1,
            "confidence": 0.92,
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="target_bowl_groups",
        question="Enumerate spatially separate target bowl groups.",
        comparison="eq",
        target=1,
    )

    result = verifier.verify(context(), predicate)

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "unknown fields" in str(result.error)


def test_grouped_verifier_rejects_ambiguous_multi_instance_aggregate() -> None:
    verifier = GuardedGroupedVisualVerifier(
        lambda _request: {
            "visible": True,
            "groups": ["three bowls on table"],
            "confidence": 0.8,
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="target_bowl_groups",
        question="Enumerate spatially separate target bowl groups.",
        comparison="eq",
        target=2,
    )

    result = verifier.verify(context(), predicate)

    assert not result.accepted
    assert result.report.status.value == "inconclusive"
    assert "grouping relation" in str(result.error)


def test_grouped_verifier_accepts_explicit_multi_instance_stack() -> None:
    verifier = GuardedGroupedVisualVerifier(
        lambda _request: {
            "visible": True,
            "groups": ["two bowls stacked at left", "one bowl at right"],
            "confidence": 0.9,
        }
    )
    predicate = ScalarVisualPredicate(
        predicate_id="target_bowl_groups",
        question="Enumerate spatially separate target bowl groups.",
        comparison="eq",
        target=2,
    )

    result = verifier.verify(context(), predicate)

    assert result.accepted
    assert result.report.status.value == "confirmed"
    assert result.report.metadata["observed_value"] == 2.0
