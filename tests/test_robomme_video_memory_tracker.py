from __future__ import annotations

import numpy as np

from scripts.track_robomme_video_memory import (
    detect_colored_objects,
    final_relation,
    identity_provenance,
    mask_loss_diagnostics,
    parse_identity,
    select_relation,
)


def test_parse_identity_extracts_color_and_relation() -> None:
    assert parse_identity("red block initially at bottom-left") == ("red", "bottom-left")


def test_color_detection_and_relation_selection() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[30:45, 160:175] = (255, 0, 0)
    frame[170:190, 45:65] = (255, 0, 0)

    candidates = detect_colored_objects(frame, "red")
    selected = select_relation(candidates, "bottom-left")

    assert len(candidates) == 2
    assert np.allclose(selected["point"], [179.5, 54.5])


def test_final_relation_labels_bottom_left_track() -> None:
    candidates = [
        {"point": [40.0, 160.0]},
        {"point": [110.0, 130.0]},
        {"point": [180.0, 55.0]},
    ]
    assert final_relation(candidates, candidates[-1]) == "bottom-left"


def test_mask_loss_gate_rejects_repeated_ambiguous_reacquisition() -> None:
    trace = [
        *({"mask_area": 100} for _ in range(4)),
        *({"mask_area": 0} for _ in range(3)),
        *({"mask_area": 100} for _ in range(4)),
        *({"mask_area": 0} for _ in range(2)),
        *({"mask_area": 100} for _ in range(4)),
    ]
    diagnostics = mask_loss_diagnostics(trace)
    assert diagnostics["loss_runs"] == [3, 2]
    assert diagnostics["reacquisitions"] == 2
    assert diagnostics["admitted"] is False


def test_mask_loss_gate_accepts_one_bounded_occlusion() -> None:
    trace = [
        *({"mask_area": 100} for _ in range(4)),
        *({"mask_area": 0} for _ in range(12)),
        *({"mask_area": 100} for _ in range(4)),
    ]
    assert mask_loss_diagnostics(trace)["admitted"] is True


def test_mask_loss_gate_rejects_empty_or_unresolved_final_observation() -> None:
    assert mask_loss_diagnostics([])["admitted"] is False
    assert mask_loss_diagnostics([{"mask_area": 100}, {"mask_area": 0}])["admitted"] is False


def test_rejected_motion_candidate_does_not_claim_vlm_memory() -> None:
    text = identity_provenance(identity="red block", has_structured_identity=False,
                               diagnostics={"admitted": False, "reacquisitions": 4})
    assert "unverified motion-selected candidate" in text
    assert "no VLM identity was supplied" in text
    assert "kept from the VLM memory" not in text


def test_tracking_provenance_preserves_accepted_and_structured_cases() -> None:
    assert identity_provenance(identity="red block", has_structured_identity=True,
                               diagnostics={"admitted": True}) == "tracked through the demonstration from red block"
    text = identity_provenance(identity="red block", has_structured_identity=True,
                               diagnostics={"admitted": False, "reacquisitions": 2})
    assert "unverified structured identity" in text
