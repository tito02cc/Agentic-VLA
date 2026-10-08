import numpy as np
import pytest

from agentic_vla.toolchain.track_quality import assess_track_masks


def masks():
    value = np.zeros((2, 16, 16), dtype=bool)
    value[0, 2:6, 2:8] = True
    value[1, 10:14, 2:8] = True
    return value


def test_clean_masks_are_only_candidates():
    raw = masks()
    result = assess_track_masks(("base", "upper"), raw, raw.copy())
    assert not result["requires_reobservation"]
    assert all(r["state"] == "tracking_candidate" for r in result["objects"].values())
    assert result["authority"] == "tracking_diagnostic_only_not_verified_identity"


def test_exclusivity_does_not_erase_raw_conflict():
    raw = masks()
    raw[1, 2:6, 2:8] = True
    output = raw.copy()
    output[1, 2:6, 2:8] = False
    result = assess_track_masks(("base", "upper"), raw, output)
    assert not np.any(output[0] & output[1])
    assert result["requires_reobservation"]
    assert all(r["state"] == "ambiguous" for r in result["objects"].values())
    assert result["pairs"][0]["fraction_of_smaller_mask"] == 1


def test_lost_mask_never_uses_previous_box():
    raw = masks()
    output = raw.copy()
    output[1] = False
    result = assess_track_masks(("base", "upper"), raw, output)
    assert result["objects"]["upper"]["state"] == "lost"
    assert result["requires_reobservation"]
    assert "box" not in result["objects"]["upper"]


@pytest.mark.parametrize("fraction", [True, 0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold(fraction):
    with pytest.raises(ValueError):
        assess_track_masks(("a", "b"), masks(), masks(), conflict_fraction=fraction)


@pytest.mark.parametrize("roles", [("a", "a"), ("", "b"), ["a", "b"], (), ("a",)])
def test_invalid_roles(roles):
    with pytest.raises(ValueError):
        assess_track_masks(roles, masks(), masks())


def test_input_validation_and_no_mutation():
    raw = masks()
    with pytest.raises(ValueError):
        assess_track_masks(("a", "b"), raw.astype(float), raw)
    with pytest.raises(ValueError):
        assess_track_masks(("a", "b"), raw, raw[:, :-1])
    bad = raw.copy()
    bad[0, 15, 15] = True
    with pytest.raises(ValueError, match="invent"):
        assess_track_masks(("a", "b"), raw, bad)
    saved = raw.copy()
    assess_track_masks(("a", "b"), raw, raw)
    np.testing.assert_array_equal(raw, saved)
