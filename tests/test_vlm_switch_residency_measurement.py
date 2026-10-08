"""Contracts for the VLM-switch / residency measurement tool.

Two properties carry the conclusions this tool produces, so both are pinned:

* a missing per-phase timing must fail loudly, because item C's whole point is
  that VLA inference, VLM generation and weight transfer are accounted for
  separately rather than collapsed into one number;
* the state comparison must detect any change in the VLA action, since "VLM
  switching is output-preserving" is exactly what it is used to assert.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_vlm_switch_and_residency",
    REPO_ROOT / "scripts/measure_vlm_switch_and_residency.py",
)
measure = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(measure)


class _Planner:
    """Stands in for CpuStagedVisionPlanner, returning chosen timings."""

    def __init__(self, timings: dict[str, float]) -> None:
        self._timings = timings

    def decide(self, **_kwargs):
        return {
            "text": "The gripper is above the bowls.",
            "input_tokens": 100,
            "output_tokens": 12,
            "timings": dict(self._timings),
        }


def _full_timings() -> dict[str, float]:
    return {key: 1.0 for key in measure.PHASE_KEYS}


def test_every_phase_must_be_reported() -> None:
    """A collapsed or partial timing dict must not silently pass through."""

    frames = {"head": np.zeros((4, 4, 3), dtype=np.uint8)}
    result = measure.planner_call(_Planner(_full_timings()), frames, 16)
    assert set(measure.PHASE_KEYS) <= set(result["timings"])

    incomplete = _full_timings()
    del incomplete["vlm_generate_ms"]
    with pytest.raises(RuntimeError, match="did not report phases"):
        measure.planner_call(_Planner(incomplete), frames, 16)


def test_transfer_phases_are_separate_from_generation() -> None:
    """Weight movement must never be counted as generation, or the split lies."""

    assert "vlm_generate_ms" not in measure.TRANSFER_PHASES
    assert "vlm_load_ms" not in measure.TRANSFER_PHASES
    assert set(measure.TRANSFER_PHASES) == {
        "vla_to_cpu_ms",
        "vlm_to_gpu_ms",
        "vlm_to_cpu_ms",
        "vla_restore_ms",
    }
    # Every transfer phase is also reported individually.
    assert set(measure.TRANSFER_PHASES) <= set(measure.PHASE_KEYS)


def _state(arrays: list[np.ndarray]) -> dict:
    return {
        "action_sha256": [measure.replay.action_sha256(a) for a in arrays],
        "vla_latency": measure.summarize([1.0] * len(arrays)),
        "_arrays": arrays,
    }


def test_identical_states_compare_as_output_preserving() -> None:
    arrays = [np.arange(16 * 14, dtype=np.float32).reshape(16, 14) for _ in range(3)]
    reference = _state(arrays)
    candidate = _state([a.copy() for a in arrays])

    result = measure.compare_states(reference, candidate)

    assert result["bit_identical_actions"] == 3
    assert result["bit_identical_fraction"] == 1.0
    assert result["max_abs_diff_over_payloads"] == 0.0
    assert result["digest_sequence_equal"] is True


def test_a_single_perturbed_element_is_detected() -> None:
    """The comparison must not smooth over a one-element change."""

    arrays = [np.zeros((16, 14), dtype=np.float32) for _ in range(3)]
    perturbed = [a.copy() for a in arrays]
    perturbed[1][7, 3] = np.float32(1e-4)

    result = measure.compare_states(_state(arrays), _state(perturbed))

    assert result["bit_identical_actions"] == 2
    assert result["digest_sequence_equal"] is False
    assert result["max_abs_diff_over_payloads"] == pytest.approx(1e-4)


def test_summarize_reports_spread_not_just_a_mean() -> None:
    stats = measure.summarize([10.0, 20.0, 30.0])

    assert stats == {
        "n": 3,
        "min_ms": 10.0,
        "p50_ms": 20.0,
        "max_ms": 30.0,
        "mean_ms": 20.0,
    }
    assert measure.summarize([]) is None


def test_head_frame_is_contiguous_and_unmodified() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    payload = {"examples": [{"image": [image, image, image], "lang": "x"}]}

    frame = measure.head_frame(payload)

    np.testing.assert_array_equal(frame, image)
    assert frame.flags["C_CONTIGUOUS"]
