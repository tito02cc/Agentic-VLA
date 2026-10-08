from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_starvla_ddim import (
    _fixed_examples,
    summarize_latencies,
    summarize_paired_profile,
)


def test_summarize_latencies_reports_nearest_rank_percentiles() -> None:
    summary = summarize_latencies([10.0, 20.0, 30.0, 40.0])

    assert summary["calls"] == 4
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 20.0
    assert summary["p95_ms"] == 40.0


def test_summarize_latencies_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_latencies([])


def test_fixed_input_matches_starvla_pi_v3_state_contract() -> None:
    example = _fixed_examples()[0]

    assert example["state"].shape == (1, 14)
    assert len(example["image"]) == 3
    assert all(image.shape == (224, 224, 3) for image in example["image"])


def test_paired_profile_reports_latency_and_action_drift() -> None:
    baseline_actions = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]
    candidate_actions = [np.array([0.5, 1.0]), np.array([1.0, 3.0])]

    summary = summarize_paired_profile(
        [100.0, 120.0],
        [80.0, 100.0],
        baseline_actions,
        candidate_actions,
    )

    assert summary["pairs"] == 2
    assert summary["candidate_minus_baseline_mean_ms"] == -20.0
    assert summary["candidate_mean_latency_reduction_percent"] == pytest.approx(
        18.181818
    )
    assert summary["action_mae_mean"] == 0.375
