import pytest

from scripts.summarize_robodojo_optimize_ablation import _wilson_interval


def test_wilson_interval_for_three_of_three_is_wide() -> None:
    lower, upper = _wilson_interval(3, 3)

    assert lower == pytest.approx(0.4385029682449546)
    assert upper == 1.0


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = _wilson_interval(2, 3)

    assert lower < 2 / 3 < upper
