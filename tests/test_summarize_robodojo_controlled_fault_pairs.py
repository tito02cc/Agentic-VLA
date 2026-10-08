from scripts.summarize_robodojo_controlled_fault_pairs import (
    _mcnemar_exact_two_sided,
    _wilson_interval,
)


def test_mcnemar_three_one_way_conversions_is_not_significant() -> None:
    assert _mcnemar_exact_two_sided(0, 3) == 0.25


def test_mcnemar_no_discordant_pairs_is_one() -> None:
    assert _mcnemar_exact_two_sided(0, 0) == 1.0


def test_wilson_three_of_three_remains_wide() -> None:
    lower, upper = _wilson_interval(3, 3)
    assert 0.43 < lower < 0.44
    assert upper == 1.0
