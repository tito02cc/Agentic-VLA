import pytest

from scripts.run_robomme_feedback_pilot import memory_cases, planner_protocol_failure, regression_cases, repick_cases


def test_fixed_paired_regression_alternates_order():
    cases = regression_cases([1, 2, 3, 4, 5])
    assert len(cases) == 10
    assert cases[0][1] == "raw_every"
    assert cases[2][1] == "observe_every"
    for episode in range(1, 6):
        pair = [case for case in cases if case[0] == episode]
        assert {case[1] for case in pair} == {"raw_every", "observe_every"}
        assert all(case[-1] == "every_chunk" for case in pair)


@pytest.mark.parametrize("episodes", [[0, 1, 2, 3, 4], [1, 1, 2, 3, 4], [1], [1, 2, 3, 4, 50]])
def test_reject_invalid_regression(episodes):
    with pytest.raises(ValueError):
        regression_cases(episodes)


def test_memory_matrix_has_five_complete_pairs_and_alternates_order():
    cases = memory_cases()
    assert len(cases) == 10
    assert cases[0][1] == "observe_memory_off"
    assert cases[2][1] == "observe_memory_on"
    for episode in range(5):
        pair = [case for case in cases if case[0] == episode]
        assert {case[1] for case in pair} == {"observe_memory_off", "observe_memory_on"}
        assert all(case[2:] == ("carve", "execution_chunks", "every_chunk") for case in pair)


def test_protocol_failure_does_not_hide_transport_or_gpu_errors():
    assert planner_protocol_failure({"failure": "grounded Planner output contains a malformed point token"})
    assert planner_protocol_failure({"failure": "unsupported VideoUnmaskSwap action skill"})
    for message in (None, "CUDA out of memory", "grounded Planner request failed: connection refused"):
        assert not planner_protocol_failure({"failure": message})


def test_memory_extension_preserves_registered_episode_order():
    cases = memory_cases([10, 11, 12, 13, 14])
    assert [case[0] for case in cases] == [ep for ep in range(10, 15) for _ in range(2)]
    assert cases[0][1].endswith("_off")
    assert cases[2][1].endswith("_on")


def test_repick_matrix_keeps_raw_and_no_memory_controls():
    cases = repick_cases()
    assert len(cases) == 6
    assert cases[0] == (11, "raw_every", "raw", "legacy_predictions", "every_chunk")
    assert cases[-1] == (12, "raw_every", "raw", "legacy_predictions", "every_chunk")
    for ep in (11, 12):
        assert {c[1] for c in cases if c[0] == ep} == {"raw_every", "observe_memory_off", "observe_memory_on"}


@pytest.mark.parametrize("episodes", [[0], [0, 0, 1, 2, 3], [-1, 0, 1, 2, 3], [46, 47, 48, 49, 50]])
def test_memory_extension_rejects_invalid_episode_sets(episodes):
    with pytest.raises(ValueError, match="five distinct"):
        memory_cases(episodes)
