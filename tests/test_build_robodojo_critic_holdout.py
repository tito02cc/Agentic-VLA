from scripts.build_robodojo_critic_holdout import _sample_indices


def test_sample_indices_reserve_only_terminal_frame_for_safe_stop() -> None:
    samples = _sample_indices(101)

    assert samples[-1] == ("terminal", 100, "safe_stop")
    assert len(samples) == 6
    assert all(target == "continue" for _, _, target in samples[:-1])
    assert len({index for _, index, _ in samples}) == len(samples)
