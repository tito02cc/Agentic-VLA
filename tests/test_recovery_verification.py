import numpy as np
import pytest

from agentic_vla.toolchain import GuardedVisualVerifier
from agentic_vla.toolchain.recovery_verification import RecoveryVerificationLifecycle

IMAGE = np.zeros((32, 48, 3), dtype=np.uint8)


def begun():
    gate = RecoveryVerificationLifecycle()
    gate.observe(10)
    gate.begin(step=10, expected_outcome="The bowl is in the stack", image=IMAGE)
    return gate


def test_generation_is_not_execution_and_fresh_observation_is_required():
    gate = begun()
    assert not gate.ready(step=10, work_remaining=0)
    gate.issued_chunk(step=10, execute_steps=16)
    assert not gate.ready(step=26, work_remaining=0)
    gate.observe(25)
    assert not gate.ready(step=25, work_remaining=0)
    gate.observe(26)
    assert not gate.ready(step=26, work_remaining=1)
    assert gate.ready(step=26, work_remaining=0)


@pytest.mark.parametrize("status", ["confirmed", "contradicted", "inconclusive"])
def test_post_action_report_is_bounded_not_task_success(status):
    gate = begun()
    gate.issued_chunk(step=10, execute_steps=16)
    gate.issued_chunk(step=26, execute_steps=16)
    gate.observe(42)
    calls = []

    def infer(request):
        calls.append(request)
        return {
            "status": status,
            "confidence": 0.9,
            "observed_outcome": "Visible bowls provide the observation",
        }

    result = gate.verify(
        step=42,
        work_remaining=0,
        image=IMAGE,
        task_instruction="Stack bowls",
        verifier=GuardedVisualVerifier(infer),
    )
    assert result["status"] == status and len(calls) == 1
    assert result["verification_step"] >= result["attempt"]["last_chunk_end"]
    assert result["authority"] == "visual_report_not_task_success_or_memory_promotion"
    assert result["attempt"]["before_rgb_sha256"] == result["after_rgb_sha256"]
    assert gate.pending is None and gate.checks == 1
    assert (
        gate.verify(
            step=42,
            work_remaining=0,
            image=IMAGE,
            task_instruction="Stack bowls",
            verifier=None,
        )
        is None
    )


def test_deadline_does_not_promote_late_confirmation():
    gate = begun()
    gate.issued_chunk(step=10, execute_steps=1)
    gate.observe(11)
    clock = iter([0, 13])
    gate.clock = lambda: next(clock)
    result = gate.verify(
        step=11,
        work_remaining=0,
        image=IMAGE,
        task_instruction="Stack bowls",
        verifier=GuardedVisualVerifier(
            lambda _: {
                "status": "confirmed",
                "confidence": 0.9,
                "observed_outcome": "One stack is visible",
            }
        ),
    )
    assert result["status"] == "inconclusive" and not result["accepted"]
    assert "deadline" in result["error"]


def test_overlapping_recovery_and_budget_are_rejected():
    gate = begun()
    with pytest.raises(ValueError, match="pending"):
        gate.begin(step=10, expected_outcome="Stack bowls", image=IMAGE)
    gate.close()
    gate.checks = 2
    assert not gate.available
    with pytest.raises(ValueError, match="budget"):
        gate.begin(step=10, expected_outcome="Stack bowls", image=IMAGE)


def test_early_reset_retains_unverified_attempt_receipt():
    gate = begun()
    gate.issued_chunk(step=10, execute_steps=16)
    result = gate.close()
    assert result["status"] == "unverified_at_episode_reset"
    assert result["pending"]["last_chunk_end"] == 26
    assert gate.pending is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_checks": 0},
        {"max_checks": True},
        {"deadline_s": float("nan")},
        {"deadline_s": 0},
    ],
)
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        RecoveryVerificationLifecycle(**kwargs)
