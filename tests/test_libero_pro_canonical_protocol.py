from __future__ import annotations

import pytest

from scripts.run_agentic_vla_libero_pro_canonical import _instruction_receipt_risk


def test_stale_subgoal_risk_uses_only_runtime_receipt_metadata() -> None:
    risk = _instruction_receipt_risk("leave both moka pots on the table")

    assert risk.event == "stale_subgoal"
    assert risk.bucket == "high"
    assert risk.evidence == {
        "source": "runtime_instruction_receipt",
        "task_revision": 1,
        "cached_subgoal_revision": 0,
        "evaluator_state_used": False,
    }


def test_stale_subgoal_risk_rejects_empty_fault() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _instruction_receipt_risk("  ")
