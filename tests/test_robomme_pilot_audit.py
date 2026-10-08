import hashlib
import json

import pytest

from scripts.audit_robomme_feedback_pilot import audit, latency_stats


def test_latency_summary_does_not_treat_missing_samples_as_zero():
    assert latency_stats([]) == {"count": 0, "mean": None, "p50": None, "p95": None}
    stats = latency_stats([1, 3, 5])
    assert stats["count"] == 3
    assert stats["mean"] == stats["p50"] == 3
    assert stats["p95"] == pytest.approx(4.8)


@pytest.mark.parametrize("complete", [False, True])
def test_complete_batch_requires_registered_run_coverage(tmp_path, complete):
    content = b"# frozen runner\n"
    (tmp_path / "runner_snapshot.py").write_bytes(content)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "runner_sha256": hashlib.sha256(content).hexdigest(),
        "commands": {"ep0_condition": []}, "claim_boundary": "development only",
    }))
    (tmp_path / "summary.json").write_text(json.dumps({
        "complete": complete, "failure": None if complete else "stopped before rollout",
    }))
    if complete:
        with pytest.raises(ValueError, match="registered runs"):
            audit(tmp_path)
    else:
        result = audit(tmp_path)
        assert not result["complete"]
        assert result["audited_runs"] == 0
        assert result["expected_runs"] == 1
