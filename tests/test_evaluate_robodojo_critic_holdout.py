from scripts.evaluate_robodojo_critic_holdout import _metrics


def test_metrics_treat_only_confirmed_completion_as_safe_stop() -> None:
    rows = [
        {
            "expected_complete": False,
            "protocol_valid": True,
            "predicted_safe_stop": False,
            "predicted_control": "continue",
            "control_target": "continue",
            "latency_ms": 10.0,
        },
        {
            "expected_complete": True,
            "protocol_valid": True,
            "predicted_safe_stop": True,
            "predicted_control": "safe_stop",
            "control_target": "safe_stop",
            "latency_ms": 20.0,
        },
    ]

    metrics = _metrics(rows)

    assert metrics["protocol_valid_rate"] == 1.0
    assert metrics["incomplete_false_safe_stop_rate"] == 0.0
    assert metrics["terminal_safe_stop_recall"] == 1.0
    assert metrics["control_accuracy"] == 1.0
