from __future__ import annotations

from scripts.summarize_carve_ral_memory_breadth import (
    _aggregate,
    _arm_directories,
    _mcnemar_exact,
    _percentile,
    _stratified_bootstrap_reduction,
)


def test_arm_directories_are_protocol_suffix_aware() -> None:
    assert _arm_directories("a5") == {
        "direct_9b": "direct9_a5",
        "memory_9b": "memory9_a5",
        "memory_4b": "memory4_a5",
    }
    assert _arm_directories("_a5") == _arm_directories("a5")


def test_aggregate_reports_failure_and_plan_consistency() -> None:
    base = {
        "episode_steps": 100,
        "wall_time_s": 10.0,
        "planner_requests": 1,
        "startup_planner_attempts": 1,
        "event_planner_attempts": 0,
        "planner_latency_ms": 1000.0,
        "critic_calls": 1,
        "critic_latency_ms": 500.0,
        "semantic_latency_ms": 1500.0,
        "semantic_retries": 0,
        "recovery_attempts": 0,
        "vla_calls": 10,
        "vla_latency_mean_ms": 50.0,
        "vla_latency_p95_ms": 55.0,
        "vla_latency_max_ms": 60.0,
        "vla_deadline_misses": 0,
        "superseded_planner_decisions": 0,
    }
    rows = [
        {
            **base,
            "success": True,
            "status": "success",
            "task_plan_completed": True,
            "procedure_warm_start_id": "procedure-1",
        },
        {
            **base,
            "success": False,
            "status": "safe_stop",
            "task_plan_completed": True,
            "procedure_warm_start_id": None,
        },
    ]

    summary = _aggregate(rows)

    assert summary["status_counts"] == {"safe_stop": 1, "success": 1}
    assert summary["procedure_warm_starts"] == 1
    assert summary["plan_evaluator_mismatches"] == 1


def test_aggregate_allows_fail_closed_cell_without_vla_calls() -> None:
    row = {
        "success": False,
        "status": "safe_stop",
        "task_plan_completed": False,
        "procedure_warm_start_id": None,
        "episode_steps": 0,
        "wall_time_s": 2.0,
        "planner_requests": 1,
        "startup_planner_attempts": 2,
        "event_planner_attempts": 0,
        "planner_latency_ms": 2000.0,
        "critic_calls": 0,
        "critic_latency_ms": 0.0,
        "semantic_latency_ms": 2000.0,
        "semantic_retries": 0,
        "recovery_attempts": 0,
        "vla_calls": 0,
        "vla_latency_mean_ms": 0.0,
        "vla_latency_p95_ms": 0.0,
        "vla_latency_max_ms": 0.0,
        "vla_deadline_misses": 0,
        "superseded_planner_decisions": 0,
    }

    assert _aggregate([row])["vla_latency_mean_ms"] is None


def test_mcnemar_exact_handles_balanced_and_one_sided_pairs() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(1, 1) == 1.0
    assert _mcnemar_exact(0, 6) == 0.03125


def test_percentile_uses_nearest_rank() -> None:
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([], 0.95) == 0.0


def test_stratified_bootstrap_is_deterministic_and_task_balanced() -> None:
    reference = [
        {"task_id": task, "trial": trial, "wall_time_s": 10.0}
        for task in (0, 8)
        for trial in (1, 2, 3)
    ]
    candidate = [
        {"task_id": task, "trial": trial, "wall_time_s": 5.0}
        for task in (0, 8)
        for trial in (1, 2, 3)
    ]

    assert _stratified_bootstrap_reduction(
        reference, candidate, samples=100, seed=7
    ) == (50.0, 50.0)
