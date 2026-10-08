from __future__ import annotations

from scripts.finalize_robodojo_nominal_run import (
    _summarize_agentic,
    _summarize_runtime,
)


def test_missing_runtime_trace_is_not_reported_as_zero_calls() -> None:
    summary = _summarize_runtime([], trace_available=False)

    assert summary["trace_available"] is False
    assert summary["vla_calls"] is None
    assert summary["request_level_action_seeded"] is None
    assert summary["compute_phase_calls"] is None
    assert summary["first_call_latency_ms"] is None
    assert summary["steady_state_model_latency_p95_ms"] is None
    assert summary["total_model_latency_ms"] is None
    assert summary["observed_inference_steps"] is None
    assert summary["observed_execute_horizons"] is None
    assert summary["inference_step_control_verified"] is None


def test_empty_observed_runtime_trace_remains_an_observed_zero() -> None:
    summary = _summarize_runtime([], trace_available=True)

    assert summary["trace_available"] is True
    assert summary["vla_calls"] == 0
    assert summary["request_level_action_seeded"] is False
    assert summary["compute_phase_calls"] == {}
    assert summary["first_call_latency_ms"] is None
    assert summary["steady_state_model_latency_p95_ms"] is None
    assert summary["total_model_latency_ms"] == 0
    assert summary["observed_inference_steps"] == []
    assert summary["observed_execute_horizons"] == []
    assert summary["inference_step_control_verified"] is False


def test_runtime_summary_separates_first_call_from_steady_state() -> None:
    runtime = [
        {
            "model_latency_ms": 900.0,
            "applied_controls": {"inference_steps": 6, "max_actions": 16},
            "metadata": {
                "inference_steps_parameter": "num_inference_steps",
                "optimization_profile": {"profile": {"profile_id": "fast-h16"}}
            },
        },
        {
            "model_latency_ms": 300.0,
            "applied_controls": {"inference_steps": 6, "max_actions": 16},
            "metadata": {},
        },
        {
            "model_latency_ms": 320.0,
            "applied_controls": {"inference_steps": 10, "max_actions": 16},
            "metadata": {},
        },
    ]

    summary = _summarize_runtime(runtime, trace_available=True)

    assert summary["first_call_latency_ms"] == 900.0
    assert summary["steady_state_model_latency_p50_ms"] == 300.0
    assert summary["steady_state_model_latency_p95_ms"] == 320.0
    assert summary["steady_state_model_latency_max_ms"] == 320.0
    assert summary["total_model_latency_ms"] == 1520.0
    assert summary["observed_inference_steps"] == [6, 10]
    assert summary["observed_execute_horizons"] == [16]
    assert summary["optimization_profile_ids"] == ["fast-h16"]
    assert summary["observed_inference_step_parameters"] == [
        "num_inference_steps"
    ]
    assert summary["inference_step_control_verified"] is True


def test_agentic_summary_separates_two_stage_recovery() -> None:
    planner = [
        {"timestep": 674, "trigger": "no_progress", "action": "fresh_vla_replan"},
        {
            "timestep": 675,
            "trigger": "no_progress",
            "accepted": True,
            "decision": {"intent": "vla_act"},
            "control_action": "planner_vla_replan",
        },
    ]
    monitor = [
        {"timestep": 674, "assessment": {"event": "no_progress"}},
        {"timestep": 675, "assessment": {"event": "no_progress"}},
    ]

    summary = _summarize_agentic(planner, monitor)

    assert summary["interventions"] == 2
    assert summary["low_cost_replans"] == 1
    assert summary["semantic_planner_calls"] == 1
    assert summary["accepted_semantic_interventions"] == 1
    assert summary["planner_control_actions"] == ["planner_vla_replan"]
    assert summary["first_no_progress_event_step"] == 674


def test_staged_vlm_receipts_count_startup_and_critic_without_fake_interventions():
    rows = [
        {"event": "planner_residency_receipt", "ok": True,
         "action_model_restored": True, "model_path": "/models/vlm",
         "timings": {"total_ms": 25_000, "vlm_generate_ms": 6_000}},
        {"event": "planner_residency_receipt", "ok": True,
         "action_model_restored": True, "model_path": "/models/vlm",
         "timings": {"total_ms": 18_000, "vlm_generate_ms": 1_000}},
    ]
    summary = _summarize_agentic(rows, [])
    assert summary["semantic_planner_calls"] == 0
    assert summary["interventions"] == 0
    staged = summary["staged_vlm"]
    assert staged["trace_available"] is True
    assert staged["rpc_calls"] == 2
    assert staged["successful_rpc_calls"] == 2
    assert staged["action_model_restored_calls"] == 2
    assert staged["total_residency_and_generation_ms"] == 43_000
    assert staged["total_generation_ms"] == 7_000


def test_absent_staged_receipts_are_unknown_not_zero_calls():
    staged = _summarize_agentic([], [])["staged_vlm"]
    assert staged["trace_available"] is False
    assert staged["rpc_calls"] is None
    assert staged["total_residency_and_generation_ms"] is None


def test_failed_staged_receipt_does_not_invent_missing_latency():
    staged = _summarize_agentic([
        {"event": "planner_residency_receipt", "ok": False,
         "error": "load failed", "timings": None},
    ], [])["staged_vlm"]
    assert staged["rpc_calls"] == 1
    assert staged["successful_rpc_calls"] == 0
    assert staged["action_model_restored_calls"] == 0
    assert staged["total_residency_and_generation_ms"] is None
