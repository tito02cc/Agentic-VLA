"""Regression tests for the RoboDojo B0/C1/C2/C3 matrix aggregator.

The aggregator is the only place the four-condition natural-task comparison is
computed, so these tests pin the properties that keep it from manufacturing a
result: correct pairing, honest denominators, and refusal to pool inputs that
were never part of the pre-registered matrix.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "combine_robodojo_condition_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location("combine_robodojo_condition_matrix", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


def _episode(layout_id: int, success: bool, *, traced: bool, steps: int, agentic: bool) -> dict:
    return {
        "episode_index": layout_id,
        "layout_id": layout_id,
        "success": success,
        "score": 1.0 if success else 0.0,
        "video_frames": 420 if success else 801,
        "runtime": {
            "trace_available": traced,
            "vla_calls": (26 if success else 50) if traced else None,
            "model_latency_p50_ms": 300.0 if traced else None,
            "model_latency_p95_ms": 900.0 if traced else None,
            "first_call_latency_ms": 900.0 if traced else None,
            "steady_state_model_latency_p50_ms": (235.0 if steps == 2 else 307.0) if traced else None,
            "steady_state_model_latency_p95_ms": (265.0 if steps == 2 else 315.0) if traced else None,
            "deadline_misses": 0 if traced else None,
            "observed_inference_steps": [steps] if traced else None,
            "observed_execute_horizons": [16] if traced else None,
            "inference_step_control_verified": True if traced else None,
        },
        "agentic": {
            "semantic_checks": 2 if agentic else 0,
            "interventions": 1 if (agentic and not success) else 0,
            "low_cost_replans": 1 if agentic else 0,
            "semantic_planner_calls": 1 if agentic else 0,
            "accepted_semantic_interventions": 1 if (agentic and success) else 0,
            "monitor_events": 3 if agentic else 0,
            "first_no_progress_event_step": 320 if (agentic and not success) else None,
        },
    }


def _write_run(
    root: Path,
    condition: str,
    flags: list[int],
    *,
    task: str = "stack_bowls",
    layout_set: int = 0,
    schema: str = "carve.robodojo.nominal.v2",
) -> Path:
    traced = condition != "B0"
    steps = 2 if condition in ("C1", "C3") else 4
    agentic = condition in ("C2", "C3")
    episodes = [
        _episode(layout, bool(flag), traced=traced, steps=steps, agentic=agentic)
        for layout, flag in enumerate(flags)
    ]
    successes = sum(flags)
    run = root / f"{task}_{condition}_set{layout_set}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "condition": condition,
                "benchmark": "RoboDojo",
                "task": task,
                "layout_id": 0,
                "seed": layout_set,
                "layout_set": layout_set,
                "policy": "StarVLA PI-v3",
                "runtime": {"trace_available": traced},
                "agentic": {},
                "official_result": {
                    "success": None,
                    "successes": successes,
                    "trials": len(flags),
                    "success_rate": successes / len(flags),
                },
                "aggregate": {"episodes": len(flags), "successes": successes},
                "episodes": episodes,
            }
        )
    )
    return run


def _run(tmp_path: Path, runs: list[Path], *extra: str) -> dict:
    import sys

    output = tmp_path / "matrix.json"
    argv = sys.argv
    sys.argv = ["combine_robodojo_condition_matrix.py"]
    for run in runs:
        sys.argv += ["--run", str(run)]
    sys.argv += ["--output", str(output), *extra]
    try:
        matrix.main()
    finally:
        sys.argv = argv
    return json.loads(output.read_text())


class TestInputValidation:
    def test_rejects_tuned_variant_condition(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path, "C2", [1, 0])
        payload = json.loads((run / "summary.json").read_text())
        payload["condition"] = "C2_deadline_gated"
        (run / "summary.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="not one of"):
            matrix._load_summary(run / "summary.json")

    def test_rejects_non_nominal_schema(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path, "C3", [1], schema="carve.robodojo.controlled-fault.v2")
        with pytest.raises(ValueError, match="expected a nominal"):
            matrix._load_summary(run / "summary.json")

    def test_accepts_legacy_v1_single_episode(self, tmp_path: Path) -> None:
        run = tmp_path / "legacy"
        run.mkdir()
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "carve.robodojo.nominal.v1",
                    "condition": "B0",
                    "task": "stack_bowls",
                    "layout_id": 3,
                    "seed": 1,
                    "runtime": {"trace_available": False},
                    "agentic": {},
                    "official_result": {"success": True, "score": 1.0, "video_frames": 400},
                }
            )
        )
        rows = matrix._episode_rows(matrix._load_summary(run / "summary.json"))
        assert len(rows) == 1
        assert rows[0]["layout_id"] == 3
        assert rows[0]["layout_set"] == 1
        assert rows[0]["success"] is True


class TestPairingAndDenominators:
    def test_four_condition_matrix(self, tmp_path: Path) -> None:
        runs = [
            _write_run(tmp_path, "B0", [0, 0, 0, 1, 0, 0]),
            _write_run(tmp_path, "C1", [0, 0, 1, 1, 0, 0]),
            _write_run(tmp_path, "C2", [1, 0, 1, 1, 1, 0]),
            _write_run(tmp_path, "C3", [1, 1, 1, 1, 1, 0]),
        ]
        payload = _run(tmp_path, runs, "--expected-episodes-per-condition", "6")

        assert payload["overall"]["B0"]["successes"] == 1
        assert payload["overall"]["C3"]["successes"] == 5
        assert all(
            payload["overall"][c]["episodes"] == 6 for c in ("B0", "C1", "C2", "C3")
        )
        assert payload["pairing_audit"]["layouts_with_all_conditions"] == 6
        assert payload["pairing_audit"]["strict_trajectory_pairing_verified"] is False

        b0_c3 = payload["pairwise"]["B0_to_C3"]
        assert b0_c3["fail_to_success"] == 4
        assert b0_c3["success_to_fail"] == 0
        assert b0_c3["paired_episodes"] == 6
        # Exact McNemar with 0 vs 4 discordant pairs.
        assert b0_c3["mcnemar_exact_two_sided_p"] == pytest.approx(0.125)

    def test_baseline_without_trace_reports_none_not_zero(self, tmp_path: Path) -> None:
        runs = [_write_run(tmp_path, "B0", [0, 1]), _write_run(tmp_path, "C1", [1, 1])]
        payload = _run(tmp_path, runs)
        b0 = payload["overall"]["B0"]
        assert b0["episodes_with_runtime_trace"] == 0
        assert b0["vla_calls_total"] is None
        assert b0["deadline_misses_total"] is None
        # Control steps come from video frames, which a baseline still has.
        assert b0["control_steps_total"] == 801 + 420

    def test_only_shared_layouts_are_compared(self, tmp_path: Path) -> None:
        runs = [
            _write_run(tmp_path, "B0", [0, 0, 0, 0]),
            _write_run(tmp_path, "C3", [1, 1]),
        ]
        payload = _run(tmp_path, runs, "--expected-episodes-per-condition", "4")
        assert payload["pairwise"]["B0_to_C3"]["paired_episodes"] == 2
        assert payload["pairing_audit"]["layouts_with_all_conditions"] == 2
        missing = payload["pairing_audit"]["layouts_missing_conditions"]
        assert set(missing) == {"stack_bowls/set0/layout2", "stack_bowls/set0/layout3"}
        matches = payload["protocol"]["episode_count_matches_preregistration"]
        assert matches == {"B0": True, "C3": False}

    def test_layouts_from_different_sets_are_not_paired(self, tmp_path: Path) -> None:
        runs = [
            _write_run(tmp_path, "B0", [0, 0], layout_set=0),
            _write_run(tmp_path, "C3", [1, 1], layout_set=1),
        ]
        payload = _run(tmp_path, runs)
        assert "B0_to_C3" not in payload["pairwise"]
        assert payload["pairing_audit"]["layouts_with_all_conditions"] == 0


class TestDuplicateCells:
    def test_duplicates_fail_by_default(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path, "B0", [0, 1])
        with pytest.raises(ValueError, match="duplicate condition/layout cells"):
            _run(tmp_path, [run, run])

    def test_allow_flag_keeps_one_and_denominators_stay_consistent(self, tmp_path: Path) -> None:
        b0 = _write_run(tmp_path, "B0", [0, 1])
        c3 = _write_run(tmp_path, "C3", [1, 1])
        payload = _run(tmp_path, [b0, b0, c3], "--allow-duplicate-cells")
        assert payload["overall"]["B0"]["episodes"] == 2
        assert payload["overall"]["B0"]["successes"] == 1
        assert len(payload["protocol"]["duplicate_condition_layout_cells"]) == 2
        assert payload["pairwise"]["B0_to_C3"]["paired_episodes"] == 2


class TestClaimBoundary:
    def test_records_preregistration_hash(self, tmp_path: Path) -> None:
        prereg = tmp_path / "prereg.md"
        prereg.write_text("frozen protocol\n")
        runs = [_write_run(tmp_path, "B0", [0]), _write_run(tmp_path, "C3", [1])]
        payload = _run(tmp_path, runs, "--preregistration", str(prereg))
        import hashlib

        assert payload["protocol"]["preregistration_sha256"] == hashlib.sha256(
            prereg.read_bytes()
        ).hexdigest()

    def test_claim_boundary_disclaims_strict_pairing_and_realtime(self, tmp_path: Path) -> None:
        runs = [_write_run(tmp_path, "B0", [0]), _write_run(tmp_path, "C3", [1])]
        payload = _run(tmp_path, runs)
        boundary = payload["claim_boundary"]
        assert "Strict per-trajectory pairing is not" in boundary
        assert "hard real-time" in boundary
        assert "after each episode" in boundary
