#!/usr/bin/env python3
"""Aggregate a pre-registered RoboDojo B0/C1/C2/C3 condition matrix.

The existing RoboDojo summarizers each handle exactly two conditions:
``summarize_robodojo_optimize_ablation`` requires a B0/C1 pair and
``summarize_robodojo_controlled_fault_pairs`` handles C1/C3. Nothing aggregated
the four-condition natural-task matrix, so this fills that gap.

Episodes are paired on ``(task, layout_set, layout_id)``. Independent Isaac
processes make byte-identical trajectory replay unverifiable, so this reports
same-layout pairing rather than claiming strict per-trajectory pairing.

Statistics reuse the RoboMME helpers verbatim so the two benchmarks stay
comparable: exact McNemar, paired bootstrap, and Wilson intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_robomme_c2_c3_ablation import (  # noqa: E402
    mcnemar_exact_two_sided,
    paired_bootstrap_interval,
    wilson_interval,
)

CANONICAL_CONDITIONS = ("B0", "C1", "C2", "C3")
NOMINAL_SCHEMAS = {"carve.robodojo.nominal.v1", "carve.robodojo.nominal.v2"}
# Ordered so each comparison isolates one added mechanism.
COMPARISONS = (
    ("B0", "C1"),  # Optimize Runtime alone
    ("B0", "C2"),  # Agentic Harness alone
    ("B0", "C3"),  # both
    ("C2", "C3"),  # Optimize Runtime on top of Agentic
    ("C1", "C3"),  # Agentic on top of Optimize
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema not in NOMINAL_SCHEMAS:
        raise ValueError(
            f"{path}: expected a nominal RoboDojo summary, got schema {schema!r}. "
            "Controlled-fault and ablation summaries are not matrix inputs."
        )
    condition = str(payload.get("condition", ""))
    if condition not in CANONICAL_CONDITIONS:
        raise ValueError(
            f"{path}: condition {condition!r} is not one of {CANONICAL_CONDITIONS}. "
            "Tuned variants must not be pooled into a pre-registered matrix."
        )
    payload["_summary_path"] = str(path)
    return payload


def _episode_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one run into per-episode rows keyed for pairing."""
    task = summary["task"]
    condition = summary["condition"]
    layout_set = summary.get("layout_set", summary.get("seed"))
    source = summary["_summary_path"]

    episodes = summary.get("episodes")
    if not episodes:
        # A v1 summary predating the multi-episode finalizer.
        official = summary.get("official_result") or {}
        episodes = [
            {
                "episode_index": 0,
                "layout_id": summary.get("layout_id"),
                "success": bool(official.get("success")),
                "score": official.get("score"),
                "video_frames": official.get("video_frames"),
                "runtime": summary.get("runtime") or {},
                "agentic": summary.get("agentic") or {},
            }
        ]

    rows = []
    for episode in episodes:
        runtime = episode.get("runtime") or {}
        agentic = episode.get("agentic") or {}
        rows.append(
            {
                "task": task,
                "condition": condition,
                "layout_set": layout_set,
                "layout_id": episode.get("layout_id"),
                "episode_index": episode.get("episode_index"),
                "success": bool(episode.get("success")),
                "score": episode.get("score"),
                "video_frames": episode.get("video_frames"),
                "vla_calls": runtime.get("vla_calls"),
                "trace_available": runtime.get("trace_available"),
                "model_latency_p50_ms": runtime.get("model_latency_p50_ms"),
                "model_latency_p95_ms": runtime.get("model_latency_p95_ms"),
                "first_call_latency_ms": runtime.get("first_call_latency_ms"),
                "steady_state_p50_ms": runtime.get("steady_state_model_latency_p50_ms"),
                "steady_state_p95_ms": runtime.get("steady_state_model_latency_p95_ms"),
                "deadline_misses": runtime.get("deadline_misses"),
                "observed_inference_steps": runtime.get("observed_inference_steps"),
                "observed_execute_horizons": runtime.get("observed_execute_horizons"),
                "inference_step_control_verified": runtime.get(
                    "inference_step_control_verified"
                ),
                "semantic_checks": agentic.get("semantic_checks"),
                "interventions": agentic.get("interventions"),
                "low_cost_replans": agentic.get("low_cost_replans"),
                "semantic_planner_calls": agentic.get("semantic_planner_calls"),
                "accepted_semantic_interventions": agentic.get(
                    "accepted_semantic_interventions"
                ),
                "monitor_events": agentic.get("monitor_events"),
                "first_no_progress_event_step": agentic.get(
                    "first_no_progress_event_step"
                ),
                "source_summary": source,
            }
        )
    return rows


def _condition_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(row["success"] for row in rows)
    trials = len(rows)
    traced = [row for row in rows if row["trace_available"]]
    vla_calls = [row["vla_calls"] for row in traced if row["vla_calls"] is not None]
    frames = [row["video_frames"] for row in rows if row["video_frames"] is not None]
    misses = [row["deadline_misses"] for row in traced if row["deadline_misses"] is not None]
    steps = sorted({
        step
        for row in traced
        for step in (row["observed_inference_steps"] or [])
    })
    horizons = sorted({
        horizon
        for row in traced
        for horizon in (row["observed_execute_horizons"] or [])
    })
    unverified = [
        row["layout_id"]
        for row in traced
        if row["inference_step_control_verified"] is False
    ]
    return {
        "episodes": trials,
        "successes": successes,
        "success_rate": successes / trials if trials else None,
        "success_wilson95": wilson_interval(successes, trials),
        "episodes_with_runtime_trace": len(traced),
        "vla_calls_total": sum(vla_calls) if vla_calls else None,
        "vla_calls_mean": _mean([float(value) for value in vla_calls]),
        "first_call_latency_mean_ms": _mean(
            [row["first_call_latency_ms"] for row in traced if row["first_call_latency_ms"] is not None]
        ),
        "steady_state_p50_mean_ms": _mean(
            [row["steady_state_p50_ms"] for row in traced if row["steady_state_p50_ms"] is not None]
        ),
        "steady_state_p95_mean_ms": _mean(
            [row["steady_state_p95_ms"] for row in traced if row["steady_state_p95_ms"] is not None]
        ),
        "deadline_misses_total": sum(misses) if misses else None,
        "control_steps_total": sum(frames) if frames else None,
        "control_steps_mean": _mean([float(value) for value in frames]),
        "observed_inference_steps": steps,
        "observed_execute_horizons": horizons,
        "layout_ids_with_unverified_step_control": unverified,
        "semantic_checks_total": sum(row["semantic_checks"] or 0 for row in rows),
        "interventions_total": sum(row["interventions"] or 0 for row in rows),
        "low_cost_replans_total": sum(row["low_cost_replans"] or 0 for row in rows),
        "semantic_planner_calls_total": sum(row["semantic_planner_calls"] or 0 for row in rows),
        "accepted_semantic_interventions_total": sum(
            row["accepted_semantic_interventions"] or 0 for row in rows
        ),
        "monitor_events_total": sum(row["monitor_events"] or 0 for row in rows),
        "episodes_with_no_progress_event": sum(
            row["first_no_progress_event_step"] is not None for row in rows
        ),
    }


def _compare(
    by_key: dict[tuple, dict[str, dict[str, Any]]], left: str, right: str
) -> dict[str, Any] | None:
    """Compare two conditions over layouts where both actually ran."""
    shared = sorted(
        key for key, cells in by_key.items() if left in cells and right in cells
    )
    if not shared:
        return None

    deltas: list[int] = []
    left_only = 0
    right_only = 0
    both = 0
    neither = 0
    conversions: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for key in shared:
        left_success = by_key[key][left]["success"]
        right_success = by_key[key][right]["success"]
        deltas.append(int(right_success) - int(left_success))
        if left_success and right_success:
            both += 1
        elif left_success and not right_success:
            left_only += 1
            regressions.append(
                {"task": key[0], "layout_set": key[1], "layout_id": key[2]}
            )
        elif right_success and not left_success:
            right_only += 1
            conversions.append(
                {"task": key[0], "layout_set": key[1], "layout_id": key[2]}
            )
        else:
            neither += 1

    left_successes = sum(by_key[key][left]["success"] for key in shared)
    right_successes = sum(by_key[key][right]["success"] for key in shared)
    return {
        "paired_episodes": len(shared),
        "left_successes": left_successes,
        "right_successes": right_successes,
        "success_rate_delta": (right_successes - left_successes) / len(shared),
        "both_success": both,
        "neither_success": neither,
        "left_success_to_right_fail": left_only,
        "right_success_to_left_fail": right_only,
        "fail_to_success": right_only,
        "success_to_fail": left_only,
        "paired_bootstrap95": paired_bootstrap_interval(deltas, 100_000),
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(left_only, right_only),
        "discordant_pairs": left_only + right_only,
        "conversion_layouts": conversions,
        "regression_layouts": regressions,
    }


def _pairing_audit(by_key: dict[tuple, dict[str, dict[str, Any]]], conditions: list[str]) -> dict[str, Any]:
    complete = [key for key, cells in by_key.items() if all(c in cells for c in conditions)]
    partial = {
        f"{key[0]}/set{key[1]}/layout{key[2]}": sorted(set(conditions) - set(cells))
        for key, cells in by_key.items()
        if not all(c in cells for c in conditions)
    }
    return {
        "pairing_key": "task + layout_set + layout_id",
        "layouts_with_all_conditions": len(complete),
        "layouts_missing_conditions": partial,
        "strict_trajectory_pairing_verified": False,
        "pairing_note": (
            "Independent Isaac Sim processes are not bit-reproducible, so this is "
            "same-layout pairing with pre-registered action seeds, not strict "
            "per-trajectory pairing."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a RoboDojo B0/C1/C2/C3 matrix into one paired summary."
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Run artifact directory or summary.json. Repeat for every run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-csv", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="Frozen protocol document recorded in the summary by SHA256.",
    )
    parser.add_argument(
        "--expected-episodes-per-condition",
        type=int,
        help="Pre-registered episode count; a mismatch is recorded, not silently accepted.",
    )
    parser.add_argument(
        "--allow-duplicate-cells",
        action="store_true",
        help="Keep the first episode for a repeated condition/layout cell instead of failing.",
    )
    args = parser.parse_args()

    summaries = [_load_summary(path) for path in args.run]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(_episode_rows(summary))
    if not rows:
        raise ValueError("no episodes found across the provided runs")

    tasks = sorted({row["task"] for row in rows})
    conditions = [c for c in CANONICAL_CONDITIONS if any(r["condition"] == c for r in rows)]

    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = (row["task"], row["layout_set"], row["layout_id"])
        cell = by_key.setdefault(key, {})
        if row["condition"] in cell:
            duplicates.append(
                f"{row['task']}/set{row['layout_set']}/layout{row['layout_id']}/{row['condition']}"
            )
            continue
        cell[row["condition"]] = row

    if duplicates and not args.allow_duplicate_cells:
        raise ValueError(
            f"{len(duplicates)} duplicate condition/layout cells: "
            f"{duplicates[:5]}{' ...' if len(duplicates) > 5 else ''}. A "
            "pre-registered matrix has one episode per condition and layout. "
            "Pass --allow-duplicate-cells to keep the first of each and record "
            "the rest as skipped."
        )

    # Aggregate from the de-duplicated cells so the per-condition denominators
    # match the ones the paired comparisons use.
    rows = [row for cell in by_key.values() for row in cell.values()]
    overall = {
        condition: _condition_aggregate([r for r in rows if r["condition"] == condition])
        for condition in conditions
    }
    per_task = {
        task: {
            condition: _condition_aggregate(
                [r for r in rows if r["task"] == task and r["condition"] == condition]
            )
            for condition in conditions
            if any(r["task"] == task and r["condition"] == condition for r in rows)
        }
        for task in tasks
    }

    pairwise = {}
    for left, right in COMPARISONS:
        if left in conditions and right in conditions:
            comparison = _compare(by_key, left, right)
            if comparison is not None:
                pairwise[f"{left}_to_{right}"] = comparison

    protocol: dict[str, Any] = {
        "conditions_present": conditions,
        "expected_episodes_per_condition": args.expected_episodes_per_condition,
        "episode_count_matches_preregistration": (
            None
            if args.expected_episodes_per_condition is None
            else {
                condition: overall[condition]["episodes"]
                == args.expected_episodes_per_condition
                for condition in conditions
            }
        ),
        "duplicate_condition_layout_cells": duplicates,
        "source_runs": [summary["_summary_path"] for summary in summaries],
    }
    if args.preregistration is not None:
        import hashlib

        digest = hashlib.sha256(args.preregistration.read_bytes()).hexdigest()
        protocol["preregistration_path"] = str(args.preregistration)
        protocol["preregistration_sha256"] = digest

    payload = {
        "schema_version": "carve.robodojo.condition-matrix.v1",
        "benchmark": "RoboDojo",
        "policy": "StarVLA PI-v3",
        "tasks": tasks,
        "protocol": protocol,
        "pairing_audit": _pairing_audit(by_key, conditions),
        "overall": overall,
        "per_task": per_task,
        "pairwise": pairwise,
        "claim_boundary": (
            "Same-layout paired comparison across pre-registered RoboDojo layouts. "
            "Success labels come from the official evaluator after each episode "
            "ends; no online trigger reads reward, success, object pose or "
            "simulator private state. Strict per-trajectory pairing is not "
            "claimed. Latency figures are single-model measurements, not a "
            "robot-level hard real-time guarantee."
        ),
        "episodes": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")

    if args.episodes_csv is not None:
        fields = [key for key in rows[0] if not key.startswith("_")]
        args.episodes_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.episodes_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {args.episodes_csv}")

    print()
    print(f"{'condition':<10}{'succ/trials':<14}{'rate':<9}{'wilson95'}")
    for condition in conditions:
        cell = overall[condition]
        low, high = cell["success_wilson95"]
        print(
            f"{condition:<10}{cell['successes']}/{cell['episodes']:<11}"
            f"{cell['success_rate']:.1%}    [{low:.1%}, {high:.1%}]"
        )
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicate condition/layout cells were skipped")
    print()
    for name, comparison in pairwise.items():
        print(
            f"{name}: delta {comparison['success_rate_delta']:+.1%} over "
            f"{comparison['paired_episodes']} paired episodes, "
            f"{comparison['fail_to_success']} gained / {comparison['success_to_fail']} lost, "
            f"McNemar p={comparison['mcnemar_exact_two_sided_p']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
