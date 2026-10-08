#!/usr/bin/env python3
"""Summarize paired RoboDojo controlled-fault runs with protocol checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    p = successes / trials
    scale = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / scale
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / scale
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar_exact_two_sided(c1_only: int, c3_only: int) -> float:
    discordant = c1_only + c3_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(c1_only, c3_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "summary.json"
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    payload["_summary_path"] = str(resolved)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_pair(c1: dict[str, Any], c3: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("benchmark", "task", "layout_id", "seed", "policy"):
        if c1.get(field) != c3.get(field):
            errors.append(f"pair mismatch for {field}: {c1.get(field)!r} != {c3.get(field)!r}")
    for label, row in (("C1", c1), ("C3", c3)):
        fault = row.get("controlled_fault", {})
        if fault.get("enabled") is not True or fault.get("injected_at_step") != 640:
            errors.append(f"{label} does not contain the frozen step-640 fault")
        runtime = row.get("runtime", {})
        if runtime.get("trace_available") is not True:
            errors.append(f"{label} is missing runtime trace evidence")
        if runtime.get("inference_step_control_verified") is not True:
            errors.append(f"{label} did not verify native inference-step control")
    c1_runtime = c1.get("runtime", {})
    if c1_runtime.get("observed_inference_steps") != [2]:
        errors.append("C1 must contain only the Flow2 fast path")
    if c1_runtime.get("compute_phase_calls", {}).get("recovery_boost", 0) != 0:
        errors.append("C1 unexpectedly contains recovery compute")
    c3_runtime = c3.get("runtime", {})
    if c3_runtime.get("observed_inference_steps") != [2, 4]:
        errors.append("C3 must contain both Flow2 and Flow4")
    if c3_runtime.get("compute_phase_calls", {}).get("recovery_boost") != 2:
        errors.append("C3 must contain exactly two recovery-boost calls")
    if c3.get("controlled_fault", {}).get("first_no_progress_event_step") is None:
        errors.append("C3 has no recorded no-progress event")
    if c3.get("agentic", {}).get("interventions", 0) < 1:
        errors.append("C3 has no recorded Agentic intervention")
    c3_agentic = c3.get("agentic", {})
    if c3_agentic.get("low_cost_replans") != 1:
        errors.append("C3 must contain exactly one low-cost fresh-chunk replan")
    if c3_agentic.get("semantic_planner_calls") != 1:
        errors.append("C3 must contain exactly one semantic Planner call")
    if c3_agentic.get("accepted_semantic_interventions") != 1:
        errors.append("C3 must contain exactly one accepted semantic intervention")
    ticket_path = Path(c3["_summary_path"]).parent / "planner_replay_ticket.json"
    if not ticket_path.is_file():
        errors.append("C3 is missing its frozen Planner replay ticket")
    return errors


def summarize(c1_paths: list[Path], c3_paths: list[Path]) -> dict[str, Any]:
    if len(c1_paths) != len(c3_paths) or not c1_paths:
        raise ValueError("C1 and C3 require the same non-zero number of runs")

    def index_by_layout_set(paths: list[Path]) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for path in paths:
            row = _load(path)
            layout_set = int(row["seed"])
            if layout_set in rows:
                raise ValueError(f"duplicate layout set: {layout_set}")
            rows[layout_set] = row
        return rows

    c1_rows = index_by_layout_set(c1_paths)
    c3_rows = index_by_layout_set(c3_paths)
    if set(c1_rows) != set(c3_rows):
        raise ValueError("C1 and C3 layout sets do not match")

    pairs = []
    for layout_set in sorted(c1_rows):
        c1 = c1_rows[layout_set]
        c3 = c3_rows[layout_set]
        errors = _validate_pair(c1, c3)
        ticket_path = Path(c3["_summary_path"]).parent / "planner_replay_ticket.json"
        pairs.append(
            {
                "layout_set": layout_set,
                "layout_id_within_set": int(c1["layout_id"]),
                "seed": c1["seed"],
                "valid": not errors,
                "validation_errors": errors,
                "c1_success": bool(c1["official_result"]["success"]),
                "c3_success": bool(c3["official_result"]["success"]),
                "c1_video_frames": int(c1["official_result"]["video_frames"]),
                "c3_video_frames": int(c3["official_result"]["video_frames"]),
                "no_progress_step": c3["controlled_fault"]["first_no_progress_event_step"],
                "c3_compute_phase_calls": c3["runtime"]["compute_phase_calls"],
                "c3_low_cost_replans": c3["agentic"]["low_cost_replans"],
                "c3_semantic_planner_calls": c3["agentic"]["semantic_planner_calls"],
                "c3_accepted_semantic_interventions": c3["agentic"][
                    "accepted_semantic_interventions"
                ],
                "planner_replay_ticket_sha256": (
                    _sha256(ticket_path) if ticket_path.is_file() else None
                ),
                "c1_summary": c1["_summary_path"],
                "c3_summary": c3["_summary_path"],
            }
        )
    invalid = [row for row in pairs if not row["valid"]]
    if invalid:
        raise ValueError(f"controlled-fault protocol validation failed: {invalid}")
    ticket_hashes = {row["planner_replay_ticket_sha256"] for row in pairs}
    if len(ticket_hashes) != 1:
        raise ValueError(f"C3 replay tickets do not match: {sorted(ticket_hashes)}")

    trials = len(pairs)
    c1_successes = sum(row["c1_success"] for row in pairs)
    c3_successes = sum(row["c3_success"] for row in pairs)
    c1_only = sum(row["c1_success"] and not row["c3_success"] for row in pairs)
    c3_only = sum(row["c3_success"] and not row["c1_success"] for row in pairs)
    return {
        "schema_version": "carve.robodojo.controlled-fault-pairs.v1",
        "benchmark": pairs and c1_rows[pairs[0]["layout_set"]]["benchmark"],
        "task": pairs and c1_rows[pairs[0]["layout_set"]]["task"],
        "policy": pairs and c1_rows[pairs[0]["layout_set"]]["policy"],
        "protocol": {
            "fault": "stale_action_hold",
            "fault_step": 640,
            "nominal_profile": "Flow2/h16",
            "recovery_profile": "Flow4/h16",
            "recovery_calls": 2,
            "semantic_decision": "fixed accepted VLM replay ticket",
            "planner_replay_ticket_sha256": next(iter(ticket_hashes)),
        },
        "pairs": pairs,
        "aggregate": {
            "pairs": trials,
            "c1_successes": c1_successes,
            "c1_success_rate": c1_successes / trials,
            "c1_success_wilson95": _wilson_interval(c1_successes, trials),
            "c3_successes": c3_successes,
            "c3_success_rate": c3_successes / trials,
            "c3_success_wilson95": _wilson_interval(c3_successes, trials),
            "failure_to_success": c3_only,
            "success_to_failure": c1_only,
            "paired_success_rate_delta": (c3_successes - c1_successes) / trials,
            "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(c1_only, c3_only),
        },
        "claim_boundary": (
            "Controlled-fault mechanism evidence only. The three official layout sets "
            "do not establish natural-task gains or statistical significance."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# RoboDojo controlled-fault paired result",
        "",
        "| Layout | C1 | C3 | No-progress step | C3 compute calls |",
        "|---:|---:|---:|---:|---|",
    ]
    for pair in summary["pairs"]:
        phases = pair["c3_compute_phase_calls"]
        rows.append(
            f"| {pair['layout_set']} | {'success' if pair['c1_success'] else 'fail'} | "
            f"{'success' if pair['c3_success'] else 'fail'} | {pair['no_progress_step']} | "
            f"Flow2={phases.get('nominal_fast_path', 0)}, "
            f"Flow4={phases.get('recovery_boost', 0)} |"
        )
    aggregate = summary["aggregate"]
    rows.extend(
        [
            "",
            f"C1: {aggregate['c1_successes']}/{aggregate['pairs']}; "
            f"C3: {aggregate['c3_successes']}/{aggregate['pairs']}.",
            f"Failure-to-success: {aggregate['failure_to_success']}; "
            f"success-to-failure: {aggregate['success_to_failure']}.",
            f"Exact McNemar two-sided p={aggregate['mcnemar_exact_two_sided_p']:.3f}.",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1", type=Path, nargs="+", required=True)
    parser.add_argument("--c3", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.c1, args.c3)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
