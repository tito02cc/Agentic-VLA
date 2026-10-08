#!/usr/bin/env python3
"""Build the canonical four-suite RoboMME CARVE pilot report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = RESULT_ROOT / "robomme_c3_pilot_canonical_20260831"

METHOD_RESULTS = {
    **{
        ("StopCube", episode): path
        for episode, path in {
            0: "robomme_c3_pilot_4suite_v2_20260831/StopCube_ep0/summary.json",
            1: "robomme_c3_pilot_4suite_v2_20260831/StopCube_ep1/summary.json",
            2: "robomme_stopcube_ep2_gate_20260831/StopCube_ep2/summary.json",
            3: "robomme_stopcube_ep34_gate_20260831/StopCube_ep3/summary.json",
            4: "robomme_stopcube_ep34_gate_20260831/StopCube_ep4/summary.json",
        }.items()
    },
    **{
        ("VideoRepick", episode): path
        for episode, path in {
            0: "robomme_c3_videorepick_event_v8_20260831/VideoRepick_ep0/summary.json",
            1: "robomme_c3_videorepick_event_v8_20260831/VideoRepick_ep1/summary.json",
            2: "robomme_c3_videorepick_event_v8_20260831/VideoRepick_ep2/summary.json",
            3: "robomme_c3_videorepick_event_v10_ep3_20260831/VideoRepick_ep3/summary.json",
            4: "robomme_c3_videorepick_event_v9_ep4_20260831/VideoRepick_ep4/summary.json",
        }.items()
    },
    **{
        ("RouteStick", episode):
        f"robomme_c3_routestick_trajectory_v1_20260831/RouteStick_ep{episode}/summary.json"
        for episode in range(5)
    },
    **{
        ("VideoUnmaskSwap", episode):
        f"robomme_c3_unmask_swap_event_v6_20260831/VideoUnmaskSwap_ep{episode}/summary.json"
        for episode in range(5)
    },
}

ORACLE_DIAGNOSTICS = {
    ("StopCube", 1): "robomme_stopcube_oracle_timing_ep1_diagnostic_20260831/summary.json",
    ("StopCube", 3): "robomme_stopcube_oracle_timing_ep3_diagnostic_20260831/summary.json",
    ("StopCube", 4): "robomme_stopcube_oracle_timing_ep4_diagnostic_20260831/summary.json",
    ("VideoRepick", 3): "robomme_videorepick_oracle_ep3_diagnostic_20260831/summary.json",
    ("RouteStick", 2): "robomme_routestick_oracle_diagnostic_20260831/RouteStick_ep2/summary.json",
    ("RouteStick", 3): "robomme_routestick_oracle_diagnostic_20260831/RouteStick_ep3/summary.json",
}


def read(path: str) -> dict[str, Any]:
    source = RESULT_ROOT / path
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def local_video(path: str | None) -> Path | None:
    if not path:
        return None
    if path.startswith("/workspace/"):
        return PROJECT_ROOT / path.removeprefix("/workspace/")
    return Path(path)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for key, result_path in METHOD_RESULTS.items():
        task, episode = key
        payload = read(result_path)
        if bool(payload.get("privileged_online_subgoal_used")):
            raise ValueError(f"canonical method result is privileged: {result_path}")
        oracle_path = ORACLE_DIAGNOSTICS.get(key)
        oracle = read(oracle_path) if oracle_path else None
        success = bool(payload.get("success"))
        if success:
            classification = "deployable_success"
        elif oracle is not None and not bool(oracle.get("success")):
            classification = "checkpoint_upper_bound_failure"
        elif oracle is not None and bool(oracle.get("success")):
            classification = "agentic_gap"
        else:
            classification = "unresolved_failure"
        video = local_video(payload.get("video_path"))
        rows.append(
            {
                "task": task,
                "episode": episode,
                "success": success,
                "classification": classification,
                "executed_steps": payload.get("executed_steps"),
                "policy_calls": payload.get("policy_calls"),
                "planner_calls": payload.get("planner_calls"),
                "planner_call_reduction_fraction": payload.get(
                    "planner_call_reduction_fraction"
                ),
                "method_summary": str(RESULT_ROOT / result_path),
                "video": str(video) if video else None,
                "video_exists": bool(video and video.is_file()),
                "oracle_diagnostic_summary": (
                    str(RESULT_ROOT / oracle_path) if oracle_path else None
                ),
                "oracle_success": oracle.get("success") if oracle else None,
                "privileged_online_subgoal_used": False,
            }
        )

    per_task: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    for task, values in grouped.items():
        reductions = [
            float(row["planner_call_reduction_fraction"])
            for row in values
            if row["planner_call_reduction_fraction"] is not None
        ]
        per_task[task] = {
            "successes": sum(bool(row["success"]) for row in values),
            "episodes": len(values),
            "success_rate": sum(bool(row["success"]) for row in values) / len(values),
            "mean_planner_call_reduction_fraction": (
                sum(reductions) / len(reductions) if reductions else None
            ),
        }

    ceiling_failures = sum(
        row["classification"] == "checkpoint_upper_bound_failure" for row in rows
    )
    eligible = [
        row
        for row in rows
        if row["classification"] != "checkpoint_upper_bound_failure"
    ]
    summary = {
        "protocol": "carve.robomme.c3_pilot.canonical.v1",
        "claim_boundary": (
            "All canonical method episodes are deployable and oracle-free. Oracle "
            "runs are separate diagnostics used only to classify checkpoint ceilings."
        ),
        "overall": {
            "successes": sum(bool(row["success"]) for row in rows),
            "episodes": len(rows),
            "raw_success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "checkpoint_upper_bound_failures": ceiling_failures,
            "ceiling_eligible_successes": sum(bool(row["success"]) for row in eligible),
            "ceiling_eligible_episodes": len(eligible),
            "ceiling_eligible_success_rate": (
                sum(bool(row["success"]) for row in eligible) / len(eligible)
            ),
            "agentic_gaps": sum(row["classification"] == "agentic_gap" for row in rows),
            "videos_present": sum(bool(row["video_exists"]) for row in rows),
        },
        "per_task": per_task,
        "episodes": rows,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT_ROOT / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RoboMME C3 Pilot Canonical Results",
        "",
        summary["claim_boundary"],
        "",
        "## Overall",
        "",
        f"- Raw deployable success: {summary['overall']['successes']}/20 "
        f"({summary['overall']['raw_success_rate']:.1%}).",
        f"- PI0.5 checkpoint upper-bound failures: {ceiling_failures}/20.",
        f"- Success on ceiling-eligible episodes: "
        f"{summary['overall']['ceiling_eligible_successes']}/"
        f"{summary['overall']['ceiling_eligible_episodes']} "
        f"({summary['overall']['ceiling_eligible_success_rate']:.1%}).",
        f"- Remaining Agentic gap: {summary['overall']['agentic_gaps']} episode.",
        f"- Method videos present: {summary['overall']['videos_present']}/20.",
        "",
        "## By Task",
        "",
        "| Task | Success | Mean Planner-call reduction |",
        "|---|---:|---:|",
    ]
    for task in ("StopCube", "VideoRepick", "RouteStick", "VideoUnmaskSwap"):
        value = per_task[task]
        lines.append(
            f"| {task} | {value['successes']}/{value['episodes']} | "
            f"{value['mean_planner_call_reduction_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Failure Attribution",
            "",
            "| Task | Episode | Classification | Oracle diagnostic |",
            "|---|---:|---|---:|",
        ]
    )
    for row in rows:
        if row["success"]:
            continue
        oracle_value = row["oracle_success"]
        lines.append(
            f"| {row['task']} | {row['episode']} | {row['classification']} | "
            f"{oracle_value} |"
        )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
