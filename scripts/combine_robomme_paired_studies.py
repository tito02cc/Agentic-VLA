#!/usr/bin/env python3
"""Combine audited RoboMME paired-study summaries without rewriting rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from summarize_robomme_c2_c3_ablation import (
    mcnemar_exact_two_sided,
    paired_bootstrap_interval,
    wilson_interval,
)


CONDITIONS = ("B1_raw", "C2_agentic", "C3_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "carve.robomme.b1_c2_c3_paired_study.v2":
        raise ValueError(f"unsupported paired-study protocol: {path}")
    return payload


def aggregate(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    successes = sum(bool(row[f"{condition}_success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_wilson95": wilson_interval(successes, len(rows)),
        "planner_calls": sum(int(row[f"{condition}_planner_calls"]) for row in rows),
        "policy_calls": sum(int(row[f"{condition}_policy_calls"]) for row in rows),
        "wall_time_s": sum(float(row[f"{condition}_wall_time_s"]) for row in rows),
    }


def compare(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    deltas = [
        int(bool(row[f"{right}_success"])) - int(bool(row[f"{left}_success"]))
        for row in rows
    ]
    right_only = sum(delta == 1 for delta in deltas)
    left_only = sum(delta == -1 for delta in deltas)
    return {
        "success_rate_delta": sum(deltas) / len(deltas),
        "left_fail_to_right_success": right_only,
        "left_success_to_right_fail": left_only,
        "paired_bootstrap95": paired_bootstrap_interval(deltas, 100_000),
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(left_only, right_only),
    }


def main() -> int:
    args = parse_args()
    studies = [(path.resolve(), load(path)) for path in args.studies]
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    task_order: list[str] = []
    for study_path, study in studies:
        for row in study["episodes"]:
            task = str(row["task"])
            key = (task, int(row["episode"]))
            if key in rows_by_key:
                raise ValueError(f"duplicate paired episode {key} from {study_path}")
            rows_by_key[key] = dict(row)
            if task not in task_order:
                task_order.append(task)
    rows = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda x: (task_order.index(x[0]), x[1]))]
    if not rows:
        raise ValueError("no paired episodes to combine")

    overall = {condition: aggregate(rows, condition) for condition in CONDITIONS}
    per_task = {
        task: {
            condition: aggregate(
                [row for row in rows if row["task"] == task], condition
            )
            for condition in CONDITIONS
        }
        for task in task_order
    }
    pairwise = {
        "B1_raw_to_C2_agentic": compare(rows, "B1_raw", "C2_agentic"),
        "B1_raw_to_C3_full": compare(rows, "B1_raw", "C3_full"),
        "C2_agentic_to_C3_full": compare(rows, "C2_agentic", "C3_full"),
    }
    report = {
        "protocol": "carve.robomme.combined_paired_studies.v1",
        "claim_boundary": (
            f"Combined {len(rows)} paired RoboMME episodes from {len(studies)} audited "
            "studies using one frozen GroundSG PI0.5 and Qwen3-VL-4B checkpoint."
        ),
        "source_studies": [str(path) for path, _ in studies],
        "overall": overall,
        "per_task": per_task,
        "pairwise": pairwise,
        "episodes": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "B1_raw": "B1 raw VLM+VLA",
        "C2_agentic": "C2 full Harness, every-chunk",
        "C3_full": "C3 full Harness, selective runtime",
    }
    lines = [
        "# Combined RoboMME Paired Study",
        "",
        report["claim_boundary"],
        "",
        "| Condition | Success | Planner calls | Policy calls | Wall time |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        value = overall[condition]
        lines.append(
            f"| {labels[condition]} | {value['successes']}/{value['episodes']} "
            f"({value['success_rate']:.1%}) | {value['planner_calls']} | "
            f"{value['policy_calls']} | {value['wall_time_s']:.1f} s |"
        )
    lines.extend(["", "## By Task", "", "| Task | B1 | C2 | C3 |", "|---|---:|---:|---:|"])
    for task in task_order:
        count = per_task[task]["B1_raw"]["episodes"]
        lines.append(
            f"| {task} | {per_task[task]['B1_raw']['successes']}/{count} | "
            f"{per_task[task]['C2_agentic']['successes']}/{count} | "
            f"{per_task[task]['C3_full']['successes']}/{count} |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for name, value in pairwise.items():
        lines.append(
            f"- {name}: delta {value['success_rate_delta']:+.1%}, transitions "
            f"{value['left_fail_to_right_success']}/{value['left_success_to_right_fail']}, "
            f"bootstrap 95% CI [{value['paired_bootstrap95'][0]:+.1%}, "
            f"{value['paired_bootstrap95'][1]:+.1%}], McNemar p="
            f"{value['mcnemar_exact_two_sided_p']:.6f}."
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "pairwise": pairwise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
