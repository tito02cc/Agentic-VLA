#!/usr/bin/env python3
"""Summarize the paired RoboMME raw, Agentic and optimized pilot conditions."""

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


TASK_ORDER = ("StopCube", "VideoRepick", "RouteStick", "VideoUnmaskSwap")
CONDITION_ORDER = ("B1_raw", "C2_agentic", "C3_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b1-root",
        type=Path,
        default=Path("results/robomme_b1_raw_policy_20260831"),
    )
    parser.add_argument(
        "--c2-root",
        type=Path,
        default=Path("results/robomme_c2_agentic_every_chunk_20260831"),
    )
    parser.add_argument(
        "--c3-episodes",
        type=Path,
        default=Path("results/robomme_c3_pilot_canonical_20260831/episodes.csv"),
    )
    parser.add_argument(
        "--c3-root",
        type=Path,
        help="Load C3 episode summaries directly instead of an episodes CSV.",
    )
    parser.add_argument("--b1-extension-root", type=Path, nargs="+")
    parser.add_argument("--c2-extension-root", type=Path, nargs="+")
    parser.add_argument("--c3-extension-root", type=Path, nargs="+")
    parser.add_argument("--episode-count", type=int, default=5)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_ORDER),
        help="Ordered task set expected in every paired condition.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/robomme_b1_c2_c3_paired_pilot_20260831"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_root(root: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    records = {}
    for path in sorted(root.glob("*/summary.json")):
        payload = read_json(path)
        key = (str(payload["task"]), int(payload["episode"]))
        records[key] = (path.resolve(), payload)
    return records


def load_c3(path: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    records = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summary_path = Path(row["method_summary"])
            payload = read_json(summary_path)
            key = (str(payload["task"]), int(payload["episode"]))
            records[key] = (summary_path.resolve(), payload)
    return records


def extend_records(
    records: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
    extension_roots: list[Path] | None,
) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    combined = dict(records)
    for extension_root in extension_roots or []:
        for key, value in load_root(extension_root).items():
            if key in combined:
                raise ValueError(f"duplicate paired episode in extension: {key}")
            combined[key] = value
    return combined


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(record["success"]) for record in records)
    planner_calls = sum(int(record["planner_calls"]) for record in records)
    policy_calls = sum(int(record["policy_calls"]) for record in records)
    wall_time_s = sum(float(record["wall_time_s"]) for record in records)
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "success_wilson95": wilson_interval(successes, len(records)),
        "planner_calls": planner_calls,
        "policy_calls": policy_calls,
        "wall_time_s": wall_time_s,
    }


def paired_comparison(
    left: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
    right: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    keys = sorted(left)
    deltas = [
        int(bool(right[key][1]["success"])) - int(bool(left[key][1]["success"]))
        for key in keys
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
    conditions = {
        "B1_raw": extend_records(load_root(args.b1_root), args.b1_extension_root),
        "C2_agentic": extend_records(load_root(args.c2_root), args.c2_extension_root),
        "C3_full": extend_records(
            load_root(args.c3_root) if args.c3_root is not None else load_c3(args.c3_episodes),
            args.c3_extension_root,
        ),
    }
    task_order = tuple(args.tasks)
    expected = {
        (task, episode)
        for task in task_order
        for episode in range(args.episode_count)
    }
    for name, records in conditions.items():
        if set(records) != expected:
            missing = sorted(expected - set(records))
            extra = sorted(set(records) - expected)
            raise ValueError(f"{name} has missing={missing}, extra={extra}")

    for key, (_, payload) in conditions["B1_raw"].items():
        if payload.get("planner_profile") != "raw":
            raise ValueError(f"B1 profile is not raw: {key}")
        if payload.get("task_memory") is not None:
            raise ValueError(f"B1 received task memory: {key}")
        if payload.get("planner_schedule", {}).get("mode") != "every_chunk":
            raise ValueError(f"B1 schedule is not every_chunk: {key}")
        if payload.get("temporal_monitor", {}).get("events"):
            raise ValueError(f"B1 temporal monitor produced events: {key}")
        if any(item.get("tool_grounding") for item in payload.get("planner_trace", [])):
            raise ValueError(f"B1 used a CARVE grounding tool: {key}")

    overall = {
        name: aggregate([payload for _, payload in records.values()])
        for name, records in conditions.items()
    }
    per_task = {
        task: {
            name: aggregate(
                [
                    payload
                    for (record_task, _), (_, payload) in records.items()
                    if record_task == task
                ]
            )
            for name, records in conditions.items()
        }
        for task in task_order
    }
    pairwise = {
        "B1_raw_to_C2_agentic": paired_comparison(
            conditions["B1_raw"], conditions["C2_agentic"]
        ),
        "B1_raw_to_C3_full": paired_comparison(
            conditions["B1_raw"], conditions["C3_full"]
        ),
        "C2_agentic_to_C3_full": paired_comparison(
            conditions["C2_agentic"], conditions["C3_full"]
        ),
    }
    episodes = []
    for task in task_order:
        for episode in range(args.episode_count):
            key = (task, episode)
            row: dict[str, Any] = {"task": task, "episode": episode}
            for name in CONDITION_ORDER:
                path, payload = conditions[name][key]
                row[f"{name}_success"] = bool(payload["success"])
                row[f"{name}_policy_calls"] = int(payload["policy_calls"])
                row[f"{name}_planner_calls"] = int(payload["planner_calls"])
                row[f"{name}_wall_time_s"] = float(payload["wall_time_s"])
                row[f"{name}_summary"] = str(path)
            episodes.append(row)

    report = {
        "protocol": "carve.robomme.b1_c2_c3_paired_study.v2",
        "claim_boundary": (
            f"Paired {len(expected)}-episode RoboMME mechanism study with one frozen GroundSG "
            "PI0.5 and Qwen3-VL-4B checkpoint. B1 removes CARVE memory, tools and "
            "process control; C2 enables the full Harness; C3 adds selective runtime."
        ),
        "overall": overall,
        "per_task": per_task,
        "pairwise": pairwise,
        "episodes": episodes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)

    lines = [
        "# RoboMME Paired B1/C2/C3 Study",
        "",
        report["claim_boundary"],
        "",
        "## Main Table",
        "",
        "| Condition | Success | Planner calls | Policy calls | Wall time |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "B1_raw": "B1 raw VLM+VLA",
        "C2_agentic": "C2 full Harness, every-chunk",
        "C3_full": "C3 full Harness, selective runtime",
    }
    for name in CONDITION_ORDER:
        value = overall[name]
        lines.append(
            f"| {labels[name]} | {value['successes']}/{len(expected)} ({value['success_rate']:.1%}) | "
            f"{value['planner_calls']} | {value['policy_calls']} | {value['wall_time_s']:.1f} s |"
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | B1 raw | C2 Agentic | C3 full |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in task_order:
        lines.append(
            f"| {task} | {per_task[task]['B1_raw']['successes']}/{args.episode_count} | "
            f"{per_task[task]['C2_agentic']['successes']}/{args.episode_count} | "
            f"{per_task[task]['C3_full']['successes']}/{args.episode_count} |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for name, value in pairwise.items():
        lines.append(
            f"- {name}: delta {value['success_rate_delta']:+.1%}, "
            f"transitions {value['left_fail_to_right_success']}/"
            f"{value['left_success_to_right_fail']}, bootstrap 95% CI "
            f"[{value['paired_bootstrap95'][0]:+.1%}, "
            f"{value['paired_bootstrap95'][1]:+.1%}], McNemar p="
            f"{value['mcnemar_exact_two_sided_p']:.4f}."
        )
    lines.extend(
        [
            "",
            "B1 is audited to contain no task memory, CARVE grounding tool or "
            "temporal-monitor event. This is a fixed pilot, not full-benchmark coverage.",
            "",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": overall, "pairwise": pairwise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
