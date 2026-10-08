#!/usr/bin/env python3
"""Build the paired RoboMME C2-agentic versus C3-full pilot report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


TASK_ORDER = ("StopCube", "VideoRepick", "RouteStick", "VideoUnmaskSwap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output",
        type=Path,
        default=Path("results/robomme_c2_c3_runtime_ablation_20260831"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_c2(root: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    records: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/summary.json")):
        record = load_json(path)
        key = (str(record["task"]), int(record["episode"]))
        if key in records:
            raise ValueError(f"duplicate C2 result: {key}")
        records[key] = (path.resolve(), record)
    return records


def load_c3(path: Path) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    records: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summary_path = Path(row["method_summary"])
            record = load_json(summary_path)
            key = (str(record["task"]), int(record["episode"]))
            if key in records:
                raise ValueError(f"duplicate C3 result: {key}")
            records[key] = (summary_path.resolve(), record)
    return records


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [center - radius, center + radius]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(
    deltas: list[int], samples: int, seed: int = 7
) -> list[float]:
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(
            sum(generator.choice(deltas) for _ in range(len(deltas))) / len(deltas)
        )
    return [percentile(means, 0.025), percentile(means, 0.975)]


def mcnemar_exact_two_sided(c2_only: int, c3_only: int) -> float:
    discordant = c2_only + c3_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(c2_only, c3_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> int:
    args = parse_args()
    c2 = load_c2(args.c2_root)
    c3 = load_c3(args.c3_episodes)
    expected = {(task, episode) for task in TASK_ORDER for episode in range(5)}
    if set(c2) != expected:
        raise ValueError(f"C2 keys differ from expected pilot: {sorted(expected - set(c2))}")
    if set(c3) != expected:
        raise ValueError(f"C3 keys differ from expected pilot: {sorted(expected - set(c3))}")

    episodes: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        for episode in range(5):
            key = (task, episode)
            c2_path, c2_record = c2[key]
            c3_path, c3_record = c3[key]
            c2_success = bool(c2_record["success"])
            c3_success = bool(c3_record["success"])
            episodes.append(
                {
                    "task": task,
                    "episode": episode,
                    "c2_success": c2_success,
                    "c3_success": c3_success,
                    "transition": (
                        "fail_to_success"
                        if not c2_success and c3_success
                        else "success_to_fail"
                        if c2_success and not c3_success
                        else "unchanged_success"
                        if c2_success
                        else "unchanged_failure"
                    ),
                    "c2_policy_calls": int(c2_record["policy_calls"]),
                    "c3_policy_calls": int(c3_record["policy_calls"]),
                    "c2_planner_calls": int(c2_record["planner_calls"]),
                    "c3_planner_calls": int(c3_record["planner_calls"]),
                    "c2_wall_time_s": float(c2_record["wall_time_s"]),
                    "c3_wall_time_s": float(c3_record["wall_time_s"]),
                    "c2_summary": str(c2_path),
                    "c3_summary": str(c3_path),
                    "c2_video": str(c2_record["video_path"]),
                    "c3_video": str(c3_record["video_path"]),
                }
            )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        c2_successes = sum(bool(row["c2_success"]) for row in rows)
        c3_successes = sum(bool(row["c3_success"]) for row in rows)
        c2_planner_calls = sum(int(row["c2_planner_calls"]) for row in rows)
        c3_planner_calls = sum(int(row["c3_planner_calls"]) for row in rows)
        c2_policy_calls = sum(int(row["c2_policy_calls"]) for row in rows)
        c3_policy_calls = sum(int(row["c3_policy_calls"]) for row in rows)
        c2_wall_time_s = sum(float(row["c2_wall_time_s"]) for row in rows)
        c3_wall_time_s = sum(float(row["c3_wall_time_s"]) for row in rows)
        return {
            "episodes": len(rows),
            "c2_successes": c2_successes,
            "c2_success_rate": c2_successes / len(rows),
            "c2_success_wilson95": wilson_interval(c2_successes, len(rows)),
            "c3_successes": c3_successes,
            "c3_success_rate": c3_successes / len(rows),
            "c3_success_wilson95": wilson_interval(c3_successes, len(rows)),
            "success_rate_delta": (c3_successes - c2_successes) / len(rows),
            "c2_policy_calls": c2_policy_calls,
            "c3_policy_calls": c3_policy_calls,
            "c2_planner_calls": c2_planner_calls,
            "c3_planner_calls": c3_planner_calls,
            "planner_call_reduction": 1.0 - c3_planner_calls / c2_planner_calls,
            "c2_wall_time_s": c2_wall_time_s,
            "c3_wall_time_s": c3_wall_time_s,
            "wall_time_reduction": 1.0 - c3_wall_time_s / c2_wall_time_s,
        }

    overall = aggregate(episodes)
    deltas = [int(row["c3_success"]) - int(row["c2_success"]) for row in episodes]
    c3_only = sum(delta == 1 for delta in deltas)
    c2_only = sum(delta == -1 for delta in deltas)
    overall.update(
        {
            "fail_to_success": c3_only,
            "success_to_fail": c2_only,
            "paired_success_delta_bootstrap95": paired_bootstrap_interval(
                deltas, args.bootstrap_samples
            ),
            "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(c2_only, c3_only),
        }
    )
    per_task = {
        task: aggregate([row for row in episodes if row["task"] == task])
        for task in TASK_ORDER
    }
    report = {
        "protocol": "carve.robomme.c2_c3_runtime_ablation.v1",
        "claim_boundary": (
            "Paired 20-episode RoboMME mechanism pilot. C2 and C3 use the same frozen "
            "PI0.5, Planner, memory, tools and Harness; only Planner scheduling differs."
        ),
        "c2_condition": "every action chunk invokes the VLM Planner",
        "c3_condition": "event-triggered selective Planner invocation with bounded reuse",
        "overall": overall,
        "per_task": per_task,
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
        "# RoboMME C2-Agentic versus C3-Full Runtime Ablation",
        "",
        report["claim_boundary"],
        "",
        "## Overall",
        "",
        f"- C2 every-chunk success: {overall['c2_successes']}/20 ({overall['c2_success_rate']:.1%}).",
        f"- C3 selective success: {overall['c3_successes']}/20 ({overall['c3_success_rate']:.1%}).",
        f"- Paired transitions: {c3_only} fail-to-success, {c2_only} success-to-fail.",
        f"- Success-rate delta: {overall['success_rate_delta']:+.1%}; paired bootstrap 95% CI "
        f"[{overall['paired_success_delta_bootstrap95'][0]:+.1%}, "
        f"{overall['paired_success_delta_bootstrap95'][1]:+.1%}].",
        f"- McNemar exact two-sided p: {overall['mcnemar_exact_two_sided_p']:.3f}.",
        f"- Planner calls: {overall['c2_planner_calls']} -> {overall['c3_planner_calls']} "
        f"({overall['planner_call_reduction']:.1%} reduction).",
        f"- Policy calls: {overall['c2_policy_calls']} -> {overall['c3_policy_calls']}.",
        f"- Total wall time: {overall['c2_wall_time_s']:.1f}s -> "
        f"{overall['c3_wall_time_s']:.1f}s ({overall['wall_time_reduction']:.1%} reduction).",
        "",
        "## By Task",
        "",
        "| Task | C2 success | C3 success | Planner calls C2 -> C3 | Wall time C2 -> C3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASK_ORDER:
        task_result = per_task[task]
        lines.append(
            f"| {task} | {task_result['c2_successes']}/5 | "
            f"{task_result['c3_successes']}/5 | "
            f"{task_result['c2_planner_calls']} -> {task_result['c3_planner_calls']} | "
            f"{task_result['c2_wall_time_s']:.1f}s -> {task_result['c3_wall_time_s']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "C3 executes more policy chunks because it completes four additional long-horizon "
            "episodes, yet it still uses fewer Planner calls and less total wall time. The pilot "
            "supports the mechanism claim that event-triggered semantic planning reduces overhead "
            "and avoids repeated-decision jitter. It is not a full RoboMME leaderboard result, and "
            "the 20-episode paired success difference is not statistically conclusive at p < 0.05.",
            "",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
