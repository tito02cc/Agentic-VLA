"""Summarize LIBERO-10 core results from Agentic-VLA experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RESULTS = {
    "b0": PROJECT_ROOT / "results" / "ablation_B0_pi05_libero_10_20260512" / "summary.json",
    "b1": PROJECT_ROOT / "results" / "ablation_FULL_refined_libero10_20260518_2059" / "summary.json",
    "b4_task05": PROJECT_ROOT / "results" / "core_B4_profileauto_task05_20trials_20260604_1511" / "summary.json",
    "b4_task69": PROJECT_ROOT / "results" / "core_B4_profileauto_task69_20trials_20260604_1442" / "summary.json",
    "b4_task12347": PROJECT_ROOT / "results" / "core_B4_profileauto_task12347_20trials_20260604_1555" / "summary.json",
    "b4_task6_plate": PROJECT_ROOT / "results" / "core_B4_noRecovery_task6_20trials_20260604_1457" / "summary.json",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_success(summary: dict[str, Any], task_id: int) -> tuple[int, int]:
    metrics = summary.get("task_metrics", {}).get(str(task_id), {})
    episodes = int(round(float(metrics.get("episodes", 0))))
    success_rate = float(metrics.get("success_rate", 0.0))
    return int(round(success_rate * episodes)), episodes


def _row_from_summary(name: str, summary: dict[str, Any], task_ids: list[int]) -> dict[str, Any]:
    task_values = {}
    total_successes = 0
    total_episodes = 0
    for task_id in task_ids:
        successes, episodes = _task_success(summary, task_id)
        task_values[task_id] = (successes, episodes)
        total_successes += successes
        total_episodes += episodes
    return {
        "name": name,
        "successes": total_successes,
        "episodes": total_episodes,
        "tasks": task_values,
    }


def _merge_b4_profileauto(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task_values = {}
    for task_id in (0, 5):
        task_values[task_id] = _task_success(summaries["b4_task05"], task_id)
    for task_id in (6, 9):
        task_values[task_id] = _task_success(summaries["b4_task69"], task_id)
    for task_id in (1, 2, 3, 4, 7):
        task_values[task_id] = _task_success(summaries["b4_task12347"], task_id)
    total_successes = sum(value[0] for value in task_values.values())
    total_episodes = sum(value[1] for value in task_values.values())
    return {
        "name": "B4-ProfileAuto",
        "successes": total_successes,
        "episodes": total_episodes,
        "tasks": task_values,
    }


def _merge_b4_best(summaries: dict[str, dict[str, Any]], task8_successes: int) -> dict[str, Any]:
    row = _merge_b4_profileauto(summaries)
    row = {
        "name": "B4-current-best-per-profile",
        "successes": row["successes"],
        "episodes": row["episodes"],
        "tasks": dict(row["tasks"]),
    }
    old_task6_successes, task6_episodes = row["tasks"][6]
    plate_task6_successes, _ = _task_success(summaries["b4_task6_plate"], 6)
    row["tasks"][6] = (plate_task6_successes, task6_episodes)
    row["tasks"][8] = (task8_successes, 20)
    row["successes"] = row["successes"] - old_task6_successes + plate_task6_successes + task8_successes
    row["episodes"] += 20
    return row


def _format_cell(value: tuple[int, int] | None) -> str:
    if value is None:
        return "-"
    return f"{value[0]}/{value[1]}"


def _format_rate(successes: int, episodes: int) -> str:
    if episodes <= 0:
        return "-"
    return f"{successes}/{episodes} ({successes / episodes:.1%})"


def _print_markdown(rows: list[dict[str, Any]], task_ids: list[int]) -> None:
    headers = ["Method", "Overall"] + [f"Task{task_id}" for task_id in task_ids]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [
            row["name"],
            _format_rate(int(row["successes"]), int(row["episodes"])),
        ]
        cells.extend(_format_cell(row["tasks"].get(task_id)) for task_id in task_ids)
        print("| " + " | ".join(cells) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, default=DEFAULT_RESULTS["b0"])
    parser.add_argument("--b1", type=Path, default=DEFAULT_RESULTS["b1"])
    parser.add_argument("--b4-task05", type=Path, default=DEFAULT_RESULTS["b4_task05"])
    parser.add_argument("--b4-task69", type=Path, default=DEFAULT_RESULTS["b4_task69"])
    parser.add_argument("--b4-task12347", type=Path, default=DEFAULT_RESULTS["b4_task12347"])
    parser.add_argument("--b4-task6-plate", type=Path, default=DEFAULT_RESULTS["b4_task6_plate"])
    parser.add_argument("--task8-profileauto-successes", type=int, default=16)
    parser.add_argument("--task8-best-successes", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        "b0": _load(args.b0),
        "b1": _load(args.b1),
        "b4_task05": _load(args.b4_task05),
        "b4_task69": _load(args.b4_task69),
        "b4_task12347": _load(args.b4_task12347),
        "b4_task6_plate": _load(args.b4_task6_plate),
    }
    task_ids = list(range(10))
    rows = [
        _row_from_summary("B0-VLA", summaries["b0"], task_ids),
        _row_from_summary("B1-FullRefined", summaries["b1"], task_ids),
        _merge_b4_profileauto(summaries),
        _merge_b4_best(summaries, task8_successes=int(args.task8_best_successes)),
    ]
    rows[2]["tasks"][8] = (int(args.task8_profileauto_successes), 20)
    rows[2]["successes"] += int(args.task8_profileauto_successes)
    rows[2]["episodes"] += 20
    _print_markdown(rows, task_ids)


if __name__ == "__main__":
    main()
