#!/usr/bin/env python3
"""Summarize a diagnostic RoboMME online-oracle capability survey."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


TASK_SUITES = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Persistent": ["ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "Referential": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Behavior": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
TASK_TO_SUITE = {
    task: suite for suite, tasks in TASK_SUITES.items() for task in tasks
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def resolve_video(root: Path, value: str) -> Path:
    if value.startswith("/workspace/"):
        project_root = root.parents[1]
        return project_root / value.removeprefix("/workspace/")
    path = Path(value)
    return path if path.is_absolute() else root.parents[1] / path


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    expected_path = root / "expected_tasks.txt"
    expected = (
        [line.strip() for line in expected_path.read_text().splitlines() if line.strip()]
        if expected_path.exists()
        else []
    )

    episodes: list[dict[str, object]] = []
    for path in sorted(root.glob("*_ep*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task = str(row["task"])
        video = resolve_video(root, str(row.get("video_path", "")))
        episodes.append(
            {
                "task": task,
                "suite": TASK_TO_SUITE.get(task, "Unknown"),
                "episode": int(row["episode"]),
                "success": bool(row["success"]),
                "executed_steps": int(row.get("executed_steps", 0)),
                "policy_calls": int(row.get("policy_calls", 0)),
                "wall_time_s": float(row.get("wall_time_s", 0.0)),
                "unique_subgoal_count": len(row.get("unique_subgoals", [])),
                "video": str(video),
                "video_exists": video.is_file() and video.stat().st_size > 0,
            }
        )

    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        by_task[str(row["task"])].append(row)

    task_rows = []
    task_order = expected or sorted(by_task)
    for task in task_order:
        rows = by_task.get(task, [])
        task_rows.append(
            {
                "task": task,
                "suite": TASK_TO_SUITE.get(task, "Unknown"),
                "episodes": len(rows),
                "successes": sum(bool(row["success"]) for row in rows),
                "success_rate": (
                    sum(bool(row["success"]) for row in rows) / len(rows) if rows else None
                ),
                "mean_steps": (
                    statistics.fmean(int(row["executed_steps"]) for row in rows)
                    if rows
                    else None
                ),
                "mean_wall_time_s": (
                    statistics.fmean(float(row["wall_time_s"]) for row in rows)
                    if rows
                    else None
                ),
                "videos_valid": sum(bool(row["video_exists"]) for row in rows),
            }
        )

    completed_tasks = {str(row["task"]) for row in episodes}
    summary = {
        "purpose": "diagnostic online-oracle capability gate; not a deployable CARVE result",
        "privileged_online_subgoal_used": True,
        "expected_tasks": expected,
        "completed_tasks": sorted(completed_tasks),
        "missing_tasks": [task for task in expected if task not in completed_tasks],
        "episode_count": len(episodes),
        "successes": sum(bool(row["success"]) for row in episodes),
        "success_rate": (
            sum(bool(row["success"]) for row in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "valid_videos": sum(bool(row["video_exists"]) for row in episodes),
        "tasks": task_rows,
        "episodes": episodes,
    }
    (root / "survey_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# RoboMME Online-Oracle Capability Survey",
        "",
        "> Diagnostic gate only. It uses privileged online grounded subgoals and is not a deployable CARVE result.",
        "",
        f"- Episodes: {len(episodes)}",
        f"- Successes: {summary['successes']}/{len(episodes)}",
        f"- Valid videos: {summary['valid_videos']}/{len(episodes)}",
        f"- Missing tasks: {', '.join(summary['missing_tasks']) or 'none'}",
        "",
        "| Suite | Task | Success | Mean steps | Mean wall time (s) | Video |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        rate = "-" if row["success_rate"] is None else f"{100 * row['success_rate']:.1f}%"
        steps = "-" if row["mean_steps"] is None else f"{row['mean_steps']:.1f}"
        wall = "-" if row["mean_wall_time_s"] is None else f"{row['mean_wall_time_s']:.1f}"
        lines.append(
            f"| {row['suite']} | {row['task']} | {row['successes']}/{row['episodes']} ({rate}) "
            f"| {steps} | {wall} | {row['videos_valid']}/{row['episodes']} |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("episode_count", "successes", "missing_tasks", "valid_videos")}, indent=2))


if __name__ == "__main__":
    main()
