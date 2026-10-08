#!/usr/bin/env python3
"""Plot the paired RoboMME B1/C2/C3 success and efficiency results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("B1_raw", "C2_agentic", "C3_full")
LABELS = ("Raw VLM+VLA", "Agentic Harness", "Harness + Runtime")
COLORS = ("#4C78A8", "#F28E2B", "#2A9D8F")
TASK_LABELS = {
    "StopCube": "Stop",
    "VideoRepick": "Repick",
    "RouteStick": "Route",
    "VideoUnmaskSwap": "Unmask",
    "MoveCube": "Move",
    "PickHighlight": "Highlight",
    "ButtonUnmask": "Button",
    "BinFill": "BinFill",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    overall = report["overall"]
    per_task = report["per_task"]
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 4, figsize=(16.2, 3.45))

    rates = np.asarray([overall[name]["success_rate"] for name in CONDITIONS])
    intervals = np.asarray([overall[name]["success_wilson95"] for name in CONDITIONS])
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    axes[0].bar(LABELS, rates * 100, color=COLORS, width=0.68)
    axes[0].errorbar(
        np.arange(3), rates * 100, yerr=errors * 100, fmt="none", color="#222222",
        capsize=3, linewidth=1.1,
    )
    for index, value in enumerate(rates):
        axes[0].text(index, value * 100 + 2.0, f"{value:.1%}", ha="center", weight="bold")
    axes[0].set_ylim(0, 80)
    axes[0].set_ylabel("Task success (%)")
    paired_episodes = overall[CONDITIONS[0]]["episodes"]
    axes[0].set_title(
        f"(a) Overall success (n={paired_episodes})", loc="left", weight="bold"
    )
    axes[0].tick_params(axis="x", rotation=22)

    tasks = [task for task in TASK_LABELS if task in per_task]
    x = np.arange(len(tasks))
    width = 0.24
    for index, (condition, label, color) in enumerate(zip(CONDITIONS, LABELS, COLORS)):
        values = [per_task[task][condition]["success_rate"] * 100 for task in tasks]
        axes[1].bar(x + (index - 1) * width, values, width, label=label, color=color)
    axes[1].set_xticks(x, [TASK_LABELS[task] for task in tasks])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Task success (%)")
    axes[1].set_title("(b) Success by capability", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].tick_params(axis="x", labelsize=7, rotation=25)

    planner_calls = [overall[name]["planner_calls"] for name in CONDITIONS]
    axes[2].bar(LABELS, planner_calls, color=COLORS, width=0.68)
    for index, value in enumerate(planner_calls):
        axes[2].text(index, value + 9, str(value), ha="center", weight="bold")
    axes[2].set_ylim(0, max(planner_calls) * 1.22)
    axes[2].set_ylabel("Total Planner calls")
    axes[2].set_title("(c) Semantic planning cost", loc="left", weight="bold")
    axes[2].tick_params(axis="x", rotation=22)

    wall = np.asarray(
        [overall[name]["wall_time_s"] / overall[name]["episodes"] for name in CONDITIONS]
    )
    axes[3].bar(LABELS, wall, color=COLORS, width=0.68)
    for index, value in enumerate(wall):
        axes[3].text(index, value + 0.8, f"{value:.1f}s", ha="center", weight="bold")
    axes[3].set_ylim(0, max(wall) * 1.22)
    axes[3].set_ylabel("Mean wall time / episode (s)")
    axes[3].set_title("(d) End-to-end runtime", loc="left", weight="bold")
    axes[3].tick_params(axis="x", rotation=22)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)
    figure.suptitle(
        "CARVE-VLA paired RoboMME study: Agentic capability and runtime efficiency",
        fontsize=11, weight="bold", y=1.02,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
