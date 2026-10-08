#!/usr/bin/env python3
"""Create paper-ready plots for the full paired LIBERO-Pro study."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("frozen_vla", "fixed_recovery", "agentic")
LABELS = ("Frozen VLA", "Fixed Recovery", "Full Agentic")
COLORS = ("#4B5563", "#0F766E", "#B45309")
SUITE_LABELS = {
    "libero_10": "Standard",
    "libero_10_object": "Object",
    "libero_10_swap": "Position",
    "libero_10_task": "Task logic",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    suites = list(SUITE_LABELS)
    rates = np.asarray(
        [[100.0 * data["suites"][suite][method]["success_rate"] for method in METHODS]
         for suite in suites]
    )
    totals = data["aggregate"]
    reference_calls = totals["frozen_vla"]["vla_calls"]
    reference_wall = totals["frozen_vla"]["episode_wall_s"]
    efficiency = np.asarray(
        [
            [100.0 * totals[m]["vla_calls"] / reference_calls for m in METHODS],
            [100.0 * totals[m]["episode_wall_s"] / reference_wall for m in METHODS],
        ]
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)

    x = np.arange(len(suites))
    width = 0.24
    for index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        bars = axes[0].bar(
            x + (index - 1) * width,
            rates[:, index],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[0].bar_label(bars, fmt="%.0f", padding=1, fontsize=7)
    axes[0].set_title("Task success across paired suites")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_xticks(x, [SUITE_LABELS[s] for s in suites])
    axes[0].set_ylim(0, 108)
    axes[0].grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")

    ex = np.arange(2)
    for index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        bars = axes[1].bar(
            ex + (index - 1) * width,
            efficiency[:, index],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].bar_label(bars, fmt="%.1f", padding=1, fontsize=7)
    axes[1].axhline(100.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[1].set_title("Execution cost relative to Frozen VLA")
    axes[1].set_ylabel("Relative cost (%)")
    axes[1].set_xticks(ex, ["VLA calls", "Episode wall time"])
    axes[1].set_ylim(0, 130)
    axes[1].grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[1].set_axisbelow(True)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"libero_pro_full_study.{suffix}", bbox_inches="tight")
    plt.close(fig)

    with (args.output_root / "task_success.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "task_id", *METHODS])
        for task in data["tasks"]:
            writer.writerow(
                [task["suite"], task["task_id"], *[
                    task["methods"][method]["successes"] for method in METHODS
                ]]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
