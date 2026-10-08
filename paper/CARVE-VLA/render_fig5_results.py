#!/usr/bin/env python3
"""Render the retained LIBERO-10 comparison used by the CARVE-VLA paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "generated" / "fig5_libero10_data.csv"
OUTPUT = ROOT / "figures" / "fig5_libero10_task_comparison_v2.png"


def main() -> None:
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    tasks = [row["task"] for row in rows]
    baseline = np.asarray([float(row["pi05_libero"]) for row in rows])
    carve = np.asarray([float(row["carve_vla"]) for row in rows])
    delta = carve - baseline

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.0, 5.25), dpi=220, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.12)

    left = fig.add_subplot(grid[0, 0])
    x = np.arange(len(tasks))
    width = 0.36
    left.bar(
        x - width / 2,
        baseline,
        width,
        color="#9ca3af",
        label=r"$\pi_{0.5}$ baseline (180/200)",
    )
    left.bar(
        x + width / 2,
        carve,
        width,
        color="#2563eb",
        label="CARVE-VLA (185/200)",
    )
    left.set_xticks(x, tasks)
    left.set_ylim(0, 105)
    left.set_ylabel("Success rate (%)")
    left.set_title("LIBERO-10 task-level comparison", weight="bold", pad=9)
    left.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    left.set_axisbelow(True)
    left.legend(frameon=False, loc="lower left")
    left.axvspan(7.52, 8.48, color="#fee2e2", alpha=0.5, zorder=0)
    left.annotate(
        "dominant weak task: 55% -> 75%",
        xy=(8, 75),
        xytext=(6.65, 52),
        fontsize=9,
        color="#991b1b",
        arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 1.0},
    )

    right = fig.add_subplot(grid[0, 1])
    colors = ["#16a34a" if value > 0 else "#dc2626" if value < 0 else "#9ca3af" for value in delta]
    bars = right.barh(x, delta, color=colors, height=0.65)
    right.axvline(0, color="#374151", linewidth=1.1)
    right.set_yticks(x, tasks)
    right.invert_yaxis()
    right.set_xlim(-7, 23)
    right.set_xlabel("Change from baseline (percentage points)")
    right.set_title("Effect is concentrated, not uniform", weight="bold", pad=9)
    right.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.3)
    right.set_axisbelow(True)
    for bar, value in zip(bars, delta, strict=True):
        if value == 0:
            xpos, align = 0.6, "left"
        elif value > 0:
            xpos, align = value + 0.5, "left"
        else:
            xpos, align = value - 0.5, "right"
        right.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.0f}" if value else "0",
            va="center",
            ha=align,
            fontsize=9,
            color="#111827",
        )
    right.text(
        0.02,
        -0.14,
        "Green: improvement   Red: regression   Gray: unchanged",
        transform=right.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.08)
    print(OUTPUT)


if __name__ == "__main__":
    main()
