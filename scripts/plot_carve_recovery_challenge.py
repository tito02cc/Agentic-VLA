#!/usr/bin/env python3
"""Plot paired outcomes and PI0.5 inference cost for the recovery challenge."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


BRANCHES = ["continue", "accurate", "recovery", "physical_recovery"]
SHORT_LABELS = ["Continue", "Replan", "Prompt retry", "Physical recovery"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    scenarios = summary["scenarios"]
    matrix = np.zeros((len(scenarios), len(BRANCHES)), dtype=int)
    labels: list[list[str]] = []
    for row_index, scenario in enumerate(scenarios):
        row_labels = []
        for column_index, branch in enumerate(BRANCHES):
            if scenario["safe_stops"].get(branch, False):
                matrix[row_index, column_index] = 2
                row_labels.append("STOP")
            elif scenario["outcomes"].get(branch, False):
                matrix[row_index, column_index] = 1
                row_labels.append("PASS")
            else:
                matrix[row_index, column_index] = 0
                row_labels.append("FAIL")
        labels.append(row_labels)

    fig, (ax_outcome, ax_cost) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    cmap = ListedColormap(["#d95f5f", "#4b9b72", "#e1a340"])
    ax_outcome.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax_outcome.set_xticks(range(len(BRANCHES)), SHORT_LABELS, rotation=18, ha="right")
    ax_outcome.set_yticks(
        range(len(scenarios)),
        [f"T{row['task_id']} {row['trigger']}" for row in scenarios],
    )
    ax_outcome.set_title("Exact-state intervention outcomes")
    for row_index, row in enumerate(labels):
        for column_index, label in enumerate(row):
            ax_outcome.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if label != "STOP" else "#202020",
                fontsize=9,
                fontweight="bold",
            )
    ax_outcome.tick_params(length=0)

    calls = [summary["branches"][branch]["vla_calls"] for branch in BRANCHES]
    bars = ax_cost.barh(
        range(len(BRANCHES)),
        calls,
        color=["#607d8b", "#4c78a8", "#8f6cb1", "#4b9b72"],
        height=0.62,
    )
    ax_cost.set_yticks(range(len(BRANCHES)), SHORT_LABELS)
    ax_cost.invert_yaxis()
    ax_cost.set_xlabel("Total PI0.5 calls")
    ax_cost.set_title("Inference cost across three states")
    ax_cost.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax_cost.set_axisbelow(True)
    for bar, value in zip(bars, calls, strict=True):
        ax_cost.text(
            value + max(calls) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
        )
    ax_cost.set_xlim(0, max(calls) * 1.17)

    fig.suptitle("CARVE PI0.5 Recovery Challenge", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Real LIBERO MuJoCo, identical restored states, admitted SMVE profile; STOP denotes fail-closed rejection.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
