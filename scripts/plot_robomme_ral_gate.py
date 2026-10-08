#!/usr/bin/env python3
"""Plot paired RoboMME outcomes from a machine-generated summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if data.get("protocol") != "carve.robomme.ral_gate_pilot_summary.v1":
        raise ValueError("expected a RoboMME paired pilot summary")
    episodes = data["episodes"]
    if not episodes:
        raise ValueError("no paired episodes")

    ink, teal, amber, red = "#213444", "#177c74", "#b36c27", "#bd4b44"
    fig, (ax_matrix, ax_delta) = plt.subplots(
        1, 2, figsize=(9.2, 4.25), gridspec_kw={"width_ratios": [1.03, 1.42]}
    )
    fig.patch.set_facecolor("white")
    y = np.arange(len(episodes))
    ep_labels = [str(row["episode"]) for row in episodes]
    for index, case in enumerate(episodes):
        for column, arm in enumerate("ABC"):
            success = case["arms"][arm]["success"]
            ax_matrix.scatter(column, index, s=104, marker="o" if success else "X",
                              c=teal if success else red, zorder=3,
                              edgecolors="white" if success else "none", linewidths=0.8)
    ax_matrix.set_xlim(-0.55, 2.55)
    ax_matrix.set_ylim(len(episodes)-0.5, -0.5)
    ax_matrix.set_xticks(range(3), ["A: Raw", "B: Harness", "C: + Memory"])
    ax_matrix.set_yticks(y, ep_labels)
    ax_matrix.set_ylabel("Preselected episode ID", color=ink)
    ax_matrix.set_title("Official task completion", fontsize=12, fontweight="bold", color=ink, loc="left")
    ax_matrix.grid(axis="y", color="#e2e7e9", linewidth=0.75)
    ax_matrix.set_axisbelow(True)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    for index, case in enumerate(episodes):
        raw, memory = case["arms"]["A"], case["arms"]["C"]
        if not (raw["success"] and memory["success"]):
            ax_delta.text(0, index, "FAIL: C lost a baseline success", ha="center",
                          va="center", color=red, fontsize=8.3, fontweight="bold")
            continue
        delta = memory["executed_steps"] - raw["executed_steps"]
        ax_delta.barh(index, delta, height=0.53,
                      color=teal if delta < 0 else amber, zorder=3)
        offset = -4 if delta < 0 else 4
        ax_delta.text(delta + offset, index, f"{delta:+d}",
                      ha="right" if delta < 0 else "left", va="center",
                      fontsize=8.4, color=ink)
    ax_delta.axvline(0, color=ink, linewidth=0.95)
    ax_delta.set_xlim(-250, 250)
    ax_delta.set_ylim(len(episodes)-0.5, -0.5)
    ax_delta.set_yticks(y, [])
    ax_delta.set_xlabel("C minus A executed steps (both successful only)", color=ink)
    ax_delta.set_title("Paired execution cost", fontsize=12, fontweight="bold", color=ink, loc="left")
    ax_delta.grid(axis="x", color="#e2e7e9", linewidth=0.75)
    ax_delta.set_axisbelow(True)
    for spine in ax_delta.spines.values():
        spine.set_visible(False)
    fig.suptitle("VideoUnmask: memory saves steps in some pairs, but harms one",
                 x=0.065, y=0.99, ha="left", color=ink, fontsize=13.2, fontweight="bold")
    fig.text(0.066, 0.015, "Development pilot, 8 matched initial states. Shorter early failure is not a speedup.",
             fontsize=8.5, color="#59656c")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.19, wspace=0.2)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
