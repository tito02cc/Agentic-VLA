#!/usr/bin/env python3
"""Build paper framework diagrams for Agentic-VLA Runtime."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = Path("results/paper_assets_20260609/figures")

COLORS = {
    "input": "#d9ead3",
    "vla": "#cfe2f3",
    "agent": "#fce5cd",
    "light": "#eadcf8",
    "trace": "#f4cccc",
    "neutral": "#f8f9fa",
    "line": "#2f3a45",
    "success": "#d9ead3",
    "risk": "#f4cccc",
}


def setup_ax(width: float = 15.0, height: float = 8.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def box(ax, xy, wh, text, fc, ec=None, fontsize=10, weight="normal", radius=0.08):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.4,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1f2933",
        wrap=True,
    )
    return patch


def label(ax, xy, text, fontsize=12, weight="bold", ha="center"):
    ax.text(xy[0], xy[1], text, ha=ha, va="center", fontsize=fontsize, fontweight=weight, color="#17202a")


def arrow(ax, start, end, text=None, rad=0.0, color=None, fontsize=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        tx = (start[0] + end[0]) / 2
        ty = (start[1] + end[1]) / 2
        ax.text(tx, ty + 2.2, text, ha="center", va="center", fontsize=fontsize, color="#34495e")
    return patch


def diamond(ax, center, size, text, fc, fontsize=9):
    cx, cy = center
    w, h = size
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=COLORS["line"], linewidth=1.4)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color="#1f2933", wrap=True)
    return patch


def save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig1_system_overview():
    fig, ax = setup_ax(16, 8.5)
    label(ax, (50, 96), "Agentic-VLA Runtime: System Overview", fontsize=15)

    box(ax, (5, 64), (18, 10), "RGB observation", COLORS["input"])
    box(ax, (5, 48), (18, 10), "Robot state", COLORS["input"])
    box(ax, (5, 32), (18, 10), "Language instruction", COLORS["input"])

    box(ax, (38, 44), (24, 17), "Frozen VLA executor\n(pi0.5 / OpenPI)\nAction chunk", COLORS["vla"], fontsize=11, weight="bold")

    box(ax, (30, 75), (12, 9), "Progress\nverifier", COLORS["agent"], fontsize=9)
    box(ax, (45, 75), (12, 9), "GraphRAG\npriors", COLORS["agent"], fontsize=9)
    box(ax, (60, 75), (12, 9), "EvoKAM\nmemory", COLORS["agent"], fontsize=9)
    box(ax, (75, 75), (13, 9), "Risk / context\ngate", COLORS["agent"], fontsize=9)
    box(ax, (73, 58), (15, 10), "Recovery /\ntransition interface", COLORS["agent"], fontsize=9)

    box(ax, (30, 15), (15, 10), "Criticality\nestimator", COLORS["light"], fontsize=9)
    box(ax, (48, 15), (15, 10), "Action chunk\nbuffer", COLORS["light"], fontsize=9)
    box(ax, (66, 15), (15, 10), "Cached suffix\nreuse", COLORS["light"], fontsize=9)
    box(ax, (73, 32), (15, 10), "Full-call\nfallback", COLORS["light"], fontsize=9)

    box(ax, (91, 48), (7, 10), "Robot\naction", COLORS["success"], fontsize=9)
    box(ax, (87, 12), (10, 10), "Trace\nlogs", COLORS["trace"], fontsize=9)

    for y in [69, 53, 37]:
        arrow(ax, (23, y), (38, 52))
    arrow(ax, (62, 52), (91, 53))
    arrow(ax, (50, 61), (36, 75))
    arrow(ax, (50, 61), (51, 75))
    arrow(ax, (50, 61), (66, 75))
    arrow(ax, (72, 75), (79, 68))
    arrow(ax, (73, 63), (62, 57), rad=-0.15, text="prompt / control update")

    arrow(ax, (50, 44), (37, 25))
    arrow(ax, (45, 20), (48, 20))
    arrow(ax, (63, 20), (66, 20))
    arrow(ax, (73, 25), (76, 32))
    arrow(ax, (80, 42), (62, 48), rad=0.15, text="critical states")
    arrow(ax, (73, 15), (91, 17))
    arrow(ax, (88, 53), (92, 22), rad=-0.2)

    label(ax, (55, 88), "Agentic Policy Harness", fontsize=11)
    label(ax, (55, 7), "CAQ-Lite Realtime Runtime", fontsize=11)
    save(fig, "fig1_system_overview_agentic_vla_runtime")


def fig2_agentic_recovery_flow():
    fig, ax = setup_ax(15.8, 8.4)
    label(ax, (50, 96), "Agentic Harness: Progress-Gated Recovery", fontsize=15)

    box(ax, (5, 60), (14, 12), "Observation\n+ task", COLORS["input"], fontsize=10)
    box(ax, (24, 60), (16, 12), "Progress\nverifier", COLORS["agent"], fontsize=10)
    diamond(ax, (50, 66), (15, 16), "Risky or\nstalled?", COLORS["neutral"], fontsize=9)
    box(ax, (64, 76), (18, 11), "Memory / prior\nretrieval", COLORS["agent"], fontsize=10)
    box(ax, (64, 57), (18, 11), "Recovery prompt\nand expert route", COLORS["agent"], fontsize=10)
    box(ax, (64, 36), (18, 11), "Continue frozen\nVLA execution", COLORS["vla"], fontsize=10)
    box(ax, (88, 57), (9, 11), "Action", COLORS["success"], fontsize=10)

    box(ax, (16, 17), (18, 10), "Trace:\nprogress events", COLORS["trace"], fontsize=9)
    box(ax, (40, 17), (18, 10), "Trace:\nrecovery count", COLORS["trace"], fontsize=9)
    box(ax, (64, 17), (18, 10), "Trace:\nfinal phase", COLORS["trace"], fontsize=9)

    arrow(ax, (19, 66), (24, 66))
    arrow(ax, (40, 66), (42.5, 66))
    arrow(ax, (57.5, 70), (64, 81), text="yes")
    arrow(ax, (57.5, 65), (64, 63), text="yes")
    arrow(ax, (50, 58), (64, 42), text="no", rad=-0.12)
    arrow(ax, (82, 81), (88, 67), rad=-0.2)
    arrow(ax, (82, 63), (88, 63))
    arrow(ax, (82, 42), (91, 57), rad=0.12)
    arrow(ax, (32, 60), (25, 27), rad=0.0)
    arrow(ax, (73, 57), (49, 27), rad=0.2)
    arrow(ax, (92, 57), (73, 27), rad=0.18)

    ax.text(
        50,
        8,
        "The harness does not replace the VLA. It decides when execution requires memory, recovery, or normal continuation.",
        ha="center",
        va="center",
        fontsize=10,
        color="#34495e",
    )
    save(fig, "fig2_agentic_recovery_flow")


def fig3_caq_lite_timeline():
    fig, ax = setup_ax(16, 7.2)
    label(ax, (50, 96), "CAQ-Lite: Criticality-Aware VLA Call Scheduling", fontsize=15)

    ax.plot([8, 92], [50, 50], color=COLORS["line"], linewidth=1.6)
    ax.text(8, 44, "time", ha="left", va="center", fontsize=10, color="#34495e")

    full_calls = [14, 25, 52, 62, 78]
    skips = [18, 21, 30, 34, 38, 84, 88]
    for x in full_calls:
        box(ax, (x - 2.4, 58), (4.8, 13), "Full\nVLA", COLORS["vla"], fontsize=8)
        arrow(ax, (x, 58), (x, 51.5), color="#3d5a80")
    for x in skips:
        box(ax, (x - 2.0, 31), (4.0, 10), "Reuse", COLORS["light"], fontsize=8)
        arrow(ax, (x, 41), (x, 48.5), color="#6f42c1")

    ax.add_patch(Rectangle((45, 25), 23, 47, facecolor="#fff2cc", edgecolor="#b7950b", alpha=0.45, linewidth=1.2))
    ax.text(56.5, 69, "critical interval", ha="center", va="center", fontsize=10, color="#7d6608", fontweight="bold")
    ax.text(56.5, 23, "post-nudge / recovery lockout", ha="center", va="center", fontsize=9, color="#7d6608")

    box(ax, (6, 74), (24, 10), "Low-risk phase:\nreuse cached suffix", COLORS["success"], fontsize=9)
    box(ax, (38, 74), (22, 10), "Perturbation or high risk:\nforce full VLA calls", COLORS["risk"], fontsize=9)
    box(ax, (69, 74), (24, 10), "Recovered low-risk phase:\nreuse resumes", COLORS["success"], fontsize=9)

    arrow(ax, (18, 74), (21, 64), rad=-0.2)
    arrow(ax, (49, 74), (52, 64), rad=-0.2)
    arrow(ax, (81, 74), (84, 41), rad=0.2)

    box(ax, (9, 11), (21, 10), "Metric:\nfull VLA calls / ep", COLORS["neutral"], fontsize=9)
    box(ax, (39.5, 11), (21, 10), "Metric:\nmiss@deadline", COLORS["neutral"], fontsize=9)
    box(ax, (70, 11), (21, 10), "Metric:\nsuccess under deadline", COLORS["neutral"], fontsize=9)
    arrow(ax, (22, 31), (19, 21), rad=-0.2)
    arrow(ax, (52, 58), (50, 21), rad=0.05)
    arrow(ax, (82, 31), (80, 21), rad=0.2)

    ax.text(
        50,
        4,
        "CAQ-Lite reduces blocking calls only when the harness marks the state as low-risk; critical phases fall back to full VLA execution.",
        ha="center",
        va="center",
        fontsize=10,
        color="#34495e",
    )
    save(fig, "fig3_caq_lite_timeline")


def main():
    fig1_system_overview()
    fig2_agentic_recovery_flow()
    fig3_caq_lite_timeline()
    print(f"Wrote diagrams to {OUT_DIR}")


if __name__ == "__main__":
    main()
