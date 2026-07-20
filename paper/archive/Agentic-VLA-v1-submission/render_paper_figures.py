from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


def _box(ax, xy, width, height, text, fc, ec="#1f2937", text_color="#111827", fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, text=None, color="#4b5563", fontsize=9, text_offset=(0, 0)):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color)
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=fontsize, color=color, ha="center", va="center")


def draw_method_overview():
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Agentic-VLA: Inference-Time Augmentation around a Frozen Fine-Tuned VLA", fontsize=16, weight="bold")
    ax.text(0.03, 0.91, "The base policy remains unchanged; agentic modules intervene only when process control is needed.", fontsize=10.5, color="#374151")

    _box(ax, (0.03, 0.58), 0.18, 0.22, "Inputs\nagentview image\nwrist image\nrobot state\nlanguage instruction", fc="#dbeafe")
    _box(ax, (0.27, 0.55), 0.22, 0.28, "Frozen fine-tuned VLA\npi05_libero policy server\nchunked action inference", fc="#dcfce7")
    _box(ax, (0.55, 0.74), 0.18, 0.14, "Transition Agent\nstall / state-gap detection\ntransition prompt", fc="#fef3c7")
    _box(ax, (0.55, 0.53), 0.18, 0.14, "Scene Priors / Memory\nobject priors\nepisodic cues", fc="#fde68a")
    _box(ax, (0.55, 0.32), 0.18, 0.14, "Critic / Retry\nperiodic checking\nrecovery + retry", fc="#fecaca")
    _box(ax, (0.79, 0.55), 0.17, 0.25, "Unified execution loop\naction rollout\nprogress monitoring\nreplan / resume", fc="#e9d5ff")
    _box(ax, (0.27, 0.12), 0.22, 0.16, "Official LIBERO environment\nreal rollout execution\nsuccess signal + videos", fc="#e5e7eb")

    _arrow(ax, (0.21, 0.69), (0.27, 0.69), "observation + prompt", text_offset=(0, 0.03))
    _arrow(ax, (0.49, 0.69), (0.55, 0.81))
    _arrow(ax, (0.49, 0.69), (0.55, 0.60))
    _arrow(ax, (0.49, 0.69), (0.55, 0.39))
    _arrow(ax, (0.73, 0.81), (0.79, 0.72), "transition actions", text_offset=(0.01, 0.04))
    _arrow(ax, (0.73, 0.60), (0.79, 0.67), "prompt augmentation", text_offset=(0.02, -0.04))
    _arrow(ax, (0.73, 0.39), (0.79, 0.61), "recovery / retry", text_offset=(0.03, -0.02))
    _arrow(ax, (0.875, 0.55), (0.875, 0.31), "execute", text_offset=(0.03, 0))
    _arrow(ax, (0.49, 0.20), (0.79, 0.55), "environment feedback", text_offset=(0.01, -0.03))
    _arrow(ax, (0.79, 0.55), (0.49, 0.55), "replan if needed", text_offset=(0, -0.04))

    ax.text(0.03, 0.04, "Key idea: Agentic-VLA repairs missing process control without retraining the VLA backbone.", fontsize=10.5, color="#111827")

    out = FIG_DIR / "fig_method_overview.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_transition_gap():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    titles = ["Baseline Rollout", "With Transition Agent"]
    subtitles = [
        "State gap causes stall near subtask boundary",
        "Intermediate transition reconnects the rollout",
    ]

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.text(0.3, 9.4, titles[i], fontsize=15, weight="bold")
        ax.text(0.3, 8.85, subtitles[i], fontsize=10, color="#4b5563")

        ax.plot([1.0, 3.0], [2.0, 4.0], color="#9ca3af", linewidth=2, linestyle="--")
        ax.plot([7.0, 9.0], [6.0, 8.0], color="#9ca3af", linewidth=2, linestyle="--")
        ax.text(0.8, 1.2, "Subtask A\nstable region", fontsize=10)
        ax.text(7.0, 8.3, "Subtask B\nstable region", fontsize=10)

        if i == 0:
            xs = [1.2, 2.2, 3.2, 4.0, 4.6, 4.2, 4.7, 4.3, 4.6, 4.4]
            ys = [2.1, 2.9, 3.8, 4.6, 5.0, 4.8, 5.1, 4.7, 5.0, 4.8]
            ax.plot(xs, ys, color="#ef4444", linewidth=2.5, marker="o", markersize=4)
            ax.text(4.9, 5.5, "stall / oscillation", fontsize=10, color="#b91c1c")
            ax.annotate("", xy=(4.6, 5.1), xytext=(5.9, 6.2), arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5))
        else:
            xs = [1.2, 2.2, 3.2, 4.1, 5.2, 6.1, 7.0, 8.0, 8.8]
            ys = [2.1, 2.9, 3.8, 4.6, 5.3, 6.0, 6.6, 7.3, 7.9]
            ax.plot(xs, ys, color="#2563eb", linewidth=2.5, marker="o", markersize=4)
            ax.text(4.4, 5.9, "transition prompt\nintermediate motion", fontsize=10, color="#1d4ed8")
            ax.annotate("", xy=(5.2, 5.3), xytext=(3.8, 7.0), arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.5))

        ax.scatter([1.2], [2.1], s=60, color="#10b981", zorder=3)
        ax.scatter([8.8 if i == 1 else 4.4], [7.9 if i == 1 else 4.8], s=60, color="#111827", zorder=3)
        ax.text(1.0, 1.6, "start", fontsize=9)
        ax.text(8.65 if i == 1 else 4.2, 4.2 if i == 0 else 8.3, "end", fontsize=9)

    fig.suptitle("Transition Agent bridges subtask boundaries where chunked VLA execution alone may stall", fontsize=16, weight="bold", y=0.98)
    out = FIG_DIR / "fig_transition_gap.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_critic_retry_flow():
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Critic / Retry Flow for Local Failure Recovery", fontsize=16, weight="bold")
    ax.text(0.03, 0.90, "The critic does not replace the policy; it only intervenes when execution appears inconsistent or stalled.", fontsize=10.5, color="#374151")

    _box(ax, (0.04, 0.52), 0.16, 0.18, "Current rollout state\nimages + state + task prompt", fc="#dbeafe")
    _box(ax, (0.27, 0.52), 0.16, 0.18, "Periodic critic check\nQwen3-VL or fallback\nprogress judgment", fc="#fee2e2")
    _box(ax, (0.50, 0.64), 0.16, 0.14, "Healthy progress\nkeep current plan", fc="#dcfce7")
    _box(ax, (0.50, 0.40), 0.16, 0.18, "Likely failure\nstuck / incomplete /\nwrong contact", fc="#fecaca")
    _box(ax, (0.73, 0.64), 0.18, 0.14, "No intervention\ncontinue rollout", fc="#e5e7eb")
    _box(ax, (0.73, 0.40), 0.18, 0.18, "Recovery + retry\nclear plan\nissue recovery prompt\nresume from safer state", fc="#fde68a")
    _box(ax, (0.35, 0.10), 0.28, 0.14, "Outcome statistics\nretry count\nretry-to-success rate\nextra episode cost", fc="#e9d5ff")

    _arrow(ax, (0.20, 0.61), (0.27, 0.61), "observe")
    _arrow(ax, (0.43, 0.61), (0.50, 0.71), "healthy")
    _arrow(ax, (0.43, 0.61), (0.50, 0.49), "failure")
    _arrow(ax, (0.66, 0.71), (0.73, 0.71), "continue")
    _arrow(ax, (0.66, 0.49), (0.73, 0.49), "recover")
    _arrow(ax, (0.82, 0.40), (0.82, 0.24), "log")
    _arrow(ax, (0.82, 0.64), (0.58, 0.24), "measure")

    out = FIG_DIR / "fig_critic_retry_flow.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_failure_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Failure Taxonomy Used for Agentic-VLA Analysis", fontsize=16, weight="bold")
    ax.text(0.03, 0.91, "Each module is tied to a concrete long-horizon failure mode rather than a vague reasoning claim.", fontsize=10.5, color="#374151")

    _box(ax, (0.08, 0.74), 0.22, 0.13, "Transition failure\nstall / oscillation at subtask boundary", fc="#fef3c7")
    _box(ax, (0.39, 0.74), 0.22, 0.13, "Contact or grasp failure\nreaches region but cannot manipulate stably", fc="#fde68a")
    _box(ax, (0.70, 0.74), 0.22, 0.13, "Completion failure\npartial progress but wrong final state", fc="#fecaca")
    _box(ax, (0.24, 0.48), 0.22, 0.13, "Recovery failure\nrollout drifts and cannot re-enter productive state", fc="#fee2e2")
    _box(ax, (0.54, 0.48), 0.22, 0.13, "Target alignment failure\nlocalization / approach mismatch", fc="#dbeafe")

    _box(ax, (0.10, 0.16), 0.20, 0.14, "A2 Transition Agent\nbridge state gaps", fc="#fef3c7")
    _box(ax, (0.40, 0.16), 0.20, 0.14, "A3 Scene Priors / Memory\nimprove approach and contact hints", fc="#fde68a")
    _box(ax, (0.70, 0.16), 0.20, 0.14, "A4 Critic / Retry\nrecover after drift or local failure", fc="#fecaca")

    _arrow(ax, (0.19, 0.74), (0.20, 0.30))
    _arrow(ax, (0.50, 0.74), (0.50, 0.30))
    _arrow(ax, (0.81, 0.74), (0.80, 0.30))
    _arrow(ax, (0.35, 0.48), (0.80, 0.30))
    _arrow(ax, (0.65, 0.48), (0.50, 0.30))

    ax.text(0.03, 0.04, "Interpretation goal: if a module is effective, it should disproportionately reduce the failure modes it is designed to address.", fontsize=10.2, color="#111827")

    out = FIG_DIR / "fig_failure_taxonomy.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_eval_protocol():
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Real Rollout Evaluation and Evidence Pipeline", fontsize=16, weight="bold")
    ax.text(0.03, 0.90, "All reported numbers come from official LIBERO execution rather than mock scoring or offline estimates.", fontsize=10.5, color="#374151")

    _box(ax, (0.03, 0.52), 0.16, 0.18, "Official LIBERO suites\nlibero_object\nlibero_goal\nlibero_10", fc="#dbeafe")
    _box(ax, (0.24, 0.52), 0.18, 0.18, "Ablation variants\nA1 / A2 / A3 / A4 / Full", fc="#e9d5ff")
    _box(ax, (0.47, 0.52), 0.18, 0.18, "OpenPI policy server\nreal policy inference\nwebsocket rollout loop", fc="#dcfce7")
    _box(ax, (0.70, 0.52), 0.18, 0.18, "Per-episode outputs\nsuccess flag\nvideo\nintervention counts", fc="#fde68a")

    _box(ax, (0.13, 0.16), 0.22, 0.16, "Task-level summaries\nper-task success rate\nweak-task evidence", fc="#fef3c7")
    _box(ax, (0.41, 0.16), 0.22, 0.16, "Mechanism statistics\ntransitions\nretries\nepisode length", fc="#fee2e2")
    _box(ax, (0.69, 0.16), 0.22, 0.16, "Paper tables and figures\nmain table\nper-task table\nqualitative cases", fc="#e5e7eb")

    _arrow(ax, (0.19, 0.61), (0.24, 0.61))
    _arrow(ax, (0.42, 0.61), (0.47, 0.61))
    _arrow(ax, (0.65, 0.61), (0.70, 0.61))
    _arrow(ax, (0.79, 0.52), (0.24, 0.32), "aggregate", text_offset=(0.00, -0.02))
    _arrow(ax, (0.79, 0.52), (0.52, 0.32), "measure", text_offset=(0.00, 0.01))
    _arrow(ax, (0.24, 0.16), (0.69, 0.24), "support", text_offset=(0.0, 0.03))
    _arrow(ax, (0.63, 0.24), (0.69, 0.24))

    out = FIG_DIR / "fig_eval_protocol.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        draw_method_overview(),
        draw_transition_gap(),
        draw_critic_retry_flow(),
        draw_failure_taxonomy(),
        draw_eval_protocol(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
