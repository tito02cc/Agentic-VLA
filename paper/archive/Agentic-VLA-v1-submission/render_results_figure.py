from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULTS_ROOT = ROOT.parent.parent / "results"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _task_rate(summary: dict, task_id: str) -> float:
    return 100.0 * float(summary["task_metrics"][task_id]["success_rate"])


def main():
    baseline = _load(RESULTS_ROOT / "ablation_B0_pi05_libero_10_20260512" / "summary.json")
    full = _load(RESULTS_ROOT / "ablation_FULL_refined_libero10_20260518_2059" / "summary.json")
    wo_graph_t8 = _load(RESULTS_ROOT / "ablation_FULL_tuned_v2_wo_graph_task8_20260516_112247" / "summary.json")
    wo_critic_t8 = _load(RESULTS_ROOT / "ablation_FULL_tuned_v2_wo_critic_task8_20260516_112247" / "summary.json")
    wo_transition_t8 = _load(RESULTS_ROOT / "ablation_FULL_tuned_v2_wo_transition_task8_20260516_112247" / "summary.json")

    task_labels = [f"T{i}" for i in range(10)]
    baseline_rates = [_task_rate(baseline, str(i)) for i in range(10)]
    full_rates = [_task_rate(full, str(i)) for i in range(10)]

    diag_labels = [
        "A1 baseline (20)",
        "Full refined (20)",
        "w/o Graph+Mem. (10)",
        "w/o Critic (10)",
        "w/o Transition (10)",
    ]
    diag_values = [
        _task_rate(baseline, "8"),
        _task_rate(full, "8"),
        _task_rate(wo_graph_t8, "8"),
        _task_rate(wo_critic_t8, "8"),
        _task_rate(wo_transition_t8, "8"),
    ]
    diag_colors = [
        "#9ca3af",
        "#2563eb",
        "#8b5cf6",
        "#6366f1",
        "#14b8a6",
    ]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(12.8, 5.4), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.05], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(task_labels))
    width = 0.36
    ax1.bar(x - width / 2, baseline_rates, width=width, color="#9ca3af", label="A1 baseline")
    ax1.bar(x + width / 2, full_rates, width=width, color="#2563eb", label="Full Agentic-VLA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Main Comparison on libero_10", fontsize=12, weight="bold", pad=8)
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax1.legend(frameon=False, loc="lower left")
    ax1.axvspan(7.5, 8.5, color="#fee2e2", alpha=0.45)
    ax1.annotate(
        "dominant weak task",
        xy=(8, 100),
        xytext=(7.55, 103),
        fontsize=8.5,
        color="#991b1b",
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "color": "#991b1b", "lw": 0.8},
    )

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(diag_labels))
    ax2.barh(y, diag_values, color=diag_colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(diag_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Task-8 Success Rate (%)")
    ax2.set_title("Task-8 Ablation (mixed trial counts)", fontsize=12, weight="bold", pad=8)
    ax2.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax2.tick_params(axis="y", labelsize=9)
    for yi, val in enumerate(diag_values):
        ax2.text(min(val + 1.0, 97.0), yi, f"{val:.0f}", va="center", fontsize=8.8, color="#111827")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_libero10_task_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
