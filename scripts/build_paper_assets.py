#!/usr/bin/env python3
"""Build paper-ready tables and figures from Agentic-VLA result summaries."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "paper_assets_20260609"
FIG_DIR = OUT_DIR / "figures"


def nested_get(data: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt_metric(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}{suffix}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_row(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text())
    rel_dir = summary_path.parent.relative_to(ROOT)
    trace = data.get("trace_aggregate", {})
    inference = trace.get("inference", {})
    realtime = trace.get("realtime", {})
    lightweight = trace.get("lightweight", {})
    recovery = trace.get("recovery", {})
    gpu = trace.get("gpu", {})
    flags = data.get("flags", {})
    seed_match = re.search(r"seed(\d+)", str(rel_dir))
    tasks = data.get("evaluated_task_ids", [])
    if isinstance(tasks, list):
        task_ids = ",".join(str(x) for x in tasks)
    else:
        task_ids = str(tasks)
    total_episodes = safe_int(data.get("total_episodes"))
    total_successes = safe_int(data.get("total_successes"))
    success_rate = (
        total_successes / total_episodes if total_episodes else data.get("overall_success_rate")
    )
    return {
        "result_dir": str(rel_dir),
        "name": summary_path.parent.name,
        "seed": seed_match.group(1) if seed_match else "",
        "ablation_tag": data.get("ablation_tag", ""),
        "method_tag": flags.get("method_tag", data.get("ablation_tag", "")),
        "task_suite": data.get("task_suite", ""),
        "task_ids": task_ids,
        "trials_per_task": data.get("trials_per_task", ""),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": success_rate,
        "avg_episode_length": data.get("avg_episode_length"),
        "full_vla_calls_total": inference.get("full_vla_calls_total"),
        "skipped_vla_calls_total": inference.get("skipped_vla_calls_total"),
        "full_vla_calls_per_episode": inference.get("full_vla_calls_per_episode"),
        "skipped_vla_calls_per_episode": inference.get("skipped_vla_calls_per_episode"),
        "skipped_vla_call_ratio": lightweight.get("skipped_vla_call_ratio"),
        "episode_wall_sec_mean": inference.get("episode_wall_sec_mean"),
        "vla_latency_ms_mean": inference.get("vla_latency_ms_mean"),
        "vla_latency_ms_p95": inference.get("vla_latency_ms_p95"),
        "ttfa_ms_mean": realtime.get("ttfa_ms_mean"),
        "deadline_miss_rate_mean": realtime.get("deadline_miss_rate_mean"),
        "success_under_deadline": realtime.get("success_under_deadline"),
        "fast_path_ratio_mean": realtime.get("fast_path_ratio_mean"),
        "recoveries_triggered_total": recovery.get("recoveries_triggered_total"),
        "recoveries_successful_total": recovery.get("recoveries_successful_total"),
        "recovery_precision": recovery.get("recovery_precision"),
        "peak_mem_gb_max": gpu.get("peak_mem_gb_max"),
        "ba_harness": flags.get("ba_harness", ""),
        "transition": flags.get("transition", ""),
        "critic": flags.get("critic", ""),
        "graph_rag": flags.get("graph_rag", ""),
        "control_deadline_ms": flags.get("control_deadline_ms", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_runs(pattern: str) -> list[Path]:
    return sorted(p / "summary.json" for p in RESULTS_DIR.glob(pattern) if (p / "summary.json").exists())


@dataclass(frozen=True)
class GroupSpec:
    table: str
    method: str
    glob_pattern: str
    note: str = ""


GROUPS = [
    GroupSpec(
        "task2_joint_conservative",
        "B0-VLA",
        "joint2x2_b0_vla_replan5_noLight_midNudge003_task2_10trials_seed*_20260608",
    ),
    GroupSpec(
        "task2_joint_conservative",
        "B0-VLA-Light",
        "joint2x2_b0_vla_light_safe80_replan5_reuseMax2_midNudge003_task2_10trials_seed*_20260608",
    ),
    GroupSpec(
        "task2_joint_conservative",
        "B4-Agentic",
        "b4_agentic_contextGate_replan5_noLight_midNudge_task2_10trials_seed*_20260608",
    ),
    GroupSpec(
        "task2_joint_conservative",
        "B4-Agentic-Light",
        "b4_agentic_light_safeLockout80_contextGate_replan5_reuseMax2_midNudge_task2_10trials_seed*_20260608",
    ),
    GroupSpec(
        "task2_runtime_candidate",
        "B0-VLA-Light-final-default",
        "finalTask2_b0_vla_light_defaultReplan_reuseMax2_midNudge003_10trials_seed*_20260609",
        "single seed currently available",
    ),
    GroupSpec(
        "task2_runtime_candidate",
        "B4-Agentic-Light-final-noTransition",
        "finalTask2_b4_agentic_light_noTransition_defaultReplan_reuseMax2_midNudge003_10trials_seed*_20260609",
        "fast no-transition candidate; recovery precision is not the main claim",
    ),
    GroupSpec(
        "lightweight_generalization",
        "B0-VLA",
        "lightTable_b0_vla_noLight_replan5_midNudge003_tasks157_5trials_seed*_202606*",
    ),
    GroupSpec(
        "lightweight_generalization",
        "B0-VLA-Light",
        "lightTable_b0_vla_light_replan5_reuseMax2_midNudge003_tasks157_5trials_seed*_202606*",
    ),
    GroupSpec(
        "caq_proxy_stress",
        "B0-VLA-CAQ-Proxy",
        "caqStress_b0_vla_caq_proxy_eager_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260609",
    ),
    GroupSpec(
        "caq_proxy_stress",
        "B4-Agentic-CAQ-Proxy",
        "caqStress_b4_agentic_caq_proxy_eager_safe80_noTransition_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260609",
    ),
    GroupSpec(
        "recovery_stress_strong",
        "B4-Agentic",
        "recoveryStress_b4_agentic_contextGate_replan5_noLight_midNudge005_task2_20trials_seed*_202606*",
        "Task2 strong perturbation, mid_nudge_xy=0.05",
    ),
    GroupSpec(
        "recovery_stress_strong",
        "B4-Agentic-Light",
        "recoveryStress_b4_agentic_light_safe80_replan5_reuseMax2_midNudge005_task2_20trials_seed*_202606*",
        "Task2 strong perturbation, mid_nudge_xy=0.05; action reuse with safety lockout",
    ),
    GroupSpec(
        "libero10_coverage",
        "B0-VLA-clean",
        "clean_B0_pytorchNoCompile_libero10_10trials_seed7_20260605_123000",
    ),
    GroupSpec(
        "libero10_coverage",
        "B4-Agentic-clean",
        "clean_B4RTProfileV2_pytorchNoCompile_libero10_10trials_seed7_20260605_132000",
    ),
    GroupSpec(
        "libero10_coverage",
        "B4-Agentic-Light-perturb",
        "finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609",
        "mid_nudge_xy=0.03 perturbation coverage",
    ),
    GroupSpec(
        "libero10_midnudge_snapshot",
        "B0-VLA",
        "paperSnapshot_b0_vla_noLight_replan5_midNudge003_libero10_5trials_seed7_20260610",
        "LIBERO-10 all tasks, mid_nudge_xy=0.03, seed 7",
    ),
    GroupSpec(
        "libero10_midnudge_snapshot",
        "B0-VLA-Light",
        "paperSnapshot_b0_vla_light_replan5_reuseMax2_midNudge003_libero10_5trials_seed7_20260610",
        "same protocol as B0-VLA; action reuse enabled",
    ),
    GroupSpec(
        "libero10_midnudge_snapshot",
        "B4-Agentic",
        "paperSnapshot_b4_agentic_noLight_safeLong_noTransition_replan5_midNudge003_libero10_5trials_seed7_20260610",
        "same protocol as B0-VLA; safe long-horizon, no transition",
    ),
    GroupSpec(
        "libero10_midnudge_snapshot",
        "B4-Agentic-Light",
        "finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609",
        "same task/trial/seed protocol from prior coverage run; safe long-horizon, no transition, action reuse",
    ),
]


WEIGHTED_METRICS = [
    "full_vla_calls_per_episode",
    "skipped_vla_calls_per_episode",
    "skipped_vla_call_ratio",
    "episode_wall_sec_mean",
    "vla_latency_ms_mean",
    "vla_latency_ms_p95",
    "ttfa_ms_mean",
    "deadline_miss_rate_mean",
    "success_under_deadline",
    "fast_path_ratio_mean",
    "avg_episode_length",
]


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        value = safe_float(row.get(field))
        weight = safe_float(row.get("total_episodes")) or 0.0
        if value is None or weight <= 0:
            continue
        num += value * weight
        den += weight
    return num / den if den else None


def aggregate_group(spec: GroupSpec) -> dict[str, Any]:
    summary_paths = find_runs(spec.glob_pattern)
    rows = [extract_row(p) for p in summary_paths]
    total_episodes = sum(safe_int(row.get("total_episodes")) for row in rows)
    total_successes = sum(safe_int(row.get("total_successes")) for row in rows)
    full_total = sum(safe_int(row.get("full_vla_calls_total")) for row in rows)
    skipped_total = sum(safe_int(row.get("skipped_vla_calls_total")) for row in rows)
    rec_total = sum(safe_int(row.get("recoveries_triggered_total")) for row in rows)
    rec_success = sum(safe_int(row.get("recoveries_successful_total")) for row in rows)
    out: dict[str, Any] = {
        "table": spec.table,
        "method": spec.method,
        "num_runs": len(rows),
        "seeds": ",".join(row["seed"] for row in rows if row.get("seed")),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": total_successes / total_episodes if total_episodes else None,
        "full_vla_calls_total": full_total if full_total else None,
        "skipped_vla_calls_total": skipped_total if skipped_total else None,
        "skipped_vla_call_ratio": (
            skipped_total / (full_total + skipped_total)
            if (full_total + skipped_total) > 0
            else weighted_mean(rows, "skipped_vla_call_ratio")
        ),
        "recoveries_triggered_total": rec_total if rec_total else None,
        "recoveries_successful_total": rec_success if rec_total else None,
        "recovery_precision": rec_success / rec_total if rec_total else None,
        "peak_mem_gb_max": max(
            [v for v in (safe_float(row.get("peak_mem_gb_max")) for row in rows) if v is not None],
            default=None,
        ),
        "source_dirs": "; ".join(row["result_dir"] for row in rows),
        "note": spec.note,
    }
    for metric in WEIGHTED_METRICS:
        if metric == "skipped_vla_call_ratio":
            continue
        out[metric] = weighted_mean(rows, metric)
    return out


def markdown_table(rows: list[dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", ""]
    headers = [
        "Method",
        "Runs",
        "Success",
        "Full VLA/ep",
        "Skipped/ep",
        "Skip ratio",
        "Wall/ep",
        "Miss@80ms",
        "SUD@80ms",
        "Recovery precision",
        "Peak GB",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        success = (
            f"{row['total_successes']}/{row['total_episodes']} "
            f"({fmt_float(row.get('success_rate'), 3)})"
        )
        rec = fmt_float(row.get("recovery_precision"), 3)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['method']}`",
                    str(row.get("num_runs", "")),
                    success,
                    fmt_metric(row.get("full_vla_calls_per_episode"), 2),
                    fmt_metric(row.get("skipped_vla_calls_per_episode"), 2),
                    fmt_float(row.get("skipped_vla_call_ratio"), 3),
                    fmt_metric(row.get("episode_wall_sec_mean"), 2, " s"),
                    fmt_float(row.get("deadline_miss_rate_mean"), 3),
                    fmt_float(row.get("success_under_deadline"), 3),
                    rec,
                    fmt_metric(row.get("peak_mem_gb_max"), 2),
                ]
            )
            + " |"
        )
    lines.append("")
    notes = [row for row in rows if row.get("note")]
    if notes:
        lines.append("Notes:")
        for row in notes:
            lines.append(f"- `{row['method']}`: {row['note']}")
        lines.append("")
    return "\n".join(lines)


def write_selected_tables(aggregates: list[dict[str, Any]]) -> None:
    titles = {
        "task2_joint_conservative": "Task2 Joint 2x2 Conservative Main Table",
        "task2_runtime_candidate": "Task2 Fast Runtime Candidate",
        "lightweight_generalization": "Lightweight Runtime Generalization",
        "caq_proxy_stress": "CAQ-Proxy Strong Perturbation Stress",
        "recovery_stress_strong": "Task2 Strong Perturbation Recovery Stress",
        "libero10_coverage": "LIBERO-10 Coverage Snapshot",
        "libero10_midnudge_snapshot": "LIBERO-10 Mid-Nudge Fair Snapshot",
    }
    for table, title in titles.items():
        rows = [row for row in aggregates if row["table"] == table]
        if not rows:
            continue
        (OUT_DIR / f"table_{table}.md").write_text(markdown_table(rows, title))


def extract_taskwise_tables() -> None:
    def write_taskwise_snapshot(
        targets: dict[str, str],
        csv_name: str,
        md_name: str,
        title: str,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for dirname, method in targets.items():
            path = RESULTS_DIR / dirname / "summary.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for task_id, metrics in sorted(data.get("task_metrics", {}).items(), key=lambda x: int(x[0])):
                rows.append(
                    {
                        "method": method,
                        "task_id": task_id,
                        "success_rate": metrics.get("success_rate"),
                        "episodes": metrics.get("episodes"),
                        "avg_episode_length": metrics.get("avg_episode_length"),
                    }
                )
        fields = ["method", "task_id", "success_rate", "episodes", "avg_episode_length"]
        write_csv(OUT_DIR / csv_name, rows, fields)
        lines = [f"# {title}", ""]
        lines.append("| Method | Task | Success rate | Episodes | Avg len |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| `{row['method']}` | {row['task_id']} | "
                f"{fmt_float(row['success_rate'], 3)} | {fmt_metric(row['episodes'], 0)} | "
                f"{fmt_metric(row['avg_episode_length'], 1)} |"
            )
        lines.append("")
        (OUT_DIR / md_name).write_text("\n".join(lines))

    write_taskwise_snapshot(
        {
            "clean_B0_pytorchNoCompile_libero10_10trials_seed7_20260605_123000": "B0-VLA-clean",
            "clean_B4RTProfileV2_pytorchNoCompile_libero10_10trials_seed7_20260605_132000": "B4-Agentic-clean",
            "finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609": "B4-Agentic-Light-perturb",
        },
        "taskwise_libero10_snapshot.csv",
        "table_taskwise_libero10_snapshot.md",
        "LIBERO-10 Taskwise Snapshot",
    )
    write_taskwise_snapshot(
        {
            "paperSnapshot_b0_vla_noLight_replan5_midNudge003_libero10_5trials_seed7_20260610": "B0-VLA",
            "paperSnapshot_b0_vla_light_replan5_reuseMax2_midNudge003_libero10_5trials_seed7_20260610": "B0-VLA-Light",
            "paperSnapshot_b4_agentic_noLight_safeLong_noTransition_replan5_midNudge003_libero10_5trials_seed7_20260610": "B4-Agentic",
            "finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609": "B4-Agentic-Light",
        },
        "taskwise_libero10_midnudge_snapshot.csv",
        "table_taskwise_libero10_midnudge_snapshot.md",
        "LIBERO-10 Mid-Nudge Taskwise Snapshot",
    )

def make_figures(aggregates: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        (OUT_DIR / "figure_generation_skipped.txt").write_text(str(exc))
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#2F5D7C", "#3FA88C", "#D66A3A", "#7C4D79", "#6B7280"]

    def rows_for(table: str) -> list[dict[str, Any]]:
        return [row for row in aggregates if row["table"] == table]

    task2 = rows_for("task2_joint_conservative")
    if task2:
        labels = [row["method"] for row in task2]
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(8.2, 4.3), dpi=180)
        ax1.bar(x - 0.18, [row["success_rate"] for row in task2], 0.36, color=colors[1], label="success")
        ax1.bar(
            x + 0.18,
            [row["success_under_deadline"] for row in task2],
            0.36,
            color=colors[0],
            label="SUD@80ms",
        )
        ax1.set_ylim(0, 1.08)
        ax1.set_ylabel("Rate")
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            [row["full_vla_calls_per_episode"] for row in task2],
            color=colors[2],
            marker="o",
            linewidth=2,
            label="full VLA calls/ep",
        )
        ax2.set_ylabel("Full VLA calls per episode")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=False)
        ax1.set_title("Task2: robustness and realtime tradeoff")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "task2_joint_success_sud_calls.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
        ax.scatter(
            [row["deadline_miss_rate_mean"] for row in task2],
            [row["success_rate"] for row in task2],
            s=[110 + 6 * row["full_vla_calls_per_episode"] for row in task2],
            c=colors[: len(task2)],
            alpha=0.85,
        )
        for row in task2:
            ax.annotate(
                row["method"],
                (row["deadline_miss_rate_mean"], row["success_rate"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
        ax.set_xlabel("Deadline miss rate @80ms (lower is better)")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0.75, 1.04)
        ax.grid(True, alpha=0.25)
        ax.set_title("Task2 deployment frontier")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "task2_deadline_frontier.png")
        plt.close(fig)

    light = rows_for("lightweight_generalization")
    if light:
        labels = [row["method"] for row in light]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), dpi=180)
        metrics = [
            ("success_rate", "Success rate", 1.0),
            ("full_vla_calls_per_episode", "Full VLA/ep", None),
            ("deadline_miss_rate_mean", "Miss@80ms", None),
        ]
        for ax, (field, title, ymax) in zip(axes, metrics):
            ax.bar(x, [row[field] for row in light], color=colors[: len(light)])
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            if ymax is not None:
                ax.set_ylim(0, ymax)
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle("Lightweight runtime generalization on Task1/5/7", y=1.03)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "lightweight_generalization_tasks157.png")
        plt.close(fig)

    caq = rows_for("caq_proxy_stress")
    if caq:
        labels = [row["method"] for row in caq]
        x = np.arange(len(labels))
        width = 0.22
        fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=180)
        ax.bar(x - width, [row["success_rate"] for row in caq], width, label="success", color=colors[1])
        ax.bar(x, [row["success_under_deadline"] for row in caq], width, label="SUD@80ms", color=colors[0])
        ax.bar(x + width, [row["skipped_vla_call_ratio"] for row in caq], width, label="skip ratio", color=colors[2])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title("CAQ-Proxy stress under stronger perturbation")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "caq_proxy_stress.png")
        plt.close(fig)

    recovery = rows_for("recovery_stress_strong")
    if recovery:
        labels = [row["method"] for row in recovery]
        x = np.arange(len(labels))
        width = 0.22
        fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=180)
        ax.bar(
            x - width,
            [row["success_rate"] for row in recovery],
            width,
            label="success",
            color=colors[1],
        )
        ax.bar(
            x,
            [row["success_under_deadline"] for row in recovery],
            width,
            label="SUD@80ms",
            color=colors[0],
        )
        ax.bar(
            x + width,
            [row["recovery_precision"] or 0.0 for row in recovery],
            width,
            label="recovery precision",
            color=colors[2],
        )
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title("Task2 strong perturbation recovery stress")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "recovery_stress_strong.png")
        plt.close(fig)

    taskwise_path = OUT_DIR / "taskwise_libero10_snapshot.csv"
    if taskwise_path.exists():
        task_rows = list(csv.DictReader(taskwise_path.open()))
        by_method: dict[str, list[dict[str, Any]]] = {}
        for row in task_rows:
            by_method.setdefault(row["method"], []).append(row)
        fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=180)
        width = 0.25
        task_ids = sorted({int(row["task_id"]) for row in task_rows})
        x = np.arange(len(task_ids))
        for i, (method, rows) in enumerate(by_method.items()):
            values_by_task = {int(row["task_id"]): float(row["success_rate"]) for row in rows}
            ax.bar(
                x + (i - 1) * width,
                [values_by_task.get(task_id, 0.0) for task_id in task_ids],
                width,
                label=method,
                color=colors[i % len(colors)],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in task_ids])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("LIBERO-10 task id")
        ax.set_ylabel("Success rate")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title("LIBERO-10 taskwise snapshot")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "libero10_taskwise_snapshot.png")
        plt.close(fig)


def write_report(all_rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    missing = [spec for spec in GROUPS if not find_runs(spec.glob_pattern)]
    lines = [
        "# Paper Asset Build Report",
        "",
        "Date: 2026-06-10",
        "",
        "## Generated files",
        "",
        "- `all_summaries.csv`: flattened index of every `results/**/summary.json`.",
        "- `selected_aggregates.csv`: aggregated paper-candidate tables.",
        "- `table_*.md`: Markdown tables ready for report/paper drafting.",
        "- `taskwise_libero10_snapshot.csv` and `.md`: taskwise success snapshot.",
        "- `figures/*.png`: paper/report figure drafts.",
        "",
        "## Selection policy",
        "",
        "- Main Task2 table uses conservative 3-seed 2x2 runs.",
        "- Fast no-transition runtime runs are kept as a separate candidate table.",
        "- CAQ-Proxy stress is reported separately because recovery was not triggered.",
        "- Recovery stress reports strong Task2 perturbations where recovery is triggered.",
        "- LIBERO-10 coverage mixes clean and perturbed settings, so it is a snapshot rather than a single fair main table.",
        "",
        "## Counts",
        "",
        f"- Total flattened summaries: `{len(all_rows)}`.",
        f"- Aggregated paper rows: `{len(aggregates)}`.",
        "",
    ]
    if missing:
        lines.extend(["## Missing selected groups", ""])
        for spec in missing:
            lines.append(f"- `{spec.table}` / `{spec.method}`: `{spec.glob_pattern}`")
        lines.append("")
    else:
        lines.extend(["## Missing selected groups", "", "- None.", ""])
    (OUT_DIR / "asset_build_report.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(RESULTS_DIR.glob("**/summary.json"))
    all_rows = [extract_row(path) for path in summary_paths]
    all_fields = list(all_rows[0].keys()) if all_rows else []
    write_csv(OUT_DIR / "all_summaries.csv", all_rows, all_fields)

    aggregates = [aggregate_group(spec) for spec in GROUPS]
    aggregate_fields = [
        "table",
        "method",
        "num_runs",
        "seeds",
        "total_successes",
        "total_episodes",
        "success_rate",
        "full_vla_calls_per_episode",
        "skipped_vla_calls_per_episode",
        "skipped_vla_call_ratio",
        "episode_wall_sec_mean",
        "vla_latency_ms_mean",
        "vla_latency_ms_p95",
        "ttfa_ms_mean",
        "deadline_miss_rate_mean",
        "success_under_deadline",
        "recoveries_triggered_total",
        "recoveries_successful_total",
        "recovery_precision",
        "peak_mem_gb_max",
        "source_dirs",
        "note",
    ]
    write_csv(OUT_DIR / "selected_aggregates.csv", aggregates, aggregate_fields)
    write_selected_tables(aggregates)
    extract_taskwise_tables()
    make_figures(aggregates)
    write_report(all_rows, aggregates)
    print(f"Wrote paper assets to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
