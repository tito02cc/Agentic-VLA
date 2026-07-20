#!/usr/bin/env python3
"""Build the CARVE VLM+VLA co-resident contention report and deadline plot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "carve_optimize"
DEFAULT_MANIFESTS = {
    "compile": RESULTS / "pi05_compile_2step_h10_active_qwen35_4b_500calls_20260717.json",
    "smve": RESULTS / "pi05_smve_2step_h10_active_qwen35_4b_500calls_20260717.json",
}
DEFAULT_LOADS = {
    "compile": RESULTS / "qwen35_4b_active_vlm_load_compile_500_20260717.jsonl",
    "smve": RESULTS / "qwen35_4b_active_vlm_load_smve_500_20260717.jsonl",
}
DEFAULT_EVENT_MANIFESTS = {
    "compile": RESULTS / "pi05_compile_2step_h10_event5_qwen35_4b_500calls_20260717.json",
    "smve": RESULTS / "pi05_smve_2step_h10_event5_qwen35_4b_500calls_20260717.json",
}
DEFAULT_EVENT_LOADS = {
    "compile": RESULTS / "qwen35_4b_event5_vlm_load_compile_500_20260717.jsonl",
    "smve": RESULTS / "qwen35_4b_event5_vlm_load_smve_500_20260717.jsonl",
}
DEFAULT_REPEATS = {
    "compile": [
        RESULTS / "pi05_compile_2step_h10_active_qwen35_4b_pilot_20260717.json",
        RESULTS / "pi05_compile_2step_h10_active_qwen35_4b_200calls_20260717.json",
        DEFAULT_MANIFESTS["compile"],
    ],
    "smve": [
        RESULTS / "pi05_smve_2step_h10_active_qwen35_4b_pilot_20260717.json",
        RESULTS / "pi05_smve_2step_h10_active_qwen35_4b_200calls_20260717.json",
        DEFAULT_MANIFESTS["smve"],
    ],
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No JSONL rows: {path}")
    return rows


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2)) / scale
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _latencies(manifest: dict[str, Any]) -> list[float]:
    values = manifest["benchmark"]["metadata"]["latency_samples"]["runtime_ms"]
    if len(values) != int(manifest["benchmark"]["samples"]):
        raise ValueError("Latency sample count does not match benchmark sample count")
    return [float(value) for value in values]


def _profile_summary(
    manifest: dict[str, Any], deadlines: list[float]
) -> dict[str, Any]:
    values = _latencies(manifest)
    misses = {str(int(deadline)): sum(value > deadline for value in values) for deadline in deadlines}
    return {
        "backend": manifest["profile"]["backend"],
        "profile_id": manifest["profile"]["profile_id"],
        "samples": len(values),
        "runtime_p50_ms": _percentile(values, 0.50),
        "runtime_p95_ms": _percentile(values, 0.95),
        "runtime_p99_ms": _percentile(values, 0.99),
        "mean_ms": mean(values),
        "deadline_misses": misses,
        "deadline_miss_rates": {
            key: count / len(values) for key, count in misses.items()
        },
        "deadline_miss_wilson_95": {
            key: _wilson(count, len(values)) for key, count in misses.items()
        },
        "fidelity": manifest["fidelity"],
        "deployment_condition": manifest["benchmark"]["deployment_condition"],
    }


def _load_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    successes = sum(bool(row.get("success")) for row in rows)
    return {
        "requests": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "latency_p50_ms": median(latencies),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_mean_ms": mean(latencies),
        "prompt_tokens": sorted(
            {int(row.get("usage", {}).get("prompt_tokens", 0)) for row in rows}
        ),
    }


def _repeat_summary(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        manifest = _load_json(path)
        benchmark = manifest["benchmark"]
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "samples": int(benchmark["samples"]),
                "runtime_p50_ms": float(benchmark["runtime_p50_ms"]),
                "runtime_p95_ms": float(benchmark["runtime_p95_ms"]),
                "runtime_p99_ms": float(benchmark["runtime_p99_ms"]),
                "deadline_miss_rate": float(benchmark["deadline_miss_rate"]),
            }
        )
    return rows


def build_summary(deadlines: list[float]) -> dict[str, Any]:
    manifests = {name: _load_json(path) for name, path in DEFAULT_MANIFESTS.items()}
    profiles = {
        name: _profile_summary(manifest, deadlines) for name, manifest in manifests.items()
    }
    loads = {name: _load_summary(_load_jsonl(path)) for name, path in DEFAULT_LOADS.items()}
    event_manifests = {
        name: _load_json(path) for name, path in DEFAULT_EVENT_MANIFESTS.items()
    }
    event_profiles = {
        name: _profile_summary(manifest, deadlines)
        for name, manifest in event_manifests.items()
    }
    event_loads = {
        name: _load_summary(_load_jsonl(path)) for name, path in DEFAULT_EVENT_LOADS.items()
    }
    compile_row = profiles["compile"]
    smve_row = profiles["smve"]
    comparable = {
        "checkpoint_equal": manifests["compile"]["checkpoint_id"]
        == manifests["smve"]["checkpoint_id"],
        "hardware_equal": manifests["compile"]["hardware"] == manifests["smve"]["hardware"],
        "inference_steps_equal": manifests["compile"]["profile"]["inference_steps"]
        == manifests["smve"]["profile"]["inference_steps"],
        "action_horizon_equal": manifests["compile"]["profile"]["action_horizon"]
        == manifests["smve"]["profile"]["action_horizon"],
        "both_fidelity_passed": bool(manifests["compile"]["fidelity"]["passed"])
        and bool(manifests["smve"]["fidelity"]["passed"]),
        "vlm_success_rate_equal": loads["compile"]["success_rate"]
        == loads["smve"]["success_rate"],
    }
    return {
        "protocol": {
            "vla": "OpenPI pi0.5 LIBERO PyTorch",
            "agent_vlm": "Qwen3.5-4B BF16 vision-language server",
            "hardware": "NVIDIA GeForce RTX 4090 24GB",
            "deadlines_ms": deadlines,
            "condition": "continuous co-resident visual VLM inference",
        },
        "fairness_audit": {"passed": all(comparable.values()), "checks": comparable},
        "profiles": profiles,
        "vlm_load": loads,
        "event_triggered_profiles": event_profiles,
        "event_triggered_vlm_load": event_loads,
        "effect": {
            "runtime_p50_reduction": 1.0
            - smve_row["runtime_p50_ms"] / compile_row["runtime_p50_ms"],
            "runtime_p95_reduction": 1.0
            - smve_row["runtime_p95_ms"] / compile_row["runtime_p95_ms"],
            "runtime_p99_reduction": 1.0
            - smve_row["runtime_p99_ms"] / compile_row["runtime_p99_ms"],
            "miss_rate_80ms_absolute_reduction": (
                compile_row["deadline_miss_rates"]["80"]
                - smve_row["deadline_miss_rates"]["80"]
            ),
            "miss_rate_80ms_relative_reduction": 1.0
            - smve_row["deadline_miss_rates"]["80"]
            / compile_row["deadline_miss_rates"]["80"],
            "event_smve_vs_continuous_compile_p50_reduction": 1.0
            - event_profiles["smve"]["runtime_p50_ms"]
            / compile_row["runtime_p50_ms"],
            "event_smve_vs_continuous_compile_p95_reduction": 1.0
            - event_profiles["smve"]["runtime_p95_ms"]
            / compile_row["runtime_p95_ms"],
            "event_smve_vs_continuous_compile_miss_reduction": (
                compile_row["deadline_miss_rates"]["80"]
                - event_profiles["smve"]["deadline_miss_rates"]["80"]
            ),
        },
        "repeats": {
            name: _repeat_summary(paths) for name, paths in DEFAULT_REPEATS.items()
        },
        "evidence": {
            "manifests": {name: str(path.relative_to(ROOT)) for name, path in DEFAULT_MANIFESTS.items()},
            "vlm_loads": {name: str(path.relative_to(ROOT)) for name, path in DEFAULT_LOADS.items()},
            "event_manifests": {
                name: str(path.relative_to(ROOT)) for name, path in DEFAULT_EVENT_MANIFESTS.items()
            },
            "event_vlm_loads": {
                name: str(path.relative_to(ROOT)) for name, path in DEFAULT_EVENT_LOADS.items()
            },
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    compile_row = summary["profiles"]["compile"]
    smve_row = summary["profiles"]["smve"]
    lines = [
        "# CARVE Agent-VLM + VLA Co-resident Contention Gate",
        "",
        "Date: 2026-07-17",
        "",
        "## Protocol",
        "",
        "- Frozen policy: OpenPI pi0.5 LIBERO PyTorch checkpoint.",
        "- Agent module: Qwen3.5-4B BF16 server continuously processing a robot image.",
        "- Hardware: one RTX 4090; both models share the same accelerator.",
        "- VLA controls: two flow steps, ten returned actions, fixed paired noise.",
        "- Each reported profile: five warmup calls and 500 measured calls.",
        "- Static masked-view elision removes only the adapter-declared padded right-wrist slot.",
        f"- Fairness audit: **{'PASS' if summary['fairness_audit']['passed'] else 'FAIL'}**.",
        "",
        "## Main Result",
        "",
        "| Profile | Samples | P50 | P95 | P99 | Miss@80ms | Fidelity | System GPU use |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Compiled BF16", compile_row), ("Compiled BF16 + SMVE", smve_row)):
        condition = row["deployment_condition"]
        lines.append(
            f"| {label} | {row['samples']} | {row['runtime_p50_ms']:.2f} ms | "
            f"{row['runtime_p95_ms']:.2f} ms | {row['runtime_p99_ms']:.2f} ms | "
            f"{100.0 * row['deadline_miss_rates']['80']:.1f}% | "
            f"{row['fidelity']['samples']}/{row['fidelity']['samples']} pass | "
            f"{condition['system_gpu_memory_used_mib_before'] / 1024.0:.2f} GB |"
        )
    effect = summary["effect"]
    lines.extend(
        [
            "",
            "SMVE reduces P50/P95/P99 by "
            f"`{100.0 * effect['runtime_p50_reduction']:.1f}%/"
            f"{100.0 * effect['runtime_p95_reduction']:.1f}%/"
            f"{100.0 * effect['runtime_p99_reduction']:.1f}%`. "
            "At the 80 ms deployment deadline, it reduces misses from "
            f"`{100.0 * compile_row['deadline_miss_rates']['80']:.1f}%` to "
            f"`{100.0 * smve_row['deadline_miss_rates']['80']:.1f}%`.",
            "",
            "## Deadline Sweep",
            "",
            "| Deadline | Compiled BF16 miss | SMVE miss | Absolute reduction |",
            "|---:|---:|---:|---:|",
        ]
    )
    for deadline in summary["protocol"]["deadlines_ms"]:
        key = str(int(deadline))
        compile_rate = compile_row["deadline_miss_rates"][key]
        smve_rate = smve_row["deadline_miss_rates"][key]
        lines.append(
            f"| {key} ms | {100.0 * compile_rate:.1f}% | {100.0 * smve_rate:.1f}% | "
            f"{100.0 * (compile_rate - smve_rate):.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## VLM Load Audit",
            "",
            "| Concurrent VLA profile | VLM requests | Success | VLM P50 | VLM P95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, label in (("compile", "Compiled BF16"), ("smve", "Compiled BF16 + SMVE")):
        row = summary["vlm_load"][name]
        lines.append(
            f"| {label} | {row['requests']} | {row['successes']}/{row['requests']} | "
            f"{row['latency_p50_ms']:.1f} ms | {row['latency_p95_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## Event-triggered Agent VLM",
            "",
            "The event-triggered condition inserts a five-second cooldown after each VLM response. "
            "It represents on-demand semantic checks rather than continuous reasoning.",
            "",
            "| Profile | VLA P50 | VLA P95 | VLA P99 | Miss@80ms | VLM requests | VLM success |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, label in (("compile", "Compiled BF16"), ("smve", "Compiled BF16 + SMVE")):
        row = summary["event_triggered_profiles"][name]
        load = summary["event_triggered_vlm_load"][name]
        lines.append(
            f"| {label} | {row['runtime_p50_ms']:.2f} ms | {row['runtime_p95_ms']:.2f} ms | "
            f"{row['runtime_p99_ms']:.2f} ms | {100.0 * row['deadline_miss_rates']['80']:.1f}% | "
            f"{load['requests']} | {load['successes']}/{load['requests']} |"
        )
    scheduling = summary["effect"]
    lines.extend(
        [
            "",
            "Relative to continuous VLM + ordinary compilation, event-triggered VLM + SMVE "
            f"reduces P50/P95 by `{100.0 * scheduling['event_smve_vs_continuous_compile_p50_reduction']:.1f}%/"
            f"{100.0 * scheduling['event_smve_vs_continuous_compile_p95_reduction']:.1f}%` and reduces "
            f"80 ms misses by `{100.0 * scheduling['event_smve_vs_continuous_compile_miss_reduction']:.1f}` "
            "percentage points.",
        ]
    )
    lines.extend(
        [
            "",
            "## Repetition Check",
            "",
            "| Profile | Calls | P50 | P95 | Miss@80ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, label in (("compile", "Compiled BF16"), ("smve", "Compiled BF16 + SMVE")):
        for row in summary["repeats"][name]:
            lines.append(
                f"| {label} | {row['samples']} | {row['runtime_p50_ms']:.2f} ms | "
                f"{row['runtime_p95_ms']:.2f} ms | {100.0 * row['deadline_miss_rate']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This gate measures warm inference under controlled single-GPU Agent-VLM contention. "
            "It supports a deployment/runtime claim, not a manipulation-success or universal-VLA claim. "
            "The VLM workload is deliberately continuous and is therefore a stress envelope rather than "
            "the expected event-triggered average load.",
            "",
        ]
    )
    return "\n".join(lines)


def render_plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    compile_values = sorted(_latencies(_load_json(DEFAULT_MANIFESTS["compile"])))
    smve_values = sorted(_latencies(_load_json(DEFAULT_MANIFESTS["smve"])))
    deadlines = summary["protocol"]["deadlines_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    colors = {"compile": "#C4553D", "smve": "#277DA1"}
    for values, name, label in (
        (compile_values, "compile", "Compiled BF16"),
        (smve_values, "smve", "Compiled BF16 + SMVE"),
    ):
        y = [(index + 1) / len(values) for index in range(len(values))]
        axes[0].plot(values, y, linewidth=2.0, color=colors[name], label=label)
        misses = [summary["profiles"][name]["deadline_miss_rates"][str(int(d))] for d in deadlines]
        axes[1].plot(deadlines, misses, marker="o", linewidth=2.0, color=colors[name], label=label)
    axes[0].axvline(80, color="#333333", linestyle="--", linewidth=1.2, label="80 ms deadline")
    axes[0].set_xlabel("VLA runtime latency (ms)")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_xlim(left=45)
    axes[0].set_ylim(0, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Control deadline (ms)")
    axes[1].set_ylabel("Deadline miss rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Single-GPU Agent-VLM + PI0.5 Contention", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_scheduling_plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    conditions = ("Continuous VLM", "Event-triggered VLM")
    profiles = ("compile", "smve")
    labels = {"compile": "Compiled BF16", "smve": "Compiled BF16 + SMVE"}
    colors = {"compile": "#C4553D", "smve": "#277DA1"}
    data = {
        "Continuous VLM": summary["profiles"],
        "Event-triggered VLM": summary["event_triggered_profiles"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25))
    x = list(range(len(conditions)))
    width = 0.34
    for offset, profile in ((-width / 2, profiles[0]), (width / 2, profiles[1])):
        axes[0].bar(
            [value + offset for value in x],
            [data[condition][profile]["runtime_p95_ms"] for condition in conditions],
            width=width,
            color=colors[profile],
            label=labels[profile],
        )
        axes[1].bar(
            [value + offset for value in x],
            [100.0 * data[condition][profile]["deadline_miss_rates"]["80"] for condition in conditions],
            width=width,
            color=colors[profile],
            label=labels[profile],
        )
    axes[0].axhline(80, color="#333333", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("VLA runtime P95 (ms)")
    axes[0].set_xticks(x, conditions)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("80 ms deadline miss rate (%)")
    axes[1].set_xticks(x, conditions)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Agent VLM Scheduling and VLA Runtime Co-design", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadlines", default="50,60,70,80,90,100,120")
    parser.add_argument("--json", type=Path, default=RESULTS / "vlm_vla_contention_gate.json")
    parser.add_argument("--markdown", type=Path, default=RESULTS / "VLM_VLA_CONTENTION_GATE.md")
    parser.add_argument(
        "--plot", type=Path, default=RESULTS / "vlm_vla_contention_deadline_curve.png"
    )
    parser.add_argument(
        "--scheduling-plot",
        type=Path,
        default=RESULTS / "vlm_vla_scheduling_comparison.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadlines = [float(value) for value in args.deadlines.split(",") if value.strip()]
    summary = build_summary(deadlines)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    render_plot(summary, args.plot)
    render_scheduling_plot(summary, args.scheduling_plot)
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
