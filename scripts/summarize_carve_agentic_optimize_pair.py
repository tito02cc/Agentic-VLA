#!/usr/bin/env python3
"""Summarize the paired PI0.5 Agentic Harness and Optimize Runtime experiment."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROFILE_ORDER = ("eager_bf16", "compiled_bf16", "compiled_smve")
PROFILE_LABELS = {
    "eager_bf16": "Eager BF16",
    "compiled_bf16": "Compiled BF16",
    "compiled_smve": "Compiled BF16 + SMVE",
}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _verified(branch: dict[str, Any]) -> bool:
    outcome = (branch.get("physical_recovery") or {}).get("outcome", {})
    return bool(outcome.get("verified", False)) or outcome.get("status") == "succeeded"


def summarize(root: pathlib.Path) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    outcome_vectors: list[tuple[bool, ...]] = []
    for profile_name in PROFILE_ORDER:
        profile_dir = root / profile_name
        branches = _load_json(profile_dir / "paired_branches.json")
        traces = _load_jsonl(profile_dir / "paired_branches_policy_calls.jsonl")
        online = _load_json(profile_dir / "online_controller" / "results.json")
        rows = [record["branches"][0] for record in branches["records"]]
        outcomes = tuple(bool(row["success_within_horizon"]) for row in rows)
        outcome_vectors.append(outcomes)
        runtime_ms = [float(trace["runtime_latency_ms"]) for trace in traces]
        model_ms = [float(trace["model_latency_ms"]) for trace in traces]
        misses = sum(bool(trace["deadline_miss"]) for trace in traces)
        optimization_profile = next(
            row.get("optimization_profile") for row in rows if row.get("optimization_profile")
        )
        admission = optimization_profile.get("admission", {})
        online_trace = online.get("trace_aggregate", {})
        online_recovery = online_trace.get("recovery", {})
        profiles[profile_name] = {
            "label": PROFILE_LABELS[profile_name],
            "profile_id": optimization_profile.get("profile", {}).get("profile_id"),
            "admission_accepted": bool(admission.get("accepted", False)),
            "admission_status": admission.get("status"),
            "exact_state_successes": sum(outcomes),
            "exact_state_scenarios": len(rows),
            "exact_state_outcomes": list(outcomes),
            "verified_recoveries": sum(_verified(row) for row in rows),
            "safe_stops": sum(bool(row.get("safe_stop", False)) for row in rows),
            "vla_calls": len(traces),
            "model_p50_ms": _percentile(model_ms, 50),
            "model_p95_ms": _percentile(model_ms, 95),
            "runtime_p50_ms": _percentile(runtime_ms, 50),
            "runtime_p95_ms": _percentile(runtime_ms, 95),
            "deadline_misses": misses,
            "deadline_calls": len(traces),
            "deadline_miss_rate": misses / len(traces),
            "online_successes": int(online.get("total_successes", 0)),
            "online_episodes": int(online.get("total_episodes", 0)),
            "online_physical_triggered": int(
                online_recovery.get("physical_recoveries_triggered_total", 0)
            ),
            "online_physical_verified": int(
                online_recovery.get("physical_recoveries_verified_total", 0)
            ),
            "online_vla_p95_ms": online_trace.get("inference", {}).get(
                "vla_latency_ms_p95"
            ),
            "online_control_miss_rate": online_trace.get("realtime", {}).get(
                "deadline_miss_rate_mean"
            ),
        }

    eager = profiles["eager_bf16"]
    compiled = profiles["compiled_bf16"]
    smve = profiles["compiled_smve"]
    parity = all(vector == outcome_vectors[0] for vector in outcome_vectors[1:])
    complete = bool(
        not eager["admission_accepted"]
        and compiled["admission_accepted"]
        and smve["admission_accepted"]
        and all(row["exact_state_scenarios"] == 2 for row in profiles.values())
    )
    return {
        "experiment": "CARVE-PI0.5-Agentic-Optimize-Pair",
        "claim": (
            "same-state Agentic physical recovery under research-reference eager "
            "BF16 and deployment-admitted compiled BF16/SMVE profiles"
        ),
        "profiles": profiles,
        "comparisons": {
            "exact_state_outcome_parity": parity,
            "compiled_runtime_p95_reduction_vs_eager": (
                1.0 - compiled["runtime_p95_ms"] / eager["runtime_p95_ms"]
            ),
            "smve_runtime_p95_reduction_vs_eager": (
                1.0 - smve["runtime_p95_ms"] / eager["runtime_p95_ms"]
            ),
            "smve_runtime_p95_reduction_vs_compiled": (
                1.0 - smve["runtime_p95_ms"] / compiled["runtime_p95_ms"]
            ),
        },
        "gates": {
            "evidence_complete": complete,
            "outcome_parity": parity,
            "compiled_deadline_compliant": compiled["runtime_p95_ms"] <= 80.0,
            "smve_deadline_compliant": smve["runtime_p95_ms"] <= 80.0,
            "eager_reference_misses_deadline": eager["runtime_p95_ms"] > 80.0,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    gates = summary["gates"]
    lines = [
        "# CARVE PI0.5 Agentic-Optimize Pair",
        "",
        "This experiment restores the same two LIBERO MuJoCo stall states and runs "
        "the same bounded physical-recovery branch under three PI0.5 execution profiles.",
        "Eager BF16 is an explicit unpromoted research reference; compiled profiles "
        "retain their deployment-admission requirements.",
        "",
        "| Profile | Admission | Exact success | Verified recovery | VLA calls | Runtime P50 | Runtime P95 | Miss@80ms | Online lifecycle |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PROFILE_ORDER:
        row = summary["profiles"][name]
        admission = "promoted" if row["admission_accepted"] else "research reference"
        lines.append(
            f"| {row['label']} | {admission} | "
            f"{row['exact_state_successes']}/{row['exact_state_scenarios']} | "
            f"{row['verified_recoveries']}/{row['exact_state_scenarios']} | "
            f"{row['vla_calls']} | {row['runtime_p50_ms']:.2f} ms | "
            f"{row['runtime_p95_ms']:.2f} ms | "
            f"{row['deadline_misses']}/{row['deadline_calls']} | "
            f"{row['online_successes']}/{row['online_episodes']}, recovery "
            f"{row['online_physical_verified']}/{row['online_physical_triggered']} |"
        )
    comparisons = summary["comparisons"]
    lines.extend(
        [
            "",
            "## Paired Conclusions",
            "",
            f"- Exact-state outcome parity: `{'PASS' if comparisons['exact_state_outcome_parity'] else 'FAIL'}`.",
            "- Compiled BF16 runtime-P95 reduction versus eager: "
            f"`{100.0 * comparisons['compiled_runtime_p95_reduction_vs_eager']:.1f}%`.",
            "- Compiled BF16 + SMVE runtime-P95 reduction versus eager: "
            f"`{100.0 * comparisons['smve_runtime_p95_reduction_vs_eager']:.1f}%`.",
            "- SMVE runtime-P95 reduction versus ordinary compiled BF16: "
            f"`{100.0 * comparisons['smve_runtime_p95_reduction_vs_compiled']:.1f}%`.",
            "",
            "## Gates",
            "",
        ]
    )
    for name, passed in gates.items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "The experiment is a coupled systems test, not a benchmark-wide success-rate estimate. "
            "A failed parity or deadline gate is retained as a negative result rather than tuned away.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot(summary: dict[str, Any], output: pathlib.Path) -> None:
    labels = [PROFILE_LABELS[name] for name in PROFILE_ORDER]
    p50 = [summary["profiles"][name]["runtime_p50_ms"] for name in PROFILE_ORDER]
    p95 = [summary["profiles"][name]["runtime_p95_ms"] for name in PROFILE_ORDER]
    misses = [
        100.0 * summary["profiles"][name]["deadline_miss_rate"]
        for name in PROFILE_ORDER
    ]
    x = np.arange(len(labels))
    fig, (ax_latency, ax_miss) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    width = 0.34
    ax_latency.bar(x - width / 2, p50, width, label="P50", color="#6a8290")
    ax_latency.bar(x + width / 2, p95, width, label="P95", color="#3f6da4")
    ax_latency.axhline(80.0, color="#c64e4e", linestyle="--", linewidth=1.5, label="80 ms deadline")
    ax_latency.set_xticks(x, labels, rotation=15, ha="right")
    ax_latency.set_ylabel("Runtime latency (ms)")
    ax_latency.set_title("Same-state Agentic recovery latency")
    ax_latency.legend(frameon=False)
    ax_latency.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax_latency.set_axisbelow(True)

    colors = ["#9b6a6a", "#4c78a8", "#4b9b72"]
    bars = ax_miss.bar(x, misses, color=colors, width=0.58)
    ax_miss.set_xticks(x, labels, rotation=15, ha="right")
    ax_miss.set_ylabel("Deadline misses (%)")
    ax_miss.set_title("80 ms policy-call deadline")
    ax_miss.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax_miss.set_axisbelow(True)
    for bar, value in zip(bars, misses, strict=True):
        ax_miss.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(1.0, max(misses) * 0.02),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_miss.set_ylim(0, max(5.0, max(misses) * 1.15))

    fig.suptitle("CARVE Agentic Harness + Optimize Runtime", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    parser.add_argument("--output-plot", type=pathlib.Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.root)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    plot(summary, args.output_plot)
    print(json.dumps(summary["gates"], indent=2))
    return 0 if summary["gates"]["evidence_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
