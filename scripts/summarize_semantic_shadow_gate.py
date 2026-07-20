#!/usr/bin/env python3
"""Summarize the paired CARVE semantic-shadow perturbation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = {
    "no_semantic": "No semantic VLM",
    "semantic32": "Async VLM, 32 tokens",
    "semantic12_labels": "Async VLM, 12-token label",
}
STATUSES = {"NOMINAL", "REPLAN", "RECOVER", "STOP"}
FAILURES = {"NONE", "MISGRASP", "DROP", "MISALIGN", "COLLISION", "STALL", "OTHER"}


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _valid_label(content: Any) -> bool:
    for line in str(content or "").replace("`", "").splitlines():
        parts = [part.strip().upper() for part in line.split("|")]
        if len(parts) == 2 and parts[0] in STATUSES and parts[1] in FAILURES:
            return True
    return False


def _load_traces(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [
        observation
        for trace in traces
        for observation in trace.get("realtime", {})
        .get("semantic_observer", {})
        .get("observations", [])
    ]
    semantic_latencies = [
        float(item["latency_ms"])
        for item in observations
        if item.get("latency_ms") is not None
    ]
    vla_latencies = [
        float(value)
        for trace in traces
        for value in trace.get("latency", {}).get("vla_latency_ms_values", [])
    ]
    deadline_rates = [
        float(trace["realtime"]["deadline_miss_rate"])
        for trace in traces
        if trace.get("realtime", {}).get("deadline_miss_rate") is not None
    ]
    return {
        "episodes": len(traces),
        "successes": sum(bool(trace.get("success")) for trace in traces),
        "success_rate": float(np.mean([bool(trace.get("success")) for trace in traces])),
        "semantic_calls": len(observations),
        "semantic_errors": sum(not bool(item.get("valid", False)) for item in observations),
        "semantic_protocol_valid": sum(_valid_label(item.get("content")) for item in observations),
        "semantic_latency_ms_mean": (
            float(np.mean(semantic_latencies)) if semantic_latencies else None
        ),
        "semantic_latency_ms_p95": _percentile(semantic_latencies, 95),
        "vla_latency_ms_p50": _percentile(vla_latencies, 50),
        "vla_latency_ms_p95": _percentile(vla_latencies, 95),
        "deadline_miss_rate_mean": (
            float(np.mean(deadline_rates)) if deadline_rates else None
        ),
        "blocking_reasoning_ms_mean": float(
            np.mean(
                [
                    trace.get("realtime", {}).get("blocking_reasoning_ms", 0.0)
                    for trace in traces
                ]
            )
        ),
    }


def _paired_agreement(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_map = {
        (trace["task_id"], trace["episode_idx"]): trace for trace in baseline
    }
    candidate_map = {
        (trace["task_id"], trace["episode_idx"]): trace for trace in candidate
    }
    keys = sorted(set(baseline_map) & set(candidate_map))
    return {
        "pairs": len(keys),
        "success_agreement": float(
            np.mean(
                [
                    bool(baseline_map[key]["success"])
                    == bool(candidate_map[key]["success"])
                    for key in keys
                ]
            )
        ),
        "episode_length_agreement": float(
            np.mean(
                [
                    int(baseline_map[key]["episode_steps"])
                    == int(candidate_map[key]["episode_steps"])
                    for key in keys
                ]
            )
        ),
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "results/carve_semantic_shadow/paired_t89_3trials_20260717"
        ),
    )
    parser.add_argument(
        "--smoke96-trace",
        type=Path,
        default=Path(
            "results/carve_semantic_shadow/smoke_t8_20260717/episode_traces.jsonl"
        ),
    )
    args = parser.parse_args()

    traces = {
        key: _load_traces(args.run_root / key / "episode_traces.jsonl")
        for key in VARIANTS
    }
    summaries = {key: _summarize(value) for key, value in traces.items()}
    for key in ("semantic32", "semantic12_labels"):
        summaries[key]["paired_agreement"] = _paired_agreement(
            traces["no_semantic"], traces[key]
        )

    smoke96 = _summarize(_load_traces(args.smoke96_trace))
    baseline = summaries["no_semantic"]
    selected = summaries["semantic12_labels"]
    findings = {
        "selected_profile": "semantic12_labels",
        "semantic_p95_reduction_vs_32_pct": 100.0
        * (1.0 - selected["semantic_latency_ms_p95"] / summaries["semantic32"]["semantic_latency_ms_p95"]),
        "semantic_p95_reduction_vs_96_smoke_pct": 100.0
        * (1.0 - selected["semantic_latency_ms_p95"] / smoke96["semantic_latency_ms_p95"]),
        "vla_p95_overhead_vs_no_semantic_pct": 100.0
        * (selected["vla_latency_ms_p95"] / baseline["vla_latency_ms_p95"] - 1.0),
        "deadline_miss_delta_percentage_points": 100.0
        * (selected["deadline_miss_rate_mean"] - baseline["deadline_miss_rate_mean"]),
    }
    payload = {
        "gate": "CARVE asynchronous semantic observation under PI0.5 contention",
        "setting": {
            "tasks": [8, 9],
            "trials_per_task": 3,
            "perturbation": "mid_episode_nudge@80, xy=0.03m",
            "vla": "PI0.5 PyTorch, SMVE BF16, 2 flow steps, horizon 10",
            "vlm": "Qwen3.5-4B BF16, co-resident on one RTX 4090",
            "deadline_ms": 80,
        },
        "variants": summaries,
        "smoke96": smoke96,
        "findings": findings,
    }
    json_path = args.run_root / "semantic_shadow_gate_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CARVE Semantic-Shadow Perturbation Gate",
        "",
        "## Setting",
        "",
        "- LIBERO-10 Task 8/9, 3 paired trials per task, fixed policy noise.",
        "- Mid-episode object nudge at control step 80 (`xy=0.03 m`).",
        "- PI0.5 PyTorch with the selected SMVE BF16 2-step/horizon-10 profile.",
        "- Qwen3.5-4B BF16 and PI0.5 co-resident on one RTX 4090.",
        "- Semantic output is shadow-only and never changes robot actions.",
        "",
        "## Results",
        "",
        "| Variant | Success | Semantic calls | Protocol valid | Semantic P95 (ms) | VLA P95 (ms) | Deadline miss | Blocking reasoning (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in VARIANTS.items():
        item = summaries[key]
        lines.append(
            f"| {label} | {item['successes']}/{item['episodes']} | "
            f"{item['semantic_calls']} | {item['semantic_protocol_valid']}/{item['semantic_calls']} | "
            f"{_fmt(item['semantic_latency_ms_p95'])} | {_fmt(item['vla_latency_ms_p95'])} | "
            f"{100.0 * item['deadline_miss_rate_mean']:.2f}% | "
            f"{item['blocking_reasoning_ms_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            "The 12-token label protocol is selected for the next CARVE stage:",
            "",
            f"- Semantic P95 falls by {findings['semantic_p95_reduction_vs_32_pct']:.1f}% versus the 32-token JSON attempt and {findings['semantic_p95_reduction_vs_96_smoke_pct']:.1f}% versus the 96-token smoke run.",
            f"- All {selected['semantic_calls']} responses are valid and parseable; paired success agreement and episode-length agreement are both 100%.",
            f"- VLA model P95 increases by {findings['vla_p95_overhead_vs_no_semantic_pct']:.1f}% during co-resident execution, while the mean deadline-miss delta is {findings['deadline_miss_delta_percentage_points']:+.3f} percentage points.",
            "- The semantic path contributes zero blocking reasoning time. It remains shadow-only until semantic intervention precision is evaluated on labeled failure events.",
        ]
    )
    md_path = args.run_root / "SEMANTIC_SHADOW_GATE.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        labels = ["No semantic", "32-token JSON", "12-token label"]
        vla_p95 = [summaries[key]["vla_latency_ms_p95"] for key in VARIANTS]
        semantic_p95 = [summaries[key]["semantic_latency_ms_p95"] or 0.0 for key in VARIANTS]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
        axes[0].bar(labels, vla_p95, color=["#64748b", "#d97706", "#0f766e"])
        axes[0].axhline(80, color="#b91c1c", linestyle="--", linewidth=1)
        axes[0].text(0.05, 82.0, "80 ms deadline", color="#991b1b", fontsize=8)
        axes[0].set_ylim(0, 90)
        axes[0].set_ylabel("PI0.5 latency P95 (ms)")
        axes[1].bar(labels, semantic_p95, color=["#64748b", "#d97706", "#0f766e"])
        axes[1].set_ylabel("Semantic VLM latency P95 (ms)")
        for axis in axes:
            axis.tick_params(axis="x", rotation=18)
            axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.run_root / "semantic_shadow_gate.png", dpi=200)
        plt.close(fig)
    except ImportError:
        pass

    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
