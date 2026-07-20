#!/usr/bin/env python3
"""Audit and summarize the reproducible CARVE LIBERO-10 comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


VARIANTS = ("baseline", "agentic")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / scale
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mcnemar_exact(baseline_only: int, agentic_only: int) -> float:
    discordant = baseline_only + agentic_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, agentic_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _episode_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["episode_idx"])


def _profile_value(row: dict[str, Any], key: str) -> Any:
    profile = row.get("metadata", {}).get("optimization_profile", {}).get("profile", {})
    return profile.get(key)


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted(set(values), key=lambda value: str(value))


def _audit_calls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error") is not None]
    return {
        "calls": len(rows),
        "profile_ids": _sorted_unique(_profile_value(row, "profile_id") for row in rows),
        "backends": _sorted_unique(_profile_value(row, "backend") for row in rows),
        "inference_steps": _sorted_unique(
            row.get("applied_controls", {}).get("inference_steps") for row in rows
        ),
        "max_actions": _sorted_unique(
            row.get("applied_controls", {}).get("max_actions") for row in rows
        ),
        "policy_deadline_misses": sum(bool(row.get("deadline_miss")) for row in rows),
        "errors": len(errors),
    }


def _variant_metrics(
    result: dict[str, Any], traces: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in traces)
    episodes = len(traces)
    vla_latencies = [
        float(value)
        for row in traces
        for value in row.get("latency", {}).get("vla_latency_ms_values", [])
    ]
    episode_wall = [float(row["latency"]["episode_wall_sec"]) for row in traces]
    control_steps = sum(int(row["realtime"]["control_steps"]) for row in traces)
    deadline_misses = sum(int(row["realtime"]["deadline_misses"]) for row in traces)
    return {
        "status": result.get("status"),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_wilson_95": _wilson(successes, episodes),
        "episode_steps_total": sum(int(row["episode_steps"]) for row in traces),
        "vla_calls_total": sum(int(row["vla_calls"]) for row in traces),
        "vla_calls_per_episode": mean(float(row["vla_calls"]) for row in traces),
        "vla_latency_ms_p50": median(vla_latencies),
        "vla_latency_ms_p95": _percentile(vla_latencies, 0.95),
        "episode_wall_sec_mean": mean(episode_wall),
        "control_deadline_misses": deadline_misses,
        "control_steps": control_steps,
        "control_deadline_miss_rate": deadline_misses / control_steps,
        "recoveries_triggered": sum(int(row.get("recoveries_triggered", 0)) for row in traces),
        "recoveries_successful": sum(int(row.get("recoveries_successful", 0)) for row in traces),
        "physical_recoveries_triggered": sum(
            int(row.get("physical_recoveries_triggered", 0)) for row in traces
        ),
        "physical_recoveries_verified": sum(
            int(row.get("physical_recoveries_verified", 0)) for row in traces
        ),
        "physical_recovery_actions": sum(
            int(row.get("physical_recovery_actions", 0)) for row in traces
        ),
        "peak_gpu_mem_gb": max(float(row.get("gpu", {}).get("peak_mem_gb", 0.0)) for row in traces),
        "policy_call_audit": _audit_calls(calls),
    }


def _task_table(
    baseline: dict[tuple[int, int], dict[str, Any]],
    agentic: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    task_ids = sorted({task_id for task_id, _ in baseline})
    for task_id in task_ids:
        keys = sorted(key for key in baseline if key[0] == task_id)
        baseline_successes = sum(bool(baseline[key]["success"]) for key in keys)
        agentic_successes = sum(bool(agentic[key]["success"]) for key in keys)
        rows.append(
            {
                "task_id": task_id,
                "episodes": len(keys),
                "baseline_successes": baseline_successes,
                "agentic_successes": agentic_successes,
                "delta_successes": agentic_successes - baseline_successes,
            }
        )
    return rows


def summarize(run_root: Path) -> dict[str, Any]:
    raw: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        root = run_root / variant
        raw[variant] = {
            "result": _load_json(root / "results.json"),
            "traces": _load_jsonl(root / "episode_traces.jsonl"),
            "calls": _load_jsonl(root / "policy_calls.jsonl"),
        }

    indexed = {
        variant: {_episode_key(row): row for row in raw[variant]["traces"]}
        for variant in VARIANTS
    }
    duplicate_counts = {
        variant: len(raw[variant]["traces"]) - len(indexed[variant]) for variant in VARIANTS
    }
    baseline_keys = set(indexed["baseline"])
    agentic_keys = set(indexed["agentic"])
    if duplicate_counts != {"baseline": 0, "agentic": 0}:
        raise ValueError(f"Duplicate episode keys detected: {duplicate_counts}")
    if baseline_keys != agentic_keys:
        raise ValueError(
            "Episode pairing mismatch: "
            f"baseline_only={sorted(baseline_keys - agentic_keys)}, "
            f"agentic_only={sorted(agentic_keys - baseline_keys)}"
        )

    paired = Counter()
    for key in sorted(baseline_keys):
        baseline_success = bool(indexed["baseline"][key]["success"])
        agentic_success = bool(indexed["agentic"][key]["success"])
        if baseline_success and agentic_success:
            paired["both_success"] += 1
        elif baseline_success:
            paired["baseline_only"] += 1
        elif agentic_success:
            paired["agentic_only"] += 1
        else:
            paired["both_fail"] += 1

    variants = {
        variant: _variant_metrics(
            raw[variant]["result"], raw[variant]["traces"], raw[variant]["calls"]
        )
        for variant in VARIANTS
    }
    comparable_fields = ("profile_ids", "backends", "inference_steps", "max_actions")
    mismatches = [
        field
        for field in comparable_fields
        if variants["baseline"]["policy_call_audit"][field]
        != variants["agentic"]["policy_call_audit"][field]
    ]
    errors = sum(
        int(variants[variant]["policy_call_audit"]["errors"]) for variant in VARIANTS
    )
    fairness_passed = not mismatches and errors == 0

    total = len(baseline_keys)
    baseline_rate = float(variants["baseline"]["success_rate"])
    agentic_rate = float(variants["agentic"]["success_rate"])
    return {
        "run_root": str(run_root.resolve()),
        "fairness_audit": {
            "passed": fairness_passed,
            "paired_episodes": total,
            "configuration_mismatches": mismatches,
            "policy_errors": errors,
        },
        "variants": variants,
        "paired_outcomes": dict(paired),
        "effect": {
            "absolute_success_rate_delta": agentic_rate - baseline_rate,
            "relative_success_rate_delta": (
                agentic_rate / baseline_rate - 1.0 if baseline_rate > 0.0 else None
            ),
            "mcnemar_exact_p": _mcnemar_exact(
                paired["baseline_only"], paired["agentic_only"]
            ),
        },
        "tasks": _task_table(indexed["baseline"], indexed["agentic"]),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["variants"]["baseline"]
    agentic = summary["variants"]["agentic"]
    paired = summary["paired_outcomes"]
    lines = [
        "# CARVE LIBERO-10 Reproducible Gate",
        "",
        f"- Fairness audit: **{'PASS' if summary['fairness_audit']['passed'] else 'FAIL'}**",
        f"- Paired episodes: **{summary['fairness_audit']['paired_episodes']}**",
        f"- Baseline: **{baseline['successes']}/{baseline['episodes']} ({_pct(baseline['success_rate'])})**",
        f"- CARVE Agentic: **{agentic['successes']}/{agentic['episodes']} ({_pct(agentic['success_rate'])})**",
        f"- Absolute delta: **{100.0 * summary['effect']['absolute_success_rate_delta']:+.1f} pp**",
        f"- Exact paired McNemar p: **{summary['effect']['mcnemar_exact_p']:.4g}**",
        "",
        "## Per-task Success",
        "",
        "| Task | Episodes | Baseline | CARVE Agentic | Delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task_id']} | {row['episodes']} | {row['baseline_successes']} | "
            f"{row['agentic_successes']} | {row['delta_successes']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Paired Outcomes",
            "",
            f"- Both succeed: {paired.get('both_success', 0)}",
            f"- Baseline only: {paired.get('baseline_only', 0)}",
            f"- CARVE Agentic only: {paired.get('agentic_only', 0)}",
            f"- Both fail: {paired.get('both_fail', 0)}",
            "",
            "## Runtime and Recovery",
            "",
            "| Metric | Baseline | CARVE Agentic |",
            "|---|---:|---:|",
            f"| VLA calls / episode | {baseline['vla_calls_per_episode']:.2f} | {agentic['vla_calls_per_episode']:.2f} |",
            f"| VLA latency P50 (ms) | {baseline['vla_latency_ms_p50']:.2f} | {agentic['vla_latency_ms_p50']:.2f} |",
            f"| VLA latency P95 (ms) | {baseline['vla_latency_ms_p95']:.2f} | {agentic['vla_latency_ms_p95']:.2f} |",
            f"| Episode wall time (s) | {baseline['episode_wall_sec_mean']:.2f} | {agentic['episode_wall_sec_mean']:.2f} |",
            f"| Physical recoveries | {baseline['physical_recoveries_triggered']} | {agentic['physical_recoveries_triggered']} |",
            f"| Verified recoveries | {baseline['physical_recoveries_verified']} | {agentic['physical_recoveries_verified']} |",
            f"| Peak GPU memory (GB) | {baseline['peak_gpu_mem_gb']:.2f} | {agentic['peak_gpu_mem_gb']:.2f} |",
            "",
            "## Configuration Audit",
            "",
            "```json",
            json.dumps(
                {
                    variant: summary["variants"][variant]["policy_call_audit"]
                    for variant in VARIANTS
                },
                indent=2,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--json", type=Path, help="Output JSON path; defaults under run root")
    parser.add_argument("--markdown", type=Path, help="Output Markdown path; defaults under run root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.run_root)
    json_path = args.json or args.run_root / "comparison.json"
    markdown_path = args.markdown or args.run_root / "COMPARISON.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    if not summary["fairness_audit"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
