"""Summarize realtime deadline stress from episode trace JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_GROUPS = {
    "B0-VLA": [
        PROJECT_ROOT / "results" / f"joint2x2_b0_vla_replan5_noLight_midNudge003_task2_10trials_seed{seed}_20260608" / "episode_traces.jsonl"
        for seed in (7, 17, 27)
    ],
    "B0-VLA-Light": [
        PROJECT_ROOT / "results" / f"joint2x2_b0_vla_light_safe80_replan5_reuseMax2_midNudge003_task2_10trials_seed{seed}_20260608" / "episode_traces.jsonl"
        for seed in (7, 17, 27)
    ],
    "B4-Agentic": [
        PROJECT_ROOT / "results" / f"b4_agentic_contextGate_replan5_noLight_midNudge_task2_10trials_seed{seed}_20260608" / "episode_traces.jsonl"
        for seed in (7, 17, 27)
    ],
    "B4-Agentic-Light": [
        PROJECT_ROOT / "results" / f"b4_agentic_light_safeLockout80_contextGate_replan5_reuseMax2_midNudge_task2_10trials_seed{seed}_20260608" / "episode_traces.jsonl"
        for seed in (7, 17, 27)
    ],
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize_group(traces: list[dict[str, Any]], deadlines: list[float]) -> dict[str, Any]:
    latencies: list[float] = []
    full_calls: list[float] = []
    skipped_calls: list[float] = []
    episode_wall: list[float] = []
    successes = 0

    deadline_rows: dict[float, dict[str, float]] = {}
    for deadline in deadlines:
        miss_rates: list[float] = []
        sud_values: list[float] = []
        for trace in traces:
            realtime = trace.get("realtime", {}) or {}
            latency = trace.get("latency", {}) or {}
            values = [float(v) for v in latency.get("vla_latency_ms_values", []) or []]
            control_steps = int(realtime.get("control_steps") or trace.get("episode_steps") or 0)
            misses = sum(1 for value in values if value > deadline)
            miss_rate = float(misses / control_steps) if control_steps > 0 else 0.0
            success = 1.0 if bool(trace.get("success")) else 0.0
            miss_rates.append(miss_rate)
            sud_values.append(success * (1.0 - miss_rate))
        deadline_rows[deadline] = {
            "deadline_miss": float(_mean(miss_rates) or 0.0),
            "sud": float(_mean(sud_values) or 0.0),
        }

    for trace in traces:
        latency = trace.get("latency", {}) or {}
        light = trace.get("lightweight", {}) or {}
        latencies.extend(float(v) for v in latency.get("vla_latency_ms_values", []) or [])
        full_calls.append(float(trace.get("full_vla_calls") or light.get("full_vla_calls") or 0.0))
        skipped_calls.append(float(trace.get("skipped_vla_calls") or light.get("skipped_vla_calls") or 0.0))
        if latency.get("episode_wall_sec") is not None:
            episode_wall.append(float(latency["episode_wall_sec"]))
        successes += int(bool(trace.get("success")))

    return {
        "episodes": len(traces),
        "successes": successes,
        "success_rate": successes / len(traces) if traces else 0.0,
        "full_calls": _mean(full_calls),
        "skipped_calls": _mean(skipped_calls),
        "episode_wall": _mean(episode_wall),
        "vla_p50": _percentile(latencies, 0.50),
        "vla_p95": _percentile(latencies, 0.95),
        "vla_p99": _percentile(latencies, 0.99),
        "deadlines": deadline_rows,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deadlines",
        default="50,80,100,200,400,500",
        help="Comma-separated deadline thresholds in milliseconds.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=TRACE[,TRACE...]",
        help="Custom group definition. Can be repeated. Defaults to the Task2 joint2x2 groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadlines = [float(item) for item in str(args.deadlines).split(",") if item.strip()]

    groups: dict[str, list[Path]]
    if args.group:
        groups = {}
        for item in args.group:
            name, raw_paths = item.split("=", 1)
            groups[name] = [Path(path) for path in raw_paths.split(",") if path]
    else:
        groups = DEFAULT_GROUPS

    summaries: dict[str, dict[str, Any]] = {}
    for name, paths in groups.items():
        traces: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            traces.extend(_load_jsonl(path))
        summaries[name] = _summarize_group(traces, deadlines)

    base_headers = [
        "Method",
        "Success",
        "Full calls/ep",
        "Skipped/ep",
        "Wall sec/ep",
        "VLA p50",
        "VLA p95",
        "VLA p99",
    ]
    deadline_headers = []
    for deadline in deadlines:
        label = str(int(deadline)) if float(deadline).is_integer() else str(deadline)
        deadline_headers.extend([f"Miss@{label}ms", f"SUD@{label}ms"])

    headers = base_headers + deadline_headers
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for name, summary in summaries.items():
        cells = [
            name,
            f"{summary['successes']}/{summary['episodes']}",
            _fmt(summary["full_calls"], 2),
            _fmt(summary["skipped_calls"], 2),
            _fmt(summary["episode_wall"], 2),
            _fmt(summary["vla_p50"], 1),
            _fmt(summary["vla_p95"], 1),
            _fmt(summary["vla_p99"], 1),
        ]
        for deadline in deadlines:
            row = summary["deadlines"][deadline]
            cells.append(_fmt(row["deadline_miss"], 3))
            cells.append(_fmt(row["sud"], 3))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
