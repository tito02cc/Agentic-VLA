#!/usr/bin/env python3
"""Aggregate trace-backed RoboDojo B0/C1 Optimize Runtime pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "carve.robodojo.nominal.v1":
        raise ValueError(f"unsupported summary schema: {path}")
    return payload


def _index(paths: list[Path]) -> dict[int, dict]:
    indexed = {}
    for path in paths:
        payload = _load(path)
        seed = int(payload["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate seed {seed}: {path}")
        indexed[seed] = payload
    return indexed


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return the two-sided Wilson score interval as probabilities."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _runtime_profile(runtime: dict) -> str:
    steps = runtime.get("observed_inference_steps")
    horizons = runtime.get("observed_execute_horizons")
    if not horizons:
        return "trace-backed runtime profile (controls unavailable)"
    horizon_label = "/".join(str(value) for value in horizons)
    if not runtime.get("inference_step_control_verified"):
        return (
            "native inference steps (requested step control unverified), "
            f"execute horizon {horizon_label}"
        )
    if not steps:
        return f"verified inference-step control (value unavailable), execute horizon {horizon_label}"
    step_label = "/".join(str(value) for value in steps)
    parameters = runtime.get("observed_inference_step_parameters") or []
    step_family = "Flow" if parameters == ["num_inference_steps"] else "DDIM"
    return f"{step_family}{step_label}, execute horizon {horizon_label}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--runtime", type=Path, action="append", required=True)
    parser.add_argument("--baseline-horizon", type=int, default=16)
    parser.add_argument("--baseline-profile-label", default="native Flow4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline_horizon <= 0:
        raise ValueError("baseline horizon must be positive")

    baselines = _index(args.baseline)
    runtimes = _index(args.runtime)
    if set(baselines) != set(runtimes):
        raise ValueError("baseline and runtime summaries must have identical seeds")
    runtime_profiles = {
        _runtime_profile(payload["runtime"]) for payload in runtimes.values()
    }
    if len(runtime_profiles) != 1:
        raise ValueError(
            "all runtime summaries in one ablation must use the same observed profile"
        )
    runtime_profile = next(iter(runtime_profiles))

    rows = []
    for seed in sorted(baselines):
        baseline = baselines[seed]
        runtime = runtimes[seed]
        if baseline["task"] != runtime["task"]:
            raise ValueError(f"task mismatch for seed {seed}")
        if baseline["condition"] != "B0" or runtime["condition"] != "C1":
            raise ValueError(f"expected B0/C1 pair for seed {seed}")
        runtime_calls = runtime["runtime"]["vla_calls"]
        if runtime_calls is None or not runtime["runtime"]["trace_available"]:
            raise ValueError(f"C1 seed {seed} requires an observed runtime trace")
        frames = int(baseline["official_result"]["video_frames"])
        estimated_baseline_calls = math.ceil((frames - 1) / args.baseline_horizon)
        rows.append(
            {
                "seed": seed,
                "task": baseline["task"],
                "baseline_success": baseline["official_result"]["success"],
                "runtime_success": runtime["official_result"]["success"],
                "baseline_score": baseline["official_result"]["score"],
                "runtime_score": runtime["official_result"]["score"],
                "baseline_vla_calls_estimated": estimated_baseline_calls,
                "runtime_vla_calls_observed": runtime_calls,
                "call_reduction_percent": 100.0
                * (estimated_baseline_calls - runtime_calls)
                / estimated_baseline_calls,
                "runtime_p50_ms": runtime["runtime"]["model_latency_p50_ms"],
                "runtime_p95_ms": runtime["runtime"]["model_latency_p95_ms"],
                "runtime_first_call_ms": runtime["runtime"].get(
                    "first_call_latency_ms"
                ),
                "runtime_steady_p50_ms": runtime["runtime"].get(
                    "steady_state_model_latency_p50_ms"
                ),
                "runtime_steady_p95_ms": runtime["runtime"].get(
                    "steady_state_model_latency_p95_ms"
                ),
                "runtime_total_model_ms": runtime["runtime"].get(
                    "total_model_latency_ms"
                ),
                "runtime_inference_steps": runtime["runtime"].get(
                    "observed_inference_steps"
                ),
                "runtime_execute_horizons": runtime["runtime"].get(
                    "observed_execute_horizons"
                ),
                "runtime_inference_step_control_verified": runtime["runtime"].get(
                    "inference_step_control_verified"
                ),
            }
        )

    baseline_calls = sum(row["baseline_vla_calls_estimated"] for row in rows)
    runtime_calls = sum(row["runtime_vla_calls_observed"] for row in rows)
    episode_count = len(rows)
    baseline_successes = sum(row["baseline_success"] for row in rows)
    runtime_successes = sum(row["runtime_success"] for row in rows)
    recovered = sum(
        not row["baseline_success"] and row["runtime_success"] for row in rows
    )
    regressed = sum(
        row["baseline_success"] and not row["runtime_success"] for row in rows
    )
    output = {
        "schema_version": "carve.robodojo.optimize-ablation.v1",
        "conditions": {
            "B0": f"{args.baseline_profile_label}, execute horizon {args.baseline_horizon}",
            "C1": runtime_profile,
        },
        "rows": rows,
        "aggregate": {
            "episodes_per_condition": episode_count,
            "baseline_successes": baseline_successes,
            "runtime_successes": runtime_successes,
            "baseline_success_rate": baseline_successes / episode_count,
            "runtime_success_rate": runtime_successes / episode_count,
            "baseline_success_rate_wilson95": _wilson_interval(
                baseline_successes, episode_count
            ),
            "runtime_success_rate_wilson95": _wilson_interval(
                runtime_successes, episode_count
            ),
            "success_rate_delta_percentage_points": 100.0
            * (runtime_successes - baseline_successes)
            / episode_count,
            "baseline_failure_to_runtime_success": recovered,
            "baseline_success_to_runtime_failure": regressed,
            "baseline_mean_video_frames": mean(
                int(baselines[seed]["official_result"]["video_frames"])
                for seed in sorted(baselines)
            ),
            "runtime_mean_video_frames": mean(
                int(runtimes[seed]["official_result"]["video_frames"])
                for seed in sorted(runtimes)
            ),
            "baseline_vla_calls_estimated": baseline_calls,
            "runtime_vla_calls_observed": runtime_calls,
            "call_reduction_percent": 100.0
            * (baseline_calls - runtime_calls)
            / baseline_calls,
        },
        "claim_boundary": (
            "B0 calls are estimated from control frames and the configured baseline "
            "horizon because official baseline traces are unavailable. C1 calls and "
            "latencies are observed. This small block is deployment evidence, not a "
            "statistical task-success claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
