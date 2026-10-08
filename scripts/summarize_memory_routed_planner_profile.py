#!/usr/bin/env python3
"""Summarize direct and memory-routed Planner profiles in the staged loop."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def events(workspace: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (workspace / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def aggregate(summary_paths: list[pathlib.Path]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    vla: list[dict[str, Any]] = []
    planner_latencies: list[float] = []
    critic_latencies: list[float] = []
    for path in summary_paths:
        for episode in load(path)["episodes"]:
            workspace = pathlib.Path(episode["workspace"])
            for event in events(workspace):
                payload = event.get("payload", {})
                if event.get("event_type") == "planner_result" and payload.get("accepted"):
                    planner_latencies.append(float(payload["elapsed_ms"]))
                if (
                    event.get("event_type") == "tool_result"
                    and payload.get("tool") == "vla_act"
                    and payload.get("accepted")
                ):
                    vla.append(dict(payload["output"]["primitive_outcome"]["metadata"]))
            critic_latencies.extend(
                float(item["elapsed_ms"])
                for item in episode.get("visual_critic_receipts", [])
                if item.get("accepted")
            )
            episodes.append(
                {
                    "trial": int(episode["trial"]),
                    "success": bool(episode["success"]),
                    "episode_steps": int(episode["episode_steps"]),
                    "elapsed_s": float(episode["elapsed_s"]),
                    "planner_calls": int(episode["planner_calls"]),
                    "critic_calls": int(episode["visual_critic"]["calls"]),
                    "recovery_attempts": int(episode["recovery_attempts"]),
                    "task_plan_completed": bool(episode["task_plan"]["completed"]),
                    "procedure_warm_start_id": episode.get("procedure_warm_start_id"),
                    "workspace": str(workspace),
                    "video": str(episode["video"]),
                }
            )
    runtime = [float(item["runtime_latency_ms"]) for item in vla]
    return {
        "episodes": episodes,
        "metrics": {
            "successes": sum(item["success"] for item in episodes),
            "trials": len(episodes),
            "task_plans_completed": sum(item["task_plan_completed"] for item in episodes),
            "wall_time_mean_s": statistics.mean(item["elapsed_s"] for item in episodes),
            "planner_calls": sum(item["planner_calls"] for item in episodes),
            "planner_latency_total_ms": sum(planner_latencies),
            "planner_latency_mean_ms": (
                statistics.mean(planner_latencies) if planner_latencies else 0.0
            ),
            "critic_calls": sum(item["critic_calls"] for item in episodes),
            "critic_latency_mean_ms": (
                statistics.mean(critic_latencies) if critic_latencies else 0.0
            ),
            "critic_latency_p95_ms": (
                percentile(critic_latencies, 0.95) if critic_latencies else 0.0
            ),
            "recovery_attempts": sum(item["recovery_attempts"] for item in episodes),
            "vla_calls": len(vla),
            "vla_runtime_mean_ms": statistics.mean(runtime),
            "vla_runtime_p95_ms": percentile(runtime, 0.95),
            "vla_runtime_max_ms": max(runtime),
            "vla_deadline_misses": sum(bool(item["deadline_miss"]) for item in vla),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-summary", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--reference-aggregate", type=pathlib.Path, required=True)
    parser.add_argument("--direct-summary", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    memory_all = aggregate(args.memory_summary)
    reference = load(args.reference_aggregate)
    common_trials = set(int(value) for value in reference["protocol"]["trials"])
    common_paths = [
        path
        for path in args.memory_summary
        if int(load(path)["episodes"][0]["trial"]) in common_trials
    ]
    memory_common = aggregate(common_paths)
    direct_episode = load(args.direct_summary)["episodes"][0]
    direct_workspace = pathlib.Path(direct_episode["workspace"])
    direct_planner = next(
        event["payload"]
        for event in events(direct_workspace)
        if event.get("event_type") == "planner_result"
    )
    reference_metrics = reference["aggregate"]
    memory_metrics = memory_common["metrics"]
    reference_planner_calls = sum(
        int(episode["planner_calls"]) for episode in reference["episodes"]
    )
    reference_planner_wait_ms = (
        float(reference_metrics["accepted_planner_latency_mean_ms"])
        * reference_planner_calls
    )
    payload = {
        "schema_version": "carve-memory-routed-planner-profile-v1",
        "protocol": {
            "benchmark": "LIBERO-PRO libero_10_object task 8",
            "common_trials": sorted(common_trials),
            "extra_memory_trial": 1,
            "vla_profile": "pi05-torch_compile_masked_views-bf16-2step-h10",
            "deadline_ms": 80.0,
        },
        "direct_qwen4b_bf16": {
            "success": bool(direct_episode["success"]),
            "planner_accepted": bool(direct_planner["accepted"]),
            "attempt_count": int(direct_planner["attempt_count"]),
            "elapsed_ms": float(direct_planner["elapsed_ms"]),
            "error": direct_planner["error"],
            "decision": "rejected_as_direct_long_plan_profile",
        },
        "qwen9b_nf4_reference": reference,
        "memory_routed_qwen4b_bf16_common": memory_common,
        "memory_routed_qwen4b_bf16_all": memory_all,
        "qwen4b_bf16_receipt": load(args.receipt),
        "paired_system_changes": {
            "success_delta": memory_metrics["successes"] - reference_metrics["successes"],
            "planner_calls_delta": (
                memory_metrics["planner_calls"] - reference_planner_calls
            ),
            "planner_wait_reduction_percent": 100.0 * (
                1.0
                - memory_metrics["planner_latency_total_ms"]
                / max(1.0, reference_planner_wait_ms)
            ),
            "wall_time_reduction_percent": 100.0 * (
                1.0 - memory_metrics["wall_time_mean_s"] / reference_metrics["wall_time_mean_s"]
            ),
            "vla_p95_change_percent": 100.0 * (
                memory_metrics["vla_runtime_p95_ms"]
                / reference_metrics["vla_runtime_latency_p95_ms"]
                - 1.0
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_system_changes"], indent=2))
    print(json.dumps(memory_all["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
