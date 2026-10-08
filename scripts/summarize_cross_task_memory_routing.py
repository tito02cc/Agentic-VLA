#!/usr/bin/env python3
"""Aggregate the factorized cross-task procedure-memory routing study."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def load_events(workspace: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (workspace / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def aggregate(paths: list[pathlib.Path]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    planner_latencies: list[float] = []
    critic_latencies: list[float] = []
    vla_rows: list[dict[str, Any]] = []
    startup_planner_calls = 0
    event_planner_calls = 0
    for path in paths:
        summary = load(path)["episodes"][0]
        workspace = pathlib.Path(summary["workspace"])
        events = load_events(workspace)
        submitted_timesteps = {
            str(event["payload"]["ticket_id"]): int(event["payload"]["timestep"])
            for event in events
            if event.get("event_type") == "planner_submitted"
        }
        for event in events:
            payload = event.get("payload", {})
            if event.get("event_type") == "planner_result":
                calls = int(payload["attempt_count"])
                planner_latencies.append(float(payload["elapsed_ms"]))
                if submitted_timesteps.get(str(payload["ticket_id"])) == 0:
                    startup_planner_calls += calls
                else:
                    event_planner_calls += calls
            if (
                event.get("event_type") == "tool_result"
                and payload.get("tool") == "vla_act"
                and payload.get("accepted")
            ):
                vla_rows.append(
                    dict(payload["output"]["primitive_outcome"]["metadata"])
                )
        critic_latencies.extend(
            float(row["elapsed_ms"])
            for row in summary.get("visual_critic_receipts", [])
            if row.get("accepted")
        )
        episodes.append(
            {
                "trial": int(summary["trial"]),
                "success": bool(summary["success"]),
                "task_plan_completed": bool(summary["task_plan"]["completed"]),
                "episode_steps": int(summary["episode_steps"]),
                "elapsed_s": float(summary["elapsed_s"]),
                "planner_calls": int(summary["planner_calls"]),
                "critic_calls": int(summary["visual_critic"]["calls"]),
                "recovery_attempts": int(summary["recovery_attempts"]),
                "procedure_warm_start_id": summary.get("procedure_warm_start_id"),
                "workspace": str(workspace),
                "video": str(summary["video"]),
            }
        )
    runtime = [float(row["runtime_latency_ms"]) for row in vla_rows]
    return {
        "episodes": sorted(episodes, key=lambda item: item["trial"]),
        "metrics": {
            "successes": sum(item["success"] for item in episodes),
            "trials": len(episodes),
            "task_plans_completed": sum(
                item["task_plan_completed"] for item in episodes
            ),
            "episode_steps_mean": statistics.mean(
                item["episode_steps"] for item in episodes
            ),
            "wall_time_mean_s": statistics.mean(
                item["elapsed_s"] for item in episodes
            ),
            "startup_planner_calls": startup_planner_calls,
            "event_planner_calls": event_planner_calls,
            "planner_calls": startup_planner_calls + event_planner_calls,
            "planner_latency_total_ms": sum(planner_latencies),
            "planner_latency_mean_ms": (
                statistics.mean(planner_latencies) if planner_latencies else 0.0
            ),
            "planner_latency_p95_ms": (
                percentile(planner_latencies, 0.95) if planner_latencies else 0.0
            ),
            "critic_calls": len(critic_latencies),
            "critic_latency_mean_ms": statistics.mean(critic_latencies),
            "critic_latency_p95_ms": percentile(critic_latencies, 0.95),
            "recovery_attempts": sum(
                item["recovery_attempts"] for item in episodes
            ),
            "vla_calls": len(vla_rows),
            "vla_runtime_mean_ms": statistics.mean(runtime),
            "vla_runtime_p95_ms": percentile(runtime, 0.95),
            "vla_runtime_max_ms": max(runtime),
            "vla_deadline_misses": sum(
                bool(row["deadline_miss"]) for row in vla_rows
            ),
        },
    }


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-9b", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--memory-9b", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--memory-4b", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--direct-4b", type=pathlib.Path, required=True)
    parser.add_argument("--discovery", type=pathlib.Path, required=True)
    parser.add_argument("--memory", type=pathlib.Path, required=True)
    parser.add_argument("--receipt-9b", type=pathlib.Path, required=True)
    parser.add_argument("--receipt-4b", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    direct_9b = aggregate(args.direct_9b)
    memory_9b = aggregate(args.memory_9b)
    memory_4b = aggregate(args.memory_4b)
    direct_4b_summary = load(args.direct_4b)["episodes"][0]
    direct_4b_workspace = pathlib.Path(direct_4b_summary["workspace"])
    direct_4b_planner = next(
        event["payload"]
        for event in load_events(direct_4b_workspace)
        if event.get("event_type") == "planner_result"
    )
    verified_memory = load(args.memory)
    record = verified_memory["records"][0]
    d9 = direct_9b["metrics"]
    m9 = memory_9b["metrics"]
    m4 = memory_4b["metrics"]
    payload = {
        "schema_version": "carve-cross-task-memory-routing-v1",
        "protocol": {
            "benchmark": "LIBERO-PRO implementation, standard libero_10 task 3",
            "instruction": "put the black bowl in the bottom drawer of the cabinet and close it",
            "discovery_trial": 0,
            "paired_heldout_trials": [1, 7, 9],
            "semantic_checkpoints": [130, 190],
            "deadline_ms": 80.0,
        },
        "verified_procedure": {
            "procedure_id": record["procedure_id"],
            "verification_result": record["verification_result"],
            "step_count": len(record["steps"]),
            "steps": record["steps"],
            "discovery_summary": str(args.discovery),
        },
        "direct_9b_nf4": direct_9b,
        "memory_9b_nf4": memory_9b,
        "memory_4b_bf16": memory_4b,
        "direct_4b_bf16_qualification": {
            "trial": int(direct_4b_summary["trial"]),
            "success": bool(direct_4b_summary["success"]),
            "status": direct_4b_summary["status"],
            "episode_steps": int(direct_4b_summary["episode_steps"]),
            "planner_accepted": bool(direct_4b_planner["accepted"]),
            "planner_attempt_count": int(direct_4b_planner["attempt_count"]),
            "planner_elapsed_ms": float(direct_4b_planner["elapsed_ms"]),
            "error": direct_4b_planner["error"],
            "decision": "rejected_before_physical_execution",
        },
        "paired_changes": {
            "memory_9b_vs_direct_9b": {
                "success_delta": m9["successes"] - d9["successes"],
                "startup_planner_calls_delta": (
                    m9["startup_planner_calls"] - d9["startup_planner_calls"]
                ),
                "total_planner_calls_delta": m9["planner_calls"] - d9["planner_calls"],
                "wall_time_reduction_percent": reduction(
                    d9["wall_time_mean_s"], m9["wall_time_mean_s"]
                ),
            },
            "memory_4b_vs_memory_9b": {
                "success_delta": m4["successes"] - m9["successes"],
                "critic_latency_reduction_percent": reduction(
                    m9["critic_latency_mean_ms"], m4["critic_latency_mean_ms"]
                ),
                "wall_time_reduction_percent": reduction(
                    m9["wall_time_mean_s"], m4["wall_time_mean_s"]
                ),
            },
            "memory_4b_vs_direct_9b": {
                "success_delta": m4["successes"] - d9["successes"],
                "startup_planner_calls_delta": (
                    m4["startup_planner_calls"] - d9["startup_planner_calls"]
                ),
                "total_planner_calls_delta": m4["planner_calls"] - d9["planner_calls"],
                "wall_time_reduction_percent": reduction(
                    d9["wall_time_mean_s"], m4["wall_time_mean_s"]
                ),
            },
        },
        "profile_receipts": {
            "qwen3.5_9b_uniform_nf4": load(args.receipt_9b),
            "qwen3.5_4b_bf16": load(args.receipt_4b),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_changes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
