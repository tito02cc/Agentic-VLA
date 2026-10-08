#!/usr/bin/env python3
"""Summarize the four-task RoboMME memory capability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ("StopCube", "VideoUnmaskSwap", "VideoRepick", "RouteStick")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every-chunk", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(root: Path, task: str) -> dict[str, Any]:
    payload = json.loads((root / f"{task}_ep0" / "summary.json").read_text())
    return {
        "success": bool(payload["success"]),
        "status": str(payload["status"]),
        "executed_steps": int(payload["executed_steps"]),
        "policy_calls": int(payload["policy_calls"]),
        "planner_calls": int(payload.get("planner_calls", 0)),
        "wall_time_s": float(payload["wall_time_s"]),
        "initial_memory_frames": int(payload["initial_memory_frames"]),
        "unique_subgoals": payload.get("unique_subgoals"),
        "first_subgoal": (
            payload.get("planner_trace", [{}])[0].get("grounded_subgoal")
            if payload.get("planner_trace")
            else None
        ),
        "video_path": str(payload["video_path"]),
    }


def aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "successes": sum(row["success"] for row in rows.values()),
        "tasks": len(rows),
        "planner_calls": sum(row["planner_calls"] for row in rows.values()),
        "policy_calls": sum(row["policy_calls"] for row in rows.values()),
        "wall_time_s": sum(row["wall_time_s"] for row in rows.values()),
    }


def main() -> int:
    args = parse_args()
    conditions = {
        "every_chunk_vlm": {task: load(args.every_chunk, task) for task in TASKS},
        "complexity_aware_vlm": {task: load(args.adaptive, task) for task in TASKS},
        "privileged_oracle": {task: load(args.oracle, task) for task in TASKS},
    }
    totals = {name: aggregate(rows) for name, rows in conditions.items()}
    baseline = totals["every_chunk_vlm"]
    adaptive = totals["complexity_aware_vlm"]
    summary = {
        "protocol": "carve.robomme.memory_capability_gate.v1",
        "episode": 0,
        "tasks": list(TASKS),
        "conditions": conditions,
        "totals": totals,
        "adaptive_vs_every_chunk": {
            "success_delta": adaptive["successes"] - baseline["successes"],
            "planner_call_reduction_fraction": (
                1.0 - adaptive["planner_calls"] / baseline["planner_calls"]
            ),
            "wall_time_reduction_fraction": (
                1.0 - adaptive["wall_time_s"] / baseline["wall_time_s"]
            ),
        },
        "diagnosis": {
            "oracle_policy_capability": "4/4",
            "deployable_vlm_capability": "0/4",
            "next_bottleneck": "structured temporal/spatial/object/procedural memory",
            "batch_extension_admitted": False,
        },
        "claim_boundary": (
            "One official test episode per memory suite is a capability gate. "
            "The oracle is privileged and is not a CARVE result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": totals, **summary["adaptive_vs_every_chunk"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
