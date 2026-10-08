#!/usr/bin/env python3
"""Summarize the paired RoboMME grounded-Planner scheduling ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--uniform-reuse", type=Path, required=True)
    parser.add_argument("--adaptive-reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_condition(root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in sorted(root.glob("MoveCube_ep*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = Path(str(payload["video_path"]))
        if not video.exists() and str(video).startswith("/workspace/"):
            video = Path.cwd() / str(video).removeprefix("/workspace/")
        episodes.append(
            {
                "episode": int(payload["episode"]),
                "success": bool(payload["success"]),
                "status": str(payload["status"]),
                "executed_steps": int(payload["executed_steps"]),
                "policy_calls": int(payload["policy_calls"]),
                "planner_calls": int(payload["planner_calls"]),
                "planner_reuse_hits": int(payload.get("planner_reuse_hits", 0)),
                "wall_time_s": float(payload["wall_time_s"]),
                "planner_p95_ms": payload["planner_latency_ms"]["p95"],
                "video_path": str(video),
                "video_sha256": sha256(video),
            }
        )
    if [item["episode"] for item in episodes] != list(range(5)):
        raise ValueError(f"expected MoveCube episodes 0..4 under {root}")
    return episodes


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "successes": sum(item["success"] for item in episodes),
        "episodes": len(episodes),
        "planner_calls": sum(item["planner_calls"] for item in episodes),
        "planner_reuse_hits": sum(item["planner_reuse_hits"] for item in episodes),
        "policy_calls": sum(item["policy_calls"] for item in episodes),
        "wall_time_s": sum(item["wall_time_s"] for item in episodes),
        "mean_episode_planner_p95_ms": statistics.fmean(
            float(item["planner_p95_ms"])
            for item in episodes
            if item["planner_p95_ms"] is not None
        ),
    }


def reduction(new: float, reference: float) -> float:
    return 1.0 - new / reference


def main() -> int:
    args = parse_args()
    conditions = {
        "every_chunk_bf16": load_condition(args.baseline),
        "uniform_reuse_2": load_condition(args.uniform_reuse),
        "complexity_aware_reuse_2": load_condition(args.adaptive_reuse),
    }
    aggregates = {name: aggregate(rows) for name, rows in conditions.items()}
    baseline = aggregates["every_chunk_bf16"]
    adaptive = aggregates["complexity_aware_reuse_2"]

    paired_rows = []
    for episode in range(5):
        paired_rows.append(
            {
                "episode": episode,
                **{
                    name: next(row for row in rows if row["episode"] == episode)
                    for name, rows in conditions.items()
                },
            }
        )

    successful_pairs = [
        row
        for row in paired_rows
        if row["every_chunk_bf16"]["success"]
        and row["complexity_aware_reuse_2"]["success"]
    ]
    baseline_success_calls = sum(
        row["every_chunk_bf16"]["planner_calls"] for row in successful_pairs
    )
    adaptive_success_calls = sum(
        row["complexity_aware_reuse_2"]["planner_calls"] for row in successful_pairs
    )
    baseline_success_wall = sum(
        row["every_chunk_bf16"]["wall_time_s"] for row in successful_pairs
    )
    adaptive_success_wall = sum(
        row["complexity_aware_reuse_2"]["wall_time_s"] for row in successful_pairs
    )

    summary = {
        "protocol": "carve.robomme.planner_scheduling_ablation.v1",
        "task": "MoveCube",
        "episode_ids": list(range(5)),
        "fixed_components": {
            "action_policy": "Yinpei/mme_vla_suite symbolic-grounded-subgoal/79999",
            "planner": "Qwen3-VL-4B-Instruct + official GroundSG LoRA, BF16",
            "action_horizon": 16,
            "policy_reset": "official client reset restores the fixed policy seed per episode",
        },
        "conditions": aggregates,
        "paired_episodes": paired_rows,
        "complexity_aware_vs_every_chunk": {
            "success_delta": adaptive["successes"] - baseline["successes"],
            "planner_call_reduction_fraction_all": reduction(
                adaptive["planner_calls"], baseline["planner_calls"]
            ),
            "wall_time_reduction_fraction_all": reduction(
                adaptive["wall_time_s"], baseline["wall_time_s"]
            ),
            "matched_success_episode_ids": [row["episode"] for row in successful_pairs],
            "planner_call_reduction_fraction_matched_successes": reduction(
                adaptive_success_calls, baseline_success_calls
            ),
            "wall_time_reduction_fraction_matched_successes": reduction(
                adaptive_success_wall, baseline_success_wall
            ),
        },
        "admission": {
            "success_non_inferior": adaptive["successes"] >= baseline["successes"],
            "planner_calls_lower": adaptive["planner_calls"] < baseline["planner_calls"],
            "wall_time_lower": adaptive["wall_time_s"] < baseline["wall_time_s"],
            "uniform_reuse_rejected": (
                aggregates["uniform_reuse_2"]["successes"] < baseline["successes"]
            ),
        },
        "claim_boundary": (
            "Five paired official RoboMME MoveCube test episodes establish a "
            "mechanism gate, not benchmark-wide superiority."
        ),
    }
    summary["admission"]["passed"] = all(
        summary["admission"][key]
        for key in ("success_non_inferior", "planner_calls_lower", "wall_time_lower")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["complexity_aware_vs_every_chunk"], indent=2))
    print(json.dumps(summary["admission"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
