#!/usr/bin/env python3
"""Summarize paired fixed, deployable-VLM, and oracle RoboMME episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--vlm-root", type=Path)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_episodes(root: Path, *, method: str) -> dict[int, dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("MoveCube_ep*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = int(payload["episode"])
        video = path.parent / Path(payload["video_path"]).name
        if not video.is_file():
            raise FileNotFoundError(video)
        status_key = "evaluator_status" if method == "fixed" else "status"
        status = str(payload[status_key])
        planner_trace = payload.get("planner_trace", [])
        unique_subgoals = list(
            dict.fromkeys(
                str(item["history_text"])
                for item in planner_trace
                if isinstance(item, dict) and item.get("history_text")
            )
        )
        episodes[episode] = {
            "episode": episode,
            "success": status == "success",
            "status": status,
            "executed_steps": int(payload["executed_steps"]),
            "policy_calls": int(payload["policy_calls"]),
            "policy_latency_p95_ms": float(payload["policy_latency_ms"]["p95"]),
            "wall_time_s": float(payload["wall_time_s"]),
            "subgoal": payload.get("fixed_subgoal") if method == "fixed" else None,
            "unique_subgoals": payload.get("unique_subgoals", unique_subgoals),
            "planner_calls": int(payload.get("planner_calls", 0)),
            "planner_latency_p95_ms": payload.get("planner_latency_ms", {}).get("p95"),
            "summary": str(path),
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        }
    if not episodes:
        raise ValueError(f"no episode summaries found in {root}")
    return episodes


def method_summary(episodes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    values = list(episodes.values())
    successes = [row for row in values if row["success"]]
    planner_latencies = [
        float(row["planner_latency_p95_ms"])
        for row in values
        if row["planner_latency_p95_ms"] is not None
    ]
    return {
        "episodes": len(values),
        "successes": len(successes),
        "success_rate": len(successes) / len(values),
        "mean_policy_latency_p95_ms": statistics.fmean(
            row["policy_latency_p95_ms"] for row in values
        ),
        "mean_steps_all": statistics.fmean(row["executed_steps"] for row in values),
        "mean_steps_success_only": (
            statistics.fmean(row["executed_steps"] for row in successes)
            if successes
            else None
        ),
        "mean_policy_calls_all": statistics.fmean(row["policy_calls"] for row in values),
        "mean_planner_calls_all": statistics.fmean(row["planner_calls"] for row in values),
        "mean_planner_latency_p95_ms": (
            statistics.fmean(planner_latencies) if planner_latencies else None
        ),
    }


def main() -> int:
    args = parse_args()
    fixed = load_episodes(args.fixed_root, method="fixed")
    oracle = load_episodes(args.oracle_root, method="oracle")
    if fixed.keys() != oracle.keys():
        raise ValueError("fixed and oracle episode identities do not match")
    vlm = load_episodes(args.vlm_root, method="vlm") if args.vlm_root else None
    if vlm is not None and fixed.keys() != vlm.keys():
        raise ValueError("fixed and VLM episode identities do not match")

    paired = [
        {
            "episode": episode,
            "fixed": fixed[episode],
            **({"deployable_vlm": vlm[episode]} if vlm is not None else {}),
            "oracle": oracle[episode],
        }
        for episode in sorted(fixed)
    ]
    output = {
        "schema_version": "carve.robomme.planning_gap.v2",
        "benchmark": "RoboMME",
        "task": "MoveCube",
        "claim_boundary": {
            "fixed_text": (
                "Accepted Planner-output interface pilot with one static generic "
                "subgoal; not the complete event-triggered CARVE Planner."
            ),
            "oracle": (
                "Privileged online grounded-subgoal upper bound; not deployable and "
                "not reported as CARVE task performance."
            ),
            "deployable_vlm": (
                "Official RoboMME Qwen3-VL GroundSG adapter behind CARVE's guarded "
                "provider boundary; no evaluator or oracle fields enter inference."
            ),
            "interpretation": (
                "The paired gap measures the value left for dynamic stage selection "
                "and visual point grounding while holding the action policy fixed."
            ),
        },
        "fixed_text": method_summary(fixed),
        **({"deployable_vlm": method_summary(vlm)} if vlm is not None else {}),
        "online_oracle": method_summary(oracle),
        "paired_episodes": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
