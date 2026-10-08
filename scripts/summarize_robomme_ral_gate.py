#!/usr/bin/env python3
"""Summarize preselected RoboMME A/B/C pairs without dropping failures."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
from statistics import mean, median


ARMS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_mcnemar(rescues: int, harms: int) -> float | None:
    discordant = rescues + harms
    if not discordant:
        return None
    tail = sum(comb(discordant, index) for index in range(min(rescues, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(args: argparse.Namespace) -> dict:
    episodes = []
    for episode in range(args.episode_start, args.episode_end + 1):
        rows = {}
        paths = {}
        for arm in ARMS:
            matches = [root / f"{args.task}_ep{episode}_{arm}" / "summary.json"
                       for root in args.results_root]
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise ValueError(f"expected one {args.task} ep{episode} {arm} summary, found {matches}")
            path = matches[0]
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("task") != args.task or value.get("episode") != episode:
                raise ValueError(f"wrong task/episode in {path}")
            if type(value.get("success")) is not bool:
                raise ValueError(f"missing official success in {path}")
            rows[arm], paths[arm] = value, str(path)
        hashes = {row["initial_observation_sha256"] for row in rows.values()}
        if len(hashes) != 1:
            raise ValueError(f"unpaired initial observation for ep{episode}")
        episodes.append({
            "episode": episode,
            "initial_observation_sha256": next(iter(hashes)),
            "arms": {arm: {
                "success": row["success"], "status": row["status"],
                "failure": row["failure"], "executed_steps": row["executed_steps"],
                "planner_calls": row["planner_calls"], "policy_calls": row["policy_calls"],
                "wall_time_s": row["wall_time_s"],
                "memory_admitted": (row.get("memory_provenance") or {}).get("admitted"),
                "summary_path": paths[arm],
            } for arm, row in rows.items()},
        })
    totals = {}
    for arm in ARMS:
        entries = [case["arms"][arm] for case in episodes]
        totals[arm] = {
            "successes": sum(entry["success"] for entry in entries),
            "n": len(entries),
            "failure_reasons": dict(Counter(entry["failure"] for entry in entries if entry["failure"])),
            "mean_steps_all": mean(entry["executed_steps"] for entry in entries),
            "mean_planner_calls_all": mean(entry["planner_calls"] for entry in entries),
            "mean_policy_calls_all": mean(entry["policy_calls"] for entry in entries),
            "mean_wall_time_s_all": mean(entry["wall_time_s"] for entry in entries),
        }
    pairs = {}
    for arm in ("B", "C"):
        rescues = [case["episode"] for case in episodes
                   if not case["arms"]["A"]["success"] and case["arms"][arm]["success"]]
        harms = [case["episode"] for case in episodes
                 if case["arms"]["A"]["success"] and not case["arms"][arm]["success"]]
        both = [case for case in episodes
                if case["arms"]["A"]["success"] and case["arms"][arm]["success"]]
        pairs[f"A_vs_{arm}"] = {
            "rescues": rescues,
            "harms": harms,
            "exact_mcnemar_p_exploratory": exact_mcnemar(len(rescues), len(harms)),
            "both_success_episodes": [case["episode"] for case in both],
            "median_wall_time_s_both_success": {
                key: median(case["arms"][key]["wall_time_s"] for case in both) if both else None
                for key in ("A", arm)
            },
            "median_steps_both_success": {
                key: median(case["arms"][key]["executed_steps"] for case in both) if both else None
                for key in ("A", arm)
            },
        }
    return {
        "protocol": "carve.robomme.ral_gate_pilot_summary.v1",
        "claim_scope": "development pilot; code versions differ across roots; not frozen confirmatory evidence",
        "task": args.task,
        "episodes": episodes,
        "totals": totals,
        "paired_vs_raw": pairs,
    }


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"task": result["task"], "totals": result["totals"],
                      "paired_vs_raw": result["paired_vs_raw"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
