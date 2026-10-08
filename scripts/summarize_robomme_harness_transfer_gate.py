#!/usr/bin/env python3
"""Audit the frozen RouteStick/PickHighlight raw-vs-harness comparison."""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/robomme/harness_transfer_gate_20260924"
TASKS = ("RouteStick", "PickHighlight")
EPISODES = range(22, 38)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing video: {path}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams or streams[0].get("width", 0) <= 0 or streams[0].get("height", 0) <= 0:
        raise ValueError(f"undecodable video: {path}")


def exact_mcnemar(rescues: int, harms: int) -> float:
    discordant = rescues + harms
    if discordant == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(discordant, k) for k in range(min(rescues, harms) + 1)) / 2**discordant)


def summarize(rows: list[dict]) -> dict:
    both = [row for row in rows if row["R"]["success"] and row["H"]["success"]]
    rescues = sum(row["rescue"] for row in rows)
    harms = sum(row["harm"] for row in rows)
    return {
        "pairs": len(rows),
        "raw_success": sum(row["R"]["success"] for row in rows),
        "harness_success": sum(row["H"]["success"] for row in rows),
        "rescues": rescues,
        "harms": harms,
        "mcnemar_exact_two_sided_p": exact_mcnemar(rescues, harms),
        "budget_exhausted": {arm: sum(row[arm]["budget_exhausted"] for row in rows) for arm in ("R", "H")},
        "both_success_pairs": len(both),
        "both_success_cost": {
            key: {arm: sum(row[arm][key] for row in both) for arm in ("R", "H")}
            for key in ("wall_time_s", "planner_calls", "policy_calls", "steps")
        },
    }


def main() -> int:
    output = ROOT / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    order = (ROOT / "run_order.log").read_text(encoding="utf-8")
    rows = []
    for task in TASKS:
        seen = set()
        for episode in EPISODES:
            name = f"{task}_ep{episode}"
            preflight = read(ROOT / "preflight" / name / "summary.json")
            initial = preflight["initial_observation_sha256"]
            if initial in seen:
                raise ValueError(f"duplicate preflight: {name}")
            seen.add(initial)
            summaries = {}
            row = {"task": task, "episode": episode, "instruction": preflight["instruction"]}
            for arm, profile in (("R", "raw"), ("H", "carve")):
                run = ROOT / "rollouts" / f"{name}_{arm}"
                summary = read(run / "summary.json")
                verify_video(run / f"{name}_vlm_groundsg.mp4")
                if summary["task"] != task or summary["episode"] != episode:
                    raise ValueError(f"wrong task/episode: {name}_{arm}")
                if summary["initial_observation_sha256"] != initial:
                    raise ValueError(f"initial state mismatch: {name}_{arm}")
                if summary["failure"] is not None:
                    raise ValueError(f"method/infrastructure error: {name}_{arm}: {summary['failure']}")
                if summary["status"] not in {"success", "fail", "ongoing"}:
                    raise ValueError(f"unknown official status: {name}_{arm}")
                if summary["status"] == "ongoing" and summary["executed_steps"] < 1300:
                    raise ValueError(f"unfinished rollout: {name}_{arm}")
                config = summary["run_config"]
                if config["planner_profile"] != profile or config["planner_schedule"] != "every_chunk":
                    raise ValueError(f"method mismatch: {name}_{arm}")
                if config["task_memory_file"] is not None or summary["memory_provenance"] is not None:
                    raise ValueError(f"unexpected structured memory: {name}_{arm}")
                if config["demo_history_mode"] != "official" or config["planner_image_format"] != "png":
                    raise ValueError(f"presentation mismatch: {name}_{arm}")
                if summary["max_steps"] != 1300 or summary["action_horizon"] != 16:
                    raise ValueError(f"control budget mismatch: {name}_{arm}")
                if summary["video_frames"] <= 0:
                    raise ValueError(f"empty frames: {name}_{arm}")
                if order.count(f"starting {name}_{arm} ") != 1:
                    raise ValueError(f"missing/duplicate order entry: {name}_{arm}")
                summaries[arm] = summary
                row[arm] = {
                    "status": summary["status"],
                    "success": summary["success"],
                    "budget_exhausted": summary["status"] == "ongoing",
                    "steps": summary["executed_steps"],
                    "planner_calls": summary["planner_calls"],
                    "policy_calls": summary["policy_calls"],
                    "wall_time_s": summary["wall_time_s"],
                    "video": str((run / f"{name}_vlm_groundsg.mp4").relative_to(PROJECT)),
                }
            for key in ("runner_sha256", "policy_label", "planner_label"):
                if summaries["R"][key] != summaries["H"][key]:
                    raise ValueError(f"pair {key} mismatch: {name}")
            first, second = (("R", "H") if episode % 2 == 0 else ("H", "R"))
            if order.index(f"starting {name}_{first} ") > order.index(f"starting {name}_{second} "):
                raise ValueError(f"wrong execution order: {name}")
            row["rescue"] = not row["R"]["success"] and row["H"]["success"]
            row["harm"] = row["R"]["success"] and not row["H"]["success"]
            rows.append(row)
    analysis = {
        "protocol": "carve.robomme.frozen_harness_transfer.v1",
        "claim_boundary": "Purposefully selected positive/risk task families; not a random benchmark sample or full memory/Optimize evaluation.",
        "tasks": list(TASKS),
        "episode_range": [EPISODES.start, EPISODES.stop - 1],
        "source_sha256": digest(ROOT / "source_sha256.txt"),
        "overall": summarize(rows),
        "per_task": {task: summarize([row for row in rows if row["task"] == task]) for task in TASKS},
        "rows": rows,
    }
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": analysis["overall"], "per_task": analysis["per_task"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
