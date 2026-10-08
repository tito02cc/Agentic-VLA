#!/usr/bin/env python3
"""Audit and summarize the preregistered VideoUnmaskSwap ep15-19 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts/robomme/unmaskswap_fixed_five_20260923"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path) -> dict:
    rows = []
    runner_sha = None
    policy_label = None
    planner_label = None
    shared_memory_load_s = None
    for episode in range(15, 20):
        name = f"VideoUnmaskSwap_ep{episode}"
        preflight_dir = root / "preflight" / name
        preflight = read_json(preflight_dir / "summary.json")
        memory_path = root / "memory" / f"{name}_memory.json"
        memory = read_json(memory_path)
        a_dir = root / "rollouts" / f"{name}_A"
        b_dir = root / "rollouts" / f"{name}_B"
        a = read_json(a_dir / "summary.json")
        b = read_json(b_dir / "summary.json")
        a_config = a["run_config"]
        b_config = b["run_config"]
        ignored = {"output", "task_memory_file"}
        if {key: value for key, value in a_config.items() if key not in ignored} != {
            key: value for key, value in b_config.items() if key not in ignored
        }:
            raise ValueError(f"episode {episode} A/B settings differ beyond memory")
        if a_config.get("task_memory_file") is not None or Path(b_config["task_memory_file"]).name != memory_path.name:
            raise ValueError(f"episode {episode} A/B memory path mismatch")
        if memory.get("admission", {}).get("admitted") is not True:
            raise ValueError(f"episode {episode} memory was not admitted")
        if memory["source_demo_sha256"] != preflight["initial_demo_sha256"]:
            raise ValueError(f"episode {episode} preflight source SHA mismatch")
        if hashlib.sha256((preflight_dir / "initial_demo_front.mp4").read_bytes()).hexdigest() != preflight["initial_demo_sha256"]:
            raise ValueError(f"episode {episode} preflight video changed")
        provenance = b.get("memory_provenance") or {}
        if not provenance.get("admitted") or not provenance.get("used"):
            raise ValueError(f"episode {episode} B did not use admitted identity memory")
        if provenance.get("source_demo_sha256") != preflight["initial_demo_sha256"]:
            raise ValueError(f"episode {episode} B memory source mismatch")
        if provenance.get("current_demo_sha256") != preflight["initial_demo_sha256"]:
            raise ValueError(f"episode {episode} B live demonstration mismatch")
        if provenance.get("memory_sha256") != hashlib.sha256(memory_path.read_bytes()).hexdigest():
            raise ValueError(f"episode {episode} B memory file changed")
        if shared_memory_load_s is None:
            shared_memory_load_s = memory.get("shared_model_load_s")
        for arm, data, directory in (("A", a, a_dir), ("B", b, b_dir)):
            if data["task"] != "VideoUnmaskSwap" or data["episode"] != episode:
                raise ValueError(f"episode {episode} {arm} task binding mismatch")
            if data["instruction"] != preflight["instruction"]:
                raise ValueError(f"episode {episode} {arm} instruction mismatch")
            if data["initial_observation_sha256"] != preflight["initial_observation_sha256"]:
                raise ValueError(f"episode {episode} {arm} initial observation mismatch")
            if data["success"] is not (data["status"] == "success"):
                raise ValueError(f"episode {episode} {arm} success/status mismatch")
            if data["failure"] is not None:
                raise ValueError(f"episode {episode} {arm} infrastructure failure: {data['failure']}")
            video_path = directory / f"{name}_vlm_groundsg.mp4"
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise ValueError(f"episode {episode} {arm} missing video")
            if runner_sha is None:
                runner_sha = data["runner_sha256"]
                policy_label = data["policy_label"]
                planner_label = data["planner_label"]
            if (data["runner_sha256"], data["policy_label"], data["planner_label"]) != (
                runner_sha, policy_label, planner_label
            ):
                raise ValueError(f"episode {episode} {arm} runner/model mismatch")
        rows.append({
            "episode": episode,
            "instruction": preflight["instruction"],
            "initial_observation_sha256": preflight["initial_observation_sha256"],
            "memory_sha256": provenance["memory_sha256"],
            "memory_compile_s": memory["compile_time_s"],
            "A": {key: a[key] for key in ("success", "status", "executed_steps", "planner_calls", "policy_calls", "wall_time_s")},
            "B": {key: b[key] for key in ("success", "status", "executed_steps", "planner_calls", "policy_calls", "wall_time_s")},
            "first_subgoal_A": a["planner_trace"][0]["grounded_subgoal"],
            "first_subgoal_B": b["planner_trace"][0]["grounded_subgoal"],
            "video_A": str(a_dir / f"{name}_vlm_groundsg.mp4"),
            "video_B": str(b_dir / f"{name}_vlm_groundsg.mp4"),
        })
    rescued = sum(not row["A"]["success"] and row["B"]["success"] for row in rows)
    harmed = sum(row["A"]["success"] and not row["B"]["success"] for row in rows)
    both_success = [row for row in rows if row["A"]["success"] and row["B"]["success"]]
    return {
        "protocol": "carve.robomme.unmaskswap_fixed_five.audit.v1",
        "scope": "preregistered_unused_within_family_episodes_15_to_19",
        "claim_boundary": "Within-family paired quality evidence only; not cross-task generalization or runtime speedup.",
        "runner_sha256": runner_sha,
        "policy_label": policy_label,
        "planner_label": planner_label,
        "order": ["15:A,B", "16:B,A", "17:A,B", "18:B,A", "19:A,B"],
        "episodes": len(rows),
        "success_A": sum(row["A"]["success"] for row in rows),
        "success_B": sum(row["B"]["success"] for row in rows),
        "rescued": rescued,
        "harmed": harmed,
        "both_success": len(both_success),
        "memory_admitted": len(rows),
        "memory_compile_total_s": sum(row["memory_compile_s"] for row in rows),
        "shared_memory_model_load_s": shared_memory_load_s,
        "both_success_steps_A": sum(row["A"]["executed_steps"] for row in both_success),
        "both_success_steps_B": sum(row["B"]["executed_steps"] for row in both_success),
        "both_success_planner_calls_A": sum(row["A"]["planner_calls"] for row in both_success),
        "both_success_planner_calls_B": sum(row["B"]["planner_calls"] for row in both_success),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    summary = summarize(args.root)
    output = args.root / "analysis.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
