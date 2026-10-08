#!/usr/bin/env python3
"""Audit the preselected, same-service B/D identity-conflict development pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/robomme/identity_conflict_dev_20260923"
MEMORY_ROOTS = {
    16: PROJECT_ROOT / "artifacts/robomme/unmaskswap_fixed_five_20260923",
    20: PROJECT_ROOT / "artifacts/robomme/service_block_pilot_20260923",
    21: PROJECT_ROOT / "artifacts/robomme/service_block_pilot_20260923",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    seconds = float(result.stdout.strip())
    if seconds <= 0:
        raise ValueError(f"empty video: {path}")
    return seconds


def summarize(root: Path) -> dict:
    warmup = read_json(root / "rollouts/warmup_ep15_B/summary.json")
    if warmup["failure"] is not None or warmup["planner_calls"] < 1 or warmup["policy_calls"] < 1:
        raise ValueError("model warmup failed")
    pairs = []
    runner_sha = None
    labels = None
    for episode, memory_root in MEMORY_ROOTS.items():
        name = f"VideoUnmaskSwap_ep{episode}"
        memory_file = memory_root / "memory" / f"{name}_memory.json"
        memory_sha = hashlib.sha256(memory_file.read_bytes()).hexdigest()
        memory = read_json(memory_file)
        if memory["admission"]["admitted"] is not True:
            raise ValueError(f"{name}: memory is not admitted")
        arm_data = {}
        for arm in ("B", "D"):
            directory = root / "rollouts" / f"{name}_{arm}"
            summary = read_json(directory / "summary.json")
            config = summary["run_config"]
            provenance = summary["memory_provenance"]
            if summary["task"] != "VideoUnmaskSwap" or summary["episode"] != episode:
                raise ValueError(f"{name} {arm}: wrong task/episode")
            if summary["instruction"] != memory["instruction"]:
                raise ValueError(f"{name} {arm}: wrong instruction")
            if bool(config["verified_identity_conflict"]) != (arm == "D"):
                raise ValueError(f"{name} {arm}: wrong intervention setting")
            if config["grounding_authority"] != "observe_only" or config["planner_schedule"] != "every_chunk":
                raise ValueError(f"{name} {arm}: protocol drift")
            if Path(config["task_memory_file"]).name != memory_file.name:
                raise ValueError(f"{name} {arm}: wrong memory file")
            if provenance["memory_sha256"] != memory_sha or not provenance["used"]:
                raise ValueError(f"{name} {arm}: memory binding failed")
            if provenance["current_demo_sha256"] != memory["source_demo_sha256"]:
                raise ValueError(f"{name} {arm}: demo binding failed")
            if summary["success"] is not (summary["status"] == "success"):
                raise ValueError(f"{name} {arm}: inconsistent status")
            current_labels = (summary["policy_label"], summary["planner_label"])
            if runner_sha is None:
                runner_sha, labels = summary["runner_sha256"], current_labels
            if summary["runner_sha256"] != runner_sha or current_labels != labels:
                raise ValueError(f"{name} {arm}: runner/model labels changed")
            video = directory / f"{name}_vlm_groundsg.mp4"
            corrections = [
                {"step": item["step"], **item["identity_conflict"]}
                for item in summary["planner_trace"]
                if (item.get("identity_conflict") or {}).get("resolved")
            ]
            if arm == "B" and corrections:
                raise ValueError(f"{name}: baseline was corrected")
            arm_data[arm] = {
                "summary": summary,
                "config": config,
                "outcome": {
                    "status": summary["status"], "success": summary["success"],
                    "failure": summary["failure"], "steps": summary["executed_steps"],
                    "planner_calls": summary["planner_calls"],
                    "vla_calls": summary["policy_calls"],
                    "wall_s": summary["wall_time_s"],
                    "corrections": corrections,
                    "video": str(video), "video_duration_s": video_seconds(video),
                },
            }
        if arm_data["B"]["summary"]["initial_observation_sha256"] != arm_data["D"]["summary"]["initial_observation_sha256"]:
            raise ValueError(f"{name}: initial observations differ")
        ignored = {"output", "verified_identity_conflict"}
        for key, value in arm_data["B"]["config"].items():
            if key not in ignored and value != arm_data["D"]["config"][key]:
                raise ValueError(f"{name}: B/D protocol differs at {key}")
        pairs.append({
            "episode": episode,
            "instruction": arm_data["B"]["summary"]["instruction"],
            "initial_observation_sha256": arm_data["B"]["summary"]["initial_observation_sha256"],
            "memory_sha256": memory_sha,
            "arms": {arm: arm_data[arm]["outcome"] for arm in ("B", "D")},
        })
    both_success = [p for p in pairs if p["arms"]["B"]["success"] and p["arms"]["D"]["success"]]
    return {
        "protocol": "carve.robomme.verified_identity_conflict.development.v1",
        "scope": "seen_ep16_20_21_same_service_development_not_holdout",
        "runner_sha256": runner_sha,
        "policy_label": labels[0], "planner_label": labels[1],
        "success_B": sum(p["arms"]["B"]["success"] for p in pairs),
        "success_D": sum(p["arms"]["D"]["success"] for p in pairs),
        "B_to_D_rescues": sum(not p["arms"]["B"]["success"] and p["arms"]["D"]["success"] for p in pairs),
        "B_to_D_harms": sum(p["arms"]["B"]["success"] and not p["arms"]["D"]["success"] for p in pairs),
        "both_success_count": len(both_success),
        "both_success_wall_B_s": sum(p["arms"]["B"]["wall_s"] for p in both_success),
        "both_success_wall_D_s": sum(p["arms"]["D"]["wall_s"] for p in both_success),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    for pair in result["pairs"]:
        print(pair["episode"], {
            arm: (value["status"], value["steps"], len(value["corrections"]))
            for arm, value in pair["arms"].items()
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
