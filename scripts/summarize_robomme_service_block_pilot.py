#!/usr/bin/env python3
"""Audit every preselected RoboMME service-block pilot outcome and video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts/robomme/service_block_pilot_20260923"
CASES = {
    "VideoUnmaskSwap": ((20, 21), ("A", "B", "C")),
    "VideoUnmask": ((20, 21), ("A", "B")),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_seconds(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(proc.stdout.strip())
    if duration <= 0:
        raise ValueError(f"empty video: {path}")
    return duration


def summarize(root: Path) -> dict:
    rows = []
    runner_sha = None
    labels = None
    for block in ("block1", "block2"):
        warmup = read_json(root / "rollouts" / block / "VideoUnmaskSwap_ep15_B/summary.json")
        if warmup["failure"] is not None or warmup["planner_calls"] < 1 or warmup["policy_calls"] < 1:
            raise ValueError(f"{block}: model warmup failed")
        for task, (episodes, arms) in CASES.items():
            for episode in episodes:
                name = f"{task}_ep{episode}"
                preflight = read_json(root / "preflight" / name / "summary.json")
                memory_path = root / "memory" / f"{name}_memory.json"
                memory = read_json(memory_path)
                if memory["admission"]["admitted"] is not True:
                    raise ValueError(f"{name}: memory admission changed")
                if memory["source_demo_sha256"] != preflight["initial_demo_sha256"]:
                    raise ValueError(f"{name}: memory source differs from preflight")
                if hashlib.sha256((root / "preflight" / name / "initial_demo_front.mp4").read_bytes()).hexdigest() != preflight["initial_demo_sha256"]:
                    raise ValueError(f"{name}: preflight video changed")
                arm_data = {}
                for arm in arms:
                    directory = root / "rollouts" / block / f"{name}_{arm}"
                    data = read_json(directory / "summary.json")
                    config = data["run_config"]
                    if data["task"] != task or data["episode"] != episode:
                        raise ValueError(f"{block} {name} {arm}: task binding mismatch")
                    if data["instruction"] != preflight["instruction"]:
                        raise ValueError(f"{block} {name} {arm}: instruction mismatch")
                    if data["initial_observation_sha256"] != preflight["initial_observation_sha256"]:
                        raise ValueError(f"{block} {name} {arm}: initial observation mismatch")
                    if data["success"] is not (data["status"] == "success"):
                        raise ValueError(f"{block} {name} {arm}: status mismatch")
                    expected_schedule = "selective" if arm == "C" else "every_chunk"
                    if config["planner_schedule"] != expected_schedule or bool(config["verified_point_reuse"]) != (arm == "C"):
                        raise ValueError(f"{block} {name} {arm}: schedule mismatch")
                    provenance = data.get("memory_provenance")
                    if arm == "A":
                        if config["task_memory_file"] is not None or data.get("task_memory") is not None:
                            raise ValueError(f"{block} {name} {arm}: baseline received memory")
                    else:
                        if Path(config["task_memory_file"]).name != memory_path.name:
                            raise ValueError(f"{block} {name} {arm}: wrong memory file")
                        if (provenance or {}).get("memory_sha256") != hashlib.sha256(memory_path.read_bytes()).hexdigest():
                            raise ValueError(f"{block} {name} {arm}: memory hash mismatch")
                        if (provenance or {}).get("current_demo_sha256") != preflight["initial_demo_sha256"]:
                            raise ValueError(f"{block} {name} {arm}: live demo source mismatch")
                        if bool(provenance["used"]) != (task == "VideoUnmaskSwap"):
                            raise ValueError(f"{block} {name} {arm}: memory motion gate contradicted preflight")
                    current_labels = (data["policy_label"], data["planner_label"])
                    if runner_sha is None:
                        runner_sha, labels = data["runner_sha256"], current_labels
                    if data["runner_sha256"] != runner_sha or current_labels != labels:
                        raise ValueError(f"{block} {name} {arm}: runner/model labels differ")
                    video = directory / f"{name}_vlm_groundsg.mp4"
                    duration = video_seconds(video)
                    arm_data[arm] = (data, config, {
                        "success": data["success"], "status": data["status"],
                        "failure": data["failure"], "steps": data["executed_steps"],
                        "planner_calls": data["planner_calls"], "vla_calls": data["policy_calls"],
                        "reuse_hits": data["planner_reuse_hits"],
                        "wall_s": data["wall_time_s"], "rollout_s": data["timing"]["rollout_s"],
                        "planner_request_s": data["timing"]["planner_requests"]["total_ms"] / 1000.0,
                        "vla_request_s": data["timing"]["policy_requests"]["total_ms"] / 1000.0,
                        "video": str(video), "video_duration_s": duration,
                    })
                common = {"output", "task_memory_file", "memory_use_policy", "planner_schedule", "verified_point_reuse"}
                configs = [{k: v for k, v in arm_data[a][1].items() if k not in common} for a in arms]
                if any(config != configs[0] for config in configs[1:]):
                    raise ValueError(f"{block} {name}: arms differ beyond intended method factors")
                row = {
                    "block": block, "task": task, "episode": episode,
                    "instruction": preflight["instruction"],
                    "memory_compile_s": memory["compile_time_s"],
                    "arms": {a: arm_data[a][2] for a in arms},
                }
                if task == "VideoUnmaskSwap":
                    target = memory["memory"]["required_color_order"][0]
                    points = memory["memory"]["hidden_container_points"]
                    first_points = {}
                    for arm in arms:
                        trace = arm_data[arm][0]["planner_trace"]
                        point = trace[0]["model_points"][0] if trace and trace[0]["model_points"] else None
                        distances = {
                            color: round(math.dist(point, memory_point), 2)
                            for color, memory_point in points.items()
                        } if point else {}
                        first_points[arm] = {
                            "planner_point_row_col": point,
                            "target_memory_distance_px": distances.get(target),
                            "nearest_memory_color": min(distances, key=distances.get) if distances else None,
                        }
                    row["first_target_memory_consistency"] = {
                        "target_color": target,
                        "memory_point_row_col": points[target],
                        "arms": first_points,
                        "interpretation_limit": "memory-compilation consistency, not simulator ground truth",
                    }
                rows.append(row)

    swap = [r for r in rows if r["task"] == "VideoUnmaskSwap"]
    unmask = [r for r in rows if r["task"] == "VideoUnmask"]
    bc_both = [r for r in swap if r["arms"]["B"]["success"] and r["arms"]["C"]["success"]]
    ab_both = [r for r in unmask if r["arms"]["A"]["success"] and r["arms"]["B"]["success"]]
    return {
        "protocol": "carve.robomme.service_block_pilot.audit.v1",
        "scope": "preselected_ep20_21_two_service_blocks_small_feasibility_not_RAL_confirmation",
        "model_digest_verified": (root / "model_integrity.log").exists(),
        "runner_sha256": runner_sha,
        "policy_label": labels[0], "planner_label": labels[1],
        "swap_success_A": sum(r["arms"]["A"]["success"] for r in swap),
        "swap_success_B": sum(r["arms"]["B"]["success"] for r in swap),
        "swap_success_C": sum(r["arms"]["C"]["success"] for r in swap),
        "swap_A_to_B_rescues": sum(not r["arms"]["A"]["success"] and r["arms"]["B"]["success"] for r in swap),
        "swap_A_to_B_harms": sum(r["arms"]["A"]["success"] and not r["arms"]["B"]["success"] for r in swap),
        "swap_B_to_C_rescues": sum(not r["arms"]["B"]["success"] and r["arms"]["C"]["success"] for r in swap),
        "swap_B_to_C_harms": sum(r["arms"]["B"]["success"] and not r["arms"]["C"]["success"] for r in swap),
        "swap_B_C_both_success": len(bc_both),
        "swap_B_C_both_success_planner_B": sum(r["arms"]["B"]["planner_calls"] for r in bc_both),
        "swap_B_C_both_success_planner_C": sum(r["arms"]["C"]["planner_calls"] for r in bc_both),
        "swap_B_C_both_success_wall_B_s": sum(r["arms"]["B"]["wall_s"] for r in bc_both),
        "swap_B_C_both_success_wall_C_s": sum(r["arms"]["C"]["wall_s"] for r in bc_both),
        "unmask_success_A": sum(r["arms"]["A"]["success"] for r in unmask),
        "unmask_success_gated_B": sum(r["arms"]["B"]["success"] for r in unmask),
        "unmask_A_to_B_harms": sum(r["arms"]["A"]["success"] and not r["arms"]["B"]["success"] for r in unmask),
        "unmask_A_B_both_success": len(ab_both),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for row in result["rows"]:
        print(row["block"], row["task"], row["episode"], {
            arm: (value["status"], value["steps"], value["planner_calls"])
            for arm, value in row["arms"].items()
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
