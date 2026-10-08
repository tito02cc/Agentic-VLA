#!/usr/bin/env python3
"""Package one RoboDojo controlled-fault run into an auditable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def video_frames(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return int(output.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("shadow", "fixed", "adaptive"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    official = args.official_run.resolve()
    artifact.mkdir(parents=True, exist_ok=True)

    shutil.copy2(official / "_result.json", artifact / "result.json")
    for video in official.glob("*.mp4"):
        shutil.copy2(video, artifact / video.name)

    runtime = load_jsonl(artifact / "runtime_trace.jsonl")
    planner_path = artifact / "planner_trace.jsonl"
    planner = load_jsonl(planner_path) if planner_path.exists() else []
    monitor = load_jsonl(artifact / "monitor_trace.jsonl")
    result = json.loads((artifact / "result.json").read_text())
    detail = result["details"]["0"]
    latencies = [float(row["model_latency_ms"]) for row in runtime]
    boost = [
        row
        for row in runtime
        if row["metadata"]["controller"]["compute_phase"] == "recovery_boost"
    ]
    critic = [row for row in planner if row.get("role") == "critic"]
    accepted = [row for row in critic if row.get("accepted")]
    monitor_events = [
        row for row in monitor if row.get("assessment", {}).get("event") == "no_progress"
    ]

    head_video = next(artifact.glob("episode_*_cam_head_*.mp4"))
    frames = video_frames(head_video)
    selected = sorted({0, min(320, frames - 1), min(639, frames - 1), min(690, frames - 1), min(738, frames - 1), frames - 1})
    expression = "+".join(f"eq(n\\,{frame})" for frame in selected)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(head_video),
            "-vf",
            f"select={expression},scale=480:-1,tile=3x2",
            "-vsync",
            "vfr",
            "-frames:v",
            "1",
            str(artifact / "head_contact_sheet.png"),
        ],
        check=True,
    )

    first = runtime[0]
    server_metadata = first["metadata"]["optimization_profile"]["server_metadata"]
    last_critic = accepted[-1] if accepted else None
    condition_name = {
        "shadow": "shadow_fault_no_intervention",
        "fixed": "C3_full_fault_fixed_compute",
        "adaptive": "C3_full_fault_adaptive_compute",
    }[args.condition]
    summary = {
        "schema_version": "carve.robodojo.controlled_fault.v1",
        "condition": condition_name,
        "benchmark": "RoboDojo",
        "task": "build_tower",
        "layout_id": int(detail["layout_id"]),
        "seed": args.seed,
        "seed_semantics": "StarVLA policy-server RNG seed; RoboDojo environment seed remains 0",
        "policy": "StarVLA PI-v3",
        "runtime": {
            "vla_calls": len(runtime),
            "nominal_calls": len(runtime) - len(boost),
            "recovery_boost_calls": len(boost),
            "model_latency_p50_ms": nearest_rank(latencies, 0.5),
            "model_latency_p95_ms": nearest_rank(latencies, 0.95),
            "deadline_misses": sum(bool(row.get("deadline_miss")) for row in runtime),
            "recovery_boost_timesteps": [int(row["timestep"]) for row in boost],
        },
        "agentic": {
            "fault_step": 640,
            "first_monitor_event_step": int(monitor_events[0]["timestep"]) if monitor_events else None,
            "critic_calls": len(critic),
            "accepted_critic_calls": len(accepted),
            "critic_status": last_critic["report"]["status"] if last_critic else None,
            "critic_confidence": last_critic["report"]["confidence"] if last_critic else None,
            "critic_latency_ms": last_critic["elapsed_ms"] if last_critic else None,
        },
        "official_result": {
            "success": bool(detail["success"]),
            "success_rate": float(result["success_rate"]),
            "score": float(detail["score"]),
            "video_frames": frames,
        },
        "reproducibility": {
            "environment_seed": 0,
            "policy_server_seed": server_metadata.get("runtime_seed"),
            "runtime_seed_visible_in_handshake": "runtime_seed" in server_metadata,
        },
        "claim_boundary": "This run verifies the official evaluator, event-triggered Critic, bounded recovery, runtime traces, and videos. Aggregate success claims must use the complete multi-seed matrix rather than this episode alone.",
    }
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    checksum_lines = []
    for path in sorted(p for p in artifact.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}")
    (artifact / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


if __name__ == "__main__":
    main()
