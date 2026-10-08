#!/usr/bin/env python3
"""Audit RoboMME public-demo preflight before any paired model rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_frame_sha256(video: Path) -> str:
    frame = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "ppm", "-"],
        check=True, capture_output=True,
    ).stdout
    if not frame.startswith(b"P6\n"):
        raise ValueError(f"unable to decode the first RGB frame: {video}")
    return hashlib.sha256(frame).hexdigest()


def prior_rollouts(project_root: Path, preflight_root: Path) -> dict[tuple[str, int], list[str]]:
    seen: dict[tuple[str, int], list[str]] = {}
    for base in (project_root / "results", project_root / "artifacts"):
        if not base.exists():
            continue
        for path in base.rglob("summary.json"):
            if path.is_relative_to(preflight_root):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            task, episode = data.get("task"), data.get("episode")
            if isinstance(task, str) and type(episode) is int:
                seen.setdefault((task, episode), []).append(str(path.relative_to(project_root)))
    return seen


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    preflight_root = args.preflight_root.resolve()
    if not 0 <= args.episode_start <= args.episode_end < 50:
        raise ValueError("RoboMME test episode IDs must be in [0, 49]")
    prior = prior_rollouts(project_root, preflight_root)
    records = []
    issues = []
    for task in args.task:
        frame_hashes: dict[str, int] = {}
        for episode in range(args.episode_start, args.episode_end + 1):
            case = preflight_root / f"{task}_ep{episode}"
            summary = json.loads((case / "summary.json").read_text(encoding="utf-8"))
            video = case / "initial_demo_front.mp4"
            if (summary.get("protocol") != "carve.robomme.initial_demo.v1"
                    or summary.get("task") != task
                    or summary.get("episode") != episode
                    or summary.get("demo_history_mode") != "official"):
                issues.append(f"{task} ep{episode}: invalid preflight identity or history mode")
            frame_hash = first_frame_sha256(video)
            if frame_hash in frame_hashes:
                issues.append(f"{task} ep{episode}: first frame duplicates ep{frame_hashes[frame_hash]}")
            frame_hashes[frame_hash] = episode
            old_paths = prior.get((task, episode), [])
            if old_paths:
                issues.append(f"{task} ep{episode}: appears in prior summary: {old_paths[0]}")
            records.append({
                "task": task,
                "episode": episode,
                "instruction": summary["instruction"],
                "demo_frames": summary["planner_demo_frames"],
                "first_frame_sha256": frame_hash,
                "public_demo_sha256": sha256_file(video),
                "prior_summary_paths": old_paths,
            })
    result = {
        "protocol": "carve.robomme.holdout_preflight.v1",
        "model_rollouts_started": False,
        "tasks": args.task,
        "episodes": [args.episode_start, args.episode_end],
        "records": records,
        "issues": issues,
        "admitted_for_paired_gate": not issues,
        "scope_note": "First-frame uniqueness and prior summary search do not prove all simulator states are distinct.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(records), "issues": issues,
                      "admitted": not issues, "output": str(args.output)}, ensure_ascii=False))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
