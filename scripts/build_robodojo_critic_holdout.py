#!/usr/bin/env python3
"""Build a pre-registered three-view RoboDojo completion-Critic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


TASK_INSTRUCTIONS = {
    "build_tower": "Build a tower using the wooden blocks and wooden boards.",
    "put_bottles_into_dustbin": (
        "Pick up the bottles and throw them into the dustbin, using handover when needed."
    ),
}
CAMERAS = ("head", "left_wrist", "right_wrist")
NEGATIVE_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(frame_count: int) -> list[tuple[str, int, str]]:
    if frame_count < 20:
        raise ValueError("completion-Critic videos must contain at least 20 frames")
    samples = [
        (f"p{int(fraction * 100):02d}", min(frame_count - 2, int((frame_count - 1) * fraction)), "continue")
        for fraction in NEGATIVE_FRACTIONS
    ]
    samples.append(("terminal", frame_count - 1, "safe_stop"))
    return samples


def _read_frame(video: Path, frame_index: int) -> tuple[Any, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= frame_index < frame_count:
            raise IndexError(f"frame {frame_index} outside [0, {frame_count}) for {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"unable to decode frame {frame_index} from {video}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_count
    finally:
        capture.release()


def _load_run(run_dir: Path, split: str, image_root: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "carve.robodojo.nominal.v1":
        raise ValueError(f"unsupported RoboDojo summary: {summary_path}")
    if not summary["official_result"]["success"]:
        raise ValueError(f"Critic completion dataset requires a successful run: {run_dir}")
    task = str(summary["task"])
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(f"missing task instruction for {task}")

    videos = {
        camera: run_dir / f"episode_0000000_cam_{camera}_success.mp4"
        for camera in CAMERAS
    }
    for video in videos.values():
        if not video.is_file():
            raise FileNotFoundError(video)
    expected_frames = int(summary["official_result"]["video_frames"])
    records: list[dict[str, Any]] = []
    for tag, frame_index, control_target in _sample_indices(expected_frames):
        sample_id = f"{split}_{task}_set{summary['seed']}_{summary['condition'].lower()}_{tag}"
        sample_dir = image_root / split / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, str] = {}
        for camera, video in videos.items():
            frame, decoded_count = _read_frame(video, frame_index)
            if decoded_count != expected_frames:
                raise ValueError(
                    f"video/summary frame mismatch for {video}: {decoded_count} != {expected_frames}"
                )
            output = sample_dir / f"{camera}.png"
            if not cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"unable to write frame: {output}")
            frames[f"current_{camera}"] = str(output.relative_to(image_root.parent))
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "task": task,
                "task_instruction": TASK_INSTRUCTIONS[task],
                "condition": summary["condition"],
                "layout_set": int(summary["seed"]),
                "frame_index": frame_index,
                "video_frames": expected_frames,
                "normalized_progress": frame_index / (expected_frames - 1),
                "control_target": control_target,
                "expected_complete": control_target == "safe_stop",
                "frames": frames,
                "source_videos": {
                    camera: {
                        "path": str(video),
                        "sha256": _sha256(video),
                    }
                    for camera, video in videos.items()
                },
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, action="append", default=[])
    parser.add_argument("--holdout-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.development_run or not args.holdout_run:
        raise ValueError("at least one development and one holdout run are required")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = output.parent / "images"
    records: list[dict[str, Any]] = []
    for split, runs in (
        ("development", args.development_run),
        ("holdout", args.holdout_run),
    ):
        for run in runs:
            records.extend(_load_run(run, split, image_root))
    payload = {
        "schema_version": "carve.robodojo.critic-holdout.v1",
        "expected_outcome": (
            "The complete requested task is visibly finished, with every required object "
            "and stage completed."
        ),
        "sampling": {
            "negative_progress_fractions": list(NEGATIVE_FRACTIONS),
            "positive_frame": "official first-success terminal frame",
            "views": list(CAMERAS),
        },
        "label_policy": (
            "Only the official first-success terminal frame permits safe_stop. Earlier "
            "frames were evaluated before terminal success and must continue or replan."
        ),
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "development_samples": sum(r["split"] == "development" for r in records),
                "holdout_samples": sum(r["split"] == "holdout" for r in records),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
