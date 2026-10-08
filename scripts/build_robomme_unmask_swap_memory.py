#!/usr/bin/env python3
"""Compile VideoUnmask family demonstrations into color-to-container memory."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.track_robomme_video_memory import (
    DEFAULT_SAM2_ROOT,
    detect_colored_objects,
    mask_loss_diagnostics,
    read_video,
)
from agentic_vla.benchmarks.robomme_memory import MemoryUncertainError


COLORS = ("red", "green", "blue")
UNMASK_TASK_PROTOCOLS = {
    "VideoUnmask": "carve.robomme.video_unmask_memory.v1",
    "VideoUnmaskSwap": "carve.robomme.unmask_swap_memory.v1",
}
OVERLAY_COLORS = {
    "red": np.asarray([255, 64, 64]),
    "green": np.asarray([64, 255, 64]),
    "blue": np.asarray([64, 128, 255]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    )
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def detect_white_containers(frame: np.ndarray) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = np.asarray(
        (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 130), dtype=np.uint8
    )
    mask[:35] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not (
            100 <= area <= 700
            and 10 <= width <= 40
            and 10 <= height <= 40
        ):
            continue
        candidates.append(
            {
                "point": [float(centroids[index][1]), float(centroids[index][0])],
                "box": [x, y, x + width - 1, y + height - 1],
                "area": area,
            }
        )
    return candidates


def first_cover_frame(frames: list[np.ndarray], minimum: int = 3) -> tuple[int, list[dict[str, Any]]]:
    for index, frame in enumerate(frames):
        candidates = detect_white_containers(frame)
        if len(candidates) >= minimum:
            return index, candidates
    raise MemoryUncertainError("could not find the initial covered-container frame")


def associate_colors(
    initial_frame: np.ndarray, containers: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    associations: dict[str, dict[str, Any]] = {}
    used: set[int] = set()
    for color in COLORS:
        colored = detect_colored_objects(initial_frame, color)
        if len(colored) != 1:
            raise MemoryUncertainError(f"expected one {color} cube, found {len(colored)}")
        source = np.asarray(colored[0]["point"], dtype=np.float32)
        choices = [
            (index, candidate)
            for index, candidate in enumerate(containers)
            if index not in used
        ]
        selected_index, selected = min(
            choices,
            key=lambda item: np.linalg.norm(
                np.asarray(item[1]["point"], dtype=np.float32) - source
            ),
        )
        distance = float(
            np.linalg.norm(np.asarray(selected["point"], dtype=np.float32) - source)
        )
        if distance > 20.0:
            raise MemoryUncertainError(f"{color} cube-to-container association is too far: {distance:.1f}")
        used.add(selected_index)
        associations[color] = {**selected, "source_cube_point": colored[0]["point"]}
    return associations


def target_colors(instruction: str) -> list[str]:
    return [value.lower() for value in re.findall(r"\b(red|green|blue)\b", instruction)]


def _compile_case(
    *,
    predictor: Any,
    summary_path: Path,
    output_root: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    import torch

    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task = str(summary["task"])
    episode = int(summary["episode"])
    instruction = str(summary["instruction"])
    video_path = summary_path.parent / "initial_demo_front.mp4"
    with tempfile.TemporaryDirectory(prefix=f"carve_unmask_ep{episode}_") as temporary:
        all_frame_dir = Path(temporary) / "all"
        track_frame_dir = Path(temporary) / "track"
        all_frame_dir.mkdir()
        track_frame_dir.mkdir()
        frames, fps = read_video(video_path, all_frame_dir)
        cover_index, containers = first_cover_frame(frames)
        associations = associate_colors(frames[0], containers)
        tracked_frames = frames[cover_index:]
        for index, frame in enumerate(tracked_frames):
            cv2.imwrite(
                str(track_frame_dir / f"{index:05d}.jpg"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        state = predictor.init_state(video_path=str(track_frame_dir))
        traces: dict[str, list[dict[str, Any]]] = {color: [] for color in COLORS}
        overlays = [frame.copy() for frame in tracked_frames]
        object_ids = {index + 1: color for index, color in enumerate(COLORS)}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if predictor.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            for obj_id, color in object_ids.items():
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.asarray(associations[color]["box"], dtype=np.float32),
                )
            for frame_index, output_ids, mask_logits in predictor.propagate_in_video(state):
                for mask_index, obj_id in enumerate(output_ids):
                    color = object_ids[int(obj_id)]
                    mask = (mask_logits[mask_index] > 0).squeeze().cpu().numpy()
                    rows, cols = np.where(mask)
                    point = [float(rows.mean()), float(cols.mean())] if len(rows) else None
                    traces[color].append(
                        {
                            "frame": int(frame_index + cover_index),
                            "point": point,
                            "mask_area": int(len(rows)),
                        }
                    )
                    overlays[frame_index][mask] = (
                        0.55 * overlays[frame_index][mask]
                        + 0.45 * OVERLAY_COLORS[color]
                    ).astype(np.uint8)

    tracking: dict[str, Any] = {}
    for color in COLORS:
        diagnostics = mask_loss_diagnostics(traces[color])
        final_point = traces[color][-1]["point"] if traces[color] else None
        if final_point is None or not diagnostics["admitted"]:
            raise MemoryUncertainError(f"SAM2 tracking was not admitted for {color}")
        tracking[color] = {
            "source_cube_point_row_col": [
                round(float(value), 2) for value in associations[color]["source_cube_point"]
            ],
            "initial_container_box_xyxy": associations[color]["box"],
            "final_point_row_col": [round(float(value), 2) for value in final_point],
            "admission": diagnostics,
            "trace": traces[color],
        }

    requested = target_colors(instruction)
    point_text = "; ".join(
        f"{color}-hidden container is at <{tracking[color]['final_point_row_col'][0]:.1f}, "
        f"{tracking[color]['final_point_row_col'][1]:.1f}>"
        for color in COLORS
    )
    planner_hint = (
        "Demonstration RGB+SAM2 container identity memory at the end of the public video: "
        f"{point_text}. Required pick order: {', '.join(requested)}. "
        "For each requested color, pick its tracked container and then put it down."
    )
    output_root.mkdir(parents=True, exist_ok=True)
    overlay_path = output_root / f"{task}_ep{episode}_sam2_container_memory.mp4"
    imageio.mimsave(overlay_path, overlays, fps=fps)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    output = {
        "protocol": UNMASK_TASK_PROTOCOLS[task],
        "task": task,
        "episode": episode,
        "source_demo": str(video_path),
        "evaluator_or_oracle_fields_used": False,
        "compile_time_s": time.perf_counter() - started,
        "compile_time_scope": "video decode, RGB association, SAM2 tracking, overlay encoding; excludes shared model load and simulator reset",
        "memory": {
            "planner_hint": planner_hint,
            "hidden_container_points": {
                color: tracking[color]["final_point_row_col"] for color in COLORS
            },
            "required_color_order": requested,
            "video_instance_tracking": {
                "cover_frame": cover_index,
                "tracker": "SAM2.1 Hiera Tiny",
                "checkpoint": str(checkpoint),
                "objects": tracking,
                "overlay_video": str(overlay_path),
            },
        },
    }
    return output


def compile_case(*, predictor: Any, summary_path: Path, output_root: Path, checkpoint: Path,
                 shared_model_load_s: float | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["task"] not in UNMASK_TASK_PROTOCOLS:
        raise ValueError("this compiler only supports VideoUnmask and VideoUnmaskSwap")
    output_path = output_root / f"{summary['task']}_ep{summary['episode']}_memory.json"
    if output_path.exists():
        raise FileExistsError("memory already exists; do not overwrite evidence")
    source = (summary_path.parent / "initial_demo_front.mp4").resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        output = _compile_case(predictor=predictor, summary_path=summary_path,
                               output_root=output_root, checkpoint=checkpoint)
        output["admission"] = {"admitted": True, "reason": None}
    except MemoryUncertainError as error:
        output = {"protocol": UNMASK_TASK_PROTOCOLS[summary["task"]],
                  "task": summary["task"], "episode": summary["episode"],
                  "evaluator_or_oracle_fields_used": False, "memory": {},
                  "admission": {"admitted": False, "reason": str(error)}}
    output.update(source_demo=str(source), source_demo_sha256=source_hash,
                  instruction=summary["instruction"], demo_history_mode=summary.get("demo_history_mode", "legacy_full"),
                  compile_time_s=time.perf_counter() - started,
                  compile_time_scope="source validation, decode, association, tracking, overlay; excludes shared model load",
                  shared_model_load_s=shared_model_load_s)
    if source.is_relative_to(PROJECT_ROOT):
        output["source_demo_relative"] = str(source.relative_to(PROJECT_ROOT))
    output_root.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2)
        destination.write("\n")
    return output


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.output_root):
        pass
    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    load_started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        args.model_config, str(args.checkpoint), device=args.device
    )
    if predictor.device.type == "cuda":
        import torch
        torch.cuda.synchronize()
    shared_model_load_s = time.perf_counter() - load_started
    for summary_path in args.summary:
        output = compile_case(
            predictor=predictor,
            summary_path=summary_path,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            shared_model_load_s=shared_model_load_s,
        )
        print(
            json.dumps(
                {
                    "task": output["task"],
                    "episode": output["episode"],
                    "admission": output["admission"],
                    "hidden_container_points": output["memory"].get("hidden_container_points"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
