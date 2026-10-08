#!/usr/bin/env python3
"""Track a structured-identity or motion-selected public demonstration object."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "third_party" / "sam2"
COLOR_RANGES = {
    "red": (((0, 170, 70), (7, 255, 255)), ((170, 170, 70), (179, 255, 255))),
    "green": (((35, 80, 40), (90, 255, 255)),),
    "blue": (((90, 80, 40), (135, 255, 255)),),
}
RELATIONS = (
    "bottom-left",
    "bottom-right",
    "top-left",
    "top-right",
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost",
    "middle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--structured-memory", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    )
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--motion-select",
        action="store_true",
        help="Select the initially manipulated same-color instance from demo motion.",
    )
    return parser.parse_args()


def initial_identity(memory: dict[str, Any]) -> str:
    for step in memory.get("demonstrated_sequence", []):
        if isinstance(step, str):
            match = re.search(
                r"\b(?:grasp(?:ed)?|pick(?:ed)?(?:\s+up)?)\s+(?:the\s+)?(.+)",
                step,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().lower()
            continue
        if not isinstance(step, dict):
            continue
        value = step.get("object_identity") or step.get("object") or step.get("target")
        if isinstance(value, str) and value.strip() and value.strip().lower() != "button":
            return value.strip().lower()
        for interaction in step.get("object_interactions", []):
            if not isinstance(interaction, dict):
                continue
            action = str(interaction.get("action", "")).lower()
            value = interaction.get("object") or interaction.get("object_identity")
            if (
                ("grasp" in action or "pick" in action)
                and isinstance(value, str)
                and value.strip()
            ):
                location = interaction.get("location")
                suffix = f" at {location.strip()}" if isinstance(location, str) else ""
                return f"{value.strip()}{suffix}".lower()
    raise ValueError("structured memory contains no demonstrated object identity")


def parse_identity(identity: str) -> tuple[str, str]:
    color = next((value for value in COLOR_RANGES if value in identity), None)
    relation = next((value for value in RELATIONS if value in identity), None)
    if color is None or relation is None:
        raise ValueError(f"identity must contain a supported color and relation: {identity}")
    return color, relation


def detect_colored_objects(frame: np.ndarray, color: str) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(
            hsv,
            np.asarray(lower, dtype=np.uint8),
            np.asarray(upper, dtype=np.uint8),
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[dict[str, Any]] = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        fill = area / max(1, width * height)
        if not (
            8 <= width <= 45
            and 8 <= height <= 45
            and 50 <= area <= 1200
            and fill >= 0.30
        ):
            continue
        candidates.append(
            {
                "point": [
                    float(centroids[index][1]),
                    float(centroids[index][0]),
                ],
                "box": [x, y, x + width - 1, y + height - 1],
                "area": area,
            }
        )
    return candidates


def select_relation(candidates: list[dict[str, Any]], relation: str) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no colored object candidates were detected")
    if relation == "top-left":
        return min(candidates, key=lambda item: sum(item["point"]))
    if relation == "top-right":
        return max(candidates, key=lambda item: item["point"][1] - item["point"][0])
    if relation == "bottom-left":
        return max(candidates, key=lambda item: item["point"][0] - item["point"][1])
    if relation == "bottom-right":
        return max(candidates, key=lambda item: sum(item["point"]))
    if relation == "topmost":
        return min(candidates, key=lambda item: item["point"][0])
    if relation == "bottommost":
        return max(candidates, key=lambda item: item["point"][0])
    if relation == "leftmost":
        return min(candidates, key=lambda item: item["point"][1])
    if relation == "rightmost":
        return max(candidates, key=lambda item: item["point"][1])
    center = np.mean([item["point"] for item in candidates], axis=0)
    return min(candidates, key=lambda item: np.linalg.norm(np.asarray(item["point"]) - center))


def select_motion_candidate(
    frames: list[np.ndarray],
    candidates: list[dict[str, Any]],
    *,
    patch_radius: int = 15,
    change_threshold: float = 30.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the instance whose initial neighborhood changes persistently."""

    if not frames or not candidates:
        raise ValueError("motion selection requires frames and candidates")
    video = np.stack(frames).astype(np.float32)
    height, width = video.shape[1:3]
    scores: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row, col = (int(round(value)) for value in candidate["point"])
        row_start, row_end = max(0, row - patch_radius), min(height, row + patch_radius + 1)
        col_start, col_end = max(0, col - patch_radius), min(width, col + patch_radius + 1)
        patch = video[:, row_start:row_end, col_start:col_end]
        difference = np.mean(np.abs(patch - patch[0]), axis=(1, 2, 3))
        scores.append(
            {
                "candidate_index": index,
                "point": [round(float(value), 2) for value in candidate["point"]],
                "changed_frames": int(np.sum(difference > change_threshold)),
                "mean_change": float(np.mean(difference)),
                "p95_change": float(np.percentile(difference, 95)),
            }
        )
    ranked = sorted(
        scores,
        key=lambda item: (item["changed_frames"], item["p95_change"]),
        reverse=True,
    )
    selected = candidates[int(ranked[0]["candidate_index"])]
    runner_up = int(ranked[1]["changed_frames"]) if len(ranked) > 1 else 0
    diagnostic = {
        "method": "initial_patch_persistence",
        "change_threshold": change_threshold,
        "patch_radius": patch_radius,
        "scores": scores,
        "selected_candidate_index": int(ranked[0]["candidate_index"]),
        "changed_frame_margin": int(ranked[0]["changed_frames"]) - runner_up,
    }
    return selected, diagnostic


def final_relation(candidates: list[dict[str, Any]], target: dict[str, Any]) -> str:
    if len(candidates) == 1:
        return "only"
    point = np.asarray(target["point"], dtype=np.float32)
    rows = np.asarray([item["point"][0] for item in candidates])
    cols = np.asarray([item["point"][1] for item in candidates])
    row_rank = int(np.argsort(rows).tolist().index(int(np.argmin(np.abs(rows - point[0])))))
    col_rank = int(np.argsort(cols).tolist().index(int(np.argmin(np.abs(cols - point[1])))))
    last = len(candidates) - 1
    if row_rank == 0:
        return "top-left" if col_rank == 0 and len(candidates) > 2 else "topmost"
    if row_rank == last:
        if col_rank == 0:
            return "bottom-left"
        if col_rank == last:
            return "bottom-right"
        return "bottommost"
    if col_rank == 0:
        return "leftmost"
    if col_rank == last:
        return "rightmost"
    return "middle"


def mask_loss_diagnostics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [int(item["mask_area"]) == 0 for item in trace]
    runs = [len(list(group)) for value, group in itertools.groupby(missing) if value]
    return {
        "missing_frames": int(sum(missing)),
        "missing_fraction": float(sum(missing) / max(1, len(missing))),
        "loss_runs": runs,
        "reacquisitions": len(runs),
        "admitted": bool(missing) and not missing[-1] and len(runs) <= 1 and max(runs or [0]) <= 30,
    }


def identity_provenance(*, identity: str, has_structured_identity: bool, diagnostics: dict[str, Any]) -> str:
    if diagnostics["admitted"]:
        return f"tracked through the demonstration from {identity}"
    source = "retained only as an unverified structured identity" if has_structured_identity else (
        "retained only as an unverified motion-selected candidate; no VLM identity was supplied"
    )
    return f"{source}; video tracking was rejected after {diagnostics['reacquisitions']} mask loss segments"


def read_video(path: Path, frame_dir: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        cv2.imwrite(str(frame_dir / f"{index:05d}.jpg"), bgr)
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames, fps


def track_case(
    *,
    predictor: Any,
    summary_path: Path,
    memory_path: Path | None,
    output_root: Path,
    checkpoint: Path,
    motion_select: bool = False,
) -> dict[str, Any]:
    import torch

    if memory_path is None and not motion_select:
        raise ValueError("absent structured memory requires motion selection")
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path else {}
    memory = memory_payload.get("memory", memory_payload)
    identity_error = None
    try:
        identity = initial_identity(memory)
        color, relation = parse_identity(identity)
    except ValueError as error:
        if not motion_select:
            raise
        identity = ""
        color = ""
        relation = "motion-selected"
        identity_error = str(error)
    video_path = summary_path.parent / "initial_demo_front.mp4"
    task = str(summary["task"])
    episode = int(summary["episode"])
    with tempfile.TemporaryDirectory(prefix=f"carve_{task}_ep{episode}_") as temporary:
        frame_dir = Path(temporary)
        frames, fps = read_video(video_path, frame_dir)
        if color:
            initial_candidates = detect_colored_objects(frames[0], color)
        else:
            initial_candidates = []
            for candidate_color in COLOR_RANGES:
                for candidate in detect_colored_objects(frames[0], candidate_color):
                    initial_candidates.append({**candidate, "candidate_color": candidate_color})
        motion_diagnostic = None
        if motion_select:
            selected, motion_diagnostic = select_motion_candidate(
                frames, initial_candidates
            )
            if not color:
                color = str(selected["candidate_color"])
                identity = f"motion-selected {color} block"
                motion_diagnostic["identity_fallback_reason"] = identity_error
        else:
            selected = select_relation(initial_candidates, relation)
        state = predictor.init_state(video_path=str(frame_dir))
        trace: list[dict[str, Any]] = []
        overlays: list[np.ndarray] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=np.asarray(selected["box"], dtype=np.float32),
            )
            for frame_index, _, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[0] > 0).squeeze().cpu().numpy()
                rows, cols = np.where(mask)
                point = None
                if len(rows):
                    point = [float(rows.mean()), float(cols.mean())]
                trace.append(
                    {
                        "frame": int(frame_index),
                        "point": point,
                        "mask_area": int(len(rows)),
                    }
                )
                overlay = frames[frame_index].copy()
                overlay[mask] = (
                    0.55 * overlay[mask] + 0.45 * np.asarray([0, 255, 255])
                ).astype(np.uint8)
                if point is not None:
                    cv2.circle(
                        overlay,
                        (int(round(point[1])), int(round(point[0]))),
                        4,
                        (255, 255, 255),
                        -1,
                    )
                overlays.append(overlay)
        final_candidates = detect_colored_objects(frames[-1], color)
        final_point = np.asarray(trace[-1]["point"], dtype=np.float32)
        matched = min(
            final_candidates,
            key=lambda item: np.linalg.norm(np.asarray(item["point"]) - final_point),
        )
        tracked_relation = final_relation(final_candidates, matched)
        diagnostics = mask_loss_diagnostics(trace)
        resolved_relation = tracked_relation if diagnostics["admitted"] else relation

    output_root.mkdir(parents=True, exist_ok=True)
    overlay_path = output_root / f"{task}_ep{episode}_sam2_memory_track.mp4"
    imageio.mimsave(overlay_path, overlays, fps=fps)
    provenance = identity_provenance(
        identity=identity, has_structured_identity=identity_error is None, diagnostics=diagnostics
    )
    planner_hint = (
        f"The persistent target identity is {resolved_relation} {color} block, "
        f"{provenance}. Task sequence: {str(summary['instruction']).strip()}"
    )
    output = {
        "protocol": "carve.robomme.video_instance_memory.v1",
        "task": task,
        "episode": episode,
        "source_memory": str(memory_path) if memory_path else None,
        "source_demo": str(video_path),
        "source_demo_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "instruction": str(summary["instruction"]),
        "compile_time_s": time.perf_counter() - started,
        "compile_time_scope": "decode, motion selection, SAM2 tracking and overlay encoding; excludes shared model load and simulator reset",
        "evaluator_or_oracle_fields_used": False,
        "memory": {
            "planner_hint": planner_hint,
            "video_instance_tracking": {
                "target_color": color,
                "initial_identity": identity,
                "initial_box_xyxy": selected["box"],
                "initial_selection": motion_diagnostic or {
                    "method": "vlm_relation",
                    "relation": relation,
                },
                "resolved_identity": f"{resolved_relation} {color} block",
                "tracked_identity_before_admission": f"{tracked_relation} {color} block",
                "final_point_row_col": [round(value, 2) for value in matched["point"]],
                "admission": diagnostics,
                "tracker": "SAM2.1 Hiera Tiny",
                "checkpoint": str(checkpoint),
                "frame_count": len(trace),
                "trace": trace,
                "overlay_video": str(overlay_path),
            },
        },
    }
    source = video_path.resolve()
    if source.is_relative_to(PROJECT_ROOT):
        output["source_demo_relative"] = str(source.relative_to(PROJECT_ROOT))
    output_path = output_root / f"{task}_ep{episode}_memory.json"
    with output_path.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2)
        destination.write("\n")
    return output


def main() -> int:
    args = parse_args()
    if not args.structured_memory and args.motion_select:
        args.structured_memory = [None] * len(args.summary)
    if len(args.summary) != len(args.structured_memory):
        raise ValueError("--summary and --structured-memory counts must match")
    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        args.model_config,
        str(args.checkpoint),
        device=args.device,
    )
    for summary_path, memory_path in zip(args.summary, args.structured_memory):
        output = track_case(
            predictor=predictor,
            summary_path=summary_path,
            memory_path=memory_path,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            motion_select=args.motion_select,
        )
        tracking = output["memory"]["video_instance_tracking"]
        print(
            json.dumps(
                {
                    "task": output["task"],
                    "episode": output["episode"],
                    "initial_identity": tracking["initial_identity"],
                    "resolved_identity": tracking["resolved_identity"],
                    "final_point_row_col": tracking["final_point_row_col"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
