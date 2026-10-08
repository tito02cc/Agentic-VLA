"""Admit demonstration-derived identity memory without evaluator state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


class MemoryUncertainError(ValueError):
    """A valid, bound observation did not produce usable identity evidence."""


def planner_demo_frames(frames: list, mode: str) -> list:
    if mode == "official":
        return frames[:-1]
    if mode == "legacy_full":
        return frames
    raise ValueError("unknown demonstration history mode")


def validate_identity_memory(
    document: dict[str, Any], *, task: str, episode: int, instruction: str,
    hint_style: str = "precise",
) -> dict[str, Any]:
    if hint_style not in {"precise", "coarse_grid"}:
        raise ValueError("unsupported identity memory hint style")
    if task == "VideoRepick":
        if hint_style != "precise":
            raise ValueError("coarse grid hint is only supported for the VideoUnmask family")
        return validate_repick_memory(document, task=task, episode=episode, instruction=instruction)
    protocols = {
        "VideoUnmask": "carve.robomme.video_unmask_memory.v1",
        "VideoUnmaskSwap": "carve.robomme.unmask_swap_memory.v1",
    }
    if task not in protocols or document.get("protocol") != protocols[task]:
        raise ValueError("unsupported identity memory protocol/task")
    if document.get("task") != task or document.get("episode") != episode:
        raise ValueError("identity memory belongs to another task/episode")
    if document.get("evaluator_or_oracle_fields_used") is not False:
        raise ValueError("memory must declare public demonstration provenance")
    if "instruction" in document and document["instruction"] != instruction:
        raise ValueError("identity memory instruction mismatch")
    admission = document.get("admission")
    if admission is not None:
        if not isinstance(admission, dict) or type(admission.get("admitted")) is not bool:
            raise ValueError("malformed memory admission")
        if not admission["admitted"]:
            if not isinstance(admission.get("reason"), str) or not admission["reason"].strip():
                raise ValueError("rejected memory must state its uncertainty reason")
            if document.get("instruction") != instruction:
                raise ValueError("rejected memory must bind the full instruction")
            if document.get("memory") != {}:
                raise ValueError("rejected memory must not contain partially admitted hints")
            raise MemoryUncertainError(admission["reason"])
    memory = document["memory"]
    requested = re.findall(r"\b(red|green|blue)\b", instruction.lower())
    if not requested or memory.get("required_color_order") != requested:
        raise ValueError("identity memory instruction/order mismatch")
    points = {}
    for color in ("red", "green", "blue"):
        point = np.asarray(memory["hidden_container_points"][color], dtype=float)
        track = memory["video_instance_tracking"]["objects"][color]
        if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any((point < 0) | (point >= 256)):
            raise ValueError("invalid identity point")
        if type(track["admission"].get("admitted")) is not bool:
            raise ValueError("malformed tracking admission")
        if not track["admission"]["admitted"]:
            raise ValueError("identity tracking was not admitted; recompile an explicit rejection record")
        last = np.asarray(track["trace"][-1]["point"], dtype=float)
        if last.shape != (2,) or not np.allclose(point, last, atol=0.011, rtol=0):
            raise ValueError("identity point does not match its source trace")
        points[color] = point.tolist()
    if hint_style == "coarse_grid":
        cells = {color: (int(point[0] // 32), int(point[1] // 32))
                 for color, point in points.items()}
        if len(set(cells.values())) != len(cells):
            raise MemoryUncertainError("two hidden-container identities share one coarse grid cell")
        locations = "; ".join(
            f"{color}-hidden container in grid cell (row {row}, column {col})"
            for color, (row, col) in cells.items()
        )
        coordinate_context = (
            "(256x256 front RGB divided into an 8x8 grid; row and column indices "
            "start at 0 from the top left): "
        )
        grounding_instruction = (
            "These cells are rough historical identity regions, not grasp points, "
            "verified current positions or completed actions. Use the current image "
            "to choose a fresh exact point for the next grounded subgoal."
        )
    else:
        locations = "; ".join(
            f"{color}-hidden container at <|box_start|>"
            f"({int(round(point[0] * 1000 / 256))},{int(round(point[1] * 1000 / 256))})<|box_end|>"
            for color, point in points.items()
        )
        coordinate_context = "(GroundSG normalized row,column coordinates, 0 to 1000): "
        grounding_instruction = (
            "These are historical positions, not verified current positions "
            "or completed actions. Use the current image for the next grounded subgoal."
        )
    # Only structured observations enter the prompt, not arbitrary cached instructions.
    location_context = (
        "at the end of the swap" if task == "VideoUnmaskSwap"
        else ("in its final covered frame" if hint_style == "coarse_grid"
              else "at the end of the public demonstration")
    )
    return {
        "planner_hint": (
            f"Identity memory from the public demonstration, {location_context} "
            + coordinate_context + locations + ". " + grounding_instruction
        ),
        "hidden_container_points": points,
        "required_color_order": requested,
    }


def validate_repick_memory(
    document: dict[str, Any], *, task: str, episode: int, instruction: str,
) -> dict[str, Any]:
    if document.get("protocol") != "carve.robomme.video_instance_memory.v1":
        raise ValueError("unsupported VideoRepick memory protocol")
    if document.get("task") != task or document.get("episode") != episode:
        raise ValueError("identity memory belongs to another task/episode")
    if document.get("instruction") != instruction or document.get("evaluator_or_oracle_fields_used") is not False:
        raise ValueError("VideoRepick instruction/provenance mismatch")
    track = document["memory"]["video_instance_tracking"]
    selection = track["initial_selection"]
    color = track.get("target_color")
    if (track["admission"].get("admitted") is not True
            or selection.get("method") != "initial_patch_persistence"
            or selection.get("changed_frame_margin", 0) <= 0
            or color not in {"red", "green", "blue"}):
        raise ValueError("VideoRepick identity selection/tracking was not admitted")
    point = np.asarray(track["trace"][-1]["point"], dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any((point < 0) | (point >= 256)):
        raise ValueError("invalid VideoRepick tracked point")
    resolved_identity = track.get("resolved_identity")
    if (not isinstance(resolved_identity, str)
            or not re.fullmatch(r"(?:topmost|bottommost|leftmost|rightmost|middle|"
                                r"bottom-left|bottom-right|top-left|top-right) "
                                r"(?:red|green|blue) block", resolved_identity)
            or not resolved_identity.endswith(f"{color} block")):
        raise ValueError("invalid VideoRepick tracked identity")
    return {"planner_hint": (
        f"Identity memory from the public demonstration: the motion-selected target "
        f"was the {resolved_identity} in the final demonstration frame. "
        "This historical relation identifies the object; it is not a current position "
        "or a completed repetition. Ground the next action point from the current image."
    )}


def identity_motion_evidence(document: dict[str, Any]) -> dict[str, Any]:
    """Measure identity relocation from public demonstration tracks only."""

    if document.get("task") not in {"VideoUnmask", "VideoUnmaskSwap"}:
        raise ValueError("motion gate requires VideoUnmask-family memory")
    objects = document["memory"]["video_instance_tracking"]["objects"]
    deltas = []
    for color in ("red", "green", "blue"):
        trace = objects[color]["trace"]
        if not trace:
            raise MemoryUncertainError(f"missing {color} container track")
        first = np.asarray(trace[0]["point"], dtype=float)
        last = np.asarray(trace[-1]["point"], dtype=float)
        if (first.shape != (2,) or last.shape != (2,)
                or not np.all(np.isfinite(first)) or not np.all(np.isfinite(last))):
            raise MemoryUncertainError(f"invalid {color} track endpoints")
        deltas.append(last - first)
    motion = np.asarray(deltas, dtype=float)
    common_motion = np.median(motion, axis=0)
    relative = np.linalg.norm(motion - common_motion, axis=1)
    threshold_px = 16.0
    return {
        "source": "public_demo_sam2_track_endpoints",
        "common_motion_row_col_px": common_motion.tolist(),
        "relative_motion_px": dict(zip(("red", "green", "blue"), relative.tolist())),
        "max_relative_motion_px": float(np.max(relative)),
        "threshold_px": threshold_px,
        "use_memory": bool(np.max(relative) >= threshold_px),
    }


def load_verified_identity_memory(
    path: Path, *, current_demo: Path, task: str, episode: int, instruction: str,
    on_uncertain: str = "error", hint_style: str = "precise",
    memory_use_policy: str = "always",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import imageio.v2 as imageio

    if on_uncertain not in {"error", "without_memory"}:
        raise ValueError("unknown uncertainty policy")
    if memory_use_policy not in {"always", "motion_gate"}:
        raise ValueError("unknown memory use policy")
    if memory_use_policy == "motion_gate" and hint_style != "precise":
        raise ValueError("motion gate must use the original precise memory hint")
    data = path.read_bytes()
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError("memory document must be an object")
    # Verify the source even when tracking failed. A foreign cache is not uncertainty.
    source = Path(document["source_demo"])
    if "source_demo_relative" in document:
        root = Path(__file__).resolve().parents[2]
        relative = Path(document["source_demo_relative"])
        if relative.is_absolute():
            raise ValueError("source_demo_relative must be project-relative")
        source = (root / relative).resolve()
        source.relative_to(root)
    source_bytes = source.read_bytes()
    current_bytes = current_demo.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    current_hash = hashlib.sha256(current_bytes).hexdigest()
    if document.get("source_demo_sha256", source_hash) != source_hash:
        raise ValueError("identity memory source hash mismatch")
    match = "encoded_bytes"
    if source_hash != current_hash:
        source_reader = imageio.get_reader(source)
        current_reader = imageio.get_reader(current_demo)
        from itertools import zip_longest
        count = 0
        try:
            for left, right in zip_longest(source_reader, current_reader):
                if left is None or right is None or not np.array_equal(left, right):
                    raise ValueError("cached identity memory demonstration does not match this episode")
                count += 1
        finally:
            source_reader.close()
            current_reader.close()
        if not count:
            raise ValueError("empty identity memory demonstration")
        match = "decoded_rgb_exact"
    costs = {}
    for key in ("compile_time_s", "shared_model_load_s"):
        value = document.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not np.isfinite(value) or value < 0):
            raise ValueError(f"invalid memory cost: {key}")
        costs[key] = value
    reason = None
    motion_evidence = None
    try:
        admitted_memory = validate_identity_memory(
            document, task=task, episode=episode, instruction=instruction,
            hint_style=hint_style,
        )
        if memory_use_policy == "motion_gate":
            motion_evidence = identity_motion_evidence(document)
    except MemoryUncertainError as error:
        if on_uncertain == "error":
            raise
        admitted_memory, reason = None, str(error)
    memory = admitted_memory
    if memory is not None and motion_evidence is not None and not motion_evidence["use_memory"]:
        memory = None
    return memory, {
        "memory_sha256": hashlib.sha256(data).hexdigest(),
        "source_demo": str(source), "source_demo_sha256": source_hash,
        "current_demo_sha256": current_hash, "demonstration_match": match,
        "source": "public_initial_demonstration_rgb",
        "memory_kind": "episode_identity", "completion_evidence": False,
        "hint_style": hint_style, "memory_use_policy": memory_use_policy,
        "motion_evidence": motion_evidence,
        "offline_compilation_cost_included": False,
        "admitted": admitted_memory is not None,
        "used": memory is not None,
        "fallback": "without_identity_memory" if admitted_memory is None else None,
        "rejection_reason": reason,
        **costs,
    }
