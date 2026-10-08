#!/usr/bin/env python3
"""Locate control and visible-rollout divergence in the saved RoboMME B/C batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/robomme/unmaskswap_fixed_five_20260923"
NEW = ROOT / "artifacts/robomme/unmaskswap_runtime_bc_20260923"


def load_summary(root: Path, episode: int, arm: str) -> dict:
    path = root / "rollouts" / f"VideoUnmaskSwap_ep{episode}_{arm}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def first_difference(left: list[dict], right: list[dict], key: str) -> int | None:
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a[key] != b[key]:
            return index
    return min(len(left), len(right)) + 1 if len(left) != len(right) else None


def compare_video(path_a: Path, path_b: Path) -> dict:
    a = cv2.VideoCapture(str(path_a))
    b = cv2.VideoCapture(str(path_b))
    if not a.isOpened() or not b.isOpened():
        raise FileNotFoundError(f"cannot read videos: {path_a}, {path_b}")
    first_pixel = None
    first_visible = None
    first_large = None
    first_16_identical = True
    compared = 0
    try:
        while True:
            ok_a, frame_a = a.read()
            ok_b, frame_b = b.read()
            if not (ok_a and ok_b):
                break
            if frame_a.shape != frame_b.shape:
                raise ValueError("compared videos have different frame dimensions")
            diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
            mae = float(diff.mean())
            if first_pixel is None and np.any(diff):
                first_pixel = compared
            if first_visible is None and mae > 0.1:
                first_visible = compared
            if first_large is None and mae > 1.0:
                first_large = compared
            if compared < 16 and np.any(diff):
                first_16_identical = False
            compared += 1
    finally:
        a.release()
        b.release()
    if compared == 0:
        raise ValueError("videos have no comparable frames")
    return {
        "frames_compared": compared,
        "first_pixel_difference_frame_zero_based": first_pixel,
        "first_mae_gt_0_1_frame_zero_based": first_visible,
        "first_mae_gt_1_frame_zero_based": first_large,
        "first_16_frames_identical": first_16_identical,
        "visibility_note": "H.264 decoded RGB differences; exact action SHA can differ without visible pixels changing",
    }


def video_path(root: Path, episode: int, arm: str) -> Path:
    name = f"VideoUnmaskSwap_ep{episode}"
    return root / "rollouts" / f"{name}_{arm}" / f"{name}_vlm_groundsg.mp4"


def comparison(a: dict, b: dict, a_video: Path, b_video: Path) -> dict:
    if a["initial_observation_sha256"] != b["initial_observation_sha256"]:
        raise ValueError("initial observation changed")
    if (a["memory_provenance"] or {}).get("memory_sha256") != (
        b["memory_provenance"] or {}
    ).get("memory_sha256"):
        raise ValueError("identity memory changed")
    input_index = first_difference(a["policy_request_audit"], b["policy_request_audit"], "input_sha256")
    action_index = first_difference(a["policy_request_audit"], b["policy_request_audit"], "actions_sha256")
    subgoal_index = first_difference(a["schedule_trace"], b["schedule_trace"], "grounded_subgoal")
    return {
        "first_policy_input_sha_divergence_request_one_based": input_index,
        "first_policy_action_sha_divergence_request_one_based": action_index,
        "first_grounded_subgoal_divergence_chunk_one_based": subgoal_index,
        "subgoal_at_divergence": (
            {
                "step_a": a["schedule_trace"][subgoal_index - 1]["step"],
                "step_b": b["schedule_trace"][subgoal_index - 1]["step"],
                "a": a["schedule_trace"][subgoal_index - 1]["grounded_subgoal"],
                "b": b["schedule_trace"][subgoal_index - 1]["grounded_subgoal"],
            }
            if subgoal_index is not None
            and subgoal_index <= min(len(a["schedule_trace"]), len(b["schedule_trace"]))
            else None
        ),
        "video": compare_video(a_video, b_video),
    }


def audit() -> dict:
    rows = []
    for episode in range(15, 20):
        old_b = load_summary(OLD, episode, "B")
        new_b = load_summary(NEW, episode, "B")
        new_c = load_summary(NEW, episode, "C")
        if {k: v for k, v in old_b["run_config"].items() if k != "output"} != {
            k: v for k, v in new_b["run_config"].items() if k != "output"
        }:
            raise ValueError(f"ep{episode}: old and new B settings changed")
        rows.append({
            "episode": episode,
            "status_old_B": old_b["status"],
            "status_new_B": new_b["status"],
            "status_new_C": new_c["status"],
            "old_B_vs_new_B": comparison(
                old_b, new_b, video_path(OLD, episode, "B"), video_path(NEW, episode, "B")
            ),
            "new_B_vs_new_C": comparison(
                new_b, new_c, video_path(NEW, episode, "B"), video_path(NEW, episode, "C")
            ),
        })
    return {
        "protocol": "carve.robomme.runtime_divergence_audit.v1",
        "scope": "CPU_only_saved_inputs_actions_subgoals_and_H264_frames",
        "old_runner_sha256": load_summary(OLD, 15, "B")["runner_sha256"],
        "new_runner_sha256": load_summary(NEW, 15, "B")["runner_sha256"],
        "old_weight_digest_available": False,
        "new_weight_digest_path": str(NEW / "model_sha256.txt"),
        "interpretation_limit": (
            "An action-byte mismatch is not proof of material motion divergence. "
            "Old service weights and server environment were not cryptographically snapshotted; "
            "the cause of cross-service drift is unresolved."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=NEW / "repro_audit.json")
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        old = row["old_B_vs_new_B"]
        new = row["new_B_vs_new_C"]
        print(
            row["episode"], row["status_old_B"], row["status_new_B"], row["status_new_C"],
            "old/new B first subgoal", old["first_grounded_subgoal_divergence_chunk_one_based"],
            "old/new B first visible frame", old["video"]["first_mae_gt_0_1_frame_zero_based"],
            "new B/C first subgoal", new["first_grounded_subgoal_divergence_chunk_one_based"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
