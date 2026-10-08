#!/usr/bin/env python3
"""Run RoboMME with a deployable VLM grounded-subgoal provider and PI0.5."""

from __future__ import annotations

import argparse
import base64
import copy
from contextlib import ExitStack
import hashlib
import io
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio.v3 as iio
import cv2
import numpy as np

from agentic_vla.benchmarks.robomme_runtime import pack_robomme_state
from agentic_vla.benchmarks.robomme_memory import load_verified_identity_memory, planner_demo_frames
from agentic_vla.runtime import (
    GroundedSubgoalScheduleConfig,
    GroundedSubgoalScheduleDecision,
    GroundedSubgoalScheduleMode,
    GroundedSubgoalScheduler,
)
from scripts.run_robomme_policy_admission import (
    pack_history,
    paired_frame,
    percentile,
    wait_for_reset,
)


BOX_PATTERN = re.compile(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>")
SCALED_POINT_PATTERN = re.compile(r"<(\d+),\s*(\d+)>")
UNMASK_SWAP_PICK_PATTERN = re.compile(
    r"pick up the container at <(\d+),\s*(\d+)> that hides the (red|green|blue) cube"
)
GROUNDING_RANGE = range(0, 256)
GROUND_SG_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of grounded language subgoals"
)
VISUAL_GROUNDING_VERBS = re.compile(
    r"\b(pick(?: up)?|grasp|press (?:the )?button|place .+ (?:target|onto)|move to .+button)\b",
    re.IGNORECASE,
)
RELATIONAL_OBJECT_ACTIONS = re.compile(r"\b(pick(?: up)?|grasp)\b", re.IGNORECASE)
REPETITION_WORDS = {"once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "four": 4}
ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


class StopCubeTemporalMonitor:
    """Count visible cube-to-target arrivals without simulator-private state."""

    ENTER_DISTANCE_PX = 18.0
    EXIT_DISTANCE_PX = 25.0
    INITIAL_TURNAROUND_MARGIN_STEPS = 4

    def __init__(
        self, task_goal: str, initial_frame: np.ndarray, action_horizon: int = 16
    ) -> None:
        normalized = task_goal.lower()
        self.required = self._required_occurrence(normalized)
        self.enabled = bool(
            self.required is not None
            and "press" in normalized
            and "button" in normalized
            and "reaches the target" in normalized
        )
        self.count = 0
        self.inside = False
        self.stage = "prepare"
        self.prepare_chunks = 0
        self.prepare_chunks_required = 3
        self.press_lead_steps = int(action_horizon) + 2
        self.event_steps: list[int] = []
        self.predicted_target_step: int | None = None
        self.period_estimate_source: str | None = None
        self.anticipation_emitted = False
        self.ready_to_press = False
        self.target_point: np.ndarray | None = None
        self.cube_point: np.ndarray | None = None
        self.cube_hue: int | None = None
        self.trace: list[dict[str, Any]] = []
        self.initialization_error: str | None = None
        if self.enabled:
            try:
                self._initialize(np.asarray(initial_frame, dtype=np.uint8))
            except ValueError as error:
                self.initialization_error = str(error)
                self.enabled = False

    @staticmethod
    def _required_occurrence(text: str) -> int | None:
        words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        for word, count in words.items():
            if re.search(rf"\b{word}\s+time\b", text):
                return count
        match = re.search(r"\b([1-9])(?:st|nd|rd|th)?\s+time\b", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _hue_distance(hue: np.ndarray, reference: int) -> np.ndarray:
        values = hue.astype(np.int16)
        return np.minimum((values - reference) % 180, (reference - values) % 180)

    @staticmethod
    def _colored_components(frame: np.ndarray) -> list[dict[str, Any]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        height, width = hsv.shape[:2]
        roi = hsv[max(0, height // 10) : int(height * 0.9), : int(width * 0.88)]
        background_hue = int(np.median(roi[:, :, 0]))
        hue_distance = StopCubeTemporalMonitor._hue_distance(
            hsv[:, :, 0], background_hue
        )
        saturated_foreground = (
            (hue_distance > 20)
            & (hsv[:, :, 1] > 70)
            & (hsv[:, :, 2] > 50)
        )
        low_contrast_foreground = (
            (hue_distance > 2)
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 1] < 145)
            & (hsv[:, :, 2] > 150)
        )
        mask = np.asarray(
            saturated_foreground | low_contrast_foreground,
            dtype=np.uint8,
        ) * 255
        mask[: max(0, height // 10), :] = 0
        mask[int(height * 0.9) :, :] = 0
        mask[:, int(width * 0.88) :] = 0
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        components: list[dict[str, Any]] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if not (15 <= area <= 1200 and component_width <= 50 and component_height <= 50):
                continue
            row = float(centroids[index][1])
            col = float(centroids[index][0])
            components.append(
                {
                    "area": area,
                    "point": np.asarray([row, col], dtype=np.float32),
                    "hue": int(hsv[int(round(row)), int(round(col)), 0]),
                }
            )
        return components

    def _initialize(self, frame: np.ndarray) -> None:
        components = self._colored_components(frame)
        clusters: list[list[dict[str, Any]]] = []
        for component in components:
            cluster = next(
                (
                    values
                    for values in clusters
                    if np.linalg.norm(values[0]["point"] - component["point"]) <= 12.0
                ),
                None,
            )
            if cluster is None:
                clusters.append([component])
            else:
                cluster.append(component)
        target_candidates = [values for values in clusters if len(values) >= 2]
        if not target_candidates:
            raise ValueError("temporal monitor could not identify the bullseye target")
        target = max(
            target_candidates,
            key=lambda values: sum(int(item["area"]) for item in values),
        )
        weights = np.asarray([item["area"] for item in target], dtype=np.float32)
        self.target_point = np.average(
            np.stack([item["point"] for item in target]), axis=0, weights=weights
        )
        cube_candidates = [
            component
            for component in components
            if np.linalg.norm(component["point"] - self.target_point) > 20.0
        ]
        if not cube_candidates:
            raise ValueError("temporal monitor could not identify the moving cube")
        cube = max(cube_candidates, key=lambda item: int(item["area"]))
        self.cube_point = cube["point"].copy()
        self.cube_hue = int(cube["hue"])

    @property
    def active(self) -> bool:
        return self.enabled and self.target_point is not None and self.cube_hue is not None

    def observe(self, frame: np.ndarray, step: int) -> dict[str, Any] | None:
        if not self.active:
            return None
        hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        hue_distance = self._hue_distance(hsv[:, :, 0], int(self.cube_hue))
        mask = np.asarray(
            (hue_distance <= 10)
            & (hsv[:, :, 1] > 70)
            & (hsv[:, :, 2] > 50),
            dtype=np.uint8,
        ) * 255
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates: list[tuple[int, np.ndarray]] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if 20 <= area <= 800:
                candidates.append(
                    (
                        area,
                        np.asarray(
                            [centroids[index][1], centroids[index][0]],
                            dtype=np.float32,
                        ),
                    )
                )
        if not candidates:
            return None
        if self.cube_point is None:
            _, point = max(candidates, key=lambda value: value[0])
        elif self.inside:
            assert self.target_point is not None
            separated = [
                value
                for value in candidates
                if np.linalg.norm(value[1] - self.target_point)
                >= self.EXIT_DISTANCE_PX
            ]
            if separated:
                _, point = min(
                    separated,
                    key=lambda value: np.linalg.norm(value[1] - self.cube_point),
                )
            else:
                _, point = min(
                    candidates,
                    key=lambda value: np.linalg.norm(value[1] - self.cube_point),
                )
        else:
            _, point = min(
                candidates,
                key=lambda value: np.linalg.norm(value[1] - self.cube_point),
            )
        self.cube_point = point.copy()
        assert self.target_point is not None
        distance = float(np.linalg.norm(self.cube_point - self.target_point))
        entered = distance <= self.ENTER_DISTANCE_PX and not self.inside
        was_ready_to_press = self.ready_to_press
        if entered:
            self.count += 1
            self.event_steps.append(int(step))
        if distance >= self.EXIT_DISTANCE_PX:
            self.inside = False
        elif distance <= self.ENTER_DISTANCE_PX:
            self.inside = True
        event = {
            "step": int(step),
            "distance_px": distance,
            "entered_target": entered,
            "observed_occurrences": self.count,
            "required_occurrences": self.required,
            "cube_point": self.cube_point.tolist(),
            "target_point": self.target_point.tolist(),
            "anticipated_target": False,
        }
        if entered:
            self.trace.append(event)
            if self.count >= int(self.required):
                self.ready_to_press = True
            return None if was_ready_to_press else event
        if (
            not self.anticipation_emitted
            and self.stage == "wait"
            and self.count == int(self.required) - 1
            and self.event_steps
        ):
            if len(self.event_steps) >= 2:
                periods = np.diff(np.asarray(self.event_steps, dtype=np.int32))
                period = int(round(float(np.median(periods))))
                self.period_estimate_source = "rgb_interarrival_median"
            else:
                first_arrival_step = int(self.event_steps[0])
                period = 2 * (
                    first_arrival_step + self.INITIAL_TURNAROUND_MARGIN_STEPS
                )
                self.period_estimate_source = "rgb_initial_half_cycle"
            self.predicted_target_step = int(self.event_steps[-1] + period)
            if int(step) >= self.predicted_target_step - self.press_lead_steps:
                self.anticipation_emitted = True
                self.ready_to_press = True
                event["anticipated_target"] = True
                event["predicted_target_step"] = self.predicted_target_step
                event["press_lead_steps"] = self.press_lead_steps
                event["period_estimate_source"] = self.period_estimate_source
                self.trace.append(event)
                return event
        return None

    def prompt_context(self) -> str | None:
        if not self.active:
            return None
        if self.stage == "prepare":
            return (
                "Temporal execution stage: first move to the top of the button at "
                "its current-image point to prepare, without pressing it."
            )
        if self.ready_to_press:
            return (
                f"Temporal event monitor: {self.count}/{self.required} completed target "
                f"arrivals; the next arrival is due by step {self.predicted_target_step}. "
                "Start pressing the button immediately to account for action latency."
            )
        return (
            f"Temporal event monitor: {self.count}/{self.required} target arrivals "
            "have been observed from RGB. The robot is prepared above the button; "
            "remain static until the required count is reached."
        )

    def validate(self, text: str) -> str | None:
        if not self.active:
            return None
        normalized = text.lower()
        action = RepetitionProgressController._action(text)
        move_to_button = "move" in normalized and "button" in normalized
        remain_static = "remain static" in normalized
        if self.stage == "prepare" and not move_to_button:
            return "the robot must first move to the top of the button to prepare"
        if self.stage == "wait" and not self.ready_to_press and not remain_static:
            return "the prepared robot must remain static until the required count"
        if action == "press" and not self.ready_to_press:
            return "the temporal monitor has not reached the required target-arrival count"
        if self.ready_to_press and action != "press":
            return "the temporal monitor reached the required count; press the button now"
        return None

    def record_executed_subgoal(self, text: str) -> bool:
        if not self.active or self.stage != "prepare":
            return False
        normalized = text.lower()
        if "move" in normalized and "button" in normalized:
            self.prepare_chunks += 1
            if self.prepare_chunks >= self.prepare_chunks_required:
                self.stage = "wait"
                return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8011)
    parser.add_argument(
        "--planner-endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument("--planner-model", default="robomme-groundsg-qwen3vl4b")
    parser.add_argument(
        "--planner-profile",
        choices=("carve", "raw"),
        default="carve",
        help="Use the full CARVE Planner/Harness contract or the raw GroundSG baseline.",
    )
    parser.add_argument("--planner-timeout-s", type=float, default=120.0)
    parser.add_argument("--planner-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--execution-feedback",
        choices=("legacy_predictions", "execution_chunks"),
        default="legacy_predictions",
        help="Keep historical prediction counters or update procedures from executed chunks.",
    )
    parser.add_argument(
        "--procedure-authority",
        choices=("legacy_override", "observe_only"),
        default="legacy_override",
        help="Preserve legacy overrides or use gripper evidence only to request visual review.",
    )
    parser.add_argument(
        "--semantic-transition-confirmations",
        type=int,
        default=1,
        help="Consecutive Planner decisions required before changing primitive type.",
    )
    parser.add_argument(
        "--planner-context", choices=("augmented", "native"), default="augmented",
        help="Keep execution receipts outside the learned Planner prompt in native mode.",
    )
    parser.add_argument(
        "--grounding-authority", choices=("override", "observe_only"), default="override",
        help="Record visual tool proposals without rewriting the learned policy's target point.",
    )
    parser.add_argument(
        "--planner-demo-mode", choices=("always", "once_with_memory"), default="always",
        help="Omit repeated demonstration video only after binding structured identity memory.",
    )
    parser.add_argument("--demo-history-mode", choices=("legacy_full", "official"), default="legacy_full")
    parser.add_argument("--planner-image-format", choices=("jpeg", "png"), default="jpeg")
    parser.add_argument("--memory-on-uncertain", choices=("error", "without_memory"), default="error")
    parser.add_argument("--task", default="MoveCube")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--planner-schedule",
        choices=[mode.value for mode in GroundedSubgoalScheduleMode],
        default=GroundedSubgoalScheduleMode.EVERY_CHUNK.value,
    )
    parser.add_argument("--planner-min-reuse-chunks", type=int, default=1)
    parser.add_argument("--planner-max-reuse-chunks", type=int, default=2)
    parser.add_argument("--planner-visual-change-threshold", type=float, default=0.09)
    parser.add_argument("--planner-gripper-change-threshold", type=float, default=0.15)
    parser.add_argument(
        "--planner-max-reusable-points",
        type=int,
        default=1,
        help="Refresh every chunk when a grounded subgoal has more points.",
    )
    parser.add_argument(
        "--verified-point-reuse",
        action="store_true",
        help="Reuse a VideoUnmask-family pick only after current-RGB point revalidation at a chunk boundary.",
    )
    parser.add_argument(
        "--verified-identity-conflict",
        action="store_true",
        help="At a chunk boundary, resolve a pick point that matches a different tracked identity only when both containers are visible in current RGB.",
    )
    parser.add_argument(
        "--stage-receipt-hint",
        action="store_true",
        help="After an executed first-target put, tell the Planner the next ordered target and gripper-process evidence without rewriting its action.",
    )
    parser.add_argument(
        "--task-memory-file",
        type=Path,
        help="Optional deployable structured memory extracted from the initial observation.",
    )
    parser.add_argument(
        "--memory-hint-style", choices=("precise", "coarse_grid"), default="precise",
        help="VLM-visible identity representation; coarse_grid omits exact historical point tokens.",
    )
    parser.add_argument(
        "--memory-use-policy", choices=("always", "motion_gate"), default="always",
        help="Use identity memory only when public-demo tracks show persistent relative relocation.",
    )
    parser.add_argument(
        "--relational-color-grounding",
        action="store_true",
        help="Validate one color/relation target with a local perception tool.",
    )
    parser.add_argument(
        "--button-grounding",
        action="store_true",
        help="Refine a Planner-proposed button point to the local button center.",
    )
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--planner-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--container-project-root",
        type=Path,
        default=Path("/workspace"),
    )
    parser.add_argument(
        "--host-project-root",
        type=Path,
        default=Path("/home/admin1/ct/CARVE-VLA"),
    )
    return parser.parse_args()


def encode_image(frame: np.ndarray, image_format: str = "jpeg") -> str:
    buffer = io.BytesIO()
    if image_format not in {"jpeg", "png"}:
        raise ValueError("unknown Planner image format")
    options = {"extension": ".jpg", "quality": 90} if image_format == "jpeg" else {"extension": ".png"}
    iio.imwrite(buffer, np.asarray(frame, dtype=np.uint8), **options)
    return f"data:image/{image_format};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_planner(client: Any, payload: dict, *, send_demo: bool) -> dict:
    request = urllib.request.Request(client.endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    client.request_count += 1
    entry = {"request_index": client.request_count, "demo_video_sent": send_demo,
             "output_status": "unvalidated", "usage": None, "error": None}
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=client.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Planner response must be a JSON object")
        entry["usage"] = result.get("usage")
        return result
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        entry.update(output_status="request_failed", error=str(error))
        raise RuntimeError(f"grounded Planner request failed: {error}") from error
    finally:
        entry["request_latency_ms"] = (time.perf_counter() - started) * 1000
        client.request_audit.append(entry)


def planner_response_text(client: Any, payload: dict) -> str:
    try:
        value = payload["choices"][0]["message"]["content"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("grounded Planner returned no text")
        return value.strip()
    except (KeyError, IndexError, TypeError, ValueError) as error:
        client.request_audit[-1].update(output_status="rejected", error=str(error))
        raise RuntimeError("grounded Planner returned no text") from error


def parse_planner_output(client: Any, raw: str) -> tuple[str, str, list[list[int]]]:
    try:
        return parse_grounded_subgoal(raw)
    except ValueError as error:
        client.request_audit[-1].update(output_status="rejected", error=str(error))
        raise


def planner_visible_path(path: Path, args: argparse.Namespace) -> Path:
    absolute = path.resolve()
    container_root = args.container_project_root.resolve()
    try:
        relative = absolute.relative_to(container_root)
    except ValueError as exc:
        raise ValueError(f"output must be under {container_root}") from exc
    return args.host_project_root.resolve() / relative


def route_stick_marker_col(frame: np.ndarray) -> float | None:
    """Locate RoboMME's red route marker in one front-camera frame."""

    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = np.asarray(
        ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 172))
        & (hsv[:, :, 1] > 150)
        & (hsv[:, :, 2] > 140),
        dtype=np.uint8,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, float]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        row = float(centroids[index][1])
        if 20 <= area <= 180 and row > 35.0:
            candidates.append((area, float(centroids[index][0])))
    return max(candidates)[1] if candidates else None


def route_stick_target_sequence(frames: list[np.ndarray]) -> list[float]:
    """Compress a demonstrated red-marker trajectory into stable target columns."""

    columns = [route_stick_marker_col(frame) for frame in frames]
    observed = np.asarray([value for value in columns if value is not None])
    if observed.size < 4:
        return []
    peaks: list[tuple[int, float]] = []
    for center in np.arange(16.0, 240.0, 4.0):
        support = int(np.sum(np.abs(observed - center) <= 3.0))
        if support >= 4:
            peaks.append((support, float(center)))
    stable_centers: list[float] = []
    for _, center in sorted(peaks, reverse=True):
        if all(abs(center - selected) > 12.0 for selected in stable_centers):
            stable_centers.append(center)
    if len(stable_centers) < 2:
        return []

    labels: list[float | None] = []
    for value in columns:
        if value is None:
            labels.append(None)
            continue
        nearest = min(stable_centers, key=lambda center: abs(center - value))
        labels.append(nearest if abs(nearest - value) <= 10.0 else None)
    runs: list[list[float | None | int]] = []
    for label in labels:
        if not runs or runs[-1][0] != label:
            runs.append([label, 1])
        else:
            runs[-1][1] = int(runs[-1][1]) + 1

    sequence: list[float] = []
    nonempty = [index for index, run in enumerate(runs) if run[0] is not None]
    final_nonempty = nonempty[-1] if nonempty else -1
    for index, (label, length) in enumerate(runs):
        if label is None or (int(length) < 3 and index != final_nonempty):
            continue
        value = float(label)
        if not sequence or value != sequence[-1]:
            sequence.append(value)
    return sequence


def hidden_container_grounding(
    frame: np.ndarray,
    color: str,
    task_memory: dict[str, Any] | None,
    *,
    refine_current_rgb: bool = True,
) -> dict[str, Any] | None:
    """Refine a remembered swapped-container identity in the current RGB frame."""

    points = (task_memory or {}).get("hidden_container_points")
    if not isinstance(points, dict) or color not in points:
        return None
    remembered = np.asarray(points[color], dtype=np.float32)
    candidates: list[np.ndarray] = []
    if refine_current_rgb:
        hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        mask = np.asarray(
            (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 130), dtype=np.uint8
        )
        mask[:35] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            _, _, width, height, area = (int(value) for value in stats[index])
            if 100 <= area <= 700 and 10 <= width <= 40 and 10 <= height <= 40:
                candidates.append(
                    np.asarray(
                        [centroids[index][1], centroids[index][0]], dtype=np.float32
                    )
                )
    point = (
        min(candidates, key=lambda value: np.linalg.norm(value - remembered))
        if candidates
        else remembered
    )
    distance = float(np.linalg.norm(point - remembered))
    if distance > 35.0:
        return None
    return {
        "tool": "hidden_container_memory_grounding",
        "source": (
            "demonstration_rgb_sam2_and_current_rgb"
            if refine_current_rgb
            else "demonstration_rgb_sam2_instance_lock"
        ),
        "color": color,
        "remembered_point": [float(value) for value in remembered],
        "point": [int(round(float(point[0]))), int(round(float(point[1])))],
        "refinement_distance": distance,
        "current_rgb_match": bool(candidates),
    }


def verified_identity_conflict(
    frame: np.ndarray,
    color: str,
    model_point: list[int],
    task_memory: dict[str, Any] | None,
    *,
    expected_color: str | None,
    grasp_observed: bool,
) -> dict[str, Any]:
    """Resolve only an identity contradiction with two current-image matches."""

    receipt: dict[str, Any] = {"resolved": False, "reason": "not_admitted"}
    points = (task_memory or {}).get("hidden_container_points")
    if (
        not isinstance(points, dict)
        or color != expected_color
        or grasp_observed
        or len(model_point) != 2
    ):
        return receipt
    target = hidden_container_grounding(frame, color, task_memory)
    if target is None or not target["current_rgb_match"] or target["refinement_distance"] > 6.0:
        return {"resolved": False, "reason": "target_not_current_rgb_verified"}
    target_point = np.asarray(target["point"], dtype=float)
    proposed = np.asarray(model_point, dtype=float)
    if np.linalg.norm(proposed - target_point) <= 12.0:
        return {"resolved": False, "reason": "model_matches_target"}
    for other_color in points:
        if other_color == color:
            continue
        other = hidden_container_grounding(frame, other_color, task_memory)
        if other is None or not other["current_rgb_match"] or other["refinement_distance"] > 6.0:
            continue
        other_point = np.asarray(other["point"], dtype=float)
        if np.linalg.norm(target_point - other_point) < 24.0:
            continue
        if np.linalg.norm(proposed - other_point) <= 12.0:
            return {
                "resolved": True,
                "reason": "model_point_matches_other_tracked_identity",
                "color": color,
                "other_color": other_color,
                "model_point": list(model_point),
                "point": list(target["point"]),
                "target_current_rgb_distance_px": target["refinement_distance"],
                "other_current_rgb_distance_px": other["refinement_distance"],
            }
    return {"resolved": False, "reason": "no_verified_identity_conflict"}


def build_button_unmask_memory(
    frame: np.ndarray, task_goal: str
) -> dict[str, Any] | None:
    """Store visible cube identities before ButtonUnmask covers them."""

    colors = list(dict.fromkeys(re.findall(r"\b(red|green|blue)\b", task_goal.lower())))
    if not colors:
        return None
    points: dict[str, list[float]] = {}
    for color in colors:
        grounding = relational_color_grounding(frame, f"pick up the {color} cube")
        if grounding is None:
            return None
        points[color] = [float(value) for value in grounding["point"]]
    hint_entries = "; ".join(
        f"{color} at <{point[0]:.1f}, {point[1]:.1f}>"
        for color, point in points.items()
    )
    return {
        "planner_hint": (
            f"Event-persistent identity memory before the button event: {hint_entries}. "
            "After a cube is covered, pick the container at its remembered physical "
            "location and preserve the requested color order."
        ),
        "hidden_container_points": points,
        "required_color_order": colors,
        "event_memory": {
            "source": "initial_current_rgb",
            "event": "button_press_then_occlusion",
            "targets": [
                {"color": color, "point_row_col": points[color]}
                for color in colors
            ],
            "evaluator_or_oracle_fields_used": False,
        },
    }


class RouteStickTrajectoryController:
    """Translate a demonstrated image trajectory into robot-frame route sides."""

    _SIDE = re.compile(r"nearest\s+(left|right)\s+target", re.IGNORECASE)

    def __init__(self, target_cols: list[float] | None = None) -> None:
        self.target_cols = list(target_cols or [])
        self.stage = 0

    @property
    def active(self) -> bool:
        return len(self.target_cols) >= 2

    def apply(self, text: str, frame: np.ndarray) -> tuple[str, dict[str, Any] | None]:
        if not self.active or self._SIDE.search(text) is None:
            return text, None
        current_col = route_stick_marker_col(frame)
        if current_col is not None:
            while (
                self.stage + 1 < len(self.target_cols)
                and abs(current_col - self.target_cols[self.stage + 1]) <= 9.0
            ):
                self.stage += 1
        target_index = min(self.stage + 1, len(self.target_cols) - 1)
        delta_image = self.target_cols[target_index] - self.target_cols[self.stage]
        robot_side = "right" if delta_image < 0 else "left"
        rewritten = self._SIDE.sub(f"nearest {robot_side} target", text, count=1)
        return rewritten, {
            "tool": "route_trajectory",
            "source": "demonstration_rgb",
            "stage": self.stage,
            "current_marker_col": current_col,
            "current_target_col": self.target_cols[self.stage],
            "next_target_col": self.target_cols[target_index],
            "robot_side": robot_side,
            "rewritten": rewritten != text,
        }


def parse_grounded_subgoal(raw: str) -> tuple[str, str, list[list[int]]]:
    """Convert optional Qwen point tokens into RoboMME's subgoal contract."""

    matches = BOX_PATTERN.findall(raw)
    remainder = BOX_PATTERN.sub("", raw)
    if "<|box_start|>" in remainder or "<|box_end|>" in remainder:
        raise ValueError("grounded Planner output contains a malformed point token")
    points = [
        [int(int(x) * 256 / 1000), int(int(y) * 256 / 1000)]
        for x, y in matches
    ]
    if any(value not in GROUNDING_RANGE for point in points for value in point):
        raise ValueError("grounded Planner output contains an out-of-range point")
    grounded = BOX_PATTERN.sub(
        lambda match: f"<{int(int(match.group(1)) * 256 / 1000)}, "
        f"{int(int(match.group(2)) * 256 / 1000)}>",
        raw,
    ).strip()
    history_text = BOX_PATTERN.sub("<bbox>", raw).strip()
    if not grounded:
        raise ValueError("grounded Planner output is empty")
    if len(grounded) > 320:
        raise ValueError("grounded Planner output is unreasonably long")
    return grounded, history_text, points


def requires_visual_grounding(text: str) -> bool:
    """Return whether the subgoal normally requires a current-image point."""

    return VISUAL_GROUNDING_VERBS.search(text) is not None


def unmask_swap_skill_rejection(grounded: str, points: list[list[int]]) -> str | None:
    """Validate the official skill vocabulary, not target identity or task progress."""

    text = grounded.strip().removesuffix(".").lower()
    if text == "put down the container" and not points:
        return None
    match = UNMASK_SWAP_PICK_PATTERN.fullmatch(text)
    if match and points == [[int(match[1]), int(match[2])]] and all(
        value in GROUNDING_RANGE for value in points[0]
    ):
        return None
    return (
        "unsupported VideoUnmask action skill; answer with exactly one executable "
        "subgoal: `put down the container`, or a `pick up the container` "
        "instruction with one current-image GroundSG point and the correct "
        "hidden cube color. Do not repeat the question, combine skills, or "
        "claim task completion"
    )


def cacheable_unmask_pick(result: dict[str, Any], memory: dict[str, Any] | None) -> bool:
    match = UNMASK_SWAP_PICK_PATTERN.fullmatch(str(result.get("grounded_subgoal", "")).lower())
    points = result.get("points")
    return bool(
        match
        and points == [[int(match[1]), int(match[2])]]
        and isinstance((memory or {}).get("hidden_container_points"), dict)
        and match[3] in memory["hidden_container_points"]
    )


def revalidate_unmask_pick(
    frame: np.ndarray,
    result: dict[str, Any],
    memory: dict[str, Any] | None,
    last_feedback: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fail closed before reusing a grounded pick; this never authorizes stage completion."""

    if not cacheable_unmask_pick(result, memory):
        return None, {"reason": "unsupported_grounded_pick"}
    if last_feedback is None or last_feedback.get("grasp_observed") is not False:
        return None, {"reason": "grasp_state_not_clear"}
    match = UNMASK_SWAP_PICK_PATTERN.fullmatch(str(result["grounded_subgoal"]).lower())
    assert match is not None
    observed = hidden_container_grounding(frame, match[3], memory)
    if observed is None or not observed["current_rgb_match"]:
        return None, {"reason": "current_rgb_target_missing"}
    point = observed["point"]
    previous = result["points"][0]
    shift = float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(previous, dtype=float)))
    if shift > 4.0:
        return None, {"reason": "grounded_point_shifted", "shift_px": shift}
    updated = dict(result)
    updated["points"] = [point]
    updated["grounded_subgoal"] = (
        f"pick up the container at <{point[0]}, {point[1]}> that hides the {match[3]} cube"
    )
    return updated, {"reason": "current_rgb_revalidated", "point": point, "shift_px": shift}


def repick_skill_rejection(grounded: str, points: list[list[int]]) -> str | None:
    """Accept learned skills without enforcing an ordinal or completion transition."""
    text = grounded.strip().removesuffix(".").lower()
    if text == "put it down" and not points:
        return None
    match = re.fullmatch(
        r"(?:pick up the correct cube at <(\d+),\s*(\d+)> for the "
        r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time"
        r"|press the button at <(\d+),\s*(\d+)> to finish)", text,
    )
    if match:
        coords = [int(value) for value in match.groups() if value is not None]
        if points == [coords] and all(value in GROUNDING_RANGE for value in coords):
            return None
    return (
        "unsupported VideoRepick action skill; for a pick, write 'pick up the correct "
        "cube at' followed by two numeric current-image coordinates inside angle brackets, "
        "then 'for the first time' (or the applicable repetition number). "
        "For release, write 'put it down'. For the final press, write 'press the button at' "
        "followed by two numeric current-image coordinates and 'to finish'. "
        "Never write placeholder words as coordinates or infer completion from gripper closure"
    )


def native_skill_rejection(task: str, grounded: str, points: list[list[int]]) -> str | None:
    if task in {"VideoUnmask", "VideoUnmaskSwap"}:
        return unmask_swap_skill_rejection(grounded, points)
    if task == "VideoRepick":
        return repick_skill_rejection(grounded, points)
    return None


def relational_color_grounding(frame: np.ndarray, text: str) -> dict[str, Any] | None:
    """Ground a color object selected by a simple spatial relation."""

    normalized = text.lower()
    identity_match = re.search(
        r"persistent target identity is\s+(.+?)(?:\.|persistent evidence:|task sequence:|$)",
        normalized,
    )
    identity_text = identity_match.group(1) if identity_match else normalized
    color_ranges = {
        "green": ((35, 80, 40), (90, 255, 255)),
        "blue": ((90, 80, 40), (135, 255, 255)),
        "red": ((0, 200, 120), (6, 255, 255)),
    }
    color = next((name for name in color_ranges if name in identity_text), None)
    relations = (
        "bottom-left",
        "bottom-right",
        "top-left",
        "top-right",
        "topmost",
        "bottommost",
        "leftmost",
        "rightmost",
    )
    relation = next((name for name in relations if name in identity_text), None)
    if color is None:
        return None

    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [
        {
            "point": [int(round(centroids[index][1])), int(round(centroids[index][0]))],
            "area": int(stats[index, cv2.CC_STAT_AREA]),
        }
        for index in range(1, count)
        if 20 <= int(stats[index, cv2.CC_STAT_AREA]) <= 2000
        and 4 <= int(stats[index, cv2.CC_STAT_WIDTH]) <= 40
        and 4 <= int(stats[index, cv2.CC_STAT_HEIGHT]) <= 40
    ]
    if not candidates:
        return None
    if relation is None:
        if len(candidates) != 1:
            return None
        selected = candidates[0]
        relation = "unique-visible"
    elif relation == "top-left":
        selected = min(candidates, key=lambda item: sum(item["point"]))
    elif relation == "top-right":
        selected = max(candidates, key=lambda item: item["point"][1] - item["point"][0])
    elif relation == "bottom-left":
        selected = max(candidates, key=lambda item: item["point"][0] - item["point"][1])
    elif relation == "bottom-right":
        selected = max(candidates, key=lambda item: sum(item["point"]))
    elif relation == "topmost":
        selected = min(candidates, key=lambda item: item["point"][0])
    elif relation == "bottommost":
        selected = max(candidates, key=lambda item: item["point"][0])
    elif relation == "leftmost":
        selected = min(candidates, key=lambda item: item["point"][1])
    else:
        selected = max(candidates, key=lambda item: item["point"][1])
    return {
        "tool": (
            "unique_color_grounding"
            if relation == "unique-visible"
            else "relational_color_grounding"
        ),
        "color": color,
        "relation": relation,
        "point": selected["point"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def button_center_grounding(
    frame: np.ndarray,
    text: str,
    model_points: list[list[int]],
) -> dict[str, Any] | None:
    """Refine a coarse semantic button point with a compact local detector."""

    normalized = text.lower()
    if not (
        "button" in normalized and ("press" in normalized or "move" in normalized)
    ):
        return None
    hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    masks = (
        ("dark_inset", cv2.inRange(hsv, (0, 0, 50), (179, 80, 180))),
        ("gray_base", cv2.inRange(hsv, (0, 0, 100), (179, 70, 245))),
    )
    coarse = (
        np.asarray(model_points[0], dtype=np.float32) if model_points else None
    )
    candidates: list[dict[str, Any]] = []
    for detector, mask in masks:
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if not (5 <= width <= 30 and 5 <= height <= 30 and 20 <= area <= 500):
                continue
            point = [
                int(round(centroids[index][1])),
                int(round(centroids[index][0])),
            ]
            if point[0] < 10:
                continue
            distance = (
                float(np.linalg.norm(np.asarray(point, dtype=np.float32) - coarse))
                if coarse is not None
                else None
            )
            if distance is None or distance <= 24.0:
                candidates.append(
                    {
                        "point": point,
                        "distance_to_model_point": distance,
                        "area": area,
                        "detector": detector,
                    }
                )
        if candidates and coarse is not None:
            break
    if not candidates:
        return None
    if coarse is not None:
        selected = min(
            candidates,
            key=lambda item: float(item["distance_to_model_point"]),
        )
    else:
        gray_candidates = [
            item for item in candidates if item["detector"] == "gray_base"
        ]
        selected = max(gray_candidates or candidates, key=lambda item: int(item["area"]))
    return {
        "tool": "button_center_grounding",
        "point": selected["point"],
        "model_point": list(model_points[0]) if model_points else None,
        "distance_to_model_point": selected["distance_to_model_point"],
        "detector": selected["detector"],
        "source": "coarse_point_refinement" if model_points else "semantic_fallback",
        "candidate_count": len(candidates),
    }


class RepetitionProgressController:
    """Maintain an auditable repetition index from semantic stage changes."""

    def __init__(self, memory_hint: str) -> None:
        normalized = memory_hint.lower()
        self.target = next(
            (
                count
                for word, count in REPETITION_WORDS.items()
                if re.search(rf"\b{word}\s+times?\b", normalized)
            ),
            None,
        )
        self.index = 1
        self.last_action: str | None = None

    @staticmethod
    def _action(text: str) -> str | None:
        normalized = text.lower()
        if re.search(r"\b(pick(?: up)?|grasp)\b", normalized):
            return "pick"
        if re.search(
            r"\b(?:put(?:\s+(?:it|the\s+\w+))?(?:\s+down)?|place)\b",
            normalized,
        ):
            return "put"
        if re.search(r"\bpress\b", normalized):
            return "press"
        return None

    def apply(self, grounded: str, history: str) -> tuple[str, str, dict[str, Any] | None]:
        action = self._action(grounded)
        if self.target is None or action is None:
            self.last_action = action or self.last_action
            return grounded, history, None
        previous = self.last_action
        if action == "pick" and previous == "put":
            self.index += 1
        self.last_action = action
        if action != "pick" or self.index > self.target:
            return grounded, history, {
                "target_repetitions": self.target,
                "current_repetition": self.index,
                "previous_action": previous,
                "action": action,
                "rewritten": False,
            }
        ordinal = ORDINAL_WORDS.get(self.index, f"{self.index}th")
        pattern = re.compile(r"\b(first|second|third|fourth|\d+th)\s+time\b", re.IGNORECASE)
        rewritten_grounded, count = pattern.subn(f"{ordinal} time", grounded)
        rewritten_history = pattern.sub(f"{ordinal} time", history)
        if count == 0:
            rewritten_grounded = f"{grounded} for the {ordinal} time"
            rewritten_history = f"{history} for the {ordinal} time"
        return rewritten_grounded, rewritten_history, {
            "target_repetitions": self.target,
            "current_repetition": self.index,
            "previous_action": previous,
            "action": action,
            "rewritten": rewritten_grounded != grounded,
        }


class RepeatedProcedureController:
    """Gate repeated pick/put cycles with deployable gripper events."""

    def __init__(
        self,
        task_text: str,
        *,
        final_pick_terminal: bool = False,
        minimum_pick_predictions: int = 1,
        grasp_confirmation_predictions: int = 0,
        pick_prediction_schedule: tuple[int, ...] | None = None,
    ) -> None:
        normalized = task_text.lower()
        self.target = next(
            (
                count
                for word, count in REPETITION_WORDS.items()
                if re.search(rf"\b{word}\s+times?\b", normalized)
            ),
            None,
        )
        self.active = bool(
            self.target
            and self.target > 1
            and re.search(r"\b(?:pick(?: up)?|grasp)\b", normalized)
            and re.search(r"\b(?:put|place)\b", normalized)
        )
        self.final_pick_terminal = final_pick_terminal
        self.pick_prediction_schedule = tuple(pick_prediction_schedule or ())
        self.minimum_pick_predictions = max(
            1,
            (
                self.pick_prediction_schedule[0]
                if self.pick_prediction_schedule
                else minimum_pick_predictions
            ),
        )
        self.pick_predictions = 0
        self.grasp_confirmation_predictions = max(0, grasp_confirmation_predictions)
        self.grasp_confirmations = 0
        self.cycle = 1
        self.stage = "pick"
        self.open_reference: float | None = None
        self.current_gripper: float | None = None
        self.grasp_observed = False
        self.release_observed = False

    def observe_execution(self, gripper: float) -> None:
        if not self.active:
            return
        value = float(gripper)
        self.current_gripper = value
        if self.open_reference is None:
            self.open_reference = value
        else:
            self.open_reference = max(self.open_reference, value)
        close_delta = max(0.004, abs(self.open_reference) * 0.2)
        reopen_tolerance = max(0.0015, abs(self.open_reference) * 0.05)
        if self.stage == "pick" and value <= self.open_reference - close_delta:
            self.grasp_observed = True
        if (
            self.stage == "put"
            and self.grasp_observed
            and value >= self.open_reference - reopen_tolerance
        ):
            self.release_observed = True

    def prompt_context(self) -> str | None:
        if not self.active:
            return None
        completion = self.grasp_observed if self.stage == "pick" else self.release_observed
        return (
            f"Repeated procedure constraint: cycle {self.cycle}/{self.target}, "
            f"current stage {self.stage}, execution monitor completion {completion}. "
            f"Pick evidence {self.pick_predictions}/{self.minimum_pick_predictions}. "
            f"Grasp confirmations {self.grasp_confirmations}/"
            f"{self.grasp_confirmation_predictions}. "
            "Do not advance until the current physical primitive is complete."
        )

    def validate(self, text: str) -> str | None:
        if not self.active:
            return None
        action = RepetitionProgressController._action(text)
        if action is None:
            return None
        if self.stage == "pick":
            final_pick = (
                self.final_pick_terminal
                and self.cycle >= int(self.target or 0)
            )
            pick_ready = (
                self.minimum_pick_predictions == 1
                or self.pick_predictions >= self.minimum_pick_predictions
            ) and self.grasp_confirmations >= self.grasp_confirmation_predictions
            if action == "pick" and self.grasp_observed and pick_ready and not final_pick:
                return "the current repeated pick is complete; advance to put"
            if action == "put" and final_pick:
                return "the final target must remain grasped until task success"
            if action == "put" and not self.grasp_observed:
                return "the current repeated pick has no confirmed gripper closure"
            if action == "put" and not pick_ready:
                return "the current pick has not met its minimum execution evidence"
            if action == "press":
                return "the current repeated pick/put cycle is not complete"
            return None
        if self.stage == "put":
            if action == "put" and self.release_observed:
                return "the current repeated put is complete; advance to the next stage"
            if action == "pick":
                if not self.release_observed:
                    return "the current repeated put has no confirmed gripper reopening"
                if self.cycle >= int(self.target):
                    return "all repeated cycles are complete; continue with press"
            if action == "press":
                if self.cycle < int(self.target):
                    return "the requested number of pick/put cycles is not complete"
                if not self.release_observed:
                    return "the final put has no confirmed gripper reopening"
            return None
        if self.stage == "press" and action in {"pick", "put"}:
            return "the repeated procedure has reached press and cannot regress"
        return None

    def stage_fallback(self) -> str | None:
        if not self.active:
            return None
        if self.stage == "pick":
            if (
                self.grasp_observed
                and self.final_pick_terminal
                and self.cycle >= int(self.target or 0)
            ):
                return "continue_pick"
            return (
                "advance_put"
                if self.grasp_observed
                and (
                    self.minimum_pick_predictions == 1
                    or self.pick_predictions >= self.minimum_pick_predictions
                )
                and self.grasp_confirmations >= self.grasp_confirmation_predictions
                else "continue_pick"
            )
        if self.stage == "put":
            if not self.release_observed:
                return "continue_put"
            if self.cycle < int(self.target):
                return "advance_pick"
            return "advance_press"
        return None

    def commit(self, text: str) -> dict[str, Any] | None:
        if not self.active:
            return None
        action = RepetitionProgressController._action(text)
        previous_cycle = self.cycle
        previous_stage = self.stage
        if self.stage == "pick" and action == "pick":
            self.pick_predictions += 1
            if self.grasp_observed:
                self.grasp_confirmations += 1
        if self.stage == "pick" and action == "put" and self.grasp_observed:
            self.stage = "put"
        elif self.stage == "put" and action == "pick" and self.release_observed:
            if self.cycle < int(self.target):
                self.cycle += 1
                self.stage = "pick"
                self.grasp_observed = False
                self.release_observed = False
                self.pick_predictions = 1
                self.grasp_confirmations = 0
                if self.pick_prediction_schedule:
                    schedule_index = min(
                        self.cycle - 1, len(self.pick_prediction_schedule) - 1
                    )
                    self.minimum_pick_predictions = max(
                        1, self.pick_prediction_schedule[schedule_index]
                    )
        elif self.stage == "put" and action == "press" and self.release_observed:
            if self.cycle >= int(self.target):
                self.stage = "press"
        return {
            "target_cycles": self.target,
            "previous_cycle": previous_cycle,
            "current_cycle": self.cycle,
            "previous_stage": previous_stage,
            "current_stage": self.stage,
            "observed_action": action,
            "open_reference": self.open_reference,
            "current_gripper": self.current_gripper,
            "grasp_observed": self.grasp_observed,
            "release_observed": self.release_observed,
            "pick_predictions": self.pick_predictions,
            "grasp_confirmations": self.grasp_confirmations,
        }


class OrderedProcedureController:
    """Enforce monotonic semantic progress for explicit one-cycle procedures."""

    def __init__(self, task_goal: str) -> None:
        normalized = task_goal.lower()
        has_numeric_repetition = any(
            re.search(rf"\b{word}\s+times?\b", normalized)
            for word in REPETITION_WORDS
        )
        if has_numeric_repetition:
            self.actions: list[str] = []
        else:
            occurrences: list[tuple[int, str]] = []
            patterns = {
                "pick": r"\b(?:pick(?: up)?|grasp)\b",
                "put": r"\b(?:put down|put it down|place)\b",
                "press": r"\bpress\b.{0,20}\bbutton\b",
            }
            for action, pattern in patterns.items():
                match = re.search(pattern, normalized)
                if match is not None:
                    occurrences.append((match.start(), action))
            self.actions = [action for _, action in sorted(occurrences)]
        self.stage = 0
        self.open_reference: float | None = None
        self.current_gripper: float | None = None
        self.grasp_observed = False
        self.release_observed = False
        self.stage_predictions = 0
        # A press primitive must include contact and withdrawal before the next
        # spatial skill starts; otherwise the arm can drag newly revealed objects.
        self.press_prediction_budget = 6

    def observe_execution(self, gripper: float) -> None:
        """Update primitive completion from deployable gripper proprioception."""

        value = float(gripper)
        self.current_gripper = value
        if self.open_reference is None:
            self.open_reference = value
        else:
            self.open_reference = max(self.open_reference, value)
        close_delta = max(0.004, abs(self.open_reference) * 0.2)
        reopen_tolerance = max(0.0015, abs(self.open_reference) * 0.05)
        if value <= self.open_reference - close_delta:
            self.grasp_observed = True
        if self.grasp_observed and value >= self.open_reference - reopen_tolerance:
            self.release_observed = True

    def execution_status(self) -> dict[str, Any]:
        return {
            "open_reference": self.open_reference,
            "current_gripper": self.current_gripper,
            "grasp_observed": self.grasp_observed,
            "release_observed": self.release_observed,
            "stage_predictions": self.stage_predictions,
            "press_prediction_budget": self.press_prediction_budget,
        }

    def stage_complete(self) -> tuple[bool, str]:
        action = self.actions[self.stage]
        if action == "pick":
            return self.grasp_observed, "gripper closure"
        if action == "put":
            return self.release_observed, "gripper reopening"
        if action == "press":
            return (
                self.stage_predictions >= self.press_prediction_budget,
                f"{self.press_prediction_budget} executed press chunks",
            )
        return self.stage_predictions >= 1, "one executed action chunk"

    @property
    def active(self) -> bool:
        return len(self.actions) >= 2

    def prompt_context(self) -> str | None:
        if not self.active:
            return None
        completion, evidence = self.stage_complete()
        return (
            "Ordered procedure constraint: "
            + " -> ".join(self.actions)
            + f". Current admitted stage: {self.actions[self.stage]}. "
            f"Execution monitor completion for this stage: {completion} "
            f"(evidence: {evidence}; admitted predictions: {self.stage_predictions}). "
            "Do not regress to an earlier stage or advance before monitor completion."
        )

    def validate(self, text: str) -> str | None:
        if not self.active:
            return None
        action = RepetitionProgressController._action(text)
        if action not in self.actions:
            return None
        index = self.actions.index(action)
        if index < self.stage:
            expected = self.actions[min(self.stage + 1, len(self.actions) - 1)]
            return (
                f"the ordered procedure has already reached {self.actions[self.stage]} "
                f"and cannot regress to {action}; continue with {expected}"
            )
        if index == self.stage + 1:
            completion, event = self.stage_complete()
            if not completion:
                return (
                    f"the execution monitor has not confirmed {event} for "
                    f"{self.actions[self.stage]}; continue the current stage"
                )
        if index > self.stage + 1:
            return (
                f"the ordered procedure cannot skip from {self.actions[self.stage]} "
                f"directly to {action}"
            )
        return None

    def stage_fallback(self) -> str | None:
        if not self.active:
            return None
        complete, _ = self.stage_complete()
        return None if complete else "continue_ordered_stage"

    def commit(self, text: str) -> dict[str, Any] | None:
        if not self.active:
            return None
        action = RepetitionProgressController._action(text)
        previous = self.actions[self.stage]
        if action in self.actions:
            index = self.actions.index(action)
            if index == self.stage + 1:
                self.stage = index
                self.stage_predictions = 1
            elif index == self.stage:
                self.stage_predictions += 1
        return {
            "ordered_actions": list(self.actions),
            "previous_stage": previous,
            "current_stage": self.actions[self.stage],
            "observed_action": action,
            "execution_monitor": self.execution_status(),
        }


class SemanticTransitionGate:
    """Debounce semantic primitive transitions without using evaluator state."""

    def __init__(self, confirmations: int = 1) -> None:
        if confirmations <= 0:
            raise ValueError("semantic transition confirmations must be positive")
        self.confirmations = confirmations
        self.current: dict[str, Any] | None = None
        self.pending_action: str | None = None
        self.pending_count = 0

    def apply(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_action = RepetitionProgressController._action(
            str(candidate["grounded_subgoal"])
        )
        current_action = (
            RepetitionProgressController._action(str(self.current["grounded_subgoal"]))
            if self.current is not None
            else None
        )
        if (
            self.current is None
            or self.confirmations == 1
            or candidate_action is None
            or current_action is None
            or candidate_action == current_action
        ):
            self.current = dict(candidate)
            self.pending_action = None
            self.pending_count = 0
            return dict(candidate), {
                "candidate_action": candidate_action,
                "accepted_action": candidate_action,
                "pending_count": 0,
                "required_confirmations": self.confirmations,
                "transition_accepted": True,
            }

        if candidate_action == self.pending_action:
            self.pending_count += 1
        else:
            self.pending_action = candidate_action
            self.pending_count = 1
        if self.pending_count >= self.confirmations:
            self.current = dict(candidate)
            self.pending_action = None
            self.pending_count = 0
            return dict(candidate), {
                "candidate_action": candidate_action,
                "accepted_action": candidate_action,
                "pending_count": self.confirmations,
                "required_confirmations": self.confirmations,
                "transition_accepted": True,
            }
        assert self.current is not None
        return dict(self.current), {
            "candidate_action": candidate_action,
            "accepted_action": current_action,
            "pending_count": self.pending_count,
            "required_confirmations": self.confirmations,
            "transition_accepted": False,
        }


class RawGroundedPlannerClient:
    """Official-style GroundSG Planner without CARVE repair, tools or control."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_s: float,
        demo_video: Path | None,
        task_goal: str,
        task_name: str | None = None,
        image_format: str = "jpeg",
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.demo_video = demo_video
        self.task_goal = task_goal
        self.task_name = task_name
        self.image_format = image_format
        self.history_text: list[str] = []
        self.history_points: list[list[int]] = []
        self.request_count = 0
        self.request_audit: list[dict[str, Any]] = []
        self.last_response_text: str | None = None

    def infer(
        self, frame: np.ndarray, robot_state: np.ndarray | None = None
    ) -> dict[str, Any]:
        del robot_state
        history = "; ".join(
            f"{index + 1}. {value}" for index, value in enumerate(self.history_text)
        )
        if history:
            prompt = (
                f"The task goal is: {self.task_goal}\n"
                "The history of previous predicted grounded language subgoals are: "
                f"{history}\n"
            )
        else:
            prompt = (
                f"The task goal is: {self.task_goal}\n"
                "This is the initial turn for prediction\n"
            )
        content: list[dict[str, Any]] = []
        if self.demo_video is not None:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": str(self.demo_video)},
                }
            )
        content.extend(
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image(frame, self.image_format)}},
                {
                    "type": "text",
                    "text": (
                        "What's the next grounded language subgoal based on current "
                        "observation?"
                    ),
                },
            ]
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": GROUND_SG_SYSTEM_PROMPT,
                },
                {"role": "user", "content": content},
            ],
            "objects": {"ref": [], "bbox": self.history_points},
            "temperature": 0,
            "max_tokens": 128,
        }
        started = time.perf_counter()
        response_payload = request_planner(self, payload, send_demo=self.demo_video is not None)
        raw = planner_response_text(self, response_payload)
        self.last_response_text = raw
        grounded, history_text, points = parse_planner_output(self, raw)
        reason = native_skill_rejection(self.task_name, grounded, points)
        if reason:
            self.request_audit[-1].update(output_status="rejected", error=reason)
            raise ValueError(reason)
        self.request_audit[-1]["output_status"] = "accepted"
        if not self.history_text or self.history_text[-1] != history_text:
            self.history_text.append(history_text)
            self.history_points.extend(points)
        return {
            "raw": raw,
            "grounded_subgoal": grounded,
            "history_text": history_text,
            "points": points,
            "model_points": [list(point) for point in points],
            "tool_grounding": None,
            "progress_update": None,
            "transition_gate": None,
            "procedure_update": None,
            "repeated_procedure_update": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "repair_attempts": 0,
            "monitor_fallback": None,
            "route_update": None,
        }


class GroundedPlannerClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_s: float,
        demo_video: Path | None,
        task_name: str,
        task_goal: str,
        route_target_cols: list[float] | None = None,
        task_memory: dict[str, Any] | None = None,
        repair_attempts: int = 1,
        relational_color_grounding_enabled: bool = False,
        button_grounding_enabled: bool = False,
        semantic_transition_confirmations: int = 1,
        temporal_monitor: StopCubeTemporalMonitor | None = None,
        execution_feedback: str = "legacy_predictions",
        procedure_authority: str = "legacy_override",
        planner_context: str = "augmented",
        grounding_authority: str = "override",
        verified_identity_conflict_enabled: bool = False,
        stage_receipt_hint_enabled: bool = False,
        planner_demo_mode: str = "always",
        image_format: str = "jpeg",
    ) -> None:
        if execution_feedback not in {"legacy_predictions", "execution_chunks"}:
            raise ValueError("invalid execution feedback mode")
        if procedure_authority not in {"legacy_override", "observe_only"}:
            raise ValueError("invalid procedure authority")
        if procedure_authority == "observe_only" and (
            execution_feedback != "execution_chunks" or task_name not in {"PickXtimes", "VideoUnmask", "VideoUnmaskSwap", "VideoRepick"}
        ):
            raise ValueError("observe_only admitted only for supported tasks with execution_chunks")
        if task_name in {"VideoUnmask", "VideoUnmaskSwap", "VideoRepick"} and procedure_authority == "observe_only" and (
            planner_context != "native" or grounding_authority != "observe_only"
        ):
            raise ValueError("identity task observation authority requires native context and observation-only grounding")
        if planner_context not in {"augmented", "native"}:
            raise ValueError("invalid planner context")
        if planner_context == "native" and (
            procedure_authority != "observe_only"
            or (task_memory is not None and task_name not in {"VideoUnmask", "VideoUnmaskSwap", "VideoRepick"})
            or (temporal_monitor is not None and temporal_monitor.active)
        ):
            raise ValueError("native context requires observe_only without injected task memory/temporal context")
        self.planner_context = planner_context
        if grounding_authority not in {"override", "observe_only"}:
            raise ValueError("invalid grounding authority")
        if grounding_authority == "observe_only" and planner_context != "native":
            raise ValueError("observation-only grounding requires native context")
        if verified_identity_conflict_enabled and (
            task_name != "VideoUnmaskSwap"
            or planner_context != "native"
            or procedure_authority != "observe_only"
            or grounding_authority != "observe_only"
        ):
            raise ValueError("verified identity conflict requires native observation-only VideoUnmaskSwap")
        self.grounding_authority = grounding_authority
        self.verified_identity_conflict_enabled = verified_identity_conflict_enabled
        required_colors = (task_memory or {}).get("required_color_order")
        if stage_receipt_hint_enabled and (
            task_name not in {"VideoUnmask", "VideoUnmaskSwap"}
            or planner_context != "native"
            or procedure_authority != "observe_only"
            or grounding_authority != "observe_only"
            or not isinstance(required_colors, list)
            or len(required_colors) < 2
        ):
            raise ValueError("stage receipt hint requires native observation-only multi-target Unmask memory")
        self.stage_receipt_hint_enabled = stage_receipt_hint_enabled
        self.first_put_executed = False
        if planner_demo_mode not in {"always", "once_with_memory"}:
            raise ValueError("invalid planner demonstration mode")
        if planner_demo_mode == "once_with_memory" and (
            task_name != "VideoUnmaskSwap" or planner_context != "native"
            or task_memory is None or demo_video is None
        ):
            raise ValueError("once_with_memory requires native VideoUnmaskSwap with memory and source demo")
        self.planner_demo_mode = planner_demo_mode
        self.image_format = image_format
        self.request_audit: list[dict[str, Any]] = []
        self.request_count = 0
        self.last_response_text: str | None = None
        self.execution_feedback = execution_feedback
        self.procedure_authority = procedure_authority
        self.executed_chunks = 0
        self.executed_steps = 0
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.demo_video = demo_video
        self.task_name = task_name
        self.task_goal = task_goal
        self.route_trajectory = RouteStickTrajectoryController(route_target_cols)
        self.task_memory = task_memory
        if repair_attempts < 0:
            raise ValueError("repair_attempts must be non-negative")
        self.repair_attempts = repair_attempts
        self.relational_color_grounding_enabled = relational_color_grounding_enabled
        self.button_grounding_enabled = button_grounding_enabled
        self.cached_button_grounding: dict[str, Any] | None = None
        self.cached_repeated_pick_grounding: dict[str, Any] | None = None
        self.cached_repeated_pick_cycle: int | None = None
        self.transition_gate = SemanticTransitionGate(semantic_transition_confirmations)
        self.temporal_monitor = temporal_monitor
        self.history_text: list[str] = []
        self.history_points: list[list[int]] = []
        memory_hint = str((task_memory or {}).get("planner_hint", ""))
        procedure_description = f"{task_goal} {memory_hint}"
        if isinstance(required_colors, list) and len(required_colors) > 1:
            count_word = {2: "two", 3: "three", 4: "four"}.get(
                len(required_colors), str(len(required_colors))
            )
            procedure_description += f" repeat the pick and put cycle {count_word} times"
        self.repetition_progress = RepetitionProgressController(memory_hint)
        self.repeated_procedure = RepeatedProcedureController(
            procedure_description,
            final_pick_terminal=self.task_name in {"VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask"},
            grasp_confirmation_predictions=(
                1 if self.task_name in {"VideoUnmask", "VideoUnmaskSwap"} else 0
            ),
            pick_prediction_schedule=(
                (5, 2, 2, 2) if self.task_name == "VideoRepick" and procedure_authority == "legacy_override" else None
            ),
        )
        self.ordered_procedure = OrderedProcedureController(task_goal)

    def _repeated_control_enabled(self) -> bool:
        """Delay a repeated suffix until an explicit prefix primitive completes."""

        return not (
            self.task_name == "ButtonUnmask"
            and self.ordered_procedure.active
            and self.ordered_procedure.actions[self.ordered_procedure.stage] == "press"
        )

    def _system_prompt(self) -> str:
        if self.planner_context == "native":
            return GROUND_SG_SYSTEM_PROMPT
        base = (
            "You are a helpful assistant to help guide the robot to complete "
            "the task by predicting one grounded language subgoal at a time."
        )
        if self.task_name == "RouteStick":
            return (
                base
                + " For RouteStick, output exactly one executable primitive in "
                "this learned form: move to the nearest [left/right] target by "
                "circling around the stick [clockwise/counterclockwise]. Infer "
                "left/right and direction from the demonstration, current image, "
                "and completed-subgoal history. Do not name stick colors and do "
                "not paraphrase the primitive. Directions are in the robot base "
                "frame: robot-left appears on image-right in the front camera, "
                "and robot-right appears on image-left."
            )
        if self.task_name == "VideoUnmaskSwap":
            return (
                base
                + " For VideoUnmaskSwap, output exactly one learned primitive: "
                "pick up the container at <y, x> that hides the "
                "[red/green/blue] cube, or put down the container. Use the "
                "demonstration identity memory after the swap and preserve the "
                "requested color order. Do not output wait or uncover."
            )
        return base

    def _required_hidden_color(self) -> str | None:
        values = (self.task_memory or {}).get("required_color_order")
        if not isinstance(values, list) or not values:
            return None
        index = self.repeated_procedure.cycle - 1
        if (
            self.repeated_procedure.stage == "put"
            and self.repeated_procedure.release_observed
            and self.repeated_procedure.cycle < len(values)
        ):
            index = self.repeated_procedure.cycle
        index = min(max(0, index), len(values) - 1)
        value = str(values[index]).lower()
        return value if value in {"red", "green", "blue"} else None

    def _stage_receipt_hint(self) -> str | None:
        if not self.stage_receipt_hint_enabled or not self.first_put_executed:
            return None
        colors = self.task_memory["required_color_order"]
        next_index = min(self.repeated_procedure.cycle, len(colors) - 1)
        next_color = str(colors[next_index]).lower()
        if next_color not in {"red", "green", "blue"}:
            return None
        release = self.repeated_procedure.release_observed
        return (
            "Execution-stage receipt (process evidence, not task success): a put-down "
            "chunk has run after the first target was attempted. Gripper reopening "
            f"observed: {'yes' if release else 'no'}. The next distinct target in "
            f"the requested order is the {next_color} cube. Inspect the current image: "
            "continue putting down if the container is still held; only after visual "
            f"placement evidence proceed to the {next_color} target. Do not treat a gripper event as proof that "
            "placement succeeded, and do not restart the first target as the second.\n"
        )

    def observe_robot_state(self, robot_state: np.ndarray) -> None:
        state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
        if state.size != 8 or not np.all(np.isfinite(state)):
            raise ValueError("execution feedback requires finite 8-D robot state")
        self.ordered_procedure.observe_execution(float(state[-1]))
        self.repeated_procedure.observe_execution(float(state[-1]))

    def record_executed_chunk(
        self,
        subgoal: str,
        states: list[np.ndarray],
        *,
        chunk_id: int,
        start_step: int,
    ) -> dict[str, Any]:
        """Consume an ordered execution receipt, never a model completion claim."""
        if self.execution_feedback != "execution_chunks":
            raise ValueError("execution receipt requires execution_chunks mode")
        if (
            type(chunk_id) is not int
            or type(start_step) is not int
            or chunk_id != self.executed_chunks + 1
            or start_step != self.executed_steps
        ):
            raise ValueError("execution receipt is duplicate, stale, or out of order")
        observed = np.asarray(states, dtype=np.float32)
        if (
            observed.ndim != 2
            or observed.shape[0] == 0
            or observed.shape[1] != 8
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError("execution receipt requires non-empty finite N x 8 states")
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise ValueError("execution receipt requires an executed subgoal")

        previous_evidence = (
            self.repeated_procedure.grasp_observed,
            self.repeated_procedure.release_observed,
        )
        # Enter the executed stage before consuming its post-action observations.
        ordered = self.ordered_procedure.commit(subgoal)
        repeated = (
            self.repeated_procedure.commit(subgoal)
            if self._repeated_control_enabled()
            else None
        )
        self.repetition_progress.apply(subgoal, subgoal)
        for state in observed:
            self.observe_robot_state(state)
        if (
            self.stage_receipt_hint_enabled
            and self.repeated_procedure.cycle == 1
            and RepetitionProgressController._action(subgoal) == "put"
            and self.repeated_procedure.grasp_observed
        ):
            self.first_put_executed = True
        self.executed_chunks += 1
        self.executed_steps += len(observed)

        boundary_reason = None
        if self.repeated_procedure.active and self._repeated_control_enabled():
            boundary_reason = self.repeated_procedure.validate(subgoal)
        if self.ordered_procedure.active:
            complete, evidence = self.ordered_procedure.stage_complete()
            if complete and self.ordered_procedure.stage < len(self.ordered_procedure.actions) - 1:
                boundary_reason = boundary_reason or f"ordered stage evidence: {evidence}"
        if self.procedure_authority == "observe_only":
            current_evidence = (
                self.repeated_procedure.grasp_observed,
                self.repeated_procedure.release_observed,
            )
            boundary_reason = (
                "gripper process evidence changed; visual review required, not completion"
                if current_evidence != previous_evidence else None
            )
        execution_event = (
            "gripper_review" if self.procedure_authority == "observe_only" else "procedure_boundary"
        ) if boundary_reason else None
        return {
            "chunk_id": chunk_id,
            "start_step": start_step,
            "end_step": self.executed_steps,
            "executed_steps": len(observed),
            "grounded_subgoal": subgoal,
            "procedure_update": ordered,
            "repeated_procedure_update": repeated,
            "post_execution_monitor": self.ordered_procedure.execution_status(),
            "repeated_stage": self.repeated_procedure.stage,
            "repeated_cycle": self.repeated_procedure.cycle,
            "pick_executed_chunks": self.repeated_procedure.pick_predictions,
            "grasp_observed": self.repeated_procedure.grasp_observed,
            "release_observed": self.repeated_procedure.release_observed,
            "boundary_reason": boundary_reason,
            "execution_event": execution_event,
            "procedure_authority": self.procedure_authority,
            "source": "executed_chunk_and_public_robot_state",
            "semantic_success_verified": False,
        }

    def infer(
        self, frame: np.ndarray, robot_state: np.ndarray | None = None
    ) -> dict[str, Any]:
        if self.button_grounding_enabled and self.cached_button_grounding is None:
            self.cached_button_grounding = button_center_grounding(
                frame, "press the button", []
            )
        if robot_state is not None and self.execution_feedback == "legacy_predictions":
            state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
            if state.size:
                self.ordered_procedure.observe_execution(float(state[-1]))
                self.repeated_procedure.observe_execution(float(state[-1]))
        history = "; ".join(
            f"{index + 1}. {value}" for index, value in enumerate(self.history_text)
        )
        if history:
            prompt = (
                f"The task goal is: {self.task_goal}\n"
                "The history of previous predicted grounded language subgoals are: "
                f"{history}\n"
            )
        else:
            prompt = (
                f"The task goal is: {self.task_goal}\n"
                "This is the initial turn for prediction\n"
            )
        if self.task_memory is not None:
            memory_hint = self.task_memory.get("planner_hint")
            memory_text = (
                str(memory_hint).strip()
                if isinstance(memory_hint, str) and memory_hint.strip()
                else json.dumps(self.task_memory, ensure_ascii=True)
            )
            prompt += (
                "Demonstration-derived identity/order constraint: "
                f"{memory_text}\n"
                "Use the constraint only to disambiguate the task. Answer the "
                "usual GroundSG question below in the original learned format. "
                "For a visible-object action, include its current-image point.\n"
            )
        stage_receipt_hint = self._stage_receipt_hint()
        if stage_receipt_hint is not None:
            prompt += stage_receipt_hint
        enforce_procedures = self.procedure_authority == "legacy_override"
        procedure_context = self.ordered_procedure.prompt_context() if enforce_procedures else None
        if procedure_context is not None:
            prompt += procedure_context + "\n"
        repeated_context = (
            self.repeated_procedure.prompt_context()
            if enforce_procedures and self._repeated_control_enabled()
            else None
        )
        if repeated_context is not None:
            prompt += repeated_context + "\n"
        if self.planner_context == "native":
            pass  # Receipts remain in the audit state, not the trained model's prompt.
        elif self.procedure_authority == "observe_only":
            prompt += (
                f"Execution receipt: {self.executed_chunks} action chunks and "
                f"{self.executed_steps} control steps have run. These are not completed "
                "task repetitions. Gripper closure/reopening is not proof that the object "
                "was lifted or placed on the target. Select the next subgoal using the "
                "current image, task goal and history.\n"
            )
        elif self.execution_feedback == "execution_chunks":
            prompt += (
                "Procedure counters measure executed action chunks, not model predictions. "
                "Gripper closure/reopening are process signals, not proof of grasping the "
                "correct object or completing the task. Check the current image as well.\n"
            )
        temporal_context = (
            self.temporal_monitor.prompt_context()
            if self.temporal_monitor is not None
            else None
        )
        if temporal_context is not None:
            prompt += temporal_context + "\n"
        content = []
        send_demo = self.demo_video is not None and (
            self.planner_demo_mode == "always" or self.request_count == 0
        )
        if send_demo:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": str(self.demo_video)},
                }
            )
        content.extend([
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image(frame, self.image_format)}},
            {
                "type": "text",
                "text": "What's the next grounded language subgoal based on current observation?",
            },
        ])
        started = time.perf_counter()
        repair_reason: str | None = None
        monitor_fallback: str | None = None
        route_update: dict[str, Any] | None = None
        tool_grounding: dict[str, Any] | None = None
        raw = ""
        accepted = False
        for attempt in range(self.repair_attempts + 1):
            attempt_content = list(content)
            if repair_reason is not None:
                attempt_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"The previous answer `{raw}` was rejected because "
                            f"{repair_reason}. Return one corrected GroundSG subgoal "
                            "for the current image."
                        ),
                    }
                )
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {"role": "user", "content": attempt_content},
                ],
                "objects": {"ref": [], "bbox": self.history_points},
                "temperature": 0,
                "max_tokens": 128,
            }
            response_payload = request_planner(self, payload, send_demo=send_demo)
            raw = planner_response_text(self, response_payload)
            self.last_response_text = raw
            tool_grounding = None
            raw, route_update = self.route_trajectory.apply(raw, frame)
            required_hidden_color = self._required_hidden_color() if enforce_procedures else None
            if (
                self.task_name in {"VideoUnmaskSwap", "ButtonUnmask"}
                and required_hidden_color is not None
                and RepetitionProgressController._action(raw) == "pick"
            ):
                raw = re.sub(
                    r"\b(red|green|blue)\b",
                    required_hidden_color,
                    raw,
                    count=1,
                    flags=re.IGNORECASE,
                )
            grounded, history_text, points = parse_planner_output(self, raw)
            grounding_context = raw
            if self.task_memory is not None:
                grounding_context += " " + str(self.task_memory.get("planner_hint", ""))
            if (
                self.relational_color_grounding_enabled
                and RELATIONAL_OBJECT_ACTIONS.search(raw) is not None
            ):
                current_cycle = self.repeated_procedure.cycle
                if self.cached_repeated_pick_cycle != current_cycle:
                    self.cached_repeated_pick_grounding = None
                    self.cached_repeated_pick_cycle = current_cycle
                candidate_grounding = relational_color_grounding(
                    frame, grounding_context
                )
                if (
                    enforce_procedures
                    and self.repeated_procedure.grasp_observed
                    and self.cached_repeated_pick_grounding is not None
                ):
                    tool_grounding = self.cached_repeated_pick_grounding
                else:
                    tool_grounding = candidate_grounding
                    if tool_grounding is not None:
                        self.cached_repeated_pick_grounding = tool_grounding
            if (
                self.task_name in {"VideoUnmaskSwap", "ButtonUnmask"}
                and required_hidden_color is not None
                and RepetitionProgressController._action(raw) == "pick"
            ):
                tool_grounding = hidden_container_grounding(
                    frame,
                    required_hidden_color,
                    self.task_memory,
                    refine_current_rgb=not self.repeated_procedure.grasp_observed,
                )
            if (
                self.temporal_monitor is not None
                and self.temporal_monitor.active
                and self.temporal_monitor.ready_to_press
                and RepetitionProgressController._action(raw) == "press"
                and self.temporal_monitor.target_point is not None
            ):
                target = self.temporal_monitor.target_point
                tool_grounding = {
                    "tool": "temporal_target_grounding",
                    "point": [int(round(float(target[0]))), int(round(float(target[1])))],
                    "source": "rgb_temporal_monitor",
                }
            missing_press_point = (
                not points
                and RepetitionProgressController._action(raw) == "press"
                and "button" in raw.lower()
            )
            if (
                tool_grounding is None
                and (self.button_grounding_enabled or missing_press_point)
            ):
                tool_grounding = button_center_grounding(frame, raw, points)
                if tool_grounding is not None:
                    self.cached_button_grounding = tool_grounding
            grounding_valid = (
                self.task_memory is None
                or bool(points)
                or tool_grounding is not None
                or not requires_visual_grounding(raw)
            )
            if not grounding_valid:
                repair_reason = "the visually targeted action contains no grounding point"
                continue
            if not enforce_procedures:
                skill_rejection = native_skill_rejection(self.task_name, grounded, points)
                if skill_rejection is not None:
                    repair_reason = skill_rejection
                    self.request_audit[-1].update(output_status="rejected", error=repair_reason)
                    continue
            procedure_rejection = self.ordered_procedure.validate(raw) if enforce_procedures else None
            if procedure_rejection is not None:
                repair_reason = procedure_rejection
                continue
            repeated_rejection = (
                self.repeated_procedure.validate(raw)
                if enforce_procedures and self._repeated_control_enabled()
                else None
            )
            if repeated_rejection is not None:
                repair_reason = repeated_rejection
                continue
            temporal_rejection = (
                self.temporal_monitor.validate(raw)
                if self.temporal_monitor is not None
                else None
            )
            if temporal_rejection is not None:
                repair_reason = temporal_rejection
                continue
            accepted = True
            self.request_audit[-1]["output_status"] = "accepted"
            break
        if not accepted:
            if not enforce_procedures:
                raise RuntimeError(f"grounded Planner repair failed without procedural override: {repair_reason}")
            fallback = (
                self.repeated_procedure.stage_fallback()
                if self._repeated_control_enabled()
                else None
            )
            ordered_fallback = self.ordered_procedure.stage_fallback()
            cached = self.transition_gate.current
            if fallback in {"advance_put", "continue_put"}:
                raw = grounded = history_text = "put it down"
                points = []
                tool_grounding = None
                monitor_fallback = fallback
                accepted = True
            elif (
                fallback == "continue_pick"
                and cached is not None
                and RepetitionProgressController._action(
                    str(cached["grounded_subgoal"])
                )
                == "pick"
            ):
                raw = str(cached["grounded_subgoal"])
                grounded = raw
                history_text = str(cached["history_text"])
                points = [list(point) for point in cached["points"]]
                tool_grounding = cached["tool_grounding"]
                monitor_fallback = fallback
                accepted = True
            elif fallback == "advance_pick":
                next_cycle = self.repeated_procedure.cycle + 1
                ordinal = ORDINAL_WORDS.get(next_cycle, f"{next_cycle}th")
                required_hidden_color = self._required_hidden_color()
                if self.task_name == "VideoUnmaskSwap" and required_hidden_color is not None:
                    raw = grounded = (
                        "pick up the container that hides the "
                        f"{required_hidden_color} cube"
                    )
                else:
                    raw = grounded = f"pick up the correct cube for the {ordinal} time"
                history_text = raw
                points = []
                if self.task_name == "VideoUnmaskSwap" and required_hidden_color is not None:
                    tool_grounding = hidden_container_grounding(
                        frame, required_hidden_color, self.task_memory
                    )
                else:
                    grounding_context = self.task_goal + " " + raw + " " + str(
                        (self.task_memory or {}).get("planner_hint", "")
                    )
                    tool_grounding = relational_color_grounding(frame, grounding_context)
                if tool_grounding is None:
                    raise RuntimeError(
                        "execution monitor could not ground the next repeated pick"
                    )
                monitor_fallback = fallback
                accepted = True
            elif fallback == "advance_press":
                if self.task_name == "VideoUnmaskSwap":
                    raise RuntimeError(
                        "pick/put plan completed without environment success"
                    )
                raw = grounded = history_text = "press the button to finish"
                points = []
                tool_grounding = (
                    button_center_grounding(frame, raw, points)
                    or self.cached_button_grounding
                )
                if tool_grounding is None:
                    raise RuntimeError(
                        "execution monitor could not ground the final button press"
                    )
                monitor_fallback = fallback
                accepted = True
            elif (
                ordered_fallback == "continue_ordered_stage"
                and cached is not None
                and RepetitionProgressController._action(
                    str(cached["grounded_subgoal"])
                )
                == self.ordered_procedure.actions[self.ordered_procedure.stage]
            ):
                raw = str(cached["grounded_subgoal"])
                grounded = raw
                history_text = str(cached["history_text"])
                points = [list(point) for point in cached["points"]]
                tool_grounding = cached["tool_grounding"]
                monitor_fallback = ordered_fallback
                accepted = True
            else:
                raise RuntimeError(f"grounded Planner repair failed: {repair_reason}")
        model_points = [list(point) for point in points]
        tool_grounding_proposal = copy.deepcopy(tool_grounding)
        if self.grounding_authority == "observe_only":
            tool_grounding = None
        identity_conflict = None
        if self.verified_identity_conflict_enabled and self.task_memory is not None:
            pick = UNMASK_SWAP_PICK_PATTERN.fullmatch(grounded.lower())
            if pick is not None and points:
                identity_conflict = verified_identity_conflict(
                    frame, pick[3], points[0], self.task_memory,
                    expected_color=self._required_hidden_color(),
                    grasp_observed=self.repeated_procedure.grasp_observed,
                )
                if identity_conflict["resolved"]:
                    point = identity_conflict["point"]
                    grounded = (
                        f"pick up the container at <{point[0]}, {point[1]}> "
                        f"that hides the {pick[3]} cube"
                    )
                    points = [list(point)]
        if (
            tool_grounding is not None
            and tool_grounding.get("tool") == "hidden_container_memory_grounding"
        ):
            point = list(tool_grounding["point"])
            color = str(tool_grounding["color"])
            grounded = (
                f"pick up the container at <{point[0]}, {point[1]}> "
                f"that hides the {color} cube"
            )
            history_text = (
                f"pick up the container at <bbox> that hides the {color} cube"
            )
            points = [point]
        if tool_grounding is not None:
            point = list(tool_grounding["point"])
            if points:
                grounded = SCALED_POINT_PATTERN.sub(
                    f"<{point[0]}, {point[1]}>", grounded, count=1
                )
                points[0] = point
            else:
                grounded = re.sub(
                    r"\s+at\s+<\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?>",
                    "",
                    grounded,
                    count=1,
                ).strip()
                grounded = f"{grounded} at <{point[0]}, {point[1]}>"
                history_text = f"{history_text} at <bbox>"
                points = [point]
        if (
            self.temporal_monitor is not None
            and self.temporal_monitor.active
            and self.temporal_monitor.ready_to_press
            and RepetitionProgressController._action(grounded) == "press"
        ):
            point = points[0]
            grounded = (
                "press the button to stop the cube on the target "
                f"at <{point[0]}, {point[1]}>"
            )
            history_text = "press the button to stop the cube on the target at <bbox>"
        selected, transition_gate = self.transition_gate.apply(
            {
                "grounded_subgoal": grounded,
                "history_text": history_text,
                "points": points,
                "model_points": model_points,
                "tool_grounding": tool_grounding,
            }
        )
        grounded = str(selected["grounded_subgoal"])
        history_text = str(selected["history_text"])
        points = [list(point) for point in selected["points"]]
        model_points = [list(point) for point in selected["model_points"]]
        tool_grounding = selected["tool_grounding"]
        progress = (
            copy.deepcopy(self.repetition_progress)
            if self.execution_feedback == "execution_chunks"
            else self.repetition_progress
        )
        progress_update = None
        if enforce_procedures:
            grounded, history_text, progress_update = progress.apply(
                grounded, history_text
            )
        procedure_update = (
            self.ordered_procedure.commit(grounded)
            if self.execution_feedback == "legacy_predictions"
            else None
        )
        repeated_procedure_update = (
            self.repeated_procedure.commit(grounded)
            if self.execution_feedback == "legacy_predictions" and self._repeated_control_enabled()
            else None
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not self.history_text or self.history_text[-1] != history_text:
            self.history_text.append(history_text)
            self.history_points.extend(points)
        return {
            "raw": raw.strip(),
            "grounded_subgoal": grounded,
            "history_text": history_text,
            "points": points,
            "model_points": model_points,
            "tool_grounding": tool_grounding,
            "tool_grounding_proposal": tool_grounding_proposal,
            "identity_conflict": identity_conflict,
            "stage_receipt_hint": stage_receipt_hint,
            "grounding_authority": (
                "verified_identity_conflict"
                if identity_conflict is not None and identity_conflict["resolved"]
                else self.grounding_authority
            ),
            "demo_video_sent": send_demo,
            "progress_update": progress_update,
            "transition_gate": transition_gate,
            "procedure_update": procedure_update,
            "repeated_procedure_update": repeated_procedure_update,
            "latency_ms": elapsed_ms,
            "repair_attempts": attempt,
            "monitor_fallback": monitor_fallback,
            "procedure_authority": self.procedure_authority,
            "route_update": route_update,
        }


def current_observation(observation: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    front = np.asarray(observation["front_rgb_list"][-1], dtype=np.uint8)
    wrist = np.asarray(observation["wrist_rgb_list"][-1], dtype=np.uint8)
    state = pack_robomme_state(
        observation["joint_state_list"][-1],
        observation["gripper_state_list"][-1],
    )
    return front, wrist, state


def initial_history(observation: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    front = [np.asarray(value, dtype=np.uint8) for value in observation["front_rgb_list"]]
    wrist = [np.asarray(value, dtype=np.uint8) for value in observation["wrist_rgb_list"]]
    states = [
        pack_robomme_state(joints, gripper)
        for joints, gripper in zip(
            observation["joint_state_list"], observation["gripper_state_list"]
        )
    ]
    return front, wrist, states


def run_episode(args: argparse.Namespace, resources: ExitStack) -> int:
    if args.max_steps <= 0 or args.action_horizon <= 0:
        raise ValueError("max steps and action horizon must be positive")
    if args.planner_profile == "raw" and args.execution_feedback != "legacy_predictions":
        raise ValueError("raw Planner baseline must not enable CARVE execution feedback")
    if args.procedure_authority == "observe_only" and (
        args.planner_profile != "carve"
        or args.task not in {"PickXtimes", "VideoUnmask", "VideoUnmaskSwap", "VideoRepick"}
        or args.execution_feedback != "execution_chunks"
    ):
        raise ValueError("observe_only requires carve admitted task execution_chunks")
    if args.planner_context == "native" and (
        args.planner_profile != "carve" or args.procedure_authority != "observe_only"
        or (args.task_memory_file is not None and args.task not in {"VideoUnmask", "VideoUnmaskSwap", "VideoRepick"})
    ):
        raise ValueError("native context requires carve observe_only without injected task memory")
    if args.grounding_authority == "observe_only" and args.planner_context != "native":
        raise ValueError("observation-only grounding requires native context")
    if args.planner_demo_mode == "once_with_memory" and (
        args.task != "VideoUnmaskSwap" or args.planner_context != "native"
        or args.task_memory_file is None
    ):
        raise ValueError("once_with_memory requires native VideoUnmaskSwap and bound memory")
    if args.verified_point_reuse and (
        args.task not in {"VideoUnmask", "VideoUnmaskSwap"}
        or args.planner_context != "native"
        or args.procedure_authority != "observe_only"
        or args.planner_schedule != GroundedSubgoalScheduleMode.SELECTIVE.value
        or args.planner_max_reusable_points != 0
        or args.task_memory_file is None
    ):
        raise ValueError("verified point reuse requires native VideoUnmask-family memory and selective scheduling")
    if args.verified_identity_conflict and (
        args.task not in {"VideoUnmask", "VideoUnmaskSwap"}
        or args.planner_profile != "carve"
        or args.planner_context != "native"
        or args.procedure_authority != "observe_only"
        or args.grounding_authority != "observe_only"
        or args.task_memory_file is None
        or args.memory_hint_style != "precise"
    ):
        raise ValueError("verified identity conflict requires precise native observation-only Swap memory")
    if args.stage_receipt_hint and (
        args.task not in {"VideoUnmask", "VideoUnmaskSwap"}
        or args.planner_profile != "carve"
        or args.planner_context != "native"
        or args.procedure_authority != "observe_only"
        or args.grounding_authority != "observe_only"
        or args.task_memory_file is None
        or args.memory_hint_style != "precise"
    ):
        raise ValueError("stage receipt hint requires precise native observation-only Unmask memory")
    if args.memory_hint_style == "coarse_grid" and (
        args.task not in {"VideoUnmask", "VideoUnmaskSwap"}
        or args.task_memory_file is None
        or args.planner_profile != "carve"
        or args.planner_context != "native"
    ):
        raise ValueError("coarse grid hints require native VideoUnmask-family identity memory")
    if args.memory_use_policy == "motion_gate" and (
        args.task not in {"VideoUnmask", "VideoUnmaskSwap"}
        or args.task_memory_file is None
        or args.planner_profile != "carve"
        or args.planner_context != "native"
        or args.planner_demo_mode != "always"
        or args.memory_hint_style != "precise"
    ):
        raise ValueError("motion gate requires native VideoUnmask-family precise memory and full demo")
    if args.memory_on_uncertain == "without_memory" and (
        args.planner_context != "native" or args.planner_demo_mode != "always"
    ):
        raise ValueError("memory fallback requires native context with the original demonstration")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output is nonempty; use a fresh run directory")
    if (
        args.planner_profile == "raw"
        and args.planner_schedule != GroundedSubgoalScheduleMode.EVERY_CHUNK.value
    ):
        raise ValueError("raw Planner baseline requires --planner-schedule every_chunk")
    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    started = time.perf_counter()
    client = MMEVLAWebsocketClientPolicy(args.policy_host, args.policy_port)
    wait_for_reset(client)
    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=args.max_steps,
    )
    episode_count = builder.get_episode_num()
    if not 0 <= args.episode < episode_count:
        raise ValueError(f"episode {args.episode} is outside {args.task} test range [0, {episode_count})")
    env = builder.make_env_for_episode(args.episode)
    resources.callback(env.close)
    observation, info = env.reset()
    goal = info["task_goal"]
    instruction = str(goal[0] if isinstance(goal, (list, tuple)) else goal).strip()
    history_front, history_wrist, history_states = initial_history(observation)
    initial_observation_sha256 = hashlib.sha256(
        np.asarray(history_front, dtype=np.uint8).tobytes()
        + np.asarray(history_wrist, dtype=np.uint8).tobytes()
        + np.asarray(history_states, dtype=np.float32).tobytes()
        + instruction.encode("utf-8")
    ).hexdigest()
    initial_memory_frames = len(history_front)
    exec_start_idx = initial_memory_frames - 1
    frames = [paired_frame(front, wrist) for front, wrist in zip(history_front, history_wrist)]
    temporal_monitor = StopCubeTemporalMonitor(
        instruction, history_front[-1], action_horizon=args.action_horizon
    )

    args.output.mkdir(parents=True, exist_ok=True)
    demo_path = args.output / "initial_demo_front.mp4"
    demo_frames = planner_demo_frames(history_front, args.demo_history_mode)
    demo_started = time.perf_counter()
    if demo_frames:
        imageio.mimsave(demo_path, demo_frames, fps=30)
    demo_encoding_s = time.perf_counter() - demo_started
    task_memory = None
    memory_provenance = None
    memory_started = time.perf_counter()
    if args.task_memory_file is not None and args.planner_context == "native":
        task_memory, memory_provenance = load_verified_identity_memory(
            args.task_memory_file, current_demo=demo_path, task=args.task,
            episode=args.episode, instruction=instruction,
            on_uncertain=args.memory_on_uncertain, hint_style=args.memory_hint_style,
            memory_use_policy=args.memory_use_policy,
        )
    elif args.task_memory_file is not None:
        task_memory = json.loads(args.task_memory_file.read_text(encoding="utf-8"))
        if not isinstance(task_memory, dict):
            raise ValueError("task memory must be a JSON object")
        if isinstance(task_memory.get("memory"), dict):
            task_memory = task_memory["memory"]
    elif args.planner_profile == "carve" and args.task == "ButtonUnmask":
        task_memory = build_button_unmask_memory(history_front[-1], instruction)
        if task_memory is None:
            raise RuntimeError("could not build ButtonUnmask identity memory from RGB")
    memory_admission_s = time.perf_counter() - memory_started
    demo_video = (
        planner_visible_path(demo_path, args) if initial_memory_frames >= 2 and demo_frames else None
    )
    if args.planner_profile == "raw":
        if task_memory is not None:
            raise ValueError("raw Planner baseline must not receive CARVE task memory")
        if args.relational_color_grounding or args.button_grounding:
            raise ValueError("raw Planner baseline must not enable CARVE grounding tools")
        planner = RawGroundedPlannerClient(
            endpoint=args.planner_endpoint,
            model=args.planner_model,
            timeout_s=args.planner_timeout_s,
            demo_video=demo_video,
            task_goal=instruction,
            task_name=args.task if args.demo_history_mode == "official" else None,
            image_format=args.planner_image_format,
        )
    else:
        planner = GroundedPlannerClient(
            endpoint=args.planner_endpoint,
            model=args.planner_model,
            timeout_s=args.planner_timeout_s,
            demo_video=demo_video,
            task_name=args.task,
            task_goal=instruction,
            route_target_cols=(
                route_stick_target_sequence(history_front)
                if args.task == "RouteStick"
                else None
            ),
            task_memory=task_memory,
            repair_attempts=args.planner_repair_attempts,
            relational_color_grounding_enabled=args.relational_color_grounding,
            button_grounding_enabled=args.button_grounding,
            semantic_transition_confirmations=args.semantic_transition_confirmations,
            temporal_monitor=temporal_monitor,
            execution_feedback=args.execution_feedback,
            procedure_authority=args.procedure_authority,
            planner_context=args.planner_context,
            grounding_authority=args.grounding_authority,
            verified_identity_conflict_enabled=args.verified_identity_conflict,
            stage_receipt_hint_enabled=args.stage_receipt_hint,
            planner_demo_mode=args.planner_demo_mode,
            image_format=args.planner_image_format,
        )
        if args.execution_feedback == "execution_chunks":
            planner.observe_robot_state(history_states[-1])
    schedule_config = GroundedSubgoalScheduleConfig(
        mode=GroundedSubgoalScheduleMode(args.planner_schedule),
        min_reuse_chunks=args.planner_min_reuse_chunks,
        max_reuse_chunks=args.planner_max_reuse_chunks,
        visual_change_threshold=args.planner_visual_change_threshold,
        gripper_change_threshold=args.planner_gripper_change_threshold,
    )
    if args.planner_max_reusable_points < 0:
        raise ValueError("planner_max_reusable_points must be non-negative")
    scheduler = GroundedSubgoalScheduler(schedule_config)

    policy_latencies: list[float] = []
    planner_latencies: list[float] = []
    planner_trace: list[dict[str, Any]] = []
    schedule_trace: list[dict[str, Any]] = []
    execution_feedback_trace: list[dict[str, Any]] = []
    cached_planner_result: dict[str, Any] | None = None
    policy_calls = 0
    policy_request_audit: list[dict[str, Any]] = []
    steps = 0
    terminated = False
    truncated = False
    failure: str | None = None
    pending_execution_event: str | None = None
    rollout_started = time.perf_counter()
    try:
        while steps < args.max_steps and not (terminated or truncated):
            try:
                response = client.add_buffer(pack_history(history_front, history_states, exec_start_idx))
                if not response.get("add_buffer_finished", False):
                    raise RuntimeError("policy rejected the history buffer")
            except Exception as error:
                failure = str(error)
                break

            front, wrist, state = current_observation(observation)
            schedule_decision = scheduler.decide(
                frame=front,
                state=state,
                execution_event=pending_execution_event,
            )
            pending_execution_event = None
            reuse_validation = None
            reuse_validation_latency_ms = None
            if (
                args.verified_point_reuse
                and not schedule_decision.invoke
                and cached_planner_result is not None
                and cached_planner_result["points"]
            ):
                validation_started = time.perf_counter()
                validated, reuse_validation = revalidate_unmask_pick(
                    front,
                    cached_planner_result,
                    task_memory,
                    execution_feedback_trace[-1] if execution_feedback_trace else None,
                )
                reuse_validation_latency_ms = (time.perf_counter() - validation_started) * 1000.0
                if validated is None:
                    schedule_decision = GroundedSubgoalScheduleDecision(
                        invoke=True,
                        reason=f"point_revalidation:{reuse_validation['reason']}",
                        chunks_since_planner=schedule_decision.chunks_since_planner,
                        visual_change=schedule_decision.visual_change,
                        gripper_change=schedule_decision.gripper_change,
                        execution_event=schedule_decision.execution_event,
                    )
                else:
                    cached_planner_result = validated
            planner_trace_index: int | None = None
            if schedule_decision.invoke:
                try:
                    planner_result = planner.infer(front, robot_state=state)
                except Exception as exc:
                    failure = str(exc)
                    if planner.request_audit and planner.request_audit[-1]["output_status"] == "unvalidated":
                        planner.request_audit[-1].update(output_status="rejected", error=failure)
                    break
                planner_latencies.append(float(planner_result["latency_ms"]))
                planner_trace_index = len(planner_trace)
                planner_trace.append(
                    {
                        "step": steps,
                        "trigger_reason": schedule_decision.reason,
                        "schedule": schedule_decision.to_dict(),
                        **planner_result,
                    }
                )
                cached_planner_result = planner_result
                scheduler.record_planner_result(
                    frame=front,
                    state=state,
                    reuse_permitted=(
                        len(planner_result["points"])
                        <= args.planner_max_reusable_points
                        or (
                            args.verified_point_reuse
                            and cacheable_unmask_pick(planner_result, task_memory)
                        )
                    ),
                )
            elif cached_planner_result is None:
                failure = "Planner scheduler admitted reuse without a cached subgoal"
                break

            assert cached_planner_result is not None
            subgoal = str(cached_planner_result["grounded_subgoal"])
            schedule_trace.append(
                {
                    "step": steps,
                    "planner_invoked": schedule_decision.invoke,
                    "planner_trace_index": planner_trace_index,
                    "source_planner_trace_index": len(planner_trace) - 1,
                    "decision": schedule_decision.to_dict(),
                    "reuse_validation": reuse_validation,
                    "reuse_validation_latency_ms": reuse_validation_latency_ms,
                    "grounded_subgoal": subgoal,
                    "history_text": str(cached_planner_result["history_text"]),
                }
            )
            payload = {
                "observation/image": front,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": instruction,
                "simple_subgoal": subgoal,
                "grounded_subgoal": subgoal,
            }
            policy_input_sha256 = hashlib.sha256(
                np.ascontiguousarray(front).tobytes()
                + np.ascontiguousarray(wrist).tobytes()
                + np.ascontiguousarray(state).tobytes()
                + instruction.encode("utf-8")
                + subgoal.encode("utf-8")
            ).hexdigest()
            infer_started = time.perf_counter()
            request_error = None
            try:
                response = client.infer(payload)
            except Exception as error:
                request_error = failure = str(error)
            finally:
                elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
                policy_request_audit.append({"request_index": len(policy_request_audit) + 1,
                                             "latency_ms": elapsed_ms, "error": request_error,
                                             "input_sha256": policy_input_sha256})
            if request_error is not None:
                break
            policy_latencies.append(elapsed_ms)
            policy_calls += 1
            actions = np.asarray(response["actions"], dtype=np.float32)[: args.action_horizon]
            if (
                actions.ndim != 2
                or actions.shape[0] == 0
                or actions.shape[1] != 8
                or not np.all(np.isfinite(actions))
            ):
                failure = "policy returned empty, non-finite, or non-Nx8 actions"
                break
            policy_request_audit[-1]["actions_sha256"] = hashlib.sha256(
                np.ascontiguousarray(actions).tobytes()
            ).hexdigest()

            history_front.clear()
            history_wrist.clear()
            history_states.clear()
            exec_start_idx = 0
            chunk_start_step = steps
            for action in actions:
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                front, wrist, state = current_observation(observation)
                history_front.append(front.copy())
                history_wrist.append(wrist.copy())
                history_states.append(state.copy())
                frames.append(paired_frame(front, wrist))
                if args.planner_profile == "carve":
                    temporal_event = temporal_monitor.observe(front, steps)
                    if temporal_event is not None and temporal_monitor.ready_to_press:
                        pending_execution_event = "temporal_target_occurrence"
                        break
                if steps >= args.max_steps or terminated or truncated:
                    break
            scheduler.record_chunk_executed()
            if args.execution_feedback == "execution_chunks":
                feedback = planner.record_executed_chunk(
                    subgoal,
                    history_states,
                    chunk_id=len(execution_feedback_trace) + 1,
                    start_step=chunk_start_step,
                )
                execution_feedback_trace.append(feedback)
                pending_execution_event = pending_execution_event or feedback["execution_event"]
            if (
                args.planner_profile == "carve"
                and temporal_monitor.record_executed_subgoal(subgoal)
            ):
                pending_execution_event = "temporal_prepare_complete"

        rollout_time_s = time.perf_counter() - rollout_started
        video_path = args.output / f"{args.task}_ep{args.episode}_vlm_groundsg.mp4"
        video_started = time.perf_counter()
        imageio.mimsave(video_path, frames, fps=30)
        video_encoding_s = time.perf_counter() - video_started
        preparation_s = rollout_started - started
        build_s = 0.0 if args.task_memory_file is None else (memory_provenance or {}).get("compile_time_s")
        planner_request_ms = [item["request_latency_ms"] for item in planner.request_audit]
        policy_request_ms = [item["latency_ms"] for item in policy_request_audit]
        status = str(info.get("status", "unknown"))
        summary = {
            "protocol": (
                "carve.robomme.vlm_groundsg.execution_feedback.v1"
                if args.execution_feedback == "execution_chunks"
                else "carve.robomme.vlm_groundsg.v2"
            ),
            "deployable_method": True,
            "privileged_online_subgoal_used": False,
            "claim_boundary": (
                "Raw official-style Qwen GroundSG baseline without CARVE memory, "
                "grounding tools, semantic repair, procedure control or temporal monitor."
                if args.planner_profile == "raw"
                else "Official RoboMME Qwen GroundSG adapter used through a CARVE "
                "provider boundary; no evaluator or oracle subgoal enters Planner or VLA."
            ),
            "policy_label": args.policy_label,
            "planner_label": args.planner_label,
            "planner_profile": args.planner_profile,
            "execution_feedback_mode": args.execution_feedback,
            "procedure_authority": args.procedure_authority,
            "execution_feedback_trace": execution_feedback_trace,
            "initial_observation_sha256": initial_observation_sha256,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_steps": args.max_steps,
            "action_horizon": args.action_horizon,
            "run_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "task": args.task,
            "episode": args.episode,
            "instruction": instruction,
            "status": status,
            "success": status == "success",
            "failure": failure,
            "initial_memory_frames": initial_memory_frames,
            "planner_demo_frames": len(demo_frames),
            "policy_calls": policy_calls,
            "policy_request_audit": policy_request_audit,
            "planner_calls": len(planner_trace),
            "planner_request_count": planner.request_count,
            "planner_request_audit": getattr(planner, "request_audit", []),
            "planner_last_response_on_failure": planner.last_response_text if failure else None,
            "planner_reuse_hits": sum(
                not bool(item["planner_invoked"]) for item in schedule_trace
            ),
            "planner_call_reduction_fraction": (
                1.0 - len(planner_trace) / policy_calls if policy_calls else None
            ),
            "executed_steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "planner_trace": planner_trace,
            "planner_schedule": {
                "mode": schedule_config.mode.value,
                "min_reuse_chunks": schedule_config.min_reuse_chunks,
                "max_reuse_chunks": schedule_config.max_reuse_chunks,
                "visual_change_threshold": schedule_config.visual_change_threshold,
                "gripper_change_threshold": schedule_config.gripper_change_threshold,
                "max_reusable_points": args.planner_max_reusable_points,
                "verified_point_reuse": args.verified_point_reuse,
            },
            "task_memory": task_memory,
            "memory_provenance": memory_provenance,
            "temporal_monitor": {
                "active": temporal_monitor.active,
                "required_occurrences": temporal_monitor.required,
                "observed_occurrences": temporal_monitor.count,
                "stage": temporal_monitor.stage,
                "prepare_chunks": temporal_monitor.prepare_chunks,
                "prepare_chunks_required": temporal_monitor.prepare_chunks_required,
                "press_lead_steps": temporal_monitor.press_lead_steps,
                "predicted_target_step": temporal_monitor.predicted_target_step,
                "period_estimate_source": temporal_monitor.period_estimate_source,
                "ready_to_press": temporal_monitor.ready_to_press,
                "target_point": (
                    temporal_monitor.target_point.tolist()
                    if temporal_monitor.target_point is not None
                    else None
                ),
                "cube_hue": temporal_monitor.cube_hue,
                "initialization_error": temporal_monitor.initialization_error,
                "events": temporal_monitor.trace,
            },
            "schedule_trace": schedule_trace,
            "video_frames": len(frames),
            "video_path": str(video_path),
            "policy_latency_ms": {
                "mean": statistics.fmean(policy_latencies) if policy_latencies else None,
                "p50": percentile(policy_latencies, 50.0),
                "p95": percentile(policy_latencies, 95.0),
                "max": max(policy_latencies) if policy_latencies else None,
            },
            "planner_latency_ms": {
                "mean": statistics.fmean(planner_latencies) if planner_latencies else None,
                "p50": percentile(planner_latencies, 50.0),
                "p95": percentile(planner_latencies, 95.0),
                "max": max(planner_latencies) if planner_latencies else None,
            },
            "wall_time_s": time.perf_counter() - started,
            "rollout_time_s": rollout_time_s,
            "timing": {
                "preparation_s": preparation_s,
                "demo_encoding_s": demo_encoding_s,
                "memory_admission_s": memory_admission_s,
                "rollout_s": rollout_time_s,
                "video_encoding_s": video_encoding_s,
                "memory_build_s": build_s,
                "task_with_memory_build_s": None if build_s is None else preparation_s + rollout_time_s + build_s,
                "service_startup_included": False,
                "shared_memory_model_load_s": (memory_provenance or {}).get("shared_model_load_s"),
                "memory_build_scope": "charged once per standalone first-use case, even when artifact is reused",
                "planner_requests": request_latency_summary(planner_request_ms),
                "policy_requests": request_latency_summary(policy_request_ms),
                "point_revalidation": request_latency_summary([
                    float(item["reuse_validation_latency_ms"])
                    for item in schedule_trace
                    if item["reuse_validation_latency_ms"] is not None
                ]),
            },
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0 if failure is None else 2
    finally:
        resources.close()


def request_latency_summary(samples: list[float]) -> dict[str, Any]:
    return {"samples_ms": samples, "count": len(samples), "first_ms": samples[0] if samples else None,
            "warm_count": max(0, len(samples) - 1), "warm_p50_ms": percentile(samples[1:], 50),
            "warm_p95_ms": percentile(samples[1:], 95), "total_ms": sum(samples)}


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        with ExitStack() as resources:
            return run_episode(args, resources)
    except Exception as error:
        if (not isinstance(error, FileExistsError) and not (args.output / "summary.json").exists()
                and not (args.output / "failure.json").exists()):
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / "failure.json").open("x") as output:
                json.dump({"task": args.task, "episode": args.episode,
                           "error_type": type(error).__name__, "error": str(error),
                           "elapsed_s": time.perf_counter() - started}, output, indent=2)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
