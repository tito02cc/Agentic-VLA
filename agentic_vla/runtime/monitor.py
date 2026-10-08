"""Deployable execution-risk signals for frozen embodied policies."""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class MonitorConfig:
    """Thresholds for an auditable, training-free execution monitor."""

    window_size: int = 4
    warmup_steps: int = 0
    command_threshold: float = 0.03
    state_response_threshold: float = 0.002
    visual_response_threshold: float = 0.004
    stale_action_steps: int = 8
    no_progress_steps: int | None = None
    idle_command_threshold: float = 0.01
    idle_state_response_threshold: float = 0.015
    idle_visual_response_threshold: float = 0.004
    low_slack_ms: float = 30.0
    stall_weight: float = 0.45
    staleness_weight: float = 0.20
    uncertainty_weight: float = 0.20
    deadline_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.stale_action_steps <= 0:
            raise ValueError("stale_action_steps must be positive")
        if self.no_progress_steps is not None and self.no_progress_steps <= 0:
            raise ValueError("no_progress_steps must be positive or disabled")
        idle_thresholds = (
            self.idle_command_threshold,
            self.idle_state_response_threshold,
            self.idle_visual_response_threshold,
        )
        if any(value < 0 for value in idle_thresholds):
            raise ValueError("idle thresholds must be non-negative")
        weights = (
            self.stall_weight,
            self.staleness_weight,
            self.uncertainty_weight,
            self.deadline_weight,
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("monitor weights must be non-negative with positive sum")


@dataclasses.dataclass(frozen=True)
class RiskAssessment:
    """Normalized risk and observable evidence for one control step."""

    score: float
    bucket: str
    event: str | None
    components: Mapping[str, float]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ExecutionRiskMonitor:
    """Detect poor action response without simulator object poses or rewards."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._commands: deque[float] = deque(maxlen=self.config.window_size)
        self._state_responses: deque[float] = deque(maxlen=self.config.window_size)
        self._visual_responses: deque[float] = deque(maxlen=self.config.window_size)
        self._previous_state: np.ndarray | None = None
        self._previous_frame: np.ndarray | None = None
        self._last_event: str | None = None
        self._event_streak = 0
        self._idle_streak = 0
        self._updates = 0

    def reset(self) -> None:
        self._commands.clear()
        self._state_responses.clear()
        self._visual_responses.clear()
        self._previous_state = None
        self._previous_frame = None
        self._last_event = None
        self._event_streak = 0
        self._idle_streak = 0
        self._updates = 0

    @staticmethod
    def _vector(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("monitor input contains non-finite values")
        return array

    @staticmethod
    def _frame(value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim not in (2, 3):
            raise ValueError("frame must be a 2-D or 3-D array")
        if not np.all(np.isfinite(array)):
            raise ValueError("frame contains non-finite values")
        return array.astype(np.float32, copy=False)

    def update(
        self,
        *,
        proprio: Any,
        commanded_action: Any,
        frame: Any | None = None,
        action_age_steps: int = 0,
        uncertainty: float | None = None,
        deadline_slack_ms: float | None = None,
    ) -> RiskAssessment:
        state = self._vector(proprio)
        action = self._vector(commanded_action)
        image = self._frame(frame)
        if action_age_steps < 0:
            raise ValueError("action_age_steps must be non-negative")

        command = float(np.linalg.norm(action[:-1] if action.size > 1 else action))
        state_response = 0.0
        if self._previous_state is not None:
            if self._previous_state.shape != state.shape:
                raise ValueError("proprio shape changed within an episode")
            state_response = float(np.linalg.norm(state - self._previous_state))

        visual_response = 0.0
        if image is not None and self._previous_frame is not None:
            if self._previous_frame.shape != image.shape:
                raise ValueError("frame shape changed within an episode")
            scale = 255.0 if max(float(image.max()), float(self._previous_frame.max())) > 1.5 else 1.0
            visual_response = float(np.mean(np.abs(image - self._previous_frame)) / scale)

        self._commands.append(command)
        self._state_responses.append(state_response)
        self._visual_responses.append(visual_response)
        self._previous_state = state.copy()
        self._previous_frame = None if image is None else image.copy()
        self._updates += 1

        warm = bool(
            len(self._commands) >= self.config.window_size
            and self._updates > self.config.warmup_steps
        )
        mean_command = float(np.mean(self._commands))
        mean_state_response = float(np.mean(self._state_responses))
        mean_visual_response = float(np.mean(self._visual_responses))
        stalled = bool(
            warm
            and mean_command >= self.config.command_threshold
            and mean_state_response < self.config.state_response_threshold
            and (image is None or mean_visual_response < self.config.visual_response_threshold)
        )

        idle = bool(
            warm
            and self.config.no_progress_steps is not None
            and mean_command < self.config.idle_command_threshold
            and mean_state_response < self.config.idle_state_response_threshold
            and (image is None or mean_visual_response < self.config.idle_visual_response_threshold)
        )
        self._idle_streak = self._idle_streak + 1 if idle else 0
        no_progress = bool(
            self.config.no_progress_steps is not None
            and self._idle_streak >= self.config.no_progress_steps
        )

        stall_risk = 1.0 if stalled else 0.0
        no_progress_risk = 1.0 if no_progress else 0.0
        movement_risk = max(stall_risk, no_progress_risk)
        staleness_risk = min(1.0, action_age_steps / self.config.stale_action_steps)
        uncertainty_risk = 0.0 if uncertainty is None else min(1.0, max(0.0, float(uncertainty)))
        if deadline_slack_ms is None:
            deadline_risk = 0.0
        else:
            deadline_risk = min(
                1.0,
                max(0.0, (self.config.low_slack_ms - float(deadline_slack_ms)) / self.config.low_slack_ms),
            )
        weights = np.asarray(
            [
                self.config.stall_weight,
                self.config.staleness_weight,
                self.config.uncertainty_weight,
                self.config.deadline_weight,
            ],
            dtype=np.float64,
        )
        values = np.asarray(
            [movement_risk, staleness_risk, uncertainty_risk, deadline_risk],
            dtype=np.float64,
        )
        score = float(np.dot(weights, values) / weights.sum())
        bucket = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
        event = (
            "stall"
            if stalled
            else "no_progress"
            if no_progress
            else "stale_action"
            if staleness_risk >= 1.0
            else None
        )
        if event is not None and event == self._last_event:
            self._event_streak += 1
        elif event is not None:
            self._event_streak = 1
        else:
            self._event_streak = 0
        self._last_event = event
        return RiskAssessment(
            score=score,
            bucket=bucket,
            event=event,
            components={
                "stall": stall_risk,
                "no_progress": no_progress_risk,
                "staleness": staleness_risk,
                "uncertainty": uncertainty_risk,
                "deadline": deadline_risk,
            },
            evidence={
                "window_ready": warm,
                "warmup_steps_remaining": max(
                    0,
                    self.config.warmup_steps - self._updates + 1,
                ),
                "mean_command": mean_command,
                "mean_state_response": mean_state_response,
                "mean_visual_response": mean_visual_response,
                "action_age_steps": int(action_age_steps),
                "idle_streak": self._idle_streak,
                "deadline_slack_ms": deadline_slack_ms,
                "event_streak": self._event_streak,
            },
        )
