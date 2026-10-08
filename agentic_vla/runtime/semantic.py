"""Deadline-aware asynchronous semantic observation for CARVE."""

from __future__ import annotations

import dataclasses
import enum
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np


class GroundedSubgoalScheduleMode(str, enum.Enum):
    """Planner invocation schedules for grounded VLA subgoals."""

    EVERY_CHUNK = "every_chunk"
    SELECTIVE = "selective"


@dataclasses.dataclass(frozen=True)
class GroundedSubgoalScheduleConfig:
    """Conservative reuse limits for expensive grounded Planner calls."""

    mode: GroundedSubgoalScheduleMode = GroundedSubgoalScheduleMode.SELECTIVE
    min_reuse_chunks: int = 1
    max_reuse_chunks: int = 2
    visual_change_threshold: float = 0.09
    gripper_change_threshold: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", GroundedSubgoalScheduleMode(self.mode))
        if self.min_reuse_chunks < 0:
            raise ValueError("min_reuse_chunks must be non-negative")
        if self.max_reuse_chunks <= 0:
            raise ValueError("max_reuse_chunks must be positive")
        if self.min_reuse_chunks > self.max_reuse_chunks:
            raise ValueError("min_reuse_chunks must not exceed max_reuse_chunks")
        if self.visual_change_threshold < 0:
            raise ValueError("visual_change_threshold must be non-negative")
        if self.gripper_change_threshold < 0:
            raise ValueError("gripper_change_threshold must be non-negative")


@dataclasses.dataclass(frozen=True)
class GroundedSubgoalScheduleDecision:
    """Planner-call decision at a chunk boundary, not semantic completion authority."""

    invoke: bool
    reason: str
    chunks_since_planner: int
    visual_change: float | None = None
    gripper_change: float | None = None
    execution_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class GroundedSubgoalScheduler:
    """Reuse grounded subgoals until deployable evidence requires replanning.

    The scheduler never infers task semantics. It only controls when the VLM is
    invoked, using chunk age, image change, gripper change and explicit
    execution events. Invocation does not authorize interrupting a skill or
    declaring it complete. Callers own execution-boundary and verification
    gates; the VLM proposes the next semantic stage and grounded points.
    """

    def __init__(self, config: GroundedSubgoalScheduleConfig | None = None) -> None:
        self.config = config or GroundedSubgoalScheduleConfig()
        self._has_plan = False
        self._chunks_since_planner = 0
        self._reference_frame: np.ndarray | None = None
        self._reference_state: np.ndarray | None = None
        self._reuse_permitted = True

    @property
    def chunks_since_planner(self) -> int:
        return self._chunks_since_planner

    def reset(self) -> None:
        self._has_plan = False
        self._chunks_since_planner = 0
        self._reference_frame = None
        self._reference_state = None
        self._reuse_permitted = True

    @staticmethod
    def _frame(value: Any) -> np.ndarray:
        frame = np.asarray(value)
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2-D or 3-D array")
        if not np.all(np.isfinite(frame)):
            raise ValueError("frame contains non-finite values")
        return frame.astype(np.float32, copy=False)

    @staticmethod
    def _state(value: Any) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32).reshape(-1)
        if state.size == 0:
            raise ValueError("state must not be empty")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains non-finite values")
        return state

    @staticmethod
    def _visual_change(current: np.ndarray, reference: np.ndarray) -> float:
        if current.shape != reference.shape:
            raise ValueError("frame shape changed within an episode")
        scale = 255.0 if max(float(current.max()), float(reference.max())) > 1.5 else 1.0
        return float(np.mean(np.abs(current - reference)) / scale)

    def decide(
        self,
        *,
        frame: Any,
        state: Any,
        execution_event: str | None = None,
    ) -> GroundedSubgoalScheduleDecision:
        image = self._frame(frame)
        proprio = self._state(state)
        event = str(execution_event or "").strip().lower() or None

        if not self._has_plan:
            return GroundedSubgoalScheduleDecision(
                True,
                "task_start",
                self._chunks_since_planner,
                execution_event=event,
            )
        if self.config.mode is GroundedSubgoalScheduleMode.EVERY_CHUNK:
            return GroundedSubgoalScheduleDecision(
                True,
                "every_chunk_baseline",
                self._chunks_since_planner,
                execution_event=event,
            )
        if not self._reuse_permitted:
            return GroundedSubgoalScheduleDecision(
                True,
                "precision_sensitive_grounding",
                self._chunks_since_planner,
                execution_event=event,
            )

        assert self._reference_frame is not None
        assert self._reference_state is not None
        visual_change = self._visual_change(image, self._reference_frame)
        if proprio.shape != self._reference_state.shape:
            raise ValueError("state shape changed within an episode")
        gripper_change = float(abs(proprio[-1] - self._reference_state[-1]))
        metrics = {
            "chunks_since_planner": self._chunks_since_planner,
            "visual_change": visual_change,
            "gripper_change": gripper_change,
            "execution_event": event,
        }

        if event is not None:
            return GroundedSubgoalScheduleDecision(
                True,
                f"execution_event:{event}",
                **metrics,
            )
        if self._chunks_since_planner >= self.config.max_reuse_chunks:
            return GroundedSubgoalScheduleDecision(
                True,
                "reuse_budget_exhausted",
                **metrics,
            )
        if self._chunks_since_planner >= self.config.min_reuse_chunks:
            if gripper_change >= self.config.gripper_change_threshold:
                return GroundedSubgoalScheduleDecision(
                    True,
                    "gripper_state_changed",
                    **metrics,
                )
            if visual_change >= self.config.visual_change_threshold:
                return GroundedSubgoalScheduleDecision(
                    True,
                    "visual_context_changed",
                    **metrics,
                )
        return GroundedSubgoalScheduleDecision(
            False,
            "grounded_subgoal_reuse_admitted",
            **metrics,
        )

    def record_planner_result(
        self,
        *,
        frame: Any,
        state: Any,
        reuse_permitted: bool = True,
    ) -> None:
        self._reference_frame = self._frame(frame).copy()
        self._reference_state = self._state(state).copy()
        self._chunks_since_planner = 0
        self._has_plan = True
        self._reuse_permitted = bool(reuse_permitted)

    def record_chunk_executed(self) -> None:
        if not self._has_plan:
            raise RuntimeError("cannot record reuse before the first Planner result")
        self._chunks_since_planner += 1


@dataclasses.dataclass(frozen=True)
class SemanticScheduleConfig:
    """Guard expensive semantic calls behind events and available slack."""

    cooldown_steps: int = 100
    min_deadline_slack_ms: float = 20.0
    allowed_events: frozenset[str] = frozenset(
        {"perturbation", "recovery_failed", "repeated_failure"}
    )

    def __post_init__(self) -> None:
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if self.min_deadline_slack_ms < 0:
            raise ValueError("min_deadline_slack_ms must be non-negative")
        if not self.allowed_events:
            raise ValueError("allowed_events must not be empty")
        if any(not str(event).strip() for event in self.allowed_events):
            raise ValueError("allowed_events must not contain empty event names")


@dataclasses.dataclass(frozen=True)
class SemanticScheduleDecision:
    invoke: bool
    reason: str
    event: str


class DeadlineAwareSemanticScheduler:
    """Decide whether an event warrants a non-blocking semantic observation."""

    def __init__(self, config: SemanticScheduleConfig | None = None) -> None:
        self.config = config or SemanticScheduleConfig()
        self._last_submission_step: int | None = None

    def reset(self) -> None:
        self._last_submission_step = None

    def decide(
        self,
        *,
        event: str | None,
        timestep: int,
        deadline_slack_ms: float | None,
        request_pending: bool = False,
    ) -> SemanticScheduleDecision:
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        normalized_event = str(event or "").strip().lower()
        if not normalized_event:
            return SemanticScheduleDecision(False, "no semantic event", normalized_event)
        if normalized_event not in self.config.allowed_events:
            return SemanticScheduleDecision(False, "event is not enabled", normalized_event)
        if request_pending:
            return SemanticScheduleDecision(False, "semantic request already pending", normalized_event)
        if (
            self._last_submission_step is not None
            and timestep - self._last_submission_step < self.config.cooldown_steps
        ):
            return SemanticScheduleDecision(False, "semantic cooldown active", normalized_event)
        if (
            deadline_slack_ms is not None
            and float(deadline_slack_ms) < self.config.min_deadline_slack_ms
        ):
            return SemanticScheduleDecision(False, "insufficient deadline slack", normalized_event)
        return SemanticScheduleDecision(True, "event and deadline gate passed", normalized_event)

    def record_submission(self, timestep: int) -> None:
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        self._last_submission_step = int(timestep)


@dataclasses.dataclass(frozen=True)
class SemanticObservationContext:
    episode_id: str | int
    submitted_timestep: int
    event: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    ticket_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.submitted_timestep < 0:
            raise ValueError("submitted_timestep must be non-negative")
        if not self.event.strip():
            raise ValueError("event must not be empty")


@dataclasses.dataclass(frozen=True)
class SemanticObservationResult:
    context: SemanticObservationContext
    response: Any | None
    elapsed_s: float
    error: Exception | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.response is not None


class AsyncSemanticObserver:
    """Run at most one semantic observation outside the robot control path."""

    def __init__(self, observe: Callable[[Mapping[str, Any]], Any]) -> None:
        if not callable(observe):
            raise TypeError("observe must be callable")
        self._observe = observe
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="carve-semantic")
        self._future: Future[tuple[Any | None, float, Exception | None]] | None = None
        self._context: SemanticObservationContext | None = None
        self._closed = False

    @property
    def pending(self) -> bool:
        return self._future is not None

    @property
    def ready(self) -> bool:
        return self._future is not None and self._future.done()

    @property
    def context(self) -> SemanticObservationContext | None:
        return self._context

    def submit(
        self,
        request: Mapping[str, Any],
        context: SemanticObservationContext,
    ) -> str:
        if self._closed:
            raise RuntimeError("semantic observer is closed")
        if self._future is not None:
            raise RuntimeError("a semantic observation is already active")
        if not isinstance(request, Mapping):
            raise TypeError("semantic request must be a mapping")
        self._context = context
        self._future = self._executor.submit(self._run, dict(request))
        return context.ticket_id

    def take(self, *, wait: bool = True) -> SemanticObservationResult | None:
        if self._future is None or self._context is None:
            return None
        if not wait and not self._future.done():
            return None
        response, elapsed_s, error = self._future.result()
        result = SemanticObservationResult(
            context=self._context,
            response=response,
            elapsed_s=elapsed_s,
            error=error,
        )
        self._future = None
        self._context = None
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _run(self, request: Mapping[str, Any]) -> tuple[Any | None, float, Exception | None]:
        started_s = time.perf_counter()
        try:
            response = self._observe(request)
            error = None
        except Exception as exc:  # The control loop records failure without blocking or retrying.
            response = None
            error = exc
        return response, time.perf_counter() - started_s, error

    def __enter__(self) -> "AsyncSemanticObserver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
