"""Deadline-aware asynchronous semantic observation for CARVE."""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


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
