"""Capability-aware CARVE runtime and lightweight compute controller."""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable

from .adapter import FallbackMode, PolicyAdapter
from .contracts import ActionChunk, InferenceControls, InferenceRequest, RuntimeTrace


@dataclasses.dataclass(frozen=True)
class ComputeDecision:
    controls: InferenceControls
    risk_bucket: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ComputeControllerConfig:
    low_inference_steps: int = 1
    high_inference_steps: int = 2
    low_risk_max_actions: int = 8
    high_risk_max_actions: int = 2
    high_risk_threshold: float = 0.65
    low_slack_ms: float = 30.0

    def __post_init__(self) -> None:
        if self.low_inference_steps <= 0 or self.high_inference_steps <= 0:
            raise ValueError("inference step budgets must be positive")
        if self.low_risk_max_actions <= 0 or self.high_risk_max_actions <= 0:
            raise ValueError("action budgets must be positive")
        if not 0.0 <= self.high_risk_threshold <= 1.0:
            raise ValueError("high_risk_threshold must be in [0, 1]")


class RiskDeadlineController:
    """Deterministic first-stage controller for auditable experiments."""

    def __init__(self, config: ComputeControllerConfig | None = None) -> None:
        self.config = config or ComputeControllerConfig()

    def decide(
        self,
        *,
        risk_score: float,
        deadline_ms: float | None,
        deadline_slack_ms: float | None,
    ) -> ComputeDecision:
        risk = min(1.0, max(0.0, float(risk_score)))
        low_slack = (
            deadline_slack_ms is not None
            and float(deadline_slack_ms) < self.config.low_slack_ms
        )
        high_risk = risk >= self.config.high_risk_threshold

        if high_risk and not low_slack:
            return ComputeDecision(
                controls=InferenceControls(
                    inference_steps=self.config.high_inference_steps,
                    max_actions=self.config.high_risk_max_actions,
                    deadline_ms=deadline_ms,
                ),
                risk_bucket="high",
                reason="high risk with sufficient deadline slack",
            )
        if high_risk:
            return ComputeDecision(
                controls=InferenceControls(
                    inference_steps=self.config.low_inference_steps,
                    max_actions=self.config.high_risk_max_actions,
                    deadline_ms=deadline_ms,
                ),
                risk_bucket="high",
                reason="high risk but low deadline slack",
            )
        return ComputeDecision(
            controls=InferenceControls(
                inference_steps=self.config.low_inference_steps,
                max_actions=self.config.low_risk_max_actions,
                deadline_ms=deadline_ms,
            ),
            risk_bucket="low",
            reason="low risk fast path",
        )


class CarveRuntime:
    """Execute one policy adapter while recording capability and deadline evidence."""

    def __init__(
        self,
        adapter: PolicyAdapter,
        *,
        fallback_mode: FallbackMode = "graceful",
        trace_capacity: int = 10_000,
        trace_sink: Callable[[RuntimeTrace], None] | None = None,
    ) -> None:
        if trace_capacity <= 0:
            raise ValueError("trace_capacity must be positive")
        self.adapter = adapter
        self.fallback_mode = fallback_mode
        self.trace_sink = trace_sink
        self._traces: deque[RuntimeTrace] = deque(maxlen=trace_capacity)

    @property
    def traces(self) -> tuple[RuntimeTrace, ...]:
        return tuple(self._traces)

    @property
    def last_trace(self) -> RuntimeTrace | None:
        return self._traces[-1] if self._traces else None

    def reset(self, episode_id: str | int | None = None) -> None:
        self.adapter.reset(episode_id)

    def _record_trace(self, trace: RuntimeTrace) -> None:
        self._traces.append(trace)
        if self.trace_sink is not None:
            self.trace_sink(trace)

    def infer(self, request: InferenceRequest) -> ActionChunk:
        started_s = time.perf_counter()
        queue_age_ms = max(0.0, (started_s - request.created_at_s) * 1000.0)
        negotiated = self.adapter.negotiate(
            request.controls,
            fallback_mode=self.fallback_mode,
        )
        applied_request = request.with_controls(negotiated.controls)
        requested = dataclasses.asdict(request.controls)
        applied = dataclasses.asdict(negotiated.controls)
        action_age_value = request.metadata.get("action_age_steps")
        action_age_steps = (
            int(action_age_value) if action_age_value is not None else None
        )
        task_cycle_value = request.metadata.get("task_cycle_latency_ms")
        task_cycle_latency_ms = (
            float(task_cycle_value) if task_cycle_value is not None else None
        )

        try:
            chunk = self.adapter.infer(applied_request)
            chunk = chunk.limited(negotiated.controls.max_actions)
        except Exception as exc:
            runtime_ms = (time.perf_counter() - started_s) * 1000.0
            self._record_trace(
                RuntimeTrace(
                    request_id=request.request_id,
                    adapter_id=self.adapter.adapter_id,
                    policy_family=self.adapter.capabilities.policy_family.value,
                    episode_id=request.episode_id,
                    timestep=request.timestep,
                    requested_controls=requested,
                    applied_controls=applied,
                    dropped_controls=negotiated.dropped,
                    queue_age_ms=queue_age_ms,
                    runtime_latency_ms=runtime_ms,
                    model_latency_ms=0.0,
                    action_count=0,
                    deadline_ms=negotiated.controls.deadline_ms,
                    deadline_miss=bool(
                        negotiated.controls.deadline_ms is not None
                        and runtime_ms > negotiated.controls.deadline_ms
                    ),
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    reaction_latency_ms=queue_age_ms + runtime_ms,
                    task_cycle_latency_ms=task_cycle_latency_ms,
                    action_age_steps=action_age_steps,
                )
            )
            raise

        runtime_ms = (time.perf_counter() - started_s) * 1000.0
        deadline_ms = negotiated.controls.deadline_ms
        trace_metadata = dict(chunk.metadata)
        trace_context = request.metadata.get("trace_context")
        if isinstance(trace_context, Mapping):
            trace_metadata["controller"] = dict(trace_context)
        if request.agentic is not None:
            trace_metadata["agentic_event"] = dict(request.agentic)
        stage_values = chunk.metadata.get("stage_latencies_ms", {})
        stage_latencies_ms = (
            {str(name): float(value) for name, value in stage_values.items()}
            if isinstance(stage_values, Mapping)
            else {}
        )
        trace = RuntimeTrace(
            request_id=request.request_id,
            adapter_id=self.adapter.adapter_id,
            policy_family=self.adapter.capabilities.policy_family.value,
            episode_id=request.episode_id,
            timestep=request.timestep,
            requested_controls=requested,
            applied_controls=applied,
            dropped_controls=negotiated.dropped,
            queue_age_ms=queue_age_ms,
            runtime_latency_ms=runtime_ms,
            model_latency_ms=chunk.model_latency_ms,
            action_count=chunk.action_count,
            deadline_ms=deadline_ms,
            deadline_miss=bool(deadline_ms is not None and runtime_ms > deadline_ms),
            success=True,
            metadata=trace_metadata,
            reaction_latency_ms=queue_age_ms + runtime_ms,
            task_cycle_latency_ms=task_cycle_latency_ms,
            action_age_steps=action_age_steps,
            stage_latencies_ms=stage_latencies_ms,
        )
        self._record_trace(trace)
        chunk.metadata.setdefault("runtime_trace", trace.to_dict())
        return chunk

    def trace_dicts(self) -> list[Mapping[str, Any]]:
        return [trace.to_dict() for trace in self._traces]
