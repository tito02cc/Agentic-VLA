"""Backend-neutral warm inference benchmarking for prepared CARVE profiles."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from agentic_vla.runtime import CarveRuntime, InferenceRequest

from .contracts import BenchmarkReport
from .plugins import PreparedPolicy


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    warmup_calls: int = 2
    measured_calls: int = 20
    record_samples: bool = False

    def __post_init__(self) -> None:
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be non-negative")
        if self.measured_calls <= 0:
            raise ValueError("measured_calls must be positive")


class BenchmarkRunner:
    """Measure the same CARVE runtime path used during deployment."""

    def __init__(
        self,
        prepared: PreparedPolicy,
        *,
        config: BenchmarkConfig | None = None,
        synchronize: Callable[[], None] | None = None,
        peak_vram_gb: Callable[[], float] | None = None,
    ) -> None:
        self.prepared = prepared
        self.config = config or BenchmarkConfig()
        self.synchronize = synchronize
        self.peak_vram_gb = peak_vram_gb

    def run(self, request_factory: Callable[[int], InferenceRequest]) -> BenchmarkReport:
        runtime = CarveRuntime(self.prepared.adapter, fallback_mode="strict")
        total_calls = self.config.warmup_calls + self.config.measured_calls
        runtime_latencies: list[float] = []
        model_latencies: list[float] = []
        action_counts: list[int] = []
        deadline_misses: list[bool] = []

        for index in range(total_calls):
            request = request_factory(index)
            if not isinstance(request, InferenceRequest):
                raise TypeError("request_factory must return InferenceRequest")
            request = request.with_controls(self.prepared.apply_profile(request.controls))
            if self.synchronize is not None:
                self.synchronize()
            runtime.infer(request)
            if self.synchronize is not None:
                self.synchronize()
            if index < self.config.warmup_calls:
                continue
            trace = runtime.last_trace
            if trace is None:
                raise RuntimeError("runtime did not produce a trace")
            runtime_latencies.append(trace.runtime_latency_ms)
            model_latencies.append(trace.model_latency_ms)
            action_counts.append(trace.action_count)
            deadline_misses.append(trace.deadline_miss)

        metadata = {
            "model_id": self.prepared.model_id,
            "backend_id": self.prepared.backend_id,
            "profile_id": self.prepared.profile.profile_id,
            "warmup_calls": self.config.warmup_calls,
            "backend_metadata": dict(self.prepared.metadata),
        }
        if self.config.record_samples:
            metadata["latency_samples"] = {
                "runtime_ms": runtime_latencies,
                "model_ms": model_latencies,
                "deadline_miss": deadline_misses,
            }

        return BenchmarkReport(
            samples=len(runtime_latencies),
            runtime_p50_ms=float(np.percentile(runtime_latencies, 50)),
            runtime_p95_ms=float(np.percentile(runtime_latencies, 95)),
            runtime_p99_ms=float(np.percentile(runtime_latencies, 99)),
            model_p50_ms=float(np.percentile(model_latencies, 50)),
            model_p95_ms=float(np.percentile(model_latencies, 95)),
            mean_action_count=float(np.mean(action_counts)),
            deadline_miss_rate=float(np.mean(deadline_misses)),
            peak_vram_gb=None if self.peak_vram_gb is None else float(self.peak_vram_gb()),
            metadata=metadata,
        )
