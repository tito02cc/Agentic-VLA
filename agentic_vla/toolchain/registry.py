"""Permissioned dispatch for CARVE's embodied-agent tool ecosystem."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import (
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
    validate_tool_input,
)


ToolHandler = Callable[[Mapping[str, Any], ToolExecutionContext], Mapping[str, Any]]
ToolTraceSink = Callable[[ToolResult], None]


class EmbodiedToolRegistry:
    """Register and execute typed tools behind CARVE safety and budget gates."""

    def __init__(self, *, trace_sink: ToolTraceSink | None = None) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}
        self._calls: dict[tuple[str, str], int] = defaultdict(int)
        self._operation_lock = threading.Lock()
        self._trace_sink = trace_sink

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate CARVE tool: {spec.name}")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self._tools[spec.name] = (spec, handler)

    def specs(self, *, allowed_tools: tuple[str, ...] | None = None) -> tuple[dict[str, Any], ...]:
        allowed = None if allowed_tools is None else set(allowed_tools)
        return tuple(
            spec.to_dict()
            for name, (spec, _) in sorted(self._tools.items())
            if allowed is None or name in allowed
        )

    def reset_episode(self, episode_id: str | int) -> None:
        prefix = str(episode_id)
        for key in tuple(self._calls):
            if key[0] == prefix:
                del self._calls[key]

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        started = time.perf_counter()
        if call.episode_id != context.episode_id:
            return self._reject(call, started, "tool call belongs to another episode")
        if call.timestep != context.timestep:
            return self._reject(call, started, "tool call is stale for this timestep")
        entry = self._tools.get(call.name)
        if entry is None:
            return self._reject(call, started, f"unknown tool: {call.name}")
        spec, handler = entry
        if spec.name not in context.allowed_tools:
            return self._reject(call, started, f"tool is not allowed: {spec.name}")
        if spec.requires_safe_boundary and not context.at_safe_boundary:
            return self._reject(call, started, "tool requires a safe execution boundary")
        counter_key = (str(context.episode_id), spec.name)
        if (
            spec.max_calls_per_episode is not None
            and self._calls[counter_key] >= spec.max_calls_per_episode
        ):
            return self._reject(call, started, "tool call budget exhausted")
        try:
            validate_tool_input(call.arguments, spec.input_schema)
        except (TypeError, ValueError) as exc:
            return self._reject(call, started, str(exc))
        if not self._operation_lock.acquire(blocking=False):
            return self._reject(call, started, "another embodied tool is active")
        try:
            self._calls[counter_key] += 1
            output = handler(call.arguments, context)
            if not isinstance(output, Mapping):
                raise TypeError("tool handler must return a mapping")
            result = ToolResult(
                call_id=call.call_id,
                name=spec.name,
                accepted=True,
                output=dict(output),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            result = ToolResult(
                call_id=call.call_id,
                name=spec.name,
                accepted=False,
                output={},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
            )
        finally:
            self._operation_lock.release()
        self._emit(result)
        return result

    def _reject(self, call: ToolCall, started: float, error: str) -> ToolResult:
        result = ToolResult(
            call_id=call.call_id,
            name=call.name,
            accepted=False,
            output={},
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )
        self._emit(result)
        return result

    def _emit(self, result: ToolResult) -> None:
        if self._trace_sink is not None:
            self._trace_sink(result)
