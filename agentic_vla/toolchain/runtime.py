"""Bind core tools and optional read-only inspection to deployment-owned handlers."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agentic_vla.runtime.agent import AgentIntent, HighLevelAgentDecision

from .catalog import core_tool_specs
from .contracts import (
    PrimitiveOutcome,
    PrimitiveStatus,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    VerificationReport,
    reject_direct_action_fields,
)
from .inspection import VisualEvidenceStore
from .object_memory import ObjectEvidenceMemory
from .registry import EmbodiedToolRegistry
from .workspace import RunWorkspace


ObservationHandler = Callable[[ToolExecutionContext], Mapping[str, Any]]
MemoryHandler = Callable[[str, int, ToolExecutionContext], Mapping[str, Any]]
VerificationHandler = Callable[[str, ToolExecutionContext], VerificationReport]
SafeHoldHandler = Callable[[str, ToolExecutionContext], Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class PrimitiveExecutionReport:
    """Executor-local result before the runtime attaches tool-call identity."""

    status: PrimitiveStatus | str
    started_timestep: int
    ended_timestep: int
    expected_outcome: str = ""
    observed_outcome: str = ""
    requires_semantic_check: bool = False
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            status = (
                self.status
                if isinstance(self.status, PrimitiveStatus)
                else PrimitiveStatus(str(self.status).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("unsupported primitive execution status") from exc
        if self.started_timestep < 0 or self.ended_timestep < self.started_timestep:
            raise ValueError("primitive report timesteps must be ordered")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("primitive report metadata must be a mapping")
        reject_direct_action_fields(self.metadata, path="primitive_report.metadata")
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "started_timestep": self.started_timestep,
            "ended_timestep": self.ended_timestep,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "requires_semantic_check": self.requires_semantic_check,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PrimitiveExecutionReport":
        return cls(**dict(values))


VlaPrimitiveHandler = Callable[
    [str, ToolExecutionContext], PrimitiveExecutionReport
]
SkillPrimitiveHandler = Callable[
    [str, Mapping[str, Any], ToolExecutionContext], PrimitiveExecutionReport
]


@dataclasses.dataclass(frozen=True)
class EmbodiedToolBindings:
    """Deployment-provided implementations behind the stable tool vocabulary."""

    observe: ObservationHandler
    retrieve_memory: MemoryHandler
    vla_act: VlaPrimitiveHandler
    run_skill: SkillPrimitiveHandler
    verify: VerificationHandler
    safe_hold: SafeHoldHandler


class CanonicalToolRuntime:
    """Execute all planner tools through one permissioned and auditable path."""

    def __init__(
        self,
        *,
        bindings: EmbodiedToolBindings,
        workspace: RunWorkspace,
        tool_call_budgets: Mapping[str, int] | None = None,
        visual_evidence: VisualEvidenceStore | None = None,
        object_evidence: ObjectEvidenceMemory | None = None,
    ) -> None:
        self.bindings = bindings
        self.workspace = workspace
        self.registry = EmbodiedToolRegistry()
        budgets = dict(tool_call_budgets or {})
        catalog_names = {spec.name for spec in core_tool_specs()}
        if visual_evidence is not None:
            if not isinstance(visual_evidence, VisualEvidenceStore):
                raise TypeError("visual_evidence must be a VisualEvidenceStore")
            catalog_names.add(visual_evidence.tool_spec.name)
        if object_evidence is not None:
            if not isinstance(object_evidence, ObjectEvidenceMemory):
                raise TypeError("object_evidence must be an ObjectEvidenceMemory")
            catalog_names.add(object_evidence.tool_spec.name)
        unknown = sorted(set(budgets) - catalog_names)
        if unknown:
            raise ValueError(f"tool_call_budgets contains unknown tools: {unknown}")
        if any(int(value) < 0 for value in budgets.values()):
            raise ValueError("tool call budgets must be non-negative")
        handlers = {
            "observe": self._observe,
            "retrieve_memory": self._retrieve_memory,
            "vla_act": self._vla_act,
            "run_skill": self._run_skill,
            "verify": self._verify,
            "safe_hold": self._safe_hold,
            "finish": self._finish,
        }
        for spec in core_tool_specs():
            if spec.name in budgets:
                spec = dataclasses.replace(
                    spec, max_calls_per_episode=int(budgets[spec.name])
                )
            self.registry.register(spec, handlers[spec.name])
        if visual_evidence is not None:
            spec = visual_evidence.tool_spec
            if spec.name in budgets:
                spec = dataclasses.replace(
                    spec, max_calls_per_episode=min(int(budgets[spec.name]), visual_evidence.max_inspections)
                )
            self.registry.register(spec, visual_evidence.inspect_region)
        if object_evidence is not None:
            spec = object_evidence.tool_spec
            if spec.name in budgets:
                spec = dataclasses.replace(spec, max_calls_per_episode=min(int(budgets[spec.name]), spec.max_calls_per_episode))
            self.registry.register(spec, object_evidence.read_tool)

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ToolExecutionContext,
        call_id: str | None = None,
    ) -> ToolResult:
        call = ToolCall(
            call_id=call_id or uuid.uuid4().hex,
            name=name,
            arguments=dict(arguments),
            episode_id=context.episode_id,
            timestep=context.timestep,
        )
        result = self.registry.execute(call, context)
        if result.accepted and name in {"vla_act", "run_skill"}:
            report = PrimitiveExecutionReport.from_dict(result.output)
            primitive_name = (
                "vla_act" if name == "vla_act" else str(arguments["skill_id"])
            )
            outcome = PrimitiveOutcome(
                call_id=call.call_id,
                primitive_name=primitive_name,
                status=report.status,
                episode_id=context.episode_id,
                started_timestep=report.started_timestep,
                ended_timestep=report.ended_timestep,
                expected_outcome=report.expected_outcome,
                observed_outcome=report.observed_outcome,
                requires_semantic_check=report.requires_semantic_check,
                metadata=report.metadata,
            )
            result = dataclasses.replace(
                result, output={"primitive_outcome": outcome.to_planner_dict()}
            )
        self.workspace.append_recipe(call, result)
        self.workspace.append_event(
            "tool_result",
            {
                "call_id": result.call_id,
                "tool": result.name,
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "output": dict(result.output),
                "error": result.error,
            },
            source="toolchain",
        )
        if result.accepted and name == "finish":
            self.workspace.finish(
                status=str(arguments["status"]), summary=str(arguments["summary"])
            )
        return result

    def invoke_decision(
        self,
        decision: HighLevelAgentDecision,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult | None:
        """Map a guarded model decision to the same seven-tool boundary."""

        if decision.intent is AgentIntent.CONTINUE:
            return None
        if decision.intent is AgentIntent.VLA_ACT:
            return self.invoke(
                "vla_act",
                {
                    "instruction": decision.vla_instruction,
                    "expected_outcome": decision.expected_outcome,
                },
                context=context,
            )
        if decision.intent is AgentIntent.RUN_SKILL:
            return self.invoke(
                "run_skill",
                {
                    "skill_id": decision.skill_id,
                    "skill_args": dict(decision.skill_args),
                    "expected_outcome": decision.expected_outcome,
                },
                context=context,
            )
        return self.invoke(
            "safe_hold", {"reason": decision.rationale}, context=context
        )

    @staticmethod
    def primitive_outcome(
        result: ToolResult, *, episode_id: str | int
    ) -> PrimitiveOutcome | None:
        payload = result.output.get("primitive_outcome")
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("primitive_outcome must be a mapping")
        return PrimitiveOutcome(
            call_id=str(payload["call_id"]),
            primitive_name=str(payload["primitive_name"]),
            status=str(payload["status"]),
            episode_id=episode_id,
            started_timestep=int(payload["started_timestep"]),
            ended_timestep=int(payload["ended_timestep"]),
            expected_outcome=str(payload.get("expected_outcome", "")),
            observed_outcome=str(payload.get("observed_outcome", "")),
            requires_semantic_check=bool(
                payload.get("requires_semantic_check", False)
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def _observe(
        self, _arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        return self._bounded_output(self.bindings.observe(context))

    def _retrieve_memory(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        limit = int(arguments.get("limit", 8))
        if limit <= 0:
            raise ValueError("memory retrieval limit must be positive")
        return self._bounded_output(
            self.bindings.retrieve_memory(str(arguments["query"]), limit, context)
        )

    def _vla_act(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        report = self.bindings.vla_act(str(arguments["instruction"]), context)
        expected = str(arguments.get("expected_outcome", "")).strip()
        if expected:
            report = dataclasses.replace(
                report,
                expected_outcome=expected,
            )
        return report.to_dict()

    def _run_skill(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        report = self.bindings.run_skill(
            str(arguments["skill_id"]), dict(arguments["skill_args"]), context
        )
        expected = str(arguments.get("expected_outcome", "")).strip()
        if expected:
            report = dataclasses.replace(
                report,
                expected_outcome=expected,
            )
        return report.to_dict()

    def _verify(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        report = self.bindings.verify(
            str(arguments["expected_outcome"]), context
        )
        if not isinstance(report, VerificationReport):
            raise TypeError("verify binding must return VerificationReport")
        return report.to_dict()

    def _safe_hold(
        self, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        return self._bounded_output(
            self.bindings.safe_hold(str(arguments["reason"]), context)
        )

    @staticmethod
    def _finish(
        arguments: Mapping[str, Any], _context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        return {"status": str(arguments["status"]), "summary": str(arguments["summary"])}

    @staticmethod
    def _bounded_output(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("tool binding must return a mapping")
        reject_direct_action_fields(value, path="tool_output")
        return dict(value)
