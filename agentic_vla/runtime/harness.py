"""Reusable multi-rate Agentic harness orchestration for CARVE."""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .agent import (
    AgentIntent,
    AsyncGuardedHighLevelAgent,
    AsyncHighLevelPlannerResult,
    HighLevelAgentContext,
    HighLevelPlannerTicket,
)
from .controller import ExecutionMode, JointDecision, JointRecoveryComputeController
from .knowledge import (
    AgenticKnowledgeProvider,
    SemanticSceneGraph,
    parse_semantic_scene_graph,
)
from .monitor import RiskAssessment
from .recovery import FailureEpisodeRecord, FailureMemory
from agentic_vla.toolchain import PrimitiveOutcome, PrimitiveStatus


class HarnessState(str, enum.Enum):
    """Explicit execution states shared by simulator and robot adapters."""

    EXECUTE_FAST = "execute_fast"
    VERIFY = "verify"
    RECOVER = "recover"
    PLAN_AT_SAFE_BOUNDARY = "plan_at_safe_boundary"
    SAFE_HOLD = "safe_hold"
    STOP = "stop"


class TaskStartPolicy(str, enum.Enum):
    """Complete semantics for optional task-start VLM reasoning."""

    EVENT_ONLY = "event_only"
    STARTUP_SHADOW = "startup_shadow"
    STARTUP_WAIT = "startup_wait"


class PrimitiveBoundaryPolicy(str, enum.Enum):
    """When semantic planning is requested after a bounded primitive returns."""

    EVENT_ONLY = "event_only"
    SELECTIVE = "selective"
    EVERY_PRIMITIVE = "every_primitive"


@dataclasses.dataclass
class HarnessCounters:
    """Independent episode counters; no counter aliases another concept."""

    failure_streak: int = 0
    recovery_attempts: int = 0
    planner_calls: int = 0
    semantic_retries: int = 0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if min(
            self.failure_streak,
            self.recovery_attempts,
            self.planner_calls,
            self.semantic_retries,
        ) < 0:
            raise ValueError("harness counters must be non-negative")

    def record_failure(self) -> None:
        self.failure_streak += 1

    def clear_failures(self) -> None:
        self.failure_streak = 0

    def record_recovery_attempt(self) -> None:
        self.recovery_attempts += 1

    def record_planner_call(self) -> None:
        self.planner_calls += 1

    def record_semantic_retry(self) -> None:
        self.semantic_retries += 1

    def to_dict(self) -> dict[str, int]:
        self._validate()
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SafeHoldRequest:
    """One adapter-level request to keep the robot stationary."""

    episode_id: str | int
    reason: str
    heartbeat_hz: float = 10.0
    timeout_s: float = 10.0
    entered_at_s: float = dataclasses.field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("safe-hold reason must not be empty")
        if self.heartbeat_hz <= 0 or self.timeout_s <= 0:
            raise ValueError("safe-hold heartbeat and timeout must be positive")


class SafeHoldAdapter(Protocol):
    """Robot-specific stationary-command and heartbeat contract."""

    def enter(self, request: SafeHoldRequest) -> None: ...

    def heartbeat(self, request: SafeHoldRequest) -> bool: ...

    def release(self, request: SafeHoldRequest, *, reason: str) -> None: ...

    def stop(self, request: SafeHoldRequest, *, reason: str) -> None: ...


class PausedExecutionSafeHoldAdapter:
    """Safe-boundary adapter for simulators whose world is paused while waiting."""

    def __init__(self) -> None:
        self.active: SafeHoldRequest | None = None
        self.events: list[dict[str, object]] = []

    def enter(self, request: SafeHoldRequest) -> None:
        if self.active is not None:
            raise RuntimeError("safe hold is already active")
        self.active = request
        self.events.append(
            {
                "event": "enter",
                "episode_id": request.episode_id,
                "reason": request.reason,
                "at_s": request.entered_at_s,
            }
        )

    def heartbeat(self, request: SafeHoldRequest) -> bool:
        if self.active != request:
            return False
        elapsed = time.perf_counter() - request.entered_at_s
        alive = elapsed <= request.timeout_s
        self.events.append(
            {
                "event": "heartbeat",
                "episode_id": request.episode_id,
                "elapsed_s": elapsed,
                "alive": alive,
            }
        )
        return alive

    def release(self, request: SafeHoldRequest, *, reason: str) -> None:
        self._finish(request, event="release", reason=reason)

    def stop(self, request: SafeHoldRequest, *, reason: str) -> None:
        self._finish(request, event="stop", reason=reason)

    def _finish(self, request: SafeHoldRequest, *, event: str, reason: str) -> None:
        if self.active != request:
            raise RuntimeError("safe-hold request is not active")
        self.events.append(
            {
                "event": event,
                "episode_id": request.episode_id,
                "reason": str(reason),
                "at_s": time.perf_counter(),
            }
        )
        self.active = None


@dataclasses.dataclass(frozen=True)
class HarnessTransition:
    """One auditable state-machine transition."""

    state: HarnessState
    joint: JointDecision | None = None
    planner: AsyncHighLevelPlannerResult | None = None
    ticket: HighLevelPlannerTicket | None = None
    decision_applied: bool = False
    stale: bool = False
    timed_out: bool = False
    reason: str = ""


PlannerFactory = Callable[[str | int], AsyncGuardedHighLevelAgent]


class AsyncAgenticHarnessController:
    """Episode-scoped orchestration around a deterministic recovery controller.

    The controller owns lifecycle and safety semantics, but never invokes a VLA
    or executes a skill. Callers apply the returned typed transition through
    their policy and skill adapters.
    """

    def __init__(
        self,
        controller: JointRecoveryComputeController,
        *,
        planner_factory: PlannerFactory | None = None,
        task_start_policy: TaskStartPolicy | str = TaskStartPolicy.EVENT_ONLY,
        safe_hold_adapter: SafeHoldAdapter | None = None,
        failure_memory: FailureMemory | None = None,
        knowledge_provider: AgenticKnowledgeProvider | None = None,
        memory_retrieval_limit: int = 8,
        planner_cooldown_steps: int = 50,
        semantic_retry_budget: int = 2,
        primitive_boundary_policy: PrimitiveBoundaryPolicy | str = PrimitiveBoundaryPolicy.EVENT_ONLY,
        safe_hold_timeout_s: float = 10.0,
        safe_hold_heartbeat_hz: float = 10.0,
    ) -> None:
        if not isinstance(controller, JointRecoveryComputeController):
            raise TypeError("controller must be a JointRecoveryComputeController")
        if safe_hold_timeout_s <= 0 or safe_hold_heartbeat_hz <= 0:
            raise ValueError("safe-hold settings must be positive")
        if memory_retrieval_limit <= 0:
            raise ValueError("memory_retrieval_limit must be positive")
        if planner_cooldown_steps < 0 or semantic_retry_budget < 0:
            raise ValueError("planner cooldown and semantic retry budget must be non-negative")
        self.controller = controller
        self.planner_factory = planner_factory
        self.task_start_policy = TaskStartPolicy(task_start_policy)
        self.safe_hold_adapter = safe_hold_adapter or PausedExecutionSafeHoldAdapter()
        self.failure_memory = (
            failure_memory if failure_memory is not None else FailureMemory()
        )
        self.knowledge_provider = knowledge_provider
        self.memory_retrieval_limit = int(memory_retrieval_limit)
        self.planner_cooldown_steps = int(planner_cooldown_steps)
        self.semantic_retry_budget = int(semantic_retry_budget)
        self.primitive_boundary_policy = PrimitiveBoundaryPolicy(primitive_boundary_policy)
        self.safe_hold_timeout_s = float(safe_hold_timeout_s)
        self.safe_hold_heartbeat_hz = float(safe_hold_heartbeat_hz)
        self.state = HarnessState.STOP
        self.counters = HarnessCounters()
        self._episode_id: str | int | None = None
        self._generation = 0
        self._planner: AsyncGuardedHighLevelAgent | None = None
        self._purpose: str | None = None
        self._hold: SafeHoldRequest | None = None
        self._scene_graph: SemanticSceneGraph | None = None
        self._last_planner_submission_step: int | None = None

    @property
    def episode_id(self) -> str | int | None:
        return self._episode_id

    @property
    def planner_pending(self) -> bool:
        return self._planner is not None and self._planner.pending

    @property
    def planner_ticket(self) -> HighLevelPlannerTicket | None:
        return None if self._planner is None else self._planner.ticket

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def scene_graph(self) -> SemanticSceneGraph | None:
        return self._scene_graph

    def start_episode(
        self,
        episode_id: str | int,
        *,
        task_start_context: HighLevelAgentContext | None = None,
    ) -> HarnessTransition:
        self.close_episode(reason="episode_replaced")
        self._generation += 1
        self._episode_id = episode_id
        self.counters = HarnessCounters()
        self._scene_graph = None
        self._last_planner_submission_step = None
        self._planner = (
            self.planner_factory(episode_id) if self.planner_factory is not None else None
        )
        self.state = HarnessState.EXECUTE_FAST

        if self.task_start_policy is TaskStartPolicy.EVENT_ONLY:
            return HarnessTransition(
                state=self.state,
                reason="task-start planning disabled; event-triggered planner only",
            )
        if task_start_context is None:
            raise ValueError("task_start_context is required by the selected task-start policy")
        self._require_current_context(task_start_context)
        ticket = self._submit(task_start_context, purpose=self.task_start_policy.value)
        if self.task_start_policy is TaskStartPolicy.STARTUP_WAIT:
            self._enter_hold("awaiting task-start semantic plan")
            self.state = HarnessState.PLAN_AT_SAFE_BOUNDARY
        return HarnessTransition(
            state=self.state,
            ticket=ticket,
            reason=f"submitted {self.task_start_policy.value}",
        )

    def route(
        self,
        risk: RiskAssessment,
        *,
        context: HighLevelAgentContext,
        deadline_ms: float | None,
        deadline_slack_ms: float | None,
        cached_actions: int = 0,
    ) -> HarnessTransition:
        self._require_current_context(context)
        joint = self.controller.decide(
            risk,
            deadline_ms=deadline_ms,
            deadline_slack_ms=deadline_slack_ms,
            cached_actions=cached_actions,
            repeated_failures=self.counters.failure_streak,
            recovery_attempts=self.counters.recovery_attempts,
            planner_available=self._planner is not None,
        )
        if joint.mode is ExecutionMode.RECOVERY:
            self.state = HarnessState.RECOVER
        elif joint.mode is ExecutionMode.PLANNER:
            self.state = HarnessState.PLAN_AT_SAFE_BOUNDARY
            if self._semantic_retry_budget_exhausted():
                self._enter_hold("semantic retry budget exhausted")
                self.state = HarnessState.SAFE_HOLD
                return HarnessTransition(
                    state=self.state,
                    joint=joint,
                    reason="semantic retry budget exhausted",
                )
            if self._planner_cooldown_active(context.timestep):
                self.state = HarnessState.EXECUTE_FAST
                return HarnessTransition(
                    state=self.state,
                    joint=dataclasses.replace(
                        joint,
                        mode=ExecutionMode.ACCURATE_VLA,
                        reason="semantic planner cooldown active",
                    ),
                    reason="semantic planner cooldown active",
                )
            if self.planner_pending:
                return HarnessTransition(
                    state=self.state,
                    joint=joint,
                    ticket=self.planner_ticket,
                    reason="planner request already pending",
                )
            ticket = self._submit(context, purpose="event_escalation")
            self.counters.record_semantic_retry()
            self._enter_hold(joint.reason)
            return HarnessTransition(
                state=self.state,
                joint=joint,
                ticket=ticket,
                reason=joint.reason,
            )
        elif joint.mode is ExecutionMode.SAFE_STOP:
            self._enter_hold(joint.reason)
            self.state = HarnessState.SAFE_HOLD
        else:
            self.state = (
                HarnessState.VERIFY
                if joint.request_verification
                else HarnessState.EXECUTE_FAST
            )
        return HarnessTransition(state=self.state, joint=joint, reason=joint.reason)

    def poll_planner(self) -> HarnessTransition | None:
        if self._planner is None:
            return None
        completed = self._planner.take(wait=False)
        if completed is None:
            return None
        return self._resolve_planner(completed)

    def request_planner(
        self,
        context: HighLevelAgentContext,
        *,
        purpose: str,
        reason: str,
        respect_cooldown: bool = True,
        count_as_retry: bool = True,
    ) -> HarnessTransition:
        """Submit one explicit semantic check at an adapter-declared safe boundary."""

        if not purpose.strip() or not reason.strip():
            raise ValueError("planner purpose and reason must not be empty")
        self._require_current_context(context)
        if self._planner is None:
            raise RuntimeError("no high-level planner is configured")
        if count_as_retry and self._semantic_retry_budget_exhausted():
            self._enter_hold("semantic retry budget exhausted")
            self.state = HarnessState.SAFE_HOLD
            return HarnessTransition(
                state=self.state,
                reason="semantic retry budget exhausted",
            )
        if respect_cooldown and self._planner_cooldown_active(context.timestep):
            return HarnessTransition(
                state=HarnessState.EXECUTE_FAST,
                reason="semantic planner cooldown active",
            )
        if self.planner_pending:
            return HarnessTransition(
                state=HarnessState.PLAN_AT_SAFE_BOUNDARY,
                ticket=self.planner_ticket,
                reason="planner request already pending",
            )
        ticket = self._submit(context, purpose=purpose)
        if count_as_retry:
            self.counters.record_semantic_retry()
        self._enter_hold(reason)
        self.state = HarnessState.PLAN_AT_SAFE_BOUNDARY
        return HarnessTransition(
            state=self.state,
            ticket=ticket,
            reason=reason,
        )

    def record_primitive_boundary(
        self,
        context: HighLevelAgentContext,
        outcome: PrimitiveOutcome,
    ) -> HarnessTransition:
        """Optionally ask the VLM to verify or replan when a primitive finishes."""

        self._require_current_context(context)
        if outcome.episode_id != self._episode_id:
            raise ValueError("primitive outcome does not match the active episode")
        if outcome.ended_timestep != context.timestep:
            raise ValueError("primitive outcome and planner context timesteps differ")

        should_plan = outcome.abnormal
        if self.primitive_boundary_policy is PrimitiveBoundaryPolicy.SELECTIVE:
            should_plan = should_plan or outcome.requires_semantic_check
        elif self.primitive_boundary_policy is PrimitiveBoundaryPolicy.EVERY_PRIMITIVE:
            should_plan = True

        if not should_plan:
            self.state = HarnessState.EXECUTE_FAST
            return HarnessTransition(
                state=self.state,
                reason="primitive boundary does not require semantic planning",
            )
        if self._planner is None:
            self.state = HarnessState.SAFE_HOLD if outcome.abnormal else HarnessState.EXECUTE_FAST
            if outcome.abnormal:
                self._enter_hold("abnormal primitive completed without a semantic planner")
            return HarnessTransition(
                state=self.state,
                reason="no high-level planner is configured for primitive boundary",
            )

        trigger = (
            "primitive_failed"
            if outcome.status is PrimitiveStatus.FAILED
            else "primitive_timeout"
            if outcome.status is PrimitiveStatus.TIMEOUT
            else "primitive_interrupted"
            if outcome.status is PrimitiveStatus.INTERRUPTED
            else "primitive_completed"
        )
        grounded = dataclasses.replace(
            context,
            trigger=trigger,
            last_primitive=outcome.to_planner_dict(),
        )
        return self.request_planner(
            grounded,
            purpose="primitive_boundary_verification",
            reason=f"verify {outcome.primitive_name} after {outcome.status.value}",
            respect_cooldown=False,
        )

    def await_planner(self, *, timeout_s: float) -> HarnessTransition:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._planner is None or not self._planner.pending:
            raise RuntimeError("no planner request is pending")
        ticket = self._planner.ticket
        completed = self._planner.take(wait=True, timeout_s=timeout_s)
        if completed is None:
            self._enter_hold("planner boundary timeout")
            self.state = HarnessState.SAFE_HOLD
            return HarnessTransition(
                state=self.state,
                ticket=ticket,
                timed_out=True,
                reason="planner boundary timeout",
            )
        return self._resolve_planner(completed)

    def record_failure(self) -> None:
        self.counters.record_failure()

    def clear_failures(self) -> None:
        self.counters.clear_failures()

    def record_recovery_attempt(self) -> None:
        self.counters.record_recovery_attempt()

    def record_semantic_retry(self) -> None:
        self.counters.record_semantic_retry()

    def record_failure_evidence(self, record: FailureEpisodeRecord) -> None:
        """Store typed evidence for later planner context, never for execution."""

        if self._episode_id is None:
            raise RuntimeError("start_episode must be called first")
        if record.episode_id != self._episode_id:
            raise ValueError("failure record does not match the active harness episode")
        self.failure_memory.record(record)

    def hold_heartbeat(self) -> bool:
        if self._hold is None:
            return False
        alive = self.safe_hold_adapter.heartbeat(self._hold)
        if not alive:
            self.safe_hold_adapter.stop(self._hold, reason="safe-hold timeout")
            self._hold = None
            self.state = HarnessState.STOP
        return alive

    def stop(self, *, reason: str) -> None:
        if self._hold is None and self._episode_id is not None:
            self._enter_hold(reason)
        if self._hold is not None:
            self.safe_hold_adapter.stop(self._hold, reason=reason)
            self._hold = None
        self.state = HarnessState.STOP

    def close_episode(self, *, reason: str = "episode_end") -> None:
        if self._planner is not None:
            self._planner.close(wait=False)
            self._planner = None
        if self._hold is not None:
            self.safe_hold_adapter.stop(self._hold, reason=reason)
            self._hold = None
        self._purpose = None
        self._episode_id = None
        self.state = HarnessState.STOP

    def close(self) -> None:
        self.close_episode(reason="harness_closed")

    def _submit(
        self,
        context: HighLevelAgentContext,
        *,
        purpose: str,
    ) -> HighLevelPlannerTicket:
        if self._planner is None:
            raise RuntimeError("no high-level planner is configured")
        self._require_current_context(context)
        context = self._with_agentic_knowledge(self._with_failure_memory(context))
        ticket = self._planner.submit(context)
        self._last_planner_submission_step = int(context.timestep)
        self._purpose = purpose
        self.counters.record_planner_call()
        return ticket

    def _with_failure_memory(
        self,
        context: HighLevelAgentContext,
    ) -> HighLevelAgentContext:
        failure_type = str(context.risk.get("event") or context.trigger)
        records = self.failure_memory.retrieve(
            context_fingerprint=context.memory_context_fingerprint or None,
            failure_type=failure_type or None,
            deployment_profile_id=context.deployment_profile_id or None,
            limit=self.memory_retrieval_limit,
        )
        if not records:
            return context
        structured = context.memory_records + tuple(record.to_dict() for record in records)
        return dataclasses.replace(context, memory_records=structured)

    def _planner_cooldown_active(self, timestep: int) -> bool:
        return bool(
            self._last_planner_submission_step is not None
            and timestep - self._last_planner_submission_step
            < self.planner_cooldown_steps
        )

    def _semantic_retry_budget_exhausted(self) -> bool:
        return self.counters.semantic_retries >= self.semantic_retry_budget

    def _with_agentic_knowledge(
        self,
        context: HighLevelAgentContext,
    ) -> HighLevelAgentContext:
        if self.knowledge_provider is None:
            return context
        graph: SemanticSceneGraph | Mapping[str, Any] | None
        graph = context.scene_graph or self._scene_graph
        bundle = self.knowledge_provider.build(
            task_instruction=context.task_instruction,
            scene_graph=graph,
            failure_type=str(context.risk.get("event") or context.trigger),
            timestep=context.timestep,
        )
        return dataclasses.replace(
            context,
            scene_graph=(
                {}
                if bundle.scene_graph is None
                else bundle.scene_graph.to_dict()
            ),
            affordance_retrievals=(
                context.affordance_retrievals + bundle.affordance_retrievals
            ),
            procedural_retrievals=(
                context.procedural_retrievals + bundle.procedural_retrievals
            ),
        )

    def _resolve_planner(
        self,
        completed: AsyncHighLevelPlannerResult,
    ) -> HarnessTransition:
        current = self._episode_id
        stale = current is None or completed.ticket.context.episode_id != current
        if stale:
            return HarnessTransition(
                state=self.state,
                planner=completed,
                stale=True,
                reason="planner result belongs to a different episode",
            )

        purpose = self._purpose
        self._purpose = None
        result = completed.result
        decision = result.decision
        if result.accepted and decision.scene_graph_update:
            self._scene_graph = parse_semantic_scene_graph(
                decision.scene_graph_update,
                source="vlm_observation",
                timestep=completed.ticket.context.timestep,
            )
        if purpose == TaskStartPolicy.STARTUP_SHADOW.value:
            self.state = HarnessState.EXECUTE_FAST
            return HarnessTransition(
                state=self.state,
                planner=completed,
                decision_applied=False,
                reason="task-start shadow decision recorded",
            )

        if not result.accepted or decision.intent is AgentIntent.SAFE_STOP:
            self._enter_hold(result.error or decision.rationale)
            self.state = HarnessState.SAFE_HOLD
            return HarnessTransition(
                state=self.state,
                planner=completed,
                decision_applied=True,
                reason=result.error or decision.rationale,
            )

        if self._hold is not None:
            self.safe_hold_adapter.release(
                self._hold,
                reason=f"accepted planner intent {decision.intent.value}",
            )
            self._hold = None

        if decision.intent is AgentIntent.RUN_SKILL:
            mode = ExecutionMode.RECOVERY
            state = HarnessState.RECOVER
        elif decision.intent is AgentIntent.CONTINUE and purpose == TaskStartPolicy.STARTUP_WAIT.value:
            mode = ExecutionMode.FAST_VLA
            state = HarnessState.EXECUTE_FAST
        else:
            mode = ExecutionMode.ACCURATE_VLA
            state = HarnessState.EXECUTE_FAST
        self.state = state
        return HarnessTransition(
            state=state,
            joint=self._joint_from_planner(mode, completed),
            planner=completed,
            decision_applied=True,
            reason=decision.rationale,
        )

    def _joint_from_planner(
        self,
        mode: ExecutionMode,
        completed: AsyncHighLevelPlannerResult,
    ) -> JointDecision:
        context = completed.ticket.context
        risk = RiskAssessment(
            score=float(context.risk.get("score", 0.0)),
            bucket=str(context.risk.get("bucket", "low")),
            event=context.risk.get("event"),
            components=dict(context.risk.get("components", {})),
            evidence=dict(context.risk.get("evidence", {})),
        )
        base = self.controller.decide(
            risk,
            deadline_ms=None,
            deadline_slack_ms=context.deadline_slack_ms,
            repeated_failures=0,
            recovery_attempts=self.counters.recovery_attempts,
            planner_available=self._planner is not None,
        )
        return dataclasses.replace(
            base,
            mode=mode,
            reason=completed.result.decision.rationale,
            recovery_skill_id=completed.result.decision.skill_id,
        )

    def _enter_hold(self, reason: str) -> None:
        if self._episode_id is None:
            raise RuntimeError("cannot enter safe hold without an active episode")
        if self._hold is not None:
            return
        request = SafeHoldRequest(
            episode_id=self._episode_id,
            reason=reason,
            heartbeat_hz=self.safe_hold_heartbeat_hz,
            timeout_s=self.safe_hold_timeout_s,
        )
        self.safe_hold_adapter.enter(request)
        self._hold = request

    def _require_current_context(self, context: HighLevelAgentContext) -> None:
        if self._episode_id is None:
            raise RuntimeError("start_episode must be called first")
        if context.episode_id != self._episode_id:
            raise ValueError(
                "high-level context episode does not match the active harness episode"
            )
