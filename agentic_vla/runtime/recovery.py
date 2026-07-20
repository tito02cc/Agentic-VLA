"""Stateful, auditable physical-recovery contracts for CARVE."""

from __future__ import annotations

import dataclasses
import enum
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import ActionSpec


class RecoveryStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_VERIFICATION = "awaiting_verification"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SAFE_STOP = "safe_stop"


@dataclasses.dataclass(frozen=True)
class RecoveryPhase:
    """One bounded physical phase followed by optional deployable verification."""

    name: str
    actions: tuple[tuple[float, ...], ...]
    request_verification: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("recovery phase name must not be empty")
        if not self.actions:
            raise ValueError("recovery phase must contain at least one action")


@dataclasses.dataclass(frozen=True)
class RecoveryPlan:
    """A model-independent physical skill instantiated for one action contract."""

    skill_id: str
    trigger_events: frozenset[str]
    action_spec: ActionSpec
    phases: tuple[RecoveryPhase, ...]
    max_actions: int

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("recovery skill_id must not be empty")
        if not self.trigger_events:
            raise ValueError("recovery plan must declare at least one trigger event")
        if not self.phases:
            raise ValueError("recovery plan must contain at least one phase")
        if self.max_actions <= 0:
            raise ValueError("recovery max_actions must be positive")
        if self.action_count > self.max_actions:
            raise ValueError(
                f"recovery plan has {self.action_count} actions, exceeding {self.max_actions}"
            )
        if not self.phases[-1].request_verification:
            raise ValueError("final recovery phase must request verification")
        for phase in self.phases:
            self.action_spec.validate(phase.actions)

    @property
    def action_count(self) -> int:
        return sum(len(phase.actions) for phase in self.phases)


@dataclasses.dataclass(frozen=True)
class RecoveryContext:
    episode_id: str | int
    trigger_event: str
    attempt: int
    timestep: int | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    session_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not str(self.trigger_event).strip():
            raise ValueError("trigger_event must not be empty")
        if self.attempt < 0:
            raise ValueError("recovery attempt must be non-negative")
        if self.timestep is not None and self.timestep < 0:
            raise ValueError("recovery timestep must be non-negative")


@dataclasses.dataclass(frozen=True)
class RecoveryCommand:
    session_id: str
    skill_id: str
    phase_index: int
    phase_name: str
    actions: tuple[tuple[float, ...], ...]
    request_verification: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RecoveryOutcome:
    session_id: str
    episode_id: str | int
    trigger_event: str
    skill_id: str
    attempt: int
    status: RecoveryStatus
    actions_executed: int
    request_replan: bool
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["status"] = self.status.value
        return payload


class RecoveryMemory:
    """Bounded episodic memory of physical-recovery outcomes."""

    def __init__(self, max_records: int = 256) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._records: deque[RecoveryOutcome] = deque(maxlen=max_records)

    def record(self, outcome: RecoveryOutcome) -> None:
        if outcome.status not in {
            RecoveryStatus.SUCCEEDED,
            RecoveryStatus.FAILED,
            RecoveryStatus.SAFE_STOP,
        }:
            raise ValueError("only terminal recovery outcomes may be recorded")
        self._records.append(outcome)

    def recent(
        self,
        *,
        trigger_event: str | None = None,
        skill_id: str | None = None,
    ) -> tuple[RecoveryOutcome, ...]:
        return tuple(
            record
            for record in self._records
            if (trigger_event is None or record.trigger_event == trigger_event)
            and (skill_id is None or record.skill_id == skill_id)
        )

    def success_rate(self, *, trigger_event: str, skill_id: str) -> float | None:
        records = self.recent(trigger_event=trigger_event, skill_id=skill_id)
        if not records:
            return None
        return sum(record.status == RecoveryStatus.SUCCEEDED for record in records) / len(records)


class StatefulRecoveryExecutor:
    """Execute one bounded plan without hiding verification or fallback state."""

    def __init__(self, memory: RecoveryMemory | None = None) -> None:
        self.memory = memory or RecoveryMemory()
        self._plan: RecoveryPlan | None = None
        self._context: RecoveryContext | None = None
        self._phase_index = 0
        self._actions_executed = 0
        self._status = RecoveryStatus.IDLE

    @property
    def status(self) -> RecoveryStatus:
        return self._status

    def start(self, plan: RecoveryPlan, context: RecoveryContext) -> None:
        if self._status in {RecoveryStatus.RUNNING, RecoveryStatus.AWAITING_VERIFICATION}:
            raise RuntimeError("a recovery session is already active")
        if context.trigger_event not in plan.trigger_events:
            raise ValueError(
                f"skill {plan.skill_id!r} does not support event {context.trigger_event!r}"
            )
        self._plan = plan
        self._context = context
        self._phase_index = 0
        self._actions_executed = 0
        self._status = RecoveryStatus.RUNNING

    def next_command(self) -> RecoveryCommand:
        plan, context = self._active()
        if self._status == RecoveryStatus.AWAITING_VERIFICATION:
            raise RuntimeError("verification must be resolved before the next recovery command")
        if self._phase_index >= len(plan.phases):
            raise RuntimeError("recovery plan exhausted without a terminal verification")
        phase_index = self._phase_index
        phase = plan.phases[phase_index]
        self._phase_index += 1
        self._actions_executed += len(phase.actions)
        self._status = (
            RecoveryStatus.AWAITING_VERIFICATION
            if phase.request_verification
            else RecoveryStatus.RUNNING
        )
        return RecoveryCommand(
            session_id=context.session_id,
            skill_id=plan.skill_id,
            phase_index=phase_index,
            phase_name=phase.name,
            actions=phase.actions,
            request_verification=phase.request_verification,
        )

    def resolve_verification(
        self,
        success: bool,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> RecoveryOutcome:
        plan, context = self._active()
        if self._status != RecoveryStatus.AWAITING_VERIFICATION:
            raise RuntimeError("recovery is not awaiting verification")
        status = RecoveryStatus.SUCCEEDED if success else RecoveryStatus.FAILED
        outcome = RecoveryOutcome(
            session_id=context.session_id,
            episode_id=context.episode_id,
            trigger_event=context.trigger_event,
            skill_id=plan.skill_id,
            attempt=context.attempt,
            status=status,
            actions_executed=self._actions_executed,
            request_replan=bool(success),
            evidence=dict(evidence or {}),
        )
        self._status = status
        self.memory.record(outcome)
        return outcome

    def abort(
        self,
        *,
        reason: str,
        safe_stop: bool = True,
        evidence: Mapping[str, Any] | None = None,
    ) -> RecoveryOutcome:
        plan, context = self._active()
        if not reason.strip():
            raise ValueError("abort reason must not be empty")
        status = RecoveryStatus.SAFE_STOP if safe_stop else RecoveryStatus.FAILED
        payload = dict(evidence or {})
        payload["reason"] = reason
        outcome = RecoveryOutcome(
            session_id=context.session_id,
            episode_id=context.episode_id,
            trigger_event=context.trigger_event,
            skill_id=plan.skill_id,
            attempt=context.attempt,
            status=status,
            actions_executed=self._actions_executed,
            request_replan=False,
            evidence=payload,
        )
        self._status = status
        self.memory.record(outcome)
        return outcome

    def _active(self) -> tuple[RecoveryPlan, RecoveryContext]:
        if self._plan is None or self._context is None or self._status == RecoveryStatus.IDLE:
            raise RuntimeError("no active recovery session")
        if self._status in {
            RecoveryStatus.SUCCEEDED,
            RecoveryStatus.FAILED,
            RecoveryStatus.SAFE_STOP,
        }:
            raise RuntimeError("recovery session is already terminal")
        return self._plan, self._context


def build_cartesian_retreat_plan(
    action_spec: ActionSpec,
    last_action: Sequence[float],
    *,
    hold_steps: int = 2,
    retreat_steps: int = 4,
    lift_steps: int = 4,
    settle_steps: int = 2,
    retreat_scale: float = 0.5,
    lift_delta: float = 0.15,
) -> RecoveryPlan:
    """Build a bounded retract-lift-reobserve skill for normalized 7-D deltas."""

    if action_spec.action_dim != 7 or "delta_cartesian" not in action_spec.representation:
        raise ValueError("cartesian retreat requires a 7-D delta_cartesian action contract")
    step_counts = (hold_steps, retreat_steps, lift_steps, settle_steps)
    if any(value <= 0 for value in step_counts):
        raise ValueError("recovery phase step counts must be positive")
    if not 0.0 < retreat_scale <= 1.0:
        raise ValueError("retreat_scale must be in (0, 1]")
    if lift_delta <= 0.0:
        raise ValueError("lift_delta must be positive")

    previous = np.asarray(last_action, dtype=np.float32).reshape(-1)
    if previous.size != action_spec.action_dim:
        raise ValueError(
            f"last_action has dimension {previous.size}, expected {action_spec.action_dim}"
        )
    if not np.all(np.isfinite(previous)):
        raise ValueError("last_action contains a non-finite value")
    # Model outputs can slightly exceed normalized controller bounds. Project the
    # recovery seed so every generated physical command satisfies its own contract.
    if action_spec.minimum is not None:
        previous = np.maximum(previous, float(action_spec.minimum))
    if action_spec.maximum is not None:
        previous = np.minimum(previous, float(action_spec.maximum))
    action_spec.validate((previous,))
    gripper = float(previous[-1])
    hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32)
    retreat = hold.copy()
    retreat[:3] = -previous[:3] * float(retreat_scale)
    lift = hold.copy()
    lift[2] = float(lift_delta)

    def repeated(action: np.ndarray, count: int) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(value) for value in action) for _ in range(count))

    phases = (
        RecoveryPhase("stabilize", repeated(hold, hold_steps)),
        RecoveryPhase("retract", repeated(retreat, retreat_steps)),
        RecoveryPhase("lift", repeated(lift, lift_steps)),
        RecoveryPhase(
            "settle_reobserve",
            repeated(hold, settle_steps),
            request_verification=True,
        ),
    )
    return RecoveryPlan(
        skill_id="cartesian_retract_lift_reobserve",
        trigger_events=frozenset({"stall", "slip", "misgrasp", "contact"}),
        action_spec=action_spec,
        phases=phases,
        max_actions=sum(step_counts),
    )
