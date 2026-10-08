"""Joint recovery and inference-compute decisions for CARVE."""

from __future__ import annotations

import dataclasses
import enum

from .contracts import InferenceControls
from .monitor import RiskAssessment


class ExecutionMode(str, enum.Enum):
    REUSE = "reuse"
    FAST_VLA = "fast_vla"
    ACCURATE_VLA = "accurate_vla"
    RECOVERY = "recovery"
    PLANNER = "planner"
    SAFE_STOP = "safe_stop"


@dataclasses.dataclass(frozen=True)
class JointControllerConfig:
    fast_inference_steps: int | None = 2
    accurate_inference_steps: int | None = 2
    low_risk_commit: int = 10
    high_risk_commit: int = 10
    accurate_min_slack_ms: float = 30.0
    emergency_slack_ms: float = 5.0
    planner_after_failures: int = 2
    max_recovery_attempts: int = 2
    stall_recovery_streak: int = 2
    no_progress_planner_streak: int = 2
    physical_recovery_skill: str = "cartesian_retract_lift_reobserve"
    release_recovery_skill: str = "release_retract_lift_reobserve"

    def __post_init__(self) -> None:
        inference_steps = (
            self.fast_inference_steps,
            self.accurate_inference_steps,
        )
        if any(value is not None and value <= 0 for value in inference_steps):
            raise ValueError("inference step budgets must be positive or disabled")
        if self.low_risk_commit <= 0 or self.high_risk_commit <= 0:
            raise ValueError("inference and commit budgets must be positive")
        if self.stall_recovery_streak < 2:
            raise ValueError("stall_recovery_streak must be at least 2")
        if self.no_progress_planner_streak < 2:
            raise ValueError("no_progress_planner_streak must be at least 2")
        if not self.physical_recovery_skill.strip():
            raise ValueError("physical_recovery_skill must not be empty")
        if not self.release_recovery_skill.strip():
            raise ValueError("release_recovery_skill must not be empty")


@dataclasses.dataclass(frozen=True)
class JointDecision:
    mode: ExecutionMode
    controls: InferenceControls
    request_verification: bool
    reason: str
    risk: RiskAssessment
    recovery_skill_id: str | None = None

    def to_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["mode"] = self.mode.value
        return payload


class JointRecoveryComputeController:
    """Map one risk assessment to both execution mode and model budget."""

    _PHYSICAL_FAILURES = frozenset({"slip", "misgrasp", "contact"})

    def __init__(self, config: JointControllerConfig | None = None) -> None:
        self.config = config or JointControllerConfig()

    def decide(
        self,
        risk: RiskAssessment,
        *,
        deadline_ms: float | None,
        deadline_slack_ms: float | None,
        cached_actions: int = 0,
        repeated_failures: int = 0,
        recovery_attempts: int = 0,
        planner_available: bool = False,
        reuse_permitted: bool = True,
    ) -> JointDecision:
        if cached_actions < 0 or repeated_failures < 0 or recovery_attempts < 0:
            raise ValueError("controller counters must be non-negative")
        slack = float("inf") if deadline_slack_ms is None else float(deadline_slack_ms)
        high_risk = risk.bucket == "high"

        if high_risk and slack <= self.config.emergency_slack_ms:
            return self._decision(
                ExecutionMode.SAFE_STOP,
                risk,
                deadline_ms,
                self.config.fast_inference_steps,
                1,
                True,
                "high risk with emergency deadline slack",
            )
        if repeated_failures >= self.config.planner_after_failures:
            mode = ExecutionMode.PLANNER if planner_available else ExecutionMode.SAFE_STOP
            reason = "repeated failure escalation" if planner_available else "repeated failures without planner"
            return self._decision(
                mode,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.high_risk_commit,
                True,
                reason,
            )
        event_streak = int(risk.evidence.get("event_streak", 0))
        if risk.event == "no_progress" and event_streak >= self.config.no_progress_planner_streak:
            if planner_available:
                return self._decision(
                    ExecutionMode.PLANNER,
                    risk,
                    deadline_ms,
                    self.config.accurate_inference_steps,
                    self.config.high_risk_commit,
                    True,
                    f"semantic progress check after {event_streak} idle windows",
                )
            return self._decision(
                ExecutionMode.ACCURATE_VLA,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.high_risk_commit,
                True,
                "no progress detected without a semantic planner",
            )
        if risk.event == "no_progress":
            return self._decision(
                ExecutionMode.ACCURATE_VLA,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.low_risk_commit,
                True,
                "request a fresh observation-conditioned action chunk before escalation",
            )
        if risk.event == "stall" and event_streak >= self.config.stall_recovery_streak:
            if recovery_attempts < self.config.max_recovery_attempts:
                return self._decision(
                    ExecutionMode.RECOVERY,
                    risk,
                    deadline_ms,
                    self.config.fast_inference_steps,
                    self.config.high_risk_commit,
                    True,
                    f"confirmed stall after {event_streak} monitor windows",
                    recovery_skill_id=self.config.physical_recovery_skill,
                )
            mode = ExecutionMode.PLANNER if planner_available else ExecutionMode.SAFE_STOP
            reason = (
                "confirmed stall after recovery budget exhaustion"
                if planner_available
                else "confirmed stall without remaining recovery or planner"
            )
            return self._decision(
                mode,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.high_risk_commit,
                True,
                reason,
            )
        if risk.event == "stall" and slack >= self.config.accurate_min_slack_ms:
            return self._decision(
                ExecutionMode.ACCURATE_VLA,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.low_risk_commit,
                True,
                "defer stall confirmation to the next accurate replan",
            )
        if (
            risk.event in self._PHYSICAL_FAILURES
            and recovery_attempts < self.config.max_recovery_attempts
        ):
            recovery_skill = (
                self.config.release_recovery_skill
                if risk.event in {"slip", "misgrasp"}
                else self.config.physical_recovery_skill
            )
            return self._decision(
                ExecutionMode.RECOVERY,
                risk,
                deadline_ms,
                self.config.fast_inference_steps,
                self.config.high_risk_commit,
                True,
                f"physical recovery for {risk.event}",
                recovery_skill_id=recovery_skill,
            )
        if high_risk and slack >= self.config.accurate_min_slack_ms:
            return self._decision(
                ExecutionMode.ACCURATE_VLA,
                risk,
                deadline_ms,
                self.config.accurate_inference_steps,
                self.config.high_risk_commit,
                True,
                "high risk with sufficient compute slack",
            )
        if risk.bucket == "low" and cached_actions > 0 and reuse_permitted:
            return self._decision(
                ExecutionMode.REUSE,
                risk,
                deadline_ms,
                self.config.fast_inference_steps,
                min(cached_actions, self.config.low_risk_commit),
                False,
                "low risk cached-action fast path",
            )
        reason = "deadline-aware fast inference"
        if cached_actions > 0 and not reuse_permitted:
            reason = "cached actions invalidated; fresh deadline-aware inference"
        return self._decision(
            ExecutionMode.FAST_VLA,
            risk,
            deadline_ms,
            self.config.fast_inference_steps,
            self.config.high_risk_commit if risk.bucket != "low" else self.config.low_risk_commit,
            risk.bucket != "low",
            reason,
        )

    @staticmethod
    def _decision(
        mode: ExecutionMode,
        risk: RiskAssessment,
        deadline_ms: float | None,
        inference_steps: int | None,
        max_actions: int,
        request_verification: bool,
        reason: str,
        recovery_skill_id: str | None = None,
    ) -> JointDecision:
        return JointDecision(
            mode=mode,
            controls=InferenceControls(
                inference_steps=inference_steps,
                max_actions=max_actions,
                deadline_ms=deadline_ms,
            ),
            request_verification=request_verification,
            reason=reason,
            risk=risk,
            recovery_skill_id=recovery_skill_id,
        )
