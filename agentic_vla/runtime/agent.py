"""Guarded high-level multimodal agent contracts for CARVE."""

from __future__ import annotations

import base64
import dataclasses
import enum
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .controller import ExecutionMode, JointDecision, JointRecoveryComputeController
from .monitor import RiskAssessment


class AgentIntent(str, enum.Enum):
    """Actions that a high-level agent may request from the harness."""

    CONTINUE = "continue"
    VLA_ACT = "vla_act"
    RUN_SKILL = "run_skill"
    SAFE_STOP = "safe_stop"


@dataclasses.dataclass(frozen=True)
class HighLevelAgentContext:
    """Deployable evidence exposed to a high-level multimodal agent."""

    task_instruction: str
    trigger: str
    episode_id: str | int
    timestep: int
    frames: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    robot_state: Sequence[float] = ()
    risk: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    current_subgoal: str = ""
    failure_history: tuple[str, ...] = ()
    memory: tuple[str, ...] = ()
    available_skills: tuple[str, ...] = ()
    remaining_retries: int = 0
    remaining_recoveries: int = 0
    deadline_slack_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.task_instruction.strip():
            raise ValueError("task_instruction must not be empty")
        if not self.trigger.strip():
            raise ValueError("trigger must not be empty")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if self.remaining_retries < 0 or self.remaining_recoveries < 0:
            raise ValueError("remaining budgets must be non-negative")
        skills = tuple(str(skill).strip() for skill in self.available_skills)
        if any(not skill for skill in skills):
            raise ValueError("available_skills must not contain empty names")
        if len(skills) != len(set(skills)):
            raise ValueError("available_skills must be unique")
        object.__setattr__(self, "available_skills", skills)


@dataclasses.dataclass(frozen=True)
class HighLevelAgentDecision:
    """A typed high-level decision that never contains raw robot actions."""

    intent: AgentIntent
    rationale: str
    confidence: float
    subgoal: str = ""
    vla_instruction: str | None = None
    skill_id: str | None = None
    skill_args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    expected_outcome: str = ""
    memory_note: str = ""

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.intent is AgentIntent.VLA_ACT:
            if self.vla_instruction is None or not self.vla_instruction.strip():
                raise ValueError("vla_act requires vla_instruction")
            if self.skill_id is not None:
                raise ValueError("vla_act must not include skill_id")
        elif self.intent is AgentIntent.RUN_SKILL:
            if self.skill_id is None or not self.skill_id.strip():
                raise ValueError("run_skill requires skill_id")
            if self.vla_instruction is not None:
                raise ValueError("run_skill must not include vla_instruction")
        elif self.vla_instruction is not None or self.skill_id is not None:
            raise ValueError("continue and safe_stop must not select an executor")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["intent"] = self.intent.value
        return payload


@dataclasses.dataclass(frozen=True)
class HighLevelAgentResult:
    """One auditable model call and its guarded decision."""

    decision: HighLevelAgentDecision
    accepted: bool
    elapsed_ms: float
    raw_output: Any | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class HighLevelAgentConfig:
    """Bound model usage and reject uncertain interventions."""

    max_calls_per_episode: int = 3
    minimum_intervention_confidence: float = 0.55
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.max_calls_per_episode <= 0:
            raise ValueError("max_calls_per_episode must be positive")
        if not 0.0 <= self.minimum_intervention_confidence <= 1.0:
            raise ValueError("minimum_intervention_confidence must be in [0, 1]")


class PlannerCallable(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> Any: ...


_DECISION_FIELDS = frozenset(
    {
        "intent",
        "rationale",
        "confidence",
        "subgoal",
        "vla_instruction",
        "skill_id",
        "skill_args",
        "expected_outcome",
        "memory_note",
    }
)
_FORBIDDEN_ACTION_FIELDS = frozenset(
    {"actions", "action", "joint_targets", "joint_positions", "torques", "trajectory"}
)


def build_high_level_agent_request(context: HighLevelAgentContext) -> dict[str, Any]:
    """Build a provider-neutral multimodal planning request."""

    skills = list(context.available_skills)
    system_prompt = (
        "You are the high-level multimodal planner inside a robot execution harness. "
        "You may select only a frozen VLA call, one registered analytic skill, continued "
        "execution, or a safe stop. Never emit joint commands, torques, trajectories, or "
        "unregistered skills. Return one compact JSON object, without Markdown. Required "
        "fields are intent, rationale, confidence, subgoal, vla_instruction, skill_id, and "
        "skill_args. intent must be one of continue, vla_act, run_skill, safe_stop. Use null "
        "for vla_instruction or skill_id when it does not apply. Keep rationale, subgoal, "
        "and vla_instruction below 12 words each. confidence is mandatory and must be a "
        "number from 0 to 1; a response without it is invalid. Do not repeat the scene "
        "description. Minimal valid example: "
        '{"intent":"vla_act","rationale":"target is visible","confidence":0.8,'
        '"subgoal":"grasp mug","vla_instruction":"grasp the mug","skill_id":null,'
        '"skill_args":{}}'
    )
    state = [float(value) for value in context.robot_state]
    user_payload = {
        "task_instruction": context.task_instruction,
        "trigger": context.trigger,
        "episode_id": context.episode_id,
        "timestep": context.timestep,
        "robot_state": state,
        "risk": dict(context.risk),
        "current_subgoal": context.current_subgoal,
        "failure_history": list(context.failure_history),
        "memory": list(context.memory),
        "available_skills": skills,
        "remaining_retries": context.remaining_retries,
        "remaining_recoveries": context.remaining_recoveries,
        "deadline_slack_ms": context.deadline_slack_ms,
    }
    return {
        "system_prompt": system_prompt,
        "user_prompt": json.dumps(user_payload, ensure_ascii=True, separators=(",", ":")),
        "frames": dict(context.frames),
    }


def parse_high_level_agent_decision(
    raw_output: Any,
    context: HighLevelAgentContext,
) -> HighLevelAgentDecision:
    """Parse and validate a model decision against the current capability set."""

    candidate = raw_output
    if isinstance(candidate, Mapping) and "intent" not in candidate and "content" in candidate:
        candidate = candidate["content"]
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("planner code block is not a single closed JSON block")
            opening = lines[0].strip().lower()
            if opening not in {"```", "```json"}:
                raise ValueError("planner code block must be JSON")
            text = "\n".join(lines[1:-1]).strip()
            if "```" in text:
                raise ValueError("planner output contains multiple code blocks")
        candidate = json.loads(text)
    if not isinstance(candidate, Mapping):
        raise ValueError("planner output must be a JSON object")

    keys = frozenset(str(key) for key in candidate)
    forbidden = keys & _FORBIDDEN_ACTION_FIELDS
    if forbidden:
        raise ValueError(f"planner output contains forbidden action fields: {sorted(forbidden)}")
    unknown = keys - _DECISION_FIELDS
    if unknown:
        raise ValueError(f"planner output contains unknown fields: {sorted(unknown)}")

    try:
        intent = AgentIntent(str(candidate.get("intent", "")).strip().lower())
    except ValueError as exc:
        raise ValueError("planner intent is not allowed") from exc

    skill_id = candidate.get("skill_id")
    if skill_id is not None:
        if not isinstance(skill_id, str):
            raise ValueError("skill_id must be a string or null")
        skill_id = skill_id.strip()
    vla_instruction = candidate.get("vla_instruction")
    if vla_instruction is not None:
        if not isinstance(vla_instruction, str):
            raise ValueError("vla_instruction must be a string or null")
        vla_instruction = vla_instruction.strip()
    skill_args = candidate.get("skill_args", {})
    if skill_args is None:
        skill_args = {}
    if not isinstance(skill_args, Mapping):
        raise ValueError("skill_args must be a JSON object")
    if intent is AgentIntent.RUN_SKILL and skill_id not in context.available_skills:
        raise ValueError(f"skill is not available in this state: {skill_id}")

    return HighLevelAgentDecision(
        intent=intent,
        rationale=str(candidate.get("rationale", "")).strip(),
        confidence=float(candidate.get("confidence", 0.0)),
        subgoal=str(candidate.get("subgoal", "")).strip(),
        vla_instruction=vla_instruction,
        skill_id=skill_id,
        skill_args=dict(skill_args),
        expected_outcome=str(candidate.get("expected_outcome", "")).strip(),
        memory_note=str(candidate.get("memory_note", "")).strip(),
    )


class GuardedHighLevelAgent:
    """Call a multimodal planner behind schema, capability, and budget gates."""

    def __init__(
        self,
        infer: PlannerCallable,
        config: HighLevelAgentConfig | None = None,
    ) -> None:
        if not callable(infer):
            raise TypeError("infer must be callable")
        self._infer = infer
        self.config = config or HighLevelAgentConfig()
        self._episode_id: str | int | None = None
        self._calls = 0

    @property
    def calls_in_episode(self) -> int:
        return self._calls

    def reset(self, episode_id: str | int | None = None) -> None:
        self._episode_id = episode_id
        self._calls = 0

    def decide(self, context: HighLevelAgentContext) -> HighLevelAgentResult:
        if context.episode_id != self._episode_id:
            self.reset(context.episode_id)
        if self._calls >= self.config.max_calls_per_episode:
            return self._rejected("high-level agent call budget exhausted")

        self._calls += 1
        started = time.perf_counter()
        raw_output: Any | None = None
        try:
            raw_output = self._infer(build_high_level_agent_request(context))
            decision = parse_high_level_agent_decision(raw_output, context)
            intervention = decision.intent not in {AgentIntent.CONTINUE, AgentIntent.SAFE_STOP}
            if intervention and decision.confidence < self.config.minimum_intervention_confidence:
                raise ValueError("planner intervention confidence is below threshold")
            return HighLevelAgentResult(
                decision=decision,
                accepted=True,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                raw_output=raw_output,
            )
        except Exception as exc:
            result = self._rejected(str(exc), raw_output=raw_output)
            return dataclasses.replace(
                result,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _rejected(
        self,
        error: str,
        *,
        raw_output: Any | None = None,
    ) -> HighLevelAgentResult:
        intent = AgentIntent.SAFE_STOP if self.config.fail_closed else AgentIntent.CONTINUE
        decision = HighLevelAgentDecision(
            intent=intent,
            rationale=f"guarded planner fallback: {error}",
            confidence=1.0,
        )
        return HighLevelAgentResult(
            decision=decision,
            accepted=False,
            elapsed_ms=0.0,
            raw_output=raw_output,
            error=error,
        )


class OpenAICompatibleVisionPlanner:
    """Send a provider-neutral planner request to a local multimodal endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_s: float = 30.0,
        max_tokens: int = 192,
        jpeg_quality: int = 85,
    ) -> None:
        if not endpoint.strip() or not model.strip():
            raise ValueError("endpoint and model must not be empty")
        if timeout_s <= 0 or max_tokens <= 0:
            raise ValueError("timeout_s and max_tokens must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.jpeg_quality = int(jpeg_quality)

    def __call__(self, request: Mapping[str, Any]) -> str:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": str(request["user_prompt"])}
        ]
        for name, frame in dict(request.get("frames", {})).items():
            content.append({"type": "text", "text": f"Camera view: {name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(frame)},
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(request["system_prompt"])},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"high-level VLM request failed: {exc}") from exc
        content_text = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError("high-level VLM returned no text")
        return content_text.strip()

    def _image_data_url(self, frame: Any) -> str:
        import imageio.v3 as iio
        import numpy as np

        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim not in (2, 3):
            raise ValueError("planner frame must be a 2-D or 3-D image")
        buffer = io.BytesIO()
        iio.imwrite(buffer, image, extension=".jpg", quality=self.jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + encoded


@dataclasses.dataclass(frozen=True)
class AgenticHarnessDecision:
    """Joint real-time decision plus an optional high-level escalation."""

    joint: JointDecision
    high_level: HighLevelAgentResult | None = None


class AgenticHarnessController:
    """Connect deterministic real-time control to a guarded high-level agent."""

    def __init__(
        self,
        controller: JointRecoveryComputeController,
        high_level_agent: GuardedHighLevelAgent | None = None,
    ) -> None:
        self.controller = controller
        self.high_level_agent = high_level_agent

    def decide(
        self,
        risk: RiskAssessment,
        *,
        context: HighLevelAgentContext,
        deadline_ms: float | None,
        deadline_slack_ms: float | None,
        cached_actions: int = 0,
        repeated_failures: int = 0,
        recovery_attempts: int = 0,
    ) -> AgenticHarnessDecision:
        joint = self.controller.decide(
            risk,
            deadline_ms=deadline_ms,
            deadline_slack_ms=deadline_slack_ms,
            cached_actions=cached_actions,
            repeated_failures=repeated_failures,
            recovery_attempts=recovery_attempts,
            planner_available=self.high_level_agent is not None,
        )
        if joint.mode is not ExecutionMode.PLANNER:
            return AgenticHarnessDecision(joint=joint)
        assert self.high_level_agent is not None
        return AgenticHarnessDecision(
            joint=joint,
            high_level=self.high_level_agent.decide(context),
        )
