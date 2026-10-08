"""Guarded low-frequency VLM verification for symbolic robot outcomes."""

from __future__ import annotations

import dataclasses
import json
import math
import operator
import re
import time
from collections.abc import Mapping
from typing import Any

from agentic_vla.runtime.agent import PlannerCallable

from .contracts import VerificationReport, VerificationStatus
from ._json import decode_json_object


@dataclasses.dataclass(frozen=True)
class VisualVerificationContext:
    task_instruction: str
    expected_outcome: str
    frames: Mapping[str, Any]
    timestep: int
    active_stage: str = ""
    require_visual_change: bool = False
    minimum_mean_pixel_change: float = 1.0

    def __post_init__(self) -> None:
        if not self.task_instruction.strip() or not self.expected_outcome.strip():
            raise ValueError("verification task and expected outcome must not be empty")
        if self.timestep < 0:
            raise ValueError("verification timestep must be non-negative")
        if not self.frames:
            raise ValueError("visual verification requires at least one frame")
        if self.minimum_mean_pixel_change < 0:
            raise ValueError("minimum visual change must be non-negative")


@dataclasses.dataclass(frozen=True)
class VisualVerificationResult:
    report: VerificationReport
    accepted: bool
    elapsed_ms: float
    error: str | None = None
    raw_output: Any | None = None


@dataclasses.dataclass(frozen=True)
class ScalarVisualPredicate:
    """A planner-declared visual scalar whose truth is decided by the harness."""

    predicate_id: str
    question: str
    comparison: str
    target: float
    unit: str = "count"
    tolerance: float = 0.0
    minimum_valid_value: float | None = None
    maximum_valid_value: float | None = None

    def __post_init__(self) -> None:
        if not self.predicate_id.strip() or not self.question.strip():
            raise ValueError("visual predicate id and question must not be empty")
        if self.comparison not in {"eq", "le", "ge", "lt", "gt"}:
            raise ValueError("unsupported visual predicate comparison")
        if self.tolerance < 0:
            raise ValueError("visual predicate tolerance must be non-negative")
        if (
            self.minimum_valid_value is not None
            and self.maximum_valid_value is not None
            and self.minimum_valid_value > self.maximum_valid_value
        ):
            raise ValueError("visual predicate valid range is reversed")


@dataclasses.dataclass(frozen=True)
class ScalarVisualObservation:
    """Action-free scalar evidence extracted by a VLM from camera observations."""

    predicate_id: str
    visible: bool
    value: float | None
    confidence: float
    evidence: str

    def __post_init__(self) -> None:
        if not self.predicate_id.strip() or not self.evidence.strip():
            raise ValueError("visual observation id and evidence must not be empty")
        if self.visible and self.value is None:
            raise ValueError("a visible scalar observation requires a value")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("visual observation confidence must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class GroupedVisualObservation:
    """Visible groups enumerated by a VLM and counted by the harness."""

    visible: bool
    groups: tuple[str, ...]
    confidence: float
    member_counts: tuple[int | None, ...] = ()
    group_relations: tuple[str, ...] = ()
    all_targets_visible: bool | None = None

    def __post_init__(self) -> None:
        if self.visible and not self.groups:
            raise ValueError("a visible grouped observation requires at least one group")
        if not self.visible and self.groups:
            raise ValueError("an occluded grouped observation cannot enumerate groups")
        if any(not item.strip() for item in self.groups):
            raise ValueError("group descriptions must not be empty")
        if len(self.groups) > 16:
            raise ValueError("grouped observation exceeds the bounded group count")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("grouped observation confidence must be in [0, 1]")


def build_visual_verification_request(
    context: VisualVerificationContext,
) -> dict[str, Any]:
    """Build a provider-neutral, action-free verification request."""

    return {
        "system_prompt": (
            "You are the low-frequency visual Critic in a robot execution harness. "
            "Judge only whether expected_outcome is supported by the current camera "
            "images. Return one JSON object with status, observed_outcome, and "
            "confidence. status must be confirmed, contradicted, or inconclusive. "
            "confidence must be a JSON number between 0 and 1, never a word such as "
            "low, medium, or high. "
            "Use confirmed only when the visible evidence directly supports the full "
            "predicate. Use contradicted only when visible evidence clearly conflicts. "
            "Use inconclusive for occlusion, ambiguity, missing views, partial progress, "
            "or predicates that are not visually decidable. Never infer hidden state, "
            "reward, simulator state, metric poses, or robot actions. When camera names "
            "start with before_ and current_, compare the paired views but judge the "
            "predicate only in current_ views; visible change is supporting evidence, not "
            "proof by itself. Output JSON only, without Markdown. Keep observed_outcome "
            "to one concrete sentence of at most 16 words so the response cannot be "
            "truncated by a bounded robot-runtime token budget. For confirmed, state "
            "positive visible evidence and do not use a negation or failure phrase."
        ),
        "user_prompt": json.dumps(
            {
                "task_instruction": context.task_instruction,
                "active_stage": context.active_stage,
                "expected_outcome": context.expected_outcome,
                "timestep": context.timestep,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "frames": dict(context.frames),
    }


def parse_visual_verification_report(raw_output: Any) -> VerificationReport:
    candidate = raw_output
    if isinstance(candidate, Mapping) and "status" not in candidate and "content" in candidate:
        candidate = candidate["content"]
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("Critic code block is not one closed JSON object")
            text = "\n".join(lines[1:-1]).strip()
        candidate = json.loads(text)
    if not isinstance(candidate, Mapping):
        raise ValueError("Critic output must be a JSON object")
    allowed = {"status", "observed_outcome", "confidence"}
    unknown = set(candidate) - allowed
    if unknown:
        raise ValueError(f"Critic output contains unknown fields: {sorted(unknown)}")
    return VerificationReport(
        status=str(candidate.get("status", "")),
        observed_outcome=str(candidate.get("observed_outcome", "")),
        confidence=float(candidate.get("confidence", 0.0)),
    )


_NEGATIVE_VISUAL_EVIDENCE = re.compile(
    r"\b(?:not|isn't|aren't|wasn't|weren't|cannot|can't|failed|failure|"
    r"incomplete|unfinished|missing|separate|separated|apart|unchanged)\b",
    flags=re.IGNORECASE,
)

_UNOBSERVABLE_EVIDENCE = re.compile(
    r"\b(?:not\s+(?:clearly\s+)?visible|cannot\s+(?:see|determine|tell)|"
    r"can't\s+(?:see|determine|tell)|out\s+of\s+(?:the\s+)?(?:view|frame)|"
    r"outside\s+(?:the\s+)?(?:camera\s+)?(?:view|frame)|occluded|"
    r"not\s+(?:clearly\s+)?discernible|unclear|ambiguous)\b",
    flags=re.IGNORECASE,
)


def build_visual_evidence_request(context: VisualVerificationContext) -> dict[str, Any]:
    """Keep the task context, but require visual evidence before a host decision."""
    request = build_visual_verification_request(context)
    request["system_prompt"] = (
        "You inspect robot camera images for one expected_outcome. The full task is "
        "context, not evidence that any action happened. Judge only expected_outcome; "
        "do not substitute another object, destination, or task stage. Return one JSON "
        "object with exactly visibility, relation, evidence_views, observed_outcome, confidence. "
        "visibility is clear, occluded, out_of_view, or ambiguous: clear means ALL visual "
        "evidence needed for this relation is visible, not merely that an object exists. "
        "relation is supported, refuted, or unknown. Use refuted only for a directly "
        "visible conflicting state; not seeing an object never proves absence or failure. "
        "If visibility is not clear, relation must be unknown. evidence_views is an array "
        "of current camera names actually supporting the observation; use [] if none. "
        "Historical before_ views cannot establish the current state. Do not infer hidden "
        "poses, rewards, physical contact, or task completion. observed_outcome is one "
        "visible fact of at most 20 words. confidence is a JSON number in [0,1]. "
        "Do not output a status or suggest robot actions. Output JSON only."
    )
    return request


def parse_visual_evidence_report(raw_output: Any, context: VisualVerificationContext) -> VerificationReport:
    candidate = raw_output
    if isinstance(candidate, Mapping) and "content" in candidate and "visibility" not in candidate:
        candidate = candidate["content"]
    candidate = decode_json_object(candidate)
    required = {"visibility", "relation", "evidence_views", "observed_outcome", "confidence"}
    if set(candidate) != required:
        raise ValueError("visual evidence requires exactly the five declared fields")
    visibility, relation = candidate["visibility"], candidate["relation"]
    if visibility not in ("clear", "occluded", "out_of_view", "ambiguous"):
        raise ValueError("invalid visual evidence visibility")
    if relation not in ("supported", "refuted", "unknown"):
        raise ValueError("invalid visual evidence relation")
    confidence = candidate["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("visual evidence confidence must be a finite JSON number in [0,1]")
    evidence = candidate["observed_outcome"]
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 500:
        raise ValueError("visual evidence needs a bounded nonempty observation")
    views = candidate["evidence_views"]
    if (not isinstance(views, list) or any(not isinstance(v, str) for v in views)
            or len(views) != len(set(views))):
        raise ValueError("evidence_views must be a unique array of camera names")
    if any(v not in context.frames or v.startswith("before_") for v in views):
        raise ValueError("visual evidence cites an unavailable or historical camera")
    status = VerificationStatus.INCONCLUSIVE
    veto = None
    if visibility != "clear":
        veto = "insufficient_visibility"
    elif relation != "unknown":
        if not views:
            raise ValueError("definitive relation requires a current evidence view")
        status = VerificationStatus.CONFIRMED if relation == "supported" else VerificationStatus.CONTRADICTED
    return VerificationReport(status=status, observed_outcome=evidence, confidence=confidence,
        metadata={"protocol": "evidence", "visibility": visibility, "relation": relation,
                  "evidence_views": views, "visibility_veto": veto,
                  "evidence_is_model_reported": True})


def _reject_internally_inconsistent_confirmation(
    report: VerificationReport,
) -> None:
    """Fail closed when a definitive label contradicts its own evidence text."""

    if report.status is not VerificationStatus.INCONCLUSIVE and _UNOBSERVABLE_EVIDENCE.search(report.observed_outcome):
        raise ValueError("definitive Critic status conflicts with unobservable evidence")

    if (
        report.status is VerificationStatus.CONFIRMED
        and _NEGATIVE_VISUAL_EVIDENCE.search(report.observed_outcome)
    ):
        raise ValueError(
            "confirmed Critic status conflicts with negative observed evidence"
        )


def build_scalar_visual_predicate_request(
    context: VisualVerificationContext,
    predicate: ScalarVisualPredicate,
) -> dict[str, Any]:
    """Ask a VLM for evidence only; it cannot decide task success."""

    return {
        "system_prompt": (
            "You extract one observable scalar from robot camera images. Answer the "
            "question without deciding task success, recovery, or robot actions. Return "
            "JSON only with predicate_id, visible, value, confidence, and evidence. Set "
            "visible=false and value=null when occlusion or ambiguity prevents a reliable "
            "measurement. evidence must be one short sentence grounded in visible pixels."
        ),
        "user_prompt": json.dumps(
            {
                "task_instruction": context.task_instruction,
                "active_stage": context.active_stage,
                "predicate_id": predicate.predicate_id,
                "question": predicate.question,
                "unit": predicate.unit,
                "minimum_valid_value": predicate.minimum_valid_value,
                "maximum_valid_value": predicate.maximum_valid_value,
                "timestep": context.timestep,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "frames": dict(context.frames),
    }


def build_grouped_visual_predicate_request(
    context: VisualVerificationContext,
    predicate: ScalarVisualPredicate,
    *,
    expected_object_count: int | None = None,
) -> dict[str, Any]:
    """Request group enumeration so the model cannot invent the final count."""

    request = {
        "system_prompt": (
            "You enumerate spatially separate visible groups in robot camera images. "
            "Answer the question without deciding task success or robot actions. Return "
            "exactly one compact JSON object with only visible, groups, and confidence. "
            "groups must contain one short string per spatially separate target group. "
            "Example: {\"visible\":true,\"groups\":[\"green bowl at left\","
            "\"white bowl at right\"],\"confidence\":0.9}. Nested or touching target "
            "objects belong to one group. Every pair of target objects that are visibly "
            "separated by free space must be listed as two entries, even when they have "
            "the same type, color, or support surface. Never combine several separated "
            "instances into one phrase such as 'three bowls on the table'. Do not return "
            "the same target instances twice or add an aggregate label alongside its "
            "members. Do not return a numeric count, status, group objects, or Markdown. "
            "Use visible=false and "
            "groups=[] only when reliable "
            "enumeration is impossible."
        ),
        "user_prompt": json.dumps(
            {
                "task_instruction": context.task_instruction,
                "active_stage": context.active_stage,
                "predicate_id": predicate.predicate_id,
                "question": predicate.question,
                "unit": predicate.unit,
                "minimum_valid_value": predicate.minimum_valid_value,
                "maximum_valid_value": predicate.maximum_valid_value,
                "timestep": context.timestep,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "frames": dict(context.frames),
    }
    if expected_object_count is not None:
        request["system_prompt"] = (
            "Inspect target objects in ONE current robot camera view. Return JSON only "
            "with visible, groups, confidence, member_counts, group_relations, and "
            "all_targets_visible. groups is a list of short descriptions, one per "
            "spatial group. member_counts and group_relations have one entry per group. "
            "Each member count is a positive integer or null if not visibly countable. "
            "Each relation is singleton, stacked, nested, touching, or unknown. "
            "singleton means exactly one object; stacked/nested require multiple "
            "visibly countable objects, not an instruction to stack them. "
            "Separate objects must be separate groups, even on the same surface. "
            "Do not double-count. Touching side by side is NOT stacked or nested. "
            "all_targets_visible is a boolean: false for omitted, occluded or uncertain "
            "targets. Do not invent hidden members to match the expected inventory. "
            "If nothing is reliably visible, use visible=false and empty arrays. "
            "Include ALL six keys, including visible even when true. Example format "
            "for one visible target with others occluded (not an answer for this image): "
            '{"visible":true,"groups":["bowl at left"],"confidence":0.8,'
            '"member_counts":[1],"group_relations":["singleton"],"all_targets_visible":false}. '
            "Report visual evidence only, not actions or overall task completion."
        )
        payload = json.loads(request["user_prompt"])
        payload["expected_object_count"] = expected_object_count
        payload["inventory_source"] = "caller-declared task inventory, not simulator state"
        request["user_prompt"] = json.dumps(payload, ensure_ascii=True)
    return request


def parse_scalar_visual_observation(raw_output: Any) -> ScalarVisualObservation:
    candidate = raw_output
    if isinstance(candidate, Mapping) and "predicate_id" not in candidate and "content" in candidate:
        candidate = candidate["content"]
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("VLM code block is not one closed JSON object")
            text = "\n".join(lines[1:-1]).strip()
        candidate = json.loads(text)
    if not isinstance(candidate, Mapping):
        raise ValueError("scalar visual observation must be a JSON object")
    allowed = {"predicate_id", "visible", "value", "confidence", "evidence"}
    unknown = set(candidate) - allowed
    if unknown:
        raise ValueError(f"scalar visual observation contains unknown fields: {sorted(unknown)}")
    value = candidate.get("value")
    return ScalarVisualObservation(
        predicate_id=str(candidate.get("predicate_id", "")),
        visible=bool(candidate.get("visible", False)),
        value=None if value is None else float(value),
        confidence=float(candidate.get("confidence", 0.0)),
        evidence=str(candidate.get("evidence", "")),
    )


def parse_grouped_visual_observation(
    raw_output: Any, *, require_coverage: bool = False,
) -> GroupedVisualObservation:
    candidate = raw_output
    if isinstance(candidate, Mapping) and "predicate_id" not in candidate and "content" in candidate:
        candidate = candidate["content"]
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("VLM code block is not one closed JSON object")
            text = "\n".join(lines[1:-1]).strip()
        candidate = json.loads(text)
    if not isinstance(candidate, Mapping):
        raise ValueError("grouped visual observation must be a JSON object")
    allowed = {"visible", "groups", "confidence"}
    if require_coverage:
        allowed |= {"member_counts", "group_relations", "all_targets_visible"}
        if set(candidate) != allowed:
            raise ValueError("coverage observation must contain every requested field")
    unknown = set(candidate) - allowed
    if unknown:
        raise ValueError(f"grouped visual observation contains unknown fields: {sorted(unknown)}")
    raw_groups = candidate.get("groups", [])
    if not isinstance(raw_groups, list):
        raise ValueError("grouped visual observation groups must be an array")
    if any(not isinstance(item, str) for item in raw_groups):
        raise ValueError("each visual group must be a short string")
    counts, relations, complete = (), (), None
    if require_coverage:
        counts = candidate["member_counts"]
        relations = candidate["group_relations"]
        complete = candidate["all_targets_visible"]
        if type(candidate["visible"]) is not bool or type(complete) is not bool:
            raise ValueError("coverage visibility fields must be JSON booleans")
        if not isinstance(counts, list) or not isinstance(relations, list):
            raise ValueError("coverage counts and relations must be arrays")
        if len(counts) != len(raw_groups) or len(relations) != len(raw_groups):
            raise ValueError("coverage arrays must align with groups")
        if any(c is not None and (type(c) is not int or not 1 <= c <= 16) for c in counts):
            raise ValueError("group member counts must be bounded integers or null")
        if any(r not in ("singleton", "stacked", "nested", "touching", "unknown") for r in relations):
            raise ValueError("unsupported group relation")
        if any((r == "singleton" and c != 1) or
               (r in ("stacked", "nested", "touching") and c == 1)
               for c, r in zip(counts, relations, strict=True)):
            raise ValueError("group relation conflicts with member count")
    multi_instance = re.compile(
        r"\b(?:two|three|four|five|six|seven|eight|nine|ten|multiple|several|[2-9]|1[0-6])\b",
        re.IGNORECASE,
    )
    grouping_relation = re.compile(
        r"\b(?:stack(?:ed|ing)?|nest(?:ed|ing)?|touch(?:ed|ing)?|together|"
        r"cluster(?:ed|ing)?|pile(?:d|ing)?|inside|within)\b",
        re.IGNORECASE,
    )
    ambiguous = [
        item
        for item in raw_groups
        if multi_instance.search(item) and not grouping_relation.search(item)
    ]
    if ambiguous:
        raise ValueError(
            "a multi-instance visual group must state the visible grouping relation"
        )
    return GroupedVisualObservation(
        visible=bool(candidate.get("visible", False)),
        groups=tuple(raw_groups),
        confidence=float(candidate.get("confidence", 0.0)),
        member_counts=tuple(counts),
        group_relations=tuple(relations),
        all_targets_visible=complete,
    )


def evaluate_scalar_visual_predicate(
    predicate: ScalarVisualPredicate,
    observation: ScalarVisualObservation,
    *,
    minimum_confidence: float = 0.55,
) -> VerificationReport:
    """Deterministically derive predicate truth from model-extracted evidence."""

    if observation.predicate_id != predicate.predicate_id:
        raise ValueError("visual observation does not match requested predicate")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must be in [0, 1]")
    metadata = {
        "predicate_id": predicate.predicate_id,
        "comparison": predicate.comparison,
        "target": float(predicate.target),
        "unit": predicate.unit,
        "visible": bool(observation.visible),
        "observed_value": observation.value,
    }
    if (
        not observation.visible
        or observation.value is None
        or observation.confidence < minimum_confidence
    ):
        return VerificationReport(
            status=VerificationStatus.INCONCLUSIVE,
            observed_outcome=observation.evidence,
            confidence=float(observation.confidence),
            metadata=metadata,
        )

    value = float(observation.value)
    if (
        predicate.minimum_valid_value is not None
        and value < predicate.minimum_valid_value
    ) or (
        predicate.maximum_valid_value is not None
        and value > predicate.maximum_valid_value
    ):
        raise ValueError(
            f"observed value {value:g} is outside the physically valid range "
            f"[{predicate.minimum_valid_value}, {predicate.maximum_valid_value}]"
        )
    target = float(predicate.target)
    tolerance = float(predicate.tolerance)
    comparisons = {
        "eq": lambda left, right: abs(left - right) <= tolerance,
        "le": operator.le,
        "ge": operator.ge,
        "lt": operator.lt,
        "gt": operator.gt,
    }
    met = bool(comparisons[predicate.comparison](value, target))
    return VerificationReport(
        status=(VerificationStatus.CONFIRMED if met else VerificationStatus.CONTRADICTED),
        observed_outcome=observation.evidence,
        confidence=float(observation.confidence),
        metadata=metadata,
    )


class GuardedVisualVerifier:
    """Fail to inconclusive when a VLM cannot provide admissible evidence."""

    def __init__(
        self,
        infer: PlannerCallable,
        *,
        minimum_confidence: float = 0.55,
        protocol: str = "legacy",
    ) -> None:
        if not callable(infer):
            raise TypeError("visual verifier infer must be callable")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum confidence must be in [0, 1]")
        self.infer = infer
        if protocol not in {"legacy", "evidence"}:
            raise ValueError("invalid visual verifier protocol")
        self.protocol = protocol
        self.minimum_confidence = float(minimum_confidence)
        self.calls = 0
        self.accepted_calls = 0
        self.total_latency_ms = 0.0

    def build_request(self, context: VisualVerificationContext) -> dict[str, Any]:
        builder = build_visual_evidence_request if self.protocol == "evidence" else build_visual_verification_request
        return builder(context)

    def parse_report(self, raw_output: Any, context: VisualVerificationContext) -> VerificationReport:
        if self.protocol == "evidence":
            return parse_visual_evidence_report(raw_output, context)
        return parse_visual_verification_report(raw_output)

    def verify(self, context: VisualVerificationContext) -> VisualVerificationResult:
        started = time.perf_counter()
        raw_output = None
        error = None
        try:
            self.calls += 1
            raw_output = self.infer(self.build_request(context))
            report = self.parse_report(raw_output, context)
            _reject_internally_inconsistent_confirmation(report)
            visual_change = _paired_visual_change(context.frames)
            if (
                context.require_visual_change
                and report.status is VerificationStatus.CONFIRMED
                and (
                    visual_change is None
                    or visual_change < context.minimum_mean_pixel_change
                )
            ):
                report = VerificationReport(
                    status=VerificationStatus.INCONCLUSIVE,
                    observed_outcome=(
                        "semantic confirmation vetoed because paired views show "
                        "insufficient visual change"
                    ),
                    confidence=0.0,
                    metadata={"mean_pixel_change": visual_change},
                )
            if (
                report.status is not VerificationStatus.INCONCLUSIVE
                and report.confidence < self.minimum_confidence
            ):
                raise ValueError("definitive Critic result is below confidence threshold")
            accepted = True
            self.accepted_calls += 1
        except Exception as exc:
            error = str(exc)
            accepted = False
            report = VerificationReport(
                status=VerificationStatus.INCONCLUSIVE,
                observed_outcome=f"visual Critic unavailable: {error}",
                confidence=0.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_latency_ms += elapsed_ms
        return VisualVerificationResult(
            report=report,
            accepted=accepted,
            elapsed_ms=elapsed_ms,
            error=error,
            raw_output=raw_output,
        )

    def metrics(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "accepted_calls": self.accepted_calls,
            "total_latency_ms": self.total_latency_ms,
            "mean_latency_ms": (
                0.0 if self.calls == 0 else self.total_latency_ms / self.calls
            ),
        }


class GuardedScalarVisualVerifier:
    """Extract scalar visual evidence and keep the final decision deterministic."""

    def __init__(
        self,
        infer: PlannerCallable,
        *,
        minimum_confidence: float = 0.55,
    ) -> None:
        if not callable(infer):
            raise TypeError("scalar visual verifier infer must be callable")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.infer = infer
        self.minimum_confidence = float(minimum_confidence)
        self.calls = 0
        self.accepted_calls = 0
        self.total_latency_ms = 0.0

    def verify(
        self,
        context: VisualVerificationContext,
        predicate: ScalarVisualPredicate,
    ) -> VisualVerificationResult:
        started = time.perf_counter()
        raw_output = None
        error = None
        try:
            self.calls += 1
            raw_output = self.infer(
                build_scalar_visual_predicate_request(context, predicate)
            )
            observation = parse_scalar_visual_observation(raw_output)
            report = evaluate_scalar_visual_predicate(
                predicate,
                observation,
                minimum_confidence=self.minimum_confidence,
            )
            accepted = True
            self.accepted_calls += 1
        except Exception as exc:
            error = str(exc)
            accepted = False
            report = VerificationReport(
                status=VerificationStatus.INCONCLUSIVE,
                observed_outcome=f"scalar visual verifier unavailable: {error}",
                confidence=0.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_latency_ms += elapsed_ms
        return VisualVerificationResult(
            report=report,
            accepted=accepted,
            elapsed_ms=elapsed_ms,
            error=error,
            raw_output=raw_output,
        )

    def metrics(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "accepted_calls": self.accepted_calls,
            "total_latency_ms": self.total_latency_ms,
            "mean_latency_ms": (
                0.0 if self.calls == 0 else self.total_latency_ms / self.calls
            ),
        }


class GuardedGroupedVisualVerifier:
    """Count model-enumerated visual groups inside the deterministic harness."""

    def __init__(
        self,
        infer: PlannerCallable,
        *,
        minimum_confidence: float = 0.55,
        expected_object_count: int | None = None,
    ) -> None:
        if not callable(infer):
            raise TypeError("grouped visual verifier infer must be callable")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.infer = infer
        if expected_object_count is not None and (
            type(expected_object_count) is not int or not 1 <= expected_object_count <= 16
        ):
            raise ValueError("expected object count must be an integer in [1, 16]")
        self.expected_object_count = expected_object_count
        self.minimum_confidence = float(minimum_confidence)
        self.calls = 0
        self.accepted_calls = 0
        self.total_latency_ms = 0.0

    def verify(
        self,
        context: VisualVerificationContext,
        predicate: ScalarVisualPredicate,
    ) -> VisualVerificationResult:
        started = time.perf_counter()
        raw_output = None
        error = None
        try:
            self.calls += 1
            raw_output = self.infer(
                build_grouped_visual_predicate_request(
                    context, predicate, expected_object_count=self.expected_object_count,
                )
            )
            grouped = parse_grouped_visual_observation(
                raw_output, require_coverage=self.expected_object_count is not None,
            )
            coverage_valid = None
            if self.expected_object_count is not None:
                coverage_valid = bool(
                    grouped.visible and grouped.all_targets_visible
                    and all(c is not None for c in grouped.member_counts)
                    and sum(c or 0 for c in grouped.member_counts) == self.expected_object_count
                    and all(r in ("singleton", "stacked", "nested") for r in grouped.group_relations)
                )
            usable = grouped.visible and coverage_valid is not False
            observation = ScalarVisualObservation(
                predicate_id=predicate.predicate_id,
                visible=usable,
                value=float(len(grouped.groups)) if usable else None,
                confidence=grouped.confidence,
                evidence=(
                    "; ".join(grouped.groups)
                    if grouped.groups
                    else "target groups are not reliably visible"
                ),
            )
            report = evaluate_scalar_visual_predicate(
                predicate,
                observation,
                minimum_confidence=self.minimum_confidence,
            )
            metadata = dict(report.metadata)
            metadata["groups"] = list(grouped.groups)
            metadata.update({
                "measurement_scope": "visual_group_progress_only",
                "task_completion_authorized": False,
                "expected_object_count": self.expected_object_count,
                "coverage_valid": coverage_valid,
                "member_counts": list(grouped.member_counts),
                "group_relations": list(grouped.group_relations),
                "all_targets_visible": grouped.all_targets_visible,
                "observed_group_count": len(grouped.groups) if grouped.visible else None,
            })
            report = VerificationReport(
                status=report.status,
                observed_outcome=report.observed_outcome,
                confidence=report.confidence,
                metadata=metadata,
            )
            accepted = True
            self.accepted_calls += 1
        except Exception as exc:
            error = str(exc)
            accepted = False
            report = VerificationReport(
                status=VerificationStatus.INCONCLUSIVE,
                observed_outcome=f"grouped visual verifier unavailable: {error}",
                confidence=0.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_latency_ms += elapsed_ms
        return VisualVerificationResult(
            report=report,
            accepted=accepted,
            elapsed_ms=elapsed_ms,
            error=error,
            raw_output=raw_output,
        )

    def metrics(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "accepted_calls": self.accepted_calls,
            "total_latency_ms": self.total_latency_ms,
            "mean_latency_ms": (
                0.0 if self.calls == 0 else self.total_latency_ms / self.calls
            ),
        }


def _paired_visual_change(frames: Mapping[str, Any]) -> float | None:
    """Return the largest mean absolute difference across named view pairs."""

    import numpy as np

    changes: list[float] = []
    for name, before in frames.items():
        if not name.startswith("before_"):
            continue
        current = frames.get("current_" + name.removeprefix("before_"))
        if current is None:
            continue
        before_array = np.asarray(before, dtype=np.float32)
        current_array = np.asarray(current, dtype=np.float32)
        if before_array.shape != current_array.shape:
            continue
        changes.append(float(np.mean(np.abs(current_array - before_array))))
    return None if not changes else max(changes)
