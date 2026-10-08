"""Action-free snapshots of ordered, cumulative visual task predicates."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping
from typing import Any

from agentic_vla.runtime.agent import PlannerCallable

from ._json import decode_json_object
from .contracts import VerificationReport, VerificationStatus
from .verifier import _reject_internally_inconsistent_confirmation


@dataclasses.dataclass(frozen=True)
class VisualStagePredicate:
    stage_id: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("visual stage id must be a nonempty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("visual stage description must be a nonempty string")


@dataclasses.dataclass(frozen=True)
class VisualProgressContext:
    task_instruction: str
    predicates: tuple[VisualStagePredicate, ...]
    frames: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.task_instruction.strip():
            raise ValueError("task instruction must not be empty")
        if not 1 <= len(self.predicates) <= 8:
            raise ValueError("progress snapshot requires one to eight predicates")
        if any(not isinstance(item, VisualStagePredicate) for item in self.predicates):
            raise TypeError("unsupported visual stage predicate")
        ids = [item.stage_id for item in self.predicates]
        if len(set(ids)) != len(ids):
            raise ValueError("visual stage ids must be unique")
        if not 1 <= len(self.frames) <= 3:
            raise ValueError("progress snapshot requires one to three camera frames")


@dataclasses.dataclass(frozen=True)
class VisualProgressResult:
    reports: Mapping[str, VerificationReport]
    confirmed_prefix: tuple[str, ...]
    accepted: bool
    elapsed_ms: float
    error: str | None = None
    raw_output: Any | None = None


def build_visual_progress_request(context: VisualProgressContext) -> dict[str, Any]:
    return {
        "system_prompt": (
            "Inspect the current robot camera image for EACH listed cumulative visual "
            "predicate independently. Describe visible geometry, support and placement, "
            "not intentions, actions, task success, or physical stability. Return JSON only: "
            '{"stages":[{"stage_id":"...","state":"present|absent|unknown",'
            '"confidence":0.9,"evidence":"short visible evidence"}]}. '
            "Return every requested stage_id exactly once, with no extra fields. "
            "present means the entire predicate is visibly supported now; absent means "
            "visible geometry clearly lacks the required structure; unknown means occlusion "
            "or ambiguity prevents deciding. A later stage does not excuse checking earlier "
            "predicates. Do not assume progress from stage order. Partial progress is NOT "
            "a failure and does not request a retry. A robot holding a piece near its target "
            "does not establish that it rests on that target. Never infer hidden state, "
            "simulator poses, rewards, contact forces, or dynamic stability from a still "
            "image. confidence must be a JSON number in [0,1]. evidence must be at most "
            "16 words per stage; present evidence must state a positive visible relation."
        ),
        "user_prompt": json.dumps(
            {
                "task_instruction": context.task_instruction,
                "predicates": [dataclasses.asdict(item) for item in context.predicates],
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "frames": dict(context.frames),
        "required_frame_names": tuple(context.frames),
    }


def _parse_progress(raw: Any, context: VisualProgressContext, threshold: float):
    candidate = decode_json_object(raw)
    if not isinstance(candidate, Mapping) or set(candidate) != {"stages"}:
        raise ValueError("progress output must contain only stages")
    stages = candidate["stages"]
    expected = {item.stage_id for item in context.predicates}
    if not isinstance(stages, list) or len(stages) != len(expected):
        raise ValueError("progress output must cover every requested stage")
    reports = {}
    for row in stages:
        if not isinstance(row, Mapping) or set(row) != {
            "stage_id",
            "state",
            "confidence",
            "evidence",
        }:
            raise ValueError("progress stage has missing or unknown fields")
        stage = row["stage_id"]
        if not isinstance(stage, str) or stage not in expected or stage in reports:
            raise ValueError("progress stage id is unknown or duplicated")
        state, confidence, evidence = row["state"], row["confidence"], row["evidence"]
        if not isinstance(state, str) or state not in {"present", "absent", "unknown"}:
            raise ValueError("unsupported visual state")
        if (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be a finite JSON number in [0,1]")
        if (
            not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 1024
        ):
            raise ValueError("visual evidence must be nonempty and bounded")
        # Absence or uncertainty never grants retry, failure, or safe-stop authority.
        report = VerificationReport(
            status=VerificationStatus.CONFIRMED
            if state == "present" and confidence >= threshold
            else VerificationStatus.INCONCLUSIVE,
            observed_outcome=evidence,
            confidence=float(confidence),
            metadata={"visual_state": state, "stage_id": stage},
        )
        _reject_internally_inconsistent_confirmation(report)
        reports[stage] = report
    ordered = {item.stage_id: reports[item.stage_id] for item in context.predicates}
    prefix = []
    for stage, report in ordered.items():
        if report.status is not VerificationStatus.CONFIRMED:
            break
        prefix.append(stage)
    return ordered, tuple(prefix)


class GuardedProgressVerifier:
    """One bounded VLM call, no ledger mutations or robot control authority."""

    def __init__(self, infer: PlannerCallable, *, minimum_confidence: float = 0.55):
        if not callable(infer):
            raise TypeError("progress infer must be callable")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be in [0,1]")
        self.infer = infer
        self.minimum_confidence = float(minimum_confidence)

    def verify(self, context: VisualProgressContext) -> VisualProgressResult:
        started = time.perf_counter()
        raw, error = None, None
        try:
            raw = self.infer(build_visual_progress_request(context))
            reports, prefix = _parse_progress(raw, context, self.minimum_confidence)
            accepted = True
        except Exception as exc:  # noqa: BLE001 - provider errors must leave progress unconfirmed
            error = str(exc)
            accepted, prefix = False, ()
            reports = {
                item.stage_id: VerificationReport(
                    status=VerificationStatus.INCONCLUSIVE,
                    observed_outcome="progress verification unavailable",
                    confidence=0.0,
                )
                for item in context.predicates
            }
        return VisualProgressResult(
            reports=reports,
            confirmed_prefix=prefix,
            accepted=accepted,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=error,
            raw_output=raw,
        )
