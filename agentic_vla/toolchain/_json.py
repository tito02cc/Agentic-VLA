"""Bounded JSON envelopes for model-selected observations, without repair."""

import json
from collections.abc import Mapping
from typing import Any


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def decode_json_object(raw: Any, *, max_chars: int = 16384) -> Mapping[str, Any]:
    """Accept one bare object or one complete JSON fence; never extract prose."""
    candidate = raw
    if isinstance(candidate, str):
        if len(candidate) > max_chars:
            raise ValueError("model JSON exceeds response budget")
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if (
                len(lines) < 3
                or lines[0].strip().lower() not in {"```", "```json"}
                or lines[-1].strip() != "```"
            ):
                raise ValueError("model JSON code block is malformed or incomplete")
            text = "\n".join(lines[1:-1]).strip()
            if "```" in text:
                raise ValueError("model output contains multiple code blocks")
        candidate = json.loads(
            text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant
        )
    if not isinstance(candidate, Mapping):
        raise ValueError("model output must be a JSON object")  # noqa: TRY004 - JSON shape validation
    return candidate
