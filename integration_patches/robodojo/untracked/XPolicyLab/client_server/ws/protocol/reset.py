"""Versioned contracts for auditable episode-reset messages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


RESET_RECEIPT_SCHEMA_VERSION = "xpolicylab.episode-reset.v1"
RESET_CAPABILITY_NAME = "episode_reset_receipt"


def canonical_reset_context(reset_context: Mapping[str, Any] | None) -> bytes:
    """Return a deterministic representation or fail on non-JSON context."""

    if reset_context is None:
        reset_context = {}
    if not isinstance(reset_context, Mapping):
        raise TypeError("reset_context must be a mapping or None")
    try:
        encoded = json.dumps(
            dict(reset_context),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reset_context must contain finite JSON values") from exc
    return encoded.encode("utf-8")


def reset_context_digest(reset_context: Mapping[str, Any] | None) -> str:
    return hashlib.sha256(canonical_reset_context(reset_context)).hexdigest()


def reset_event_id(
    *, server_instance_id: str, server_reset_generation: int, context_digest: str
) -> str:
    material = (
        f"{server_instance_id}:{server_reset_generation}:{context_digest}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _require_nonempty_string(receipt: Mapping[str, Any], field: str) -> str:
    value = receipt.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"reset receipt {field} must be a non-empty string")
    return value


def _require_sha256(receipt: Mapping[str, Any], field: str) -> str:
    value = _require_nonempty_string(receipt, field)
    if len(value) != 64:
        raise ValueError(f"reset receipt {field} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"reset receipt {field} must be a SHA256 hex digest"
        ) from exc
    return value


def validate_reset_receipt(
    receipt: Any,
    reset_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the server's reset attestation before committing the boundary.

    The guarantee is at-least-once per physical episode, not exactly-once: a
    crash between this response and the client's durable acknowledgement makes
    the next attempt allocate a fresh token, so that episode is reset again and
    only the last reset is the attested one. What this function checks is that a
    reset was applied for exactly this context, and that the response is either
    that application or a replay of it.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("policy reset did not return a receipt mapping")
    validated = dict(receipt)
    if validated.get("reset_receipt_schema_version") != RESET_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported reset receipt schema")

    if reset_context is None:
        context = {}
    elif not isinstance(reset_context, Mapping):
        raise TypeError("reset_context must be a mapping or None")
    else:
        context = dict(reset_context)
    expected_context_digest = reset_context_digest(context)
    if validated.get("context_digest") != expected_context_digest:
        raise ValueError("reset receipt context digest mismatch")

    # The digest binds the entire context, while explicit echo checks make the
    # episode identity and environment assignment independently auditable.  A
    # canonical JSON comparison keeps booleans distinct from integers (Python's
    # normal equality would otherwise consider True equal to 1). The server
    # normalizes absent list-valued legacy context fields to empty lists.
    for field in (
        "episode_id",
        "episode_seq",
        "reset_session_id",
        "active_env_ids",
        "layout_seeds",
    ):
        if field not in validated:
            raise ValueError(f"reset receipt is missing context echo {field}")
        expected_value = context.get(field)
        if field in {"active_env_ids", "layout_seeds"} and field not in context:
            expected_value = []
        try:
            echoed = canonical_reset_context({"value": validated[field]})
            expected = canonical_reset_context({"value": expected_value})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reset receipt context echo {field} is not finite JSON"
            ) from exc
        if echoed != expected:
            raise ValueError(f"reset receipt {field} mismatch")

    for field in (
        "applied_once",
        "applied_this_request",
        "response_replayed",
    ):
        if type(validated.get(field)) is not bool:
            raise ValueError(f"reset receipt {field} must be a boolean")
    if validated["applied_once"] is not True:
        raise ValueError("reset receipt does not prove one applied reset")
    valid_response_state = (
        validated["applied_this_request"] is True
        and validated["response_replayed"] is False
    ) or (
        validated["applied_this_request"] is False
        and validated["response_replayed"] is True
    )
    if not valid_response_state:
        raise ValueError("reset receipt has an inconsistent replay state")

    generation = validated.get("server_reset_generation")
    if type(generation) is not int or generation <= 0:
        raise ValueError("reset receipt server_reset_generation must be positive")
    server_instance_id = _require_nonempty_string(
        validated, "server_instance_id"
    )
    _require_nonempty_string(validated, "model_module_id")
    _require_sha256(validated, "model_code_sha256")
    event_id = _require_sha256(validated, "reset_event_id")
    expected_event_id = reset_event_id(
        server_instance_id=server_instance_id,
        server_reset_generation=generation,
        context_digest=expected_context_digest,
    )
    if event_id != expected_event_id:
        raise ValueError("reset receipt event id mismatch")
    return validated
