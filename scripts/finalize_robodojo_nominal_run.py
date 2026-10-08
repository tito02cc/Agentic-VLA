#!/usr/bin/env python3
"""Package one official RoboDojo nominal run with CARVE traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBODOJO_ROOT = _REPO_ROOT / "third_party" / "robodojo_official"


def _load_contract_module(name: str, relative_path: str) -> ModuleType:
    """Load one reset-contract module straight from the vendored source file.

    The audit only needs the contract definitions, which are pure stdlib. Their
    packages are not importable here: ``XPolicyLab/client_server/ws/__init__.py``
    re-exports the websocket client with checkout-rooted absolute imports and
    pulls in the msgpack wire codec, which this analysis venv does not install.
    Importing by path keeps a single source of truth for the schema without
    dragging the transport stack (and its dependencies) into the finalizer.
    """
    path = _ROBODOJO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the reset contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reset_protocol = _load_contract_module(
    "_carve_reset_protocol",
    "XPolicyLab/client_server/ws/protocol/reset.py",
)
_starvla_reset_contract = _load_contract_module(
    "_carve_starvla_reset_contract",
    "XPolicyLab/policy/starVLA/reset_contract.py",
)

RESET_RECEIPT_SCHEMA_VERSION = _reset_protocol.RESET_RECEIPT_SCHEMA_VERSION
validate_reset_receipt = _reset_protocol.validate_reset_receipt
STARVLA_RESET_STATE_FIELDS = _starvla_reset_contract.STARVLA_RESET_STATE_FIELDS
STARVLA_RESET_STATE_VERSION = _starvla_reset_contract.STARVLA_RESET_STATE_VERSION
validate_starvla_reset_state = _starvla_reset_contract.validate_starvla_reset_state


MODEL_RESET_SCHEMA_VERSION = "carve.policy-reset.v1"
MODEL_MODULE_ID = "XPolicyLab.policy.starVLA.model"
# Digest of the contract file this audit loaded its expectations from. It is
# compared against the launcher-recorded value so the expectation side of the
# audit is pinned too, not only the receipt side.
_RESET_CONTRACT_SHA256 = hashlib.sha256(
    (_ROBODOJO_ROOT / "XPolicyLab/policy/starVLA/reset_contract.py").read_bytes()
).hexdigest()
POLICY_RESET_LEDGER_SCHEMA_VERSION = "robodojo.policy-reset-ledger.v1"
POLICY_RESET_AUDIT_SCHEMA_VERSION = "carve.robodojo.policy-reset-audit.v2"
RUN_COMPLETENESS_SCHEMA_VERSION = "carve.robodojo.run-completeness.v1"
_RESET_CONTEXT_FIELDS = (
    "episode_id",
    "episode_seq",
    "reset_session_id",
    "active_env_ids",
    "layout_seeds",
)
_MISSING = object()


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_audit_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Read reset evidence without losing diagnostics on one malformed row."""
    if not path.is_file():
        return [], []
    rows: list[dict] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError) as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, Mapping):
            errors.append(f"{path.name}:{line_number}: receipt row is not an object")
            continue
        rows.append(dict(row))
    return rows, errors


def _read_audit_mapping(path: Path) -> tuple[dict | None, str | None]:
    """Read one audit input as a JSON object, returning an attributable error."""
    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (TypeError, ValueError) as exc:
        return None, f"{path.name}: invalid JSON: {exc}"
    if not isinstance(value, Mapping):
        return None, f"{path.name}: top-level JSON value is not an object"
    return dict(value), None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _config_value(
    config: Mapping[str, Any] | None,
    paths: tuple[tuple[str, ...], ...],
) -> tuple[Any, str | None]:
    if config is None:
        return _MISSING, None
    for path in paths:
        value: Any = config
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value, ".".join(path)
    return _MISSING, None


def _deduplicated_errors(errors: list[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def _audit_run_completeness(
    result: Mapping[str, Any],
    episodes: list[dict],
    *,
    expected_episodes: int | None,
) -> dict:
    """Check that this condition actually produced the episodes it claims.

    The reset audit deliberately tolerates a reset event with no scored episode,
    because RoboDojo abandons physical episodes after their reset for reasons that
    do not weaken per-episode isolation. That tolerance removes the only thing
    that used to fail a truncated condition, so completeness needs its own gate:
    a run whose retries were exhausted mid-matrix would otherwise enter the
    confirmatory aggregation with a perfectly valid reset audit and fewer episodes
    than the protocol pre-registered.
    """

    episode_count = len(episodes)
    successes = sum(episode["success"] for episode in episodes)
    reported_eval_time = result.get("eval_time")
    reported_success_rate = result.get("success_rate")
    errors: list[str] = []
    invariants: dict[str, bool] = {}

    def check(name: str, condition: Any, message: str) -> None:
        passed = bool(condition)
        invariants[name] = passed
        if not passed:
            errors.append(f"{name}: {message}")

    check(
        "expected_episodes_declared",
        expected_episodes is not None,
        "--expected-episodes is required to certify a pre-registered block",
    )
    check(
        "episode_count_matches_expected",
        expected_episodes is None or episode_count == expected_episodes,
        f"summarized {episode_count} episodes, expected {expected_episodes}",
    )
    check(
        "official_eval_time_matches_details",
        type(reported_eval_time) is int and reported_eval_time == episode_count,
        f"official eval_time={reported_eval_time!r} disagrees with "
        f"{episode_count} scored details",
    )
    recomputed_rate = successes / episode_count if episode_count else None
    check(
        "official_success_rate_matches_details",
        recomputed_rate is not None
        and isinstance(reported_success_rate, (int, float))
        and not isinstance(reported_success_rate, bool)
        and math.isclose(
            float(reported_success_rate), recomputed_rate, rel_tol=0.0, abs_tol=1e-9
        ),
        f"official success_rate={reported_success_rate!r} disagrees with "
        f"{successes}/{episode_count} recomputed from details",
    )

    return {
        "schema_version": RUN_COMPLETENESS_SCHEMA_VERSION,
        "valid": all(invariants.values()),
        "expected_episodes": expected_episodes,
        "summarized_episodes": episode_count,
        "official_eval_time": reported_eval_time,
        "successes": successes,
        "recomputed_success_rate": recomputed_rate,
        "official_success_rate": reported_success_rate,
        "invariants": invariants,
        "errors": errors,
        **invariants,
    }


def _audit_policy_resets(
    details: list[dict],
    reset_trace: list[dict],
    *,
    reset_trace_present: bool,
    ledger: dict | None,
    ledger_present: bool,
    run_config: dict | None,
    run_config_source: str | None,
    input_errors: list[str],
) -> dict:
    """Validate reset evidence at the unique server event boundary.

    One physical reset can cover several parallel environments, so detail
    receipts are grouped by ``reset_event_id`` before sequence, trace, and
    ledger invariants are evaluated.
    """

    errors = list(input_errors)
    invariants: dict[str, bool] = {}

    def add_error(message: str) -> None:
        errors.append(message)

    def check(name: str, condition: Any, message: str) -> bool:
        passed = bool(condition)
        invariants[name] = passed
        if not passed:
            add_error(f"{name}: {message}")
        return passed

    require_audited_reset, require_path = _config_value(
        run_config,
        (
            ("require_audited_reset",),
            ("policy_reset", "require_audited_reset"),
            ("reset_audit", "require_audited_reset"),
        ),
    )
    expected_module, expected_module_path = _config_value(
        run_config,
        (
            ("expected_model_module_id",),
            ("expected_model_module",),
            ("policy_reset", "expected_model_module_id"),
            ("policy_reset", "expected", "model_module_id"),
            ("reset_audit", "expected_model_module_id"),
        ),
    )
    expected_hash, expected_hash_path = _config_value(
        run_config,
        (
            ("expected_model_code_sha256",),
            ("expected_model_sha256",),
            ("expected_model_hash",),
            ("policy_reset", "expected_model_code_sha256"),
            ("policy_reset", "expected", "model_code_sha256"),
            ("reset_audit", "expected_model_code_sha256"),
        ),
    )
    expected_model_schema, expected_model_schema_path = _config_value(
        run_config,
        (
            ("expected_model_reset_schema_version",),
            ("expected_model_reset_schema",),
            ("expected_model_schema_version",),
            ("policy_reset", "expected_model_reset_schema_version"),
            ("policy_reset", "expected", "model_reset_schema_version"),
            ("reset_audit", "expected_model_reset_schema_version"),
        ),
    )
    expected_receipt_schema, expected_receipt_schema_path = _config_value(
        run_config,
        (
            ("expected_reset_receipt_schema_version",),
            ("policy_reset", "expected_reset_receipt_schema_version"),
            ("policy_reset", "expected", "reset_receipt_schema_version"),
            ("reset_audit", "expected_reset_receipt_schema_version"),
        ),
    )
    expected_state_schema, expected_state_schema_path = _config_value(
        run_config,
        (
            ("expected_state_inventory_version",),
            ("policy_reset", "expected_state_inventory_version"),
            ("policy_reset", "expected", "state_inventory_version"),
            ("reset_audit", "expected_state_inventory_version"),
        ),
    )
    # The official RoboDojo output directory is shared across conditions, so a
    # condition that exits before creating its own run directory would otherwise
    # be finalized against the previous condition's evidence. That evidence is
    # internally consistent, so only the run id can catch the mislabelling.
    expected_run_id, expected_run_id_path = _config_value(
        run_config,
        (
            ("robodojo_run_id",),
            ("run_id",),
            ("policy_reset", "robodojo_run_id"),
        ),
    )
    # The inventory now lives only in reset_contract.py, which is outside
    # the model_code_sha256 scope and is also the file this audit imports to build
    # its expectations. Pinning it to the launcher-recorded digest keeps an
    # equal-count substitution in that file from changing receipt and expectation
    # together and silently.
    expected_contract_hash, expected_contract_hash_path = _config_value(
        run_config,
        (
            ("expected_reset_contract_sha256",),
            ("policy_reset", "expected_reset_contract_sha256"),
        ),
    )

    check(
        "audit_inputs_parseable",
        not input_errors,
        "one or more reset audit inputs could not be parsed",
    )
    check(
        "run_config_present",
        run_config_source is not None,
        "run_config.json is required for an audited run",
    )
    check(
        "run_config_requires_audited_reset",
        require_path is not None and require_audited_reset is True,
        "require_audited_reset must exist and be exactly true",
    )
    check(
        "expected_model_identity_present",
        expected_module_path is not None
        and expected_hash_path is not None
        and expected_model_schema_path is not None,
        "expected model module, code hash, and reset schema must all be recorded",
    )
    check(
        "expected_model_module_matches_fixed_contract",
        expected_module == MODEL_MODULE_ID,
        f"expected model module must be {MODEL_MODULE_ID}",
    )
    check(
        "expected_model_schema_matches_fixed_contract",
        expected_model_schema == MODEL_RESET_SCHEMA_VERSION,
        f"expected model reset schema must be {MODEL_RESET_SCHEMA_VERSION}",
    )
    check(
        "expected_model_hash_is_sha256",
        _is_sha256(expected_hash),
        "expected model code hash must be a SHA256 hex digest",
    )
    check(
        "expected_receipt_schema_matches_fixed_contract",
        expected_receipt_schema_path is None
        or expected_receipt_schema == RESET_RECEIPT_SCHEMA_VERSION,
        f"configured receipt schema must be {RESET_RECEIPT_SCHEMA_VERSION}",
    )
    check(
        "expected_state_schema_matches_fixed_contract",
        expected_state_schema_path is None
        or expected_state_schema == STARVLA_RESET_STATE_VERSION,
        f"configured state inventory version must be {STARVLA_RESET_STATE_VERSION}",
    )
    check(
        "reset_contract_hash_matches_run_config",
        expected_contract_hash_path is not None
        and _is_sha256(expected_contract_hash)
        and expected_contract_hash == _RESET_CONTRACT_SHA256,
        "run_config must pin the reset-contract digest this audit loaded "
        f"({_RESET_CONTRACT_SHA256})",
    )

    groups: dict[str, list[tuple[int, dict, dict]]] = {}
    event_order: list[str] = []
    detail_receipt_count = 0
    detail_receipts_complete = True
    detail_event_ids_present = True
    for episode_index, detail in enumerate(details):
        receipt = detail.get("policy_reset_receipt")
        if not isinstance(receipt, Mapping):
            detail_receipts_complete = False
            detail_event_ids_present = False
            add_error(
                f"episode {episode_index}: policy_reset_receipt is missing or not an object"
            )
            continue
        detail_receipt_count += 1
        receipt_dict = dict(receipt)
        event_id = receipt_dict.get("reset_event_id")
        if not isinstance(event_id, str) or not event_id:
            detail_event_ids_present = False
            add_error(f"episode {episode_index}: reset_event_id is missing")
            continue
        if event_id not in groups:
            groups[event_id] = []
            event_order.append(event_id)
        groups[event_id].append((episode_index, detail, receipt_dict))

    check(
        "detail_receipts_present_for_all_episodes",
        detail_receipts_complete and detail_receipt_count == len(details),
        "every official detail must carry a reset receipt",
    )
    check(
        "detail_reset_event_ids_present",
        detail_event_ids_present,
        "every detail receipt must identify its reset event",
    )
    check(
        "unique_reset_events_present",
        bool(event_order),
        "no unique reset event could be reconstructed from details",
    )

    representative_receipts: list[dict] = []
    event_summaries: list[dict] = []
    groups_identical = bool(event_order)
    batch_contexts_well_formed = bool(event_order)
    active_env_ids_match = bool(event_order)
    batch_layouts_match = bool(event_order)

    for event_id in event_order:
        records = groups[event_id]
        fingerprints: list[str] = []
        canonical_failed = False
        for episode_index, _, receipt in records:
            try:
                fingerprints.append(_json_fingerprint(receipt))
            except (TypeError, ValueError) as exc:
                canonical_failed = True
                add_error(
                    f"episode {episode_index}: receipt is not canonical JSON: {exc}"
                )
        group_identical = (
            not canonical_failed
            and len(fingerprints) == len(records)
            and len(set(fingerprints)) == 1
        )
        if not group_identical:
            groups_identical = False
            add_error(
                f"reset event {event_id}: detail receipts are not fully identical"
            )

        representative = records[0][2]
        representative_receipts.append(representative)
        active_env_ids = representative.get("active_env_ids")
        layout_seeds = representative.get("layout_seeds")
        active_valid = (
            isinstance(active_env_ids, list)
            and all(type(value) is int and value >= 0 for value in active_env_ids)
            and len(set(active_env_ids)) == len(active_env_ids)
        )
        rows_valid = isinstance(layout_seeds, list) and all(
            isinstance(row, Mapping)
            and set(row) == {"env_idx", "layout_id"}
            and type(row.get("env_idx")) is int
            and row["env_idx"] >= 0
            and type(row.get("layout_id")) is int
            for row in (layout_seeds if isinstance(layout_seeds, list) else [])
        )
        episode_layouts = [detail.get("layout_id") for _, detail, _ in records]
        detail_layouts_valid = all(type(value) is int for value in episode_layouts)
        context_valid = active_valid and rows_valid and detail_layouts_valid
        if not context_valid:
            batch_contexts_well_formed = False
            add_error(f"reset event {event_id}: malformed batch reset context")

        layout_env_ids = (
            [row["env_idx"] for row in layout_seeds] if rows_valid else []
        )
        layout_ids = (
            [row["layout_id"] for row in layout_seeds] if rows_valid else []
        )
        active_matches_event = (
            context_valid
            and active_env_ids == layout_env_ids
            and len(set(layout_env_ids)) == len(layout_env_ids)
        )
        if not active_matches_event:
            active_env_ids_match = False
            add_error(
                f"reset event {event_id}: active_env_ids do not match layout_seeds"
            )
        layouts_match_event = (
            context_valid and Counter(episode_layouts) == Counter(layout_ids)
        )
        if not layouts_match_event:
            batch_layouts_match = False
            add_error(
                f"reset event {event_id}: layout_seeds do not match grouped official episodes"
            )

        event_summaries.append(
            {
                "reset_event_id": event_id,
                "episode_id": representative.get("episode_id"),
                "episode_seq": representative.get("episode_seq"),
                "reset_session_id": representative.get("reset_session_id"),
                "server_instance_id": representative.get("server_instance_id"),
                "server_reset_generation": representative.get(
                    "server_reset_generation"
                ),
                "detail_episode_indexes": [index for index, _, _ in records],
                "detail_layout_ids": episode_layouts,
                "active_env_ids": active_env_ids,
                "layout_seeds": layout_seeds,
                "receipt_fingerprint": (
                    fingerprints[0] if fingerprints else None
                ),
                "detail_receipt_fingerprints": fingerprints,
                "response_replayed": representative.get("response_replayed"),
            }
        )

    check(
        "detail_receipts_identical_per_event",
        groups_identical,
        "all copies of a batch receipt must have identical canonical JSON",
    )
    check(
        "batch_contexts_well_formed",
        batch_contexts_well_formed,
        "active_env_ids and layout_seeds must use the strict batch shape",
    )
    check(
        "active_env_ids_match_layout_seeds",
        active_env_ids_match,
        "active environment ids must exactly match layout seed env_idx values",
    )
    check(
        "batch_layouts_match_episode_details",
        batch_layouts_match,
        "each event's layout seeds must exactly cover its grouped detail layouts",
    )

    # Identity, ordering, continuity and the receipt contract are properties of
    # every reset the server applied, and reset_receipts.jsonl is the record of
    # those. Scored episodes are a strict subset: RoboDojo abandons a physical
    # episode after its reset already happened (PhysX broken env, PhysX fatal
    # restart, an unstable env dropped from eval_envs, or any mid-rollout
    # exception that consumes the seed), which legitimately leaves a reset event
    # with no scored detail. Auditing over the scored subset alone would fail an
    # otherwise valid run and force a full re-run of the condition.
    audit_basis = reset_trace if reset_trace_present else representative_receipts
    # Scored receipts are always held to the contract even when the trace is
    # missing, so an absent trace cannot silently skip contract validation. When
    # the trace is present it already contains every applied reset, and the scored
    # receipts are checked to be a byte-identical subsequence of it, so adding
    # them again would only duplicate work and errors.
    if reset_trace_present:
        contract_receipts = list(reset_trace)
    else:
        contract_receipts = list(representative_receipts)

    generic_valid = bool(contract_receipts)
    starvla_valid = bool(contract_receipts)
    wire_schema_valid = bool(contract_receipts)
    model_schema_valid = bool(contract_receipts)
    inventory_complete = bool(contract_receipts)
    module_valid = bool(contract_receipts)
    model_hash_valid = bool(contract_receipts) and _is_sha256(expected_hash)
    replay_state_valid = bool(contract_receipts)

    # Every reset the server applied is held to the contract, not just the ones
    # that produced a scored episode. An abandoned attempt cannot weaken the
    # isolation of the scored episodes, but a receipt claiming a different
    # adapter, schema or non-empty state is evidence that the run was not the
    # configuration it says it was, so it must still fail the audit.
    for receipt in contract_receipts:
        event_id = str(receipt.get("reset_event_id"))
        reset_context = {field: receipt.get(field) for field in _RESET_CONTEXT_FIELDS}
        try:
            validate_reset_receipt(receipt, reset_context)
        except Exception as exc:
            generic_valid = False
            add_error(f"reset event {event_id}: generic receipt validation failed: {exc}")
        try:
            validate_starvla_reset_state(receipt)
        except Exception as exc:
            starvla_valid = False
            add_error(f"reset event {event_id}: StarVLA state validation failed: {exc}")

        if receipt.get("reset_receipt_schema_version") != RESET_RECEIPT_SCHEMA_VERSION:
            wire_schema_valid = False
        if receipt.get("model_reset_schema_version") != MODEL_RESET_SCHEMA_VERSION:
            model_schema_valid = False
        if not (
            receipt.get("state_inventory_version")
            == STARVLA_RESET_STATE_VERSION
            and receipt.get("state_inventory")
            == list(STARVLA_RESET_STATE_FIELDS)
        ):
            inventory_complete = False
        if receipt.get("model_module_id") != MODEL_MODULE_ID:
            module_valid = False
        if receipt.get("model_code_sha256") != expected_hash:
            model_hash_valid = False
        if not (
            receipt.get("applied_once") is True
            and (
                (
                    receipt.get("applied_this_request") is True
                    and receipt.get("response_replayed") is False
                )
                or (
                    receipt.get("applied_this_request") is False
                    and receipt.get("response_replayed") is True
                )
            )
        ):
            replay_state_valid = False

    check(
        "generic_reset_receipts_valid",
        generic_valid,
        "validate_reset_receipt must accept every unique event",
    )
    check(
        "starvla_reset_states_valid",
        starvla_valid,
        "validate_starvla_reset_state must accept every unique event",
    )
    check(
        "reset_receipt_schema_matches",
        wire_schema_valid,
        f"every receipt must use {RESET_RECEIPT_SCHEMA_VERSION}",
    )
    check(
        "model_reset_schema_matches",
        model_schema_valid,
        f"every receipt must use {MODEL_RESET_SCHEMA_VERSION}",
    )
    check(
        "state_inventory_complete",
        inventory_complete,
        f"every receipt must carry the complete ordered {len(STARVLA_RESET_STATE_FIELDS)}-item StarVLA inventory",
    )
    check(
        "model_module_id_matches",
        module_valid,
        f"every receipt must identify {MODEL_MODULE_ID}",
    )
    check(
        "model_code_hash_matches_run_config",
        model_hash_valid,
        "every receipt model hash must match the expected run_config hash",
    )
    check(
        "receipt_replay_states_valid",
        replay_state_valid,
        "each event must be either newly applied or a valid replay response",
    )

    server_instances = [
        receipt.get("server_instance_id") for receipt in audit_basis
    ]
    reset_sessions = [
        receipt.get("reset_session_id") for receipt in audit_basis
    ]
    episode_ids = [receipt.get("episode_id") for receipt in audit_basis]
    episode_sequences = [
        receipt.get("episode_seq") for receipt in audit_basis
    ]
    server_generations = [
        receipt.get("server_reset_generation") for receipt in audit_basis
    ]

    server_instance_consistent = bool(server_instances) and all(
        isinstance(value, str) and value for value in server_instances
    ) and len(set(server_instances)) == 1
    reset_session_consistent = bool(reset_sessions) and all(
        isinstance(value, str) and value for value in reset_sessions
    ) and len(set(reset_sessions)) == 1
    episode_ids_unique = bool(episode_ids) and all(
        isinstance(value, str) and value for value in episode_ids
    ) and len(set(episode_ids)) == len(episode_ids)
    sequences_valid = bool(episode_sequences) and all(
        type(value) is int and value > 0 for value in episode_sequences
    )
    sequences_ordered = sequences_valid and all(
        later > earlier
        for earlier, later in zip(
            episode_sequences, episode_sequences[1:], strict=False
        )
    )

    run_ids: list[str] = []
    episode_ids_bound = bool(audit_basis)
    for episode_id, sequence, session_id in zip(
        episode_ids, episode_sequences, reset_sessions, strict=False
    ):
        if not (
            isinstance(episode_id, str)
            and type(sequence) is int
            and sequence > 0
            and isinstance(session_id, str)
            and session_id
        ):
            episode_ids_bound = False
            continue
        suffix = f":{session_id}:batch-{sequence:07d}"
        if not episode_id.endswith(suffix) or len(episode_id) <= len(suffix):
            episode_ids_bound = False
            continue
        run_ids.append(episode_id[: -len(suffix)])
    run_id_consistent = (
        episode_ids_bound
        and len(run_ids) == len(audit_basis)
        and len(set(run_ids)) == 1
    )
    # Exactly 1..N over every applied reset. A gap would mean the server applied
    # a reset this run never recorded; a duplicate would mean two records share
    # one application. Abandoned episodes keep their generation, so they occupy a
    # slot here instead of punching a hole in the sequence.
    server_generation_exact = server_generations == list(
        range(1, len(audit_basis) + 1)
    )

    check(
        "server_instance_consistent",
        server_instance_consistent,
        "all applied resets must come from one exclusive server instance",
    )
    check(
        "reset_session_consistent",
        reset_session_consistent,
        "all reset events must use one non-empty reset session",
    )
    check(
        "episode_ids_unique",
        episode_ids_unique,
        "unique reset events must have unique episode ids",
    )
    check(
        "episode_sequences_unique_and_increasing",
        sequences_ordered,
        "episode_seq values must be unique and increase in event order",
    )
    check(
        "episode_ids_match_run_session_sequence",
        run_id_consistent,
        "episode ids must bind one run/session to their exact episode sequence",
    )
    check(
        "run_id_matches_run_config",
        expected_run_id_path is not None
        and isinstance(expected_run_id, str)
        and bool(expected_run_id)
        and run_id_consistent
        and run_ids[0] == expected_run_id,
        "the evidence must come from the run this artifact was configured for, "
        "not from another condition's official output directory",
    )
    check(
        "server_reset_generations_exact_1_to_n",
        server_generation_exact,
        "exclusive-server generations must be exactly 1..N with no gaps or duplicates",
    )

    expected_canonical: list[bytes] = []
    expected_fingerprints: list[str] = []
    trace_canonical: list[bytes] = []
    trace_fingerprints: list[str] = []
    # Row indexes are carried alongside the canonical bytes: a row that cannot be
    # canonicalized is skipped, so positions in trace_canonical would otherwise
    # drift from positions in reset_trace and misattribute orphan event ids.
    trace_row_indexes: list[int] = []
    trace_canonical_valid = True
    for receipt in representative_receipts:
        try:
            canonical = _canonical_json(receipt)
        except (TypeError, ValueError) as exc:
            trace_canonical_valid = False
            add_error(f"detail receipt cannot be canonicalized: {exc}")
            continue
        expected_canonical.append(canonical)
        expected_fingerprints.append(hashlib.sha256(canonical).hexdigest())
    for row_index, receipt in enumerate(reset_trace):
        try:
            canonical = _canonical_json(receipt)
        except (TypeError, ValueError) as exc:
            trace_canonical_valid = False
            add_error(
                f"reset trace row {row_index + 1} cannot be canonicalized: {exc}"
            )
            continue
        trace_canonical.append(canonical)
        trace_fingerprints.append(hashlib.sha256(canonical).hexdigest())
        trace_row_indexes.append(row_index)
    trace_event_ids = [row.get("reset_event_id") for row in reset_trace]
    trace_event_ids_unique = bool(trace_event_ids) and all(
        isinstance(value, str) and value for value in trace_event_ids
    ) and len(set(trace_event_ids)) == len(trace_event_ids)

    # Every scored episode's receipt must appear in the trace, byte-identical and
    # in the same order. Extra trace rows are abandoned physical episodes, which
    # only add cleaning and cannot weaken per-episode isolation, so they are
    # reported rather than treated as a failure.
    trace_position_by_canonical: dict[bytes, int] = {}
    for position, canonical in enumerate(trace_canonical):
        trace_position_by_canonical.setdefault(canonical, position)
    detail_positions: list[int] = []
    detail_is_subsequence = trace_canonical_valid and bool(expected_canonical)
    for index, canonical in enumerate(expected_canonical):
        position = trace_position_by_canonical.get(canonical)
        if position is None:
            detail_is_subsequence = False
            add_error(
                f"scored reset event {index} is absent from the receipt trace "
                "or does not match it byte-for-byte"
            )
            continue
        detail_positions.append(position)
    if detail_positions != sorted(detail_positions):
        detail_is_subsequence = False
        add_error("scored reset events do not follow the recorded trace order")

    matched_positions = set(detail_positions)
    # Translate back through trace_row_indexes so the reported ids name the rows
    # that were actually unmatched, even if an earlier row failed to canonicalize.
    orphan_positions = [
        trace_row_indexes[position]
        for position in range(len(trace_canonical))
        if position not in matched_positions
    ]
    orphan_event_ids = [
        trace_event_ids[row_index]
        for row_index in orphan_positions
        if row_index < len(trace_event_ids)
    ]
    detail_event_ids_in_trace = all(
        event_id in set(trace_event_ids) for event_id in event_order
    )
    event_ids_unique = (
        bool(event_order)
        and len(set(event_order)) == len(event_order)
        and trace_event_ids_unique
        and detail_event_ids_in_trace
    )

    check(
        "reset_receipt_trace_present",
        reset_trace_present,
        "reset_receipts.jsonl is required for an audited run",
    )
    check(
        "trace_event_ids_unique",
        trace_event_ids_unique,
        "trace must contain each applied reset exactly once",
    )
    check(
        "event_ids_unique",
        event_ids_unique,
        "scored reset events must be unique and present in the trace",
    )
    check(
        "scored_events_are_ordered_subsequence_of_trace",
        detail_is_subsequence,
        "every scored episode's receipt must appear in the trace, byte-identical "
        "and in recorded order",
    )

    ledger_schema_matches = bool(ledger) and (
        ledger.get("schema_version") == POLICY_RESET_LEDGER_SCHEMA_VERSION
    )
    ledger_acknowledged = bool(ledger) and ledger.get("phase") == "acknowledged"
    # The ledger records the last reset the client acknowledged, which is the last
    # applied reset rather than the last scored one: an abandoned final attempt
    # still acknowledged its receipt.
    last_receipt = audit_basis[-1] if audit_basis else None
    sole_run_id = run_ids[0] if run_id_consistent else None
    sole_session = reset_sessions[0] if reset_session_consistent else None
    ledger_run_matches = bool(ledger) and sole_run_id is not None and (
        ledger.get("run_id") == sole_run_id
    )
    ledger_session_matches = bool(ledger) and sole_session is not None and (
        ledger.get("session_id") == sole_session
    )
    ledger_sequence_matches = bool(ledger) and last_receipt is not None and (
        type(ledger.get("last_episode_seq")) is int
        and ledger.get("last_episode_seq") == last_receipt.get("episode_seq")
    )
    ledger_last_event_matches = bool(ledger) and last_receipt is not None and all(
        ledger.get(field) == last_receipt.get(field)
        for field in (
            "episode_id",
            "context_digest",
            "reset_event_id",
            "server_instance_id",
            "server_reset_generation",
        )
    )

    check(
        "policy_reset_ledger_present",
        ledger_present,
        "policy_reset_ledger.json is required for an audited run",
    )
    check(
        "ledger_schema_matches",
        ledger_schema_matches,
        f"ledger schema must be {POLICY_RESET_LEDGER_SCHEMA_VERSION}",
    )
    check(
        "ledger_acknowledged",
        ledger_acknowledged,
        "final ledger phase must be acknowledged",
    )
    check(
        "ledger_run_matches",
        ledger_run_matches,
        "ledger run_id must match the run encoded by reset episode ids",
    )
    check(
        "ledger_session_matches",
        ledger_session_matches,
        "ledger session_id must match every reset receipt",
    )
    check(
        "ledger_sequence_matches",
        ledger_sequence_matches,
        "ledger last_episode_seq must match the final applied reset",
    )
    check(
        "ledger_last_event_matches",
        ledger_last_event_matches,
        "ledger must point to the final event/context/server acknowledgement",
    )

    errors = _deduplicated_errors(errors)
    valid = all(invariants.values())
    expected_values = {
        "model_module_id": None if expected_module is _MISSING else expected_module,
        "model_code_sha256": None if expected_hash is _MISSING else expected_hash,
        "model_reset_schema_version": (
            None if expected_model_schema is _MISSING else expected_model_schema
        ),
        "reset_receipt_schema_version": (
            RESET_RECEIPT_SCHEMA_VERSION
            if expected_receipt_schema is _MISSING
            else expected_receipt_schema
        ),
        "state_inventory_version": (
            STARVLA_RESET_STATE_VERSION
            if expected_state_schema is _MISSING
            else expected_state_schema
        ),
    }
    audit = {
        "schema_version": POLICY_RESET_AUDIT_SCHEMA_VERSION,
        "valid": valid,
        # State the guarantee at the strength the evidence supports, so a reader
        # does not infer more from words like "audited" than the mechanism proves.
        "guarantee": (
            "Every scored episode was preceded by a recorded, server-attested "
            "reset that reported all StarVLA per-episode containers empty. Per "
            "physical episode the guarantee is at-least-once, not exactly-once: "
            "a crash between the reset RPC and the ledger acknowledgement makes "
            "the next attempt allocate a fresh token, so that episode is reset "
            "again and only the last reset is the attested one."
        ),
        "evidence_limits": (
            "reset_event_id is an unkeyed digest over fields that are all present "
            "in the receipt, so it proves internal consistency, not that the "
            "server executed the clearing. The reported counts are self-reported "
            "by the policy process. Provenance covers only the adapter source "
            "listed in provenance_scope. Orphan resets are counted but not "
            "attributed: this audit does not reconcile them against the run's "
            "abandoned, unstable or restart accounting. Episode-count "
            "completeness is not certified here; see run_completeness. "
            "after.high_level_calls is vacuous under B0 and C1, which construct "
            "no GuardedHighLevelAgent."
        ),
        "provenance_scope": ["XPolicyLab/policy/starVLA/model.py"],
        "provenance_excludes": [
            "XPolicyLab/policy/starVLA/reset_contract.py",
            "agentic_vla/ (GuardedHighLevelAgent, GuardedVisualVerifier)",
            "policy checkpoint weights",
            "effective carve_* runtime overrides",
        ],
        "strict_mode_requested_by_run_config": require_audited_reset is True,
        "run_config_source": run_config_source,
        "run_config_fields": {
            "require_audited_reset": require_path,
            "expected_model_module_id": expected_module_path,
            "expected_model_code_sha256": expected_hash_path,
            "expected_model_reset_schema_version": expected_model_schema_path,
            "expected_reset_receipt_schema_version": expected_receipt_schema_path,
            "expected_state_inventory_version": expected_state_schema_path,
            "robodojo_run_id": expected_run_id_path,
            "expected_reset_contract_sha256": expected_contract_hash_path,
        },
        "audited_run_id": expected_run_id if expected_run_id_path else None,
        "reset_contract_sha256": _RESET_CONTRACT_SHA256,
        "expected": expected_values,
        "detail_episodes": len(details),
        "detail_receipts": detail_receipt_count,
        "unique_reset_events": len(representative_receipts),
        "reset_receipt_trace_rows": len(reset_trace),
        # Resets that were applied but produced no scored episode: abandoned or
        # restarted physical attempts. Reported for completeness; they cannot
        # weaken per-episode isolation, so they do not fail the audit.
        "orphan_reset_events": len(orphan_positions),
        "orphan_reset_event_ids": orphan_event_ids,
        "scored_reset_events": len(detail_positions),
        "server_instance_ids": sorted(
            {value for value in server_instances if isinstance(value, str)}
        ),
        "reset_session_ids": sorted(
            {value for value in reset_sessions if isinstance(value, str)}
        ),
        "server_reset_generations": server_generations,
        "episode_sequences": episode_sequences,
        "run_ids": sorted(set(run_ids)),
        "newly_applied_events": sum(
            receipt.get("applied_this_request") is True
            and receipt.get("response_replayed") is False
            for receipt in representative_receipts
        ),
        "replayed_response_events": sum(
            receipt.get("applied_this_request") is False
            and receipt.get("response_replayed") is True
            for receipt in representative_receipts
        ),
        "detail_event_fingerprints": expected_fingerprints,
        "trace_event_fingerprints": trace_fingerprints,
        "events": event_summaries,
        "invariants": invariants,
        "errors": errors,
    }
    # Keep every invariant directly addressable for simple downstream gates while
    # retaining the structured map for schema-aware consumers.
    audit.update(invariants)
    return audit


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _video_frames(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return int(output.strip())


def _copy_if_distinct(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _write_contact_sheet(video: Path, destination: Path, frame_count: int) -> None:
    selected = sorted(
        {
            0,
            frame_count // 4,
            frame_count // 2,
            3 * frame_count // 4,
            frame_count - 1,
        }
    )
    expression = "+".join(f"eq(n\\,{frame})" for frame in selected)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select={expression},scale=480:-1,tile={len(selected)}x1",
            "-vsync",
            "vfr",
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def _segment_episodes(rows: list[dict]) -> list[list[dict]]:
    """Split one run's trace into per-episode blocks.

    ``episode_id`` in the StarVLA trace is the parallel-environment index, not a
    sequential episode counter, so it is always 0 for the single-environment
    protocol and cannot delimit episodes. The policy adapter clears
    ``step_by_env`` in ``reset()``, so ``timestep`` restarts at 0 for each new
    episode. Only a strict decrease starts a new block: recovery can legitimately
    emit two chunks at the same timestep, which must stay in one episode.
    """
    segments: list[list[dict]] = []
    current: list[dict] = []
    previous: int | None = None
    for row in rows:
        timestep = row.get("timestep")
        if timestep is None:
            current.append(row)
            continue
        timestep = int(timestep)
        if previous is not None and timestep < previous:
            segments.append(current)
            current = []
        current.append(row)
        previous = timestep
    if current:
        segments.append(current)
    return segments


def _align_segments(segments: list[list[dict]], episode_count: int) -> list[list[dict]]:
    """Pad or flag trace segments so they line up with the official episodes.

    A baseline run has no trace at all, and an interrupted run can have fewer
    segments than episodes. Never silently reassign blocks: return empty blocks
    for the episodes that have no trace evidence.
    """
    if not segments:
        return [[] for _ in range(episode_count)]
    if len(segments) == episode_count:
        return segments
    aligned = segments[:episode_count]
    aligned.extend([] for _ in range(max(0, episode_count - len(segments))))
    return aligned


def _planner_episode_segments(
    planner: list[dict],
    monitor_segments: list[list[dict]],
    details: list[dict],
    *,
    has_orphan_resets: bool = False,
) -> tuple[list[list[dict]], dict]:
    """Attribute sparse events by reset identity or dense monitor time windows.

    A missing planner call does not create a segment. Timestep decreases in
    the sparse planner trace therefore cannot identify all episode boundaries.
    """
    count = len(details)
    empty = [[] for _ in details]

    def unavailable(reason: str):
        return empty, {"available": False, "method": "unavailable", "reason": reason}

    if not planner:
        return empty, {"available": True, "method": "no_planner_rows"}
    if any("policy_reset_generation" in row for row in planner):
        generations = []
        for detail in details:
            receipt = detail.get("policy_reset_receipt")
            generations.append(
                receipt.get("reset_generation") if isinstance(receipt, Mapping) else None
            )
        if any(type(value) is not int or value < 1 for value in generations):
            return unavailable("missing official reset generations")
        if len(set(generations)) != count:
            return unavailable("multiple environments share a reset generation")
        indices = {generation: index for index, generation in enumerate(generations)}
        assigned = [[] for _ in details]
        for row in planner:
            generation = row.get("policy_reset_generation")
            if type(generation) is not int or generation not in indices:
                return unavailable("planner row has no matching scored reset")
            assigned[indices[generation]].append(row)
        return assigned, {"available": True, "method": "policy_reset_generation"}
    if has_orphan_resets:
        return unavailable("legacy trace includes unscored resets")
    if count == 1:
        return [planner], {"available": True, "method": "single_episode"}
    if len(monitor_segments) != count or any(not block for block in monitor_segments):
        return unavailable("dense monitor does not cover every episode")

    def timestamp(row):
        value = row.get("timestamp_s")
        return value if type(value) in (int, float) and math.isfinite(value) else None

    monitor_times = [timestamp(row) for block in monitor_segments for row in block]
    if any(value is None for value in monitor_times) or any(
        right < left for left, right in zip(monitor_times, monitor_times[1:])
    ):
        return unavailable("monitor timestamps are missing or non-monotonic")
    windows = [(timestamp(block[0]), timestamp(block[-1])) for block in monitor_segments]
    assigned = [[] for _ in details]
    previous = None
    for row in planner:
        value = timestamp(row)
        if value is None or (previous is not None and value < previous):
            return unavailable("planner timestamps are missing or non-monotonic")
        candidates = [index for index, (start, end) in enumerate(windows) if start <= value <= end]
        if len(candidates) != 1:
            return unavailable("planner event is outside an unambiguous monitor window")
        assigned[candidates[0]].append(row)
        previous = value
    return assigned, {"available": True, "method": "dense_monitor_timestamps"}


def _summarize_runtime(
    runtime: list[dict],
    *,
    trace_available: bool,
    segments: list[list[dict]] | None = None,
) -> dict:
    """Keep an unavailable baseline trace distinct from an observed zero."""
    if not trace_available:
        return {
            "trace_available": False,
            "vla_calls": None,
            "model_latency_p50_ms": None,
            "model_latency_p95_ms": None,
            "model_latency_max_ms": None,
            "first_call_latency_ms": None,
            "per_episode_first_call_latency_ms": None,
            "steady_state_model_latency_p50_ms": None,
            "steady_state_model_latency_p95_ms": None,
            "steady_state_model_latency_max_ms": None,
            "steady_state_excludes_per_episode_first_call": None,
            "total_model_latency_ms": None,
            "deadline_misses": None,
            "request_level_action_seeded": None,
            "first_action_seed": None,
            "last_action_seed": None,
            "compute_phase_calls": None,
            "observed_inference_steps": None,
            "observed_execute_horizons": None,
            "optimization_profile_ids": None,
            "observed_inference_step_parameters": None,
            "inference_step_control_verified": None,
        }

    latencies = [float(row["model_latency_ms"]) for row in runtime]
    # Every episode pays a cold first call. Pooling those into the steady-state
    # percentiles would inflate them once a run holds more than one episode.
    if segments:
        per_episode_first_call = [
            float(block[0]["model_latency_ms"]) for block in segments if block
        ]
        steady_state_latencies = [
            float(row["model_latency_ms"])
            for block in segments
            for row in block[1:]
        ]
        steady_state_scoped = True
    else:
        per_episode_first_call = latencies[:1]
        steady_state_latencies = latencies[1:]
        steady_state_scoped = False
    action_seeds = [
        row.get("metadata", {}).get("action_seed")
        for row in runtime
        if row.get("metadata", {}).get("action_seed") is not None
    ]
    compute_phases: dict[str, int] = {}
    for row in runtime:
        phase = str(
            row.get("metadata", {}).get("controller", {}).get(
                "compute_phase", "unspecified"
            )
        )
        compute_phases[phase] = compute_phases.get(phase, 0) + 1
    inference_steps = sorted(
        {
            int(row["applied_controls"]["inference_steps"])
            for row in runtime
            if row.get("applied_controls", {}).get("inference_steps") is not None
        }
    )
    execute_horizons = sorted(
        {
            int(row["applied_controls"]["max_actions"])
            for row in runtime
            if row.get("applied_controls", {}).get("max_actions") is not None
        }
    )
    profile_ids = sorted(
        {
            str(row["metadata"]["optimization_profile"]["profile"]["profile_id"])
            for row in runtime
            if row.get("metadata", {})
            .get("optimization_profile", {})
            .get("profile", {})
            .get("profile_id")
        }
    )
    inference_step_parameters = sorted(
        {
            str(row["metadata"]["inference_steps_parameter"])
            for row in runtime
            if row.get("metadata", {}).get("inference_steps_parameter")
        }
    )
    return {
        "trace_available": True,
        "vla_calls": len(runtime),
        "model_latency_p50_ms": _percentile(latencies, 0.50),
        "model_latency_p95_ms": _percentile(latencies, 0.95),
        "model_latency_max_ms": max(latencies) if latencies else None,
        "first_call_latency_ms": latencies[0] if latencies else None,
        "per_episode_first_call_latency_ms": per_episode_first_call,
        "steady_state_model_latency_p50_ms": _percentile(
            steady_state_latencies, 0.50
        ),
        "steady_state_model_latency_p95_ms": _percentile(
            steady_state_latencies, 0.95
        ),
        "steady_state_model_latency_max_ms": (
            max(steady_state_latencies) if steady_state_latencies else None
        ),
        "steady_state_excludes_per_episode_first_call": steady_state_scoped,
        "total_model_latency_ms": sum(latencies),
        "deadline_misses": sum(bool(row.get("deadline_miss")) for row in runtime),
        "request_level_action_seeded": bool(action_seeds),
        "first_action_seed": action_seeds[0] if action_seeds else None,
        "last_action_seed": action_seeds[-1] if action_seeds else None,
        "compute_phase_calls": compute_phases,
        "observed_inference_steps": inference_steps,
        "observed_execute_horizons": execute_horizons,
        "optimization_profile_ids": profile_ids,
        "observed_inference_step_parameters": inference_step_parameters,
        "inference_step_control_verified": bool(inference_step_parameters),
    }


def _summarize_staged_vlm(planner: list[dict]) -> dict:
    receipts = [row for row in planner if row.get("event") == "planner_residency_receipt"]

    def total(key):
        values = [
            row["timings"].get(key) if isinstance(row.get("timings"), Mapping) else None
            for row in receipts
        ]
        if not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            for value in values
        ):
            return None
        return sum(values)

    return {
        "trace_available": bool(receipts),
        "rpc_calls": len(receipts) if receipts else None,
        "successful_rpc_calls": sum(row.get("ok") is True for row in receipts) if receipts else None,
        "action_model_restored_calls": sum(
            row.get("action_model_restored") is True for row in receipts
        ) if receipts else None,
        "total_residency_and_generation_ms": total("total_ms"),
        "total_generation_ms": total("vlm_generate_ms"),
        "model_paths": sorted({str(row["model_path"]) for row in receipts if row.get("model_path")}),
        "scope": "Staged semantic RPC receipts, including startup Planner and scheduled Critic; not recovery-decision count.",
    }


def _summarize_agentic(planner: list[dict], monitor: list[dict]) -> dict:
    semantic_rows = [
        row
        for row in planner
        if row.get("role") in {"scalar_visual_critic", "critic"}
    ]
    task_plan_checks = [
        row
        for row in planner
        if row.get("trigger") == "scheduled_task_plan_checkpoint"
    ]
    task_plan_installs = [
        row
        for row in planner
        if row.get("action") == "task_plan_installed"
        or row.get("trigger") == "task_plan_install"
    ]
    accepted_control_actions = {
        "planner_vla_replan",
        "planner_continue_with_fresh_vla_chunk",
        "planner_safe_stop_admitted",
        "semantic_planner_safe_stop_admitted",
    }
    actual_control_actions = accepted_control_actions | {
        "fresh_vla_replan", "task_preserving_vla_replan",
        "semantic_planner_fallback_replan", "semantic_subgoal_vla_replan",
        "semantic_subgoal_blocked_task_only_replan", "safe_stop_confirmed_complete",
        "task_plan_retry_active_stage", "task_plan_advance",
        "task_plan_complete_continue_original",
    }
    # Planner and Critic may log the same committed recovery. Count that control
    # event once, and never count unknown/rejected/observation-only log labels.
    control_events = set()
    for index, row in enumerate(planner):
        action = row.get("control_action", row.get("action"))
        if action not in actual_control_actions:
            continue
        if all(row.get(key) is not None for key in ("policy_reset_generation", "env_idx", "timestep")):
            identity = (row["policy_reset_generation"], row["env_idx"], row["timestep"], action)
        else:
            identity = ("legacy_unbound_row", index)
        control_events.add(identity)
    semantic_planner_rows = [
        row
        for row in planner
        if row.get("trigger") in {"no_progress", "semantic_no_progress"}
        and isinstance(row.get("decision"), dict)
    ]
    accepted_semantic_interventions = [
        row
        for row in semantic_planner_rows
        if row.get("accepted") is True
        and row.get("control_action", row.get("action")) in accepted_control_actions
    ]
    monitor_events = [
        row
        for row in monitor
        if row.get("assessment", {}).get("event") not in {None, "none"}
    ]
    no_progress_events = [
        row
        for row in monitor_events
        if row.get("assessment", {}).get("event") == "no_progress"
    ]
    return {
        "semantic_checks": len(semantic_rows) + len(task_plan_checks),
        "interventions": len(control_events),
        "interventions_scope": "Distinct committed generation/env/timestep/action events; unbound legacy rows cannot be deduplicated.",
        "low_cost_replans": sum(
            row.get("action") == "fresh_vla_replan" for row in planner
        ),
        "semantic_planner_calls": len(semantic_planner_rows),
        "semantic_planner_calls_scope": "no_progress recovery decisions only; excludes startup and scheduled checks",
        "staged_vlm": _summarize_staged_vlm(planner),
        "accepted_semantic_interventions": len(accepted_semantic_interventions),
        "planner_control_actions": [
            row.get("control_action", row.get("action"))
            for row in semantic_planner_rows
        ],
        "monitor_events": len(monitor_events),
        "first_no_progress_event_step": (
            int(no_progress_events[0]["timestep"]) if no_progress_events else None
        ),
        "task_plan_installs": len(task_plan_installs),
        "task_plan_checks": len(task_plan_checks),
        "task_plan_check_statuses": [
            row.get("report", {}).get("status") for row in task_plan_checks
        ],
        "task_plan_actions": [row.get("action") for row in task_plan_checks],
        "all_recorded_actions": [
            row.get("action") for row in planner if row.get("action")
        ],
        "task_plan_verification_modes": sorted(
            {
                str(row.get("verification_mode"))
                for row in task_plan_checks
                if row.get("verification_mode")
            }
        ),
        "task_plan_verifier_sources": sorted(
            {
                str(row.get("verifier_source"))
                for row in task_plan_checks
                if row.get("verifier_source")
            }
        ),
        "task_plan_sources": sorted(
            {
                str(row.get("source"))
                for row in task_plan_installs
                if row.get("source")
            }
        ),
        "vla_instruction_capabilities": sorted(
            {
                str(row.get("vla_instruction_capability"))
                for row in planner
                if row.get("vla_instruction_capability")
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--fault-step", type=int)
    parser.add_argument("--require-valid-reset-audit", action="store_true")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        help=(
            "Pre-registered episode count for this block. Required by "
            "--require-valid-reset-audit: the reset audit tolerates abandoned "
            "physical episodes, so completeness needs its own gate."
        ),
    )
    args = parser.parse_args()
    if args.expected_episodes is not None and args.expected_episodes < 1:
        raise ValueError("--expected-episodes must be a positive integer")

    artifact = args.artifact.resolve()
    official = args.official_run.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    official_result = official / "_result.json"
    if not official_result.is_file():
        official_result = official / "result.json"
    if not official_result.is_file():
        raise FileNotFoundError(f"official result is missing under {official}")
    _copy_if_distinct(official_result, artifact / "result.json")
    for evidence_name in ("reset_receipts.jsonl", "policy_reset_ledger.json"):
        evidence_path = official / evidence_name
        if evidence_path.is_file():
            _copy_if_distinct(evidence_path, artifact / evidence_name)
    for video in official.glob("*.mp4"):
        _copy_if_distinct(video, artifact / video.name)

    result = json.loads((artifact / "result.json").read_text())
    details = result["details"]
    if not details:
        raise ValueError(f"official result under {official} contains no episodes")
    # RoboDojo keys details by stringified episode index; order by that index so
    # trace blocks and videos line up with the official layout order.
    ordered_keys = sorted(details, key=lambda key: int(key))
    ordered_details = [details[key] for key in ordered_keys]

    reset_receipts_path = artifact / "reset_receipts.jsonl"
    reset_receipt_trace, reset_trace_errors = _read_audit_jsonl(
        reset_receipts_path
    )
    ledger_path = artifact / "policy_reset_ledger.json"
    ledger, ledger_error = _read_audit_mapping(ledger_path)
    artifact_run_config = artifact / "run_config.json"
    official_run_config = official / "run_config.json"
    if artifact_run_config.is_file():
        run_config_path: Path | None = artifact_run_config
    elif official_run_config.is_file():
        run_config_path = official_run_config
    else:
        run_config_path = None
    if run_config_path is None:
        run_config = None
        run_config_error = None
    else:
        run_config, run_config_error = _read_audit_mapping(run_config_path)
    audit_input_errors = list(reset_trace_errors)
    if ledger_error:
        audit_input_errors.append(ledger_error)
    if run_config_error:
        audit_input_errors.append(run_config_error)

    reset_audit = _audit_policy_resets(
        ordered_details,
        reset_receipt_trace,
        reset_trace_present=reset_receipts_path.is_file(),
        ledger=ledger,
        ledger_present=ledger_path.is_file(),
        run_config=run_config,
        run_config_source=str(run_config_path) if run_config_path else None,
        input_errors=audit_input_errors,
    )
    reset_audit_path = artifact / "policy_reset_audit.json"
    reset_audit_path.write_text(
        json.dumps(reset_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.require_valid_reset_audit and not reset_audit["valid"]:
        errors = reset_audit.get("errors", [])
        diagnostic = "; ".join(str(error) for error in errors[:3])
        if len(errors) > 3:
            diagnostic += f"; ... ({len(errors)} errors total)"
        raise ValueError(f"policy reset audit is invalid: {diagnostic}")

    runtime_trace_path = artifact / "runtime_trace.jsonl"
    runtime_trace_available = runtime_trace_path.is_file()
    runtime = _jsonl(runtime_trace_path)
    planner = _jsonl(artifact / "planner_trace.jsonl")
    monitor = _jsonl(artifact / "monitor_trace.jsonl")

    episode_count = len(ordered_keys)
    single_episode = episode_count == 1
    runtime_segments = _align_segments(_segment_episodes(runtime), episode_count)
    raw_monitor_segments = _segment_episodes(monitor)
    planner_segments, planner_attribution = _planner_episode_segments(
        planner, raw_monitor_segments, ordered_details,
        has_orphan_resets=bool(reset_audit["orphan_reset_events"]),
    )
    monitor_segments = _align_segments(raw_monitor_segments, episode_count)
    trace_segments_aligned = (
        not runtime_trace_available
        or len(_segment_episodes(runtime)) == episode_count
    )

    agentic = _summarize_agentic(planner, monitor)

    episodes: list[dict] = []
    for index, key in enumerate(ordered_keys):
        detail = details[key]
        videos = sorted(artifact.glob(f"episode_{index:07d}_cam_head_*.mp4"))
        frame_count = _video_frames(videos[0]) if videos else None
        contact_sheet = None
        if videos and frame_count:
            contact_sheet = (
                "head_contact_sheet.png"
                if single_episode
                else f"head_contact_sheet_ep{index:03d}.png"
            )
            _write_contact_sheet(videos[0], artifact / contact_sheet, frame_count)
        episode_agentic = _summarize_agentic(
            planner_segments[index], monitor_segments[index]
        )
        if not planner_attribution["available"]:
            for metric in episode_agentic:
                if metric not in {"monitor_events", "first_no_progress_event_step"}:
                    episode_agentic[metric] = None
        episode_agentic["planner_attribution_available"] = planner_attribution["available"]
        episodes.append(
            {
                "episode_index": index,
                "layout_id": int(detail["layout_id"]),
                "success": bool(detail["success"]),
                "score": float(detail["score"]),
                "video_frames": frame_count,
                "head_contact_sheet": contact_sheet,
                "policy_reset_receipt": detail.get("policy_reset_receipt"),
                "runtime": _summarize_runtime(
                    runtime_segments[index],
                    trace_available=runtime_trace_available
                    and bool(runtime_segments[index]),
                ),
                "agentic": episode_agentic,
                "first_no_progress_event_step": episode_agentic[
                    "first_no_progress_event_step"
                ],
            }
        )

    run_completeness = _audit_run_completeness(
        result, episodes, expected_episodes=args.expected_episodes
    )
    completeness_path = artifact / "run_completeness.json"
    completeness_path.write_text(
        json.dumps(run_completeness, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.require_valid_reset_audit and not run_completeness["valid"]:
        # Fail before publishing summary.json: a truncated block that still has a
        # valid reset audit must not be adoptable by the matrix aggregation.
        diagnostic = "; ".join(str(error) for error in run_completeness["errors"][:3])
        raise ValueError(f"run completeness check failed: {diagnostic}")

    successes = sum(episode["success"] for episode in episodes)
    frame_counts = [
        episode["video_frames"]
        for episode in episodes
        if episode["video_frames"] is not None
    ]
    aggregate = {
        "episodes": episode_count,
        "successes": successes,
        "success_rate": successes / episode_count,
        "official_success_rate": float(result["success_rate"]),
        "layout_ids": [episode["layout_id"] for episode in episodes],
        "successful_layout_ids": [
            episode["layout_id"] for episode in episodes if episode["success"]
        ],
        "failed_layout_ids": [
            episode["layout_id"] for episode in episodes if not episode["success"]
        ],
        "total_video_frames": sum(frame_counts) if frame_counts else None,
        "mean_video_frames": (
            sum(frame_counts) / len(frame_counts) if frame_counts else None
        ),
        "trace_segments_aligned_with_episodes": trace_segments_aligned,
        "runtime_trace_segments": len(_segment_episodes(runtime)),
        "planner_trace_attribution": planner_attribution,
        "policy_reset_audit_valid": reset_audit["valid"],
        "run_completeness_valid": run_completeness["valid"],
    }

    if args.fault_step is not None:
        schema_version = "carve.robodojo.controlled-fault.v2"
    elif single_episode:
        schema_version = "carve.robodojo.nominal.v1"
    else:
        # Bump the schema so single-episode consumers reject a multi-episode run
        # instead of silently reading only its first episode.
        schema_version = "carve.robodojo.nominal.v2"

    if single_episode:
        claim_boundary = (
            "One controlled-fault episode. It is causal mechanism evidence only; "
            "aggregate recovery claims require paired repetitions."
            if args.fault_step is not None
            else "One official nominal episode. It is mechanism evidence only; "
            "aggregate success claims require the pre-registered repeated matrix."
        )
    else:
        claim_boundary = (
            f"{episode_count} official RoboDojo episodes on layout set "
            f"{args.seed}, evaluated in ascending layout_id order. Run-level "
            "runtime and agentic counters pool every episode; per-episode values "
            "are in 'episodes'. Cross-condition claims require the "
            "pre-registered paired matrix and its aggregation script."
        )

    summary = {
        "schema_version": schema_version,
        "condition": args.condition,
        "benchmark": "RoboDojo",
        "task": args.task,
        # Retained for single-episode consumers; multi-episode runs must read
        # aggregate.layout_ids instead.
        "layout_id": episodes[0]["layout_id"],
        "seed": args.seed,
        "layout_set": args.seed,
        "policy": "StarVLA PI-v3",
        "controlled_fault": {
            "enabled": args.fault_step is not None,
            "type": "stale_action_hold" if args.fault_step is not None else None,
            "injected_at_step": args.fault_step,
            "first_no_progress_event_step": (
                agentic["first_no_progress_event_step"]
            ),
        },
        "runtime": _summarize_runtime(
            runtime,
            trace_available=runtime_trace_available,
            segments=[block for block in runtime_segments if block] or None,
        ),
        "agentic": agentic,
        "policy_reset_audit": reset_audit,
        "run_completeness": run_completeness,
        "official_result": {
            "success": bool(episodes[0]["success"]) if single_episode else None,
            "successes": successes,
            "trials": episode_count,
            "success_rate": float(result["success_rate"]),
            "score": episodes[0]["score"] if single_episode else None,
            "video_frames": episodes[0]["video_frames"] if single_episode else None,
        },
        "aggregate": aggregate,
        "episodes": episodes,
        "claim_boundary": claim_boundary,
    }
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    checksum_lines = []
    for path in sorted(
        p for p in artifact.iterdir() if p.is_file() and p.name != "SHA256SUMS"
    ):
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        )
    (artifact / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


if __name__ == "__main__":
    main()
