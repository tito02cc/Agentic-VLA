"""Regression tests for the durable policy-reset token ledger.

``PolicyResetLedger`` is the only thing standing between a crashed RoboDojo
process and a reused episode token, so the tests below pin its durability
ordering (pending token on disk *before* the reset RPC), its resume behaviour,
and the exact receipt it is willing to accept as an acknowledgement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# The root unit-test environment does not install XPolicyLab's wire-codec
# dependency, and importing the ledger pulls in ``client_server.ws``. These
# tests never encode a frame, so a minimal stub keeps them codec-independent.
try:
    import msgpack_numpy  # noqa: F401
except ModuleNotFoundError:
    msgpack_numpy_stub = ModuleType("msgpack_numpy")
    msgpack_numpy_stub.encode = lambda value: value
    msgpack_numpy_stub.decode = lambda value: value
    sys.modules["msgpack_numpy"] = msgpack_numpy_stub


ROBODOJO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "robodojo_official"
XPOLICYLAB_ROOT = ROBODOJO_ROOT / "XPolicyLab"
for root in (str(ROBODOJO_ROOT), str(XPOLICYLAB_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from client_server.ws.protocol.reset import (  # noqa: E402
    reset_context_digest,
    reset_event_id,
)
from src.eval_client.policy_reset_state import (  # noqa: E402
    POLICY_RESET_LEDGER_SCHEMA_VERSION,
    PolicyResetLedger,
)


RUN_ID = "stack_bowls-C3-seed1"
SERVER_INSTANCE_ID = "server-instance-abc"
LEDGER_NAME = "policy_reset_ledger.json"


def _ledger(tmp_path: Path, *, run_id: str = RUN_ID, **kwargs: Any) -> PolicyResetLedger:
    return PolicyResetLedger(tmp_path / LEDGER_NAME, run_id=run_id, **kwargs)


def _on_disk(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / LEDGER_NAME).read_text(encoding="utf-8"))


def _assignment(env_ids: list[int]) -> dict[str, Any]:
    return {
        "active_env_ids": list(env_ids),
        "layout_seeds": [
            {"env_idx": env_idx, "layout_id": env_idx} for env_idx in env_ids
        ],
    }


def _receipt_for(
    context: dict[str, Any],
    *,
    server_instance_id: str = SERVER_INSTANCE_ID,
    generation: int = 1,
) -> dict[str, Any]:
    """The subset of a wire receipt the ledger actually acknowledges."""

    digest = reset_context_digest(context)
    return {
        "episode_id": context["episode_id"],
        "episode_seq": context["episode_seq"],
        "reset_session_id": context["reset_session_id"],
        "context_digest": digest,
        "server_instance_id": server_instance_id,
        "server_reset_generation": generation,
        "reset_event_id": reset_event_id(
            server_instance_id=server_instance_id,
            server_reset_generation=generation,
            context_digest=digest,
        ),
    }


# ---------------------------------------------------------------------------
# Creation, persistence and resume
# ---------------------------------------------------------------------------


def test_new_ledger_persists_session_and_sequence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    persisted = _on_disk(tmp_path)
    assert persisted["schema_version"] == POLICY_RESET_LEDGER_SCHEMA_VERSION
    assert persisted["run_id"] == RUN_ID
    assert persisted["phase"] == "idle"
    assert persisted["last_episode_seq"] == 0
    assert persisted["session_id"] == ledger.session_id
    assert len(ledger.session_id) == 32
    int(ledger.session_id, 16)
    assert ledger.last_episode_seq == 0

    # A fresh handle on the same run must adopt the persisted session, not mint
    # a new one: the session id is part of every episode token.
    reloaded = _ledger(tmp_path)
    assert reloaded.session_id == ledger.session_id
    assert reloaded.last_episode_seq == 0
    assert _on_disk(tmp_path)["session_id"] == ledger.session_id


def test_reloaded_ledger_keeps_session_and_continues_sequence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    session_id = ledger.session_id

    first = ledger.begin_episode(**_assignment([0]))
    ledger.acknowledge(_receipt_for(first, generation=1))
    assert ledger.last_episode_seq == 1

    reloaded = _ledger(tmp_path)
    assert reloaded.session_id == session_id
    assert reloaded.last_episode_seq == 1
    assert reloaded.snapshot()["phase"] == "acknowledged"

    second = reloaded.begin_episode(**_assignment([0, 1]))
    assert second["episode_seq"] == 2
    assert second["reset_session_id"] == session_id
    assert second["episode_id"] == f"{RUN_ID}:{session_id}:batch-0000002"
    assert second["episode_id"] != first["episode_id"]

    reloaded.acknowledge(_receipt_for(second, generation=2))
    assert _on_disk(tmp_path)["last_episode_seq"] == 2

    # A third handle keeps counting from the durable sequence.
    third = _ledger(tmp_path).begin_episode(**_assignment([0]))
    assert third["episode_seq"] == 3


def test_ledger_rejects_a_different_run_id(tmp_path: Path) -> None:
    _ledger(tmp_path)

    with pytest.raises(ValueError, match="run_id mismatch"):
        _ledger(tmp_path, run_id="stack_bowls-C3-seed2")


def test_ledger_rejects_an_unsupported_schema(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    state = _on_disk(tmp_path)
    state["schema_version"] = "robodojo.policy-reset-ledger.v0"
    (tmp_path / LEDGER_NAME).write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported policy reset ledger schema"):
        _ledger(tmp_path)

    # The rejected file is still the one the previous handle wrote.
    assert _on_disk(tmp_path)["session_id"] == ledger.session_id


def test_ledger_rejects_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / LEDGER_NAME).write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to load policy reset ledger"):
        _ledger(tmp_path)


def test_ledger_rejects_a_non_object_document(tmp_path: Path) -> None:
    (tmp_path / LEDGER_NAME).write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="policy reset ledger state must be a mapping"):
        _ledger(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        pytest.param({"session_id": ""}, "invalid session_id", id="empty_session"),
        pytest.param({"last_episode_seq": -1}, "invalid sequence", id="negative_seq"),
        pytest.param({"last_episode_seq": 1.0}, "invalid sequence", id="float_seq"),
        pytest.param({"phase": "done"}, "invalid phase", id="unknown_phase"),
        pytest.param(
            {"episode_id": "leftover"},
            "idle policy reset ledger contains episode state",
            id="idle_with_episode_state",
        ),
    ],
)
def test_ledger_rejects_inconsistent_persisted_state(
    tmp_path: Path, mutation: dict[str, Any], match: str
) -> None:
    _ledger(tmp_path)
    state = _on_disk(tmp_path)
    state.update(mutation)
    (tmp_path / LEDGER_NAME).write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _ledger(tmp_path)


def test_ledger_rejects_an_inconsistent_acknowledged_state(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = ledger.begin_episode(**_assignment([0]))
    ledger.acknowledge(_receipt_for(context))

    state = _on_disk(tmp_path)
    state["server_reset_generation"] = 7  # no longer matches the event id
    (tmp_path / LEDGER_NAME).write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="event is inconsistent"):
        _ledger(tmp_path)


def test_ledger_rejects_a_resume_snapshot_from_another_session(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    snapshot = ledger.snapshot()
    snapshot["session_id"] = "f" * 32

    with pytest.raises(ValueError, match="policy reset snapshot session mismatch"):
        _ledger(tmp_path, resume_snapshot=snapshot)


def test_ledger_rejects_a_sequence_that_trails_the_resume_point(tmp_path: Path) -> None:
    _ledger(tmp_path)

    with pytest.raises(ValueError, match="sequence trails resume state"):
        _ledger(tmp_path, initial_sequence=4)


def test_ledger_validates_constructor_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        _ledger(tmp_path, run_id="")
    with pytest.raises(ValueError, match="initial_sequence must be a non-negative"):
        _ledger(tmp_path, initial_sequence=-1)


# ---------------------------------------------------------------------------
# Token allocation
# ---------------------------------------------------------------------------


def test_begin_episode_persists_the_pending_token_before_the_rpc(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    context = ledger.begin_episode(**_assignment([0, 1]))

    # Nothing has been acknowledged yet: this is exactly the state a crash
    # between `begin_episode` and the reset RPC would leave behind, and it must
    # already be on disk so the token can never be handed out twice.
    persisted = _on_disk(tmp_path)
    assert persisted["phase"] == "pending"
    assert persisted["last_episode_seq"] == 1
    assert persisted["episode_id"] == context["episode_id"]
    assert persisted["context_digest"] == reset_context_digest(context)
    assert persisted["reset_event_id"] is None
    assert persisted["server_instance_id"] is None
    assert persisted["server_reset_generation"] is None

    assert context == {
        "episode_id": f"{RUN_ID}:{ledger.session_id}:batch-0000001",
        "episode_seq": 1,
        "reset_session_id": ledger.session_id,
        "active_env_ids": [0, 1],
        "layout_seeds": [
            {"env_idx": 0, "layout_id": 0},
            {"env_idx": 1, "layout_id": 1},
        ],
    }

    # The caller receives a copy; mutating it must not rewrite history.
    context["active_env_ids"].append(9)
    context["episode_id"] = "tampered"
    assert _on_disk(tmp_path)["episode_id"] == persisted["episode_id"]
    assert ledger.snapshot()["episode_id"] == persisted["episode_id"]


def test_pending_token_is_never_reused_for_a_second_episode(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    first = ledger.begin_episode(**_assignment([0]))
    # A retry loop that re-enters begin_episode (e.g. after an unstable scene)
    # must burn the token rather than reuse it.
    second = ledger.begin_episode(**_assignment([0]))

    assert second["episode_seq"] == first["episode_seq"] + 1
    assert second["episode_id"] != first["episode_id"]
    assert reset_context_digest(second) != reset_context_digest(first)
    with pytest.raises(ValueError, match="does not match pending ledger episode"):
        ledger.acknowledge(_receipt_for(first))


@pytest.mark.parametrize(
    ("assignment", "match"),
    [
        pytest.param(
            {"active_env_ids": [], "layout_seeds": []},
            "non-empty list",
            id="empty_active_env_ids",
        ),
        pytest.param(
            {"active_env_ids": (0,), "layout_seeds": [{"env_idx": 0, "layout_id": 0}]},
            "non-empty list",
            id="tuple_active_env_ids",
        ),
        pytest.param(
            {"active_env_ids": [0, 0], "layout_seeds": [{"env_idx": 0, "layout_id": 0}] * 2},
            "must not contain duplicates",
            id="duplicate_env_ids",
        ),
        pytest.param(
            {"active_env_ids": [-1], "layout_seeds": [{"env_idx": -1, "layout_id": 0}]},
            "non-negative integers",
            id="negative_env_id",
        ),
        pytest.param(
            {"active_env_ids": [0, 1], "layout_seeds": [{"env_idx": 0, "layout_id": 0}]},
            "must match active_env_ids length",
            id="length_mismatch",
        ),
        pytest.param(
            {"active_env_ids": [0], "layout_seeds": [{"env_idx": 0}]},
            "only env_idx and layout_id",
            id="incomplete_seed_row",
        ),
        pytest.param(
            {
                "active_env_ids": [0, 1],
                "layout_seeds": [
                    {"env_idx": 1, "layout_id": 1},
                    {"env_idx": 0, "layout_id": 0},
                ],
            },
            "order must match active_env_ids",
            id="seed_order_mismatch",
        ),
    ],
)
def test_begin_episode_rejects_malformed_assignments(
    tmp_path: Path, assignment: dict[str, Any], match: str
) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(ValueError, match=match):
        ledger.begin_episode(**assignment)

    # A rejected assignment must not consume a sequence number.
    assert ledger.last_episode_seq == 0
    assert _on_disk(tmp_path)["phase"] == "idle"


# ---------------------------------------------------------------------------
# Acknowledgement
# ---------------------------------------------------------------------------


def test_acknowledge_commits_the_matching_receipt(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = ledger.begin_episode(**_assignment([0]))
    receipt = _receipt_for(context, generation=3)

    ledger.acknowledge(receipt)

    persisted = _on_disk(tmp_path)
    assert persisted["phase"] == "acknowledged"
    assert persisted["reset_event_id"] == receipt["reset_event_id"]
    assert persisted["server_instance_id"] == SERVER_INSTANCE_ID
    assert persisted["server_reset_generation"] == 3
    assert persisted["episode_id"] == context["episode_id"]
    assert persisted["context_digest"] == reset_context_digest(context)
    assert ledger.snapshot() == persisted


def test_acknowledge_requires_a_pending_episode(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = {
        "episode_id": f"{RUN_ID}:{ledger.session_id}:batch-0000001",
        "episode_seq": 1,
        "reset_session_id": ledger.session_id,
        "active_env_ids": [0],
        "layout_seeds": [{"env_idx": 0, "layout_id": 0}],
    }

    with pytest.raises(ValueError, match="has no pending episode"):
        ledger.acknowledge(_receipt_for(context))

    pending = ledger.begin_episode(**_assignment([0]))
    ledger.acknowledge(_receipt_for(pending))
    # Double acknowledgement is a protocol error, not an idempotent no-op.
    with pytest.raises(ValueError, match="has no pending episode"):
        ledger.acknowledge(_receipt_for(pending))


def _conflicting_receipts(context: dict[str, Any], session_id: str) -> list[tuple[str, Any, str]]:
    valid = _receipt_for(context)
    other_context = dict(context)
    other_context["active_env_ids"] = [0, 1]
    other_context["layout_seeds"] = [
        {"env_idx": 0, "layout_id": 0},
        {"env_idx": 1, "layout_id": 1},
    ]

    cases: list[tuple[str, Any, str]] = [
        ("not_a_mapping", ["receipt"], "reset receipt must be a mapping"),
    ]

    wrong_episode = dict(valid)
    wrong_episode["episode_id"] = f"{RUN_ID}:{session_id}:batch-0000009"
    cases.append(
        ("wrong_episode_id", wrong_episode, "does not match pending ledger episode")
    )

    missing_episode = dict(valid)
    missing_episode.pop("episode_id")
    cases.append(
        ("missing_episode_id", missing_episode, "does not match pending ledger episode")
    )

    wrong_digest = dict(valid)
    wrong_digest["context_digest"] = reset_context_digest(other_context)
    cases.append(
        ("wrong_context_digest", wrong_digest, "does not match pending ledger context")
    )

    wrong_sequence = dict(valid)
    wrong_sequence["episode_seq"] = 2
    cases.append(
        ("wrong_episode_seq", wrong_sequence, "sequence does not match pending ledger")
    )

    bool_sequence = dict(valid)
    bool_sequence["episode_seq"] = True
    cases.append(
        ("bool_episode_seq", bool_sequence, "sequence does not match pending ledger")
    )

    wrong_session = dict(valid)
    wrong_session["reset_session_id"] = "0" * 32
    cases.append(
        ("wrong_session", wrong_session, "session does not match pending ledger")
    )

    bad_event = dict(valid)
    bad_event["reset_event_id"] = "not-a-digest"
    cases.append(("invalid_event_id", bad_event, "invalid event id"))

    empty_server = dict(valid)
    empty_server["server_instance_id"] = ""
    cases.append(("empty_server_id", empty_server, "invalid server instance id"))

    zero_generation = dict(valid)
    zero_generation["server_reset_generation"] = 0
    cases.append(
        ("zero_generation", zero_generation, "invalid server reset generation")
    )

    bool_generation = dict(valid)
    bool_generation["server_reset_generation"] = True
    cases.append(
        ("bool_generation", bool_generation, "invalid server reset generation")
    )

    # A well-formed event id that was computed over a different generation must
    # not be accepted for this pending token.
    mismatched_event = dict(valid)
    mismatched_event["reset_event_id"] = _receipt_for(context, generation=2)[
        "reset_event_id"
    ]
    cases.append(
        ("event_id_for_other_generation", mismatched_event, "does not match pending ledger")
    )

    foreign_event = dict(valid)
    foreign_event["reset_event_id"] = _receipt_for(
        context, server_instance_id="another-server"
    )["reset_event_id"]
    cases.append(
        ("event_id_from_other_server", foreign_event, "does not match pending ledger")
    )

    return cases


def test_acknowledge_rejects_conflicting_receipts(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = ledger.begin_episode(**_assignment([0]))

    for case_id, receipt, match in _conflicting_receipts(context, ledger.session_id):
        with pytest.raises(ValueError, match=match):
            ledger.acknowledge(receipt)
        # A rejected acknowledgement leaves the token pending, never committed.
        persisted = _on_disk(tmp_path)
        assert persisted["phase"] == "pending", case_id
        assert persisted["reset_event_id"] is None, case_id

    # The correct receipt is still accepted afterwards.
    ledger.acknowledge(_receipt_for(context))
    assert _on_disk(tmp_path)["phase"] == "acknowledged"


def test_pending_ledger_survives_a_restart_and_can_still_be_acknowledged(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    context = ledger.begin_episode(**_assignment([0, 1]))
    receipt = _receipt_for(context, generation=5)

    # Simulate a crash after the server applied the reset but before the
    # acknowledgement was recorded.
    resumed = _ledger(tmp_path, initial_sequence=0)
    assert resumed.session_id == ledger.session_id
    assert resumed.last_episode_seq == context["episode_seq"]
    assert resumed.snapshot()["phase"] == "pending"

    resumed.acknowledge(receipt)

    persisted = _on_disk(tmp_path)
    assert persisted["phase"] == "acknowledged"
    assert persisted["reset_event_id"] == receipt["reset_event_id"]
    assert persisted["server_reset_generation"] == 5


def test_ledger_writes_are_atomic_and_leave_no_temporary_file(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    context = ledger.begin_episode(**_assignment([0]))
    ledger.acknowledge(_receipt_for(context))

    assert sorted(path.name for path in tmp_path.iterdir()) == [LEDGER_NAME]
