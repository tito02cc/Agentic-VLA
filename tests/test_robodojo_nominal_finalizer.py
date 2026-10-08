"""Regression tests for the multi-episode RoboDojo nominal finalizer.

The finalizer used to read only ``result.json["details"]["0"]``, so a run that
evaluated more than one official layout was silently summarized as a single
episode. These tests pin the two properties that made that bug dangerous:
episode segmentation, and keeping each episode's cold first call out of the
steady-state percentiles.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBODOJO_ROOT = _REPO_ROOT / "third_party" / "robodojo_official"

# The finalizer imports the reset contract from the vendored RoboDojo tree,
# whose package __init__ chain pulls in the websocket codec. Neither the extra
# sys.path entry nor the codec dependency is available in the unit-test venv,
# so bootstrap both before loading the module. Nothing here encodes a frame.
try:
    import msgpack_numpy  # noqa: F401
except ModuleNotFoundError:
    _msgpack_numpy_stub = ModuleType("msgpack_numpy")
    _msgpack_numpy_stub.encode = lambda value: value
    _msgpack_numpy_stub.decode = lambda value: value
    sys.modules["msgpack_numpy"] = _msgpack_numpy_stub

for _root in (str(_ROBODOJO_ROOT), str(_ROBODOJO_ROOT / "XPolicyLab")):
    if _root not in sys.path:
        sys.path.insert(0, _root)

_MODULE_PATH = _REPO_ROOT / "scripts" / "finalize_robodojo_nominal_run.py"
_SPEC = importlib.util.spec_from_file_location("finalize_robodojo_nominal_run", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
finalizer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(finalizer)

from XPolicyLab.client_server.ws.protocol.reset import (  # noqa: E402
    RESET_RECEIPT_SCHEMA_VERSION,
    reset_context_digest,
    reset_event_id,
)
from XPolicyLab.policy.starVLA.reset_contract import (  # noqa: E402
    STARVLA_RESET_STATE_FIELDS,
    STARVLA_RESET_STATE_VERSION,
)


def _row(timestep: int, latency: float, *, phase: str = "nominal_fast_path") -> dict:
    return {
        "timestep": timestep,
        "model_latency_ms": latency,
        "episode_id": 0,
        "deadline_miss": False,
        "applied_controls": {"inference_steps": 2, "max_actions": 16},
        "metadata": {
            "action_seed": 1000 + timestep,
            "controller": {"compute_phase": phase},
            "inference_steps_parameter": "num_ddim_steps",
        },
    }


def test_sparse_planner_events_skip_middle_episode_without_shifting():
    monitor = [
        [{"timestamp_s": start, "timestep": 1}, {"timestamp_s": start + 5, "timestep": 500}]
        for start in (10, 20, 30)
    ]
    planner = [{"timestamp_s": 12, "timestep": 165}, {"timestamp_s": 32, "timestep": 350}]
    segments, audit = finalizer._planner_episode_segments(planner, monitor, [{}, {}, {}])
    assert audit["available"] is True
    assert segments == [[planner[0]], [], [planner[1]]]


@pytest.mark.parametrize("change", ["missing_time", "gap", "orphan", "incomplete", "clock_backwards"])
def test_sparse_planner_attribution_fails_closed(change):
    monitor = [
        [{"timestamp_s": 10, "timestep": 1}, {"timestamp_s": 15, "timestep": 500}],
        [{"timestamp_s": 20, "timestep": 1}, {"timestamp_s": 25, "timestep": 500}],
    ]
    planner = [{"timestamp_s": 12, "timestep": 165}]
    if change == "missing_time":
        planner[0].pop("timestamp_s")
    elif change == "gap":
        planner[0]["timestamp_s"] = 18
    elif change == "incomplete":
        monitor.pop()
    elif change == "clock_backwards":
        monitor[1][0]["timestamp_s"] = 9
    segments, audit = finalizer._planner_episode_segments(
        planner, monitor, [{}, {}], has_orphan_resets=change == "orphan"
    )
    assert audit["available"] is False
    assert segments == [[], []]


def test_planner_explicit_generation_works_without_monitor():
    details = [{"policy_reset_receipt": {"reset_generation": generation}} for generation in (1, 2, 3)]
    planner = [{"policy_reset_generation": 3, "timestep": 0}]
    segments, audit = finalizer._planner_episode_segments(planner, [], details)
    assert audit["method"] == "policy_reset_generation"
    assert segments == [[], [], planner]
    planner[0]["policy_reset_generation"] = 4
    assert finalizer._planner_episode_segments(planner, [], details)[1]["available"] is False


def test_planner_identity_without_official_receipt_is_unavailable():
    _, audit = finalizer._planner_episode_segments(
        [{"policy_reset_generation": 1}], [], [{"policy_reset_receipt": None}]
    )
    assert audit["available"] is False


def test_procedure_retrieval_is_not_a_control_intervention():
    result = finalizer._summarize_agentic([{"action": "retrieve_procedure_context"}], [])
    assert result["interventions"] == 0


def test_semantic_planner_and_critic_count_one_committed_recovery():
    event = {"policy_reset_generation": 1, "env_idx": 0, "timestep": 320,
             "action": "semantic_subgoal_blocked_task_only_replan"}
    rows = [{**event, "role": "vlm_planner"}, {**event, "role": "scalar_visual_critic"},
            {**event, "policy_reset_generation": 2}]
    assert finalizer._summarize_agentic(rows, [])["interventions"] == 2


@pytest.mark.parametrize("action", ["critic_inconclusive_no_intervention",
    "critic_unavailable_continue_original", "critic_budget_exhausted_continue_original",
    "critic_confirmed_continue_original_no_stop_authority", "planner_safe_stop_rejected_no_authority",
    "task_plan_shadow_recorded", "unknown_label", "semantic_planner_continue_original",
    "semantic_planner_safe_stop_rejected_no_authority"])
def test_non_execution_records_never_count_as_interventions(action):
    assert finalizer._summarize_agentic([{"action": action}], [])["interventions"] == 0


def test_committed_planner_control_is_not_counted_twice():
    row = {"policy_reset_generation": 1, "env_idx": 0, "timestep": 320,
           "action": "planner_vla_replan", "control_action": "planner_vla_replan",
           "trigger": "no_progress", "decision": {}, "accepted": True}
    assert finalizer._summarize_agentic([row], [])["interventions"] == 1


def test_scalar_planner_stop_is_one_committed_control_event():
    event = {"policy_reset_generation": 1, "env_idx": 0, "timestep": 480,
             "action": "semantic_planner_safe_stop_admitted"}
    rows = [{**event, "role": "vlm_planner"}, {**event, "role": "scalar_visual_critic"}]
    assert finalizer._summarize_agentic(rows, [])["interventions"] == 1


class TestSegmentEpisodes:
    def test_splits_on_timestep_reset(self) -> None:
        rows = [_row(0, 900.0), _row(16, 300.0), _row(0, 910.0), _row(16, 310.0), _row(32, 320.0)]
        segments = finalizer._segment_episodes(rows)
        assert [len(block) for block in segments] == [2, 3]

    def test_repeated_timestep_stays_in_one_episode(self) -> None:
        """Recovery can emit a second chunk at the same timestep."""
        rows = [_row(0, 900.0), _row(16, 300.0), _row(16, 480.0, phase="recovery"), _row(32, 310.0)]
        segments = finalizer._segment_episodes(rows)
        assert len(segments) == 1
        assert len(segments[0]) == 4

    def test_empty_trace_yields_no_segments(self) -> None:
        assert finalizer._segment_episodes([]) == []

    def test_single_episode_is_one_segment(self) -> None:
        rows = [_row(step, 300.0) for step in (0, 16, 32, 48)]
        assert len(finalizer._segment_episodes(rows)) == 1


class TestAlignSegments:
    def test_missing_trace_gives_empty_block_per_episode(self) -> None:
        assert finalizer._align_segments([], 3) == [[], [], []]

    def test_exact_match_passes_through(self) -> None:
        segments = [[_row(0, 1.0)], [_row(0, 2.0)]]
        assert finalizer._align_segments(segments, 2) == segments

    def test_short_trace_is_padded_not_reassigned(self) -> None:
        first = [_row(0, 1.0)]
        aligned = finalizer._align_segments([first], 3)
        assert aligned[0] == first
        assert aligned[1] == [] and aligned[2] == []


class TestSteadyStateExcludesColdCalls:
    def test_per_episode_first_calls_are_excluded(self) -> None:
        segments = [
            [_row(0, 900.0), _row(16, 300.0), _row(32, 302.0)],
            [_row(0, 910.0), _row(16, 304.0)],
        ]
        rows = [row for block in segments for row in block]
        summary = finalizer._summarize_runtime(rows, trace_available=True, segments=segments)

        assert summary["per_episode_first_call_latency_ms"] == [900.0, 910.0]
        assert summary["steady_state_excludes_per_episode_first_call"] is True
        # 900/910 must not reach the steady-state percentiles.
        assert summary["steady_state_model_latency_max_ms"] == 304.0
        # The unscoped percentile still sees them, which is why scoping matters.
        assert summary["model_latency_p95_ms"] == 910.0
        assert summary["vla_calls"] == 5
        assert summary["deadline_misses"] == 0

    def test_without_segments_falls_back_to_run_level(self) -> None:
        rows = [_row(0, 900.0), _row(16, 300.0)]
        summary = finalizer._summarize_runtime(rows, trace_available=True)
        assert summary["steady_state_excludes_per_episode_first_call"] is False
        assert summary["per_episode_first_call_latency_ms"] == [900.0]

    def test_unavailable_trace_reports_none_not_zero(self) -> None:
        summary = finalizer._summarize_runtime([], trace_available=False)
        assert summary["trace_available"] is False
        assert summary["vla_calls"] is None
        assert summary["deadline_misses"] is None
        assert summary["per_episode_first_call_latency_ms"] is None


class TestSchemaVersioning:
    """A multi-episode run must not be readable as a single-episode summary."""

    @staticmethod
    def _write_run(tmp_path: Path, episode_count: int) -> Path:
        details = {
            str(index): {
                "layout_id": index,
                "success": index == 1,
                "score": 1.0 if index == 1 else 0.0,
            }
            for index in range(episode_count)
        }
        successes = sum(detail["success"] for detail in details.values())
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "success_rate": successes / episode_count,
                    "eval_time": episode_count,
                    "score": successes / episode_count,
                    "details": details,
                }
            )
        )
        return tmp_path

    def _finalize(self, tmp_path: Path, episode_count: int) -> dict:
        self._write_run(tmp_path, episode_count)
        import sys

        argv = sys.argv
        sys.argv = [
            "finalize_robodojo_nominal_run.py",
            "--condition",
            "C3",
            "--task",
            "stack_bowls",
            "--seed",
            "1",
            "--artifact",
            str(tmp_path),
            "--official-run",
            str(tmp_path),
        ]
        try:
            finalizer.main()
        finally:
            sys.argv = argv
        return json.loads((tmp_path / "summary.json").read_text())

    def test_single_episode_keeps_v1(self, tmp_path: Path) -> None:
        summary = self._finalize(tmp_path, 1)
        assert summary["schema_version"] == "carve.robodojo.nominal.v1"
        # v1 consumers read these scalars directly.
        assert summary["official_result"]["success"] is False
        assert summary["layout_id"] == 0

    def test_multi_episode_bumps_to_v2(self, tmp_path: Path) -> None:
        summary = self._finalize(tmp_path, 3)
        assert summary["schema_version"] == "carve.robodojo.nominal.v2"
        # Scalars that cannot describe 3 episodes are nulled so a v1 reader
        # cannot quietly treat episode 0 as the whole run.
        assert summary["official_result"]["success"] is None
        assert summary["official_result"]["successes"] == 1
        assert summary["official_result"]["trials"] == 3
        assert summary["aggregate"]["layout_ids"] == [0, 1, 2]
        assert summary["aggregate"]["successful_layout_ids"] == [1]
        assert summary["aggregate"]["failed_layout_ids"] == [0, 2]
        assert len(summary["episodes"]) == 3

    def test_every_episode_is_summarized(self, tmp_path: Path) -> None:
        summary = self._finalize(tmp_path, 5)
        assert [episode["episode_index"] for episode in summary["episodes"]] == [0, 1, 2, 3, 4]
        assert [episode["layout_id"] for episode in summary["episodes"]] == [0, 1, 2, 3, 4]
        assert summary["aggregate"]["episodes"] == 5

    def test_empty_details_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "result.json").write_text(
            json.dumps({"success_rate": 0.0, "eval_time": 0, "score": 0.0, "details": {}})
        )
        import sys

        argv = sys.argv
        sys.argv = [
            "finalize_robodojo_nominal_run.py",
            "--condition",
            "B0",
            "--task",
            "stack_bowls",
            "--seed",
            "0",
            "--artifact",
            str(tmp_path),
            "--official-run",
            str(tmp_path),
        ]
        try:
            with pytest.raises(ValueError, match="no episodes"):
                finalizer.main()
        finally:
            sys.argv = argv


RUN_ID = "stack_bowls-C3-seed1"
SESSION_ID = "9f" * 16
SERVER_INSTANCE_ID = "server-instance-1"
STARVLA_MODEL_PATH = (
    _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "starVLA" / "model.py"
)
# The audited run pins the exact adapter source the server loaded, so the
# fixture uses the real digest rather than an invented one.
MODEL_CODE_SHA256 = hashlib.sha256(STARVLA_MODEL_PATH.read_bytes()).hexdigest()
RESET_CONTRACT_PATH = (
    _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "starVLA" / "reset_contract.py"
)
# The 31-item inventory lives in this file, outside model.py's digest, and this
# is also the file the finalizer imports to build its expectations. Computed from
# the real source for the same reason as above: a literal would let the fixture
# and the audit drift apart without any test noticing.
RESET_CONTRACT_SHA256 = hashlib.sha256(RESET_CONTRACT_PATH.read_bytes()).hexdigest()

_UNSET = object()


def _reset_receipt(
    *,
    sequence: int,
    generation: int,
    layout_ids: list[int],
    env_ids: list[int] | None = None,
    run_id: str = RUN_ID,
    session_id: str = SESSION_ID,
    server_instance_id: str = SERVER_INSTANCE_ID,
    model_module_id: str = finalizer.MODEL_MODULE_ID,
    model_code_sha256: str = MODEL_CODE_SHA256,
    replayed: bool = False,
) -> dict:
    """Build a complete, contract-legal receipt from the shared constants."""

    if env_ids is None:
        env_ids = list(range(len(layout_ids)))
    context = {
        "episode_id": f"{run_id}:{session_id}:batch-{sequence:07d}",
        "episode_seq": sequence,
        "reset_session_id": session_id,
        "active_env_ids": list(env_ids),
        "layout_seeds": [
            {"env_idx": env_idx, "layout_id": layout_id}
            for env_idx, layout_id in zip(env_ids, layout_ids, strict=True)
        ],
    }
    context_digest = reset_context_digest(context)
    return {
        "reset_receipt_schema_version": RESET_RECEIPT_SCHEMA_VERSION,
        "model_reset_schema_version": finalizer.MODEL_RESET_SCHEMA_VERSION,
        "state_inventory_version": STARVLA_RESET_STATE_VERSION,
        "state_inventory": list(STARVLA_RESET_STATE_FIELDS),
        **context,
        "context_digest": context_digest,
        "evaluation_id": run_id,
        "trial_id": f"stack_bowls-{run_id}",
        "action_case_id": "stack_bowls_case",
        "repeat_index": None,
        "server_instance_id": server_instance_id,
        "server_reset_generation": generation,
        "reset_event_id": reset_event_id(
            server_instance_id=server_instance_id,
            server_reset_generation=generation,
            context_digest=context_digest,
        ),
        "model_module_id": model_module_id,
        "model_code_sha256": model_code_sha256,
        "applied_once": True,
        "applied_this_request": not replayed,
        "response_replayed": replayed,
        "applied": not replayed,
        "replayed": replayed,
        "reset_generation": sequence,
        "before": {
            "state_entries": {name: 1 for name in STARVLA_RESET_STATE_FIELDS},
            "high_level_calls": 3,
        },
        "after": {
            "state_entries": {name: 0 for name in STARVLA_RESET_STATE_FIELDS},
            "high_level_calls": 0,
        },
    }


def _default_events() -> list[tuple[dict, list[int]]]:
    """Three single-env episodes: the ordinary audited nominal run."""

    return [
        (
            _reset_receipt(sequence=index + 1, generation=index + 1, layout_ids=[index]),
            [index],
        )
        for index in range(3)
    ]


def _batch_event() -> list[tuple[dict, list[int]]]:
    """One physical reset covering three parallel official layouts."""

    return [
        (
            _reset_receipt(
                sequence=1, generation=1, env_ids=[0, 1, 2], layout_ids=[5, 6, 7]
            ),
            [5, 6, 7],
        )
    ]


def _abandoned_run(
    abandoned_index: int, *, applied: int = 4
) -> tuple[list[tuple[dict, list[int]]], list[tuple[dict, list[int]]]]:
    """``applied`` resets of which one physical episode was never scored.

    RoboDojo abandons a physical episode *after* its reset already happened (a
    broken PhysX env, a fatal restart, an unstable env dropped from eval_envs, or
    any mid-rollout exception that consumes the seed). The reset stays in the
    trace and keeps its generation, but no official detail references it.

    Returns ``(applied_events, scored_events)``: the first drives the trace and
    the ledger, the second drives result.json.
    """

    events = [
        (
            _reset_receipt(
                sequence=index + 1, generation=index + 1, layout_ids=[index]
            ),
            [index],
        )
        for index in range(applied)
    ]
    scored = [
        event for index, event in enumerate(events) if index != abandoned_index
    ]
    return events, scored


def _details_for(events: list[tuple[dict, list[int]]]) -> list[dict]:
    details: list[dict] = []
    for receipt, layout_ids in events:
        for layout_id in layout_ids:
            details.append(
                {
                    "layout_id": layout_id,
                    "success": False,
                    "score": 0.0,
                    # Every env covered by one batch reset stores the same
                    # receipt, so the copies must stay byte-identical.
                    "policy_reset_receipt": copy.deepcopy(receipt),
                }
            )
    return details


def _mutated(receipt: dict, path: tuple[str, ...], value: Any) -> dict:
    """Return a copy of one receipt with a single nested field replaced."""

    mutated = copy.deepcopy(receipt)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def _run_config(**overrides: Any) -> dict:
    config = {
        "require_audited_reset": True,
        "expected_model_module_id": finalizer.MODEL_MODULE_ID,
        "expected_model_code_sha256": MODEL_CODE_SHA256,
        "expected_model_reset_schema_version": finalizer.MODEL_RESET_SCHEMA_VERSION,
        "expected_reset_receipt_schema_version": RESET_RECEIPT_SCHEMA_VERSION,
        "expected_state_inventory_version": STARVLA_RESET_STATE_VERSION,
        # Bind the artifact to the one official run directory its evidence may
        # come from, and to the exact contract file the audit loaded.
        "robodojo_run_id": RUN_ID,
        "expected_reset_contract_sha256": RESET_CONTRACT_SHA256,
    }
    config.update(overrides)
    return config


def _run_config_without(*keys: str) -> dict:
    """The launcher-written config with specific bindings left out."""

    config = _run_config()
    for key in keys:
        del config[key]
    return config


def _ledger_for(receipt: dict, **overrides: Any) -> dict:
    run_id = receipt["episode_id"].split(":", 1)[0]
    ledger = {
        "schema_version": finalizer.POLICY_RESET_LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "session_id": receipt["reset_session_id"],
        "last_episode_seq": receipt["episode_seq"],
        "phase": "acknowledged",
        "episode_id": receipt["episode_id"],
        "context_digest": receipt["context_digest"],
        "reset_event_id": receipt["reset_event_id"],
        "server_instance_id": receipt["server_instance_id"],
        "server_reset_generation": receipt["server_reset_generation"],
    }
    ledger.update(overrides)
    return ledger


def _audit(
    events: list[tuple[dict, list[int]]] | None = None,
    *,
    details: list[dict] | None = None,
    trace: list[dict] | object = _UNSET,
    ledger: dict | None | object = _UNSET,
    run_config: dict | None | object = _UNSET,
    trace_present: bool = True,
    ledger_present: bool = True,
    run_config_source: str | None = "run_config.json",
    input_errors: tuple[str, ...] = (),
) -> dict:
    """Audit an in-memory evidence bundle the way ``main`` audits files."""

    events = _default_events() if events is None else events
    receipts = [receipt for receipt, _ in events]
    if details is None:
        details = _details_for(events)
    if trace is _UNSET:
        trace = [copy.deepcopy(receipt) for receipt in receipts]
    if ledger is _UNSET:
        ledger = _ledger_for(receipts[-1])
    if run_config is _UNSET:
        run_config = _run_config()
    # Round-trip through JSON so the audit sees exactly the value types it
    # would read back from result.json / reset_receipts.jsonl.
    payload = json.loads(
        json.dumps(
            {
                "details": details,
                "trace": trace,
                "ledger": ledger,
                "run_config": run_config,
            }
        )
    )
    return finalizer._audit_policy_resets(
        payload["details"],
        payload["trace"],
        reset_trace_present=trace_present,
        ledger=payload["ledger"],
        ledger_present=ledger_present,
        run_config=payload["run_config"],
        run_config_source=run_config_source,
        input_errors=list(input_errors),
    )


def _failed_invariants(audit: dict) -> list[str]:
    return sorted(
        name for name, passed in audit["invariants"].items() if passed is False
    )


def _write_evidence(
    tmp_path: Path,
    events: list[tuple[dict, list[int]]] | None = None,
    *,
    details: list[dict] | None = None,
    trace: list[dict] | None | object = _UNSET,
    trace_text: str | None = None,
    ledger: dict | None | object = _UNSET,
    run_config: dict | None | object = _UNSET,
) -> list[dict]:
    """Write one evidence bundle on disk exactly as an audited run leaves it.

    ``events`` is the record of applied resets (the trace and the ledger anchor);
    ``details`` defaults to one scored episode per covered layout but can be
    passed explicitly when a physical episode was abandoned after its reset.
    """

    events = _default_events() if events is None else events
    receipts = [receipt for receipt, _ in events]
    if details is None:
        details = _details_for(events)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "success_rate": 0.0,
                "eval_time": len(details),
                "score": 0.0,
                "details": {
                    str(index): detail for index, detail in enumerate(details)
                },
            }
        ),
        encoding="utf-8",
    )
    if trace_text is not None:
        (tmp_path / "reset_receipts.jsonl").write_text(trace_text, encoding="utf-8")
    else:
        rows = receipts if trace is _UNSET else trace
        if rows is not None:
            (tmp_path / "reset_receipts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
    if ledger is not None:
        state = _ledger_for(receipts[-1]) if ledger is _UNSET else ledger
        (tmp_path / "policy_reset_ledger.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    if run_config is not None:
        config = _run_config() if run_config is _UNSET else run_config
        (tmp_path / "run_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    return details


def _run_finalizer(
    tmp_path: Path,
    *,
    require_valid_reset_audit: bool = False,
    expected_episodes: int | None = None,
) -> None:
    argv = sys.argv
    sys.argv = [
        "finalize_robodojo_nominal_run.py",
        "--condition",
        "C3",
        "--task",
        "stack_bowls",
        "--seed",
        "1",
        "--artifact",
        str(tmp_path),
        "--official-run",
        str(tmp_path),
    ]
    if require_valid_reset_audit:
        sys.argv.append("--require-valid-reset-audit")
    # The strict gate also requires the pre-registered episode count: the reset
    # audit tolerates abandoned physical episodes, so a truncated block would
    # otherwise pass it. Kept optional so the "not declared" case is testable.
    if expected_episodes is not None:
        sys.argv.extend(["--expected-episodes", str(expected_episodes)])
    try:
        finalizer.main()
    finally:
        sys.argv = argv


class TestResetReceiptAudit:
    def test_valid_episode_receipts_are_audited(self) -> None:
        audit = _audit()

        assert _failed_invariants(audit) == []
        assert audit["valid"] is True
        assert audit["errors"] == []
        assert audit["schema_version"] == finalizer.POLICY_RESET_AUDIT_SCHEMA_VERSION
        assert audit["strict_mode_requested_by_run_config"] is True
        assert audit["detail_episodes"] == 3
        assert audit["detail_receipts"] == 3
        assert audit["unique_reset_events"] == 3
        assert audit["reset_receipt_trace_rows"] == 3
        assert audit["server_instance_ids"] == [SERVER_INSTANCE_ID]
        assert audit["reset_session_ids"] == [SESSION_ID]
        assert audit["server_reset_generations"] == [1, 2, 3]
        assert audit["episode_sequences"] == [1, 2, 3]
        assert audit["run_ids"] == [RUN_ID]
        assert audit["newly_applied_events"] == 3
        assert audit["replayed_response_events"] == 0
        assert audit["expected"] == {
            "model_module_id": finalizer.MODEL_MODULE_ID,
            "model_code_sha256": MODEL_CODE_SHA256,
            "model_reset_schema_version": finalizer.MODEL_RESET_SCHEMA_VERSION,
            "reset_receipt_schema_version": RESET_RECEIPT_SCHEMA_VERSION,
            "state_inventory_version": STARVLA_RESET_STATE_VERSION,
        }
        assert audit["detail_event_fingerprints"] == audit["trace_event_fingerprints"]
        assert [event["detail_layout_ids"] for event in audit["events"]] == [
            [0],
            [1],
            [2],
        ]
        # The optional reobservation state participates in the full contract.
        assert len(STARVLA_RESET_STATE_FIELDS) == 36
        assert "reobservation_by_env" in STARVLA_RESET_STATE_FIELDS
        # The stall-evidence scheduling state is episode state too: a leaked
        # "last counted observation" would let episode N+1 inherit N's schedule.
        assert "semantic_last_counted_step_by_env" in STARVLA_RESET_STATE_FIELDS
        assert audit["state_inventory_complete"] is True

    def test_valid_receipts_survive_the_full_finalizer(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)

        _run_finalizer(tmp_path, require_valid_reset_audit=True, expected_episodes=3)

        summary = json.loads((tmp_path / "summary.json").read_text())
        audit = summary["policy_reset_audit"]
        assert audit["valid"] is True
        assert summary["aggregate"]["policy_reset_audit_valid"] is True
        assert all(
            episode["policy_reset_receipt"] is not None
            for episode in summary["episodes"]
        )
        # The standalone audit artifact is always written, and matches summary.
        standalone = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert standalone == audit

    def test_replayed_response_event_is_still_a_pass(self) -> None:
        events = _default_events()
        events[1] = (
            _reset_receipt(
                sequence=2, generation=2, layout_ids=[1], replayed=True
            ),
            [1],
        )
        audit = _audit(events)

        assert _failed_invariants(audit) == []
        assert audit["valid"] is True
        assert audit["newly_applied_events"] == 2
        assert audit["replayed_response_events"] == 1
        assert audit["receipt_replay_states_valid"] is True

    def test_batch_receipt_covering_multiple_episodes_is_valid(self) -> None:
        audit = _audit(_batch_event())

        assert _failed_invariants(audit) == []
        assert audit["valid"] is True
        assert audit["detail_episodes"] == 3
        assert audit["detail_receipts"] == 3
        assert audit["unique_reset_events"] == 1
        assert audit["reset_receipt_trace_rows"] == 1
        assert audit["events"][0]["detail_episode_indexes"] == [0, 1, 2]
        assert audit["events"][0]["detail_layout_ids"] == [5, 6, 7]
        assert audit["events"][0]["active_env_ids"] == [0, 1, 2]

    def test_batch_receipt_survives_the_full_finalizer(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, _batch_event())

        _run_finalizer(tmp_path, require_valid_reset_audit=True, expected_episodes=3)

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["policy_reset_audit"]["valid"] is True
        assert [episode["layout_id"] for episode in summary["episodes"]] == [5, 6, 7]
        assert summary["policy_reset_audit"]["unique_reset_events"] == 1


class TestAbandonedPhysicalEpisodes:
    """An applied reset with no scored episode is evidence, not a defect.

    Continuity is a property of every reset the server applied, so the audit
    bases identity, ordering and 1..N on reset_receipts.jsonl and requires the
    scored episodes to be an ordered subsequence of it. Auditing continuity over
    the scored subset instead used to condemn an otherwise valid condition and
    force a full re-run, which is the regression these tests pin.
    """

    @pytest.mark.parametrize(
        "abandoned_index",
        [
            pytest.param(0, id="abandoned_first"),
            pytest.param(1, id="abandoned_second"),
            pytest.param(2, id="abandoned_third"),
            pytest.param(3, id="abandoned_last"),
        ],
    )
    def test_orphan_reset_event_is_reported_not_failed(
        self, abandoned_index: int
    ) -> None:
        events, scored = _abandoned_run(abandoned_index)
        orphan = events[abandoned_index][0]

        audit = _audit(events, details=_details_for(scored))

        assert _failed_invariants(audit) == []
        assert audit["valid"] is True
        assert audit["errors"] == []
        assert audit["orphan_reset_events"] == 1
        assert audit["orphan_reset_event_ids"] == [orphan["reset_event_id"]]
        assert audit["scored_reset_events"] == 3
        # Four applied resets, three scored episodes, no hole in the trace.
        assert audit["reset_receipt_trace_rows"] == 4
        assert audit["unique_reset_events"] == 3
        assert audit["detail_episodes"] == 3
        assert audit["detail_receipts"] == 3
        assert audit["server_reset_generations"] == [1, 2, 3, 4]
        assert audit["episode_sequences"] == [1, 2, 3, 4]
        assert audit["server_reset_generations_exact_1_to_n"] is True
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is True
        # The orphan is only in the trace, never in the scored event list.
        scored_event_ids = [event["reset_event_id"] for event in audit["events"]]
        assert orphan["reset_event_id"] not in scored_event_ids
        assert len(scored_event_ids) == 3

    def test_no_abandoned_episode_reports_zero_orphans(self) -> None:
        audit = _audit()

        assert audit["valid"] is True
        assert audit["orphan_reset_events"] == 0
        assert audit["orphan_reset_event_ids"] == []
        assert audit["scored_reset_events"] == 3

    def test_ledger_anchors_on_the_abandoned_final_reset(self) -> None:
        """The ledger records the last *applied* reset, not the last scored one."""

        events, scored = _abandoned_run(3)
        orphan = events[3][0]

        audit = _audit(
            events, details=_details_for(scored), ledger=_ledger_for(orphan)
        )

        assert _failed_invariants(audit) == []
        assert audit["valid"] is True
        assert audit["ledger_sequence_matches"] is True
        assert audit["ledger_last_event_matches"] is True
        assert audit["orphan_reset_event_ids"] == [orphan["reset_event_id"]]

    def test_ledger_must_not_stop_at_the_last_scored_reset(self) -> None:
        events, scored = _abandoned_run(3)

        audit = _audit(
            events,
            details=_details_for(scored),
            ledger=_ledger_for(scored[-1][0]),
        )

        assert audit["valid"] is False
        assert audit["ledger_sequence_matches"] is False
        assert audit["ledger_last_event_matches"] is False

    @pytest.mark.parametrize(
        "abandoned_index",
        [
            pytest.param(0, id="abandoned_first"),
            pytest.param(1, id="abandoned_middle"),
            pytest.param(3, id="abandoned_last"),
        ],
    )
    def test_abandoned_episode_survives_the_full_finalizer(
        self, tmp_path: Path, abandoned_index: int
    ) -> None:
        """The strict gate must publish this run instead of condemning it."""

        events, scored = _abandoned_run(abandoned_index)
        _write_evidence(tmp_path, events, details=_details_for(scored))

        # Three episodes were pre-registered and three were scored: the fourth
        # applied reset never produced an episode, which is what makes this run
        # complete *and* auditable.
        _run_finalizer(tmp_path, require_valid_reset_audit=True, expected_episodes=3)

        summary = json.loads((tmp_path / "summary.json").read_text())
        audit = summary["policy_reset_audit"]
        assert audit["valid"] is True
        assert summary["aggregate"]["policy_reset_audit_valid"] is True
        assert audit["orphan_reset_events"] == 1
        assert audit["orphan_reset_event_ids"] == [
            events[abandoned_index][0]["reset_event_id"]
        ]
        assert audit["scored_reset_events"] == 3
        assert len(summary["episodes"]) == 3
        assert summary["aggregate"]["layout_ids"] == [
            layout for _, layouts in scored for layout in layouts
        ]
        standalone = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert standalone == audit

    def test_a_hole_in_the_trace_still_fails(self) -> None:
        """An unrecorded applied reset is not the same thing as an orphan.

        Same scored episodes as the orphan case, but the abandoned reset's trace
        row is gone, so the trace itself is missing an application.
        """

        events, scored = _abandoned_run(1)
        trace = [
            copy.deepcopy(receipt)
            for index, (receipt, _) in enumerate(events)
            if index != 1
        ]

        audit = _audit(events, details=_details_for(scored), trace=trace)

        assert audit["valid"] is False
        assert audit["server_reset_generations_exact_1_to_n"] is False
        assert audit["server_reset_generations"] == [1, 3, 4]
        assert audit["orphan_reset_events"] == 0

    @pytest.mark.parametrize(
        ("overrides", "failed_invariant"),
        [
            pytest.param(
                {"server_instance_id": "server-instance-2"},
                "server_instance_consistent",
                id="server_instance",
            ),
            pytest.param(
                {"session_id": "ab" * 16},
                "reset_session_consistent",
                id="reset_session",
            ),
        ],
    )
    def test_orphan_trace_rows_are_held_to_the_same_identity(
        self, overrides: dict, failed_invariant: str
    ) -> None:
        """Being unscored does not exempt an applied reset from the audit."""

        events, scored = _abandoned_run(1)
        events[1] = (
            _reset_receipt(sequence=2, generation=2, layout_ids=[1], **overrides),
            [1],
        )

        audit = _audit(events, details=_details_for(scored))

        assert audit["valid"] is False
        assert audit[failed_invariant] is False
        # Every scored receipt still agrees with its trace row.
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is True

    @pytest.mark.parametrize(
        ("path", "value", "failed_invariant"),
        [
            pytest.param(
                ("model_module_id",),
                "XPolicyLab.policy.demo_policy.model",
                "model_module_id_matches",
                id="model_module_id",
            ),
            pytest.param(
                ("model_code_sha256",),
                "c" * 64,
                "model_code_hash_matches_run_config",
                id="model_code_sha256",
            ),
            pytest.param(
                ("after", "state_entries", "step_by_env"),
                1,
                "starvla_reset_states_valid",
                id="state_entry_left_populated",
            ),
            pytest.param(
                ("after", "high_level_calls"),
                2,
                "starvla_reset_states_valid",
                id="high_level_budget_not_reset",
            ),
            pytest.param(
                ("reset_receipt_schema_version",),
                "xpolicylab.episode-reset.v0",
                "reset_receipt_schema_matches",
                id="reset_receipt_schema_version",
            ),
        ],
    )
    def test_orphan_trace_row_contract_violations_condemn_the_run(
        self, path: tuple[str, ...], value: Any, failed_invariant: str
    ) -> None:
        """The contract is checked on every applied reset, not the scored subset.

        Tolerating an abandoned attempt is about *continuity*: it cannot weaken
        the isolation of the scored episodes. It is not an exemption from the
        contract, so an unscored receipt that claims a different adapter, a
        different schema or a container it never emptied must still fail.
        """

        events, scored = _abandoned_run(1)
        orphan = _mutated(events[1][0], path, value)
        events[1] = (orphan, events[1][1])

        audit = _audit(events, details=_details_for(scored))

        assert audit["valid"] is False
        assert audit[failed_invariant] is False
        # The violating receipt exists only in the trace: no official detail
        # points at it, and the scored receipts are untouched.
        assert audit["orphan_reset_event_ids"] == [orphan["reset_event_id"]]
        assert audit["scored_reset_events"] == 3
        assert audit["detail_receipts_present_for_all_episodes"] is True
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is True


class TestOrphanEventAttribution:
    """A reported orphan id must name the trace row that was really unmatched.

    ``_read_audit_jsonl`` parses with ``json.loads``, which accepts ``NaN``,
    while ``_canonical_json`` requires finite JSON. A row that passes the first
    and fails the second is dropped from the canonical list, so positions in that
    list stop lining up with rows in the trace. Reporting orphans by canonical
    position rather than by row index therefore misattributed the ids, and could
    name an event that a scored episode actually points at.
    """

    @staticmethod
    def _run_with_an_uncanonicalizable_row() -> tuple[
        list[tuple[dict, list[int]]], list[tuple[dict, list[int]]]
    ]:
        """Four applied resets: row 0 non-finite, rows 1-2 scored, row 3 orphan.

        The offset matters: with one row skipped, canonical position 2 is trace
        row 3, while the scored rows sit at positions 0 and 1. Reading the id at
        the position instead of at the row index names a scored event.
        """

        events = [
            (
                _reset_receipt(
                    sequence=index + 1, generation=index + 1, layout_ids=[index]
                ),
                [index],
            )
            for index in range(4)
        ]
        events[0] = (_mutated(events[0][0], ("repeat_index",), float("nan")), [0])
        return events, [events[1], events[2]]

    def test_uncanonicalizable_row_never_orphans_a_scored_event(
        self, tmp_path: Path
    ) -> None:
        events, scored = self._run_with_an_uncanonicalizable_row()
        _write_evidence(tmp_path, events, details=_details_for(scored))

        _run_finalizer(tmp_path)

        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        scored_event_ids = {receipt["reset_event_id"] for receipt, _ in scored}
        # The row parsed as JSON, so this is a canonicalization failure and must
        # be reported as one rather than as an unreadable input.
        assert audit["audit_inputs_parseable"] is True
        assert any(
            "reset trace row 1 cannot be canonicalized" in error
            for error in audit["errors"]
        )
        # The invariant under test: whatever is reported as an orphan, it is
        # never an event a scored episode points at.
        assert set(audit["orphan_reset_event_ids"]).isdisjoint(scored_event_ids)
        assert [event["reset_event_id"] for event in audit["events"]] == [
            receipt["reset_event_id"] for receipt, _ in scored
        ]
        # A trace row the run cannot canonicalize is still a defect: the scored
        # receipts can no longer be proven to be a subsequence of the trace.
        assert audit["valid"] is False
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is False


class TestResetReceiptAuditRejections:
    """Every way an audited run can be wrong must fail the audit."""

    @pytest.mark.parametrize("field", STARVLA_RESET_STATE_FIELDS)
    def test_missing_inventory_entry_fails(self, field: str) -> None:
        events = _default_events()
        receipt, layouts = events[1]
        receipt = copy.deepcopy(receipt)
        del receipt["after"]["state_entries"][field]
        events[1] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["starvla_reset_states_valid"] is False
        assert any("state-entry keys mismatch" in error for error in audit["errors"])

    @pytest.mark.parametrize("field", STARVLA_RESET_STATE_FIELDS)
    def test_non_zero_inventory_entry_fails(self, field: str) -> None:
        events = _default_events()
        receipt, layouts = events[2]
        receipt = copy.deepcopy(receipt)
        receipt["after"]["state_entries"][field] = 1
        events[2] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["starvla_reset_states_valid"] is False
        assert any(
            "left episode state populated" in error for error in audit["errors"]
        )

    def test_unreset_high_level_budget_fails(self) -> None:
        events = _default_events()
        receipt, layouts = events[0]
        receipt = copy.deepcopy(receipt)
        receipt["after"]["high_level_calls"] = 2
        events[0] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["starvla_reset_states_valid"] is False
        assert any("high-level budget" in error for error in audit["errors"])

    def test_truncated_state_inventory_fails(self) -> None:
        events = _default_events()
        receipt, layouts = events[0]
        receipt = copy.deepcopy(receipt)
        receipt["state_inventory"] = list(STARVLA_RESET_STATE_FIELDS[:-1])
        events[0] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["state_inventory_complete"] is False
        assert audit["starvla_reset_states_valid"] is False

    @pytest.mark.parametrize(
        ("generations", "case"),
        [
            pytest.param([1, 2, 4], "gap", id="gap"),
            pytest.param([1, 2, 2], "duplicate", id="duplicate_tail"),
            pytest.param([1, 1, 2], "duplicate", id="duplicate_head"),
            pytest.param([2, 3, 4], "offset", id="does_not_start_at_one"),
            pytest.param([3, 2, 1], "descending", id="descending"),
        ],
    )
    def test_generation_sequence_must_be_exactly_one_to_n(
        self, generations: list[int], case: str
    ) -> None:
        """1..N is read off the trace of applied resets, not the scored subset.

        The trace is the basis, so a gap here means the server applied a reset
        this run never recorded. An abandoned episode cannot produce one: it keeps
        its trace row and its generation, which ``TestAbandonedPhysicalEpisodes``
        pins separately.
        """

        del case
        events = [
            (
                _reset_receipt(
                    sequence=index + 1,
                    generation=generation,
                    layout_ids=[index],
                ),
                [index],
            )
            for index, generation in enumerate(generations)
        ]
        trace = [copy.deepcopy(receipt) for receipt, _ in events]

        audit = _audit(events, trace=trace)

        assert audit["valid"] is False
        assert audit["server_reset_generations_exact_1_to_n"] is False
        # The reported generations are exactly the trace rows, in trace order.
        assert audit["server_reset_generations"] == generations
        assert [row["server_reset_generation"] for row in trace] == generations
        # Only continuity broke: the scored episodes still match the trace.
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is True
        assert audit["orphan_reset_events"] == 0

    def test_trace_generation_gap_fails_even_with_contiguous_scored_events(
        self,
    ) -> None:
        """A hole in the trace fails while the scored details look perfect."""

        events = [
            (
                _reset_receipt(
                    sequence=index + 1, generation=generation, layout_ids=[index]
                ),
                [index],
            )
            for index, generation in enumerate((1, 2, 4))
        ]

        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["server_reset_generations_exact_1_to_n"] is False
        assert audit["server_reset_generations"] == [1, 2, 4]
        assert audit["detail_receipts_present_for_all_episodes"] is True
        assert audit["episode_sequences_unique_and_increasing"] is True

    def test_changed_server_instance_fails(self) -> None:
        events = _default_events()
        events[1] = (
            _reset_receipt(
                sequence=2,
                generation=2,
                layout_ids=[1],
                server_instance_id="server-instance-2",
            ),
            [1],
        )
        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["server_instance_consistent"] is False
        # The receipts are individually well formed; only exclusivity broke.
        assert audit["generic_reset_receipts_valid"] is True
        assert audit["server_instance_ids"] == [
            SERVER_INSTANCE_ID,
            "server-instance-2",
        ]

    def test_changed_reset_session_fails(self) -> None:
        events = _default_events()
        events[2] = (
            _reset_receipt(
                sequence=3, generation=3, layout_ids=[2], session_id="ab" * 16
            ),
            [2],
        )
        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["reset_session_consistent"] is False

    def test_wrong_model_module_id_fails(self) -> None:
        events = _default_events()
        events[0] = (
            _reset_receipt(
                sequence=1,
                generation=1,
                layout_ids=[0],
                model_module_id="XPolicyLab.policy.demo_policy.model",
            ),
            [0],
        )
        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["model_module_id_matches"] is False

    def test_model_code_hash_must_match_run_config(self) -> None:
        events = _default_events()
        events[1] = (
            _reset_receipt(
                sequence=2,
                generation=2,
                layout_ids=[1],
                model_code_sha256="c" * 64,
            ),
            [1],
        )
        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["model_code_hash_matches_run_config"] is False

    def test_run_config_hash_must_be_a_digest(self) -> None:
        audit = _audit(run_config=_run_config(expected_model_code_sha256="deadbeef"))

        assert audit["valid"] is False
        assert audit["expected_model_hash_is_sha256"] is False
        assert audit["model_code_hash_matches_run_config"] is False

    @pytest.mark.parametrize(
        ("overrides", "failed_invariant"),
        [
            pytest.param(
                {"require_audited_reset": False},
                "run_config_requires_audited_reset",
                id="strict_mode_off",
            ),
            pytest.param(
                {"expected_model_module_id": "XPolicyLab.policy.demo_policy.model"},
                "expected_model_module_matches_fixed_contract",
                id="wrong_expected_module",
            ),
            pytest.param(
                {"expected_model_reset_schema_version": "carve.policy-reset.v0"},
                "expected_model_schema_matches_fixed_contract",
                id="wrong_expected_model_schema",
            ),
            pytest.param(
                {"expected_reset_receipt_schema_version": "xpolicylab.episode-reset.v0"},
                "expected_receipt_schema_matches_fixed_contract",
                id="wrong_expected_receipt_schema",
            ),
            pytest.param(
                {"expected_state_inventory_version": "starvla.episode-state.v0"},
                "expected_state_schema_matches_fixed_contract",
                id="wrong_expected_state_schema",
            ),
        ],
    )
    def test_run_config_expectations_are_enforced(
        self, overrides: dict, failed_invariant: str
    ) -> None:
        audit = _audit(run_config=_run_config(**overrides))

        assert audit["valid"] is False
        assert audit[failed_invariant] is False

    def test_missing_run_config_fails(self) -> None:
        audit = _audit(run_config=None, run_config_source=None)

        assert audit["valid"] is False
        assert audit["run_config_present"] is False
        assert audit["run_config_requires_audited_reset"] is False
        assert audit["expected_model_identity_present"] is False

    @pytest.mark.parametrize(
        ("mutate", "failed_invariant", "case"),
        [
            pytest.param(
                lambda receipts: list(reversed(receipts)),
                "scored_events_are_ordered_subsequence_of_trace",
                "reordered",
                id="reordered_rows",
            ),
            pytest.param(
                lambda receipts: receipts[:-1],
                "scored_events_are_ordered_subsequence_of_trace",
                "absent",
                id="scored_receipt_absent_from_trace",
            ),
            pytest.param(
                lambda receipts: receipts + [copy.deepcopy(receipts[-1])],
                "trace_event_ids_unique",
                "duplicated",
                id="duplicated_row",
            ),
            pytest.param(
                lambda receipts: receipts[:1]
                + [{**copy.deepcopy(receipts[1]), "applied": False}]
                + receipts[2:],
                "scored_events_are_ordered_subsequence_of_trace",
                "mutated",
                id="mutated_field",
            ),
            pytest.param(
                lambda receipts: [
                    {**copy.deepcopy(row), "extra_field": "sneaky"} for row in receipts
                ],
                "scored_events_are_ordered_subsequence_of_trace",
                "augmented",
                id="added_field",
            ),
            pytest.param(
                lambda receipts: receipts[:2]
                + [
                    {
                        key: value
                        for key, value in copy.deepcopy(receipts[2]).items()
                        if key != "repeat_index"
                    }
                ],
                "scored_events_are_ordered_subsequence_of_trace",
                "pruned",
                id="dropped_field",
            ),
        ],
    )
    def test_trace_must_agree_with_detail_receipts(
        self, mutate, failed_invariant: str, case: str
    ) -> None:
        """Extra trace rows are tolerated; disagreeing ones are not.

        A scored episode's receipt must appear in the trace byte-for-byte and in
        the recorded order, so any absent, reordered, mutated or augmented row
        breaks the subsequence invariant rather than being quietly accepted.
        """

        del case
        events = _default_events()
        trace = mutate([copy.deepcopy(receipt) for receipt, _ in events])

        audit = _audit(events, trace=trace)

        assert audit["valid"] is False
        assert audit[failed_invariant] is False

    def test_scored_events_out_of_trace_order_fail(self) -> None:
        """Present in the trace is not enough: the order must match too."""

        events = _default_events()
        # Trace order 1,2,3; scored order 2,1,3.
        details = _details_for([events[1], events[0], events[2]])

        audit = _audit(events, details=details)

        assert audit["valid"] is False
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is False
        # The trace itself is intact, so continuity and uniqueness still hold:
        # only the scored-to-trace alignment failed.
        assert audit["trace_event_ids_unique"] is True
        assert audit["server_reset_generations_exact_1_to_n"] is True
        assert audit["server_reset_generations"] == [1, 2, 3]

    def test_missing_trace_fails(self) -> None:
        audit = _audit(trace=[], trace_present=False)

        assert audit["valid"] is False
        assert audit["reset_receipt_trace_present"] is False
        assert audit["trace_event_ids_unique"] is False
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is False
        assert audit["scored_reset_events"] == 0

    def test_missing_ledger_fails(self) -> None:
        audit = _audit(ledger=None, ledger_present=False)

        assert audit["valid"] is False
        assert audit["policy_reset_ledger_present"] is False
        assert audit["ledger_schema_matches"] is False
        assert audit["ledger_acknowledged"] is False

    def test_unacknowledged_ledger_fails(self) -> None:
        receipts = [receipt for receipt, _ in _default_events()]
        audit = _audit(ledger=_ledger_for(receipts[-1], phase="pending"))

        assert audit["valid"] is False
        assert audit["ledger_acknowledged"] is False
        # Nothing else about the ledger changed, so the rest still holds.
        assert audit["ledger_run_matches"] is True
        assert audit["ledger_session_matches"] is True
        assert audit["ledger_last_event_matches"] is True

    @pytest.mark.parametrize(
        ("overrides", "failed_invariant"),
        [
            pytest.param(
                {"schema_version": "robodojo.policy-reset-ledger.v0"},
                "ledger_schema_matches",
                id="schema",
            ),
            pytest.param({"run_id": "other-run"}, "ledger_run_matches", id="run_id"),
            pytest.param(
                {"session_id": "ab" * 16}, "ledger_session_matches", id="session"
            ),
            pytest.param(
                {"last_episode_seq": 2}, "ledger_sequence_matches", id="sequence"
            ),
            pytest.param(
                {"reset_event_id": "d" * 64},
                "ledger_last_event_matches",
                id="event_id",
            ),
            pytest.param(
                {"server_reset_generation": 2},
                "ledger_last_event_matches",
                id="generation",
            ),
            pytest.param(
                {"context_digest": "e" * 64},
                "ledger_last_event_matches",
                id="context_digest",
            ),
        ],
    )
    def test_ledger_must_point_at_the_final_event(
        self, overrides: dict, failed_invariant: str
    ) -> None:
        receipts = [receipt for receipt, _ in _default_events()]
        audit = _audit(ledger=_ledger_for(receipts[-1], **overrides))

        assert audit["valid"] is False
        assert audit[failed_invariant] is False

    def test_missing_detail_receipt_fails(self) -> None:
        events = _default_events()
        details = _details_for(events)
        details[1]["policy_reset_receipt"] = None

        audit = _audit(events, details=details)

        assert audit["valid"] is False
        assert audit["detail_receipts_present_for_all_episodes"] is False
        assert audit["detail_receipts"] == 2

    def test_detail_receipt_without_event_id_fails(self) -> None:
        events = _default_events()
        details = _details_for(events)
        del details[0]["policy_reset_receipt"]["reset_event_id"]

        audit = _audit(events, details=details)

        assert audit["valid"] is False
        assert audit["detail_reset_event_ids_present"] is False

    def test_batch_copies_must_be_identical(self) -> None:
        events = _batch_event()
        details = _details_for(events)
        # Same reset event, but one env recorded a different score of evidence.
        details[2]["policy_reset_receipt"]["applied"] = False

        audit = _audit(events, details=details)

        assert audit["valid"] is False
        assert audit["detail_receipts_identical_per_event"] is False

    def test_batch_layouts_must_cover_the_grouped_episodes(self) -> None:
        events = _batch_event()
        details = _details_for(events)
        details[1]["layout_id"] = 99

        audit = _audit(events, details=details)

        assert audit["valid"] is False
        assert audit["batch_layouts_match_episode_details"] is False

    def test_active_env_ids_must_match_layout_seeds(self) -> None:
        receipt = _reset_receipt(
            sequence=1, generation=1, env_ids=[0, 1], layout_ids=[5, 6]
        )
        receipt["active_env_ids"] = [0, 2]
        # Keep the digest/event id honest so only the cross-check fails.
        context = {
            field: receipt[field]
            for field in (
                "episode_id",
                "episode_seq",
                "reset_session_id",
                "active_env_ids",
                "layout_seeds",
            )
        }
        receipt["context_digest"] = reset_context_digest(context)
        receipt["reset_event_id"] = reset_event_id(
            server_instance_id=receipt["server_instance_id"],
            server_reset_generation=receipt["server_reset_generation"],
            context_digest=receipt["context_digest"],
        )

        audit = _audit([(receipt, [5, 6])])

        assert audit["valid"] is False
        assert audit["active_env_ids_match_layout_seeds"] is False

    def test_episode_id_must_encode_run_session_and_sequence(self) -> None:
        events = _default_events()
        receipt, layouts = events[1]
        receipt = copy.deepcopy(receipt)
        receipt["episode_id"] = "freeform-episode-2"
        context = {
            field: receipt[field]
            for field in (
                "episode_id",
                "episode_seq",
                "reset_session_id",
                "active_env_ids",
                "layout_seeds",
            )
        }
        receipt["context_digest"] = reset_context_digest(context)
        receipt["reset_event_id"] = reset_event_id(
            server_instance_id=receipt["server_instance_id"],
            server_reset_generation=receipt["server_reset_generation"],
            context_digest=receipt["context_digest"],
        )
        events[1] = (receipt, layouts)

        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["episode_ids_match_run_session_sequence"] is False

    def test_repeated_episode_sequence_fails(self) -> None:
        events = [
            (_reset_receipt(sequence=1, generation=1, layout_ids=[0]), [0]),
            (_reset_receipt(sequence=1, generation=2, layout_ids=[1]), [1]),
        ]
        audit = _audit(events)

        assert audit["valid"] is False
        assert audit["episode_sequences_unique_and_increasing"] is False
        assert audit["episode_ids_unique"] is False

    def test_wrong_wire_schema_fails(self) -> None:
        events = _default_events()
        receipt, layouts = events[0]
        receipt = copy.deepcopy(receipt)
        receipt["reset_receipt_schema_version"] = "xpolicylab.episode-reset.v0"
        events[0] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["reset_receipt_schema_matches"] is False
        assert audit["generic_reset_receipts_valid"] is False

    def test_wrong_model_reset_schema_fails(self) -> None:
        events = _default_events()
        receipt, layouts = events[0]
        receipt = copy.deepcopy(receipt)
        receipt["model_reset_schema_version"] = "carve.policy-reset.v0"
        events[0] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["model_reset_schema_matches"] is False

    def test_inconsistent_replay_state_fails(self) -> None:
        events = _default_events()
        receipt, layouts = events[1]
        receipt = copy.deepcopy(receipt)
        receipt["applied_this_request"] = False
        receipt["response_replayed"] = False
        events[1] = (receipt, layouts)

        audit = _audit(events, trace=[copy.deepcopy(r) for r, _ in events])

        assert audit["valid"] is False
        assert audit["receipt_replay_states_valid"] is False
        assert audit["generic_reset_receipts_valid"] is False

    def test_unparseable_inputs_are_reported(self) -> None:
        audit = _audit(input_errors=("reset_receipts.jsonl:2: invalid JSON",))

        assert audit["valid"] is False
        assert audit["audit_inputs_parseable"] is False
        assert "reset_receipts.jsonl:2: invalid JSON" in audit["errors"]

    def test_malformed_trace_row_is_attributed_end_to_end(
        self, tmp_path: Path
    ) -> None:
        receipts = [receipt for receipt, _ in _default_events()]
        trace_text = (
            json.dumps(receipts[0])
            + "\n"
            + "{not json\n"
            + json.dumps(receipts[1])
            + "\n"
            + json.dumps(receipts[2])
            + "\n"
        )
        _write_evidence(tmp_path, trace_text=trace_text)

        _run_finalizer(tmp_path)

        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is False
        assert audit["audit_inputs_parseable"] is False
        assert any(
            "reset_receipts.jsonl:2" in error for error in audit["errors"]
        )


class TestRequireValidResetAuditGate:
    def test_invalid_audit_blocks_the_summary(self, tmp_path: Path) -> None:
        events = _default_events()
        receipt, layouts = events[1]
        receipt = copy.deepcopy(receipt)
        receipt["after"]["state_entries"]["semantic_calls_by_env"] = 4
        events[1] = (receipt, layouts)
        _write_evidence(tmp_path, events)

        with pytest.raises(ValueError, match="policy reset audit is invalid"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=3
            )

        # The gate must fail before any summary is published ...
        assert not (tmp_path / "summary.json").exists()
        # ... and must still leave the diagnostic artifact behind.
        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is False
        assert audit["starvla_reset_states_valid"] is False
        assert audit["errors"]

    def test_missing_evidence_blocks_the_summary(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, trace=None, ledger=None, run_config=None)

        with pytest.raises(ValueError, match="policy reset audit is invalid"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=3
            )

        assert not (tmp_path / "summary.json").exists()
        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is False
        assert audit["reset_receipt_trace_present"] is False
        assert audit["policy_reset_ledger_present"] is False
        assert audit["run_config_present"] is False

    def test_invalid_audit_without_the_gate_still_summarizes(
        self, tmp_path: Path
    ) -> None:
        _write_evidence(tmp_path, trace=None, ledger=None, run_config=None)

        _run_finalizer(tmp_path)

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["policy_reset_audit"]["valid"] is False
        assert summary["aggregate"]["policy_reset_audit_valid"] is False
        assert (tmp_path / "policy_reset_audit.json").is_file()


class TestRunIdBinding:
    """The artifact must be bound to the official run it was configured for.

    RoboDojo writes every condition into the same output tree, so a condition
    that exits before creating its own run directory would otherwise be finalized
    against the previous condition's evidence. That evidence is internally
    consistent — receipts, trace, ledger and continuity all agree — so nothing
    except the run id can catch the mislabelling.
    """

    def test_matching_run_id_is_recorded(self) -> None:
        audit = _audit()

        assert audit["run_id_matches_run_config"] is True
        assert audit["audited_run_id"] == RUN_ID
        assert audit["run_ids"] == [RUN_ID]
        assert audit["run_config_fields"]["robodojo_run_id"] == "robodojo_run_id"

    @pytest.mark.parametrize(
        ("run_config", "audited_run_id"),
        [
            pytest.param(
                _run_config(robodojo_run_id="stack_bowls-C2-seed1"),
                "stack_bowls-C2-seed1",
                id="another_condition",
            ),
            pytest.param(_run_config(robodojo_run_id=""), "", id="empty_run_id"),
            pytest.param(
                _run_config_without("robodojo_run_id"), None, id="missing_run_id"
            ),
        ],
    )
    def test_unbound_or_foreign_run_id_fails(
        self, run_config: dict, audited_run_id: str | None
    ) -> None:
        audit = _audit(run_config=run_config)

        assert audit["valid"] is False
        assert audit["run_id_matches_run_config"] is False
        assert audit["audited_run_id"] == audited_run_id
        # The evidence itself is beyond reproach, which is the whole point: the
        # binding is the only invariant that can reject a foreign run directory.
        assert audit["run_ids"] == [RUN_ID]
        assert audit["episode_ids_match_run_session_sequence"] is True
        assert audit["scored_events_are_ordered_subsequence_of_trace"] is True
        assert audit["server_reset_generations_exact_1_to_n"] is True

    def test_foreign_official_directory_blocks_the_summary(
        self, tmp_path: Path
    ) -> None:
        _write_evidence(
            tmp_path,
            run_config=_run_config(robodojo_run_id="stack_bowls-C2-seed1"),
        )

        with pytest.raises(ValueError, match="policy reset audit is invalid"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=3
            )

        assert not (tmp_path / "summary.json").exists()
        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is False
        assert audit["run_id_matches_run_config"] is False


class TestResetContractBinding:
    """The expectation side of the audit is pinned too, not only the receipts.

    The 31-item inventory lives in ``reset_contract.py``, which is outside the
    ``model_code_sha256`` scope and is also the file the finalizer imports to
    build its expectations. Without this binding, an equal-count substitution in
    that file would move the receipt and the expectation together and pass.
    """

    def test_the_audit_pins_the_contract_file_it_loaded(self) -> None:
        assert len(RESET_CONTRACT_SHA256) == 64
        assert finalizer._RESET_CONTRACT_SHA256 == RESET_CONTRACT_SHA256
        # A distinct file, so the two digests can never be confused for each other.
        assert RESET_CONTRACT_SHA256 != MODEL_CODE_SHA256

    def test_matching_contract_digest_is_recorded(self) -> None:
        audit = _audit()

        assert audit["reset_contract_hash_matches_run_config"] is True
        assert audit["reset_contract_sha256"] == RESET_CONTRACT_SHA256
        assert (
            audit["run_config_fields"]["expected_reset_contract_sha256"]
            == "expected_reset_contract_sha256"
        )

    @pytest.mark.parametrize(
        "run_config",
        [
            pytest.param(
                _run_config_without("expected_reset_contract_sha256"), id="missing"
            ),
            pytest.param(
                _run_config(expected_reset_contract_sha256="deadbeef"),
                id="not_a_sha256",
            ),
            pytest.param(
                _run_config(expected_reset_contract_sha256="a" * 64),
                id="wrong_digest",
            ),
            pytest.param(
                _run_config(expected_reset_contract_sha256=MODEL_CODE_SHA256),
                id="model_py_digest_instead",
            ),
        ],
    )
    def test_unpinned_or_wrong_contract_digest_fails(self, run_config: dict) -> None:
        audit = _audit(run_config=run_config)

        assert audit["valid"] is False
        assert audit["reset_contract_hash_matches_run_config"] is False
        # The digest the audit actually loaded is still reported, so the
        # disagreement is diagnosable from the artifact alone.
        assert audit["reset_contract_sha256"] == RESET_CONTRACT_SHA256
        # Nothing about the receipts changed.
        assert audit["state_inventory_complete"] is True
        assert audit["starvla_reset_states_valid"] is True

    def test_missing_run_config_reports_both_bindings_unmet(self) -> None:
        audit = _audit(run_config=None, run_config_source=None)

        assert audit["valid"] is False
        assert audit["run_id_matches_run_config"] is False
        assert audit["reset_contract_hash_matches_run_config"] is False
        assert audit["audited_run_id"] is None
        assert audit["run_config_fields"]["robodojo_run_id"] is None
        assert audit["run_config_fields"]["expected_reset_contract_sha256"] is None


def _completeness(
    *,
    episode_successes: tuple[bool, ...] = (False, False, False),
    eval_time: Any = _UNSET,
    success_rate: Any = _UNSET,
    expected_episodes: int | None = 3,
) -> dict:
    """Audit completeness the way ``main`` does: result.json against episodes."""

    episodes = [
        {"episode_index": index, "success": success}
        for index, success in enumerate(episode_successes)
    ]
    successes = sum(episode_successes)
    recomputed = successes / len(episodes) if episodes else 0.0
    result = {
        "eval_time": len(episodes) if eval_time is _UNSET else eval_time,
        "success_rate": recomputed if success_rate is _UNSET else success_rate,
    }
    return finalizer._audit_run_completeness(
        result, episodes, expected_episodes=expected_episodes
    )


class TestRunCompletenessAudit:
    """Episode-count completeness is a separate guarantee from reset isolation.

    The reset audit deliberately tolerates an applied reset with no scored
    episode, which is exactly what a condition whose retries were exhausted
    mid-matrix looks like. Nothing in that audit can therefore notice a truncated
    block, so completeness is checked against the pre-registered count and
    against RoboDojo's own reported totals.
    """

    def test_complete_block_passes(self) -> None:
        completeness = _completeness()

        assert completeness["valid"] is True
        assert completeness["errors"] == []
        assert all(completeness["invariants"].values())
        assert (
            completeness["schema_version"]
            == finalizer.RUN_COMPLETENESS_SCHEMA_VERSION
        )
        assert completeness["expected_episodes"] == 3
        assert completeness["summarized_episodes"] == 3
        assert completeness["official_eval_time"] == 3
        assert completeness["successes"] == 0
        assert completeness["recomputed_success_rate"] == 0.0

    def test_complete_block_with_successes_passes(self) -> None:
        completeness = _completeness(
            episode_successes=(True, False, True, False), expected_episodes=4
        )

        assert completeness["valid"] is True
        assert completeness["successes"] == 2
        assert completeness["recomputed_success_rate"] == 0.5
        assert completeness["official_success_rate"] == 0.5
        assert completeness["official_success_rate_matches_details"] is True

    def test_truncated_block_fails_the_expected_count(self) -> None:
        completeness = _completeness(expected_episodes=4)

        assert completeness["valid"] is False
        assert completeness["episode_count_matches_expected"] is False
        assert completeness["summarized_episodes"] == 3
        assert completeness["expected_episodes"] == 4
        assert any("expected 4" in error for error in completeness["errors"])
        # RoboDojo's own counter agrees with the details it wrote, so only the
        # pre-registered count can reveal that the block stopped early.
        assert completeness["official_eval_time_matches_details"] is True
        assert completeness["official_success_rate_matches_details"] is True

    @pytest.mark.parametrize(
        "eval_time",
        [
            pytest.param(2, id="fewer_than_details"),
            pytest.param(4, id="more_than_details"),
            pytest.param(3.0, id="float_not_int"),
            pytest.param(True, id="bool_not_int"),
            pytest.param(None, id="absent"),
            pytest.param("3", id="string"),
        ],
    )
    def test_official_eval_time_must_match_the_detail_count(
        self, eval_time: Any
    ) -> None:
        completeness = _completeness(eval_time=eval_time)

        assert completeness["valid"] is False
        assert completeness["official_eval_time_matches_details"] is False
        assert completeness["official_eval_time"] == eval_time
        # The count itself and the recomputed rate are unaffected.
        assert completeness["episode_count_matches_expected"] is True
        assert completeness["official_success_rate_matches_details"] is True

    @pytest.mark.parametrize(
        ("episode_successes", "success_rate"),
        [
            pytest.param((False, False, False), 0.1, id="zero_run_reported_nonzero"),
            pytest.param((True, False, False), 0.0, id="success_reported_as_zero"),
            pytest.param((True, False, False, False), 0.5, id="wrong_denominator"),
            pytest.param((True, False, False), None, id="absent"),
            pytest.param((True, False, False), "0.3333", id="string"),
            pytest.param((True, False, False), True, id="bool_not_number"),
        ],
    )
    def test_official_success_rate_must_match_the_details(
        self, episode_successes: tuple[bool, ...], success_rate: Any
    ) -> None:
        completeness = _completeness(
            episode_successes=episode_successes,
            success_rate=success_rate,
            expected_episodes=len(episode_successes),
        )

        assert completeness["valid"] is False
        assert completeness["official_success_rate_matches_details"] is False
        assert completeness["official_success_rate"] == success_rate
        assert completeness["episode_count_matches_expected"] is True
        assert completeness["official_eval_time_matches_details"] is True

    def test_matching_success_rate_passes_with_scored_successes(self) -> None:
        completeness = _completeness(
            episode_successes=(True, False, False), expected_episodes=3
        )

        assert completeness["valid"] is True
        assert completeness["official_success_rate_matches_details"] is True
        assert completeness["successes"] == 1
        assert completeness["recomputed_success_rate"] == pytest.approx(1 / 3)

    def test_undeclared_expected_episodes_fails(self) -> None:
        completeness = _completeness(expected_episodes=None)

        assert completeness["valid"] is False
        assert completeness["expected_episodes_declared"] is False
        assert completeness["expected_episodes"] is None
        # Nothing else is wrong: an undeclared block simply cannot be certified.
        assert completeness["episode_count_matches_expected"] is True
        assert completeness["official_eval_time_matches_details"] is True
        assert completeness["official_success_rate_matches_details"] is True


class TestRunCompletenessGate:
    """A truncated block must not reach the confirmatory aggregation.

    The gate runs before ``summary.json`` is written, because that file is what
    the matrix aggregation adopts. The diagnostic artifact is still published so
    the failure is inspectable without re-running the condition.
    """

    def test_dropped_scored_episode_fails_completeness_not_the_reset_audit(
        self, tmp_path: Path
    ) -> None:
        # Four applied resets, four trace rows, but only three scored episodes:
        # indistinguishable from an abandoned physical episode as far as reset
        # isolation goes, which is why the reset audit stays valid here.
        events, scored = _abandoned_run(2)
        _write_evidence(tmp_path, events, details=_details_for(scored))

        with pytest.raises(ValueError, match="run completeness check failed"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=4
            )

        # Nothing adoptable was published ...
        assert not (tmp_path / "summary.json").exists()
        # ... and both diagnostics were.
        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is True
        completeness = json.loads((tmp_path / "run_completeness.json").read_text())
        assert (
            completeness["schema_version"]
            == finalizer.RUN_COMPLETENESS_SCHEMA_VERSION
        )
        assert completeness["valid"] is False
        assert completeness["episode_count_matches_expected"] is False
        assert completeness["expected_episodes"] == 4
        assert completeness["summarized_episodes"] == 3
        assert completeness["errors"]

    def test_official_eval_time_disagreement_blocks_the_summary(
        self, tmp_path: Path
    ) -> None:
        details = _write_evidence(tmp_path)
        result = json.loads((tmp_path / "result.json").read_text())
        result["eval_time"] = len(details) + 1
        (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")

        with pytest.raises(ValueError, match="run completeness check failed"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=3
            )

        assert not (tmp_path / "summary.json").exists()
        completeness = json.loads((tmp_path / "run_completeness.json").read_text())
        assert completeness["valid"] is False
        assert completeness["official_eval_time_matches_details"] is False
        assert completeness["official_eval_time"] == 4
        assert completeness["summarized_episodes"] == 3

    def test_undeclared_expected_episodes_blocks_the_summary(
        self, tmp_path: Path
    ) -> None:
        _write_evidence(tmp_path)

        with pytest.raises(ValueError, match="run completeness check failed"):
            _run_finalizer(tmp_path, require_valid_reset_audit=True)

        assert not (tmp_path / "summary.json").exists()
        # The reset evidence is perfect, so this is the only gate that fires.
        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["valid"] is True
        completeness = json.loads((tmp_path / "run_completeness.json").read_text())
        assert completeness["valid"] is False
        assert completeness["expected_episodes_declared"] is False
        assert completeness["expected_episodes"] is None

    @pytest.mark.parametrize(
        "expected_episodes",
        [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
    )
    def test_non_positive_expected_episodes_is_rejected(
        self, tmp_path: Path, expected_episodes: int
    ) -> None:
        _write_evidence(tmp_path)

        with pytest.raises(ValueError, match="positive integer"):
            _run_finalizer(
                tmp_path,
                require_valid_reset_audit=True,
                expected_episodes=expected_episodes,
            )

        # Rejected before any evidence was read, so nothing was published.
        assert not (tmp_path / "summary.json").exists()
        assert not (tmp_path / "run_completeness.json").exists()
        assert not (tmp_path / "policy_reset_audit.json").exists()

    def test_completeness_failure_without_the_gate_still_summarizes(
        self, tmp_path: Path
    ) -> None:
        """Diagnostic runs stay usable; the flag is what makes it a gate."""

        _write_evidence(tmp_path)

        _run_finalizer(tmp_path)

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["run_completeness"]["valid"] is False
        assert summary["run_completeness"]["expected_episodes_declared"] is False
        assert summary["aggregate"]["run_completeness_valid"] is False
        # The reset audit is a separate verdict and stays true.
        assert summary["aggregate"]["policy_reset_audit_valid"] is True
        standalone = json.loads((tmp_path / "run_completeness.json").read_text())
        assert standalone == summary["run_completeness"]

    def test_complete_block_is_published_with_both_verdicts(
        self, tmp_path: Path
    ) -> None:
        _write_evidence(tmp_path)

        _run_finalizer(tmp_path, require_valid_reset_audit=True, expected_episodes=3)

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["aggregate"]["run_completeness_valid"] is True
        assert summary["aggregate"]["policy_reset_audit_valid"] is True
        completeness = summary["run_completeness"]
        assert completeness["valid"] is True
        assert completeness["expected_episodes"] == 3
        assert completeness["summarized_episodes"] == 3
        assert completeness["official_eval_time"] == 3
        assert len(summary["episodes"]) == 3


class TestCanonicalizedEvidenceOnDisk:
    """Audit the evidence in the shape the run actually writes it.

    ``EvalEnv`` appends receipts with ``json.dumps(..., sort_keys=True)`` and
    RoboDojo's result writer canonicalizes as well, so every nested mapping in
    the persisted receipt is alphabetical rather than in registry order. The
    fixtures above serialize with the default ``sort_keys=False``, which is why a
    real 3-episode C3 run failed this audit on all four receipts while the whole
    test suite was green. These cases pin the on-disk shape.
    """

    @staticmethod
    def _write_canonicalized(tmp_path: Path, events, expected_episodes: int) -> None:
        details = _details_for(events)
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "success_rate": 0.0,
                    "eval_time": len(details),
                    "score": 0.0,
                    "details": {
                        str(index): detail for index, detail in enumerate(details)
                    },
                },
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        (tmp_path / "reset_receipts.jsonl").write_text(
            "".join(
                json.dumps(receipt, sort_keys=True, default=str) + "\n"
                for receipt, _ in events
            ),
            encoding="utf-8",
        )
        (tmp_path / "policy_reset_ledger.json").write_text(
            json.dumps(_ledger_for(events[-1][0]), sort_keys=True), encoding="utf-8"
        )
        (tmp_path / "run_config.json").write_text(
            json.dumps(_run_config(), sort_keys=True), encoding="utf-8"
        )
        del expected_episodes

    def test_sort_keys_persistence_still_passes_both_gates(
        self, tmp_path: Path
    ) -> None:
        events = _default_events()
        self._write_canonicalized(tmp_path, events, 3)

        # Sanity: the fixture really is in the reordered on-disk shape.
        persisted = json.loads(
            (tmp_path / "reset_receipts.jsonl").read_text().splitlines()[0]
        )
        assert tuple(persisted["after"]["state_entries"]) == tuple(
            sorted(STARVLA_RESET_STATE_FIELDS)
        )
        assert persisted["state_inventory"] == list(STARVLA_RESET_STATE_FIELDS)

        _run_finalizer(tmp_path, require_valid_reset_audit=True, expected_episodes=3)

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["policy_reset_audit"]["valid"] is True
        assert summary["policy_reset_audit"]["errors"] == []
        assert summary["run_completeness"]["valid"] is True

    def test_sort_keys_persistence_still_rejects_leftover_state(
        self, tmp_path: Path
    ) -> None:
        """Tolerating key order must not tolerate a container left populated."""

        events = _default_events()
        events[1] = (
            _mutated(
                events[1][0], ("after", "state_entries", "obs_by_env"), 1
            ),
            events[1][1],
        )
        self._write_canonicalized(tmp_path, events, 3)

        with pytest.raises(ValueError, match="policy reset audit is invalid"):
            _run_finalizer(
                tmp_path, require_valid_reset_audit=True, expected_episodes=3
            )

        audit = json.loads((tmp_path / "policy_reset_audit.json").read_text())
        assert audit["starvla_reset_states_valid"] is False
        assert not (tmp_path / "summary.json").exists()
