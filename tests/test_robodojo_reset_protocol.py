"""Regression tests for the audited episode-reset contract.

Three layers are pinned here, each against the real implementation:

* ``XPolicyLab.policy.starVLA.model.Model.reset`` — the keyword-only reset that
  must empty every registered per-episode container and the high-level budget.
* ``client_server.ws.model_server.PolicyServer._handle_reset`` — exactly-once
  application of a logical episode token, with server-owned evidence.
* ``client_server.ws.model_client.WsModelClient`` — the env-side commit point,
  which may only clear its step counter after the receipt validates.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# The root unit-test environment does not install XPolicyLab's wire-codec
# dependency. These tests call handlers directly and never encode a frame, so a
# minimal import stub keeps protocol-state tests independent of the codec extra.
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

from client_server.ws.model_client import WsModelClient  # noqa: E402
from client_server.ws.model_server import PolicyServer  # noqa: E402
from client_server.ws.protocol.exceptions import ErrorCode  # noqa: E402
from client_server.ws.protocol.messages import MessageType  # noqa: E402
from client_server.ws.protocol.reset import (  # noqa: E402
    RESET_CAPABILITY_NAME,
    RESET_RECEIPT_SCHEMA_VERSION,
    reset_context_digest,
    reset_event_id,
)
from client_server.ws.protocol.schemas import Frame  # noqa: E402
from XPolicyLab.policy.starVLA import deploy as starvla_deploy  # noqa: E402
from XPolicyLab.policy.starVLA.model import Model  # noqa: E402
from XPolicyLab.policy.starVLA.reset_contract import (  # noqa: E402
    STARVLA_RESET_STATE_FIELDS,
    STARVLA_RESET_STATE_VERSION,
)


# ---------------------------------------------------------------------------
# A. Model-side reset
# ---------------------------------------------------------------------------


class _ResettableMonitor:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _BudgetedAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    @property
    def calls_in_episode(self) -> int:
        return self.calls

    def reset(self, _episode_id=None) -> None:
        self.calls = 0
        self.reset_calls += 1


def _populate_episode_state(model: Model, agent: _BudgetedAgent) -> _ResettableMonitor:
    """Put exactly one entry in every registered per-episode container."""

    model.obs_by_env = {0: {"vision": {"head": {"color": "rgb-frame"}}}}
    model.action_chunks_by_env = {0: ["action-chunk"]}
    model.chunk_start_by_env = {0: 16}
    model._active_execute_horizon_by_env = {0: 16}
    model.step_by_env = {0: 320}
    model._last_action_by_env = {0: [0.0]}
    model._last_planner_step_by_env = {0: 256}
    model._recovery_compute_remaining_by_env = {0: 2}
    model._semantic_last_check_by_env = {0: 256}
    model._semantic_last_counted_step_by_env = {0: 256}
    model._semantic_unchanged_since_step_by_env = {0: 96}
    model._semantic_last_periodic_step_by_env = {0: 256}
    model._semantic_last_value_by_env = {0: 1.0}
    model._semantic_stale_checks_by_env = {0: 2}
    model._semantic_calls_by_env = {0: 3}
    model._semantic_recoveries_by_env = {0: 1}
    model._visual_critic_calls_by_env = {0: 3}
    model._visual_critic_disabled_envs = {0}
    model._high_level_agent_disabled_envs = {0}
    model._semantic_guidance_remaining_by_env = {0: 2}
    model._semantic_guidance_instruction_by_env = {0: "retry stack bowls"}
    model._agent_memory_by_env = {0: ["failure"]}
    model._task_instruction_by_env = {0: "stack bowls"}
    model._task_plan_by_env = {0: object()}
    model._task_plan_attempted_envs = {0}
    model._task_plan_abandoned_envs = {0}
    model._task_plan_last_check_by_env = {0: 256}
    model._task_plan_checks_by_env = {0: 2}
    model._task_plan_contradiction_streak_by_env = {0: ("top", 2)}
    model._task_plan_protocol_failures_by_env = {0: 1}
    model._safe_stop_envs = {0}
    model._fault_recovered_envs = {0}
    model._fault_announced_envs = {0}
    monitor = _ResettableMonitor()
    model._risk_monitors = {0: monitor}
    model._reobservation_by_env = {0: _ResettableMonitor()}
    from agentic_vla.toolchain.recovery_verification import RecoveryVerificationLifecycle
    model._recovery_verification_by_env = {0: RecoveryVerificationLifecycle()}
    model._planner_trace_path = None
    model._high_level_agent = agent
    agent.calls = 3
    return monitor


def _bare_starvla_model() -> Model:
    model = Model.__new__(Model)
    model._reset_generation = 0
    model._latest_env_idx_list = [0]
    model.action_dim = 14
    model.input_color_order = "rgb"
    model.unnorm_key = "arx_x5"
    return model


def test_model_reset_context_is_keyword_only() -> None:
    parameters = inspect.signature(Model.reset).parameters
    assert list(parameters) == ["self", "reset_context"]
    assert parameters["reset_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["reset_context"].default is None


def test_model_reset_rejects_positional_reset_context() -> None:
    model = _bare_starvla_model()
    _populate_episode_state(model, _BudgetedAgent())

    with pytest.raises(TypeError):
        model.reset({"episode_id": "run:session:batch-0000001"})

    # A rejected call must not have half-applied the boundary.
    assert model.obs_by_env == {0: {"vision": {"head": {"color": "rgb-frame"}}}}
    assert model._reset_generation == 0


def test_model_reset_rejects_non_mapping_reset_context() -> None:
    model = _bare_starvla_model()
    _populate_episode_state(model, _BudgetedAgent())

    with pytest.raises(TypeError, match="reset_context must be a dict or None"):
        model.reset(reset_context=[("episode_id", "x")])

    assert model.step_by_env == {0: 320}
    assert model._reset_generation == 0


def test_model_reset_clears_every_episode_state_for_four_episodes() -> None:
    model = _bare_starvla_model()
    agent = _BudgetedAgent()

    for episode_seq in range(1, 5):
        monitor = _populate_episode_state(model, agent)
        receipt = model.reset(
            reset_context={
                "episode_id": f"run:session:batch-{episode_seq:07d}",
                "episode_seq": episode_seq,
                "reset_session_id": "session",
                "active_env_ids": [0],
                "layout_seeds": [{"env_idx": 0, "layout_id": episode_seq - 1}],
            }
        )

        assert receipt["model_reset_schema_version"] == "carve.policy-reset.v1"
        assert receipt["state_inventory_version"] == STARVLA_RESET_STATE_VERSION
        assert receipt["state_inventory"] == list(STARVLA_RESET_STATE_FIELDS)
        assert receipt["reset_generation"] == episode_seq
        assert receipt["episode_seq"] == episode_seq
        assert receipt["active_env_ids"] == [0]

        # Every registered container held exactly one entry before the reset.
        before = receipt["before"]["state_entries"]
        assert tuple(before) == STARVLA_RESET_STATE_FIELDS
        assert len(before) == 36
        assert set(before.values()) == {1}
        assert receipt["before"]["high_level_calls"] == 3

        # ... and all of them are empty afterwards, together with the agent budget.
        after = receipt["after"]["state_entries"]
        assert tuple(after) == STARVLA_RESET_STATE_FIELDS
        assert len(after) == 36
        assert set(after.values()) == {0}
        assert receipt["after"]["high_level_calls"] == 0

        # Reset must reach the live attributes, not a copy of them.
        assert model.obs_by_env == {}
        assert model.step_by_env == {}
        assert model._risk_monitors == {}
        assert model._visual_critic_disabled_envs == set()
        assert model._task_plan_attempted_envs == set()
        assert monitor.reset_calls == 1
        assert agent.calls_in_episode == 0
        assert agent.reset_calls == episode_seq

        # Reset changes lifecycle state, not the RGB/action contract.
        assert model.action_dim == 14
        assert model.input_color_order == "rgb"
        assert model.unnorm_key == "arx_x5"
        assert model._latest_env_idx_list == [0]

    assert model._reset_generation == 4


def test_model_reset_without_context_still_returns_a_receipt() -> None:
    model = _bare_starvla_model()
    _populate_episode_state(model, _BudgetedAgent())

    receipt = model.reset()

    assert receipt["episode_id"] is None
    assert receipt["episode_seq"] is None
    assert receipt["active_env_ids"] == []
    assert set(receipt["after"]["state_entries"].values()) == {0}


def test_model_reset_refuses_a_missing_high_level_agent_attribute() -> None:
    """A renamed agent attribute must raise, not report a phantom zero.

    ``high_level_calls`` is read through ``getattr(..., 0)``, so a rename would
    otherwise produce ``before=0 / after=0``, satisfy the receipt contract, and
    silently skip the agent's own per-episode reset. The attribute is therefore
    required to exist for the same reason the 31 containers are.
    """

    model = _bare_starvla_model()
    agent = _BudgetedAgent()
    monitor = _populate_episode_state(model, agent)
    del model._high_level_agent

    with pytest.raises(RuntimeError, match="_high_level_agent"):
        model.reset(reset_context=_reset_context())

    # The boundary must not have been half-applied: the refusal happens before
    # anything is cleared, so the episode is still the one it was.
    assert model.obs_by_env == {0: {"vision": {"head": {"color": "rgb-frame"}}}}
    assert model.step_by_env == {0: 320}
    assert model._risk_monitors == {0: monitor}
    assert model._safe_stop_envs == {0}
    assert monitor.reset_calls == 0
    assert agent.reset_calls == 0
    assert agent.calls_in_episode == 3
    assert model._reset_generation == 0


def test_model_reset_accepts_a_none_high_level_agent() -> None:
    """B0 and C1 build no agent at all, which is a legitimate configuration."""

    model = _bare_starvla_model()
    agent = _BudgetedAgent()
    monitor = _populate_episode_state(model, agent)
    model._high_level_agent = None

    receipt = model.reset(reset_context=_reset_context())

    assert receipt["before"]["high_level_calls"] == 0
    assert receipt["after"]["high_level_calls"] == 0
    assert set(receipt["after"]["state_entries"].values()) == {0}
    assert tuple(receipt["after"]["state_entries"]) == STARVLA_RESET_STATE_FIELDS
    assert model._reset_generation == 1
    assert monitor.reset_calls == 1
    # The detached agent is not reset by a run that no longer holds it.
    assert agent.reset_calls == 0


# ---------------------------------------------------------------------------
# B. Server-side exactly-once application
# ---------------------------------------------------------------------------


_EPISODE_ID = "run-1:session-1:batch-0000001"


def _reset_context(
    *,
    episode_id: str = _EPISODE_ID,
    episode_seq: int = 1,
    active_env_ids: list[int] | None = None,
    layout_seeds: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    if active_env_ids is None:
        active_env_ids = [0]
    if layout_seeds is None:
        layout_seeds = [{"env_idx": env_idx, "layout_id": env_idx} for env_idx in active_env_ids]
    return {
        "episode_id": episode_id,
        "episode_seq": episode_seq,
        "reset_session_id": "session-1",
        "active_env_ids": list(active_env_ids),
        "layout_seeds": [dict(row) for row in layout_seeds],
    }


def _reset_frame(
    request_id: str,
    reset_context: dict[str, Any] | None,
    *,
    evaluation_id: str = "eval-1",
    trial_id: str | None = "stack_bowls-run",
    action_case_id: str | None = "stack_bowls_case",
    repeat_index: int | None = None,
) -> Frame:
    payload: dict[str, Any] = {"trial_id": trial_id}
    if reset_context is not None:
        payload["reset_context"] = dict(reset_context)
    return Frame(
        message_type=MessageType.RESET,
        request_id=request_id,
        evaluation_id=evaluation_id,
        action_case_id=action_case_id,
        trial_id=trial_id,
        repeat_index=repeat_index,
        payload=payload,
    )


class _ContextAwareModel:
    """Minimal stand-in for the StarVLA adapter's keyword-only reset."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def reset(self, *, reset_context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.contexts.append(dict(reset_context or {}))
        return {
            "model_reset_schema_version": "carve.policy-reset.v1",
            "state_inventory_version": STARVLA_RESET_STATE_VERSION,
            "state_inventory": list(STARVLA_RESET_STATE_FIELDS),
            "reset_generation": len(self.contexts),
            "after": {
                "state_entries": {name: 0 for name in STARVLA_RESET_STATE_FIELDS},
                "high_level_calls": 0,
            },
        }


def _run_reset_frames(model: Any, frames: list[Frame]) -> tuple[PolicyServer, list[Frame]]:
    async def scenario() -> tuple[PolicyServer, list[Frame]]:
        server = PolicyServer(model)
        responses = [await server.process_frame(frame) for frame in frames]
        return server, responses

    return asyncio.run(scenario())


def test_server_applies_logical_episode_reset_exactly_once() -> None:
    model = _ContextAwareModel()
    context = _reset_context()
    second_context = _reset_context(
        episode_id="run-1:session-1:batch-0000002", episode_seq=2
    )
    server, (first, replay, second) = _run_reset_frames(
        model,
        [
            _reset_frame("request-a", context),
            _reset_frame("request-b", context),
            _reset_frame("request-c", second_context),
        ],
    )

    # Two logical episodes, so the model was reset exactly twice.
    assert len(model.contexts) == 2
    assert model.contexts[0] == context
    assert model.contexts[1] == second_context

    first_receipt = first.payload["result"]
    replay_receipt = replay.payload["result"]
    second_receipt = second.payload["result"]

    assert first.message_type == MessageType.RESET_RESULT
    assert replay.message_type == MessageType.RESET_RESULT

    assert (
        first_receipt["applied_once"],
        first_receipt["applied_this_request"],
        first_receipt["response_replayed"],
    ) == (True, True, False)
    assert (
        replay_receipt["applied_once"],
        replay_receipt["applied_this_request"],
        replay_receipt["response_replayed"],
    ) == (True, False, True)
    assert (
        second_receipt["applied_once"],
        second_receipt["applied_this_request"],
        second_receipt["response_replayed"],
    ) == (True, True, False)

    # Legacy aliases keep their per-response meaning.
    assert (first_receipt["applied"], first_receipt["replayed"]) == (True, False)
    assert (replay_receipt["applied"], replay_receipt["replayed"]) == (False, True)

    # The replay is the same server event, not a new one.
    assert replay_receipt["reset_event_id"] == first_receipt["reset_event_id"]
    assert (
        replay_receipt["server_reset_generation"]
        == first_receipt["server_reset_generation"]
        == 1
    )
    assert replay_receipt["context_digest"] == first_receipt["context_digest"]
    assert second_receipt["server_reset_generation"] == 2
    assert second_receipt["reset_event_id"] != first_receipt["reset_event_id"]

    # Server-owned identity, and the digest/event id actually bind the context.
    assert first_receipt["server_instance_id"] == server._instance_id
    assert first_receipt["context_digest"] == reset_context_digest(context)
    assert first_receipt["reset_event_id"] == reset_event_id(
        server_instance_id=server._instance_id,
        server_reset_generation=1,
        context_digest=reset_context_digest(context),
    )
    assert first_receipt["reset_receipt_schema_version"] == RESET_RECEIPT_SCHEMA_VERSION
    for field, expected in context.items():
        assert first_receipt[field] == expected
    assert first_receipt["evaluation_id"] == "eval-1"
    assert first_receipt["trial_id"] == "stack_bowls-run"
    assert first_receipt["action_case_id"] == "stack_bowls_case"


def test_server_receipt_identifies_model_code_without_absolute_paths() -> None:
    model = _ContextAwareModel()
    _server, (response,) = _run_reset_frames(
        model, [_reset_frame("request-a", _reset_context())]
    )
    receipt = response.payload["result"]

    assert receipt["model_module_id"] == _ContextAwareModel.__module__
    expected_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert receipt["model_code_sha256"] == expected_hash

    # A filesystem path would leak the operator's layout into shared evidence.
    assert "model_module_file" not in receipt

    def _absolute_paths(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.startswith("/") else []
        if isinstance(value, dict):
            return [hit for item in value.values() for hit in _absolute_paths(item)]
        if isinstance(value, (list, tuple)):
            return [hit for item in value for hit in _absolute_paths(item)]
        return []

    assert _absolute_paths(receipt) == []


def test_server_rejects_reused_episode_token_with_changed_layout_seeds() -> None:
    model = _ContextAwareModel()
    context = _reset_context(active_env_ids=[0, 1])
    tampered_layout = _reset_context(active_env_ids=[0, 1])
    tampered_layout["layout_seeds"] = [
        {"env_idx": 0, "layout_id": 0},
        {"env_idx": 1, "layout_id": 99},
    ]
    tampered_envs = _reset_context(active_env_ids=[0, 2])
    tampered_envs["layout_seeds"] = [
        {"env_idx": 0, "layout_id": 0},
        {"env_idx": 2, "layout_id": 1},
    ]

    _server, (first, layout_error, env_error) = _run_reset_frames(
        model,
        [
            _reset_frame("request-a", context),
            _reset_frame("request-b", tampered_layout),
            _reset_frame("request-c", tampered_envs),
        ],
    )

    assert first.message_type == MessageType.RESET_RESULT
    for response in (layout_error, env_error):
        assert response.message_type == MessageType.ERROR
        assert response.payload["code"] == ErrorCode.INVALID_FRAME.value
        assert "reused with different context" in response.payload["message"]

    # A rejected replay must not touch the model or the generation counter.
    assert len(model.contexts) == 1
    assert model.contexts[0] == context
    assert first.payload["result"]["server_reset_generation"] == 1


@pytest.mark.parametrize(
    "frame_kwargs",
    [
        pytest.param({"evaluation_id": "eval-2"}, id="evaluation_id"),
        pytest.param({"trial_id": "other-run"}, id="trial_id"),
        pytest.param({"action_case_id": "other_case"}, id="action_case_id"),
        pytest.param({"repeat_index": 1}, id="repeat_index"),
    ],
)
def test_server_isolates_episode_tokens_across_run_scopes(
    frame_kwargs: dict[str, Any],
) -> None:
    model = _ContextAwareModel()
    context = _reset_context()

    _server, (first, scoped, replay) = _run_reset_frames(
        model,
        [
            _reset_frame("request-a", context),
            _reset_frame("request-b", context, **frame_kwargs),
            _reset_frame("request-c", context, **frame_kwargs),
        ],
    )

    # The same episode token in a different run scope is a different episode.
    assert len(model.contexts) == 2
    first_receipt = first.payload["result"]
    scoped_receipt = scoped.payload["result"]
    replay_receipt = replay.payload["result"]

    assert first_receipt["applied_this_request"] is True
    assert scoped_receipt["applied_this_request"] is True
    assert scoped_receipt["server_reset_generation"] == 2
    assert scoped_receipt["reset_event_id"] != first_receipt["reset_event_id"]

    # The scoped token is still deduplicated within its own scope.
    assert replay_receipt["response_replayed"] is True
    assert replay_receipt["reset_event_id"] == scoped_receipt["reset_event_id"]


def test_server_keeps_legacy_seed_options_model_out_of_the_reset_context() -> None:
    class LegacySeedOptionsModel:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

        def reset(self, seed=None, options=None):
            self.calls.append((seed, options))
            return None

    model = LegacySeedOptionsModel()
    context = _reset_context()
    _server, (response,) = _run_reset_frames(model, [_reset_frame("legacy", context)])

    # A legacy signature has no reset_context parameter, so the server must not
    # smuggle the context into `seed`/`options`.
    assert model.calls == [(None, None)]
    receipt = response.payload["result"]
    assert response.message_type == MessageType.RESET_RESULT
    assert receipt["episode_id"] == _EPISODE_ID
    assert receipt["applied_this_request"] is True
    assert receipt["context_digest"] == reset_context_digest(context)


def test_server_keeps_legacy_no_argument_model_compatible() -> None:
    class LegacyModel:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self):
            self.calls += 1

    model = LegacyModel()
    _server, (response, replay) = _run_reset_frames(
        model,
        [
            _reset_frame("legacy", _reset_context()),
            _reset_frame("legacy-replay", _reset_context()),
        ],
    )

    assert model.calls == 1
    receipt = response.payload["result"]
    assert receipt["episode_id"] == _EPISODE_ID
    assert receipt["applied"] is True
    assert receipt["applied_this_request"] is True
    assert replay.payload["result"]["response_replayed"] is True


def test_server_advertises_the_audited_reset_capability() -> None:
    async def scenario():
        server = PolicyServer(_ContextAwareModel())
        return await server.process_frame(
            Frame(
                message_type=MessageType.HELLO,
                request_id="hello-1",
                evaluation_id="eval-1",
            )
        )

    response = asyncio.run(scenario())

    assert response.message_type == MessageType.HELLO_ACK
    assert response.payload["capabilities"] == {
        RESET_CAPABILITY_NAME: RESET_RECEIPT_SCHEMA_VERSION
    }


# ---------------------------------------------------------------------------
# C. Client-side commit point
# ---------------------------------------------------------------------------


def _server_issued_receipt(context: dict[str, Any]) -> dict[str, Any]:
    """Build a wire receipt the way the real server builds one."""

    _server, (response,) = _run_reset_frames(
        _ContextAwareModel(), [_reset_frame("request-a", context)]
    )
    return dict(response.payload["result"])


class _FakeProtocolClient:
    def __init__(
        self,
        *,
        receipt: Any = None,
        fail: bool = False,
        capabilities: dict[str, Any] | None = None,
    ) -> None:
        self.fail = fail
        self.receipt = receipt
        self.reset_context: dict[str, Any] | None = None
        self.reset_calls = 0
        self.closed = False
        self.connected = False
        self.server_capabilities = (
            {RESET_CAPABILITY_NAME: RESET_RECEIPT_SCHEMA_VERSION}
            if capabilities is None
            else dict(capabilities)
        )

    async def connect(self, handshake: bool = False) -> None:
        self.connected = bool(handshake) or True

    async def reset(self, **kwargs: Any):
        self.reset_calls += 1
        self.reset_context = kwargs["reset_context"]
        if self.fail:
            raise RuntimeError("server reset failed")
        return SimpleNamespace(payload={"result": self.receipt})

    async def close(self) -> None:
        self.closed = True


def _synchronous_ws_client(
    protocol_client: _FakeProtocolClient, *, require_audited_reset: bool = True
) -> WsModelClient:
    """A WsModelClient wired to a fake transport, without the loop thread."""

    client = WsModelClient.__new__(WsModelClient)
    client._client = protocol_client
    client._step = 7
    client.last_reset_receipt = None
    client.trial_id = "stack_bowls-run"
    client.action_case_id = "stack_bowls_case"
    client.repeat_index = None
    client.require_audited_reset = require_audited_reset
    client._run = lambda coroutine: asyncio.run(coroutine)
    return client


def test_client_clears_step_only_after_reset_ack() -> None:
    context = _reset_context()
    receipt = _server_issued_receipt(context)

    failing = _synchronous_ws_client(_FakeProtocolClient(fail=True))
    with pytest.raises(RuntimeError, match="server reset failed"):
        failing.call(func_name="reset", obs=context)
    assert failing._step == 7
    assert failing.last_reset_receipt is None

    protocol_client = _FakeProtocolClient(receipt=receipt)
    successful = _synchronous_ws_client(protocol_client)
    validated = successful.call(func_name="reset", obs=context)

    assert successful._step == 0
    assert successful.last_reset_receipt == validated
    assert validated == receipt
    assert protocol_client.reset_context == context


def _malformed_receipts() -> list[tuple[str, Any]]:
    context = _reset_context()
    valid = _server_issued_receipt(context)

    cases: list[tuple[str, Any]] = [
        ("not_a_mapping", None),
        ("list_instead_of_mapping", [valid]),
    ]
    for field in (
        "reset_receipt_schema_version",
        "context_digest",
        "episode_id",
        "episode_seq",
        "reset_session_id",
        "active_env_ids",
        "layout_seeds",
        "applied_once",
        "applied_this_request",
        "response_replayed",
        "server_instance_id",
        "server_reset_generation",
        "reset_event_id",
        "model_module_id",
        "model_code_sha256",
    ):
        missing = dict(valid)
        missing.pop(field)
        cases.append((f"missing_{field}", missing))

    tampered_digest = dict(valid)
    tampered_digest["context_digest"] = "0" * 64
    cases.append(("wrong_context_digest", tampered_digest))

    tampered_event = dict(valid)
    tampered_event["reset_event_id"] = "1" * 64
    cases.append(("wrong_reset_event_id", tampered_event))

    short_event = dict(valid)
    short_event["reset_event_id"] = "abc"
    cases.append(("short_reset_event_id", short_event))

    not_applied = dict(valid)
    not_applied["applied_once"] = False
    cases.append(("applied_once_false", not_applied))

    inconsistent = dict(valid)
    inconsistent["response_replayed"] = True
    cases.append(("applied_and_replayed", inconsistent))

    boolish = dict(valid)
    boolish["applied_once"] = 1
    cases.append(("applied_once_is_int", boolish))

    zero_generation = dict(valid)
    zero_generation["server_reset_generation"] = 0
    cases.append(("zero_generation", zero_generation))

    bool_generation = dict(valid)
    bool_generation["server_reset_generation"] = True
    cases.append(("bool_generation", bool_generation))

    wrong_schema = dict(valid)
    wrong_schema["reset_receipt_schema_version"] = "xpolicylab.episode-reset.v0"
    cases.append(("unsupported_schema", wrong_schema))

    echo_mismatch = dict(valid)
    echo_mismatch["active_env_ids"] = [0, 1]
    cases.append(("active_env_ids_echo_mismatch", echo_mismatch))

    empty_hash = dict(valid)
    empty_hash["model_code_sha256"] = ""
    cases.append(("empty_model_code_sha256", empty_hash))

    non_hex_hash = dict(valid)
    non_hex_hash["model_code_sha256"] = "z" * 64
    cases.append(("non_hex_model_code_sha256", non_hex_hash))

    return cases


@pytest.mark.parametrize(
    ("case_id", "receipt"),
    [pytest.param(case_id, receipt, id=case_id) for case_id, receipt in _malformed_receipts()],
)
def test_client_rejects_malformed_reset_ack_without_committing(
    case_id: str, receipt: Any
) -> None:
    del case_id
    context = _reset_context()
    protocol_client = _FakeProtocolClient(receipt=receipt)
    client = _synchronous_ws_client(protocol_client)

    with pytest.raises((TypeError, ValueError)):
        client.call(func_name="reset", obs=context)

    assert protocol_client.reset_calls == 1
    assert client._step == 7
    assert client.last_reset_receipt is None


def test_client_validates_versioned_receipt_even_without_strict_mode() -> None:
    context = _reset_context()
    receipt = _server_issued_receipt(context)
    receipt.pop("model_module_id")
    protocol_client = _FakeProtocolClient(receipt=receipt)
    client = _synchronous_ws_client(protocol_client, require_audited_reset=False)

    with pytest.raises(ValueError, match="model_module_id"):
        client.call(func_name="reset", obs=context)

    assert client._step == 7
    assert client.last_reset_receipt is None


def _construct_ws_client(protocol_client: _FakeProtocolClient, *, strict: bool) -> WsModelClient:
    return WsModelClient(
        url="ws://127.0.0.1:19000",
        evaluation_id="eval-1",
        trial_id="stack_bowls-run",
        action_case_id="stack_bowls_case",
        require_audited_reset=strict,
        client=protocol_client,
    )


def test_strict_client_fails_construction_without_advertised_capability() -> None:
    for capabilities in (
        {},
        {RESET_CAPABILITY_NAME: "xpolicylab.episode-reset.v0"},
        {"something_else": RESET_RECEIPT_SCHEMA_VERSION},
    ):
        protocol_client = _FakeProtocolClient(capabilities=capabilities)
        with pytest.raises(
            RuntimeError, match="does not advertise the audited reset contract"
        ):
            _construct_ws_client(protocol_client, strict=True)
        assert protocol_client.connected is True


def test_strict_client_accepts_advertised_capability() -> None:
    protocol_client = _FakeProtocolClient()
    client = _construct_ws_client(protocol_client, strict=True)
    try:
        assert client.supports_audited_reset is True
        assert client.require_audited_reset is True
    finally:
        client.close()
    assert protocol_client.closed is True


def test_client_rejects_non_bool_strict_reset_flag() -> None:
    with pytest.raises(TypeError, match="require_audited_reset must be a bool"):
        WsModelClient(
            url="ws://127.0.0.1:19000",
            evaluation_id="eval-1",
            trial_id="stack_bowls-run",
            require_audited_reset=1,  # type: ignore[arg-type]
            client=_FakeProtocolClient(),
        )


def test_starvla_rollout_skips_acknowledged_reset_and_keeps_legacy_fallback() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, *, func_name, **_kwargs):
            self.calls.append(func_name)
            return {"applied": True}

    acknowledged_env = SimpleNamespace(
        policy_reset_acknowledged=lambda: True,
        _policy_reset_receipt={"applied": True},
    )
    acknowledged_client = Client()
    assert (
        starvla_deploy._ensure_episode_reset(
            acknowledged_env, acknowledged_client
        )
        == {"applied": True}
    )
    assert acknowledged_client.calls == []

    legacy_env = SimpleNamespace()
    legacy_client = Client()
    starvla_deploy._ensure_episode_reset(legacy_env, legacy_client)
    assert legacy_client.calls == ["reset"]


# ---------------------------------------------------------------------------
# E. EvalEnv's strict num_envs gate
#
# `require_audited_reset` with `num_envs != 1` must be refused: the strongest
# per-episode isolation evidence in the receipt is `after.high_level_calls`, a
# single scalar on one shared GuardedHighLevelAgent whose budget is keyed by env
# index, so parallel envs reset each other's budget and the receipt would pass
# every structural check while the claim it certifies is false.
#
# WHY THESE ARE SOURCE-LEVEL, NOT RUNTIME, ASSERTIONS: `EvalEnv` is declared
# inside `create_eval_env` and subclasses a task class resolved from Isaac Lab,
# which is not importable on a CPU-only runner, so the class cannot be
# instantiated (and instantiating it would stand up a simulation). The tests
# below therefore assert over the parsed source: the guard exists, it raises
# PolicyResetProtocolError, and it runs *before* the ledger file and the
# websocket client are constructed. `test_strict_num_envs_condition_...` does
# execute the guard's own condition, by compiling the predicate straight out of
# the production AST rather than restating it, but the surrounding `__init__` is
# never run. This is deliberately labelled as static coverage rather than
# dressed up as a runtime test of EvalEnv.
# ---------------------------------------------------------------------------

import ast  # noqa: E402

EVAL_ENV_PATH = ROBODOJO_ROOT / "src" / "eval_client" / "eval_env.py"


def _eval_env_init() -> ast.FunctionDef:
    tree = ast.parse(EVAL_ENV_PATH.read_text(encoding="utf-8"))
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_eval_env"
    )
    eval_env = next(
        node
        for node in factory.body
        if isinstance(node, ast.ClassDef) and node.name == "EvalEnv"
    )
    return next(
        node
        for node in eval_env.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def _is_self_attribute(node: ast.expr, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _num_envs_guard(init: ast.FunctionDef) -> ast.If:
    """The `require_audited_reset and num_envs != 1` refusal in `__init__`.

    Takes the parsed `__init__` so callers that also need statement positions
    compare nodes from one tree instead of two independent parses.
    """

    guards = []
    for node in ast.walk(init):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)):
            continue
        if not any(
            _is_self_attribute(value, "require_audited_reset") for value in test.values
        ):
            continue
        compares_num_envs = any(
            isinstance(value, ast.Compare)
            and _is_self_attribute(value.left, "num_envs")
            and len(value.ops) == 1
            and isinstance(value.ops[0], ast.NotEq)
            and isinstance(value.comparators[0], ast.Constant)
            and value.comparators[0].value == 1
            for value in test.values
        )
        if compares_num_envs:
            guards.append(node)
    assert len(guards) == 1, (
        "expected exactly one `require_audited_reset and num_envs != 1` guard in "
        f"EvalEnv.__init__, found {len(guards)}"
    )
    return guards[0]


def _statement_index(function: ast.FunctionDef, node: ast.AST) -> int:
    """Index in `function.body` of the statement that contains `node`."""

    for index, statement in enumerate(function.body):
        if any(child is node for child in ast.walk(statement)):
            return index
    raise AssertionError(f"{node!r} is not inside {function.name}")


def _first_construction(function: ast.FunctionDef, callee: str) -> ast.Call:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
    ]
    assert calls, f"EvalEnv.__init__ no longer constructs {callee}"
    return min(calls, key=lambda call: call.lineno)


def test_strict_reset_guard_is_declared_in_eval_env_source() -> None:
    """The guard must exist and refuse with the protocol error."""

    guard = _num_envs_guard(_eval_env_init())
    raises = [node for node in ast.walk(guard) if isinstance(node, ast.Raise)]

    assert len(raises) == 1
    exception = raises[0].exc
    assert isinstance(exception, ast.Call)
    assert isinstance(exception.func, ast.Name)
    assert exception.func.id == "PolicyResetProtocolError"
    # The refusal aborts a queued run, so it has to say what it wants. The
    # message mixes literal and f-string fragments, so read the literal parts
    # rather than evaluating a node that references `self`.
    message = "".join(
        part.value
        for part in ast.walk(exception.args[0])
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )
    assert "num_envs=1" in message
    assert "not per-env isolated" in message


def test_strict_num_envs_guard_runs_before_any_audited_reset_state_is_built() -> None:
    """Refuse before touching the ledger file or opening the policy socket.

    Ordering is the whole point: a ledger written under `num_envs > 1` would be
    evidence of an audited run, and a HELLO handshake would advertise the audited
    capability, for a configuration the mechanism cannot honour.
    """

    init = _eval_env_init()
    guard = _num_envs_guard(init)
    ledger = _first_construction(init, "PolicyResetLedger")
    ws_client = _first_construction(init, "WsModelClient")

    # Straight-line statement order in `__init__`, not just line numbers: the
    # guard must be an earlier statement than the ones that build the evidence.
    guard_index = _statement_index(init, guard)
    assert guard_index < _statement_index(init, ledger)
    assert guard_index < _statement_index(init, ws_client)
    assert guard.lineno < ledger.lineno < ws_client.lineno


@pytest.mark.parametrize(
    ("require_audited_reset", "num_envs", "refuses"),
    [
        pytest.param(True, 1, False, id="strict_single_env_allowed"),
        pytest.param(True, 2, True, id="strict_two_envs_refused"),
        pytest.param(True, 8, True, id="strict_eight_envs_refused"),
        pytest.param(False, 1, False, id="legacy_single_env_allowed"),
        pytest.param(False, 4, False, id="legacy_parallel_untouched"),
    ],
)
def test_strict_num_envs_condition_matches_the_intended_truth_table(
    require_audited_reset: bool, num_envs: int, refuses: bool
) -> None:
    """Execute the production condition itself, compiled from its own AST.

    This runs the real predicate text (so a rewrite that inverts it fails here)
    without constructing EvalEnv, which needs Isaac Lab and a live simulation.
    """

    predicate = compile(
        ast.Expression(body=_num_envs_guard(_eval_env_init()).test),
        str(EVAL_ENV_PATH),
        "eval",
    )
    scope = SimpleNamespace(
        require_audited_reset=require_audited_reset, num_envs=num_envs
    )

    # The only thing evaluated here is the guard's own condition expression,
    # lifted from eval_env.py; `self` is the fake scope above.
    assert bool(eval(predicate, {"self": scope})) is refuses
