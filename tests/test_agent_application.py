"""Application contract tests, not model or robot-success experiments."""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agentic_vla.application.api import create_app
from agentic_vla.application.control_plane import (
    ExecutionDisabled,
    LangGraphToolService,
    RequestConflict,
    RequestInvalid,
)
from agentic_vla.configuration import CarveRunConfig
from agentic_vla.external_agent import ExternalToolAgentGateway
from agentic_vla.runtime import HighLevelAgentContext
from agentic_vla.session import CarveAgentSession
from agentic_vla.toolchain import (
    EmbodiedToolBindings,
    PrimitiveExecutionReport,
    RunWorkspace,
    ToolExecutionContext,
    VerificationReport,
)


@pytest.fixture
def setup_service(tmp_path):
    calls = []
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    context = [
        ToolExecutionContext(
            episode_id="test-episode",
            timestep=8,
            at_safe_boundary=True,
            allowed_tools=config.harness.allowed_tools,
            deployment_profile_id=config.vla.deployment_profile_id,
        )
    ]

    def observe(ctx):
        calls.append(("observe", ctx.timestep))
        return {"evidence_kind": "contract_test_fixture", "risk_bucket": "low"}

    def vla_act(instruction, ctx):
        calls.append(("vla_act", instruction))
        return PrimitiveExecutionReport(
            status="interrupted",
            started_timestep=ctx.timestep,
            ended_timestep=ctx.timestep,
            observed_outcome="test fixture: no model or robot executed",
        )

    bindings = EmbodiedToolBindings(
        observe=observe,
        retrieve_memory=lambda query, limit, _ctx: {
            "query": query,
            "records": [],
            "limit": limit,
        },
        vla_act=vla_act,
        run_skill=lambda name, _args, ctx: vla_act(name, ctx),
        verify=lambda expected, _ctx: VerificationReport(
            status="inconclusive",
            observed_outcome=f"unverified fixture: {expected}",
            confidence=0.0,
        ),
        safe_hold=lambda reason, _ctx: {"holding": True, "reason": reason},
    )
    session = CarveAgentSession(
        config,
        bindings=bindings,
        workspace=RunWorkspace(tmp_path / "run", config.to_run_manifest()),
    )
    session.start(
        HighLevelAgentContext(
            task_instruction="put the mug on the plate",
            trigger="task_start",
            episode_id="test-episode",
            timestep=8,
            risk={"score": 0.0, "bucket": "low", "components": {}, "evidence": {}},
        )
    )
    gateway = ExternalToolAgentGateway(session, context_source=lambda: context[0])

    def make_service(allow_execution=False, path=None):
        return LangGraphToolService(
            gateway,
            ledger_path=path or tmp_path / "requests.sqlite3",
            allow_execution=allow_execution,
        )

    yield make_service, gateway, context, calls
    session.close()


def request(request_id="call-1", name="observe", arguments=None):
    return {
        "request_id": request_id,
        "name": name,
        "arguments": arguments or {},
        "expected_episode_id": "test-episode",
        "expected_timestep": 8,
    }


def test_real_langgraph_dispatches_through_canonical_gateway(setup_service):
    make, gateway, _context, calls = setup_service
    service = make()
    direct = gateway.invoke(request("direct"))
    receipt = service.invoke(request())
    assert receipt["status"] == "completed" and not receipt["replayed"]
    assert receipt["response"]["accepted"] == direct["accepted"]
    assert receipt["response"]["result"]["output"] == direct["result"]["output"]
    assert calls == [("observe", 8), ("observe", 8)]
    assert {"admit", "dispatch", "persist_receipt"} <= set(
        service.graph.get_graph().nodes
    )


def test_receipt_survives_reopen_without_reexecuting_or_returning_live_observation(
    setup_service,
):
    make, _gateway, context, calls = setup_service
    first = make().invoke(request())
    context[0] = dataclasses.replace(context[0], timestep=99)
    replay = make().invoke(request())
    assert replay["replayed"] and replay["timestep"] == 8
    assert replay["response"] == first["response"]
    assert calls == [("observe", 8)]
    with pytest.raises(RequestConflict, match="different payload"):
        make().invoke({**request(), "expected_timestep": 99})


def test_physical_request_is_not_repeated(setup_service):
    make, _gateway, _context, calls = setup_service
    service = make(allow_execution=True)
    payload = request(
        name="vla_act", arguments={"instruction": "put the mug on the plate"}
    )
    assert service.invoke(payload)["response"]["accepted"]
    assert service.invoke(payload)["replayed"]
    assert len(calls) == 1


def test_read_only_service_hides_and_blocks_state_changes(setup_service):
    make, _gateway, _context, calls = setup_service
    service = make()
    assert {t["name"] for t in service.catalog()["tools"]} == {
        "observe",
        "verify",
        "retrieve_memory",
    }
    with pytest.raises(ExecutionDisabled):
        service.invoke(request(name="vla_act", arguments={"instruction": "move"}))
    assert not calls


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": ""},
        {"request_id": "../x"},
        {"expected_timestep": True},
        {"expected_timestep": 1.5},
        {"expected_episode_id": True},
        {"expected_episode_id": []},
        {"name": None},
        {"schema_version": True},
        {"arguments": {"actions": [[0] * 7]}},
        {"arguments": {"nested": {"object_pose": [1, 2, 3]}}},
        {"arguments": {"value": float("nan")}},
        {"unknown": 1},
        {"name": "arbitrary_shell"},
    ],
)
def test_invalid_requests_never_reach_tools(setup_service, changes):
    make, _gateway, _context, calls = setup_service
    with pytest.raises(RequestInvalid):
        make().invoke({**request(), **changes})
    assert not calls


def test_stale_input_rejection_is_preserved(setup_service):
    make, _gateway, context, calls = setup_service
    context[0] = dataclasses.replace(context[0], timestep=9)
    receipt = make().invoke(request())
    assert not receipt["response"]["accepted"]
    assert "stale" in receipt["response"]["error"]
    assert receipt["status"] == "completed"  # Transport completion is not task success.
    assert not calls


def test_safe_boundary_gate_is_preserved(setup_service):
    make, _gateway, context, calls = setup_service
    context[0] = dataclasses.replace(context[0], at_safe_boundary=False)
    receipt = make(True).invoke(
        request(name="vla_act", arguments={"instruction": "move"})
    )
    assert not receipt["response"]["accepted"]
    assert "safe" in receipt["response"]["error"]
    assert not calls


def test_crash_before_receipt_blocks_new_requests_after_reopen(
    setup_service, monkeypatch
):
    make, _gateway, _context, calls = setup_service
    service = make()

    def disk_failure(*_args):
        raise OSError("simulated receipt persistence failure")

    monkeypatch.setattr(service.ledger, "complete", disk_failure)
    with pytest.raises(OSError):
        service.invoke(request())
    assert calls == [("observe", 8)]
    reopened = make()
    assert reopened.ledger.get("call-1")["status"] == "pending"
    for payload in [request(), request("new-call")]:
        with pytest.raises(RequestConflict):
            reopened.invoke(payload)
    assert len(calls) == 1


def test_dispatch_exception_marks_indeterminate(setup_service, monkeypatch):
    make, gateway, _context, _calls = setup_service

    def fail(_payload):
        raise RuntimeError("simulated interrupted network reply")

    monkeypatch.setattr(gateway, "invoke", fail)
    service = make()
    with pytest.raises(RequestConflict, match="unknown"):
        service.invoke(request())
    assert service.ledger.get("call-1")["status"] == "indeterminate"
    with pytest.raises(RequestConflict):
        make().invoke(request("another"))


def test_concurrent_services_share_single_flight_ledger(setup_service, monkeypatch):
    make, gateway, _context, calls = setup_service
    entered, release = threading.Event(), threading.Event()
    original = gateway.invoke

    def slow(payload):
        entered.set()
        assert release.wait(3)
        return original(payload)

    monkeypatch.setattr(gateway, "invoke", slow)
    first, second = make(), make()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first.invoke, request())
        try:
            assert entered.wait(3)
            for payload in [request(), request("second")]:
                with pytest.raises(RequestConflict):
                    second.invoke(payload)
        finally:
            release.set()
        assert future.result(timeout=3)["response"]["accepted"]
    assert len(calls) == 1


def test_ledger_rejects_another_run_or_permission_mode(setup_service, monkeypatch):
    make, gateway, _context, _calls = setup_service
    make()
    with pytest.raises(RequestConflict, match="configuration"):
        make(True)
    catalog = copy.deepcopy(gateway.catalog())
    catalog["run_id"] = "different-run"
    monkeypatch.setattr(gateway, "catalog", lambda: catalog)
    with pytest.raises(RequestConflict, match="configuration"):
        make()


def test_http_auth_validation_status_and_idempotency(setup_service):
    make, _gateway, _context, calls = setup_service
    token = "test-only-token-not-a-secret-000000"
    with TestClient(create_app(make(), api_token=token)) as client:
        assert client.get("/health").json()["robot_health_checked"] is False
        assert client.get("/tools").status_code == 401
        assert client.get("/openapi.json").status_code == 401
        client.headers["Authorization"] = f"Bearer {token}"
        assert client.get("/tools").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/requests/missing").status_code == 404
        assert (
            client.post(
                "/requests", json={**request(), "expected_timestep": "8"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/requests",
                json=request(name="vla_act", arguments={"instruction": "move"}),
            ).status_code
            == 403
        )
        first = client.post("/requests", json=request())
        assert first.status_code == 200 and first.json()["response"]["accepted"]
        assert client.get("/requests/call-1").json()["status"] == "completed"
        assert client.post("/requests", json=request()).json()["replayed"]
        assert (
            client.post(
                "/requests", json={**request(), "expected_timestep": 9}
            ).status_code
            == 409
        )
    assert len(calls) == 1


def test_http_requires_explicit_strong_token(setup_service):
    make, _gateway, _context, _calls = setup_service
    with pytest.raises(ValueError, match="token"):
        create_app(make(), api_token="")


def test_core_remains_independent_of_optional_application_packages():
    code = """
import sys
from agentic_vla.session import CarveAgentSession
from agentic_vla.assembly import build_profile_admitted_tool_bindings
assert not any(name.split('.')[0] in {'langgraph', 'fastapi'} for name in sys.modules)
assert 'agentic_vla.application' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("retrieve_memory", {"query": "put the mug", "limit": 2}),
        ("verify", {"expected_outcome": "mug on plate"}),
        ("vla_act", {"instruction": "put the mug on the plate"}),
    ],
)
def test_tool_outputs_and_context_match_direct_path(setup_service, name, arguments):
    make, gateway, context, _calls = setup_service
    context[0] = dataclasses.replace(
        context[0], inference_controls={"inference_steps": 2}
    )
    before = copy.deepcopy(context[0])
    payload = request("via-graph", name, arguments)
    unchanged = copy.deepcopy(payload)
    direct = gateway.invoke(request("direct", name, arguments))
    receipt = make(True).invoke(payload)
    assert receipt["response"]["accepted"] == direct["accepted"]
    actual_output = copy.deepcopy(receipt["response"]["result"]["output"])
    direct_output = copy.deepcopy(direct["result"]["output"])
    if name == "vla_act":
        assert actual_output["primitive_outcome"].pop("call_id")
        assert direct_output["primitive_outcome"].pop("call_id")
    assert actual_output == direct_output
    assert context[0] == before and payload == unchanged


def test_execution_permission_cannot_be_enabled_by_a_string(setup_service):
    make, _gateway, _context, _calls = setup_service
    with pytest.raises(ValueError, match="boolean"):
        make("false")
