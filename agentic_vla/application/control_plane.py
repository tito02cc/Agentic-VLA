"""LangGraph tool turns with durable, fail-closed execution receipts.

This is an external-agent transport, not a planner or a robot task queue.
The existing session and tool gateway retain all execution authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_vla.external_agent import ExternalToolAgentGateway, ExternalToolRequest


class RequestConflict(RuntimeError):
    """A request cannot be safely executed or automatically retried."""


class RequestInvalid(ValueError):
    """The transport request does not match the external-tool contract."""


class ExecutionDisabled(PermissionError):
    """This service has not opted into state-changing tools."""


class ToolTurn(TypedDict, total=False):
    request: dict[str, Any]
    replayed: bool
    receipt: dict[str, Any]
    response: dict[str, Any]


def normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RequestInvalid("request must be a JSON object")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_.:-]{1,128}", request_id
    ):
        raise RequestInvalid(
            "request_id must be an explicit 1-128 character identifier"
        )
    episode = payload.get("expected_episode_id")
    if type(episode) not in (int, str) or (
        isinstance(episode, str) and not episode.strip()
    ):
        raise RequestInvalid("expected_episode_id must be a nonempty string or integer")
    timestep = payload.get("expected_timestep")
    if type(timestep) is not int or timestep < 0:
        raise RequestInvalid("expected_timestep must be a non-negative integer")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise RequestInvalid("name must be a nonempty string")
    if type(payload.get("schema_version", 1)) is not int:
        raise RequestInvalid("schema_version must be an integer")
    try:
        request = ExternalToolRequest.from_dict(payload)
        return json.loads(json.dumps(dataclasses.asdict(request), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestInvalid(str(exc)) from exc


class RequestLedger:
    """One durable run identity; an unresolved dispatch blocks all new turns.

    A crash between dispatch and receipt persistence cannot be distinguished
    from a completed physical action. Never clear or replay that pending row
    automatically, even when another process reopens this database.
    """

    def __init__(self, path: str | Path, identity: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity_json = json.dumps(dict(identity), sort_keys=True, allow_nan=False)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, value TEXT)"
            )
            db.execute("INSERT OR IGNORE INTO identity VALUES (1, ?)", (identity_json,))
            if (
                db.execute("SELECT value FROM identity WHERE id=1").fetchone()[0]
                != identity_json
            ):
                raise RequestConflict(
                    "ledger belongs to a different run or service configuration"
                )
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                request_json TEXT NOT NULL, status TEXT NOT NULL,
                response_json TEXT, started_at REAL NOT NULL, finished_at REAL,
                error TEXT)""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5.0)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        encoded = json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM requests WHERE request_id=?", (request["request_id"],)
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise RequestConflict(
                        "request_id was already used with a different payload"
                    )
                if row["status"] != "completed":
                    raise RequestConflict(
                        "request is unresolved; inspect execution before any retry"
                    )
                return self._receipt(row)
            unresolved = db.execute(
                "SELECT request_id FROM requests WHERE status != 'completed' LIMIT 1"
            ).fetchone()
            if unresolved is not None:
                raise RequestConflict(
                    "another request is active or indeterminate; new dispatch is blocked"
                )
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, 'pending', NULL, ?, NULL, NULL)",
                (request["request_id"], fingerprint, encoded, time.time()),
            )
        return None

    def complete(self, request_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(dict(response), sort_keys=True, allow_nan=False)
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE requests SET status='completed', response_json=?, finished_at=? WHERE request_id=? AND status='pending'",
                (encoded, time.time(), request_id),
            )
            if cursor.rowcount != 1:
                raise RequestConflict("request is not pending")
        return self.get(request_id)

    def mark_indeterminate(self, request_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET status='indeterminate', error=? WHERE request_id=? AND status='pending'",
                ("dispatch raised; physical outcome must be inspected", request_id),
            )

    def get(self, request_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._receipt(row)

    @staticmethod
    def _receipt(row: sqlite3.Row) -> dict[str, Any]:
        request = json.loads(row["request_json"])
        return {
            "request_id": row["request_id"],
            "status": row["status"],
            "tool": request["name"],
            "episode_id": request["expected_episode_id"],
            "timestep": request["expected_timestep"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "response": json.loads(row["response_json"])
            if row["response_json"]
            else None,
        }


class LangGraphToolService:
    """A bounded tool-turn graph over an already-bound, active CARVE session.

    No model decisions are invented here. Codex or another external planner
    submits typed intents. SQLite provides dispatch receipts, NOT checkpoints
    of the physical world or learned/semantic task memory.
    """

    READ_ONLY_TOOLS = frozenset({"observe", "retrieve_memory", "verify"})

    def __init__(
        self,
        gateway: ExternalToolAgentGateway,
        *,
        ledger_path: str | Path,
        allow_execution: bool = False,
    ) -> None:
        if type(allow_execution) is not bool:
            raise ValueError("allow_execution must be an explicit boolean")
        self.gateway = gateway
        self.allow_execution = allow_execution
        catalog = gateway.catalog()
        self.identity = {
            "schema_version": 1,
            "langgraph_version": version("langgraph"),
            "run_id": catalog["run_id"],
            "config_fingerprint": catalog["config_fingerprint"],
            "allow_execution": allow_execution,
            "tool_catalog": catalog["tools"],
        }
        self.ledger = RequestLedger(ledger_path, self.identity)
        graph = StateGraph(ToolTurn)
        graph.add_node("admit", self._admit)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("persist_receipt", self._persist_receipt)
        graph.add_edge(START, "admit")
        graph.add_conditional_edges(
            "admit",
            lambda s: "cached" if s["replayed"] else "new",
            {"cached": END, "new": "dispatch"},
        )
        graph.add_edge("dispatch", "persist_receipt")
        graph.add_edge("persist_receipt", END)
        # Never attach automatic retries/checkpoint replay to a physical tool.
        self.graph = graph.compile()

    def catalog(self) -> dict[str, Any]:
        catalog = self.gateway.catalog()
        catalog["tools"] = [
            tool
            for tool in catalog["tools"]
            if self.allow_execution or tool["name"] in self.READ_ONLY_TOOLS
        ]
        return {
            **catalog,
            "allow_execution": self.allow_execution,
            "orchestrator": "langgraph",
            "persistence": "sqlite_tool_receipts",
        }

    def invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = normalize_request(payload)
        state = self.graph.invoke({"request": request}, {"recursion_limit": 8})
        return {**state["receipt"], "replayed": state["replayed"]}

    def _admit(self, state: ToolTurn) -> ToolTurn:
        request = state["request"]
        allowed = {tool["name"] for tool in self.gateway.catalog()["tools"]}
        if request["name"] not in allowed:
            raise RequestInvalid("unknown or disallowed tool")
        if not self.allow_execution and request["name"] not in self.READ_ONLY_TOOLS:
            raise ExecutionDisabled("state-changing tools are disabled on this service")
        receipt = self.ledger.reserve(request)
        if receipt is not None:
            return {"replayed": True, "receipt": receipt}
        return {"replayed": False}

    def _dispatch(self, state: ToolTurn) -> ToolTurn:
        try:
            response = self.gateway.invoke(state["request"])
        except Exception as exc:
            self.ledger.mark_indeterminate(state["request"]["request_id"])
            raise RequestConflict(
                "dispatch outcome unknown; automatic replay is disabled"
            ) from exc
        return {"response": response}

    def _persist_receipt(self, state: ToolTurn) -> ToolTurn:
        return {
            "receipt": self.ledger.complete(
                state["request"]["request_id"], state["response"]
            )
        }
