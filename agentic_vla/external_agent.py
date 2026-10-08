"""Safe tool gateway for external coding agents such as Codex."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agentic_vla.session import CarveAgentSession
from agentic_vla.toolchain import ToolExecutionContext, reject_direct_action_fields


@dataclasses.dataclass(frozen=True)
class ExternalToolRequest:
    """Agent-provided intent without agent-provided execution authority."""

    name: str
    arguments: Mapping[str, Any]
    expected_episode_id: str | int
    expected_timestep: int
    request_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported external-tool request schema")
        if not self.name.strip() or not self.request_id.strip():
            raise ValueError("external-tool name and request_id must not be empty")
        if self.expected_timestep < 0:
            raise ValueError("expected_timestep must be non-negative")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("external-tool arguments must be a mapping")
        reject_direct_action_fields(self.arguments, path="external_tool.arguments")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalToolRequest":
        allowed = {
            "schema_version",
            "request_id",
            "name",
            "arguments",
            "expected_episode_id",
            "expected_timestep",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"external-tool request has unknown fields: {unknown}")
        return cls(**dict(payload))


class ExternalToolAgentGateway:
    """Expose discoverable CARVE tools while the Harness owns live context."""

    def __init__(
        self,
        session: CarveAgentSession,
        *,
        context_source: Callable[[], ToolExecutionContext],
    ) -> None:
        if not session.external_tool_agent:
            raise ValueError("external tool gateway requires external_coding_agent mode")
        if not callable(context_source):
            raise TypeError("context_source must be callable")
        self.session = session
        self.context_source = context_source

    def catalog(self) -> dict[str, Any]:
        """Return a provider-neutral tool catalog for a coding-agent prompt."""

        return {
            "schema_version": 1,
            "protocol": "carve.external-tools.v1",
            "run_id": self.session.config.run_id,
            "config_fingerprint": self.session.config.fingerprint,
            "tools": list(
                self.session.tools.registry.specs(
                    allowed_tools=self.session.config.harness.allowed_tools
                )
            ),
        }

    def invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = ExternalToolRequest.from_dict(payload)
        context = self.context_source()
        stale_reason = None
        if request.expected_episode_id != context.episode_id:
            stale_reason = "external request belongs to another episode"
        elif request.expected_timestep != context.timestep:
            stale_reason = "external request is stale for the current timestep"
        if stale_reason is not None:
            response = {
                "schema_version": 1,
                "request_id": request.request_id,
                "accepted": False,
                "error": stale_reason,
                "result": None,
            }
            self.session.workspace.append_event(
                "external_tool_rejected", response, source="external_agent_gateway"
            )
            return response

        result = self.session.invoke_external_tool(
            request.name,
            dict(request.arguments),
            context=context,
        )
        response = {
            "schema_version": 1,
            "request_id": request.request_id,
            "accepted": result.accepted,
            "error": result.error,
            "result": result.to_dict(),
        }
        self.session.workspace.append_transcript(
            role="tool",
            content=json.dumps(response, sort_keys=True),
            metadata={
                "request_id": request.request_id,
                "tool": request.name,
                "episode_id": context.episode_id,
                "timestep": context.timestep,
            },
        )
        return response
