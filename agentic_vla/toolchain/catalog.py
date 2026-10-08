"""Canonical agent-facing tool specifications for a CARVE deployment."""

from __future__ import annotations

from .contracts import ToolEffect, ToolSpec


def core_tool_specs() -> tuple[ToolSpec, ...]:
    """Return the stable tool vocabulary; deployments provide the handlers."""

    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    return (
        ToolSpec(
            name="observe",
            description="Read the latest deployable robot observation and monitor evidence.",
            input_schema=empty,
            effect=ToolEffect.OBSERVE,
        ),
        ToolSpec(
            name="retrieve_memory",
            description="Retrieve bounded affordance, procedure, and failure context.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            effect=ToolEffect.MEMORY,
        ),
        ToolSpec(
            name="vla_act",
            description="Request one admitted frozen-VLA action chunk for a typed subgoal.",
            input_schema={
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "minLength": 1},
                    "expected_outcome": {"type": "string"},
                },
                "required": ["instruction"],
                "additionalProperties": False,
            },
            effect=ToolEffect.VLA,
            requires_safe_boundary=True,
        ),
        ToolSpec(
            name="run_skill",
            description="Execute one registered, bounded physical skill by identifier.",
            input_schema={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "minLength": 1},
                    "skill_args": {"type": "object"},
                    "expected_outcome": {"type": "string"},
                },
                "required": ["skill_id", "skill_args"],
                "additionalProperties": False,
            },
            effect=ToolEffect.SKILL,
            requires_safe_boundary=True,
            max_calls_per_episode=3,
        ),
        ToolSpec(
            name="verify",
            description="Request deployable verification of an expected symbolic outcome.",
            input_schema={
                "type": "object",
                "properties": {
                    "expected_outcome": {"type": "string", "minLength": 1},
                },
                "required": ["expected_outcome"],
                "additionalProperties": False,
            },
            effect=ToolEffect.OBSERVE,
        ),
        ToolSpec(
            name="safe_hold",
            description="Enter the adapter-defined stationary safe-hold lifecycle.",
            input_schema={
                "type": "object",
                "properties": {"reason": {"type": "string", "minLength": 1}},
                "required": ["reason"],
                "additionalProperties": False,
            },
            effect=ToolEffect.LIFECYCLE,
        ),
        ToolSpec(
            name="finish",
            description="Finish the current episode with an auditable status and summary.",
            input_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "success",
                            "failure",
                            "stuck",
                            "safe_stop",
                            "completed",
                        ],
                    },
                    "summary": {"type": "string", "minLength": 1},
                },
                "required": ["status", "summary"],
                "additionalProperties": False,
            },
            effect=ToolEffect.LIFECYCLE,
        ),
    )
