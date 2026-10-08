"""Assembly tests for embedded and external CARVE planners."""

from __future__ import annotations

import json

import pytest

from agentic_vla.assembly import build_planner_factory
from agentic_vla.configuration import CarveRunConfig, PlannerRunConfig


def test_external_codex_uses_no_hidden_planner_callable() -> None:
    config = CarveRunConfig.load("configs/carve_pi05_codex_tool_agent.json")
    assert build_planner_factory(config.planner) is None


def test_scripted_factory_is_episode_scoped() -> None:
    config = PlannerRunConfig(
        mode="scripted",
        planner_id="scripted-baseline",
        model="fixture",
        max_calls_per_episode=2,
    )

    def infer(_request):
        return json.dumps(
            {
                "intent": "continue",
                "rationale": "nominal execution",
                "confidence": 1.0,
                "subgoal": "",
                "vla_instruction": None,
                "skill_id": None,
                "skill_args": {},
                "failure_type": "",
                "scene_graph_update": {},
            }
        )

    factory = build_planner_factory(config, scripted_infer=infer)
    assert factory is not None
    first = factory("episode-a")
    second = factory("episode-b")
    assert first is not second
    first.close()
    second.close()


def test_planner_modes_reject_wrong_construction_path() -> None:
    external = CarveRunConfig.load(
        "configs/carve_pi05_codex_tool_agent.json"
    ).planner
    with pytest.raises(ValueError, match="tool boundary"):
        build_planner_factory(external, scripted_infer=lambda _: {})

    scripted = PlannerRunConfig(
        mode="scripted", planner_id="scripted", model="fixture"
    )
    with pytest.raises(ValueError, match="requires scripted_infer"):
        build_planner_factory(scripted)
