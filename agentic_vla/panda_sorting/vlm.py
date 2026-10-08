"""Low-frequency VLM planner boundary for the Panda sorting task.

The VLM may choose semantic subgoals and registered recovery skills. It cannot
produce continuous robot actions; those remain the responsibility of the
closed-loop executor.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from agentic_vla.runtime import (
    AgentIntent,
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    HighLevelAgentResult,
    OpenAICompatibleVisionPlanner,
    PlannerProviderConfig,
    build_vision_planner,
)

from .environment import PandaSortingObservation


SORTING_SKILLS = (
    "reobserve_scene",
    "wide_reapproach",
    "retract_and_regrasp",
    "verify_destination",
)


def build_sorting_context(
    observation: PandaSortingObservation,
    *,
    task_instruction: str,
    episode_id: str | int,
    timestep: int,
    trigger: str,
    current_subgoal: str,
    memory: tuple[str, ...] = (),
    memory_records: tuple[Mapping[str, Any], ...] = (),
    risk: Mapping[str, Any] | None = None,
    remaining_retries: int = 1,
    remaining_recoveries: int = 1,
    available_skills: tuple[str, ...] = SORTING_SKILLS,
    allowed_intents: tuple[AgentIntent, ...] = (
        AgentIntent.CONTINUE,
        AgentIntent.VLA_ACT,
        AgentIntent.RUN_SKILL,
        AgentIntent.SAFE_STOP,
    ),
) -> HighLevelAgentContext:
    """Build a planner request from deployable observation channels only."""
    frames = {
        "external_rgb": np.asarray(observation.external_rgb),
        "wrist_rgb": np.asarray(observation.wrist_rgb),
    }
    if observation.front_rgb is not None:
        frames["front_rgb"] = np.asarray(observation.front_rgb)
    return HighLevelAgentContext(
        task_instruction=task_instruction,
        trigger=trigger,
        episode_id=episode_id,
        timestep=timestep,
        frames=frames,
        robot_state=tuple(float(value) for value in observation.proprio),
        risk=dict(risk or {"event": None, "bucket": "low", "score": 0.0}),
        current_subgoal=current_subgoal,
        memory=memory,
        memory_records=memory_records,
        available_skills=available_skills,
        allowed_intents=allowed_intents,
        remaining_retries=remaining_retries,
        remaining_recoveries=remaining_recoveries,
        deadline_slack_ms=None,
    )


def make_openai_compatible_sorting_planner(
    *,
    endpoint: str,
    model: str,
    timeout_s: float = 30.0,
    max_calls_per_episode: int = 3,
) -> GuardedHighLevelAgent:
    """Create a schema-guarded VLM planner for event-boundary decisions."""
    return GuardedHighLevelAgent(
        OpenAICompatibleVisionPlanner(
            endpoint=endpoint,
            model=model,
            timeout_s=timeout_s,
            max_tokens=160,
        ),
        HighLevelAgentConfig(
            max_calls_per_episode=max_calls_per_episode,
            minimum_intervention_confidence=0.55,
            fail_closed=True,
        ),
    )


def make_sorting_planner(
    *,
    provider: str,
    endpoint: str,
    model: str,
    api_key_env: str = "",
    timeout_s: float = 30.0,
    max_calls_per_episode: int = 3,
) -> GuardedHighLevelAgent:
    """Create a vendor-neutral schema-guarded multimodal planner."""

    return GuardedHighLevelAgent(
        build_vision_planner(
            PlannerProviderConfig(
                provider=provider,
                endpoint=endpoint,
                model=model,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
                max_tokens=160,
            )
        ),
        HighLevelAgentConfig(
            max_calls_per_episode=max_calls_per_episode,
            minimum_intervention_confidence=0.55,
            fail_closed=True,
        ),
    )


def shadow_plan(
    planner: GuardedHighLevelAgent,
    observation: PandaSortingObservation,
    *,
    task_instruction: str,
    episode_id: str | int,
    trigger: str = "task_start",
) -> HighLevelAgentResult:
    """Call the VLM for audit only; this helper intentionally executes nothing."""
    context = build_sorting_context(
        observation,
        task_instruction=task_instruction,
        episode_id=episode_id,
        timestep=0,
        trigger=trigger,
        current_subgoal="inspect sorting scene",
    )
    return planner.decide(context)
