"""Factories that assemble configured CARVE components without benchmark branches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agentic_vla.configuration import (
    CarveRunConfig,
    PlannerExecutionMode,
    PlannerRunConfig,
)
from agentic_vla.optimization import ProfileManifest
from agentic_vla.runtime import (
    AsyncGuardedHighLevelAgent,
    CarveRuntime,
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    build_vision_planner,
)
from agentic_vla.runtime.agent import PlannerCallable
from agentic_vla.runtime.harness import PlannerFactory
from agentic_vla.toolchain.runtime import (
    EmbodiedToolBindings,
    MemoryHandler,
    ObservationHandler,
    SafeHoldHandler,
    SkillPrimitiveHandler,
    VerificationHandler,
)
from agentic_vla.toolchain.vla import (
    ActionChunkExecutor,
    InferenceMetadataSource,
    ObservationSource,
    ProfileAdmittedVlaPrimitive,
)


def build_planner_factory(
    config: PlannerRunConfig,
    *,
    scripted_infer: PlannerCallable | None = None,
    environ: Mapping[str, str] | None = None,
) -> PlannerFactory | None:
    """Build an episode-scoped planner or leave authority to a coding agent.

    An external coding agent, such as Codex, drives the seven tools directly and
    therefore has no hidden in-process VLM callable. Embedded Qwen-VL and scripted
    baselines use the same guarded asynchronous planner lifecycle.
    """

    if config.mode is PlannerExecutionMode.EXTERNAL_CODING_AGENT:
        if scripted_infer is not None:
            raise ValueError("external coding agents must use the tool boundary")
        return None
    if config.mode is PlannerExecutionMode.SCRIPTED:
        if scripted_infer is None:
            raise ValueError("scripted planner mode requires scripted_infer")
        infer = scripted_infer
    else:
        if scripted_infer is not None:
            raise ValueError("embedded VLM mode does not accept scripted_infer")
        provider = config.provider_config()
        if provider is None:
            raise ValueError("embedded VLM provider configuration is missing")
        infer = build_vision_planner(provider, environ=environ)

    def factory(episode_id: str | int) -> AsyncGuardedHighLevelAgent:
        agent = GuardedHighLevelAgent(
            infer,
            HighLevelAgentConfig(
                max_calls_per_episode=config.max_calls_per_episode,
                max_grounding_repairs=config.max_grounding_repairs,
                minimum_intervention_confidence=(
                    config.minimum_intervention_confidence
                ),
                fail_closed=True,
            ),
        )
        agent.reset(episode_id)
        return AsyncGuardedHighLevelAgent(agent)

    return factory


def build_profile_admitted_tool_bindings(
    config: CarveRunConfig,
    *,
    runtime: CarveRuntime,
    observation_source: ObservationSource,
    action_executor: ActionChunkExecutor,
    observe: ObservationHandler,
    retrieve_memory: MemoryHandler,
    run_skill: SkillPrimitiveHandler,
    verify: VerificationHandler,
    safe_hold: SafeHoldHandler,
    profile_manifest: ProfileManifest | str | Path | None = None,
    inference_metadata_source: InferenceMetadataSource | None = None,
) -> EmbodiedToolBindings:
    """Assemble the seven-tool deployment boundary around an admitted VLA."""

    manifest = profile_manifest
    if isinstance(manifest, (str, Path)):
        manifest = ProfileManifest.load(manifest)
    vla_primitive = ProfileAdmittedVlaPrimitive(
        config=config,
        runtime=runtime,
        observation_source=observation_source,
        action_executor=action_executor,
        profile_manifest=manifest,
        inference_metadata_source=inference_metadata_source,
    )
    return EmbodiedToolBindings(
        observe=observe,
        retrieve_memory=retrieve_memory,
        vla_act=vla_primitive,
        run_skill=run_skill,
        verify=verify,
        safe_hold=safe_hold,
    )
