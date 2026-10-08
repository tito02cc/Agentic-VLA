#!/usr/bin/env python3
"""Run auditable Agentic VLA episodes in the official LIBERO-PRO simulator."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from collections.abc import Mapping
from typing import Any

import imageio.v2 as imageio
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_ROOT = pathlib.Path(
    os.environ.get("AGENTIC_VLA_OPENPI_ROOT", "/home/admin1/openpi")
).expanduser()
LIBERO_ROOT = pathlib.Path(
    os.environ.get(
        "AGENTIC_VLA_LIBERO_ROOT",
        "/home/admin1/ct/benchmark-sources/LIBERO-PRO",
    )
).expanduser()
for path in (
    PROJECT_ROOT,
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
    LIBERO_ROOT,
):
    sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from agentic_vla.assembly import build_profile_admitted_tool_bindings  # noqa: E402
from agentic_vla.benchmarks import (  # noqa: E402
    LiberoEmbodiedSkillLibrary,
    LiberoRuntimeAdapter,
)
from agentic_vla.benchmarks.libero_pro import (  # noqa: E402
    LIBERO_PRO_SUITES,
    build_libero_pro_haa_index,
    validate_libero_pro_root,
)
from agentic_vla.configuration import (  # noqa: E402
    ArtifactRunConfig,
    BenchmarkRunConfig,
    CarveRunConfig,
    PlannerExecutionMode,
)
from agentic_vla.runtime import (  # noqa: E402
    ActionSpec,
    AgentIntent,
    AgenticKnowledgeProvider,
    CarveRuntime,
    ExecutionRiskMonitor,
    HarnessState,
    HighLevelAgentContext,
    JointControllerConfig,
    JointRecoveryComputeController,
    MonitorConfig,
    PrimitiveBoundaryPolicy,
    ProceduralTaskMemory,
    ProceduralStep,
    RecoverySkillRegistry,
    RiskAssessment,
    TaskStartPolicy,
    build_vision_planner,
    compose_grounded_vla_instruction,
)
from agentic_vla.runtime.adapters import Pi05Adapter  # noqa: E402
from agentic_vla.session import CarveAgentSession  # noqa: E402
from agentic_vla.toolchain import (  # noqa: E402
    PrimitiveExecutionReport,
    PrimitiveStatus,
    RunWorkspace,
    ToolExecutionContext,
    VerificationReport,
    GuardedVisualVerifier,
    VisualVerificationContext,
)


DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/carve_pi05_qwen_local.json"
    )
    parser.add_argument(
        "--suite", default="libero_10_object", choices=LIBERO_PRO_SUITES
    )
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--trial-offset",
        type=int,
        default=0,
        help=(
            "Start from this official initial-state index. Use 0 for a reference "
            "episode and 1 or greater for held-out procedural-memory evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--method",
        choices=("agentic", "fixed_recovery", "frozen_vla"),
        default="agentic",
        help="Agentic Harness, bounded fixed recovery, or matched frozen-VLA control.",
    )
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=18081)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--monitor-warmup-steps",
        type=int,
        default=6,
        help="Control steps observed before the ExecutionRiskMonitor may intervene.",
    )
    parser.add_argument(
        "--stall-recovery-streak",
        type=int,
        default=2,
        help="Consecutive monitor windows required before physical stall recovery.",
    )
    parser.add_argument(
        "--restore-snapshot",
        type=pathlib.Path,
        default=None,
        help="Evaluator-only JSON/NPZ simulator snapshot used as the episode start state.",
    )
    parser.add_argument(
        "--stale-vla-instruction",
        default="",
        help=(
            "Controlled fault: replace the cached low-level VLA instruction while "
            "preserving the benchmark task contract. Agentic runs invoke the VLM "
            "planner at task start; frozen-VLA controls retain the stale instruction."
        ),
    )
    parser.add_argument(
        "--task-start-planner",
        action="store_true",
        help=(
            "Invoke the bounded VLM Planner once at the initial safe boundary for "
            "nominal semantic grounding. Available only for --method agentic."
        ),
    )
    parser.add_argument(
        "--semantic-checkpoint-step",
        action="append",
        type=int,
        default=[],
        help=(
            "Repeatable control step at which Agentic requests one low-frequency "
            "VLM replan at the next primitive boundary."
        ),
    )
    parser.add_argument(
        "--semantic-planner-only",
        action="store_true",
        help=(
            "Ablation: keep explicit task-start/checkpoint VLM calls but mask "
            "Monitor-triggered recovery and escalation."
        ),
    )
    parser.add_argument(
        "--procedure-memory",
        type=pathlib.Path,
        default=None,
        help="Persistent verified procedural-memory JSON shared across episodes.",
    )
    parser.add_argument(
        "--promote-successful-procedure",
        action="store_true",
        help=(
            "After private evaluator success, compile the action-free primitive "
            "trace into --procedure-memory. Intended for a reference seed."
        ),
    )
    parser.add_argument(
        "--procedure-warm-start",
        action="store_true",
        help=(
            "Install the highest-ranked verified symbolic procedure at episode "
            "start and skip task-start plan generation."
        ),
    )
    parser.add_argument(
        "--task-family",
        default="",
        help="Symbolic task family stored with a promoted reference procedure.",
    )
    parser.add_argument(
        "--memory-object",
        action="append",
        default=[],
        help="Object category used to retrieve a promoted procedure; repeatable.",
    )
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "libero_pro_canonical_20260824",
    )
    parser.add_argument(
        "--run-id-prefix", default="agentic-vla-libero-pro-canonical"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-episode summaries and archive incomplete workspaces.",
    )
    return parser.parse_args()


def _configure_libero_paths(root: pathlib.Path) -> None:
    package_root = root / "libero" / "libero"
    if not package_root.is_dir():
        raise FileNotFoundError(f"LIBERO package root not found: {package_root}")
    config_root = PROJECT_ROOT / "results" / "_runtime" / "libero_pro_canonical"
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "assets": str(package_root / "assets"),
        "bddl_files": str(package_root / "bddl_files"),
        "benchmark_root": str(package_root),
        "datasets": str(root / "libero" / "datasets"),
        "init_states": str(package_root / "init_files"),
    }
    (config_root / "config.yaml").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _make_environment(task: Any, *, seed: int, size: int = 256) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    environment = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=size,
        camera_widths=size,
        camera_depths=True,
    )
    environment.seed(seed)
    return environment


def _planner_context(
    *,
    task_instruction: str,
    episode_id: str,
    observation: Any,
    risk: Any,
    session: CarveAgentSession | None,
    config: CarveRunConfig,
    current_subgoal: str,
    available_skills: tuple[str, ...],
    available_skill_specs: tuple[Mapping[str, Any], ...] = (),
) -> HighLevelAgentContext:
    counters = None if session is None else session.harness.counters
    retries = 0 if counters is None else counters.semantic_retries
    recoveries = 0 if counters is None else counters.recovery_attempts
    return HighLevelAgentContext(
        task_instruction=task_instruction,
        trigger=str(risk.event or "control_boundary"),
        episode_id=episode_id,
        timestep=observation.timestep,
        frames=_semantic_frames(observation),
        robot_state=observation.robot_state,
        risk=risk.to_dict(),
        current_subgoal=current_subgoal,
        available_skills=available_skills,
        available_skill_specs=available_skill_specs,
        remaining_retries=max(0, config.harness.retry_budget - retries),
        remaining_recoveries=max(0, config.harness.recovery_budget - recoveries),
        deployment_profile_id=config.vla.deployment_profile_id,
        deadline_slack_ms=config.optimize.deadline_ms,
        task_plan=(
            {} if session is None else session.task_plan.to_dict()
        ),
    )


def _semantic_frames(observation: Any) -> dict[str, Any]:
    """Select the global semantic view without changing VLA policy inputs."""

    frames = dict(observation.planner_frames)
    if "agentview" in frames:
        return {"agentview": frames["agentview"]}
    name = sorted(frames)[0]
    return {name: frames[name]}


def _tool_context(
    config: CarveRunConfig, *, episode_id: str, timestep: int
) -> ToolExecutionContext:
    return ToolExecutionContext(
        episode_id=episode_id,
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=config.harness.allowed_tools,
        deployment_profile_id=config.vla.deployment_profile_id,
    )


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _load_restore_snapshot(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    metadata_path = path.expanduser().resolve()
    if metadata_path.suffix != ".json" or not metadata_path.is_file():
        raise FileNotFoundError(f"restore snapshot metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("restore snapshot metadata must be a JSON object")
    array_name = str(metadata.get("array_file", "")).strip()
    array_path = metadata_path.with_name(array_name)
    if not array_name or not array_path.is_file():
        raise FileNotFoundError(f"restore snapshot array not found: {array_path}")
    with np.load(array_path) as arrays:
        required = {"sim_state", "last_action"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise KeyError(f"restore snapshot is missing arrays: {missing}")
        sim_state = np.asarray(arrays["sim_state"], dtype=np.float64).copy()
        last_action = np.asarray(arrays["last_action"], dtype=np.float32).reshape(-1).copy()
    if sim_state.ndim != 1 or last_action.shape != (7,):
        raise ValueError("restore snapshot must contain flat sim_state and 7-D last_action")
    risk = metadata.get("controller", {}).get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("restore snapshot must contain deployable controller risk")
    return {
        "metadata_path": metadata_path,
        "metadata": dict(metadata),
        "sim_state": sim_state,
        "last_action": last_action,
        "risk": dict(risk),
    }


def _risk_from_snapshot(values: Mapping[str, Any]) -> RiskAssessment:
    return RiskAssessment(
        score=float(values["score"]),
        bucket=str(values["bucket"]),
        event=None if values.get("event") is None else str(values["event"]),
        components=dict(values.get("components", {})),
        evidence={**dict(values.get("evidence", {})), "source": "deployable_snapshot_monitor"},
    )


def _disable_intervention(risk: RiskAssessment) -> RiskAssessment:
    return dataclasses.replace(
        risk,
        score=0.0,
        bucket="low",
        event=None,
        components={name: 0.0 for name in risk.components},
        evidence={**risk.evidence, "intervention_disabled": True},
    )


def _instruction_receipt_risk(stale_instruction: str) -> RiskAssessment:
    """Expose a deployable instruction-channel fault without evaluator semantics."""

    if not stale_instruction.strip():
        raise ValueError("stale instruction must not be empty")
    return RiskAssessment(
        score=0.75,
        bucket="high",
        event="stale_subgoal",
        components={"instruction_integrity": 0.75},
        evidence={
            "source": "runtime_instruction_receipt",
            "task_revision": 1,
            "cached_subgoal_revision": 0,
            "evaluator_state_used": False,
        },
    )


def _run_episode(
    *,
    base_config: CarveRunConfig,
    suite_name: str,
    task_id: int,
    trial: int,
    seed: int,
    task: Any,
    initial_state: Any,
    restore_snapshot: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    snapshot_suffix = (
        ""
        if restore_snapshot is None
        else f"-restore-{restore_snapshot['metadata_path'].stem}"
    )
    stale_instruction = str(args.stale_vla_instruction).strip()
    fault_suffix = "-stale-subgoal" if stale_instruction else ""
    startup_suffix = "-startup-planner" if args.task_start_planner else ""
    checkpoint_steps = tuple(sorted(set(args.semantic_checkpoint_step)))
    checkpoint_suffix = "-semantic-checkpoint" if checkpoint_steps else ""
    semantic_only_suffix = "-semantic-only" if args.semantic_planner_only else ""
    run_id = (
        f"{args.run_id_prefix}-{args.method}-{suite_name}"
        f"-t{task_id}-r{trial}-s{seed}{snapshot_suffix}{fault_suffix}{startup_suffix}"
        f"{checkpoint_suffix}{semantic_only_suffix}"
    )
    episode_summary_path = args.results_root / "_episode_summaries" / f"{run_id}.json"
    if args.resume and episode_summary_path.is_file():
        resumed = json.loads(episode_summary_path.read_text(encoding="utf-8"))
        if not isinstance(resumed, Mapping):
            raise TypeError(f"episode summary must be a JSON object: {episode_summary_path}")
        video_path = pathlib.Path(str(resumed.get("video", "")))
        workspace_path = pathlib.Path(str(resumed.get("workspace", "")))
        if not video_path.is_file() or not (workspace_path / "run_summary.json").is_file():
            raise RuntimeError(f"completed episode artifacts are missing for {run_id}")
        return dict(resumed)
    planner_config = base_config.planner
    if args.method == "fixed_recovery":
        planner_config = dataclasses.replace(
            base_config.planner,
            mode=PlannerExecutionMode.EXTERNAL_CODING_AGENT,
            planner_id="disabled-fixed-recovery",
            provider=None,
            model="",
            endpoint="",
            api_key_env="",
        )
    config = dataclasses.replace(
        base_config,
        run_id=run_id,
        planner=planner_config,
        harness=dataclasses.replace(
            base_config.harness,
            task_start_policy=(
                TaskStartPolicy.STARTUP_WAIT
                if args.method == "agentic"
                and (stale_instruction or args.task_start_planner)
                else TaskStartPolicy.EVENT_ONLY
            ),
            primitive_boundary_policy=(
                PrimitiveBoundaryPolicy.EVENT_ONLY
                if args.method == "fixed_recovery"
                else PrimitiveBoundaryPolicy.SELECTIVE
            ),
            recovery_budget=1 if args.method in {"agentic", "fixed_recovery"} else 0,
            retry_budget=2,
        ),
        benchmark=BenchmarkRunConfig(
            environment_id="libero-pro",
            suite_id=suite_name,
            task_id=f"task-{task_id}:trial-{trial}",
            seed=seed,
            max_steps=args.max_steps,
            record_video=True,
        ),
        artifacts=ArtifactRunConfig(
            results_root=str(args.results_root), persist_frames=False
        ),
    )
    if config.workspace_path.exists():
        if not args.resume:
            raise FileExistsError(f"run workspace already exists: {config.workspace_path}")
        archive_root = args.results_root / "_partial_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / f"{run_id}-{time.time_ns()}"
        config.workspace_path.rename(archive_path)

    env = _make_environment(task, seed=seed)
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction=task.language,
        initial_state=initial_state,
        image_transform=lambda image: image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, 256, 256)
        ),
        max_steps=args.max_steps,
        settle_steps=0 if restore_snapshot is not None else args.settle_steps,
        settle_action=DUMMY_ACTION,
        restore_state=(
            None if restore_snapshot is None else restore_snapshot["sim_state"]
        ),
    )
    observation = adapter.reset()
    critic_reference_frames = _semantic_frames(observation)
    client = WebsocketClientPolicy(host=args.vla_host, port=args.vla_port)
    action_spec = ActionSpec(
        action_dim=7,
        representation="normalized_delta_cartesian_pose",
        coordinate_frame="robot_base",
        gripper_convention="-1=close,+1=open",
        control_frequency_hz=20.0,
    )
    runtime = CarveRuntime(
        Pi05Adapter(
            client,
            adapter_id=config.vla.adapter_id,
            precision="bf16",
            max_action_horizon=10,
            action_spec=action_spec,
        ),
        fallback_mode="strict",
    )
    monitor = ExecutionRiskMonitor(
        MonitorConfig(window_size=3, warmup_steps=args.monitor_warmup_steps)
    )
    recovery_skills = RecoverySkillRegistry.with_default_skills()
    haa_index = build_libero_pro_haa_index()
    procedure_memory_path = (
        None
        if args.procedure_memory is None
        else args.procedure_memory.expanduser().resolve()
    )
    procedural_memory = (
        ProceduralTaskMemory.load(procedure_memory_path)
        if procedure_memory_path is not None and procedure_memory_path.is_file()
        else ProceduralTaskMemory()
    )
    visual_critic = None
    if config.planner.mode is PlannerExecutionMode.EMBEDDED_VLM:
        provider = config.planner.provider_config()
        if provider is not None:
            visual_critic = GuardedVisualVerifier(
                build_vision_planner(provider),
                minimum_confidence=config.planner.minimum_intervention_confidence,
            )
    critic_receipts: list[dict[str, Any]] = []
    memory_records_before = len(procedural_memory)
    last_action = np.asarray(
        DUMMY_ACTION if restore_snapshot is None else restore_snapshot["last_action"],
        dtype=np.float32,
    )
    pending_risk = None
    verification = VerificationReport(
        status="inconclusive",
        observed_outcome="no primitive has requested verification",
        confidence=0.0,
    )

    def record_control_step(step_observation: Any, action: np.ndarray) -> None:
        nonlocal last_action, pending_risk, risk
        last_action = np.asarray(action, dtype=np.float32).copy()
        assessment = monitor.update(
            proprio=step_observation.robot_state,
            commanded_action=last_action,
            frame=step_observation.planner_frames["agentview"],
            deadline_slack_ms=max(
                0.0,
                args.deadline_ms
                - (0.0 if runtime.last_trace is None else runtime.last_trace.runtime_latency_ms),
            ),
        )
        if args.method == "frozen_vla" or args.semantic_planner_only:
            assessment = _disable_intervention(assessment)
        risk = assessment
        if assessment.event is not None and (
            pending_risk is None or assessment.score >= pending_risk.score
        ):
            pending_risk = assessment
        frames.append(adapter.render_video_frame(size=args.video_size))

    def action_executor(actions: Any, context: ToolExecutionContext):
        nonlocal last_action
        rows = np.asarray(actions, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError(f"PI0.5 action chunk has invalid shape: {rows.shape}")
        report = adapter.execute_action_chunk(rows, context, on_step=record_control_step)
        return report

    embodied_skills = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: last_action,
        on_step=record_control_step,
    )
    recovery_specs = (
        {
            "skill_id": "cartesian_retract_lift_reobserve",
            "description": "Retract opposite the last motion, lift, and reobserve after a stall.",
            "arguments": {},
            "postcondition": "tool clears the suspected contact and returns a fresh observation",
        },
        {
            "skill_id": "release_retract_lift_reobserve",
            "description": "Open the gripper, retract, lift, and reobserve after a failed grasp.",
            "arguments": {},
            "postcondition": "gripper and tool clear the failed contact for replanning",
        },
    )
    available_skill_ids = tuple(
        dict.fromkeys((*embodied_skills.skill_ids, *recovery_skills.skill_ids))
    )
    available_skill_specs = tuple((*embodied_skills.planner_specs, *recovery_specs))

    def run_skill(
        skill_id: str, arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> PrimitiveExecutionReport:
        nonlocal last_action, pending_risk, risk, verification
        if skill_id in embodied_skills.skill_ids:
            report = embodied_skills.execute(skill_id, arguments, context)
            verification = VerificationReport(
                status=(
                    "inconclusive"
                    if report.status is PrimitiveStatus.SUCCEEDED
                    else "contradicted"
                ),
                observed_outcome=report.observed_outcome,
                confidence=0.6 if report.status is PrimitiveStatus.SUCCEEDED else 0.95,
                metadata={"skill_id": skill_id, **dict(report.metadata)},
            )
            monitor.reset()
            pending_risk = None
            return report
        before = np.asarray(adapter.observe().robot_state, dtype=np.float32)
        plan = recovery_skills.build(skill_id, action_spec, last_action)
        actions = np.asarray(
            [action for phase in plan.phases for action in phase.actions],
            dtype=np.float32,
        )
        execution = adapter.execute_action_chunk(
            actions,
            context,
            on_step=record_control_step,
        )
        after = np.asarray(adapter.observe().robot_state, dtype=np.float32)
        response = float(np.linalg.norm(after[:6] - before[:6]))
        moved = bool(np.isfinite(response) and response >= 0.005)
        verification = VerificationReport(
            status="inconclusive" if moved else "contradicted",
            observed_outcome=(
                f"physical response {response:.6f}; task semantics require VLM review"
                if moved
                else f"recovery produced insufficient physical response {response:.6f}"
            ),
            confidence=0.5 if moved else 0.95,
            metadata={"physical_response": response, "skill_id": skill_id},
        )
        # Recovery is a deliberate execution-regime change. Its commanded motion
        # must not be carried into the next VLA primitive as evidence of a
        # continuing policy stall.
        monitor.reset()
        pending_risk = None
        current = adapter.observe()
        risk = monitor.update(
            proprio=current.robot_state,
            commanded_action=last_action,
            frame=current.planner_frames["agentview"],
            deadline_slack_ms=args.deadline_ms,
        )
        return PrimitiveExecutionReport(
            status=(PrimitiveStatus.SUCCEEDED if moved else PrimitiveStatus.FAILED),
            started_timestep=context.timestep,
            ended_timestep=execution.ended_timestep,
            expected_outcome=(
                "robot tool has retracted and lifted clear of the suspected contact; "
                "the manipulation task may still be incomplete"
            ),
            observed_outcome=verification.observed_outcome,
            requires_semantic_check=True,
            metadata={
                "skill_id": skill_id,
                "physical_response": response,
                "executed_steps": execution.metadata.get("executed_steps", 0),
            },
        )

    def fixed_noise(context: ToolExecutionContext) -> Mapping[str, Any]:
        generator = np.random.default_rng(
            np.random.SeedSequence([seed, task_id, trial, context.timestep])
        )
        return {"noise": generator.standard_normal((10, 32), dtype=np.float32)}

    def verify_expected_outcome(
        expected_outcome: str,
        context: ToolExecutionContext,
    ) -> VerificationReport:
        nonlocal verification
        if visual_critic is None:
            return verification
        current = adapter.observe()
        result = visual_critic.verify(
            VisualVerificationContext(
                task_instruction=str(task.language),
                expected_outcome=expected_outcome,
                frames=_semantic_frames(current),
                timestep=context.timestep,
                active_stage="",
            )
        )
        verification = result.report
        critic_receipts.append(
            {
                "timestep": context.timestep,
                "expected_outcome": expected_outcome,
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "report": result.report.to_dict(),
            }
        )
        return verification

    bindings = build_profile_admitted_tool_bindings(
        config,
        runtime=runtime,
        observation_source=adapter.policy_observation,
        action_executor=action_executor,
        observe=lambda _context: {
            "frame_id": adapter.observe().frame_id,
            "timestep": adapter.timestep,
            "available_views": sorted(adapter.observe().planner_frames),
            "robot_state_dim": len(adapter.observe().robot_state),
        },
        retrieve_memory=lambda query, limit, _context: {
            "query": query,
            "affordance_records": list(
                haa_index.retrieve(
                    task_instruction=task.language,
                    failure_type=query,
                    limit=limit,
                )
            ),
            "procedural_records": list(
                procedural_memory.retrieve(
                    task_instruction=task.language,
                    limit=min(limit, 2),
                )
            ),
        },
        run_skill=run_skill,
        verify=verify_expected_outcome,
        safe_hold=lambda reason, _context: {"holding": True, "reason": reason},
        inference_metadata_source=fixed_noise,
    )
    manifest = config.to_run_manifest()
    manifest = dataclasses.replace(
        manifest,
        metadata={
            **manifest.metadata,
            "experiment_method": args.method,
            "restore_snapshot_id": (
                None
                if restore_snapshot is None
                else restore_snapshot["metadata_path"].stem
            ),
            "restore_state_role": (
                None if restore_snapshot is None else "evaluator_restore_only"
            ),
            "restore_snapshot_fingerprint": (
                None
                if restore_snapshot is None
                else restore_snapshot["metadata"].get("snapshot_fingerprint")
            ),
            "controlled_fault": (
                None
                if not stale_instruction
                else {
                    "id": "stale_subgoal_v1",
                    "injection_surface": "cached_low_level_vla_instruction",
                    "task_contract_preserved": True,
                    "detection_source": "runtime_instruction_receipt",
                    "evaluator_state_used": False,
                    "stale_instruction": stale_instruction,
                }
            ),
            "task_start_planner": bool(args.task_start_planner),
            "semantic_checkpoint_steps": list(checkpoint_steps),
            "semantic_planner_only": bool(args.semantic_planner_only),
            "monitor_warmup_steps": int(args.monitor_warmup_steps),
            "stall_recovery_streak": int(args.stall_recovery_streak),
        },
    )
    workspace = RunWorkspace(config.workspace_path, manifest)
    knowledge_provider = AgenticKnowledgeProvider(haa_index, procedural_memory)
    session = CarveAgentSession(
        config,
        bindings=bindings,
        controller=JointRecoveryComputeController(
            JointControllerConfig(
                planner_after_failures=max(1, config.harness.retry_budget),
                max_recovery_attempts=config.harness.recovery_budget,
                stall_recovery_streak=args.stall_recovery_streak,
            )
        ),
        knowledge_provider=knowledge_provider,
        workspace=workspace,
    )
    episode_id = (
        f"{suite_name}:{task_id}:{trial}:{seed}{snapshot_suffix}{fault_suffix}"
        f"{startup_suffix}{checkpoint_suffix}{semantic_only_suffix}"
    )
    current_instruction = stale_instruction or str(task.language)
    current_subgoal = stale_instruction or "execute"
    frames = [adapter.render_video_frame(size=args.video_size)]
    risk = (
        monitor.update(
            proprio=observation.robot_state,
            commanded_action=last_action,
            frame=observation.planner_frames["agentview"],
            deadline_slack_ms=args.deadline_ms,
        )
        if restore_snapshot is None
        else _risk_from_snapshot(restore_snapshot["risk"])
    )
    if stale_instruction:
        risk = _instruction_receipt_risk(stale_instruction)
    if args.method == "frozen_vla" or args.semantic_planner_only:
        risk = _disable_intervention(risk)
    start_context = _planner_context(
        task_instruction=task.language,
        episode_id=episode_id,
        observation=observation,
        risk=risk,
        session=None,
        config=config,
        current_subgoal=current_subgoal,
        available_skills=available_skill_ids,
        available_skill_specs=available_skill_specs,
    )
    if args.task_start_planner and risk.event is None:
        start_context = dataclasses.replace(start_context, trigger="task_start")
    session.start(start_context)
    warm_start_procedure_id = None
    if args.procedure_warm_start:
        retrieved = procedural_memory.retrieve(
            task_instruction=str(task.language),
            limit=1,
        )
        if not retrieved:
            raise RuntimeError("procedure warm-start found no matching verified record")
        record = retrieved[0]
        if str(record.get("verification_result", "")) != "verified":
            raise RuntimeError("procedure warm-start requires a verified record")
        steps = tuple(
            ProceduralStep(
                stage=str(step["stage"]),
                intent=str(step["intent"]),
                subgoal=str(step["subgoal"]),
                expected_outcome=str(step["expected_outcome"]),
                skill_id=(
                    None if step.get("skill_id") is None else str(step["skill_id"])
                ),
                constraints=tuple(str(value) for value in step.get("constraints", [])),
            )
            for step in record["steps"]
        )
        warm_start_procedure_id = str(record["procedure_id"])
        session.install_retrieved_task_plan(
            steps,
            available_skills=available_skill_ids,
            timestep=observation.timestep,
            procedure_id=warm_start_procedure_id,
            task_instruction=str(task.language),
        )
        session.select_active_task_plan_step(
            timestep=observation.timestep,
            source="verified_procedure_memory",
        )
        active = session.task_plan.active
        assert active is not None
        current_subgoal = active.step.subgoal
        current_instruction = compose_grounded_vla_instruction(
            task.language,
            subgoal=current_subgoal,
            planner_instruction=current_subgoal,
        )
    started = time.perf_counter()
    status = "stuck"
    reason = "maximum control steps reached"
    requested_checkpoints: set[int] = set()
    harness_stage_advances_without_planner = 0
    inconclusive_checkpoints_continued_without_planner = 0
    contradicted_checkpoints_deferred_to_next_observation = 0
    monitor_resets_after_stage_advance = 0
    final_stage_confirmations_without_planner = 0
    private_evaluator_plan_confirmations = 0
    try:
        while adapter.timestep < args.max_steps:
            observation = adapter.observe()
            planner_context = _planner_context(
                task_instruction=task.language,
                episode_id=episode_id,
                observation=observation,
                risk=risk,
                session=session,
                config=config,
                current_subgoal=current_subgoal,
                available_skills=available_skill_ids,
                available_skill_specs=available_skill_specs,
            )
            due_checkpoint = next(
                (
                    step
                    for step in checkpoint_steps
                    if step <= adapter.timestep and step not in requested_checkpoints
                ),
                None,
            )
            if due_checkpoint is not None:
                requested_checkpoints.add(due_checkpoint)
                active_plan_step = session.task_plan.active
                advanced_to_next_stage = False
                continue_current_stage = False
                if visual_critic is not None and active_plan_step is not None:
                    paired_frames = {
                        **{
                            f"before_{name}": frame
                            for name, frame in critic_reference_frames.items()
                        },
                        **{
                            f"current_{name}": frame
                            for name, frame in _semantic_frames(observation).items()
                        },
                    }
                    critic_result = visual_critic.verify(
                        VisualVerificationContext(
                            task_instruction=str(task.language),
                            expected_outcome=(
                                active_plan_step.step.expected_outcome
                            ),
                            frames=paired_frames,
                            timestep=observation.timestep,
                            active_stage=active_plan_step.step.stage,
                            require_visual_change=True,
                        )
                    )
                    critic_receipts.append(
                        {
                            "timestep": observation.timestep,
                            "expected_outcome": (
                                active_plan_step.step.expected_outcome
                            ),
                            "active_stage": active_plan_step.step.stage,
                            "accepted": critic_result.accepted,
                            "elapsed_ms": critic_result.elapsed_ms,
                            "error": critic_result.error,
                            "report": critic_result.report.to_dict(),
                        }
                    )
                    session.verify_active_plan_step(
                        critic_result.report,
                        timestep=observation.timestep,
                    )
                    planner_context = dataclasses.replace(
                        planner_context,
                        task_plan=session.task_plan.to_dict(),
                    )
                    if critic_result.report.status.value == "confirmed":
                        critic_reference_frames = _semantic_frames(observation)
                        next_stage = session.task_plan.active
                        if next_stage is not None:
                            session.select_active_task_plan_step(
                                timestep=observation.timestep,
                                source="verified_plan_progression",
                            )
                            current_subgoal = next_stage.step.subgoal
                            current_instruction = compose_grounded_vla_instruction(
                                task.language,
                                subgoal=current_subgoal,
                                planner_instruction=current_subgoal,
                            )
                            planner_context = dataclasses.replace(
                                planner_context,
                                current_subgoal=current_subgoal,
                                task_plan=session.task_plan.to_dict(),
                            )
                            advanced_to_next_stage = True
                            harness_stage_advances_without_planner += 1
                            # A verified semantic transition starts a different
                            # physical primitive. Do not carry motion statistics
                            # from the completed stage into the next one.
                            monitor.reset()
                            pending_risk = None
                            monitor_resets_after_stage_advance += 1
                        else:
                            # The semantic ledger is complete. Give the private
                            # evaluator/control loop one bounded settling chunk;
                            # there is no remaining stage for a Planner to choose.
                            continue_current_stage = True
                            final_stage_confirmations_without_planner += 1
                    elif critic_result.report.status.value == "inconclusive":
                        # Inconclusive visual evidence neither authorizes a plan
                        # transition nor justifies another semantic model call.
                        # Keep the active physical stage and gather more evidence.
                        continue_current_stage = True
                        inconclusive_checkpoints_continued_without_planner += 1
                    elif any(
                        step > due_checkpoint and step not in requested_checkpoints
                        for step in checkpoint_steps
                    ):
                        # A fixed-time checkpoint can precede completion on a
                        # slower but still progressing rollout. One predeclared
                        # later observation provides bounded grace; the final
                        # contradiction still escalates through the Planner.
                        session.select_active_task_plan_step(
                            timestep=observation.timestep,
                            source="scheduled_checkpoint_observation_grace",
                        )
                        continue_current_stage = True
                        contradicted_checkpoints_deferred_to_next_observation += 1
                if advanced_to_next_stage or continue_current_stage:
                    transition = session.route_execution(
                        risk,
                        planner_context=planner_context,
                        deadline_ms=args.deadline_ms,
                        deadline_slack_ms=max(
                            0.0,
                            args.deadline_ms
                            - (
                                0.0
                                if runtime.last_trace is None
                                else runtime.last_trace.runtime_latency_ms
                            ),
                        ),
                    )
                    if transition.ticket is not None:
                        transition = session.await_planner()
                else:
                    checkpoint_context = dataclasses.replace(
                        planner_context,
                        trigger="scheduled_semantic_checkpoint",
                    )
                    transition = session.request_semantic_checkpoint(
                        checkpoint_context,
                        reason=(
                            f"scheduled semantic checkpoint at step {due_checkpoint}"
                        ),
                    )
                    if transition.ticket is not None:
                        transition = session.await_planner()
            elif session.harness.planner_pending:
                transition = session.await_planner()
            else:
                transition = session.route_execution(
                    risk,
                    planner_context=planner_context,
                    deadline_ms=args.deadline_ms,
                    deadline_slack_ms=max(
                        0.0,
                        args.deadline_ms
                        - (
                            0.0
                            if runtime.last_trace is None
                            else runtime.last_trace.runtime_latency_ms
                        ),
                    ),
                )
                if transition.ticket is not None:
                    transition = session.await_planner()
            if transition.planner is not None:
                decision = transition.planner.result.decision
                current_subgoal = decision.subgoal or current_subgoal
                if decision.vla_instruction:
                    current_instruction = compose_grounded_vla_instruction(
                        task.language,
                        subgoal=current_subgoal,
                        planner_instruction=decision.vla_instruction,
                    )
                elif (
                    decision.intent is AgentIntent.CONTINUE
                    and session.task_plan.active is not None
                    and decision.subgoal
                ):
                    current_instruction = compose_grounded_vla_instruction(
                        task.language,
                        subgoal=current_subgoal,
                        planner_instruction=decision.subgoal,
                    )
                result = session.apply_planner_transition(
                    transition,
                    context=_tool_context(
                        config, episode_id=episode_id, timestep=adapter.timestep
                    ),
                )
            else:
                result = session.apply_execution_transition(
                    transition,
                    context=_tool_context(
                        config, episode_id=episode_id, timestep=adapter.timestep
                    ),
                    vla_instruction=current_instruction,
                )
            if result is None:
                if transition.state in {HarnessState.SAFE_HOLD, HarnessState.STOP}:
                    status = "safe_stop"
                    reason = transition.reason
                    break
                continue
            if result.name == "safe_hold":
                status = "safe_stop"
                reason = str(result.output.get("reason", transition.reason))
                break
            if result.name not in {"vla_act", "run_skill"}:
                continue

            post = adapter.observe()
            completion = session.finalize_primitive(
                result,
                planner_context=_planner_context(
                    task_instruction=task.language,
                    episode_id=episode_id,
                    observation=post,
                    risk=risk,
                    session=session,
                    config=config,
                    current_subgoal=current_subgoal,
                    available_skills=available_skill_ids,
                    available_skill_specs=available_skill_specs,
                ),
            )
            if pending_risk is not None:
                risk = pending_risk
                pending_risk = None
            private = adapter.private_result()
            if private.task_success:
                while session.task_plan.active is not None:
                    active_stage = session.task_plan.active.step.stage
                    session.verify_active_plan_step(
                        VerificationReport(
                            status="confirmed",
                            observed_outcome=(
                                "official evaluator reported full task completion"
                            ),
                            confidence=1.0,
                            metadata={
                                "source": "private_evaluator_posthoc",
                                "planner_visible": False,
                                "stage": active_stage,
                            },
                        ),
                        timestep=adapter.timestep,
                        source="private_evaluator_posthoc",
                    )
                    private_evaluator_plan_confirmations += 1
                status = "success"
                reason = "LIBERO evaluator reported task completion"
                break
            if completion.transition.state in {HarnessState.SAFE_HOLD, HarnessState.STOP}:
                status = "safe_stop"
                reason = completion.transition.reason
                break

        video_path = workspace.root / "episode.mp4"
        imageio.mimwrite(video_path, frames, fps=args.video_fps, quality=7)
        workspace.register_artifact(
            "episode.mp4", path=video_path, kind="episode_video"
        )
        private = adapter.private_result()
        evaluator_path = workspace.root / "evaluator_result.json"
        _write_json(evaluator_path, private.to_private_dict())
        workspace.register_artifact(
            "evaluator_result.json",
            path=evaluator_path,
            kind="private_evaluator_result",
        )
        promoted_procedure_id = None
        if private.task_success and args.promote_successful_procedure:
            record = session.promote_verified_procedure(
                task_family=args.task_family.strip() or str(task.language),
                object_categories=tuple(args.memory_object),
                task_verification=VerificationReport(
                    status="confirmed",
                    observed_outcome=(
                        "trusted LIBERO evaluator confirmed task completion"
                    ),
                    confidence=1.0,
                ),
                notes=(f"reference suite={suite_name} task={task_id} seed={seed}",),
            )
            assert procedure_memory_path is not None
            procedural_memory.save(procedure_memory_path)
            promoted_procedure_id = record.procedure_id

        session.finish(
            status="safe_stop" if status == "safe_stop" else "completed",
            summary=(
                reason
                if status == "safe_stop"
                else "Control rollout ended; task outcome is held by the private evaluator."
            ),
            timestep=adapter.timestep,
        )
        episode_summary = {
            "run_id": run_id,
            "method": args.method,
            "restore_snapshot_id": (
                None
                if restore_snapshot is None
                else restore_snapshot["metadata_path"].stem
            ),
            "controlled_fault": "stale_subgoal_v1" if stale_instruction else None,
            "task_start_planner": bool(args.task_start_planner),
            "semantic_checkpoint_steps": list(checkpoint_steps),
            "semantic_checkpoints_requested": sorted(requested_checkpoints),
            "harness_stage_advances_without_planner": (
                harness_stage_advances_without_planner
            ),
            "inconclusive_checkpoints_continued_without_planner": (
                inconclusive_checkpoints_continued_without_planner
            ),
            "contradicted_checkpoints_deferred_to_next_observation": (
                contradicted_checkpoints_deferred_to_next_observation
            ),
            "monitor_resets_after_stage_advance": (
                monitor_resets_after_stage_advance
            ),
            "final_stage_confirmations_without_planner": (
                final_stage_confirmations_without_planner
            ),
            "private_evaluator_plan_confirmations": (
                private_evaluator_plan_confirmations
            ),
            "semantic_planner_only": bool(args.semantic_planner_only),
            "monitor_warmup_steps": int(args.monitor_warmup_steps),
            "stall_recovery_streak": int(args.stall_recovery_streak),
            "initial_vla_instruction": stale_instruction or str(task.language),
            "final_vla_instruction": current_instruction,
            "suite": suite_name,
            "task_id": task_id,
            "trial": trial,
            "seed": seed,
            "success": private.task_success,
            "status": status,
            "episode_steps": private.episode_steps,
            "elapsed_s": time.perf_counter() - started,
            "planner_calls": session.harness.counters.planner_calls,
            "semantic_retries": session.harness.counters.semantic_retries,
            "recovery_attempts": session.harness.counters.recovery_attempts,
            "procedure_memory_path": (
                None if procedure_memory_path is None else str(procedure_memory_path)
            ),
            "procedure_memory_records_before": memory_records_before,
            "procedure_memory_records_after": len(procedural_memory),
            "procedure_warm_start_id": warm_start_procedure_id,
            "knowledge_retrieval": knowledge_provider.retrieval_metrics(),
            "task_plan": session.task_plan.to_dict(),
            "visual_critic": (
                None if visual_critic is None else visual_critic.metrics()
            ),
            "visual_critic_receipts": critic_receipts,
            "promoted_procedure_id": promoted_procedure_id,
            "workspace": str(workspace.root),
            "video": str(video_path),
        }
        episode_summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(episode_summary_path, episode_summary)
        return episode_summary
    finally:
        session.close(reason="canonical-libero-pro-runner-exit")
        adapter.close()


def main() -> int:
    args = _parse_args()
    if (
        args.trials <= 0
        or args.max_steps <= 0
        or args.trial_offset < 0
        or args.monitor_warmup_steps < 0
        or args.stall_recovery_streak < 2
    ):
        raise ValueError(
            "trials/max_steps must be positive; trial offset and monitor warmup "
            "must be non-negative"
        )
    if args.restore_snapshot is not None and args.stale_vla_instruction.strip():
        raise ValueError(
            "--restore-snapshot and --stale-vla-instruction are separate protocols"
        )
    if args.task_start_planner and args.method != "agentic":
        raise ValueError("--task-start-planner requires --method agentic")
    if args.task_start_planner and args.stale_vla_instruction.strip():
        raise ValueError(
            "stale-subgoal runs already require startup planning; do not combine flags"
        )
    if any(step <= 0 for step in args.semantic_checkpoint_step):
        raise ValueError("--semantic-checkpoint-step values must be positive")
    if args.semantic_checkpoint_step and args.method != "agentic":
        raise ValueError("semantic checkpoints require --method agentic")
    if args.semantic_planner_only and args.method != "agentic":
        raise ValueError("--semantic-planner-only requires --method agentic")
    if args.semantic_planner_only and not (
        args.task_start_planner or args.semantic_checkpoint_step
    ):
        raise ValueError(
            "--semantic-planner-only requires task-start or checkpoint planning"
        )
    if args.promote_successful_procedure:
        if args.method != "agentic":
            raise ValueError("procedure promotion requires --method agentic")
        if args.procedure_memory is None:
            raise ValueError(
                "procedure promotion requires --procedure-memory"
            )
        if not args.memory_object:
            raise ValueError(
                "procedure promotion requires at least one --memory-object"
            )
    if args.procedure_warm_start:
        if args.method != "agentic":
            raise ValueError("procedure warm-start requires --method agentic")
        if args.procedure_memory is None or not args.procedure_memory.is_file():
            raise ValueError(
                "procedure warm-start requires an existing --procedure-memory"
            )
        if args.task_start_planner or args.stale_vla_instruction.strip():
            raise ValueError(
                "procedure warm-start replaces task-start/stale-subgoal planning"
            )
    validate_libero_pro_root(LIBERO_ROOT, args.suite)
    _configure_libero_paths(LIBERO_ROOT)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks})")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trial_offset + args.trials > len(initial_states):
        raise ValueError(
            "requested trial range exceeds the available official initial states: "
            f"offset={args.trial_offset}, trials={args.trials}, "
            f"available={len(initial_states)}"
        )
    base_config = CarveRunConfig.load(args.config)
    args.results_root.mkdir(parents=True, exist_ok=True)
    restore_snapshot = _load_restore_snapshot(args.restore_snapshot)
    if restore_snapshot is not None:
        metadata = restore_snapshot["metadata"]
        if int(metadata.get("task_id", -1)) != args.task_id:
            raise ValueError("restore snapshot task_id does not match --task-id")
        if str(metadata.get("instruction", "")).strip() != str(task.language).strip():
            raise ValueError("restore snapshot instruction does not match benchmark task")
    summaries = []
    for trial in range(args.trial_offset, args.trial_offset + args.trials):
        summaries.append(
            _run_episode(
                base_config=base_config,
                suite_name=args.suite,
                task_id=args.task_id,
                trial=trial,
                seed=args.seed,
                task=task,
                initial_state=initial_states[trial],
                restore_snapshot=restore_snapshot,
                args=args,
            )
        )
        print(json.dumps(summaries[-1], indent=2))
    aggregate = {
        "protocol": "agentic_vla_libero_pro_canonical_v1",
        "claim_boundary": "official LIBERO-PRO physics with private evaluator truth",
        "suite": args.suite,
        "task_id": args.task_id,
        "method": args.method,
        "restore_snapshot_id": (
            None
            if restore_snapshot is None
            else restore_snapshot["metadata_path"].stem
        ),
        "controlled_fault": (
            "stale_subgoal_v1" if args.stale_vla_instruction.strip() else None
        ),
        "task_start_planner": bool(args.task_start_planner),
        "semantic_checkpoint_steps": sorted(set(args.semantic_checkpoint_step)),
        "semantic_planner_only": bool(args.semantic_planner_only),
        "monitor_warmup_steps": int(args.monitor_warmup_steps),
        "stall_recovery_streak": int(args.stall_recovery_streak),
        "task_contract_instruction": str(task.language),
        "stale_vla_instruction": args.stale_vla_instruction.strip() or None,
        "trials": args.trials,
        "trial_offset": args.trial_offset,
        "resume_enabled": bool(args.resume),
        "successes": sum(bool(item["success"]) for item in summaries),
        "episodes": summaries,
    }
    _write_json(args.results_root / "summary.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
