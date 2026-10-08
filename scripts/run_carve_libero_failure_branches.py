#!/usr/bin/env python3
"""Run matched pi0.5 decisions from saved LIBERO failure snapshots."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import time
from typing import Any

import imageio
import numpy as np

from agentic_vla.runtime import (
    ActionSpec,
    AgentIntent,
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    OpenAICompatibleVisionPlanner,
    RecoveryContext,
    RecoveryMemory,
    StatefulRecoveryExecutor,
    build_cartesian_retreat_plan,
)
from scripts.run_agentic_vla_libero import (
    _build_policy_payload,
    _build_recovery_prompt,
    _create_client,
    _make_env,
    _require_runtime,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
BRANCHES = ("continue", "fast", "accurate", "recovery")
SUPPORTED_BRANCHES = (
    *BRANCHES,
    "task_contract",
    "physical_recovery",
    "vlm_physical_recovery",
    "vlm_replan",
)


def _state_digest(state: Any) -> str:
    quantized = np.rint(np.asarray(state, dtype=np.float64).reshape(-1) * 1e9).astype(np.int64)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def _prepare_images(observation, image_tools, resize_size: int):
    base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    base_policy = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(base, resize_size, resize_size)
    )
    wrist_policy = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    return base, base_policy, wrist_policy


def _execute_physical_recovery(
    *,
    env,
    observation,
    cached_actions: np.ndarray,
    last_action: np.ndarray | None,
    snapshot_id: str,
    trigger: str,
    minimum_state_response: float,
    remaining_horizon: int,
) -> tuple[Any, bool, int, list[np.ndarray], dict[str, Any]]:
    if cached_actions.ndim == 2 and cached_actions.shape[1] == 7 and len(cached_actions):
        recovery_reference_action = cached_actions[0]
    elif last_action is not None and np.asarray(last_action).shape == (7,):
        recovery_reference_action = np.asarray(last_action, dtype=np.float32)
    else:
        raise ValueError("physical recovery requires a cached or last executed 7-D action")
    action_spec = ActionSpec(
        action_dim=7,
        representation="normalized_delta_cartesian_pose",
        coordinate_frame="robot_base",
        gripper_convention="-1=close,+1=open",
        control_frequency_hz=20.0,
        minimum=-1.0,
        maximum=1.0,
    )
    plan = build_cartesian_retreat_plan(action_spec, recovery_reference_action)
    if plan.action_count > remaining_horizon:
        raise ValueError("branch horizon is shorter than the physical recovery plan")
    executor = StatefulRecoveryExecutor(RecoveryMemory(max_records=4))
    executor.start(
        plan,
        RecoveryContext(
            episode_id=f"branch:{snapshot_id}:physical_recovery",
            trigger_event=trigger,
            attempt=0,
        ),
    )
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    frames = []
    phase_records = []
    executed_steps = 0
    done = bool(env.check_success())
    outcome = None
    for _ in plan.phases:
        command = executor.next_command()
        phase_start = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        phase_steps = 0
        environment_terminated = False
        for action in command.actions:
            frames.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
            observation, _, environment_terminated, _ = env.step(list(action))
            executed_steps += 1
            phase_steps += 1
            done = bool(env.check_success())
            if environment_terminated:
                break
        phase_end = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        phase_records.append(
            {
                **command.to_dict(),
                "executed_actions": phase_steps,
                "eef_response": float(np.linalg.norm(phase_end - phase_start)),
                "environment_terminated": bool(environment_terminated),
            }
        )
        if environment_terminated and not command.request_verification:
            outcome = executor.abort(
                reason="environment terminated before physical recovery verification",
                safe_stop=not done,
                evidence={"task_success": done},
            )
            break
        if command.request_verification:
            total_response = float(np.linalg.norm(phase_end - initial_eef))
            verified = bool(
                np.isfinite(total_response)
                and total_response >= minimum_state_response
                and (not environment_terminated or done)
            )
            outcome = executor.resolve_verification(
                verified,
                evidence={
                    "eef_state_response": total_response,
                    "minimum_state_response": minimum_state_response,
                    "task_success": done,
                    "environment_terminated": bool(environment_terminated),
                },
            )
            break
    if outcome is None:
        raise RuntimeError("physical recovery plan ended without an outcome")
    record = {
        "skill_id": plan.skill_id,
        "planned_actions": plan.action_count,
        "phase_records": phase_records,
        "outcome": outcome.to_dict(),
    }
    return observation, done, executed_steps, frames, record


def _run_branch(
    *,
    env,
    client,
    image_tools,
    sim_state: np.ndarray,
    cached_actions: np.ndarray,
    last_action: np.ndarray | None,
    task_id: int,
    snapshot_id: str,
    trigger: str,
    instruction: str,
    execution_instruction: str,
    use_cached_prefix: bool,
    recovery_prompt: str,
    branch: str,
    horizon: int,
    resize_size: int,
    replan_steps: int,
    recovery_chunks: int,
    physical_minimum_state_response: float,
    fixed_inference_steps: int,
    deadline_ms: float,
    noise_seed: int,
    high_level_agent: GuardedHighLevelAgent | None,
    video_path: pathlib.Path | None,
) -> dict[str, Any]:
    env.reset()
    observation = env.regenerate_obs_from_state(sim_state.copy())
    if hasattr(client, "runtime"):
        client.runtime.reset(f"{snapshot_id}:{branch}")
    action_plan: collections.deque = collections.deque()
    if branch == "continue" and use_cached_prefix and cached_actions.ndim == 2:
        action_plan.extend(cached_actions.tolist())
    frames = []
    vla_calls = 0
    agentic_retry_calls = 0
    inference_ms = []
    applied_step_budgets = []
    deadline_misses = []
    optimization_profile = None
    physical_recovery = None
    physical_recovery_error = None
    high_level_result = None
    done = bool(env.check_success())
    environment_terminated = False
    started = time.perf_counter()
    executed_steps = 0
    noise_rng = np.random.default_rng(int(noise_seed))

    uses_physical_recovery = branch in {"physical_recovery", "vlm_physical_recovery"}
    if uses_physical_recovery and not done:
        try:
            observation, done, recovery_steps, recovery_frames, physical_recovery = (
                _execute_physical_recovery(
                    env=env,
                    observation=observation,
                    cached_actions=cached_actions,
                    last_action=last_action,
                    snapshot_id=snapshot_id,
                    trigger=trigger,
                    minimum_state_response=physical_minimum_state_response,
                    remaining_horizon=horizon,
                )
            )
            executed_steps += recovery_steps
            frames.extend(recovery_frames)
        except ValueError as error:
            physical_recovery_error = str(error)

    if (
        branch == "vlm_physical_recovery"
        and physical_recovery is not None
        and physical_recovery_error is None
        and physical_recovery["outcome"].get("request_replan", False)
    ):
        if high_level_agent is None:
            raise ValueError("vlm_physical_recovery requires a high-level planner")
        base, _, wrist = _prepare_images(observation, image_tools, resize_size)
        proprio = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(-1),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            )
        )
        high_level_result = high_level_agent.decide(
            HighLevelAgentContext(
                task_instruction=str(instruction),
                trigger="replan_after_physical_recovery",
                episode_id=f"branch:{snapshot_id}:{branch}",
                timestep=int(executed_steps),
                frames={"agentview": base, "wrist": wrist},
                robot_state=tuple(float(value) for value in proprio),
                risk={
                    "event": trigger,
                    "bucket": "high",
                    "score": 0.85,
                    "components": {"physical_recovery": 1.0},
                    "evidence": {
                        "recovery_verified": True,
                        "task_success": False,
                        "replan_required": True,
                    },
                },
                current_subgoal="task incomplete; choose next VLA subgoal",
                failure_history=(str(trigger),),
                memory=(
                    f"skill={physical_recovery['skill_id']};"
                    f"status={physical_recovery['outcome']['status']};"
                    "task_success=false;replan_required=true",
                ),
                available_skills=(),
                remaining_retries=1,
                remaining_recoveries=0,
                deadline_slack_ms=float(deadline_ms),
            )
        )
        if (
            high_level_result.accepted
            and high_level_result.decision.intent is AgentIntent.VLA_ACT
        ):
            recovery_prompt = str(high_level_result.decision.vla_instruction)
        elif (
            not high_level_result.accepted
            or high_level_result.decision.intent is AgentIntent.SAFE_STOP
        ):
            done = False
            physical_recovery_error = "guarded VLM rejected post-recovery replan"

    if branch == "vlm_replan" and not done:
        if high_level_agent is None:
            raise ValueError("vlm_replan requires a high-level planner")
        base, _, wrist = _prepare_images(observation, image_tools, resize_size)
        proprio = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(-1),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            )
        )
        high_level_result = high_level_agent.decide(
            HighLevelAgentContext(
                task_instruction=str(instruction),
                trigger=str(trigger),
                episode_id=f"branch:{snapshot_id}:{branch}",
                timestep=0,
                frames={"agentview": base, "wrist": wrist},
                robot_state=tuple(float(value) for value in proprio),
                risk={
                    "event": trigger,
                    "bucket": "high",
                    "score": 0.85,
                    "components": {"restored_failure": 1.0},
                    "evidence": {"task_success": False, "replan_required": True},
                },
                current_subgoal="task incomplete; choose next VLA subgoal",
                failure_history=(str(trigger),),
                memory=("restored failure state;task_success=false;replan_required=true",),
                available_skills=(),
                remaining_retries=1,
                remaining_recoveries=0,
                deadline_slack_ms=float(deadline_ms),
            )
        )
        if (
            high_level_result.accepted
            and high_level_result.decision.intent is AgentIntent.VLA_ACT
        ):
            recovery_prompt = str(high_level_result.decision.vla_instruction)
        elif (
            not high_level_result.accepted
            or high_level_result.decision.intent is AgentIntent.SAFE_STOP
        ):
            physical_recovery_error = "guarded VLM rejected restored-state replan"

    physical_allows_replan = bool(
        physical_recovery_error is None
        and (
            physical_recovery is None
            or physical_recovery["outcome"].get("request_replan", False)
        )
    )
    while executed_steps < horizon and not done and physical_allows_replan:
        base, base_policy, wrist_policy = _prepare_images(
            observation, image_tools, resize_size
        )
        frames.append(base)
        if not action_plan:
            use_recovery_prompt = branch == "recovery" and vla_calls < recovery_chunks
            use_vlm_recovery_prompt = (
                branch in {"vlm_physical_recovery", "vlm_replan"}
                and vla_calls == 0
                and high_level_result is not None
                and high_level_result.accepted
                and high_level_result.decision.intent is AgentIntent.VLA_ACT
            )
            default_execution_prompt = (
                instruction
                if branch in {"task_contract", "vlm_physical_recovery", "vlm_replan"}
                else execution_instruction
            )
            prompt = (
                recovery_prompt
                if use_recovery_prompt or use_vlm_recovery_prompt
                else default_execution_prompt
            )
            agentic_request = None
            if use_recovery_prompt:
                agentic_request = {
                    "event": trigger,
                    "action": "retry",
                    "reason": "paired_failure_state_branch",
                }
            elif uses_physical_recovery and vla_calls == 0:
                agentic_request = {
                    "event": trigger,
                    "action": "replan_after_physical_recovery",
                    "reason": physical_recovery["outcome"]["status"],
                    "skill_id": physical_recovery["skill_id"],
                }
            elif branch == "vlm_replan" and vla_calls == 0:
                agentic_request = {
                    "event": trigger,
                    "action": "planner_replan",
                    "reason": "guarded_vlm_replan",
                }
            requested_steps = (
                fixed_inference_steps
                if fixed_inference_steps > 0
                else 2
                if branch
                in {
                    "accurate",
                    "recovery",
                    "physical_recovery",
                    "vlm_physical_recovery",
                    "vlm_replan",
                }
                else 1
            )
            commit = (
                2
                if branch
                in {
                    "accurate",
                    "recovery",
                    "physical_recovery",
                    "vlm_physical_recovery",
                    "vlm_replan",
                }
                else replan_steps
            )
            payload = _build_policy_payload(
                obs=observation,
                base_img_p=base_policy,
                wrist_img_p=wrist_policy,
                prompt=prompt,
                episode_id=f"branch:{snapshot_id}:{branch}",
                timestep=executed_steps,
                agentic_req=agentic_request,
            )
            payload["runtime_controls"] = {
                "inference_steps": requested_steps,
                "max_actions": commit,
                "deadline_ms": deadline_ms,
            }
            payload["runtime_trace_context"] = {
                "experiment": "E3-failure-state-branch",
                "snapshot_id": snapshot_id,
                "branch": branch,
                "requested_steps": requested_steps,
            }
            payload["runtime_noise"] = noise_rng.standard_normal(
                (10, 32), dtype=np.float32
            )
            infer_started = time.perf_counter()
            response = client.infer(payload)
            inference_ms.append((time.perf_counter() - infer_started) * 1000.0)
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"invalid branch action shape {actions.shape}")
            action_plan.extend(actions[:commit].tolist())
            vla_calls += 1
            agentic_retry_calls += int(use_recovery_prompt)
            if hasattr(client, "runtime") and client.runtime.last_trace is not None:
                trace_metadata = client.runtime.last_trace.metadata
                applied_step_budgets.append(
                    client.runtime.last_trace.applied_controls.get("inference_steps")
                )
                deadline_misses.append(bool(client.runtime.last_trace.deadline_miss))
                if optimization_profile is None:
                    optimization_profile = trace_metadata.get("optimization_profile")

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        observation, _, environment_terminated, _ = env.step(action.tolist())
        executed_steps += 1
        done = bool(env.check_success())
        if environment_terminated and not done:
            break

    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=20)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    final_state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    return {
        "branch": branch,
        "success_within_horizon": bool(done),
        "environment_terminated": bool(environment_terminated),
        "executed_steps": executed_steps,
        "vla_calls": vla_calls,
        "agentic_retry_calls": agentic_retry_calls,
        "inference_ms_total": float(sum(inference_ms)),
        "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else 0.0,
        "branch_wall_ms": elapsed_ms,
        "applied_step_budgets": applied_step_budgets,
        "deadline_miss_count": int(sum(deadline_misses)),
        "deadline_call_count": len(deadline_misses),
        "optimization_profile": optimization_profile,
        "physical_recovery": physical_recovery,
        "physical_recovery_error": physical_recovery_error,
        "high_level_planner": (
            None
            if high_level_result is None
            else {
                "accepted": bool(high_level_result.accepted),
                "elapsed_ms": float(high_level_result.elapsed_ms),
                "error": high_level_result.error,
                "decision": high_level_result.decision.to_dict(),
            }
        ),
        "safe_stop": physical_recovery_error is not None,
        "used_cached_prefix": bool(
            branch == "continue" and use_cached_prefix and cached_actions.size > 0
        ),
        "video_path": str(video_path) if video_path is not None and frames else None,
        "final_state_digest": _state_digest(final_state),
        "execution_instruction": execution_instruction,
        "effective_initial_instruction": (
            instruction
            if branch in {"task_contract", "vlm_physical_recovery", "vlm_replan"}
            else execution_instruction
        ),
        "controlled_fault": (
            "semantic_stale_stage_v1"
            if branch != "task_contract" and execution_instruction != instruction
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-snapshots", type=int, default=15)
    parser.add_argument(
        "--snapshot-names",
        nargs="*",
        default=(),
        help="Exact snapshot stems to run; defaults to the first --max-snapshots files.",
    )
    parser.add_argument(
        "--branches",
        default=",".join(BRANCHES),
        help=(
            "Comma-separated subset of continue,fast,accurate,recovery,"
            "task_contract,physical_recovery,vlm_physical_recovery,vlm_replan."
        ),
    )
    parser.add_argument("--branch-horizon", type=int, default=80)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--recovery-chunks", type=int, default=2)
    parser.add_argument("--physical-minimum-state-response", type=float, default=1e-4)
    parser.add_argument(
        "--fixed-inference-steps",
        type=int,
        default=0,
        help="Use one prevalidated flow-step count for every branch; 0 keeps branch defaults.",
    )
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=7,
        help=(
            "Experiment-level seed mixed with each restored-state identifier to "
            "produce deterministic but independently repeatable VLA noise."
        ),
    )
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--vlm-endpoint", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=128)
    parser.add_argument(
        "--execution-instruction-map",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional JSON object mapping task IDs to a controlled stale low-level "
            "instruction. VLM branches retain the correct task contract and may repair it."
        ),
    )
    parser.add_argument(
        "--disable-cached-prefix",
        action="store_true",
        help="Start controlled semantic-fault branches from a fresh VLA query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.physical_minimum_state_response <= 0:
        raise ValueError("--physical-minimum-state-response must be positive")
    benchmark, get_libero_path, offscreen_env, _, image_tools, _ = _require_runtime()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    snapshot_dir = pathlib.Path(args.snapshot_dir)
    selected_branches = tuple(item.strip() for item in args.branches.split(",") if item.strip())
    unknown_branches = sorted(set(selected_branches) - set(SUPPORTED_BRANCHES))
    if not selected_branches or unknown_branches:
        raise ValueError(f"invalid --branches selection: {unknown_branches or selected_branches}")
    use_vlm_branch = bool(
        {"vlm_physical_recovery", "vlm_replan"} & set(selected_branches)
    )
    if use_vlm_branch and (not args.vlm_endpoint or not args.vlm_model):
        raise ValueError(
            "--vlm-endpoint and --vlm-model are required by vlm_physical_recovery"
        )
    if args.vlm_timeout_sec <= 0 or args.vlm_max_tokens <= 0:
        raise ValueError("VLM timeout and max tokens must be positive")
    execution_instruction_map: dict[str, str] = {}
    if args.execution_instruction_map is not None:
        value = json.loads(args.execution_instruction_map.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) and item.strip()
            for key, item in value.items()
        ):
            raise ValueError("execution-instruction-map must be a JSON string map")
        execution_instruction_map = {key: item.strip() for key, item in value.items()}
    if args.snapshot_names:
        metadata_paths = [snapshot_dir / f"{pathlib.Path(name).stem}.json" for name in args.snapshot_names]
        missing = [str(path) for path in metadata_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing requested snapshots: {missing}")
    else:
        metadata_paths = sorted(snapshot_dir.glob("*.json"))[: args.max_snapshots]
    if not metadata_paths:
        raise FileNotFoundError(f"no failure snapshots in {args.snapshot_dir}")
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_name(f"{output_path.stem}_policy_calls.jsonl")
    trace_path.unlink(missing_ok=True)
    client = _create_client(
        args.host,
        args.port,
        carve_runtime=True,
        carve_trace_jsonl=str(trace_path),
    )
    high_level_agent = (
        GuardedHighLevelAgent(
            OpenAICompatibleVisionPlanner(
                endpoint=str(args.vlm_endpoint),
                model=str(args.vlm_model),
                timeout_s=float(args.vlm_timeout_sec),
                max_tokens=int(args.vlm_max_tokens),
            ),
            HighLevelAgentConfig(max_calls_per_episode=1),
        )
        if use_vlm_branch
        else None
    )
    records = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(metadata_path.with_name(metadata["array_file"]))
        task_id = int(metadata["task_id"])
        task = suite.get_task(task_id)
        env = _make_env(task, get_libero_path, offscreen_env, seed=int(args.noise_seed))
        try:
            env.reset()
            recovery_prompt = _build_recovery_prompt(metadata["instruction"], {})
            branch_results = []
            for branch in selected_branches:
                video_path = (
                    output_path.parent
                    / f"{output_path.stem}_videos"
                    / f"{metadata_path.stem}_{branch}.mp4"
                    if args.save_videos
                    else None
                )
                branch_results.append(
                    _run_branch(
                        env=env,
                        client=client,
                        image_tools=image_tools,
                        sim_state=np.asarray(arrays["sim_state"], dtype=np.float64),
                        cached_actions=np.asarray(arrays["cached_actions"], dtype=np.float32),
                        last_action=(
                            np.asarray(arrays["last_action"], dtype=np.float32)
                            if "last_action" in arrays
                            else None
                        ),
                        task_id=task_id,
                        snapshot_id=metadata_path.stem,
                        trigger=str(metadata["trigger"]),
                        instruction=metadata["instruction"],
                        execution_instruction=execution_instruction_map.get(
                            str(task_id), metadata["instruction"]
                        ),
                        use_cached_prefix=not args.disable_cached_prefix,
                        recovery_prompt=recovery_prompt,
                        branch=branch,
                        horizon=int(args.branch_horizon),
                        resize_size=int(args.resize_size),
                        replan_steps=int(args.replan_steps),
                        recovery_chunks=int(args.recovery_chunks),
                        physical_minimum_state_response=float(
                            args.physical_minimum_state_response
                        ),
                        fixed_inference_steps=int(args.fixed_inference_steps),
                        deadline_ms=float(args.deadline_ms),
                        noise_seed=int.from_bytes(
                            hashlib.sha256(
                                f"{args.noise_seed}:{metadata_path.stem}".encode("utf-8")
                        ).digest()[:8],
                            byteorder="little",
                            signed=False,
                        ),
                        high_level_agent=high_level_agent,
                        video_path=video_path,
                    )
                )
            successful = [item["branch"] for item in branch_results if item["success_within_horizon"]]
            records.append(
                {
                    "snapshot": metadata_path.stem,
                    "task_id": task_id,
                    "trigger": metadata["trigger"],
                    "successful_branches": successful,
                    "oracle_success": bool(successful),
                    "branches": branch_results,
                }
            )
            output_path.write_text(
                json.dumps(
                    {
                        "experiment": "E3-failure-state-branch-pilot",
                        "status": "running",
                        "snapshot_count": len(records),
                        "branches": list(selected_branches),
                        "branch_horizon": int(args.branch_horizon),
                        "fixed_inference_steps": int(args.fixed_inference_steps),
                        "deadline_ms": float(args.deadline_ms),
                        "noise_seed": int(args.noise_seed),
                        "policy_call_trace": str(trace_path),
                        "records": records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            env.close()
            arrays.close()

    payload = {
        "experiment": "E3-failure-state-branch-pilot",
        "status": "completed",
        "snapshot_count": len(records),
        "branches": list(selected_branches),
        "branch_horizon": int(args.branch_horizon),
        "fixed_inference_steps": int(args.fixed_inference_steps),
        "deadline_ms": float(args.deadline_ms),
        "policy_call_trace": str(trace_path),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
