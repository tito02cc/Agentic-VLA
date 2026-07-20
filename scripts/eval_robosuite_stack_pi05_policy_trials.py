#!/usr/bin/env python3
"""Run matched multi-trial robosuite Stack evaluation for a pi0.5 policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.policies import policy_config
from openpi.training import config as openpi_config

from scripts.eval_robosuite_stack_pi05_policy import (
    DEFAULT_PROMPT,
    obs_to_openpi_state,
    json_default,
)
from scripts.eval_robosuite_stack_policy import geometric_stack_retry_action
from scripts.robosuite_stack_policy import extract_image
from scripts.run_robosuite_deployment_pilot import get_progress, make_env


DEFAULT_POLICY_DIR = "checkpoints/pi05_robosuite_stack_smoke/robosuite_stack_pi05_head_plus_300step/300"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_robosuite_stack_smoke")
    parser.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--state-mode", choices=("proprio", "privileged_stack"), default="proprio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=430)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--compute-policy", choices=("fixed", "phase_adaptive"), default="fixed")
    parser.add_argument("--adaptive-low-steps", type=int, default=1)
    parser.add_argument("--adaptive-high-steps", type=int, default=2)
    parser.add_argument(
        "--adaptive-high-risk-phases",
        default="grasp_or_lift,transport",
        help="Comma-separated progress phases that receive the high compute budget.",
    )
    parser.add_argument(
        "--adaptive-escalate-after-stall-steps",
        type=int,
        default=15,
        help="Use the high compute budget after this many steps without progress; negative disables escalation.",
    )
    parser.add_argument(
        "--inference-deadline-ms",
        type=float,
        default=200.0,
        help="Per-call soft inference deadline used only for runtime instrumentation.",
    )
    parser.add_argument("--noise-mode", choices=("default", "zero", "fixed"), default="fixed")
    parser.add_argument("--camera-name", default="frontview")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-tag", default="pi05")
    parser.add_argument("--max-action-abs", type=float, default=1.0)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--retry-skill", choices=("none", "geometric_stack"), default="none")
    parser.add_argument("--min-recovery-step", type=int, default=60)
    parser.add_argument("--recovery-stall-steps", type=int, default=30)
    parser.add_argument("--post-recovery-lockout-steps", type=int, default=80)
    parser.add_argument("--max-recoveries-per-episode", type=int, default=1)
    parser.add_argument("--max-retry-skill-steps", type=int, default=260)
    parser.add_argument("--save-traces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-first-success-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def load_policy(args: argparse.Namespace):
    cfg = openpi_config.get_config(args.policy_config)
    model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
    cfg = dataclasses.replace(cfg, model=model_cfg)
    return policy_config.create_trained_policy(
        cfg,
        Path(args.policy_dir).expanduser().resolve(),
        sample_kwargs={"num_steps": int(args.num_inference_steps)},
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )


def make_noise(args: argparse.Namespace, episode_seed: int, call_index: int) -> np.ndarray | None:
    if args.noise_mode == "default":
        return None
    shape = (int(args.action_horizon), int(args.action_dim))
    if args.noise_mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(int(episode_seed) + int(call_index) * 9973)
    return rng.standard_normal(shape).astype(np.float32)


def infer_chunk(
    policy: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
    *,
    episode_seed: int,
    call_index: int,
    step: int,
    num_inference_steps: int,
) -> tuple[np.ndarray, float]:
    image = extract_image(obs, camera_name=args.camera_name)
    payload = {
        "observation/image": image,
        "observation/wrist_image": image.copy(),
        "observation/state": obs_to_openpi_state(
            obs,
            state_mode=args.state_mode,
            step=step,
            horizon=int(args.horizon),
        ),
        "prompt": args.prompt,
    }
    start = time.perf_counter()
    policy_sample_kwargs = getattr(policy, "_sample_kwargs", None)
    if not isinstance(policy_sample_kwargs, dict):
        raise TypeError("Dynamic compute policy requires a local OpenPI Policy with mutable sample kwargs")
    previous_num_steps = policy_sample_kwargs.get("num_steps")
    policy_sample_kwargs["num_steps"] = int(num_inference_steps)
    try:
        output = policy.infer(payload, noise=make_noise(args, episode_seed, call_index))
    finally:
        if previous_num_steps is None:
            policy_sample_kwargs.pop("num_steps", None)
        else:
            policy_sample_kwargs["num_steps"] = previous_num_steps
    wall_ms = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(output["actions"], dtype=np.float32)
    return actions, float(output.get("policy_timing", {}).get("infer_ms", wall_ms))


def select_inference_steps(
    args: argparse.Namespace,
    progress: dict[str, Any],
    *,
    steps_since_improvement: int,
) -> int:
    if args.compute_policy == "fixed":
        return max(1, int(args.num_inference_steps))

    high_risk_phases = {
        phase.strip()
        for phase in str(args.adaptive_high_risk_phases).split(",")
        if phase.strip()
    }
    phase_is_high_risk = str(progress.get("phase", "")) in high_risk_phases
    stall_threshold = int(args.adaptive_escalate_after_stall_steps)
    stall_requires_compute = stall_threshold >= 0 and int(steps_since_improvement) >= stall_threshold
    if phase_is_high_risk or stall_requires_compute:
        return max(1, int(args.adaptive_high_steps))
    return max(1, int(args.adaptive_low_steps))


def compact_trace_row(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "eef_pos": np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float32),
        "cubeA_pos": np.asarray(obs.get("cubeA_pos", []), dtype=np.float32),
        "cubeB_pos": np.asarray(obs.get("cubeB_pos", []), dtype=np.float32),
        "gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32),
    }


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_episode(
    *,
    env: Any,
    policy: Any,
    args: argparse.Namespace,
    episode_id: int,
    episode_seed: int,
    record_video: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    if hasattr(env, "seed"):
        try:
            env.seed(episode_seed)
        except Exception:
            pass
    obs = env.reset()
    frames: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    inference_steps_used: list[int] = []
    inference_deadline_misses = 0
    full_policy_calls = 0
    actions_applied = 0
    success = False
    failure_reason = "horizon_reached"
    best_cube_z = float(np.asarray(obs.get("cubeA_pos", [0.0, 0.0, 0.0]), dtype=np.float32)[2])
    last_improvement_step = 0
    final_progress: dict[str, Any] | None = None
    recovery_cooldown_until = 0
    recoveries_triggered = 0
    recoveries_successful = 0
    retry_active = False
    retry_step = 0
    retry_skill_steps = 0
    start = time.perf_counter()

    step = 0
    while step < int(args.horizon):
        current_cube_z = float(np.asarray(obs.get("cubeA_pos", [0.0, 0.0, 0.0]), dtype=np.float32)[2])
        if current_cube_z > best_cube_z + 1e-5:
            last_improvement_step = step
        best_cube_z = max(best_cube_z, current_cube_z)
        progress_before_chunk = get_progress("Stack", obs, best_cube_z)

        can_trigger_recovery = (
            bool(args.agentic)
            and str(args.retry_skill) == "geometric_stack"
            and not retry_active
            and recoveries_triggered < int(args.max_recoveries_per_episode)
            and step >= int(args.min_recovery_step)
            and step >= recovery_cooldown_until
            and progress_before_chunk.get("phase") not in {"transport", "done"}
            and step - last_improvement_step >= int(args.recovery_stall_steps)
        )
        if can_trigger_recovery:
            recoveries_triggered += 1
            recovery_cooldown_until = step + int(args.post_recovery_lockout_steps)
            retry_active = True
            retry_step = 0

        chunk: np.ndarray | None = None
        chunk_infer_ms = 0.0
        selected_inference_steps: int | None = None
        if not retry_active:
            selected_inference_steps = select_inference_steps(
                args,
                progress_before_chunk,
                steps_since_improvement=step - last_improvement_step,
            )
            chunk, chunk_infer_ms = infer_chunk(
                policy,
                obs,
                args,
                episode_seed=episode_seed,
                call_index=full_policy_calls,
                step=step,
                num_inference_steps=selected_inference_steps,
            )
            full_policy_calls += 1
            infer_ms.append(chunk_infer_ms)
            inference_steps_used.append(selected_inference_steps)
            if float(args.inference_deadline_ms) > 0 and chunk_infer_ms > float(args.inference_deadline_ms):
                inference_deadline_misses += 1
            if chunk.ndim != 2 or chunk.shape[1] < 7:
                failure_reason = f"bad_action_shape_{chunk.shape}"
                break

        for action_idx in range(int(args.replan_steps)):
            if retry_active:
                raw_action = geometric_stack_retry_action(obs, retry_step)
                retry_step += 1
                retry_skill_steps += 1
            else:
                assert chunk is not None
                if action_idx >= chunk.shape[0]:
                    break
                raw_action = np.asarray(chunk[action_idx, :7], dtype=np.float32).copy()

            action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
            action = np.clip(action, -float(args.max_action_abs), float(args.max_action_abs)).astype(np.float32)
            if record_video:
                frames.append(extract_image(obs, camera_name=args.camera_name))
            progress_before_step = get_progress("Stack", obs, best_cube_z)
            obs_before = compact_trace_row(obs)
            obs, _, _, _ = env.step(action)
            cube_z = float(np.asarray(obs.get("cubeA_pos", [0.0, 0.0, 0.0]), dtype=np.float32)[2])
            best_cube_z = max(best_cube_z, cube_z)
            final_progress = get_progress("Stack", obs, best_cube_z)
            if args.save_traces:
                trace_rows.append(
                    {
                        "episode_id": int(episode_id),
                        "seed": int(episode_seed),
                        "step": int(step),
                        "policy_call": int(full_policy_calls - 1),
                        "chunk_action_index": int(action_idx),
                        "chunk_infer_ms": float(chunk_infer_ms),
                        "num_inference_steps": selected_inference_steps,
                        "retry_active": bool(retry_active),
                        "retry_step": int(retry_step),
                        "recoveries_triggered": int(recoveries_triggered),
                        "obs_before": obs_before,
                        "progress_before_step": progress_before_step,
                        "progress_after_step": final_progress,
                    }
                )
            actions_applied += 1
            step += 1
            if hasattr(env, "_check_success") and env._check_success():
                success = True
                failure_reason = ""
                break
            if retry_active:
                if final_progress.get("phase") in {"transport", "done"} and recoveries_successful < recoveries_triggered:
                    recoveries_successful = recoveries_triggered
                if retry_step >= int(args.max_retry_skill_steps):
                    retry_active = False
            if step >= int(args.horizon):
                break

        print(
            f"[trial] method={args.method_tag} ep={episode_id} step={step}/{args.horizon} "
            f"calls={full_policy_calls} phase={progress_before_chunk.get('phase')} "
            f"retry={retry_active} rec={recoveries_successful}/{recoveries_triggered} "
            f"success={success}",
            flush=True,
        )
        if success:
            break

    steady = infer_ms[1:]
    step_histogram = {
        str(value): inference_steps_used.count(value)
        for value in sorted(set(inference_steps_used))
    }
    row = {
        "method_tag": args.method_tag,
        "episode_id": int(episode_id),
        "seed": int(episode_seed),
        "success": bool(success),
        "failure_reason": failure_reason,
        "steps": int(actions_applied),
        "full_policy_calls": int(full_policy_calls),
        "compute_policy": args.compute_policy,
        "inference_steps_mean": mean([float(value) for value in inference_steps_used]),
        "inference_steps_histogram": step_histogram,
        "high_compute_calls": sum(int(value > int(args.adaptive_low_steps)) for value in inference_steps_used),
        "inference_deadline_misses": int(inference_deadline_misses),
        "inference_deadline_miss_rate": (
            inference_deadline_misses / len(inference_steps_used) if inference_steps_used else None
        ),
        "recoveries_triggered": int(recoveries_triggered),
        "recoveries_successful": int(recoveries_successful),
        "retry_skill_steps": int(retry_skill_steps),
        "policy_infer_ms_mean": mean(infer_ms),
        "policy_infer_ms_p95": pct(infer_ms, 95.0),
        "steady_policy_infer_ms_mean": mean(steady),
        "steady_policy_infer_ms_p95": pct(steady, 95.0),
        "final_progress": final_progress,
        "best_cube_z": float(best_cube_z),
        "wall_sec": time.perf_counter() - start,
    }
    return row, trace_rows, frames


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for row in rows if row["success"])
    return {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "steps_mean": mean([float(row["steps"]) for row in rows]),
        "full_policy_calls_mean": mean([float(row["full_policy_calls"]) for row in rows]),
        "inference_steps_mean": mean(
            [float(row["inference_steps_mean"]) for row in rows if row["inference_steps_mean"] is not None]
        ),
        "high_compute_calls_total": sum(int(row["high_compute_calls"]) for row in rows),
        "inference_deadline_misses_total": sum(int(row["inference_deadline_misses"]) for row in rows),
        "inference_deadline_miss_rate": (
            sum(int(row["inference_deadline_misses"]) for row in rows)
            / sum(int(row["full_policy_calls"]) for row in rows)
            if sum(int(row["full_policy_calls"]) for row in rows)
            else None
        ),
        "recoveries_triggered_total": sum(int(row["recoveries_triggered"]) for row in rows),
        "recoveries_successful_total": sum(int(row["recoveries_successful"]) for row in rows),
        "retry_skill_steps_mean": mean([float(row["retry_skill_steps"]) for row in rows]),
        "policy_infer_ms_mean": mean(
            [float(row["policy_infer_ms_mean"]) for row in rows if row["policy_infer_ms_mean"] is not None]
        ),
        "steady_policy_infer_ms_mean": mean(
            [float(row["steady_policy_infer_ms_mean"]) for row in rows if row["steady_policy_infer_ms_mean"] is not None]
        ),
        "steady_policy_infer_ms_p95_mean": mean(
            [float(row["steady_policy_infer_ms_p95"]) for row in rows if row["steady_policy_infer_ms_p95"] is not None]
        ),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    policy = load_policy(args)
    policy_load_sec = time.perf_counter() - load_start
    print(f"[policy] loaded_sec={policy_load_sec:.3f}")

    env = make_env(
        "Stack",
        int(args.horizon),
        record_video=True,
        camera_name=args.camera_name,
        camera_height=int(args.image_size),
        camera_width=int(args.image_size),
    )

    rows: list[dict[str, Any]] = []
    wrote_success_video = False
    traces_dir = out_dir / "traces"
    if args.save_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode_id in range(int(args.trials)):
            episode_seed = int(args.seed) + episode_id
            should_record = bool(args.save_first_success_video) and not wrote_success_video
            row, trace_rows, frames = run_episode(
                env=env,
                policy=policy,
                args=args,
                episode_id=episode_id,
                episode_seed=episode_seed,
                record_video=should_record,
            )
            if args.save_traces:
                (traces_dir / f"episode_{episode_id:03d}.json").write_text(
                    json.dumps(trace_rows, indent=2, default=json_default),
                    encoding="utf-8",
                )
            if should_record and row["success"] and frames:
                video_path = out_dir / f"episode_{episode_id:03d}_success.mp4"
                imageio.mimsave(video_path, frames, fps=int(args.video_fps))
                row["video_path"] = str(video_path)
                wrote_success_video = True
            rows.append(row)
            with (out_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=json_default) + "\n")
    finally:
        env.close()

    summary = {
        "method_tag": args.method_tag,
        "policy_config": args.policy_config,
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "prompt": args.prompt,
        "state_mode": args.state_mode,
        "seed": int(args.seed),
        "trials": int(args.trials),
        "horizon": int(args.horizon),
        "replan_steps": int(args.replan_steps),
        "num_inference_steps": int(args.num_inference_steps),
        "compute_policy": args.compute_policy,
        "adaptive_low_steps": int(args.adaptive_low_steps),
        "adaptive_high_steps": int(args.adaptive_high_steps),
        "adaptive_high_risk_phases": args.adaptive_high_risk_phases,
        "adaptive_escalate_after_stall_steps": int(args.adaptive_escalate_after_stall_steps),
        "inference_deadline_ms": float(args.inference_deadline_ms),
        "noise_mode": args.noise_mode,
        "agentic": bool(args.agentic),
        "retry_skill": args.retry_skill,
        "policy_load_sec": policy_load_sec,
        "aggregate": aggregate(rows),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        "episodes": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, default=json_default))
    print(f"[summary] {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
