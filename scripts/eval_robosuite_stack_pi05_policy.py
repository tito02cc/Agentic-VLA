#!/usr/bin/env python3
"""Run a short robosuite Stack closed-loop smoke with an OpenPI/pi0.5 policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.policies import policy_config
from openpi.training import config as openpi_config

from scripts.eval_robosuite_stack_policy import geometric_stack_retry_action
from scripts.robosuite_stack_policy import extract_image
from scripts.run_robosuite_deployment_pilot import get_progress, make_env
from scripts.convert_robosuite_stack_hdf5_to_lerobot import quat_to_axis_angle_wxyz


DEFAULT_POLICY_DIR = (
    "checkpoints/pi05_robosuite_stack_smoke/"
    "robosuite_stack_pi05_base_head_only_smoke_2step/2"
)
DEFAULT_PROMPT = "stack the red cube on the green cube"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


def obs_to_openpi_state(
    obs: dict[str, Any],
    *,
    state_mode: str = "proprio",
    step: int = 0,
    horizon: int = 1,
) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(1, 3)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(1, 4)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(1, 2)
    axis_angle = quat_to_axis_angle_wxyz(eef_quat)
    base_state = np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=-1)[0].astype(np.float32)
    if state_mode == "proprio":
        return base_state
    if state_mode == "privileged_stack":
        object_state = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
        if object_state.shape[0] != 23:
            raise ValueError(f"expected 23D object-state, got {object_state.shape}")
        denom = max(1, int(horizon) - 1)
        step_fraction = np.asarray([float(step) / float(denom)], dtype=np.float32)
        return np.concatenate([base_state, object_state, step_fraction], axis=-1).astype(np.float32)
    raise ValueError(f"unknown state_mode={state_mode!r}")


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


def make_noise(args: argparse.Namespace, call_index: int) -> np.ndarray | None:
    if args.noise_mode == "default":
        return None
    shape = (int(args.action_horizon), int(args.action_dim))
    if args.noise_mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(int(args.seed) + int(call_index) * 9973)
    return rng.standard_normal(shape).astype(np.float32)


def infer_chunk(
    policy,
    obs: dict[str, Any],
    *,
    camera_name: str,
    prompt: str,
    noise: np.ndarray | None,
    state_mode: str,
    step: int,
    horizon: int,
) -> tuple[np.ndarray, float]:
    image = extract_image(obs, camera_name=camera_name)
    payload = {
        "observation/image": image,
        "observation/wrist_image": image.copy(),
        "observation/state": obs_to_openpi_state(obs, state_mode=state_mode, step=step, horizon=horizon),
        "prompt": prompt,
    }
    start = time.perf_counter()
    output = policy.infer(payload, noise=noise)
    wall_ms = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(output["actions"], dtype=np.float32)
    return actions, float(output.get("policy_timing", {}).get("infer_ms", wall_ms))


def compact_obs_trace(obs: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eef_pos": np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float32),
        "gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32),
    }
    for key in ("cubeA_pos", "cubeB_pos", "cube_pos"):
        if key in obs:
            row[key] = np.asarray(obs[key], dtype=np.float32)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_robosuite_stack_smoke")
    parser.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--state-mode", choices=("proprio", "privileged_stack"), default="proprio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--noise-mode", choices=("default", "zero", "fixed"), default="default")
    parser.add_argument("--camera-name", default="frontview")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-dir", default="results/robosuite_stack_pi05_closed_loop_smoke_20260611")
    parser.add_argument("--max-action-abs", type=float, default=1.0)
    parser.add_argument("--save-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--retry-skill", choices=("none", "geometric_stack"), default="none")
    parser.add_argument("--min-recovery-step", type=int, default=60)
    parser.add_argument("--recovery-stall-steps", type=int, default=30)
    parser.add_argument("--post-recovery-lockout-steps", type=int, default=80)
    parser.add_argument("--max-recoveries-per-episode", type=int, default=1)
    parser.add_argument("--max-retry-skill-steps", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    frames: list[np.ndarray] = []
    infer_ms: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    full_policy_calls = 0
    actions_applied = 0
    success = False
    failure_reason = "horizon_reached"
    best_cube_z = 0.0
    last_improvement_step = 0
    final_progress: dict[str, Any] | None = None
    recovery_cooldown_until = 0
    recoveries_triggered = 0
    recoveries_successful = 0
    retry_active = False
    retry_step = 0
    retry_skill_steps = 0
    start = time.perf_counter()

    try:
        if hasattr(env, "seed"):
            try:
                env.seed(args.seed)
            except Exception:
                pass
        obs = env.reset()
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
            if not retry_active:
                chunk, chunk_infer_ms = infer_chunk(
                    policy,
                    obs,
                    camera_name=args.camera_name,
                    prompt=args.prompt,
                    noise=make_noise(args, full_policy_calls),
                    state_mode=args.state_mode,
                    step=step,
                    horizon=int(args.horizon),
                )
                full_policy_calls += 1
                infer_ms.append(chunk_infer_ms)
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
                raw_action = np.asarray(raw_action, dtype=np.float32).copy()
                action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
                action = np.clip(action, -float(args.max_action_abs), float(args.max_action_abs)).astype(np.float32)
                frames.append(extract_image(obs, camera_name=args.camera_name))
                obs_before_step = compact_obs_trace(obs)
                progress_before_step = get_progress("Stack", obs, best_cube_z)
                obs, _, _, _ = env.step(action)
                cube_z = float(np.asarray(obs.get("cubeA_pos", [0.0, 0.0, 0.0]), dtype=np.float32)[2])
                best_cube_z = max(best_cube_z, cube_z)
                final_progress = get_progress("Stack", obs, best_cube_z)
                if args.save_trace:
                    trace_rows.append(
                        {
                            "step": int(step),
                            "policy_call": int(full_policy_calls - 1),
                            "chunk_action_index": int(action_idx),
                            "chunk_infer_ms": float(chunk_infer_ms),
                            "raw_action": raw_action,
                            "applied_action": action,
                            "retry_active": bool(retry_active),
                            "retry_step": int(retry_step),
                            "recoveries_triggered": int(recoveries_triggered),
                            "obs_before": obs_before_step,
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
                f"[eval] step={step}/{args.horizon} calls={full_policy_calls} "
                f"last_infer_ms={chunk_infer_ms:.1f} "
                f"phase={progress_before_chunk.get('phase')} "
                f"retry={retry_active} rec={recoveries_successful}/{recoveries_triggered} "
                f"success={success}",
                flush=True,
            )
            if success:
                break
    finally:
        env.close()

    if frames:
        video_path = out_dir / "pi05_closed_loop_smoke.mp4"
        imageio.mimsave(video_path, frames, fps=20)
    else:
        video_path = None

    summary = {
        "policy_config": args.policy_config,
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "prompt": args.prompt,
        "state_mode": args.state_mode,
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "replan_steps": int(args.replan_steps),
        "num_inference_steps": int(args.num_inference_steps),
        "noise_mode": args.noise_mode,
        "agentic": bool(args.agentic),
        "retry_skill": args.retry_skill,
        "success": bool(success),
        "failure_reason": failure_reason,
        "steps": int(actions_applied),
        "full_policy_calls": int(full_policy_calls),
        "recoveries_triggered": int(recoveries_triggered),
        "recoveries_successful": int(recoveries_successful),
        "retry_skill_steps": int(retry_skill_steps),
        "policy_load_sec": policy_load_sec,
        "policy_infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
        "policy_infer_ms_p95": float(np.percentile(infer_ms, 95)) if infer_ms else None,
        "final_progress": final_progress,
        "best_cube_z": best_cube_z,
        "wall_sec": time.perf_counter() - start,
        "video_path": str(video_path) if video_path is not None else None,
        "trace_path": str(out_dir / "trace.json") if args.save_trace else None,
        "closed_loop_claim": "integration_smoke_only_not_success_rate_benchmark",
    }
    if args.save_trace:
        trace_path = out_dir / "trace.json"
        trace_path.write_text(json.dumps(trace_rows, indent=2, default=json_default), encoding="utf-8")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
