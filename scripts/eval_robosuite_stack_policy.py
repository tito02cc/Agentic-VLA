#!/usr/bin/env python3
"""Evaluate a trained robosuite Stack action-chunk policy in real MuJoCo simulation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robosuite_stack_policy import (
    NormalizationStats,
    StackChunkPolicyNet,
    denormalize_actions,
    extract_image,
    extract_state,
    make_step_feature,
    prepare_image_tensor,
    prepare_state_tensor,
)
from scripts.run_robosuite_deployment_pilot import get_progress, make_env, should_trigger_recovery, try_apply_nudge


@dataclass
class EvalConfig:
    method_tag: str
    agentic: bool
    light: bool
    reuse_max_actions: int
    control_deadline_ms: float
    post_perturbation_lockout_steps: int
    post_recovery_lockout_steps: int
    recovery_stall_steps: int
    max_recoveries_per_episode: int
    min_recovery_step: int
    retry_skill: str
    max_retry_skill_steps: int


@dataclass
class EvalTrace:
    method_tag: str
    seed: int
    episode_id: int
    success: bool = False
    steps: int = 0
    full_policy_calls: int = 0
    reused_actions: int = 0
    control_steps: int = 0
    deadline_misses: int = 0
    policy_latency_ms: list[float] = field(default_factory=list)
    recoveries_triggered: int = 0
    recoveries_successful: int = 0
    retry_skill_steps: int = 0
    perturbation_event: dict[str, Any] | None = None
    final_progress: dict[str, Any] | None = None
    best_cube_z: float = 0.0
    wall_sec: float = 0.0
    failure_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        miss = self.deadline_misses / self.control_steps if self.control_steps else None
        return {
            "method_tag": self.method_tag,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "success": self.success,
            "steps": self.steps,
            "full_policy_calls": self.full_policy_calls,
            "reused_actions": self.reused_actions,
            "control_steps": self.control_steps,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": miss,
            "policy_latency_ms_mean": float(np.mean(self.policy_latency_ms)) if self.policy_latency_ms else None,
            "policy_latency_ms_p95": float(np.percentile(self.policy_latency_ms, 95)) if self.policy_latency_ms else None,
            "recoveries_triggered": self.recoveries_triggered,
            "recoveries_successful": self.recoveries_successful,
            "retry_skill_steps": self.retry_skill_steps,
            "perturbation_event": self.perturbation_event,
            "final_progress": self.final_progress,
            "best_cube_z": self.best_cube_z,
            "wall_sec": self.wall_sec,
            "failure_reason": self.failure_reason,
        }


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[StackChunkPolicyNet, NormalizationStats]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    stats = NormalizationStats.from_dict(ckpt["stats"])
    model = StackChunkPolicyNet(
        state_dim=int(ckpt["state_dim"]),
        chunk_size=int(ckpt["chunk_size"]),
        action_dim=int(ckpt["action_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stats


@torch.no_grad()
def infer_chunk(
    *,
    model: StackChunkPolicyNet,
    stats: NormalizationStats,
    obs: dict[str, Any],
    step: int,
    horizon: int,
    camera_name: str,
    device: torch.device,
) -> np.ndarray:
    image = extract_image(obs, camera_name=camera_name)
    state = np.concatenate([extract_state(obs), make_step_feature(step, horizon)], axis=0)
    image_t = prepare_image_tensor(image, device)
    state_t = prepare_state_tensor(state, stats, device)
    pred = model(image_t, state_t).detach().cpu().numpy()[0]
    chunk = denormalize_actions(pred, stats)
    chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
    chunk[:, 3:6] = 0.0
    chunk[:, 6] = np.where(chunk[:, 6] >= 0.0, 1.0, -1.0)
    return chunk


def geometric_stack_retry_action(obs: dict[str, Any], retry_step: int) -> np.ndarray:
    """Conservative grasp-and-stack retry skill used by the Agentic critic."""
    cube_a = np.asarray(obs["cubeA_pos"], dtype=np.float64)
    cube_b = np.asarray(obs["cubeB_pos"], dtype=np.float64)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    action = np.zeros(7, dtype=np.float32)

    if cube_a[2] < 0.90:
        # If the first grasp misses, cycle through open-descend-close-lift again.
        cycle = retry_step if retry_step < 150 else (retry_step - 150) % 140
        if cycle < 35:
            target = cube_a + np.array([0.0, 0.0, 0.14])
            grip = -1.0
        elif cycle < 80:
            target = cube_a + np.array([0.0, 0.0, 0.002])
            grip = -1.0
        elif cycle < 115:
            target = cube_a + np.array([0.0, 0.0, 0.0])
            grip = 1.0
        else:
            target = np.array([cube_a[0], cube_a[1], 1.12])
            grip = 1.0
    else:
        xy_to_stack = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
        place_z = float(cube_b[2] + 0.058)
        if xy_to_stack > 0.018:
            target = np.array([cube_b[0], cube_b[1], 1.12])
            grip = 1.0
        elif eef[2] > place_z + 0.012:
            target = np.array([cube_b[0], cube_b[1], place_z])
            grip = 1.0
        elif cube_a[2] - cube_b[2] > 0.070:
            target = np.array([cube_b[0], cube_b[1], place_z])
            grip = 1.0
        elif xy_to_stack <= 0.026:
            target = np.array([cube_b[0], cube_b[1], cube_b[2] + 0.055])
            grip = -1.0
        else:
            target = np.array([cube_b[0], cube_b[1], place_z])
            grip = 1.0

    action[:3] = np.clip((target - eef) * 10.0, -1.0, 1.0)
    action[3:6] = 0.0
    action[6] = grip
    return action


def run_episode(
    *,
    env: Any,
    model: StackChunkPolicyNet,
    stats: NormalizationStats,
    cfg: EvalConfig,
    seed: int,
    episode_id: int,
    horizon: int,
    perturb_step: int,
    perturb_xy: float,
    camera_name: str,
    device: torch.device,
    video_writer: Any | None,
    video_render_every: int,
) -> EvalTrace:
    rng = random.Random(seed * 1009 + episode_id)
    np.random.seed(seed * 1009 + episode_id)
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass
    obs = env.reset()
    trace = EvalTrace(method_tag=cfg.method_tag, seed=seed, episode_id=episode_id)
    action_buffer: list[np.ndarray] = []
    reuse_lockout_until = 0
    recovery_cooldown_until = 0
    retry_active = False
    retry_step = 0
    perturbation_step: int | None = None
    best_cube_z = float(obs["cubeA_pos"][2])
    last_improvement_step = 0
    start_time = time.perf_counter()

    def maybe_write_frame(step_idx: int, current_obs: dict[str, Any]) -> None:
        if video_writer is None or step_idx % max(1, int(video_render_every)) != 0:
            return
        video_writer.append_data(extract_image(current_obs, camera_name=camera_name))

    maybe_write_frame(0, obs)

    for step in range(horizon):
        loop_start = time.perf_counter()
        current_cube_z = float(obs["cubeA_pos"][2])
        if current_cube_z > best_cube_z + 1e-5:
            last_improvement_step = step
        best_cube_z = max(best_cube_z, current_cube_z)
        progress = get_progress("Stack", obs, best_cube_z)
        trace.final_progress = progress
        trace.best_cube_z = best_cube_z

        if perturb_xy > 0 and perturbation_step is None and step == perturb_step:
            event = try_apply_nudge(env, "Stack", rng, perturb_xy)
            trace.perturbation_event = event
            perturbation_step = step
            if event.get("applied"):
                obs = env._get_observations()
            if cfg.light:
                reuse_lockout_until = max(reuse_lockout_until, step + cfg.post_perturbation_lockout_steps)
            action_buffer.clear()

        # Reuse the same recovery predicate as the runtime pilot via a minimal adapter.
        recovery_adapter = type(
            "RecoveryAdapter",
            (),
            {
                "agentic": cfg.agentic,
                "max_recoveries_per_episode": cfg.max_recoveries_per_episode,
                "recovery_stall_steps": cfg.recovery_stall_steps,
            },
        )()
        trace_adapter = type("TraceAdapter", (), {"recoveries_triggered": trace.recoveries_triggered})()
        can_trigger_recovery = step >= cfg.min_recovery_step
        if can_trigger_recovery and should_trigger_recovery(
            cfg=recovery_adapter,
            trace=trace_adapter,
            progress=progress,
            last_improvement_step=last_improvement_step,
            step=step,
            perturbation_step=perturbation_step,
            recovery_cooldown_until=recovery_cooldown_until,
        ):
            trace.recoveries_triggered += 1
            recovery_cooldown_until = step + cfg.post_recovery_lockout_steps
            reuse_lockout_until = max(reuse_lockout_until, step + cfg.post_recovery_lockout_steps)
            action_buffer.clear()
            if cfg.retry_skill == "geometric_stack":
                retry_active = True
                retry_step = 0

        if retry_active and cfg.retry_skill == "geometric_stack":
            action = geometric_stack_retry_action(obs, retry_step)
            retry_step += 1
            trace.retry_skill_steps += 1
            action_buffer.clear()
            if retry_step >= cfg.max_retry_skill_steps or progress["phase"] == "done":
                retry_active = False
        else:
            can_reuse = (
            cfg.light
            and action_buffer
            and step >= reuse_lockout_until
            and trace.reused_actions < cfg.reuse_max_actions * max(1, trace.full_policy_calls)
            )
            if can_reuse:
                action = action_buffer.pop(0)
                trace.reused_actions += 1
            else:
                call_start = time.perf_counter()
                chunk = infer_chunk(
                    model=model,
                    stats=stats,
                    obs=obs,
                    step=step,
                    horizon=horizon,
                    camera_name=camera_name,
                    device=device,
                )
                trace.policy_latency_ms.append((time.perf_counter() - call_start) * 1000.0)
                trace.full_policy_calls += 1
                action = chunk[0]
                action_buffer = list(chunk[1 : 1 + cfg.reuse_max_actions]) if cfg.light else []

        obs, reward, done, info = env.step(action)
        maybe_write_frame(step + 1, obs)
        trace.steps = step + 1
        trace.control_steps += 1
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        if loop_ms > cfg.control_deadline_ms:
            trace.deadline_misses += 1

        if trace.recoveries_triggered > trace.recoveries_successful:
            if get_progress("Stack", obs, best_cube_z)["phase"] in {"grasp_or_lift", "transport", "done"}:
                trace.recoveries_successful = trace.recoveries_triggered

        if hasattr(env, "_check_success") and env._check_success():
            trace.success = True
            break

    trace.wall_sec = time.perf_counter() - start_time
    if not trace.success:
        trace.failure_reason = str(trace.final_progress.get("phase", "unknown")) if trace.final_progress else "unknown"
    return trace


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    successes = sum(1 for r in rows if r["success"])
    full = [float(r["full_policy_calls"]) for r in rows]
    reuse = [float(r["reused_actions"]) for r in rows]
    miss = [float(r["deadline_miss_rate"]) for r in rows if r["deadline_miss_rate"] is not None]
    lat = [float(r["policy_latency_ms_mean"]) for r in rows if r["policy_latency_ms_mean"] is not None]
    lat95 = [float(r["policy_latency_ms_p95"]) for r in rows if r["policy_latency_ms_p95"] is not None]
    rec_t = sum(int(r["recoveries_triggered"]) for r in rows)
    rec_s = sum(int(r["recoveries_successful"]) for r in rows)
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "full_policy_calls_per_episode": float(np.mean(full)) if full else None,
        "reused_actions_per_episode": float(np.mean(reuse)) if reuse else None,
        "reuse_ratio": float(sum(reuse) / (sum(full) + sum(reuse))) if (sum(full) + sum(reuse)) else None,
        "deadline_miss_rate_mean": float(np.mean(miss)) if miss else None,
        "policy_latency_ms_mean": float(np.mean(lat)) if lat else None,
        "policy_latency_ms_p95_mean": float(np.mean(lat95)) if lat95 else None,
        "recoveries_triggered_total": rec_t,
        "recoveries_successful_total": rec_s,
        "recovery_precision": rec_s / rec_t if rec_t else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--horizon", type=int, default=430)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", default="frontview")
    parser.add_argument("--method-tag", default="RS-Learned-Agentic-Light")
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--reuse-max-actions", type=int, default=2)
    parser.add_argument("--control-deadline-ms", type=float, default=80.0)
    parser.add_argument("--post-perturbation-lockout-steps", type=int, default=40)
    parser.add_argument("--post-recovery-lockout-steps", type=int, default=40)
    parser.add_argument("--recovery-stall-steps", type=int, default=25)
    parser.add_argument("--max-recoveries-per-episode", type=int, default=1)
    parser.add_argument("--min-recovery-step", type=int, default=120)
    parser.add_argument("--retry-skill", default="none", choices=["none", "geometric_stack"])
    parser.add_argument("--max-retry-skill-steps", type=int, default=260)
    parser.add_argument("--perturb-step", type=int, default=260)
    parser.add_argument("--perturb-xy", type=float, default=0.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-render-every", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_json)
    trace_path = Path(args.trace_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video_path) if args.video_path else results_path.with_suffix(".mp4")
    if args.record_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_policy(Path(args.checkpoint), device)
    cfg = EvalConfig(
        method_tag=str(args.method_tag),
        agentic=bool(args.agentic),
        light=bool(args.light),
        reuse_max_actions=int(args.reuse_max_actions),
        control_deadline_ms=float(args.control_deadline_ms),
        post_perturbation_lockout_steps=int(args.post_perturbation_lockout_steps),
        post_recovery_lockout_steps=int(args.post_recovery_lockout_steps),
        recovery_stall_steps=int(args.recovery_stall_steps),
        max_recoveries_per_episode=int(args.max_recoveries_per_episode),
        min_recovery_step=int(args.min_recovery_step),
        retry_skill=str(args.retry_skill),
        max_retry_skill_steps=int(args.max_retry_skill_steps),
    )

    env = make_env(
        "Stack",
        int(args.horizon),
        record_video=True,
        camera_name=str(args.camera_name),
        camera_height=int(args.image_size),
        camera_width=int(args.image_size),
    )
    video_writer = None
    if args.record_video:
        import imageio.v2 as imageio

        video_writer = imageio.get_writer(str(video_path), fps=int(args.video_fps), macro_block_size=1)

    traces: list[EvalTrace] = []
    try:
        with trace_path.open("w", encoding="utf-8") as f:
            for episode_id in range(int(args.trials)):
                ep_seed = int(args.seed) + episode_id
                trace = run_episode(
                    env=env,
                    model=model,
                    stats=stats,
                    cfg=cfg,
                    seed=ep_seed,
                    episode_id=episode_id,
                    horizon=int(args.horizon),
                    perturb_step=int(args.perturb_step),
                    perturb_xy=float(args.perturb_xy),
                    camera_name=str(args.camera_name),
                    device=device,
                    video_writer=video_writer if episode_id == 0 else None,
                    video_render_every=int(args.video_render_every),
                )
                traces.append(trace)
                row = trace.to_json()
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"[eval] ep={episode_id} success={trace.success} steps={trace.steps} "
                    f"full={trace.full_policy_calls} reuse={trace.reused_actions} "
                    f"miss={row['deadline_miss_rate']:.3f} rec={trace.recoveries_successful}/{trace.recoveries_triggered}",
                    flush=True,
                )
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    rows = [t.to_json() for t in traces]
    summary = {
        "benchmark": "robosuite_stack_learned_policy",
        "checkpoint": str(args.checkpoint),
        "method_tag": str(args.method_tag),
        "trials": int(args.trials),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "image_size": int(args.image_size),
        "perturb_xy": float(args.perturb_xy),
        "runtime_config": cfg.__dict__,
        "aggregate": aggregate(rows),
        "trace_jsonl": str(trace_path),
        "video_path": str(video_path) if args.record_video else None,
    }
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
