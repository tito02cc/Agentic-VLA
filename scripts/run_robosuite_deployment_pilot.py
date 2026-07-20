#!/usr/bin/env python3
"""Run a robosuite/MuJoCo deployment pilot for Agentic-VLA runtime logic.

This is a real-physics simulator deployment smoke benchmark, not a VLA SOTA
benchmark.  It uses a scripted action-chunk backend to verify that the
Agentic/CAQ-Lite runtime loop, perturbation handling, recovery lockout,
realtime metrics, and video recording can run outside the LIBERO runner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _safe_p95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


@dataclass
class RuntimeConfig:
    method_tag: str
    agentic: bool = False
    light: bool = False
    replan_steps: int = 5
    chunk_size: int = 5
    reuse_max_actions: int = 2
    control_deadline_ms: float = 80.0
    simulated_full_call_latency_ms: float = 0.0
    post_perturbation_lockout_steps: int = 40
    post_recovery_lockout_steps: int = 40
    recovery_stall_steps: int = 25
    max_recoveries_per_episode: int = 1


@dataclass
class EpisodeTrace:
    method_tag: str
    task: str
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
    perturbation_event: dict[str, Any] | None = None
    final_progress: dict[str, Any] | None = None
    best_cube_z: float = 0.0
    wall_sec: float = 0.0
    failure_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        miss = self.deadline_misses / self.control_steps if self.control_steps else None
        return {
            "method_tag": self.method_tag,
            "task": self.task,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "success": self.success,
            "steps": self.steps,
            "full_policy_calls": self.full_policy_calls,
            "reused_actions": self.reused_actions,
            "control_steps": self.control_steps,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": miss,
            "success_under_deadline": (1.0 if self.success else 0.0) * (1.0 - miss)
            if miss is not None
            else None,
            "policy_latency_ms_mean": _safe_mean(self.policy_latency_ms),
            "policy_latency_ms_p95": _safe_p95(self.policy_latency_ms),
            "recoveries_triggered": self.recoveries_triggered,
            "recoveries_successful": self.recoveries_successful,
            "perturbation_event": self.perturbation_event,
            "final_progress": self.final_progress,
            "best_cube_z": self.best_cube_z,
            "wall_sec": self.wall_sec,
            "failure_reason": self.failure_reason,
        }


class ScriptedChunkPolicy:
    """Small action-chunk policy for robosuite Lift/Stack.

    The policy mimics a frozen action generator interface: a full call observes
    state and returns a chunk of 7D OSC_POSE actions.
    """

    def __init__(self, task: str, chunk_size: int = 5):
        self.task = str(task)
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _lift_phase(obs: dict[str, Any], step: int) -> str:
        cube = np.asarray(obs["cube_pos"], dtype=np.float64)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        xy_dist = float(np.linalg.norm(eef[:2] - cube[:2]))
        z_err = float(abs(eef[2] - (cube[2] + 0.025)))
        if cube[2] > 0.91:
            return "lifted"
        if step < 35:
            return "approach_above"
        if xy_dist < 0.025 and z_err < 0.035:
            return "grasp_or_lift"
        return "approach_grasp"

    @staticmethod
    def _lift_action(obs: dict[str, Any], step: int, recovery_boost: bool = False) -> np.ndarray:
        cube = np.asarray(obs["cube_pos"], dtype=np.float64)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        phase = ScriptedChunkPolicy._lift_phase(obs, step)
        action = np.zeros(7, dtype=np.float64)

        if phase == "approach_above":
            target = cube + np.array([0.0, 0.0, 0.16])
            grip = -1.0
        elif phase == "approach_grasp":
            target = cube + np.array([0.0, 0.0, 0.005 if not recovery_boost else 0.0])
            grip = 1.0 if step > 70 else -1.0
        elif phase == "grasp_or_lift":
            target = np.array([cube[0], cube[1], 1.10 if recovery_boost else 1.05])
            grip = 1.0
        else:
            target = np.array([cube[0], cube[1], 1.08])
            grip = 1.0

        # OSC_POSE position command is normalized; keep rotations zero.
        action[:3] = np.clip((target - eef) * 10.0, -1.0, 1.0)
        action[6] = grip
        return action

    @staticmethod
    def _stack_action(obs: dict[str, Any], step: int, recovery_boost: bool = False) -> np.ndarray:
        cube_a = np.asarray(obs["cubeA_pos"], dtype=np.float64)
        cube_b = np.asarray(obs["cubeB_pos"], dtype=np.float64)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float64)

        if step < 50:
            target = cube_a + np.array([0.0, 0.0, 0.16])
            grip = -1.0
        elif step < 105:
            target = cube_a + np.array([0.0, 0.0, 0.005 if not recovery_boost else 0.0])
            grip = -1.0
        elif step < 140:
            target = cube_a + np.array([0.0, 0.0, 0.0])
            grip = 1.0
        elif step < 210:
            target = np.array([cube_a[0], cube_a[1], 1.12])
            grip = 1.0
        elif step < 280:
            target = np.array([cube_b[0], cube_b[1], 1.13])
            grip = 1.0
        elif step < 335:
            place_z = cube_b[2] + (0.055 if not recovery_boost else 0.058)
            target = np.array([cube_b[0], cube_b[1], place_z])
            grip = 1.0
        elif step < 370:
            target = np.array([cube_b[0], cube_b[1], cube_b[2] + 0.055])
            grip = -1.0
        else:
            target = np.array([cube_b[0], cube_b[1], 1.15])
            grip = -1.0

        action[:3] = np.clip((target - eef) * 10.0, -1.0, 1.0)
        action[6] = grip
        return action

    def _single_action(self, obs: dict[str, Any], step: int, recovery_boost: bool = False) -> np.ndarray:
        if self.task == "Stack":
            return self._stack_action(obs, step, recovery_boost=recovery_boost)
        return self._lift_action(obs, step, recovery_boost=recovery_boost)

    def infer_chunk(self, obs: dict[str, Any], step: int, recovery_boost: bool = False) -> list[np.ndarray]:
        chunk: list[np.ndarray] = []
        fake_obs = dict(obs)
        for i in range(self.chunk_size):
            chunk.append(self._single_action(fake_obs, step + i, recovery_boost=recovery_boost))
        return chunk


def make_env(
    task: str,
    horizon: int,
    *,
    record_video: bool = False,
    camera_name: str = "frontview",
    camera_height: int = 256,
    camera_width: int = 256,
):
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller_config = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        task,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=bool(record_video),
        use_camera_obs=bool(record_video),
        camera_names=camera_name,
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        render_camera=camera_name,
        ignore_done=True,
        horizon=horizon,
        control_freq=20,
        hard_reset=False,
    )


def get_lift_progress(obs: dict[str, Any], best_cube_z: float) -> dict[str, Any]:
    cube = np.asarray(obs["cube_pos"], dtype=np.float64)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    xy_dist = float(np.linalg.norm(eef[:2] - cube[:2]))
    z_lift = float(cube[2])
    if z_lift > 0.91:
        phase = "done"
    elif xy_dist < 0.03 and abs(eef[2] - (cube[2] + 0.025)) < 0.04:
        phase = "grasp_or_lift"
    elif xy_dist < 0.08:
        phase = "approach_grasp"
    else:
        phase = "approach_above"
    return {
        "phase": phase,
        "cube_z": z_lift,
        "best_cube_z": float(best_cube_z),
        "eef_cube_xy_dist": xy_dist,
        "eef_z": float(eef[2]),
    }


def get_stack_progress(obs: dict[str, Any], best_cube_z: float) -> dict[str, Any]:
    cube_a = np.asarray(obs["cubeA_pos"], dtype=np.float64)
    cube_b = np.asarray(obs["cubeB_pos"], dtype=np.float64)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    xy_ab = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
    z_gap = float(cube_a[2] - cube_b[2])
    eef_a_xy = float(np.linalg.norm(eef[:2] - cube_a[:2]))
    if xy_ab < 0.025 and 0.035 <= z_gap <= 0.075:
        phase = "done"
    elif cube_a[2] > 0.96 and eef_a_xy < 0.07:
        phase = "transport"
    elif eef_a_xy < 0.04:
        phase = "grasp_or_lift"
    elif eef_a_xy < 0.10:
        phase = "approach_cubeA"
    else:
        phase = "approach_above"
    return {
        "phase": phase,
        "cubeA_z": float(cube_a[2]),
        "cubeB_z": float(cube_b[2]),
        "best_cubeA_z": float(best_cube_z),
        "cubeA_cubeB_xy_dist": xy_ab,
        "cubeA_cubeB_z_gap": z_gap,
        "eef_cubeA_xy_dist": eef_a_xy,
        "eef_z": float(eef[2]),
    }


def get_progress(task: str, obs: dict[str, Any], best_cube_z: float) -> dict[str, Any]:
    if task == "Stack":
        return get_stack_progress(obs, best_cube_z)
    return get_lift_progress(obs, best_cube_z)


def try_apply_nudge(env: Any, task: str, rng: random.Random, xy: float) -> dict[str, Any]:
    event = {"type": "mid_episode_nudge", "applied": False, "dx": 0.0, "dy": 0.0}
    dx = rng.uniform(-xy, xy)
    dy = rng.uniform(-xy, xy)
    event.update({"dx": dx, "dy": dy})
    if task == "Stack":
        candidates = ["cubeB_joint0", "cubeB_joint", "cubeA_joint0", "cubeA_joint"]
    else:
        candidates = ["cube_joint0", "cube_joint", "cube"]
    for joint_name in candidates:
        try:
            qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64)
            if qpos.size >= 3:
                qpos[0] += dx
                qpos[1] += dy
                env.sim.data.set_joint_qpos(joint_name, qpos)
                env.sim.forward()
                event.update({"applied": True, "joint": joint_name})
                return event
        except Exception:
            continue
    # Fallback: record the event even if the free joint name is unavailable.
    event.update({"reason": "joint_not_found"})
    return event


def should_trigger_recovery(
    *,
    cfg: RuntimeConfig,
    trace: EpisodeTrace,
    progress: dict[str, Any],
    last_improvement_step: int,
    step: int,
    perturbation_step: int | None,
    recovery_cooldown_until: int,
) -> bool:
    if not cfg.agentic:
        return False
    if trace.recoveries_triggered >= cfg.max_recoveries_per_episode:
        return False
    if step < recovery_cooldown_until:
        return False
    if perturbation_step is not None and step in {perturbation_step, perturbation_step + 1}:
        return True
    if progress["phase"] in {"transport", "done"}:
        return False
    if progress["phase"] != "done" and step - last_improvement_step >= cfg.recovery_stall_steps:
        return True
    return False


def run_episode(
    *,
    env: Any,
    task: str,
    cfg: RuntimeConfig,
    seed: int,
    episode_id: int,
    horizon: int,
    perturb_step: int,
    perturb_xy: float,
    video_writer: Any | None = None,
    video_camera: str = "frontview",
    video_render_every: int = 2,
) -> EpisodeTrace:
    rng = random.Random(seed * 1009 + episode_id)
    np.random.seed(seed * 1009 + episode_id)
    obs = env.reset()
    policy = ScriptedChunkPolicy(task=task, chunk_size=cfg.chunk_size)
    trace = EpisodeTrace(method_tag=cfg.method_tag, task=task, seed=seed, episode_id=episode_id)

    action_buffer: list[np.ndarray] = []
    reuse_lockout_until = 0
    recovery_cooldown_until = 0
    recovery_active = False
    recovery_trigger_step: int | None = None
    perturbation_step: int | None = None
    best_cube_z = float(obs["cubeA_pos"][2] if task == "Stack" else obs["cube_pos"][2])
    last_improvement_step = 0
    start_time = time.perf_counter()

    def maybe_write_frame(step_idx: int, current_obs: dict[str, Any]) -> None:
        if video_writer is None or step_idx % max(1, int(video_render_every)) != 0:
            return
        frame_key = f"{video_camera}_image"
        if frame_key not in current_obs:
            return
        frame = np.asarray(current_obs[frame_key])
        if frame.ndim == 3 and frame.shape[-1] >= 3:
            # robosuite returns OpenGL-style images; flip to conventional video coordinates.
            video_writer.append_data(frame[::-1, :, :3])

    maybe_write_frame(0, obs)

    for step in range(horizon):
        loop_start = time.perf_counter()
        current_cube_z = float(obs["cubeA_pos"][2] if task == "Stack" else obs["cube_pos"][2])
        best_cube_z = max(best_cube_z, current_cube_z)
        progress = get_progress(task, obs, best_cube_z)
        trace.final_progress = progress
        trace.best_cube_z = best_cube_z
        if current_cube_z >= best_cube_z - 1e-6:
            last_improvement_step = step

        if perturb_xy > 0 and perturbation_step is None and step == perturb_step:
            event = try_apply_nudge(env, task, rng, perturb_xy)
            trace.perturbation_event = event
            perturbation_step = step
            if event.get("applied"):
                obs = env._get_observations()
            if cfg.light:
                reuse_lockout_until = max(reuse_lockout_until, step + cfg.post_perturbation_lockout_steps)
            action_buffer.clear()

        if should_trigger_recovery(
            cfg=cfg,
            trace=trace,
            progress=progress,
            last_improvement_step=last_improvement_step,
            step=step,
            perturbation_step=perturbation_step,
            recovery_cooldown_until=recovery_cooldown_until,
        ):
            trace.recoveries_triggered += 1
            recovery_active = True
            recovery_trigger_step = step
            recovery_cooldown_until = step + cfg.post_recovery_lockout_steps
            reuse_lockout_until = max(reuse_lockout_until, step + cfg.post_recovery_lockout_steps)
            action_buffer.clear()

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
            if cfg.simulated_full_call_latency_ms > 0:
                time.sleep(cfg.simulated_full_call_latency_ms / 1000.0)
            chunk = policy.infer_chunk(obs, step, recovery_boost=recovery_active)
            latency_ms = (time.perf_counter() - call_start) * 1000.0
            trace.policy_latency_ms.append(latency_ms)
            trace.full_policy_calls += 1
            action = chunk[0]
            if cfg.light:
                action_buffer = chunk[1 : 1 + cfg.reuse_max_actions]
            else:
                action_buffer = []

        obs, reward, done, info = env.step(action)
        maybe_write_frame(step + 1, obs)
        trace.steps = step + 1
        trace.control_steps += 1
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        if loop_ms > cfg.control_deadline_ms:
            trace.deadline_misses += 1

        if recovery_active and recovery_trigger_step is not None:
            if get_progress(task, obs, best_cube_z)["phase"] in {"grasp_or_lift", "transport", "done"} and step > recovery_trigger_step:
                recovery_active = False
                trace.recoveries_successful += 1

        if hasattr(env, "_check_success") and env._check_success():
            trace.success = True
            break

    trace.wall_sec = time.perf_counter() - start_time
    if not trace.success:
        if trace.final_progress:
            trace.failure_reason = str(trace.final_progress.get("phase", "unknown"))
        else:
            trace.failure_reason = "unknown"
    if trace.recoveries_triggered and trace.success and trace.recoveries_successful == 0:
        trace.recoveries_successful = 1
    return trace


def aggregate(traces: list[EpisodeTrace], cfg: RuntimeConfig, task: str) -> dict[str, Any]:
    rows = [t.to_json() for t in traces]
    n = len(rows)
    successes = sum(1 for r in rows if r["success"])
    full_calls = [r["full_policy_calls"] for r in rows]
    reused = [r["reused_actions"] for r in rows]
    misses = [r["deadline_miss_rate"] for r in rows]
    sud = [r["success_under_deadline"] for r in rows]
    wall = [r["wall_sec"] for r in rows]
    rec_t = sum(int(r["recoveries_triggered"]) for r in rows)
    rec_s = sum(int(r["recoveries_successful"]) for r in rows)
    full_total = sum(full_calls)
    reused_total = sum(reused)
    return {
        "method_tag": cfg.method_tag,
        "task": task,
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "full_policy_calls_per_episode": _safe_mean([float(x) for x in full_calls]),
        "reused_actions_per_episode": _safe_mean([float(x) for x in reused]),
        "reuse_ratio": reused_total / (full_total + reused_total) if (full_total + reused_total) else None,
        "wall_sec_mean": _safe_mean(wall),
        "deadline_miss_rate_mean": _safe_mean(misses),
        "success_under_deadline": _safe_mean(sud),
        "policy_latency_ms_mean": _safe_mean(
            [r["policy_latency_ms_mean"] for r in rows if r["policy_latency_ms_mean"] is not None]
        ),
        "policy_latency_ms_p95_mean": _safe_mean(
            [r["policy_latency_ms_p95"] for r in rows if r["policy_latency_ms_p95"] is not None]
        ),
        "recoveries_triggered_total": rec_t,
        "recoveries_successful_total": rec_s,
        "recovery_precision": rec_s / rec_t if rec_t else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Stack", choices=["Lift", "Stack"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=430)
    parser.add_argument("--perturb-step", type=int, default=260)
    parser.add_argument("--perturb-xy", type=float, default=0.03)
    parser.add_argument("--method-tag", default="RS-Agentic-Light")
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--reuse-max-actions", type=int, default=2)
    parser.add_argument("--control-deadline-ms", type=float, default=80.0)
    parser.add_argument("--simulated-full-call-latency-ms", type=float, default=0.0)
    parser.add_argument("--post-perturbation-lockout-steps", type=int, default=40)
    parser.add_argument("--post-recovery-lockout-steps", type=int, default=40)
    parser.add_argument("--recovery-stall-steps", type=int, default=25)
    parser.add_argument("--max-recoveries-per-episode", type=int, default=1)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-render-every", type=int, default=2)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig(
        method_tag=args.method_tag,
        agentic=bool(args.agentic),
        light=bool(args.light),
        replan_steps=int(args.replan_steps),
        chunk_size=int(args.chunk_size),
        reuse_max_actions=int(args.reuse_max_actions),
        control_deadline_ms=float(args.control_deadline_ms),
        simulated_full_call_latency_ms=float(args.simulated_full_call_latency_ms),
        post_perturbation_lockout_steps=int(args.post_perturbation_lockout_steps),
        post_recovery_lockout_steps=int(args.post_recovery_lockout_steps),
        recovery_stall_steps=int(args.recovery_stall_steps),
        max_recoveries_per_episode=int(args.max_recoveries_per_episode),
    )
    results_path = Path(args.results_json)
    trace_path = Path(args.trace_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video_path) if args.video_path else results_path.with_suffix(".mp4")
    if args.record_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.task,
        args.horizon,
        record_video=bool(args.record_video),
        camera_name=str(args.video_camera),
        camera_height=int(args.video_height),
        camera_width=int(args.video_width),
    )
    traces: list[EpisodeTrace] = []
    video_writer = None
    if args.record_video:
        import imageio.v2 as imageio

        video_writer = imageio.get_writer(str(video_path), fps=int(args.video_fps), macro_block_size=1)
    try:
        with trace_path.open("w", encoding="utf-8") as f:
            for episode_id in range(int(args.trials)):
                ep_seed = int(args.seed) + episode_id
                trace = run_episode(
                    env=env,
                    task=args.task,
                    cfg=cfg,
                    seed=ep_seed,
                    episode_id=episode_id,
                    horizon=int(args.horizon),
                    perturb_step=int(args.perturb_step),
                    perturb_xy=float(args.perturb_xy),
                    video_writer=video_writer if episode_id == 0 else None,
                    video_camera=str(args.video_camera),
                    video_render_every=int(args.video_render_every),
                )
                traces.append(trace)
                f.write(json.dumps(trace.to_json(), default=_json_default) + "\n")
                f.flush()
                print(
                    f"[{cfg.method_tag}] ep={episode_id} success={trace.success} "
                    f"steps={trace.steps} full={trace.full_policy_calls} reuse={trace.reused_actions} "
                    f"miss={trace.to_json()['deadline_miss_rate']:.3f} rec={trace.recoveries_successful}/{trace.recoveries_triggered}",
                    flush=True,
                )
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    summary = {
        "benchmark": "robosuite_deployment_pilot",
        "task": args.task,
        "seed": int(args.seed),
        "trials": int(args.trials),
        "runtime_config": cfg.__dict__,
        "aggregate": aggregate(traces, cfg, args.task),
        "trace_jsonl": str(trace_path),
        "video_path": str(video_path) if args.record_video else None,
    }
    results_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
