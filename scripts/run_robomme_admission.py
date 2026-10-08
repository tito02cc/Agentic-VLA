#!/usr/bin/env python3
"""Run CARVE's no-policy RoboMME environment admission gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from agentic_vla.benchmarks.robomme_runtime import RoboMMERuntimeAdapter
from agentic_vla.toolchain import ToolExecutionContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="MoveCube")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_frame(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.asarray(front, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)),
        axis=1,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    started = time.perf_counter()
    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=args.steps,
    )
    env = builder.make_env_for_episode(args.episode)
    adapter = RoboMMERuntimeAdapter(
        env,
        task_id=args.task,
        episode_id=args.episode,
        max_steps=args.steps,
    )
    observation = adapter.reset()
    frames = [
        make_frame(front, wrist)
        for front, wrist in zip(
            adapter.initial_memory["front_rgb"],
            adapter.initial_memory["wrist_rgb"],
        )
    ]

    base_action = np.asarray(
        [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
        dtype=np.float32,
    )

    try:
        while adapter.timestep < args.steps and not adapter.terminated:
            context = ToolExecutionContext(
                episode_id=f"{args.task}:{args.episode}",
                timestep=adapter.timestep,
                at_safe_boundary=True,
                allowed_tools=(),
                deployment_profile_id="robomme-admission-no-policy",
            )
            remaining = args.steps - adapter.timestep
            chunk = np.repeat(base_action[None, :], min(16, remaining), axis=0)

            def capture(current, _action) -> None:
                frames.append(
                    make_frame(
                        current.planner_frames["front"],
                        current.planner_frames["wrist"],
                    )
                )

            adapter.execute_action_chunk(chunk, context, on_step=capture)

        result = adapter.private_result()
        args.output.mkdir(parents=True, exist_ok=True)
        video_path = args.output / f"{args.task}_ep{args.episode}_admission.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        summary = {
            "gate": "robomme_environment_admission",
            "diagnostic_only": True,
            "policy_loaded": False,
            "task": args.task,
            "episode": args.episode,
            "instruction": adapter.task_instruction,
            "initial_memory_frames": len(adapter.initial_memory["front_rgb"]),
            "executed_steps": adapter.timestep,
            "video_frames": len(frames),
            "video_path": str(video_path),
            "observation_keys": sorted(observation.policy_observation),
            "state_dim": len(observation.robot_state),
            "terminated": result.terminated,
            "evaluator_status": result.evaluator_metadata.get("status"),
            "wall_time_s": time.perf_counter() - started,
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
