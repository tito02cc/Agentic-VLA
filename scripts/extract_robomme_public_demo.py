#!/usr/bin/env python3
"""Extract a RoboMME episode's public initial demonstration without policy calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from agentic_vla.benchmarks.robomme_memory import planner_demo_frames
from scripts.run_robomme_vlm_groundsg import initial_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VideoUnmaskSwap")
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--demo-history-mode", choices=("official",), default="official")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def extract_episode(args: argparse.Namespace, episode: int) -> dict[str, object]:
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    output = args.output_root / f"{args.task}_ep{episode}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"public demonstration already exists: {output}")
    builder = BenchmarkEnvBuilder(
        env_id=args.task, dataset="test", action_space="joint_angle",
        gui_render=False, max_steps=args.max_steps,
    )
    episode_count = builder.get_episode_num()
    if not 0 <= episode < episode_count:
        raise ValueError(f"episode {episode} is outside {args.task} test range [0, {episode_count})")
    env = builder.make_env_for_episode(episode)
    try:
        observation, info = env.reset()
        goal = info["task_goal"]
        instruction = str(goal[0] if isinstance(goal, (list, tuple)) else goal).strip()
        front, wrist, states = initial_history(observation)
        initial_observation_sha256 = hashlib.sha256(
            np.asarray(front, dtype=np.uint8).tobytes()
            + np.asarray(wrist, dtype=np.uint8).tobytes()
            + np.asarray(states, dtype=np.float32).tobytes()
            + instruction.encode("utf-8")
        ).hexdigest()
        output.mkdir(parents=True, exist_ok=True)
        video_path = output / "initial_demo_front.mp4"
        demo_frames = planner_demo_frames(front, args.demo_history_mode)
        if demo_frames:
            imageio.mimsave(video_path, demo_frames, fps=30)
        summary: dict[str, object] = {
            "protocol": "carve.robomme.public_initial_demo.v1",
            "task": args.task,
            "episode": episode,
            "instruction": instruction,
            "demo_history_mode": args.demo_history_mode,
            "initial_memory_frames": len(front),
            "initial_observation_sha256": initial_observation_sha256,
            "initial_demo_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest() if demo_frames else None,
            "policy_calls": 0,
            "planner_calls": 0,
            "executed_steps": 0,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    for episode in args.episode:
        print(json.dumps(extract_episode(args, episode), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
