#!/usr/bin/env python3
"""Collect only RoboMME initial demonstration observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.benchmarks.robomme_memory import planner_demo_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--demo-history-mode", choices=("legacy_full", "official"), default="legacy_full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    for episode in args.episodes:
        output = args.output_root / f"{args.task}_ep{episode}"
        output.mkdir(parents=True, exist_ok=False)
        env = BenchmarkEnvBuilder(
            env_id=args.task,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
            max_steps=1,
        ).make_env_for_episode(episode)
        try:
            observation, info = env.reset()
            frames = [
                np.asarray(frame, dtype=np.uint8)
                for frame in observation["front_rgb_list"]
            ]
            goal = info["task_goal"]
            instruction = str(
                goal[0] if isinstance(goal, (list, tuple)) else goal
            ).strip()
            video = output / "initial_demo_front.mp4"
            demo = planner_demo_frames(frames, args.demo_history_mode)
            if not demo:
                raise ValueError("the task supplies no initial demonstration")
            imageio.mimsave(video, demo, fps=30)
            summary = {
                "protocol": "carve.robomme.initial_demo.v1",
                "task": args.task,
                "episode": episode,
                "instruction": instruction,
                "initial_memory_frames": len(frames),
                "demo_history_mode": args.demo_history_mode,
                "planner_demo_frames": len(demo),
                "video_path": str(video),
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary))
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
