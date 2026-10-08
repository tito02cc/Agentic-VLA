#!/usr/bin/env python3
"""Run one explicit RoboMME online-oracle upper-bound episode."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from agentic_vla.benchmarks.robomme_runtime import pack_robomme_state
from scripts.run_robomme_policy_admission import (
    pack_history,
    paired_frame,
    percentile,
    wait_for_reset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--task", default="MoveCube")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def current_observation(observation: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    front = np.asarray(observation["front_rgb_list"][-1], dtype=np.uint8)
    wrist = np.asarray(observation["wrist_rgb_list"][-1], dtype=np.uint8)
    state = pack_robomme_state(
        observation["joint_state_list"][-1],
        observation["gripper_state_list"][-1],
    )
    return front, wrist, state


def initial_history(observation: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    front = [np.asarray(value, dtype=np.uint8) for value in observation["front_rgb_list"]]
    wrist = [np.asarray(value, dtype=np.uint8) for value in observation["wrist_rgb_list"]]
    states = [
        pack_robomme_state(joints, gripper)
        for joints, gripper in zip(
            observation["joint_state_list"], observation["gripper_state_list"]
        )
    ]
    return front, wrist, states


def task_instruction(info: dict[str, object]) -> str:
    value = info["task_goal"]
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value).strip()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.action_horizon <= 0:
        raise ValueError("max steps and action horizon must be positive")

    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    started = time.perf_counter()
    client = MMEVLAWebsocketClientPolicy(args.host, args.port)
    wait_for_reset(client)
    env = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=args.max_steps,
    ).make_env_for_episode(args.episode)
    observation, info = env.reset()
    instruction = task_instruction(info)
    history_front, history_wrist, history_states = initial_history(observation)
    initial_history_frames = len(history_front)
    exec_start_idx = initial_history_frames - 1
    frames = [paired_frame(front, wrist) for front, wrist in zip(history_front, history_wrist)]
    policy_latency_ms: list[float] = []
    policy_calls = 0
    history_batches = 0
    steps = 0
    terminated = False
    truncated = False
    subgoals: list[str] = []
    subgoal_trace: list[dict[str, object]] = []

    try:
        while steps < args.max_steps and not (terminated or truncated):
            response = client.add_buffer(
                pack_history(history_front, history_states, exec_start_idx)
            )
            if not response.get("add_buffer_finished", False):
                raise RuntimeError("policy rejected the history buffer")
            history_batches += 1

            front, wrist, state = current_observation(observation)
            subgoal = str(info["grounded_subgoal_online"]).strip()
            subgoals.append(subgoal)
            subgoal_trace.append({"step": steps, "grounded_subgoal": subgoal})
            payload = {
                "observation/image": front,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": instruction,
                "simple_subgoal": subgoal,
                "grounded_subgoal": subgoal,
            }
            infer_started = time.perf_counter()
            response = client.infer(payload)
            policy_latency_ms.append((time.perf_counter() - infer_started) * 1000.0)
            policy_calls += 1
            actions = np.asarray(response["actions"], dtype=np.float32)[: args.action_horizon]

            history_front.clear()
            history_wrist.clear()
            history_states.clear()
            exec_start_idx = 0
            for action in actions:
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                front, wrist, state = current_observation(observation)
                history_front.append(front.copy())
                history_wrist.append(wrist.copy())
                history_states.append(state.copy())
                frames.append(paired_frame(front, wrist))
                if steps >= args.max_steps or terminated or truncated:
                    break

        args.output.mkdir(parents=True, exist_ok=True)
        video_path = args.output / f"{args.task}_ep{args.episode}_oracle_upper_bound.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        status = str(info.get("status", "unknown"))
        unique_subgoals = list(dict.fromkeys(subgoals))
        summary = {
            "gate": "robomme_online_oracle_upper_bound",
            "deployable_method": False,
            "privileged_online_subgoal_used": True,
            "policy_label": args.policy_label,
            "task": args.task,
            "episode": args.episode,
            "instruction": instruction,
            "status": status,
            "success": status == "success",
            "initial_memory_frames": initial_history_frames,
            "policy_calls": policy_calls,
            "history_batches": history_batches,
            "executed_steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "subgoal_queries": len(subgoals),
            "unique_subgoals": unique_subgoals,
            "subgoal_trace": subgoal_trace,
            "video_frames": len(frames),
            "video_path": str(video_path),
            "policy_latency_ms": {
                "mean": statistics.fmean(policy_latency_ms),
                "p50": percentile(policy_latency_ms, 50),
                "p95": percentile(policy_latency_ms, 95),
                "max": max(policy_latency_ms),
            },
            "wall_time_s": time.perf_counter() - started,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
