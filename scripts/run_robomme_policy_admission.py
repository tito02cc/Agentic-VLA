#!/usr/bin/env python3
"""Run one official RoboMME episode through a remote PI0.5-style policy."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from agentic_vla.benchmarks.robomme_runtime import RoboMMERuntimeAdapter
from agentic_vla.toolchain import ToolExecutionContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--task", default="MoveCube")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--use-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subgoal", default=None)
    parser.add_argument("--policy-label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def paired_frame(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.asarray(front, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)),
        axis=1,
    )


def pack_history(
    images: list[np.ndarray], states: list[np.ndarray], exec_start_idx: int
) -> dict[str, object]:
    return {
        "images": np.stack(images, axis=0).astype(np.uint8)[:, None],
        "state": np.stack(states, axis=0).astype(np.float32),
        "add_buffer": True,
        "exec_start_idx": int(exec_start_idx),
    }


def wait_for_reset(client: object, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        response = client.reset()
        if isinstance(response, dict) and response.get("reset_finished", False):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("policy reset did not finish before the deadline")
        time.sleep(0.1)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.action_horizon <= 0:
        raise ValueError("max steps and action horizon must be positive")

    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    started = time.perf_counter()
    client = MMEVLAWebsocketClientPolicy(args.host, args.port)
    server_metadata = {
        str(key): value
        for key, value in client.get_server_metadata().items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
    wait_for_reset(client)

    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=args.max_steps,
    )
    env = builder.make_env_for_episode(args.episode)
    adapter = RoboMMERuntimeAdapter(
        env,
        task_id=args.task,
        episode_id=args.episode,
        max_steps=args.max_steps,
    )
    adapter.reset()

    history_images = [frame.copy() for frame in adapter.initial_memory["front_rgb"]]
    history_states = [state.copy() for state in adapter.initial_memory["robot_state"]]
    initial_history_frames = len(history_images)
    exec_start_idx = initial_history_frames - 1
    frames = [
        paired_frame(front, wrist)
        for front, wrist in zip(
            adapter.initial_memory["front_rgb"], adapter.initial_memory["wrist_rgb"]
        )
    ]
    inference_ms: list[float] = []
    policy_calls = 0
    added_history_batches = 0

    try:
        while adapter.timestep < args.max_steps and not adapter.terminated:
            if args.use_history:
                response = client.add_buffer(
                    pack_history(history_images, history_states, exec_start_idx)
                )
                if not response.get("add_buffer_finished", False):
                    raise RuntimeError("policy rejected the history buffer")
                added_history_batches += 1

            observation = adapter.observe()
            payload = dict(observation.policy_observation)
            payload["prompt"] = adapter.task_instruction
            if args.subgoal:
                payload["simple_subgoal"] = args.subgoal
                payload["grounded_subgoal"] = args.subgoal

            infer_started = time.perf_counter()
            response = client.infer(payload)
            inference_ms.append((time.perf_counter() - infer_started) * 1000.0)
            policy_calls += 1
            if "actions" not in response:
                raise KeyError("policy response does not contain actions")
            actions = np.asarray(response["actions"], dtype=np.float32)
            actions = actions[: args.action_horizon]

            # The server now owns the submitted history. Accumulate only the
            # subsequently executed trajectory for the next policy request.
            history_images.clear()
            history_states.clear()
            exec_start_idx = 0

            context = ToolExecutionContext(
                episode_id=f"{args.task}:{args.episode}",
                timestep=adapter.timestep,
                at_safe_boundary=True,
                allowed_tools=(),
                deployment_profile_id="robomme-policy-admission",
            )

            def capture(current, _action) -> None:
                frames.append(
                    paired_frame(
                        current.planner_frames["front"],
                        current.planner_frames["wrist"],
                    )
                )
                history_images.append(current.planner_frames["front"].copy())
                history_states.append(np.asarray(current.robot_state, dtype=np.float32))

            adapter.execute_action_chunk(actions, context, on_step=capture)

        result = adapter.private_result()
        args.output.mkdir(parents=True, exist_ok=True)
        video_path = args.output / f"{args.task}_ep{args.episode}_policy_admission.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        summary = {
            "gate": "robomme_policy_admission",
            "task_success_claim": False,
            "policy_label": args.policy_label,
            "policy_endpoint": f"ws://{args.host}:{args.port}",
            "server_metadata": server_metadata,
            "task": args.task,
            "episode": args.episode,
            "instruction": adapter.task_instruction,
            "use_history": args.use_history,
            "subgoal_supplied": bool(args.subgoal),
            "fixed_subgoal": args.subgoal,
            "initial_memory_frames": initial_history_frames,
            "policy_calls": policy_calls,
            "history_batches": added_history_batches,
            "executed_steps": adapter.timestep,
            "evaluator_status": result.evaluator_metadata.get("status"),
            "terminated": result.terminated,
            "video_frames": len(frames),
            "video_path": str(video_path),
            "policy_latency_ms": {
                "mean": statistics.fmean(inference_ms) if inference_ms else None,
                "p50": percentile(inference_ms, 50),
                "p95": percentile(inference_ms, 95),
                "max": max(inference_ms) if inference_ms else None,
            },
            "wall_time_s": time.perf_counter() - started,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
