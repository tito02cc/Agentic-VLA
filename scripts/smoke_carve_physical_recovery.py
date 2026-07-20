#!/usr/bin/env python3
"""Run a deterministic CARVE physical-recovery skill from a LIBERO snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import imageio
import numpy as np

from agentic_vla.runtime import (
    ActionSpec,
    RecoveryContext,
    RecoveryMemory,
    StatefulRecoveryExecutor,
    build_cartesian_retreat_plan,
)
from scripts.run_agentic_vla_libero import _make_env, _require_runtime


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT
    / "results/carve_t689_paired_5states_v6/joint/failure_snapshots"
    / "task06_episode001_step0013.npz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=pathlib.Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "results/carve_recovery/physical_recovery_smoke.json",
    )
    parser.add_argument("--video", type=pathlib.Path, default=None)
    parser.add_argument("--minimum-state-response", type=float, default=1e-4)
    return parser.parse_args()


def _digest(state: np.ndarray) -> str:
    values = np.rint(np.asarray(state, dtype=np.float64).reshape(-1) * 1e9).astype(np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _run_once(
    *,
    env,
    sim_state: np.ndarray,
    plan,
    metadata: dict,
    minimum_state_response: float,
    video_path: pathlib.Path | None,
) -> dict:
    env.reset()
    observation = env.regenerate_obs_from_state(sim_state.copy())
    initial_state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    task_success_before = bool(env.check_success())
    memory = RecoveryMemory(max_records=4)
    executor = StatefulRecoveryExecutor(memory)
    executor.start(
        plan,
        RecoveryContext(
            episode_id=f"physical-smoke:{metadata['task_id']}:{metadata['episode_id']}",
            trigger_event=str(metadata["trigger"]),
            attempt=0,
            timestep=int(metadata["timestep"]),
            metadata={"snapshot": metadata["array_file"]},
        ),
    )

    frames = []
    phase_records = []
    environment_terminated = False
    for _ in plan.phases:
        command = executor.next_command()
        phase_start = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        executed = 0
        for action in command.actions:
            frame = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
            frames.append(frame)
            observation, _, environment_terminated, _ = env.step(list(action))
            executed += 1
            if environment_terminated:
                break
        phase_end = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        phase_records.append(
            {
                **command.to_dict(),
                "executed_actions": executed,
                "eef_response": float(np.linalg.norm(phase_end - phase_start)),
                "environment_terminated": bool(environment_terminated),
            }
        )
        if environment_terminated and not command.request_verification:
            outcome = executor.abort(
                reason="environment terminated before recovery verification",
                safe_stop=not bool(env.check_success()),
            )
            break
        if command.request_verification:
            total_response = float(
                np.linalg.norm(
                    np.asarray(observation["robot0_eef_pos"], dtype=np.float64) - initial_eef
                )
            )
            verified = bool(
                np.isfinite(total_response)
                and total_response >= minimum_state_response
                and (not environment_terminated or bool(env.check_success()))
            )
            outcome = executor.resolve_verification(
                verified,
                evidence={
                    "eef_state_response": total_response,
                    "minimum_state_response": minimum_state_response,
                    "task_success": bool(env.check_success()),
                    "environment_terminated": bool(environment_terminated),
                },
            )
            break
    else:
        raise RuntimeError("physical recovery ended without an outcome")

    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=20)
    final_state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    return {
        "initial_state_digest": _digest(initial_state),
        "final_state_digest": _digest(final_state),
        "final_state": final_state.tolist(),
        "phase_records": phase_records,
        "outcome": outcome.to_dict(),
        "task_success_before": task_success_before,
        "task_success_after": bool(env.check_success()),
        "video": str(video_path) if video_path is not None else None,
    }


def main() -> int:
    args = _parse_args()
    if args.minimum_state_response <= 0:
        raise ValueError("--minimum-state-response must be positive")
    metadata_path = args.snapshot.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(args.snapshot, allow_pickle=False) as arrays:
        sim_state = np.asarray(arrays["sim_state"], dtype=np.float64)
        cached_actions = np.asarray(arrays["cached_actions"], dtype=np.float32)
    if cached_actions.ndim != 2 or cached_actions.shape[1] != 7 or not len(cached_actions):
        raise ValueError(f"snapshot has invalid cached actions: {cached_actions.shape}")

    action_spec = ActionSpec(
        action_dim=7,
        representation="normalized_delta_cartesian_pose",
        coordinate_frame="robot_base",
        gripper_convention="-1=close,+1=open",
        control_frequency_hz=20.0,
        minimum=-1.0,
        maximum=1.0,
    )
    plan = build_cartesian_retreat_plan(action_spec, cached_actions[0])

    benchmark, get_libero_path, offscreen_env, _, _, _ = _require_runtime()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(int(metadata["task_id"]))
    env = _make_env(task, get_libero_path, offscreen_env, seed=7)
    try:
        first = _run_once(
            env=env,
            sim_state=sim_state,
            plan=plan,
            metadata=metadata,
            minimum_state_response=float(args.minimum_state_response),
            video_path=args.video,
        )
        second = _run_once(
            env=env,
            sim_state=sim_state,
            plan=plan,
            metadata=metadata,
            minimum_state_response=float(args.minimum_state_response),
            video_path=None,
        )
    finally:
        env.close()

    final_delta = float(
        np.max(
            np.abs(
                np.asarray(first.pop("final_state"), dtype=np.float64)
                - np.asarray(second.pop("final_state"), dtype=np.float64)
            )
        )
    )
    deterministic = bool(
        first["final_state_digest"] == second["final_state_digest"] and final_delta <= 1e-9
    )
    passed = bool(
        deterministic
        and first["outcome"]["status"] == "succeeded"
        and second["outcome"]["status"] == "succeeded"
    )
    payload = {
        "experiment": "stateful-physical-recovery-smoke",
        "snapshot": args.snapshot.name,
        "trigger": metadata["trigger"],
        "skill_id": plan.skill_id,
        "action_count": plan.action_count,
        "phase_names": [phase.name for phase in plan.phases],
        "deterministic_replay": deterministic,
        "final_state_max_abs_delta": final_delta,
        "first": first,
        "second": second,
        "passed": passed,
        "claim_boundary": "physical-response and lifecycle smoke; not task recovery success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "skill_id": plan.skill_id,
                "deterministic_replay": deterministic,
                "first_outcome": first["outcome"]["status"],
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
