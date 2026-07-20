#!/usr/bin/env python3
"""Validate reproducible state branching in a real LIBERO MuJoCo task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_ROOT = PROJECT_ROOT / "LIBERO"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from agentic_vla.experiments import capture_simulator_snapshot, restore_simulator_snapshot
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def _digest(array: Any) -> str:
    value = np.asarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _stable_state_digest(array: Any) -> str:
    value = np.rint(np.asarray(array, dtype=np.float64) * 1e9).astype(np.int64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _rollout(env: Any, snapshot: Any, actions: np.ndarray) -> dict[str, Any]:
    observation = restore_simulator_snapshot(env, snapshot)
    done = False
    for action in actions:
        observation, _, done, _ = env.step(action.tolist())
    state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    image = np.asarray(observation["agentview_image"])
    return {
        "state": state,
        "state_digest": _stable_state_digest(state),
        "image": image,
        "image_digest": _digest(image),
        "done": bool(done),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--branch-steps", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "carve_e1_libero_branch_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    init_index = args.init_state_id % len(init_states)
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(init_states[init_index])
        for _ in range(5):
            observation, _, _, _ = env.step([0.0] * 7)

        snapshot = capture_simulator_snapshot(
            env,
            task_id=args.task_id,
            episode_id=init_index,
            timestep=5,
        )
        branch_a_actions = np.zeros((args.branch_steps, 7), dtype=np.float64)
        branch_b_actions = np.zeros((args.branch_steps, 7), dtype=np.float64)
        branch_b_actions[:, 0] = 0.15

        branch_a_first = _rollout(env, snapshot, branch_a_actions)
        branch_b = _rollout(env, snapshot, branch_b_actions)
        branch_a_repeat = _rollout(env, snapshot, branch_a_actions)
        repeat_error = float(
            np.max(np.abs(branch_a_first["state"] - branch_a_repeat["state"]))
        )
        branch_distance = float(np.linalg.norm(branch_a_first["state"] - branch_b["state"]))
        image_difference = np.abs(
            branch_a_first["image"].astype(np.int16)
            - branch_a_repeat["image"].astype(np.int16)
        )
        image_max_abs_error = int(image_difference.max())
        image_mean_abs_error = float(image_difference.mean())
        reproducible = bool(
            np.allclose(
                branch_a_first["state"],
                branch_a_repeat["state"],
                rtol=1e-10,
                atol=1e-10,
            )
            and image_mean_abs_error <= 0.01
        )
        diverged = branch_distance > 1e-6
        payload = {
            "experiment": "E1-libero-state-branch-smoke",
            "task_suite": "libero_10",
            "task_id": args.task_id,
            "task_name": task.name,
            "task_description": task.language,
            "seed": args.seed,
            "init_state_id": init_index,
            "branch_steps": args.branch_steps,
            "snapshot_fingerprint": snapshot.fingerprint,
            "branch_a_first_state_digest": branch_a_first["state_digest"],
            "branch_a_repeat_state_digest": branch_a_repeat["state_digest"],
            "branch_b_state_digest": branch_b["state_digest"],
            "branch_a_image_digest": branch_a_first["image_digest"],
            "branch_a_repeat_image_digest": branch_a_repeat["image_digest"],
            "branch_b_image_digest": branch_b["image_digest"],
            "repeat_max_abs_error": repeat_error,
            "repeat_image_max_abs_error": image_max_abs_error,
            "repeat_image_mean_abs_error": image_mean_abs_error,
            "branch_state_l2_distance": branch_distance,
            "same_branch_reproducible": reproducible,
            "different_branches_diverged": diverged,
            "passed": reproducible and diverged,
            "controller_privileged_state": False,
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        printable = dict(payload)
        print(json.dumps(printable, indent=2))
        if not payload["passed"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
