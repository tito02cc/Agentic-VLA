#!/usr/bin/env python3
"""Create evaluator-only LIBERO snapshots from official initial states."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_ROOT = PROJECT_ROOT / "LIBERO"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from agentic_vla.experiments import capture_simulator_snapshot
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def _parse_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids or len(ids) != len(set(ids)) or any(item < 0 for item in ids):
        raise argparse.ArgumentTypeError("IDs must be unique non-negative integers")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=_parse_ids, required=True)
    parser.add_argument("--init-state-ids", type=_parse_ids, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.settle_steps < 0 or args.resolution <= 0:
        raise ValueError("settle steps must be non-negative and resolution positive")
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.resolution,
            camera_widths=args.resolution,
        )
        try:
            env.seed(args.seed)
            for state_id in args.init_state_ids:
                if state_id >= len(init_states):
                    raise IndexError(
                        f"task {task_id} has {len(init_states)} states, got {state_id}"
                    )
                env.reset()
                observation = env.set_init_state(init_states[state_id])
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step([0.0] * 7)
                snapshot = capture_simulator_snapshot(
                    env,
                    task_id=task_id,
                    episode_id=state_id,
                    timestep=args.settle_steps,
                )
                stem = f"task{task_id:02d}_state{state_id:02d}_step{args.settle_steps:04d}"
                array_path = args.output_root / f"{stem}.npz"
                metadata_path = args.output_root / f"{stem}.json"
                arrays = {
                    "sim_state": snapshot.state,
                    "cached_actions": np.empty((0, 7), dtype=np.float32),
                    "last_action": np.zeros(7, dtype=np.float32),
                }
                for key, value in observation.items():
                    arrays[f"observation_{key}"] = np.asarray(value)
                np.savez_compressed(array_path, **arrays)
                metadata = {
                    "schema_version": 2,
                    "task_id": task_id,
                    "episode_id": state_id,
                    "timestep": args.settle_steps,
                    "trigger": "stale_action",
                    "instruction": task.language,
                    "snapshot_fingerprint": snapshot.fingerprint,
                    "controller": {
                        "source": "official_initial_state",
                        "settle_steps": args.settle_steps,
                        "seed": args.seed,
                    },
                    "sim_state_role": "evaluator_restore_only",
                    "controller_uses_privileged_state": False,
                    "array_file": array_path.name,
                }
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                manifest.append(
                    {
                        "snapshot": stem,
                        "task_id": task_id,
                        "init_state_id": state_id,
                        "instruction": task.language,
                        "fingerprint": snapshot.fingerprint,
                    }
                )
        finally:
            env.close()
    payload = {
        "schema_version": "carve-libero-initial-snapshots-v1",
        "task_suite": "libero_10",
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "snapshot_count": len(manifest),
        "snapshots": manifest,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
