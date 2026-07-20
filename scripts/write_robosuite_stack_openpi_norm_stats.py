#!/usr/bin/env python3
"""Write OpenPI norm stats for the converted robosuite Stack dataset.

This avoids the slow image-heavy LeRobot pass for a first pi0.5 integration
smoke.  It uses the same state mapping as
``convert_robosuite_stack_hdf5_to_lerobot.py`` and writes the standard
OpenPI ``norm_stats.json`` format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from openpi.shared import normalize

from scripts.convert_robosuite_stack_hdf5_to_lerobot import build_openpi_state


DEFAULT_INPUT = Path("results/robosuite_stack_e2e_v2_20260611/stack_demos_100eps.hdf5")
DEFAULT_OUTPUT = Path("assets/pi05_robosuite_stack_smoke/agentic-vla/robosuite_stack")


def episode_group(h5: h5py.File) -> h5py.Group:
    if "episodes" in h5:
        return h5["episodes"]
    if "data" in h5:
        return h5["data"]
    raise ValueError("HDF5 file must contain an 'episodes' or 'data' group")


def sort_episode_keys(keys: list[str]) -> list[str]:
    def key_fn(name: str) -> tuple[int, str]:
        tail = name.rsplit("_", 1)[-1]
        return (int(tail) if tail.isdigit() else 10**9, name)

    return sorted(keys, key=key_fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-mode", choices=("proprio", "privileged_stack"), default="proprio")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frame_count = 0
    episode_count = 0

    with h5py.File(args.input_file, "r") as h5:
        group = episode_group(h5)
        for key in sort_episode_keys(list(group.keys())):
            ep = group[key]
            states = build_openpi_state(ep["states"][:], state_mode=args.state_mode)
            actions = np.asarray(ep["actions"][:], dtype=np.float32)
            if args.max_frames is not None:
                remaining = int(args.max_frames) - frame_count
                if remaining <= 0:
                    break
                states = states[:remaining]
                actions = actions[:remaining]
            if states.shape[0] == 0:
                continue
            state_stats.update(states)
            action_stats.update(actions)
            frame_count += int(states.shape[0])
            episode_count += 1

    if frame_count < 2:
        raise ValueError("need at least two frames to compute norm stats")

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    normalize.save(args.output_dir, norm_stats)
    print(
        f"[norm] episodes={episode_count} frames={frame_count} "
        f"output={args.output_dir / 'norm_stats.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
