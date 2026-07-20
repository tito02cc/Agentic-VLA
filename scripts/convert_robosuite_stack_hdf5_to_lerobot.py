#!/usr/bin/env python3
"""Convert robosuite Stack HDF5 trajectories to a pi0.5/LeRobot dataset.

The current robosuite collector stores real MuJoCo RGB frames, 56D flattened
state, and 7D OSC_POSE actions.  OpenPI's LIBERO policy path expects:

- image: third-person RGB image
- wrist_image: wrist RGB image
- state: 8D [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
- actions: 7D action

The robosuite HDF5 does not include a wrist camera, so this converter duplicates
the front-view image into ``wrist_image`` for the first pi0.5 integration smoke.
This should be replaced by a true wrist camera capture for paper-quality
robosuite fine-tuning/evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_REPO_ID = "agentic-vla/robosuite_stack"
DEFAULT_PROMPT = "stack the red cube on the green cube"


def quat_to_axis_angle_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"expected quaternion dim 4, got {q.shape}")

    w, x, y, z = [q[..., i] for i in range(4)]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.maximum(norm, 1e-9)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    w = np.clip(w, -1.0, 1.0)

    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1e-12, 1.0 - w * w))
    axis = np.stack([x / s, y / s, z / s], axis=-1)
    return (axis * angle[..., None]).astype(np.float32)


def build_openpi_state(flat_states: np.ndarray, *, state_mode: str = "proprio") -> np.ndarray:
    """Extract the 8D OpenPI/LIBERO state from robosuite flattened state.

    The first 32 dims are robosuite ``robot0_proprio-state``:
    - 21:24 robot0_eef_pos
    - 24:28 robot0_eef_quat in wxyz order
    - 28:30 robot0_gripper_qpos

    ``privileged_stack`` appends robosuite's 23D object-state and the 1D
    step-fraction feature. This keeps the final state at 32D, matching pi0.5's
    native state/action dimensionality, and is used only as a diagnostic
    state-observability experiment.
    """

    states = np.asarray(flat_states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 30:
        raise ValueError(f"expected flattened state with at least 30 dims, got {states.shape}")

    eef_pos = states[:, 21:24]
    eef_quat = states[:, 24:28]
    gripper_qpos = states[:, 28:30]
    eef_axis_angle = quat_to_axis_angle_wxyz(eef_quat)
    base_state = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=-1)
    if state_mode == "proprio":
        openpi_state = base_state
    elif state_mode == "privileged_stack":
        if states.shape[1] < 56:
            raise ValueError(f"privileged_stack expects at least 56 dims, got {states.shape}")
        object_state = states[:, 32:55]
        step_fraction = states[:, 55:56]
        openpi_state = np.concatenate([base_state, object_state, step_fraction], axis=-1)
    else:
        raise ValueError(f"unknown state_mode={state_mode!r}")
    return openpi_state.astype(np.float32, copy=False)


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


def load_episodes(
    input_file: Path,
    *,
    max_episodes: int | None,
    state_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with h5py.File(input_file, "r") as h5:
        root_attrs = {key: h5.attrs[key] for key in h5.attrs.keys()}
        group = episode_group(h5)
        for key in sort_episode_keys(list(group.keys())):
            ep = group[key]
            images = np.asarray(ep["images"], dtype=np.uint8)
            flat_states = np.asarray(ep["states"], dtype=np.float32)
            actions = np.asarray(ep["actions"], dtype=np.float32)
            if not (images.shape[0] == flat_states.shape[0] == actions.shape[0]):
                raise ValueError(
                    f"{key}: length mismatch images={images.shape[0]} "
                    f"states={flat_states.shape[0]} actions={actions.shape[0]}"
                )
            episodes.append(
                {
                    "name": key,
                    "images": images,
                    "state": build_openpi_state(flat_states, state_mode=state_mode),
                    "actions": actions,
                    "attrs": {attr_key: ep.attrs[attr_key] for attr_key in ep.attrs.keys()},
                }
            )
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
    if not episodes:
        raise ValueError("no episodes selected")
    return episodes, root_attrs


def import_lerobot_dataset():
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit("Run this script inside the OpenPI/LeRobot environment.") from exc
    return LeRobotDataset, Path(HF_LEROBOT_HOME)


def create_dataset(args: argparse.Namespace):
    LeRobotDataset, hf_home = import_lerobot_dataset()
    root = args.root or (hf_home / args.repo_id)
    root = root.expanduser().resolve()
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite")
        shutil.rmtree(root)

    features = {
        "image": {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (int(args.state_dim),),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="franka_panda",
        fps=args.fps,
        features=features,
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    return dataset, root


def write_dataset(episodes: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    args.state_dim = int(episodes[0]["state"].shape[-1])
    dataset, root = create_dataset(args)
    try:
        for ep_idx, episode in enumerate(episodes):
            images = episode["images"]
            states = episode["state"]
            actions = episode["actions"]
            print(f"[write] {episode['name']} frames={states.shape[0]}")
            for frame_idx in range(states.shape[0]):
                image = images[frame_idx]
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": image.copy(),
                        "state": states[frame_idx],
                        "actions": actions[frame_idx],
                        "task": args.prompt,
                    }
                )
            dataset.save_episode()
            if args.progress_every > 0 and (ep_idx + 1) % args.progress_every == 0:
                print(f"[progress] episodes={ep_idx + 1}/{len(episodes)}")
    finally:
        stop = getattr(dataset, "stop_image_writer", None)
        if callable(stop):
            stop()
    return root


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def write_summary(
    root: Path,
    episodes: list[dict[str, Any]],
    root_attrs: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    first_state = episodes[0]["state"]
    first_actions = episodes[0]["actions"]
    summary = {
        "repo_id": args.repo_id,
        "root": str(root),
        "input_file": str(args.input_file),
        "prompt": args.prompt,
        "fps": args.fps,
        "source_attrs": root_attrs,
        "episode_count": len(episodes),
        "frame_count": int(sum(ep["state"].shape[0] for ep in episodes)),
        "image_shape": list(episodes[0]["images"].shape[1:]),
        "state_dim": int(first_state.shape[-1]),
        "action_dim": int(first_actions.shape[-1]),
        "state_mapping": {
            "state_mode": args.state_mode,
            "eef_pos": "flat_state[21:24]",
            "eef_quat_wxyz": "flat_state[24:28]",
            "eef_axis_angle": "quat_to_axis_angle_wxyz(flat_state[24:28])",
            "gripper_qpos": "flat_state[28:30]",
            "object_state": "flat_state[32:55] when state_mode=privileged_stack",
            "step_fraction": "flat_state[55:56] when state_mode=privileged_stack",
        },
        "wrist_image_source": "duplicated_frontview_image",
        "episodes": [
            {
                "name": ep["name"],
                "frames": int(ep["state"].shape[0]),
                "attrs": ep["attrs"],
            }
            for ep in episodes
        ],
    }
    path = root / "robosuite_stack_lerobot_conversion_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(f"[summary] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--state-mode", choices=("proprio", "privileged_stack"), default="proprio")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser().resolve()
    if args.root is not None:
        args.root = args.root.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    episodes, root_attrs = load_episodes(
        args.input_file,
        max_episodes=args.max_episodes,
        state_mode=args.state_mode,
    )
    print(f"[load] episodes={len(episodes)} frames={sum(ep['state'].shape[0] for ep in episodes)}")
    root = write_dataset(episodes, args)
    write_summary(root, episodes, root_attrs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
