#!/usr/bin/env python3
"""Collect real robosuite/MuJoCo Stack demonstrations from the scripted teacher."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robosuite_stack_policy import extract_image, extract_state, make_step_feature
from scripts.run_robosuite_deployment_pilot import ScriptedChunkPolicy, make_env, try_apply_nudge


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


def collect_episode(
    *,
    env: Any,
    seed: int,
    horizon: int,
    chunk_size: int,
    perturb: bool,
    perturb_step: int,
    perturb_xy: float,
    camera_name: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass

    obs = env.reset()
    teacher = ScriptedChunkPolicy(task="Stack", chunk_size=chunk_size)
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    step_features: list[np.ndarray] = []
    perturbation_event: dict[str, Any] | None = None

    for step in range(horizon):
        if perturb and perturbation_event is None and step == perturb_step:
            perturbation_event = try_apply_nudge(env, "Stack", rng, perturb_xy)
            if perturbation_event.get("applied"):
                obs = env._get_observations()

        chunk = np.asarray(teacher.infer_chunk(obs, step), dtype=np.float32)
        action = chunk[0].astype(np.float32)
        images.append(extract_image(obs, camera_name=camera_name))
        state = extract_state(obs)
        states.append(np.concatenate([state, make_step_feature(step, horizon)], axis=0))
        actions.append(action)
        action_chunks.append(chunk)
        step_features.append(make_step_feature(step, horizon))

        obs, reward, done, info = env.step(action)
        if hasattr(env, "_check_success") and env._check_success():
            return {
                "success": True,
                "steps": step + 1,
                "images": np.stack(images, axis=0),
                "states": np.stack(states, axis=0),
                "actions": np.stack(actions, axis=0),
                "action_chunks": np.stack(action_chunks, axis=0),
                "step_features": np.stack(step_features, axis=0),
                "perturbation_event": perturbation_event,
            }

    return {
        "success": False,
        "steps": horizon,
        "images": np.stack(images, axis=0) if images else np.empty((0, 0, 0, 3), dtype=np.uint8),
        "states": np.stack(states, axis=0) if states else np.empty((0, 0), dtype=np.float32),
        "actions": np.stack(actions, axis=0) if actions else np.empty((0, 7), dtype=np.float32),
        "action_chunks": np.stack(action_chunks, axis=0) if action_chunks else np.empty((0, chunk_size, 7), dtype=np.float32),
        "step_features": np.stack(step_features, axis=0) if step_features else np.empty((0, 1), dtype=np.float32),
        "perturbation_event": perturbation_event,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-hdf5", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--target-successes", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=45)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--horizon", type=int, default=430)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", default="frontview")
    parser.add_argument("--perturb-prob", type=float, default=0.25)
    parser.add_argument("--perturb-step", type=int, default=260)
    parser.add_argument("--perturb-xy", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_hdf5)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        "Stack",
        int(args.horizon),
        record_video=True,
        camera_name=str(args.camera_name),
        camera_height=int(args.image_size),
        camera_width=int(args.image_size),
    )

    accepted = 0
    attempts = 0
    total_frames = 0
    start = time.perf_counter()
    per_episode: list[dict[str, Any]] = []
    try:
        with h5py.File(output_path, "w") as h5:
            h5.attrs["task"] = "Stack"
            h5.attrs["simulator"] = "robosuite"
            h5.attrs["physics"] = "MuJoCo"
            h5.attrs["robot"] = "Panda"
            h5.attrs["controller"] = "OSC_POSE"
            h5.attrs["chunk_size"] = int(args.chunk_size)
            h5.attrs["horizon"] = int(args.horizon)
            h5.attrs["image_size"] = int(args.image_size)
            h5.attrs["camera_name"] = str(args.camera_name)
            h5.attrs["state_keys"] = json.dumps(["robot0_proprio-state", "object-state", "step_fraction"])
            episodes_group = h5.create_group("episodes")

            while accepted < int(args.target_successes) and attempts < int(args.max_attempts):
                ep_seed = int(args.seed) + attempts
                perturb = random.Random(ep_seed).random() < float(args.perturb_prob)
                result = collect_episode(
                    env=env,
                    seed=ep_seed,
                    horizon=int(args.horizon),
                    chunk_size=int(args.chunk_size),
                    perturb=perturb,
                    perturb_step=int(args.perturb_step),
                    perturb_xy=float(args.perturb_xy),
                    camera_name=str(args.camera_name),
                )
                attempts += 1
                row = {
                    "attempt": attempts,
                    "seed": ep_seed,
                    "success": bool(result["success"]),
                    "steps": int(result["steps"]),
                    "perturb": bool(perturb),
                    "perturbation_event": result["perturbation_event"],
                }
                if result["success"]:
                    group = episodes_group.create_group(f"episode_{accepted:04d}")
                    group.create_dataset("images", data=result["images"], compression="gzip", compression_opts=1)
                    group.create_dataset("states", data=result["states"], compression="gzip", compression_opts=1)
                    group.create_dataset("actions", data=result["actions"], compression="gzip", compression_opts=1)
                    group.create_dataset("action_chunks", data=result["action_chunks"], compression="gzip", compression_opts=1)
                    group.attrs["seed"] = ep_seed
                    group.attrs["steps"] = int(result["steps"])
                    group.attrs["perturb"] = bool(perturb)
                    group.attrs["perturbation_event"] = json.dumps(result["perturbation_event"], default=_json_default)
                    accepted += 1
                    total_frames += int(result["steps"])
                    row["accepted_index"] = accepted - 1
                per_episode.append(row)
                print(
                    f"[collect] attempt={attempts} accepted={accepted}/{args.target_successes} "
                    f"success={result['success']} steps={result['steps']} perturb={perturb}",
                    flush=True,
                )
    finally:
        env.close()

    summary = {
        "output_hdf5": str(output_path),
        "target_successes": int(args.target_successes),
        "accepted_successes": accepted,
        "attempts": attempts,
        "total_frames": total_frames,
        "horizon": int(args.horizon),
        "chunk_size": int(args.chunk_size),
        "image_size": int(args.image_size),
        "camera_name": str(args.camera_name),
        "perturb_prob": float(args.perturb_prob),
        "wall_sec": time.perf_counter() - start,
        "episodes": per_episode,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
