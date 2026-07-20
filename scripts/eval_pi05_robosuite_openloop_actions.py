#!/usr/bin/env python3
"""Evaluate an OpenPI/pi0.5 robosuite policy against recorded teacher actions."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi.policies import policy_config
from openpi.training import config as openpi_config

from scripts.convert_robosuite_stack_hdf5_to_lerobot import build_openpi_state


DEFAULT_DATASET = "results/robosuite_stack_e2e_v2_20260611/stack_demos_100eps.hdf5"
DEFAULT_POLICY_DIR = "checkpoints/pi05_robosuite_stack_smoke/robosuite_stack_pi05_head_plus_300step/300"
DEFAULT_PROMPT = "stack the red cube on the green cube"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


def sort_episode_keys(keys: list[str]) -> list[str]:
    def key_fn(name: str) -> tuple[int, str]:
        tail = name.rsplit("_", 1)[-1]
        return (int(tail) if tail.isdigit() else 10**9, name)

    return sorted(keys, key=key_fn)


def load_policy(args: argparse.Namespace):
    cfg = openpi_config.get_config(args.policy_config)
    model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
    cfg = dataclasses.replace(cfg, model=model_cfg)
    return policy_config.create_trained_policy(
        cfg,
        Path(args.policy_dir).expanduser().resolve(),
        sample_kwargs={"num_steps": int(args.num_inference_steps)},
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )


def sample_indices(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--policy-config", default="pi05_robosuite_stack_smoke")
    parser.add_argument("--policy-dir", default=DEFAULT_POLICY_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--state-mode", choices=("proprio", "privileged_stack"), default="proprio")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--noise-mode", choices=("default", "zero", "fixed"), default="default")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def make_noise(args: argparse.Namespace, sample_index: int) -> np.ndarray | None:
    if args.noise_mode == "default":
        return None
    shape = (int(args.action_horizon), int(args.action_dim))
    if args.noise_mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(int(args.seed) + int(sample_index) * 9973)
    return rng.standard_normal(shape).astype(np.float32)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    policy_load_start = time.perf_counter()
    policy = load_policy(args)
    policy_load_sec = time.perf_counter() - policy_load_start
    rows: list[dict[str, Any]] = []
    infer_ms: list[float] = []

    with h5py.File(args.dataset, "r") as h5:
        group = h5["episodes"] if "episodes" in h5 else h5["data"]
        episode_keys = sort_episode_keys(list(group.keys()))[: int(args.episodes)]
        sample_index = 0
        for ep_key in episode_keys:
            ep = group[ep_key]
            images = ep["images"]
            states = build_openpi_state(ep["states"][:], state_mode=args.state_mode)
            actions = np.asarray(ep["actions"][:], dtype=np.float32)
            max_start = max(1, actions.shape[0] - int(args.action_horizon))
            for frame_idx in sample_indices(max_start, int(args.frames_per_episode), rng):
                image = np.asarray(images[int(frame_idx)], dtype=np.uint8)
                target = actions[int(frame_idx) : int(frame_idx) + int(args.action_horizon)]
                if target.shape[0] < int(args.action_horizon):
                    pad = np.repeat(target[-1:], int(args.action_horizon) - target.shape[0], axis=0)
                    target = np.concatenate([target, pad], axis=0)
                obs = {
                    "observation/image": image,
                    "observation/wrist_image": image.copy(),
                    "observation/state": states[int(frame_idx)],
                    "prompt": args.prompt,
                }
                infer_start = time.perf_counter()
                output = policy.infer(obs, noise=make_noise(args, sample_index))
                sample_index += 1
                wall_ms = (time.perf_counter() - infer_start) * 1000.0
                pred = np.asarray(output["actions"], dtype=np.float32)[: int(args.action_horizon), :7]
                infer_ms.append(float(output.get("policy_timing", {}).get("infer_ms", wall_ms)))
                err = pred - target[:, :7]
                rows.append(
                    {
                        "episode": ep_key,
                        "frame": int(frame_idx),
                        "first_action_l2": float(np.linalg.norm(err[0])),
                        "chunk_mse": float(np.mean(err**2)),
                        "chunk_mae": float(np.mean(np.abs(err))),
                        "target_first": target[0, :7],
                        "pred_first": pred[0],
                    }
                )
                print(
                    f"[openloop] {ep_key} frame={int(frame_idx)} "
                    f"first_l2={rows[-1]['first_action_l2']:.4f} "
                    f"chunk_mse={rows[-1]['chunk_mse']:.4f}",
                    flush=True,
                )

    summary = {
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "prompt": args.prompt,
        "policy_load_sec": policy_load_sec,
        "samples": len(rows),
        "state_mode": args.state_mode,
        "num_inference_steps": int(args.num_inference_steps),
        "noise_mode": args.noise_mode,
        "seed": int(args.seed),
        "first_action_l2_mean": float(np.mean([r["first_action_l2"] for r in rows])) if rows else None,
        "first_action_l2_median": float(np.median([r["first_action_l2"] for r in rows])) if rows else None,
        "chunk_mse_mean": float(np.mean([r["chunk_mse"] for r in rows])) if rows else None,
        "chunk_mae_mean": float(np.mean([r["chunk_mae"] for r in rows])) if rows else None,
        "policy_infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
        "policy_infer_ms_p95": float(np.percentile(infer_ms, 95)) if infer_ms else None,
        "rows": rows,
    }
    output_json = Path(args.output_json) if args.output_json else (
        Path(args.policy_dir).expanduser().resolve().parent / "openloop_action_eval.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2, default=json_default))
    print(f"[summary] {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
