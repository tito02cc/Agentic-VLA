#!/usr/bin/env python3
"""Sweep pi0.5 inference steps on recorded robosuite Stack frames.

The script loads a PyTorch OpenPI/pi0.5 policy once and mutates the sampling
step count between evaluations.  This keeps the latency-quality sweep focused on
runtime inference cost instead of repeatedly paying checkpoint load time.
"""

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
from scripts.eval_pi05_robosuite_openloop_actions import (
    DEFAULT_DATASET,
    DEFAULT_POLICY_DIR,
    DEFAULT_PROMPT,
    json_default,
    make_noise,
    sample_indices,
    sort_episode_keys,
)


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
    parser.add_argument("--num-inference-steps", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--noise-mode", choices=("default", "zero", "fixed"), default="fixed")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_policy(args: argparse.Namespace, initial_steps: int):
    cfg = openpi_config.get_config(args.policy_config)
    model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
    cfg = dataclasses.replace(cfg, model=model_cfg)
    return policy_config.create_trained_policy(
        cfg,
        Path(args.policy_dir).expanduser().resolve(),
        sample_kwargs={"num_steps": int(initial_steps)},
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )


def set_policy_steps(policy: Any, steps: int) -> None:
    if not hasattr(policy, "_sample_kwargs"):
        raise AttributeError("OpenPI policy object does not expose _sample_kwargs")
    policy._sample_kwargs["num_steps"] = int(steps)


def collect_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    samples: list[dict[str, Any]] = []
    with h5py.File(args.dataset, "r") as h5:
        group = h5["episodes"] if "episodes" in h5 else h5["data"]
        episode_keys = sort_episode_keys(list(group.keys()))[: int(args.episodes)]
        for ep_key in episode_keys:
            ep = group[ep_key]
            images = ep["images"]
            states = build_openpi_state(ep["states"][:], state_mode=args.state_mode)
            actions = np.asarray(ep["actions"][:], dtype=np.float32)
            max_start = max(1, actions.shape[0] - int(args.action_horizon))
            for frame_idx in sample_indices(max_start, int(args.frames_per_episode), rng):
                target = actions[int(frame_idx) : int(frame_idx) + int(args.action_horizon)]
                if target.shape[0] < int(args.action_horizon):
                    pad = np.repeat(target[-1:], int(args.action_horizon) - target.shape[0], axis=0)
                    target = np.concatenate([target, pad], axis=0)
                samples.append(
                    {
                        "episode": ep_key,
                        "frame": int(frame_idx),
                        "image": np.asarray(images[int(frame_idx)], dtype=np.uint8),
                        "state": np.asarray(states[int(frame_idx)], dtype=np.float32),
                        "target": target.astype(np.float32),
                    }
                )
    return samples


def infer_one(policy: Any, args: argparse.Namespace, sample: dict[str, Any], sample_index: int) -> tuple[np.ndarray, float]:
    obs = {
        "observation/image": sample["image"],
        "observation/wrist_image": sample["image"].copy(),
        "observation/state": sample["state"],
        "prompt": args.prompt,
    }
    infer_start = time.perf_counter()
    output = policy.infer(obs, noise=make_noise(args, sample_index))
    wall_ms = (time.perf_counter() - infer_start) * 1000.0
    pred = np.asarray(output["actions"], dtype=np.float32)[: int(args.action_horizon), :7]
    infer_ms = float(output.get("policy_timing", {}).get("infer_ms", wall_ms))
    return pred, infer_ms


def evaluate_steps(policy: Any, args: argparse.Namespace, samples: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    set_policy_steps(policy, steps)
    for warmup_idx in range(max(0, int(args.warmup_calls))):
        infer_one(policy, args, samples[0], 1_000_000 + int(steps) * 100 + warmup_idx)

    rows: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    per_dim_abs: list[np.ndarray] = []
    first_gripper_correct = 0
    chunk_gripper_correct = 0
    chunk_gripper_total = 0

    for sample_index, sample in enumerate(samples):
        pred, one_infer_ms = infer_one(policy, args, sample, sample_index)
        target = np.asarray(sample["target"], dtype=np.float32)[:, :7]
        err = pred - target
        infer_ms.append(one_infer_ms)
        per_dim_abs.append(np.mean(np.abs(err), axis=0))

        pred_gripper = np.where(pred[:, 6] >= 0.0, 1.0, -1.0)
        target_gripper = np.where(target[:, 6] >= 0.0, 1.0, -1.0)
        first_gripper_correct += int(pred_gripper[0] == target_gripper[0])
        chunk_gripper_correct += int(np.sum(pred_gripper == target_gripper))
        chunk_gripper_total += int(target_gripper.size)

        row = {
            "episode": sample["episode"],
            "frame": int(sample["frame"]),
            "first_action_l2": float(np.linalg.norm(err[0])),
            "chunk_mse": float(np.mean(err**2)),
            "chunk_mae": float(np.mean(np.abs(err))),
            "first_gripper_match": bool(pred_gripper[0] == target_gripper[0]),
            "target_first": target[0],
            "pred_first": pred[0],
        }
        rows.append(row)

    return {
        "num_inference_steps": int(steps),
        "samples": len(rows),
        "first_action_l2_mean": float(np.mean([r["first_action_l2"] for r in rows])) if rows else None,
        "first_action_l2_median": float(np.median([r["first_action_l2"] for r in rows])) if rows else None,
        "chunk_mse_mean": float(np.mean([r["chunk_mse"] for r in rows])) if rows else None,
        "chunk_mae_mean": float(np.mean([r["chunk_mae"] for r in rows])) if rows else None,
        "per_dim_mae_mean": np.mean(per_dim_abs, axis=0) if per_dim_abs else None,
        "first_gripper_sign_accuracy": first_gripper_correct / len(rows) if rows else None,
        "chunk_gripper_sign_accuracy": chunk_gripper_correct / chunk_gripper_total if chunk_gripper_total else None,
        "policy_infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
        "policy_infer_ms_p50": float(np.percentile(infer_ms, 50)) if infer_ms else None,
        "policy_infer_ms_p95": float(np.percentile(infer_ms, 95)) if infer_ms else None,
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    steps_list = [int(step) for step in args.num_inference_steps]
    samples = collect_samples(args)
    if not samples:
        raise RuntimeError("no evaluation samples collected")

    policy_load_start = time.perf_counter()
    policy = load_policy(args, initial_steps=steps_list[0])
    policy_load_sec = time.perf_counter() - policy_load_start

    sweep: list[dict[str, Any]] = []
    for steps in steps_list:
        step_start = time.perf_counter()
        result = evaluate_steps(policy, args, samples, steps)
        result["wall_sec"] = time.perf_counter() - step_start
        sweep.append(result)
        print(
            f"[sweep] steps={steps} first_l2={result['first_action_l2_mean']:.4f} "
            f"mse={result['chunk_mse_mean']:.4f} infer={result['policy_infer_ms_mean']:.1f}ms "
            f"grip={result['first_gripper_sign_accuracy']:.2f}",
            flush=True,
        )

    summary = {
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "prompt": args.prompt,
        "policy_load_sec": policy_load_sec,
        "state_mode": args.state_mode,
        "noise_mode": args.noise_mode,
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "frames_per_episode": int(args.frames_per_episode),
        "action_horizon": int(args.action_horizon),
        "warmup_calls": int(args.warmup_calls),
        "sweep": sweep,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(f"[summary] {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
