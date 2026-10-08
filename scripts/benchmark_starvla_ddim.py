#!/usr/bin/env python3
"""Benchmark StarVLA flow-step profiles on one fixed, resident-model input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("at least one latency is required")
    return {
        "calls": len(values),
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.pstdev(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_paired_profile(
    baseline_latencies: list[float],
    candidate_latencies: list[float],
    baseline_actions: list[np.ndarray],
    candidate_actions: list[np.ndarray],
) -> dict[str, float | int | list[float]]:
    lengths = {
        len(baseline_latencies),
        len(candidate_latencies),
        len(baseline_actions),
        len(candidate_actions),
    }
    if len(lengths) != 1 or not baseline_latencies:
        raise ValueError("paired profiles require equal non-empty sample counts")
    latency_deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_latencies, candidate_latencies)
    ]
    mean_delta = statistics.fmean(latency_deltas)
    if len(latency_deltas) > 1:
        half_width = 1.96 * statistics.stdev(latency_deltas) / math.sqrt(len(latency_deltas))
    else:
        half_width = 0.0
    action_mae = [
        float(np.mean(np.abs(candidate - baseline)))
        for baseline, candidate in zip(baseline_actions, candidate_actions)
    ]
    action_max_abs = [
        float(np.max(np.abs(candidate - baseline)))
        for baseline, candidate in zip(baseline_actions, candidate_actions)
    ]
    baseline_mean = statistics.fmean(baseline_latencies)
    return {
        "pairs": len(latency_deltas),
        "candidate_minus_baseline_mean_ms": mean_delta,
        "candidate_minus_baseline_normal_approx_95ci_ms": [
            mean_delta - half_width,
            mean_delta + half_width,
        ],
        "candidate_mean_latency_reduction_percent": (
            100.0 * -mean_delta / baseline_mean
        ),
        "action_mae_mean": statistics.fmean(action_mae),
        "action_mae_p95": _percentile(action_mae, 0.95),
        "action_max_abs_p95": _percentile(action_max_abs, 0.95),
    }


def _fixed_examples() -> list[dict]:
    rng = np.random.default_rng(20260902)
    images = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    state = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 2,
        dtype=np.float32,
    )
    return [
        {
            "image": images,
            "lang": "Build a stable tower with the available blocks.",
            "state": state[None, :],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 6])
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--measure-cycles", type=int, default=20)
    parser.add_argument("--unnorm-key", default="arx_x5")
    parser.add_argument("--seed-base", type=int, default=20260902)
    args = parser.parse_args()

    if len(set(args.steps)) != len(args.steps) or any(step <= 0 for step in args.steps):
        raise ValueError("--steps must contain distinct positive integers")
    if args.warmup_cycles < 1 or args.measure_cycles < 2:
        raise ValueError("use at least one warmup and two measured cycles")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    examples = _fixed_examples()
    wrapper = PolicyServerWrapper(
        ckpt_path=str(args.checkpoint.resolve()),
        device="cuda",
        use_bf16=True,
    )
    torch.cuda.reset_peak_memory_stats()

    warmups: dict[int, list[float]] = {step: [] for step in args.steps}
    measured: dict[int, list[float]] = {step: [] for step in args.steps}
    output_hashes: dict[int, list[str]] = {step: [] for step in args.steps}
    measured_actions: dict[int, list[np.ndarray]] = {
        step: [] for step in args.steps
    }

    def invoke(step: int, action_seed: int) -> tuple[float, str, np.ndarray]:
        started = time.perf_counter()
        result = wrapper.predict_action(
            examples=examples,
            unnorm_key=args.unnorm_key,
            do_sample=False,
            use_ddim=True,
            num_inference_steps=step,
            action_seed=action_seed,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        actions = np.ascontiguousarray(result["actions"])
        return latency_ms, hashlib.sha256(actions.tobytes()).hexdigest(), actions

    for cycle in range(args.warmup_cycles):
        order = args.steps if cycle % 2 == 0 else list(reversed(args.steps))
        for step in order:
            latency, _, _ = invoke(step, args.seed_base + cycle)
            warmups[step].append(latency)

    for cycle in range(args.measure_cycles):
        order = args.steps if cycle % 2 == 0 else list(reversed(args.steps))
        for step in order:
            latency, output_hash, actions = invoke(
                step, args.seed_base + args.warmup_cycles + cycle
            )
            measured[step].append(latency)
            output_hashes[step].append(output_hash)
            measured_actions[step].append(actions)

    gpu = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "carve.starvla.flow-step-microbenchmark.v1",
        "benchmark_design": {
            "resident_model": True,
            "fixed_multimodal_input": True,
            "paired_action_seed_per_cycle": True,
            "alternating_profile_order": True,
            "warmup_cycles": args.warmup_cycles,
            "measure_cycles": args.measure_cycles,
            "steps": args.steps,
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "precision": "bf16",
            "unnorm_key": args.unnorm_key,
        },
        "hardware": {
            "gpu": gpu.name,
            "gpu_total_memory_bytes": gpu.total_memory,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
        },
        "profiles": {
            str(step): {
                "warmup_ms": warmups[step],
                "steady_state": summarize_latencies(measured[step]),
                "latency_samples_ms": measured[step],
                "unique_action_hashes": len(set(output_hashes[step])),
            }
            for step in args.steps
        },
        "paired_comparisons": {
            f"{args.steps[0]}_vs_{candidate}": summarize_paired_profile(
                measured[args.steps[0]],
                measured[candidate],
                measured_actions[args.steps[0]],
                measured_actions[candidate],
            )
            for candidate in args.steps[1:]
        },
        "claim_boundary": (
            "This isolates resident-model inference latency on one fixed synthetic "
            "multimodal input. It does not measure task success or closed-loop latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
