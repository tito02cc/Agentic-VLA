#!/usr/bin/env python3
"""Decompose OpenVLA BF16 action latency into vision, prefill, and decode work."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_vla.runtime.policies import (  # noqa: E402
    HuggingFaceOpenVlaPolicy,
    OpenVlaLoadConfig,
    align_openvla_action_token_mask,
)


DEFAULT_CHECKPOINT = ROOT / "checkpoints/openvla-7b-finetuned-libero-10"
DEFAULT_REPLAY = ROOT / "results/carve_semantic_precision/temporal_pairs_t89/manifest.json"
DEFAULT_OUTPUT = (
    ROOT / "results/carve_optimize/openvla_4090_20260718/openvla_critical_path.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay-manifest", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--measured-calls", type=int, default=10)
    parser.add_argument("--compile-language-model", action="store_true")
    parser.add_argument("--align-action-token-mask", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_replay(path: pathlib.Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in manifest["records"]:
        snapshot_id = str(record["snapshot_id"])
        if snapshot_id in seen or record["severity"] != "no_op":
            continue
        image_path = ROOT / str(record["after_image"])
        samples.append(
            {
                "episode_id": snapshot_id,
                "instruction": str(record["instruction"]),
                "image": Image.open(image_path).convert("RGB"),
            }
        )
        seen.add(snapshot_id)
    if not samples:
        raise SystemExit(f"No no-op replay observations found in {path}")
    return samples


class CudaModuleTimer:
    """Record sequential CUDA time for selected forward modules."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._pending: dict[str, list[Any]] = defaultdict(list)
        self._pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self._handles: list[Any] = []

    def add(self, name: str, module: Any) -> None:
        def before(_module: Any, _args: Any) -> None:
            event = self._torch.cuda.Event(enable_timing=True)
            event.record()
            self._pending[name].append(event)

        def after(_module: Any, _args: Any, _output: Any) -> None:
            start = self._pending[name].pop()
            end = self._torch.cuda.Event(enable_timing=True)
            end.record()
            self._pairs[name].append((start, end))

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def reset(self) -> None:
        self._pending.clear()
        self._pairs.clear()

    def elapsed(self, name: str) -> list[float]:
        return [float(start.elapsed_time(end)) for start, end in self._pairs[name]]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def summarize(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records]
    return {
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def main() -> int:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise SystemExit("Critical-path decomposition requires a CUDA device")

    import torch

    samples = load_replay(args.replay_manifest.resolve())
    policy = HuggingFaceOpenVlaPolicy.from_pretrained(
        OpenVlaLoadConfig(
            checkpoint=args.checkpoint_dir.resolve(),
            device=args.device,
            precision="bf16",
            align_action_token_mask=args.align_action_token_mask,
            attn_implementation="sdpa",
        )
    )
    compile_wrap_seconds = 0.0
    if args.compile_language_model:
        compile_started = time.perf_counter()
        policy.model.language_model = torch.compile(
            policy.model.language_model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=args.compile_dynamic,
        )
        compile_wrap_seconds = time.perf_counter() - compile_started

    warmup_wall_seconds: list[float] = []
    for index in range(args.warmup_calls):
        sample = samples[index % len(samples)]
        warmup_started = time.perf_counter()
        policy.infer({"image": sample["image"], "instruction": sample["instruction"]})
        warmup_wall_seconds.append(time.perf_counter() - warmup_started)

    timer = CudaModuleTimer(torch)
    timer.add("vision", policy.model.vision_backbone)
    timer.add("projector", policy.model.projector)
    timer.add("language_model", policy.model.language_model)
    records: list[dict[str, Any]] = []
    try:
        for index in range(args.measured_calls):
            sample = samples[index % len(samples)]
            preprocess_started = time.perf_counter()
            image = policy._prepare_image(sample["image"])
            inputs = policy._processor(policy._prompt(sample["instruction"]), image)
            if args.align_action_token_mask:
                align_openvla_action_token_mask(inputs, torch_module=torch)
            inputs = inputs.to(args.device, dtype=torch.bfloat16)
            torch.cuda.synchronize(args.device)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

            timer.reset()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            total_start.record()
            with torch.inference_mode():
                action = policy.model.predict_action(
                    **inputs,
                    unnorm_key=policy.config.unnorm_key,
                    do_sample=False,
                )
            total_end.record()
            torch.cuda.synchronize(args.device)
            wall_ms = (time.perf_counter() - wall_started) * 1000.0

            language_calls = timer.elapsed("language_model")
            if not language_calls:
                raise RuntimeError("No language-model calls were recorded")
            records.append(
                {
                    "episode_id": sample["episode_id"],
                    "preprocess_and_transfer_ms": preprocess_ms,
                    "model_wall_ms": wall_ms,
                    "model_cuda_span_ms": float(total_start.elapsed_time(total_end)),
                    "vision_ms": sum(timer.elapsed("vision")),
                    "projector_ms": sum(timer.elapsed("projector")),
                    "prefill_ms": language_calls[0],
                    "decode_ms": sum(language_calls[1:]),
                    "decode_calls": max(0, len(language_calls) - 1),
                    "language_calls": len(language_calls),
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                }
            )
    finally:
        timer.close()

    metric_keys = (
        "preprocess_and_transfer_ms",
        "model_wall_ms",
        "model_cuda_span_ms",
        "vision_ms",
        "projector_ms",
        "prefill_ms",
        "decode_ms",
    )
    summary = {key: summarize(records, key) for key in metric_keys}
    mean_total = summary["model_cuda_span_ms"]["mean_ms"]
    summary["mean_cuda_share_percent"] = {
        key: 100.0 * summary[key]["mean_ms"] / mean_total
        for key in ("vision_ms", "projector_ms", "prefill_ms", "decode_ms")
    }
    summary["mean_language_calls"] = float(
        statistics.fmean(record["language_calls"] for record in records)
    )
    summary["mean_decode_calls"] = float(
        statistics.fmean(record["decode_calls"] for record in records)
    )

    payload = {
        "schema_version": 1,
        "diagnostic": "openvla_bf16_cuda_critical_path_v1",
        "device": args.device,
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "attention": "sdpa",
        "align_action_token_mask": args.align_action_token_mask,
        "compile_language_model": args.compile_language_model,
        "compile_mode": args.compile_mode if args.compile_language_model else None,
        "compile_dynamic": args.compile_dynamic if args.compile_language_model else None,
        "compile_wrap_seconds": compile_wrap_seconds,
        "warmup_wall_seconds": warmup_wall_seconds,
        "warmup_calls": args.warmup_calls,
        "measured_calls": args.measured_calls,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
