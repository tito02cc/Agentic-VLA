#!/usr/bin/env python3
"""bitsandbytes W8 smoke test on real OpenPI checkpoint tensors.

This does not load the full pi0.5 policy. It loads selected Linear weights from
model.safetensors, wraps them with bitsandbytes Linear8bitLt, and compares random
forward outputs with a BF16/FP32 torch Linear reference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any


DEFAULT_WEIGHT_PATTERNS = [
    "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
]


def _pick_weights(checkpoint_dir: pathlib.Path, patterns: list[str], max_params: int) -> list[dict[str, Any]]:
    from safetensors import safe_open

    path = checkpoint_dir / "model.safetensors"
    selected = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for pattern in patterns:
            matches = [key for key in sorted(keys) if pattern in key]
            for key in matches:
                shape = list(handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    continue
                params = int(shape[0] * shape[1])
                if params <= max_params:
                    selected.append({"name": key, "shape": shape, "params": params})
                    break
    return selected


def _bench_linear(weight, *, batch: int, seq: int, repeats: int) -> dict[str, Any]:
    import torch
    import bitsandbytes as bnb

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes Linear8bitLt smoke")

    out_features, in_features = weight.shape
    device = torch.device("cuda")
    x = torch.randn(batch, seq, in_features, dtype=torch.float16, device=device)
    fp = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.float16)
    with torch.no_grad():
        fp.weight.copy_(weight.to(dtype=torch.float16, device=device))

    q = bnb.nn.Linear8bitLt(in_features, out_features, bias=False, has_fp16_weights=False).to(device)
    with torch.no_grad():
        q.weight = bnb.nn.Int8Params(
            weight.to(dtype=torch.float16, device=device).contiguous(),
            requires_grad=False,
            has_fp16_weights=False,
        )

    # Warmup and quantization state initialization.
    with torch.no_grad():
        y_fp = fp(x)
        y_q = q(x)
    torch.cuda.synchronize()

    fp_times = []
    q_times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            y_fp = fp(x)
            torch.cuda.synchronize()
            fp_times.append((time.perf_counter() - start) * 1000)

        for _ in range(repeats):
            start = time.perf_counter()
            y_q = q(x)
            torch.cuda.synchronize()
            q_times.append((time.perf_counter() - start) * 1000)

    diff = (y_fp.float() - y_q.float()).abs()
    denom = y_fp.float().abs().mean().clamp_min(1e-6)
    return {
        "shape": [int(out_features), int(in_features)],
        "input_shape": [batch, seq, int(in_features)],
        "fp_ms_mean": sum(fp_times) / len(fp_times),
        "q_ms_mean": sum(q_times) / len(q_times),
        "speedup": (sum(fp_times) / len(fp_times)) / max(sum(q_times) / len(q_times), 1e-9),
        "abs_err_mean": float(diff.mean().item()),
        "abs_err_p95": float(torch.quantile(diff.flatten(), 0.95).item()),
        "rel_err_mean": float((diff.mean() / denom).item()),
        "gpu_mem_peak_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
    )
    parser.add_argument("--weight", action="append", dest="weights", default=[])
    parser.add_argument("--max-params", type=int, default=5_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    patterns = args.weights or DEFAULT_WEIGHT_PATTERNS
    selected = _pick_weights(args.checkpoint_dir, patterns, args.max_params)
    if not selected:
        raise SystemExit("No matching 2D weights under max-params")

    results = []
    path = args.checkpoint_dir / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for item in selected:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            weight = handle.get_tensor(item["name"])
            try:
                metrics = _bench_linear(weight, batch=args.batch, seq=args.seq, repeats=args.repeats)
                status = "ok"
                error = None
            except Exception as exc:  # pragma: no cover - backend dependent
                metrics = {}
                status = "failed"
                error = repr(exc)
            results.append(
                {
                    "name": item["name"],
                    "params": item["params"],
                    "status": status,
                    "error": error,
                    **metrics,
                }
            )

    report = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "backend": "bitsandbytes.Linear8bitLt",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "results": results,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
