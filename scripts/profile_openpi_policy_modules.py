#!/usr/bin/env python3
"""Profile OpenPI/pi0.5 checkpoint tensors for CAQ-Lite quantization planning.

This script intentionally starts from safetensors metadata instead of loading the
full policy into GPU memory. It is meant to answer a narrow question:
which tensors look like safe weight-only quantization candidates, and which
robot-control-sensitive tensors should stay in floating point?
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any


DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _role(name: str) -> str:
    lower = name.lower()
    if "gemma_expert" in lower or "action_expert" in lower or "action" in lower:
        return "action_or_expert"
    if any(token in lower for token in ("vision", "img", "image", "siglip", "vit")):
        return "vision"
    if any(token in lower for token in ("embed", "token")):
        return "embedding"
    if any(token in lower for token in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(token in lower for token in ("mlp", "ffn", "gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if any(token in lower for token in ("norm", "ln", "layernorm", "rms")):
        return "norm"
    if any(token in lower for token in ("head", "proj_out", "final")):
        return "head_or_output"
    if any(token in lower for token in ("paligemma", "gemma", "language", "llm")):
        return "language"
    return "other"


def _recommendation(name: str, shape: list[int], role: str) -> tuple[str, str]:
    lower = name.lower()
    if len(shape) < 2:
        return "keep_fp", "non-matrix tensor"
    if role in {"norm", "embedding", "head_or_output"}:
        return "keep_fp", f"{role} is usually sensitive under PTQ"
    if any(token in lower for token in ("action_out", "action_head", "final_action", "output_proj")):
        return "keep_fp", "direct action-output interface"
    if role == "action_or_expert":
        return "w8_candidate_guarded", "control/action expert path; quantize only behind criticality fallback"
    if role in {"mlp", "language", "vision", "attention", "other"}:
        return "w8_candidate", "matrix weight likely suitable for conservative weight-only PTQ"
    return "inspect", "unclassified matrix tensor"


def _shape_dtype(handle: Any, key: str) -> tuple[list[int], str]:
    view = handle.get_slice(key)
    shape = list(view.get_shape())
    dtype = "unknown"
    if hasattr(view, "get_dtype"):
        dtype = str(view.get_dtype())
    return shape, dtype


def profile_checkpoint(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("safetensors is required: pip install safetensors") from exc

    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise SystemExit(f"Cannot find {weight_path}")

    tensors: list[dict[str, Any]] = []
    role_stats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"tensors": 0, "params": 0})
    rec_stats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"tensors": 0, "params": 0})
    total_params = 0
    estimated_bytes = 0

    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            shape, dtype = _shape_dtype(handle, key)
            params = _numel(shape)
            dtype_key = dtype.upper().replace("TORCH.", "")
            bytes_per = DTYPE_BYTES.get(dtype_key, 0)
            role = _role(key)
            recommendation, reason = _recommendation(key, shape, role)
            total_params += params
            estimated_bytes += params * bytes_per
            role_stats[role]["tensors"] += 1
            role_stats[role]["params"] += params
            rec_stats[recommendation]["tensors"] += 1
            rec_stats[recommendation]["params"] += params
            tensors.append(
                {
                    "name": key,
                    "shape": shape,
                    "dtype": dtype,
                    "params": params,
                    "role": role,
                    "recommendation": recommendation,
                    "reason": reason,
                }
            )

    tensors.sort(key=lambda item: item["params"], reverse=True)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "weight_path": str(weight_path),
        "total_tensors": len(tensors),
        "total_params": total_params,
        "estimated_weight_gb": estimated_bytes / (1024**3) if estimated_bytes else None,
        "role_stats": dict(role_stats),
        "recommendation_stats": dict(rec_stats),
        "largest_tensors": tensors[:50],
        "all_tensors": tensors,
    }


def _format_count(params: int) -> str:
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    if params >= 1_000:
        return f"{params / 1_000:.2f}K"
    return str(params)


def write_markdown(profile: dict[str, Any], output: pathlib.Path) -> None:
    lines = [
        "# OpenPI pi0.5 Quantization Profile",
        "",
        f"- Checkpoint: `{profile['checkpoint_dir']}`",
        f"- Tensors: `{profile['total_tensors']}`",
        f"- Parameters: `{_format_count(int(profile['total_params']))}`",
    ]
    if profile["estimated_weight_gb"] is not None:
        lines.append(f"- Estimated checkpoint tensor size: `{profile['estimated_weight_gb']:.2f} GB`")
    lines.extend(["", "## Recommendation Summary", ""])
    lines.append("| Recommendation | Tensors | Params |")
    lines.append("|---|---:|---:|")
    for rec, stats in sorted(profile["recommendation_stats"].items()):
        lines.append(f"| `{rec}` | `{stats['tensors']}` | `{_format_count(stats['params'])}` |")
    lines.extend(["", "## Role Summary", ""])
    lines.append("| Role | Tensors | Params |")
    lines.append("|---|---:|---:|")
    for role, stats in sorted(profile["role_stats"].items()):
        lines.append(f"| `{role}` | `{stats['tensors']}` | `{_format_count(stats['params'])}` |")
    lines.extend(["", "## Largest Tensors", ""])
    lines.append("| Name | Shape | DType | Params | Role | Recommendation |")
    lines.append("|---|---|---|---:|---|---|")
    for item in profile["largest_tensors"]:
        shape = "x".join(str(dim) for dim in item["shape"])
        lines.append(
            f"| `{item['name']}` | `{shape}` | `{item['dtype']}` | "
            f"`{_format_count(item['params'])}` | `{item['role']}` | `{item['recommendation']}` |"
        )
    lines.extend(
        [
            "",
            "## CAQ-Lite Interpretation",
            "",
            "- `w8_candidate`: first candidates for conservative weight-only INT8 smoke tests.",
            "- `w8_candidate_guarded`: can be tested only behind criticality-aware fallback because it may affect action generation.",
            "- `keep_fp`: keep in floating point for the first paper version.",
            "- This profile is a planning artifact; task-level rollout is still required before claiming quantization results.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
    )
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--md-output", type=pathlib.Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    profile = profile_checkpoint(args.checkpoint_dir)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if args.md_output:
        write_markdown(profile, args.md_output)

    print(f"checkpoint={profile['checkpoint_dir']}")
    print(f"tensors={profile['total_tensors']} params={_format_count(int(profile['total_params']))}")
    if profile["estimated_weight_gb"] is not None:
        print(f"estimated_weight_gb={profile['estimated_weight_gb']:.2f}")
    print("recommendations:")
    for rec, stats in sorted(profile["recommendation_stats"].items()):
        print(f"  {rec}: tensors={stats['tensors']} params={_format_count(stats['params'])}")
    print("largest:")
    for item in profile["largest_tensors"][: max(0, args.top_k)]:
        shape = "x".join(str(dim) for dim in item["shape"])
        print(
            f"  {item['name']} shape={shape} params={_format_count(item['params'])} "
            f"role={item['role']} rec={item['recommendation']}"
        )


if __name__ == "__main__":
    main()
