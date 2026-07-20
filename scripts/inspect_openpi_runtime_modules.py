#!/usr/bin/env python3
"""Inspect runtime PyTorch modules for OpenPI/pi0.5 quantization smoke tests.

Unlike profile_openpi_policy_modules.py, this script instantiates the policy and
walks the actual torch.nn.Module tree. It is intentionally diagnostic: it does
not mutate the model or run LIBERO rollouts.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
from typing import Any


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


def _recommendation(name: str, module_type: str, params: int, role: str) -> tuple[str, str]:
    lower = name.lower()
    if module_type != "Linear":
        return "keep_fp", "only Linear modules are considered for first-pass weight-only quantization"
    if params == 0:
        return "skip", "no parameters"
    if role in {"embedding", "norm", "head_or_output"}:
        return "keep_fp", f"{role} is sensitive under first-pass PTQ"
    if any(token in lower for token in ("action_out_proj", "lm_head", "output")):
        return "keep_fp", "direct output interface"
    if role == "action_or_expert":
        return "w8_candidate_guarded", "action expert path; use only behind criticality fallback"
    return "w8_candidate", "Linear module suitable for conservative W8 smoke"


def _count_params(module: Any) -> int:
    return int(sum(param.numel() for param in module.parameters(recurse=False)))


def _module_dtype(module: Any) -> str:
    for param in module.parameters(recurse=False):
        return str(param.dtype).replace("torch.", "")
    return "-"


def _format_count(params: int) -> str:
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    if params >= 1_000:
        return f"{params / 1_000:.2f}K"
    return str(params)


def inspect_policy(config_name: str, checkpoint_dir: pathlib.Path, device: str) -> dict[str, Any]:
    import torch
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir, pytorch_device=device)
    model = policy._model  # noqa: SLF001 - diagnostic script.

    module_rows: list[dict[str, Any]] = []
    rec_stats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"modules": 0, "params": 0})
    role_stats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"modules": 0, "params": 0})
    type_stats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"modules": 0, "params": 0})

    for name, module in model.named_modules():
        if not name:
            continue
        module_type = type(module).__name__
        params = _count_params(module)
        role = _role(name)
        rec, reason = _recommendation(name, module_type, params, role)
        dtype = _module_dtype(module)
        row = {
            "name": name,
            "type": module_type,
            "params": params,
            "dtype": dtype,
            "role": role,
            "recommendation": rec,
            "reason": reason,
        }
        module_rows.append(row)
        rec_stats[rec]["modules"] += 1
        rec_stats[rec]["params"] += params
        role_stats[role]["modules"] += 1
        role_stats[role]["params"] += params
        type_stats[module_type]["modules"] += 1
        type_stats[module_type]["params"] += params

    module_rows.sort(key=lambda item: item["params"], reverse=True)
    return {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "total_modules": len(module_rows),
        "recommendation_stats": dict(rec_stats),
        "role_stats": dict(role_stats),
        "type_stats": dict(type_stats),
        "largest_modules": module_rows[:100],
        "all_modules": module_rows,
    }


def write_markdown(report: dict[str, Any], output: pathlib.Path) -> None:
    lines = [
        "# OpenPI pi0.5 Runtime Module Inspection",
        "",
        f"- Config: `{report['config_name']}`",
        f"- Checkpoint: `{report['checkpoint_dir']}`",
        f"- Device: `{report['device']}`",
        f"- Torch: `{report['torch_version']}`",
        f"- CUDA available: `{report['cuda_available']}`",
        f"- Modules: `{report['total_modules']}`",
        "",
        "## Recommendation Summary",
        "",
        "| Recommendation | Modules | Params |",
        "|---|---:|---:|",
    ]
    for rec, stats in sorted(report["recommendation_stats"].items()):
        lines.append(f"| `{rec}` | `{stats['modules']}` | `{_format_count(stats['params'])}` |")
    lines.extend(["", "## Role Summary", "", "| Role | Modules | Params |", "|---|---:|---:|"])
    for role, stats in sorted(report["role_stats"].items()):
        lines.append(f"| `{role}` | `{stats['modules']}` | `{_format_count(stats['params'])}` |")
    lines.extend(["", "## Largest Modules", "", "| Name | Type | DType | Params | Role | Recommendation |", "|---|---|---|---:|---|---|"])
    for item in report["largest_modules"][:50]:
        lines.append(
            f"| `{item['name']}` | `{item['type']}` | `{item['dtype']}` | "
            f"`{_format_count(item['params'])}` | `{item['role']}` | `{item['recommendation']}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--md-output", type=pathlib.Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
    report = inspect_policy(args.config, args.checkpoint_dir, args.device)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.md_output:
        write_markdown(report, args.md_output)

    print(f"config={report['config_name']} device={report['device']} modules={report['total_modules']}")
    print("recommendations:")
    for rec, stats in sorted(report["recommendation_stats"].items()):
        print(f"  {rec}: modules={stats['modules']} params={_format_count(stats['params'])}")
    print("largest:")
    for item in report["largest_modules"][: max(0, args.top_k)]:
        print(
            f"  {item['name']} type={item['type']} params={_format_count(item['params'])} "
            f"role={item['role']} rec={item['recommendation']}"
        )


if __name__ == "__main__":
    main()
