#!/usr/bin/env python3
"""Run a low-memory pi0.5 head-only smoke on robosuite Stack data.

This script is intentionally narrower than OpenPI's full fine-tuning trainer:

- it uses the normal OpenPI config/data pipeline;
- it loads the local pi0.5 base PyTorch checkpoint;
- it freezes the backbone and only trains selected lightweight heads;
- it runs a tiny number of steps to validate the integration on a single RTX
  4090;
- it runs one action-generation call and saves a small summary.

Full-model Adam fine-tuning OOMs on a 24GB 4090 because optimizer states push
memory over the device limit.  Use this smoke locally, then use LoRA/A100 for
real adaptation runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import jax
import safetensors.torch
import torch

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as normalize
import openpi.training.config as openpi_config
import openpi.training.data_loader as openpi_data


DEFAULT_CONFIG = "pi05_robosuite_stack_smoke"
DEFAULT_EXP_NAME = "robosuite_stack_pi05_base_head_only_smoke_2step"
DEFAULT_WEIGHT_PATH = "/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_isaaclab"


def json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def move_to_device(x, device: torch.device):
    if torch.is_tensor(x):
        if x.is_floating_point():
            return x.to(device=device, dtype=torch.float32)
        return x.to(device=device)
    return x


def set_trainable(model: torch.nn.Module, trainable_substrings: tuple[str, ...]) -> dict[str, Any]:
    trainable_params = 0
    frozen_params = 0
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        is_trainable = any(pattern in name for pattern in trainable_substrings)
        param.requires_grad_(is_trainable)
        count = param.numel()
        if is_trainable:
            trainable_params += count
            trainable_names.append(name)
        else:
            frozen_params += count
    return {
        "trainable_substrings": trainable_substrings,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_param_names": trainable_names,
    }


def gpu_memory() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--weight-path", default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trainable-substring", action="append", default=["action_out_proj"])
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-action-dim", type=int, default=7)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-full-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
    args = parse_args()

    cfg = dataclasses.replace(
        openpi_config.get_config(args.config_name),
        exp_name=args.exp_name,
        num_train_steps=args.steps,
        batch_size=1,
        num_workers=0,
        overwrite=args.overwrite,
        pytorch_weight_path=args.weight_path,
        wandb_enabled=False,
    )
    ckpt_dir = cfg.checkpoint_dir
    if args.overwrite and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = openpi_data.create_data_loader(cfg, framework="pytorch", shuffle=bool(args.shuffle), num_batches=args.steps)

    model_cfg = cfg.model
    if isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        object.__setattr__(model_cfg, "dtype", cfg.pytorch_training_precision)
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(model, Path(args.weight_path) / "model.safetensors")
    model.train()

    trainable_info = set_trainable(model, tuple(args.trainable_substring))
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"no trainable parameters matched {args.trainable_substring}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    losses: list[float] = []
    train_start = time.perf_counter()
    for step, (observation, actions) in enumerate(loader):
        if step >= args.steps:
            break
        observation = jax.tree.map(lambda x: move_to_device(x, device), observation)
        actions = actions.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        per_dim_loss = model(observation, actions)
        action_dim = min(int(args.loss_action_dim), per_dim_loss.shape[-1])
        selected_loss = per_dim_loss[..., :action_dim]
        if action_dim >= 7 and float(args.gripper_weight) != 1.0:
            weights = torch.ones(action_dim, device=selected_loss.device, dtype=selected_loss.dtype)
            weights[6] = float(args.gripper_weight)
            loss = (selected_loss * weights).sum() / (selected_loss.shape[0] * selected_loss.shape[1] * weights.sum())
        else:
            loss = selected_loss.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, int(args.log_every)) == 0 or step == 0 or step + 1 == args.steps:
            print(
                f"[train] step={step + 1}/{args.steps} loss={losses[-1]:.6f} "
                f"grad_norm={float(grad_norm):.6f} mem={gpu_memory()}",
                flush=True,
            )

    train_wall_sec = time.perf_counter() - train_start

    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        sampled_actions = model.sample_actions(device, observation, num_steps=args.num_inference_steps)
    inference_wall_sec = time.perf_counter() - infer_start
    sampled_actions_7d = sampled_actions[0, :, :7].detach().cpu()

    final_dir = ckpt_dir / str(args.steps)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    if args.save_full_model:
        safetensors.torch.save_model(model, final_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), final_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": args.steps,
            "config": dataclasses.asdict(cfg),
            "head_only": True,
            "trainable_info": trainable_info,
            "timestamp": time.time(),
        },
        final_dir / "metadata.pt",
    )
    data_config = loader.data_config()
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        normalize.save(final_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(ckpt_dir),
        "step_dir": str(final_dir),
        "source_weight_path": args.weight_path,
        "device": str(device),
        "head_only": True,
        "saved_full_model": bool(args.save_full_model),
        "trainable_info": trainable_info,
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_action_dim": int(args.loss_action_dim),
        "gripper_weight": float(args.gripper_weight),
        "train_wall_sec": train_wall_sec,
        "inference_num_steps": args.num_inference_steps,
        "inference_wall_sec": inference_wall_sec,
        "sampled_actions_shape": list(sampled_actions.shape),
        "sampled_actions_first_7d": sampled_actions_7d.tolist(),
        "gpu_memory": gpu_memory(),
    }
    summary_path = ckpt_dir / "head_only_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(f"[summary] {summary_path}")
    print(json.dumps({k: summary[k] for k in ("losses", "inference_wall_sec", "gpu_memory")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
