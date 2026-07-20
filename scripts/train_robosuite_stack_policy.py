#!/usr/bin/env python3
"""Train a compact RGB+state action-chunk policy on robosuite Stack demos."""

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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robosuite_stack_policy import (
    ACTION_DIM,
    NormalizationStats,
    StackChunkPolicyNet,
    normalize_actions,
    normalize_state,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InMemoryStackDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        states: np.ndarray,
        action_chunks: np.ndarray,
        stats: NormalizationStats,
    ):
        self.images = images
        self.states = normalize_state(states, stats)
        self.action_chunks = normalize_actions(action_chunks, stats)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[idx].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        state = self.states[idx].astype(np.float32)
        action = self.action_chunks[idx].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(state), torch.from_numpy(action)


def load_dataset(path: Path, val_fraction: float, seed: int, max_samples: int | None) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        ep_names = sorted(h5["episodes"].keys())
        rng = random.Random(seed)
        rng.shuffle(ep_names)
        val_count = max(1, int(round(len(ep_names) * val_fraction))) if len(ep_names) > 1 else 0
        val_eps = set(ep_names[:val_count])
        train_eps = [name for name in ep_names if name not in val_eps]
        val_eps_sorted = [name for name in ep_names if name in val_eps]

        def gather(names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            images, states, chunks = [], [], []
            for name in names:
                group = h5["episodes"][name]
                images.append(group["images"][:])
                states.append(group["states"][:])
                chunks.append(group["action_chunks"][:])
            return np.concatenate(images, axis=0), np.concatenate(states, axis=0), np.concatenate(chunks, axis=0)

        train_images, train_states, train_chunks = gather(train_eps)
        val_images, val_states, val_chunks = gather(val_eps_sorted) if val_eps_sorted else gather(train_eps[-1:])

    if max_samples is not None and max_samples > 0 and train_states.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(train_states.shape[0], size=max_samples, replace=False)
        train_images = train_images[idx]
        train_states = train_states[idx]
        train_chunks = train_chunks[idx]

    action_flat = train_chunks.reshape(-1, train_chunks.shape[-1])
    stats = NormalizationStats(
        state_mean=train_states.mean(axis=0).astype(np.float32),
        state_std=np.maximum(train_states.std(axis=0), 1e-3).astype(np.float32),
        action_mean=action_flat.mean(axis=0).astype(np.float32),
        action_std=np.maximum(action_flat.std(axis=0), 5e-2).astype(np.float32),
    )
    return {
        "train": (train_images, train_states, train_chunks),
        "val": (val_images, val_states, val_chunks),
        "stats": stats,
        "train_episodes": train_eps,
        "val_episodes": val_eps_sorted,
    }


def weighted_chunk_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target)
    weights[..., 6] = 2.5
    weights[..., 3:6] = 0.5
    return (weights * (pred - target).square()).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for images, states, chunks in loader:
        images = images.to(device=device, non_blocking=True)
        states = states.to(device=device, non_blocking=True)
        chunks = chunks.to(device=device, non_blocking=True)
        pred = model(images, states)
        losses.append(float(weighted_chunk_mse(pred, chunks).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-hdf5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    dataset_path = Path(args.dataset_hdf5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    data = load_dataset(
        dataset_path,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        max_samples=int(args.max_samples) if int(args.max_samples) > 0 else None,
    )
    stats: NormalizationStats = data["stats"]
    train_images, train_states, train_chunks = data["train"]
    val_images, val_states, val_chunks = data["val"]
    train_ds = InMemoryStackDataset(train_images, train_states, train_chunks, stats)
    val_ds = InMemoryStackDataset(val_images, val_states, val_chunks, stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = int(train_chunks.shape[1])
    state_dim = int(train_states.shape[1])
    model = StackChunkPolicyNet(state_dim=state_dim, chunk_size=chunk_size, action_dim=ACTION_DIM).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, int(args.epochs)))

    best_val = float("inf")
    history: list[dict[str, float]] = []
    best_path = output_dir / "best_policy.pt"
    final_path = output_dir / "final_policy.pt"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses = []
        for images, states, chunks in train_loader:
            images = images.to(device=device, non_blocking=True)
            states = states.to(device=device, non_blocking=True)
            chunks = chunks.to(device=device, non_blocking=True)
            pred = model(images, states)
            loss = weighted_chunk_mse(pred, chunks)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = evaluate(model, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "lr": float(scheduler.get_last_lr()[0])}
        history.append(row)
        print(f"[train] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "chunk_size": chunk_size,
                    "action_dim": ACTION_DIM,
                    "stats": stats.to_dict(),
                    "dataset_hdf5": str(dataset_path),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dim": state_dim,
            "chunk_size": chunk_size,
            "action_dim": ACTION_DIM,
            "stats": stats.to_dict(),
            "dataset_hdf5": str(dataset_path),
            "epoch": int(args.epochs),
            "val_loss": history[-1]["val_loss"] if history else None,
        },
        final_path,
    )

    summary = {
        "dataset_hdf5": str(dataset_path),
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "train_samples": int(train_states.shape[0]),
        "val_samples": int(val_states.shape[0]),
        "train_episodes": data["train_episodes"],
        "val_episodes": data["val_episodes"],
        "state_dim": state_dim,
        "chunk_size": chunk_size,
        "epochs": int(args.epochs),
        "best_val_loss": best_val,
        "wall_sec": time.perf_counter() - start,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
