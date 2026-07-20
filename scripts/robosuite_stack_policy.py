#!/usr/bin/env python3
"""Shared model and preprocessing helpers for robosuite Stack policy pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


STATE_KEYS = ("robot0_proprio-state", "object-state")
ACTION_DIM = 7


@dataclass
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationStats":
        return cls(
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
            action_mean=np.asarray(data["action_mean"], dtype=np.float32),
            action_std=np.asarray(data["action_std"], dtype=np.float32),
        )


def extract_state(obs: dict[str, Any]) -> np.ndarray:
    parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS]
    return np.concatenate(parts, axis=0).astype(np.float32)


def extract_image(obs: dict[str, Any], camera_name: str = "frontview") -> np.ndarray:
    frame = np.asarray(obs[f"{camera_name}_image"], dtype=np.uint8)
    return np.ascontiguousarray(frame[::-1, :, :3])


def make_step_feature(step: int, horizon: int) -> np.ndarray:
    denom = max(1, int(horizon) - 1)
    return np.asarray([float(step) / float(denom)], dtype=np.float32)


def normalize_state(state_with_step: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (state_with_step.astype(np.float32) - stats.state_mean) / stats.state_std


def normalize_actions(actions: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (actions.astype(np.float32) - stats.action_mean) / stats.action_std


def denormalize_actions(actions: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return actions.astype(np.float32) * stats.action_std + stats.action_mean


class StackChunkPolicyNet(nn.Module):
    """Compact RGB+state action-chunk policy for robosuite Stack."""

    def __init__(self, state_dim: int, chunk_size: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = int(state_dim)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.chunk_size * self.action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        pred = self.head(torch.cat([image_feat, state_feat], dim=-1))
        return pred.view(-1, self.chunk_size, self.action_dim)


def prepare_image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device=device)


def prepare_state_tensor(state_with_step: np.ndarray, stats: NormalizationStats, device: torch.device) -> torch.Tensor:
    norm = normalize_state(state_with_step, stats)
    return torch.from_numpy(norm).unsqueeze(0).to(device=device)
