"""Persist reproducible failure states for paired branch evaluation."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .branching import capture_simulator_snapshot


class FailureSnapshotWriter:
    """Write bounded, cooldown-gated evaluator snapshots and deployable inputs."""

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        max_per_episode: int = 3,
        minimum_gap_steps: int = 40,
    ) -> None:
        if max_per_episode <= 0:
            raise ValueError("max_per_episode must be positive")
        if minimum_gap_steps < 0:
            raise ValueError("minimum_gap_steps must be non-negative")
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_per_episode = int(max_per_episode)
        self.minimum_gap_steps = int(minimum_gap_steps)
        self._episode_key: tuple[int, int] | None = None
        self._count = 0
        self._last_timestep: int | None = None

    def reset(self, *, task_id: int, episode_id: int) -> None:
        self._episode_key = (int(task_id), int(episode_id))
        self._count = 0
        self._last_timestep = None

    def write(
        self,
        env: Any,
        *,
        task_id: int,
        episode_id: int,
        timestep: int,
        trigger: str,
        instruction: str,
        observation: Mapping[str, Any],
        cached_actions: Sequence[Any],
        controller: Mapping[str, Any],
        last_action: Sequence[Any] | None = None,
    ) -> dict[str, str] | None:
        episode_key = (int(task_id), int(episode_id))
        if self._episode_key != episode_key:
            self.reset(task_id=task_id, episode_id=episode_id)
        if self._count >= self.max_per_episode:
            return None
        if (
            self._last_timestep is not None
            and int(timestep) - self._last_timestep < self.minimum_gap_steps
        ):
            return None

        snapshot = capture_simulator_snapshot(
            env,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
        )
        stem = f"task{int(task_id):02d}_episode{int(episode_id):03d}_step{int(timestep):04d}"
        array_path = self.root / f"{stem}.npz"
        metadata_path = self.root / f"{stem}.json"
        arrays = {
            "sim_state": snapshot.state,
            "cached_actions": np.asarray(list(cached_actions), dtype=np.float32),
        }
        if last_action is not None:
            arrays["last_action"] = np.asarray(last_action, dtype=np.float32)
        for key, value in observation.items():
            arrays[f"observation_{key}"] = np.asarray(value)
        np.savez_compressed(array_path, **arrays)
        metadata = {
            "schema_version": 2,
            "task_id": int(task_id),
            "episode_id": int(episode_id),
            "timestep": int(timestep),
            "trigger": str(trigger),
            "instruction": str(instruction),
            "snapshot_fingerprint": snapshot.fingerprint,
            "controller": dict(controller),
            "sim_state_role": "evaluator_restore_only",
            "controller_uses_privileged_state": False,
            "array_file": array_path.name,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._count += 1
        self._last_timestep = int(timestep)
        return {"arrays": str(array_path), "metadata": str(metadata_path)}
