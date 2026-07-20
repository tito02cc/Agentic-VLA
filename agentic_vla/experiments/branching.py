"""Simulator-state branching without exposing privileged state to the controller."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class SimulatorSnapshot:
    state: np.ndarray
    task_id: int
    episode_id: int
    timestep: int
    fingerprint: str


def _fingerprint(state: np.ndarray) -> str:
    canonical = np.asarray(state, dtype=np.float64).reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def capture_simulator_snapshot(
    env: Any,
    *,
    task_id: int,
    episode_id: int,
    timestep: int,
) -> SimulatorSnapshot:
    """Capture evaluator state; callers must not feed it to the controller."""

    if not hasattr(env, "get_sim_state"):
        raise TypeError("environment must expose get_sim_state()")
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy().reshape(-1)
    if state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("simulator returned an invalid state")
    return SimulatorSnapshot(
        state=state,
        task_id=int(task_id),
        episode_id=int(episode_id),
        timestep=int(timestep),
        fingerprint=_fingerprint(state),
    )


def restore_simulator_snapshot(env: Any, snapshot: SimulatorSnapshot) -> Any:
    """Restore a snapshot and regenerate observations when the API permits."""

    if hasattr(env, "regenerate_obs_from_state"):
        observation = env.regenerate_obs_from_state(snapshot.state.copy())
    elif hasattr(env, "set_state"):
        env.set_state(snapshot.state.copy())
        observation = None
    else:
        raise TypeError("environment must expose regenerate_obs_from_state() or set_state()")
    restored = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    if not np.allclose(restored, snapshot.state, rtol=1e-10, atol=1e-10):
        raise RuntimeError("simulator state did not restore exactly")
    return observation
