"""Action-space fidelity gates for optimized embodied policies."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class FidelityThresholds:
    """Acceptance limits calibrated in the policy's normalized action space."""

    first_action_mae: float = 0.03
    chunk_mae: float = 0.03
    chunk_rmse: float = 0.05
    minimum_cosine: float = 0.98
    minimum_gripper_agreement: float = 0.99
    endpoint_l2: float = 0.10
    jerk_rmse: float = 0.10

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = float(getattr(self, field.name))
            if value < 0:
                raise ValueError(f"{field.name} must be non-negative")
        if self.minimum_cosine > 1.0 or self.minimum_gripper_agreement > 1.0:
            raise ValueError("minimum agreement metrics must be at most 1")


@dataclasses.dataclass(frozen=True)
class FidelityReport:
    passed: bool
    samples: int
    metrics: Mapping[str, float]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "samples": self.samples,
            "metrics": dict(self.metrics),
            "violations": list(self.violations),
        }


class ActionFidelityVerifier:
    """Compare paired action chunks without relying on language metrics."""

    def __init__(self, thresholds: FidelityThresholds | None = None) -> None:
        self.thresholds = thresholds or FidelityThresholds()

    @staticmethod
    def _actions(value: Any, name: str) -> np.ndarray:
        actions = np.asarray(value, dtype=np.float64)
        if actions.ndim != 2 or min(actions.shape) <= 0:
            raise ValueError(f"{name} actions must have shape [horizon, action_dim]")
        if not np.isfinite(actions).all():
            raise ValueError(f"{name} actions contain non-finite values")
        return actions

    def compare(self, reference: Any, candidate: Any) -> FidelityReport:
        reference_actions = self._actions(reference, "reference")
        candidate_actions = self._actions(candidate, "candidate")
        if reference_actions.shape != candidate_actions.shape:
            raise ValueError(
                "paired action chunks must have identical shapes: "
                f"{reference_actions.shape} != {candidate_actions.shape}"
            )

        difference = candidate_actions - reference_actions
        flat_reference = reference_actions.reshape(-1)
        flat_candidate = candidate_actions.reshape(-1)
        norm_product = float(np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate))
        if norm_product == 0.0:
            cosine = 1.0 if np.array_equal(reference_actions, candidate_actions) else 0.0
        else:
            cosine = float(np.dot(flat_reference, flat_candidate) / norm_product)

        pose_dims = min(3, reference_actions.shape[1])
        reference_endpoint = reference_actions[:, :pose_dims].sum(axis=0)
        candidate_endpoint = candidate_actions[:, :pose_dims].sum(axis=0)
        endpoint_l2 = float(np.linalg.norm(candidate_endpoint - reference_endpoint))

        if reference_actions.shape[0] >= 3:
            reference_jerk = np.diff(reference_actions, n=2, axis=0)
            candidate_jerk = np.diff(candidate_actions, n=2, axis=0)
            jerk_rmse = float(np.sqrt(np.mean(np.square(candidate_jerk - reference_jerk))))
        else:
            jerk_rmse = 0.0

        gripper_reference = reference_actions[:, -1] >= 0.0
        gripper_candidate = candidate_actions[:, -1] >= 0.0
        gripper_agreement = float(np.mean(gripper_reference == gripper_candidate))

        metrics = {
            "first_action_mae": float(np.mean(np.abs(difference[0]))),
            "chunk_mae": float(np.mean(np.abs(difference))),
            "chunk_rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "chunk_cosine": cosine,
            "gripper_agreement": gripper_agreement,
            "endpoint_l2": endpoint_l2,
            "jerk_rmse": jerk_rmse,
        }
        limits = self.thresholds
        violations: list[str] = []
        for metric in ("first_action_mae", "chunk_mae", "chunk_rmse", "endpoint_l2", "jerk_rmse"):
            if metrics[metric] > float(getattr(limits, metric)):
                violations.append(metric)
        if metrics["chunk_cosine"] < limits.minimum_cosine:
            violations.append("chunk_cosine")
        if metrics["gripper_agreement"] < limits.minimum_gripper_agreement:
            violations.append("gripper_agreement")
        return FidelityReport(
            passed=not violations,
            samples=reference_actions.shape[0],
            metrics=metrics,
            violations=tuple(violations),
        )
