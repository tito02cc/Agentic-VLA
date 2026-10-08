"""Single-flight asynchronous inference prefetch for chunked robot policies."""

from __future__ import annotations

import dataclasses
import math
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from .contracts import ActionSpec


@dataclasses.dataclass(frozen=True)
class ActionPrefixThresholds:
    """Action-contract limits for accepting a temporally shifted suffix."""

    continuous_rms_max: float = 0.35
    translation_endpoint_max: float = 0.35
    rotation_endpoint_max: float = 0.50
    gripper_agreement_min: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "continuous_rms_max",
            "translation_endpoint_max",
            "rotation_endpoint_max",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.gripper_agreement_min) <= 1.0:
            raise ValueError("gripper_agreement_min must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class ActionPrefixConsistency:
    """Measured agreement between executed and prefetched action prefixes."""

    accepted: bool
    reason: str
    prefix_actions: int
    continuous_rms: float
    translation_endpoint_l2: float
    rotation_endpoint_l2: float
    gripper_agreement: float
    continuous_cosine: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ActionPrefixVerifier:
    """Legacy 7-D Cartesian-delta prefix check, not a generic action validator."""

    def __init__(self, thresholds: ActionPrefixThresholds | None = None) -> None:
        self.thresholds = thresholds or ActionPrefixThresholds()

    def evaluate(
        self,
        executed_prefix: Any,
        predicted_prefix: Any,
    ) -> ActionPrefixConsistency:
        try:
            executed = np.asarray(executed_prefix, dtype=np.float64)
            predicted = np.asarray(predicted_prefix, dtype=np.float64)
        except (TypeError, ValueError):
            executed = predicted = np.empty((0, 7))
        if (
            executed.ndim != 2
            or predicted.ndim != 2
            or executed.shape != predicted.shape
            or executed.shape[0] == 0
            or executed.shape[1] != 7
            or not np.all(np.isfinite(executed))
            or not np.all(np.isfinite(predicted))
        ):
            return ActionPrefixConsistency(
                accepted=False,
                reason="invalid_prefix_contract",
                prefix_actions=int(executed.shape[0]) if executed.ndim == 2 else 0,
                continuous_rms=float("inf"),
                translation_endpoint_l2=float("inf"),
                rotation_endpoint_l2=float("inf"),
                gripper_agreement=0.0,
                continuous_cosine=0.0,
            )

        continuous_delta = predicted[:, :6] - executed[:, :6]
        continuous_rms = float(np.sqrt(np.mean(np.square(continuous_delta))))
        translation_endpoint_l2 = float(
            np.linalg.norm(np.sum(continuous_delta[:, :3], axis=0))
        )
        rotation_endpoint_l2 = float(
            np.linalg.norm(np.sum(continuous_delta[:, 3:6], axis=0))
        )
        executed_gripper = np.sign(executed[:, 6])
        predicted_gripper = np.sign(predicted[:, 6])
        gripper_agreement = float(np.mean(executed_gripper == predicted_gripper))
        executed_continuous = executed[:, :6].reshape(-1)
        predicted_continuous = predicted[:, :6].reshape(-1)
        norm_product = float(
            np.linalg.norm(executed_continuous) * np.linalg.norm(predicted_continuous)
        )
        continuous_cosine = (
            float(np.dot(executed_continuous, predicted_continuous) / norm_product)
            if norm_product > 1e-12
            else float(np.allclose(executed_continuous, predicted_continuous))
        )

        failures = []
        if continuous_rms > self.thresholds.continuous_rms_max:
            failures.append("continuous_rms")
        if translation_endpoint_l2 > self.thresholds.translation_endpoint_max:
            failures.append("translation_endpoint")
        if rotation_endpoint_l2 > self.thresholds.rotation_endpoint_max:
            failures.append("rotation_endpoint")
        if gripper_agreement < self.thresholds.gripper_agreement_min:
            failures.append("gripper")
        return ActionPrefixConsistency(
            accepted=not failures,
            reason="accepted" if not failures else "+".join(failures),
            prefix_actions=int(executed.shape[0]),
            continuous_rms=continuous_rms,
            translation_endpoint_l2=translation_endpoint_l2,
            rotation_endpoint_l2=rotation_endpoint_l2,
            gripper_agreement=gripper_agreement,
            continuous_cosine=continuous_cosine,
        )


@dataclasses.dataclass(frozen=True)
class JointPrefixConsistency:
    accepted: bool
    reason: str
    prefix_actions: int
    max_joint_error: float | None = None
    joint_rms: float | None = None
    max_gripper_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class JointPositionPrefixVerifier:
    """Compare both arms in native joint/gripper units; not a collision check.

    Thresholds must be supplied by a calibrated deployment, not inferred from
    Cartesian thresholds. Callers must align action times before this check.
    """

    def __init__(
        self,
        action_spec: ActionSpec,
        *,
        gripper_indices: tuple[int, ...],
        max_joint_error: float,
        joint_rms_max: float,
        max_gripper_error: float,
    ) -> None:
        if not isinstance(action_spec, ActionSpec) or action_spec.representation not in {
            "joint_position", "dual_arm_joint_position"
        }:
            raise ValueError("joint prefix verification requires a joint-position ActionSpec")
        indices = tuple(gripper_indices)
        if (
            not indices
            or any(type(index) is not int or not 0 <= index < action_spec.action_dim for index in indices)
            or len(set(indices)) != len(indices)
            or len(indices) == action_spec.action_dim
        ):
            raise ValueError("gripper indices must be unique, in range, and leave joint dimensions")
        if action_spec.representation == "dual_arm_joint_position" and len(indices) != 2:
            raise ValueError("dual-arm prefix verification requires two gripper indices")
        for value in (max_joint_error, joint_rms_max, max_gripper_error):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("joint prefix thresholds must be finite and positive")
        self.action_spec = action_spec
        self.gripper_indices = indices
        self.joint_indices = tuple(index for index in range(action_spec.action_dim) if index not in indices)
        self.max_joint_error = max_joint_error
        self.joint_rms_max = joint_rms_max
        self.max_gripper_error = max_gripper_error

    def evaluate(self, executed_prefix: Any, predicted_prefix: Any) -> JointPrefixConsistency:
        try:
            executed = np.asarray(executed_prefix, dtype=np.float64)
            predicted = np.asarray(predicted_prefix, dtype=np.float64)
            if executed.ndim != 2 or executed.shape != predicted.shape:
                raise ValueError("prefix shape mismatch")
            self.action_spec.validate(executed)
            self.action_spec.validate(predicted)
        except (TypeError, ValueError):
            return JointPrefixConsistency(False, "invalid_prefix_contract", 0)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = predicted - executed
            joints = delta[:, self.joint_indices]
            max_joint = float(np.max(np.abs(joints)))
            rms = float(np.sqrt(np.mean(np.square(joints))))
            max_gripper = float(np.max(np.abs(delta[:, self.gripper_indices])))
        if not all(math.isfinite(value) for value in (max_joint, rms, max_gripper)):
            return JointPrefixConsistency(False, "nonfinite_prefix_metrics", len(executed))
        failures = []
        if max_joint > self.max_joint_error:
            failures.append("joint_max")
        if rms > self.joint_rms_max:
            failures.append("joint_rms")
        if max_gripper > self.max_gripper_error:
            failures.append("gripper")
        return JointPrefixConsistency(
            not failures, "+".join(failures) or "accepted", len(executed), max_joint, rms, max_gripper
        )


@dataclasses.dataclass(frozen=True)
class PrefetchContext:
    """Lifecycle metadata needed to align or invalidate a prefetched chunk."""

    episode_id: str | int
    submitted_timestep: int
    lead_actions: int
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    ticket_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.submitted_timestep < 0:
            raise ValueError("submitted_timestep must be non-negative")
        if self.lead_actions <= 0:
            raise ValueError("lead_actions must be positive")


@dataclasses.dataclass(frozen=True)
class PrefetchResult:
    context: PrefetchContext
    response: Any | None
    elapsed_s: float
    valid: bool
    invalidation_reason: str | None = None
    error: Exception | None = None


@dataclasses.dataclass
class PrefetchDutyCycle:
    """Require fresh synchronous chunks between temporally shifted prefetches."""

    interval: int = 1
    synchronous_chunks_remaining: int = dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("prefetch interval must be positive")

    @property
    def can_submit(self) -> bool:
        return self.synchronous_chunks_remaining == 0

    def record_prefetch_consumed(self) -> None:
        self.synchronous_chunks_remaining = self.interval - 1

    def record_synchronous_chunk(self) -> None:
        self.synchronous_chunks_remaining = max(
            0,
            self.synchronous_chunks_remaining - 1,
        )


class AsyncInferencePrefetcher:
    """Run at most one policy request while cached actions remain executable."""

    def __init__(self, infer: Callable[[Mapping[str, Any]], Any]) -> None:
        if not callable(infer):
            raise TypeError("infer must be callable")
        self._infer = infer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="carve-prefetch")
        self._future: Future[tuple[Any | None, float, Exception | None]] | None = None
        self._context: PrefetchContext | None = None
        self._invalidation_reason: str | None = None
        self._closed = False

    @property
    def pending(self) -> bool:
        return self._future is not None

    @property
    def ready(self) -> bool:
        return self._future is not None and self._future.done()

    @property
    def invalidated(self) -> bool:
        return self._future is not None and self._invalidation_reason is not None

    @property
    def context(self) -> PrefetchContext | None:
        return self._context

    def submit(self, request: Mapping[str, Any], context: PrefetchContext) -> str:
        if self._closed:
            raise RuntimeError("prefetcher is closed")
        if self._future is not None:
            raise RuntimeError("a prefetch request is already active")
        if not isinstance(request, Mapping):
            raise TypeError("prefetch request must be a mapping")
        payload = dict(request)
        self._context = context
        self._invalidation_reason = None
        self._future = self._executor.submit(self._run, payload)
        return context.ticket_id

    def invalidate(self, reason: str) -> None:
        if self._future is None:
            return
        if not str(reason).strip():
            raise ValueError("invalidation reason must not be empty")
        self._invalidation_reason = str(reason)

    def take(self, *, wait: bool = True) -> PrefetchResult | None:
        if self._future is None or self._context is None:
            return None
        if not wait and not self._future.done():
            return None
        response, elapsed_s, error = self._future.result()
        result = PrefetchResult(
            context=self._context,
            response=response,
            elapsed_s=elapsed_s,
            valid=self._invalidation_reason is None and error is None,
            invalidation_reason=self._invalidation_reason,
            error=error,
        )
        self._future = None
        self._context = None
        self._invalidation_reason = None
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _run(self, request: Mapping[str, Any]) -> tuple[Any | None, float, Exception | None]:
        started_s = time.perf_counter()
        try:
            response = self._infer(request)
            error = None
        except Exception as exc:  # The control loop decides whether to retry synchronously.
            response = None
            error = exc
        return response, time.perf_counter() - started_s, error

    def __enter__(self) -> "AsyncInferencePrefetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
