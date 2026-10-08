"""Profile-admitted VLA primitive binding for the canonical tool runtime."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agentic_vla.configuration import CarveRunConfig
from agentic_vla.optimization import ProfileManifest
from agentic_vla.runtime import CarveRuntime, InferenceControls, InferenceRequest

from .contracts import PrimitiveStatus, ToolExecutionContext, reject_direct_action_fields
from .runtime import PrimitiveExecutionReport


ObservationSource = Callable[[ToolExecutionContext], Mapping[str, Any]]
InferenceMetadataSource = Callable[[ToolExecutionContext], Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class ActionExecutionReport:
    """Environment result after consuming a private VLA action chunk."""

    status: PrimitiveStatus | str
    ended_timestep: int
    observed_outcome: str
    requires_semantic_check: bool = False
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            status = (
                self.status
                if isinstance(self.status, PrimitiveStatus)
                else PrimitiveStatus(str(self.status).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("unsupported action execution status") from exc
        if self.ended_timestep < 0:
            raise ValueError("ended_timestep must be non-negative")
        if not self.observed_outcome.strip():
            raise ValueError("observed_outcome must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("action execution metadata must be a mapping")
        reject_direct_action_fields(self.metadata, path="action_execution.metadata")
        object.__setattr__(self, "status", status)


ActionChunkExecutor = Callable[
    [Any, ToolExecutionContext], ActionExecutionReport
]


def _receipt_profile_id(receipt: Mapping[str, Any]) -> str:
    profile = receipt.get("profile")
    if isinstance(profile, Mapping):
        return str(profile.get("profile_id", "")).strip()
    return str(receipt.get("profile_id", "")).strip()


class ProfileAdmittedVlaPrimitive:
    """Infer and execute one bounded VLA chunk without exposing raw actions."""

    def __init__(
        self,
        *,
        config: CarveRunConfig,
        runtime: CarveRuntime,
        observation_source: ObservationSource,
        action_executor: ActionChunkExecutor,
        profile_manifest: ProfileManifest | None = None,
        inference_metadata_source: InferenceMetadataSource | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.observation_source = observation_source
        self.action_executor = action_executor
        self.inference_metadata_source = inference_metadata_source
        manifest_path = Path(config.vla.profile_manifest_path)
        self.manifest = profile_manifest or ProfileManifest.load(manifest_path)
        self._validate_static_admission()

    @property
    def controls(self) -> InferenceControls:
        profile = self.manifest.profile
        return InferenceControls(
            inference_steps=profile.inference_steps,
            max_actions=profile.action_horizon,
            precision=profile.deployment_precision,
            reuse_context=False,
            deadline_ms=self.config.optimize.deadline_ms,
        )

    def __call__(
        self,
        instruction: str,
        context: ToolExecutionContext,
    ) -> PrimitiveExecutionReport:
        try:
            controls = self._controls_for_context(context)
            inference_metadata = (
                {}
                if self.inference_metadata_source is None
                else dict(self.inference_metadata_source(context))
            )
            reject_direct_action_fields(
                inference_metadata, path="vla_inference.metadata"
            )
            chunk = self.runtime.infer(
                InferenceRequest(
                    observation=dict(self.observation_source(context)),
                    instruction=instruction,
                    controls=controls,
                    episode_id=context.episode_id,
                    timestep=context.timestep,
                    agentic={
                        "tool": "vla_act",
                        "deployment_profile_id": context.deployment_profile_id,
                    },
                    metadata={
                        **inference_metadata,
                        "trace_context": {
                            "planner_mode": self.config.planner.mode.value,
                            "primitive_boundary_policy": (
                                self.config.harness.primitive_boundary_policy.value
                            ),
                        }
                    },
                )
            )
            trace = self.runtime.last_trace
            if trace is None:
                raise RuntimeError("VLA runtime did not produce a trace receipt")
            receipt = trace.metadata.get("optimization_profile")
            if self.config.optimize.enforce_profile_admission:
                self._validate_runtime_receipt(receipt)
            execution = self.action_executor(chunk.actions, context)
            if execution.ended_timestep < context.timestep:
                raise ValueError("action executor returned a stale timestep")
            return PrimitiveExecutionReport(
                status=execution.status,
                started_timestep=context.timestep,
                ended_timestep=execution.ended_timestep,
                expected_outcome=instruction,
                observed_outcome=execution.observed_outcome,
                requires_semantic_check=execution.requires_semantic_check,
                metadata={
                    "request_id": trace.request_id,
                    "profile_id": _receipt_profile_id(receipt or {}),
                    "backend": self.manifest.profile.backend,
                    "runtime_latency_ms": trace.runtime_latency_ms,
                    "model_latency_ms": trace.model_latency_ms,
                    "action_count": trace.action_count,
                    "deadline_ms": trace.deadline_ms,
                    "deadline_miss": trace.deadline_miss,
                    "fallback_used": (
                        _receipt_profile_id(receipt or {})
                        != self.manifest.profile.profile_id
                    ),
                    **dict(execution.metadata),
                },
            )
        except Exception as exc:
            trace = self.runtime.last_trace
            return PrimitiveExecutionReport(
                status=PrimitiveStatus.FAILED,
                started_timestep=context.timestep,
                ended_timestep=context.timestep,
                expected_outcome=instruction,
                observed_outcome=f"VLA primitive failed: {type(exc).__name__}",
                requires_semantic_check=True,
                metadata={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "runtime_latency_ms": (
                        None if trace is None else trace.runtime_latency_ms
                    ),
                    "deadline_miss": (
                        False if trace is None else trace.deadline_miss
                    ),
                },
            )

    def _validate_static_admission(self) -> None:
        expected = self.config.vla
        manifest = self.manifest
        identities = {
            "model_id": (manifest.model_id, expected.model_id),
            "adapter_id": (manifest.adapter_id, expected.adapter_id),
            "checkpoint_id": (manifest.checkpoint_id, expected.checkpoint_id),
            "profile_id": (
                manifest.profile.profile_id,
                expected.deployment_profile_id,
            ),
        }
        mismatches = [
            name for name, (actual, configured) in identities.items() if actual != configured
        ]
        if mismatches:
            raise ValueError(
                f"VLA profile manifest identity mismatch: {', '.join(mismatches)}"
            )
        if self.config.optimize.enforce_profile_admission:
            if manifest.admission.get("status") != "promoted":
                raise ValueError("VLA profile manifest is not promoted")
            gates = manifest.admission.get("gates", {})
            if not isinstance(gates, Mapping) or not gates:
                raise ValueError("promoted VLA profile has no admission gates")
            failed = [
                str(name)
                for name, gate in gates.items()
                if not isinstance(gate, Mapping) or gate.get("passed") is not True
            ]
            if failed:
                raise ValueError(f"VLA profile failed admission gates: {failed}")
        if self.runtime.adapter.adapter_id != expected.adapter_id:
            raise ValueError("runtime adapter does not match the configured VLA adapter")

    def _controls_for_context(
        self, context: ToolExecutionContext
    ) -> InferenceControls:
        """Apply Harness controls only inside the admitted deployment envelope."""

        base = self.controls
        if not context.inference_controls:
            return base
        requested = InferenceControls(**dict(context.inference_controls))
        if (
            requested.inference_steps is not None
            and requested.inference_steps != base.inference_steps
        ):
            raise ValueError("requested inference_steps are outside the admitted profile")
        if requested.precision is not None and requested.precision != base.precision:
            raise ValueError("requested precision is outside the admitted profile")
        if requested.reuse_context:
            raise ValueError("context reuse requires a separately admitted reuse profile")
        max_actions = base.max_actions
        if requested.max_actions is not None:
            if max_actions is not None and requested.max_actions > max_actions:
                raise ValueError("requested action horizon exceeds the admitted profile")
            max_actions = requested.max_actions
        return dataclasses.replace(
            base,
            max_actions=max_actions,
            deadline_ms=(
                base.deadline_ms
                if requested.deadline_ms is None
                else requested.deadline_ms
            ),
        )

    def _validate_runtime_receipt(self, receipt: Any) -> None:
        if not isinstance(receipt, Mapping):
            raise ValueError("VLA service returned no optimization-profile receipt")
        actual = _receipt_profile_id(receipt)
        accepted = {self.manifest.profile.profile_id}
        fallback = self.manifest.admission.get("fallback_profile_id")
        if self.config.optimize.allow_reference_fallback and fallback:
            accepted.add(str(fallback))
        if actual not in accepted:
            raise ValueError(
                f"VLA service profile {actual!r} is not admitted for this run"
            )
