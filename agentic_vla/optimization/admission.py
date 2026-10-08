"""Deployment admission checks for calibrated CARVE profiles."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from .contracts import (
    CoResidentBenchmarkReport,
    PlannerBenchmarkReport,
    PlannerOptimizationProfile,
    SystemOptimizationProfile,
)
from .manifest import ProfileManifest


@dataclasses.dataclass(frozen=True)
class ProfileAdmissionRequirements:
    """Minimum evidence required before a profile can serve robot requests."""

    minimum_fidelity_samples: int = 1
    maximum_deadline_miss_rate: float = 0.0
    require_closed_loop_gate: bool = True

    def __post_init__(self) -> None:
        if self.minimum_fidelity_samples <= 0:
            raise ValueError("minimum_fidelity_samples must be positive")
        if not 0.0 <= self.maximum_deadline_miss_rate <= 1.0:
            raise ValueError("maximum_deadline_miss_rate must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class ProfileAdmissionDecision:
    """Auditable result of validating one deployment manifest."""

    accepted: bool
    status: str
    scope: str
    violations: tuple[str, ...] = ()
    fallback_profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "scope": self.scope,
            "violations": list(self.violations),
            "fallback_profile_id": self.fallback_profile_id,
        }

    def require_accepted(self) -> None:
        if not self.accepted:
            detail = "; ".join(self.violations) or "profile is not admitted"
            raise ValueError(f"deployment profile admission failed: {detail}")


@dataclasses.dataclass(frozen=True)
class PlannerAdmissionRequirements:
    """Evidence thresholds for admitting an external VLM deployment."""

    minimum_samples: int = 10
    minimum_schema_valid_rate: float = 0.99
    minimum_intent_valid_rate: float = 0.99
    minimum_reference_agreement_rate: float = 0.95
    maximum_timeout_rate: float = 0.01
    maximum_vla_deadline_miss_rate: float = 0.01
    maximum_unsafe_intervention_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        rates = (
            self.minimum_schema_valid_rate,
            self.minimum_intent_valid_rate,
            self.minimum_reference_agreement_rate,
            self.maximum_timeout_rate,
            self.maximum_vla_deadline_miss_rate,
            self.maximum_unsafe_intervention_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("planner admission rates must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class PlannerAdmissionDecision:
    """Auditable semantic-runtime admission result."""

    accepted: bool
    profile_id: str
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "profile_id": self.profile_id,
            "violations": list(self.violations),
        }

    def require_accepted(self) -> None:
        if not self.accepted:
            detail = "; ".join(self.violations) or "planner profile is not admitted"
            raise ValueError(f"planner profile admission failed: {detail}")


@dataclasses.dataclass(frozen=True)
class SystemAdmissionRequirements:
    """Thresholds for a Planner/VLA pair sharing one accelerator."""

    minimum_samples: int = 10
    maximum_peak_vram_gb: float | None = None
    maximum_vla_deadline_miss_rate: float = 0.01
    maximum_planner_timeout_rate: float = 0.01
    maximum_unsafe_intervention_rate: float = 0.0
    require_fallbacks: bool = True

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if self.maximum_peak_vram_gb is not None and self.maximum_peak_vram_gb <= 0:
            raise ValueError("maximum_peak_vram_gb must be positive or null")
        for value in (
            self.maximum_vla_deadline_miss_rate,
            self.maximum_planner_timeout_rate,
            self.maximum_unsafe_intervention_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("system admission rates must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class SystemAdmissionDecision:
    """Auditable admission result for a heterogeneous shared-GPU profile."""

    accepted: bool
    profile_id: str
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "profile_id": self.profile_id,
            "violations": list(self.violations),
        }

    def require_accepted(self) -> None:
        if not self.accepted:
            detail = "; ".join(self.violations) or "system profile is not admitted"
            raise ValueError(f"system profile admission failed: {detail}")


def validate_planner_profile_admission(
    profile: PlannerOptimizationProfile,
    report: PlannerBenchmarkReport,
    requirements: PlannerAdmissionRequirements | None = None,
) -> PlannerAdmissionDecision:
    """Admit a VLM only when semantics and co-resident timing both pass."""

    requirements = requirements or PlannerAdmissionRequirements()
    violations: list[str] = []
    if report.samples < requirements.minimum_samples:
        violations.append(
            f"samples {report.samples} < {requirements.minimum_samples}"
        )
    minimum_metrics = (
        (
            "schema_valid_rate",
            report.schema_valid_rate,
            requirements.minimum_schema_valid_rate,
        ),
        (
            "intent_valid_rate",
            report.intent_valid_rate,
            requirements.minimum_intent_valid_rate,
        ),
        (
            "reference_agreement_rate",
            report.reference_agreement_rate,
            requirements.minimum_reference_agreement_rate,
        ),
    )
    for name, value, minimum in minimum_metrics:
        if value < minimum:
            violations.append(f"{name} {value:.6f} < {minimum:.6f}")
    maximum_metrics = (
        ("timeout_rate", report.timeout_rate, requirements.maximum_timeout_rate),
        (
            "co_resident_vla_deadline_miss_rate",
            report.co_resident_vla_deadline_miss_rate,
            requirements.maximum_vla_deadline_miss_rate,
        ),
        (
            "unsafe_intervention_rate",
            report.unsafe_intervention_rate,
            requirements.maximum_unsafe_intervention_rate,
        ),
    )
    for name, value, maximum in maximum_metrics:
        if value > maximum:
            violations.append(f"{name} {value:.6f} > {maximum:.6f}")
    return PlannerAdmissionDecision(
        accepted=not violations,
        profile_id=profile.profile_id,
        violations=tuple(violations),
    )


def validate_system_profile_admission(
    profile: SystemOptimizationProfile,
    report: CoResidentBenchmarkReport,
    *,
    planner_admitted: bool,
    vla_admitted: bool,
    requirements: SystemAdmissionRequirements | None = None,
) -> SystemAdmissionDecision:
    """Admit a pair only after component and shared-GPU gates both pass."""

    requirements = requirements or SystemAdmissionRequirements(
        maximum_peak_vram_gb=profile.memory_budget_gb
    )
    violations: list[str] = []
    if not planner_admitted:
        violations.append("Planner profile is not independently admitted")
    if not vla_admitted:
        violations.append("VLA profile is not independently admitted")
    if report.samples < requirements.minimum_samples:
        violations.append(f"samples {report.samples} < {requirements.minimum_samples}")
    memory_limit = requirements.maximum_peak_vram_gb
    if memory_limit is None:
        memory_limit = profile.memory_budget_gb
    if report.peak_vram_gb > memory_limit:
        violations.append(
            f"peak_vram_gb {report.peak_vram_gb:.6f} > {memory_limit:.6f}"
        )
    maximum_metrics = (
        (
            "vla_deadline_miss_rate",
            report.vla_deadline_miss_rate,
            requirements.maximum_vla_deadline_miss_rate,
        ),
        (
            "planner_timeout_rate",
            report.planner_timeout_rate,
            requirements.maximum_planner_timeout_rate,
        ),
        (
            "unsafe_intervention_rate",
            report.unsafe_intervention_rate,
            requirements.maximum_unsafe_intervention_rate,
        ),
    )
    for name, value, maximum in maximum_metrics:
        if value > maximum:
            violations.append(f"{name} {value:.6f} > {maximum:.6f}")
    if requirements.require_fallbacks:
        if profile.fallback_planner_profile_id is None:
            violations.append("Planner fallback profile is missing")
        if profile.fallback_vla_profile_id is None:
            violations.append("VLA fallback profile is missing")
    return SystemAdmissionDecision(
        accepted=not violations,
        profile_id=profile.profile_id,
        violations=tuple(violations),
    )


def _gate_passed(gates: Mapping[str, Any], name: str) -> bool:
    gate = gates.get(name)
    return isinstance(gate, Mapping) and gate.get("passed") is True


def validate_profile_admission(
    manifest: ProfileManifest,
    requirements: ProfileAdmissionRequirements | None = None,
) -> ProfileAdmissionDecision:
    """Reject profiles whose replay, realtime, or closed-loop gates are incomplete."""

    requirements = requirements or ProfileAdmissionRequirements()
    admission = manifest.admission
    status = str(admission.get("status", "unreviewed")).strip().lower()
    scope = str(admission.get("scope", "")).strip()
    fallback_value = admission.get("fallback_profile_id")
    fallback_profile_id = str(fallback_value).strip() if fallback_value is not None else None
    gates_value = admission.get("gates", {})
    gates = gates_value if isinstance(gates_value, Mapping) else {}
    violations: list[str] = []

    if status != "promoted":
        violations.append(f"admission status is {status!r}, expected 'promoted'")
    if not scope:
        violations.append("deployment scope is missing")

    if manifest.fidelity.get("passed") is not True:
        violations.append("replay fidelity did not pass")
    try:
        fidelity_samples = int(manifest.fidelity.get("samples", 0))
    except (TypeError, ValueError):
        fidelity_samples = 0
    if fidelity_samples < requirements.minimum_fidelity_samples:
        violations.append(
            "replay fidelity samples "
            f"{fidelity_samples} < {requirements.minimum_fidelity_samples}"
        )
    if not _gate_passed(gates, "replay_fidelity"):
        violations.append("replay_fidelity admission gate is missing or failed")

    try:
        deadline_miss_rate = float(manifest.benchmark["deadline_miss_rate"])
    except (KeyError, TypeError, ValueError):
        deadline_miss_rate = None
    if deadline_miss_rate is None:
        violations.append("deadline_miss_rate is missing or invalid")
    elif deadline_miss_rate > requirements.maximum_deadline_miss_rate:
        violations.append(
            "deadline_miss_rate "
            f"{deadline_miss_rate:.6f} > {requirements.maximum_deadline_miss_rate:.6f}"
        )
    if not _gate_passed(gates, "realtime"):
        violations.append("realtime admission gate is missing or failed")

    if requirements.require_closed_loop_gate and not _gate_passed(gates, "closed_loop"):
        violations.append("closed_loop admission gate is missing or failed")

    if manifest.profile.backend in {
        "torch_compile_masked_views",
        "torchao_int8_masked_views",
    }:
        has_named_contract = bool(manifest.profile.options.get("image_view_order")) and bool(
            manifest.profile.options.get("elided_image_views")
        )
        has_legacy_contract = bool(manifest.profile.options.get("elided_image_indices"))
        if not (has_named_contract or has_legacy_contract):
            violations.append("masked-view profile has no static input-view contract")
        if not fallback_profile_id:
            violations.append("masked-view profile has no fallback_profile_id")
        elif fallback_profile_id == manifest.profile.profile_id:
            violations.append("masked-view fallback must differ from the primary profile")

    return ProfileAdmissionDecision(
        accepted=not violations,
        status=status,
        scope=scope,
        violations=tuple(violations),
        fallback_profile_id=fallback_profile_id,
    )
