"""Deployment admission checks for calibrated CARVE profiles."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

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

    if manifest.profile.backend == "torch_compile_masked_views":
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
