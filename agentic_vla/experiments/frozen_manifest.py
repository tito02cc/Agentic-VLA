"""Versioned-free validation for CARVE's single frozen experiment manifest."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping


_STAGES = ("integration_smoke", "paired_recovery", "libero_plus", "libero_pro")


@dataclasses.dataclass(frozen=True)
class FrozenSystemConfig:
    model_id: str
    runtime_profile_id: str
    fallback_profile_id: str
    planner_mode: str
    control_deadline_ms: float
    profile_manifest: str

    def __post_init__(self) -> None:
        if self.model_id != "pi05":
            raise ValueError("the frozen closed-loop system currently supports pi05 only")
        if not self.runtime_profile_id or not self.fallback_profile_id:
            raise ValueError("runtime and fallback profile IDs must be present")
        if self.planner_mode != "asynchronous_guarded":
            raise ValueError("planner_mode must be asynchronous_guarded")
        if self.control_deadline_ms <= 0:
            raise ValueError("control_deadline_ms must be positive")
        if not self.profile_manifest:
            raise ValueError("profile_manifest must be present")


@dataclasses.dataclass(frozen=True)
class PreRegisteredExperiment:
    experiment_id: str
    stage: str
    snapshot_ids: tuple[str, ...]
    noise_seeds: tuple[int, ...]
    branches: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.stage not in _STAGES:
            raise ValueError(f"unsupported experiment stage: {self.stage}")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must not be empty")
        if len(self.snapshot_ids) != len(set(self.snapshot_ids)):
            raise ValueError("snapshot_ids must be unique")
        if len(self.noise_seeds) != len(set(self.noise_seeds)):
            raise ValueError("noise_seeds must be unique")
        if len(self.branches) != len(set(self.branches)):
            raise ValueError("branches must be unique")


@dataclasses.dataclass(frozen=True)
class FrozenExperimentManifest:
    schema_version: int
    system: FrozenSystemConfig
    experiments: tuple[PreRegisteredExperiment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported frozen-manifest schema version")
        if not self.experiments:
            raise ValueError("at least one pre-registered experiment is required")
        identifiers = tuple(item.experiment_id for item in self.experiments)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("experiment IDs must be unique")
        stages = tuple(item.stage for item in self.experiments)
        if stages != _STAGES:
            raise ValueError(
                "experiments must preserve the required integration, recovery, "
                "LIBERO-Plus, LIBERO-Pro order"
            )


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _integer_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_name} must be a list of integers")
    return tuple(value)


def load_frozen_experiment_manifest(path: str | Path) -> FrozenExperimentManifest:
    """Load the sole pre-registered configuration used for future GPU runs."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen manifest must be a JSON object")
    system_payload = payload.get("system")
    experiments_payload = payload.get("experiments")
    if not isinstance(system_payload, Mapping) or not isinstance(experiments_payload, list):
        raise ValueError("frozen manifest requires system and experiments")
    system = FrozenSystemConfig(
        model_id=str(system_payload.get("model_id", "")),
        runtime_profile_id=str(system_payload.get("runtime_profile_id", "")),
        fallback_profile_id=str(system_payload.get("fallback_profile_id", "")),
        planner_mode=str(system_payload.get("planner_mode", "")),
        control_deadline_ms=float(system_payload.get("control_deadline_ms", 0.0)),
        profile_manifest=str(system_payload.get("profile_manifest", "")),
    )
    experiments = []
    for item in experiments_payload:
        if not isinstance(item, Mapping):
            raise ValueError("each experiment must be a JSON object")
        experiments.append(
            PreRegisteredExperiment(
                experiment_id=str(item.get("experiment_id", "")),
                stage=str(item.get("stage", "")),
                snapshot_ids=_string_tuple(item.get("snapshot_ids"), "snapshot_ids"),
                noise_seeds=_integer_tuple(item.get("noise_seeds"), "noise_seeds"),
                branches=_string_tuple(item.get("branches"), "branches"),
                claim_boundary=str(item.get("claim_boundary", "")),
            )
        )
    return FrozenExperimentManifest(
        schema_version=int(payload.get("schema_version", 0)),
        system=system,
        experiments=tuple(experiments),
    )
