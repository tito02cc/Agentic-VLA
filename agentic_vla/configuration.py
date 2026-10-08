"""Canonical, secret-free configuration for one CARVE embodied-agent run."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentic_vla.runtime.agent import PlannerProvider, PlannerProviderConfig
from agentic_vla.runtime.harness import PrimitiveBoundaryPolicy, TaskStartPolicy
from agentic_vla.toolchain import RunManifest, core_tool_specs


RUN_CONFIG_SCHEMA_VERSION = 1
_SECRET_FIELDS = frozenset({"api_key", "access_token", "password", "secret"})


class PlannerExecutionMode(str, enum.Enum):
    """How high-level reasoning enters the common CARVE tool boundary."""

    EXTERNAL_CODING_AGENT = "external_coding_agent"
    EMBEDDED_VLM = "embedded_vlm"
    SCRIPTED = "scripted"


def _reject_unknown_fields(
    values: Mapping[str, Any],
    allowed: set[str],
    *,
    section: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{section} contains unknown fields: {unknown}")


def _reject_secrets(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _SECRET_FIELDS:
                raise ValueError(
                    f"{path}.{normalized} must not contain a secret; use an environment variable name"
                )
            _reject_secrets(item, path=f"{path}.{normalized}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secrets(item, path=f"{path}[{index}]")


@dataclasses.dataclass(frozen=True)
class PlannerRunConfig:
    """Identity and bounded authority for a coding agent or embedded VLM."""

    mode: PlannerExecutionMode | str
    planner_id: str
    provider: PlannerProvider | str | None = None
    model: str = ""
    endpoint: str = ""
    api_key_env: str = ""
    timeout_s: float = 30.0
    max_tokens: int = 192
    jpeg_quality: int = 85
    max_calls_per_episode: int = 8
    max_grounding_repairs: int = 1
    minimum_intervention_confidence: float = 0.55
    prompt_schema_version: str = "carve.planner.v1"

    def __post_init__(self) -> None:
        mode = (
            self.mode
            if isinstance(self.mode, PlannerExecutionMode)
            else PlannerExecutionMode(str(self.mode).strip().lower())
        )
        provider = self.provider
        if provider not in {None, ""} and not isinstance(provider, PlannerProvider):
            provider = PlannerProvider(str(provider).strip().lower())
        if not self.planner_id.strip() or not self.prompt_schema_version.strip():
            raise ValueError("planner_id and prompt_schema_version must not be empty")
        if self.timeout_s <= 0 or self.max_tokens <= 0 or self.max_calls_per_episode <= 0:
            raise ValueError("planner timeout, token budget, and call budget must be positive")
        if self.max_grounding_repairs < 0:
            raise ValueError("planner grounding repair budget must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("planner jpeg_quality must be in [1, 100]")
        if not 0.0 <= self.minimum_intervention_confidence <= 1.0:
            raise ValueError("minimum_intervention_confidence must be in [0, 1]")
        if mode is PlannerExecutionMode.EMBEDDED_VLM:
            if provider in {None, ""}:
                raise ValueError("embedded_vlm requires a provider")
            if not self.model.strip() or not self.endpoint.strip():
                raise ValueError("embedded_vlm requires model and endpoint")
        elif provider not in {None, ""} or self.endpoint.strip() or self.api_key_env.strip():
            raise ValueError(
                "external_coding_agent and scripted modes must use the tool boundary, not a VLM endpoint"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "provider", provider or None)

    def provider_config(self) -> PlannerProviderConfig | None:
        """Return an online VLM adapter only for the embedded execution mode."""

        if self.mode is not PlannerExecutionMode.EMBEDDED_VLM:
            return None
        return PlannerProviderConfig(
            provider=self.provider,
            model=self.model,
            endpoint=self.endpoint,
            api_key_env=self.api_key_env,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
            jpeg_quality=self.jpeg_quality,
        )


@dataclasses.dataclass(frozen=True)
class HarnessRunConfig:
    """Stable scheduling, authority, and safety settings."""

    task_start_policy: TaskStartPolicy | str = TaskStartPolicy.STARTUP_WAIT
    primitive_boundary_policy: PrimitiveBoundaryPolicy | str = (
        PrimitiveBoundaryPolicy.SELECTIVE
    )
    allowed_tools: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(spec.name for spec in core_tool_specs())
    )
    retry_budget: int = 2
    recovery_budget: int = 2
    physical_skill_budget: int | None = None
    memory_retrieval_limit: int = 8
    planner_cooldown_steps: int = 50
    safe_hold_timeout_s: float = 10.0
    safe_hold_heartbeat_hz: float = 10.0

    def __post_init__(self) -> None:
        start = (
            self.task_start_policy
            if isinstance(self.task_start_policy, TaskStartPolicy)
            else TaskStartPolicy(str(self.task_start_policy).strip().lower())
        )
        boundary = (
            self.primitive_boundary_policy
            if isinstance(self.primitive_boundary_policy, PrimitiveBoundaryPolicy)
            else PrimitiveBoundaryPolicy(
                str(self.primitive_boundary_policy).strip().lower()
            )
        )
        allowed = tuple(str(name).strip().lower() for name in self.allowed_tools)
        catalog = {spec.name for spec in core_tool_specs()}
        if not allowed or len(allowed) != len(set(allowed)):
            raise ValueError("allowed_tools must contain unique tool names")
        unknown = sorted(set(allowed) - catalog)
        if unknown:
            raise ValueError(f"allowed_tools contains unknown tools: {unknown}")
        budgets = (
            self.retry_budget,
            self.recovery_budget,
            self.planner_cooldown_steps,
        )
        if min(budgets) < 0 or (
            self.physical_skill_budget is not None
            and self.physical_skill_budget < 0
        ):
            raise ValueError("Harness budgets and cooldown must be non-negative")
        if self.memory_retrieval_limit <= 0:
            raise ValueError("memory_retrieval_limit must be positive")
        if self.safe_hold_timeout_s <= 0 or self.safe_hold_heartbeat_hz <= 0:
            raise ValueError("safe-hold settings must be positive")
        object.__setattr__(self, "task_start_policy", start)
        object.__setattr__(self, "primitive_boundary_policy", boundary)
        object.__setattr__(self, "allowed_tools", allowed)

    @property
    def effective_physical_skill_budget(self) -> int:
        """Preserve legacy recovery-only semantics unless explicitly expanded."""

        if self.physical_skill_budget is None:
            return self.recovery_budget
        return self.physical_skill_budget


@dataclasses.dataclass(frozen=True)
class VlaRunConfig:
    """Frozen VLA and admitted deployment-profile identity."""

    policy_id: str
    adapter_id: str
    model_id: str
    checkpoint_id: str
    deployment_profile_id: str
    endpoint: str = ""
    profile_manifest_path: str = ""

    def __post_init__(self) -> None:
        required = (
            self.policy_id,
            self.adapter_id,
            self.model_id,
            self.checkpoint_id,
            self.deployment_profile_id,
        )
        if any(not value.strip() for value in required):
            raise ValueError("VLA identity and deployment profile fields must not be empty")


@dataclasses.dataclass(frozen=True)
class OptimizeRunConfig:
    """Runtime enforcement settings; optimization evidence lives in its manifest."""

    enabled: bool = True
    enforce_profile_admission: bool = True
    allow_reference_fallback: bool = True
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when configured")
        if self.enabled and not self.enforce_profile_admission:
            raise ValueError("enabled Optimize Runtime must enforce profile admission")


@dataclasses.dataclass(frozen=True)
class BenchmarkRunConfig:
    """Public benchmark identity; evaluator state is deliberately absent."""

    environment_id: str
    suite_id: str
    task_id: str
    seed: int
    max_steps: int
    record_video: bool = True

    def __post_init__(self) -> None:
        required = (self.environment_id, self.suite_id, self.task_id)
        if any(not value.strip() for value in required):
            raise ValueError("benchmark identity fields must not be empty")
        if self.seed < 0 or self.max_steps <= 0:
            raise ValueError("benchmark seed must be non-negative and max_steps positive")


@dataclasses.dataclass(frozen=True)
class ArtifactRunConfig:
    results_root: str = "results/carve_runs"
    persist_frames: bool = False

    def __post_init__(self) -> None:
        if not self.results_root.strip():
            raise ValueError("results_root must not be empty")


@dataclasses.dataclass(frozen=True)
class CarveRunConfig:
    """Single source of truth for framework and benchmark runners."""

    run_id: str
    planner: PlannerRunConfig
    harness: HarnessRunConfig
    vla: VlaRunConfig
    optimize: OptimizeRunConfig
    benchmark: BenchmarkRunConfig
    artifacts: ArtifactRunConfig = dataclasses.field(default_factory=ArtifactRunConfig)
    schema_version: int = RUN_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported CARVE run config schema: {self.schema_version}")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.planner.mode is PlannerExecutionMode.EXTERNAL_CODING_AGENT:
            if self.harness.task_start_policy is not TaskStartPolicy.EVENT_ONLY:
                raise ValueError(
                    "external coding agents require event_only task start; they drive tools explicitly"
                )
            if (
                self.harness.primitive_boundary_policy
                is not PrimitiveBoundaryPolicy.EVENT_ONLY
            ):
                raise ValueError(
                    "external coding agents require event_only primitive boundaries"
                )
        _reject_secrets(self.to_dict())

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CarveRunConfig":
        _reject_secrets(values)
        _reject_unknown_fields(
            values,
            {
                "schema_version",
                "run_id",
                "planner",
                "harness",
                "vla",
                "optimize",
                "benchmark",
                "artifacts",
            },
            section="config",
        )
        sections = {
            "planner": PlannerRunConfig,
            "harness": HarnessRunConfig,
            "vla": VlaRunConfig,
            "optimize": OptimizeRunConfig,
            "benchmark": BenchmarkRunConfig,
            "artifacts": ArtifactRunConfig,
        }
        payload = dict(values)
        for name, constructor in sections.items():
            section = payload.get(name, {})
            if not isinstance(section, Mapping):
                raise TypeError(f"config.{name} must be an object")
            allowed = {field.name for field in dataclasses.fields(constructor)}
            _reject_unknown_fields(section, allowed, section=f"config.{name}")
            payload[name] = constructor(**dict(section))
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "CarveRunConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("CARVE run config root must be a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if dataclasses.is_dataclass(value):
                return {
                    field.name: normalize(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            return value

        return normalize(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def workspace_path(self) -> Path:
        return Path(self.artifacts.results_root) / self.run_id

    def to_run_manifest(self) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            environment_id=self.benchmark.environment_id,
            task_id=f"{self.benchmark.suite_id}:{self.benchmark.task_id}",
            seed=self.benchmark.seed,
            planner_id=self.planner.planner_id,
            policy_id=self.vla.policy_id,
            deployment_profile_id=self.vla.deployment_profile_id,
            schema_version=RUN_CONFIG_SCHEMA_VERSION,
            metadata={
                "config_fingerprint": self.fingerprint,
                "planner_mode": self.planner.mode.value,
                "planner_provider": (
                    self.planner.provider.value if self.planner.provider else None
                ),
                "planner_model": self.planner.model,
                "prompt_schema_version": self.planner.prompt_schema_version,
                "task_start_policy": self.harness.task_start_policy.value,
                "primitive_boundary_policy": (
                    self.harness.primitive_boundary_policy.value
                ),
                "physical_skill_budget": self.harness.effective_physical_skill_budget,
                "adapter_id": self.vla.adapter_id,
                "model_id": self.vla.model_id,
                "checkpoint_id": self.vla.checkpoint_id,
                "profile_manifest_path": self.vla.profile_manifest_path,
                "optimize_enabled": self.optimize.enabled,
                "deadline_ms": self.optimize.deadline_ms,
            },
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
