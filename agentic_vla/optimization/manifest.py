"""Machine-readable deployment manifests for calibrated CARVE profiles."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from collections.abc import Mapping
from typing import Any

from .contracts import HardwareSpec, OptimizationProfile


@dataclasses.dataclass(frozen=True)
class ProfileManifest:
    """Bind one validated profile to a model checkpoint and hardware target."""

    model_id: str
    adapter_id: str
    checkpoint_id: str
    hardware: HardwareSpec
    profile: OptimizationProfile
    benchmark: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    fidelity: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    admission: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    plugin_versions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_at: str = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("model_id", "adapter_id", "checkpoint_id", "created_at"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "adapter_id": self.adapter_id,
            "checkpoint_id": self.checkpoint_id,
            "hardware": self.hardware.to_dict(),
            "profile": self.profile.to_dict(),
            "benchmark": dict(self.benchmark),
            "fidelity": dict(self.fidelity),
            "admission": dict(self.admission),
            "plugin_versions": dict(self.plugin_versions),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ProfileManifest":
        payload = dict(values)
        payload["hardware"] = HardwareSpec.from_dict(payload["hardware"])
        payload["profile"] = OptimizationProfile.from_dict(payload["profile"])
        return cls(**payload)

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        output = pathlib.Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ProfileManifest":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("manifest root must be a JSON object")
        return cls.from_dict(payload)
