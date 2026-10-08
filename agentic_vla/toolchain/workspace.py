"""Auditable run workspace shared by CARVE planners, tools, and evaluators."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import ToolCall, ToolResult, reject_direct_action_fields


_ARTIFACT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".log", ".md", ".mp4", ".npy", ".npz", ".png", ".txt"}
)
_FINAL_STATUSES = frozenset(
    {"success", "failure", "stuck", "safe_stop", "completed"}
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclasses.dataclass(frozen=True)
class RunManifest:
    """Identity needed to reproduce one embodied-agent run."""

    run_id: str
    environment_id: str
    task_id: str
    seed: int
    planner_id: str
    policy_id: str
    deployment_profile_id: str
    schema_version: int = 1
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported CARVE run-manifest schema")
        required = (
            self.run_id,
            self.environment_id,
            self.task_id,
            self.planner_id,
            self.policy_id,
            self.deployment_profile_id,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("run-manifest identity fields must not be empty")
        if self.seed < 0:
            raise ValueError("run-manifest seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class RunWorkspace:
    """Persist one run's control evidence without exposing it as planner authority."""

    def __init__(self, root: str | Path, manifest: RunManifest) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self._lock = threading.Lock()
        self._event_sequence = self._next_event_sequence()
        artifacts_path = self.root / "artifacts.json"
        self._artifact_index: dict[str, dict[str, Any]] = (
            json.loads(artifacts_path.read_text(encoding="utf-8"))
            if artifacts_path.exists()
            else {}
        )
        manifest_path = self.root / "run_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest.to_dict():
                raise ValueError(
                    "run workspace already contains a different manifest"
                )
        else:
            self._write_json_atomic("run_manifest.json", manifest.to_dict())

    @property
    def event_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def recipe_path(self) -> Path:
        return self.root / "recipe.jsonl"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.jsonl"

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        """Append internal evidence; simulator-only fields stay evaluator-side."""

        if not event_type.strip() or not source.strip():
            raise ValueError("event type and source must not be empty")
        with self._lock:
            record = {
                "sequence": self._event_sequence,
                "timestamp_s": time.time(),
                "event_type": event_type,
                "source": source,
                "payload": dict(payload),
            }
            self._event_sequence += 1
            self._append_jsonl(self.event_path, record)
        return record

    def append_recipe(self, call: ToolCall, result: ToolResult) -> dict[str, Any]:
        """Persist symbolic tool order; low-level VLA actions are intentionally absent."""

        if call.call_id != result.call_id or call.name != result.name:
            raise ValueError("tool call and result identities do not match")
        reject_direct_action_fields(call.arguments, path="recipe.arguments")
        record = {
            "call_id": call.call_id,
            "episode_id": call.episode_id,
            "timestep": call.timestep,
            "tool": call.name,
            "arguments": dict(call.arguments),
            "accepted": result.accepted,
            "elapsed_ms": result.elapsed_ms,
            "error": result.error,
        }
        with self._lock:
            self._append_jsonl(self.recipe_path, record)
        return record

    def append_transcript(
        self,
        *,
        role: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if role not in {"system", "user", "planner", "tool"}:
            raise ValueError("unsupported transcript role")
        if not content.strip():
            raise ValueError("transcript content must not be empty")
        with self._lock:
            self._append_jsonl(
                self.transcript_path,
                {
                    "timestamp_s": time.time(),
                    "role": role,
                    "content": content,
                    "metadata": dict(metadata or {}),
                },
            )

    def register_artifact(
        self,
        name: str,
        *,
        path: str | Path,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        artifact = Path(name)
        if artifact.name != name or artifact.suffix.lower() not in _ARTIFACT_SUFFIXES:
            raise ValueError("artifact name must be a safe supported base filename")
        target = Path(path)
        try:
            relative = target.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("artifact path must remain inside the run workspace") from exc
        if not kind.strip():
            raise ValueError("artifact kind must not be empty")
        self._artifact_index[name] = {
            "path": str(relative),
            "kind": kind,
            "metadata": dict(metadata or {}),
        }
        self._write_json_atomic("artifacts.json", self._artifact_index)

    def finish(self, *, status: str, summary: str) -> Path:
        normalized = status.strip().lower()
        if normalized not in _FINAL_STATUSES:
            raise ValueError(f"unsupported final status: {status}")
        if not summary.strip():
            raise ValueError("run summary must not be empty")
        destination = self.root / "run_summary.json"
        self._write_json_atomic(
            destination.name,
            {
                "run_id": self.manifest.run_id,
                "status": normalized,
                "summary": summary,
                "finished_at_s": time.time(),
                "event_count": self._event_sequence,
                "artifacts": sorted(self._artifact_index),
            },
        )
        return destination

    def _write_json_atomic(self, name: str, payload: Any) -> None:
        destination = self.root / name
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _next_event_sequence(self) -> int:
        if not self.event_path.exists():
            return 0
        last_sequence = -1
        with self.event_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sequence = int(record["sequence"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid workspace event at line {line_number}"
                    ) from exc
                if sequence <= last_sequence:
                    raise ValueError("workspace event sequence is not increasing")
                last_sequence = sequence
        return last_sequence + 1

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=_json_default) + "\n")
