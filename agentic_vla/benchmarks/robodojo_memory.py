"""Read-only procedural context with explicit RoboDojo development/test splits."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from agentic_vla.runtime.agent import HighLevelAgentContext
from agentic_vla.runtime.knowledge import ProceduralTaskMemory


class FrozenRoboDojoProcedureMemory:
    """Retrieve existing verified procedures; never write during evaluation.

    The manifest binds the memory file and its episode provenance. Trustworthy
    procedure promotion remains the responsibility of VerifiedProcedureCompiler.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_sha256: str,
        layout_set: int,
    ) -> None:
        path = Path(manifest_path).expanduser().resolve()
        manifest_bytes = path.read_bytes()
        actual_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if not expected_sha256 or actual_hash != expected_sha256:
            raise ValueError("procedure manifest SHA256 mismatch")
        payload = json.loads(manifest_bytes)
        if not isinstance(payload, dict) or payload.get("schema_version") != (
            "carve.robodojo.procedure-memory-manifest.v1"
        ):
            raise ValueError("unsupported procedure memory manifest")
        if payload.get("benchmark") != "RoboDojo":
            raise ValueError("procedure memory benchmark must be RoboDojo")

        development = self._layout_sets(payload, "development_layout_sets")
        evaluation = self._layout_sets(payload, "evaluation_layout_sets")
        if development & evaluation:
            raise ValueError("development and evaluation layout sets overlap")
        if type(layout_set) is not int or layout_set not in evaluation:
            raise ValueError("current layout set is not a registered evaluation set")

        filename = payload.get("memory_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError("memory_file must name a local snapshot")
        memory_path = (path.parent / filename).resolve()
        if not memory_path.is_relative_to(path.parent):
            raise ValueError("memory_file must remain inside the snapshot directory")
        memory_hash = hashlib.sha256(memory_path.read_bytes()).hexdigest()
        if memory_hash != payload.get("memory_sha256"):
            raise ValueError("procedure memory SHA256 mismatch")
        memory = ProceduralTaskMemory.load(memory_path)
        if not len(memory):
            raise ValueError("procedure memory snapshot is empty")
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise ValueError("procedure provenance must be a list")
        indexed = {}
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("each procedure source must be an object")
            key = source.get("procedure_id")
            if not isinstance(key, str) or not key or key in indexed:
                raise ValueError("procedure provenance IDs must be unique")
            source_set = source.get("layout_set")
            if type(source_set) is not int or source_set not in development:
                raise ValueError("procedure source is outside the development split")
            indexed[key] = source
        if set(indexed) != {record.procedure_id for record in memory.records}:
            raise ValueError("procedure provenance does not cover the snapshot exactly")
        for record in memory.records:
            if (
                indexed[record.procedure_id].get("source_episode_id")
                != record.source_episode_id
            ):
                raise ValueError("procedure source episode identity mismatch")

        self.manifest_sha256 = actual_hash
        self.memory_sha256 = memory_hash
        self._memory = memory

    @staticmethod
    def _layout_sets(payload: dict, name: str) -> set[int]:
        values = payload.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain distinct non-negative integers")
        return set(values)

    def enrich(self, context: HighLevelAgentContext) -> HighLevelAgentContext:
        records = self._memory.retrieve(
            task_instruction=context.task_instruction, limit=2
        )
        return dataclasses.replace(
            context,
            procedural_retrievals=records,
            memory_context_fingerprint=self.manifest_sha256,
        )

    def receipt(self, context: HighLevelAgentContext) -> dict:
        return {
            "enabled": True,
            "read_only": True,
            "manifest_sha256": self.manifest_sha256,
            "memory_sha256": self.memory_sha256,
            "hit": bool(context.procedural_retrievals),
            "procedure_ids": [
                record["procedure_id"] for record in context.procedural_retrievals
            ],
            "source_episode_ids": [
                record["source_episode_id"] for record in context.procedural_retrievals
            ],
        }
