"""Deployable semantic scene graphs and affordance-aware experience retrieval."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_FORBIDDEN_PRIVILEGED_FIELDS = frozenset(
    {
        "action",
        "actions",
        "joint_positions",
        "joint_targets",
        "object_pose",
        "object_poses",
        "reward",
        "sim_state",
        "simulator_state",
        "success",
        "torques",
        "trajectory",
    }
)
_ALLOWED_RELATIONS = frozenset(
    {
        "above",
        "aligned_with",
        "below",
        "closed",
        "contains",
        "grasped_by",
        "inside",
        "left_of",
        "near",
        "occluded_by",
        "occludes",
        "open",
        "reachable",
        "right_of",
        "supported_by",
        "supports",
    }
)


def reject_privileged_semantic_fields(value: Any, *, path: str = "semantic") -> None:
    """Reject evaluator-only state and raw actions from planner-side evidence."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PRIVILEGED_FIELDS:
                raise ValueError(f"{path} contains forbidden field: {normalized}")
            reject_privileged_semantic_fields(item, path=f"{path}.{normalized}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reject_privileged_semantic_fields(item, path=f"{path}[{index}]")


def _clean_terms(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    terms = tuple(str(value).strip().lower() for value in values)
    if any(not value for value in terms):
        raise ValueError(f"{field_name} must not contain empty values")
    return terms


def _lexical_tokens(value: str) -> set[str]:
    """Normalize natural-language and identifier-style semantic labels."""

    return set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


@dataclasses.dataclass(frozen=True)
class SceneEntity:
    """One entity produced by deployable visual semantic perception."""

    entity_id: str
    category: str
    attributes: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        entity_id = self.entity_id.strip()
        category = self.category.strip().lower()
        if not entity_id or not category:
            raise ValueError("scene entity id and category must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("scene entity confidence must be in [0, 1]")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(
            self,
            "attributes",
            _clean_terms(self.attributes, field_name="scene entity attributes"),
        )


@dataclasses.dataclass(frozen=True)
class SceneRelation:
    """A qualitative relation; metric simulator poses are intentionally absent."""

    subject: str
    relation: str
    object: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        subject = self.subject.strip()
        relation = self.relation.strip().lower()
        object_id = self.object.strip()
        if not subject or not object_id:
            raise ValueError("scene relation endpoints must not be empty")
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError(f"unsupported semantic scene relation: {relation}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("scene relation confidence must be in [0, 1]")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "object", object_id)


@dataclasses.dataclass(frozen=True)
class SemanticSceneGraph:
    """Observation-side qualitative graph consumed by the high-level planner."""

    entities: tuple[SceneEntity, ...] = ()
    relations: tuple[SceneRelation, ...] = ()
    source: str = "vlm_observation"
    timestep: int = 0

    def __post_init__(self) -> None:
        if self.source not in {"vlm_observation", "rgbd_perception", "human_annotation"}:
            raise ValueError("scene graph source must be deployable or annotation-only")
        if self.timestep < 0:
            raise ValueError("scene graph timestep must be non-negative")
        entity_ids = tuple(entity.entity_id for entity in self.entities)
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("scene graph entity ids must be unique")
        known = set(entity_ids)
        for relation in self.relations:
            if relation.subject not in known or relation.object not in known:
                raise ValueError("scene relation references an unknown entity")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_semantic_scene_graph(
    payload: Mapping[str, Any],
    *,
    source: str = "vlm_observation",
    timestep: int = 0,
) -> SemanticSceneGraph:
    """Parse a VLM/perception JSON response without accepting privileged state."""

    if not isinstance(payload, Mapping):
        raise TypeError("scene graph payload must be a mapping")
    reject_privileged_semantic_fields(payload, path="scene_graph")
    raw_entities = payload.get("entities", ())
    raw_relations = payload.get("relations", ())
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        raise TypeError("scene graph entities must be a sequence")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise TypeError("scene graph relations must be a sequence")
    for index, item in enumerate(raw_entities):
        if not isinstance(item, Mapping):
            raise TypeError(f"scene graph entity {index} must be a JSON object")
    for index, item in enumerate(raw_relations):
        if not isinstance(item, Mapping):
            raise TypeError(f"scene graph relation {index} must be a JSON object")

    entities = tuple(
        SceneEntity(
            entity_id=str(item["entity_id"]),
            category=str(item["category"]),
            attributes=tuple(item.get("attributes", ())),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in raw_entities
    )
    relations = tuple(
        SceneRelation(
            subject=str(item["subject"]),
            relation=str(item["relation"]),
            object=str(item["object"]),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in raw_relations
    )
    return SemanticSceneGraph(
        entities=entities,
        relations=relations,
        source=source,
        timestep=timestep,
    )


@dataclasses.dataclass(frozen=True)
class AffordanceExperience:
    """One HAA-RAG card; it describes constraints rather than robot actions."""

    experience_id: str
    object_categories: tuple[str, ...]
    affordances: tuple[str, ...]
    grasp_regions: tuple[str, ...] = ()
    material: str = ""
    fragility: str = ""
    applicable_failures: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    outcome: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.experience_id.strip():
            raise ValueError("experience_id must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("affordance experience confidence must be in [0, 1]")
        object.__setattr__(
            self,
            "object_categories",
            _clean_terms(self.object_categories, field_name="object_categories"),
        )
        object.__setattr__(
            self,
            "affordances",
            _clean_terms(self.affordances, field_name="affordances"),
        )
        object.__setattr__(
            self,
            "grasp_regions",
            _clean_terms(self.grasp_regions, field_name="grasp_regions"),
        )
        object.__setattr__(
            self,
            "applicable_failures",
            _clean_terms(self.applicable_failures, field_name="applicable_failures"),
        )
        object.__setattr__(
            self,
            "constraints",
            tuple(str(value).strip() for value in self.constraints),
        )
        if not self.object_categories or not self.affordances:
            raise ValueError("affordance experience requires categories and affordances")
        reject_privileged_semantic_fields(self.to_dict(), path="affordance_experience")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class HAAExperienceIndex:
    """Deterministic, training-free retrieval for affordance-aware experience."""

    def __init__(self, records: Sequence[AffordanceExperience] = ()) -> None:
        self._records: dict[str, AffordanceExperience] = {}
        for record in records:
            self.add(record)

    def add(self, record: AffordanceExperience) -> None:
        if not isinstance(record, AffordanceExperience):
            raise TypeError("HAA index accepts only AffordanceExperience records")
        if record.experience_id in self._records:
            raise ValueError(f"duplicate HAA experience id: {record.experience_id}")
        self._records[record.experience_id] = record

    def retrieve(
        self,
        *,
        task_instruction: str,
        scene_graph: SemanticSceneGraph | None = None,
        failure_type: str = "",
        limit: int = 3,
    ) -> tuple[Mapping[str, Any], ...]:
        if not task_instruction.strip():
            raise ValueError("task_instruction must not be empty")
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        tokens = _lexical_tokens(task_instruction)
        visible_categories = (
            {entity.category for entity in scene_graph.entities}
            if scene_graph is not None
            else set()
        )
        failure = failure_type.strip().lower()
        candidates: list[tuple[int, bool, AffordanceExperience]] = []
        for record in self._records.values():
            category_matches = sum(
                category in tokens or category in visible_categories
                for category in record.object_categories
            )
            failure_match = bool(
                failure and failure in set(record.applicable_failures)
            )
            if category_matches == 0 and not failure_match:
                continue
            candidates.append((category_matches, failure_match, record))

        has_task_or_scene_match = any(matches > 0 for matches, _, _ in candidates)
        ranked: list[tuple[float, str, AffordanceExperience]] = []
        for category_matches, failure_match, record in candidates:
            if has_task_or_scene_match and category_matches == 0:
                continue
            score = (
                2.0 * category_matches
                + 1.5 * float(failure_match)
                + 0.25 * float(record.confidence)
            )
            ranked.append((score, record.experience_id, record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            {
                **record.to_dict(),
                "retrieval_score": score,
                "retrieval_failure": failure,
            }
            for score, _, record in ranked[:limit]
        )


_PROCEDURAL_INTENTS = frozenset({"continue", "vla_act", "run_skill", "verify"})


@dataclasses.dataclass(frozen=True)
class ProceduralStep:
    """One symbolic task stage; metric poses and low-level actions are forbidden."""

    stage: str
    intent: str
    subgoal: str
    expected_outcome: str
    skill_id: str | None = None
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stage = self.stage.strip().lower()
        intent = self.intent.strip().lower()
        subgoal = self.subgoal.strip()
        expected = self.expected_outcome.strip()
        if not stage or not subgoal or not expected:
            raise ValueError("procedural stage, subgoal, and expected outcome are required")
        if intent not in _PROCEDURAL_INTENTS:
            raise ValueError(f"unsupported procedural intent: {intent}")
        skill_id = None if self.skill_id is None else self.skill_id.strip()
        if intent == "run_skill" and not skill_id:
            raise ValueError("run_skill procedural step requires skill_id")
        if intent != "run_skill" and skill_id is not None:
            raise ValueError("only run_skill procedural steps may name a skill")
        constraints = tuple(str(value).strip() for value in self.constraints)
        if any(not value for value in constraints):
            raise ValueError("procedural constraints must not contain empty values")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "subgoal", subgoal)
        object.__setattr__(self, "expected_outcome", expected)
        object.__setattr__(self, "skill_id", skill_id)
        object.__setattr__(self, "constraints", constraints)
        reject_privileged_semantic_fields(self.to_dict(), path="procedural_step")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProceduralTaskRecord:
    """A verified symbolic procedure retrieved as planner context, never execution."""

    procedure_id: str
    task_family: str
    object_categories: tuple[str, ...]
    steps: tuple[ProceduralStep, ...]
    verification_result: str
    source_episode_id: str | int
    confidence: float = 1.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        procedure_id = self.procedure_id.strip()
        task_family = self.task_family.strip().lower()
        verification = self.verification_result.strip().lower()
        if not procedure_id or not task_family:
            raise ValueError("procedure_id and task_family must not be empty")
        if not self.steps:
            raise ValueError("procedural task record requires at least one step")
        if verification != "verified":
            raise ValueError("only deployably verified procedures may enter memory")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("procedural task confidence must be in [0, 1]")
        categories = _clean_terms(
            self.object_categories,
            field_name="procedural object_categories",
        )
        notes = tuple(str(value).strip() for value in self.notes)
        if any(not value for value in notes):
            raise ValueError("procedural notes must not contain empty values")
        object.__setattr__(self, "procedure_id", procedure_id)
        object.__setattr__(self, "task_family", task_family)
        object.__setattr__(self, "verification_result", verification)
        object.__setattr__(self, "object_categories", categories)
        object.__setattr__(self, "notes", notes)
        reject_privileged_semantic_fields(self.to_dict(), path="procedural_record")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ProceduralTaskMemory:
    """Bounded cross-episode memory of verified, symbolic task procedures."""

    def __init__(
        self,
        records: Sequence[ProceduralTaskRecord] = (),
        *,
        max_records: int = 128,
    ) -> None:
        if max_records <= 0:
            raise ValueError("procedural memory max_records must be positive")
        self._records: deque[ProceduralTaskRecord] = deque(maxlen=max_records)
        for record in records:
            self.record(record)

    def record(self, record: ProceduralTaskRecord) -> None:
        if not isinstance(record, ProceduralTaskRecord):
            raise TypeError("procedural memory accepts only ProceduralTaskRecord")
        if any(item.procedure_id == record.procedure_id for item in self._records):
            raise ValueError(f"duplicate procedure_id: {record.procedure_id}")
        self._records.append(record)

    def retrieve(
        self,
        *,
        task_instruction: str,
        scene_graph: SemanticSceneGraph | None = None,
        limit: int = 2,
    ) -> tuple[Mapping[str, Any], ...]:
        if not task_instruction.strip():
            raise ValueError("task_instruction must not be empty")
        if limit <= 0:
            raise ValueError("procedural retrieval limit must be positive")
        tokens = _lexical_tokens(task_instruction)
        visible_categories = (
            {entity.category for entity in scene_graph.entities}
            if scene_graph is not None
            else set()
        )
        ranked: list[tuple[float, str, ProceduralTaskRecord]] = []
        for record in self._records:
            family_tokens = _lexical_tokens(record.task_family)
            family_matches = len(tokens & family_tokens)
            category_matches = sum(
                _lexical_tokens(category).issubset(tokens)
                or category in visible_categories
                for category in record.object_categories
            )
            if family_matches == 0 and category_matches == 0:
                continue
            score = (
                1.5 * family_matches
                + 2.0 * category_matches
                + 0.25 * float(record.confidence)
            )
            ranked.append((score, record.procedure_id, record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            {**record.to_dict(), "retrieval_score": score}
            for score, _, record in ranked[:limit]
        )

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[ProceduralTaskRecord, ...]:
        return tuple(self._records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [record.to_dict() for record in self._records],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_records: int = 128,
    ) -> "ProceduralTaskMemory":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("unsupported procedural-memory payload")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("procedural-memory records must be a list")
        records: list[ProceduralTaskRecord] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise TypeError(f"procedural-memory record {index} must be an object")
            values = dict(raw_record)
            raw_steps = values.get("steps")
            if not isinstance(raw_steps, list):
                raise TypeError(f"procedural-memory record {index} steps must be a list")
            values["steps"] = tuple(
                ProceduralStep(**dict(step))
                for step in raw_steps
                if isinstance(step, Mapping)
            )
            if len(values["steps"]) != len(raw_steps):
                raise TypeError(
                    f"procedural-memory record {index} contains a non-object step"
                )
            for field in ("object_categories", "notes"):
                raw_values = values.get(field, ())
                if not isinstance(raw_values, list):
                    raise TypeError(
                        f"procedural-memory record {index} {field} must be a list"
                    )
                values[field] = tuple(raw_values)
            records.append(ProceduralTaskRecord(**values))
        return cls(records, max_records=max_records)


@dataclasses.dataclass(frozen=True)
class AgenticKnowledgeBundle:
    """The complete semantic evidence attached to one planner invocation."""

    scene_graph: SemanticSceneGraph | None = None
    affordance_retrievals: tuple[Mapping[str, Any], ...] = ()
    procedural_retrievals: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        reject_privileged_semantic_fields(self.to_dict(), path="knowledge_bundle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_graph": (
                None if self.scene_graph is None else self.scene_graph.to_dict()
            ),
            "affordance_retrievals": [
                dict(record) for record in self.affordance_retrievals
            ],
            "procedural_retrievals": [
                dict(record) for record in self.procedural_retrievals
            ],
        }


class AgenticKnowledgeProvider:
    """Build planner context from deployable perception and symbolic memories."""

    def __init__(
        self,
        index: HAAExperienceIndex | None = None,
        procedural_memory: ProceduralTaskMemory | None = None,
    ) -> None:
        self.index = index if index is not None else HAAExperienceIndex()
        self.procedural_memory = (
            procedural_memory
            if procedural_memory is not None
            else ProceduralTaskMemory()
        )
        self._build_calls = 0
        self._affordance_hits = 0
        self._procedural_hits = 0

    def retrieval_metrics(self) -> dict[str, int]:
        """Return aggregate planner-facing retrieval counts for run receipts."""

        return {
            "build_calls": self._build_calls,
            "affordance_hits": self._affordance_hits,
            "procedural_hits": self._procedural_hits,
        }

    def build(
        self,
        *,
        task_instruction: str,
        scene_graph: SemanticSceneGraph | Mapping[str, Any] | None,
        failure_type: str = "",
        timestep: int = 0,
        limit: int = 3,
        procedural_limit: int = 2,
    ) -> AgenticKnowledgeBundle:
        graph = scene_graph
        if isinstance(scene_graph, Mapping):
            graph = parse_semantic_scene_graph(scene_graph, timestep=timestep)
        if graph is not None and not isinstance(graph, SemanticSceneGraph):
            raise TypeError("scene_graph must be SemanticSceneGraph, mapping, or null")
        retrievals = self.index.retrieve(
            task_instruction=task_instruction,
            scene_graph=graph,
            failure_type=failure_type,
            limit=limit,
        )
        procedures = self.procedural_memory.retrieve(
            task_instruction=task_instruction,
            scene_graph=graph,
            limit=procedural_limit,
        )
        self._build_calls += 1
        self._affordance_hits += len(retrievals)
        self._procedural_hits += len(procedures)
        return AgenticKnowledgeBundle(
            scene_graph=graph,
            affordance_retrievals=retrievals,
            procedural_retrievals=procedures,
        )
