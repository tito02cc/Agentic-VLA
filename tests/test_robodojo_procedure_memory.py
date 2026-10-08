from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agentic_vla.benchmarks.robodojo_memory import FrozenRoboDojoProcedureMemory
from agentic_vla.runtime import HighLevelAgentContext, build_high_level_agent_request
from agentic_vla.runtime.knowledge import (
    ProceduralStep,
    ProceduralTaskMemory,
    ProceduralTaskRecord,
)

ROBODOJO_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "robodojo_official"
)
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))


@pytest.fixture
def snapshot(tmp_path):
    record = ProceduralTaskRecord(
        procedure_id="test-procedure",
        task_family="tower",
        object_categories=("blocks",),
        steps=(ProceduralStep("base", "vla_act", "build the base", "stable base"),),
        verification_result="verified",
        source_episode_id="development-episode",
    )
    memory_path = ProceduralTaskMemory([record]).save(tmp_path / "memory.json")
    manifest = {
        "schema_version": "carve.robodojo.procedure-memory-manifest.v1",
        "benchmark": "RoboDojo",
        "development_layout_sets": [0],
        "evaluation_layout_sets": [1, 2],
        "memory_file": "memory.json",
        "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
        "sources": [
            {
                "procedure_id": "test-procedure",
                "layout_set": 0,
                "source_episode_id": "development-episode",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def load(path, *, layout_set=1):
    return FrozenRoboDojoProcedureMemory(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layout_set=layout_set,
    )


@pytest.mark.parametrize("trigger", ["task_start", "no_progress"])
def test_memory_reaches_compact_planner_requests(snapshot, trigger):
    path, _ = snapshot
    store = load(path)
    original = HighLevelAgentContext(
        task_instruction="Build a tower using the blocks.",
        trigger=trigger,
        episode_id="evaluation-episode",
        timestep=0,
        risk={"event": "no_progress"},
    )
    context = store.enrich(original)
    request = build_high_level_agent_request(context)
    payload = json.loads(request["user_prompt"])
    assert payload["procedural_retrievals"][0]["procedure_id"] == "test-procedure"
    assert payload["memory_context_fingerprint"] == store.manifest_sha256
    assert "re-ground targets from current images" in request["system_prompt"]
    assert original.procedural_retrievals == ()
    assert store.receipt(context)["hit"] is True


def test_irrelevant_memory_is_not_inserted(snapshot):
    path, _ = snapshot
    store = load(path)
    context = store.enrich(
        HighLevelAgentContext(
            task_instruction="open the drawer",
            trigger="task_start",
            episode_id="eval",
            timestep=0,
        )
    )
    assert not store.receipt(context)["hit"]
    assert "procedural_retrievals" not in json.loads(
        build_high_level_agent_request(context)["user_prompt"]
    )


def test_snapshot_is_reusable_without_persisting_episode_state(snapshot):
    path, _ = snapshot
    before = path.read_bytes()
    memory_before = (path.parent / "memory.json").read_bytes()
    store = load(path)
    for episode in ("eval-1", "eval-2"):
        context = store.enrich(
            HighLevelAgentContext(
                task_instruction="build a tower",
                trigger="task_start",
                episode_id=episode,
                timestep=0,
            )
        )
        context.procedural_retrievals[0]["steps"][0]["subgoal"] = "caller mutation"
    fresh = store.enrich(
        HighLevelAgentContext(
            task_instruction="build a tower",
            trigger="task_start",
            episode_id="eval-3",
            timestep=0,
        )
    )
    assert fresh.procedural_retrievals[0]["steps"][0]["subgoal"] == "build the base"
    assert path.read_bytes() == before
    assert (path.parent / "memory.json").read_bytes() == memory_before


def test_rejects_development_layout_deployment(snapshot):
    path, _ = snapshot
    with pytest.raises(ValueError, match="registered evaluation"):
        load(path, layout_set=0)


def test_rejects_changed_memory_bytes(snapshot):
    path, _ = snapshot
    memory = path.parent / "memory.json"
    memory.write_text(memory.read_text() + "\n")
    with pytest.raises(ValueError, match="memory SHA256"):
        load(path)


def test_rejects_changed_manifest(snapshot):
    path, _ = snapshot
    with pytest.raises(ValueError, match="manifest SHA256"):
        FrozenRoboDojoProcedureMemory(path, expected_sha256="0" * 64, layout_set=1)


@pytest.mark.parametrize("mutation", ["overlap", "missing", "identity", "source_split"])
def test_rejects_invalid_provenance(snapshot, mutation):
    path, manifest = snapshot
    if mutation == "overlap":
        manifest["development_layout_sets"] = [0, 1]
    elif mutation == "missing":
        manifest["sources"] = []
    elif mutation == "identity":
        manifest["sources"][0]["source_episode_id"] = "another-episode"
    else:
        manifest["sources"][0]["layout_set"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load(path)


def test_bridge_retrieval_passes_context_and_emits_provenance(snapshot):
    from XPolicyLab.policy.starVLA.model import Model

    path, _ = snapshot
    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_procedure_memory_manifest": str(path),
        "carve_procedure_memory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": 1,
    }
    rows = []
    model._record_planner_row = rows.append
    model._initialize_procedure_memory(backend="openai", replay_path=None)
    original = HighLevelAgentContext(
        task_instruction="build a tower",
        trigger="task_start",
        episode_id=0,
        timestep=0,
    )
    enriched = model._with_procedural_context(original)
    assert enriched.procedural_retrievals
    assert rows[0]["procedure_ids"] == ["test-procedure"]
    assert rows[0]["read_only"] is True
    assert not original.procedural_retrievals


@pytest.mark.parametrize(
    "backend,replay", [("task_preserving", None), ("openai", "ticket.json")]
)
def test_bridge_rejects_memory_that_cannot_reach_a_live_planner(
    snapshot, backend, replay
):
    from XPolicyLab.policy.starVLA.model import Model

    path, _ = snapshot
    model = Model.__new__(Model)
    model.model_cfg = {"carve_procedure_memory_manifest": str(path)}
    with pytest.raises(ValueError, match="live semantic planner"):
        model._initialize_procedure_memory(backend=backend, replay_path=replay)


def test_bridge_reset_clears_working_state_but_retains_frozen_procedures(snapshot):
    from XPolicyLab.policy.starVLA.model import Model
    from XPolicyLab.policy.starVLA.reset_contract import STARVLA_RESET_STATE_REGISTRY

    path, _ = snapshot
    model = Model.__new__(Model)
    for _, attribute, kind in STARVLA_RESET_STATE_REGISTRY:
        setattr(model, attribute, {} if kind == "dict" else set())
    model._high_level_agent = None
    model._agent_memory_by_env[0] = ["last episode's unfinished target"]
    model._safe_stop_envs.add(0)
    model._procedure_memory = load(path)
    frozen = model._procedure_memory

    receipt = model.reset(reset_context={"episode_id": "next-episode"})

    assert receipt["before"]["state_entries"]["agent_memory_by_env"] == 1
    assert all(count == 0 for count in receipt["after"]["state_entries"].values())
    assert model._procedure_memory is frozen
    context = frozen.enrich(
        HighLevelAgentContext(
            task_instruction="build a tower",
            trigger="task_start",
            episode_id="next-episode",
            timestep=0,
        )
    )
    assert frozen.receipt(context)["procedure_ids"] == ["test-procedure"]
    assert context.memory == ()


def test_rejects_memory_path_outside_snapshot(snapshot):
    path, manifest = snapshot
    manifest["memory_file"] = "../outside.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the snapshot"):
        load(path)


def test_planner_trace_records_actual_reset_generation(tmp_path):
    from XPolicyLab.policy.starVLA.model import Model

    model = Model.__new__(Model)
    model._planner_trace_path = tmp_path / "planner.jsonl"
    model._reset_generation = 3
    row = {"action": "retrieve_procedure_context", "timestep": 0}
    model._record_planner_row(row)
    written = json.loads(model._planner_trace_path.read_text())
    assert written["policy_reset_generation"] == 3
    assert "policy_reset_generation" not in row
