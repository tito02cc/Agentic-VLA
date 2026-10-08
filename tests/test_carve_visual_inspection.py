import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.toolchain import (
    CanonicalToolRuntime,
    EmbodiedToolBindings,
    EmbodiedToolRegistry,
    RunManifest,
    RunWorkspace,
    ToolCall,
    ToolExecutionContext,
    VisualEvidenceStore,
    core_tool_specs,
)


def ctx(step=10, episode="e1"):
    return ToolExecutionContext(episode, step, False, ("inspect_region",))


def setup_store(**kwargs):
    store = VisualEvidenceStore(**kwargs)
    store.reset("e1")
    rgb = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    metadata = store.capture("head", rgb, ctx())
    arguments = {
        "frame_id": metadata["frame_id"],
        "left": 16,
        "top": 8,
        "right": 64,
        "bottom": 48,
    }
    return store, rgb, metadata, arguments


def test_native_pixels_global_context_and_json_metadata_are_preserved():
    store, rgb, metadata, arguments = setup_store()
    receipt = store.inspect_region(arguments, ctx())
    frames = store.resolve_images(receipt["inspection_id"], ctx())
    np.testing.assert_array_equal(frames["global"], rgb)
    np.testing.assert_array_equal(frames["region"], rgb[8:48, 16:64])
    assert receipt["source"] == metadata
    assert receipt["source"]["episode_id"] == "e1"
    assert receipt["authority"] == "observation_only"
    assert receipt["relationship"] == "crop_of_same_frame_not_an_independent_view"
    assert receipt["region_shape"] == [40, 48, 3]
    assert not ({"success", "confidence", "actions", "object_pose"} & set(receipt))
    json.dumps(receipt, allow_nan=False)


def test_caller_and_model_cannot_mutate_stored_sensor_image():
    store, rgb, metadata, arguments = setup_store()
    original = rgb.copy()
    rgb[:] = 0
    receipt = store.inspect_region(arguments, ctx())
    returned = store.resolve_images(receipt["inspection_id"], ctx())
    np.testing.assert_array_equal(returned["global"], original)
    returned["global"][:] = 1
    returned["region"][:] = 2
    again = store.resolve_images(receipt["inspection_id"], ctx())
    np.testing.assert_array_equal(again["global"], original)
    assert receipt["source"]["rgb_sha256"] == metadata["rgb_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"left": -1},
        {"right": 1000},
        {"bottom": 65},
        {"top": 49},
        {"right": 16},
        {"right": 17},
        {"left": True},
        {"top": 0.5},
        {"frame_id": "/etc/passwd"},
        {"reward": 1},
        {"actions": [0]},
        {"frame_id": "not-captured"},
    ],
)
def test_invalid_regions_or_fields_rejected(change):
    store, _, _, arguments = setup_store()
    with pytest.raises((ValueError, TypeError)):
        store.inspect_region({**arguments, **change}, ctx())


def test_freshness_is_rechecked_when_payload_is_delivered():
    store, _, _, arguments = setup_store(max_age_steps=2)
    receipt = store.inspect_region(arguments, ctx(12))
    with pytest.raises(ValueError, match="expired"):
        store.resolve_images(receipt["inspection_id"], ctx(13))
    with pytest.raises(ValueError, match="backwards"):
        store.resolve_images(receipt["inspection_id"], ctx(11))


def test_reset_rejects_handles_even_with_reused_episode_identifier():
    store, rgb, _, arguments = setup_store()
    receipt = store.inspect_region(arguments, ctx())
    with pytest.raises(ValueError, match="another episode"):
        store.resolve_images(receipt["inspection_id"], ctx(10, "e2"))
    store.reset("e1")
    store.capture("head", rgb, ctx())
    with pytest.raises(ValueError, match="unknown"):
        store.resolve_images(receipt["inspection_id"], ctx())
    with pytest.raises(ValueError, match="unknown"):
        store.inspect_region(arguments, ctx())


@pytest.mark.parametrize("alias", [True, 1.0])
def test_episode_identity_does_not_accept_numeric_aliases(alias):
    store = VisualEvidenceStore()
    store.reset(1)
    with pytest.raises(ValueError, match="another episode"):
        store.capture("head", np.zeros((16, 16, 3), dtype=np.uint8), ctx(0, alias))


def test_eviction_removes_region_handles_without_retaining_full_images():
    store, rgb, _, arguments = setup_store(max_frames=1)
    receipt = store.inspect_region(arguments, ctx())
    store.capture("head", rgb, ctx(11))
    with pytest.raises(ValueError, match="evicted"):
        store.resolve_images(receipt["inspection_id"], ctx(11))


def test_inspection_budget_survives_frame_eviction():
    store, rgb, _, arguments = setup_store(max_frames=1, max_inspections=1)
    store.inspect_region(arguments, ctx())
    new = store.capture("head", rgb, ctx(11))
    arguments["frame_id"] = new["frame_id"]
    with pytest.raises(ValueError, match="budget"):
        store.inspect_region(arguments, ctx(11))


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((16, 16, 3), dtype=np.float32),
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((15, 16, 3), dtype=np.uint8),
        np.zeros((1025, 1024, 3), dtype=np.uint8),
        np.zeros((16, 16), dtype=np.uint8),
        None,
    ],
)
def test_sensor_input_is_bounded_and_rgb_only(rgb):
    store = VisualEvidenceStore()
    store.reset("e1")
    with pytest.raises(ValueError):
        store.capture("head", rgb, ctx())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frames": 0},
        {"max_frames": True},
        {"max_age_steps": -1},
        {"max_age_steps": 0.5},
        {"max_inspections": 0},
        {"max_frame_pixels": 255},
    ],
)
def test_invalid_store_configuration(kwargs):
    with pytest.raises(ValueError):
        VisualEvidenceStore(**kwargs)


def test_tool_permission_and_capture_timestamp_are_independent():
    store, _, _, arguments = setup_store()
    registry = EmbodiedToolRegistry()
    registry.register(store.tool_spec, store.inspect_region)
    call = ToolCall("i1", "inspect_region", arguments, "e1", 10)
    result = registry.execute(call, dataclasses.replace(ctx(), allowed_tools=()))
    assert not result.accepted and "not allowed" in result.error
    result = registry.execute(call, ctx())
    assert result.accepted
    assert result.output["source"]["timestep"] == 10


def test_optional_runtime_tool_is_logged_and_default_vocabulary_unchanged(tmp_path):
    store, _, _, arguments = setup_store()
    bindings = EmbodiedToolBindings(*([lambda *args: {}] * 6))
    workspace = RunWorkspace(
        tmp_path, RunManifest("r1", "test", "test", 0, "none", "none", "test")
    )
    default = CanonicalToolRuntime(bindings=bindings, workspace=workspace)
    assert "inspect_region" not in default.registry.tool_names
    assert len(core_tool_specs()) == 7
    runtime = CanonicalToolRuntime(
        bindings=bindings, workspace=workspace, visual_evidence=store
    )
    result = runtime.invoke("inspect_region", arguments, context=ctx())
    assert result.accepted
    logged = [
        json.loads(line) for line in workspace.event_path.read_text().splitlines()
    ]
    assert logged[-1]["payload"]["output"] == result.output
    assert "global" not in result.output
    frames = store.resolve_images(result.output["inspection_id"], ctx())
    assert set(frames) == {"global", "region"}


@pytest.mark.parametrize("budget", [0, 1, 100])
def test_runtime_inspection_budget_cannot_exceed_store_limit(tmp_path, budget):
    store, _, _, arguments = setup_store(max_inspections=2)
    workspace = RunWorkspace(
        tmp_path, RunManifest("r1", "test", "test", 0, "none", "none", "test")
    )
    runtime = CanonicalToolRuntime(
        bindings=EmbodiedToolBindings(*([lambda *args: {}] * 6)),
        workspace=workspace,
        visual_evidence=store,
        tool_call_budgets={"inspect_region": budget},
    )
    accepted = [
        runtime.invoke("inspect_region", arguments, context=ctx()).accepted
        for _ in range(3)
    ]
    assert sum(accepted) == min(budget, 2)
