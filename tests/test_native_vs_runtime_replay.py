"""Contracts for the native-vs-wrapped payload replay.

The replay is what turns "does the CarveRuntime wrapper change VLA output" from
an open question into a measured one, so its two load-bearing helpers need to be
pinned: the payload must round-trip exactly, and the action comparison must
compare the rows that are actually executed rather than reporting a false
mismatch when the wrapper truncates a chunk.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "replay_native_vs_runtime_payloads",
    REPO_ROOT / "scripts/replay_native_vs_runtime_payloads.py",
)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)


def _write_capture(tmp_path: Path, *, corrupt: bool = False) -> dict:
    from agentic_vla.runtime.adapters.starvla import hash_starvla_request_input

    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8) for _ in range(3)]
    state = np.arange(14, dtype=np.float32)[None, :]
    actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
    payload = {
        "examples": [{"image": images, "lang": "stack the bowls", "state": state}],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "unnorm_key": "arx_x5",
        "action_seed": 101,
    }
    np.savez_compressed(
        tmp_path / "payload_0000.npz",
        image_0=images[0],
        image_1=images[1],
        image_2=images[2],
        state=state,
        actions=actions,
    )
    digest = hash_starvla_request_input(payload)
    entry = {
        "index": 0,
        "arrays": "payload_0000.npz",
        "image_views": 3,
        "has_state": True,
        "lang": "stack the bowls",
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "num_inference_steps": None,
        "unnorm_key": "arx_x5",
        "action_seed": 101,
        "input_sha256": "0" * 64 if corrupt else digest,
        "action_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest(),
        "live_inference_steps": 4,
        "live_inference_steps_source": "checkpoint_default",
        "live_model_latency_ms": 321.0,
        "episode_id": 0,
        "reset_generation": 1,
        "timestep": 0,
    }
    (tmp_path / "payload_manifest.jsonl").write_text(
        json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8"
    )
    return entry


def test_rebuilt_payload_must_reproduce_the_recorded_digest(tmp_path) -> None:
    entry = _write_capture(tmp_path)

    payload, actions = replay.rebuild_payload(tmp_path, entry)

    from agentic_vla.runtime.adapters.starvla import hash_starvla_request_input

    assert hash_starvla_request_input(payload) == entry["input_sha256"]
    assert actions.shape == (50, 14)
    # Request-level controls the server acts on must survive the round trip.
    assert payload["action_seed"] == 101
    assert payload["num_ddim_steps"] == 10
    assert payload["unnorm_key"] == "arx_x5"


def test_replay_refuses_a_payload_that_does_not_round_trip(tmp_path) -> None:
    """A silent mismatch would report a comparison on the wrong observation."""

    entry = _write_capture(tmp_path, corrupt=True)

    with pytest.raises(ValueError, match="did not round-trip"):
        replay.rebuild_payload(tmp_path, entry)


def test_comparison_uses_the_executed_prefix_when_the_wrapper_truncates() -> None:
    """Truncation to max_actions is a declared control, not an output change."""

    native = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
    wrapped = native[:16].copy()

    result = replay.compare_arrays(native, wrapped)

    assert result["identical"] is True
    assert result["prefix_sha256_equal"] is True
    assert result["rows_compared"] == 16
    assert result["wrapper_truncated_chunk"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["shape_native"] == [50, 14]
    assert result["shape_wrapped"] == [16, 14]


def test_comparison_reports_a_real_numeric_difference() -> None:
    native = np.zeros((50, 14), dtype=np.float32)
    wrapped = np.zeros((16, 14), dtype=np.float32)
    wrapped[3, 5] = 0.25

    result = replay.compare_arrays(native, wrapped)

    assert result["identical"] is False
    assert result["rows_compared"] == 16
    assert result["max_abs_diff"] == pytest.approx(0.25)
    assert result["mean_abs_diff"] > 0.0


def test_comparison_flags_an_action_dimension_mismatch() -> None:
    """A different action dim is a contract break, not a truncation."""

    result = replay.compare_arrays(
        np.zeros((50, 14), dtype=np.float32), np.zeros((16, 7), dtype=np.float32)
    )

    assert result["action_dim_mismatch"] is True
    assert result["identical"] is False


def test_manifest_loader_rejects_a_missing_or_empty_capture(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        replay.load_manifest(tmp_path)

    (tmp_path / "payload_manifest.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty payload manifest"):
        replay.load_manifest(tmp_path)


def test_in_process_client_presents_the_websocket_contract() -> None:
    """The adapter needs {"ok", "data": {"actions"}}, and examples/unnorm_key
    must be passed as named arguments the way the real server passes them."""

    seen: dict = {}

    class _Wrapper:
        def predict_action(self, *, examples, unnorm_key=None, **kwargs):
            seen["examples"] = examples
            seen["unnorm_key"] = unnorm_key
            seen["kwargs"] = kwargs
            return {"actions": np.zeros((1, 16, 14), dtype=np.float32)}

    client = replay.InProcessStarVlaClient(_Wrapper())
    response = client.predict_action(
        {
            "examples": [{"lang": "x"}],
            "unnorm_key": "arx_x5",
            "num_ddim_steps": 10,
            "action_seed": 101,
        }
    )

    assert response["ok"] is True
    assert response["data"]["actions"].shape == (1, 16, 14)
    assert seen["unnorm_key"] == "arx_x5"
    # Legacy keys are forwarded verbatim, exactly as the real server forwards
    # them, so the replay stays faithful to what the arms send.
    assert seen["kwargs"] == {"num_ddim_steps": 10, "action_seed": 101}
    assert client.calls == 1
