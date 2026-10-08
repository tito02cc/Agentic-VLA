from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from agentic_vla.runtime import (
    ExecutionRiskMonitor,
    GuardedHighLevelAgent,
    InferenceControls,
    MonitorConfig,
    RiskAssessment,
    ProceduralStep,
)
from agentic_vla.toolchain import (
    EmbodiedTaskPlan,
    GuardedScalarVisualVerifier,
    GuardedVisualVerifier,
    ScalarVisualPredicate,
)


ROBODOJO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "robodojo_official"
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from XPolicyLab.policy.starVLA.model import Model  # noqa: E402


def test_semantic_image_profile_rejects_invalid_config_before_model_loading():
    with pytest.raises(ValueError, match="carve_semantic_image_profile"):
        Model({"carve_semantic_image_profile": "unknown"})


def test_reobservation_shadow_leaves_vla_inputs_and_chunk_untouched(tmp_path):
    from types import SimpleNamespace

    model = Model.__new__(Model)
    model.carve_mode = "shadow"
    model._reset_generation = 1
    model._reobservation_by_env = {}
    model._reobservation_config = {"checkpoints": [0], "model": "/test/vlm", "max_calls": 1,
        "min_interval_steps": 16, "reply_deadline_s": 12, "label_to_role": {"board": "target"},
        "user_prompt": "Locate board", "system_prompt": "Observe only", "output_dir": str(tmp_path)}
    model.step_by_env = {0: 0}
    observation = {"lang": "original task", "state": np.zeros((1, 14), dtype=np.float32),
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)],
        "_semantic_head": np.zeros((480, 640, 3), dtype=np.uint8)}
    model.obs_by_env = {0: observation}
    chunk = np.ones((16, 14), dtype=np.float32)
    model.action_chunks_by_env = {0: chunk}
    calls = []
    def transport(request):
        calls.append(request)
        return {"ok": True, "data": {"text": "[]", "backend": "cpu_staged_vlm",
            "action_model_restored": True, "model_path": "/test/vlm"}}
    model.client = SimpleNamespace(semantic_decision=transport)
    assert model._apply_reobservation_shadow(0) is None
    assert len(calls) == 1 and model.obs_by_env[0] is observation
    assert model.action_chunks_by_env[0] is chunk and np.all(chunk == 1)
    assert model.step_by_env[0] == 0 and observation["lang"] == "original task"
    receipt = json.loads((tmp_path / "g1_env0/action_isolation.jsonl").read_text())
    assert receipt["before"] == receipt["after"] and receipt["unchanged"]
    model.carve_mode = "agentic"
    with pytest.raises(RuntimeError, match="active control"):
        model._apply_reobservation_shadow(0)


@pytest.mark.parametrize("mode", [None, "baseline", "runtime", "agentic", "full"])
def test_multiview_native_not_admitted_for_active_control(mode):
    with pytest.raises(ValueError, match="observation-only"):
        Model({"carve_semantic_image_profile": "multiview_native", "carve_mode": mode})


def test_multiview_native_observation_capture_preserves_vla_and_resets(monkeypatch):
    from XPolicyLab.policy.starVLA import model as bridge

    monkeypatch.setattr(bridge, "pack_robot_state", lambda *a, **kw: np.arange(14, dtype=np.float32))
    model = Model.__new__(Model)
    model.model_cfg, model.robot_action_dim_info = {}, {}
    model.image_size, model.include_state, model.action_type, model.action_dim = (224, 224), True, "joint", 14
    cameras = ("head", "left_wrist", "right_wrist")
    images = [np.full((480, 640, 3), v, dtype=np.uint8) for v in (17, 63, 211)]
    raw = {"instruction": "original task", "vision": {
        "cam_" + name: {"rgb": image} for name, image in zip(cameras, images, strict=True)
    }}
    model._semantic_image_profile = "vla_224"
    old = model._convert_obs(raw)
    model._semantic_image_profile = "multiview_native"
    new = model._convert_obs(raw)
    np.testing.assert_array_equal(old["state"], new["state"])
    assert old["lang"] == new["lang"]
    for a, b in zip(old["image"], new["image"], strict=True):
        np.testing.assert_array_equal(a, b)
    model.obs_by_env = {0: new}
    frames = model._semantic_frames(0, current=True)
    assert list(frames) == ["current_" + name for name in cameras]
    for name, image in zip(cameras, images, strict=True):
        frame = frames["current_" + name]
        np.testing.assert_array_equal(frame, image)
        assert not frame.flags.writeable and not np.shares_memory(frame, image)
    assert list(model._semantic_frames(0, head_only=True)) == ["head"]
    for image in images:
        image[:] = 0
    model.obs_by_env[0] = model._convert_obs(raw)
    assert all(np.all(frame == 0) for frame in model._semantic_frames(0).values())
    assert frames["current_right_wrist"][0, 0, 0] == 211
    model.obs_by_env.clear()
    with pytest.raises(KeyError):
        model._semantic_frames(0)
    model.obs_by_env[0] = new
    del new["_semantic_views"]["left_wrist"]
    with pytest.raises(KeyError, match="left_wrist"):
        model._semantic_frames(0)


def test_semantic_camera_preserves_vla_inputs_and_rgb(monkeypatch):
    from XPolicyLab.policy.starVLA import model as bridge

    state = np.arange(14, dtype=np.float32)
    monkeypatch.setattr(bridge, "pack_robot_state", lambda *args, **kwargs: state.copy())
    model = Model.__new__(Model)
    model.model_cfg = {}
    model.image_size = (224, 224)
    model.include_state = True
    model.action_type = "joint"
    model.robot_action_dim_info = {}
    model.action_dim = 14
    for shape, expected in [((480, 640), (480, 640)), ((900, 1600), (360, 640)), ((60, 80), (60, 80))]:
        rgb = np.zeros((*shape, 3), dtype=np.uint8)
        rgb[:, :, 0] = 211
        rgb[:, :, 2] = 17
        raw = {"instruction": "original task", "vision": {
            camera: {"rgb": rgb} for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        }}
        model._semantic_image_profile = "vla_224"
        original = model._convert_obs(raw)
        assert set(original) == {"lang", "image", "state"}
        model._semantic_image_profile = "head_native"
        revised = model._convert_obs(raw)
        assert revised["lang"] == original["lang"]
        np.testing.assert_array_equal(revised["state"], original["state"])
        for before, after in zip(original["image"], revised["image"], strict=True):
            np.testing.assert_array_equal(before, after)
            assert after.shape == (224, 224, 3)
        head = revised["_semantic_head"]
        assert head.shape == (*expected, 3)
        np.testing.assert_array_equal(head[0, 0], [211, 0, 17])
        assert not head.flags.writeable
        assert not np.shares_memory(head, rgb)
        model.obs_by_env = {0: revised}
        assert model._semantic_frames(0)["head"] is head
        assert model._semantic_frames(0, current=True, head_only=True)["current_head"] is head
        rgb[:] = 0
        model.obs_by_env[0] = model._convert_obs(raw)
        assert np.all(model._semantic_frames(0)["head"] == 0)
        assert head[0, 0, 0] == 211


def test_semantic_frame_default_and_missing_native_are_explicit():
    model = Model.__new__(Model)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    model.obs_by_env = {0: {"image": frames}}
    assert list(model._semantic_frames(0)) == ["head", "left_wrist", "right_wrist"]
    assert list(model._semantic_frames(0, current=True, head_only=True)) == ["current_head"]
    model._semantic_image_profile = "head_native"
    with pytest.raises(KeyError, match="_semantic_head"):
        model._semantic_frames(0)


def test_semantic_frame_never_reaches_vla_transport_or_runtime():
    for use_runtime in (False, True):
        model = Model.__new__(Model)
        images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
        state = np.zeros((1, 14), dtype=np.float32)
        model.obs_by_env = {0: {"lang": "original", "image": images, "state": state,
                                "_semantic_head": np.zeros((480, 640, 3), dtype=np.uint8),
                                "_semantic_views": {"head": np.zeros((480, 640, 3), dtype=np.uint8)}}}
        model.use_ddim = True
        model.num_ddim_steps = 4
        model.unnorm_key = None
        model.action_seed_base = 17
        model.step_by_env = {0: 160}
        model.carve_mode = "full"
        model.carve_execute_horizon = 16
        model._active_execute_horizon_by_env = {}
        model._recovery_compute_remaining_by_env = {}
        model._semantic_guidance_remaining_by_env = {}
        captured = []

        def infer(request):
            assert set(request.observation) == {"lang", "image", "state"}
            assert request.observation["image"] is images
            assert request.observation["state"] is state
            captured.append(request.metadata["raw_payload"])
            return SimpleNamespace(actions=np.zeros((16, 14), dtype=np.float32))

        def predict(payload):
            captured.append(payload)
            return {"ok": True, "data": {"actions": np.zeros((1, 50, 14), dtype=np.float32)}}

        model.client = SimpleNamespace(predict_action=predict)
        model._carve_runtime = SimpleNamespace(infer=infer) if use_runtime else None
        model._carve_prepared = SimpleNamespace(apply_profile=lambda _: InferenceControls(
            inference_steps=2, max_actions=16, precision="bf16"))
        model._infer_chunk(0)
        example = captured[0]["examples"][0]
        assert set(example) == {"lang", "image", "state"}
        assert example["image"] is images and example["state"] is state
        assert example["lang"] == "original"
        assert captured[0]["action_seed"] == 177
        assert "_semantic_head" in model.obs_by_env[0]
        assert "_semantic_views" in model.obs_by_env[0]


def _model_without_server(*, mode: str, chunks: list[np.ndarray]) -> Model:
    model = Model.__new__(Model)
    model.carve_mode = mode
    model.execute_horizon = 2
    model.carve_execute_horizon = 3
    model.action_dim = 14
    model.step_by_env = {}
    model.action_chunks_by_env = {}
    model.chunk_start_by_env = {}
    model._last_action_by_env = {}
    model._active_execute_horizon_by_env = {}
    model._vla_instruction_capability = "open_vocab_subgoal"
    model._semantic_guidance_remaining_by_env = {}
    model._semantic_guidance_instruction_by_env = {}
    pending = list(chunks)

    def infer_chunk(self, env_idx: int) -> np.ndarray:
        del self, env_idx
        return pending.pop(0)

    model._infer_chunk = MethodType(infer_chunk, model)
    return model


def _chunk(start: float, length: int) -> np.ndarray:
    return np.stack(
        [np.full(14, start + offset, dtype=np.float32) for offset in range(length)]
    )


def test_real_adapter_verifies_only_after_recovery_actions_and_new_observation(tmp_path):
    from agentic_vla.toolchain.recovery_verification import RecoveryVerificationLifecycle

    model = _model_without_server(mode="full", chunks=[])
    model._recovery_verify_config = {"max_checks": 1, "deadline_s": 12}
    model._recovery_verification_by_env = {0: RecoveryVerificationLifecycle(max_checks=1)}
    gate = model._recovery_verification_by_env[0]
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}
    model._agent_memory_by_env = {}
    model._task_instruction_by_env = {0: "Stack bowls"}
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    model.obs_by_env = {0: {"lang": "Stack bowls", "image": [image]}}
    model._planner_trace_path = tmp_path / "recovery.jsonl"
    model._record_shadow_risk = lambda *args: None
    callbacks, visual_steps = [], []
    model._apply_task_plan_checkpoint = lambda env: callbacks.append("plan")
    model._apply_scalar_semantic_checkpoint = lambda env: callbacks.append("scalar")
    def inspect(request):
        visual_steps.append(model.step_by_env[0])
        return {"status": "inconclusive", "confidence": 0.9, "observed_outcome": "Occluded bowls"}
    model._post_recovery_verifier = GuardedVisualVerifier(inspect)
    def infer(env):
        remaining = model._recovery_compute_remaining_by_env.get(env, 0)
        model._recovery_compute_remaining_by_env[env] = max(0, remaining - 1)
        return _chunk(0, 3)
    model._infer_chunk = infer
    gate.observe(0)
    model._arm_recovery_compute(0, expected_outcome="All bowls are stacked")
    for step in range(7):
        gate.observe(step)
        model._next_action_vector(0)
        assert visual_steps == ([] if step < 6 else [6])
    assert callbacks == []  # Pending, then exhausted, prevents another recovery.
    assert gate.checks == 1 and gate.pending is None
    assert model.obs_by_env[0]["lang"] == "Stack bowls"
    rows = [json.loads(line) for line in model._planner_trace_path.read_text().splitlines()]
    receipt = rows[-1]
    assert receipt["event"] == "post_recovery_verification"
    assert receipt["attempt"]["last_chunk_end"] == 6
    assert receipt["attempt"]["chunks_issued"] == 2
    assert receipt["status"] == "inconclusive"
    assert "visual_report_only_not_task_success" in model._agent_memory_by_env[0][-1]


def test_agentic_recovery_can_extend_action_commitment() -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}

    model._arm_recovery_compute(0)

    assert model._recovery_compute_remaining_by_env[0] == 2


def test_request_level_action_seed_tracks_episode_and_chunk_timestep() -> None:
    model = Model.__new__(Model)
    model.action_seed_base = 17
    model.action_dim = 14
    model.num_ddim_steps = 10
    model.use_ddim = True
    model.unnorm_key = "arx_x5"
    model.obs_by_env = {
        2: {
            "lang": "stack bowls",
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
            "state": np.zeros((1, 14), dtype=np.float32),
        }
    }
    model.step_by_env = {2: 32}
    model._carve_runtime = None
    model._carve_prepared = None
    model.carve_execute_horizon = 16
    model._active_execute_horizon_by_env = {}
    model._semantic_guidance_remaining_by_env = {}
    observed = {}

    class _Client:
        def predict_action(self, payload):
            observed.update(payload)
            return {
                "ok": True,
                "data": {"actions": np.zeros((1, 50, 14), dtype=np.float32)},
            }

    model.client = _Client()

    model._infer_chunk(2)

    assert observed["action_seed"] == 17 + 2 * 1_000_003 + 32


def test_scalar_semantic_stall_requires_repeated_observation(tmp_path: Path) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": "stack all white bowls",
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: "stack all white bowls"}
    model._planner_trace_path = tmp_path / "scalar.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 3,
            "confidence": 0.9,
            "evidence": "Three separate bowl groups are visible.",
        }
    )
    model._semantic_checkpoint_steps = 160
    model._semantic_stale_checks_required = 2
    model._semantic_max_recoveries = 1
    model._semantic_last_check_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}

    model.step_by_env = {0: 160}
    first = model._apply_scalar_semantic_checkpoint(0)
    assert not first
    assert 0 in model.action_chunks_by_env

    model.step_by_env = {0: 320}
    second = model._apply_scalar_semantic_checkpoint(0)
    assert second
    assert 0 not in model.action_chunks_by_env
    assert model._recovery_compute_remaining_by_env[0] == 2
    assert model._semantic_recoveries_by_env[0] == 1


def test_scalar_semantic_confirmation_never_forces_recovery(tmp_path: Path) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": "stack all white bowls",
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: "stack all white bowls"}
    model._planner_trace_path = tmp_path / "scalar-confirmed.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 1,
            "confidence": 0.9,
            "evidence": "One nested bowl group is visible.",
        }
    )
    model._semantic_checkpoint_steps = 160
    model._semantic_stale_checks_required = 2
    model._semantic_max_recoveries = 1
    model._semantic_last_check_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}
    model.step_by_env = {0: 160}

    intervened = model._apply_scalar_semantic_checkpoint(0)

    assert not intervened
    assert 0 in model.action_chunks_by_env
    assert model._semantic_recoveries_by_env == {}


def test_progress_only_semantic_monitor_can_detect_stale_confirmed_value(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": "stack all bowls",
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: "stack all bowls"}
    model._planner_trace_path = tmp_path / "progress-only.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="visible_structure_groups",
        question="How many visible bowl structures remain?",
        comparison="eq",
        target=1,
    )
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "visible_structure_groups",
            "visible": True,
            "value": 1,
            "confidence": 0.9,
            "evidence": "One visible bowl structure remains unchanged.",
        }
    )
    model._semantic_progress_only = True
    model._semantic_recovery_mode = "task_preserving"
    model._semantic_checkpoint_steps = 128
    model._semantic_stale_checks_required = 2
    model._semantic_max_calls_per_episode = 4
    model._semantic_max_recoveries = 1
    model._semantic_last_check_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model.carve_recovery_compute_calls = 1
    model._recovery_compute_remaining_by_env = {}

    model.step_by_env = {0: 128}
    assert not model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env = {0: 256}
    assert model._apply_scalar_semantic_checkpoint(0)

    assert model._semantic_recoveries_by_env[0] == 1
    assert 0 not in model.action_chunks_by_env
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"progress_only": true' in row


def test_semantic_deadline_recovers_incomplete_task_despite_recent_progress(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    task = "stack the three bowls together"
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: task}
    model._planner_trace_path = tmp_path / "deadline.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    values = iter((3, 2))
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": next(values),
            "confidence": 0.9,
            "evidence": "The task remains incomplete.",
        }
    )
    model._semantic_progress_only = True
    model._semantic_recovery_mode = "task_preserving"
    model._semantic_checkpoint_steps = 128
    model._semantic_deadline_step = 256
    model._semantic_stale_checks_required = 2
    model._semantic_max_calls_per_episode = 3
    model._semantic_max_recoveries = 1
    model._semantic_last_check_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model.carve_recovery_compute_calls = 0
    model._recovery_compute_remaining_by_env = {}

    model.step_by_env = {0: 128}
    assert not model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env = {0: 256}
    assert model._apply_scalar_semantic_checkpoint(0)

    assert model._semantic_recoveries_by_env[0] == 1
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"deadline_incomplete": true' in row
    assert "semantic_deadline_incomplete" in model._agent_memory_by_env[0][-1]


def test_semantic_stall_routes_through_guarded_planner_subgoal(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    task = "stack all black bowls"
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: task}
    model._planner_trace_path = tmp_path / "semantic-planner.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate black bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": 3,
            "confidence": 0.9,
            "evidence": "Three separate black bowl groups are visible.",
        }
    )
    model._semantic_recovery_mode = "planner"
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: {
            "intent": "vla_act",
            "rationale": "three bowl groups remain",
            "confidence": 0.9,
            "subgoal": "stack one black bowl group",
            "vla_instruction": "stack one remaining black bowl with the center black bowl",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "two separate black bowl groups remain",
            "memory_note": "previous command made no visible progress",
            "failure_type": "no_progress",
            "scene_graph_update": {},
            "proposed_plan": [],
        },
        # Match the online recovery behavior: reject to deterministic fallback.
        config=SimpleNamespace(
            max_calls_per_episode=3,
            max_grounding_repairs=1,
            minimum_intervention_confidence=0.55,
            fail_closed=False,
        ),
    )
    model._semantic_checkpoint_steps = 160
    model._semantic_stale_checks_required = 2
    model._semantic_max_recoveries = 1
    model._semantic_last_check_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model._task_plan_by_env = {}
    model._last_planner_step_by_env = {}
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}

    model.step_by_env = {0: 160}
    assert not model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env = {0: 320}
    assert model._apply_scalar_semantic_checkpoint(0)

    assert "Current recovery subgoal" in model.obs_by_env[0]["lang"]
    assert "stack one remaining black bowl" in model.obs_by_env[0]["lang"]
    rows = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"action": "semantic_subgoal_vla_replan"' in rows
    assert '"schema_version": "carve.robodojo.semantic-planner.v1"' in rows


def test_semantic_planner_rejection_falls_back_without_mutating_task(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="agentic", chunks=[])
    task = "stack all black bowls"
    model.model_cfg = {"carve_planner_max_calls_per_episode": 1}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model._task_instruction_by_env = {0: task}
    model._planner_trace_path = tmp_path / "semantic-fallback.jsonl"
    model._semantic_recovery_mode = "planner"
    model._semantic_max_recoveries = 1
    model._semantic_stale_checks_required = 2
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate black bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    model._agent_memory_by_env = {}
    model._task_plan_by_env = {}
    model._last_planner_step_by_env = {}
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model.step_by_env = {0: 320}
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: {"actions": [[0.0] * 14]},
        config=SimpleNamespace(
            max_calls_per_episode=1,
            max_grounding_repairs=0,
            minimum_intervention_confidence=0.55,
            fail_closed=False,
        ),
    )
    report = SimpleNamespace(
        metadata={"groups": ["left bowl", "center bowl", "right bowl"]},
        observed_outcome="observed 3 groups",
    )

    action = model._apply_semantic_recovery(
        0,
        trigger="periodic_semantic_checkpoint",
        report=report,
        observed_value=3.0,
        recoveries=0,
    )

    assert action == "semantic_planner_fallback_replan"
    assert model.obs_by_env[0]["lang"] == task
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"accepted": false' in row


def test_agentic_uses_active_recovery_horizon() -> None:
    model = _model_without_server(
        mode="agentic",
        chunks=[_chunk(0, 5), _chunk(10, 5)],
    )
    model.carve_execute_horizon = 2
    model._active_execute_horizon_by_env = {0: 4}
    model.obs_by_env = {
        0: {"state": np.zeros((1, 14), dtype=np.float32), "image": []}
    }
    model._monitor_trace_path = None

    observed = [float(model._next_action_vector(0)[0]) for _ in range(5)]

    assert observed == [0.0, 1.0, 2.0, 3.0, 10.0]


def test_agentic_only_mode_bootstraps_carve_import_path(tmp_path: Path) -> None:
    model = Model.__new__(Model)
    model.model_cfg = {"carve_root": str(tmp_path)}
    original_path = list(sys.path)
    try:
        resolved = model._ensure_carve_import_path()
        assert resolved == tmp_path.resolve()
        assert sys.path[0] == str(tmp_path.resolve())
    finally:
        sys.path[:] = original_path


def test_agentic_mode_accepts_resident_starvla_backbone(tmp_path: Path) -> None:
    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_root": str(Path(__file__).resolve().parents[1]),
        "carve_planner_backend": "shared_backbone",
        "carve_planner_trace_path": str(tmp_path / "planner.jsonl"),
        "carve_planner_max_tokens": 64,
        "carve_planner_max_images": 1,
    }
    model._planner_cooldown_steps = 128
    model._semantic_role = "planner"
    model._visual_critic = None
    model._high_level_agent = None
    model._starvla_server_meta = {"shared_semantic_generation": True}
    model.client = SimpleNamespace(
        semantic_decision=lambda _payload: {
            "ok": True,
            "data": {"text": '{"intent":"continue"}'},
        }
    )

    model._initialize_agentic_planner()

    assert model._high_level_agent is not None
    assert model._visual_critic is None


def test_staged_vlm_backend_checks_identity_and_records_restoration(tmp_path):
    import pytest

    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_root": str(Path(__file__).resolve().parents[1]),
        "carve_planner_backend": "staged_vlm",
        "carve_planner_model": "/local/qualified-vlm",
        "carve_planner_trace_path": str(tmp_path / "planner.jsonl"),
    }
    model._planner_cooldown_steps = 128
    model._semantic_role = "planner"
    model._visual_critic = None
    model._starvla_server_meta = {"staged_semantic_generation": False}
    with pytest.raises(ValueError, match="no staged VLM"):
        model._initialize_agentic_planner()
    model._starvla_server_meta = {
        "staged_semantic_generation": True, "staged_vlm_model": "/wrong/model",
    }
    with pytest.raises(ValueError, match="explicit Planner model"):
        model._initialize_agentic_planner()
    model._starvla_server_meta["staged_vlm_model"] = "/local/qualified-vlm"
    model.client = SimpleNamespace(semantic_decision=lambda payload: {
        "ok": True, "data": {
            "text": '{"intent":"continue"}', "backend": "cpu_staged_vlm",
            "action_model_restored": True, "model_path": "/local/qualified-vlm",
            "timings": {"total_ms": 1.0},
        },
    })
    model._initialize_agentic_planner()
    assert model._high_level_agent is not None
    result = model._high_level_agent._infer({
        "system_prompt": "Return JSON", "user_prompt": "inspect", "frames": {},
    })
    assert result == '{"intent":"continue"}'
    receipt = json.loads((tmp_path / "planner.jsonl").read_text())
    assert receipt["action_model_restored"] is True
    assert receipt["source"] == "cpu_staged_vlm"


def test_replay_task_plan_can_verify_with_resident_starvla_backbone(
    tmp_path: Path,
) -> None:
    ticket_path = tmp_path / "task-plan.json"
    ticket_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "source": "offline_task_plan",
                "decision": {
                    "intent": "vla_act",
                    "subgoal": "stack two bowls",
                },
                "task_plan": {
                    "steps": [
                        {
                            "stage": "partial_stack",
                            "intent": "vla_act",
                            "subgoal": "stack the left and center bowls together",
                            "expected_outcome": "two bowls form one stack",
                            "skill_id": None,
                            "constraints": [],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_root": str(Path(__file__).resolve().parents[1]),
        "carve_planner_backend": "task_preserving",
        "carve_planner_replay_path": str(ticket_path),
        "carve_planner_trace_path": str(tmp_path / "planner.jsonl"),
        "carve_task_plan_verifier_backend": "shared_backbone",
        "carve_planner_max_tokens": 64,
        "carve_planner_max_images": 1,
    }
    model._planner_cooldown_steps = 128
    model._semantic_role = "planner"
    model._task_plan_mode = "execute"
    model._visual_critic = None
    model._task_plan_visual_critic = None
    model._task_plan_grouped_visual_critic = None
    model._high_level_agent = None
    model._starvla_server_meta = {"shared_semantic_generation": True}
    model.client = SimpleNamespace(
        semantic_decision=lambda _payload: {
            "ok": True,
            "data": {"text": '{"status":"inconclusive"}'},
        }
    )

    model._initialize_agentic_planner()

    assert model._task_plan_visual_critic is not None
    assert model._task_plan_grouped_visual_critic is not None
    assert model._replay_task_plan_steps[0].stage == "partial_stack"


def test_agentic_mode_rejects_missing_resident_semantic_capability(
    tmp_path: Path,
) -> None:
    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_root": str(Path(__file__).resolve().parents[1]),
        "carve_planner_backend": "shared_backbone",
        "carve_planner_trace_path": str(tmp_path / "planner.jsonl"),
    }
    model._planner_cooldown_steps = 128
    model._semantic_role = "planner"
    model._visual_critic = None
    model._high_level_agent = None
    model._starvla_server_meta = {"shared_semantic_generation": False}
    model.client = SimpleNamespace(semantic_decision=lambda _payload: {})

    try:
        model._initialize_agentic_planner()
    except ValueError as exc:
        assert "shared semantic generation" in str(exc)
    else:
        raise AssertionError("missing shared semantic generation must fail closed")


def test_agentic_mode_supports_task_preserving_replan_ablation(
    tmp_path: Path,
) -> None:
    model = Model.__new__(Model)
    model.model_cfg = {
        "carve_root": str(Path(__file__).resolve().parents[1]),
        "carve_planner_backend": "task_preserving",
        "carve_planner_trace_path": str(tmp_path / "planner.jsonl"),
    }
    model._planner_cooldown_steps = 128
    model._semantic_role = "planner"
    model._visual_critic = None
    model._high_level_agent = None
    model._starvla_server_meta = {}
    model.client = SimpleNamespace()

    model._initialize_agentic_planner()

    assert model._high_level_agent is not None
    assert model._visual_critic is None


def test_official_baseline_keeps_fixed_execute_horizon() -> None:
    model = _model_without_server(
        mode="baseline",
        chunks=[_chunk(0, 3), _chunk(10, 3), _chunk(20, 3)],
    )

    observed = [float(model._next_action_vector(0)[0]) for _ in range(5)]

    assert observed == [0.0, 1.0, 10.0, 11.0, 20.0]


def test_carve_runtime_replans_when_chunk_exhausts_before_horizon() -> None:
    model = _model_without_server(
        mode="runtime",
        chunks=[_chunk(0, 2), _chunk(10, 2), _chunk(20, 2)],
    )

    observed = [float(model._next_action_vector(0)[0]) for _ in range(5)]

    assert observed == [0.0, 1.0, 10.0, 11.0, 20.0]


def test_recovery_compute_overrides_fast_path_for_bounded_calls() -> None:
    model = Model.__new__(Model)
    model.obs_by_env = {0: {"lang": "build the tower", "image": []}}
    model.use_ddim = True
    model.num_ddim_steps = 6
    model.unnorm_key = None
    model.step_by_env = {0: 690}
    model.carve_mode = "full"
    model.carve_execute_horizon = 32
    model.carve_recovery_num_ddim_steps = 10
    model.carve_recovery_execute_horizon = 16
    model._recovery_compute_remaining_by_env = {0: 2}
    model._carve_prepared = SimpleNamespace(
        apply_profile=lambda _controls: InferenceControls(
            inference_steps=6,
            max_actions=32,
            precision="bf16",
        )
    )
    requests = []

    def infer(request):
        requests.append(request)
        return SimpleNamespace(actions=np.zeros((16, 14), dtype=np.float32))

    model._carve_runtime = SimpleNamespace(infer=infer)

    chunk = model._infer_chunk(0)

    assert chunk.shape == (16, 14)
    assert requests[0].controls.inference_steps == 10
    assert requests[0].controls.max_actions == 16
    assert requests[0].metadata["trace_context"]["compute_phase"] == "recovery_boost"
    assert requests[0].metadata["trace_context"]["execute_horizon"] == 16
    assert model._recovery_compute_remaining_by_env[0] == 1


def test_semantic_guidance_restores_original_task_after_bounded_vla_calls(
    tmp_path: Path,
) -> None:
    model = Model.__new__(Model)
    original = "stack the three bowls together"
    guided = (
        f"{original}. Current recovery subgoal: "
        "stack the left bowl with the center bowl."
    )
    model.obs_by_env = {0: {"lang": guided, "image": []}}
    model._task_instruction_by_env = {0: original}
    model._semantic_guidance_remaining_by_env = {0: 2}
    model._semantic_guidance_instruction_by_env = {0: guided}
    model._agent_memory_by_env = {}
    model._planner_trace_path = tmp_path / "guidance.jsonl"
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = None
    model.step_by_env = {0: 320}
    model.carve_mode = "agentic"
    model.carve_execute_horizon = 16
    model.carve_recovery_num_ddim_steps = 10
    model.carve_recovery_execute_horizon = 16
    model._recovery_compute_remaining_by_env = {}
    model._active_execute_horizon_by_env = {}
    model._carve_prepared = SimpleNamespace(
        apply_profile=lambda _controls: InferenceControls(
            inference_steps=10,
            max_actions=16,
            precision="bf16",
        )
    )
    seen_instructions = []

    def infer(request):
        seen_instructions.append(request.instruction)
        return SimpleNamespace(actions=np.zeros((16, 14), dtype=np.float32))

    model._carve_runtime = SimpleNamespace(infer=infer)

    model._infer_chunk(0)
    assert model.obs_by_env[0]["lang"] == guided
    assert model._semantic_guidance_remaining_by_env[0] == 1

    # RoboDojo refreshes the observation and restores its original task text
    # before every control step. Active guidance must survive that refresh.
    model.obs_by_env[0]["lang"] = original
    model._reapply_semantic_guidance(0)
    model._infer_chunk(0)
    assert seen_instructions == [guided, guided]
    assert model.obs_by_env[0]["lang"] == original
    assert model._semantic_guidance_remaining_by_env == {}
    assert model._semantic_guidance_instruction_by_env == {}
    assert "restore_original_task" in model._planner_trace_path.read_text(
        encoding="utf-8"
    )


def test_semantic_guidance_budget_is_not_consumed_by_failed_vla_call() -> None:
    model = Model.__new__(Model)
    model.obs_by_env = {0: {"lang": "temporary recovery", "image": []}}
    model._task_instruction_by_env = {0: "original task"}
    model._semantic_guidance_remaining_by_env = {0: 2}
    model._semantic_guidance_instruction_by_env = {0: "temporary recovery"}
    model._agent_memory_by_env = {}
    model._planner_trace_path = None
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = None
    model.step_by_env = {0: 320}
    model.carve_mode = "agentic"
    model.carve_execute_horizon = 16
    model.carve_recovery_num_ddim_steps = 10
    model.carve_recovery_execute_horizon = 16
    model._recovery_compute_remaining_by_env = {}
    model._active_execute_horizon_by_env = {}
    model._carve_prepared = SimpleNamespace(
        apply_profile=lambda _controls: InferenceControls(
            inference_steps=10,
            max_actions=16,
            precision="bf16",
        )
    )
    model._carve_runtime = SimpleNamespace(
        infer=lambda _request: (_ for _ in ()).throw(RuntimeError("transport error"))
    )

    try:
        model._infer_chunk(0)
    except RuntimeError as exc:
        assert str(exc) == "transport error"
    else:
        raise AssertionError("expected transport error")

    assert model._semantic_guidance_remaining_by_env[0] == 2
    assert model._semantic_guidance_instruction_by_env[0] == "temporary recovery"
    assert model.obs_by_env[0]["lang"] == "temporary recovery"


def test_update_obs_batch_reapplies_active_semantic_guidance() -> None:
    model = Model.__new__(Model)
    model.obs_by_env = {}
    model._latest_env_idx_list = []
    model._task_instruction_by_env = {0: "original task"}
    model._semantic_guidance_remaining_by_env = {0: 2}
    model._semantic_guidance_instruction_by_env = {0: "bounded recovery task"}
    model._convert_obs = lambda obs: dict(obs["converted"])
    model._ensure_replay_task_plan = lambda _env_idx: None

    model.update_obs_batch(
        [
            {
                "env_idx": 0,
                "converted": {
                    "lang": "original task",
                    "image": [],
                },
            }
        ]
    )

    assert model.obs_by_env[0]["lang"] == "bounded recovery task"


def test_shadow_monitor_records_risk_without_changing_actions(tmp_path: Path) -> None:
    model = _model_without_server(
        mode="shadow",
        chunks=[_chunk(0, 3), _chunk(10, 3)],
    )
    model.obs_by_env = {
        0: {
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model._monitor_trace_path = tmp_path / "monitor.jsonl"
    model._monitor_factory = lambda: ExecutionRiskMonitor(
        MonitorConfig(window_size=2, warmup_steps=0)
    )
    model._risk_monitors = {}
    model._last_action_by_env = {}

    observed = [float(model._next_action_vector(0)[0]) for _ in range(4)]

    assert observed == [0.0, 1.0, 2.0, 10.0]
    rows = model._monitor_trace_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert all('"condition": "shadow_no_intervention"' in row for row in rows)


def test_agentic_no_progress_routes_through_guarded_vlm_and_replans(tmp_path: Path) -> None:
    model = _model_without_server(mode="full", chunks=[])
    model.obs_by_env = {
        0: {
            "lang": "Build a tower using the wooden blocks and wooden boards.",
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._planner_trace_path = tmp_path / "planner.jsonl"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._last_planner_step_by_env = {}
    model._visual_critic = None
    model._semantic_role = "planner"
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: {
            "intent": "vla_act",
            "rationale": "complete remaining tower stage",
            "confidence": 0.9,
            "subgoal": "complete tower top",
            "vla_instruction": "complete the remaining wooden tower structure",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "the wooden tower top is complete",
            "memory_note": "top stage was incomplete",
            "failure_type": "no_progress",
            "scene_graph_update": {},
            "proposed_plan": [],
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_agentic_progress_check(0, risk)

    assert 0 not in model.action_chunks_by_env
    assert model.obs_by_env[0]["lang"].startswith(
        "Build a tower using the wooden blocks and wooden boards."
    )
    assert "complete the remaining wooden tower structure" in model.obs_by_env[0]["lang"]
    assert 0 not in model._safe_stop_envs
    rows = model._planner_trace_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert '"intent": "vla_act"' in rows[0]
    assert '"accepted": true' in rows[0]


def test_rejected_high_level_planner_cannot_stop_or_mutate_vla(tmp_path: Path) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    original_chunk = _chunk(0, 3)
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: original_chunk}
    model._task_instruction_by_env = {0: task}
    model._planner_trace_path = tmp_path / "planner-rejected.jsonl"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._visual_critic = None
    model._semantic_role = "planner"
    model._high_level_agent_disabled_envs = set()
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: "not valid JSON"
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_agentic_progress_check(0, risk)

    assert 0 not in model._safe_stop_envs
    assert 0 in model._high_level_agent_disabled_envs
    assert model.action_chunks_by_env[0] is original_chunk
    assert model.obs_by_env[0]["lang"] == task
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"accepted": false' in row
    assert '"control_action": "planner_unavailable_continue_original"' in row


def test_high_level_planner_safe_stop_requires_explicit_authority(tmp_path: Path) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    original_chunk = _chunk(0, 3)
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: original_chunk}
    model._task_instruction_by_env = {0: task}
    model._planner_trace_path = tmp_path / "planner-stop-no-authority.jsonl"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._visual_critic = None
    model._semantic_role = "planner"
    model._semantic_safe_stop_enabled = False
    model._high_level_agent_disabled_envs = set()
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: {
            "intent": "safe_stop",
            "rationale": "the task appears complete",
            "confidence": 0.9,
            "subgoal": "",
            "vla_instruction": None,
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "",
            "memory_note": "",
            "failure_type": "none",
            "scene_graph_update": {},
            "proposed_plan": [],
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_agentic_progress_check(0, risk)

    assert 0 not in model._safe_stop_envs
    assert model.action_chunks_by_env[0] is original_chunk
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"control_action": "planner_safe_stop_rejected_no_authority"' in row


def test_agentic_recovery_is_bound_to_active_long_horizon_stage(tmp_path: Path) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.step_by_env = {0: 693}
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._planner_trace_path = tmp_path / "planner-ledger.jsonl"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._visual_critic = None
    model._semantic_role = "planner"
    model._task_instruction_by_env = {0: task}
    model.carve_recovery_compute_calls = 0
    model._recovery_compute_remaining_by_env = {}
    plan = EmbodiedTaskPlan()
    plan.install(
        (
            ProceduralStep(
                stage="top",
                intent="vla_act",
                subgoal="complete the top tower structure",
                expected_outcome="top structure is visibly complete",
            ),
        )
    )
    model._task_plan_by_env = {0: plan}
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: {
            "intent": "vla_act",
            "rationale": "finish active tower stage",
            "confidence": 0.9,
            "subgoal": "complete the top tower structure",
            "vla_instruction": "complete the top wooden tower structure",
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "top structure is visibly complete",
            "memory_note": "top stage remains incomplete",
            "failure_type": "no_progress",
            "scene_graph_update": {},
            "proposed_plan": [],
        }
    )
    risk = RiskAssessment(
        score=0.65,
        bucket="high",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_agentic_progress_check(0, risk)

    assert plan.active is not None and plan.active.attempts == 1
    assert "complete the top wooden tower structure" in model.obs_by_env[0]["lang"]
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"active_stage": "top"' in row
    assert '"attempts": 1' in row


def test_replay_ticket_installs_confirmed_prefix_and_active_stage(tmp_path: Path) -> None:
    model = Model.__new__(Model)
    model._task_plan_by_env = {}
    model._replay_task_plan_steps = ()
    model._replay_confirmed_stages = ()
    model._replay_task_plan_source = ""
    model._planner_trace_path = tmp_path / "plan-install.jsonl"
    model.step_by_env = {0: 693}
    ticket = {
        "source": "staged_vlm",
        "task_plan": {
            "source": "verified_procedural_memory",
            "confirmed_stages": ["base", "middle"],
            "steps": [
                {
                    "stage": "base",
                    "intent": "vla_act",
                    "subgoal": "assemble lower base",
                    "expected_outcome": "lower base is complete",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "middle",
                    "intent": "vla_act",
                    "subgoal": "assemble middle level",
                    "expected_outcome": "middle level is complete",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "top",
                    "intent": "vla_act",
                    "subgoal": "complete tower top",
                    "expected_outcome": "tower top is complete",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        },
    }

    model._load_replay_task_plan(ticket, replay_path=tmp_path / "ticket.json")
    plan = model._ensure_replay_task_plan(0)

    assert plan is not None and plan.active is not None
    assert plan.active.step.stage == "top"
    assert [receipt.status.value for receipt in plan.receipts] == [
        "confirmed",
        "confirmed",
        "active",
    ]


def test_replay_ticket_accepts_validated_decision_proposed_plan(
    tmp_path: Path,
) -> None:
    model = Model.__new__(Model)
    model._task_plan_by_env = {}
    model._replay_task_plan_steps = ()
    model._replay_confirmed_stages = ()
    model._replay_task_plan_source = ""
    model._planner_trace_path = tmp_path / "plan-install.jsonl"
    model.step_by_env = {0: 0}
    ticket = {
        "accepted": True,
        "decision": {
            "proposed_plan": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the left and center bowls together",
                    "expected_outcome": "two bowls form one stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the remaining bowl with the partial stack",
                    "expected_outcome": "all bowls form one stack",
                    "skill_id": None,
                    "constraints": [],
                },
            ]
        },
    }

    model._load_replay_task_plan(ticket, replay_path=tmp_path / "ticket.json")

    assert [step.stage for step in model._replay_task_plan_steps] == [
        "partial_stack",
        "complete_stack",
    ]
    assert model._replay_confirmed_stages == ()


def test_replay_ticket_execute_installs_bounded_active_stage_guidance(
    tmp_path: Path,
) -> None:
    model = Model.__new__(Model)
    task = "stack the three bowls together"
    model._task_plan_by_env = {}
    model._replay_task_plan_steps = ()
    model._replay_confirmed_stages = ()
    model._replay_task_plan_source = ""
    model._planner_trace_path = tmp_path / "plan-execute.jsonl"
    model._task_plan_mode = "execute"
    model._vla_instruction_capability = "open_vocab_subgoal"
    model._semantic_guidance_calls = 2
    model._semantic_guidance_remaining_by_env = {}
    model._semantic_guidance_instruction_by_env = {}
    model._task_instruction_by_env = {0: task}
    model.obs_by_env = {0: {"lang": task}}
    model.action_chunks_by_env = {0: _chunk(0, 2)}
    model.step_by_env = {0: 0}
    ticket = {
        "source": "staged_vlm",
        "task_plan": {
            "steps": [
                {
                    "stage": "partial_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the left bowl with the center bowl",
                    "expected_outcome": "two bowls form one partial stack",
                    "skill_id": None,
                    "constraints": [],
                },
                {
                    "stage": "complete_stack",
                    "intent": "vla_act",
                    "subgoal": "stack the remaining bowl with the partial stack",
                    "expected_outcome": "all bowls form one stack",
                    "skill_id": None,
                    "constraints": [],
                },
            ],
        },
    }

    model._load_replay_task_plan(ticket, replay_path=tmp_path / "ticket.json")
    plan = model._ensure_replay_task_plan(0)

    assert plan is not None and plan.active is not None
    assert plan.active.attempts == 1
    assert model._semantic_guidance_remaining_by_env[0] == 2
    assert "Current recovery subgoal: stack the left bowl" in model.obs_by_env[0]["lang"]
    assert 0 not in model.action_chunks_by_env


def test_task_only_vla_keeps_native_instruction_for_task_plan(
    tmp_path: Path,
) -> None:
    model = _online_plan_model(tmp_path, mode="execute")
    model._vla_instruction_capability = "task_only"
    original = model.obs_by_env[0]["lang"]

    model._maybe_initialize_online_task_plan(0)

    assert model._task_plan_by_env[0].active is not None
    assert model.obs_by_env[0]["lang"] == original
    assert model._semantic_guidance_remaining_by_env == {}
    assert model._semantic_guidance_instruction_by_env == {}


def _task_start_planner_output() -> dict:
    return {
        "intent": "vla_act",
        "rationale": "start visible bowl stack",
        "confidence": 0.9,
        "subgoal": "stack left bowl with center bowl",
        "vla_instruction": "stack the left bowl with the center bowl",
        "skill_id": None,
        "skill_args": {},
        "expected_outcome": "two bowls form one visible stack",
        "memory_note": "",
        "failure_type": "",
        "scene_graph_update": {},
        "proposed_plan": [
            {
                "stage": "first_pair",
                "intent": "vla_act",
                "subgoal": "stack left bowl with center bowl",
                "expected_outcome": "two bowls form one visible stack",
                "skill_id": None,
                "skill_args": {},
                "constraints": [],
            },
            {
                "stage": "remaining_bowl",
                "intent": "vla_act",
                "subgoal": "stack remaining bowl with stacked bowls",
                "expected_outcome": "all three bowls form one visible stack",
                "skill_id": None,
                "skill_args": {},
                "constraints": [],
            },
        ],
    }


def _online_plan_model(tmp_path: Path, *, mode: str) -> Model:
    model = Model.__new__(Model)
    task = "stack the three bowls together"
    model.model_cfg = {"carve_planner_max_calls_per_episode": 4}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model._task_instruction_by_env = {0: task}
    model._task_plan_mode = mode
    model._vla_instruction_capability = "open_vocab_subgoal"
    model._task_plan_attempted_envs = set()
    model._task_plan_abandoned_envs = set()
    model._task_plan_by_env = {}
    model._semantic_max_recoveries = 1
    model._semantic_guidance_calls = 2
    model._semantic_guidance_remaining_by_env = {}
    model._semantic_guidance_instruction_by_env = {}
    model._agent_memory_by_env = {}
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model.step_by_env = {0: 0}
    model._planner_trace_path = tmp_path / f"task-plan-{mode}.jsonl"
    model._high_level_agent = GuardedHighLevelAgent(
        lambda _request: _task_start_planner_output()
    )
    return model


@pytest.mark.parametrize("schedule,expected", [
    ((), [160, 320, 480, 640, 800, 960]),
    ((160, 512, 864), [160, 512, 864]),
])
def test_native_frames_reach_live_plan_and_checkpoints_through_late_budget(tmp_path, schedule, expected):
    model = _online_plan_model(tmp_path, mode="execute")
    head = np.zeros((480, 640, 3), dtype=np.uint8)
    model._semantic_image_profile = "head_native"
    model.obs_by_env[0]["_semantic_head"] = head
    planner_requests, critic_requests = [], []

    def planner(request):
        planner_requests.append(request)
        return _task_start_planner_output()

    def critic(request):
        critic_requests.append(request)
        return {"status": "inconclusive", "observed_outcome": "occluded", "confidence": 0.9}

    model._high_level_agent = GuardedHighLevelAgent(planner)
    model._task_plan_visual_critic = GuardedVisualVerifier(critic)
    model._task_plan_checkpoint_steps = 160
    model._task_plan_max_checks = len(expected)
    model._task_plan_check_schedule = schedule
    model._task_plan_max_attempts_per_stage = 2
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._maybe_initialize_online_task_plan(0)
    assert planner_requests[0]["frames"]["head"] is head
    assert len(planner_requests[0]["frames"]) == 1
    for step in range(0, 1051):
        model.step_by_env[0] = step
        assert not model._apply_task_plan_checkpoint(0)
    assert len(critic_requests) == len(expected)
    for request in critic_requests:
        assert list(request["frames"]) == ["current_head"]
        assert request["frames"]["current_head"] is head
    rows = [json.loads(row) for row in model._planner_trace_path.read_text().splitlines()]
    checks = [row for row in rows if row.get("trigger") == "scheduled_task_plan_checkpoint"]
    assert [row["timestep"] for row in checks] == expected
    assert all(row["check_schedule"] == list(schedule) for row in checks)
    assert all(row["semantic_image_profile"] == "head_native" for row in checks)
    assert model._task_plan_by_env[0].active.step.stage == "first_pair"


@pytest.mark.parametrize("value", ["0:160", "160:160", "160:159", "160:200", "160:320:480:640",
                                  "-1", "1.0", "true", "160:", [160, 512]])
def test_task_plan_schedule_rejects_invalid_or_overbudget_config(value):
    from XPolicyLab.policy.starVLA.runtime_config import resolve_task_plan_check_schedule
    with pytest.raises(ValueError):
        resolve_task_plan_check_schedule(value, 3, 160)


def test_task_plan_schedule_config_preserves_legacy_and_parses_explicit_steps():
    from XPolicyLab.policy.starVLA.runtime_config import resolve_task_plan_check_schedule
    assert resolve_task_plan_check_schedule(None, 3, 160) == ()
    assert resolve_task_plan_check_schedule("", 3, 160) == ()
    assert resolve_task_plan_check_schedule("160:512:864", 3, 160) == (160, 512, 864)


def test_task_plan_schedule_missing_frames_and_late_arrival_do_not_burst(tmp_path):
    model = _online_plan_model(tmp_path, mode="execute")
    model._vla_instruction_capability = "task_only"
    model._task_plan_checkpoint_steps = 160
    model._task_plan_max_checks = 3
    model._task_plan_check_schedule = (160, 512, 864)
    model._task_plan_max_attempts_per_stage = 1
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._task_plan_visual_critic = GuardedVisualVerifier(lambda _: {
        "status": "inconclusive", "observed_outcome": "occluded", "confidence": 0.9})
    original_task = model.obs_by_env[0]["lang"]
    model._maybe_initialize_online_task_plan(0)
    frames = model._semantic_frames
    model._semantic_frames = lambda *args, **kwargs: {}
    model.step_by_env[0] = 160
    assert not model._apply_task_plan_checkpoint(0)
    assert not model._task_plan_checks_by_env
    assert not model._task_plan_last_check_by_env
    model._semantic_frames = frames
    for step in (520, 521, 600, 679, 680, 681, 863, 864, 1040):
        model.step_by_env[0] = step
        assert not model._apply_task_plan_checkpoint(0)
    rows = [json.loads(line) for line in model._planner_trace_path.read_text().splitlines()]
    checks = [row["timestep"] for row in rows if row.get("trigger") == "scheduled_task_plan_checkpoint"]
    assert checks == [520, 680, 864]
    assert model._task_plan_checks_by_env == {0: 3}
    assert model.obs_by_env[0]["lang"] == original_task


def test_online_task_plan_shadow_records_without_changing_execution(
    tmp_path: Path,
) -> None:
    model = _online_plan_model(tmp_path, mode="shadow")
    original = model.obs_by_env[0]["lang"]

    model._maybe_initialize_online_task_plan(0)

    assert model._task_plan_by_env == {}
    assert model.obs_by_env[0]["lang"] == original
    assert 0 in model.action_chunks_by_env
    assert "task_plan_shadow_recorded" in model._planner_trace_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("field,value", [
    ("carve_task_plan_mode", "execute"),
    ("carve_semantic_image_profile", "vla_224"),
    ("carve_planner_backend", "replay"),
    ("carve_semantic_role", "critic"),
    ("carve_mode", "baseline"),
    ("carve_planner_replay_path", "old-plan.json"),
])
def test_inspection_rejects_non_shadow_configuration_before_loading(field, value):
    config = {
        "carve_inspection_shadow": True,
        "carve_task_plan_mode": "shadow",
        "carve_semantic_image_profile": "head_native",
        "carve_planner_backend": "staged_vlm",
        "carve_semantic_role": "planner",
        "carve_mode": "full",
    }
    config[field] = value
    with pytest.raises(ValueError, match="inspection shadow requires"):
        Model(config)


def _inspection_shadow_model(tmp_path, infer):
    model = _online_plan_model(tmp_path, mode="shadow")
    model._inspection_shadow = True
    model._semantic_image_profile = "head_native"
    model.obs_by_env[0]["_semantic_head"] = np.zeros((64, 96, 3), dtype=np.uint8)
    model._task_plan_checkpoint_steps = 16
    model._task_plan_max_checks = 2
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._maybe_initialize_online_task_plan(0)
    model._initialize_inspection_shadow(infer, "test_provider")
    return model


def test_inspection_shadow_preserves_plan_actions_and_memory_with_bounded_calls(tmp_path):
    calls = []

    def infer(request):
        calls.append(request)
        payload = json.loads(request["user_prompt"])
        if "frame" in payload:
            return {"inspect": {
                "frame_id": payload["frame"]["frame_id"],
                "left": 0, "top": 0, "right": 32, "bottom": 32,
            }}
        assert list(request["frames"]) == ["global", "region"]
        return {"stages": [{
            "stage_id": stage["stage_id"], "state": "present", "confidence": 0.9,
            "evidence": "The bowls are visibly nested",
        } for stage in payload["predicates"]]}

    model = _inspection_shadow_model(tmp_path, infer)
    original_plan = model._task_plan_by_env[0].to_dict()
    original_chunk = model.action_chunks_by_env[0].copy()
    original_lang = model.obs_by_env[0]["lang"]
    for step in range(80):
        model.step_by_env[0] = step
        assert model._apply_inspection_shadow(0) == (step in {16, 32})
        assert not model._apply_task_plan_checkpoint(0)
    assert len(calls) == 4
    assert model._task_plan_checks_by_env == {0: 2}
    assert model._task_plan_by_env[0].to_dict() == original_plan
    assert model.obs_by_env[0]["lang"] == original_lang
    assert model._agent_memory_by_env == {}
    assert model._semantic_guidance_instruction_by_env == {}
    np.testing.assert_array_equal(model.action_chunks_by_env[0], original_chunk)
    rows = [json.loads(row) for row in model._planner_trace_path.read_text().splitlines()]
    checks = [row for row in rows if row.get("event") == "inspection_progress_shadow"]
    assert [row["timestep"] for row in checks] == [16, 32]
    assert all(row["accepted"] and row["confirmed_prefix"] == ["first_pair", "remaining_bowl"] for row in checks)
    assert all(not row["control_applied"] and not row["memory_written"] for row in checks)


@pytest.mark.parametrize("change", ["reset", "observation", "step", "plan"])
def test_inspection_shadow_rejects_live_context_changes(tmp_path, change):
    calls = []

    def infer(request):
        calls.append(request)
        if change == "reset":
            model._reset_generation = 1
        if change == "observation":
            model.obs_by_env[0] = dict(model.obs_by_env[0])
        if change == "step":
            model.step_by_env[0] += 1
        if change == "plan":
            model._task_plan_by_env[0].reset()
        return {"inspect": None}

    model = _inspection_shadow_model(tmp_path, infer)
    model.step_by_env[0] = 16
    assert model._apply_inspection_shadow(0)
    assert len(calls) == 1
    row = json.loads(model._planner_trace_path.read_text().splitlines()[-1])
    assert not row["accepted"] and row["confirmed_prefix"] == []
    assert "changed during inspection" in row["error"]


def test_inspection_shadow_cannot_promote_itself_to_control(tmp_path):
    model = _inspection_shadow_model(tmp_path, lambda _: {"inspect": None})
    model._task_plan_mode = "execute"
    with pytest.raises(RuntimeError, match="not admitted"):
        model._apply_inspection_shadow(0)


def test_inspection_shadow_requires_two_image_capacity(tmp_path):
    from agentic_vla.runtime.agent import PolicyServiceVisionPlanner

    model = _online_plan_model(tmp_path, mode="shadow")
    with pytest.raises(ValueError, match="full frame and crop"):
        model._initialize_inspection_shadow(PolicyServiceVisionPlanner(lambda _: {}), "test")


def test_online_task_plan_execute_advances_only_after_visual_confirmation(
    tmp_path: Path,
) -> None:
    model = _online_plan_model(tmp_path, mode="execute")
    model.carve_mode = "agentic"
    model.carve_recovery_compute_calls = 0
    model._recovery_compute_remaining_by_env = {}
    model._task_plan_checkpoint_steps = 128
    model._task_plan_max_checks = 4
    model._task_plan_max_attempts_per_stage = 2
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._task_plan_visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "two bowls form one visible stack",
            "confidence": 0.9,
        }
    )

    model._maybe_initialize_online_task_plan(0)
    plan = model._task_plan_by_env[0]
    assert plan.active is not None and plan.active.step.stage == "first_pair"
    assert "Current recovery subgoal" in model.obs_by_env[0]["lang"]

    model.step_by_env[0] = 128
    assert model._apply_task_plan_checkpoint(0)

    assert plan.active is not None and plan.active.step.stage == "remaining_bowl"
    assert plan.receipts[0].status.value == "confirmed"
    assert "remaining bowl" in model.obs_by_env[0]["lang"]
    assert model._task_plan_checks_by_env[0] == 1


def test_task_only_confirmation_updates_ledger_without_restarting_vla(
    tmp_path: Path,
) -> None:
    model = _online_plan_model(tmp_path, mode="execute")
    model._vla_instruction_capability = "task_only"
    model.carve_mode = "agentic"
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}
    model._task_plan_checkpoint_steps = 128
    model._task_plan_max_checks = 4
    model._task_plan_max_attempts_per_stage = 2
    model._task_plan_stale_checks_required = 2
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._task_plan_contradiction_streak_by_env = {}
    model._task_plan_visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "two bowls form one visible stack",
            "confidence": 0.9,
        }
    )

    model._maybe_initialize_online_task_plan(0)
    active_chunk = _chunk(20, 3)
    model.action_chunks_by_env[0] = active_chunk
    model.step_by_env[0] = 128

    assert not model._apply_task_plan_checkpoint(0)
    assert model.action_chunks_by_env[0] is active_chunk
    assert model._recovery_compute_remaining_by_env == {}
    assert model.obs_by_env[0]["lang"] == model._task_instruction_by_env[0]
    assert '"action": "task_plan_advance_ledger_only"' in (
        model._planner_trace_path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("inspection_shadow", [False, True])
@pytest.mark.parametrize("recovery_unavailable", [False, True])
def test_task_plan_routes_no_progress_event_through_guarded_planner(
    tmp_path: Path, inspection_shadow: bool, recovery_unavailable: bool,
) -> None:
    model = _online_plan_model(tmp_path, mode="shadow" if inspection_shadow else "execute")
    model._inspection_shadow = inspection_shadow
    model._vla_instruction_capability = "task_only"
    model._safe_stop_envs = set()
    model._high_level_agent_disabled_envs = set()
    model._visual_critic = None
    model.action_dim = 14
    model.carve_mode = "full"
    model._semantic_role = "planner"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._task_plan_last_check_by_env = {}
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}
    model._fault_recovered_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    if recovery_unavailable:
        model._recovery_verification_by_env = {0: SimpleNamespace(available=False)}
    model._maybe_initialize_online_task_plan(0)
    original_plan = model._task_plan_by_env[0].to_dict()
    if inspection_shadow:
        from agentic_vla.runtime.agent import TaskPreservingRecoveryPlanner

        model._high_level_agent = GuardedHighLevelAgent(TaskPreservingRecoveryPlanner())
    active_chunk = _chunk(30, 3)
    model.action_chunks_by_env[0] = active_chunk
    model.step_by_env[0] = 320
    risk = SimpleNamespace(
        event="no_progress",
        evidence={"event_streak": 2},
        to_dict=lambda: {"event": "no_progress"},
    )

    model._apply_agentic_progress_check(0, risk)

    applied = not inspection_shadow and not recovery_unavailable
    if applied:
        assert 0 not in model.action_chunks_by_env
        assert model._recovery_compute_remaining_by_env[0] == 2
        assert model._task_plan_last_check_by_env[0] == 320
    else:
        assert model.action_chunks_by_env[0] is active_chunk
        assert model._recovery_compute_remaining_by_env == {}
        assert 0 not in model._task_plan_last_check_by_env
    assert model.obs_by_env[0]["lang"] == model._task_instruction_by_env[0]
    assert (0 in model._fault_recovered_envs) == applied
    trace = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"trigger": "no_progress"' in trace
    expected_action = (
        "planner_continue_original" if inspection_shadow else
        "planner_replan_rejected_recovery_budget" if recovery_unavailable else "planner_vla_replan"
    )
    assert f'"control_action": "{expected_action}"' in trace
    if inspection_shadow:
        assert model._task_plan_for_execution(0) is None
        assert model._task_plan_by_env[0].to_dict() == original_plan
        event = json.loads(trace.splitlines()[-1])
        assert event["task_plan_before"] == {} and event["task_plan_after"] == {}


def test_stack_task_plan_requires_repeated_contradiction_before_retry(
    tmp_path: Path,
) -> None:
    model = _online_plan_model(tmp_path, mode="execute")
    model.carve_mode = "agentic"
    model.carve_recovery_compute_calls = 0
    model._recovery_compute_remaining_by_env = {}
    model._task_plan_checkpoint_steps = 128
    model._task_plan_max_checks = 4
    model._task_plan_max_attempts_per_stage = 2
    model._task_plan_stale_checks_required = 2
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._task_plan_contradiction_streak_by_env = {}
    model._task_plan_visual_critic = GuardedVisualVerifier(
        lambda _request: (_ for _ in ()).throw(
            AssertionError("free-form verifier must not decide a known stack stage")
        )
    )
    from agentic_vla.toolchain import GuardedGroupedVisualVerifier

    grouped_requests = []

    def enumerate_groups(request):
        grouped_requests.append(request)
        return {
            "visible": True,
            "groups": [
                "black bowl at left",
                "black bowl at center",
                "black bowl at right",
            ],
            "confidence": 0.9,
        }

    model._task_plan_grouped_visual_critic = GuardedGroupedVisualVerifier(
        enumerate_groups
    )

    model._maybe_initialize_online_task_plan(0)
    plan = model._task_plan_by_env[0]
    model.action_chunks_by_env[0] = _chunk(10, 3)
    model.step_by_env[0] = 128

    assert not model._apply_task_plan_checkpoint(0)
    assert plan.active is not None and plan.active.step.stage == "first_pair"
    assert plan.active.attempts == 1
    assert 0 in model.action_chunks_by_env

    model.step_by_env[0] = 256
    assert model._apply_task_plan_checkpoint(0)
    assert plan.active is not None and plan.active.step.stage == "first_pair"
    assert plan.active.attempts == 2
    assert plan.receipts[0].status.value == "retry_required"
    assert set(grouped_requests[0]["frames"]) == {"current_head"}
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"verification_mode": "harness_group_count"' in row
    assert '"verification_frames": ["current_head"]' in row
    assert json.loads(row.splitlines()[-1])["verification_frame_shapes"] == {
        "current_head": [8, 8, 3]
    }
    assert '"observed_value": 3.0' in row
    assert '"action": "task_plan_contradiction_observed"' in row
    assert '"contradiction_streak": 2' in row


def test_task_plan_verifier_protocol_failure_opens_circuit(
    tmp_path: Path,
) -> None:
    from agentic_vla.toolchain import GuardedGroupedVisualVerifier

    model = _online_plan_model(tmp_path, mode="execute")
    model._vla_instruction_capability = "task_only"
    model.carve_mode = "agentic"
    model.carve_recovery_compute_calls = 0
    model._recovery_compute_remaining_by_env = {}
    model._task_plan_checkpoint_steps = 128
    model._task_plan_max_checks = 4
    model._task_plan_max_attempts_per_stage = 2
    model._task_plan_stale_checks_required = 2
    model._task_plan_max_protocol_failures = 1
    model._task_plan_last_check_by_env = {}
    model._task_plan_checks_by_env = {}
    model._task_plan_contradiction_streak_by_env = {}
    model._task_plan_protocol_failures_by_env = {}
    model._task_plan_visual_critic = GuardedVisualVerifier(
        lambda _request: "not valid JSON"
    )
    model._task_plan_grouped_visual_critic = GuardedGroupedVisualVerifier(
        lambda _request: "not valid JSON"
    )

    model._maybe_initialize_online_task_plan(0)
    model.step_by_env[0] = 128

    assert not model._apply_task_plan_checkpoint(0)
    assert 0 in model._task_plan_abandoned_envs
    assert model.obs_by_env[0]["lang"] == model._task_instruction_by_env[0]
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"action": "task_plan_verifier_unavailable_fallback_original"' in row
    assert '"protocol_failure_streak": 1' in row


def test_agentic_local_critic_uses_task_preserving_replan(tmp_path: Path) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._planner_trace_path = tmp_path / "critic.jsonl"
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._high_level_agent = None
    model._semantic_role = "critic"
    model.carve_recovery_compute_calls = 3
    model._recovery_compute_remaining_by_env = {}
    model._visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": "contradicted",
            "observed_outcome": "the full tower is not complete",
            "confidence": 0.9,
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_agentic_progress_check(0, risk)

    instruction = model.obs_by_env[0]["lang"]
    assert instruction.startswith(task)
    assert "unfinished task steps" in instruction
    assert 0 not in model.action_chunks_by_env
    assert 0 not in model._safe_stop_envs
    assert model._recovery_compute_remaining_by_env[0] == 3
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"role": "critic"' in row
    assert '"action": "task_preserving_vla_replan"' in row
    assert '"raw_output"' in row


@pytest.mark.parametrize("status, outcome, expected_action, allow_stop", [
    ("confirmed", "the full tower is complete", "critic_confirmed_continue_original_no_stop_authority", False),
    ("inconclusive", "grippers obscure the board and its support", "critic_inconclusive_no_intervention", False),
    ("inconclusive", "grippers obscure the board and its support", "critic_inconclusive_no_intervention", True),
])
def test_agentic_local_critic_confirmation_has_no_default_stop_authority(
    tmp_path: Path,
    status: str,
    outcome: str,
    expected_action: str,
    allow_stop: bool,
) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    original_chunk = _chunk(0, 3)
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: original_chunk}
    model._planner_trace_path = tmp_path / "critic-confirmed-observe.jsonl"
    model._agent_memory_by_env = {}
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._safe_stop_envs = set()
    model._visual_critic_calls_by_env = {}
    model._visual_critic_disabled_envs = set()
    model._last_planner_step_by_env = {}
    model._critic_safe_stop_enabled = allow_stop
    model._task_instruction_by_env = {0: task}
    model.carve_recovery_compute_calls = 3
    model._recovery_compute_remaining_by_env = {}
    model._semantic_guidance_remaining_by_env = {}
    model._visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": status,
            "observed_outcome": outcome,
            "confidence": 0.9,
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_visual_critic(0, risk)

    assert 0 not in model._safe_stop_envs
    assert model.action_chunks_by_env[0] is original_chunk
    assert model.obs_by_env[0]["lang"] == task
    assert 0 not in model._recovery_compute_remaining_by_env
    assert 0 not in model._semantic_guidance_remaining_by_env
    assert model._visual_critic_calls_by_env[0] == 1
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert f'"action": "{expected_action}"' in row


def test_agentic_local_critic_confirmation_stops_only_when_admitted(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._planner_trace_path = tmp_path / "critic-confirmed-stop.jsonl"
    model._agent_memory_by_env = {}
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._safe_stop_envs = set()
    model._visual_critic_calls_by_env = {}
    model._visual_critic_disabled_envs = set()
    model._last_planner_step_by_env = {}
    model._critic_safe_stop_enabled = True
    model._task_instruction_by_env = {0: task}
    model._visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "the full tower is complete",
            "confidence": 0.9,
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_visual_critic(0, risk)

    assert 0 in model._safe_stop_envs
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"action": "safe_stop_confirmed_complete"' in row


def test_agentic_local_critic_protocol_failure_keeps_original_execution(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    original_chunk = _chunk(0, 3)
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: original_chunk}
    model._planner_trace_path = tmp_path / "critic-invalid.jsonl"
    model._agent_memory_by_env = {}
    model._safe_stop_envs = set()
    model._risk_monitors = {0: ExecutionRiskMonitor()}
    model._semantic_guidance_remaining_by_env = {}
    model._semantic_guidance_instruction_by_env = {}
    model._visual_critic_calls_by_env = {}
    model._visual_critic_disabled_envs = set()
    model._last_planner_step_by_env = {}
    model._visual_critic = GuardedVisualVerifier(lambda _request: "not valid JSON")
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_visual_critic(0, risk)

    assert model.obs_by_env[0]["lang"] == task
    assert model.action_chunks_by_env[0] is original_chunk
    assert 0 not in model._safe_stop_envs
    assert 0 in model._visual_critic_disabled_envs
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"action": "critic_unavailable_continue_original"' in row


def test_agentic_local_critic_budget_exhaustion_does_not_safe_stop(
    tmp_path: Path,
) -> None:
    model = _model_without_server(mode="full", chunks=[])
    task = "Build a tower using the wooden blocks and wooden boards."
    original_chunk = _chunk(0, 3)
    model.model_cfg = {"carve_planner_max_calls_per_episode": 1}
    model.obs_by_env = {
        0: {
            "lang": task,
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
    }
    model.step_by_env = {0: 778}
    model.action_chunks_by_env = {0: original_chunk}
    model._planner_trace_path = tmp_path / "critic-budget.jsonl"
    model._safe_stop_envs = set()
    model._visual_critic_calls_by_env = {0: 1}
    model._visual_critic_disabled_envs = set()
    model._visual_critic = GuardedVisualVerifier(
        lambda _request: {
            "status": "confirmed",
            "observed_outcome": "tower complete",
            "confidence": 0.99,
        }
    )
    risk = RiskAssessment(
        score=0.45,
        bucket="medium",
        event="no_progress",
        components={"no_progress": 1.0},
        evidence={"event_streak": 2},
    )

    model._apply_visual_critic(0, risk)

    assert model.action_chunks_by_env[0] is original_chunk
    assert 0 not in model._safe_stop_envs
    assert 0 in model._visual_critic_disabled_envs
    assert model._visual_critic.calls == 0
    row = model._planner_trace_path.read_text(encoding="utf-8")
    assert '"action": "critic_budget_exhausted_continue_original"' in row


# ---------------------------------------------------------------------------
# Reset-contract registry
#
# `STARVLA_RESET_STATE_REGISTRY` is now the single declared source of truth for
# what "the episode was reset" means: `Model.reset` resolves it against the live
# model instead of repeating a literal container map, and
# `STARVLA_RESET_STATE_FIELDS` is derived from it. The tests below therefore
# baseline on the registry and compare it *both ways* against the per-env
# attributes the adapter actually declares, so neither a new unregistered
# container (a silent cross-episode leak) nor a registry entry the adapter no
# longer has (a container that can never be cleared) can land unnoticed.
#
# Resolution is strict on purpose: the old `getattr(self, name, {})` default
# reported `before=0 / after=0` for a renamed attribute, which satisfied every
# downstream check while the real container kept the previous episode's state.
# ---------------------------------------------------------------------------

import ast  # noqa: E402
import re  # noqa: E402


from XPolicyLab.policy.starVLA.reset_contract import (  # noqa: E402
    STARVLA_RESET_STATE_FIELDS,
    STARVLA_RESET_STATE_REGISTRY,
    STARVLA_RESET_STATE_VERSION,
    resolve_reset_state_containers,
    validate_starvla_reset_state,
)

STARVLA_MODEL_PATH = (
    ROBODOJO_ROOT / "XPolicyLab" / "policy" / "starVLA" / "model.py"
)
# Naming conventions the adapter uses for per-episode, per-environment state.
_PER_ENV_ATTRIBUTE_PATTERN = re.compile(r"(_by_env|_envs|_monitors)$")
_CONTAINER_TYPES: dict[str, type] = {"dict": dict, "set": set}


def _starvla_model_ast() -> ast.Module:
    return ast.parse(STARVLA_MODEL_PATH.read_text(encoding="utf-8"))


def _reset_state_registry() -> list[tuple[str, str, type]]:
    """`(inventory_name, attribute_name, container_type)` from the contract."""

    registry: list[tuple[str, str, type]] = []
    for name, attribute, kind in STARVLA_RESET_STATE_REGISTRY:
        assert kind in _CONTAINER_TYPES, (
            f"{name} is registered as {kind!r}; reset can only clear a dict or a set"
        )
        registry.append((name, attribute, _CONTAINER_TYPES[kind]))
    return registry


def _declared_per_env_attributes() -> set[str]:
    """Every `self.<name>` in the adapter that looks like per-env state."""

    tree = _starvla_model_ast()
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and _PER_ENV_ATTRIBUTE_PATTERN.search(node.attr)
        ):
            found.add(node.attr)
        # getattr(self, "_name", ...) never appears as an ast.Attribute.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _PER_ENV_ATTRIBUTE_PATTERN.search(node.args[1].value)
        ):
            found.add(node.args[1].value)
    return found


def test_reset_state_registry_is_well_formed() -> None:
    """The declared registry must be usable as a single source of truth."""

    names = [name for name, _, _ in STARVLA_RESET_STATE_REGISTRY]
    attributes = [attribute for _, attribute, _ in STARVLA_RESET_STATE_REGISTRY]
    kinds = {kind for _, _, kind in STARVLA_RESET_STATE_REGISTRY}

    assert len(STARVLA_RESET_STATE_REGISTRY) == 36
    # A duplicate inventory name would collapse two containers into one receipt
    # entry; a duplicate attribute would publish one container twice.
    assert len(set(names)) == len(names)
    assert len(set(attributes)) == len(attributes)
    assert kinds <= {"dict", "set"}
    assert all(name and attribute for name, attribute in zip(names, attributes))


def test_registry_validator_rejects_malformed_registries(monkeypatch) -> None:
    """Guard the guard: the contract's own import-time validator must have teeth.

    The assertions above observe that today's registry is well formed; this pins
    that a malformed one is actually refused rather than silently accepted.
    """

    import XPolicyLab.policy.starVLA.reset_contract as reset_contract

    first = reset_contract.STARVLA_RESET_STATE_REGISTRY[0]
    cases = {
        "duplicate inventory names": (first, (first[0], "_other_attr", first[2])),
        "duplicate attributes": (first, ("other_name", first[1], first[2])),
        "unknown kind": (first, ("other_name", "_other_attr", "list")),
    }
    for expected, registry in cases.items():
        monkeypatch.setattr(
            reset_contract, "STARVLA_RESET_STATE_REGISTRY", registry
        )
        with pytest.raises(RuntimeError, match=expected):
            reset_contract._validate_registry()


def test_published_inventory_is_derived_from_the_registry_in_order() -> None:
    assert STARVLA_RESET_STATE_FIELDS == tuple(
        name for name, _, _ in STARVLA_RESET_STATE_REGISTRY
    )
    assert len(STARVLA_RESET_STATE_FIELDS) == 36
    assert len(set(STARVLA_RESET_STATE_FIELDS)) == 36


def test_every_per_env_container_is_registered_for_reset() -> None:
    """Bidirectional drift check between the registry and the adapter source."""

    registered = {attribute for _, attribute, _ in STARVLA_RESET_STATE_REGISTRY}
    declared = _declared_per_env_attributes()

    unregistered = sorted(declared - registered)
    assert unregistered == [], (
        "per-episode containers are not cleared by Model.reset; register them in "
        "STARVLA_RESET_STATE_REGISTRY: " + ", ".join(unregistered)
    )
    # And the registry must not name containers the adapter no longer declares:
    # resolve_reset_state_containers would raise on every reset.
    stale = sorted(registered - declared)
    assert stale == [], (
        "STARVLA_RESET_STATE_REGISTRY names attributes model.py never sets: "
        + ", ".join(stale)
    )


def test_drift_check_would_catch_an_unregistered_container() -> None:
    """Guard the guard: the drift comparison must be able to fail.

    The check above is only meaningful if an added per-env container actually
    shows up on the unregistered side, so simulate one instead of trusting that
    the empty diff means the comparison works.
    """

    registered = {attribute for _, attribute, _ in STARVLA_RESET_STATE_REGISTRY}
    declared = _declared_per_env_attributes() | {"_brand_new_cache_by_env"}

    assert sorted(declared - registered) == ["_brand_new_cache_by_env"]
    assert _PER_ENV_ATTRIBUTE_PATTERN.search("_brand_new_cache_by_env")


def test_registered_inventory_names_track_their_attributes() -> None:
    for inventory_name, attribute, _container_type in _reset_state_registry():
        assert attribute.lstrip("_") == inventory_name, (
            f"{attribute} is published as {inventory_name!r}; keeping the names "
            "aligned is what makes the receipt readable against the adapter"
        )


def _model_with_registered_state() -> tuple[Model, list[tuple[str, str, type]]]:
    registry = _reset_state_registry()
    model = Model.__new__(Model)
    model._reset_generation = 0
    model._latest_env_idx_list = [0, 1]
    model.action_dim = 14
    model.input_color_order = "rgb"
    for _name, attribute, container_type in registry:
        if container_type is dict:
            setattr(model, attribute, {0: object(), 1: object()})
        else:
            setattr(model, attribute, {0, 1})
    # risk_monitors holds objects whose own state must be reset first.
    model._risk_monitors = {0: ExecutionRiskMonitor(), 1: ExecutionRiskMonitor()}
    model._reobservation_by_env = {0: ExecutionRiskMonitor(), 1: ExecutionRiskMonitor()}
    from agentic_vla.toolchain.recovery_verification import RecoveryVerificationLifecycle
    model._recovery_verification_by_env = {0: RecoveryVerificationLifecycle(), 1: RecoveryVerificationLifecycle()}
    model._planner_trace_path = None
    # `reset` reads the high-level agent through strict attribute access, so the
    # attribute must exist even when no agent was built. None is the value
    # `__init__` starts from and the one B0/C1 keep.
    model._high_level_agent = None
    return model, registry


def test_resolve_returns_the_live_containers_not_copies() -> None:
    """`reset` clears whatever this returns, so it must be the real objects."""

    model, registry = _model_with_registered_state()

    containers = resolve_reset_state_containers(model)

    assert tuple(containers) == STARVLA_RESET_STATE_FIELDS
    assert len(containers) == 36
    for name, attribute, container_type in registry:
        # Identity, not equality: a copy would be cleared instead of the
        # attribute the next episode reads.
        assert containers[name] is getattr(model, attribute), attribute
        assert isinstance(containers[name], container_type)


@pytest.mark.parametrize(
    "registry_index",
    [
        pytest.param(index, id=name)
        for index, (name, _, _) in enumerate(STARVLA_RESET_STATE_REGISTRY)
    ],
)
def test_resolve_rejects_a_missing_registered_attribute(registry_index: int) -> None:
    """No entry may fall back to an empty container, not just the first one."""

    model, registry = _model_with_registered_state()
    name, attribute, _kind = registry[registry_index]
    delattr(model, attribute)

    with pytest.raises(RuntimeError) as excinfo:
        resolve_reset_state_containers(model)

    # The message has to name both sides, or a rename is undebuggable.
    message = str(excinfo.value)
    assert name in message
    assert attribute in message
    assert "drifted" in message


@pytest.mark.parametrize(
    "wrong_kind",
    [
        pytest.param("dict", id="set_slot_holds_a_dict"),
        pytest.param("set", id="dict_slot_holds_a_set"),
    ],
)
def test_resolve_rejects_the_wrong_container_type(wrong_kind: str) -> None:
    model, _registry = _model_with_registered_state()
    # Pick a slot declared as the *other* kind and put the wrong container there.
    declared_kind = "set" if wrong_kind == "dict" else "dict"
    name, attribute, _kind = next(
        entry for entry in STARVLA_RESET_STATE_REGISTRY if entry[2] == declared_kind
    )
    setattr(model, attribute, {} if wrong_kind == "dict" else set())

    with pytest.raises(RuntimeError) as excinfo:
        resolve_reset_state_containers(model)

    message = str(excinfo.value)
    assert name in message
    assert attribute in message
    assert f"must be a {declared_kind}" in message


def test_reset_raises_instead_of_reporting_a_phantom_empty_container() -> None:
    """The fixed defect: a missing container must fail loudly, not report zero.

    With `getattr(self, attribute, {})` this call returned a receipt claiming
    `before=0 / after=0` for the missing container, which passed the wire
    contract, the ledger and the finalizer audit while the live state was never
    touched.
    """

    model, registry = _model_with_registered_state()
    name, attribute, _kind = registry[-1]
    _survivor_name, survivor_attribute, _survivor_kind = registry[0]
    survivor = getattr(model, survivor_attribute)
    delattr(model, attribute)

    with pytest.raises(RuntimeError) as excinfo:
        model.reset(
            reset_context={
                "episode_id": "run:session:batch-0000001",
                "episode_seq": 1,
                "reset_session_id": "session",
                "active_env_ids": [0, 1],
                "layout_seeds": [{"env_idx": 0, "layout_id": 0}],
            }
        )

    message = str(excinfo.value)
    assert name in message
    assert attribute in message
    # Resolution happens before anything is cleared, so the boundary is not
    # half-applied and the generation is not consumed.
    assert len(survivor) == 2
    assert model._reset_generation == 0


def test_reset_clears_every_registered_container_and_reports_it() -> None:
    model, registry = _model_with_registered_state()
    # References captured before the reset: `clear()` must empty these very
    # objects, not containers the model happened to rebind afterwards.
    held = {attribute: getattr(model, attribute) for _, attribute, _ in registry}

    receipt = model.reset(
        reset_context={
            "episode_id": "run:session:batch-0000001",
            "episode_seq": 1,
            "reset_session_id": "session",
            "active_env_ids": [0, 1],
            "layout_seeds": [
                {"env_idx": 0, "layout_id": 0},
                {"env_idx": 1, "layout_id": 1},
            ],
        }
    )

    assert receipt["state_inventory_version"] == STARVLA_RESET_STATE_VERSION
    assert receipt["state_inventory"] == list(STARVLA_RESET_STATE_FIELDS)
    # The receipt reports the inventory in registry order, both before & after.
    assert tuple(receipt["before"]["state_entries"]) == STARVLA_RESET_STATE_FIELDS
    assert tuple(receipt["after"]["state_entries"]) == STARVLA_RESET_STATE_FIELDS
    assert set(receipt["before"]["state_entries"].values()) == {2}
    assert set(receipt["after"]["state_entries"].values()) == {0}

    # The live attributes, not copies, must be the ones that were emptied.
    for _name, attribute, container_type in registry:
        container = getattr(model, attribute)
        assert isinstance(container, container_type)
        assert len(container) == 0, attribute
        assert container is held[attribute], attribute
        assert len(held[attribute]) == 0, attribute

    # The receipt satisfies the contract the finalizer and EvalEnv enforce.
    validate_starvla_reset_state(receipt)

    # Reset is an episode boundary, not a model reconfiguration.
    assert model.action_dim == 14
    assert model.input_color_order == "rgb"
    assert model._latest_env_idx_list == [0]


def test_reset_receipt_would_reject_an_unregistered_leftover_container() -> None:
    """Guard the guard: the contract must actually notice leftover state."""

    model, registry = _model_with_registered_state()
    receipt = model.reset()
    leaked_field = registry[-1][0]

    receipt["after"]["state_entries"][leaked_field] = 1
    with pytest.raises(ValueError, match="left episode state populated"):
        validate_starvla_reset_state(receipt)

    del receipt["after"]["state_entries"][leaked_field]
    with pytest.raises(ValueError, match="state-entry keys mismatch"):
        validate_starvla_reset_state(receipt)


def test_reset_receipt_survives_canonicalizing_serialization() -> None:
    """The contract must hold for the persisted receipt, not just the live one.

    ``EvalEnv`` appends each receipt with ``json.dumps(..., sort_keys=True)`` so
    the trace is byte-stable, and RoboDojo's own result writer canonicalizes too.
    That reorders the nested ``after.state_entries`` mapping alphabetically. An
    ordered comparison of those keys therefore passed in memory and could never
    pass on disk, which is exactly how a real 3-episode run failed the audit
    while the client-side check accepted every receipt. The ordered part of the
    contract lives in ``state_inventory``, which is a list and survives.
    """

    model, _registry = _model_with_registered_state()
    receipt = model.reset(
        reset_context={
            "episode_id": "run:session:batch-0000001",
            "episode_seq": 1,
            "reset_session_id": "session",
            "active_env_ids": [0, 1],
            "layout_seeds": [
                {"env_idx": 0, "layout_id": 0},
                {"env_idx": 1, "layout_id": 1},
            ],
        }
    )

    # In-memory the mapping is in registry order ...
    assert tuple(receipt["after"]["state_entries"]) == STARVLA_RESET_STATE_FIELDS
    validate_starvla_reset_state(receipt)

    # ... and on disk it is alphabetical, which must still validate.
    persisted = json.loads(json.dumps(receipt, sort_keys=True, default=str))
    assert tuple(persisted["after"]["state_entries"]) != STARVLA_RESET_STATE_FIELDS
    assert tuple(persisted["after"]["state_entries"]) == tuple(
        sorted(STARVLA_RESET_STATE_FIELDS)
    )
    # The ordered contract rides on the inventory list, which is unaffected.
    assert persisted["state_inventory"] == list(STARVLA_RESET_STATE_FIELDS)
    validate_starvla_reset_state(persisted)


def test_reset_receipt_keys_are_still_checked_exactly() -> None:
    """Order-insensitive must not mean membership-insensitive."""

    model, _registry = _model_with_registered_state()
    persisted = json.loads(
        json.dumps(model.reset(), sort_keys=True, default=str)
    )
    dropped = STARVLA_RESET_STATE_FIELDS[0]

    missing = json.loads(json.dumps(persisted))
    del missing["after"]["state_entries"][dropped]
    with pytest.raises(ValueError) as excinfo:
        validate_starvla_reset_state(missing)
    assert dropped in str(excinfo.value)
    assert "missing=" in str(excinfo.value)

    extra = json.loads(json.dumps(persisted))
    extra["after"]["state_entries"]["_unregistered_by_env"] = 0
    with pytest.raises(ValueError) as excinfo:
        validate_starvla_reset_state(extra)
    assert "_unregistered_by_env" in str(excinfo.value)
    assert "unexpected=" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Native-path request evidence (plan 9.21 follow-up item 1)
#
# B0 bypasses CarveRuntime, so it produced no per-request input/output/latency
# record and could not be compared with C1/C2/C3 on identical inputs. These
# tests pin the two properties that make the new record usable as evidence:
# it must not change what the baseline returns, and it must use the same digest
# helpers as the wrapped arms so the two traces are comparable.
# ---------------------------------------------------------------------------


def _native_model(tmp_path, *, trace: bool, steps: int = 4):
    """A baseline-mode Model whose only client is a recording fake."""

    model = Model.__new__(Model)
    model.action_seed_base = 101
    model.action_dim = 14
    model.num_ddim_steps = steps
    model.use_ddim = True
    model.unnorm_key = "arx_x5"
    model.carve_mode = "baseline"
    model.carve_execute_horizon = 16
    model.obs_by_env = {
        0: {
            "lang": "stack the bowls",
            "image": [
                np.full((8, 8, 3), 7, dtype=np.uint8),
                np.full((8, 8, 3), 8, dtype=np.uint8),
                np.full((8, 8, 3), 9, dtype=np.uint8),
            ],
            "state": np.arange(14, dtype=np.float32)[None, :],
        }
    }
    model.step_by_env = {0: 32}
    model._carve_runtime = None
    model._carve_prepared = None
    model._active_execute_horizon_by_env = {}
    model._semantic_guidance_remaining_by_env = {}
    model._reset_generation = 1
    model._starvla_server_meta = {
        "framework": "QwenPI_v3",
        "native_inference_steps": 4,
        "ckpt_path": "/models/steps_100000_pytorch_model.pt",
        "default_unnorm_key": "arx_x5",
    }
    model._native_trace_failures = 0
    model._native_trace_path = (tmp_path / "runtime_trace.jsonl") if trace else None

    actions = np.arange(16 * 14, dtype=np.float32).reshape(1, 16, 14)
    seen: list[dict] = []

    class _Client:
        def predict_action(self, payload):
            seen.append(payload)
            return {"ok": True, "data": {"actions": actions}}

    model.client = _Client()
    return model, seen, actions[0]


def _native_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_native_trace_is_off_by_default_and_changes_nothing(tmp_path) -> None:
    """With no path configured the baseline must behave exactly as before."""

    off, seen_off, expected = _native_model(tmp_path, trace=False)
    returned_off = off._infer_chunk(0)

    on, seen_on, _ = _native_model(tmp_path, trace=True)
    returned_on = on._infer_chunk(0)

    # Same request payload and same returned actions with tracing on or off.
    assert seen_off[0].keys() == seen_on[0].keys()
    assert seen_off[0]["action_seed"] == seen_on[0]["action_seed"]
    np.testing.assert_array_equal(returned_off, returned_on)
    np.testing.assert_array_equal(returned_on, expected)
    # Nothing is written when the trace is disabled.
    assert not (tmp_path / "runtime_trace.jsonl").exists() or _native_rows(
        tmp_path / "runtime_trace.jsonl"
    )


def test_native_trace_records_input_output_and_latency(tmp_path) -> None:
    model, seen, expected = _native_model(tmp_path, trace=True)

    model._infer_chunk(0)

    rows = _native_rows(tmp_path / "runtime_trace.jsonl")
    assert len(rows) == 1
    row = rows[0]

    # The three things B0 was missing.
    assert isinstance(row["model_latency_ms"], float) and row["model_latency_ms"] >= 0.0
    assert row["metadata"]["input_sha256"]
    assert row["metadata"]["action_sha256"]

    # Digests must come from the same helpers the wrapped arms use, or the
    # native and runtime traces cannot be compared row by row.
    from agentic_vla.runtime.adapters.starvla import (
        hash_starvla_request_input,
        summarize_starvla_request_input,
    )

    payload = seen[0]
    assert row["metadata"]["input_sha256"] == hash_starvla_request_input(payload)
    assert row["metadata"]["input_components"] == summarize_starvla_request_input(payload)
    assert (
        row["metadata"]["action_sha256"]
        == hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()
    )
    # Per-modality hashes are what localize an initial-RGB difference.
    components = row["metadata"]["input_components"]["examples"][0]
    assert len(components["image_sha256"]) == 3
    assert components["state_sha256"] and components["language_sha256"]

    assert row["success"] is True and row["error"] is None
    assert row["action_count"] == 16
    assert row["timestep"] == 32
    assert row["metadata"]["action_seed"] == 101 + 0 + 32
    assert row["metadata"]["inference_steps"] == 4
    assert row["metadata"]["controller"]["condition"] == "B0"
    assert row["metadata"]["controller"]["carve_mode"] == "baseline"
    assert row["metadata"]["path"] == "native"


def test_native_trace_leaves_wrapper_only_fields_null(tmp_path) -> None:
    """Absent measurements stay null instead of being filled with the request.

    A zero here would read as "no queueing delay observed"; null says the native
    path has no wrapper to measure, which is the honest statement.
    """

    model, _seen, _expected = _native_model(tmp_path, trace=True)
    model._infer_chunk(0)
    row = _native_rows(tmp_path / "runtime_trace.jsonl")[0]

    for field in (
        "runtime_latency_ms",
        "queue_age_ms",
        "reaction_latency_ms",
        "deadline_ms",
        "requested_controls",
        "task_cycle_latency_ms",
    ):
        assert row[field] is None, field
    assert row["metadata"]["precision"] is None
    assert row["deadline_miss"] is False
    assert row["dropped_controls"] == []


def test_native_trace_schema_matches_the_runtime_trace(tmp_path) -> None:
    """Same top-level keys as the frozen C1 runtime trace, so tooling is shared."""

    reference = (
        Path(__file__).resolve().parents[1]
        / "artifacts/robodojo/recovery_lifecycle_20260910/C1/runtime_trace.jsonl"
    )
    if not reference.is_file():
        pytest.skip("frozen C1 runtime trace not present")
    runtime_keys = set(json.loads(reference.read_text().splitlines()[0]))

    model, _seen, _expected = _native_model(tmp_path, trace=True)
    model._infer_chunk(0)
    native_keys = set(_native_rows(tmp_path / "runtime_trace.jsonl")[0])

    missing = sorted(runtime_keys - native_keys)
    assert missing == [], f"native record is missing runtime fields: {missing}"
    # Additions are allowed only when listed here, so shared tooling keeps
    # working and a new field cannot appear unnoticed. reset_generation is the
    # episode delimiter the wrapped traces never carried.
    assert sorted(native_keys - runtime_keys) == ["reset_generation"]


def test_native_trace_records_a_failed_request_then_reraises(tmp_path) -> None:
    model, _seen, _expected = _native_model(tmp_path, trace=True)

    class _Failing:
        def predict_action(self, payload):
            del payload
            return {"ok": False, "error": "server refused"}

    model.client = _Failing()

    with pytest.raises(RuntimeError, match="server refused"):
        model._infer_chunk(0)

    row = _native_rows(tmp_path / "runtime_trace.jsonl")[0]
    assert row["success"] is False
    assert "server refused" in row["error"]
    assert row["action_sha256"] is None if "action_sha256" in row else True
    assert row["metadata"]["action_sha256"] is None
    assert row["action_count"] is None


def test_native_trace_failure_does_not_break_the_control_path(tmp_path) -> None:
    """Evidence is not allowed to abort a robot episode."""

    model, _seen, expected = _native_model(tmp_path, trace=True)
    # A directory where the file should be makes every append fail.
    (tmp_path / "blocked").mkdir()
    model._native_trace_path = tmp_path / "blocked"

    returned = model._infer_chunk(0)

    np.testing.assert_array_equal(returned, expected)
    assert model._native_trace_failures == 1


def test_native_trace_tolerates_a_partially_constructed_model(tmp_path) -> None:
    """Fixtures and older snapshots without the attribute must still infer."""

    model, _seen, expected = _native_model(tmp_path, trace=False)
    del model._native_trace_path
    del model._native_trace_failures

    returned = model._infer_chunk(0)

    np.testing.assert_array_equal(returned, expected)


def test_native_trace_reports_the_step_count_the_flow_head_actually_ran(
    tmp_path,
) -> None:
    """The recorded step count must be the one that ran, not the one requested.

    The native payload carries the legacy ``num_ddim_steps`` from deploy.yml
    (10). ``QwenPI_v3.predict_action`` pops only ``num_inference_steps``, so that
    key is swallowed by ``**kwargs`` and the flow head falls back to the
    checkpoint's ``num_inference_timesteps``, which the server reports as
    ``native_inference_steps`` (4). A trace that echoed the payload would claim
    10 flow steps for a request that ran 4, and would make the B0 arm look
    compute-mismatched against the wrapped arms when it is not.
    """

    model, seen, _expected = _native_model(tmp_path, trace=True, steps=10)

    model._infer_chunk(0)

    payload = seen[0]
    assert payload["num_ddim_steps"] == 10, "payload still carries the legacy key"
    assert "num_inference_steps" not in payload, "native path sends no honoured key"

    row = _native_rows(tmp_path / "runtime_trace.jsonl")[0]
    assert row["metadata"]["inference_steps"] == 4
    assert row["applied_controls"]["inference_steps"] == 4
    assert row["metadata"]["inference_steps_parameter"] == "num_inference_steps"
    assert row["metadata"]["inference_steps_source"] == "checkpoint_default"
    # The discarded request value stays visible rather than being hidden.
    assert row["metadata"]["request_num_ddim_steps"] == 10
    assert row["metadata"]["request_num_ddim_steps_honored"] is False
    # pi_v3 is flow matching, not DDIM, whatever use_ddim says.
    assert row["metadata"]["decoding"] == "flow_matching_action_chunk"


def test_native_trace_uses_the_request_step_count_when_the_server_honors_it(
    tmp_path,
) -> None:
    """A request-level override must be reported as the effective value."""

    model, _seen, _expected = _native_model(tmp_path, trace=True, steps=10)
    model._starvla_server_meta = {
        "framework": "QwenPI_v3",
        "native_inference_steps": 4,
        "ckpt_path": "/models/steps_100000_pytorch_model.pt",
        "default_unnorm_key": "arx_x5",
    }
    original = model.client.predict_action

    class _OverridingClient:
        def predict_action(self, payload):
            payload["num_inference_steps"] = 2
            return original(payload)

    model.client = _OverridingClient()

    model._infer_chunk(0)

    row = _native_rows(tmp_path / "runtime_trace.jsonl")[0]
    assert row["metadata"]["inference_steps"] == 2
    assert row["metadata"]["inference_steps_source"] == "request"


def test_native_trace_falls_back_to_the_legacy_key_for_non_flow_frameworks(
    tmp_path,
) -> None:
    """A DDIM framework does honour num_ddim_steps, so report that value."""

    model, _seen, _expected = _native_model(tmp_path, trace=True, steps=10)
    model._starvla_server_meta = {
        "framework": "QwenPI_v2",
        "native_inference_steps": 4,
        "ckpt_path": "/models/steps_100000_pytorch_model.pt",
        "default_unnorm_key": "arx_x5",
    }

    model._infer_chunk(0)

    row = _native_rows(tmp_path / "runtime_trace.jsonl")[0]
    assert row["metadata"]["inference_steps"] == 10
    assert row["metadata"]["inference_steps_parameter"] == "num_ddim_steps"
    assert row["metadata"]["inference_steps_source"] == "request"
    assert row["metadata"]["request_num_ddim_steps_honored"] is True
    assert row["metadata"]["decoding"] == "ddim_action_chunk"


def test_native_trace_records_the_reset_generation_that_delimits_episodes(
    tmp_path,
) -> None:
    """episode_id is the env index, so episodes need the reset generation.

    Every single-env RoboDojo evaluation reports ``episode_id == 0`` for all
    requests, in the native trace and in the frozen wrapped traces alike. Without
    the reset generation, rows from episode 1 and episode 2 can only be split by
    guessing where ``timestep`` decreases.
    """

    model, _seen, _expected = _native_model(tmp_path, trace=True)
    model._reset_generation = 1
    model._infer_chunk(0)

    # Second episode: the audited reset bumps the generation.
    model._reset_generation = 2
    model.step_by_env = {0: 0}
    model._infer_chunk(0)

    rows = _native_rows(tmp_path / "runtime_trace.jsonl")
    assert [row["reset_generation"] for row in rows] == [1, 2]
    # env index stays 0, which is exactly why the generation is needed.
    assert [row["episode_id"] for row in rows] == [0, 0]


# ---------------------------------------------------------------------------
# Bounded real-payload capture. Cross-process inputs never align, so comparing
# the native and wrapped paths on identical input requires replaying saved
# payloads inside one process. The capture must be faithful enough that a replay
# re-derives the recorded input digest, and bounded enough not to fill a disk.
# ---------------------------------------------------------------------------


def _capturing_model(tmp_path, *, max_payloads: int = 16):
    model, seen, expected = _native_model(tmp_path, trace=True)
    model._native_capture_dir = tmp_path / "captured_payloads"
    model._native_capture_dir.mkdir(parents=True, exist_ok=True)
    model._native_capture_max = max_payloads
    model._native_capture_count = 0
    return model, seen, expected


def _capture_manifest(tmp_path):
    path = tmp_path / "captured_payloads/payload_manifest.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_payload_capture_is_off_by_default(tmp_path) -> None:
    """Without a capture directory nothing is written to disk."""

    model, _seen, expected = _native_model(tmp_path, trace=True)

    returned = model._infer_chunk(0)

    np.testing.assert_array_equal(returned, expected)
    assert _native_rows(tmp_path / "runtime_trace.jsonl")  # trace still written
    assert not (tmp_path / "captured_payloads").exists()


def test_captured_payload_reproduces_the_recorded_input_digest(tmp_path) -> None:
    """A replay rebuilt from the capture must hash to the recorded input.

    This is the property the whole replay rests on. If a rebuilt payload hashes
    differently, the replay is comparing paths on an observation the robot never
    saw, and any conclusion drawn from it would be wrong.
    """

    from agentic_vla.runtime.adapters.starvla import hash_starvla_request_input

    model, seen, expected = _capturing_model(tmp_path)
    model._infer_chunk(0)

    entries = _capture_manifest(tmp_path)
    assert len(entries) == 1
    entry = entries[0]

    arrays = np.load(tmp_path / "captured_payloads" / entry["arrays"])
    rebuilt = {
        "examples": [
            {
                "image": [arrays[f"image_{view}"] for view in range(entry["image_views"])],
                "lang": entry["lang"],
                "state": arrays["state"],
            }
        ],
        "do_sample": entry["do_sample"],
        "use_ddim": entry["use_ddim"],
        "num_ddim_steps": entry["num_ddim_steps"],
        "unnorm_key": entry["unnorm_key"],
        "action_seed": entry["action_seed"],
    }
    assert hash_starvla_request_input(rebuilt) == entry["input_sha256"]
    # And it must equal the digest of the payload the live client received.
    assert entry["input_sha256"] == hash_starvla_request_input(seen[0])

    # The live output is stored too, so a replay can check it reproduces the run.
    np.testing.assert_array_equal(arrays["actions"], expected)
    assert (
        entry["action_sha256"]
        == hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()
    )
    # Controls the server acts on must survive, or the replay changes compute.
    assert entry["action_seed"] == 101 + 0 + 32
    assert entry["live_inference_steps"] == 4
    assert entry["live_inference_steps_source"] == "checkpoint_default"


def test_payload_capture_stops_at_the_configured_maximum(tmp_path) -> None:
    """Capture writes full observations, so the cap has to be enforced."""

    model, _seen, _expected = _capturing_model(tmp_path, max_payloads=2)

    for step in (0, 16, 32, 48):
        model.step_by_env = {0: step}
        model._infer_chunk(0)

    assert len(_capture_manifest(tmp_path)) == 2
    assert len(list((tmp_path / "captured_payloads").glob("payload_*.npz"))) == 2
    # The trace itself is not capped: every request stays recorded.
    assert len(_native_rows(tmp_path / "runtime_trace.jsonl")) == 4


def test_failed_request_is_traced_but_not_captured(tmp_path) -> None:
    """A failed request has no action to compare, so it is not replay material."""

    model, _seen, _expected = _capturing_model(tmp_path)

    class _FailingClient:
        def predict_action(self, payload):
            return {"ok": False, "error": "boom"}

    model.client = _FailingClient()

    with pytest.raises(RuntimeError):
        model._infer_chunk(0)

    assert len(_native_rows(tmp_path / "runtime_trace.jsonl")) == 1
    assert _capture_manifest(tmp_path) == []


# ---------------------------------------------------------------------------
# Unified semantic-check scheduling (plan item 9.21.2).
#
# The frozen C3 run forced a Monitor-triggered check at step 163 when the
# periodic check had already run at step 160. Three steps apart, both counted
# toward `carve_semantic_stale_checks=2`, and the pair was treated as evidence
# of a long-term stall. A forced check must be able to look immediately, but it
# must not be able to manufacture stall evidence faster than the periodic
# schedule would, and it must never bypass the per-episode budget.
# ---------------------------------------------------------------------------


def test_stall_evidence_limits_default_to_the_checkpoint_schedule() -> None:
    """Unset overrides derive from the checkpoint interval, in one place only."""

    from XPolicyLab.policy.starVLA.model import resolve_stall_evidence_limits

    assert resolve_stall_evidence_limits(160) == (160, 320)
    # The launcher and server pass empty strings for "unset".
    assert resolve_stall_evidence_limits(160, "", "") == (160, 320)
    assert resolve_stall_evidence_limits(160, None, None) == (160, 320)
    # Explicit values win, and 0 explicitly disables a limit.
    assert resolve_stall_evidence_limits(160, 64, 500) == (64, 500)
    assert resolve_stall_evidence_limits(160, 0, 0) == (0, 0)


def _scalar_critic_model(tmp_path: Path, *, value: float = 3.0):
    """A critic-mode model whose verifier always reports the same scalar."""

    model = _model_without_server(mode="agentic", chunks=[])
    model.model_cfg = {"carve_planner_max_calls_per_episode": 3}
    model.obs_by_env = {
        0: {
            "lang": "stack all white bowls",
            "state": np.zeros((1, 14), dtype=np.float32),
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    }
    model.action_chunks_by_env = {0: _chunk(0, 3)}
    model._task_instruction_by_env = {0: "stack all white bowls"}
    model._planner_trace_path = tmp_path / "scalar.jsonl"
    model._safe_stop_envs = set()
    model._semantic_predicate = ScalarVisualPredicate(
        predicate_id="remaining_object_groups",
        question="How many separate bowl groups are visible?",
        comparison="eq",
        target=1,
    )
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": value,
            "confidence": 0.9,
            "evidence": "Bowl groups are visible.",
        }
    )
    model._semantic_checkpoint_steps = 160
    model._semantic_stale_checks_required = 2
    model._semantic_max_recoveries = 1
    model._semantic_min_stall_interval_steps = 160
    model._semantic_observation_validity_steps = 320
    model._semantic_last_check_by_env = {}
    model._semantic_last_counted_step_by_env = {}
    model._semantic_last_value_by_env = {}
    model._semantic_stale_checks_by_env = {}
    model._semantic_calls_by_env = {}
    model._semantic_recoveries_by_env = {}
    model._agent_memory_by_env = {}
    model.carve_recovery_compute_calls = 2
    model._recovery_compute_remaining_by_env = {}
    return model


def _planner_rows(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_incomplete_coverage_cannot_create_scalar_stall_or_recovery(tmp_path):
    from agentic_vla.toolchain import GuardedGroupedVisualVerifier

    model = _scalar_critic_model(tmp_path)
    model._semantic_progress_only = True
    model._visual_critic = GuardedGroupedVisualVerifier(lambda _: {
        "visible": True, "groups": ["two bowls stacked at center"],
        "member_counts": [2], "group_relations": ["stacked"],
        "all_targets_visible": True, "confidence": 0.9,
    }, expected_object_count=3)
    for step in (160, 320, 480):
        model.step_by_env[0] = step
        assert not model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_recoveries_by_env == {}
    assert model._semantic_last_value_by_env == {}
    assert model._semantic_stale_checks_by_env.get(0, 0) == 0
    assert model._semantic_calls_by_env[0] == 3
    assert all(row["report"]["status"] == "inconclusive"
               for row in _planner_rows(model._planner_trace_path))


@pytest.mark.parametrize("value,expected", [(None, 0), ("", 0), (0, 0), ("1", 1), (3, 3)])
def test_semantic_reserve_configuration(value, expected):
    from XPolicyLab.policy.starVLA.runtime_config import resolve_semantic_event_reserve
    assert resolve_semantic_event_reserve(value, 3) == expected


@pytest.mark.parametrize("value", [True, -1, 4, 1.5, "1.5", "no"])
def test_semantic_reserve_rejects_invalid_configuration(value):
    from XPolicyLab.policy.starVLA.runtime_config import resolve_semantic_event_reserve
    with pytest.raises(ValueError, match="semantic event reserve"):
        resolve_semantic_event_reserve(value, 3)


@pytest.mark.parametrize("event_name", ["no_progress", "stall", "stale_action"])
@pytest.mark.parametrize("event_steps", [
    (), tuple(range(79, 83)), tuple(range(159, 164)),
    tuple(range(100, 400)), (10, 30, 50), (159, 160, 319, 320),
])
@pytest.mark.parametrize("budget", [3, 8])
@pytest.mark.parametrize("reserve", [0, 1, 2])
def test_scalar_bridge_and_offline_pre_recovery_schedule_agree(
    tmp_path, event_name, event_steps, budget, reserve,
):
    """Synthetic timing differential, not model or task-success evidence."""
    from agentic_vla.toolchain.trigger_admission import (
        InterventionPolicy, simulate_episode_policy,
    )

    model = _scalar_critic_model(tmp_path)
    model._high_level_agent = None
    model._semantic_role = "critic"
    # Isolate admission timing from recovery, which changes later evidence.
    model._semantic_min_unchanged_steps = 10000
    model._semantic_max_calls_per_episode = budget
    model._semantic_event_reserve_calls = reserve
    model._planner_cooldown_steps = 128
    model._last_planner_step_by_env = {}
    present = set(event_steps)
    streak = 0
    for step in range(600):
        model.step_by_env[0] = step
        streak = streak + 1 if step in present else 0
        if streak:
            model._apply_agentic_progress_check(0, SimpleNamespace(
                event=event_name, evidence={"event_streak": streak},
                to_dict=lambda: {"event": event_name},
            ))
        model._apply_scalar_semantic_checkpoint(0)
    rows = [row for row in _planner_rows(model._planner_trace_path)
            if row.get("role") == "scalar_visual_critic"]
    predicted = simulate_episode_policy(
        episode_index=0, steps=600, success=False, event_steps=event_steps,
        event_name=event_name,
        policy=InterventionPolicy(max_calls_per_episode=budget, min_unchanged_steps=10000,
                                  event_reserve_calls=reserve),
    )
    assert tuple(row["timestep"] for row in rows) == predicted.checks
    assert tuple(row["timestep"] for row in rows if row["counts_toward_stall"]) == predicted.counted_checks
    assert predicted.earliest_recovery_step is None


@pytest.mark.parametrize("event_start", [600, 705])
def test_event_reserve_allows_late_check_but_not_expired_stall(tmp_path, event_start):
    from agentic_vla.toolchain.trigger_admission import InterventionPolicy, simulate_episode_policy

    model = _scalar_critic_model(tmp_path)
    model._semantic_event_reserve_calls = 1
    model._semantic_min_unchanged_steps = 320
    model._semantic_max_recoveries = 0
    model._high_level_agent = None
    model._semantic_role = "critic"
    model._last_planner_step_by_env = {}
    model._planner_cooldown_steps = 128
    events = tuple(range(event_start, event_start + 4))
    for step in range(800):
        model.step_by_env[0] = step
        if step in events:
            model._apply_agentic_progress_check(0, SimpleNamespace(
                event="no_progress", evidence={"event_streak": step - event_start + 1},
                to_dict=lambda: {"event": "no_progress"},
            ))
        model._apply_scalar_semantic_checkpoint(0)
    rows = [r for r in _planner_rows(model._planner_trace_path)
            if r.get("role") == "scalar_visual_critic"]
    expected = [160, 320, event_start + 1]
    assert [r["timestep"] for r in rows] == expected
    assert [r["call_budget_remaining_after"] for r in rows] == [2, 1, 0]
    simulated = simulate_episode_policy(
        episode_index=0, steps=800, success=False, event_steps=events,
        policy=InterventionPolicy(event_reserve_calls=1),
    )
    assert list(simulated.checks) == expected
    actual_eligible = next((r["timestep"] for r in rows if r["stale_checks"] >= 2
                            and (r["unchanged_span_steps"] or 0) >= 320), None)
    assert simulated.earliest_recovery_step == actual_eligible
    if event_start == 705:
        assert rows[-1]["previous_observation_expired"]
        assert simulated.earliest_recovery_step is None
    else:
        assert simulated.earliest_recovery_step == 601


def test_forced_scalar_calls_consume_cooldown_without_recovery(tmp_path):
    model = _scalar_critic_model(tmp_path)
    model._semantic_min_unchanged_steps = 10000
    model._planner_cooldown_steps = 128
    for step in (80, 81, 82, 207, 208):
        model.step_by_env[0] = step
        model._apply_scalar_semantic_checkpoint(0, force=True)
    rows = _planner_rows(model._planner_trace_path)
    assert [row["timestep"] for row in rows] == [80, 208]
    assert model._last_planner_step_by_env[0] == 208
    assert model._semantic_recoveries_by_env == {}


def test_forced_check_at_periodic_slot_does_not_repeat_next_step(tmp_path):
    model = _scalar_critic_model(tmp_path)
    model.step_by_env[0] = 160
    model._apply_scalar_semantic_checkpoint(0, force=True)
    model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env[0] = 161
    model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_last_periodic_step_by_env[0] == 160
    assert model._semantic_calls_by_env[0] == 1


def test_missing_frames_do_not_consume_semantic_schedule(tmp_path):
    model = _scalar_critic_model(tmp_path)
    model._semantic_frames = lambda *args, **kwargs: []
    model.step_by_env[0] = 160
    model._apply_scalar_semantic_checkpoint(0, force=True)
    assert model._semantic_last_periodic_step_by_env == {}
    assert getattr(model, "_last_planner_step_by_env", {}) == {}
    assert model._semantic_calls_by_env == {}


@pytest.mark.parametrize(
    "intent,stop_authority,expected_action",
    [
        ("safe_stop", True, "semantic_planner_safe_stop_admitted"),
        ("safe_stop", False, "semantic_planner_safe_stop_rejected_no_authority"),
        ("continue", False, "semantic_planner_continue_original"),
        ("vla_act", False, "semantic_subgoal_blocked_task_only_replan"),
    ],
)
def test_scalar_planner_intent_controls_recovery_dispatch(
    tmp_path, intent, stop_authority, expected_action,
):
    model = _scalar_critic_model(tmp_path)
    model._semantic_recovery_mode = "planner"
    model._semantic_safe_stop_enabled = stop_authority
    model._vla_instruction_capability = "task_only"
    model._task_plan_by_env = {}
    model._last_planner_step_by_env = {}
    model._high_level_agent = GuardedHighLevelAgent(lambda _: {
        "intent": intent, "rationale": "decision under test", "confidence": 0.95,
        "subgoal": "stack bowls",
        "vla_instruction": "stack a pair of bowls" if intent == "vla_act" else None,
        "skill_id": None, "skill_args": {},
        "expected_outcome": "fewer separate bowls", "memory_note": "",
        "failure_type": "no_progress", "scene_graph_update": {}, "proposed_plan": [],
    })
    chunk = model.action_chunks_by_env[0]
    model.step_by_env[0] = 160
    assert not model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env[0] = 320
    changed = model._apply_scalar_semantic_checkpoint(0)
    row = _planner_rows(model._planner_trace_path)[-1]
    assert row["action"] == expected_action
    assert (0 in model._safe_stop_envs) == (intent == "safe_stop" and stop_authority)
    assert model.obs_by_env[0]["lang"] == "stack all white bowls"
    if intent == "vla_act":
        assert changed
        assert model._semantic_recoveries_by_env[0] == 1
        assert model._recovery_compute_remaining_by_env[0] == 2
        assert 0 not in model.action_chunks_by_env
    else:
        assert not changed
        assert model._semantic_recoveries_by_env.get(0, 0) == 0
        assert model._recovery_compute_remaining_by_env == {}
        assert model.action_chunks_by_env[0] is chunk


@pytest.mark.parametrize("stage", ["risk", "task_plan", "scalar"])
def test_stop_decided_during_checkpoint_holds_before_next_action(tmp_path, stage):
    model = _scalar_critic_model(tmp_path)
    model.step_by_env[0] = 0
    calls = []

    def checkpoint(name, env):
        calls.append(name)
        if name == stage:
            model._safe_stop_envs.add(env)

    model._record_shadow_risk = lambda *_: SimpleNamespace(event="no_progress")
    model._apply_agentic_progress_check = lambda env, _: checkpoint("risk", env)
    model._apply_task_plan_checkpoint = lambda env: checkpoint("task_plan", env)
    model._apply_scalar_semantic_checkpoint = lambda env: checkpoint("scalar", env)
    model.obs_by_env[0]["state"] = np.full((1, 14), 0.25, dtype=np.float32)
    model.action_chunks_by_env[0] = np.full((3, 14), 7.0, dtype=np.float32)
    action = model._next_action_vector(0)
    np.testing.assert_array_equal(action, model.obs_by_env[0]["state"][0])
    assert model.step_by_env[0] == 1
    assert calls == ["risk", "task_plan", "scalar"][:["risk", "task_plan", "scalar"].index(stage) + 1]


def test_forced_check_too_close_cannot_manufacture_stall_evidence(
    tmp_path: Path,
) -> None:
    """The exact 160/163 pattern from the frozen C3 run must not recover."""

    model = _scalar_critic_model(tmp_path)

    model.step_by_env = {0: 160}
    assert not model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_stale_checks_by_env[0] == 1

    # Monitor forces a check three steps later.
    model.step_by_env = {0: 163}
    assert not model._apply_scalar_semantic_checkpoint(
        0, trigger="motion_no_progress", force=True
    )
    # The streak is held, not advanced: 3 steps is not a long-term stall.
    assert model._semantic_stale_checks_by_env[0] == 1
    assert model._semantic_recoveries_by_env.get(0, 0) == 0
    assert 0 in model.action_chunks_by_env

    rows = _planner_rows(tmp_path / "scalar.jsonl")
    assert len(rows) == 2
    forced = rows[1]
    assert forced["forced"] is True
    assert forced["counts_toward_stall"] is False
    assert forced["steps_since_counted_observation"] == 3
    assert forced["min_stall_interval_steps"] == 160

    # A properly spaced observation still advances the streak and recovers, so
    # the limit tightens the evidence rule without disabling recovery.
    model.step_by_env = {0: 320}
    assert model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_recoveries_by_env[0] == 1


def test_forced_check_still_observes_and_records(tmp_path: Path) -> None:
    """Tightening the evidence rule must not silence the observation."""

    model = _scalar_critic_model(tmp_path)
    model.step_by_env = {0: 160}
    model._apply_scalar_semantic_checkpoint(0)

    model.step_by_env = {0: 163}
    model._apply_scalar_semantic_checkpoint(
        0, trigger="motion_no_progress", force=True
    )

    rows = _planner_rows(tmp_path / "scalar.jsonl")
    # Two calls happened, both traced, and the budget counted both.
    assert len(rows) == 2
    assert model._semantic_calls_by_env[0] == 2
    # The freshest reading is still stored even though it did not count.
    assert model._semantic_last_value_by_env[0] == 3.0
    assert model._semantic_last_check_by_env[0] == 163
    # ... but the last *counted* observation is still the periodic one.
    assert model._semantic_last_counted_step_by_env[0] == 160


def test_forced_check_cannot_exceed_the_per_episode_budget(tmp_path: Path) -> None:
    """force must not be a way around the call budget."""

    model = _scalar_critic_model(tmp_path)
    model._semantic_max_calls_per_episode = 2

    for step in (160, 320):
        model.step_by_env = {0: step}
        model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_calls_by_env[0] == 2

    model.step_by_env = {0: 480}
    assert not model._apply_scalar_semantic_checkpoint(
        0, trigger="motion_no_progress", force=True
    )
    # Budget exhausted: no third call, forced or not.
    assert model._semantic_calls_by_env[0] == 2
    assert len(_planner_rows(tmp_path / "scalar.jsonl")) == 2


def test_expired_previous_observation_restarts_the_streak(tmp_path: Path) -> None:
    """An unchanged value across a long gap is not continuous-stall evidence."""

    model = _scalar_critic_model(tmp_path)

    model.step_by_env = {0: 160}
    model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_stale_checks_by_env[0] == 1

    # Validity window is 320 steps; this observation is 400 steps later, so the
    # previous reading can no longer evidence an uninterrupted stall.
    model.step_by_env = {0: 560}
    assert not model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_stale_checks_by_env[0] == 1
    assert model._semantic_recoveries_by_env.get(0, 0) == 0

    rows = _planner_rows(tmp_path / "scalar.jsonl")
    assert rows[-1]["previous_observation_expired"] is True
    assert rows[-1]["counts_toward_stall"] is True
    assert rows[-1]["observation_validity_steps"] == 320


def test_undecided_observation_never_contributes_to_recovery(tmp_path: Path) -> None:
    """unknown stays unknown; it must not be rounded into stall evidence."""

    model = _scalar_critic_model(tmp_path)
    # A verifier that cannot see the scene yields no usable value.
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": False,
            "value": None,
            "confidence": 0.1,
            "evidence": "The bowls are occluded.",
        }
    )

    for step in (160, 320, 480):
        model.step_by_env = {0: step}
        assert not model._apply_scalar_semantic_checkpoint(0)

    assert model._semantic_stale_checks_by_env[0] == 0
    assert model._semantic_recoveries_by_env.get(0, 0) == 0
    assert 0 in model.action_chunks_by_env


def test_forced_check_does_not_delay_the_periodic_schedule(tmp_path: Path) -> None:
    """A forced look that did not count must not consume the periodic slot.

    Keying the periodic schedule to "any check" let a Monitor-forced check at
    step 163 push the next scheduled check from 320 out to 323, delaying the
    next legitimate evidence point. The schedule follows the last periodic
    check instead.
    """

    model = _scalar_critic_model(tmp_path)

    model.step_by_env = {0: 160}
    assert not model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_last_counted_step_by_env[0] == 160

    model.step_by_env = {0: 163}
    model._apply_scalar_semantic_checkpoint(
        0, trigger="motion_no_progress", force=True
    )
    assert model._semantic_last_check_by_env[0] == 163
    assert model._semantic_last_counted_step_by_env[0] == 160

    # Exactly one checkpoint period after the last counted observation, the
    # periodic check runs -- it is not pushed back by the forced look.
    model.step_by_env = {0: 320}
    assert model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_recoveries_by_env[0] == 1


# ---------------------------------------------------------------------------
# Stall evidence as a span of steps rather than a count of checks.
#
# The deployed predicate is a coarse integer (remaining bowl groups, target 1),
# so it changes only a couple of times per episode. "Unchanged across two
# checks" is therefore weak evidence on its own, and it gets weaker the more
# often you look: doubling the check rate halved the time to declare a stall.
# The criterion is now frequency-invariant.
# ---------------------------------------------------------------------------


def test_min_unchanged_steps_default_is_stricter_than_check_counting() -> None:
    """The default deliberately requires more than two nominal readings.

    It is a policy change, not an equivalent rewrite: the old check-counting rule
    fired at a 160-step span, so this default must be justified by its own
    closed-loop evidence rather than by "behaviour unchanged".
    """

    from XPolicyLab.policy.starVLA.model import resolve_min_unchanged_steps

    # Two readings span 160 steps; the new 320-step default needs a third.
    assert resolve_min_unchanged_steps(160, 2) == 320
    assert resolve_min_unchanged_steps(160, 3) == 480
    # Launcher and server pass empty strings for "unset".
    assert resolve_min_unchanged_steps(160, 2, "") == 320
    assert resolve_min_unchanged_steps(160, 2, None) == 320
    # Explicit override wins; 0 restores pure check counting.
    assert resolve_min_unchanged_steps(160, 2, 500) == 500
    assert resolve_min_unchanged_steps(160, 2, 0) == 0


def test_frequent_checking_no_longer_shortens_time_to_stall(tmp_path: Path) -> None:
    """Checking twice as often must not make a stall appear twice as fast.

    This is the property the span criterion exists for. With a 320-step span
    required, two checks 160 steps apart are not enough on their own; the span
    has to actually elapse.
    """

    model = _scalar_critic_model(tmp_path)
    # Check twice as often as nominal, but keep the required span at the nominal
    # 320: the only thing under test is that frequency cannot substitute for it.
    model._semantic_checkpoint_steps = 80
    model._semantic_min_stall_interval_steps = 80
    model._semantic_observation_validity_steps = 0
    model._semantic_min_unchanged_steps = 320
    model._semantic_max_calls_per_episode = 6

    fired_at = []
    for step in (80, 160, 240, 320, 400):
        model.step_by_env = {0: step}
        if model._apply_scalar_semantic_checkpoint(0):
            fired_at.append(step)

    # The streak requirement is satisfied from step 160, but the span is not
    # reached until step 400, so that is when recovery may fire.
    assert fired_at == [400], fired_at
    rows = _planner_rows(tmp_path / "scalar.jsonl")
    assert [row["unchanged_span_steps"] for row in rows] == [0, 80, 160, 240, 320]
    # The row records the evidence level that fired, not the post-reset value.
    assert [row["stale_checks"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(row["min_unchanged_steps"] == 320 for row in rows)
    # Streak requirement alone was met from step 160 onward; only the span held
    # recovery back, which is precisely the frequency-invariance property.
    assert model._semantic_recoveries_by_env[0] == 1
    assert model._semantic_stale_checks_by_env[0] == 0


def test_a_changed_value_restarts_the_unchanged_span(tmp_path: Path) -> None:
    """Real progress must reset the span, not merely the streak.

    In the frozen C3 run the predicate went 2.0 -> 2.0 -> 1.0: the value did
    change, so the earlier stretch must not still count as a stall.
    """

    values = iter([3.0, 3.0, 2.0, 2.0])
    model = _scalar_critic_model(tmp_path)
    model._semantic_min_unchanged_steps = 320
    model._semantic_min_stall_interval_steps = 0
    model._semantic_observation_validity_steps = 0
    model._semantic_max_calls_per_episode = 6
    model._visual_critic = GuardedScalarVisualVerifier(
        lambda _request: {
            "predicate_id": "remaining_object_groups",
            "visible": True,
            "value": next(values),
            "confidence": 0.9,
            "evidence": "Bowl groups are visible.",
        }
    )

    for step in (160, 320, 480, 640):
        model.step_by_env = {0: step}
        model._apply_scalar_semantic_checkpoint(0)

    rows = _planner_rows(tmp_path / "scalar.jsonl")
    # 160: first reading, span 0. 320: unchanged at 3.0, span 160.
    # 480: value changed 3.0 -> 2.0, span restarts. 640: span 160 again.
    assert [row["unchanged_span_steps"] for row in rows] == [0, 160, 0, 160]
    # No recovery: the span never reached 320 without the value changing.
    assert model._semantic_recoveries_by_env.get(0, 0) == 0


def test_span_criterion_still_permits_a_genuine_long_stall(tmp_path: Path) -> None:
    """Tightening must not disable recovery for a real, sustained stall."""

    model = _scalar_critic_model(tmp_path)
    model._semantic_min_unchanged_steps = 320

    for step in (160, 320, 480):
        model.step_by_env = {0: step}
        fired = model._apply_scalar_semantic_checkpoint(0)
        if step == 480:
            assert fired, "a 320-step unchanged span must still recover"
    assert model._semantic_recoveries_by_env[0] == 1


def test_unchanged_span_is_reset_state_not_leaked_across_episodes() -> None:
    """A leaked span would let episode N+1 inherit N's apparent stall."""

    assert "semantic_unchanged_since_step_by_env" in STARVLA_RESET_STATE_FIELDS


def test_a_non_counting_check_does_not_reopen_the_periodic_gate(
    tmp_path: Path,
) -> None:
    """Scheduling and evidence-counting need separate references.

    Keying the periodic gate to the last *counted* observation left the gate open
    on every following step whenever a check did not count, so the per-episode
    call budget was spent on consecutive steps. Reachable whenever the minimum
    stall interval exceeds the checkpoint interval; found by the offline policy
    feasibility audit, not by a robot run.
    """

    model = _scalar_critic_model(tmp_path)
    model._semantic_checkpoint_steps = 80
    # Deliberately larger than the checkpoint interval: this is the combination
    # that exposed the defect.
    model._semantic_min_stall_interval_steps = 160
    model._semantic_observation_validity_steps = 0
    model._semantic_min_unchanged_steps = 0
    model._semantic_max_calls_per_episode = 5

    ran_at = []
    for step in range(400):
        model.step_by_env = {0: step}
        before = model._semantic_calls_by_env.get(0, 0)
        model._apply_scalar_semantic_checkpoint(0)
        if model._semantic_calls_by_env.get(0, 0) > before:
            ran_at.append(step)

    # One check per checkpoint interval, never two in adjacent steps.
    assert ran_at == [80, 160, 240, 320], ran_at
    assert all(b - a >= 80 for a, b in zip(ran_at, ran_at[1:]))
    # Counting still obeys the larger minimum stall interval.
    assert model._semantic_last_counted_step_by_env[0] == 240
    assert model._semantic_last_periodic_step_by_env[0] == 320


def test_a_forced_check_still_does_not_advance_the_periodic_slot(
    tmp_path: Path,
) -> None:
    """The 9.24 property must survive the scheduling fix."""

    model = _scalar_critic_model(tmp_path)

    model.step_by_env = {0: 160}
    assert not model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_last_periodic_step_by_env[0] == 160

    model.step_by_env = {0: 163}
    model._apply_scalar_semantic_checkpoint(
        0, trigger="motion_no_progress", force=True
    )
    # A forced look does not consume or shift the periodic slot.
    assert model._semantic_last_periodic_step_by_env[0] == 160

    model.step_by_env = {0: 320}
    assert model._apply_scalar_semantic_checkpoint(0)
    assert model._semantic_recoveries_by_env[0] == 1


def test_periodic_slot_is_reset_state() -> None:
    """A leaked slot would let episode N+1 skip its first scheduled check."""

    assert "semantic_last_periodic_step_by_env" in STARVLA_RESET_STATE_FIELDS


@pytest.mark.parametrize("interruption", [None, 1.0])
def test_semantic_interruption_breaks_the_entire_stall_span(tmp_path, interruption):
    model = _scalar_critic_model(tmp_path)
    model._semantic_min_unchanged_steps = 320
    model._semantic_max_calls_per_episode = 8
    values = iter([3.0, interruption, 3.0, 3.0, 3.0])

    def observe(_request):
        value = next(values)
        return {
            "predicate_id": "remaining_object_groups", "visible": value is not None,
            "value": value, "confidence": 0.9 if value is not None else 0.1,
            "evidence": "Visible bowl groups." if value is not None else "Occluded.",
        }

    model._visual_critic = GuardedScalarVisualVerifier(observe)
    fired = []
    for step in (160, 320, 480, 640, 800):
        model.step_by_env = {0: step}
        if model._apply_scalar_semantic_checkpoint(0):
            fired.append(step)
        if step == 320:
            assert 0 not in model._semantic_last_value_by_env
            assert 0 not in model._semantic_unchanged_since_step_by_env
    assert fired == [800]
    rows = _planner_rows(tmp_path / "scalar.jsonl")
    assert [row["unchanged_span_steps"] for row in rows] == [0, None, 0, 160, 320]
    assert 0 not in model._semantic_unchanged_since_step_by_env


@pytest.mark.parametrize("step", [160, 159])
def test_duplicate_or_older_forced_check_does_not_spend_budget(tmp_path, step):
    model = _scalar_critic_model(tmp_path)
    model.step_by_env = {0: 160}
    assert not model._apply_scalar_semantic_checkpoint(0)
    model.step_by_env = {0: step}
    assert not model._apply_scalar_semantic_checkpoint(0, force=True)
    assert model._semantic_calls_by_env[0] == 1
    assert model._semantic_last_check_by_env[0] == 160
    assert len(_planner_rows(tmp_path / "scalar.jsonl")) == 1


def test_progress_in_noncounting_check_breaks_the_old_streak(tmp_path):
    model = _scalar_critic_model(tmp_path)
    model._semantic_max_calls_per_episode = 5
    values = iter([3.0, 2.0, 2.0, 2.0])
    model._visual_critic = GuardedScalarVisualVerifier(lambda _request: {
        "predicate_id": "remaining_object_groups", "visible": True,
        "value": next(values), "confidence": 0.9, "evidence": "Visible bowl groups.",
    })
    for step, force in ((160, False), (163, True), (320, False)):
        model.step_by_env = {0: step}
        assert not model._apply_scalar_semantic_checkpoint(0, force=force)
        if step == 163:
            assert model._semantic_stale_checks_by_env[0] == 0
    model.step_by_env = {0: 480}
    assert model._apply_scalar_semantic_checkpoint(0)
