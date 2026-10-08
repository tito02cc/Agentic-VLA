import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def admission(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pi05_admission", scripts / "run_robodojo_pi05_admission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubPolicy:
    def __init__(self, actions):
        self.actions = actions
        self._sample_kwargs = {}
        self.calls = []

    def infer(self, observation):
        self.calls.append(observation)
        return {"actions": self.actions.copy(), "policy_timing": {"infer_ms": 1.0}}


def observation():
    return {"state": np.zeros(14), "prompt": "test instruction",
            "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)}}


@pytest.mark.parametrize("mode", ["native", "runtime"])
def test_passthrough_no_control_or_input_change(admission, tmp_path, mode):
    actions = np.zeros((50, 14), dtype=np.float32)
    policy = StubPolicy(actions)
    recorded = admission.RecordedPolicy(policy, tmp_path / "trace.jsonl", mode=mode, action_dim=14)
    obs = observation()
    output = recorded.infer(obs)
    np.testing.assert_array_equal(output["actions"], actions)
    assert len(policy.calls) == 1
    assert policy.calls[0].keys() == obs.keys()
    assert policy.calls[0]["prompt"] == obs["prompt"]
    assert policy._sample_kwargs == {}
    trace = json.loads((tmp_path / "trace.jsonl").read_text())
    assert trace["phase"] == "rollout"
    assert trace["actions_shape"] == [50, 14]
    assert trace["agent_enabled"] is False


@pytest.mark.parametrize("shape", [(16, 14), (50, 7)])
def test_wrong_shape_rejected(admission, tmp_path, shape):
    policy = StubPolicy(np.zeros(shape))
    recorded = admission.RecordedPolicy(policy, tmp_path / "trace.jsonl", mode="native", action_dim=14)
    with pytest.raises(ValueError):
        recorded.infer(observation())


def test_nonfinite_actions_rejected(admission, tmp_path):
    policy = StubPolicy(np.full((50, 14), np.nan))
    recorded = admission.RecordedPolicy(policy, tmp_path / "trace.jsonl", mode="runtime", action_dim=14)
    with pytest.raises(ValueError, match="non-finite"):
        recorded.infer(observation())


def test_warmup_is_not_rollout(admission, tmp_path):
    policy = StubPolicy(np.zeros((50, 14)))
    recorded = admission.RecordedPolicy(policy, tmp_path / "trace.jsonl", mode="native", action_dim=14)
    recorded.phase = "warmup_and_wrapper_check"
    recorded.infer(observation())
    assert json.loads((tmp_path / "trace.jsonl").read_text())["phase"] != "rollout"
    assert not (tmp_path / "initial_observation.npz").exists()


def test_initial_capture_contains_only_public_model_input(admission, tmp_path):
    recorded = admission.RecordedPolicy(StubPolicy(np.zeros((50, 14))),
                                        tmp_path / "trace.jsonl", mode="native", action_dim=14)
    obs = {**observation(), "target_label": "not-for-planner", "reward": 100}
    recorded.infer(obs)
    with np.load(tmp_path / "initial_observation.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"state", "instruction", "cam_high"}
        assert saved["instruction"].item() == obs["prompt"]
        np.testing.assert_array_equal(saved["cam_high"], obs["images"]["cam_high"])


def test_initial_capture_does_not_overwrite_prior_evidence(admission, tmp_path):
    path = tmp_path / "initial_observation.npz"
    np.savez_compressed(path, marker=np.asarray("previous episode"))
    recorded = admission.RecordedPolicy(StubPolicy(np.zeros((50, 14))),
                                        tmp_path / "trace.jsonl", mode="native", action_dim=14)
    with pytest.raises(FileExistsError, match="overwrite"):
        recorded.infer(observation())
    with np.load(path, allow_pickle=False) as saved:
        assert saved["marker"].item() == "previous episode"


def test_manual_subgoal_is_rollout_only_and_preserves_environment_instruction(admission, tmp_path):
    policy = StubPolicy(np.zeros((50, 14)))
    recorded = admission.RecordedPolicy(policy, tmp_path / "trace.jsonl", mode="native",
                                        action_dim=14, instruction_override="Put watch objects into the middle basket.")
    obs = observation()
    recorded.phase = "warmup_and_wrapper_check"
    recorded.infer(obs)
    assert policy.calls[-1]["prompt"] == "test instruction"
    recorded.phase, recorded.index = "rollout", 0
    recorded.infer(obs)
    assert obs["prompt"] == "test instruction"
    assert policy.calls[-1]["prompt"] == recorded.instruction_override
    row = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert row["environment_instruction"] == "test instruction"
    assert row["instruction_override"] == recorded.instruction_override
    assert row["agent_enabled"] is False
