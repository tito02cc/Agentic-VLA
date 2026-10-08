"""CPU adapter contracts only; these tests do not measure task success."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def port_setup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party/robodojo_official"))
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state
    from agentic_vla.benchmarks.robodojo_pi05_port import XPolicyLabPi05Port
    dims = get_robot_action_dim_info("arx_x5")
    class Environment:
        def __init__(self):
            self.reads = 0
            self.actions = []
            self.fail = False
        def get_obs(self):
            self.reads += 1
            return {"instruction": "organize the table", "state": unpack_robot_state(np.zeros(14), "joint", dims),
                    "vision": {k: np.full((4, 4, 3), self.reads, np.uint8) for k in
                               ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
        def take_action(self, action):
            self.actions.append(action)
            if self.fail:
                raise RuntimeError("partial simulator failure")
        def is_episode_end(self):
            return False
    payloads = []
    def call(func_name, **kwargs):
        if func_name == "update_obs":
            payloads.append(kwargs["obs"])
        else:
            return unpack_robot_state(np.ones((50, 14), np.float32), "joint", dims)
    env = Environment()
    return XPolicyLabPi05Port(env, SimpleNamespace(call=call)), env, payloads


def test_same_control_step_shares_immutable_snapshot(port_setup):
    port, env, payloads = port_setup
    first = port.observe()
    actions = port.infer(first.instruction)
    assert port.observe() is first
    assert env.reads == 1
    assert payloads[0]["images"]["cam_high"] is first.frames["cam_high"]
    assert actions.shape == (50, 14)
    with pytest.raises(ValueError):
        first.state[0] = 1
    with pytest.raises(ValueError):
        first.frames["cam_high"][0, 0, 0] = 5
    port.execute(actions[0])
    assert port.observe() is not first
    assert env.reads == 2
    assert first.frames["cam_high"][0, 0, 0] == 1


def test_partial_action_failure_invalidates_observation(port_setup):
    port, env, _ = port_setup
    first = port.observe()
    env.fail = True
    with pytest.raises(RuntimeError):
        port.execute(np.zeros(14))
    assert port.observe() is not first


def test_rejected_action_does_not_touch_simulation(port_setup):
    port, env, _ = port_setup
    first = port.observe()
    with pytest.raises(ValueError):
        port.execute(np.full(14, np.nan))
    assert port.observe() is first
    assert env.actions == []


def test_reference_audit_preserves_actions_and_records_inputs(tmp_path):
    import json
    from agentic_vla.benchmarks.robodojo_pi05_policy import ReferenceRuntimePolicy, array_fingerprint
    actions = np.arange(700, dtype=np.float32).reshape(50, 14)
    class Policy:
        def infer(self, observation):
            return {"actions": actions.copy()}
    trace = tmp_path / "trace.jsonl"
    policy = ReferenceRuntimePolicy(Policy(), trace)
    observation = {"prompt": "organize", "state": np.zeros(14, np.float32),
                   "images": {"cam_high": np.zeros((3, 4, 4), np.uint8)}}
    assert np.array_equal(policy.infer(observation)["actions"], actions)
    row = json.loads(trace.read_text())
    assert row["input_audit"]["state"] == array_fingerprint(observation["state"])
    assert row["input_audit"]["rng_before"] is None
    assert np.array_equal(np.load(tmp_path / row["actions_file"]), actions)
    with np.load(tmp_path / row["input_audit"]["input_file"]) as data:
        assert np.array_equal(data["cam_high"], observation["images"]["cam_high"])
