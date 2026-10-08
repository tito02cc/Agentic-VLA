from __future__ import annotations

import numpy as np
import pytest

from agentic_vla.benchmarks.robomme_runtime import (
    RoboMMERuntimeAdapter,
    pack_robomme_state,
)
from agentic_vla.toolchain import ToolExecutionContext


def _observation(value: int = 0):
    return {
        "front_rgb_list": [np.full((8, 8, 3), value, dtype=np.uint8)],
        "wrist_rgb_list": [np.full((8, 8, 3), value + 1, dtype=np.uint8)],
        "joint_state_list": [np.arange(7, dtype=np.float32)],
        "gripper_state_list": [np.asarray([0.5, 0.5], dtype=np.float32)],
    }


class FakeRoboMMEEnv:
    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self):
        return _observation(), {
            "task_goal": ["move the remembered cube"],
            "simple_subgoal_online": "privileged oracle",
        }

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 2
        return _observation(self.step_count), 0.0, done, False, {
            "status": "success" if done else "unknown",
            "grounded_subgoal_online": "privileged oracle",
        }

    def close(self):
        self.closed = True


def test_pack_robomme_state_matches_official_eight_dimensional_contract():
    state = pack_robomme_state(np.arange(7), np.asarray([0.25, 0.75]))
    assert state.shape == (8,)
    assert state[-1] == pytest.approx(0.25)


def test_adapter_exposes_only_deployable_fields_and_keeps_oracle_private():
    adapter = RoboMMERuntimeAdapter(
        FakeRoboMMEEnv(), task_id="MoveCube", episode_id=0
    )
    observation = adapter.reset()

    assert adapter.task_instruction == "move the remembered cube"
    assert set(observation.policy_observation) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    }
    assert "subgoal" not in repr(observation).lower()
    assert len(adapter.initial_memory["front_rgb"]) == 1


def test_adapter_executes_bounded_chunk_and_reports_private_success():
    adapter = RoboMMERuntimeAdapter(
        FakeRoboMMEEnv(), task_id="MoveCube", episode_id=3
    )
    adapter.reset()
    context = ToolExecutionContext(
        episode_id="MoveCube:3",
        timestep=0,
        at_safe_boundary=True,
        allowed_tools=(),
    )
    actions = np.zeros((4, 8), dtype=np.float32)
    report = adapter.execute_action_chunk(actions, context)

    assert report.metadata["executed_steps"] == 2
    assert adapter.timestep == 2
    assert adapter.private_result().task_success is True


def test_adapter_rejects_stale_context_and_invalid_action_dimension():
    adapter = RoboMMERuntimeAdapter(
        FakeRoboMMEEnv(), task_id="MoveCube", episode_id=0
    )
    adapter.reset()
    stale = ToolExecutionContext(
        episode_id="MoveCube:0",
        timestep=1,
        at_safe_boundary=True,
        allowed_tools=(),
    )
    with pytest.raises(ValueError, match="stale"):
        adapter.execute_action_chunk(np.zeros((1, 8)), stale)

    current = ToolExecutionContext(
        episode_id="MoveCube:0",
        timestep=0,
        at_safe_boundary=True,
        allowed_tools=(),
    )
    with pytest.raises(ValueError, match="8-D"):
        adapter.execute_action_chunk(np.zeros((1, 7)), current)
