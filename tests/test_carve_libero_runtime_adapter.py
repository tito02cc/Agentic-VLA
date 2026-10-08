"""CPU contract tests for the thin official-LIBERO adapter."""

from __future__ import annotations

import numpy as np

from agentic_vla.benchmarks import LiberoRuntimeAdapter
from agentic_vla.toolchain import ToolExecutionContext, core_tool_specs


class FixtureLiberoEnv:
    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    @staticmethod
    def observation():
        return {
            "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((16, 16, 3), dtype=np.uint8),
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
        }

    def reset(self):
        self.steps = 0

    def set_init_state(self, _state):
        return self.observation()

    def regenerate_obs_from_state(self, _state):
        return self.observation()

    def step(self, _action):
        self.steps += 1
        return self.observation(), 0.0, self.steps >= 3, {}

    def close(self):
        self.closed = True


def context(timestep=0):
    return ToolExecutionContext(
        episode_id="episode-1",
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="profile",
    )


def test_libero_adapter_separates_deployable_observation_and_private_result() -> None:
    env = FixtureLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="put the mug on the plate",
        initial_state=np.zeros(3),
        image_transform=lambda image: image[::2, ::2],
        max_steps=10,
    )
    observation = adapter.reset()
    assert observation.policy_observation["observation/image"].shape == (8, 8, 3)
    assert len(observation.robot_state) == 8
    assert "success" not in observation.policy_observation

    report = adapter.execute_action_chunk([[0.0] * 7] * 4, context())
    assert report.status.value == "succeeded"
    assert report.ended_timestep == 3
    assert report.metadata["executed_steps"] == 3
    assert "task_success" not in report.metadata
    assert "episode_terminated" not in report.metadata

    private = adapter.private_result()
    assert private.task_success
    assert private.episode_steps == 3
    adapter.close()
    assert env.closed


def test_libero_adapter_rejects_stale_context() -> None:
    adapter = LiberoRuntimeAdapter(
        FixtureLiberoEnv(),
        task_instruction="task",
        initial_state=np.zeros(3),
    )
    adapter.reset()
    try:
        adapter.execute_action_chunk([[0.0] * 7], context(timestep=2))
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale context must be rejected")


def test_libero_adapter_emits_deployable_observation_for_every_control_step() -> None:
    adapter = LiberoRuntimeAdapter(
        FixtureLiberoEnv(),
        task_instruction="task",
        initial_state=np.zeros(3),
    )
    adapter.reset()
    observed = []

    adapter.execute_action_chunk(
        [[0.1] * 7] * 4,
        context(),
        on_step=lambda observation, action: observed.append(
            (observation.timestep, action.copy(), observation.policy_observation)
        ),
    )

    assert [item[0] for item in observed] == [1, 2, 3]
    assert all(item[1].shape == (7,) for item in observed)
    assert all("success" not in item[2] for item in observed)


def test_libero_adapter_settles_without_exposing_or_counting_evaluator_state() -> None:
    env = FixtureLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="task",
        initial_state=np.zeros(3),
        settle_steps=2,
    )

    observation = adapter.reset()

    assert observation.timestep == 0
    assert adapter.timestep == 0
    assert env.steps == 2
    assert "success" not in observation.policy_observation


def test_libero_adapter_restores_private_simulator_state_as_episode_origin() -> None:
    env = FixtureLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="task",
        initial_state=np.zeros(3),
        restore_state=np.arange(5, dtype=np.float64),
        settle_steps=0,
    )

    observation = adapter.reset()

    assert observation.timestep == 0
    assert adapter.timestep == 0
    assert "sim_state" not in observation.policy_observation
