from __future__ import annotations

import numpy as np

from agentic_vla.benchmarks import (
    RoboTwinRecoverySkillLibrary,
    RoboTwinRuntimeAdapter,
)
from agentic_vla.toolchain import ToolExecutionContext


class FakeRoboTwinEnv:
    def __init__(self) -> None:
        self.steps = 0
        self.closed = False
        self.last_action = None
        self._planned_pose = None

    def _info(self):
        return {
            "robot_state": {
                "qpos_target14": np.arange(14, dtype=np.float32) + self.steps,
                "left_eef_pose": np.array([0.2, 0.1, 0.3, 1, 0, 0, 0]),
                "right_eef_pose": (
                    np.array([0.2, -0.1, 0.3, 1, 0, 0, 0])
                    if self._planned_pose is None
                    else self._planned_pose
                ),
            },
            "episode_status": {
                "eval_success": self.steps >= 2,
                "take_action_cnt": self.steps,
                "step_lim": 20,
                "actual_seed": 100000,
            },
        }

    def reset(self):
        self.steps = 0
        return {}, self._info()

    def step(self, action, *, action_type="qpos"):
        assert action_type == "qpos"
        self.last_action = np.asarray(action).copy()
        self.steps += 1
        success = self.steps >= 2
        return {}, float(success), success, False, self._info()

    def render_camera(self, name):
        values = {"head": 10, "left_wrist": 20, "right_wrist": 30}
        return np.full((8, 12, 3), values[name], dtype=np.uint8)

    def get_task_language(self):
        return "open the microwave"

    def plan_arm_path(self, arm, target_pose):
        assert arm in {"left", "right"}
        self._planned_pose = np.asarray(target_pose).copy()
        return {"status": "Success", "position": np.zeros((3, 6))}

    def close(self):
        self.closed = True


def _context(timestep: int) -> ToolExecutionContext:
    return ToolExecutionContext(
        episode_id="robotwin-episode",
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=("vla_act",),
        deployment_profile_id="lingbot-eager-bf16-d10-c50",
    )


def test_robotwin_adapter_builds_public_lingbot_observation() -> None:
    adapter = RoboTwinRuntimeAdapter(FakeRoboTwinEnv(), seed=100000)

    observation = adapter.reset()

    assert observation.timestep == 0
    assert observation.policy_observation["task"] == "open the microwave"
    assert observation.policy_observation["observation.state"].shape == (14,)
    assert set(observation.planner_frames) == {
        "head",
        "left_wrist",
        "right_wrist",
    }
    assert observation.frame_id == "robotwin:100000:0"


def test_robotwin_adapter_executes_qpos_chunk_and_keeps_evaluator_private() -> None:
    env = FakeRoboTwinEnv()
    adapter = RoboTwinRuntimeAdapter(env, seed=100000, max_steps=10)
    adapter.reset()
    observed_steps = []

    report = adapter.execute_action_chunk(
        np.ones((4, 14), dtype=np.float32),
        _context(0),
        on_step=lambda observation, _action: observed_steps.append(
            observation.timestep
        ),
    )

    assert report.ended_timestep == 2
    assert report.metadata["executed_steps"] == 2
    assert observed_steps == [1, 2]
    private = adapter.private_result()
    assert private.task_success is True
    assert private.episode_steps == 2
    assert "eval_success" not in adapter.observe().policy_observation


def test_robotwin_adapter_rejects_stale_context_and_closes() -> None:
    env = FakeRoboTwinEnv()
    adapter = RoboTwinRuntimeAdapter(env, seed=100000)
    adapter.reset()

    try:
        adapter.execute_action_chunk(np.ones((1, 14)), _context(1))
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale context should be rejected")

    adapter.close()
    assert env.closed is True


def test_robotwin_qpos_recovery_uses_public_path_planner() -> None:
    env = FakeRoboTwinEnv()
    adapter = RoboTwinRuntimeAdapter(env, seed=100000, max_steps=10)
    adapter.reset()
    library = RoboTwinRecoverySkillLibrary(
        adapter,
        last_action_source=lambda: np.zeros(14, dtype=np.float32),
    )

    result = library.execute(
        "qpos_retract_lift_reobserve",
        {},
        _context(0),
    )

    assert result.status.value == "succeeded"
    assert result.metadata["arm"] == "right"
    assert result.metadata["plan_status"] == "Success"
    assert result.metadata["physical_response_m"] >= 0.079
