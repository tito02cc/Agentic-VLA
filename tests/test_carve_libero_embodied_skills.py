"""CPU contracts for CARVE's LIBERO analytic embodied skills."""

from __future__ import annotations

import numpy as np

from agentic_vla.benchmarks import LiberoEmbodiedSkillLibrary, LiberoRuntimeAdapter
from agentic_vla.runtime import HighLevelAgentContext, build_high_level_agent_request
from agentic_vla.toolchain import ToolExecutionContext, core_tool_specs


class DynamicLiberoEnv:
    def __init__(self) -> None:
        self.eef = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self.commands: list[np.ndarray] = []

    def observation(self):
        return {
            "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
            "robot0_eef_pos": self.eef.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
        }

    def reset(self):
        self.commands.clear()

    def set_init_state(self, _state):
        return self.observation()

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        self.commands.append(command.copy())
        self.eef += command[:3] * 0.05
        return self.observation(), 0.0, False, {}


def tool_context(timestep: int = 0) -> ToolExecutionContext:
    return ToolExecutionContext(
        episode_id="skill-episode",
        timestep=timestep,
        at_safe_boundary=True,
        allowed_tools=tuple(spec.name for spec in core_tool_specs()),
        deployment_profile_id="profile",
    )


def test_relative_move_uses_closed_loop_adapter_without_exposing_actions() -> None:
    env = DynamicLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="move safely",
        initial_state=np.zeros(3),
        max_steps=50,
    )
    adapter.reset()
    library = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: [0.0] * 6 + [-1.0],
    )

    report = library.execute(
        "move_relative",
        {"delta_xyz_m": [0.10, -0.05, 0.05], "gripper": "keep"},
        tool_context(),
    )

    assert report.status.value == "succeeded"
    assert report.ended_timestep > 0
    assert report.metadata["controller"] == "closed_loop_osc"
    assert "actions" not in report.metadata
    assert np.linalg.norm(env.eef - np.asarray([0.10, -0.05, 1.05])) < 0.02


def test_gripper_skill_preserves_libero_open_close_convention() -> None:
    env = DynamicLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="open gripper",
        initial_state=np.zeros(3),
        max_steps=20,
    )
    adapter.reset()
    library = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: [0.0] * 7,
    )

    report = library.execute(
        "set_gripper", {"state": "open", "steps": 4}, tool_context()
    )

    assert report.status.value == "succeeded"
    assert len(env.commands) == 4
    assert all(float(command[-1]) == 1.0 for command in env.commands)


def test_planner_request_exposes_skill_contracts_not_raw_actions() -> None:
    env = DynamicLiberoEnv()
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction="task",
        initial_state=np.zeros(3),
    )
    adapter.reset()
    library = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: [0.0] * 7,
    )
    context = HighLevelAgentContext(
        task_instruction="put the mug on the plate",
        trigger="task_start",
        episode_id="episode",
        timestep=0,
        available_skills=library.skill_ids,
        available_skill_specs=library.planner_specs,
    )

    request = build_high_level_agent_request(context)

    assert "available_skill_specs" in request["user_prompt"]
    assert "does not expose raw robot actions" in request["system_prompt"]
    assert '"actions"' not in request["user_prompt"]
