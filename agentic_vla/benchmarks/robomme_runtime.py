"""Deployable-observation adapter for the official RoboMME benchmark."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from agentic_vla.toolchain import ToolExecutionContext
from agentic_vla.toolchain.vla import ActionExecutionReport

from .contracts import DeployableBenchmarkObservation, PrivateBenchmarkResult


ActionStepCallback = Callable[[DeployableBenchmarkObservation, np.ndarray], None]


def pack_robomme_state(joint_state: Any, gripper_state: Any) -> np.ndarray:
    """Pack RoboMME's public robot observation into its official 8-D contract."""

    joints = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    gripper = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
    if joints.size != 7 or gripper.size < 1:
        raise ValueError("RoboMME requires seven joints and at least one gripper value")
    state = np.concatenate((joints, gripper[:1]), dtype=np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("RoboMME robot state contains non-finite values")
    return state


class RoboMMERuntimeAdapter:
    """Keep RoboMME evaluator truth outside CARVE policy and planner inputs.

    The wrapped environment is an episode returned by the official
    ``BenchmarkEnvBuilder``. Initial demonstration frames are public benchmark
    observations and may be passed to memory modules. Online oracle subgoals and
    terminal labels remain private.
    """

    def __init__(
        self,
        env: Any,
        *,
        task_id: str,
        episode_id: int,
        max_steps: int = 1300,
    ) -> None:
        if not str(task_id).strip() or episode_id < 0 or max_steps <= 0:
            raise ValueError("task_id, episode_id, and max_steps must be valid")
        if not hasattr(env, "reset") or not hasattr(env, "step"):
            raise TypeError("RoboMME environment must expose reset and step")
        self.env = env
        self.task_id = str(task_id)
        self.episode_id = int(episode_id)
        self.max_steps = int(max_steps)
        self.task_instruction = ""
        self._observation: Mapping[str, Any] | None = None
        self._private_info: Mapping[str, Any] = {}
        self._timestep = 0
        self._terminated = False
        self._outcome = "unknown"
        self._initial_front_frames: tuple[np.ndarray, ...] = ()
        self._initial_wrist_frames: tuple[np.ndarray, ...] = ()
        self._initial_states: tuple[np.ndarray, ...] = ()

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def initial_memory(self) -> Mapping[str, tuple[np.ndarray, ...]]:
        """Public initial demonstration/observation sequence for memory modules."""

        if self._observation is None:
            raise RuntimeError("reset must be called before reading initial memory")
        return {
            "front_rgb": self._initial_front_frames,
            "wrist_rgb": self._initial_wrist_frames,
            "robot_state": self._initial_states,
        }

    def reset(self) -> DeployableBenchmarkObservation:
        observation, info = self.env.reset()
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise TypeError("RoboMME reset must return observation and info mappings")
        self._observation = observation
        self._private_info = info
        task_goal = info.get("task_goal")
        if isinstance(task_goal, Sequence) and not isinstance(task_goal, str):
            task_goal = task_goal[0] if task_goal else ""
        self.task_instruction = str(task_goal or "").strip()
        if not self.task_instruction:
            raise ValueError("RoboMME reset did not provide a task goal")
        self._timestep = 0
        self._terminated = False
        self._outcome = "unknown"

        front, wrist, states = self._read_sequences(observation)
        self._initial_front_frames = tuple(frame.copy() for frame in front)
        self._initial_wrist_frames = tuple(frame.copy() for frame in wrist)
        self._initial_states = tuple(state.copy() for state in states)
        return self.observe()

    def _read_sequences(
        self, observation: Mapping[str, Any]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        required = (
            "front_rgb_list",
            "wrist_rgb_list",
            "joint_state_list",
            "gripper_state_list",
        )
        missing = [name for name in required if name not in observation]
        if missing:
            raise KeyError(f"RoboMME observation is missing fields: {missing}")
        front = [np.asarray(value, dtype=np.uint8) for value in observation["front_rgb_list"]]
        wrist = [np.asarray(value, dtype=np.uint8) for value in observation["wrist_rgb_list"]]
        joints = list(observation["joint_state_list"])
        grippers = list(observation["gripper_state_list"])
        lengths = {len(front), len(wrist), len(joints), len(grippers)}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("RoboMME observation sequences must be non-empty and aligned")
        states = [pack_robomme_state(joint, grip) for joint, grip in zip(joints, grippers)]
        return front, wrist, states

    def observe(self) -> DeployableBenchmarkObservation:
        if self._observation is None:
            raise RuntimeError("reset must be called before observe")
        front, wrist, states = self._read_sequences(self._observation)
        current_front = np.ascontiguousarray(front[-1])
        current_wrist = np.ascontiguousarray(wrist[-1])
        current_state = states[-1]
        return DeployableBenchmarkObservation(
            policy_observation={
                "observation/image": current_front,
                "observation/wrist_image": current_wrist,
                "observation/state": current_state,
            },
            planner_frames={
                "front": current_front,
                "wrist": current_wrist,
            },
            robot_state=tuple(float(value) for value in current_state),
            timestep=self._timestep,
            frame_id=f"robomme:{self.task_id}:{self.episode_id}:{self._timestep}",
        )

    def policy_observation(self, _context: ToolExecutionContext) -> Mapping[str, Any]:
        return self.observe().policy_observation

    def execute_action_chunk(
        self,
        actions: Any,
        context: ToolExecutionContext,
        *,
        on_step: ActionStepCallback | None = None,
    ) -> ActionExecutionReport:
        if self._observation is None:
            raise RuntimeError("reset must be called before action execution")
        if context.timestep != self._timestep:
            raise ValueError("tool context timestep is stale for RoboMME")
        try:
            rows = list(actions)
        except TypeError as exc:
            raise TypeError("RoboMME action chunk must be iterable") from exc
        if not rows:
            raise ValueError("RoboMME action chunk must not be empty")

        executed = 0
        for raw_action in rows:
            if self._terminated or self._timestep >= self.max_steps:
                break
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if action.size != 8 or not np.all(np.isfinite(action)):
                raise ValueError("RoboMME actions must be finite 8-D vectors")
            observation, _reward, terminated, truncated, info = self.env.step(action)
            if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
                raise TypeError("RoboMME step must return observation and info mappings")
            self._observation = observation
            self._private_info = info
            self._timestep += 1
            executed += 1
            self._terminated = bool(terminated or truncated)
            self._outcome = str(info.get("status", "unknown"))
            if on_step is not None:
                on_step(self.observe(), action.copy())
            if self._terminated:
                break

        status = "succeeded" if executed > 0 else "interrupted"
        return ActionExecutionReport(
            status=status,
            ended_timestep=self._timestep,
            observed_outcome=(
                "action chunk executed" if executed > 0 else "no action was executed"
            ),
            requires_semantic_check=self._terminated or executed == 0,
            metadata={"executed_steps": executed},
        )

    def private_result(self) -> PrivateBenchmarkResult:
        return PrivateBenchmarkResult(
            task_success=self._outcome == "success",
            episode_steps=self._timestep,
            terminated=self._terminated,
            evaluator_metadata={
                "source": "robomme_status",
                "status": self._outcome,
                "task_id": self.task_id,
                "episode_id": self.episode_id,
            },
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

