"""RoboTwin 2.0 adapter for CARVE's public observation/action boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from agentic_vla.toolchain import ToolExecutionContext
from agentic_vla.toolchain.vla import ActionExecutionReport

from .contracts import DeployableBenchmarkObservation, PrivateBenchmarkResult


ActionStepCallback = Callable[[DeployableBenchmarkObservation, np.ndarray], None]


class RoboTwinRuntimeAdapter:
    """Own one seeded RoboTwin episode without exposing evaluator truth to agents."""

    def __init__(
        self,
        env: Any,
        *,
        seed: int,
        task_instruction: str = "",
        max_steps: int = 1000,
    ) -> None:
        if seed < 0 or max_steps <= 0:
            raise ValueError("RoboTwin seed and max_steps must be valid")
        for method in ("reset", "step", "render_camera", "get_task_language"):
            if not callable(getattr(env, method, None)):
                raise TypeError(f"RoboTwin environment must expose {method}(...)")
        self.env = env
        self.seed = int(seed)
        self.task_instruction = str(task_instruction).strip()
        self.max_steps = int(max_steps)
        self._info: Mapping[str, Any] | None = None
        self._timestep = 0
        self._terminated = False
        self._truncated = False

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    def _native(self) -> bool:
        return callable(getattr(self.env, "_sub_env", None))

    def _render(self, camera_name: str) -> np.ndarray:
        if self._native():
            frame = self.env.render_camera(camera_name, env_id=0)
        else:
            frame = self.env.render_camera(camera_name)
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(
                f"RoboTwin {camera_name} frame must have shape [H,W,C], got {array.shape}"
            )
        return np.ascontiguousarray(array[..., :3], dtype=np.uint8)

    @staticmethod
    def _status(info: Mapping[str, Any]) -> Mapping[str, Any]:
        status = info.get("episode_status")
        if not isinstance(status, Mapping):
            raise TypeError("RoboTwin info must contain episode_status")
        required = {"eval_success", "take_action_cnt", "actual_seed"}
        missing = sorted(required - set(status))
        if missing:
            raise ValueError(f"RoboTwin episode_status is missing {missing}")
        return status

    @staticmethod
    def _qpos(info: Mapping[str, Any]) -> np.ndarray:
        state = info.get("robot_state")
        if not isinstance(state, Mapping):
            raise TypeError("RoboTwin info must contain robot_state")
        qpos = np.asarray(state.get("qpos_target14"), dtype=np.float32).reshape(-1)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("RoboTwin qpos_target14 must contain 14 finite values")
        return qpos

    def reset(self) -> DeployableBenchmarkObservation:
        if self._native():
            _observation, info = self.env.reset(
                env_idx=[0], env_seeds=[self.seed]
            )
        else:
            _observation, info = self.env.reset()
        if not isinstance(info, Mapping):
            raise TypeError("RoboTwin reset info must be a mapping")
        status = self._status(info)
        if int(status["actual_seed"]) != self.seed:
            raise ValueError("RoboTwin reset did not preserve the requested seed")
        self._info = dict(info)
        self._timestep = int(status["take_action_cnt"])
        self._terminated = bool(status["eval_success"])
        self._truncated = False
        if not self.task_instruction:
            self.task_instruction = str(self.env.get_task_language()).strip()
        if not self.task_instruction:
            raise ValueError("RoboTwin task instruction must not be empty")
        return self.observe()

    def observe(self) -> DeployableBenchmarkObservation:
        if self._info is None:
            raise RuntimeError("reset must be called before observe")
        qpos = self._qpos(self._info)
        head = self._render("head")
        left = self._render("left_wrist")
        right = self._render("right_wrist")
        return DeployableBenchmarkObservation(
            policy_observation={
                "observation.images.cam_high": head,
                "observation.images.cam_left_wrist": left,
                "observation.images.cam_right_wrist": right,
                "observation.state": qpos,
                "task": self.task_instruction,
            },
            planner_frames={
                "head": head,
                "left_wrist": left,
                "right_wrist": right,
            },
            robot_state=tuple(float(value) for value in qpos),
            timestep=self._timestep,
            frame_id=f"robotwin:{self.seed}:{self._timestep}",
        )

    def policy_observation(self, _context: ToolExecutionContext) -> Mapping[str, Any]:
        return self.observe().policy_observation

    def robot_state(self) -> Mapping[str, Any]:
        """Return deployable proprioception while withholding episode labels."""

        if self._info is None:
            raise RuntimeError("reset must be called before robot_state")
        state = self._info.get("robot_state")
        if not isinstance(state, Mapping):
            raise TypeError("RoboTwin info must contain robot_state")
        return {
            str(name): np.asarray(value).copy()
            if isinstance(value, (np.ndarray, list, tuple))
            else value
            for name, value in state.items()
        }

    def plan_arm_path(self, arm: str, target_pose: Any) -> Mapping[str, Any]:
        """Call the public RoboTwin motion-planning API for a recovery tool."""

        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("target_pose must contain seven finite values")
        planner = getattr(self.env, "plan_arm_path", None)
        if not callable(planner):
            raise TypeError("RoboTwin environment does not expose plan_arm_path")
        if self._native():
            return planner(0, arm, target)
        return planner(arm, target)

    def execute_action_chunk(
        self,
        actions: Any,
        context: ToolExecutionContext,
        *,
        on_step: ActionStepCallback | None = None,
    ) -> ActionExecutionReport:
        if self._info is None:
            raise RuntimeError("reset must be called before action execution")
        if context.timestep != self._timestep:
            raise ValueError("tool context timestep is stale for RoboTwin")
        rows = np.asarray(actions, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 14:
            raise ValueError("RoboTwin qpos action chunk must have shape [N,14]")
        if not np.isfinite(rows).all():
            raise ValueError("RoboTwin action chunk contains non-finite values")

        executed = 0
        for action in rows:
            if self._terminated or self._truncated or self._timestep >= self.max_steps:
                break
            kwargs = {"action_type": "qpos"}
            if self._native():
                kwargs["env_id"] = 0
            _observation, _reward, terminated, truncated, info = self.env.step(
                action, **kwargs
            )
            if not isinstance(info, Mapping):
                raise TypeError("RoboTwin step info must be a mapping")
            status = self._status(info)
            self._info = dict(info)
            self._timestep = int(status["take_action_cnt"])
            self._terminated = bool(status["eval_success"]) or bool(
                np.asarray(terminated).any()
            )
            self._truncated = bool(np.asarray(truncated).any())
            executed += 1
            if on_step is not None:
                on_step(self.observe(), action.copy())

        status = "succeeded" if executed > 0 else "interrupted"
        return ActionExecutionReport(
            status=status,
            ended_timestep=self._timestep,
            observed_outcome=(
                "qpos action chunk executed" if executed else "no action was executed"
            ),
            requires_semantic_check=executed == 0,
            metadata={"executed_steps": executed, "action_type": "qpos"},
        )

    def private_result(self) -> PrivateBenchmarkResult:
        if self._info is None:
            raise RuntimeError("reset must be called before private_result")
        status = self._status(self._info)
        return PrivateBenchmarkResult(
            task_success=bool(status["eval_success"]),
            episode_steps=int(status["take_action_cnt"]),
            terminated=self._terminated or self._truncated,
            evaluator_metadata={
                "source": "robotwin_episode_status",
                "actual_seed": int(status["actual_seed"]),
            },
        )

    def render_video_frame(self) -> np.ndarray:
        return self._render("head")

    def close(self) -> None:
        offload = getattr(self.env, "offload", None)
        if callable(offload):
            offload(clear_cache=True)
            return
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
