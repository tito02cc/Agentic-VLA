#!/usr/bin/env python3
"""Let the guarded VLM Planner ground and execute one registered LIBERO tool."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import imageio.v2 as imageio
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi_official"
LIBERO_ROOT = pathlib.Path(
    os.environ.get("AGENTIC_VLA_LIBERO_ROOT", "/home/admin1/ct/benchmark-sources/LIBERO-PRO")
)
for path in (
    PROJECT_ROOT,
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
    LIBERO_ROOT,
):
    sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from agentic_vla.benchmarks import LiberoEmbodiedSkillLibrary, LiberoRuntimeAdapter  # noqa: E402
from agentic_vla.runtime import (  # noqa: E402
    AgentIntent,
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    OpenAICompatibleVisionPlanner,
)
from agentic_vla.toolchain import ToolExecutionContext, core_tool_specs  # noqa: E402
from scripts import run_agentic_vla_libero_pro_canonical as canonical  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10_object")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--instruction",
        default=(
            "Use the registered visual_servo_above skill to move the open gripper "
            "to a safe point 15 cm above the right moka pot without touching it."
        ),
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:18070/v1/chat/completions"
    )
    parser.add_argument(
        "--model", default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-9B"
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "vlm_embodied_tool_use_20260827",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical._configure_libero_paths(LIBERO_ROOT)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    env = canonical._make_environment(task, seed=args.seed)
    adapter = LiberoRuntimeAdapter(
        env,
        task_instruction=args.instruction,
        initial_state=initial_states[args.trial],
        max_steps=160,
    )
    observation = adapter.reset()
    frames = [adapter.render_video_frame(size=args.video_size)]
    last_action = np.asarray(canonical.DUMMY_ACTION, dtype=np.float32)

    def record_step(_observation, action) -> None:
        nonlocal last_action
        last_action = np.asarray(action, dtype=np.float32).copy()
        frames.append(adapter.render_video_frame(size=args.video_size))

    library = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: last_action,
        on_step=record_step,
    )
    agent = GuardedHighLevelAgent(
        OpenAICompatibleVisionPlanner(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
        ),
        HighLevelAgentConfig(
            max_calls_per_episode=2,
            max_grounding_repairs=1,
            minimum_intervention_confidence=0.55,
        ),
    )
    context = HighLevelAgentContext(
        task_instruction=args.instruction,
        trigger="task_start",
        episode_id=f"vlm-tool-probe:{args.suite}:{args.task_id}:{args.trial}",
        timestep=adapter.timestep,
        frames={"agentview": observation.planner_frames["agentview"]},
        robot_state=observation.robot_state,
        risk={"event": None, "bucket": "low", "score": 0.0},
        current_subgoal="move safely above the right moka pot",
        available_skills=library.skill_ids,
        available_skill_specs=library.planner_specs,
        remaining_retries=1,
        remaining_recoveries=0,
        deadline_slack_ms=80.0,
    )

    started = time.perf_counter()
    planner_result = agent.decide(context)
    report = None
    try:
        if (
            planner_result.accepted
            and planner_result.decision.intent is AgentIntent.RUN_SKILL
            and planner_result.decision.skill_id in library.skill_ids
        ):
            report = library.execute(
                planner_result.decision.skill_id,
                planner_result.decision.skill_args,
                ToolExecutionContext(
                    episode_id=context.episode_id,
                    timestep=adapter.timestep,
                    at_safe_boundary=True,
                    allowed_tools=tuple(spec.name for spec in core_tool_specs()),
                    deployment_profile_id="qwen35-9b-nf4+analytic-skill",
                ),
            )
        args.output_root.mkdir(parents=True, exist_ok=True)
        video_path = args.output_root / "vlm_grounded_tool_use.mp4"
        imageio.mimwrite(video_path, frames, fps=args.video_fps, quality=7)
        succeeded = report is not None and report.status.value == "succeeded"
        payload = {
            "protocol": "carve_vlm_embodied_tool_use_v1",
            "claim_boundary": (
                "guarded Qwen3.5-9B NF4 tool selection and RGB-D grounding in real "
                "LIBERO-PRO physics; no VLA or evaluator state used"
            ),
            "suite": args.suite,
            "task_id": args.task_id,
            "trial": args.trial,
            "scene_task": task.language,
            "probe_instruction": args.instruction,
            "planner": {
                "accepted": planner_result.accepted,
                "elapsed_ms": planner_result.elapsed_ms,
                "attempt_count": planner_result.attempt_count,
                "validation_errors": list(planner_result.validation_errors),
                "error": planner_result.error,
                "decision": planner_result.decision.to_dict(),
            },
            "execution": None if report is None else report.to_dict(),
            "succeeded": succeeded,
            "episode_steps": adapter.timestep,
            "elapsed_s": time.perf_counter() - started,
            "video": str(video_path.resolve()),
        }
        summary_path = args.output_root / "summary.json"
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if succeeded else 2
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
