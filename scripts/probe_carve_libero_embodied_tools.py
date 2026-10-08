#!/usr/bin/env python3
"""Qualify CARVE RGB-D grounding and analytic skills in real LIBERO-PRO physics."""

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
from agentic_vla.toolchain import ToolExecutionContext, core_tool_specs  # noqa: E402
from scripts import run_agentic_vla_libero_pro_canonical as canonical  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10_object")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--u", type=float, default=0.76)
    parser.add_argument("--v", type=float, default=0.64)
    parser.add_argument("--approach-height-m", type=float, default=0.15)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "libero_pro_embodied_tool_probe_20260827",
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
        task_instruction=task.language,
        initial_state=initial_states[args.trial],
        max_steps=320,
    )
    observation = adapter.reset()
    frames = [adapter.render_video_frame(size=args.video_size)]
    last_action = np.asarray(canonical.DUMMY_ACTION, dtype=np.float32)

    def record_step(step_observation, action) -> None:
        nonlocal last_action
        last_action = np.asarray(action, dtype=np.float32).copy()
        frames.append(adapter.render_video_frame(size=args.video_size))

    library = LiberoEmbodiedSkillLibrary(
        adapter,
        last_action_source=lambda: last_action,
        on_step=record_step,
    )

    def context() -> ToolExecutionContext:
        return ToolExecutionContext(
            episode_id=f"{args.suite}:{args.task_id}:{args.trial}",
            timestep=adapter.timestep,
            at_safe_boundary=True,
            allowed_tools=tuple(spec.name for spec in core_tool_specs()),
            deployment_profile_id="analytic-tool-qualification",
        )

    started = time.perf_counter()
    reports = []
    try:
        grounded = adapter.planner_pixel_to_world(u=args.u, v=args.v)
        reports.append(
            library.execute(
                "visual_servo_above",
                {
                    "u": args.u,
                    "v": args.v,
                    "approach_height_m": args.approach_height_m,
                    "gripper": "open",
                },
                context(),
            ).to_dict()
        )
        reports.append(
            library.execute(
                "rotate_wrist",
                {"delta_rad": 0.35, "gripper": "open"},
                context(),
            ).to_dict()
        )
        reports.append(
            library.execute(
                "move_relative",
                {"delta_xyz_m": [0.0, 0.0, 0.08], "gripper": "open"},
                context(),
            ).to_dict()
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        video_path = args.output_root / "embodied_tool_probe.mp4"
        imageio.mimwrite(video_path, frames, fps=args.video_fps, quality=7)
        payload = {
            "protocol": "carve_libero_embodied_tool_qualification_v1",
            "claim_boundary": "real LIBERO-PRO physics; no VLA and no evaluator state used",
            "suite": args.suite,
            "task_id": args.task_id,
            "trial": args.trial,
            "task_instruction": task.language,
            "normalized_pixel": [args.u, args.v],
            "grounded_world_point": grounded.tolist(),
            "depth_available": adapter.depth_available,
            "reports": reports,
            "all_skills_succeeded": all(report["status"] == "succeeded" for report in reports),
            "episode_steps": adapter.timestep,
            "elapsed_s": time.perf_counter() - started,
            "video": str(video_path.resolve()),
        }
        result_path = args.output_root / "summary.json"
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if payload["all_skills_succeeded"] else 2
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
