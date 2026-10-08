#!/usr/bin/env python3
"""Probe a live VLM on captured official sorting observations, without actions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.benchmarks.robodojo_sorting import SORTING_TASKS
from agentic_vla.runtime.agent import (
    GuardedHighLevelAgent, HighLevelAgentConfig, HighLevelAgentContext,
    OpenAICompatibleVisionPlanner, build_high_level_agent_request,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(SORTING_TASKS), required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=45)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    profile_url = args.endpoint.split("/v1/", 1)[0] + "/carve/profile"
    with urllib.request.urlopen(profile_url, timeout=10) as response:
        profile = json.load(response)
    if profile.get("model_id") != args.model:
        raise ValueError("connected VLM model identity does not match the probe")
    (args.output / "service_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    with np.load(args.observation, allow_pickle=False) as source:
        instruction = str(source["instruction"].item())
        state = source["state"].astype(float).tolist()
        frames = {}
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            rgb = source[key]
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.transpose(1, 2, 0)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("captured camera is not RGB")
            frames[f"current_{key}"] = rgb
            Image.fromarray(rgb).save(args.output / f"{key}.png")
    context = HighLevelAgentContext(
        task_instruction=instruction, trigger="task_start", episode_id="offline-planner-admission",
        timestep=0, frames=frames, robot_state=state, vla_instruction_mode="task_only",
        max_plan_stages=8,
        memory=(SORTING_TASKS[args.task].focus,),
        available_skills=("reobserve",),
        available_skill_specs=({"skill_id": "reobserve", "description": "Hold joint targets one control step and obtain a new observation; not grasp correction.",
                                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},),
        remaining_retries=2, remaining_recoveries=4,
    )
    request = build_high_level_agent_request(context)
    (args.output / "request.json").write_text(json.dumps({k: v for k, v in request.items() if k != "frames"}, indent=2) + "\n")
    infer = OpenAICompatibleVisionPlanner(endpoint=args.endpoint, model=args.model,
                                         timeout_s=args.timeout_s, max_tokens=args.max_tokens)
    result = GuardedHighLevelAgent(infer, HighLevelAgentConfig(
        max_calls_per_episode=2, max_grounding_repairs=1, minimum_intervention_confidence=.65,
    )).decide(context)
    report = {
        "task": args.task, "instruction": instruction, "model": args.model, "endpoint": args.endpoint,
        "generation_budget": {"timeout_s": args.timeout_s, "max_tokens": args.max_tokens},
        "observation": str(args.observation.resolve()),
        "observation_sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest(),
        "syntax_accepted": result.accepted, "error": result.error, "elapsed_ms": result.elapsed_ms,
        "attempt_count": result.attempt_count, "decision": result.decision.to_dict(),
        "raw_outputs": list(result.raw_outputs), "validation_errors": list(result.validation_errors),
        "semantic_review": "pending human/visual inspection; JSON acceptance is not correctness",
        "robot_actions_executed": 0, "simulation_episode": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
