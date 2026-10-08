#!/usr/bin/env python3
"""Run one guarded CARVE high-level VLM decision on a recorded robot frame."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import (
    GuardedHighLevelAgent,
    HighLevelAgentConfig,
    HighLevelAgentContext,
    OpenAICompatibleVisionPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--trigger", default="task_start")
    parser.add_argument("--risk-event", default="none")
    parser.add_argument("--current-subgoal", default="execute")
    parser.add_argument("--remaining-recoveries", type=int, default=1)
    parser.add_argument("--observed-group", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    parser.add_argument("--max-grounding-repairs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = pathlib.Path(args.image).expanduser().resolve()
    try:
        import imageio.v3 as iio

        frame = iio.imread(image_path)
    except ImportError:
        import numpy as np
        from PIL import Image

        frame = np.asarray(Image.open(image_path).convert("RGB"))
    agent = GuardedHighLevelAgent(
        OpenAICompatibleVisionPlanner(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_sec,
            max_tokens=args.max_tokens,
        ),
        HighLevelAgentConfig(
            max_calls_per_episode=1 + args.max_grounding_repairs,
            max_grounding_repairs=args.max_grounding_repairs,
            minimum_intervention_confidence=args.minimum_confidence,
        ),
    )
    result = agent.decide(
        HighLevelAgentContext(
            task_instruction=args.task,
            trigger=args.trigger,
            episode_id="high-level-smoke:0",
            timestep=0,
            frames={"agentview": frame},
            robot_state=(0.0,) * 9,
            risk={
                "event": None if args.risk_event == "none" else args.risk_event,
                "bucket": "high" if args.risk_event != "none" else "low",
                "score": 0.8 if args.risk_event != "none" else 0.0,
                "observed_groups": args.observed_group,
            },
            current_subgoal=args.current_subgoal,
            available_skills=(),
            remaining_retries=1,
            remaining_recoveries=args.remaining_recoveries,
            deadline_slack_ms=80.0,
        )
    )
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "attempt_count": result.attempt_count,
                "validation_errors": result.validation_errors,
                "decision": result.decision.to_dict(),
                "raw_output": result.raw_output,
                "raw_outputs": result.raw_outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
