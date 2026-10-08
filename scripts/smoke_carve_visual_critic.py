#!/usr/bin/env python3
"""Run one guarded visual-Critic decision on a recorded robot frame."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import imageio.v3 as iio

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import OpenAICompatibleVisionPlanner  # noqa: E402
from agentic_vla.toolchain import (  # noqa: E402
    GuardedVisualVerifier,
    VisualVerificationContext,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--before-image", default="")
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-outcome", required=True)
    parser.add_argument("--active-stage", default="")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = iio.imread(pathlib.Path(args.image).expanduser().resolve())
    frames = {"current_agentview": frame}
    if args.before_image:
        frames["before_agentview"] = iio.imread(
            pathlib.Path(args.before_image).expanduser().resolve()
        )
    verifier = GuardedVisualVerifier(
        OpenAICompatibleVisionPlanner(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_sec,
            max_tokens=96,
        ),
        minimum_confidence=args.minimum_confidence,
    )
    result = verifier.verify(
        VisualVerificationContext(
            task_instruction=args.task,
            expected_outcome=args.expected_outcome,
            frames=frames,
            timestep=0,
            active_stage=args.active_stage,
            require_visual_change=bool(args.before_image),
        )
    )
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "report": result.report.to_dict(),
                "raw_output": result.raw_output,
                "metrics": verifier.metrics(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
