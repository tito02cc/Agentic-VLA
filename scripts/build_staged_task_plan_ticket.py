#!/usr/bin/env python3
"""Build a guarded recovery ticket by verifying a stored task plan stage by stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain import (
    GuardedVisualVerifier,
    VerificationStatus,
    VisualVerificationContext,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18071/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("verified task plan requires non-empty steps")
    stages = [str(step.get("stage", "")).strip() for step in steps]
    if any(not stage for stage in stages) or len(stages) != len(set(stages)):
        raise ValueError("verified task plan stages must be non-empty and unique")
    return payload


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    plan = load_plan(plan_path)
    frame = np.asarray(Image.open(image_path).convert("RGB"))
    verifier = GuardedVisualVerifier(
        OpenAICompatibleVisionPlanner(
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_sec,
            max_tokens=args.max_tokens,
        ),
        minimum_confidence=args.minimum_confidence,
    )

    confirmed_stages: list[str] = []
    verification_rows: list[dict] = []
    active_step: dict | None = None
    for step in plan["steps"]:
        result = verifier.verify(
            VisualVerificationContext(
                task_instruction=str(plan["task_instruction"]),
                expected_outcome=str(step["expected_outcome"]),
                frames={"current_head": frame},
                timestep=0,
                active_stage=str(step["stage"]),
            )
        )
        verification_rows.append(
            {
                "stage": step["stage"],
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "report": result.report.to_dict(),
                "error": result.error,
                "raw_output": result.raw_output,
            }
        )
        if (
            result.accepted
            and result.report.status is VerificationStatus.CONFIRMED
        ):
            confirmed_stages.append(str(step["stage"]))
            continue
        active_step = step
        break

    if active_step is None:
        decision = {
            "intent": "safe_stop",
            "rationale": "all stored task stages are visibly complete",
            "confidence": 1.0,
            "subgoal": "",
            "vla_instruction": None,
            "skill_id": None,
            "skill_args": {},
            "expected_outcome": "",
            "memory_note": "all task-plan stages confirmed",
            "failure_type": "",
            "scene_graph_update": {},
            "proposed_plan": []
        }
    else:
        decision = {
            "intent": "vla_act",
            "rationale": "resume the first unconfirmed task-plan stage",
            "confidence": 0.9,
            "subgoal": active_step["subgoal"],
            "vla_instruction": active_step.get(
                "vla_instruction", active_step["subgoal"]
            ),
            "skill_id": active_step.get("skill_id"),
            "skill_args": {},
            "expected_outcome": active_step["expected_outcome"],
            "memory_note": (
                f"confirmed prefix={confirmed_stages}; "
                f"resume stage={active_step['stage']}"
            ),
            "failure_type": "no_progress",
            "scene_graph_update": {},
            "proposed_plan": []
        }

    ticket = {
        "schema_version": "carve.robodojo.planner-ticket.v2",
        "source": "staged_vlm_verification_of_verified_procedural_memory",
        "accepted": active_step is not None,
        "image": {
            "path": str(image_path),
            "sha256": digest(image_path),
        },
        "task_plan": {
            "plan_id": plan.get("plan_id"),
            "source": plan.get("source", "verified_procedural_memory"),
            "steps": plan["steps"],
            "confirmed_stages": confirmed_stages,
        },
        "decision": decision,
        "verification": verification_rows,
        "verifier_metrics": verifier.metrics(),
        "plan_path": str(plan_path),
        "plan_sha256": digest(plan_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ticket, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ticket, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
