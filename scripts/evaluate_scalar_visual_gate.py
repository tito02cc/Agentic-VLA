#!/usr/bin/env python3
"""Evaluate a constrained VLM scalar predicate on labeled robot frames."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain import (
    GuardedGroupedVisualVerifier,
    GuardedScalarVisualVerifier,
    ScalarVisualPredicate,
    VisualVerificationContext,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--failure-frame", type=Path, required=True)
    parser.add_argument("--success-frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--extraction-mode",
        choices=("groups", "scalar"),
        default="groups",
    )
    parser.add_argument(
        "--question",
        default=(
            "In the latest observation, how many spatially separate groups of target "
            "bowls are visible, regardless of bowl color? Bowls nested together count "
            "as one group."
        ),
    )
    parser.add_argument(
        "--task-instruction",
        default="Stack the three bowls together.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    infer = OpenAICompatibleVisionPlanner(
        endpoint=args.endpoint,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=192,
    )
    verifier_cls = (
        GuardedGroupedVisualVerifier
        if args.extraction_mode == "groups"
        else GuardedScalarVisualVerifier
    )
    verifier = verifier_cls(infer, minimum_confidence=0.55)
    predicate = ScalarVisualPredicate(
        predicate_id=(
            "target_bowl_groups"
            if args.extraction_mode == "groups"
            else "all_three_bowls_nested"
        ),
        question=args.question,
        comparison="eq",
        target=1,
        unit="groups",
    )
    samples = (
        ("natural_failure", args.failure_frame, "contradicted"),
        ("natural_success", args.success_frame, "confirmed"),
    )
    rows: list[dict[str, object]] = []
    for sample_id, frame_path, expected_status in samples:
        frame = load_rgb(frame_path)
        for repeat in range(args.repeats):
            context = VisualVerificationContext(
                task_instruction=args.task_instruction,
                expected_outcome="all three bowls form one nested stack",
                frames={"current_head": frame},
                timestep=repeat,
                active_stage="verify visible bowl grouping",
            )
            result = verifier.verify(context, predicate)
            actual_status = result.report.status.value
            rows.append(
                {
                    "sample_id": sample_id,
                    "repeat": repeat,
                    "frame": str(frame_path.resolve()),
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "matched": result.accepted and actual_status == expected_status,
                    "accepted": result.accepted,
                    "elapsed_ms": result.elapsed_ms,
                    "report": result.report.to_dict(),
                    "error": result.error,
                    "raw_output": result.raw_output,
                }
            )

    passed = bool(rows) and all(bool(row["matched"]) for row in rows)
    payload = {
        "schema_version": f"carve.{args.extraction_mode}-visual-gate.v1",
        "created_unix_s": time.time(),
        "model": args.model,
        "endpoint": args.endpoint,
        "predicate": {
            "predicate_id": predicate.predicate_id,
            "question": predicate.question,
            "comparison": predicate.comparison,
            "target": predicate.target,
            "unit": predicate.unit,
        },
        "repeats": args.repeats,
        "extraction_mode": args.extraction_mode,
        "passed": passed,
        "matched": sum(bool(row["matched"]) for row in rows),
        "total": len(rows),
        "verifier_metrics": verifier.metrics(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("passed", "matched", "total")}))
    print(args.output.resolve())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
