#!/usr/bin/env python3
"""Compile StopCube instructions into evaluator-independent count memory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def required_occurrence(instruction: str) -> int:
    text = instruction.lower()
    for word, count in ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            return count
    match = re.search(r"\b([1-9])(?:st|nd|rd|th)?\b", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"StopCube instruction has no occurrence count: {instruction}")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for summary_path in sorted(args.input_root.glob("StopCube_ep*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        count = required_occurrence(str(summary["instruction"]))
        episode = int(summary["episode"])
        output = {
            "protocol": "carve.robomme.stopcube_count_memory.v1",
            "task": "StopCube",
            "episode": episode,
            "source_instruction": str(summary["instruction"]),
            "evaluator_or_oracle_fields_used": False,
            "memory": {
                "required_target_occurrence": count,
                "planner_hint": (
                    "Remain ready and press the button on target occurrence "
                    f"{count}."
                ),
            },
        }
        destination = args.output_root / f"StopCube_ep{episode}_memory.json"
        destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"episode": episode, "required_occurrence": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
