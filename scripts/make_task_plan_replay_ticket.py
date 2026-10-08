#!/usr/bin/env python3
"""Convert an accepted CARVE task-plan gate into an auditable replay ticket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("accepted") is not True:
        raise SystemExit("source gate must contain accepted=true")
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise SystemExit("source gate must contain a decision object")
    steps = decision.get("proposed_plan")
    if not isinstance(steps, list) or not steps:
        raise SystemExit("accepted decision must contain a non-empty proposed_plan")

    ticket = {
        "schema_version": "carve.task-plan-replay.v1",
        "accepted": True,
        "source": f"guarded_vlm_gate:{source}",
        "decision": decision,
        "task_plan": {
            "source": f"guarded_vlm_gate:{source}",
            "confirmed_stages": [],
            "steps": steps,
        },
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(ticket, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
