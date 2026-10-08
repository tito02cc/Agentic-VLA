#!/usr/bin/env python3
"""Replay stored Critic outputs through the current context projection; no models."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.benchmarks.robodojo_sorting import advisory_check_for_planner


def audit(path):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = []
    for event in events:
        if event["event_type"] != "visual_critic":
            continue
        raw = event["payload"]
        if raw["authority"] != "advisory":
            raise ValueError("this replay covers advisory checks only")
        check = {key: copy.deepcopy(raw[key]) for key in (
            "timestep", "expected_outcome", "observation_sha256", "report", "accepted")}
        original = copy.deepcopy(check)
        projected = advisory_check_for_planner(check)
        passed = (
            projected["expected_outcome"] == raw["expected_outcome"]
            and projected["observation_sha256"] == raw["observation_sha256"]
            and projected["report"]["status"] == "inconclusive"
            and projected["report"]["confidence"] == 0
            and projected["report"]["metadata"]["completion_authorized"] is False
            and raw["report"]["observed_outcome"] not in json.dumps(projected)
            and check == original)
        records.append({"event_sequence": event["sequence"], "source": check,
                        "planner_projection": projected, "passed": passed})
    unchanged = hashlib.sha256(path.read_bytes()).hexdigest() == before
    return {"passed": bool(records) and all(r["passed"] for r in records) and unchanged,
            "source": str(path.resolve()), "source_sha256": before, "source_unchanged": unchanged,
            "new_model_calls": 0, "new_robot_actions": 0, "records": records,
            "scope": "read-only context contract replay, not visual accuracy or Planner efficacy evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.events)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "source_unchanged", "new_model_calls", "new_robot_actions")}))
    if not result["passed"]:
        raise SystemExit("context projection replay failed")


if __name__ == "__main__":
    main()
