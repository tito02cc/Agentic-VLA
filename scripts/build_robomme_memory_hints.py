#!/usr/bin/env python3
"""Convert verbose RoboMME task memory into typed Planner constraints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    return parser.parse_args()


def _compact(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _required_count(memory: dict, instruction: str) -> int:
    persistent = memory.get("persistent_memory", {})
    if isinstance(persistent, dict):
        explicit = persistent.get("required_repetitions")
        if isinstance(explicit, int) and explicit > 0:
            return explicit
    text = " ".join(
        (_compact(memory.get("task_constraints", "")), instruction)
    ).lower()
    words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }
    for word, value in words.items():
        if re.search(rf"\b{word}\b", text):
            return value
    match = re.search(r"\b([1-9])(?:st|nd|rd|th)?\b", text)
    return int(match.group(1)) if match else 1


def _demonstrated_identity(sequence: object) -> str | None:
    if not isinstance(sequence, list):
        return None
    for step in sequence:
        if isinstance(step, str):
            match = re.search(
                r"\b(?:grasp(?:ed)?|pick(?:ed)?(?:\s+up)?)\s+(?:the\s+)?(.+)",
                step,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
            continue
        if not isinstance(step, dict):
            continue
        direct = step.get("object_identity") or step.get("object") or step.get("target")
        action = str(step.get("action", "")).lower()
        if isinstance(direct, str) and direct.strip() and direct.strip().lower() != "button":
            if "grasp" in action or "pick" in action:
                return direct.strip()
        interactions = step.get("object_interactions", [])
        if not isinstance(interactions, list):
            continue
        for interaction in interactions:
            if not isinstance(interaction, dict):
                continue
            interaction_action = str(interaction.get("action", "")).lower()
            if not (
                "grasp" in interaction_action
                or "pick" in interaction_action
                or "grasp" in action
                or "pick" in action
            ):
                continue
            target = interaction.get("object") or interaction.get("object_identity")
            if not isinstance(target, str) or not target.strip():
                continue
            location = interaction.get("location")
            if isinstance(location, str) and location.strip():
                return f"{target.strip()} at {location.strip()}"
            return target.strip()
    return None


def hint_for(task: str, memory: dict, instruction: str = "") -> str:
    persistent = memory.get("persistent_memory", {})
    sequence = memory.get("demonstrated_sequence", [])
    if task == "VideoUnmaskSwap":
        positions = persistent.get("object_positions", {}) if isinstance(persistent, dict) else {}
        if positions:
            return (
                "The demonstrated green-cube container is "
                f"{positions.get('container_green_cube', 'unknown')}; the blue-cube "
                f"container is {positions.get('container_blue_cube', 'unknown')}. "
                "Select the green one first and the blue one second."
            )
        return (
            f"Persistent object memory: {_compact(persistent)}. "
            f"Demonstrated operation order: {_compact(sequence)}."
        )
    if task == "VideoRepick":
        identity = (
            _demonstrated_identity(sequence)
            or _compact(persistent)
            or "the demonstrated block"
        )
        constraints = memory.get("task_constraints", "repeat as instructed, then press the button")
        return (
            f"The persistent target identity is {identity}. Persistent evidence: "
            f"{_compact(persistent)}. Task sequence: {_compact(constraints)}"
        )
    if task == "StopCube":
        required = _required_count(memory, instruction)
        return f"Remain ready and press the button on target occurrence {required}."
    if task == "RouteStick":
        actions = [
            str(step.get("action", "")).strip() if isinstance(step, dict) else str(step).strip()
            for step in sequence
        ]
        actions = [value for value in actions if value]
        return (
            "Demonstrated route order: "
            + "; ".join(actions)
            + f". Persistent spatial relations: {_compact(persistent)}"
        )
    raise ValueError(f"unsupported task: {task}")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_tasks = set(args.tasks) if args.tasks else None
    for path in sorted(args.input_root.glob("*_ep*_memory.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = str(payload["task"])
        episode = int(payload["episode"])
        if selected_tasks is not None and task not in selected_tasks:
            continue
        memory = payload["memory"]
        output = {
            "protocol": "carve.robomme.planner_memory_hint.v1",
            "task": task,
            "episode": episode,
            "source_memory": str(path),
            "evaluator_or_oracle_fields_used": False,
            "memory": {
                "planner_hint": hint_for(
                    task,
                    memory,
                    str(payload.get("source", {}).get("instruction", "")),
                )
            },
        }
        destination = args.output_root / f"{task}_ep{episode}_memory.json"
        destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
