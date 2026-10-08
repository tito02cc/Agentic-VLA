#!/usr/bin/env python3
"""Compare request-seeded RoboDojo runs before admitting paired results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "runtime_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _result(root: Path) -> dict:
    return json.loads((root / "result.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows_a = _rows(args.run_a)
    rows_b = _rows(args.run_b)
    by_step_a = {int(row["timestep"]): row for row in rows_a}
    by_step_b = {int(row["timestep"]): row for row in rows_b}
    shared_steps = sorted(set(by_step_a) & set(by_step_b))
    mismatches = []
    for step in shared_steps:
        row_a = by_step_a[step]
        row_b = by_step_b[step]
        hash_a = row_a.get("metadata", {}).get("action_sha256")
        hash_b = row_b.get("metadata", {}).get("action_sha256")
        input_a = row_a.get("metadata", {}).get("input_sha256")
        input_b = row_b.get("metadata", {}).get("input_sha256")
        seed_a = row_a.get("metadata", {}).get("action_seed")
        seed_b = row_b.get("metadata", {}).get("action_seed")
        if hash_a != hash_b or input_a != input_b or seed_a != seed_b:
            mismatches.append(
                {
                    "timestep": step,
                    "action_seed_a": seed_a,
                    "action_seed_b": seed_b,
                    "action_sha256_a": hash_a,
                    "action_sha256_b": hash_b,
                    "input_sha256_a": input_a,
                    "input_sha256_b": input_b,
                }
            )

    result_a = _result(args.run_a)
    result_b = _result(args.run_b)
    report = {
        "schema_version": "carve.robodojo.reproducibility.v1",
        "run_a": str(args.run_a.resolve()),
        "run_b": str(args.run_b.resolve()),
        "shared_action_chunks": len(shared_steps),
        "matching_action_chunks": len(shared_steps) - len(mismatches),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:5],
        "official_result_a": result_a,
        "official_result_b": result_b,
        "official_results_equal": result_a == result_b,
        "admitted": bool(shared_steps) and not mismatches and result_a == result_b,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
