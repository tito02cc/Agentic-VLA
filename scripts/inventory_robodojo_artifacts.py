#!/usr/bin/env python3
"""Inventory every RoboDojo artifact directory and grade its evidence usability.

The handover document does not record the 110 directories under
artifacts/robodojo/, so it is easy to either redo that work or cite it as if it
were pre-registered evidence. This script reads what is actually on disk and
marks each run with why it can or cannot support an aggregate claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NOMINAL_SCHEMAS = {"carve.robodojo.nominal.v1", "carve.robodojo.nominal.v2"}
# v1..v26 style prefixes mean the run was one step of an interactive config
# sweep, so its outcome is selected-on and cannot be pooled as evidence.
SWEEP_PREFIX = re.compile(r"^(block|paired|probe|pilot|c2)_v\d+_")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _classify(name: str, summary: dict | None, result: dict | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if summary is None:
        reasons.append("no summary.json (run did not finalize)")
        return "unusable", reasons

    schema = summary.get("schema_version")
    if schema not in NOMINAL_SCHEMAS:
        reasons.append(f"non-nominal schema: {schema}")
        return "other", reasons

    if SWEEP_PREFIX.match(name):
        reasons.append("config-sweep run (vN prefix): outcome was selected on, not pre-registered")
    if "retry" in name:
        reasons.append("retry run: rerun after an earlier attempt")

    agg = summary.get("aggregate") or {}
    layout_ids = agg.get("layout_ids")
    episodes = agg.get("episodes", 1)
    if not layout_ids:
        layout_ids = [summary.get("layout_id")]
    if episodes == 1:
        reasons.append("single episode (eval_num=1): only layout_id 0 of 55")
    if set(layout_ids) == {0}:
        reasons.append("layout_id 0 only: no layout coverage")

    runtime = summary.get("runtime") or {}
    if runtime.get("trace_available") is False:
        reasons.append("no runtime trace (baseline): VLA calls unobserved")
    if runtime.get("inference_step_control_verified") is False:
        reasons.append("inference-step control NOT verified: condition label may not match runtime")

    horizons = runtime.get("observed_execute_horizons") or []
    if horizons and horizons != [16]:
        reasons.append(f"non-standard execute horizon {horizons}: confounded with flow steps")

    condition = str(summary.get("condition") or "")
    if condition not in {"B0", "C1", "C2", "C3", "shadow", "SHADOW"}:
        reasons.append(f"non-canonical condition label '{condition}': variant, not a matrix cell")

    grade = "mechanism_only" if reasons else "matrix_candidate"
    return grade, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/robodojo"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        summary = _load(directory / "summary.json")
        result = _load(directory / "result.json")
        grade, reasons = _classify(directory.name, summary, result)
        runtime = (summary or {}).get("runtime") or {}
        agentic = (summary or {}).get("agentic") or {}
        official = (summary or {}).get("official_result") or {}
        agg = (summary or {}).get("aggregate") or {}
        rows.append(
            {
                "directory": directory.name,
                "grade": grade,
                "reasons": reasons,
                "schema_version": (summary or {}).get("schema_version"),
                "condition": (summary or {}).get("condition"),
                "task": (summary or {}).get("task"),
                "layout_set": (summary or {}).get("seed"),
                "layout_ids": agg.get("layout_ids") or ([(summary or {}).get("layout_id")] if summary else []),
                "episodes": agg.get("episodes", 1 if summary else None),
                "successes": agg.get("successes", official.get("successes")),
                "success": official.get("success"),
                "video_frames": official.get("video_frames"),
                "vla_calls": runtime.get("vla_calls"),
                "trace_available": runtime.get("trace_available"),
                "inference_step_control_verified": runtime.get("inference_step_control_verified"),
                "observed_inference_steps": runtime.get("observed_inference_steps"),
                "observed_execute_horizons": runtime.get("observed_execute_horizons"),
                "first_action_seed": runtime.get("first_action_seed"),
                "semantic_checks": agentic.get("semantic_checks"),
                "interventions": agentic.get("interventions"),
                "monitor_events": agentic.get("monitor_events"),
                "has_result_json": result is not None,
                "video_count": len(list(directory.glob("*.mp4"))),
                "has_runtime_trace": (directory / "runtime_trace.jsonl").is_file(),
                "has_planner_trace": (directory / "planner_trace.jsonl").is_file(),
                "has_monitor_trace": (directory / "monitor_trace.jsonl").is_file(),
                "has_checksums": (directory / "SHA256SUMS").is_file(),
                "has_run_config": (directory / "run_config.json").is_file(),
            }
        )

    graded = [row for row in rows if row["schema_version"] in NOMINAL_SCHEMAS]
    by_task: dict[str, dict] = {}
    for row in graded:
        task = row["task"] or "unknown"
        entry = by_task.setdefault(task, {"episodes": 0, "successes": 0, "by_condition": {}})
        condition = row["condition"] or "unknown"
        cell = entry["by_condition"].setdefault(condition, {"runs": 0, "episodes": 0, "successes": 0})
        episodes = row["episodes"] or 0
        successes = row["successes"] if row["successes"] is not None else int(bool(row["success"]))
        cell["runs"] += 1
        cell["episodes"] += episodes
        cell["successes"] += successes
        entry["episodes"] += episodes
        entry["successes"] += successes

    payload = {
        "schema_version": "carve.robodojo.artifact-inventory.v1",
        "root": str(args.root),
        "directories": len(rows),
        "grades": {
            grade: sum(1 for row in rows if row["grade"] == grade)
            for grade in ("matrix_candidate", "mechanism_only", "unusable", "other")
        },
        "per_task_observed": by_task,
        "claim_boundary": (
            "Descriptive inventory of runs already on disk. Counts pool runs made "
            "under differing configurations and are NOT a benchmark result. Only "
            "directories graded 'matrix_candidate' could belong to a pre-registered "
            "matrix, and none of the historical single-episode runs qualify."
        ),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"directories: {len(rows)}")
    for grade, count in payload["grades"].items():
        print(f"  {grade:<18}{count}")


if __name__ == "__main__":
    main()
