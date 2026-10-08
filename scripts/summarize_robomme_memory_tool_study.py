#!/usr/bin/env python3
"""Build the auditable RoboMME video-memory/tool-routing pilot summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "robomme_memory_tool_study_20260829"
CONDITIONS = {
    "ep0_memory_free_every_chunk": ROOT
    / "results/robomme_memory_pilot_every_chunk_bf16_20260829/VideoRepick_ep0/summary.json",
    "ep0_memory_only_selective": ROOT
    / "results/robomme_memory_qwen9b_relational_adaptive_retry_20260829/VideoRepick_ep0/summary.json",
    "ep0_memory_tool_every_chunk": ROOT
    / "results/robomme_memory_qwen9b_relational_tool_every_chunk_20260829/VideoRepick_ep0/summary.json",
    "ep0_memory_tool_event_tuned": ROOT
    / "results/robomme_memory_qwen9b_relational_tool_event_tuned_20260829/VideoRepick_ep0/summary.json",
    "ep1_memory_tool": ROOT
    / "results/robomme_video_repick_multi_every_chunk_redfix_20260829/VideoRepick_ep1/summary.json",
    "ep2_progress_memory_tool": ROOT
    / "results/robomme_video_repick_ep2_every_chunk_progress_20260829/VideoRepick_ep2/summary.json",
    "ep2_button_refinement": ROOT
    / "results/robomme_video_repick_ep2_every_chunk_buttonfix_20260829/VideoRepick_ep2/summary.json",
    "ep2_transition_confirmation_negative": ROOT
    / "results/robomme_video_repick_ep2_transition_gate_20260829/VideoRepick_ep2/summary.json",
    "ep1_online_oracle_upper_bound": ROOT
    / "results/robomme_video_repick_oracle_multi_20260829/VideoRepick_ep1/summary.json",
    "ep2_online_oracle_upper_bound": ROOT
    / "results/robomme_video_repick_oracle_ep2_trace_20260829/summary.json",
}


def load_row(name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    planner_trace = payload.get("planner_trace", [])
    subgoals = [str(item.get("grounded_subgoal", "")) for item in planner_trace]
    video_path = Path(str(payload.get("video_path", "")))
    if video_path.is_absolute() and str(video_path).startswith("/workspace/"):
        video_path = ROOT / video_path.relative_to("/workspace")
    return {
        "condition": name,
        "task": payload.get("task"),
        "episode": payload.get("episode"),
        "success": bool(payload.get("success")),
        "status": payload.get("status"),
        "deployable_method": bool(payload.get("deployable_method")),
        "privileged_online_subgoal_used": bool(
            payload.get("privileged_online_subgoal_used")
        ),
        "executed_steps": payload.get("executed_steps"),
        "policy_calls": payload.get("policy_calls"),
        "planner_calls": payload.get("planner_calls"),
        "planner_reuse_hits": payload.get("planner_reuse_hits"),
        "planner_call_reduction_fraction": payload.get(
            "planner_call_reduction_fraction"
        ),
        "wall_time_s": payload.get("wall_time_s"),
        "reached_press_stage": any("press" in subgoal.lower() for subgoal in subgoals),
        "video_path": str(video_path.relative_to(ROOT)) if video_path.exists() else None,
        "summary_path": str(path.relative_to(ROOT)),
    }


def main() -> None:
    missing = [str(path) for path in CONDITIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing study inputs:\n" + "\n".join(missing))
    rows = [load_row(name, path) for name, path in CONDITIONS.items()]
    every = next(row for row in rows if row["condition"] == "ep0_memory_tool_every_chunk")
    tuned = next(row for row in rows if row["condition"] == "ep0_memory_tool_event_tuned")
    comparison = {
        "paired_episode": 0,
        "success_preserved": every["success"] and tuned["success"],
        "planner_calls_every_chunk": every["planner_calls"],
        "planner_calls_event_tuned": tuned["planner_calls"],
        "planner_call_reduction_fraction": 1.0
        - float(tuned["planner_calls"]) / float(every["planner_calls"]),
        "wall_time_every_chunk_s": every["wall_time_s"],
        "wall_time_event_tuned_s": tuned["wall_time_s"],
        "wall_time_reduction_fraction": 1.0
        - float(tuned["wall_time_s"]) / float(every["wall_time_s"]),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "carve.robomme.video_memory_tool_pilot.v1",
        "claim_boundary": (
            "Episode 0 is the admitted deployable positive case. Episodes 1 and 2 "
            "are diagnostic failures; oracle rows are privileged upper bounds only."
        ),
        "rows": rows,
        "episode0_optimize_pair": comparison,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    readme = f"""# RoboMME Video-Memory and Tool-Routing Pilot

This package is generated from the original episode summaries; it does not
replace or modify raw results.

## Admitted positive result

- `VideoRepick` episode 0 succeeds without online oracle input when CARVE uses
  offline demonstration memory, online GroundSG planning, relational visual
  grounding and frozen PI0.5 execution.
- The paired event-tuned runtime preserves success and changes Planner calls
  from {every['planner_calls']} to {tuned['planner_calls']}
  ({comparison['planner_call_reduction_fraction']:.1%} reduction), while wall
  time changes from {every['wall_time_s']:.2f} s to {tuned['wall_time_s']:.2f} s
  ({comparison['wall_time_reduction_fraction']:.1%} reduction).

## Boundaries

- Episode 1 fails because the offline memory encoder identifies the wrong
  instance relation; its online-oracle upper bound succeeds.
- Episode 2 reaches three pick/put cycles and the button stage, but does not
  satisfy the final evaluator condition. Button refinement and two-turn
  semantic transition confirmation are retained as diagnostic/negative
  ablations, not promoted defaults.
- Oracle rows use evaluator-provided online grounded subgoals and must never be
  reported as deployable CARVE results.

See `conditions.csv` for the flat table and `summary.json` for exact paths and
paired calculations.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
