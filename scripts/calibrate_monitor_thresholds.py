#!/usr/bin/env python
"""Offline calibration and admission of Monitor intervention triggers.

Why offline is exact here, not approximate
-----------------------------------------
``monitor_trace.jsonl`` records the Monitor's *window means* alongside each
decision. Those means do not depend on any threshold, so replaying the event
layer over them reproduces the deployed decision exactly for any candidate
threshold set (verified against the live Monitor in
``tests/test_trigger_admission.py``). No robot time is needed to answer "what
would this threshold have fired on".

Boundary: only thresholds may be swept. Changing ``window_size`` or
``warmup_steps`` would change the recorded means themselves, so those are not
free parameters of this analysis.

Candidate thresholds are derived from observed distribution quantiles rather than
from a hand-picked grid, because the defect this tool exists to catch is exactly
a threshold placed at the wrong scale for the signal.

Usage:
    openpi/.venv/bin/python scripts/calibrate_monitor_thresholds.py \
        --run artifacts/robodojo/ral_matrix_20260907_stack_bowls_c2_set0 \
        --run artifacts/robodojo/ral_matrix_20260907_stack_bowls_c3_set0 \
        --output artifacts/robodojo/<dir>/monitor_calibration.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_vla.runtime.monitor import MonitorConfig
from agentic_vla.toolchain import (
    EpisodeTriggerEvidence,
    MonitorEventThresholds,
    MonitorSample,
    TriggerAdmissionRequirements,
    recompute_monitor_events,
    validate_trigger_admission,
)

EVENTS = ("stall", "no_progress")


def load_run(run: Path) -> tuple[list[list[MonitorSample]], list[bool], dict[str, Any]]:
    """Return per-episode Monitor samples, per-episode outcomes, and metadata."""
    trace = run / "monitor_trace.jsonl"
    summary = run / "summary.json"
    if not trace.is_file():
        raise FileNotFoundError(f"no monitor_trace.jsonl in {run}")
    if not summary.is_file():
        raise FileNotFoundError(f"no summary.json in {run}")

    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty monitor trace: {trace}")

    # Episode boundaries: prefer the recorded reset generation, fall back to a
    # decreasing timestep, which is where the environment was reset.
    episodes: list[list[dict[str, Any]]] = [[]]
    previous = None
    previous_generation = None
    for row in rows:
        generation = row.get("policy_reset_generation")
        step = int(row["timestep"])
        new_episode = (
            generation != previous_generation
            if generation is not None and previous_generation is not None
            else (previous is not None and step < previous)
        )
        if new_episode and episodes[-1]:
            episodes.append([])
        episodes[-1].append(row)
        previous, previous_generation = step, generation

    per_episode: list[list[MonitorSample]] = []
    for block in episodes:
        if not block:
            continue
        per_episode.append(
            [
                MonitorSample(
                    timestep=int(row["timestep"]),
                    window_ready=bool(row["assessment"]["evidence"]["window_ready"]),
                    mean_command=float(row["assessment"]["evidence"]["mean_command"]),
                    mean_state_response=float(
                        row["assessment"]["evidence"]["mean_state_response"]
                    ),
                    mean_visual_response=float(
                        row["assessment"]["evidence"]["mean_visual_response"]
                    ),
                    action_age_steps=int(
                        row["assessment"]["evidence"].get("action_age_steps", 0)
                    ),
                    has_frame=True,
                )
                for row in block
            ]
        )

    details = json.loads(summary.read_text(encoding="utf-8"))["episodes"]
    outcomes = [bool(item["success"]) for item in details]
    metadata = {
        "run": str(run),
        "monitor_rows": len(rows),
        "episodes_in_trace": len(per_episode),
        "episodes_in_summary": len(outcomes),
    }
    return per_episode, outcomes, metadata


def signal_quantiles(per_episode: list[list[MonitorSample]]) -> dict[str, Any]:
    flat = [sample for block in per_episode for sample in block]
    out: dict[str, Any] = {}
    for name, values in (
        ("mean_command", [s.mean_command for s in flat]),
        ("mean_state_response", [s.mean_state_response for s in flat]),
        ("mean_visual_response", [s.mean_visual_response for s in flat]),
    ):
        array = np.asarray(values, dtype=np.float64)
        out[name] = {
            f"p{q}": float(np.percentile(array, q)) for q in (5, 10, 25, 50, 75, 90, 99)
        }
        out[name]["max"] = float(array.max())
    out["samples"] = len(flat)
    return out


def candidate_thresholds(
    quantiles: dict[str, Any], baseline: MonitorEventThresholds
) -> dict[str, MonitorEventThresholds]:
    """Build candidates anchored to the observed distribution."""
    command = quantiles["mean_command"]
    state = quantiles["mean_state_response"]
    visual = quantiles["mean_visual_response"]

    candidates = {"deployed_default": baseline}
    # The deployed stall gate needs state_response below 0.002 while the observed
    # median is an order of magnitude larger. Re-anchor it, one quantile at a
    # time, so the effect of each move is separable.
    for label, value in (("p10", state["p10"]), ("p25", state["p25"]), ("p50", state["p50"])):
        candidates[f"stall_state_{label}"] = dataclasses_replace(
            baseline, state_response_threshold=value
        )
    # Move the visual gate off the median, where it contributes variance rather
    # than discrimination.
    for label, value in (("p10", visual["p10"]), ("p25", visual["p25"])):
        candidates[f"visual_{label}"] = dataclasses_replace(
            baseline,
            visual_response_threshold=value,
            idle_visual_response_threshold=value,
        )
    # A stall gate that is re-anchored on both axes at once.
    candidates["stall_recalibrated"] = dataclasses_replace(
        baseline,
        state_response_threshold=state["p25"],
        command_threshold=command["p90"],
        visual_response_threshold=visual["p25"],
    )
    # Both triggers re-anchored together, including the idle visual gate. This is
    # the only candidate that can admit stall and no_progress at the same time,
    # so it is the one worth proposing for closed-loop validation.
    for label, state_value in (("p25", state["p25"]), ("p50", state["p50"])):
        candidates[f"both_recalibrated_state_{label}"] = dataclasses_replace(
            baseline,
            state_response_threshold=state_value,
            command_threshold=command["p90"],
            visual_response_threshold=visual["p25"],
            idle_visual_response_threshold=visual["p25"],
        )
    return candidates


def dataclasses_replace(
    thresholds: MonitorEventThresholds, **changes: Any
) -> MonitorEventThresholds:
    return dataclasses.replace(thresholds, **changes)


def evidence_for_event(
    per_episode: list[list[MonitorSample]],
    outcomes: list[bool],
    thresholds: MonitorEventThresholds,
    event: str,
) -> list[EpisodeTriggerEvidence]:
    evidence: list[EpisodeTriggerEvidence] = []
    for index, block in enumerate(per_episode):
        if index >= len(outcomes):
            break
        events = recompute_monitor_events(block, thresholds)
        steps = tuple(
            block[position].timestep
            for position, value in enumerate(events)
            if value == event
        )
        evidence.append(
            EpisodeTriggerEvidence(
                episode_index=index,
                steps=max(sample.timestep for sample in block) + 1,
                success=outcomes[index],
                event_steps=steps,
            )
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-headroom-steps", type=int, default=200)
    parser.add_argument("--cooldown-steps", type=int, default=128)
    parser.add_argument("--max-occasions-per-episode", type=float, default=3.0)
    args = parser.parse_args()

    defaults = MonitorConfig()
    baseline = MonitorEventThresholds(
        command_threshold=defaults.command_threshold,
        state_response_threshold=defaults.state_response_threshold,
        visual_response_threshold=defaults.visual_response_threshold,
        idle_command_threshold=defaults.idle_command_threshold,
        idle_state_response_threshold=defaults.idle_state_response_threshold,
        idle_visual_response_threshold=defaults.idle_visual_response_threshold,
        stale_action_steps=defaults.stale_action_steps,
        # The deployed RoboDojo monitor sets this from the execute horizon.
        no_progress_steps=32,
    )
    requirements = TriggerAdmissionRequirements(
        minimum_recovery_headroom_steps=args.recovery_headroom_steps,
        cooldown_steps=args.cooldown_steps,
        maximum_occasions_per_episode=args.max_occasions_per_episode,
    )

    per_episode: list[list[MonitorSample]] = []
    outcomes: list[bool] = []
    sources: list[dict[str, Any]] = []
    for run in args.run:
        blocks, run_outcomes, metadata = load_run(run)
        usable = min(len(blocks), len(run_outcomes))
        metadata["episodes_used"] = usable
        per_episode.extend(blocks[:usable])
        outcomes.extend(run_outcomes[:usable])
        sources.append(metadata)
        print(
            f"{run.name}: {metadata['monitor_rows']} monitor rows, "
            f"{usable} episodes usable"
        )

    quantiles = signal_quantiles(per_episode)
    candidates = candidate_thresholds(quantiles, baseline)

    failed_indices = {index for index, ok in enumerate(outcomes) if not ok}
    results: dict[str, Any] = {}
    for label, thresholds in candidates.items():
        entry: dict[str, Any] = {"thresholds": thresholds.to_dict(), "events": {}}
        selected_failures: set[int] = set()
        selected_any: set[int] = set()
        for event in EVENTS:
            evidence = evidence_for_event(per_episode, outcomes, thresholds, event)
            decision = validate_trigger_admission(
                evidence, requirements, scope=f"monitor::{event}"
            )
            entry["events"][event] = decision.to_dict()
            for item in evidence:
                if item.fired:
                    selected_any.add(item.episode_index)
                    if not item.success:
                        selected_failures.add(item.episode_index)
        # Union coverage bounds what any intervention policy built on these
        # triggers could repair: a failure no trigger selects is never attempted.
        entry["union"] = {
            "episodes_selected": len(selected_any),
            "failures_selected": len(selected_failures),
            "failure_recall": (
                round(len(selected_failures) / len(failed_indices), 4)
                if failed_indices
                else None
            ),
            "precision": (
                round(len(selected_failures) / len(selected_any), 4)
                if selected_any
                else None
            ),
        }
        results[label] = entry

    base_failure = 1 - (sum(outcomes) / len(outcomes))
    report = {
        "schema_version": "carve.monitor-threshold-calibration.v1",
        "design": {
            "offline_exact": True,
            "replayed_from": "monitor_trace window means (threshold-independent)",
            "swept": "thresholds only; window_size/warmup would change the means",
            "candidates_anchored_to": "observed signal quantiles",
        },
        "sources": sources,
        "episodes": len(outcomes),
        "base_failure_rate": round(base_failure, 4),
        "signal_quantiles": quantiles,
        "requirements": requirements.to_dict(),
        "candidates": results,
        "claim_boundary": (
            "This decides which Monitor thresholds have earned the right to "
            "trigger an intervention, from past runs. It does not measure task "
            "success under a new threshold, and an admitted trigger still has "
            "to be validated in a closed loop before any success claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output}")

    header = (
        f"{'candidate':<29}{'event':<13}{'fires':>7}{'prec':>7}"
        f"{'recall':>8}{'lift':>7}{'occ':>6}  verdict"
    )
    print(f"\nepisodes={len(outcomes)}  base failure rate={base_failure:.2f}")
    print(header)
    print("-" * len(header))
    for label, entry in results.items():
        for event, decision in entry["events"].items():
            m = decision["metrics"]
            def fmt(value, spec=".2f"):
                return "-" if value is None else format(value, spec)
            print(
                f"{label:<29}{event:<13}{m['firing_episodes']:>3}/{m['episodes']:<3}"
                f"{fmt(m['failure_precision']):>7}{fmt(m['failure_recall']):>8}"
                f"{fmt(m['failure_lift']):>7}"
                f"{fmt(m['mean_occasions_per_firing_episode'], '.1f'):>6}  "
                f"{'ADMIT' if decision['accepted'] else 'reject'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
