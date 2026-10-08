#!/usr/bin/env python
"""Audit whether an intervention policy can act inside its own budget.

``calibrate_monitor_thresholds.py`` answers "which thresholds have earned the
right to trigger". This answers the prior question that calibration exposed:
given a threshold set *and* a budget, cooldown, check interval and unchanged-span
requirement, can a recovery fire at all?

The four gates are configured independently. The deployed pairing needs the third
counted check to accumulate a 320-step span while the call budget is three, so
the window is exactly one check wide. That is the kind of coupling this audit is
for: it is invisible when each knob is reviewed alone.

Recovery feasibility here is an upper bound -- it assumes the predicate reading
never changes, i.e. the scene is maximally stall-like. A policy that cannot act
even then cannot act at all.

Usage:
    openpi/.venv/bin/python scripts/audit_intervention_policy_feasibility.py \
        --run artifacts/robodojo/ral_matrix_20260907_stack_bowls_c2_set0 \
        --run artifacts/robodojo/ral_matrix_20260907_stack_bowls_c3_set0 \
        --output artifacts/robodojo/<dir>/policy_feasibility.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentic_vla.toolchain import (  # noqa: E402
    InterventionPolicy,
    MonitorEventThresholds,
    recompute_monitor_events,
    simulate_episode_policy,
    validate_policy_feasibility,
)

# Reuse the loader and the recommended thresholds from the calibration tool
# rather than restating either.
_SPEC = importlib.util.spec_from_file_location(
    "calibrate_monitor_thresholds", REPO_ROOT / "scripts/calibrate_monitor_thresholds.py"
)
calibrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate)


def build_policies(baseline: InterventionPolicy) -> dict[str, InterventionPolicy]:
    """Candidate policies, varying only one coupled knob at a time."""
    return {
        "deployed": baseline,
        "budget_4": dataclasses.replace(baseline, max_calls_per_episode=4),
        "budget_5": dataclasses.replace(baseline, max_calls_per_episode=5),
        # Halve the check interval so the span accumulates in fewer wall steps,
        # holding the span requirement fixed.
        "checkpoint_80": dataclasses.replace(baseline, checkpoint_steps=80),
        "checkpoint_80_budget_5": dataclasses.replace(
            baseline, checkpoint_steps=80, max_calls_per_episode=5
        ),
        # Relax the span instead of the budget.
        "span_240": dataclasses.replace(baseline, min_unchanged_steps=240),
        "span_160": dataclasses.replace(baseline, min_unchanged_steps=160),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event", default="no_progress", choices=["stall", "no_progress"],
        help="event to inspect; stall does not force checks in the current bridge",
    )
    parser.add_argument("--minimum-feasible-rate", type=float, default=0.5)
    args = parser.parse_args()

    per_episode: list[list[Any]] = []
    outcomes: list[bool] = []
    sources: list[dict[str, Any]] = []
    for run in args.run:
        blocks, run_outcomes, metadata = calibrate.load_run(run)
        usable = min(len(blocks), len(run_outcomes))
        metadata["episodes_used"] = usable
        per_episode.extend(blocks[:usable])
        outcomes.extend(run_outcomes[:usable])
        sources.append(metadata)
        print(f"{run.name}: {usable} episodes usable")

    # The recommended threshold set from calibration, rebuilt from the same
    # quantiles so the two reports cannot drift apart.
    quantiles = calibrate.signal_quantiles(per_episode)
    state = quantiles["mean_state_response"]
    command = quantiles["mean_command"]
    visual = quantiles["mean_visual_response"]
    thresholds = MonitorEventThresholds(
        state_response_threshold=state["p50"],
        command_threshold=command["p90"],
        visual_response_threshold=visual["p25"],
        idle_visual_response_threshold=visual["p25"],
    )
    print(
        "thresholds (both_recalibrated_state_p50): "
        f"state={thresholds.state_response_threshold:.5f} "
        f"command={thresholds.command_threshold:.5f} "
        f"visual={thresholds.visual_response_threshold:.5f}"
    )

    event_steps_by_episode: list[tuple[int, ...]] = []
    for block in per_episode:
        events = recompute_monitor_events(block, thresholds)
        event_steps_by_episode.append(
            tuple(
                block[index].timestep
                for index, value in enumerate(events)
                if value == args.event
            )
        )
    episode_steps = [
        max(sample.timestep for sample in block) + 1 for block in per_episode
    ]

    baseline = InterventionPolicy()
    results: dict[str, Any] = {}
    for label, policy in build_policies(baseline).items():
        episodes = [
            simulate_episode_policy(
                episode_index=index,
                steps=episode_steps[index],
                success=outcomes[index],
                event_steps=event_steps_by_episode[index],
                policy=policy,
                event_name=args.event,
            )
            for index in range(len(outcomes))
        ]
        decision = validate_policy_feasibility(
            episodes,
            minimum_feasible_rate=args.minimum_feasible_rate,
            scope=f"policy::{label}",
        )
        results[label] = {
            "policy": policy.to_dict(),
            "decision": decision.to_dict(),
            "episodes": [item.to_dict() for item in episodes],
        }

    report = {
        "schema_version": "carve.intervention-policy-feasibility.v1",
        "design": {
            "offline_exact": False,
            "deployment_admission": False,
            "scope": "conditional pre-recovery scheduling with recalibrated thresholds",
            "event_routed_to_scalar_critic": args.event == "no_progress",
            "forced_check_event": args.event,
            "thresholds": thresholds.to_dict(),
            "feasibility_is_an_upper_bound": (
                "assumes the predicate never changes; a policy that cannot act "
                "under that assumption cannot act at all"
            ),
        },
        "sources": sources,
        "episode_count": len(outcomes),
        "base_failure_rate": round(1 - sum(outcomes) / len(outcomes), 4),
        "candidates": results,
        "claim_boundary": (
            "This decides whether a configuration leaves room to intervene. It "
            "does not measure whether intervening helps, and an accepted policy "
            "still requires closed-loop validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output}")

    header = (
        f"{'policy':<24}{'ckpt':>6}{'budget':>8}{'span':>6}"
        f"{'reachable':>11}{'in failures':>13}{'checks/ep':>11}"
        f"{'earliest':>10}  verdict"
    )
    print(f"\nepisodes={len(outcomes)}  forced-check event={args.event}")
    print(header)
    print("-" * len(header))
    for label, entry in results.items():
        p, m = entry["policy"], entry["decision"]["metrics"]
        print(
            f"{label:<24}{p['checkpoint_steps']:>6}{p['max_calls_per_episode']:>8}"
            f"{p['min_unchanged_steps']:>6}"
            f"{m['feasible_episodes']:>4}/{m['episodes']:<6}"
            f"{m['feasible_failed_episodes']:>6}/{m['failed_episodes']:<6}"
            f"{m['mean_checks_per_episode']:>11.2f}"
            f"{str(m['earliest_recovery_step_median']):>10}  "
            f"{'FEASIBLE' if entry['decision']['accepted'] else 'infeasible'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
