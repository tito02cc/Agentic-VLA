#!/usr/bin/env python3
"""Summarize the paired CARVE PI0.5 recovery challenge."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from statistics import mean
from typing import Any


BRANCH_LABELS = {
    "continue": "Frozen VLA continuation",
    "accurate": "Frequent VLA replan",
    "recovery": "Agentic prompt retry",
    "physical_recovery": "Physical recovery + VLA replan",
}


def _load(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _profile_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = []
    violations = []
    for record in records:
        for branch in record["branches"]:
            profile = branch.get("optimization_profile")
            if not isinstance(profile, dict):
                if not (
                    int(branch.get("vla_calls", 0)) == 0
                    and branch.get("physical_recovery_error")
                    and branch.get("safe_stop") is True
                ):
                    violations.append(
                        f"{record['snapshot']}:{branch['branch']}:missing_profile"
                    )
                continue
            profile_id = profile.get("profile", {}).get("profile_id")
            profiles.append(profile_id)
            admission = profile.get("admission", {})
            if admission.get("accepted") is not True:
                violations.append(
                    f"{record['snapshot']}:{branch['branch']}:profile_not_admitted"
                )
    return {
        "passed": not violations and bool(profiles),
        "profile_ids": sorted(set(profiles), key=str),
        "violations": violations,
    }


def _physical_recovery_verified(row: dict[str, Any]) -> bool:
    outcome = (row.get("physical_recovery") or {}).get("outcome", {})
    return bool(outcome.get("verified", False)) or outcome.get("status") == "succeeded"


def summarize(branch_payload: dict[str, Any], online_payload: dict[str, Any]) -> dict[str, Any]:
    records = list(branch_payload["records"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scenarios = []
    for record in records:
        outcomes = {}
        for branch in record["branches"]:
            grouped[branch["branch"]].append(branch)
            outcomes[branch["branch"]] = bool(branch["success_within_horizon"])
        scenarios.append(
            {
                "snapshot": record["snapshot"],
                "task_id": int(record["task_id"]),
                "trigger": record["trigger"],
                "outcomes": outcomes,
                "safe_stops": {
                    branch["branch"]: bool(branch.get("safe_stop", False))
                    for branch in record["branches"]
                },
            }
        )

    branches = {}
    for branch_name, rows in grouped.items():
        calls = sum(int(row["vla_calls"]) for row in rows)
        deadline_calls = sum(int(row["deadline_call_count"]) for row in rows)
        deadline_misses = sum(int(row["deadline_miss_count"]) for row in rows)
        executed_rows = [
            row
            for row in rows
            if int(row.get("executed_steps", 0)) > 0
            or int(row.get("vla_calls", 0)) > 0
        ]
        branches[branch_name] = {
            "label": BRANCH_LABELS.get(branch_name, branch_name),
            "successes": sum(bool(row["success_within_horizon"]) for row in rows),
            "scenarios": len(rows),
            "vla_calls": calls,
            "mean_vla_calls": calls / len(rows),
            "executed_scenarios": len(executed_rows),
            "safe_stops": sum(bool(row.get("safe_stop", False)) for row in rows),
            "mean_executed_steps_when_executed": (
                mean(float(row["executed_steps"]) for row in executed_rows)
                if executed_rows
                else None
            ),
            "mean_wall_sec_when_executed": (
                mean(float(row["branch_wall_ms"]) for row in executed_rows) / 1000.0
                if executed_rows
                else None
            ),
            "deadline_misses": deadline_misses,
            "deadline_calls": deadline_calls,
            "deadline_miss_rate": deadline_misses / deadline_calls if deadline_calls else 0.0,
            "verified_physical_recoveries": sum(
                _physical_recovery_verified(row) for row in rows
            ),
        }

    trace_aggregate = online_payload.get("trace_aggregate", {})
    online_recovery = trace_aggregate.get("recovery", {})
    online = {
        "successes": int(online_payload.get("total_successes", 0)),
        "episodes": int(online_payload.get("total_episodes", 0)),
        "physical_triggered": int(
            online_recovery.get("physical_recoveries_triggered_total", 0)
        ),
        "physical_verified": int(
            online_recovery.get("physical_recoveries_verified_total", 0)
        ),
        "physical_actions": int(online_recovery.get("physical_recovery_actions_total", 0)),
        "vla_p95_ms": trace_aggregate.get("inference", {}).get("vla_latency_ms_p95"),
        "deadline_miss_rate": trace_aggregate.get("realtime", {}).get(
            "deadline_miss_rate_mean"
        ),
    }
    profile_audit = _profile_audit(records)
    lifecycle_passed = bool(
        online["successes"] > 0
        and online["physical_triggered"] > 0
        and online["physical_triggered"] == online["physical_verified"]
        and online["physical_actions"] > 0
    )
    return {
        "experiment": "CARVE-PI0.5-Recovery-Challenge",
        "claim": (
            "paired intervention efficacy and online recovery lifecycle in real "
            "LIBERO MuJoCo; not a benchmark-wide success-rate claim"
        ),
        "scenario_count": len(records),
        "profile_audit": profile_audit,
        "scenarios": scenarios,
        "branches": branches,
        "online_controller": online,
        "gates": {
            "profile_identity": profile_audit["passed"],
            "online_lifecycle": lifecycle_passed,
            "complete": profile_audit["passed"] and lifecycle_passed,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CARVE PI0.5 Recovery Challenge",
        "",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Profile audit: `{'PASS' if summary['profile_audit']['passed'] else 'FAIL'}`",
        f"- Online lifecycle: `{'PASS' if summary['gates']['online_lifecycle'] else 'FAIL'}`",
        f"- Overall gate: `{'PASS' if summary['gates']['complete'] else 'FAIL'}`",
        "",
        "This is a paired intervention and systems experiment in real LIBERO MuJoCo, "
        "not a benchmark-wide success-rate estimate.",
        "",
        "## Paired Scenarios",
        "",
        "| Snapshot | Task | Trigger | Continue | Replan | Prompt retry | Physical recovery |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["scenarios"]:
        outcomes = row["outcomes"]

        def mark(name: str) -> str:
            if row["safe_stops"].get(name):
                return "safe stop"
            return "success" if outcomes.get(name) else "failure"

        lines.append(
            f"| `{row['snapshot']}` | {row['task_id']} | {row['trigger']} | "
            f"{mark('continue')} | {mark('accurate')} | {mark('recovery')} | "
            f"{mark('physical_recovery')} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Cost",
            "",
            "Costs are averaged only over branches that actually executed actions. "
            "A fail-closed safe stop is reported separately and never treated as zero-cost execution.",
            "",
            "| Intervention | Success | Executed | Safe stop | VLA calls | Mean steps (executed) | Mean wall (executed) | Deadline miss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("continue", "accurate", "recovery", "physical_recovery"):
        row = summary["branches"][name]
        mean_steps = row["mean_executed_steps_when_executed"]
        mean_wall = row["mean_wall_sec_when_executed"]
        mean_steps_text = f"{mean_steps:.1f}" if mean_steps is not None else "n/a"
        mean_wall_text = f"{mean_wall:.2f} s" if mean_wall is not None else "n/a"
        lines.append(
            f"| {row['label']} | {row['successes']}/{row['scenarios']} | "
            f"{row['executed_scenarios']}/{row['scenarios']} | {row['safe_stops']} | "
            f"{row['vla_calls']} | {mean_steps_text} | {mean_wall_text} | "
            f"{row['deadline_misses']}/{row['deadline_calls']} |"
        )
    physical = summary["branches"]["physical_recovery"]
    lines.extend(
        [
            "",
            "Physical-skill verification: "
            f"`{physical['verified_physical_recoveries']}/"
            f"{physical['executed_scenarios']}` executed recovery branches.",
        ]
    )
    online = summary["online_controller"]
    lines.extend(
        [
            "",
            "## Online Controller Sentinel",
            "",
            f"- Task success: `{online['successes']}/{online['episodes']}`",
            f"- Physical recovery: `{online['physical_verified']}/{online['physical_triggered']}` verified",
            f"- Physical actions: `{online['physical_actions']}`",
            f"- VLA P95: `{online['vla_p95_ms']:.2f} ms`",
            f"- Control deadline-miss rate: `{100.0 * online['deadline_miss_rate']:.2f}%`",
            "",
            "The branch study isolates intervention effects under exact state restoration; "
            "the online sentinel separately verifies automatic monitor-controller-recovery "
            "execution without simulator state entering the controller.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=pathlib.Path, required=True)
    parser.add_argument("--online", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    args = parser.parse_args()

    summary = summarize(_load(args.branches), _load(args.online))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["gates"], indent=2))
    return 0 if summary["gates"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
