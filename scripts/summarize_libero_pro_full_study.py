#!/usr/bin/env python3
"""Summarize the complete paired CARVE LIBERO-Pro study."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

from scripts.summarize_libero_pro_canonical_pair import _method_summary


METHODS = ("frozen_vla", "fixed_recovery", "agentic")
LABELS = {
    "frozen_vla": "Frozen VLA",
    "fixed_recovery": "Fixed Recovery",
    "agentic": "Full Agentic",
}


def _load(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def _percent_change(current: float, reference: float) -> float | None:
    return None if reference == 0 else 100.0 * (current - reference) / reference


def _mcnemar_exact(conversions: int, regressions: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    discordant = conversions + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(conversions, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _event_latencies(root: pathlib.Path) -> tuple[list[float], list[float]]:
    vla: list[float] = []
    planner: list[float] = []
    for events_path in root.glob("*/events.jsonl"):
        for line in events_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event_type") == "tool_result" and row.get("payload", {}).get("tool") == "vla_act":
                metadata = row["payload"]["output"]["primitive_outcome"]["metadata"]
                vla.append(float(metadata["runtime_latency_ms"]))
            elif row.get("event_type") == "planner_result":
                planner.append(float(row["payload"]["elapsed_ms"]))
    return vla, planner


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _empty_aggregate() -> dict[str, Any]:
    return {
        "successes": 0,
        "trials": 0,
        "episode_steps": 0,
        "episode_wall_s": 0.0,
        "vla_calls": 0,
        "vla_deadline_misses": 0,
        "planner_calls": 0,
        "planner_wall_ms": 0.0,
        "recovery_calls": 0,
        "safe_stops": 0,
        "vla_latencies_ms": [],
        "planner_latencies_ms": [],
    }


def _finalize(row: dict[str, Any]) -> dict[str, Any]:
    vla = row.pop("vla_latencies_ms")
    planner = row.pop("planner_latencies_ms")
    row["success_rate"] = row["successes"] / row["trials"]
    row["success_rate_wilson95"] = _wilson(row["successes"], row["trials"])
    row["vla_runtime_p50_ms"] = _percentile(vla, 0.50)
    row["vla_runtime_p95_ms"] = _percentile(vla, 0.95)
    row["vla_runtime_p99_ms"] = _percentile(vla, 0.99)
    row["planner_mean_ms"] = statistics.fmean(planner) if planner else 0.0
    row["planner_p95_ms"] = _percentile(planner, 0.95) or 0.0
    row["vla_deadline_miss_rate"] = (
        row["vla_deadline_misses"] / row["vla_calls"] if row["vla_calls"] else 0.0
    )
    row["planner_to_vla_call_ratio"] = (
        row["planner_calls"] / row["vla_calls"] if row["vla_calls"] else 0.0
    )
    row["vla_calls_per_success"] = (
        row["vla_calls"] / row["successes"] if row["successes"] else None
    )
    return row


def summarize(study_root: pathlib.Path) -> dict[str, Any]:
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for summary_path in sorted(study_root.glob("*/*/task_*/summary.json")):
        aggregate = _load(summary_path)
        suite = str(aggregate["suite"])
        task_id = int(aggregate["task_id"])
        method = str(aggregate["method"])
        if method not in METHODS:
            continue
        key = (suite, task_id, method)
        if key in cells:
            raise ValueError(f"duplicate result cell: {key}")
        row = _method_summary(summary_path.parent)
        vla, planner = _event_latencies(summary_path.parent)
        row["vla_latencies_ms"] = vla
        row["planner_latencies_ms"] = planner
        cells[key] = row
    identities = sorted({(suite, task_id) for suite, task_id, _ in cells})
    missing = [
        (suite, task_id, method)
        for suite, task_id in identities
        for method in METHODS
        if (suite, task_id, method) not in cells
    ]
    if missing:
        raise ValueError(f"incomplete method cells: {missing[:10]}")

    suite_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {method: _empty_aggregate() for method in METHODS}
    )
    totals = {method: _empty_aggregate() for method in METHODS}
    task_rows = []
    paired = {
        method: {
            "conversions": 0,
            "regressions": 0,
            "preserved_successes": 0,
            "recovery_episodes": 0,
            "recovery_successes": 0,
            "planner_episodes": 0,
        }
        for method in ("fixed_recovery", "agentic")
    }
    additive = (
        "successes", "trials", "episode_steps", "episode_wall_s", "vla_calls",
        "vla_deadline_misses", "planner_calls", "planner_wall_ms", "recovery_calls",
        "safe_stops",
    )
    for suite, task_id in identities:
        task_methods = {method: cells[(suite, task_id, method)] for method in METHODS}
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "methods": {
                    method: {
                        "successes": task_methods[method]["successes"],
                        "trials": task_methods[method]["trials"],
                    }
                    for method in METHODS
                },
            }
        )
        frozen_trials = task_methods["frozen_vla"]["per_trial"]
        for method in ("fixed_recovery", "agentic"):
            candidate_trials = task_methods[method]["per_trial"]
            if [row["trial"] for row in frozen_trials] != [
                row["trial"] for row in candidate_trials
            ]:
                raise ValueError(f"paired trial identity mismatch: {suite} task={task_id}")
            for frozen, candidate in zip(frozen_trials, candidate_trials, strict=True):
                if not frozen["success"] and candidate["success"]:
                    paired[method]["conversions"] += 1
                elif frozen["success"] and not candidate["success"]:
                    paired[method]["regressions"] += 1
                elif frozen["success"] and candidate["success"]:
                    paired[method]["preserved_successes"] += 1
                if candidate["recovery_attempts"] > 0:
                    paired[method]["recovery_episodes"] += 1
                    paired[method]["recovery_successes"] += int(candidate["success"])
                if candidate["planner_calls"] > 0:
                    paired[method]["planner_episodes"] += 1
        for method in METHODS:
            source = task_methods[method]
            for target in (suite_rows[suite][method], totals[method]):
                for key in additive:
                    target[key] += source[key]
                target["vla_latencies_ms"].extend(source["vla_latencies_ms"])
                target["planner_latencies_ms"].extend(source["planner_latencies_ms"])

    finalized_suites = {
        suite: {method: _finalize(dict(rows[method])) for method in METHODS}
        for suite, rows in sorted(suite_rows.items())
    }
    finalized_totals = {method: _finalize(dict(totals[method])) for method in METHODS}
    frozen_total = finalized_totals["frozen_vla"]
    comparisons = {}
    for method in ("fixed_recovery", "agentic"):
        row = finalized_totals[method]
        comparisons[method] = {
            "success_delta": row["successes"] - frozen_total["successes"],
            "success_rate_points": 100.0 * (row["success_rate"] - frozen_total["success_rate"]),
            "episode_steps_percent": _percent_change(row["episode_steps"], frozen_total["episode_steps"]),
            "vla_calls_percent": _percent_change(row["vla_calls"], frozen_total["vla_calls"]),
            "episode_wall_percent": _percent_change(row["episode_wall_s"], frozen_total["episode_wall_s"]),
            **paired[method],
        }
        comparisons[method]["mcnemar_exact_p"] = _mcnemar_exact(
            paired[method]["conversions"], paired[method]["regressions"]
        )
        comparisons[method]["recovery_success_rate"] = (
            paired[method]["recovery_successes"] / paired[method]["recovery_episodes"]
            if paired[method]["recovery_episodes"]
            else None
        )
    fixed_total = finalized_totals["fixed_recovery"]
    agentic_total = finalized_totals["agentic"]
    agentic_vs_fixed = {"conversions": 0, "regressions": 0, "preserved_successes": 0}
    for suite, task_id in identities:
        fixed_trials = cells[(suite, task_id, "fixed_recovery")]["per_trial"]
        agentic_trials = cells[(suite, task_id, "agentic")]["per_trial"]
        for fixed, agentic in zip(fixed_trials, agentic_trials, strict=True):
            if not fixed["success"] and agentic["success"]:
                agentic_vs_fixed["conversions"] += 1
            elif fixed["success"] and not agentic["success"]:
                agentic_vs_fixed["regressions"] += 1
            elif fixed["success"] and agentic["success"]:
                agentic_vs_fixed["preserved_successes"] += 1
    comparisons["agentic_vs_fixed"] = {
        "success_delta": agentic_total["successes"] - fixed_total["successes"],
        "success_rate_points": 100.0
        * (agentic_total["success_rate"] - fixed_total["success_rate"]),
        "episode_steps_percent": _percent_change(
            agentic_total["episode_steps"], fixed_total["episode_steps"]
        ),
        "vla_calls_percent": _percent_change(
            agentic_total["vla_calls"], fixed_total["vla_calls"]
        ),
        "episode_wall_percent": _percent_change(
            agentic_total["episode_wall_s"], fixed_total["episode_wall_s"]
        ),
        "planner_calls": agentic_total["planner_calls"],
        "planner_wall_ms": agentic_total["planner_wall_ms"],
        **agentic_vs_fixed,
        "mcnemar_exact_p": _mcnemar_exact(
            agentic_vs_fixed["conversions"], agentic_vs_fixed["regressions"]
        ),
    }
    return {
        "protocol": "libero_pro_full_suite_paired_10state_v1",
        "cell_count": len(cells),
        "task_count": len(identities),
        "episode_count": sum(row["trials"] for row in finalized_totals.values()),
        "tasks": task_rows,
        "suites": finalized_suites,
        "aggregate": finalized_totals,
        "comparisons_to_frozen": comparisons,
        "claim_boundary": (
            "Complete 10-state task coverage on the locally available LIBERO-10 and "
            "LIBERO-Pro Object/Position/Task suites; not the official 50-state leaderboard."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LIBERO-Pro Full-Suite Paired Study",
        "",
        "## Suite success",
        "",
        "| Suite | Frozen VLA | Fixed Recovery | Full Agentic |",
        "|---|---:|---:|---:|",
    ]
    for suite, methods in summary["suites"].items():
        cells = [f"{methods[m]['successes']}/{methods[m]['trials']}" for m in METHODS]
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | Success | Steps | VLA calls | P95/P99 | Miss@80 ms | Planner | Recovery | Safe stop | Wall time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = summary["aggregate"][method]
        lines.append(
            f"| {LABELS[method]} | {row['successes']}/{row['trials']} | {row['episode_steps']} | "
            f"{row['vla_calls']} | {row['vla_runtime_p95_ms']:.2f}/{row['vla_runtime_p99_ms']:.2f} ms | "
            f"{row['vla_deadline_misses']} | {row['planner_calls']} | {row['recovery_calls']} | "
            f"{row['safe_stops']} | {row['episode_wall_s']:.1f} s |"
        )
    lines.extend(["", "## Paired comparison", ""])
    for method in ("fixed_recovery", "agentic"):
        row = summary["comparisons_to_frozen"][method]
        lines.append(
            f"- **{LABELS[method]}**: success delta `{row['success_delta']}` "
            f"(`{row['success_rate_points']:+.2f}` points), conversions `{row['conversions']}`, "
            f"regressions `{row['regressions']}`, exact McNemar `p={row['mcnemar_exact_p']:.4g}`, "
            f"recovery success `{row['recovery_successes']}/{row['recovery_episodes']}`, "
            f"Planner episodes `{row['planner_episodes']}`, steps "
            f"`{row['episode_steps_percent']:+.1f}%`, VLA calls `{row['vla_calls_percent']:+.1f}%`."
        )
    direct = summary["comparisons_to_frozen"]["agentic_vs_fixed"]
    lines.append(
        f"- **Full Agentic vs. Fixed Recovery**: success delta `{direct['success_delta']}` "
        f"(`{direct['success_rate_points']:+.2f}` points), conversions "
        f"`{direct['conversions']}`, regressions `{direct['regressions']}`, exact McNemar "
        f"`p={direct['mcnemar_exact_p']:.4g}`, steps "
        f"`{direct['episode_steps_percent']:+.1f}%`, wall time "
        f"`{direct['episode_wall_percent']:+.1f}%`."
    )
    lines.extend(["", summary["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = summarize(args.study_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "study_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "REPORT.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "cell_count": result["cell_count"],
        "task_count": result["task_count"],
        "episode_count": result["episode_count"],
        "aggregate": result["aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
