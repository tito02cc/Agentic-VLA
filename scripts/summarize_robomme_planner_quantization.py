#!/usr/bin/env python3
"""Summarize paired BF16 and NF4 GroundSG Planner rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--nf4-root", type=Path, required=True)
    parser.add_argument("--bf16-vram-mib", type=float, required=True)
    parser.add_argument("--nf4-vram-mib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unique_stages(trace: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item["history_text"])
            for item in trace
            if item.get("history_text")
        )
    )


def load(root: Path) -> dict[int, dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("MoveCube_ep*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = int(payload["episode"])
        trace = payload["planner_trace"]
        episodes[episode] = {
            "episode": episode,
            "success": bool(payload["success"]),
            "status": str(payload["status"]),
            "executed_steps": int(payload["executed_steps"]),
            "policy_calls": int(payload["policy_calls"]),
            "planner_calls": int(payload["planner_calls"]),
            "policy_p95_ms": float(payload["policy_latency_ms"]["p95"]),
            "planner_p95_ms": float(payload["planner_latency_ms"]["p95"]),
            "wall_time_s": float(payload["wall_time_s"]),
            "stage_sequence": unique_stages(trace),
            "trace": trace,
            "summary": str(path),
        }
    if not episodes:
        raise ValueError(f"no summaries found in {root}")
    return episodes


def summarize(episodes: dict[int, dict[str, Any]], vram_mib: float) -> dict[str, Any]:
    rows = list(episodes.values())
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "idle_planner_vram_mib": vram_mib,
        "mean_episode_policy_p95_ms": statistics.fmean(
            row["policy_p95_ms"] for row in rows
        ),
        "mean_episode_planner_p95_ms": statistics.fmean(
            row["planner_p95_ms"] for row in rows
        ),
        "mean_steps": statistics.fmean(row["executed_steps"] for row in rows),
        "mean_wall_time_s": statistics.fmean(row["wall_time_s"] for row in rows),
    }


def common_point_error(
    bf16_trace: list[dict[str, Any]], nf4_trace: list[dict[str, Any]]
) -> tuple[list[float], int]:
    bf16_by_step = {int(item["step"]): item for item in bf16_trace}
    nf4_by_step = {int(item["step"]): item for item in nf4_trace}
    errors: list[float] = []
    matched_steps = 0
    for step in sorted(bf16_by_step.keys() & nf4_by_step.keys()):
        lhs = bf16_by_step[step]
        rhs = nf4_by_step[step]
        if lhs.get("history_text") != rhs.get("history_text"):
            continue
        lhs_points = lhs.get("points", [])
        rhs_points = rhs.get("points", [])
        if len(lhs_points) != len(rhs_points):
            continue
        matched_steps += 1
        for lhs_point, rhs_point in zip(lhs_points, rhs_points):
            errors.extend(abs(float(a) - float(b)) for a, b in zip(lhs_point, rhs_point))
    return errors, matched_steps


def main() -> int:
    args = parse_args()
    bf16 = load(args.bf16_root)
    nf4 = load(args.nf4_root)
    if bf16.keys() != nf4.keys():
        raise ValueError("BF16 and NF4 episode identities do not match")

    all_errors: list[float] = []
    stage_matches = 0
    outcome_matches = 0
    paired = []
    for episode in sorted(bf16):
        lhs = bf16[episode]
        rhs = nf4[episode]
        errors, matched_steps = common_point_error(lhs["trace"], rhs["trace"])
        all_errors.extend(errors)
        stage_match = lhs["stage_sequence"] == rhs["stage_sequence"]
        stage_matches += stage_match
        outcome_matches += lhs["success"] == rhs["success"]
        paired.append(
            {
                "episode": episode,
                "bf16_success": lhs["success"],
                "nf4_success": rhs["success"],
                "stage_sequence_match": stage_match,
                "bf16_stage_sequence": lhs["stage_sequence"],
                "nf4_stage_sequence": rhs["stage_sequence"],
                "matched_trace_steps": matched_steps,
                "mean_abs_grounding_coordinate_error_px": (
                    statistics.fmean(errors) if errors else None
                ),
                "bf16_planner_p95_ms": lhs["planner_p95_ms"],
                "nf4_planner_p95_ms": rhs["planner_p95_ms"],
            }
        )

    bf16_summary = summarize(bf16, args.bf16_vram_mib)
    nf4_summary = summarize(nf4, args.nf4_vram_mib)
    output = {
        "schema_version": "carve.robomme.planner_quantization.v1",
        "benchmark": "RoboMME",
        "task": "MoveCube",
        "claim_boundary": (
            "Post-training bitsandbytes NF4 of the Qwen3-VL base with the same "
            "GroundSG LoRA and PI0.5; quality is judged by full closed-loop "
            "episodes. The BF16 Planner request queue was isolated, but a separate "
            "Qwen process shared the GPU during part of that rerun, so the latency "
            "ratio is a conservative observation rather than an empty-GPU benchmark."
        ),
        "bf16": bf16_summary,
        "nf4": nf4_summary,
        "comparison": {
            "idle_planner_vram_reduction_ratio": 1.0
            - args.nf4_vram_mib / args.bf16_vram_mib,
            "planner_p95_ratio_nf4_over_bf16": (
                nf4_summary["mean_episode_planner_p95_ms"]
                / bf16_summary["mean_episode_planner_p95_ms"]
            ),
            "success_rate_delta_nf4_minus_bf16": (
                nf4_summary["success_rate"] - bf16_summary["success_rate"]
            ),
            "stage_sequence_agreement_rate": stage_matches / len(paired),
            "outcome_agreement_rate": outcome_matches / len(paired),
            "mean_abs_grounding_coordinate_error_px": statistics.fmean(all_errors),
            "quality_gate_passed": nf4_summary["success_rate"] >= bf16_summary["success_rate"],
            "deployment_decision": (
                "capacity_only_reject_as_default"
                if nf4_summary["success_rate"] < bf16_summary["success_rate"]
                else "admit"
            ),
        },
        "paired_episodes": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
