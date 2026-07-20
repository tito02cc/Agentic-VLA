#!/usr/bin/env python3
"""Summarize CARVE action-prefix shadow diagnostics from episode traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _point_biserial(values: list[float], labels: list[int]) -> float | None:
    if len(values) != len(labels) or len(values) < 2:
        return None
    value_mean = _mean(values)
    label_mean = _mean([float(label) for label in labels])
    numerator = sum(
        (value - value_mean) * (label - label_mean)
        for value, label in zip(values, labels)
    )
    denominator = math.sqrt(
        sum((value - value_mean) ** 2 for value in values)
        * sum((label - label_mean) ** 2 for label in labels)
    )
    return numerator / denominator if denominator > 0 else None


def _auc(failure_scores: list[float], success_scores: list[float]) -> float | None:
    if not failure_scores or not success_scores:
        return None
    wins = 0.0
    for failure_score in failure_scores:
        for success_score in success_scores:
            if failure_score > success_score:
                wins += 1.0
            elif failure_score == success_score:
                wins += 0.5
    return wins / (len(failure_scores) * len(success_scores))


def analyze(trace_path: Path) -> dict[str, Any]:
    traces = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    episodes = []
    for trace in traces:
        prefix = trace["realtime"]["async_prefetch"]["prefix_consistency"]
        checks = int(prefix["checks"])
        reject_rate = float(prefix["would_reject"] / checks) if checks else 0.0
        episodes.append(
            {
                "episode_id": trace["episode_id"],
                "task_id": int(trace["task_id"]),
                "success": bool(trace["success"]),
                "checks": checks,
                "would_reject": int(prefix["would_reject"]),
                "reject_rate": reject_rate,
                "continuous_rms_mean": prefix["continuous_rms_mean"],
                "translation_endpoint_l2_p95": prefix[
                    "translation_endpoint_l2_p95"
                ],
                "rotation_endpoint_l2_p95": prefix["rotation_endpoint_l2_p95"],
                "gripper_agreement_mean": prefix["gripper_agreement_mean"],
            }
        )

    def group(success: bool) -> dict[str, Any]:
        rows = [episode for episode in episodes if episode["success"] is success]
        return {
            "episodes": len(rows),
            "reject_rate_mean": _mean([row["reject_rate"] for row in rows]),
            "continuous_rms_mean": _mean(
                [float(row["continuous_rms_mean"]) for row in rows]
            ),
            "translation_endpoint_l2_p95_mean": _mean(
                [float(row["translation_endpoint_l2_p95"]) for row in rows]
            ),
            "rotation_endpoint_l2_p95_mean": _mean(
                [float(row["rotation_endpoint_l2_p95"]) for row in rows]
            ),
            "gripper_agreement_mean": _mean(
                [float(row["gripper_agreement_mean"]) for row in rows]
            ),
        }

    reject_rates = [episode["reject_rate"] for episode in episodes]
    failure_labels = [0 if episode["success"] else 1 for episode in episodes]
    failure_scores = [
        episode["reject_rate"] for episode in episodes if not episode["success"]
    ]
    success_scores = [
        episode["reject_rate"] for episode in episodes if episode["success"]
    ]
    return {
        "source": str(trace_path),
        "episodes": episodes,
        "groups": {
            "success": group(True),
            "failure": group(False),
        },
        "diagnostic": {
            "failure_correlation": _point_biserial(reject_rates, failure_labels),
            "failure_auc": _auc(failure_scores, success_scores),
            "supports_enforcement": False,
            "decision": (
                "Do not enforce: action-prefix reject rate does not separate "
                "successful and failed episodes in this fixed-state diagnostic."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["diagnostic"], indent=2))


if __name__ == "__main__":
    main()
