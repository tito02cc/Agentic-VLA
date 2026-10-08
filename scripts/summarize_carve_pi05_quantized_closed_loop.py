#!/usr/bin/env python3
"""Summarize matched T6/T9 PI0.5 quantized-profile closed-loop gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/carve_optimize/pi05_late_int8_closed_loop_20260825"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_optimize/pi05_quantized_closed_loop_gate.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_metrics(paths: list[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    latencies = np.asarray(
        [float(record["runtime_latency_ms"]) for record in records],
        dtype=np.float64,
    )
    return {
        "calls": len(records),
        "runtime_p50_ms": float(np.percentile(latencies, 50)),
        "runtime_p95_ms": float(np.percentile(latencies, 95)),
        "runtime_p99_ms": float(np.percentile(latencies, 99)),
        "deadline_misses": sum(bool(record["deadline_miss"]) for record in records),
    }


def branch_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for record in load(path)["records"]:
            branch = record["branches"][0]
            rows.append(
                {
                    "snapshot": record["snapshot"],
                    "success": bool(branch["success_within_horizon"]),
                    "executed_steps": int(branch["executed_steps"]),
                    "vla_calls": int(branch["vla_calls"]),
                    "deadline_misses": int(branch["deadline_miss_count"]),
                    "video_path": branch["video_path"],
                    "source": str(path),
                }
            )
    return rows


def arm(
    root: Path,
    *,
    profile_id: str,
    result_names: list[str],
    trace_names: list[str],
    decision: str,
) -> dict[str, Any]:
    rows = branch_rows([root / name for name in result_names])
    return {
        "profile_id": profile_id,
        "closed_loop": {
            "successes": sum(row["success"] for row in rows),
            "trials": len(rows),
            "passed": all(row["success"] for row in rows),
            "rows": rows,
        },
        "runtime": trace_metrics([root / name for name in trace_names]),
        "decision": decision,
    }


def main() -> int:
    args = parse_args()
    arms = [
        arm(
            args.root,
            profile_id="pi05-torch_compile_masked_views-bf16-2step-h10",
            result_names=["smve_bf16_t6.json", "smve_bf16_t9.json"],
            trace_names=[
                "smve_bf16_t6_policy_calls.jsonl",
                "smve_bf16_t9_policy_calls.jsonl",
            ],
            decision="retain_realtime_default",
        ),
        arm(
            args.root,
            profile_id="pi05-torchao_int8-bf16-2step-h10-vlm_late",
            result_names=["late_int8_t6.json", "late_int8_t9.json"],
            trace_names=[
                "late_int8_t6_policy_calls.jsonl",
                "late_int8_t9_policy_calls.jsonl",
            ],
            decision="admit_as_low_memory_tier",
        ),
        arm(
            args.root,
            profile_id="pi05-torchao_int8_masked_views-bf16-2step-h10-vlm_late",
            result_names=["int8_smve_t69.json"],
            trace_names=["int8_smve_t69_policy_calls.jsonl"],
            decision="reject_closed_loop_regression",
        ),
    ]
    payload = {
        "schema_version": "carve-pi05-quantized-closed-loop-v1",
        "protocol": {
            "snapshots": [
                "task06_episode001_step0013",
                "task09_episode001_step0013",
            ],
            "branch": "physical_recovery",
            "branch_horizon": 440,
            "fixed_noise_seed": 7,
            "inference_steps": 2,
            "action_horizon": 10,
            "deadline_ms": 80.0,
        },
        "arms": arms,
        "default_profile_id": arms[0]["profile_id"],
        "low_memory_profile_id": arms[1]["profile_id"],
        "rejected_composition_profile_id": arms[2]["profile_id"],
        "conclusion": (
            "Late-language INT8 preserves both matched recovery outcomes and is "
            "admitted only as a low-memory tier. SMVE remains the realtime default. "
            "The composed INT8+SMVE profile fails T9 and is rejected."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CARVE PI0.5 Quantized Closed-Loop Gate",
        "",
        "## Protocol",
        "",
        "- Exact T6/T9 MuJoCo failure-state restoration.",
        "- Identical physical recovery, fixed policy noise, two flow steps and ten-action horizon.",
        "- One 440-step branch per state; 80 ms runtime deadline.",
        "- Profile composition is admitted independently; component-level passes do not imply composition safety.",
        "",
        "## Results",
        "",
        "| Profile | T6 | T9 | Calls | P50 | P95 | Misses | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in arms:
        rows = {row["snapshot"]: row for row in item["closed_loop"]["rows"]}
        t6 = rows["task06_episode001_step0013"]
        t9 = rows["task09_episode001_step0013"]
        lines.append(
            f"| {item['profile_id']} | "
            f"{'success' if t6['success'] else 'failure'} ({t6['executed_steps']}) | "
            f"{'success' if t9['success'] else 'failure'} ({t9['executed_steps']}) | "
            f"{item['runtime']['calls']} | {item['runtime']['runtime_p50_ms']:.2f} ms | "
            f"{item['runtime']['runtime_p95_ms']:.2f} ms | "
            f"{item['runtime']['deadline_misses']} | {item['decision']} |"
        )
    lines.extend(["", "## Decision", "", payload["conclusion"]])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "arms": len(arms)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
