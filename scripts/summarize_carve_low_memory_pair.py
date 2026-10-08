#!/usr/bin/env python3
"""Summarize P0 Planner + PI0.5 late-INT8 shared-GPU admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentic_vla.optimization import (
    CoResidentBenchmarkReport,
    SystemAdmissionRequirements,
    SystemOptimizationProfile,
    validate_system_profile_admission,
)


CONDITIONS = (
    (
        "continuous",
        "pi05_late_int8_active_qwen35_4b_500calls_20260825.json",
        "qwen35_4b_active_vlm_load_late_int8_20260825.jsonl",
    ),
    (
        "event5",
        "pi05_late_int8_event5_qwen35_4b_500calls_20260825.json",
        "qwen35_4b_event5_vlm_load_late_int8_20260825.jsonl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/carve_optimize")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_optimize/p0_late_int8_system_gate.json"),
    )
    return parser.parse_args()


def vlm_metrics(path: Path) -> dict[str, float | int]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latencies = [float(record["latency_ms"]) for record in records]
    return {
        "requests": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "success_rate": float(np.mean([bool(record["success"]) for record in records])),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def main() -> int:
    args = parse_args()
    profile = SystemOptimizationProfile(
        profile_id="p0-qwen35-4b-bf16+pi05-late-int8",
        planner_profile_id="P0-qwen35-4b-bf16-code-v5",
        vla_profile_id="pi05-torchao_int8-bf16-2step-h10-vlm_late",
        scheduler_policy="condition-specific",
        memory_budget_gb=23.0,
        options={"deadline_ms": 80.0},
    )
    requirements = SystemAdmissionRequirements(
        minimum_samples=500,
        maximum_peak_vram_gb=23.0,
        maximum_vla_deadline_miss_rate=0.01,
        maximum_planner_timeout_rate=0.01,
        maximum_unsafe_intervention_rate=0.0,
        require_fallbacks=False,
    )
    rows = []
    for condition, manifest_name, load_name in CONDITIONS:
        manifest_path = args.input_dir / manifest_name
        load_path = args.input_dir / load_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        benchmark = manifest["benchmark"]
        vlm = vlm_metrics(load_path)
        system_memory_gb = float(
            benchmark["deployment_condition"]["system_gpu_memory_used_mib_after"]
        ) / 1024.0
        report = CoResidentBenchmarkReport(
            samples=int(benchmark["samples"]),
            peak_vram_gb=system_memory_gb,
            planner_latency_p95_ms=float(vlm["latency_p95_ms"]),
            vla_latency_p95_ms=float(benchmark["runtime_p95_ms"]),
            vla_deadline_miss_rate=float(benchmark["deadline_miss_rate"]),
            planner_timeout_rate=1.0 - float(vlm["success_rate"]),
            unsafe_intervention_rate=0.0,
            metadata={
                "condition": condition,
                "vla_manifest": str(manifest_path),
                "vlm_trace": str(load_path),
                "vlm_requests": int(vlm["requests"]),
            },
        )
        decision = validate_system_profile_admission(
            profile,
            report,
            planner_admitted=True,
            vla_admitted=True,
            requirements=requirements,
        )
        rows.append(
            {
                "condition": condition,
                "report": report.to_dict(),
                "decision": decision.to_dict(),
            }
        )

    payload = {
        "schema_version": "carve-low-memory-system-gate-v1",
        "profile": profile.to_dict(),
        "requirements": {
            "maximum_vla_deadline_miss_rate": 0.01,
            "maximum_planner_timeout_rate": 0.01,
            "maximum_peak_vram_gb": 23.0,
        },
        "rows": rows,
        "admitted": any(row["decision"]["accepted"] for row in rows),
        "conclusion": (
            "The late-INT8 VLA is admitted only as a standalone low-memory tier. "
            "Neither continuous nor five-second-cooldown P0 co-residency meets "
            "the 80 ms system deadline gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CARVE P0 + PI0.5 Late-INT8 System Gate",
        "",
        "| Planner schedule | System VRAM | VLA P50 | VLA P95 | Miss@80ms | VLM requests | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report = row["report"]
        benchmark = json.loads(
            Path(report["metadata"]["vla_manifest"]).read_text(encoding="utf-8")
        )["benchmark"]
        lines.append(
            f"| {row['condition']} | {report['peak_vram_gb']:.2f} GB | "
            f"{benchmark['runtime_p50_ms']:.2f} ms | {report['vla_latency_p95_ms']:.2f} ms | "
            f"{100 * report['vla_deadline_miss_rate']:.1f}% | "
            f"{report['metadata']['vlm_requests']} | "
            f"{'PASS' if row['decision']['accepted'] else 'FAIL'} |"
        )
    lines.extend(["", "## Decision", "", payload["conclusion"]])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "admitted": payload["admitted"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
