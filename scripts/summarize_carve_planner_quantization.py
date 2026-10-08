#!/usr/bin/env python3
"""Summarize the fixed CARVE Planner P0-P3 quantization comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_INPUTS = (
    (
        "P0",
        "Qwen3.5-4B BF16",
        "qwen35_4b_bf16_receipt.json",
        "qwen35_4b_bf16_semantic_overlay_gate_v5.json",
    ),
    (
        "P1",
        "Qwen3.5-9B BF16",
        "qwen35_9b_bf16_receipt.json",
        "qwen35_9b_bf16_semantic_overlay_gate_v5.json",
    ),
    (
        "P2",
        "Qwen3.5-9B uniform NF4",
        "qwen35_9b_uniform_nf4_receipt.json",
        "qwen35_9b_uniform_nf4_semantic_overlay_gate_v5.json",
    ),
    (
        "P3",
        "Qwen3.5-9B vision-preserving NF4",
        "qwen35_9b_vision_preserving_nf4_receipt.json",
        "qwen35_9b_vision_preserving_nf4_semantic_overlay_gate_v5.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/carve_quantization"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_quantization/planner_quantization_gate.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for profile_id, label, receipt_name, gate_name in PROFILE_INPUTS:
        receipt = load(args.input_dir / receipt_name)
        gate = load(args.input_dir / gate_name)
        metrics = gate["metrics"]
        memory = receipt["cuda_memory_after_load"]
        rows.append(
            {
                "profile_id": profile_id,
                "label": label,
                "deployment_precision": receipt["deployment_precision"],
                "module_precisions": receipt["module_precisions"],
                "preserved_modules": receipt["preserved_modules"],
                "allocated_vram_gib": float(memory["allocated_gib"]),
                "peak_allocated_vram_gib": float(memory["peak_allocated_gib"]),
                "semantic_latency_mean_ms": float(metrics["latency_ms_mean"]),
                "semantic_latency_p95_ms": float(metrics["latency_ms_p95"]),
                "protocol_valid_rate": float(metrics["protocol_valid_rate"]),
                "no_op_false_positive_rate": float(metrics["no_op_false_positive_rate"]),
                "severe_refresh_recall": float(metrics["severe_intervention_recall"]),
                "mild_refresh_rate": float(metrics["mild_intervention_rate"]),
                "semantic_gate_passed": bool(gate["passed"]),
                "receipt": str(args.input_dir / receipt_name),
                "semantic_gate": str(args.input_dir / gate_name),
            }
        )

    reference = rows[0]
    for row in rows:
        row["vram_change_vs_p0_percent"] = percent_change(
            row["allocated_vram_gib"], reference["allocated_vram_gib"]
        )
        row["p95_change_vs_p0_percent"] = percent_change(
            row["semantic_latency_p95_ms"], reference["semantic_latency_p95_ms"]
        )

    decisions = {
        "P0": "retain_as_default",
        "P1": "reject_for_24gb_co_residency_without_additional_semantic_gain",
        "P2": "retain_as_memory_candidate_only; no latency or semantic advantage",
        "P3": "retain_as_ablation; vision preservation showed no gate advantage",
    }
    for row in rows:
        row["decision"] = decisions[row["profile_id"]]

    payload = {
        "schema_version": "carve-planner-quantization-gate-v1",
        "protocol": "code_v5_short_evidence_with_deployable_change_overlay",
        "samples_per_profile": 30,
        "hardware": "NVIDIA GeForce RTX 4090 24GB",
        "rows": rows,
        "promoted_profile_id": "P0",
        "conclusion": (
            "NF4 is a memory-capacity option in the current Transformers/"
            "bitsandbytes stack, not a latency optimization. Qwen3.5-4B BF16 "
            "remains the default Planner until a kernel-backed 4-bit profile "
            "demonstrates semantic non-inferiority and lower P95."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CARVE Planner Quantization Gate",
        "",
        "## Protocol",
        "",
        "- 30 paired temporal observations: 10 no-op, 10 mild, 10 severe.",
        "- Identical `code_v5` short-evidence prompt and deployable RGB change overlay.",
        "- No simulator displacement values or evaluator labels reach the Planner.",
        "- CUDA allocated memory is measured immediately after model load; semantic latency includes request preprocessing and generation.",
        "",
        "## Results",
        "",
        "| Profile | Allocated VRAM | P95 | Valid | No-op FP | Severe recall | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile_id']} {row['label']} | "
            f"{row['allocated_vram_gib']:.2f} GiB | "
            f"{row['semantic_latency_p95_ms']:.1f} ms | "
            f"{100 * row['protocol_valid_rate']:.0f}% | "
            f"{100 * row['no_op_false_positive_rate']:.0f}% | "
            f"{100 * row['severe_refresh_recall']:.0f}% | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["conclusion"],
            "",
            "P2 reduces allocated model memory relative to 9B BF16, but its current "
            "BitsAndBytes kernels increase P95. P3 preserves the visual stack in BF16 "
            "but shows no semantic advantage over P2 on this gate. Neither is promoted "
            "as the default or described as a realtime acceleration.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "promoted_profile_id": "P0"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
