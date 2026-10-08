#!/usr/bin/env python3
"""Evaluate CARVE's short-label semantic observer on labeled temporal pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from scripts.run_agentic_vla_libero import (
    _parse_semantic_label,
    _request_semantic_observation,
)


COMPATIBLE_PHYSICAL_FAILURES = {"MISGRASP", "DROP", "MISALIGN", "COLLISION", "OTHER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/carve_semantic_precision/temporal_pairs_t89/manifest.json"),
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument(
        "--semantic-protocol",
        choices=("label_v1", "code_v2", "code_v3", "code_v4", "code_v5"),
        default="label_v1",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_semantic_precision/semantic_precision_gate.json"),
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> int:
    args = parse_args()
    if args.max_tokens <= 0 or args.timeout_sec <= 0:
        raise ValueError("max-tokens and timeout-sec must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_records: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["records"]):
        started = time.perf_counter()
        error = None
        response = None
        try:
            response = _request_semantic_observation(
                {
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "timeout_sec": float(args.timeout_sec),
                    "max_tokens": int(args.max_tokens),
                    "semantic_protocol": str(args.semantic_protocol),
                    "task": record["instruction"],
                    "event": "visual execution anomaly detected",
                    "reference_image": imageio.imread(record["before_image"]),
                    "image": imageio.imread(record["after_image"]),
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        label = _parse_semantic_label(response.get("content") if response else None)
        status = label[0] if label else None
        failure = label[1] if label else None
        output_records.append(
            {
                **record,
                "request_index": index,
                "latency_ms": elapsed_ms,
                "content": response.get("content") if response else None,
                "usage": response.get("usage") if response else None,
                "protocol_valid": label is not None,
                "predicted_status": status,
                "predicted_failure": failure,
                "predicted_intervention": status not in {None, "NOMINAL"},
                "error": error,
            }
        )

    no_op = [item for item in output_records if item["severity"] == "no_op"]
    mild = [item for item in output_records if item["severity"] == "mild"]
    severe = [item for item in output_records if item["severity"] == "severe"]
    latencies = [float(item["latency_ms"]) for item in output_records]
    metrics = {
        "samples": len(output_records),
        "protocol_valid_rate": _rate([bool(item["protocol_valid"]) for item in output_records]),
        "request_error_rate": _rate([item["error"] is not None for item in output_records]),
        "no_op_false_positive_rate": _rate(
            [bool(item["predicted_intervention"]) for item in no_op]
        ),
        "severe_intervention_recall": _rate(
            [bool(item["predicted_intervention"]) for item in severe]
        ),
        "severe_physical_failure_recall": _rate(
            [item["predicted_failure"] in COMPATIBLE_PHYSICAL_FAILURES for item in severe]
        ),
        "mild_intervention_rate": _rate(
            [bool(item["predicted_intervention"]) for item in mild]
        ),
        "latency_ms_mean": _mean(latencies),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }
    gates = {
        "protocol_valid_rate_min": 0.95,
        "no_op_false_positive_rate_max": 0.20,
        "severe_intervention_recall_min": 0.80,
    }
    gate_checks = {
        "protocol_valid": metrics["protocol_valid_rate"] >= gates["protocol_valid_rate_min"],
        "no_op_false_positive": metrics["no_op_false_positive_rate"] <= gates["no_op_false_positive_rate_max"],
        "severe_intervention_recall": metrics["severe_intervention_recall"] >= gates["severe_intervention_recall_min"],
    }
    payload = {
        "gate": (
            "CARVE semantic plan-validity precision"
            if args.semantic_protocol in {"code_v3", "code_v4", "code_v5"}
            else "CARVE semantic intervention precision"
        ),
        "model": args.model,
        "max_tokens": int(args.max_tokens),
        "semantic_protocol": str(args.semantic_protocol),
        "controller_input": "before/after RGB + generic deployable visual anomaly signal",
        "evaluator_truth_exposed_to_controller": False,
        "metrics": metrics,
        "thresholds": gates,
        "checks": gate_checks,
        "passed": all(gate_checks.values()),
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(
        "\n".join(
            [
                (
                    "# CARVE Semantic Plan-Validity Gate"
                    if args.semantic_protocol in {"code_v3", "code_v4", "code_v5"}
                    else "# CARVE Semantic Precision Gate"
                ),
                "",
                "## Protocol",
                "",
                f"- Samples: `{metrics['samples']}` temporal pairs.",
                (
                    "- Hard labels: no-op should preserve the existing plan; severe "
                    "task-object displacement should invalidate it."
                    if args.semantic_protocol in {"code_v3", "code_v4", "code_v5"}
                    else "- Hard labels: no-op should remain nominal; severe displacement should trigger intervention."
                ),
                "- Mild displacement is a gray zone and is reported without a pass/fail target.",
                "- The VLM receives only before/after RGB, task text, and a generic deployable anomaly signal.",
                "",
                "## Results",
                "",
                "| Metric | Value | Gate |",
                "|---|---:|---:|",
                f"| Protocol validity | {100 * metrics['protocol_valid_rate']:.1f}% | >=95% |",
                f"| No-op false-positive rate | {100 * metrics['no_op_false_positive_rate']:.1f}% | <=20% |",
                f"| Severe intervention recall | {100 * metrics['severe_intervention_recall']:.1f}% | >=80% |",
                f"| Severe physical-failure recall | {100 * metrics['severe_physical_failure_recall']:.1f}% | report |",
                f"| Mild intervention rate | {100 * metrics['mild_intervention_rate']:.1f}% | gray zone |",
                f"| Semantic latency P95 | {metrics['latency_ms_p95']:.2f} ms | report |",
                "",
                f"## Decision: {'PASS' if payload['passed'] else 'FAIL'}",
                "",
                (
                    "Semantic labels may proceed to a bounded intervention shadow comparison."
                    if payload["passed"]
                    else "Semantic labels remain observation-only; no action intervention is enabled."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "passed": payload["passed"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
