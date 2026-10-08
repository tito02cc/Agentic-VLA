#!/usr/bin/env python3
"""Evaluate the online CARVE visual Critic contract on a fixed RoboDojo holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.runtime import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain import (
    GuardedVisualVerifier,
    VerificationStatus,
    VisualVerificationContext,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete = [item for item in records if not item["expected_complete"]]
    terminal = [item for item in records if item["expected_complete"]]
    latencies = [float(item["latency_ms"]) for item in records]
    return {
        "samples": len(records),
        "incomplete_samples": len(incomplete),
        "terminal_samples": len(terminal),
        "protocol_valid_rate": _rate([bool(item["protocol_valid"]) for item in records]),
        "incomplete_false_safe_stop_rate": _rate(
            [bool(item["predicted_safe_stop"]) for item in incomplete]
        ),
        "terminal_safe_stop_recall": _rate(
            [bool(item["predicted_safe_stop"]) for item in terminal]
        ),
        "control_accuracy": _rate(
            [item["predicted_control"] == item["control_target"] for item in records]
        ),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18071/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "carve.robodojo.critic-holdout.v1":
        raise ValueError("unsupported Critic holdout schema")
    infer = OpenAICompatibleVisionPlanner(
        endpoint=args.endpoint,
        model=args.model,
        timeout_s=args.timeout_sec,
        max_tokens=args.max_tokens,
    )
    verifier = GuardedVisualVerifier(infer, minimum_confidence=args.minimum_confidence)
    output_records: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["records"]):
        frames = {
            name: iio.imread(manifest_path.parent / relative_path)
            for name, relative_path in record["frames"].items()
        }
        result = verifier.verify(
            VisualVerificationContext(
                task_instruction=record["task_instruction"],
                expected_outcome=manifest["expected_outcome"],
                frames=frames,
                timestep=int(record["frame_index"]),
                active_stage="check complete task progress after sustained inactivity",
            )
        )
        predicted_safe_stop = bool(
            result.accepted and result.report.status is VerificationStatus.CONFIRMED
        )
        output_records.append(
            {
                **record,
                "request_index": index,
                "protocol_valid": bool(result.accepted),
                "predicted_status": result.report.status.value,
                "predicted_confidence": result.report.confidence,
                "predicted_observation": result.report.observed_outcome,
                "predicted_safe_stop": predicted_safe_stop,
                "predicted_control": "safe_stop" if predicted_safe_stop else "continue",
                "latency_ms": result.elapsed_ms,
                "error": result.error,
                "raw_output": result.raw_output,
            }
        )

    by_split = {
        split: _metrics([item for item in output_records if item["split"] == split])
        for split in ("development", "holdout")
    }
    holdout = by_split["holdout"]
    thresholds = {
        "protocol_valid_rate_min": 0.95,
        "control_accuracy_min": 0.95,
        "incomplete_false_safe_stop_rate_max": 0.02,
        "terminal_safe_stop_recall_min": 0.90,
    }
    checks = {
        "protocol_valid": holdout["protocol_valid_rate"]
        >= thresholds["protocol_valid_rate_min"],
        "control_accuracy": holdout["control_accuracy"]
        >= thresholds["control_accuracy_min"],
        "incomplete_false_safe_stop": holdout["incomplete_false_safe_stop_rate"]
        <= thresholds["incomplete_false_safe_stop_rate_max"],
        "terminal_safe_stop_recall": holdout["terminal_safe_stop_recall"]
        >= thresholds["terminal_safe_stop_recall_min"],
    }
    per_task = {
        task: _metrics([item for item in output_records if item["task"] == task and item["split"] == "holdout"])
        for task in sorted({item["task"] for item in output_records})
    }
    receipt = None
    if args.profile_receipt:
        receipt_path = args.profile_receipt.expanduser().resolve()
        receipt = {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "payload": json.loads(receipt_path.read_text(encoding="utf-8")),
        }
    payload = {
        "schema_version": "carve.robodojo.critic-admission.v1",
        "model": args.model,
        "endpoint": args.endpoint,
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "profile_receipt": receipt,
        "decision_policy": (
            "Only an accepted CONFIRMED result may safe-stop; all errors, low-confidence "
            "answers, contradictions, and inconclusive answers continue conservatively."
        ),
        "metrics_by_split": by_split,
        "holdout_metrics_by_task": per_task,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "records": output_records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": payload["passed"], "holdout": holdout, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
